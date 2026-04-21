---
title: >-
  [论文解读] Learning Flow Fields in Attention for Controllable Person Image Generation
description: >-
  [CVPR 2025][图像生成][virtual try-on] 提出 Leffa（Learning Flow Fields in Attention），在扩散模型的注意力层中将 attention map 转换为流场并进行像素级正则化监督，显式引导 target query 关注正确的 reference key 区域，**零额外推理开销**地减少细粒度细节（纹理、文字、logo）失真，在虚拟试衣（VITON-HD、DressCode）和姿态迁移（DeepFashion）上均 SOTA。
tags:
  - CVPR 2025
  - 图像生成
  - virtual try-on
  - pose transfer
  - 扩散模型
  - flow field
  - 注意力机制
---

# Learning Flow Fields in Attention for Controllable Person Image Generation

**会议**: CVPR 2025  
**arXiv**: [2412.08486](https://arxiv.org/abs/2412.08486)  
**代码**: 待确认（Meta AI）  
**领域**: image_generation  
**关键词**: virtual try-on, pose transfer, diffusion model, flow field, attention regularization

## 一句话总结

提出 Leffa（Learning Flow Fields in Attention），在扩散模型的注意力层中将 attention map 转换为流场并进行像素级正则化监督，显式引导 target query 关注正确的 reference key 区域，**零额外推理开销**地减少细粒度细节（纹理、文字、logo）失真，在虚拟试衣（VITON-HD、DressCode）和姿态迁移（DeepFashion）上均 SOTA。

## 研究背景与动机

**领域现状**: 可控人物图像生成（虚拟试衣、姿态迁移）基于扩散模型已取得高质量结果，但在近距离观察时仍存在细粒度纹理失真（条纹方向错误、文字变形、按钮数量不对等）。

**现有方案的局限**:
1. **辅助模型方案** (IDM-VTON, OOTDiffusion): 加入 CLIP/DINOv2 特征或 warping 模型，增加模型复杂度但缺乏显式的视觉一致性监督
2. **多阶段推理** (Yang et al.): 增加推理成本
3. **根本原因**: 作者通过**可视化注意力图**发现，细节失真区域的 target query 注意力分散到错误区域，而非聚焦于 reference 中的对应位置

**关键发现**: 手动修正注意力图（将最高响应交换到正确区域），**无需任何额外训练**即可显著修复纹理失真。这启发了用显式监督来引导注意力的研究方向。

## 方法详解

### 整体框架

基于 SD1.5 的 baseline：
- 复制预训练 UNet 为 **Generative UNet**（处理源图像）和 **Reference UNet**（处理参考图像）
- 去掉文本编码器和文本交叉注意力（纯视觉条件）
- 通过**空间拼接自注意力** (Spatially Concatenated Self-Attention) 实现两个 UNet 间的特征交互

Leffa loss 作为正则化项在微调阶段加入，**无额外参数和推理开销**。

### 关键设计

#### 1. 注意力流场 (Flow Fields in Attention)

核心思路 — 将注意力图解释为空间对应关系：
- 在第 $l$ 层注意力中，$Q = F_{gen}^l$（target），$K = F_{ref}^l$（reference）
- 计算注意力图 $A^l = \text{softmax}(QK^\top / \sqrt{d} / \tau)$，跨 head 维度取平均得 $\hat{A^l}$
- 构建归一化坐标图 $C^l \in \mathbb{R}^{n^l \times 2}$（左上 $[-1,-1]$ 到右下 $[1,1]$）
- **流场** $\mathcal{F}^l = \hat{A^l} \cdot C^l$：每个 target token 加权汇聚 reference 坐标，得到其"关注"的空间位置

#### 2. 像素级流场监督 (Leffa Loss)

- 将流场双线性上采样到原图分辨率 $\mathcal{F}_{up}^l \in \mathbb{R}^{H \times W \times 2}$
- 用 $\mathcal{F}_{up}^l$ 做 grid sampling，将参考图 $I_{ref}$ warp 到目标空间得 $I_{warp}^l$
- L2 损失：$\mathcal{L}_{leffa} = \sum_{l=1}^{L} \| I_{tgt} * I_m - I_{warp}^l * I_m \|_2^2$

训练时 $I_{src} = I_{tgt}$ 同一张图，mask $I_m$ 限制为仅衣物/人体区域。

#### 3. 精心设计的适用条件

- **注意力层选择**: 仅分辨率 ≥ 原图 $1/32$ 的高分辨率注意力层参与（低分辨率 warp 不精确）
- **时间步选择**: 仅 $t < 500$（$T=1000$）时计算 Leffa loss（噪声太大时注意力无法正确对齐语义）
- **温度系数**: 使用较大 $\tau=2.0$ 使注意力更平滑，容错性更高
- **渐进训练**: 先低分辨率训练 baseline → 高分辨率训练 → **最终阶段**加入 Leffa loss 微调

### 损失函数

$\mathcal{L}_{finetune} = \mathcal{L}_{diffusion} + \lambda_{leffa} \mathcal{L}_{leffa}$

$\lambda_{leffa} = 10^{-3}$，以 Leffa loss 作为正则项，不干扰主生成质量。

## 实验关键数据

### 主实验表

**VITON-HD 虚拟试衣**:

| 方法 | Paired FID ↓ | SSIM ↑ | LPIPS ↓ | Unpaired FID ↓ |
|------|-------------|--------|---------|---------------|
| CatVTON | 5.42 | 0.870 | 0.057 | 9.02 |
| IDM-VTON | 5.76 | 0.850 | 0.063 | 9.84 |
| StableVITON | 8.23 | 0.888 | 0.073 | - |
| **Leffa** | **4.54** | **0.899** | **0.048** | **8.52** |

Paired FID 从 5.42 降至 **4.54** (−16.2%)，LPIPS 从 0.057 降至 **0.048** (−15.8%)。

**DressCode 虚拟试衣 (全品类)**:

| 方法 | Paired FID ↓ | SSIM ↑ | Unpaired FID ↓ |
|------|-------------|--------|---------------|
| CatVTON | 3.99 | 0.892 | 6.14 |
| OOTDiffusion | 4.61 | 0.885 | 12.57 |
| **Leffa** | **2.06** | **0.924** | **4.48** |

Paired FID 从 3.99 降至 **2.06** (−48.4%)，提升极为显著。

**DeepFashion 姿态迁移 (512×352)**:

| 方法 | FID ↓ | SSIM ↑ | LPIPS ↓ |
|------|-------|--------|---------|
| CFLD | 9.36 | 0.729 | 0.171 |
| PIDM | 9.81 | 0.684 | 0.192 |
| **Leffa** | **7.75** | 0.714 | **0.159** |

### 关键发现

- Leffa loss 是**模型无关**的：应用到 IDM-VTON 上 Paired FID 从 5.76 → 5.20，应用到 CatVTON 上 5.42 → 5.11
- 可视化验证：加入 Leffa 后注意力图从分散状态变为精确对准对应区域
- 温度 $\tau=2.0$ 最优：过小则梯度不稳定，过大则匹配过于模糊
- 时间步阈值 500 最优：低于 200 太严格（监督信号不足），高于 700 噪声干扰太大

## 亮点与洞察

1. **洞察深刻**: 从注意力可视化归因细节失真的根本原因，并通过手动修正实验验证因果关系——这是一个教科书级的研究方法论
2. **实现极简**: Leffa loss 仅需从已有注意力图计算流场 + L2 损失，零额外参数、零额外推理开销
3. **模型无关性**: 可即插即用到任何使用 reference 注意力的扩散模型中，通用性强
4. **统一框架**: 一个 baseline 同时处理虚拟试衣和姿态迁移两个任务，架构简洁
5. **DressCode 上极大提升** (Paired FID 2.06)：说明在复杂衣物品类上 Leffa 的细节保持优势更突出

## 局限与展望

1. 基于 SD1.5 构建，迁移到 **SDXL/SD3** 等更强基座可能进一步提升效果
2. 当参考图和目标图之间存在**严重遮挡**或**大角度变化**时，flow field 假设（单一对应关系）可能不成立
3. 仅在人物图像上验证，**通用物体/场景**的可控生成能否受益于 Leffa 未探索
4. 渐进训练策略增加了总训练步数，能否**从训练初期就加入 Leffa loss** 值得研究

## 相关工作与启发

- **IDM-VTON** (Choi et al., CVPR): 当前虚拟试衣 SOTA 之一，本文在其基础上验证了 Leffa 的通用性
- **CatVTON** (Chong et al.): 拼接自注意力范式的代表，本文 baseline 设计与之类似但更简洁
- **CFLD** (Lu et al.): 姿态迁移 SOTA，本文在 FID 上大幅超越但 SSIM 略低
- **启发**: "注意力 → 流场 → 像素级监督"的范式可推广到任何需要空间对齐的注意力机制（如 inpainting、image editing、video generation）

## 评分

⭐⭐⭐⭐⭐ — 洞察精准、方法优雅（从注意力到流场的转换非常自然）、实验全面（3 数据集 × 2 任务 × 模型无关验证）、实用性极强（零额外推理开销）。Top-tier CVPR 工作。

<!-- RELATED:START -->

## 相关论文

- [DNF: Unconditional 4D Generation with Dictionary-Based Neural Fields](dnf_unconditional_4d_generation_with_dictionary-based_neural_fields.md)
- [CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [Controllable Human Image Generation with Personalized Multi-Garments](controllable_human_image_generation_with_personalized_multi-garments.md)
- [Multi-focal Conditioned Latent Diffusion for Person Image Synthesis](multi-focal_conditioned_latent_diffusion_for_person_image_synthesis.md)
- [Consistent and Controllable Image Animation with Motion Diffusion Models](consistent_and_controllable_image_animation_with_motion_diffusion_models.md)

<!-- RELATED:END -->
