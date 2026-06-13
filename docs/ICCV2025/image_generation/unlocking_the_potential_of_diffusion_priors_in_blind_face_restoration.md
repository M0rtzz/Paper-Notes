---
title: >-
  [论文解读] Unlocking the Potential of Diffusion Priors in Blind Face Restoration
description: >-
  [ICCV 2025][图像生成][盲人脸修复] 本文提出 FLIPNET，一个基于 T2I 扩散模型的统一框架，通过翻转输入在修复模式（BoostHub 选择性融合 LQ 特征 + BFR-oriented 面部嵌入）和退化模式（从真实退化数据集学习并合成退化图像）之间切换，同时解决 HQ/LQ 分布差距和合成/真实退化差距两大难题。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "盲人脸修复"
  - "扩散先验"
  - "双模式网络"
  - "退化建模"
  - "LoRA"
---

# Unlocking the Potential of Diffusion Priors in Blind Face Restoration

**会议**: ICCV 2025  
**arXiv**: [2508.08556](https://arxiv.org/abs/2508.08556)  
**代码**: 暂无  
**领域**: 图像生成  
**关键词**: 盲人脸修复, 扩散先验, 双模式网络, 退化建模, LoRA

## 一句话总结

本文提出 FLIPNET，一个基于 T2I 扩散模型的统一框架，通过翻转输入在修复模式（BoostHub 选择性融合 LQ 特征 + BFR-oriented 面部嵌入）和退化模式（从真实退化数据集学习并合成退化图像）之间切换，同时解决 HQ/LQ 分布差距和合成/真实退化差距两大难题。

## 研究背景与动机

**盲人脸修复(BFR)的两大鸿沟**：
   - **HQ vs LQ 差距**：扩散模型在高质量图像上训练，但 BFR 需处理中度到重度退化图像
   - **合成 vs 真实退化差距**：训练用 LQ 图像由简单退化模型合成（模糊+噪声+压缩），无法模拟真实世界复杂退化

**现有方法的局限**：
   - **预处理+扩散**（DiffBIR, PMRF）：预处理模块（SwinIR）去除退化的同时也抹掉了皱纹、雀斑等面部细节
   - **条件拼接**（WaveFace）：将 LQ 图像直接拼接为输入，但退化模型有限导致真实世界表现差
   - **Real-ESRGAN 退化模型**：反复应用经典退化，但模式单一、多样性不足

**核心目标**：仅用一个 T2I 模型（+ 少量可训练参数）同时解决两个鸿沟，无需预处理模块。

## 方法详解

### 整体架构 (FLIPNET)

基于 Stable Diffusion 2.1，输入为 (HQ, LQ) 图像对。**翻转输入即可切换模式**：
- **修复模式**：HQ 为输入，LQ 为条件 → 学习从 LQ 恢复 HQ
- **退化模式**：LQ 为输入，HQ 为条件 → 学习 HQ→LQ 退化映射

训练仅更新 LoRA (rank=64) 权重 + BoostHub + Adapter，基础模型完全冻结。

### 关键设计一：BoostHub

并行于 UNet 自注意力层，通过交叉注意力选择性融合 LQ 特征：

$$F_{ro} = W_O \cdot \text{Softmax}(Q_xK_y^T/\sqrt{d})V_y$$
$$F_{joint} = \text{Self-Attn}(F_x) + \phi \cdot F_{ro}$$

- $Q_x$ 来自 HQ 特征，$K_y, V_y$ 来自 LQ 特征
- 增强权重 $\phi$ 控制 LQ 特征融入程度：$\phi=0$ 纯生成、$\phi=2$ LQ 特征主导、$\phi=1$ 最佳平衡
- 输出投影矩阵 $W_O$ 初始化为全零，避免初始引入有害影响

### 关键设计二：BFR-oriented 面部嵌入

发现 ArcFace 等人脸识别嵌入不适合 BFR——识别模型训练于区分身份而非保留细节，注意力仅集中在判别区域（眼睛）。

**两阶段训练**：
1. **重建阶段**：分别训练 HQ/LQ 自编码器，使其能从潜在空间重建输入
    - 损失：$\mathcal{L}_{ae} = \mathcal{L}_1 + \lambda_{ap}\mathcal{L}_p + \lambda_{adv}\mathcal{L}_{adv}$
2. **关联阶段**：用交叉熵损失对齐 HQ/LQ 潜在空间
    - 计算 HQ/LQ patch 特征的相似矩阵 $\mathcal{M} \in \mathbb{R}^{N \times N}$
    - 最大化对角线相似度：$\mathcal{L}^{H(L)}_{ce} = -\frac{1}{N}\sum_i\sum_j y_{i,j}\log(z_{x(y)}^{i,j})$

训练后的 LQ 编码器提取面部嵌入 $\mathbf{z}_y$，通过 Adapter $\tau$ 对齐到文本 token 维度后经交叉注意力集成。

### 关键设计三：退化模式

翻转输入后，模型从真实退化数据集（Dense-Haze, LOL, SIDD, RealBlur 约 2500 对）学习退化分布，生成多样且真实的退化图像：
- 每张 HQ 面部图像可生成 5 张不同退化变体
- 退化强度完全未知，契合"盲"修复设定
- 训练时仅使用 $\mathcal{L}_{ldm}$（不加图像级约束以避免引入噪声）

### 损失函数

修复模式：$\mathcal{L}_{rm} = \mathcal{L}_{ldm} + \lambda_{mse}\mathcal{L}_{mse} + \lambda_p\mathcal{L}_p$

其中 $\mathcal{L}_{mse}$ 和 $\mathcal{L}_p$ 对每步去噪预测 $\hat{x}_0$ 施加图像级约束。

## 实验

### CelebA-Test 定量对比

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | Deg.↓ |
|------|-------|-------|--------|------|-------|
| DiffBIR | 基线 | - | - | - | - |
| WaveFace | 基线 | - | - | - | - |
| RestoreFormer++ | 基线 | - | - | - | - |
| **FLIPNET (+O)** | 改善 | 改善 | 改善 | 改善 | 改善 |
| **FLIPNET (+O/F)** | **最优** | **最优** | **最优** | **最优** | **最优** |

### 退化建模对比

| 方法 | 分布覆盖 |
|------|---------|
| Real-ESRGAN | 聚集在有限区域，模式单一 |
| **FLIPNET 退化模式** | 广泛分布于 LFW/WIDER/WebPhoto 三个真实数据集 |

### 关键发现

- BoostHub 增强权重 $\phi=1$ 时 PSNR 和 Identity Angle 达到最佳平衡
- BFR-oriented 嵌入 vs ArcFace 嵌入：前者注意力分布更均匀（覆盖整脸而非仅眼睛），细节保持更好
- 关联阶段对齐 HQ/LQ 嵌入空间是关键——缺少此步骤会导致偏向 LQ 特征使修复质量下降
- 混合使用在线退化和离线退化（FLIPNET 退化模式）训练效果最好
- Real-ESRGAN + FLIPNET 退化的组合训练数据覆盖了更广泛的退化分布

## 亮点与洞察

1. **翻转输入切换模式**的设计极为优雅——同一架构同时解决修复和退化建模
2. **BFR-oriented 嵌入**的观察深刻：识别嵌入的判别性目标与修复的保真性目标不一致
3. **无需预处理模块**：完全端到端，避免了预处理抹去细节的问题

## 局限性

- 退化模式训练依赖非面部数据集（去雾、去噪），领域差异可能影响面部退化质量
- LoRA + BoostHub + Adapter 的训练流程较复杂
- 推理速度受扩散采样步数限制

## 相关工作

- **扩散先验 BFR**: DiffBIR, PMRF, WaveFace, PGDiff
- **生成先验 BFR**: GFP-GAN, GPEN, DAEFR
- **退化建模**: Real-ESRGAN, BSRGAN

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 翻转输入双模式设计 + BFR-oriented 嵌入
- 技术深度：⭐⭐⭐⭐⭐ — BoostHub + 嵌入对齐 + 退化模式多层设计
- 实验充分度：⭐⭐⭐⭐ — 合成+多个真实数据集
- 实用价值：⭐⭐⭐⭐ — 真实世界退化修复效果优异

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] BVINet: Unlocking Blind Video Inpainting with Zero Annotations](bvinet_unlocking_blind_video_inpainting_with_zero_annotations.md)
- [\[ICCV 2025\] FedDifRC: Unlocking the Potential of Text-to-Image Diffusion Models in Heterogeneous Federated Learning](feddifrc_unlocking_the_potential_of_text-to-image_diffusion_models_in_heterogene.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](../../CVPR2025/image_generation/osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[CVPR 2025\] SVFR: A Unified Framework for Generalized Video Face Restoration](../../CVPR2025/image_generation/svfr_a_unified_framework_for_generalized_video_face_restoration.md)

</div>

<!-- RELATED:END -->
