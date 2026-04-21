---
title: >-
  [论文解读] Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models
description: >-
  [CVPR 2025][3D视觉][3D重建增强] 提出 Difix3D+，利用微调的单步扩散模型（SD-Turbo）在训练阶段渐进式生成伪训练视角回馈 3D 表示，并在推理阶段作为实时后处理增强器，同时兼容 NeRF 和 3DGS，在 FID 上平均实现 2 倍以上提升。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D重建增强
  - 扩散模型
  - 伪影去除
  - 新视角合成
  - 神经渲染
---

# Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2503.01774](https://arxiv.org/abs/2503.01774)  
**代码**: [https://research.nvidia.com/labs/toronto-ai/difix3d](https://research.nvidia.com/labs/toronto-ai/difix3d)  
**领域**: 3D视觉  
**关键词**: 3D重建增强、扩散模型、伪影去除、新视角合成、神经渲染

## 一句话总结

提出 Difix3D+，利用微调的单步扩散模型（SD-Turbo）在训练阶段渐进式生成伪训练视角回馈 3D 表示，并在推理阶段作为实时后处理增强器，同时兼容 NeRF 和 3DGS，在 FID 上平均实现 2 倍以上提升。

## 研究背景与动机

NeRF 和 3DGS 在训练视角附近能生成高质量图像，但在稀疏观测、远离训练视角等场景下仍存在严重伪影（虚假几何、缺失区域）。核心矛盾是：(1) 逐场景优化方法缺乏数据先验，无法在欠约束区域合理"脑补"几何和外观；(2) 现有方法将扩散模型用作每步优化的评分函数（如 SDS），计算代价极高且难以扩展到大场景。本文的关键洞察是：**渲染伪影的分布与扩散模型训练时某个特定噪声水平的图像分布高度相似**，因此只需对单步扩散模型进行轻量微调即可将其转化为"3D 伪影修复器"。

## 方法详解

### 整体框架

Difix3D+ 管线分三步：(1) Difix 模型：微调 SD-Turbo 去除渲染图像中的伪影；(2) Difix3D：渐进式生成伪训练视角并回馈 3D 表示更新；(3) Difix3D+：推理时额外使用 Difix 作为实时后处理增强器。整个管线与表示形式无关，一个模型同时修复 NeRF 和 3DGS 渲染的伪影。

### 关键设计

1. **Difix：单步扩散伪影修复器**:
    - 功能：将含伪影的渲染图像转化为干净图像
    - 核心思路：基于 SD-Turbo 的 image-to-image 微调，输入为含伪影渲染图 $\tilde{I}$（而非随机高斯噪声），使用较低噪声水平 $\tau=200$（而非标准的 $\tau=1000$），引入参考视角 $I_{\text{ref}}$ 通过 cross-view reference mixing layer（将 view 维度合并到空间维度做 self-attention）提供颜色/纹理参考
    - 设计动机：实验验证渲染伪影图像与 $\tau=200$ 的加噪图像分布最接近——$\tau$ 太高会改变图像内容，太低则无法去除伪影。冻结 VAE encoder + LoRA 微调 decoder，几小时即可在单 GPU 上完成

2. **渐进式 3D 更新策略（Progressive 3D Updates）**:
    - 功能：将 Difix 修复的伪训练视角回馏 3D 表示，确保多视角一致性
    - 核心思路：从训练视角出发，每 1.5K 迭代将相机位姿稍微偏移向目标视角，渲染 → Difix 修复 → 加入训练集 → 继续优化 3D 表示。逐步扩展重建的空间范围，确保扩散模型始终有较强的条件信号
    - 设计动机：直接对远离训练视角的渲染做修复会导致扩散模型需要大量"脑补"，引入多视角不一致；渐进式策略让每一步的修复都相对简单，不一致的 3D 结构在后续优化中被消除

3. **数据构造策略（Data Curation）**:
    - 功能：构造大规模伪影-干净图像配对训练数据
    - 核心思路：四种互补策略：稀疏重建（DL3DV 跳帧训练）、循环重建（自动驾驶场景，平移轨迹→训NeRF→渲染偏移视角→再训NeRF）、交叉参考（多相机取一个训练其他评估）、欠拟合模型（减少训练epoch到25%-75%）
    - 设计动机：直接跳帧在大多数数据集上偏差不够大（保留视角仍覆盖同区域），需要更激进的策略生成足够显著的伪影

### 损失函数 / 训练策略

- $\mathcal{L} = \mathcal{L}_{\text{Recon}} + \mathcal{L}_{\text{LPIPS}} + 0.5 \cdot \mathcal{L}_{\text{Gram}}$
- Gram 矩阵损失鼓励更锐利的细节：$\mathcal{L}_{\text{Gram}} = \frac{1}{L} \sum_{l=1}^{L} \beta_l \|G_l(\hat{I}) - G_l(I)\|_2$
- 微调在单张消费级 GPU 上仅需数小时
- 推理时后处理仅需 76ms/帧（A100 GPU），比多步扩散快 10 倍以上

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | LPIPS↓ | FID↓ |
|--------|------|-------|--------|------|
| Nerfbusters | Nerfacto 基线 | 17.29 | 0.4021 | 134.65 |
| Nerfbusters | Nerfbusters | 17.72 | 0.3521 | 116.83 |
| Nerfbusters | **Difix3D+ (Nerfacto)** | **18.32** | **0.2789** | **49.44** |
| DL3DV | 3DGS 基线 | 17.18 | 0.3835 | 107.23 |
| DL3DV | **Difix3D+ (3DGS)** | **17.99** | **0.2932** | **40.86** |
| RDS (驾驶) | Nerfacto 基线 | 19.95 | 0.5300 | 91.38 |
| RDS (驾驶) | **Difix3D+ (Nerfacto)** | **21.75** | **0.4016** | **73.08** |

### 消融实验

| 配置 | PSNR↑ | LPIPS↓ | FID↓ | 说明 |
|------|-------|--------|------|------|
| Nerfacto 基线 | 17.29 | 0.4021 | 134.65 | 原始 |
| + (a) Difix 直接后处理 | 17.40 | 0.2996 | 49.87 | FID大降但不一致 |
| + (b) 一次性3D更新 | 17.97 | 0.3424 | 75.94 | PSNR升 FID反弹 |
| + (c) 渐进式3D更新 (Difix3D) | 18.08 | 0.3277 | 63.77 | 渐进优于一次性 |
| + (d) 实时后处理 (Difix3D+) | **18.32** | **0.2789** | **49.44** | 最终最优 |

### 关键发现

- 渐进式 3D 更新（c）比一次性更新（b）显著更好：LPIPS 降 0.015，FID 降 12
- Difix3D+ 实现了**约 2.7 倍 FID 改善**（Nerfbusters: 134.65→49.44），同时 PSNR 也提升了 1dB
- 同一个 Difix 模型能同时修复 NeRF 和 3DGS 的伪影，展示了强泛化性
- $\tau=200$ 是最优噪声水平：PSNR 17.73 vs $\tau=600$ 的 15.64

## 亮点与洞察

- **关键洞察极为直觉**：渲染伪影 ≈ 低噪声级别的加噪图像，使得单步扩散模型几乎"零成本"适配为伪影修复器
- **一个模型修复所有表示**：对 NeRF 和 3DGS 的伪影都有效，体现了学到的是通用的"自然图像"先验
- 训练 + 推理双阶段使用 Difix：训练阶段改善 3D 一致性，推理阶段修复残留不完美——设计理念清晰
- 计算效率高：微调数小时，无需每步查询扩散模型（比 IN2N 等快 10 倍以上）

## 局限与展望

- 后处理步骤可能引入帧间微小不一致（尽管渐进更新已大幅缓解）
- 当前数据构造策略需要为不同数据集类型设计不同方案，自动化程度有待提升
- 参考视角选择（最近训练视角）策略较简单，更智能的参考选择可能进一步提升效果
- Gram 矩阵损失权重等超参的敏感性未充分讨论

## 相关工作与启发

- 与 3DGS-Enhancer 并行工作，区别在于：渐进式更新策略 + 推理时后处理
- 与 IN2N 类方法（每步查询扩散）相比快 10 倍以上
- 噪声水平洞察可推广：其他退化类型（如低分辨率、压缩伪影）可能也有对应最优 $\tau$
- 数据构造策略（循环重建、欠拟合模型）可直接复用于其他 3D 增强研究

## 评分

- 新颖性: ⭐⭐⭐⭐ 渲染伪影≈低噪声图像的洞察新颖且有说服力，渐进式更新策略有效
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、两种3D表示、详细消融、驾驶场景验证，极为充分
- 写作质量: ⭐⭐⭐⭐⭐ NVIDIA 出品的典型高质量论文，逻辑清晰、图示精美
- 价值: ⭐⭐⭐⭐⭐ 实用性极强——通用、高效、开源，对任何 3D 重建工作都有直接价值

<!-- RELATED:START -->

## 相关论文

- [BokehDiff: Neural Lens Blur with One-Step Diffusion](../../ICCV2025/3d_vision/bokehdiff_neural_lens_blur_with_one-step_diffusion.md)
- [Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)
- [Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)

<!-- RELATED:END -->
