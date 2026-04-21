---
title: >-
  [论文解读] LayerLock: Non-collapsing Representation Learning with Progressive Freezing
description: >-
  [3D视觉] 提出 LayerLock，一种通过渐进式冻结网络层并动态切换预测目标（从像素到越来越深的中间层特征）的自监督视频表征学习方法，兼具像素预测的稳定性和潜变量预测的高效语义捕获能力，用于训练高达 4B 参数的视频模型。
tags:
  - 3D视觉
---

# LayerLock: Non-collapsing Representation Learning with Progressive Freezing

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2509.10156](https://arxiv.org/abs/2509.10156)
- **作者**: Goker Erdogan, Nikhil Parthasarathy, Catalin Ionescu, Drew A. Hudson, Alexander Lerchner, Andrew Zisserman, Mehdi S. M. Sajjadi, João Carreira
- **机构**: Google DeepMind, University of Oxford
- **领域**: 3D视觉 / 自监督表征学习
- **关键词**: 自监督学习, 视频表征, 渐进冻结, 潜变量预测, Masked Auto-Encoding, V-JEPA

## 一句话总结
提出 LayerLock，一种通过渐进式冻结网络层并动态切换预测目标（从像素到越来越深的中间层特征）的自监督视频表征学习方法，兼具像素预测的稳定性和潜变量预测的高效语义捕获能力，用于训练高达 4B 参数的视频模型。

## 研究背景与动机

### 问题定义
自监督视觉表征学习的两大范式各有利弊：
- **像素预测**（如 VideoMAE）：学习信号稳定、不会崩溃（collapse），但需要大量训练数据，且学到的特征偏向低级视觉信息，与下游任务不完全匹配
- **潜变量预测**（如 V-JEPA）：数据效率高，能学到更高级的语义特征，但需要各种训练技巧（非对称架构、EMA 教师网络、stop-gradient）来避免表征崩溃

### 关键观察
在 Video MAE 训练过程中，**ViT 的层按深度顺序收敛**：浅层先收敛，深层后收敛。作者通过冻结实验验证了这一现象——在某步冻结前 $L$ 层后继续训练，若最终 loss 接近基线 $L_{base}$，则认为这些层已"收敛"。

### 核心创新思路
利用这一层级收敛规律，提出**渐进冻结 + 动态目标切换**：训练初期预测像素（稳定），随着浅层收敛，逐步冻结这些层并切换到预测更深层的中间激活值。这样既保留了像素预测的稳定性，又逐步引入潜变量预测的语义学习优势，同时避免表征崩溃。

## 方法详解

### 整体框架
LayerLock 基于标准 MAE 架构，核心改动在于训练过程中的预测目标管理：
1. 初始阶段预测像素（标准 MAE），持续 $N_{\text{pixel}}$ 步
2. 此后每隔 $N$ 步冻结下 $k$ 层，预测目标切换到新冻结层的输出 $h_k$
3. 持续渐进直到冻结 3/4 的网络层

### 模型架构

**编码器**：标准 ViT backbone，输入视频 $x \in \mathbb{R}^{T \times H \times W \times 3}$，分割为 $2 \times 16 \times 16$ 的 patch，随机 mask 95%，线性投影到 $D$ 维。

**解码器**：复用 backbone 最后 4 层作为解码器。在 backbone 某层前拼接解码 latent tokens $z \in \mathbb{R}^{N \times D}$，通过剩余 Transformer 块处理后，用逐 patch 线性层投影到目标空间。

**3D 旋转位置编码**（3D RoPE）：将特征维度分为 4 部分，分别对时间、高度、宽度三个轴施加 1D RoPE（10%、25%、25% 比例），第四部分不加位置信息。关键发现：RoPE 作用在第一个归一化层之后比作用在 attention 中更有效。

### 渐进冻结与潜变量预测

**冻结调度**（以 ViT-G 为例）：
- 前 160K 步：预测像素（标准 MAE loss）
- 每 10K 步冻结 1 层，预测目标切换到新冻结层的输出
- 直到冻结 32 层（总 48 层的 2/3）
- 使用 L2 loss：$\text{loss} = \|h_k - \hat{h}_k\|^2$

**调度确定启发式**：当前几层的参数范数开始趋于平稳（或使用 weight decay 时开始变小），即开始冻结。

**每次目标切换时的 mini-warmup**：逐步增加学习率，避免突然切换目标导致训练不稳定（带来约 1% 的 SSv2 提升）。

### 扩展到 V-JEPA
LayerLock 也可应用于已使用潜变量预测的方法：
- 训练初期预测 EMA 教师网络的第一层激活
- 渐进冻结并切换到预测更深层激活
- 使用 ViT-L backbone + 12 层 Transformer 解码器
- 仍保留 EMA 教师网络（虽然渐进冻结本身可避免崩溃，但 EMA 在下游性能更好）

## 实验关键数据

### 主实验：LayerLock vs 基线

| 模型 | LayerLock | 参数量(M) | SSv2↑ | K700↑ | ScanNet↓ |
|------|-----------|----------|-------|-------|----------|
| 4DS-G (MAE) | ✗ | 1,848 | 63.1 | 52.1 | 1.02 |
| 4DS-G (MAE) | ✓ | 1,868 | **66.1** | **56.3** | **1.00** |
| 4DS-e (MAE) | ✗ | 3,811 | 64.6 | 54.4 | 0.94 |
| 4DS-e (MAE) | ✓ | 3,818 | **67.1** | **57.9** | 0.98 |
| V-JEPA-L | ✗ | 235 | 52.1 | 42.5 | 1.57 |
| V-JEPA-L | ✓ | 303 | **57.0** | **43.5** | **1.51** |

- SSv2 动作分类提升约 3%（MAE）和 5%（V-JEPA），非常显著
- K700 动作分类也有明显提升
- 深度估计（ScanNet）性能保持或略有提升

### 消融实验

**效率消融**：渐进冻结节省 9% 总 FLOPs + 16% 峰值内存，性能几乎无损

| 配置 | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| 基线 MAE（无冻结） | 56.1 | 0.15 |
| 渐进冻结 MAE | 56.0 | 0.16 |

**MAE + latent loss 会崩溃**：

| 模型 | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| 4DS-H | 50.1 | 0.19 |
| + latent (const weight) | 3.7 | 0.38 |
| + latent (cosine schedule) | 5.6 | 0.37 |

直接在 MAE 上添加 latent loss 而不冻结 → 严重表征崩溃！这证实了渐进冻结对避免崩溃至关重要。

**3D RoPE 消融**：

| 配置 | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| 无 RoPE | 56.1 | 0.15 |
| + 3D RoPE | 58.9 | 0.13 |
| + 3D RoPE + LayerLock | 60.1 | 0.13 |

RoPE 独立贡献约 2.8% SSv2 提升，与 LayerLock 互补叠加。

**Patch 子采样效率消融**：

| 配置 | SSv2↑ | ScanNet↓ |
|------|-------|----------|
| 基线 MAE | 63.1 | 1.02 |
| LayerLock 5% patches | 64.9 | 1.12 |
| LayerLock 100% patches | 66.1 | 1.00 |

仅用 5% 的 patches 计算 latent loss 也能显著超过基线，但深度估计稍有下降。

### 关键发现
1. **层级收敛顺序**：浅层先收敛、深层后收敛，这一观察对设计训练策略有重要指导意义
2. **单目标 vs 多目标**：每次只预测最新冻结层的输出（单目标）与同时预测所有已冻结层（多目标）性能相当，简单方案即可
3. **冻结调度**：逐层冻结（interval=2K, jump=1）效果最好，一次冻结过多层会显著降低性能
4. **更长训练受益更大**：1B 训练样本时 FLOP 效率提升可达 19%

## 亮点与洞察

1. **观察驱动的方法设计**：从"层级按深度顺序收敛"这一实证观察出发，自然推导出渐进冻结策略
2. **避免崩溃的优雅方案**：无需 asymmetric architecture、contrastive loss 等复杂技巧，仅靠渐进冻结 + 动态目标即可实现稳定的 latent 预测
3. **通用性强**：在像素预测（4DS MAE）和潜变量预测（V-JEPA）两种范式上均有效，且可扩展到 4B 参数模型
4. **计算效率**：渐进冻结不仅提升表征质量，还节省训练 FLOPs 和峰值内存
5. **类似生物视觉发育的关键期**：浅层先收敛固定的模式与神经科学中的"关键期可塑性"相呼应

## 局限性

- 冻结调度目前依赖手动启发式（观察参数范数变化），缺乏自适应机制
- 在深度估计等低级任务上的提升不如高级语义任务显著
- 仅在视频任务上验证，尚未扩展到图像自监督学习
- 训练规模极大（1B 视频 clip，256 TPU-v6），实验复现困难

## 相关工作与启发

- **vs V-JEPA**：LayerLock 通过渐进冻结提供稳定目标，可部分替代 EMA 教师网络的角色（但实验发现保留 EMA 性能更好）
- **vs FreezeOut**：FreezeOut 在监督学习中用渐进冻结加速训练，LayerLock 扩展为自监督表征学习中的 target 切换策略
- **vs 4DS**：在 4DS 的 MAE 基础上，LayerLock 通过从像素到潜变量的渐进过渡获得更好的语义特征
- **启发**：渐进冻结范式可扩展到更长视频、更高分辨率甚至更深模型，且计算效率的提升随训练规模增大而更显著

## 评分 ⭐⭐⭐⭐⭐
观察深刻，方法简洁优雅，理论动机清晰。在像素预测和潜变量预测两种主流范式上都有一致改进，且兼顾计算效率。消融实验极其全面（冻结调度、patch子采样、崩溃验证、RoPE等），为大规模视频自监督学习提供了新的训练范式。

<!-- RELATED:START -->

## 相关论文

- [Discretized Gaussian Representation for Tomographic Reconstruction](discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [Relative Illumination Fields: Learning Medium and Light Independent Underwater Scenes](relative_illumination_fields_learning_medium_and_light_independent_underwater_sc.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)

<!-- RELATED:END -->
