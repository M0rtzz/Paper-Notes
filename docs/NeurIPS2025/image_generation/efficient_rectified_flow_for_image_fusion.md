---
title: >-
  [论文解读] Efficient Rectified Flow for Image Fusion
description: >-
  [NeurIPS 2025][图像生成][Image Fusion] 提出 RFfusion，首次将 Rectified Flow 引入图像融合任务，实现无需额外训练的单步采样，同时设计面向融合的两阶段 VAE 训练策略，在速度和质量上全面超越现有扩散融合方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - Image Fusion
  - Rectified Flow
  - 扩散模型
  - VAE
  - One-Step Sampling
---

# Efficient Rectified Flow for Image Fusion

**会议**: NeurIPS 2025  
**arXiv**: [2509.16549](https://arxiv.org/abs/2509.16549)  
**代码**: [zirui0625/RFfusion](https://github.com/zirui0625/RFfusion)  
**领域**: image_generation  
**关键词**: Image Fusion, Rectified Flow, diffusion model, VAE, One-Step Sampling

## 一句话总结

提出 RFfusion，首次将 Rectified Flow 引入图像融合任务，实现无需额外训练的单步采样，同时设计面向融合的两阶段 VAE 训练策略，在速度和质量上全面超越现有扩散融合方法。

## 背景与动机

图像融合旨在将不同模态的图像（如红外/可见光、多曝光、多聚焦）合并为包含互补信息的单张图像，广泛应用于目标检测、医学诊断等场景。近年来扩散模型（DDPM）凭借强大的生成先验在图像融合中取得了显著进展，如 DDFM 和 CCF 等方法通过后验采样将融合先验注入扩散过程，有效提升了融合质量。

然而，扩散方法的核心瓶颈在于**推理效率极低**：DDFM 需要上百步采样，单张图像耗时约 22 秒；CCF 更是需要 62 秒。减少采样步数又会导致融合质量的显著下降。现有的蒸馏加速方法缺乏跨任务泛化能力，而基于 VAE 的潜空间扩散方法由于训练目标（重建）与融合目标之间存在本质差异，直接应用效果不佳。因此，亟需一种既能保持融合质量又具有泛化能力的高效采样方法。

## 核心问题

1. **采样效率**：如何将扩散融合模型从上百步采样压缩到单步，且不需要额外训练？
2. **VAE 目标不匹配**：传统 VAE 以像素级重建为目标，而融合任务需要捕获跨模态互补语义信息，如何弥合这一差异？
3. **多任务泛化**：能否用同一套模型参数在红外可见光融合（IVIF）、多曝光融合（MEF）、多聚焦融合（MFF）等多个任务上都取得优异表现？

## 方法详解

### 整体框架

RFfusion 由两个核心组件构成：基于 Rectified Flow 的单步融合网络 + 面向融合任务的两阶段 VAE。

### 1. Rectified Flow 单步融合

Rectified Flow 将前向过程视为两个数据分布之间的线性变换，通过线性插值构建直线路径：

$$x_t = (1-t) x_0 + t \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)$$

训练目标是学习速度场 $v_\theta(x_t, t)$ 来预测从噪声到真实图像的最短路径。关键观察是：用可见光图像而非纯高斯噪声作为输入，能获得更好的融合效果。

借鉴 DDFM 的后验采样机制，将融合先验通过速度场注入采样过程：

$$v_\theta(f_t | i, v) = v_\theta(f_t) + \nabla_{f_t} \log p(i, v | f_t)$$

其中融合先验通过计算融合图像与输入图像之间的观测损失来引导生成。由于 Rectified Flow 基于 ODE 框架（无随机噪声注入），采样路径为确定性轨迹，天然适合单步推理。

### 2. 面向融合的 VAE 两阶段训练

**第一阶段 — 频率感知重建训练**：微调 VAE 编码器和解码器，不涉及融合过程。引入频域相似性损失 $\mathcal{L}_{fre}$，通过 FFT 变换将图像转换到频域，计算输入与重建图像频谱之间的差异。这一阶段的核心动机是：融合任务所关注的互补语义信息与图像的高低频分量高度相关，频率损失引导 VAE 在重建时关注与融合相关的语义信息。

**第二阶段 — 融合适配联合训练**：将 VAE 集成到整体融合框架中进行联合训练，仅微调解码器。使用融合专用损失函数：

$$\mathcal{L}_{fusion} = \lambda_{int}\mathcal{L}_{int} + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{grad}\mathcal{L}_{grad} + \lambda_{color}\mathcal{L}_{color} + \lambda_{mask}\mathcal{L}_{mask}$$

其中 $\mathcal{L}_{mask}$ 是新提出的显著性掩码损失，通过显著性权重图引导网络关注显著区域，增强融合图像中互补信息的保留。

## 实验关键数据

### 推理效率对比（V100 GPU，RoadScene 数据集）

| 方法 | 推理时间 (s) | 参数量 (M) | SF↑ | AG↑ |
|------|------------|-----------|------|------|
| DDFM | 22.03 | 552.81 | 9.689 | 3.981 |
| CCF | 62.47 | 552.81 | 10.14 | 3.882 |
| Diff-IF | 2.457 | 23.47 | 13.90 | 5.179 |
| **RFfusion** | **0.308** | **65.57** | **14.00** | **5.218** |

速度相比 DDFM 提升 **71.5 倍**，相比 CCF 提升 **202.8 倍**。

### 红外可见光融合（M3FD 数据集）

相比基线 DDFM，MI 提升 +0.449，VIF 提升 +0.071，SSIM 提升 +0.047。在 TNO+RoadScene 数据集上提升更为显著：MI +1.150，VIF +0.398，SSIM +0.714。

### 多任务泛化

使用**同一套 checkpoint** 在 IVIF、MEF、MFF 三个任务上评估，无需任务特定微调。在 MEFB 数据集上 MI 达到 6.528（第一），在 MFIF 数据集上 CC 达到 0.977（第一）。

### 消融实验

两阶段训练策略完整使用时 PSNR 从 59.41 提升到 61.81；$\mathcal{L}_{fre}$ 和 $\mathcal{L}_{mask}$ 联合使用时效果最优。

## 亮点

1. **首次将 Rectified Flow 引入图像融合**，无需额外训练即可实现单步采样，推理仅 0.308 秒
2. **两阶段 VAE 训练策略**设计精巧：第一阶段通过频率损失弥合重建与融合的目标差异，第二阶段通过联合训练增强解码器的融合适配能力
3. **极强的多任务泛化能力**：同一模型在 IVIF/MEF/MFF 三类任务上均达到 SOTA 水平
4. 参数量仅 65.57M，远小于 DDFM/CCF 的 552.81M

## 局限与展望

1. 仍依赖通用图像生成任务预训练的 Rectified Flow 模型，非为融合任务专门设计，可能限制融合质量的进一步提升
2. 第一阶段 VAE 训练仅在 LLVIP 和 MSRS 两个红外可见光数据集上进行，对其他模态的适配能力有待验证
3. 论文未讨论在高分辨率输入下的表现以及实际部署场景（如边缘设备）的可行性
4. 可探索面向融合任务专门训练的 Rectified Flow 模型，而非仅利用预训练权重

## 与相关工作的对比

| 维度 | DDFM | CCF | Diff-IF | RFfusion |
|------|------|-----|---------|----------|
| 采样步数 | ~1000 | ~1000 | 多步 | **1步** |
| 是否需要额外训练 | 否 | 否 | 是 | **否** |
| 潜空间操作 | 否 | 否 | 否 | **是** |
| 多任务泛化 | 有限 | 较好 | 有限 | **强** |
| 推理速度 | 慢 | 极慢 | 中等 | **快** |

核心区别在于：RFfusion 用 Rectified Flow 的直线采样路径替代 DDPM 的迂回路径，避免了多步采样；同时用面向融合的 VAE 将操作迁移到潜空间，进一步降低计算开销。

## 启发与关联

1. **Rectified Flow 作为通用加速工具**：该方法的核心思路（直线化采样路径 → 单步推理）可推广到其他基于扩散的低级视觉任务，如图像修复、超分辨率等
2. **目标不匹配的两阶段解法**：面对 VAE 重建目标与下游任务目标不一致的问题，两阶段渐进适配是一个通用且实用的策略
3. **频域损失在融合中的价值**：频率分量与跨模态互补信息的相关性为融合任务的损失设计提供了有意义的先验

## 评分

- 新颖性: ⭐⭐⭐⭐ — Rectified Flow 首次应用于融合，两阶段 VAE 设计有新意
- 实验充分度: ⭐⭐⭐⭐ — 三类融合任务全面评估，多基准数据集，充分的消融实验
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，公式推导详尽，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 速度提升两个数量级，实用性极强

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Balanced Conic Rectified Flow](balanced_conic_rectified_flow.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[CVPR 2026\] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion](../../CVPR2026/image_generation/careflow_cyclic_adaptive_rectified_flow_for_multimodal_fusion.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](../../ICML2025/image_generation/taming_rectified_flow_for_inversion_and_editing.md)

<!-- RELATED:END -->
