---
title: >-
  [论文解读] DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][3D Gaussian Splatting] 提出基于方向一致性（Directional Consistency）的自适应密度控制方法 DC4GS，通过利用位置梯度的角度相干性来改进 3DGS 中的 primitive 分裂决策和分裂位置选择…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "自适应密度控制"
  - "方向一致性"
  - "原语分裂"
  - "场景重建"
---

# DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2510.26921](https://arxiv.org/abs/2510.26921)  
**代码**: 有（Project Page）  
**领域**: 3D Vision / 神经渲染  
**关键词**: 3D Gaussian Splatting, 自适应密度控制, 方向一致性, 原语分裂, 场景重建

## 一句话总结

提出基于方向一致性（Directional Consistency）的自适应密度控制方法 DC4GS，通过利用位置梯度的角度相干性来改进 3DGS 中的 primitive 分裂决策和分裂位置选择，在减少最多 30% primitive 数量的同时提升重建质量。

## 研究背景与动机

3D Gaussian Splatting (3DGS) 通过一组 3D 高斯原语（primitives）实现高效实时的新视角合成。其核心组件之一是自适应密度控制（ADC），负责在训练过程中对原语进行分裂（split）、克隆（clone）和剔除（prune），以适应场景的复杂结构。

**现有 ADC 的关键问题：**

**仅依赖梯度幅度**：传统 ADC 基于位置梯度的幅度来决定是否分裂，但梯度幅度大并不总意味着需要分裂——可能是多个一致方向的梯度在推动原语移动，而非需要更多原语

**随机分裂位置**：当决定分裂时，子原语的放置位置是沿原原语的协方差主轴随机偏移，这种随机性导致子原语可能无法很好地对齐局部结构

**冗余分裂**：过度分裂导致原语数量膨胀，增加存储和渲染开销

这些问题的根源在于：梯度幅度只反映了"需要多少变化"，却没有反映"变化的方向是否一致"。

## 方法详解

### 整体框架

DC4GS 的核心思想是在 ADC 的两个关键步骤中引入**方向一致性（DC）**度量：

1. **分裂判定**：结合梯度幅度和方向一致性共同决定是否分裂
2. **分裂位置**：利用方向一致性信息确定子原语的最优放置位置

### 关键设计

**1. 方向一致性度量**

对于每个高斯原语，在训练过程中累积其位置梯度。方向一致性通过梯度的**角度相干性**来衡量：

$$DC = \frac{\|\sum_{i=1}^{N} \mathbf{g}_i\|}{\sum_{i=1}^{N} \|\mathbf{g}_i\|}$$

其中 $\mathbf{g}_i$ 是第 $i$ 个训练步的位置梯度。DC 值范围在 [0, 1]：
- **DC ≈ 1**：梯度方向高度一致，原语需要平移而非分裂
- **DC ≈ 0**：梯度方向分散，说明局部结构复杂，需要分裂以提供更多表示能力

**2. 基于 DC 的分裂判定**

传统 ADC 的分裂条件：$\|\bar{\mathbf{g}}\| > \tau$（平均梯度幅度超过阈值）

DC4GS 的分裂条件：$\|\bar{\mathbf{g}}\| > \tau$ **且** $DC < \tau_{DC}$

即只有当梯度幅度大**且**方向不一致时才执行分裂。如果梯度方向高度一致（DC 高），即使幅度大，也仅执行平移或克隆，避免冗余分裂。

**3. 基于 DC 的最优分裂位置**

当需要分裂时，DC4GS 不再随机放置子原语，而是利用梯度方向信息确定最优位置：

- 分析累积梯度的方向分布，识别出梯度指向的主要方向簇
- 将子原语放置在这些方向簇对应的位置上
- 确保子原语能最佳地对齐局部结构，而非随机偏移

这种数据驱动的分裂位置选择使子原语从一开始就处于更好的位置，减少后续优化所需的迭代次数。

### 损失函数 / 训练策略

DC4GS 不改变 3DGS 的基础渲染和优化损失（L1 + D-SSIM），仅修改 ADC 策略：
- 在每次 densification 步骤中计算每个原语的 DC 值
- 使用 DC 阈值 $\tau_{DC}$ 过滤掉方向一致的原语（不分裂）
- 对需要分裂的原语，基于梯度方向信息计算最优子原语位置

该方法可以无缝集成到现有 3DGS 训练pipeline中，无需额外的训练过程。

## 实验关键数据

### 主实验

**Table 1：Mip-NeRF 360 数据集定量对比**

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | #Primitives ↓ |
|------|--------|--------|---------|---------------|
| 3DGS (原版) | 27.21 | 0.815 | 0.214 | 3.26M |
| Mini-Splatting | 27.40 | 0.820 | 0.210 | 2.80M |
| AbsGS | 27.35 | 0.818 | 0.212 | 3.10M |
| DC4GS (Ours) | **27.58** | **0.823** | **0.205** | **2.28M** |

DC4GS 使用仅约 70% 的原语数量，在所有渲染质量指标上均优于原版 3DGS 和其他密度控制方法。

**Table 2：Tanks and Temples / Deep Blending 数据集**

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 原语减少比 |
|------|--------|--------|---------|-----------|
| 3DGS | 23.14 | 0.841 | 0.183 | - |
| Compact-3DGS | 23.25 | 0.843 | 0.180 | ~15% |
| DC4GS (Ours) | **23.52** | **0.849** | **0.173** | **~30%** |

在 Tanks and Temples 等室外大场景中，DC4GS 原语减少效果更为显著（最高达 30%），同时渲染质量也有提升。

### 消融实验

**DC 判定 vs. 分裂位置的贡献分析**

| 配置 | PSNR ↑ | #Primitives ↓ |
|------|--------|---------------|
| Baseline (3DGS) | 27.21 | 3.26M |
| + DC 分裂判定 | 27.38 | 2.65M |
| + DC 分裂位置 | 27.33 | 3.10M |
| + DC 判定 + 位置 (DC4GS) | **27.58** | **2.28M** |

两个组件都有独立贡献，DC 分裂判定主要减少冗余原语，DC 分裂位置主要提升重建质量。两者组合效果最佳。

### 关键发现

1. **梯度方向比幅度更重要**：传统 ADC 仅看梯度大小，DC4GS 表明方向一致性是更关键的信号——方向一致意味着原语应移动而非分裂
2. **原语数量大幅减少**：在保持甚至提高渲染质量的前提下，原语最多减少30%，直接降低了存储和推理开销
3. **即插即用**：DC4GS 可直接替换 3DGS 的 ADC 组件，无需改动网络结构或损失函数
4. **复杂场景收益更大**：室外场景和几何细节丰富的场景中，DC 信号的区分度更高

## 亮点与洞察

- **简洁有效的观察**：梯度方向一致性这一简单的统计量包含了丰富的局部结构信息，是一个被忽视但极有价值的信号
- **减少不改质量**：与压缩方法不同，DC4GS 通过更智能的分裂策略从源头减少冗余，而非训练后剪枝
- **理论直觉清晰**：DC ≈ 1 说明多个视角对该原语的优化方向一致（应移动），DC ≈ 0 说明存在冲突（应分裂为多个子原语来分别拟合）
- **通用性**：方向一致性的思想不仅适用于 3DGS，也可推广到其他需要自适应密度控制的表示方法

## 局限与展望

1. **DC 阈值敏感性**：$\tau_{DC}$ 的选择可能因场景而异，自适应阈值选择策略有待探索
2. **与其他加速方法兼容性**：是否能与蒸馏、量化等方法叠加使用尚需验证
3. **动态场景**：当前方法针对静态场景，动态 3DGS 中的 ADC 改进是有价值的扩展方向
4. **极端场景**：对于非常稀疏或非常密集的区域，DC 信号的可靠性可能下降

## 相关工作与启发

- **3DGS (Kerbl et al. 2023)**：原始的 3D Gaussian Splatting 框架
- **Compact-3DGS**：通过后处理压缩减少原语
- **Mini-Splatting**：限制原语数量的训练策略
- **AbsGS**：基于绝对梯度的密度控制改进
- **NeRF 系列**：隐式表示的替代范式

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 — 方向一致性驱动 ADC 是简洁而有效的新视角 |
| 技术质量 | 4 — 方法清晰合理，实验全面 |
| 实验充分性 | 4 — 多数据集验证 + 消融分析 |
| 写作质量 | 4 — 动机和方法阐述清晰 |
| 影响力 | 4 — 直接改进 3DGS 核心组件，实用价值高 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Steepest Descent Density Control for Compact 3D Gaussian Splatting](../../CVPR2025/3d_vision/steepest_descent_density_control_for_compact_3d_gaussian_splatting.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](../../ECCV2024/3d_vision/pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [\[ICCV 2025\] StealthAttack: Robust 3D Gaussian Splatting Poisoning via Density-Guided Illusions](../../ICCV2025/3d_vision/stealthattack_robust_3d_gaussian_splatting_poisoning_via_density-guided_illusion.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Ref-GS: Directional Factorization for 2D Gaussian Splatting](../../CVPR2025/3d_vision/ref-gs_directional_factorization_for_2d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
