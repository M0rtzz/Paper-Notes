---
title: >-
  [论文解读] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images
description: >-
  [3D视觉] 提出 PRO（Patch Refine Once），通过分组块一致性训练（GPCT）和无偏遮罩（BFM）策略，在高分辨率图像上实现无缝的逐块深度精炼，仅需每块单次精炼即可消除边界伪影，推理速度比 PatchRefiner 快12倍。
tags:
  - 3D视觉
---

# One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.22351](https://arxiv.org/abs/2503.22351)
- **代码**: [https://kaist-viclab.github.io/One-Look-is-Enough_site](https://kaist-viclab.github.io/One-Look-is-Enough_site)
- **领域**: 3D视觉 / 深度估计
- **关键词**: High-Resolution Depth Estimation, Zero-Shot, Patch-Based Refinement, Depth Consistency, DepthAnything

## 一句话总结
提出 PRO（Patch Refine Once），通过分组块一致性训练（GPCT）和无偏遮罩（BFM）策略，在高分辨率图像上实现无缝的逐块深度精炼，仅需每块单次精炼即可消除边界伪影，推理速度比 PatchRefiner 快12倍。

## 研究背景与动机

零样本单目深度估计模型（如 DepthAnything V2）虽然泛化能力强，但在高分辨率图像上面临困境：

**分辨率失配问题**：
- 模型训练于低分辨率（384×384 或 518×518）
- 全分辨率推理：深度精度下降 + 显存爆炸
- 下采样推理：边缘细节模糊

**现有逐块方法的缺陷**：

**深度不连续问题**：块内深度连续但块间产生明显边界伪影（grid artifacts）

**测试时效率低**：PatchRefiner 需要177个块进行测试集成来缓解不连续，导致推理时间暴涨至16秒

**合成数据偏差**：训练所需的高分辨率密集深度通常来自合成数据集（如 UnrealStereo4K），其中透明物体的深度标注错误（标注为背景深度），导致训练偏差

核心动机：**能否训练一个模型，仅需每块单次精炼（One Look）就能消除边界伪影？**

## 方法详解

### 整体框架

PRO 基于残差预测架构：
1. 将下采样图像输入预训练 MDE 模型 $\Psi$ 得到粗深度 $\mathbf{D}_c$
2. 将原图切块，每块也输入 $\Psi$ 得到细深度 $\mathbf{D}_f^i$
3. 残差预测网络融合全局和局部特征，预测精炼深度：
$$\mathbf{D}_{refine}^i = \theta(\text{concat}(\mathbf{P}^i, \text{ROI}(\mathbf{D}_c, \mathbf{P}^i), \mathbf{D}_f^i))$$

### 分组块一致性训练（GPCT）

GPCT 是解决深度不连续问题的核心策略：

**训练时**：将图像裁剪为4个重叠的块（A, B, C, D），每块独立精炼后，在重叠区域施加一致性损失：

$$\mathcal{L}_{con} = \sum_{i \neq j} \frac{1}{|\Omega|} \sum_{p \in \Omega} (\mathbf{D}_{refine}^i(p) - \mathbf{D}_{refine}^j(p))^2$$

**与先前方法的关键区别**：
- PatchFusion 仅对2个对角相邻块（如 A-D 或 B-C）施加一致性约束
- GPCT 同时处理4个块，**每次反向传播都强制所有边界的一致性**，提供更全面的监督信号

**测试时**：融合深度通过简单平均：
$$\mathbf{D}_{merged}(p) = \frac{1}{N_o(p)} \sum_{i} \mathbf{D}_{refine}^i(p)$$

无需额外的测试集成步骤。

### 无偏遮罩（BFM）

BFM 解决合成数据集的标注偏差问题：

**问题来源**：UnrealStereo4K 中透明物体（如窗户）的深度被错误标注为其后方背景的深度。

**核心思路**：利用预训练 MDE 模型的先验知识识别不可靠区域。

1. 计算不可靠区域掩码：
$$\mathbf{M}_{unreliable} = \left[\max\left(\frac{N(\mathbf{D}_c)}{N(\mathbf{D}_{gt})}, \frac{N(\mathbf{D}_{gt})}{N(\mathbf{D}_c)}\right) > \tau\right]$$

其中 $N(\cdot)$ 是 min-max 归一化，$\tau=2$。

2. 提取边缘掩码（取粗深度和GT深度边缘的交集，确保保留关键边缘信息）：
$$\mathbf{M}_{edge} = \mathbf{E}_c \cap \mathbf{E}_{gt}$$

3. 最终可靠掩码：
$$\mathbf{M}_{BFM} = \mathbf{M}_{edge} \cup \sim\mathbf{M}_{unreliable}$$

仅在可靠区域上计算损失，避免模型学习到透明物体的错误深度模式。

### 总损失函数

$$\mathcal{L}_{final} = \mathcal{L}_{masked} + \lambda \mathcal{L}_{con}, \quad \lambda = 4$$

$\mathcal{L}_{masked}$ 结合 L1 + L2 + 多尺度梯度损失（4个尺度），比例为 1:1:5。

## 实验

### 主实验：零样本高分辨率深度估计

| 方法 | 时间(s) | Booster AbsRel↓ | ETH3D AbsRel↓ | Middle14 AbsRel↓ | NuScenes AbsRel↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| DepthAnythingV2 | - | 0.0307 | 0.0465 | 0.0307 | 0.106 |
| BoostingDepth | 12.7 | 0.0330 | 0.0552 | 0.0330 | 0.115 |
| PatchFusion P=177 | 36.8 | 0.0496 | 0.0723 | 0.0448 | 0.139 |
| PatchRefiner P=16 | 1.5 | 0.0348 | 0.0435 | 0.0292 | 0.107 |
| PatchRefiner P=177 | 16.2 | 0.0336 | 0.0430 | 0.0292 | 0.106 |
| **PRO (Ours)** | **1.4** | **0.0304** | **0.0422** | **0.0287** | **0.104** |

PRO 在所有指标上达到 SOTA，且推理速度最快（1.4秒 vs PatchRefiner-177 的 16.2秒，快12倍）。

### 一致性误差对比

| 方法 | 一致性误差 CE ↓ |
|:---|:---:|
| PatchFusion | 0.364 |
| PatchRefiner | 0.347 |
| **PRO** | **0.049** |

CE 降低 85.9%，证明 GPCT 显著消除了块间不连续问题。

### 消融实验

| 模型 | GPCT | BFM | Booster AbsRel↓ | CE ↓ |
|:---|:---:|:---:|:---:|:---:|
| (a) Baseline | | | 0.0385 | 0.208 |
| (b) +BFM | | ✓ | 0.0303 | 0.117 |
| (c) +GPCT | ✓ | | 0.0313 | 0.058 |
| (d) +Both | ✓ | ✓ | **0.0304** | **0.049** |

关键发现：
- BFM 在 Booster 数据集上 AbsRel 降低 21.3%（Booster 含透明物体）
- GPCT 让 CE 降低 50.4%
- 两者互补，联合使用效果最佳

### 重叠大小消融

| 重叠(px) | CE ↓ | ETH3D AbsRel↓ |
|:---:|:---:|:---:|
| 28 | 0.108 | 0.0423 |
| 112 | 0.065 | 0.0425 |
| 224 | **0.049** | **0.0422** |
| 448 | 0.060 | 0.0423 |

224像素重叠为最优值；过大重叠反而因块间内容过于相似而削弱一致性约束的效果。

## 亮点与洞察
1. **"One Look"理念**：训练时保证一致性，测试时无需集成——简洁高效的解决方案
2. **BFM 的泛化启示**：利用预训练模型的先验知识来过滤训练数据中的标注噪声，思路通用
3. **同时处理4块的 GPCT**：比仅处理对角2块的约束明显更强，CE 改善幅度大
4. **即插即用**：PRO 可无缝集成到不同基础深度估计模型中

## 局限性
- 边界回忆率（BR）指标略低于 BoostingDepth 和 PatchFusion，说明在边缘锐度上还有提升空间
- BFM 的阈值 $\tau$ 需要经验设定
- 透明物体的深度处理仍是回避而非解决（通过遮罩跳过）

## 相关工作
- 零样本深度估计：MiDaS, DepthAnything, Marigold, GeoWizard
- 高分辨率深度估计：BoostingDepth, PatchFusion, PatchRefiner
- 训练数据：UnrealStereo4K, ETH3D, Middlebury

## 评分
- **新颖性**: ⭐⭐⭐⭐ — GPCT 和 BFM 的设计巧妙，从训练策略层面解决测试难题
- **技术深度**: ⭐⭐⭐⭐ — 问题分析透彻，每个组件都有清晰的动机和验证
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个数据集零样本评估 + 全面消融
- **实用价值**: ⭐⭐⭐⭐⭐ — 12倍加速且质量更优，直接可用于部署

<!-- RELATED:START -->

## 相关论文

- [Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](../../CVPR2025/3d_vision/multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)
- [Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)

<!-- RELATED:END -->
