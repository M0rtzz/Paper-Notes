---
title: >-
  [论文解读] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction
description: >-
  [ICLR2026][3D视觉][Dynamic Gaussian Splatting] 提出 USplat4D，一种不确定性感知的动态高斯泼溅框架，通过估计每个高斯的时变不确定性并构建不确定性引导的时空图来传播可靠运动线索，显著提升了遮挡区域和极端新视角下的单目 4D 重建质量。
tags:
  - "ICLR2026"
  - "3D视觉"
  - "Dynamic Gaussian Splatting"
  - "uncertainty estimation"
  - "4D Reconstruction"
  - "Monocular"
  - "novel view synthesis"
---

# Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction

**会议**: ICLR2026  
**arXiv**: [2510.12768](https://arxiv.org/abs/2510.12768)  
**代码**: [tamu-visual-ai/usplat4d](https://tamu-visual-ai.github.io/usplat4d/)  
**领域**: 3D视觉  
**关键词**: Dynamic Gaussian Splatting, uncertainty estimation, 4D Reconstruction, Monocular, novel view synthesis

## 一句话总结

提出 USplat4D，一种不确定性感知的动态高斯泼溅框架，通过估计每个高斯的时变不确定性并构建不确定性引导的时空图来传播可靠运动线索，显著提升了遮挡区域和极端新视角下的单目 4D 重建质量。

## 背景与动机

从单目视频重建动态 3D 场景是 AR、机器人和人体运动分析等任务的基础问题，但由于遮挡和极端视角变化，该问题极具挑战性。

- **现有动态高斯泼溅方法的共性缺陷**：无论使用规范场（canonical field）、变形基（deformation bases）还是直接 4D 建模，现有方法都对所有高斯基元进行**均匀优化**，依赖深度、光流和光度一致性等 2D 监督信号。这种均匀处理忽略了一个关键事实：有些高斯被反复观测、约束充分，而另一些仅被少量观测、约束薄弱。
- **后果**：在遮挡情况下运动估计会漂移（motion drift），在极端新视角下合成质量严重退化。例如，旋转的背包在不同时刻总有一部分表面被自身遮挡，但人类仍能通过记忆和时间连续性推断出遮挡区域的外观和运动。
- **核心洞察**：当观测不完整时，重建应当以**高置信度的线索为锚点**，并通过结构化方式传播到不确定区域。置信度高的高斯应当被优先处理，并用于引导不可靠高斯的优化。

## 方法详解

### 整体框架：USplat4D

USplat4D 是一个与底层模型无关（model-agnostic）的不确定性感知框架，可集成到任何估计逐高斯运动的动态高斯泼溅方法中。整体流程分三步：

1. **动态不确定性估计**（Section 4.1）：为每个高斯在每帧估计时变不确定性分数
2. **不确定性编码的图构建**（Section 4.2）：根据不确定性将高斯划分为关键节点和非关键节点，构建时空图
3. **不确定性感知优化**（Section 4.3）：通过图结构传播可靠运动线索到不确定区域

### 关键设计 1：动态不确定性估计

**逐高斯标量不确定性**

对于光度重建损失 $\mathcal{L}_{2,t} = \sum_{h \in \Omega} \|\bar{C}_t^h - C_t^h\|_2^2$，通过对颜色参数 $c_i$ 求导并在局部最小值假设下，得到闭合形式的方差估计：

$$\sigma_{i,t}^2 = \left(\sum_{h \in \Omega_{i,t}} (T_{i,t}^h \alpha_i)^2 \right)^{-1}$$

其中 $T_{i,t}^h$ 为高斯 $i$ 在像素 $h$ 处的透射率，$\alpha_i$ 为不透明度。

为处理未收敛像素，引入逐像素收敛指示函数 $\mathbb{1}_t(h)$（颜色误差低于阈值 $\eta_c$ 时为 1）。最终标量不确定性为：

$$u_{i,t} = \mathbb{1}_{i,t} \cdot \sigma_{i,t}^2 + (1 - \mathbb{1}_{i,t}) \cdot \phi$$

直觉上，被充分观测的高斯获得低 $u_{i,t}$（高可靠性），不可靠的高斯获得高 $u_{i,t}$。

**从标量到深度感知不确定性**

单目设置中深度方向的不确定性远大于图像平面方向。标量不确定性隐含了各向同性假设，会导致沿相机轴方向过度自信，产生几何畸变。为此将图像空间误差传播到 3D，用各向异性不确定性矩阵表示：

$$\mathbf{U}_{i,t} = \mathbf{R}_{wc} \cdot \text{diag}(r_x u_{i,t}, r_y u_{i,t}, r_z u_{i,t}) \cdot \mathbf{R}_{wc}^\mathsf{T}$$

其中 $\mathbf{R}_{wc}$ 是相机到世界的旋转矩阵，$r_x, r_y, r_z$ 为轴对齐缩放因子（深度方向 $r_z$ 通常更大），将 2D 不确定性转化为考虑相机姿态和深度敏感性的 3D 不确定性。

### 关键设计 2：不确定性编码的图构建

**节点划分**：将高斯划分为少量关键节点 $\mathcal{V}_k$（低不确定性，提供运动锚点）和大量非关键节点 $\mathcal{V}_n$（从邻近关键节点继承运动）。

**关键节点选择（两阶段策略）**：

1. **3D 体素网格采样**：每帧将场景划分为 3D 体素网格，丢弃仅含高不确定性高斯的体素，在剩余体素中每个体素随机选取一个低不确定性高斯，确保空间覆盖均匀性
2. **显著周期阈值过滤**：计算每个候选高斯的"显著周期"（不确定性低于阈值的帧数），仅保留显著周期 ≥5 帧的候选者，确保时间支持充分

关键/非关键比例维持约 1:49（选取最置信的 top 2%），消融实验表明 0.5%~4% 范围内性能稳定。

**边构建**：

- **关键节点间**：采用 Uncertainty-Aware kNN (UA-kNN)，在节点最可靠帧 $\hat{t} = \arg\min_t \{u_{i,t}\}$ 处，使用 Mahalanobis 距离度量来选择邻居，确保连接的是空间接近且可靠的节点
- **非关键节点**：将其关联到整个序列中距离最近的关键节点，并继承该关键节点的邻居结构

### 关键设计 3：不确定性感知优化

**关键节点损失**：鼓励关键节点保持在预训练位置附近，使用不确定性矩阵的逆 $\mathbf{U}_{w,t,i}^{-1}$ 进行加权，确保主要沿可靠方向进行运动校正：

$$\mathcal{L}^{\text{key}} = \sum_t \sum_{i \in \mathcal{V}_k} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t}^o\|_{\mathbf{U}_{w,t,i}^{-1}} + \mathcal{L}^{\text{motion,key}}$$

**非关键节点损失**：通过双四元数混合（Dual Quaternion Blending, DQB）从邻近关键节点插值运动，同时约束非关键节点接近预训练状态和插值轨迹：

$$\mathcal{L}^{\text{non-key}} = \sum_t \sum_{i \in \mathcal{V}_n} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t}^o\|_{\mathbf{U}_{w,i}^{-1}} + \sum_t \sum_{i \in \mathcal{V}_n} \|\mathbf{p}_{i,t} - \mathbf{p}_{i,t}^{\text{DQB}}\|_{\mathbf{U}_{w,i}^{-1}} + \mathcal{L}^{\text{motion,non-key}}$$

**总损失**：$\mathcal{L}^{\text{total}} = \mathcal{L}^{\text{rgb}} + \mathcal{L}^{\text{key}} + \mathcal{L}^{\text{non-key}}$

### 训练策略

- USplat4D 采用**两阶段训练**：先用基线模型（如 SoM 或 MoSca）预训练动态高斯场，再用 USplat4D 的不确定性感知优化进行细化
- 框架是 model-agnostic 的，可插入任何估计逐高斯运动的基线方法
- 运动正则化包含等距约束、刚性约束、相对旋转约束、速度约束和加速度约束

## 实验关键数据

### DyCheck 数据集上的定量结果

| 设置 | 方法 | mPSNR↑ | mSSIM↑ | mLPIPS↓ |
|------|------|--------|--------|---------|
| 5 scenes, 1× | SC-GS | 14.13 | 0.477 | 0.49 |
| 5 scenes, 1× | Deformable 3DGS | 11.92 | 0.490 | 0.66 |
| 5 scenes, 1× | 4DGS | 13.42 | 0.490 | 0.56 |
| 5 scenes, 1× | MoDec-GS | 15.01 | 0.493 | 0.44 |
| 5 scenes, 1× | MoBlender | 16.79 | 0.650 | 0.37 |
| 5 scenes, 1× | SoM | 16.72 | 0.630 | 0.45 |
| 5 scenes, 1× | **USplat4D** | **16.85** | **0.650** | **0.38** |
| 7 scenes, 2× | Dynamic Gaussians | 7.29 | – | 0.69 |
| 7 scenes, 2× | 4DGS | 13.64 | – | 0.43 |
| 7 scenes, 2× | Gaussian Marbles | 16.72 | – | 0.41 |
| 7 scenes, 2× | MoSca | 19.32 | 0.706 | 0.26 |
| 7 scenes, 2× | **USplat4D** | **19.63** | **0.716** | **0.25** |

### Objaverse 数据集极端新视角合成结果

| 方法 | 视角范围 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|----------|-------|-------|--------|
| SoM | (0°, 60°] | 16.09 | 0.860 | 0.31 |
| USplat4D (SoM) | (0°, 60°] | **16.63** | **0.866** | **0.27** |
| SoM | (60°, 120°] | 15.58 | 0.854 | 0.32 |
| USplat4D (SoM) | (60°, 120°] | **16.57** | **0.868** | **0.27** |
| SoM | (120°, 180°] | 16.45 | 0.858 | 0.31 |
| USplat4D (SoM) | (120°, 180°] | **17.03** | **0.872** | **0.26** |
| MoSca | (0°, 60°] | 16.18 | 0.881 | 0.24 |
| USplat4D (MoSca) | (0°, 60°] | **16.22** | **0.885** | **0.22** |
| MoSca | (120°, 180°] | 15.89 | 0.876 | 0.25 |
| USplat4D (MoSca) | (120°, 180°] | **16.31** | **0.886** | **0.21** |

极端视角（120°–180°）下增益最为显著，SoM 基线上 PSNR 提升 +0.58 dB，LPIPS 改善 0.05。

### 消融实验

| 消融设置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------|-------|-------|--------|
| USplat4D (完整模型) | **19.63** | **0.716** | **0.25** |
| (a) 去掉关键节点不确定性 | 18.86 | 0.688 | 0.28 |
| (b) 去掉 UA-kNN | 19.50 | 0.711 | 0.26 |
| (c) 去掉损失加权 | 19.08 | 0.681 | 0.25 |
| (d) 去掉 3D 网格化 | 19.50 | 0.712 | 0.25 |

- **(a) 去掉不确定性引导的关键节点选择**影响最大：PSNR 下降 0.77 dB，说明不确定性是锚点选择的关键
- **(c) 去掉损失中的不确定性加权**：SSIM 下降 0.035，不可靠高斯被与可靠高斯同等强度更新导致漂移

## 亮点与洞察

1. **核心思想简洁有力**：将不确定性从辅助信号提升为框架中心，通过"高置信锚定 + 结构化传播"的范式处理遮挡和极端视角问题，具有强直觉解释力
2. **模型无关的即插即用设计**：USplat4D 可无缝集成到 SoM、MoSca 等不同基线上并稳定带来增益，体现了良好的通用性
3. **深度感知各向异性不确定性**：从标量不确定性扩展到考虑相机姿态的 3D 各向异性矩阵，有效缓解了单目重建中深度方向过度自信的问题
4. **图的自然分割能力**：关键节点图的权重矩阵经重排序后近似块对角矩阵，天然支持多物体运动分割，无需额外监督
5. **不确定性的三重角色**：在关键节点偏差加权、非关键节点插值引导、总损失平衡三个层面统一发挥作用

## 局限性

1. **依赖预训练基线质量**：USplat4D 在预训练模型基础上做细化，若基线模型初始化质量差（如严重的初始运动错误），细化效果受限
2. **视觉基础模型的计算开销和误差**：框架仍受底层视觉基础模型（深度估计、光流等）的计算开销和固有误差影响
3. **近视角增益有限**：在接近输入视角的验证视图上，USplat4D 相比强基线（如 MoBlender、SoM）提升较小（PSNR 仅 +0.13 dB），优势主要体现在极端视角
4. **超参数敏感性**：关键节点比例（2%）、显著周期阈值（5 帧）、颜色收敛阈值 $\eta_c$ 等超参需针对场景调优
5. **缺乏无纹理/快速运动场景的深入分析**：对于纹理稀疏区域和极快运动场景，不确定性估计本身可能失效

## 相关工作

| 方向 | 代表方法 | 与 USplat4D 的差异 |
|------|----------|-------------------|
| 动态高斯泼溅 (运动基) | SoM, MoSca, Marbles, 4D-Rotor | 使用低秩运动基正则化变形，但不区分高斯可靠性，遮挡下运动漂移 |
| 动态高斯泼溅 (规范场) | Deformable 3DGS, SC-GS | 通过规范空间建模运动，同样缺乏不确定性感知 |
| 场景重建中的不确定性 | SE-GS, Kim et al. (2024) | SE-GS 用于静态场景的自集成不确定性；Kim et al. 将不确定性作为辅助信号平滑运动或重加权梯度，但未整合进图结构化传播 |
| 图基运动建模 | MoSca (lifting graph), SC-GS (局部 kNN) | 使用固定距离度量构图，不考虑节点可靠性 |

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 将不确定性从辅助信号提升为统一的图构建-优化框架核心，思路新颖 |
| 技术深度 | 4 | 从标量到各向异性不确定性的推导严谨，图构建和优化设计完整 |
| 实验充分性 | 4 | 覆盖 DyCheck、DAVIS、Objaverse 三个数据集，消融全面，但定量分析主要集中在验证视图 |
| 写作质量 | 4 | 动机清晰，公式推导清楚，图示丰富 |
| 实用价值 | 4 | 模型无关的即插即用设计，实用性强，可直接增强现有方法 |
| 综合 | 4.0 | 高质量的方法论贡献，在单目 4D 重建中引入结构化不确定性建模，极端视角增益显著 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos](mono4dgs-hdr_high_dynamic_range_4d_gaussian_splatting_from_alternating-exposure_.md)
- [\[AAAI 2026\] Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [\[ICLR 2026\] MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting](moe-gs_mixture_of_experts_for_dynamic_gaussian_splatting.md)
- [\[CVPR 2026\] AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](../../CVPR2026/3d_vision/aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)

</div>

<!-- RELATED:END -->
