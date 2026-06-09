---
title: >-
  [论文解读] Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries
description: >-
  [CVPR 2025][3D视觉][深度估计] 提出一种基于新型图像块表示 Blurry-Edges 的深度估计方法，通过对散焦边界的平滑度建模，实现在极低光照（光子受限）条件下从一对不同散焦图像中鲁棒地估计物体深度，噪声鲁棒性比现有 DfD 方法高 4 倍以上。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "深度估计"
  - "散焦模糊"
  - "低光照"
  - "图像表示"
  - "深度从散焦"
---

# Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries

**会议**: CVPR 2025  
**arXiv**: [2503.23606](https://arxiv.org/abs/2503.23606)  
**代码**: [https://blurry-edges.qiguo.org/](https://blurry-edges.qiguo.org/)  
**领域**: 3D视觉  
**关键词**: 深度估计, 散焦模糊, 低光照, 图像表示, 深度从散焦

## 一句话总结

提出一种基于新型图像块表示 Blurry-Edges 的深度估计方法，通过对散焦边界的平滑度建模，实现在极低光照（光子受限）条件下从一对不同散焦图像中鲁棒地估计物体深度，噪声鲁棒性比现有 DfD 方法高 4 倍以上。

## 研究背景与动机

深度从散焦（Depth from Defocus, DfD）是一种无需主动光源的深度估计方法，具有单目紧凑的特点，适合 AR/VR、智能手机、微型机器人等空间受限的场景。然而 DfD 的核心依赖于精确估计图像的空间梯度（散焦程度的代理），这对图像噪声极度敏感。现有 DfD 方法通常假设低噪声输入（噪声标准差 ≤ 4 LSB），在暗光环境下表现很差。

本文的核心矛盾是：**DfD 需要精确的空间梯度信息，而光子受限场景下的强噪声会严重干扰梯度估计**。作者的切入角度是：不再直接估计全图的散焦程度，而是聚焦于散焦边界，设计一种参数化的图像块表示 Blurry-Edges 来显式建模边界位置、颜色和模糊程度，并通过闭式DfD方程从一对散焦图像的边界平滑度差异直接计算深度。

## 方法详解

### 整体框架

输入一对不同光学功率的散焦噪声图像 $I_+, I_-$，先将图像分成重叠的小块，通过CNN（局部阶段）独立预测每个块的Blurry-Edges表示，再通过Transformer Encoder（全局阶段）全局一致性优化，最后聚合生成全局边界图、颜色图和稀疏深度图。深度可通过后处理密化为稠密深度图。

### 关键设计

1. **Blurry-Edges 图像块表示**:
    - 功能：将图像块参数化为多层堆叠的带模糊边界楔形（wedge），每个楔形用顶点位置 $\mathbf{p}_i$、角度 $\boldsymbol{\theta}_i$、颜色 $\mathbf{c}_i$ 和边界平滑度 $\eta_i$ 描述
    - 核心思路：通过 alpha 合成渲染楔形堆叠的颜色图，使用误差函数 $\mathrm{erf}$ 建模边界的平滑过渡；每个楔形的 $\alpha$-map 为 $\alpha_i = \frac{1}{2}[1 + \mathrm{erf}(\frac{d_i}{\sqrt{2}\eta_i})]$
    - 设计动机：相比 Field-of-Junction（FoJ）仅能表示线、边、交叉等有限结构且不建模边界平滑度，Blurry-Edges 可表示多种边界结构和不同的模糊程度，为 DfD 提供直接可用的散焦线索

2. **闭式 DfD 深度方程**:
    - 功能：从一对散焦图像中对应边界的平滑度 $\eta_+, \eta_-$ 直接计算深度值
    - 核心思路：利用高斯 PSF 卷积模型，同一边界在不同光学功率下的平滑度差异仅由深度决定；通过消去纹理模糊参数 $\xi$，得到深度的闭式解 $z(\eta_+, \eta_-) = \frac{2\Sigma^2 s^2(\rho_- - \rho_+)}{\eta_+^2 - \eta_-^2 - \Sigma^2 s(\rho_+ - \rho_-)( s\rho_+ + s\rho_- - 2)}$
    - 设计动机：避免像素级的梯度计算，转而利用边界级别的参数化平滑度差异，大幅提升噪声鲁棒性

3. **局部-全局两阶段网络架构**:
    - 功能：先用 CNN 局部预测每块的 Blurry-Edges 参数，再用 Transformer Encoder 全局优化一致性
    - 核心思路：局部阶段独立处理每个块并通过岭回归求解颜色参数；全局阶段在所有块之间强制边界中心图、颜色图、颜色梯度图的一致性约束，同时确保散焦一致性（共享楔形位置和颜色，仅平滑度不同）
    - 设计动机：模块化设计实现独立训练；全局优化解决局部估计的不一致问题，类似于从 patch 级推理到全局推理的层次化策略

### 损失函数 / 训练策略

- 局部阶段损失 $\mathcal{L}_\text{local} = \sum_{i=1}^{3} \beta_i \mathbb{E}_{\mathbf{m}}(l_i)$：包含颜色误差、平滑度误差、边界定位误差三项
- 全局阶段损失 $\mathcal{L}_\text{global} = \sum_{i=1}^{7} \gamma_i \mathbb{E}_{I_\pm, \mathbf{m}}(g_i)$：包含颜色、边界位置、边界平滑度、深度的预测误差和邻域一致性共七项
- 两阶段独立训练：先训练局部CNN直到收敛，再固定局部阶段训练全局 Transformer
- 训练数据仅使用简单几何体（矩形、圆形、三角形），无需真实场景数据即可泛化到真实世界

## 实验关键数据

### 主实验

| 方法 | 类型 | 图像数 | $\delta 1$ ↑ | RMSE (cm) ↓ | AbsRel (cm) ↓ |
|------|------|--------|-------------|-------------|---------------|
| Focal Track | 稀疏 | 2 | 0.588 | 6.308 | 4.640 |
| Tang et al. | 稀疏 | 2 | 0.663 | 6.737 | 4.346 |
| **Ours (稀疏)** | 稀疏 | 2 | **0.720** | **5.281** | **3.295** |
| PhaseCam3D | 稠密 | 2 | 0.405 | 9.883 | 8.053 |
| DefocusNet | 稠密 | 5 | 0.657 | 6.092 | 4.548 |
| DFV-DFF | 稠密 | 5 | 0.518 | 8.298 | 6.707 |
| DEReD | 稠密 | 5 | 0.536 | 7.779 | 5.977 |
| **Ours-PP (稠密)** | 稠密 | 2 | **0.806** | **3.992** | **2.691** |

### 消融实验

| 配置 (Patch Size) | $\delta 1$ ↑ | RMSE (cm) ↓ | AbsRel (cm) ↓ |
|-------------------|-------------|-------------|---------------|
| $11 \times 11$ | 0.717 | 5.675 | 3.498 |
| $21 \times 21$ (最优) | **0.720** | **5.281** | **3.295** |
| $31 \times 31$ | 0.657 | 6.123 | 4.060 |

### 关键发现

- 本文方法在噪声标准差 18-19 LSB（对应极暗环境 ~80 lux）下仍能可靠估计深度，比之前方法能处理的最高噪声高 4 倍以上
- 仅用简单几何体训练即可泛化到真实世界复杂场景，无需微调
- Blurry-Edges 表示是多功能的：同时生成边界图、去噪颜色图和深度图
- 密化后处理（Ours-PP）仅用 2 张图像即超越使用 5 张图像的稠密方法

## 亮点与洞察

- **表示创新**：Blurry-Edges 是对 Field-of-Junction 的重要扩展，加入边界平滑度建模后直接获得了 DfD 的可用线索
- **闭式深度方程**：从边界平滑度直接计算深度，避免了像素级梯度的噪声敏感问题
- **极强的泛化能力**：简单几何体训练 → 真实场景推理，说明 Blurry-Edges 的参数化表示具有良好的先验归纳偏置
- 该方法证明了边界信息在噪声环境中比全局纹理信息更鲁棒

## 局限与展望

- 稀疏深度图仅沿边界估计，无纹理区域没有深度值
- 密化依赖后处理网络（U-Net），引入额外计算开销
- 楔形数量固定为 $l=2$，对复杂交叉结构可能不够
- 图像分辨率受限于 $147 \times 147$，对大分辨率图像需要分块处理

## 相关工作与启发

- Field-of-Junction（FoJ）系列工作启发了参数化图像块表示的思路，但 FoJ 缺少平滑度建模
- 解析 DfD 方法（Focal Track、Focal Flow）提供了低计算成本的深度估计框架
- 学习式 DfD（PhaseCam3D, DefocusNet）可产生稠密深度图但不够鲁棒
- 该方法的"先表示再计算"范式可推广到其他需要从噪声图像提取几何信息的任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Blurry-Edges 表示和闭式DfD方程均为原创贡献
- 实验充分度: ⭐⭐⭐⭐ 合成+真实实验完整，但真实实验规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 为低光照条件下的深度估计开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](blade_single-view_body_mesh_estimation_through_accurate_depth_estimation.md)

</div>

<!-- RELATED:END -->
