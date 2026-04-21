---
title: >-
  [论文解读] Volumetrically Consistent 3D Gaussian Rasterization
description: >-
  [CVPR 2025][3D视觉][3D高斯渲染] 本文指出 3DGS 的 splatting 渲染中存在不必要的物理近似，提出在光栅化框架内直接解析积分 3D 高斯的透射率来计算更精确的 alpha 值，既保持了光栅化的速度优势，又达到了接近光线追踪的物理精度。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯渲染
  - 体积一致性
  - 解析透射率
  - 视图合成
  - 计算机断层扫描
---

# Volumetrically Consistent 3D Gaussian Rasterization

**会议**: CVPR 2025  
**arXiv**: [2412.03378](https://arxiv.org/abs/2412.03378)  
**代码**: https://github.com/chinmay0301ucsd/Vol3DGS  
**领域**: 3D视觉  
**关键词**: 3D高斯渲染, 体积一致性, 解析透射率, 视图合成, 计算机断层扫描

## 一句话总结

本文指出 3DGS 的 splatting 渲染中存在不必要的物理近似，提出在光栅化框架内直接解析积分 3D 高斯的透射率来计算更精确的 alpha 值，既保持了光栅化的速度优势，又达到了接近光线追踪的物理精度。

## 研究背景与动机

**领域现状**：3DGS 通过将 3D 高斯投影（splatting）到屏幕空间的 2D 高斯来实现快速渲染，在视图合成任务上取得了卓越效果。光线追踪方法（如 NeRF）物理上更精确但速度慢，而 3DGS 的光栅化方法快但牺牲了物理精确性。

**现有痛点**：3DGS 的 EWA splatting 引入了三个额外近似：（1）指数透射率的线性化（$e^{-x} \approx 1-x$）；（2）假设高斯不自遮挡（no self-occlusion）；（3）协方差矩阵的仿射近似（Jacobian 近似）。这些近似导致 3DGS 难以表达不透明表面——splatting 后的 2D 高斯只有在中心点才能达到 $\alpha=1$，其他区域始终半透明。

**核心矛盾**：光栅化的速度优势和体积渲染的物理精度之间看似不可调和——但作者指出 splatting 并非光栅化的必要步骤，光栅化本身可以直接进行体积积分。

**本文目标**：在保持光栅化速度的同时，消除 splatting 引入的物理近似。

**切入角度**：利用 3D 高斯沿光线的积分可以解析求解的数学性质。

**核心 idea**：直接在 3D 空间对每个高斯沿光线方向解析积分其密度来计算透射率 $\alpha_i = 1 - \exp(-\int \kappa_i G_i(\mathbf{r}(t)) dt)$，替换 3DGS 中 splatting 得到的近似 alpha 值，作为 drop-in replacement 直接嵌入 3DGS 框架。

## 方法详解

### 整体框架

输入为 3D 高斯场景表示（密度 $\kappa_i$、均值 $\boldsymbol{\mu}_i$、协方差 $\boldsymbol{\Sigma}_i$、颜色 $c_i$），与 3DGS 共享相同的 alpha blending 框架 $C(p) = \sum_i c_i \alpha_i \prod_{j<i}(1-\alpha_j)$，但用解析计算的体积一致 alpha 值替换 3DGS 的 splatting alpha。整个流程仍在光栅化器中完成。

### 关键设计

1. **无 splatting 的 Alpha Blending 推导**:

    - 功能：证明体积渲染方程可以在不使用 splatting 的前提下表达为 alpha blending
    - 核心思路：将密度场 $\sigma(\mathbf{x}) = \sum_i \kappa_i G_i(\mathbf{x})$ 代入体积渲染方程，假设高斯不重叠且前后排序，将每个高斯的贡献分离。利用指数透射率的微分性质 $dT(t_{in}, t) = -T(t_{in}, t) \sigma(\mathbf{r}(t))$，证明内层积分恰好等于 $\alpha_i = 1 - \bar{T}_i$。最终得到与 3DGS 形式完全相同的 alpha blending 方程，但 $\alpha_i$ 的定义不同。
    - 设计动机：表明差异仅在于 alpha 值的计算方式，因此只需"换掉 alpha 计算"就能将体积一致性引入任何 3DGS 框架，实现最小改动的 drop-in 替换。

2. **解析透射率计算**:

    - 功能：精确计算每个 3D 高斯沿光线的累积不透明度
    - 核心思路：3D 高斯沿光线方向的积分可以化为 1D 高斯积分。先计算 3D 高斯在光线方向上的投影，得到 1D 参数：均值 $\gamma_j = (\boldsymbol{\mu}_j - \mathbf{o})^T \Sigma_j^{-1} \mathbf{d} / (\mathbf{d}^T \Sigma_j^{-1} \mathbf{d})$，方差 $\beta_j = 1/\sqrt{\mathbf{d}^T \Sigma_j^{-1} \mathbf{d}}$。然后对 1D 高斯积分得到误差函数，假设无限支撑后简化为 $\bar{T}_j = \exp(-\kappa_j G_j(\gamma_j \mathbf{d}) \sqrt{2\pi} \beta_j)$。
    - 设计动机：3DGS 的 splatting 丢弃了 z 方向的尺度信息（沿光线方向），使得不同 z-scale 的高斯渲染结果相同，物理上不合理。解析积分保留了 z 尺度的影响，$\beta_j$ 越大（z 方向越宽），alpha 越大。

3. **密度重参数化**:

    - 功能：解决高密度高斯梯度消失的优化问题
    - 核心思路：将密度 $\kappa$ 重参数化为 $\kappa = -\log(1 - 0.99\theta) \cdot \frac{1}{3}(\frac{1}{s_x} + \frac{1}{s_y} + \frac{1}{s_z})$，其中 $\theta \in [0,1]$。这使得小高斯自然获得高密度（更不透明），大高斯获得低密度，避免大高斯"一片模糊地遮挡一切"。同时修改了自适应密度化策略：分裂/克隆时减半密度，裁剪低密度点。
    - 设计动机：体积一致的 alpha 计算允许密度 $\kappa$ 趋向无穷来表达完全不透明表面，但这会导致梯度接近零。重参数化在保持表达能力的同时维持了梯度流。

### 损失函数 / 训练策略

使用与 3DGS 相同的损失函数（L1 + SSIM），基于 SLANG.D 光栅化器实现。禁用了 opacity reset（实验发现无帮助），增加了按高密度分裂和按低密度裁剪的策略。

## 实验关键数据

### 主实验

视图合成对比（Mip-NeRF360 / Tanks&Temples / DeepBlending）：

| 方法 | MN360 SSIM↑ | MN360 LPIPS↓ | T&T SSIM↑ | T&T LPIPS↓ | DB SSIM↑ | DB LPIPS↓ |
|---|---|---|---|---|---|---|
| 3DGS-Slang | 0.813 | 0.222 | 0.850 | 0.176 | 0.906 | 0.248 |
| GES | 0.794 | 0.250 | 0.836 | 0.198 | 0.901 | 0.252 |
| EVER | 0.825 | 0.194 | 0.870 | 0.160 | 0.908 | 0.308 |
| **Ours** | **0.813** | **0.209** | **0.854** | **0.167** | **0.908** | **0.247** |

### 消融实验

不透明物体拟合实验：

| 方法 | 分段常数纹理 LPIPS↓ | 说明 |
|---|---|---|
| 3DGS | 0.027 | 2D 高斯无法拟合锐利边界 |
| Ours | 0.005 | 高密度使高斯接近不透明，完美拟合 |

### 关键发现

- 在 SSIM 和 LPIPS 指标上一致匹配或超越 3DGS，尤其在 Tanks&Temples 上 LPIPS 从 0.176 降至 0.167
- 体积一致 alpha 使得同样数量的高斯可以更好地表达不透明表面，减少了边缘处的伪影和模糊
- 作为意外收获，方法可以直接用于断层扫描重建（CT），匹配了 SOTA 方法 R2-Gaussian 的质量但使用更少的点
- PSNR 指标上略低于 3DGS（27.30 vs 27.52 on MN360），但感知指标（SSIM/LPIPS）更优

## 亮点与洞察

1. **"drop-in replacement" 的优雅设计**：只需替换 alpha 计算即可获得体积一致性，与所有 3DGS 后续工作兼容，这种最小改动的设计理念值得借鉴
2. **splatting 近似的深度分析**：清晰地解构了 3DGS 中三层近似各自的影响，为理解 3DGS 的局限提供了理论基础
3. **从视图合成到 CT 的零迁移**：物理精确的渲染模型天然适用于需要积分密度的任务，展示了物理一致性的实用价值

## 局限与展望

- PSNR 指标上不如 3DGS（因为 PSNR 对高频细节不敏感，而模型偏好更紧凑的表示）
- 由于误差函数（erf）计算开销，FPS 略低于原版 3DGS（136 vs 159 on MN360）
- 仍保留了 per-tile 排序和不重叠假设，未完全解决体积渲染的所有近似
- 未来可以结合更完善的排序和重叠处理进一步提升物理精度

## 相关工作与启发

- **vs 3DGS**：3DGS 使用 splatting 近似，本文用解析积分替换，是 3DGS 渲染核心的直接改进
- **vs 2DGS**：2DGS 使用 2D 高斯消除仿射近似，但仍不计算精确透射率；本文方法更物理精确
- **vs EVER**：EVER 用光线追踪精确计算（包括排序和重叠），质量最高但速度慢；本文在光栅化中取得了接近的质量
- 启发：在不改变整体框架的前提下，替换核心物理模型可以获得显著改进

## 评分

- 新颖性: 7/10 — 解析积分思路在理论上自然，但推导优雅且实用价值高
- 实验充分度: 7/10 — 主流数据集覆盖全面但缺少与更多近期方法的对比
- 写作质量: 9/10 — 数学推导清晰严谨，图示直观，物理直觉解释到位
- 价值: 8/10 — 作为 3DGS 的即插即用改进，兼容性好且有 CT 应用潜力

<!-- RELATED:START -->

## 相关论文

- [StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](../../ICCV2025/3d_vision/stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)
- [Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)

<!-- RELATED:END -->
