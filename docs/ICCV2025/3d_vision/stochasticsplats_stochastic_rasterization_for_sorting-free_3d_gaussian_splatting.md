---
title: >-
  [论文解读] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting
description: >-
  [3D视觉] StochasticSplats 将随机透明度（Stochastic Transparency）引入 3DGS，通过无偏 Monte Carlo 估计替代深度排序的 alpha 混合，实现免排序、无 popping 的渲染，在 1 SPP 下比标准 CUDA 3DGS 快 4×，并可通过采样数灵活权衡质量与速度。
tags:
  - 3D视觉
---

# StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.24366](https://arxiv.org/abs/2503.24366)
- **代码**: 未公开（OpenGL + CUDA 双实现）
- **领域**: 3D视觉
- **关键词**: 3D高斯Splatting, 随机透明度, 免排序渲染, Monte Carlo估计, 体积渲染, OpenGL硬件加速

## 一句话总结
StochasticSplats 将随机透明度（Stochastic Transparency）引入 3DGS，通过无偏 Monte Carlo 估计替代深度排序的 alpha 混合，实现免排序、无 popping 的渲染，在 1 SPP 下比标准 CUDA 3DGS 快 4×，并可通过采样数灵活权衡质量与速度。

## 研究背景与动机

3D Gaussian Splatting 已成为主流辐射场方法，但其核心渲染算法——深度排序后逐序光栅化——存在多个固有限制：

**排序计算昂贵**：全局排序操作是渲染流程中的显著开销

**Popping伪影**：排序基于高斯均值深度，微小相机运动可导致排序顺序突变，产生时间不连续性

**无法灵活权衡**：对于固定表示，降低渲染分辨率反而不一定更快（每个像素覆盖更多高斯）

**Billboard 近似不准确**：将 3D 高斯投影为相机平行的 billboard，无法正确处理体积混合

**可移植性差**：多数实现使用 CUDA 定制光栅器，难以移植到非 NVIDIA 平台和标准图形管线

现有方法各自解决部分问题但互有取舍：StopThePop 解决 popping 但需复杂 CUDA 内核；EVER 用光线追踪实现精确体积混合但速度慢；Splatapult 用 OpenGL 但性能不如 CUDA。

## 方法详解

### 核心思想：随机透明度

标准 alpha 混合需要按深度排序所有图元：

$$C = \sum_{i=1}^{L} c_i \alpha_i \prod_{z_k < z_i}(1-\alpha_k)$$

随机透明度将其转化为 Monte Carlo 估计——随机采样一个图元 $i$：

$$C \approx c_i \quad \text{for} \quad i \sim P$$

其中采样概率为 $P(i) = \alpha_i \prod_{z_k < z_i}(1-\alpha_k)$。

概率与 alpha 混合权重完全一致，采样后直接取该图元颜色 $c_i$，除法抵消→估计器极其简洁。

### 实现算法

对每个像素，遍历所有高斯（无需排序）：
1. 计算当前高斯的 opacity $\alpha_i$ 和深度 $z_i$
2. 均匀采样 $u \sim \mathcal{U}(0,1)$
3. 如果 $u < \alpha_i$ 且 $z_i$ 比当前选中的更近，则选中该高斯
4. 最终像素颜色 = 选中高斯的颜色

关键：通过标准深度测试（Z-buffer）自动实现按 $P(i)$ 采样，**无需排序**。多次独立采样取平均减少估计噪声。

### 可微随机透明度

训练 3DGS 需要反向传播梯度。推导采用 detached gradient estimator：

**颜色梯度**（7式）：仅被采中的高斯 $i$ 接收颜色梯度，其他高斯梯度为零

$$\frac{\partial \mathcal{L}}{\partial c_i} = \frac{\partial \mathcal{L}}{\partial C}$$

**透明度梯度**（8式）：被采中的高斯和其前方的高斯接收 opacity 梯度

$$\frac{\partial \mathcal{L}}{\partial \alpha_i} = \frac{\partial \mathcal{L}}{\partial C} \frac{c_i}{\alpha_i}, \quad \frac{\partial \mathcal{L}}{\partial \alpha_{z_k<z_i}} = \frac{\partial \mathcal{L}}{\partial C} \frac{-c_i}{1-\alpha_{z_k<z_i}}$$

**去相关损失梯度**：使用不同随机种子分别计算 $\partial\mathcal{L}/\partial C$ 和 $\partial C/\partial\theta$，避免两个含噪项的相关性导致梯度偏差。

实际训练采用 path replay backpropagation 三阶段：
1. 渲染图像，计算 $\partial\mathcal{L}/\partial C$
2. 用不同随机种子渲染第二张图像，得到 $c_i$
3. 用相同种子重播第二次渲染，计算参数梯度

### 消除 Popping 伪影

**简化方式**：非逐像素修改深度，而是将 billboard 方向调整为线性近似高斯最大密度面：

$$\mathbf{n}^\top(\mathbf{x} - \boldsymbol{\mu}) = 0, \quad \text{where} \quad \mathbf{n} = \Sigma^{-1}(\boldsymbol{\mu} - \mathbf{o})$$

以平面近似曲面，保持硬件光栅化的深度插值能力，无额外时间开销。

**完全体积混合**（可选高质量模式）：将固定深度替换为采样的自由路径距离 $z_i \sim p(t)$，其中 $p(t)$ 基于 Beer-Lambert 定律和 3D 高斯的解析密度积分。对重叠高斯使用 decomposition tracking 取最短自由路径。

### OpenGL 实现

- Fragment shader 中通过标准深度测试自然实现采样
- 透明度测试通过 `discard` 指令在 $u \geq \alpha_i$ 时丢弃 fragment
- 多采样通过超采样（高分辨率渲染 + 降采样）实现
- 可与时间抗锯齿（TAA）结合，跨帧累积降噪

## 实验关键数据

### 主实验：渲染质量 (MipNerf360)

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS (alpha blending) | 28.99 | 0.869 | 0.185 |
| StopThePop | 28.79 | 0.870 | 0.181 |
| Ours (1 SPP) | 17.95 | 0.285 | 0.611 |
| Ours (4 SPP) | 22.81 | 0.512 | 0.493 |
| Ours (16 SPP) | 26.25 | 0.714 | 0.351 |
| Ours (64 SPP) | 27.93 | 0.819 | 0.235 |
| Ours (256 SPP) | 28.50 | 0.856 | 0.178 |
| Ours (1024 SPP) | 28.66 | 0.867 | 0.168 |

随着 SPP 增加，质量逐步逼近 alpha blending。1024 SPP 时 LPIPS 甚至优于原始 3DGS。

### 消融实验：渲染速度 (ms)

| 方法 | T1000 | RTX3090 | RTX4090 |
|------|-------|---------|---------|
| 3DGS-CUDA | 65.08 | 8.14 | 5.60 |
| 3DGS-OpenGL | 100.28 | 32.15 | 20.70 |
| StopThePop | 75.91 | 9.48 | - |
| **Ours (1 SPP)** | **16.23** | **3.25** | **1.85** |
| Ours (2 SPP) | 18.95 | 4.18 | 2.05 |
| Ours (4 SPP) | 24.71 | 6.42 | 2.86 |
| Ours (8 SPP) | 37.51 | 15.31 | 6.71 |
| Ours (16 SPP) | 61.25 | 18.48 | 8.00 |

1 SPP 在 RTX4090 上 1.85ms，比 3DGS-CUDA 快 **3.0×**；在低端 T1000 上快 **4.0×**。

### 语义定位应用 (LERF 数据集)

| 方法 | Teatime | Ramen | Waldo-Kitchen | Figurines | 平均准确率 |
|------|---------|-------|---------------|-----------|------------|
| Alpha Blending | 88.1 | 73.2 | 95.5 | 80.4 | 84.3 |
| Ours (1 SPP) | 88.1 | 73.2 | 86.4 | 80.4 | 82.0 |

即使 1 SPP 的噪声渲染也能有效支持下游感知任务，仅损失 2.3% 定位准确率。

### 关键发现

1. **降分辨率 ≠ 加速**：在 3DGS 中降低分辨率后每个像素/tile 覆盖更多高斯，渲染反而变慢，StochasticSplats 的采样数才是正确的质量-速度旋钮
2. **TAA 有效**：时间抗锯齿在 1 SPP 下显著降低渲染噪声，几乎无额外延迟
3. **i.i.d. 噪声 vs 结构性伪影**：均匀噪声在视觉上比 popping 等结构性伪影更不引人注目
4. **硬件加速关键**：OpenGL 实现的 1 SPP 渲染利用了早期丢弃、Z-buffer 等硬件特性
5. **梯度估计有效**：去相关梯度估计在 128 SPP 下即可准确近似 alpha blending 的梯度

## 亮点与洞察

1. **视角转换优雅**：将成熟的计算机图形学技术（随机透明度）引入辐射场渲染，桥接两个社区
2. **质量-速度连续体**：SPP 提供了从快速预览（1 SPP）到高质量呈现（1024 SPP）的平滑过渡，类似物理渲染
3. **可移植性**：OpenGL 实现天然跨平台、跨硬件，对 VR/AR 和移动端部署有重要意义
4. **通用梯度估计器**：反向传播推导不特定于 3DGS，可应用于任何半透明表示
5. **体积混合的正确方向**：通过自由路径采样自然处理重叠高斯的体积混合，无需射线追踪

## 局限性

1. **低 SPP 下噪声明显**：1 SPP 下 PSNR 仅 17.95，虽不影响感知但影响视觉质量指标
2. **梯度偏差**：对于非 L2 损失函数，去相关估计仅能减少但不能消除梯度偏差
3. **微调而非从头训练**：当前仅展示在已训练 3DGS 上微调 1000 步，从头训练的收敛性待验证
4. **单帧方差**：需要多帧结合 TAA 或更多采样才能达到 alpha blending 的渲染质量
5. **体积混合模式较慢**：完全体积混合需要禁用硬件早期丢弃，效率下降

## 相关工作与启发

- **Stochastic Transparency (Enderton 2010)**：实时图形学中成熟的顺序无关透明度方法，本文的直接灵感来源
- **StopThePop**：通过逐像素排序解决 popping，但计算开销大且不易移植
- **EVER**：使用恒密度椭球射线追踪实现精确体积混合，但速度远慢于光栅化
- **Path Replay Backpropagation (Vicini 2021)**：可微 Monte Carlo 方法的梯度计算范式
- **启发**：成熟的图形学技术与新兴表示的结合常能产生意想不到的优雅方案

## 评分

⭐⭐⭐⭐ (4/5)

理论推导严谨，将图形学经典技术优雅地引入 3DGS。OpenGL 实现的可移植性和质量-速度连续权衡具有实际价值。不足在于低 SPP 下质量指标差距较大，且训练方面仅展示了微调。整体是 3DGS 渲染效率领域的有价值探索。

<!-- RELATED:START -->

## 相关论文

- [SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)

<!-- RELATED:END -->
