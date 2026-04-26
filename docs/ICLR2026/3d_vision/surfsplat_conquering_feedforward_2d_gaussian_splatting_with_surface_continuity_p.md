---
title: >-
  [论文解读] SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors
description: >-
  [ICLR 2026][3D视觉][2D高斯溅射] SurfSplat 提出基于2DGS的前馈3D重建框架，通过表面连续性先验将高斯的旋转和尺度与邻域位置绑定、以及强制透明度混合策略解决颜色偏差，并引入HRRC指标揭示高分辨率下的重建质量差异。
tags:
  - ICLR 2026
  - 3D视觉
  - 2D高斯溅射
  - 前馈3D重建
  - 表面连续性
  - 高分辨率渲染一致性
  - 稀疏视角
---

# SurfSplat: Conquering Feedforward 2D Gaussian Splatting with Surface Continuity Priors

**会议**: ICLR 2026  
**arXiv**: [2602.02000](https://arxiv.org/abs/2602.02000)  
**代码**: [https://hebing-sjtu.github.io/SurfSplat-website/](https://hebing-sjtu.github.io/SurfSplat-website/)  
**领域**: 3D视觉  
**关键词**: 2D高斯溅射, 前馈3D重建, 表面连续性, 高分辨率渲染一致性, 稀疏视角

## 一句话总结

SurfSplat 提出基于2DGS的前馈3D重建框架，通过表面连续性先验将高斯的旋转和尺度与邻域位置绑定、以及强制透明度混合策略解决颜色偏差，并引入HRRC指标揭示高分辨率下的重建质量差异。

## 研究背景与动机

现有前馈3DGS方法在标准分辨率下的NVS指标看似出色，但存在一个被忽视的严重问题：

**退化的3D场景**：重建结果实际是离散的、颜色偏差的点云而非连续表面，近距离或偏轴视角下暴露严重伪影（空洞、颜色偏差、表面断裂）

**各向异性利用不足**：可学习的高斯体难以仅通过梯度监督解耦几何和纹理，导致高斯退化为近球形

**评估指标失效**：标准PSNR/SSIM/LPIPS在原始分辨率下无法捕获几何不准确性，掩盖了真实的重建质量

作者观察到：**直接训练2DGS比3DGS更具挑战**——2D高斯的面片特性使得微小几何扰动造成渲染输出的大幅偏差，在有限监督下问题加剧。

## 方法详解

### 整体框架

SurfSplat 采用双路径编码器（单视角Depth Anything V2 + 多视角交叉注意力），融合特征后通过U-Net回归中间属性（深度、尺度乘子、外观分量），最终通过表面连续性先验和强制透明度混合转换为标准2DGS属性。

### 关键设计

1. **表面连续性先验 (Surface Continuity Prior)**

   核心假设：真实场景中可见几何主要由平滑连续表面组成，空间相邻的表面元对应图像中的相邻像素。基于此约束高斯的旋转和尺度：

   **旋转**：对像素 $(h,w)$ 的3D位置 $\mathbf{p}_0$ 及其邻域，用Sobel滤波器计算两个切向量 $\mathbf{t}_1, \mathbf{t}_2$，通过叉积得到局部法线 $\mathbf{n}$，再用Rodrigues公式计算旋转矩阵：

   $$\mathbf{R} = \mathbf{I} + [\mathbf{v}]_\times + \frac{1-c}{\|\mathbf{v}\|^2}[\mathbf{v}]_\times^2$$

   其中 $\mathbf{v} = \mathbf{n}_0 \times \mathbf{n}$，$c = \mathbf{n}_0^\top \mathbf{n}$。

   **尺度**：基于图像空间邻域距离的粗估计 $\bar{\sigma}_u, \bar{\sigma}_v$，网络预测  $[1/3, 3]$ 范围内的尺度乘子 $\hat{\sigma}_u, \hat{\sigma}_v$，最终尺度 $\sigma_u = \bar{\sigma}_u \hat{\sigma}_u$。2DGS的深度轴尺度 $\sigma_w$ 固定为零。

   这样，高斯属性不是独立回归，而是从预测的3D位置推导，确保空间一致性。

2. **强制透明度混合 (Forced Alpha Blending)**

   表面连续性先验下模型倾向学习高不透明度高斯，导致遮挡的高斯在alpha混合中贡献极小，无法学习3D结构。解决方案：

   - 用上界 $\tau_{\text{opa}} < 1$（设为0.6）裁剪不透明度，确保所有高斯参与渲染
   - 将RGB颜色初始化到球谐基的DC分量
   - 对渲染输出做透明度归一化补偿：$C = C/\alpha$ 当 $\alpha \geq \tau_\alpha$

3. **HRRC 高分辨率渲染一致性指标**

   在$2\times$和$4\times$分辨率下渲染重建场景，与双三次上采样的GT比较：
   $$\text{HRRC}_{\text{metric}} = \text{metric}(\hat{I}^{HR}, \hat{I}^{GT\uparrow})$$
   
   能有效暴露稀疏性空洞、退化高斯形状和不连续性，区分真正恢复3D几何的模型与仅记忆稀疏视角的模型。

### 损失函数 / 训练策略

$$L_{\text{gs}} = \sum_{m=1}^M \left(\text{MSE}(I_{\text{render}}^m, I_{\text{gt}}^m) + \lambda \cdot \text{LPIPS}(I_{\text{render}}^m, I_{\text{gt}}^m)\right)$$

$\lambda = 0.05$，在256×256分辨率下训练。Depth Anything V2骨干学习率 $2 \times 10^{-6}$，其他层 $2 \times 10^{-4}$。

## 实验关键数据

### 主实验

| 方法 | RE10K 256 PSNR↑ | RE10K 512 PSNR↑ | RE10K 1024 PSNR↑ | RE10K Avg PSNR↑ |
|------|------|------|------|------|
| DepthSplat | **27.504** | 20.031 | 16.385 | 21.307 |
| MVSplat | 26.359 | 20.408 | 17.966 | 21.578 |
| Ours-L | 27.537 | **26.331** | **24.897** | **26.255** |

### 消融实验

| 配置 | 标准PSNR | HRRC 2× PSNR | HRRC 4× PSNR | 说明 |
|------|---------|---------|---------|------|
| 3DGS基线 (DepthSplat) | 27.504 | 20.031 | 16.385 | HRRC严重退化 |
| 2DGS (SurfSplat-L) | 27.537 | 26.331 | 24.897 | HRRC保持稳定 |
| w/o 表面先验 | 降低 | 大幅降低 | 大幅降低 | 表面不连续 |
| w/o 强制混合 | 降低 | 降低 | 降低 | 颜色偏差 |

### 关键发现

- **HRRC揭示真相**：DepthSplat在标准分辨率(27.5)表现最佳，但1024分辨率猛降到16.4；SurfSplat从27.5仅降到24.9，证明真正重建了3D结构
- MVSplat、TranSplat在HRRC下同样大幅退化（1024下仅18左右），说明3DGS前馈方法普遍存在表面退化问题
- 跨数据集（DL3DV、ScanNet）评估验证了泛化能力
- pixelSplat（多高斯/像素）在HRRC上反而较好（24.9），因为冗余高斯部分弥补了表面空洞

## 亮点与洞察

1. **问题揭示价值高**：指出了前馈3DGS领域被广泛忽视的表面退化问题，HRRC指标具有推广意义
2. **几何驱动属性预测**：用Sobel滤波+Rodrigues公式从位置推导旋转这一设计简洁优雅，物理直觉清晰
3. **2DGS在前馈场景的首次成功**：证明2DGS（面片）比3DGS（椭球）在前馈重建中更适合，提供更强的各向异性和几何精度
4. **强制混合的巧妙设计**：通过限制不透明度上界解决局部最优问题，保证多层表现力

## 局限与展望

- 在标准分辨率下仅勉强超过DepthSplat，优势主要体现在HRRC
- 每像素单高斯的设定，对复杂场景的覆盖可能不够
- HRRC指标依赖双三次上采样GT，并非真正的高分辨率GT
- 仅评估静态场景，动态场景扩展是未来方向

## 相关工作与启发

- Huang et al. (2024) 提出2DGS用于逐场景优化，SurfSplat首次将其引入前馈框架
- DepthSplat (Xu et al., 2024b) 的深度交互设计被部分继承，但SurfSplat用表面先验替代了独立属性回归
- HRRC指标的设计思路可推广到其他3D生成任务的评估

## 评分

- 新颖性: ⭐⭐⭐⭐ 表面连续性先验和HRRC指标有新意，但组件设计属于成熟技术的组合应用
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、HRRC多分辨率、多骨干变体、跨数据集评估，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，可视化对比有说服力，数学推导完整
- 价值: ⭐⭐⭐⭐ HRRC指标和表面退化问题的揭示对社区有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](../../CVPR2026/3d_vision/3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[AAAI 2026\] MeshSplat: Generalizable Sparse-View Surface Reconstruction via Gaussian Splatting](../../AAAI2026/3d_vision/meshsplat_generalizable_sparse-view_surface_reconstruction_via_gaussian_splattin.md)
- [\[AAAI 2026\] SparseSurf: Sparse-View 3D Gaussian Splatting for Surface Reconstruction](../../AAAI2026/3d_vision/sparsesurf_sparse-view_3d_gaussian_splatting_for_surface_reconstruction.md)
- [\[ICCV 2025\] SurfaceSplat: Connecting Surface Reconstruction and Gaussian Splatting](../../ICCV2025/3d_vision/surfacesplat_connecting_surface_reconstruction_and_gaussian_splatting.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)

<!-- RELATED:END -->
