---
title: >-
  [论文解读] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images
description: >-
  [ICCV 2025][3D视觉][3D高斯泼溅] 提出专门为3DGS设计的轻量图像上采样技术，利用高斯原语的解析图像梯度进行梯度感知双三次样条插值，无需深度学习推理即可实现3-4倍渲染加速，且重建质量优于标准双三次插值和DL-based上采样。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D高斯泼溅
  - 图像上采样
  - 解析梯度
  - 样条插值
  - 轻量渲染
---

# Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images

**会议**: ICCV 2025  
**arXiv**: [2503.14171](https://arxiv.org/abs/2503.14171)  
**代码**: 未公开  
**领域**: 3D视觉  
**关键词**: 3D高斯泼溅, 图像上采样, 解析梯度, 样条插值, 轻量渲染

## 一句话总结

提出专门为3DGS设计的轻量图像上采样技术，利用高斯原语的解析图像梯度进行梯度感知双三次样条插值，无需深度学习推理即可实现3-4倍渲染加速，且重建质量优于标准双三次插值和DL-based上采样。

## 研究背景与动机

尽管3DGS在新视角合成方面不断改进，但**高端GPU在高分辨率显示上仍难以维持稳定30FPS**。VR应用更为严峻——需同时渲染两张高分辨率图像。移动设备上性能更是降到不可接受的水平。

**现有上采样方案的问题**：

**传统插值**（双三次、Lanczos）：在大倍率下引入明显伪影（振铃、阶梯效应），质量下降

**深度学习上采样**（NinaSR、DLSS、SwinIR）：
   - 高质量方案不够快，无法用于实时
   - DLSS需要专用硬件+额外图像信息（深度缓冲、运动向量），3DGS无法提供
   - DL上采样会**引入幻觉伪影**和**时序不一致**（VR用户敏感）
   - PSNR反而可能下降（虽然看起来锐利但不忠于原图）

**核心洞察**：3DGS的高斯原语是**解析可微**的——可以精确计算每个像素处图像颜色的梯度，而非用有限差分近似。将这些精确梯度注入双三次样条插值，可获得远优于传统插值的结果，且计算开销极小。

## 方法详解

### 核心思想

传统双三次插值通过有限差分估计采样点处的斜率，然后拟合平滑曲线。但有限差分只是梯度的近似，在信号变化剧烈处会导致重建偏离。

本文直接利用3DGS渲染过程中可**免费获得的解析梯度**替代有限差分，获得更精确的样条拟合。

### 解析图像梯度计算

3DGS渲染的像素颜色 $I(x,y) = \sum_{i=1}^N T_i(x,y) \alpha_i(x,y) c_i$

对空间坐标微分：

$$\frac{\partial I(x,y)}{\partial x} = \sum_{i=1}^N c_i \left(\frac{\partial T_i}{\partial x} \alpha_i + T_i \frac{\partial \alpha_i}{\partial x}\right)$$

$$\frac{\partial I(x,y)}{\partial y} = \sum_{i=1}^N c_i \left(\frac{\partial T_i}{\partial y} \alpha_i + T_i \frac{\partial \alpha_i}{\partial y}\right)$$

交叉导数 $\frac{\partial^2 I}{\partial x \partial y}$ 类似推导。透射率 $T_i$ 的导数通过 $\alpha$-blending递推迭代计算：

$$\frac{\partial A_i}{\partial x} = \frac{\partial A_{i-1}}{\partial x}(1 - \alpha_i) + (1 - A_{i-1})\frac{\partial \alpha_i}{\partial x}$$

关键：这些计算可以**嵌入3DGS前向渲染流程**，仅增加边际开销。

### 梯度感知样条插值

双三次插值求解 $F = C A C^T$ 获取多项式系数 $A \in \mathbb{R}^{4 \times 4}$：

$$F = \begin{bmatrix} f(0,0) & f(0,1) & f_y(0,0) & f_y(0,1) \\ f(1,0) & f(1,1) & f_y(1,0) & f_y(1,1) \\ f_x(0,0) & f_x(0,1) & f_{xy}(0,0) & f_{xy}(0,1) \\ f_x(1,0) & f_x(1,1) & f_{xy}(1,0) & f_{xy}(1,1) \end{bmatrix}$$

传统方法：$f_x, f_y, f_{xy}$ 用有限差分估计
本文方法：$f_x, f_y, f_{xy}$ 用上述解析梯度精确给出

插值后在任意目标像素处评估：

$$p(x,y) = [1, x, x^2, x^3] A [1, y, y^2, y^3]^T$$

### 可微反向传播

梯度是完全可微的，可以集成到3DGS优化中：模型在低分辨率渲染后上采样，用上采样结果计算损失并反向传播优化高斯原语参数。

$$\frac{\partial^2 B_i}{\partial x \partial \sigma_k} = \frac{\partial^2 B_i}{\partial \alpha_k \partial x} \frac{\partial \alpha_k}{\partial \sigma_k} + \frac{\partial B_i}{\partial \alpha_k} \frac{\partial^2 \alpha_k}{\partial \sigma_k \partial x}$$

## 实验

### 渲染质量对比（4×上采样）

| 方法 | 类型 | PSNR | 额外开销 |
|:---|:---:|:---:|:---|
| 双三次插值 | 传统 | 基线 | 极低 |
| Lanczos (FSR) | 传统 | 略优于双三次 | 中等 |
| DL-based (ESRGAN等) | 深度学习 | PSNR可能下降 | 高（GPU推理） |
| **梯度感知上采样** | **本文** | **最优** | **边际** |

关键发现：DL上采样虽然看起来锐利，但由于引入幻觉细节，PSNR反而下降。

### 速度提升

| 配置 | 方法 | 渲染速度 |
|:---|:---:|:---|
| 全分辨率 | 标准3DGS | 1× (基线) |
| 1/4分辨率+上采样 | 梯度感知上采样 | **3-4×加速** |

梯度计算的额外开销通过低分辨率光栅化节省的时间完全补偿。

### 训练集成效果

| 方式 | 说明 | 质量 |
|:---|:---|:---|
| 仅渲染时上采样 | 标准训练 + 推理上采样 | 优于双三次 |
| **训练+渲染时上采样** | **低分辨率训练+上采样+反传** | **进一步提升** |

将上采样嵌入训练后，模型学会了为上采样场景优化的高斯原语形状。

### 消融：梯度类型对比

| 梯度类型 | 质量 |
|:---|:---|
| 有限差分（传统双三次） | 振铃伪影、阶梯效应 |
| **解析梯度（本文）** | **平滑、高保真** |

2D插值benchmark上定性对比显示：解析梯度消除了有限差分导致的振铃和阶梯伪影。

## 亮点与洞察

1. **零训练开销的上采样**：不引入任何神经网络，不需要预训练，仅利用3DGS本身的数学性质
2. **实现优雅**：梯度计算嵌入现有渲染流程，对任何3DGS实现都可以无侵入式集成
3. **时序稳定**：不像DL上采样会引入帧间不一致，解析梯度保证确定性结果
4. **VR友好**：轻量GPU上也可获得高分辨率输出，适合移动VR设备

## 局限性

1. 上采样倍率过大时（如8×+），插值精度仍然受限于低分辨率采样密度
2. 高频纹理区域可能仍存在一定平滑
3. 与Mip-Splatting等抗锯齿方法的协同效果需要进一步研究

## 相关工作

- **3DGS优化**：Mip-Splatting, SRGS, 梯度引导分裂
- **图像上采样**：双三次插值, Lanczos, ESRGAN, DLSS, GaussianSR
- **NeRF超分辨率**：NeRF-SR, RefSR-NeRF

## 评分

- 新颖性：⭐⭐⭐⭐ — 利用解析梯度做上采样的想法简洁而巧妙
- 技术深度：⭐⭐⭐⭐ — 前向+反向的可微推导完整，数学基础扎实
- 实验完整性：⭐⭐⭐⭐ — 多数据集、多基线、速度+质量双重评估
- 实用价值：⭐⭐⭐⭐⭐ — 即插即用，零额外训练，显著加速

<!-- RELATED:START -->

## 相关论文

- [CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)
- [EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](../../ECCV2024/3d_vision/pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)
- [OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)

<!-- RELATED:END -->
