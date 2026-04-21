---
title: >-
  [论文解读] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy
description: >-
  [ECCV 2024][3D视觉][3D高斯喷溅] 从数学角度分析3DGS中局部仿射近似引入的投影误差，推导出当高斯均值到相机中心连线为投影方向时误差最小，提出 Optimal Gaussian Splatting 投影策略。
tags:
  - ECCV 2024
  - 3D视觉
  - 3D高斯喷溅
  - 投影误差分析
  - 最优投影
  - 实时渲染
  - 新视角合成
---

# On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy

**会议**: ECCV 2024  
**arXiv**: [2402.00752](https://arxiv.org/abs/2402.00752)  
**代码**: https://letianhuang.github.io/op43dgs/ (有)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 投影误差分析, 最优投影, 实时渲染, 新视角合成

## 一句话总结

从数学上系统分析3D Gaussian Splatting中局部仿射近似引入的投影误差，证明误差函数在Gaussian均值方向与投影平面法线重合时取极小值，据此提出每个Gaussian投影到各自切平面的最优投影策略(Optimal Gaussian Splatting)，在不影响实时性能的前提下显著降低渲染伪影。

## 研究背景与动机

**领域现状**: 3D Gaussian Splatting (3D-GS)以高斯函数作为显式场景表示，通过可微光栅化实现实时渲染，已成为NeRF之后最受关注的新视角合成方法。

**现有痛点**: 高斯函数在仿射变换下封闭，但在投影变换下不封闭。3D-GS使用一阶Taylor展开的局部仿射近似来处理投影，但忽略的高阶余项会引入误差。对于远离投影平面中心的高斯，误差尤其显著，导致拉伸型和云状伪影。

**核心矛盾**: 3D-GS的各种改进（稀疏视角、存储优化、mesh重建等）层出不穷，但基本的投影误差问题一直被忽视，这是一个基础性的"地基"问题。

**本文目标**: 定量分析投影误差与高斯均值位置的关系，找到误差最小化的投影策略。

**切入角度**: 从Taylor余项出发进行数学分析，将误差表示为高斯均值极坐标的函数，通过函数优化理论求极值。

**核心 idea**: 将所有高斯投影到同一平面$z=1$是次优的，让每个高斯沿其均值到相机中心方向投影到各自的切平面可最小化投影误差。

## 方法详解

### 整体框架

Optimal Gaussian Splatting (OGS) 修改了3D-GS的投影过程：不再将所有3D高斯统一投影到相机的$z=1$平面，而是为每个高斯计算其均值到相机中心连线方向上的单位球切平面，在该切平面上进行投影。最后通过Unit Sphere Based Rasterizer将切平面上的2D高斯映射回图像像素。

### 关键设计

1. **投影误差函数推导**: 3D-GS中投影函数$\varphi(\mathbf{x}') = \mathbf{x}'(\mathbf{x_0}^\top \mathbf{x}')^{-1}$的一阶Taylor展开余项为：

$$R_1(\mathbf{x}') = \varphi(\mathbf{x}') - \varphi(\boldsymbol{\mu}') - \frac{\partial \varphi}{\partial \mathbf{x}'}(\boldsymbol{\mu}')(\mathbf{x}' - \boldsymbol{\mu}')$$

   对该余项的Frobenius范数平方取数学期望，得到仅依赖高斯均值的误差函数：

$$\epsilon(\boldsymbol{\mu}') = \int_{\mathbf{x}' \in \mathcal{X}'} \|R_1(\mathbf{x}')\|_F^2 \, d\mathbf{x}'$$

   将均值用球坐标$(\theta_\mu, \phi_\mu)$表示后，误差函数有闭合形式解。设计动机：将直觉上"远离中心误差大"的观察严格数学化。

2. **误差函数极值分析**: 对误差函数求偏导：

$$\frac{\partial \epsilon}{\partial \theta_\mu}(0,0) = 0, \quad \frac{\partial \epsilon}{\partial \phi_\mu}(0,0) = 0$$

   在$(\theta_\mu, \phi_\mu) = (0, 0)$处取极小值。这意味着当高斯均值的投影方向与投影平面法线重合时（即$\mathbf{x_0}$方向），误差最小。函数在原点附近平坦，但接近边界时急剧增大。核心发现：误差最小值大于零（因为高斯在投影变换下不封闭），但可以通过选择投影平面来最小化。

3. **最优投影策略 (Optimal Projection)**: 为每个高斯选择各自的投影切平面。切平面方程为$\mathbf{x_p}^\top \cdot (\mathbf{x}' - \mathbf{x_p}) = 0$，其中：

$$\mathbf{x_p} = \varpi(\boldsymbol{\mu}') = \boldsymbol{\mu}'(\boldsymbol{\mu}'^\top \boldsymbol{\mu}')^{-1/2}$$

   是高斯均值在单位球上的投影。最优投影函数为$\varphi_p(\mathbf{x}') = \mathbf{x}'(\mathbf{x_p}^\top \mathbf{x}')^{-1}$。对应的Jacobian矩阵$\mathbf{J_p}$由高斯均值坐标$(\mu_x, \mu_y, \mu_z)$决定，只需修改前向过程中的Jacobian计算即可实现。

4. **Unit Sphere Based Rasterizer**: 对图像像素$(u,v)$，先转换到相机空间再投影到单位球：

$$\mathbf{x}_{2D} = \varphi_p\left(\begin{bmatrix}(u-c_x)/f_x \\ (v-c_y)/f_y \\ 1\end{bmatrix}\right)$$

   然后查询该点处各切平面高斯的函数值进行alpha blending。由于投影不依赖$z=1$平面，可自然适配鱼眼相机和全景图像等不同相机模型。

5. **旋转矩阵$\mathbf{Q}$**: 由于$\mathbf{J_p}\boldsymbol{\Sigma}'\mathbf{J_p}^\top$的第三行/列不全为零（不同于原始3D-GS），需要引入可逆矩阵$\mathbf{Q}$对$\mathbf{x}_{2D}$、$\varphi_p(\boldsymbol{\mu}')$和$\mathbf{J_p}$进行左乘变换，使得可以正确提取2D协方差矩阵的逆。$\mathbf{Q}$由高斯均值坐标构造。

### 损失函数 / 训练策略

使用与原始3D-GS完全相同的默认参数和训练策略（L1 + D-SSIM损失），仅修改投影过程中的Jacobian计算和光栅化流程，保持实验的可控性。自定义CUDA核实现光栅化。

## 实验关键数据

### 主实验: 13个真实场景 (Mip-NeRF360 + Tanks & Temples + Deep Blending)

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| Plenoxels | 22.77 | 0.666 | 0.457 |
| INGP-Big | 24.93 | 0.724 | 0.336 |
| Mip-NeRF360 | 27.11 | 0.803 | 0.241 |
| 3D-GS | 26.92 | 0.832 | 0.214 |
| **Ours (OGS)** | **27.17** | **0.836** | **0.210** |

### 短焦距鲁棒性实验 (×0.2 / ×0.3 焦距, 6个场景均值)

| 指标 | 焦距 | 3D-GS | OGS (Ours) | 提升 |
|------|------|-------|------------|------|
| PSNR ↑ | ×0.2 | 15.58 | 20.46 | **+4.88** |
| PSNR ↑ | ×0.3 | 19.49 | 22.10 | **+2.61** |
| SSIM ↑ | ×0.2 | 0.499 | 0.628 | +0.129 |
| SSIM ↑ | ×0.3 | 0.630 | 0.684 | +0.054 |
| LPIPS ↓ | ×0.2 | 0.397 | 0.251 | **-0.146** |
| LPIPS ↓ | ×0.3 | 0.263 | 0.231 | -0.032 |

### 关键发现

- 在标准焦距下，OGS比3D-GS提升约0.25 PSNR（27.17 vs 26.92），超越Mip-NeRF360同时保持实时渲染性能。
- **短焦距场景优势巨大**：焦距缩减到0.2倍时，PSNR提升4.88 dB，LPIPS改善0.146，证实投影误差在广角设定下影响极为显著。
- 3D-GS的投影误差随焦距减小急剧增大（视场角扩大导致更多高斯偏离投影中心），而OGS由于采用中心径向投影，不受此影响。
- OGS可自然适配鱼眼和全景相机模型，原始3D-GS的平面投影无法支持。

## 亮点与洞察

- **理论驱动的改进**: 区别于大多数经验调参的改进，本文从严格的数学分析出发推导最优策略，有closed-form证明误差在切平面投影时取极小值。
- **极简改动，显著效果**: 核心改动仅是修改Jacobian矩阵的计算方式和光栅化流程，不改变训练参数，不增加模型复杂度。
- **广角/鱼眼适配**: 原始3D-GS在广角场景下严重退化，OGS的解决方案不仅改善质量还扩展了适用范围，对VR/AR等广角应用场景有重要意义。
- **误差可视化**: 将误差函数$\epsilon(\theta_\mu, \phi_\mu)$可视化为3D曲面，直观展示边缘区域误差爆炸性增长。

## 局限与展望

- 当前分析假设协方差恒定，仅关注均值对误差的影响。协方差（高斯形状/尺寸）对投影误差的影响需要进一步讨论。
- 切平面投影后需要额外变换到图像平面，导致训练时间略有增加（推理不受影响）。
- 对于高斯均值非常接近相机中心的退化情况（$\|\boldsymbol{\mu}'\| \to 0$），投影可能不稳定。
- 优化CUDA实现可进一步缩短训练时间开销。

## 相关工作与启发

- **3D-GS生态系统**: 本文针对3D-GS管线中一个被忽视的基础问题进行严格分析，与各种上层改进（压缩、稀疏视角、动态场景等）正交互补。
- **仿射近似的普遍性**: EWA Splatting等经典方法也使用局部仿射近似，本文的分析框架可能推广到其他splatting技术。
- **2D GS / Gaussian Surfels**: 后续工作如2DGS已转向面向surface的高斯，本文的投影策略思想可能对这些方法也有启发。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 从数学基础层面分析3D-GS的投影误差，切入角度新颖且有理论深度。
- **实验充分度**: ⭐⭐⭐⭐ — 标准场景+短焦距+多相机模型的验证全面，但场景数量可以更多。
- **写作质量**: ⭐⭐⭐⭐⭐ — 数学推导严谨清晰，从问题定义到解决方案的逻辑链完整流畅。
- **实用价值**: ⭐⭐⭐⭐ — 改动极小但效果显著，尤其对广角/VR场景有重要实用价值。
**领域**: 3D视觉  
**关键词**: 3D高斯喷溅, 投影误差分析, 最优投影, 实时渲染, 新视角合成

## 一句话总结

从数学角度分析3DGS中局部仿射近似引入的投影误差，推导出当高斯均值到相机中心连线为投影方向时误差最小，提出 Optimal Gaussian Splatting 投影策略。

## 研究背景与动机

- 3DGS 使用局部仿射近似（一阶 Taylor 展开）将3D高斯投影到图像平面
- 高斯函数在投影变换下不封闭，仿射近似必然引入误差
- 偏离图像中心的高斯投影误差更大，尤其在广角/短焦距设置下会产生拉长伪影
- 目前缺乏对该投影误差的系统分析和最优化方案

## 方法详解

### 整体框架

1. 分析投影误差函数 $\epsilon(\theta_\mu, \phi_\mu)$，建立误差与高斯均值位置的关系
2. 通过函数优化理论找到误差极小值点
3. 提出每个高斯沿其均值到相机中心方向投影到切平面的最优策略

### 关键设计

- **误差函数推导**：从 Taylor 余项 $R_1(\mathbf{x}')$ 出发，计算 Frobenius 范数的数学期望，得到误差函数 $\epsilon(\theta_\mu, \phi_\mu)$
- **极值分析**：证明当高斯均值的极坐标 $(\theta_\mu, \phi_\mu)$ 等于投影平面法线方向 $(0,0)$ 时误差取极小值
- **Optimal Projection**：每个高斯不再投影到统一的 z=1 平面，而是沿高斯均值到相机原点的方向投影到单位球的切平面
- **Unit Sphere Rasterizer**：对每个像素投射射线到单位球，查询相交的切平面高斯进行 alpha blending
- **多相机模型适配**：由于不依赖 z=1 平面，天然支持鱼眼、全景等相机模型

### 损失函数

使用3DGS原始的光度损失（L1 + D-SSIM），不引入额外损失。

## 实验关键数据

### 主实验

13个真实场景上的定量评估（Mip-NeRF360 + Tanks&Temples + Deep Blending）：

| 方法 | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ |
|------|-----------|-----------|------------|
| Plenoxels | 22.77 | 0.666 | 0.457 |
| INGP-Big | 24.93 | 0.724 | 0.336 |
| M-NeRF360 | 27.11 | 0.803 | 0.241 |
| 3D-GS | 26.92 | 0.832 | 0.214 |
| **Ours** | **27.17** | **0.836** | **0.210** |

### 消融实验

短焦距设置下的鲁棒性比较（6个场景平均）：

| 指标 | 焦距比 | 3D-GS | Ours |
|------|--------|-------|------|
| PSNR↑ | ×0.2 | 15.58 | 20.46 |
| PSNR↑ | ×0.3 | 19.49 | 22.10 |
| SSIM↑ | ×0.2 | 0.499 | 0.628 |
| LPIPS↓ | ×0.2 | 0.397 | 0.251 |

焦距缩短至0.2倍时，本方法相比原始3DGS PSNR提升约5dB。

### 关键发现

- 投影误差在图像边缘/大视角处急剧增大
- 焦距越短，视场角越大，误差越严重，产生针状和云状伪影
- 最优投影只需修改 Jacobian 计算，不影响实时渲染性能
- 天然支持鱼眼相机和全景图生成

## 亮点与洞察

- **理论驱动**：从严格数学分析出发推导最优投影，而非启发式设计
- 仅需少量代码修改即可实现，工程友好
- 解决了广角镜头下3DGS的系统性伪影问题
- 首次为3DGS提供投影误差的封闭形式表达式

## 局限性

- 切平面投影后需要坐标变换回图像空间，训练时间略有增加
- 当前分析假设协方差不变，未考虑协方差对误差的影响
- 实际改进在标准焦距设置下相对有限

## 相关工作与启发

- **3DGS**：Kerbl et al. 2023 奠定了高斯喷溅的基础
- **NeRF 系列**：MipNeRF360 等方法在全流程中不使用近似，渲染质量略高
- 启发：对基础操作的数学分析可能带来低成本高回报的改进

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 理论分析深入，最优投影推导漂亮
- 实用性：⭐⭐⭐⭐ — 实现简单但主要在广角场景获益
- 实验质量：⭐⭐⭐⭐ — 多数据集验证，短焦距实验有说服力
- 写作质量：⭐⭐⭐⭐ — 数学推导清晰

<!-- RELATED:START -->

## 相关论文

- [SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [Binomial Self-compensation for Motion Error in Dynamic 3D Scanning](binomial_self-compensation_for_motion_error_in_dynamic_3d_scanning.md)
- [Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)
- [Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)

<!-- RELATED:END -->
