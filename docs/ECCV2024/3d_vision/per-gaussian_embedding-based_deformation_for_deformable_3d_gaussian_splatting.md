---
title: >-
  [论文解读] Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][可变形3DGS] 提出基于逐高斯嵌入（Per-Gaussian Embedding）的形变表示方法，将形变定义为逐高斯潜在嵌入与时间嵌入的函数，辅以粗细形变分解和局部平滑正则化，在多个动态场景数据集上取得了质量、速度和模型容量的全面优势。
tags:
  - ECCV 2024
  - 3D视觉
  - 可变形3DGS
  - 动态场景重建
  - 逐高斯嵌入
  - 粗细形变分解
  - 新视角合成
---

# Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2404.03613](https://arxiv.org/abs/2404.03613)  
**代码**: [项目页面](https://jeongminb.github.io/e-d3dgs/)  
**领域**: 3D视觉  
**关键词**: 可变形3DGS, 动态场景重建, 逐高斯嵌入, 粗细形变分解, 新视角合成  

## 一句话总结

提出基于逐高斯嵌入（Per-Gaussian Embedding）的形变表示方法，将形变定义为逐高斯潜在嵌入与时间嵌入的函数，辅以粗细形变分解和局部平滑正则化，在多个动态场景数据集上取得了质量、速度和模型容量的全面优势。

## 研究背景与动机

1. 3D高斯溅射（3DGS）凭借可微分光栅化实现了实时、高质量的新视角合成
2. 将 3DGS 扩展到动态场景的自然方式是：定义标准3DGS + 形变场将其变形到各帧
3. 现有方法（4DGaussians, D3DGS）将形变场建模为基于坐标的函数 $f(x, y, z, t)$

**核心问题**：3DGS 本质是多个高斯场的混合体，而非单一坐标场；用单一坐标网络预测所有高斯的形变是不合理的
5. 基于坐标的方法受限于特征网格的分辨率和模型容量，即使加倍网格分辨率也仅有微小提升
6. 复杂动态场景（如快速运动的手、反射表面）中，现有方法产生模糊或失真的结果

## 方法详解

### 整体框架

E-D3DGS 框架包含三个核心组件：

1. **逐高斯嵌入**：为每个3D高斯分配独立的可学习潜在嵌入向量
2. **粗细形变分解**：通过两个解码器分别建模慢速/大幅运动和快速/精细运动
3. **局部平滑正则化**：鼓励邻近高斯具有相似的嵌入，从而产生相似的形变

### 关键设计

#### 模块一：基于嵌入的形变

摒弃基于坐标的形变场，引入以逐高斯嵌入 $\mathbf{z}_g \in \mathbb{R}^{32}$ 和时间嵌入 $\mathbf{z}_t \in \mathbb{R}^{256}$ 为输入的形变函数：

$$\mathcal{F}_\theta: (\mathbf{z}_g, \mathbf{z}_t) \rightarrow (\Delta\mathbf{x}, \Delta\mathbf{r}, \Delta\mathbf{s}, \Delta\sigma, \Delta Y)$$

其中 $\Delta\mathbf{x}$ 为位置偏移，$\Delta\mathbf{r}$ 为旋转四元数变化，$\Delta\mathbf{s}$ 为缩放变化，$\Delta\sigma$ 为不透明度变化，$\Delta Y$ 为球谐系数变化。$\mathcal{F}_\theta$ 实现为浅层 MLP + 参数专用 MLP head。

每个高斯在帧 $t$ 的参数通过将形变叠加到标准参数上得到。该设计的关键优势在于：每个高斯有独立的嵌入来编码其时间变化特性，无需依赖空间坐标。

#### 模块二：粗细形变分解

动态场景中不同部分具有不同速度的运动（如身体缓慢移动 vs 手臂快速翻炒）。为此将时间嵌入分为粗和细两种：

**时间嵌入网格**：起始为1D特征网格 $Z \in \mathbb{R}^{N \times 256}$（$N$ 帧）

- **细形变时间嵌入**：$\mathbf{z}_t^f = \text{interp}(Z, t)$，使用原始分辨率
- **粗形变时间嵌入**：将 $Z$ 线性下采样5倍去除高频，再插值得到 $\mathbf{z}_t^c$

最终形变为两个解码器输出之和：

$$\Delta = \mathcal{F}_{\theta_c}(\mathbf{z}_g, \mathbf{z}_t^c) + \mathcal{F}_{\theta_f}(\mathbf{z}_g, \mathbf{z}_t^f)$$

- **粗形变解码器** $\mathcal{F}_{\theta_c}$：负责大尺度/慢速运动（如头部和躯干移动）
- **细形变解码器** $\mathcal{F}_{\theta_f}$：负责快速/精细运动（如手臂、夹子、阴影）

训练初期，$Z_t^f$ 从与 $Z_t^c$ 相同的 $N/5$ 分辨率开始，在10K迭代内逐步增长到 $N$ 分辨率。

#### 模块三：局部平滑正则化

构成动态物体的邻近高斯应具有局部相似的形变。通过约束逐高斯嵌入的局部平滑性来隐式实现：

$$\mathcal{L}^{\text{emb\_reg}} = \frac{1}{k|\mathcal{S}|} \sum_{i \in \mathcal{S}} \sum_{j \in \text{KNN}_{i;k}} w_{i,j} \|\mathbf{z}_{g_i} - \mathbf{z}_{g_j}\|_2$$

其中权重因子 $w_{i,j} = \exp(-\lambda_w \|\mu_j - \mu_i\|_2^2)$，$\lambda_w = 2000$，$k = 20$。KNN 集合仅在密度化发生时更新，以降低计算成本。

与直接约束物理属性（刚性、旋转）的方法不同，本文通过嵌入空间的平滑性隐式约束形变一致性，允许更好地捕捉动态物体的纹理和细节。

### 损失函数 / 训练策略

- **渲染损失**：L1 + 周期性 DSSIM
- 不透明度正则化权重 $1.0 \times 10^{-4}$，取消了3DGS原始的 opacity reset
- 解码器 MLP：128 隐藏单元，2层
- 解码器学习率从 $1.6 \times 10^{-4}$ 指数衰减到 $1.6 \times 10^{-5}$
- 逐高斯嵌入学习率 $2.5 \times 10^{-3}$
- **高效训练策略**：基于误差的帧采样 + 摄像机视角均匀覆盖 + 周期性 DSSIM 密度化

## 实验关键数据

### 主实验

**Neural 3D Video 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | 模型大小↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| DyNeRF | 29.58 | - | 0.083 | 0.01 | 56 MB |
| K-Planes | 30.86 | 0.939 | 0.096 | 0.13 | 309 MB |
| 4DGS | 31.19 | 0.940 | 0.051 | 33.7 | 8700 MB |
| 4DGaussians | 30.71 | 0.935 | 0.056 | 51.9 | 59 MB |
| **Ours** | **31.31** | **0.945** | **0.037** | **74.5** | **35 MB** |

**Technicolor Light Field 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | 模型大小↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| DyNeRF | 31.80 | - | 0.140 | 0.02 | 0.6 MB |
| HyperReel | 32.32 | 0.899 | 0.118 | 0.91 | 289 MB |
| 4DGaussians | 29.62 | 0.844 | 0.176 | 34.8 | 51 MB |
| **Ours** | **33.24** | **0.907** | **0.100** | **60.8** | **77 MB** |

**HyperNeRF 数据集**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | 模型大小↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| HyperNeRF | 22.29 | 0.598 | 0.153 | <1 | 15 MB |
| D3DGS | 22.40 | 0.598 | 0.275 | 6.95 | 309 MB |
| 4DGaussians | 25.03 | 0.682 | 0.281 | 96.3 | 60 MB |
| **Ours** | **25.43** | **0.697** | **0.231** | **139.3** | **33 MB** |

### 消融实验

**粗细形变分解消融**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| **Ours (粗+细)** | **29.70** | **0.933** | **0.041** |
| 仅粗形变 | 29.48 | 0.931 | 0.044 |
| 仅细形变 | 29.23 | 0.932 | 0.043 |
| Ours + 坐标注入 | 29.60 | 0.931 | 0.045 |

**局部平滑正则化消融**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| 无嵌入正则 | 32.26 | 0.951 | 0.037 |
| **+ 嵌入正则（ours）** | **32.34** | **0.952** | **0.036** |
| + 物理约束正则 | 32.08 | 0.950 | 0.036 |

### 关键发现

1. **坐标注入有害**：将高斯坐标 $\mathbf{x}$ 作为额外输入注入解码器反而降低质量（29.60 vs 29.70），验证了应当去除坐标依赖的论点
2. **粗细分解互补**：仅用粗或细解码器都会导致动态区域模糊，组合使用效果最佳
3. **嵌入正则优于物理正则**：隐式的嵌入平滑约束（+0.08 PSNR）优于直接的物理刚性约束（-0.18 PSNR），因为后者过度限制了形变的灵活性
4. **嵌入维度影响小**：$\mathbf{z}_g$ 从16维到64维，$\mathbf{z}_t$ 从128到512维，性能变化不超过 0.18 PSNR
5. **全面优势**：在三个数据集上同时取得最优质量、最快渲染和最小模型的平衡

## 亮点与洞察

- 从 3DGS 的本质出发（高斯混合体而非单一场），为每个高斯定义独立形变，设计逻辑清晰
- 粗细形变分解通过简单的时间网格下采样实现，无需复杂的多尺度架构
- 模型仅 35MB（Neural 3D Video），显著小于 4DGS 的 8700MB，体现了嵌入表示的高效性
- 渲染速度达 74.5 FPS，远超 NeRF 基线和部分 3DGS 基线

## 局限性

- 在随意拍摄的单目视频上（如 HyperNeRF 中的挑战设置）仍然效果不佳
- 逐高斯嵌入的内存占用随高斯数量线性增长
- KNN 近邻查找仅在密度化时更新，可能导致快速变形场景中的不一致
- 需要 COLMAP 重建的初始点云，对初始化质量有依赖

## 相关工作与启发

- **4DGaussians**：使用 HexPlane 解码特征进行时间形变，坐标依赖的代表方法
- **D3DGS**：使用隐式函数处理时间和位置，同为形变3DGS方法
- **Nerfies / HyperNeRF**：使用逐帧形变码，但用于 NeRF 而非 3DGS
- **Dynamic 3D Gaussians**：引入局部刚性正则化，启发了本文的局部平滑正则化
- **启发**：逐元素嵌入的思路可推广到其他基于离散元素的表示（如 3D Mesh 形变）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] SAGS: Structure-Aware 3D Gaussian Splatting](sags_structure-aware_3d_gaussian_splatting.md)
- [\[ECCV 2024\] On the Error Analysis of 3D Gaussian Splatting and an Optimal Projection Strategy](on_the_error_analysis_of_3d_gaussian_splatting_and_an_optimal_projection_strateg.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[ECCV 2024\] MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images](mvsplat_efficient_3d_gaussian_splatting_from_sparse_multi-view_images.md)
- [\[ECCV 2024\] Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Splatting](pixel-gs_density_control_with_pixel-aware_gradient_for_3d_gaussian_splatting.md)

<!-- RELATED:END -->
