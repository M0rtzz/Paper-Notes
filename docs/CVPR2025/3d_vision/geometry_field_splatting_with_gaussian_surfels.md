---
title: >-
  [论文解读] Geometry Field Splatting with Gaussian Surfels
description: >-
  [CVPR 2025][3D视觉][高斯表面元] 本文将 Geometry Field（几何场）理论引入 Gaussian Surfel 框架，推导出高效且近乎精确的可微渲染算法用于不透明表面重建，同时解决了 surfel 聚集时的损失不连续问题，并采用基于反射向量的潜在表示来更好地处理高光表面。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "高斯表面元"
  - "几何场"
  - "表面重建"
  - "可微渲染"
  - "新视角合成"
---

# Geometry Field Splatting with Gaussian Surfels

**会议**: CVPR 2025  
**arXiv**: [2411.17067](https://arxiv.org/abs/2411.17067)  
**代码**: 无  
**领域**: 3D视觉 / 表面重建  
**关键词**: 高斯表面元, 几何场, 表面重建, 可微渲染, 新视角合成

## 一句话总结
本文将 Geometry Field（几何场）理论引入 Gaussian Surfel 框架，推导出高效且近乎精确的可微渲染算法用于不透明表面重建，同时解决了 surfel 聚集时的损失不连续问题，并采用基于反射向量的潜在表示来更好地处理高光表面。

## 研究背景与动机

**领域现状**：从多视角图像重建不透明表面的几何结构是计算机视觉的经典问题。近年来，NeRF 和 3D Gaussian Splatting (3DGS) 带来了体积渲染/辐射场领域的革命——3DGS 通过高斯核的 splatting 实现了快速的新视角合成，而 2DGS/Gaussian Surfel 方法将 3D 高斯退化为 2D 平面 surfel 以更好地拟合表面。

**现有痛点**：（1）3DGS 及其变体使用体积密度（volume density）来建模不透明表面，但体密度是"烟雾"模型——对不透明固体来说是近似的，导致表面提取时出现噪声和厚度；（2）当前的 surfel 渲染算法包含多处近似——Taylor 级数展开、忽略自衰减等，降低了梯度精度；（3）当多个 surfel 靠近同一表面时，渲染颜色关于 surfel 颜色的梯度会出现不连续跳变，导致优化不稳定。

**核心矛盾**：不透明表面需要的是"表面"而非"体积"——体密度模型天然不适合精确表面重建；而 surfel 方法虽然方向正确，但当前渲染算法的近似和不连续性限制了重建精度。

**本文目标**：将几何场（Geometry Field）理论与 Gaussian Surfel 结合，推导出精确的、梯度连续的可微渲染算法，实现高质量不透明表面重建。

**切入角度**：作者从 Geometry Field（最近提出的用于随机不透明表面建模的理论框架）出发。几何场将不透明表面表示为体密度的一种特殊形式，可以转换为体积密度但对表面建模更精确。关键观察是——使用 Gaussian Surfel 来参数化几何场（而非体积密度），可以自然地获得面向表面的归纳偏置。

**核心 idea**：用 Gaussian Surfel 来 splat 几何场而非体积密度，推导出高效精确的渲染公式，解决近似误差和梯度不连续问题。

## 方法详解

### 整体框架
给定多视角图像，模型使用一组 Gaussian Surfel（2D 高斯平面元素）来参数化场景的几何场。每个 surfel 有位置、法线、尺度、颜色/外观等属性。通过几何场的 splatting 渲染公式从任意视角渲染图像，与真实图像比较计算损失，通过梯度反传优化 surfel 参数。最终的表面通过几何场的零水平集提取。

### 关键设计

1. **几何场 Splatting 渲染公式（Geometry Field Splatting）**:

    - 功能：将几何场理论与高斯 surfel 表示结合，推导出精确高效的可微渲染算法
    - 核心思路：传统 3DGS 方法使用 alpha-compositing 来累加高斯核的贡献，其中 alpha 值基于体积密度的近似。本文直接从几何场出发推导渲染方程。几何场 $G(\mathbf{x})$ 定义在 3D 空间中，其值表示点 $\mathbf{x}$ 位于表面内侧的概率。对于由高斯 surfel 参数化的几何场，射线上的几何场值可以解析计算。渲染像素颜色时，沿射线对几何场的导数进行积分：$C = \int_0^{\infty} G'(t) \cdot c(t) dt$，其中 $G'(t)$ 是射线参数 $t$ 处的几何场梯度（即该处是表面的概率密度），$c(t)$ 是对应位置的颜色。这一公式对 Gaussian Surfel 可以高效计算，且移除了现有方法中的 Taylor 展开近似和忽略自衰减的假设
    - 设计动机：精确渲染公式意味着更准确的梯度信号，直接改善优化收敛速度和最终重建质量。移除近似也消除了因近似误差导致的表面噪声

2. **颜色连续性保证（Color Continuity Guarantee）**:

    - 功能：确保当 surfel 聚集靠近表面时渲染颜色是 surfel 颜色的连续函数，避免优化跳变
    - 核心思路：当两个 surfel 沿射线非常接近时，它们的前后排序可能因微小扰动而翻转，导致渲染颜色发生突变（因为 alpha-compositing 是顺序相关的）。作者通过将几何场的渲染积分进行分析，证明了在几何场框架下可以保证渲染颜色关于 surfel 颜色属性的连续性。具体方法是在射线方向上对重叠 surfel 的颜色贡献使用加权平均而非硬排序：$c_{\text{blend}} = \sum_i w_i c_i$，其中权重 $w_i$ 由几何场梯度（而非顺序依赖的 alpha）决定，因此是连续的
    - 设计动机：损失函数的不连续性是梯度优化的大敌——会导致振荡和不收敛。保证连续性使得优化过程更稳定，尤其在 surfel 密集的精细表面区域

3. **基于反射向量的潜在外观表示（Latent Reflection Vector Representation）**:

    - 功能：更好地建模高光/镜面反射表面的视角依赖外观
    - 核心思路：传统 3DGS 使用球谐函数（SH）编码的颜色来表示视角依赖的外观，但 SH 是在球面坐标（视线方向）上展开的。本文改用反射向量（reflection vector）作为 SH 的参数化基础——利用 surfel 法线计算视线的反射方向 $\mathbf{r} = 2(\mathbf{n} \cdot \mathbf{v})\mathbf{n} - \mathbf{v}$，然后在反射向量上展开 SH。此外，不直接预测 RGB 颜色，而是预测一个潜在特征向量（latent），再通过一个小型 MLP 解码为最终颜色
    - 设计动机：高光反射的方向特性与反射向量的相关性远强于与视线方向的相关性。SH 在视线方向上需要高阶才能拟合尖锐高光，但在反射方向上低阶就足够。潜在表示进一步增加了表达能力

### 损失函数 / 训练策略
- L1 颜色重建损失 + SSIM 损失：$L = (1-\lambda)L_1 + \lambda L_{\text{SSIM}}$
- 法线一致性正则化：鼓励相邻 surfel 的法线方向一致
- 深度正则化（可选）：对有深度先验的数据使用
- 采用与 3DGS 相同的自适应密度控制策略（clone/split/prune）

## 实验关键数据

### 主实验
**DTU 数据集（表面重建质量，Chamfer Distance mm）**:

| 方法 | CD↓ | PSNR↑ | SSIM↑ |
|------|-----|-------|-------|
| NeuS | 0.89 | 31.0 | 0.95 |
| 3DGS | 1.52 | 34.2 | **0.97** |
| 2DGS | 0.76 | 32.8 | 0.96 |
| GOF | 0.72 | 33.1 | 0.96 |
| **GFSplatting (Ours)** | **0.63** | **33.5** | 0.96 |

**Mip-NeRF 360 数据集（新视角合成质量）**:

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| 3DGS | 27.4 | 0.81 | 0.22 |
| 2DGS | 27.0 | 0.80 | 0.23 |
| **GFSplatting** | **27.6** | **0.82** | **0.21** |

### 消融实验

| 配置 | DTU CD↓ | PSNR↑ | 说明 |
|------|---------|-------|------|
| Full model | 0.63 | 33.5 | 完整模型 |
| 使用传统体密度 splatting | 0.76 | 33.0 | 退化为 2DGS 类似方法 |
| w/o 颜色连续性 | 0.70 | 33.1 | 表面细节处重建退化 |
| w/o 反射向量表示 | 0.65 | 32.8 | 高光表面渲染质量下降 |
| SH 颜色 (非 latent) | 0.64 | 33.0 | latent 表示提供了额外表达力 |

### 关键发现
- **几何场 splatting 是最大贡献**：相比传统体密度 splatting（2DGS），DTU 上 Chamfer Distance 从 0.76 降到 0.63，提升 17%
- **颜色连续性保证**在 surfel 密集区域（如精细几何结构）改善明显，去掉后 CD 上升 11%
- **反射向量表示**主要在高光物体上有优势，对漫反射表面贡献有限
- 整体方法在表面重建质量（CD）上明显优于现有 GS 方法，同时保持了具有竞争力的渲染质量

## 亮点与洞察
- **从理论到实践**——几何场理论本身是一个优美的数学框架，本文成功地将其落地为实用的可微渲染算法。理论上的精确性直接转化为了实验上的提升
- **连续性分析**很有深度——指出并解决了 surfel 排序不连续导致的优化问题，这是一个被前人忽略但实际影响很大的问题
- **反射向量的使用**虽非全新，但与 latent 表示的结合在 Gaussian Splatting 框架中是首次，可以被其他 GS 方法直接借鉴

## 局限与展望
- 虽然表面重建质量超过了 2DGS/GOF，但新视角合成质量的提升幅度有限（PSNR 仅提升 ~0.5dB）
- 几何场理论目前假设不透明表面，对半透明物体（如烟雾、玻璃）不适用
- 计算开销比 3DGS 稍大（因为更精确的渲染公式），但仍保持实时渲染速率
- 法线估计的精度对反射向量表示的效果有直接影响，法线不准会导致反射方向也不准

## 相关工作与启发
- **vs 2DGS**: 2DGS 同样使用 surfel 但基于体密度 splatting，本文在理论上更精确（几何场vs体密度），实验上重建质量也更优
- **vs GOF (Gaussian Opacity Field)**: GOF 也探索了从 3DGS 提取更好表面的方法，但仍基于体密度框架。GFSplatting 的几何场公式更适合不透明表面
- **vs NeuS/VolSDF**: 这些基于 SDF 的隐式方法也针对表面重建设计，但计算速度慢得多。GFSplatting 继承了 GS 的快速渲染优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 将几何场理论引入 GS 框架，理论贡献扎实，渲染公式推导有深度
- 实验充分度: ⭐⭐⭐⭐ DTU + Mip-NeRF 360 标准评测，消融覆盖各模块
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰严谨，但理论部分门槛较高
- 价值: ⭐⭐⭐⭐ 推动了 GS 表面重建的理论基础，但实际提升幅度中等

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)
- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2025\] Improving Gaussian Splatting with Localized Points Management](improving_gaussian_splatting_with_localized_points_management.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)

</div>

<!-- RELATED:END -->
