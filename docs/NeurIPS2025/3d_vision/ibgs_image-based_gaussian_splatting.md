---
title: >-
  [论文解读] IBGS: Image-Based Gaussian Splatting
description: >-
  [NeurIPS 2025][3D视觉][3D高斯泼溅] 提出基于图像的高斯泼溅方法（IBGS），通过从邻近训练图像中学习颜色残差来增强标准3DGS的渲染质量，在不增加存储开销的前提下显著提升高频细节和视角依赖效果的建模能力。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 3D高斯泼溅
  - 新视角合成
  - 图像基渲染
  - 颜色残差
  - 视角依赖效果
---

# IBGS: Image-Based Gaussian Splatting

**会议**: NeurIPS 2025  
**arXiv**: [2511.14357](https://arxiv.org/abs/2511.14357)  
**代码**: [GitHub](https://hoangchuongnguyen.github.io/ibgs) (有)  
**领域**: 3D视觉 / 新视角合成  
**关键词**: 3D高斯泼溅, 新视角合成, 图像基渲染, 颜色残差, 视角依赖效果

## 一句话总结

提出基于图像的高斯泼溅方法（IBGS），通过从邻近训练图像中学习颜色残差来增强标准3DGS的渲染质量，在不增加存储开销的前提下显著提升高频细节和视角依赖效果的建模能力。

## 研究背景与动机

**领域现状**: 3D高斯泼溅（3DGS）已成为新视角合成（NVS）的主流方法，因其高质量渲染和快速优化而广受关注。但每个高斯球在给定视角只能表示单一颜色，且低阶球谐函数（SH）难以捕捉复杂的视角依赖效果。

**现有痛点**: 现有改进方法要么使用全局纹理映射（在复杂场景中失效），要么使用逐高斯纹理映射（存储开销随纹理分辨率平方增长），且仍然无法处理视角依赖效果。

**核心矛盾**: 如何在不显著增加存储的前提下同时建模高频细节和视角依赖效果？

**本文目标**: 利用训练图像中已有的高频细节和视角信息来增强渲染。

**切入角度**: 受传统图像基渲染（IBR）技术启发，将3DGS与图像基渲染结合。

**核心idea**: 像素颜色 = 标准3DGS基础颜色 + 从邻近视角图像学习的颜色残差。

## 方法详解

### 整体框架

IBGS将每个像素的最终颜色建模为两部分之和：

$$\mathbf{c}^{\text{final}}(\mathbf{p}) = \underbrace{\sum_{i=1}^{N} w_i \Psi_l(\mathbf{h}_i, \mathbf{v}_i)}_{\text{基础颜色 } \mathbf{c}(\mathbf{p})} + \underbrace{\mathcal{F}(\mathbf{c}(\mathbf{p}), \mathbf{d}(\mathbf{p}), \{\Delta\mathbf{c}_m\}_{m=1}^{M}, \{\Delta\mathbf{d}_m\}_{m=1}^{M})}_{\text{颜色残差 } \Delta\mathbf{c}(\mathbf{p})}$$

其中基础颜色来自标准3DGS光栅化，颜色残差由轻量级网络从邻近源视图中预测。

### 关键设计

1. **源视图特征提取**:

    - 对目标像素 $\mathbf{p}$，计算射线与高斯球的交点 $\mathbf{x}_i(\mathbf{p})$
    - 交点由射线与高斯中心和法向量定义的平面的交集计算：
    $\mathbf{x}_i(\mathbf{p}) = \mathbf{o} + \frac{\mathbf{n}_i^T(\boldsymbol{\mu}_i - \mathbf{o})}{\mathbf{n}_i^T \mathbf{d}(\mathbf{p})} \mathbf{d}(\mathbf{p})$
    - 仅投影累计透射率接近0.5的 $K$ 个中位交点（过滤浮动高斯噪声）
    - 将交点投影到邻近源视图获取颜色，计算加权平均warped颜色与基础颜色的差值：
    $\Delta\mathbf{c}_m(\mathbf{p}) = \mathbf{c}_m^{\text{warp}}(\mathbf{p}) - \mathbf{c}(\mathbf{p})$

2. **颜色残差预测网络**:

    - 采用**PointNet风格**的逐像素特征提取器处理每个源视图的颜色差异和相机差异特征
    - 对多视图特征做max-pooling聚合，得到 $\mathbf{F} \in \mathbb{R}^{H \times W \times 32}$ 的特征图
    - 使用**9层 $3\times3$ 卷积解码器**预测颜色残差图 $\Delta\mathbf{C}$
    - 网络极轻量，不影响渲染速度

3. **曝光校正模块**:

    - 解决现代相机自动曝光导致的跨视图亮度不一致问题
    - 假设邻近位置有相似光照条件，通过最小二乘拟合仿射变换矩阵：
    $\mathbf{A}^{\star} = \arg\min_{\mathbf{A}} \sum_{\mathbf{p}} \left\| \mathbf{A} \begin{bmatrix} \mathbf{c}(\mathbf{p}) \\ 1 \end{bmatrix} - \mathbf{c}_1^{\text{warp}}(\mathbf{p}) \right\|_2^2$
    - 关键优势：可推广到任意新视角（现有方法仅校正训练视角）

4. **基于可见性的源视图选择**: 通过深度一致性检查排除目标点不可见的源视图：
    $\frac{|z(\mathbf{x}(\mathbf{p})) - z(\mathbf{x}_s^{\text{warp}}(\mathbf{p}))|}{z(\mathbf{x}(\mathbf{p})) + z(\mathbf{x}_s^{\text{warp}}(\mathbf{p}))} \leq \tau$

### 损失函数 / 训练策略

总损失函数包含三项：
$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_1 \mathcal{L}_{\text{photo}} + \lambda_2 \mathcal{L}_{\text{normal}}$$

- **颜色渲染损失** $\mathcal{L}_{\text{rgb}}$：对基础图像和最终图像分别计算L1+SSIM损失，权重 $\gamma$ 从1.0逐步降至0.5
- **多视图颜色一致性损失** $\mathcal{L}_{\text{photo}}$：强制warped颜色与真实图像的一致性，鼓励精确的像素匹配
- **法向一致性损失** $\mathcal{L}_{\text{normal}}$：改善几何质量
- 前7000次迭代仅用RGB损失，之后激活 $\lambda_1=0.3, \lambda_2=0.03$
- SH阶数 $l=2$，中位交点数 $K=4$，候选源视图 $S=4$，可见源视图 $M=3$

## 实验关键数据

### 主实验

在Mip-NeRF360、Tanks and Temples (TNT)、Deep Blending三个标准NVS基准上的对比：

| 方法 | Mip-NeRF360 PSNR↑ | TNT PSNR↑ | Deep Blending PSNR↑ | TNT #Gauss(M) | TNT Mem(MB) |
|------|-------------------|-----------|---------------------|---------------|-------------|
| 3DGS | 27.69 | 23.11 | 29.53 | 1.75 | 415 |
| SuperGauss | 27.31 | 23.72 | 28.83 | 1.50 | 502 |
| TexturedGauss | 27.35 | 24.26 | 28.33 | - | - |
| **IBGS (Ours)** | **28.33** | **24.84** | **30.12** | **0.75** | **143** |

在具有挑战性的Shiny数据集（镜面高光、反射、光盘衍射）上：

| 场景 | 3DGS PSNR | SuperGauss PSNR | **IBGS PSNR** |
|------|----------|-----------------|--------------|
| Guitars（镜面高光） | 29.37 | 30.43 | **35.65** |
| Lab（反射） | 29.17 | 29.38 | **35.06** |
| CD（衍射） | 29.10 | 29.49 | **35.23** |

Shiny数据集上PSNR提升超过**5.2 dB**，同时高斯数量更少。

### 消融实验

| 设置 | TNT PSNR↑ | Mip-NeRF360 PSNR↑ |
|------|----------|-------------------|
| Full | **24.84** | **28.33** |
| 仅基础颜色（无残差） | 23.06 | 27.08 |
| 无颜色一致性损失 | 24.70 | 28.31 |
| 用完整warped颜色代替差值作为输入 | 24.61 | 28.21 |
| 无曝光校正 | 24.28 | - |

### 关键发现

1. 颜色残差模块提供约**1.8 dB**的PSNR提升（TNT），是最关键的组件
2. 使用差值 $\Delta\mathbf{c}_m$ 而非完整warped颜色作为网络输入效果更好
3. 高不透明度剪枝（阈值0.05）下IBGS几乎无质量损失，而3DGS显著退化——说明IBGS的高斯更集中在真实表面附近
4. 在Mip-NeRF360和TNT上，IBGS至少减少**62%**高斯数量和**42%**存储，同时质量更优

## 亮点与洞察

- **基础颜色+残差**的分解思想优雅：基础颜色处理大部分外观，残差补充SH无法捕捉的细节
- 仅投影中位交点的设计巧妙过滤了浮动高斯的噪声，同时促进了高斯向真实表面对齐
- 曝光校正可推广到新视角，解决了现有方法仅能校正训练视角的限制
- 9层轻量CNN + PointNet聚合的残差网络设计在效率和质量间取得出色平衡
- 在Shiny数据集上5+ dB的巨大提升证明了该方法在视角依赖效果建模上的根本性优势

## 局限与展望

1. **稀疏视角场景**表现可能不佳，因为需要密集的像素对应关系来预测残差
2. 额外的渲染计算导致**渲染速度低于**原始3DGS，运行时内存也更高
3. 依赖训练图像作为"纹理源"，在训练视角覆盖不到的区域可能退化
4. 可与高斯压缩/量化方法结合以进一步减少存储

## 相关工作与启发

- **传统IBR**（Light Field, IBRNet）：IBGS巧妙地将IBR"借用像素"的思想与3DGS的高效光栅化结合
- **2DGS**：IBGS借鉴了其中位交点和法向一致性损失的设计
- **TexturedGaussian/SuperGaussian**：逐高斯纹理映射方法，存储开销大且无法处理视角依赖效果
- 启发：在场景表示中，利用已有观测图像作为"纹理库"是比学习纹理映射更高效的策略

## 评分

- 新颖性: ⭐⭐⭐⭐ IBR与3DGS的结合思路新颖，颜色残差设计自然
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集，详细消融和定性分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导完整
- 价值: ⭐⭐⭐⭐⭐ 在多个基准上刷新SOTA，实用价值高

<!-- RELATED:START -->

## 相关论文

- [Anti-Aliased 2D Gaussian Splatting](anti-aliased_2d_gaussian_splatting.md)
- [Gaussian Splatting for Efficient Satellite Image Photogrammetry (EOGS)](../../CVPR2025/3d_vision/gaussian_splatting_for_efficient_satellite_image_photogrammetry.md)
- [Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)

<!-- RELATED:END -->
