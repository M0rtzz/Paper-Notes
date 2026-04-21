---
title: >-
  [论文解读] GaussReg: Fast 3D Registration with Gaussian Splatting
description: >-
  [ECCV 2024][3D视觉][Gaussian Splatting] 首次探索3D Gaussian Splatting场景之间的配准问题，提出粗到精的GaussReg框架——粗阶段利用点云配准方法估计初始变换，精阶段通过渲染图像提取体积特征进行精细对齐，速度比HLoc快44倍且精度可比。
tags:
  - ECCV 2024
  - 3D视觉
  - Gaussian Splatting
  - 3D配准
  - 粗到精
  - 图像引导
  - 场景重建
---

# GaussReg: Fast 3D Registration with Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2407.05254](https://arxiv.org/abs/2407.05254)  
**代码**: https://jiahao620.github.io/gaussreg (有)  
**领域**: 3D视觉  
**关键词**: Gaussian Splatting, 3D配准, 粗到精, 图像引导, 场景重建

## 一句话总结

首次探索3D Gaussian Splatting场景之间的配准问题，提出粗到精的GaussReg框架——粗阶段利用点云配准方法估计初始变换，精阶段通过渲染图像提取体积特征进行精细对齐，速度比HLoc快44倍且精度可比。

## 研究背景与动机

**领域现状**: 点云配准已经发展较为成熟（ICP、GeoTransformer等）。NeRF和3D Gaussian Splatting (3DGS) 作为新兴3D场景表示方法迅速崛起，大规模场景重建中对子场景间的配准需求日益迫切。

**现有痛点**: NeRF的配准方法主要分两类：(a) NeRFuser使用SfM方法对渲染图像重新估计位姿，非常耗时；(b) DReg-NeRF将隐式辐射场转为显式体素再配准，但难以处理无界场景且受体素分辨率限制。

**核心矛盾**: 3DGS提供了类似点云的显式表示，理论上可直接用点云配准方法处理，但GS点云本身噪声大、存在几何畸变，直接粗配准精度不高；而GS内含丰富图像信息尚未被利用。

**本文目标**: 如何快速准确地配准两个3D Gaussian Splatting场景模型？

**切入角度**: 利用GS的双重属性——既有点云的几何结构（用于粗配准），又能高效渲染图像（用于精配准）。

**核心 idea**: 结合点云配准做粗对齐 + 利用GS渲染图像提取体积特征做精对齐的两阶段框架。

## 方法详解

### 整体框架

GaussReg由两个阶段组成：**粗配准（Coarse Registration）** 和 **图像引导精配准（Image-Guided Fine Registration）**。

输入为两个重叠场景A和B的GS模型 $Gaussian_A$ 和 $Gaussian_B$，以及各自训练图像的相机位姿 $\{C_i^A\}$ 和 $\{C_j^B\}$。目标是估计刚性变换 $\{s, R, T\}$（含缩放因子）将场景B对齐到A。

### 关键设计

1. **粗配准模块**: 从GS模型提取点云 $Points_A$, $Points_B$，利用不透明度 $\alpha > 0.7$ 筛选可靠点，每个点以 $(x, y, z, \alpha, r, g, b)$ 作为7通道输入。采用GeoTransformer的工作流——通过共享KPConv-FPN提取多尺度特征，粗层特征 $F^{low}$ 用于超点匹配，细层特征 $F^{high}$ 用于ICP点匹配。训练时对输入GS点云不仅做旋转/平移增强，还做**尺度增强**以应对单目视频重建的尺度不确定性。损失函数沿用GeoTransformer的overlap-aware circle loss和point matching loss。

2. **重叠图像选择（Overlap Image Selection）**: 精配准的前置步骤——在粗配准结果基础上高效选出两个场景中具有高重叠率的相机子集。流程分3步：

    - 从各自训练位姿中均匀采样30个，利用粗变换 $\{s_c, R_c, T_c\}$ 将B的相机对齐到A坐标系；
    - 在30×30的相机对中，先用**相机朝向余弦相似度**筛出top-k对（$k=10$），再利用GS快速渲染低分辨率深度图计算**视角共享面积**，找到重叠最大的相机对 $(C_{i0}^a, \hat{C}_{j0}^b)$；
    - 在最佳相机对附近各选一组训练相机并渲染图像集 $I_A$, $I_B$。

3. **图像引导3D特征提取（I3D Feature Extraction）**: 借鉴多视图立体视觉（MVS）原理。将渲染图像送入2D CNN得到参考特征 $Ref_A$ 和源特征 $\{Src_A^k\}$，通过可微单应性构建cost volume $Cost_A$。经3D CNN正则化后得到概率体 $P_A \in \mathbb{R}^{D \times H \times W}$ 和特征体 $F_A \in \mathbb{R}^{C \times D \times H \times W}$。对每个像素 $p$，选取概率最大的深度假设层：

$$l_0 = \arg\max_{l=0,...,D-1} \{P_A^l(p) + P_A^{l+1}(p)\}$$

并通过概率加权插值得到cost volume特征 $F_A^{Cost}(p)$、深度 $Depth_A(p)$ 和置信度 $Conf_A(p)$。最终将 $Ref_A$、$F_A^{Cost}$ 和 $Depth_A$ 拼接经卷积后，用置信度过滤得到高低分辨率特征：

$$f_A^{high} = Conv(Concat(Ref_A, F_A^{Cost}, Depth_A))[Conf_A > Mean(Conf_A)]$$

再投影到3D空间进行配准。

### 损失函数 / 训练策略

精配准的总损失由两部分组成：

$$L_{total} = \lambda L_{depth} + L_{regis}$$

其中 $\lambda = 10$，$L_{depth}$ 为交叉熵损失监督概率体：

$$L_{depth} = \sum_{p \in \Omega_A} -P_A^{gt}(p) \log P_A(p) + \sum_{p \in \Omega_B} -P_B^{gt}(p) \log P_B(p)$$

$L_{regis}$ 与粗配准中的配准损失一致。两个网络分别训练40个epoch，学习率从 $1e\text{-}4$ 开始每个epoch指数衰减0.05。精配准中渲染 $n=5$ 张图像，深度假设数 $D=64$。

### GS融合与过滤

配准后合并两个GS模型：对 $Gaussian_B$ 的位置、旋转、尺度分别施加变换 $\{s_f, R_f, T_f\}$，其中球谐系数(SH)的旋转利用SH的线性变换性质通过伪逆矩阵计算。融合时保留各自靠近中心的高斯，滤除边界处的冗余高斯。

## 实验关键数据

### 主实验

**ScanNet-GSReg数据集（82个测试场景）**：

| 方法 | RRE↓ | RTE↓ | RSE↓ | 成功率↑ | 耗时(s)↓ |
|------|------|------|------|---------|----------|
| HLoc (SP+SG) | 2.725 | 0.099 | 0.098 | 75.6% | 212.3 |
| FGR | 157.126 | 3.328 | 0.268 | 100% | 3.4 |
| REGTR | 80.095 | 2.768 | 0.408 | 100% | 3.5 |
| **GaussReg** | **2.827** | **0.042** | **0.032** | **100%** | **4.8** |

**Objaverse数据集（44个对象）**：

| 方法 | RRE↓ | ATE↓ |
|------|------|------|
| FGR | 61.59 | 13.50 |
| REGTR | 113.78 | 43.31 |
| DReg-NeRF | 9.67 | 3.85 |
| **GaussReg (w/o fine)** | **2.47** | **3.46** |

### 消融实验

| 配置 | RRE↓ | RTE↓ | RSE↓ | 成功率↑ | 耗时(s)↓ |
|------|------|------|------|---------|----------|
| HLoc | 2.725 | 0.099 | 0.098 | 75.6% | 212.3 |
| 仅粗配准 | 3.403 | 0.061 | 0.034 | 100% | 3.7 |
| 粗配准+HLoc精化 | 1.104 | 0.186 | 0.278 | 51.2% | 206.8 |
| **完整GaussReg** | **2.827** | **0.042** | **0.032** | **100%** | **4.8** |

Top-k重叠图像选择消融：$k=10$ 为最佳平衡点，$k<10$ 精度下降，$k>10$ 收益甚微。

### 关键发现

- GS点云噪声大，传统点云方法（FGR、REGTR）难以直接配准，GaussReg速度比HLoc快**44倍**（4.8s vs 212.3s），且成功率100%
- 图像引导精配准比单纯用HLoc精化效果更好，成功率100% vs 51.2%
- 在Objaverse上仅用粗配准即超越DReg-NeRF，说明对GS特征的利用（opacity通道）有效

## 亮点与洞察

- **首创性**: 首个针对3DGS场景配准的方法，发现GS点云的"双重属性"（几何+渲染）可被充分利用
- **实用价值**: 44倍加速且成功率从75.6%提升到100%，具有显著工程意义
- **设计思路清晰**: 粗到精的框架简洁有效，图像引导利用了GS快速渲染的优势
- **完整benchmark**: 构建了ScanNet-GSReg（1379场景）和GSReg（10个野外场景）两个评测集

## 局限与展望

- GS融合策略简单，两个场景光照条件不同时融合边界可能出现不一致
- 精配准需要训练图像的相机位姿信息，限制了纯GS模型（无位姿信息）的使用
- 未探索端到端训练两个阶段的可能性
- 消融显示去掉I3D特征提取后深度精度更好但配准反而更差，说明图像特征和几何特征的融合机制仍有优化空间

## 相关工作与启发

- **GeoTransformer**: 粗配准的backbone，提供了成熟的点云配准管线
- **DReg-NeRF**: NeRF配准的代表方法，但受限于体素化表示
- **MVSNet**: 多视图立体视觉思路被借鉴用于I3D特征提取
- 启发：GS的快速渲染能力可以为更多下游任务提供额外信息（如深度补全、语义标注等）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个GS配准方法，粗到精思路合理但非颠覆性创新
- 实验充分度: ⭐⭐⭐⭐ 多数据集评测+完整消融+自建benchmark
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ 44倍加速+100%成功率，实用性很强，为GS大规模场景重建奠定基础

<!-- RELATED:START -->

## 相关论文

- [MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo](mvsgaussian_fast_generalizable_gaussian_splatting_reconstruction_from_multi-view.md)
- [TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting](per-gaussian_embedding-based_deformation_for_deformable_3d_gaussian_splatting.md)
- [Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)

<!-- RELATED:END -->
