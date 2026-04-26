---
title: >-
  [论文解读] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting
description: >-
  [CVPR2026][3D视觉][3D超分辨率] 将3D超分辨率(3DSR)重新定义为从稀疏低分辨率视图到高分辨率3DGS的**前馈映射**问题，通过高斯偏移学习和特征精炼实现高保真HR 3DGS重建，无需逐场景优化即可实现强零样本泛化。
tags:
  - CVPR2026
  - 3D视觉
  - 3D超分辨率
  - 3D高斯溅射
  - 前馈重建
  - 高斯偏移学习
  - 稀疏视图重建
---

# SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting

**会议**: CVPR2026  
**arXiv**: [2602.24020](https://arxiv.org/abs/2602.24020)  
**代码**: [项目主页](https://xiangfeng66.github.io/SR3R/)  
**领域**: 3d_vision  
**关键词**: 3D超分辨率, 3D高斯溅射, 前馈重建, 高斯偏移学习, 稀疏视图重建

## 一句话总结

将3D超分辨率(3DSR)重新定义为从稀疏低分辨率视图到高分辨率3DGS的**前馈映射**问题，通过高斯偏移学习和特征精炼实现高保真HR 3DGS重建，无需逐场景优化即可实现强零样本泛化。

## 背景与动机

- **核心痛点**: 现有3DSR方法依赖密集LR输入和预训练2D超分模型生成伪HR图像，再用伪HR标签逐场景优化HR 3DGS，存在三大根本限制：
  1. **高频先验受限**: 高频知识仅来源于2DSR模型的先验，无法学习3D特有的高频几何/纹理结构
  2. **重建保真度有天花板**: 伪HR标签质量本身决定了重建上限
  3. **计算开销大**: 密集多视图合成 + 逐场景迭代优化，无法跨场景泛化
- **关键观察**: 前馈3DGS重建模型已能从稀疏视图直接预测3DGS，但其重建质量严重受输入分辨率限制——能否将3DSR也做成前馈映射，从大规模多场景数据中学习3D特有的高频先验？
- **范式转换**: 从"逐场景HR 3DGS自优化"转向"泛化的HR 3DGS前馈预测"，根本改变了3DSR获取高频知识的方式

## 方法详解

### 整体框架

SR3R采用**即插即用**的设计，整体流程分四步：

1. **LR 3DGS重建**: 使用任意前馈3DGS骨干（如NoPoSplat/DepthSplat）从2张LR视图得到LR 3DGS $\mathcal{G}^{\text{LR}}$
2. **高斯密集化**: 通过Gaussian Shuffle Split将 $\mathcal{G}^{\text{LR}}$ 密集化为 $\mathcal{G}^{\text{Dense}}$，作为结构脚手架
3. **映射网络**: LR图像上采样后经ViT编码器+特征精炼+ViT解码器提取多视图融合特征
4. **高斯偏移学习**: 预测从 $\mathcal{G}^{\text{Dense}}$ 到 $\mathcal{G}^{\text{HR}}$ 的残差偏移，得到最终HR 3DGS

核心公式——前馈映射定义：

$$f_{\boldsymbol{\theta}}: \{(\boldsymbol{I}^{v}_{lr}, \boldsymbol{K}^{v})\}_{v=1}^{V} \mapsto \mathcal{G}^{\text{HR}}$$

其中每个3D高斯原语参数化为 $(\boldsymbol{\mu}, \alpha, \boldsymbol{r}, \boldsymbol{s}, \boldsymbol{c})$，分别对应中心位置、不透明度、四元数旋转、缩放和球谐系数。

### 关键设计一：Gaussian Shuffle Split 密集化

对 $\mathcal{G}^{\text{LR}}$ 中每个高斯原语，沿其三个主轴的正负方向生成6个子高斯，提供更精细的结构脚手架：

$$\boldsymbol{\mu}_{j,k} = \boldsymbol{\mu}_j + \beta \, R_j \, \boldsymbol{e}_k \odot \boldsymbol{s}_j, \quad k=1,\dots,6$$

- $R_j$ 为四元数 $\boldsymbol{r}_j$ 对应的旋转矩阵，$\boldsymbol{e}_k$ 为正/负主轴单位向量
- $\beta = 0.5$ 控制偏移幅度；子高斯沿偏移轴的尺度缩小为原来的 $\frac{1}{4}$
- **仅对opacity > 0.5的高斯执行**，聚焦结构显著区域
- 密集化后 $\mathcal{G}^{\text{Dense}}$ 包含 $N = 6M$ 个原语（$M$ 为LR高斯数量）

### 关键设计二：特征精炼模块（Feature Refinement）

上采样后的LR图像包含插值产生的模糊/幻觉高频模式，直接使用会导致3D几何/纹理伪影。特征精炼模块通过**双向交叉注意力**将ViT编码特征与预训练3DGS骨干的几何感知特征对齐：

$$\mathbf{U}_{o \leftarrow p} = \text{softmax}\!\left(\frac{(\boldsymbol{t}_{\text{en}} \boldsymbol{W}^o_Q)(\boldsymbol{t}_{\text{pre}} \boldsymbol{W}^p_K)^\top}{\sqrt{d}}\right)(\boldsymbol{t}_{\text{pre}} \boldsymbol{W}^p_V)$$

$$\mathbf{U}_{p \leftarrow o} = \text{softmax}\!\left(\frac{(\boldsymbol{t}_{\text{pre}} \boldsymbol{W}^p_Q)(\boldsymbol{t}_{\text{en}} \boldsymbol{W}^o_K)^\top}{\sqrt{d}}\right)(\boldsymbol{t}_{\text{en}} \boldsymbol{W}^o_V)$$

两个方向的注意力输出拼接后经全连接层融合，生成精炼特征 $\boldsymbol{t}_{ca}$。核心思路：**将3DGS骨干的可靠3D几何先验传递到2D特征空间**，抑制上采样引入的模糊性。

### 关键设计三：高斯偏移学习（Gaussian Offset Learning）

这是SR3R性能提升最关键的模块。核心思想：**不直接回归绝对高斯参数，而是预测从 $\mathcal{G}^{\text{Dense}}$ 到 $\mathcal{G}^{\text{HR}}$ 的残差偏移**。

具体流程：
1. 将每个密集高斯中心 $\boldsymbol{\mu}_i$ 投影到图像平面获取2D坐标 $\boldsymbol{p}_i$
2. 从ViT解码器特征图 $\boldsymbol{t}_{de}$ 中查询 $\boldsymbol{p}_i$ 位置的局部特征 $\boldsymbol{F}_i$
3. 聚合高斯中心、查询特征和相机内参，送入PointTransformerV3进行空间推理：

$$\boldsymbol{F} = \Phi_{\text{PTv3}}\!\left([\boldsymbol{\mu}_i;\, \{\boldsymbol{F}_i\}_{i=1}^{N};\, \boldsymbol{K}]\right)$$

4. 经Gaussian Head (轻量MLP) 预测残差偏移：

$$\Delta G = (\Delta\boldsymbol{\mu},\, \Delta\boldsymbol{\alpha},\, \Delta\boldsymbol{r},\, \Delta\boldsymbol{s},\, \Delta\boldsymbol{c}) = \Psi_{\text{GH}}(\boldsymbol{F})$$

5. 残差组合得到最终HR 3DGS：

$$\mathcal{G}^{\text{HR}} = \mathcal{G}^{\text{Dense}} + \Delta\mathcal{G}$$

**设计动机**: $\mathcal{G}^{\text{Dense}}$ 已提供可靠的粗结构脚手架，剩余差异主要是局部高频信号。学习偏移而非绝对参数将搜索空间约束在局部，显著提升训练稳定性和重建锐度。

### 关键设计四：ViT解码器跨视图融合

精炼特征 $\boldsymbol{t}_{ca}$ 经ViT解码器处理：
- **视图内自注意力**: 聚合全局上下文信息
- **视图间交叉注意力**: 融合跨视图互补信息，缓解位姿不准确或视图重叠不足导致的不一致性

### 损失函数

采用像素级MSE重建损失和感知一致性LPIPS损失的组合：

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + 0.05 \cdot \mathcal{L}_{\text{LPIPS}}$$

通过可微高斯光栅化端到端训练。

## 实验

### 实验设置

- **数据集**: RealEstate10K (RE10K, 室内)、ACID (室外无人机)、DTU (物体中心)、ScanNet++ (室内)
- **超分倍率**: 4× (64×64 → 256×256)
- **骨干**: NoPoSplat、DepthSplat
- **训练**: 75K迭代，batch=8，lr=2.5e-5，4×RTX 5090

### 主实验结果

| 方法 | 数据集 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯参数量 |
|------|--------|--------|--------|---------|-----------|
| NoPoSplat | RE10K | 21.33 | 0.612 | 0.307 | 2.7M |
| Up-NoPoSplat | RE10K | 23.37 | 0.771 | 0.251 | 44.5M |
| **SR3R (NoPoSplat)** | **RE10K** | **24.79** | **0.827** | **0.188** | **16.5M** |
| DepthSplat | RE10K | 23.15 | 0.699 | 0.281 | 2.3M |
| Up-DepthSplat | RE10K | 24.71 | 0.793 | 0.244 | 38.3M |
| **SR3R (DepthSplat)** | **RE10K** | **26.25** | **0.856** | **0.165** | **14.2M** |
| NoPoSplat | ACID | 21.45 | 0.606 | 0.531 | 2.7M |
| Up-NoPoSplat | ACID | 23.91 | 0.692 | 0.384 | 44.5M |
| **SR3R (NoPoSplat)** | **ACID** | **25.54** | **0.746** | **0.283** | **16.5M** |
| DepthSplat | ACID | 23.80 | 0.624 | 0.437 | 2.3M |
| Up-DepthSplat | ACID | 25.32 | 0.721 | 0.322 | 38.3M |
| **SR3R (DepthSplat)** | **ACID** | **27.02** | **0.797** | **0.261** | **14.2M** |

**关键发现**: SR3R在PSNR上平均提升1.4-3.5dB，同时高斯参数量仅为直接上采样的37%-63%（16.5M vs 44.5M）。

### 零样本泛化实验（RE10K → DTU）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 重建时间 |
|------|--------|--------|---------|---------|
| SRGS (逐场景优化) | 12.42 | 0.327 | 0.598 | 300s |
| FSGS+SRGS (逐场景优化) | 13.72 | 0.444 | 0.481 | 420s |
| NoPoSplat | 12.63 | 0.343 | 0.581 | 0.01s |
| Up-NoPoSplat | 16.64 | 0.598 | 0.369 | 0.16s |
| **SR3R (NoPoSplat)** | **17.24** | **0.607** | **0.291** | **1.69s** |

SR3R不仅超越所有前馈基线，还**超越了需要逐场景优化的SRGS/FSGS+SRGS**（PSNR +3.5dB），且推理速度快177-248倍。

### 消融实验

| 组件 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 高斯参数 |
|------|--------|--------|---------|---------|
| NoPoSplat (基线) | 21.33 | 0.612 | 0.307 | 2.7M |
| + 上采样 | 23.37 | 0.771 | 0.251 | 44.5M |
| + 交叉注意力 | 23.50 | 0.784 | 0.237 | 44.5M |
| + 高斯偏移 (无PTv3) | 24.45 | 0.808 | 0.211 | 16.5M |
| + PTv3 (**完整SR3R**) | **24.79** | **0.827** | **0.188** | **16.5M** |

**关键发现**:
1. **高斯偏移学习贡献最大**: +0.95 PSNR，同时将高斯参数从44.5M降至16.5M
2. 交叉注意力特征精炼提升结构一致性（+0.13 PSNR, LPIPS −0.014）
3. PTv3多尺度空间推理进一步提升锐度（+0.35 PSNR, LPIPS −0.023）
4. 各组件互补，逐步提升重建质量

### 上采样策略鲁棒性

| 上采样方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-----------|--------|--------|---------|
| Bilinear | 24.59 | 0.795 | 0.204 |
| Bicubic | 24.66 | 0.817 | 0.193 |
| SwinIR | 24.79 | 0.827 | 0.188 |
| HAT | 24.78 | 0.819 | 0.183 |

即使使用最简单的Bilinear插值，SR3R也已超越所有前馈基线，表明框架不依赖特定上采样设计。

## 亮点

- 🔄 **范式颠覆**: 将3DSR从"逐场景优化+2DSR伪监督"转向"大规模跨场景前馈预测"，根本改变高频知识获取方式
- 🔌 **即插即用**: 可与任意前馈3DGS骨干配合，设计优雅且实用性强
- 📐 **偏移学习 > 直接回归**: 通过学习残差偏移而非绝对参数，在提升质量的同时将高斯参数减少至37%
- 🎯 **零样本泛化**: 在未见场景上超越逐场景优化方法，且速度快2个数量级
- ⚡ **高效实用**: 从仅2张LR视图即可完成HR 3D重建

## 局限与展望

- 推理时间(1.69s)虽远快于优化方法(300+s)，但相比基础前馈模型(0.01s)仍慢约100倍，实时应用受限
- 仅验证了4×超分，更高倍率(8×/16×)的效果未知
- 密集化策略（固定6个子高斯）较为启发式，自适应密集化可能更优
- 训练需要4×RTX 5090，计算资源门槛较高
- 仅在室内/室外/物体中心场景上验证，大规模户外场景(如自动驾驶)的泛化能力待验证

## 评分

- 新颖性: ⭐⭐⭐⭐ — 3DSR的前馈映射范式转换思路新颖，高斯偏移学习设计巧妙
- 实验充分度: ⭐⭐⭐⭐ — 3个数据集+零样本泛化+消融+上采样策略分析，较为完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，motivation阐述充分，公式规范
- 价值: ⭐⭐⭐⭐ — 为3DSR领域提供了新范式，实用性强且即插即用设计利于推广

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_model_for_feed-forward_4d_synthesis.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Off The Grid: Detection of Primitives for Feed-Forward 3D Gaussian Splatting](off_the_grid_detection_of_primitives_for_feed-forward_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)

<!-- RELATED:END -->
