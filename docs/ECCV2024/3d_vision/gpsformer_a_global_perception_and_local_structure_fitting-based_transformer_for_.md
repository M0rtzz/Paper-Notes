---
title: >-
  [论文解读] GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding
description: >-
  [ECCV 2024][3D视觉][点云理解] 提出GPSFormer，通过全局感知模块(GPM)学习短程和长程依赖关系，并利用Taylor级数启发的局部结构拟合卷积(LSFConv)精确建模局部几何信息，在ScanObjectNN上以2.36M参数达到95.4%准确率，超越所有监督学习和预训练方法。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "点云理解"
  - "全局感知"
  - "可变形图卷积"
  - "局部结构拟合"
  - "Taylor级数"
---

# GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding

**会议**: ECCV 2024  
**arXiv**: [2407.13519](https://arxiv.org/abs/2407.13519)  
**代码**: [https://github.com/changshuowang/GPSFormer](https://github.com/changshuowang/GPSFormer)  
**领域**: 3D视觉  
**关键词**: 点云理解, 全局感知, 可变形卷积, 局部结构拟合, Taylor级数

## 一句话总结

提出GPSFormer，通过全局感知模块(GPM)学习点云短程和长程依赖，结合Taylor级数启发的局部结构拟合卷积(LSFConv)精确捕获局部几何细节，在ScanObjectNN上以纯监督学习方式达到95.4%准确率，超越所有预训练方法。

## 研究背景与动机

**领域现状**: 点云理解在自动驾驶、机器人、公共安全等领域应用广泛，但点云的无序性和不规则性使得有效提取形状信息极具挑战。已有方法包括间接表示（多视图/体素）、直接处理（PointNet系列）、Transformer方法和预训练方法。

**现有痛点**: 
   - 多视图/体素方法丢失几何信息且计算开销大
   - PointNet忽略局部结构信息
   - 局部特征聚合方法忽略点间长程依赖
   - Transformer方法难以同时建模短程依赖、长程依赖和局部结构
   - 预训练方法依赖外部数据，训练成本高

**核心矛盾**: 如何在不依赖外部数据的情况下，同时捕获全局上下文和精细局部结构信息？

**本文目标**: 直接从点云中学习丰富的上下文形状感知和精确的局部几何结构。

**切入角度**: 先用全局感知增强上下文表征，再用Taylor级数多项式拟合精细分析局部结构。

**核心 idea**: 全局感知(自适应可变形图卷积+交叉注意力+多头注意力)引导局部结构拟合(Taylor级数分解为低阶基础+高阶细化)。

## 方法详解

### 整体框架

GPSFormer由GPS Block级联构成，每个GPS Block包含两个核心组件：
- **Global Perception Module (GPM)**: 先通过ADGConv学习短程依赖，再通过RCA和MHA学习长程依赖
- **Local Structure Fitting Convolution (LSFConv)**: 受Taylor级数启发，通过低阶卷积和高阶卷积分别捕获局部结构的基础形状和细节信息

分类任务使用3个GPS Block级联，分割任务使用5个GPS Block构建U-Net编码器。

### 关键设计

1. **Adaptive Deformable Graph Convolution (ADGConv)**: 

    - **功能**: 在特征空间中灵活构建局部邻域，学习相似特征间的短程依赖
    - **核心思路**: 为每个采样点学习一个特征偏移 $\Delta(f_i) = \phi(f_i)$，使得变换后特征 $\hat{f}_i = f_i + \Delta(f_i)$ 可以在整个特征空间"漫游"，以 $\hat{f}_i$ 为中心构建动态图:
    $f_i^a = \mathcal{A}(\{\mathcal{T}(\hat{f}_i, f_j) \mid f_j \in \mathcal{N}(\hat{f}_i)\})$
    - **设计动机**: 传统动态图卷积的感受野受K值影响大，K小则局限于局部坐标邻域，K大则引入语义无关点。自适应偏移使模型能在相似组件之间灵活学习。

2. **Residual Cross-Attention (RCA) + Multi-Head Attention (MHA)**:

    - **功能**: RCA融合变换前后特征，MHA捕获全局长程依赖
    - **核心思路**: 
    $f_i^r = f_i^a + \text{Attn}(\hat{f}_i, f_i^a, f_i^a)$
    $f_i^g = \text{Softmax}\left(\frac{Q_i K^T}{\sqrt{h}}\right) V$
    - **设计动机**: ADGConv提供短程依赖特征，RCA通过交叉注意力丰富上下文理解，MHA进一步建立全局依赖关系，形成"短程→融合→长程"的递进式全局感知。

3. **Local Structure Fitting Convolution (LSFConv)**:

    - **功能**: 基于Taylor级数将局部结构表示分解为低频和高频信息，精细捕获局部几何
    - **核心思路**: Taylor级数 $f(x) = f(a) + \sum_{n=1}^{\infty} a_n (x-a)^n$ 分解为：
        - **低阶卷积(LOConv)**: $f_i^L = \mathcal{A}(\{\phi(f_j)\}_{j=1}^K)$ — 拟合局部结构平坦部分和整体趋势
        - **高阶卷积(HOConv)**: $f_i^H = \mathcal{A}(\{\mathcal{T}(f_i, f_j)\}_{j=1}^K)$ — 拟合边缘和细节
        - 其中仿射基函数:
    $\mathcal{T}(f_i, f_j) = \left(\frac{w_j \cdot (f_j - f_i)}{|w_j \cdot (f_j - f_i)|}\right)^s \cdot |w_j \cdot (f_j - f_i)|^p$
    - $s=1, p=1$ 时退化为ABF; $s=0, p=2$ 时退化为RBF; $p$ 可学习
    - **设计动机**: 低阶项捕获局部基本形状和趋势（低频），高阶项捕获边缘和精细特征（高频）。显式引入几何先验 $h(p_i, p_j) = [p_i, p_j, p_j - p_i, \|p_i, p_j\|]$ 学习权重 $w_j$。

### 损失函数 / 训练策略

- 分类：CrossEntropy损失 + Voting策略可进一步提升性能
- 分割：U-Net风格编解码器 + skip connection + 反向插值
- 多尺度局部邻域：球形查询半径 $\{0.1, 0.2, 0.4\}$，对应邻域点数 $\{8, 16, 32\}$，各stage参数一致

## 实验关键数据

### 主实验

| 数据集 | 指标 | GPSFormer | 之前SOTA(监督) | 之前SOTA(预训练) | 提升 |
|--------|------|-----------|---------------|-----------------|------|
| ScanObjectNN | OA | **95.4%** | DeLA 90.4% | PointGPT 93.6% | +5.0% / +1.8% |
| ScanObjectNN | mAcc | **93.8%** | DeLA 89.3% | ULIP-2 91.2% | +4.5% / +2.6% |
| ModelNet40 | OA | 94.2% | PointMLP 94.5% | PointGPT 94.9% | 竞争力强 |
| ShapeNetPart | cls mIoU | **85.4%** | SPoTr 85.4% | - | 持平 |
| ShapeNetPart | ins mIoU | 86.8% | SPoTr 87.2% | - | 接近 |

### 消融实验

| 配置 | OA(%) | 说明 |
|------|-------|------|
| ADGConv only | 93.2 | 有效提取特征空间局部特征 |
| RCA only | 88.7 | 全局关系建模 |
| MHA only | 89.6 | 长程依赖捕获 |
| ADGConv + RCA | 94.4 | 短程+全局融合互补 |
| ADGConv + MHA | 95.0 | 短程+长程有效组合 |
| ADGConv + RCA + MHA (Full) | **95.4** | 三模块协同最优 |

| HOConv参数设置 | OA(%) | 说明 |
|---------------|-------|------|
| ABF (s=1,p=1) | 92.8 | 基线仿射函数 |
| RBF (s=0,p=2) | 93.2 | 径向基函数 |
| s=0, p learnable | 94.6 | 可学习p提升显著 |
| s=1, p learnable | **95.4** | 最优配置 |

### 关键发现

- GPSFormer仅2.36M参数、0.7G FLOPs，远小于PointMLP(12.6M, 31.4G)，但准确率高出10%
- GPSFormer-elite仅0.68M参数即可达93.3%，超越大部分预训练方法
- ADGConv最优邻域大小K=20，过大或过小都不利于语义信息学习
- t-SNE可视化显示GPSFormer比Point-BERT更好地缩小类内距离，特征分布更紧凑

## 亮点与洞察

- **全局到局部的分层设计**: GPM先建立全局上下文，再用LSFConv精细分析局部结构，形成有效的信息流
- **Taylor级数的创新应用**: 将局部结构表示建模为多项式拟合问题，低阶+高阶分别捕获不同频率信息，既有理论基础又有实践效果
- **参数效率极高**: 2.36M参数实现95.4%准确率，证明精巧的模块设计比堆砌参数更重要
- **纯监督超越预训练**: 不依赖任何外部数据或预训练即超越所有预训练方法，体现了架构本身的强大表达能力

## 局限与展望

- ModelNet40上已接近饱和，该数据集区分能力有限
- 分割任务性能虽然竞争力强但未明显超越SOTA
- 未探索与预训练方法的结合，可能进一步提升上限
- 计算在特征空间建图的额外开销未详细分析

## 相关工作与启发

- **可变形卷积的迁移**: 将2D可变形卷积思想迁移到点云特征空间，通过学习偏移实现灵活的邻域构建
- **Taylor级数在深度学习中的应用**: 为局部特征提取提供了新的数学工具，低/高频分解思路可启发其他领域
- **小模型高性能路线**: 与大模型/预训练趋势形成对比，证明在特定任务上精巧设计可以替代大规模训练

## 评分

- 新颖性: ⭐⭐⭐⭐ ADGConv和Taylor级数启发的LSFConv设计新颖，有理论支撑
- 实验充分度: ⭐⭐⭐⭐⭐ 三个任务+详尽消融+复杂度分析+可视化，非常完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数学推导完整，但部分公式较为繁杂
- 价值: ⭐⭐⭐⭐ 在ScanObjectNN上的突破性结果有重要参考价值，参数效率令人印象深刻
# GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding

**会议**: ECCV 2024  
**arXiv**: [2407.13519](https://arxiv.org/abs/2407.13519)  
**代码**: [https://github.com/changshuowang/GPSFormer](https://github.com/changshuowang/GPSFormer)  
**领域**: 3D视觉  
**关键词**: 点云理解, 全局感知, 可变形图卷积, 局部结构拟合, Taylor级数

## 一句话总结

提出GPSFormer，通过全局感知模块(GPM)学习短程和长程依赖关系，并利用Taylor级数启发的局部结构拟合卷积(LSFConv)精确建模局部几何信息，在ScanObjectNN上以2.36M参数达到95.4%准确率，超越所有监督学习和预训练方法。

## 研究背景与动机

1. **领域现状**: 点云理解已广泛应用于自动驾驶、机器人和公共安全等领域，但由于点云的无序性和不规则性，有效提取形状信息仍是挑战。现有方法包括基于多视图/体素的间接方法、基于局部特征聚合的直接方法、以及基于预训练的方法。
2. **现有痛点**: (1) PointNet等方法忽略了局部结构信息；(2) PointNet++等局部聚合方法忽略了点之间的长程依赖关系；(3) 现有Transformer方法很少同时考虑短程依赖、长程依赖和局部结构建模；(4) 预训练方法虽然提升了性能，但需要大量外部数据且训练时间长。
3. **核心矛盾**: 如何在不依赖外部数据的情况下，同时捕获全局上下文信息和精细的局部几何结构。
4. **本文目标**: 设计一个纯监督学习的Transformer，同时学习全局感知和局部结构拟合，以精确捕获点云的形状信息。
5. **切入角度**: 将全局感知分解为短程依赖(ADGConv)和长程依赖(MHA)，将局部结构建模转化为Taylor级数的多项式拟合问题。
6. **核心 idea**: 用自适应可变形图卷积+多头注意力捕获多尺度全局依赖，用Taylor级数拟合局部几何结构的低阶基础和高阶细节信息。

## 方法详解

### 整体框架

GPSFormer由两个核心组件堆叠构成：**全局感知模块(GPM)** 和 **局部结构拟合卷积(LSFConv)**。分类任务使用3级GPS块级联(特征维度64→128→256)，分割任务使用5级GPS块编码器+U-Net解码器。每个GPS块中，先用GPM进行全局感知，再通过FPS采样+LSFConv进行局部结构拟合。

### 关键设计

1. **自适应可变形图卷积(ADGConv)**: 学习特征空间中相似特征的短程依赖。核心思路是为每个采样点学习一个特征偏移 $\Delta(f_i) = \phi(f_i)$，将特征变换为 $\hat{f}_i = f_i + \Delta(f_i)$，然后以变换后的特征为中心构建动态图进行特征聚合。这种"漫游"机制使点特征能够在整个特征空间中灵活导航，构建合适的局部邻域，避免了K值选取对感受野的影响。设计动机是传统KNN构建的动态图受K值影响大：K小则局限于局部，K大则引入语义无关点。

2. **残差交叉注意力(RCA)**: 融合变换前后的特征信息。公式为 $f_i^r = f_i^a + \text{Attn}(\hat{f}_i, f_i^a, f_i^a)$，其中 $\hat{f}_i$ 作为Query，ADGConv输出 $f_i^a$ 作为Key和Value。通过残差连接增强上下文结构理解。

3. **多头注意力(MHA)**: 捕获特征空间中所有位置间的长程依赖关系。RCA的输出 $f_i^r$ 经过标准自注意力机制 $f_i^g = \text{Softmax}(\frac{Q_i K^T}{\sqrt{h}})V$ 进一步增强全局表征能力。

4. **局部结构拟合卷积(LSFConv)**: 受Taylor级数 $f(x) = f(a) + \sum_{n=1}^{\infty} a_n(x-a)^n$ 启发，将局部结构表示分解为低阶分量和高阶分量：

   $$f(\{f_j\}_{j=1}^{K}) \approx f_i^L + f_i^H = \mathcal{A}(\{\phi(f_j)\}_{j=1}^{K}) + \mathcal{A}(\{\mathcal{T}(f_i, f_j)\}_{j=1}^{K})$$

   其中 $f_i^L$ 为**低阶卷积(LOConv)**，拟合局部结构的平坦部分和整体趋势；$f_i^H$ 为**高阶卷积(HOConv)**，拟合边缘和细节部分。HOConv使用新颖的仿射基函数：

   $$\mathcal{T}(f_i, f_j) = \left(\frac{w_j \cdot (f_j - f_i)}{|w_j \cdot (f_j - f_i)|}\right)^s \cdot |w_j \cdot (f_j - f_i)|^p$$

   当 $s=1, p=1$ 时退化为ABF，$s=0, p=2$ 时退化为RBF，通过可学习参数 $p$ 获得更强的表达能力。

5. **显式几何结构引入**: 权重 $w_j = \xi(h(p_i, p_j))$，其中 $h(p_i, p_j) = [p_i, p_j, p_j - p_i, \|p_i, p_j\|]$，利用采样点与邻近点的坐标交互信息学习权重，增强局部形状感知。

### 损失函数 / 训练策略

- 分类任务使用交叉熵损失，支持voting机制（多次随机采样取多数投票）
- LSFConv在每个阶段采用多尺度策略：球查询构建多尺度半径 $\{0.1, 0.2, 0.4\}$，对应邻域点数 $\{8, 16, 32\}$
- 多尺度参数在各阶段保持一致，避免参数调优

## 实验关键数据

### 主实验

| 数据集 | 指标 | GPSFormer | 之前SOTA | 提升 |
|--------|------|-----------|----------|------|
| ScanObjectNN (PB_T50_RS) | OA | **95.4%** | DeLA 90.4% (监督) / PointGPT 93.6% (预训练) | +5.0% / +1.8% |
| ScanObjectNN (PB_T50_RS) | mAcc | **93.8%** | DeLA 89.3% (监督) | +4.5% |
| ModelNet40 | OA | 94.2% | PointGPT 94.9% (预训练) | 接近饱和 |
| ShapeNetPart | class mIoU | **85.4%** | SPoTr 85.4% | 持平最佳 |
| ShapeNetPart | inst mIoU | 86.8% | SPoTr 87.2% | -0.4% |

### 消融实验

| 配置 | OA (%) | 说明 |
|------|--------|------|
| 仅ADGConv | 93.2 | 有效提取局部特征 |
| 仅RCA | 88.7 | 贡献全局关系 |
| 仅MHA | 89.6 | 捕获依赖关系 |
| ADGConv + RCA | 94.4 | 互补性显著 |
| ADGConv + MHA | 95.0 | 验证有效性 |
| ADGConv + RCA + MHA (完整) | **95.4** | 三者协同最优 |

| HOConv参数设置 | OA (%) | 说明 |
|---------------|--------|------|
| ABF (s=1, p=1固定) | 92.8 | 退化为仿射基函数 |
| RBF (s=0, p=2固定) | 93.2 | 退化为径向基函数 |
| s=0, p可学习 | 94.6 | 自适应参数有效 |
| s=1, p可学习 | **95.4** | 最优配置 |

### 关键发现

- GPSFormer-elite（0.68M参数）仍可达93.3% OA，表明模型设计高效
- ADGConv的邻域大小K=20为最优，过大或过小都会降低性能
- 模型仅2.36M参数、0.7G FLOPS，远低于MVTN(11.2M)和PointMLP(12.6M)
- 在Few-shot设置下（5-way 10-shot）GPSFormer在ScanObjectNN上达89.3%，远超PointNeXt的72.4%

## 亮点与洞察

- **纯监督学习超越预训练方法**：不依赖任何外部数据，仅靠精妙的结构设计就超越了PointGPT等预训练方法，说明模型结构本身的表达能力至关重要
- **Taylor级数的巧妙应用**：将局部结构拟合问题转化为多项式拟合，低阶拟合平坦部分、高阶拟合边缘细节，思路优雅且有效
- **极致的参数效率**：2.36M参数实现95.4%准确率，参数效率极高
- **可变形图卷积的创新**：通过学习特征偏移实现特征空间漫游，比传统KNN更灵活

## 局限与展望

- ModelNet40上性能接近饱和（94.2%），但该数据集已不足以区分方法优劣
- 分割任务性能略低于SPoTr，可能需要针对分割任务优化网络结构
- 多尺度策略参数固定（半径0.1/0.2/0.4），可探索自适应多尺度机制
- 未探索预训练+GPSFormer的结合，可能进一步提升上限

## 相关工作与启发

- 与DGCNN的动态图卷积相比，ADGConv通过特征偏移实现更灵活的邻域构建
- Taylor级数拟合思路可推广到其他需要局部结构建模的任务
- GPM的短程+长程依赖建模框架可启发其他3D理解任务的模型设计

## 评分

- 新颖性: ⭐⭐⭐⭐ Taylor级数拟合局部结构和自适应可变形图卷积都是有新意的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 三个任务+详细消融+few-shot+模型复杂度分析+可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ 条理清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ 纯监督方法超越预训练方法具有重要意义，参数效率出色

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](../../CVPR2025/3d_vision/pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)
- [\[ECCV 2024\] SegPoint: Segment Any Point Cloud via Large Language Model](segpoint_segment_any_point_cloud_via_large_language_model.md)
- [\[CVPR 2025\] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](../../CVPR2025/3d_vision/colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)

</div>

<!-- RELATED:END -->
