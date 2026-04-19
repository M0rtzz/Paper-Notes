---
title: >-
  [论文解读] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning
description: >-
  [NeurIPS 2025][自监督学习][自监督学习] 本文首次将自监督学习（SSL）应用于冰晶图像的潜在表征学习，通过在大规模云粒子图像上预训练ViT，学习冰晶形态的连续潜在表征，并用vMF浓度参数量化冰晶多样性，实现30倍计算效率提升的同时取得最佳分类准确率84.39%。
tags:
  - NeurIPS 2025
  - 自监督学习
  - 冰晶形态
  - 气候科学
  - Transformer
  - 数据策化
---

# Understanding Ice Crystal Habit Diversity with Self-Supervised Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.07688](https://arxiv.org/abs/2509.07688)  
**代码**: 无  
**领域**: 自监督学习 / AI for Science  
**关键词**: 自监督学习, 冰晶形态, 气候科学, 视觉Transformer, 数据策化

## 一句话总结

本文首次将自监督学习（SSL）应用于冰晶图像的潜在表征学习，通过在大规模云粒子图像上预训练ViT，学习冰晶形态的连续潜在表征，并用vMF浓度参数量化冰晶多样性，实现30倍计算效率提升的同时取得最佳分类准确率84.39%。

## 研究背景与动机

**领域现状**：云是气候模型中最大的不确定性来源之一，含冰云由于冰晶形态（habit）的高度多样性尤其难以建模。冰晶的微物理特性影响粒子与辐射的相互作用以及气动力学，进而在多尺度上影响全球辐射强迫、降水和云的时空分布。

**现有痛点**：目前对冰晶形态的研究主要依赖云粒子成像仪（CPI）拍摄的数百万张图像。传统方法使用图像处理技术提取几何特征（如长宽比、圆度等），或使用有监督ML进行分类。但这些方法存在两个根本问题：（1）需要大量人工标注，成本极高；（2）依赖预定义的形态类别，无法捕捉连续的形态变化和类内多样性。

**核心矛盾**：冰晶形态本质上是连续分布的，而现有分析方法要么依赖离散类别，要么需要昂贵的人工标注，导致对冰晶多样性的理解受限。

**本文目标** 如何无需人工标注就能学习冰晶形态的有意义表征？如何用数据驱动的方式量化冰晶的形态多样性？

**切入角度**：作者观察到CPI图像天然存在由冰晶形态决定的聚类结构，这与基于聚类的SSL方法（DINO系列）的假设高度吻合。因此可以利用SSL在无标注的情况下学习有物理意义的表征。

**核心 idea**：用DINO系列的自监督ViT在大规模CPI数据集上学习冰晶的连续潜在表征，替代传统的离散分类和几何特征提取方法。

## 方法详解

### 整体框架

输入为320万张无标注的CPI图像（CPI-3M数据集），通过iBOT-vMF自监督方法预训练ViT-Small模型，输出384维的潜在嵌入向量。这些向量可用于下游任务（如形态分类、多样性量化）。整个pipeline分为三个阶段：数据策化→高效预训练→下游应用。

### 关键设计

1. **基于vMF分布的SSL预训练（iBOT-vMF）**:

    - 功能：在无标注CPI图像上学习冰晶形态的潜在表征
    - 核心思路：采用teacher-student自蒸馏框架，student模型学习匹配teacher的聚类分配。关键是引入von Mises-Fisher（vMF）分布的归一化，使嵌入向量自然分布在超球面上。对CPI图像做了特定的数据增强调整：去除饱和度和色调抖动（单色图像）、加入随机垂直翻转（冰晶可自由旋转）、减小随机裁剪的宽高比变化范围（保留冰晶针状特征）
    - 设计动机：vMF分布假设与冰晶的形态聚类结构天然匹配，且vMF的浓度参数$\kappa$可直接用于量化多样性

2. **层级采样数据策化（Hierarchical Sampling）**:

    - 功能：解决CPI数据集的严重类不平衡问题
    - 核心思路：在学到的潜在空间中进行层级采样，从320万张图像中策化出120万张更均匀分布的子集（CPI-H-1M），使各形态类别在潜在空间中更均衡分布
    - 设计动机：DINO系列方法在不平衡数据上预训练效果较差，是已知瓶颈问题。策化后的数据集虽然只有原始的1/3，但训练效果更好

3. **高效预训练策略（ImageNet初始化+短训练）**:

    - 功能：用约30倍更少的计算资源达到最佳性能
    - 核心思路：用ImageNet预训练的iBOT权重初始化模型，仅在CPI-H-1M上微调10个epoch（而非从头训练100个epoch）。利用了ImageNet预训练特征可跨域迁移的发现
    - 设计动机：直接在CPI-3M上预训练100个epoch计算开销大，而ImageNet特征已经能很好地迁移到CPI图像，只需少量领域适应

### 损失函数 / 训练策略

训练使用iBOT的标准交叉熵损失，student网络通过梯度更新，teacher网络通过student的EMA更新。预训练batch size为1024。冰晶多样性用vMF的浓度参数$\hat{\kappa} = \frac{\bar{R}(p - \bar{R}^2)}{1 - \bar{R}^2}$估计，其中$\bar{R}$为归一化嵌入向量的平均长度。

## 实验关键数据

### 主实验

本文的主要评估任务是用学到的表征在CPI-21K（21000张手工标注的测试集）上进行分类：

| SSL方法 | 预训练数据 | Epoch数 | ImageNet初始化 | kNN(%) | 逻辑回归(%) |
|---------|-----------|---------|---------------|--------|------------|
| DINOv3 | LVD-1689M | 1000 | ✗ | 74.83 | 81.83 |
| iBOT | ImageNet | 800 | ✗ | 78.33 | 82.00 |
| iBOT-vMF | CPI-3M | 100 | ✗ | 75.05 | 81.00 |
| iBOT-vMF | CPI-H-1M | 100 | ✗ | 77.67 | 83.17 |
| iBOT-vMF | CPI-H-1M | 10 | ✓ | **81.56** | **84.39** |

对比基线：使用13个几何特征的逻辑回归分类器仅达到65%准确率，远低于SSL表征的84.39%。

### 消融实验

| 配置 | 分类准确率(%) | 说明 |
|------|-------------|------|
| 几何特征基线 | 65.00 | 传统图像处理特征 |
| ImageNet SSL直接用 | 82.00 | 跨域迁移性不错 |
| CPI-3M从头训 | 81.00 | 数据不平衡影响 |
| CPI-H-1M策化后训 | 83.17 | 策化提升+2.17% |
| 策化+初始化+短训 | **84.39** | 30x计算效率提升 |

### 关键发现

- **数据策化贡献最大**：从CPI-3M到CPI-H-1M，在1/3的数据上训练反而效果更好（83.17 vs 81.00），证明类不平衡是SSL的主要瓶颈
- **ImageNet迁移出人意料地好**：纯ImageNet预训练模型在CPI分类上达到82%，说明自然图像特征对CPI图像有良好迁移性
- **PCA投影显示线性可分性**：384维嵌入在PCA投影后呈现清晰的三类聚类，说明学到的特征近似线性可分
- **冰晶多样性随环境变化**：温度升高→多样性增加（$\kappa$降低）；粒子越大→多样性降低（$\kappa$升高）。不同外场试验间差异显著

## 亮点与洞察

- **vMF分布 + 冰晶聚类的天然匹配**：利用vMF的浓度参数$\kappa$直接量化形态多样性，比传统Shannon熵更自然、更连续。这个思路可推广到其他具有聚类结构的科学图像领域
- **"先策化再短训"的高效范式**：在大数据集的潜在空间中做层级采样，然后用ImageNet初始化+短epoch训练，实现30x计算节省。这对计算资源有限的科学领域很有参考价值
- **SSL驱动的"无假设"多样性量化**：不需要预定义形态类别就能量化多样性，避免了人为分类带来的信息损失

## 局限与展望

- **数据集规模有限**：320万张CPI图像对于SSL来说并不算大，可能限制了表征质量
- **仅用ViT-Small**：更大模型可能学到更好的表征，但受限于计算资源
- **下游验证仅限分类**：多样性量化是定性展示，缺少与地面真值的定量对比
- **未探索异常检测和罕见形态发现**：作者提到的未来方向，利用SSL表征检测错标样本或发现罕见冰晶形态
- **缺少与其他SSL方法的充分对比**：如MAE、SimCLR等方法在CPI数据上的表现未知

## 相关工作与启发

- **vs 传统几何特征方法**：传统方法用长宽比等13个手工特征，准确率仅65%，而SSL表征达到84.39%，说明端到端学习的特征远优于手工设计
- **vs DINOv3大规模预训练**：DINOv3在17亿自然图像上预训练，在CPI上反而不如领域内策化数据训练（81.83 vs 84.39），说明领域适应比数据规模更重要
- **vs 有监督CNN分类（Przybylo et al.）**：之前的VGG16有监督方法需要大量标注，SSL方法完全无需标注即可达到可比效果

## 评分

- 新颖性: ⭐⭐⭐ 首次将SSL应用于冰晶形态分析，但SSL方法本身不是新的
- 实验充分度: ⭐⭐⭐ 分类验证和多样性分析都做了，但缺少更多定量对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，问题和方法的连接自然
- 价值: ⭐⭐⭐ 对气候科学领域有实际价值，但方法创新性有限

<!-- RELATED:START -->

## 相关论文

- [T-REGS: Minimum Spanning Tree Regularization for Self-Supervised Learning](t-regs_minimum_spanning_tree_regularization_for_self-supervised_learning.md)
- [Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [Clustering Properties of Self-Supervised Learning](../../ICML2025/self_supervised/clustering_properties_of_self-supervised_learning.md)
- [Collapse-Proof Non-Contrastive Self-Supervised Learning](../../ICML2025/self_supervised/collapse-proof_non-contrastive_self-supervised_learning.md)
- [MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning](../../ICCV2025/self_supervised/mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning.md)

<!-- RELATED:END -->
