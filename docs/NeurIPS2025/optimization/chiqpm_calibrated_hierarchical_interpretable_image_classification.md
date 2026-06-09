---
title: >-
  [论文解读] CHiQPM: Calibrated Hierarchical Interpretable Image Classification
description: >-
  [NeurIPS 2025][优化/理论][可解释机器学习] CHiQPM 提出一种校准的层次化可解释图像分类方法，通过二次规划选择和分配特征给类别，构建层次化解释路径，并内置可解释的 Conformal Prediction 集合预测，在保持黑盒模型 99% 准确率的同时提供全局和局部可解释性。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "可解释机器学习"
  - "层次化解释"
  - "共形预测"
  - "图像分类"
  - "人机互补"
---

# CHiQPM: Calibrated Hierarchical Interpretable Image Classification

**会议**: NeurIPS 2025  
**arXiv**: [2511.20779](https://arxiv.org/abs/2511.20779)  
**代码**: 无  
**领域**: 可解释AI / 图像分类  
**关键词**: 可解释机器学习, 层次化解释, 共形预测, 图像分类, 人机互补

## 一句话总结

CHiQPM 提出一种校准的层次化可解释图像分类方法，通过二次规划选择和分配特征给类别，构建层次化解释路径，并内置可解释的 Conformal Prediction 集合预测，在保持黑盒模型 99% 准确率的同时提供全局和局部可解释性。

## 研究背景与动机

**领域现状**：深度学习在医疗诊断、自动驾驶等安全关键领域愈发普及，可解释模型（interpretable-by-design）是实现可信 AI 的重要方向。QPM 系列方法通过紧凑的二值类别表示（每个类仅用少量特征）提供全局可解释性，能对比地解释类间差异。

**现有痛点**：
   - QPM 虽然能生成对比性类别表示，但这种对比性比较稀缺（如 CUB-2011 上平均仅 0.13% 的类对具有对比性）
   - 现有可解释模型缺乏局部解释的确定性量化——saliency map 无法传达置信度
   - 原型网络（如 ProtoTree）看似可解释，但其相似度空间是自由学习的，导致人类无法真正预测其行为
   - Conformal Prediction (CP) 方法能提供集合预测保证，但预测集通常包含语义不相关的类别

**核心矛盾**：可解释性与准确率之间的 trade-off——更紧凑的表示更易解释但准确率受限；CP 方法生成的集合缺乏语义连贯性

**本文解决什么**：
   - 如何在保持高准确率的同时增加对比性类对比例？
   - 如何提供层次化的局部解释，类似人类推理过程？
   - 如何让 CP 集合预测产生语义连贯的类别集合？

**切入角度**：将特征检测与层次化类别结构结合，利用 QPM 的二值特征分配自然形成类别层次树，在该树上进行遍历即可得到语义连贯的集合预测。

**核心idea**：在 QPM 框架中引入层次化约束和 Feature Grounding Loss，使特征分配自然形成可遍历的类别层次结构，将 CP 融入这一层次结构，实现可解释的校准集合预测。

## 方法详解

### 整体框架

CHiQPM 的 pipeline 分为五个阶段：
1. **训练稠密模型**：使用 Feature Diversity Loss $\mathcal{L}_{div}$ 训练黑盒基础模型，确保初始特征图激活在图像的不同位置
2. **计算 QP 常量**：计算类-特征相似度矩阵 $\mathbf{A}$、特征-特征相似度矩阵 $\mathbf{R}$ 和线性偏置项 $\mathbf{b}$，以及相似类集合 $\mathbb{K}$
3. **求解带层次约束的 QP**：在 QPM 的基础上添加层次化约束，联合优化特征选择 $\mathbf{s} \in \{0,1\}^{n_f}$ 和类-特征分配 $\mathbf{W} \in \{0,1\}^{n_c \times n_f}$
4. **微调特征**：使用 Feature Grounding Loss $\mathcal{L}_{feat}$ 配合 ReLU 激活微调特征
5. **校准**：使用 CP 方法校准层次化集合预测器

### 关键设计

**带层次约束的二次规划**：QPM 原始的 QP 目标是最大化选定特征与类别的相关性，同时保持特征的独特性和局部性。CHiQPM 在此基础上增加了层次化约束，要求特征分配 $\mathbf{W}^*$ 能在类别间形成有意义的层次结构。具体来说，QP 约束确保每个类恰好被分配 $k$ 个特征（紧凑性），且这些特征的组合能唯一标识该类。层次约束还鼓励相似类共享更多特征，形成类别树种具有语义意义的分组。

**Feature Grounding Loss $\mathcal{L}_{feat}$**：这是本文的关键创新之一。传统特征可能是多义的（polysemantic），即一个特征同时检测多个人类概念。$\mathcal{L}_{feat}$ 配合 ReLU 激活函数，促使学到的特征更具接地性（grounded）和稀疏性。这意味着每个特征更可能对应单一的人类可理解概念（如"红眼"特征在 Figure 1 中区分两种黑鸟）。ReLU 的引入确保特征激活是非负的，使得"未检测到特征"有明确的语义含义。

**层次化局部解释**：给定一个输入图像，CHiQPM 构建一个样本特定的解释层次树。每个层级 $n$ 对应于使用前 $n$ 个最显著特征进行分类：
- 层级 1：仅用最强特征，可能匹配多个共享该特征的类
- 层级 2：加入第二个特征，进一步缩小候选类集合
- 直到层级 $k$：唯一确定一个类

这种层次结构回答了多个问题：这张图片中检测到了哪些有意义的特征？每个特征如何逐步缩小候选类集合？应该预测哪个集合以保证目标覆盖率？

**内置可解释 Conformal Prediction**：CHiQPM 独特地将 CP 融入其层次化解释中。预测集定义为在层次树的某个层级 $n$ 与最可能类共享前 $n$ 个显著特征的所有类。通过 CP 校准，系统可以动态为每个样本选择合适的层级：对于容易的样本，可能只需要层级 $k$（单个类）；对于困难的样本，可能退回到层级 2 或 1，预测一个语义连贯的类组（如所有黑鸟）。

### 损失函数 / 训练策略

总训练流程：
1. 阶段一：$\mathcal{L}_{CE} + \lambda_{div} \mathcal{L}_{div}$ 训练稠密模型
2. 阶段二：求解 QP 获得最优特征选择和分配
3. 阶段三：$\mathcal{L}_{CE} + \lambda_{feat} \mathcal{L}_{feat}$ 微调压缩模型，配合 ReLU 使特征更接地

## 实验关键数据

### 主实验

CHiQPM 在多个数据集和架构上评估，核心结果：

| 数据集 | 架构 | 黑盒准确率 | CHiQPM准确率 | 准确率保持比 | 对比性类对比例 |
|--------|------|-----------|-------------|-------------|-------------|
| CUB-2011 | ResNet-50 | ~82% | ~81% | 99%+ | 显著提升 vs QPM |
| ImageNet-1K | ResNet-50 | ~76% | ~75% | ~99% | 对比性差距减半 |
| CUB-2011 | 多架构 | 各异 | 接近黑盒 | 99%+ | 大多数类获得对比解释 |

关键发现：CHiQPM 作为点预测器达到 SOTA 准确率，保持了非可解释模型 99% 以上的准确率。在 ImageNet-1K 上，与黑盒基线的差距减少了一半以上。

### 消融实验

| 组件 | 对比性比例变化 | 准确率变化 | 覆盖率效率 |
|------|-------------|-----------|-----------|
| 基础 QPM | 基线 | 基线 | - |
| + 层次约束 | ↑ 显著 | 持平 | 可用 |
| + Feature Grounding Loss | ↑ | 持平或↑ | 改善 |
| + ReLU 激活 | ↑ | 持平 | 改善 |
| 完整 CHiQPM | 最优 | 最优 | 最优 |

CP 集合预测效率对比：CHiQPM 的内置 CP 方法在 CUB-2011 上（每类 5/50 个特征）的覆盖率-集合大小曲线与标准 CP 方法（THR、APS）竞争性媲美，同时预测集具有语义连贯性。

### 关键发现

1. **可解释性无需牺牲准确率**：CHiQPM 保持 99% 以上的黑盒模型准确率，同时提供全面的全局和局部可解释性
2. **对比性大幅提升**：相比 QPM，CHiQPM 显著增加了具有对比性解释的类对比例，使更多类别可以通过特征差异来区分
3. **层次解释的认知合理性**：层次化解释更接近人类推理方式——先识别大类（黑鸟），再细分具体种类
4. **语义连贯的集合预测**：CP 集合内的类别在层次结构中相邻，具有语义相似性

## 亮点与洞察

- **将 CP 与可解释模型自然融合**：不是事后套用 CP，而是模型结构本身支持层次化集合预测，这是一个优雅的设计
- **Feature Grounding Loss 解决多义性问题**：直接针对可解释 ML 中特征多义性这一根本问题提出解决方案
- **可扩展到 ImageNet**：在 1000 类的 ImageNet 上仍能工作，证明了方法的可扩展性
- **人机互补视角**：不同于替代人类的 AI，CHiQPM 设计为辅助人类专家决策的工具

## 局限与展望

- QP 求解的计算开销随类别数增长，在超大规模分类任务上可能受限
- 特征的语义接地性仍依赖于基础模型的表示质量
- 层次深度与特征数 $k$ 直接相关，在特征极少时层次结构可能过于浅
- 论文主要在视觉分类任务上验证，其他模态（文本、多模态）的适用性待探索

## 相关工作与启发

- QPM / Q-SENN 系列：CHiQPM 的直接前身，本文在其基础上增加层次化和校准能力
- Conformal Prediction：分布无关的预测集方法，CHiQPM 首次将其与可解释模型内在结合
- 原型网络（ProtoPNet, ProtoTree）：另一类可解释模型，但相似度空间的不可预测性是其固有问题
- Concept Bottleneck Models（CBM）：使用人类概念作为瓶颈，但需要概念标注

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 层次化 CP + 可解释模型的融合是创新点
- **技术深度**: ⭐⭐⭐⭐ — QP 优化与 CP 理论结合紧密
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集多架构验证，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图示直观
- **实用性**: ⭐⭐⭐⭐ — 不牺牲准确率的可解释性在安全关键领域有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit](../../ICCV2025/optimization/learning_interpretable_queries_for_explainable_image_classification_with_informa.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

</div>

<!-- RELATED:END -->
