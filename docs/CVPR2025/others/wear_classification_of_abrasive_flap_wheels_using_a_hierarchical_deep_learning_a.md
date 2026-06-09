---
title: >-
  [论文解读] Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach
description: >-
  [CVPR 2025][砂轮磨损分类] 本文提出一种基于 EfficientNetV2 的分层视觉分类框架，将砂布翼轮的磨损状态分解为三个层级（使用状态→磨损类型→严重程度），在各子任务上取得 93.8%~99.3% 的分类精度。
tags:
  - "CVPR 2025"
  - "砂轮磨损分类"
  - "层级分类"
  - "迁移学习"
  - "EfficientNetV2"
  - "Grad-CAM"
---

# Wear Classification of Abrasive Flap Wheels using a Hierarchical Deep Learning Approach

**会议**: CVPR 2025  
**arXiv**: [2603.12852](https://arxiv.org/abs/2603.12852)  
**代码**: 无  
**领域**: 其他  
**关键词**: 砂轮磨损分类、层级分类、迁移学习、EfficientNetV2、Grad-CAM

## 一句话总结
本文提出一种基于 EfficientNetV2 的分层视觉分类框架，将砂布翼轮的磨损状态分解为三个层级（使用状态→磨损类型→严重程度），在各子任务上取得 93.8%~99.3% 的分类精度。

## 研究背景与动机

**领域现状**：砂布翼轮（abrasive flap wheel）广泛用于航空航天和模具制造中的复杂自由曲面精加工，由于其柔性特点能适配不同曲面形状。自动化磨削中的工具状态监控是实现自适应过程控制的关键。

**现有痛点**：砂布翼轮因柔性结构产生复杂且高度变异的磨损模式（凹形/凸形轮廓变化、翼片撕裂等），传统基于规则的图像处理方法在面对这种高方差磨损模式时容易出错。现有的 CNN 磨损监测方法主要针对铣刀、车刀等刚性工具，缺乏针对柔性工具磨损特征的专用框架。

**核心矛盾**：砂布翼轮的磨损是多维度的（几何形状、翼片完整性、严重程度），而单一多类别分类器需要为每种磨损组合准备足够训练数据，造成数据碎片化问题。

**本文目标**：设计一个多层级分类框架，将磨损检测分解为多个独立子决策，每个子网络只负责一个特定分类任务。

**切入角度**：不同磨损特征之间存在逻辑依赖关系，层级分解不仅能降低每个子网络的分类难度，还能通过逻辑一致性检查来检测误分类。

**核心 idea**：将砂布翼轮磨损分析分解为三层层级决策树（使用状态→轮廓类型+翼片撕裂→严重程度），每层使用独立的 EfficientNetV2 分类器，并利用层级间逻辑约束提升鲁棒性。

## 方法详解

### 整体框架
输入为砂布翼轮的径向和轴向透视图像，通过三层层级分类依次判断：Level 1 判断使用状态（新/旧），Level 2 判断磨损轮廓类型（矩形/凹形/凸形）和翼片撕裂（有/无），Level 3 对凹形或凸形进行严重程度评估（部分/完全变形）。

### 关键设计

1. **层级分类架构**:

    - 功能：将复杂的多类别磨损分类分解为多个二元或三元子分类任务
    - 核心思路：不同于 monolithic 分类器，层级分类让每个子网络专注于特定磨损维度。Level 1 用径向图像判断新旧，Level 2 同时做轮廓分类和翼片撕裂检测，Level 3 做严重程度细分。各子网络可独立训练，避免类别组合爆炸导致的数据不足。
    - 设计动机：不同磨损特征最适合的拍摄角度不同——翼片撕裂从轴向最易观察，轮廓变化需要径向视角。层级结构允许不同子网络使用不同视角输入。

2. **逻辑一致性检查**:

    - 功能：检测层级间矛盾的分类结果
    - 核心思路：定义了 11 种一致结果路径和 3 种矛盾路径。例如 Level 1 判断为"新"但 Level 2 检测到翼片撕裂或非矩形轮廓即为逻辑矛盾。矛盾结果可触发重新检测或人工审核。
    - 设计动机：利用磨损的物理先验知识来为分类结果提供额外验证，提升工业环境中的可靠性。

3. **基于 EfficientNetV2 的迁移学习**:

    - 功能：在有限数据集上实现高精度分类
    - 核心思路：使用 ImageNet 预训练的 EfficientNetV2 作为特征提取器。翼片撕裂检测使用 EfficientNetV2-L，其余子网络使用 EfficientNetV2-S。使用 AdamW 优化器，学习率和权重衰减均为 $1 \times 10^{-4}$。
    - 设计动机：工业场景中标注数据有限（约 13,240 张图像），迁移学习能充分利用预训练特征。

### 损失函数 / 训练策略
各子网络独立训练，使用标准交叉熵损失。训练过程中对图像进行随机旋转、亮度和对比度调整等数据增强。轴向图像转为灰度图用于翼片撕裂检测。

## 实验关键数据

### 主实验
在 105 个不同磨损状态的砂布翼轮共 13,240 张图像上评估：

| 子任务 | 准确率 | F1-score | AUC |
|--------|--------|----------|-----|
| 使用状态分类 (Level 1) | 98.6% | 0.983 | 0.999 |
| 翼片轮廓分类 (Level 2) | 95.4% | 0.954 | 0.99 |
| 翼片撕裂检测 (Level 2) | 93.8% | 0.935 | 0.98 |
| 凹形严重程度 (Level 3) | 99.3% | 0.993 | 1.00 |
| 凸形严重程度 (Level 3) | 95.0% | 0.948 | ≥0.98 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 层级分类（本文） | 各子任务 93.8-99.3% | 每个子任务独立优化 |
| 逻辑一致性检查 | 可检测矛盾路径 | 提升工业可靠性 |
| EfficientNetV2-L (撕裂) | 93.8% | 更大模型提升难区分任务精度 |
| EfficientNetV2-S (其他) | >95% | 轻量模型已足够 |

### 关键发现
- 翼片撕裂检测（93.8%）是最难的子任务，误分类的平均置信度（0.76）明显低于总体平均（0.92），可利用置信度阈值进一步过滤
- 凸形轮廓分类中"完全→部分"的单向误分类模式表明部分与完全凸形之间的视觉边界是渐变的
- Grad-CAM 可视化验证了模型学到了物理相关特征（翼片边缘形状、轮廓变化区域）
- 旋转对称性导致部分凹凸误分类，建议增加垂直翻转数据增强

## 亮点与洞察
- **层级分解思路**：将复杂的多属性分类任务分解为多个简单子决策，避免了类别组合爆炸的数据需求问题，可迁移到其他工业检测场景。
- **逻辑一致性检查**：利用领域先验在层级间建立约束关系检测误分类，是一种轻量但有效的鲁棒性提升方式。
- **Grad-CAM 验证物理特征**：不仅用于解释模型，还验证模型是否学到物理相关的磨损特征。

## 局限与展望
- 数据集规模较小（约 13K 图像），小样本类别泛化能力有待验证
- 缺乏与 monolithic 多类别分类器的直接对比实验
- 翼片撕裂检测精度最低，未尝试注意力机制或目标检测方法
- 层级间错误传播问题未深入分析

## 相关工作与启发
- **vs 直接多类别分类**: 层级分类分解降低子任务难度和数据需求，但可能引入错误传播
- **vs 间接方法（力/振动信号）**: 直接视觉检测解耦了监测与工件几何复杂度
- 这篇论文展示了工业检测问题与深度学习结合的实用案例

## 评分
- 新颖性: ⭐⭐⭐ 方法上无突破，主要是层级分类+迁移学习的工程组合
- 实验充分度: ⭐⭐⭐ 缺少与端到端多类别分类器的对比
- 写作质量: ⭐⭐⭐⭐ 工业背景描述清晰，Grad-CAM 分析详尽
- 价值: ⭐⭐⭐ 对砂布翼轮磨损检测有实用价值，通用性有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](potential_field_based_deep_metric_learning.md)
- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[ICML 2025\] Learning-Augmented Hierarchical Clustering](../../ICML2025/others/learning-augmented_hierarchical_clustering.md)
- [\[ICML 2025\] Function Encoders: A Principled Approach to Transfer Learning in Hilbert Spaces](../../ICML2025/others/function_encoders_a_principled_approach_to_transfer_learning_in_hilbert_spaces.md)
- [\[AAAI 2026\] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment](../../AAAI2026/others/deep_incomplete_multi-view_clustering_via_hierarchical_imputation_and_alignment.md)

</div>

<!-- RELATED:END -->
