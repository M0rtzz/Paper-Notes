---
title: >-
  [论文解读] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection
description: >-
  [NeurIPS 2025 (Workshop: Foundation Models for the Brain and Body)][人体理解][半监督学习] 提出一种半监督框架，利用鼠标轨迹作为弱监督信号预训练眼动表征，然后在标注数据上微调以区分阅读与扫描行为，在推理时仅使用眼动信号，实现免手操作的辅助阅读检测。
tags:
  - "NeurIPS 2025 (Workshop: Foundation Models for the Brain and Body)"
  - 人体理解
  - 半监督学习
  - 眼动追踪
  - 屏幕放大
  - 阅读行为分类
  - 可访问性
---

# Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection

**会议**: NeurIPS 2025 (Workshop: Foundation Models for the Brain and Body)  
**arXiv**: [2509.19574](https://arxiv.org/abs/2509.19574)  
**代码**: 无  
**领域**: Human Understanding / 人机交互  
**关键词**: 半监督学习, 眼动追踪, 屏幕放大, 阅读行为分类, 可访问性

## 一句话总结

提出一种半监督框架，利用鼠标轨迹作为弱监督信号预训练眼动表征，然后在标注数据上微调以区分阅读与扫描行为，在推理时仅使用眼动信号，实现免手操作的辅助阅读检测。

## 研究背景与动机

**领域现状**：屏幕放大是低视力用户阅读的重要辅助手段，但放大后视窗仅能显示几个词或几行文字，用户需要频繁拖动鼠标滚动视窗。

**现有痛点**：大多数基于眼动的自动滚动控制系统依赖手工设计的启发式规则，泛化性差；放大环境下的眼动轨迹碎片化且噪声大，使得仅从眼动区分阅读和扫描非常困难。

**核心矛盾**：精确的意图推断需要高质量的行为标注，但标注成本高昂；鼠标在推理时不可用（目标是免手操作），但鼠标轨迹携带丰富的行为语义信息。

**本文目标**：如何在推理时仅依赖眼动信号，鲁棒地区分阅读（reading）和扫描（scanning）行为。

**切入角度**：利用鼠标运动作为预训练阶段的弱监督目标，学习意图感知的眼动表征；联合建模原始眼动和补偿眼动（compensated gaze）两个互补视角。

**核心idea**：鼠标引导的半监督预训练 + 原始/补偿眼动的双流跨注意力融合。

## 方法详解

### 整体框架

框架分为两阶段：
- **预训练阶段（Pretext）**：用未标注的眼动数据预测鼠标速度（2D 回归任务），学习行为感知的眼动表征。
- **微调阶段（Downstream）**：将预训练好的编码器迁移，替换回归头为分类头，在有标注数据上微调进行阅读/扫描二分类。

两阶段共享相同的骨干网络：双流跨注意力融合 + Transformer 编码器。

### 关键设计

1. **双流眼动输入（Raw + Compensated Gaze）**

    - **功能**：同时输入原始眼动坐标（放大视窗坐标系）和补偿眼动坐标（映射回原始屏幕坐标系）。
    - **为什么**：原始眼动保留局部眼球运动的细粒度动力学特征，补偿眼动恢复全局空间连续性（行和段落的对齐关系），两者互补。
    - **怎么做**：两路眼动分别经三层 1D CNN（kernel=3, 64维）编码，然后通过两个跨注意力块融合（Q=g, K/V=c 和 Q=c, K/V=g），捕获互补信息。
    - **区别**：首次在阅读行为分类任务中融合原始和补偿眼动两个流。

2. **鼠标引导的半监督预训练**

    - **功能**：在预训练阶段，模型学习从无标注眼动序列预测鼠标的二维速度。
    - **为什么**：扫描时鼠标活动显著增加（用户需要重新定位视窗），鼠标运动反映了用户的主动意图决策。这提供了无需人工标注的弱监督信号。
    - **怎么做**：线性回归头 + MSE 损失预测鼠标 2D 速度；微调时替换为分类头 + 交叉熵损失。
    - **区别**：鼠标信号仅在训练时使用，推理时完全基于眼动，实现免手操作。

3. **Transformer 时序建模**

    - 融合后的表征送入三层 Transformer 编码器（64维, 4头注意力），建模时序依赖。
    - 输入窗口为 0.2 秒（120Hz 下 24 步），采用滑动重叠窗口，标签取窗口最后时间点的标注。

### 损失函数 / 训练策略

- **预训练**：MSE 损失，预测鼠标 2D 速度
- **微调**：加权交叉熵损失（处理阅读/扫描的类别不均衡）
- **微调策略**：部分微调（仅更新最后三层 Transformer）vs. 全量微调（更新所有参数）
- 优化器：Adam（lr=3e-4, weight_decay=0.01）
- 评估：留一被试交叉验证（leave-one-subject-out）

## 实验关键数据

### 数据集

基于 Tang et al. 的数据集，包含低视力被试在全屏放大条件下阅读文本文档和网页的同步眼动 + 鼠标记录。眼动 120Hz（Tobii Spectrum），鼠标 10Hz。

### 主实验

#### 不同输入配置的监督学习结果（文本数据集）

| 输入类型 | Overall F1 | Reading F1 | Scanning F1 |
|---------|-----------|-----------|------------|
| Random baseline | 40.91 | 56.04 | 25.79 |
| Compensated only | 67.22 | 87.69 | 46.75 |
| Gaze only | 75.06 | 89.31 | 60.81 |
| **Gaze + Comp. (ours)** | **80.02** | **91.27** | **68.78** |
| Mouse only | 52.85 | 70.29 | 35.41 |
| Mouse + Gaze + Comp. | 83.64 | 91.17 | 76.10 |

#### 半监督 vs 监督学习结果

| 方法 | Text Overall | Text Reading | Text Scanning | Web Overall | Web Reading | Web Scanning |
|-----|-------------|-------------|--------------|------------|------------|-------------|
| Supervised | 80.02 | 91.27 | 68.78 | 62.49 | 60.27 | 64.59 |
| Semi-supervised (Partial) | 81.93 | 91.56 | 72.29 | 64.51 | 62.59 | 66.42 |
| **Semi-supervised (Full)** | **85.97** | **93.13** | **78.80** | **70.01** | **68.39** | **71.62** |

### 消融实验

| 消融维度 | 结果 |
|---------|------|
| 原始眼动 vs 补偿眼动 vs 融合 | 融合（80.02）> 原始（75.06）> 补偿（67.22） |
| 部分微调 vs 全量微调 | 全量（85.97）> 部分（81.93）> 监督（80.02） |
| 鼠标信号贡献 | 鼠标独立弱（52.85），但作为监督信号提升至 85.97 |

### 关键发现

- 半监督预训练在文本数据集上提升 **6.0% F1**（80.02→85.97），在更具挑战性的网页数据集上提升 **7.5% F1**（62.49→70.01）。
- 即使仅部分微调也超过全监督基线，说明预训练学到了高质量的行为表征。
- Scanning 类别（少数类）获益最大：全监督 68.78→半监督 78.80（+10.02 F1）。

## 亮点与洞察

- **鼠标作为弱监督信号的巧妙利用**：鼠标只在训练时用于引导表征学习，推理时完全不需要——这完美契合辅助功能场景下"免手操作"的需求。
- **双流互补建模**：原始眼动和补偿眼动的融合是合理的工程设计，前者保留局部细节，后者恢复全局结构。
- **可访问性应用导向明确**：直接面向低视力用户的屏幕放大场景，有清晰的应用价值。

## 局限与展望

- 仅在全屏放大（full-lens）条件下实验，未测试局部放大等其他放大模式。
- 数据集规模有限（留一被试交叉验证说明被试数少），泛化性有待验证。
- 0.2 秒的窗口长度是否最优没有专门讨论。
- 未与近期的 Transformer 阅读行为分类方法（如 Yang et al. 2025）在相同数据上直接对比。
- 未来可将意图分类器扩展为实时自动滚动控制器。

## 相关工作与启发

- **自监督眼动表征学习**：之前的工作主要用于眼部图像的注视估计或 EOG 信号的粗粒度行为识别，本文首次将半监督学习用于帧级阅读行为分类。
- **鼠标-眼动对齐**：以往主要用于注意力分析和阅读深度预测，本文创新地将鼠标作为预训练目标。
- **对其他辅助技术的启发**：类似的弱监督预训练策略可推广到其他多模态人机交互场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ 鼠标引导预训练的idea简洁有效，双流融合为首次应用
- 实验充分度: ⭐⭐⭐ 作为 workshop paper 实验较完整，但数据规模和基线对比有限
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 面向辅助功能的实用场景，半监督策略可推广

<!-- RELATED:START -->

## 相关论文

- [SignRep: Enhancing Self-Supervised Sign Representations](../../ICCV2025/human_understanding/signrep_enhancing_self-supervised_sign_representations.md)
- [Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](../../ECCV2024/human_understanding/pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)
- [SPROD: Spurious-Aware Prototype Refinement for Reliable Out-of-Distribution Detection](spurious-aware_prototype_refinement_for_reliable_out-of-distribution_detection.md)
- [Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](../../CVPR2025/human_understanding/enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)

<!-- RELATED:END -->
