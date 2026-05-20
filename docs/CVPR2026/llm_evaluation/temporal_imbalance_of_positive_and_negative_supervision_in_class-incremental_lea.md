---
title: >-
  [论文解读] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning
description: >-
  [CVPR2026][LLM评测][Class-Incremental Learning] 提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重…
tags:
  - "CVPR2026"
  - "LLM评测"
  - "Class-Incremental Learning"
  - "catastrophic forgetting"
  - "Temporal Imbalance"
  - "Loss Reweighting"
  - "continual learning"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning

**会议**: CVPR2026  
**arXiv**: [2603.02280](https://arxiv.org/abs/2603.02280)  
**代码**: 待确认  
**领域**: LLM评测  
**关键词**: Class-Incremental Learning, catastrophic forgetting, Temporal Imbalance, Loss Reweighting, continual learning

## 一句话总结

提出时序不平衡（Temporal Imbalance）这一被忽视的类增量学习偏差来源，并设计 Temporal-Adjusted Loss（TAL）通过时间衰减记忆核动态降低旧类的负监督权重，以即插即用的方式显著缓解灾难性遗忘。

## 研究背景与动机

**类增量学习（CIL）的核心挑战**：模型需顺序学习新类，但旧类数据不可再访问，导致灾难性遗忘——模型对新类预测偏差严重。

**现有方法局限于类别不平衡视角**：主流方法（Balanced Fine-tuning、Prototype-based Classifier、Output Layer Calibration）将预测偏差归因于新旧类的类别不平衡，仅在分类头层面做修正。

**忽视了时序不平衡**：即使旧类之间样本数相同，因正样本到达时间不同，较早的类在训练后期累积了更多负监督，导致 precision 高但 recall 低的不对称现象。

**时序偏差影响全局参数**：CIL 中训练数据的时序顺序引入系统性偏差，不仅局限于分类头，而是影响整个模型（含 backbone 特征空间）。

**图1 的关键示例**：Task 2 中类 A 和类 B 样本数相同，但类 A 的正样本集中在 Task 0，类 B 在 Task 1——类 A 遭受更严重的遗忘，证明类别平衡不能解释所有偏差。

**缺乏损失层面的时序建模**：尽管时序衰减在时间序列预测、强化学习、在线学习等领域广泛使用，CIL 中尚无在损失函数层面显式建模正负监督时序不平衡的工作。

## 方法详解

### 整体框架

TAL 的核心思路：为每个类 $k$ 维护一个时序正监督强度 $Q_k[N]$，通过指数衰减记忆核追踪该类近期的正监督状态；然后在交叉熵损失中根据 $Q_k[N]$ 动态降权负监督，使旧类免受过度抑制。

### 关键设计

**1. 时序正监督强度 $Q_k[N]$**

- 对每个类 $k$，定义监督极性序列 $a_k[n] \in \{+1, -1\}$（正样本 +1，负样本 -1）
- 引入指数衰减记忆核 $f[n] = \lambda^{n+1}$（$0 < \lambda < 1$），$\lambda$ 为记忆参数
- $Q_k[N] = \sum_{n=0}^{N-1} f[N-1-n] \cdot a_k[n]$，即衰减核与监督序列的离散卷积
- 递推形式：$Q_k[N+1] = \lambda(Q_k[N] + a_k[N])$，时间和空间复杂度均为 $\mathcal{O}(1)$
- 上界 $Q_{\max} = \lambda / (1-\lambda)$

**2. 时序不平衡定理（Theorem 1）**

两个样本总数相同的类，正样本出现更早的类在训练结束时 $Q$ 值更小，即受到更多负监督压力，表现为高 precision 低 recall。

**3. Temporal-Adjusted Loss**

$$\ell_{\text{TAL}}(y, z, Q[N]) = -\log \frac{e^{z_y}}{e^{z_y} + \alpha \sum_{k \neq y} w(Q_k[N]) \cdot e^{z_k}}$$

其中权重函数 $w(Q_k[N]) = (Q_k[N] / Q_{\max})^r$：

- $Q_k$ 小（旧类近期缺乏正监督）→ $w$ 小 → 负监督被抑制
- $Q_k$ 接近 $Q_{\max}$（新类正监督充足）→ $w$ 接近 1 → 保留完整负监督
- 指数 $r > 0$ 控制权重函数的陡峭程度

**4. 频率对齐参数 $\alpha$**

- 保证在时序均匀且类别平衡条件下 TAL 退化为标准交叉熵
- 由类别数 $C$ 和指数 $r$ 唯一确定，无需额外调参

### 损失函数特性

- 仅有 $\lambda$（衰减速率）和 $r$（权重陡峭度）两个超参数
- 对 $Q$ 的更新一致性：负监督更新也乘以 $w(Q_k)$
- 即插即用：不修改模型架构，可直接替换 CE loss 集成到现有 CIL 框架

## 实验

### 主实验：多数据集多基线一致提升

| 方法 | CIFAR-100 10-task $A_{\text{Mean}}$ | CIFAR-100 10-task $A_{\text{Last}}$ | ImageNet-100 10-task $A_{\text{Mean}}$ | ImageNet-100 10-task $A_{\text{Last}}$ |
|------|------|------|------|------|
| iCaRL | 58.76 | 45.39 | 43.71 | 24.38 |
| iCaRL + TAL | **60.82** | **47.36** | **52.19** | **32.78** |
| DER | 63.53 | 50.75 | 52.25 | 40.28 |
| DER + TAL | **66.33** | **53.82** | **54.57** | **42.62** |
| TagFex | 65.97 | 55.99 | 54.73 | 41.70 |
| TagFex + TAL | **68.68** | **57.91** | **57.05** | **43.01** |

在 CIFAR-100、ImageNet-100、Food101 三个数据集的 10-task 和 20-task 设置下，TAL 对所有五个基线（iCaRL、FOSTER、MEMO、DER、TagFex）均带来一致且显著的提升。

### 消融实验

| $r$ \ $\lambda$ | 0.99 $A_{\text{Mean}}$ | 0.995 $A_{\text{Mean}}$ | 0.999 $A_{\text{Mean}}$ |
|------|------|------|------|
| 0.5 | 62.12 | 61.27 | 61.72 |
| 1.0 | 62.60 | **63.36** | 62.46 |
| 2.0 | 62.51 | 60.24 | 62.85 |
| CE baseline | 59.96 | - | - |

- 最优超参组合为 $\lambda = 0.995, r = 1.0$
- 在宽泛的超参范围内 TAL 均优于 CE，表明对超参选择具有鲁棒性
- $r$ 过大（如 5.0）会过度抑制新类导致性能下降

### 关键发现

1. **Precision-Recall 不对称是普遍现象**：在 iCaRL、DER、MEMO、TagFex 等方法中，早期类均呈现高 precision 低 recall 的模式（图2c）
2. **TAL 影响特征空间**：UMAP 可视化显示 TAL 缓解了旧类特征被新类侵占的问题，说明其效果不局限于分类头（图5）
3. **TAL 非均匀保护旧类**：不同旧类获得不同程度的保护，较新的旧类甚至可能被轻微抑制（图7）
4. **计算开销极小**：额外开销仅约 0.8%，得益于 $Q$ 的 $\mathcal{O}(1)$ 递推更新

## 亮点

- **新颖视角**：首次从时序不平衡角度形式化分析 CIL 中的预测偏差，区别于传统的类别不平衡解释
- **理论扎实**：建立了时序监督模型，证明了时序不平衡定理和 TAL 在平衡条件下退化为 CE 的性质
- **即插即用**：仅修改损失函数，不改架构，可无缝集成到任意 CIL 方法中
- **全局效果**：不仅修正分类头，还改善 backbone 特征空间的类间分布
- **广泛验证**：5 个基线 × 3 个数据集 × 2 种 task 设置，一致有效

## 局限性

- 衰减核固定为指数形式，实际中 $\lambda$ 可能随任务阶段变化
- 指数衰减随时间趋于零，可能无法完全捕捉已学表示的持续影响
- 未探索非参数化或自适应的时序建模形式
- 实验仅在中等规模数据集（100 类）上验证，大规模场景有待测试

## 相关工作

- **CIL 预测偏差修正**：Balanced Fine-tuning [49]、iCaRL 原型分类器 [29]、Weight Alignment [50]——均局限于分类头修正
- **时序建模**：指数平滑（Holt-Winters）、强化学习中的 Eligibility Traces / TD(λ) [37]、在线学习中的 ADWIN [4]——共享"近期数据权重更大"的直觉，但未用于 CIL 损失设计
- **动态架构方法**：DER [50]、MEMO [55]、TagFex [52]——扩展网络容量适应新任务，TAL 可作为正交补充

## 评分

- 新颖性: ⭐⭐⭐⭐ — 时序不平衡视角新颖且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ — 多基线多数据集全面验证，消融设计合理
- 写作质量: ⭐⭐⭐⭐ — 从问题定义到方法推导逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ — 即插即用特性使其实用性强，新视角可启发后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](../../ECCV2024/llm_evaluation/versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[ICLR 2026\] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](../../ICLR2026/llm_evaluation/accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)
- [\[ICML 2026\] HEDP: A Hybrid Energy-Distance Prompt-based Framework for Domain Incremental Learning](../../ICML2026/llm_evaluation/hedp_a_hybrid_energy-distance_prompt-based_framework_for_domain_incremental_lear.md)
- [\[ICLR 2026\] In-Context Learning of Temporal Point Processes with Foundation Inference Models](../../ICLR2026/llm_evaluation/in-context_learning_of_temporal_point_processes_with_foundation_inference_models.md)

</div>

<!-- RELATED:END -->
