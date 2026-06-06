---
title: >-
  [论文解读] Mitigating Shortcut Learning with InterpoLated Learning
description: >-
  [shortcut learning] 提出 InterpoLated Learning (InterpoLL)，通过将多数样本的表示与同类少数样本的表示进行插值，削弱模型对虚假关联（shortcut）的依赖，显著提升少数样本上的泛化能力。
tags:
  - "shortcut learning"
  - "interpolation"
  - "representation learning"
  - "minority generalization"
  - "ERM"
---

# Mitigating Shortcut Learning with InterpoLated Learning

| 会议/期刊 | 年份 | 论文链接 | 代码 |
|----------|------|---------|------|
| ACL 2025 | 2025 | [arXiv 2507.05527](https://arxiv.org/abs/2507.05527) | - |

**领域**: 自然语言理解 / 鲁棒性  
**关键词**: shortcut learning, interpolation, representation learning, minority generalization, ERM

## 一句话总结

提出 InterpoLated Learning (InterpoLL)，通过将多数样本的表示与同类少数样本的表示进行插值，削弱模型对虚假关联（shortcut）的依赖，显著提升少数样本上的泛化能力。

## 研究背景与动机

**问题定义**: 经验风险最小化 (ERM) 训练的模型倾向于利用 shortcut（即训练数据中输入特征与标签之间的虚假相关性），例如 MNLI 中 "entailment" 样本通常具有高词重叠率。模型在多数样本上表现良好，但在 shortcut 不成立的少数样本上严重失败。

**现有方法的局限性**:
- 数据增强方法（合成少数样本）和样本加权方法（上调少数样本权重）**主要改进分类层**，并不能学到不同于 ERM 的表示，甚至可能强化 shortcut 特征
- 许多方法依赖**辅助模型**，增加计算开销和超参数调优复杂度
- 部分方法需要**事先知道少数/多数样本标签**（group annotation），实际场景中难以获取

**本文动机**: 设计一种无需 group annotation、模型无关、能真正改善表示学习的 shortcut 缓解方法。

## 方法详解

### 整体框架

InterpoLL 由两个阶段组成：

1. **推断少数/多数样本**: 使用一个欠参数化的辅助模型 $f_\phi$（如 TinyBERT）对训练集分类——被误分类的样本视为少数样本 $g_{\min}$，正确分类的视为多数样本 $g_{\maj}$
2. **插值训练**: 对每个 mini-batch 中的多数样本，在表示层面与同类少数样本进行插值，然后用插值后的表示计算损失并更新模型

### 关键设计

- **类内少数样本插值**: 对多数样本 $(x_i, y_i) \in g_{\maj}$，随机选取一个标签相同的少数样本 $(x_j, y_j) \in g_{\min}$，在编码器输出空间进行线性插值：$z_i = (1-\lambda) f_{\text{enc}}(x_i) + \lambda f_{\text{enc}}(x_j)$
- **受限插值比例**: $\lambda \sim \text{Uniform}(0, 0.5)$，确保多数样本表示只被轻微修改，保持对多数样本的拟合能力，同时引入少数样本的 shortcut-mitigating 特征
- **标签不变性**: 由于 $x_i$ 和 $x_j$ 属于同一类别，插值后标签保持不变，无需修改标签

### 损失函数

使用标准的交叉熵损失，仅在前向传播中用插值表示 $z_i$ 替代原始 $f_{\text{enc}}(x_i)$，反向传播正常进行：

$$J_{\text{ERM}}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_{\text{cls}}(z_i), y_i)$$

## 实验

### 主实验结果

**自然语言推理任务 (NLI)**:

| 方法 | MNLI-ID | MNLI-OOD | FEVER-ID | FEVER-OOD | QQP-ID | QQP-OOD | Avg-OOD |
|------|---------|----------|----------|-----------|--------|---------|---------|
| ERM | 84.9 | 62.4 | 88.4 | 55.9 | 90.2 | 33.8 | 50.7 |
| GroupDRO (需要group) | 84.3 | 72.5 | 87.5 | 64.1 | 89.5 | 52.9 | 63.2 |
| InterpoLL (无需group) | 84.6 | **75.6** | 87.8 | **68.7** | 89.8 | **56.9** | **67.1** |

**文本分类任务**:

| 方法 | FDCL18-Avg | FDCL18-Minority | CivilComments-Minority | Avg-Minority |
|------|-----------|-----------------|----------------------|--------------|
| ERM | 81.3 | 35.6 | 63.5 | 49.6 |
| GroupDRO | 76.2 | 57.3 | 69.5 | 63.4 |
| InterpoLL | 78.8 | **61.2** | **73.9** | **67.6** |

### 消融/分析实验

**跨架构泛化性** (MNLI → HANS/PAWS/Sym 等 OOD 集合):

| 模型 | ERM-Avg | InterpoLL-Avg | 提升 |
|------|---------|---------------|------|
| BERT-large | 61.7 | 67.7 | +6.0 |
| RoBERTa-large | 65.4 | 71.9 | +6.5 |
| T5-large | 69.9 | 76.0 | +6.1 |
| T5-3B | 70.8 | 77.3 | +6.5 |

**域泛化 (GLUE-X)**: InterpoLL 在 6 个任务上平均提升 3.1%，超越次优方法 Minimax 2.5%。

### 关键发现

1. InterpoLL 在**无需 group 标注**的情况下，显著超越需要 group 标注的方法（如 GroupDRO）
2. 改进在 encoder、encoder-decoder、decoder-only 三种架构上**一致有效**
3. InterpoLL 不仅改善分类层，还**减少了表示中的 shortcut 特征**
4. 运行时间与 ERM **基本一致**，无显著额外开销

## 亮点

- 方法简洁优雅：仅通过表示插值即可有效缓解 shortcut learning，不需要复杂的对抗训练或多阶段流程
- 无需 group annotation 但超越需要 group 信息的方法，实用价值高
- 跨架构、跨任务的一致性提升，展示了方法的通用性
- 提供了详细的分析，包括表示中 shortcut 特征的 probing 实验

## 局限性

- 辅助模型推断少数/多数样本存在噪声，依赖辅助模型的质量
- 插值比例 $\lambda$ 的范围 $[0, 0.5]$ 是固定的，可能不是所有任务的最优选择
- 主要在 NLU 任务上验证，对生成任务的适用性未知
- 需要训练一个辅助模型来推断样本类别

## 相关工作

- **Shortcut 缓解**: GroupDRO (Sagawa et al., 2019)、JTT (Liu et al., 2021)、DFR (Kirichenko et al., 2023)
- **Mixup 系列**: Zhang et al. (2018) 提出的 Mixup 在输入空间混合，而 InterpoLL 在表示空间对特定样本对进行混合
- **样本加权**: Conf-reg (Utama et al., 2020)、Weak-learn (Sanh et al., 2021)

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 7 |
| 实用性 | 8 |
| 实验充分度 | 9 |
| 写作质量 | 8 |
| 总分 | 8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation](../../ICML2026/others/disco_mitigating_bias_in_deep_learning_with_conditional_distance_correlation.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[ACL 2025\] Value Residual Learning](value_residual_learning.md)
- [\[ACL 2025\] Retrospective Learning from Interactions](retrospective_learning_from_interactions.md)
- [\[ACL 2025\] Learning to Reason from Feedback at Test-Time](learning_to_reason_from_feedback_at_test-time.md)

</div>

<!-- RELATED:END -->
