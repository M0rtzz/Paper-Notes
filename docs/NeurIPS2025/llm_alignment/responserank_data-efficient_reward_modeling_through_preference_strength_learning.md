---
title: >-
  [论文解读] ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning
description: >-
  [NeurIPS 2025][LLM对齐][奖励模型] 提出 ResponseRank 方法,通过利用偏好强度的代理信号（如响应时间和标注者一致性）的局部相对差异来鲁棒地学习效用差值,显著提升奖励模型的样本效率。 现有痛点 现有痛点：领域现状：RLHF 中的二元偏好选择（A 优于 B）仅传达偏好方向,不包含强度信息…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "奖励模型"
  - "偏好强度"
  - "响应排名"
  - "RLHF"
  - "样本效率"
---

# ResponseRank: Data-Efficient Reward Modeling through Preference Strength Learning

**会议**: NeurIPS 2025

**arXiv**: [2512.25023](https://arxiv.org/abs/2512.25023)

**代码**: 无

**领域**: LLM对齐

**关键词**: 奖励模型, 偏好强度, 响应排名, RLHF, 样本效率

## 一句话总结

提出 ResponseRank 方法,通过利用偏好强度的代理信号（如响应时间和标注者一致性）的局部相对差异来鲁棒地学习效用差值,显著提升奖励模型的样本效率。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：RLHF 中的二元偏好选择（A 优于 B）仅传达偏好方向,不包含强度信息。但偏好强度对于不确定性下的决策和偏好模型泛化至关重要：

**信息丢失**: "强烈偏好苹果" 与 "略微偏好苹果" 在二元标注中无法区分

**代理信号不可靠**: 响应时间、标注者一致性等代理信号虽可反映强度,但噪声大、存在混杂因素

**绝对值无意义**: 不同标注者、不同问题的响应时间绝对值不可比较

**样本效率低**: 不利用强度信息导致需要更多标注数据

## 方法详解

### 整体框架

ResponseRank 通过在精心构建的分层（strata）内比较代理信号的相对差异,将偏好强度信息转化为排名约束,用于训练更准确的奖励模型。

### 关键设计

**1. 局部分层策略**

- 将样本按特征分层（如相同 prompt、相同标注者）
- 仅在**同一层内**比较代理信号,控制系统性偏差
- 避免跨层比较带来的混杂效应（如不同 prompt 难度差异）

**2. 相对强度排名**

- 在每层中,按代理信号值从大到小排序
- 代理信号高（如响应时间短、标注者一致性高）→ 强偏好
- 生成排名约束: 若 $(x_i, y_i^w, y_i^l)$ 的偏好比 $(x_j, y_j^w, y_j^l)$ 强,则 $|r(y_i^w) - r(y_i^l)| > |r(y_j^w) - r(y_j^l)|$

**3. 排名约束训练**

在标准 BT 损失之上添加排名损失：
$$\mathcal{L}_{\text{rank}} = \sum_{(i,j) \in \text{rank-pairs}} \max(0, \Delta_j - \Delta_i + \text{margin})$$

其中 $\Delta_i = r(y_i^w) - r(y_i^l)$ 是效用差值。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{BT}} + \lambda \mathcal{L}_{\text{rank}}$$

- $\mathcal{L}_{\text{BT}}$: 标准 Bradley-Terry 偏好损失
- $\mathcal{L}_{\text{rank}}$: 基于偏好强度的排名约束
- $\lambda$: 权衡系数

## 实验关键数据

### 主实验

合成偏好学习 (带模拟响应时间):

| 方法 | 20% 数据 | 50% 数据 | 100% 数据 |
|------|---------|---------|----------|
| BT (标准) | 0.62 | 0.71 | 0.78 |
| Weighted BT | 0.65 | 0.73 | 0.79 |
| ResponseRank | **0.72** | **0.78** | **0.82** |

语言模型奖励学习 (标注者一致性作为代理, RewardBench Accuracy):

| 方法 | Chat | Safety | Reasoning | 平均 |
|------|------|--------|-----------|------|
| BT 标准 | 72.5 | 80.3 | 68.2 | 73.7 |
| Margin-BT | 73.8 | 81.2 | 69.5 | 74.8 |
| ResponseRank | **76.2** | **83.5** | **72.1** | **77.3** |

### 消融实验

不同代理信号质量下的鲁棒性:

| 噪声水平 | BT | Weighted BT | ResponseRank |
|---------|-----|-------------|-------------|
| 无噪声 | 0.78 | 0.85 | **0.86** |
| 低噪声 | 0.78 | 0.82 | **0.84** |
| 中噪声 | 0.78 | 0.78 | **0.82** |
| 高噪声 | 0.78 | 0.72 | **0.80** |

### 关键发现

1. ResponseRank 仅用 20% 数据即达到标准BT使用 100% 数据的性能水平,样本效率提升 5 倍
2. 局部分层是鲁棒性的关键——Weighted BT 在高噪声下反而比标准BT差
3. Pearson Distance Correlation (PDC) 指标能有效区分序数准确性和基数效用学习
4. 在 RL 控制任务中,利用模拟回报作为代理信号同样有效

## 亮点与洞察

- **最小假设**: 仅假设代理信号在局部有效,不对其全局分布做假设
- **鲁棒设计**: 噪声越大时相比 Weighted BT 优势越明显
- **新指标 PDC**: Pearson Distance Correlation 分离了序数和基数学习

## 局限与展望

1. 分层策略需要足够多的同层样本,小规模数据集可能不适用
2. 代理信号的选择（响应时间 vs 一致性 vs 其他）缺乏系统比较
3. 排名约束仅考虑成对关系,未利用多样本的全序信息
4. 与 DPO 等直接偏好优化方法的集成未探索

## 相关工作与启发

- **RLHF**: 标准的人类反馈强化学习框架
- **Preference Learning with RT**: 响应时间辅助偏好学习的相关工作
- **学习排名 (Learning to Rank)**: 信息检索中的排名学习方法

## 评分

- ⭐ 创新性: 8/10 — 局部分层+相对排名的设计简洁有效
- ⭐ 实用性: 8/10 — 5倍样本效率提升对实际标注很有价值
- ⭐ 写作质量: 8/10 — 实验涵盖合成、语言、RL三个场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Provably Efficient Online RLHF with One-Pass Reward Modeling](provably_efficient_online_rlhf_with_one-pass_reward_modeling.md)
- [\[ACL 2025\] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](../../ACL2025/llm_alignment/rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)
- [\[NeurIPS 2025\] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)
- [\[NeurIPS 2025\] What Makes a Reward Model a Good Teacher? An Optimization Perspective](what_makes_a_reward_model_a_good_teacher_an_optimization_perspective.md)

</div>

<!-- RELATED:END -->
