---
title: >-
  [论文解读] Incentivizing Time-Aware Fairness in Data Sharing
description: >-
  [NeurIPS 2025][AI安全][数据共享] 提出了一个时间感知的数据共享框架，设计了新的激励机制（F6-F8）和两种奖励方案（时间感知奖励累计和时间感知数据估值），保证早加入协作的参与方能获得更高价值的奖励，同时兼顾公平性和个体理性。
tags:
  - NeurIPS 2025
  - AI安全
  - 数据共享
  - 公平性
  - 时间感知激励
  - Shapley值
  - 合作博弈论
---

# Incentivizing Time-Aware Fairness in Data Sharing

**会议**: NeurIPS 2025  
**arXiv**: [2510.09240](https://arxiv.org/abs/2510.09240)  
**代码**: 无  
**领域**: ai_safety  
**关键词**: 数据共享, 公平性, 时间感知激励, Shapley值, 合作博弈论

## 一句话总结

提出了一个时间感知的数据共享框架，设计了新的激励机制（F6-F8）和两种奖励方案（时间感知奖励累计和时间感知数据估值），保证早加入协作的参与方能获得更高价值的奖励，同时兼顾公平性和个体理性。

## 研究背景与动机

**领域现状**：协作机器学习（CML）中，多方聚合数据训练模型可以获得更好的性能。现有框架如 Shapley 值等已经建立了公平性和个体理性等激励机制来鼓励参与。

**现有痛点**：所有现有框架都假设所有参与方同时加入协作，但现实中参与方会因数据清洗时间、法律障碍或信息不对称等原因在不同时间加入。现有框架无法激励参与方尽早贡献数据。

**核心矛盾**：传统公平性要求"贡献相同价值数据的参与方获得相同奖励"，但时间感知视角下"早加入者应获得更高奖励"——两者存在冲突。

**本文目标**：如何设计既满足公平性又能激励早期参与的奖励机制。

**切入角度**：从合作博弈论出发，引入"时间"维度重新定义激励条件，设计兼容的数学框架。

**核心 idea**：早加入者承担更高风险且能促进"观望者"参与，因此应获得更高奖励。通过为公平性激励添加"同时加入"的前提条件来解决与时间感知激励的冲突。

## 方法详解

### 整体框架

框架包含三个阶段：(1) 数据估值：使用估值函数 $v$ 衡量各联盟的数据价值；(2) 奖励值决定：使用两种时间感知方案之一计算奖励值 $r_i$；(3) 奖励实现：通过似然调温或子集选择生成实际的模型奖励。

### 关键设计

1. **时间感知激励条件设计 (F1-F8)**：在传统的 5 个激励条件基础上进行了精心修改和扩展：

    - **F3\* 等时对称性**：添加"同时加入"前提条件，仅当 $t_i = t_j$ 且边际贡献相同时要求 $r_i = r_j$。
    - **F6\# 必要性**：如果某参与方的数据缺失会使任何联盟的价值为零，则该方必须获得与他人相同的奖励，无论加入时间。这保护了持有关键数据的参与方不因迟到被惩罚。
    - **F7\# 时间单调性**：在其他条件不变时，早加入的参与方奖励不低于晚加入时的奖励：$(t_i' < t_i) \implies r_i' \geq r_i$。
    - **F8\# 严格时间单调性**：当参与方数据对其前序者有增量价值时，早加入奖励严格更高。

2. **时间感知奖励累计（Time-Aware Reward Cumulation）**：将协作时间段划分为多个区间，每个区间视为一个独立博弈。参与方最终奖励为各区间 Shapley 值的加权和：
    $r_i = \sum_{\tau=0}^{T} w^{(\tau)} \varphi_i^{(\tau)}$
   权重 $w^{(t)} = \beta^t / \sum_{\tau=0}^{T}\beta^\tau$，通过参数 $\beta$ 控制对时间的强调程度。$\beta \to \infty$ 时退化为标准 Shapley 值。该方法满足 F1-F8。

3. **时间感知数据估值（Time-Aware Data Valuation）**：定义参与方合作能力为 $\lambda_i = e^{-\gamma t_i}$，使用修改后的估值函数：
    $v_{C,\mathbf{t}} = \sum_{T \subseteq C, |T| \geq 2} d(v,T) \min_{i \in T}\{e^{-\gamma t_i}\} + \sum_{i \in C} d(v,\{i\})$
   其中 $d(v,T)$ 是 Harsanyi 红利。参数 $\gamma \in (0,1]$ 控制时间影响。$\gamma = 0$ 时退化为标准情况。该方法同样满足 F1-F8。

4. **数据估值函数的性质要求 (A1-A3)**：

    - **非负性 A1**：$v_C \geq 0$
    - **单调性 A2**：$B \subseteq C \implies v_C \geq v_B$
    - **超加性 A3**：$v_{B \cup C} \geq v_B + v_C$（不重叠联盟）
   
   条件信息增益（Conditional IG）和次模函数的对偶均满足 A1-A3。

### 损失函数 / 训练策略

奖励实现阶段采用两种方法：
- **似然调温法**：对目标参与方，使用自身似然和其他方数据的调温似然更新后验，精确实现条件 IG 下的奖励值
- **子集选择法**：仅在聚合数据的一个子集上训练模型作为奖励，适用于任意估值函数但仅近似实现

## 实验关键数据

### 主实验

在三个数据集上验证（Friedman 合成数据、California Housing、MNIST），$n=3$ 个参与方。

**Friedman 数据集（$v_1 \approx v_2 > v_3$）核心观察：**

| $t_1$ | 方法 | $r_1^*$ | $r_2^*$ | $r_3^*$ | 满足 F2/F8 |
|--------|------|---------|---------|---------|------------|
| 0 | Shapley | 35.88 | 34.64 | 28.40 | F2✓/F8✗ |
| 0 | 累计 ($\beta$=1) | 35.88 | 34.64 | 28.40 | F2✓/F8✓ |
| 3 | 累计 ($\beta$=1) | ~31 | ~33 | ~27 | F2✓/F8✓ |
| 3 | 估值 ($\gamma$=0.5) | ~31 | ~33 | ~27 | F2✓/F8✓ |

### 消融实验

**参数敏感性分析（Friedman 数据集）：**

| 参数 | 效果 |
|------|------|
| $\beta$ 减小 | 更强调早期参与，早到者奖励差距增大 |
| $\beta \to \infty$ | 退化为标准 Shapley 值，时间无关 |
| $\gamma$ 增大 | 早到者优势更大，晚到惩罚更重 |
| $\gamma = 0$ | 时间无关，等同于标准情况 |

### 关键发现

- 当数据价值差距小时，时间因素主导奖励分配。即使数据更有价值的参与方，如果加入太晚仍可能获得低于他人的奖励
- 当数据价值差距大时（$v_3 \ll v_1$），数据质量始终主导，参与方 1 无论多晚加入都获得高于参与方 3 的奖励
- 两种方案都保证个体理性（每方奖励不低于自身数据价值），且模型性能（MNLP）与奖励值正相关
- 天真的方法（Shapley 值除以加入时间）会违反个体理性 F2 和必要性 F6

## 亮点与洞察

- **问题定义的严谨性出色**：8 个激励条件的定义互相兼容，解决冲突的方式（等时前提条件、弱不等式）非常优雅
- **两种方案的互补设计**：时间感知奖励累计从"奖励分配后"引入时间，时间感知数据估值从"数据估值前"引入时间
- **对偶估值函数与机器遗忘的关联**是一个深刻的洞察
- 框架在数据价值和时间价值之间取得了良好平衡

## 局限与展望

- 计算复杂度仍然较高：Shapley 值精确计算需要指数级评估，虽然提出了高效估计方向但未详细验证
- 假设数据价值随时间不变（排除了时间序列等数据类型），限制了适用范围
- 实验规模有限（仅 3 和 10 个参与方），实际場景中参与方可能更多
- 未考虑参与方可能策略性地谎报加入时间的博弈行为

## 相关工作与启发

- Béal et al. [2022] 是唯一类似的研究早到激励的合作博弈工作，但设定和方案完全不同
- 与联邦学习中的激励设计互补——本文考虑数据共享场景，FL 场景中的时间激励仍待探索
- 思路可推广到众筹、数据市场等需要激励早期参与的场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统地将时间维度引入数据共享公平性，激励条件设计严谨创新
- 实验充分度: ⭐⭐⭐⭐ 三个数据集验证了理论性质，但缺少大规模和实际应用场景实验
- 写作质量: ⭐⭐⭐⭐⭐ 數学严谨，命题定义清晰，论证逻辑通顺
- 价值: ⭐⭐⭐⭐ 在数据共享和协作ML领域有重要理论意义，但距离实际部署仍有距离

<!-- RELATED:START -->

## 相关论文

- [Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs](../../ICML2025/ai_safety/is_your_model_fairly_certain_uncertainty-aware_fairness_evaluation_for_llms.md)
- [Fairness under Competition](fairness_under_competition.md)
- [FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation](faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)
- [MARS: A Malignity-Aware Backdoor Defense in Federated Learning](mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)

<!-- RELATED:END -->
