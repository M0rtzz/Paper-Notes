---
title: >-
  [论文解读] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework
description: >-
  [ICLR 2026][LLM对齐][RLHF] 提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "RLHF"
  - "NLHF"
  - "偏好学习"
  - "社会选择理论"
  - "人群比例对齐"
  - "公理化框架"
---

# Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework

**会议**: ICLR 2026  
**arXiv**: [2506.05619](https://arxiv.org/abs/2506.05619)  
**代码**: 补充材料中包含实验代码  
**领域**: AI Alignment / Social Choice Theory  
**关键词**: RLHF, NLHF, 偏好学习, 社会选择理论, 人群比例对齐, 公理化框架  

## 一句话总结
提出基于社会选择理论公理的偏好学习框架，从成对比较数据中推断评估者人群分布的可行集，构造满足人群比例对齐(PPA)和人群有界可操纵性(PBM)公理的策略。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：1. RLHF 依赖 Bradley-Terry 模型将偏好压缩为标量奖励，在偏好不一致/循环偏好场景下失效
2. NLHF 将偏好学习建模为博弈，找 Nash 均衡策略，但仍不能按比例反映评估者分布
3. **核心问题**：当两组评估者对两个选项的偏好接近 50:50 时（如 50+ε vs 50−ε），RLHF 和 NLHF 都会输出确定性策略选择微弱多数方，完全忽略少数群体
4. 现有多元对齐方法（mixture-based、steerable models）通常需要显式的评估者群组标签，实际中难以获取
5. 已有公理化方法（如随机独裁制 Random Dictatorship）虽满足比例对齐，但无法仅从成对比较数据实现
6. 本文目标：不需额外群组信息，仅从成对比较数据实现比例对齐

## 方法详解

### 整体框架

整个方法把"对齐"重新定义成一个社会选择问题：不再把偏好压成标量奖励，而是先从成对比较数据里反推出评估者人群分布的可行集合，再在这个集合上构造一个能按比例反映各群体份额、且抗操纵的随机策略。落地上分两步——先估计每个选项的人群份额上界形成可行集，再据此分配选择概率，并用一组公理来界定这个策略应当满足的性质。

### 关键设计

**1. 可行人群分布集推断：在没有群组标签时反推人群结构。** 多元对齐方法通常要求显式的评估者群组标签，但实际只能拿到成对比较数据。本文绕开这一点：对每个选项 $y_i$，定义 $u_i = \min_{y \neq y_i} P(y_i \succ y)$，即 $y_i$ 在所有对决中胜率的最小值，作为支持 $y_i$ 的人群份额的一个上界——因为任何把 $y_i$ 排在首位的群体，至少要让 $y_i$ 赢下所有对手。把这些上界拼起来就得到真实人群分布 $w$ 的多面体外近似 $\bar{\mathcal{W}}(P) = \{w \in \Delta(\mathcal{Y}) \mid w_i \leq u_i\}$。这样无需任何额外标注，仅凭比较概率就刻画出"人群可能长什么样"的集合，但代价是选项数多时该外近似会偏宽松。

**2. 比例分配策略 $F^*$：让选择概率正比于人群份额。** 有了上界后，最自然的对齐策略是按份额比例随机选择：$\pi(y_i) = u_i / \sum_j u_j$。这与 RLHF/NLHF 的确定性输出形成对比——当两群人对两个选项的偏好接近 $50{+}\epsilon$ 比 $50{-}\epsilon$ 时，确定性策略会彻底丢掉少数方，而比例策略会以接近一半的概率照顾到弱势群体。该策略本质上在最坏情况的比例失配上取保守解，保证少数偏好不被多数微弱优势抹平。

**3. 四条公理：界定"好策略"应满足的性质。** 为了说明 $F^*$ 的合理性，本文给出四条公理作为评判标准。**单调性**要求提升某选项的排名不会降低它被选中的概率；**Pareto 效率**要求若所有人都偏好 $y$ 胜 $y'$，策略应倾向 $y$。真正核心的是后两条量化公理：$\alpha$-PPA（人群比例对齐）要求 $\pi(y_k)/w_k^\sigma \geq \alpha(\sigma)$，即每个群体首选项的被选概率至少弱比例于其人群份额；$\gamma$-PBM（人群有界可操纵性）要求任何群体通过策略性误报偏好所能攫取的策略增益被 $\gamma_1 w_k^\sigma + \gamma_2$ 上界约束住，从而非多数群体无法靠操纵翻身成多数。本文进一步证明 RLHF 与 NLHF 会违反任意强度的 PPA 和 PBM，而 $F^*$ 在这两条公理上有理论保证。

**4. Softmax 松弛 $F^\beta$：在比例对齐与多数一致间可调权衡。** 纯比例策略虽然公平，却可能在该选多数方时显得"太软"。为此引入温度参数 $\beta$ 做松弛：$\pi(y_i) = u_i \exp(\beta u_i) / \sum_j u_j \exp(\beta u_j)$。$\beta = 0$ 时退化回原始的比例策略 $F^*$；$\beta \to \infty$ 时概率质量集中到上界最大的选项，收敛为 minimax Condorcet 方法。$\beta$ 越大越偏向多数、胜率越高，但 PPA 随之下降——这条权衡被后续实验直接验证。

## 实验

### 表格实验：MovieLens 电影推荐


### 主实验

| 方法 | 胜率 | PPA 水平 | PBM 增益 |
|------|------|----------|----------|
| RLHF | 0.7784 | 0 | 0.0611 |
| NLHF | 0.7712 | 0 | 0.0124 |
| $F^\beta$($\beta=1$) | ~0.60 | 0.4869 | 8.9e-4 |

- β 增大时胜率升高但 PPA 下降，验证理论预测的权衡关系
- 提出方法在 β≤10 时操纵抗性显著优于基线

### LLM 实验：Qwen2.5-3B-Instruct


### 消融实验

| 数据集 | β=0 PPA | DPO PPA |
|--------|---------|---------|
| Synthetic-Color | 0.0883 | 0.0000 |
| Alpaca-Expertise | 0.1428 | 0.1321 |
| Alpaca-Style | 0.5012 | 0.3786 |

- 合成数据上权衡明显；Alpaca 数据因 GPT-4.1 注释噪声效果较弱
- 计算代价与 RLHF 相当，高于 DPO

## 亮点与洞察
- 理论严谨：证明 RLHF/NLHF 违反任意强度的 PPA 和 PBM 公理
- 仅需成对比较数据即可推断人群分布可行集，不需要群组标签
- Softmax 松弛提供比例对齐与 Condorcet 一致性的可调权衡
- 操纵抗性有理论保证：非多数群体无法通过策略性误报获得多数地位

## 局限与展望
- PPA 仅关注各群组首选项的选择概率，忽略低排名偏好
- LLM 场景下评估 PPA 水平仍是开放问题（logit 估计 vs 群组分类均有噪声）
- 两阶段函数近似方法计算开销不低于 RLHF，需开发直接策略优化版本
- 外近似 $\bar{\mathcal{W}}$ 在选项数多时可能过于宽松

## 相关工作
- **RLHF / DPO**：等价于最大 Borda 规则，确定性选择胜者
- **NLHF**：等价于最大彩票(Maximal Lotteries)，满足 Pareto 但不满足 PPA
- **Random Dictatorship**：完美 PPA 但不可从成对比较实现
- **多元对齐**（Sorensen 2024, Chen 2024）：需要显式群组标签
- **抗操纵机制**（Buening 2025, Park 2024）：追求严格策略防篡改，本文约束群体层面增益

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling](beyond_pairwise_empowering_llm_alignment_with_ranked_choice_modeling.md)
- [\[ICLR 2026\] General Exploratory Bonus for Optimistic Exploration in RLHF](general_exploratory_bonus_for_optimistic_exploration_in_rlhf.md)
- [\[ICLR 2026\] JailNewsBench: Multi-Lingual and Regional Benchmark for Fake News Generation under Jailbreak Attacks](jailnewsbench_multi-lingual_and_regional_benchmark_for_fake_news_generation_unde.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)
- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)

</div>

<!-- RELATED:END -->
