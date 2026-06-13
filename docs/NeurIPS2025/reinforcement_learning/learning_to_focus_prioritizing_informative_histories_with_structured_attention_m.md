---
title: >-
  [论文解读] Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][部分可观测RL] 提出两种结构化时序先验（Memory-Length Prior和Gaussian Distributional Prior）嵌入Transformer世界模型的自注意力机制中，在部分可观测RL环境下，Gaussian Attention在Atari 100k基准上相对UniZero提升77%的人类归一化均分，且计算开销几乎为零。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "部分可观测RL"
  - "Transformer"
  - "注意力先验"
  - "高斯注意力"
  - "样本效率"
---

# Learning to Focus: Prioritizing Informative Histories with Structured Attention Mechanisms in Partially Observable Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2511.06946](https://arxiv.org/abs/2511.06946)  
**代码**: [GitHub](https://github.com/daniallegue/learning-to-focus)  
**领域**: 强化学习  
**关键词**: 部分可观测RL, Transformer世界模型, 注意力先验, 高斯注意力, 样本效率

## 一句话总结

提出两种结构化时序先验（Memory-Length Prior和Gaussian Distributional Prior）嵌入Transformer世界模型的自注意力机制中，在部分可观测RL环境下，Gaussian Attention在Atari 100k基准上相对UniZero提升77%的人类归一化均分，且计算开销几乎为零。

## 研究背景与动机

Transformer作为世界模型在基于模型的强化学习（MBRL）中越来越受欢迎，特别是UniZero将MuZero的循环动力学替换为Transformer骨架，利用掩码自注意力捕捉长程依赖。然而，存在一个根本性的假设错配：

- **NLP场景**：数据丰富、平衡，长程依赖频繁出现，标准自注意力可以隐式学到
- **RL场景**：轨迹稀疏、奖励驱动，绝大部分transition不包含有用信息，仅极少数关键转移驱动决策

标准自注意力对所有历史token赋予均等的初始权重，在数据稀缺的RL环境中难以快速学到哪些transition真正重要。这导致Transformer世界模型在低数据场景（如Atari 100k）下样本效率低下。

核心思路：**能否将结构化的时间先验直接编码进自注意力，让模型一开始就知道"关注哪里"？**

## 方法详解

### 整体框架

在UniZero世界模型的动力学头（dynamics head）的自注意力层中引入两种归纳偏置（inductive bias）：Memory-Length Prior和Distributional Prior。动力学头负责根据历史潜在状态-动作对预测下一个潜在状态 $\hat{z}_{t+1}$ 和即时奖励 $\hat{r}_t$。

### 关键设计

1. **Memory-Length Prior（自适应注意力）**：受部分可观测环境中有限有效记忆假设启发。每个注意力头 $h$ 学习一个标量参数 $s_h$，通过softplus变换为正的回溯跨度 $L_h = \text{softplus}(s_h)$。构造硬掩码：
$$M_{ij}^{(h)} = \begin{cases} 0, & i - j \leq L_h \\ -\infty, & i - j > L_h \end{cases}$$
注意力权重变为 $\text{Attention}^{(h)} = \text{softmax}\left(\frac{Q^{(h)} K^{(h)\top}}{\sqrt{d_k}} + M^{(h)}\right)$。添加 $\ell_1$ 惩罚防止所有跨度无限增长，鼓励学习最小但充分的窗口。

2. **Distributional Prior（高斯注意力）**：在部分可观测环境中，仅稀疏的token子集对预测有贡献。每个头学习高斯参数 $\mu_h, \sigma_h > 0$，定义位置核：
$$G_{ij}^{(h)} = -\frac{(i - j - \mu_h)^2}{2\sigma_h^2}$$
加入到注意力logits中作为偏置项。无约束的 $\mu_h$ 和 $\sigma_h$ 让每个头获得光滑的、可学习的显著性分布——可以聚焦于特定偏移量处，也可以扩展到广域注意力。不同头捕获不同的时间尺度。

3. **组合先验（Gaussian Adaptive Attention）**：将两种先验相加 $B_{ij}^{(h)} = G_{ij}^{(h)} + M_{ij}^{(h)}$，在有限窗口内保持光滑显著性。但实验表明硬截断会切断高斯尾部，反而损害性能。

### 损失函数 / 训练策略

- 训练遵循UniZero的联合模型-策略优化框架，使用soft-target世界模型稳定学习
- 自适应注意力跨度使用 $\ell_1$、$\ell_2$ 或 max-norm 正则化，其中 $\ell_2$ 泛化最稳健
- 高斯先验初始化为 $\mu_h=6, \sigma_h=1$，窄先验（$\sigma_h=1$）始终优于宽先验（$\sigma_h=3$）

## 实验关键数据

### 主实验

Atari 100k基准，26个游戏，5个随机种子，与UniZero和MuZero对比：

| 方法 | 归一化均分 (HNS Mean) | 归一化中位数 (HNS Median) | 胜出游戏数 |
|------|---------------------|-------------------------|-----------|
| MuZero | 0.44 | 0.13 | — |
| UniZero ST (Baseline) | 0.13 | 0.05 | — |
| Adaptive UniZero | 0.095 | 0.05 | 部分 |
| **Gaussian UniZero** | **0.23** | **0.10** | **19/26** |
| Gaussian Adaptive UniZero | 0.00 | 0.02 | 极少 |

代表性游戏得分对比：

| 游戏 | UniZero | Gaussian UniZero | 提升 |
|------|---------|-----------------|------|
| KungFuMaster | 2019 | **9424** | +367% |
| Kangaroo | 843 | **1636** | +94% |
| Assault | 342 | **487** | +42% |
| Jamesbond | 202 | **362** | +79% |

### 消融实验

| 配置 | Pong | MsPacman | Jamesbond | Freeway |
|------|------|----------|-----------|---------|
| $L_h=2$ | -18.5 | 716.7 | 180.0 | 2.2 |
| $L_h=6$ | -19.6 | **1103.3** | 156.7 | 0.7 |
| $\sigma_h=1$ (窄) | -7.9 | 726.7 | **362.1** | 0.1 |
| $\sigma_h=3$ (宽) | -15.1 | 638.7 | 196.7 | 0.0 |
| $\ell_1$ 正则 | 中等 | 中等 | 中等 | 偶尔最优 |
| $\ell_2$ 正则 | **最稳健** | **最稳健** | 中等 | 中等 |

### 关键发现

- **高斯先验远优于硬截断窗口**：光滑的位置权重可以灵活适应不同时间依赖结构，而硬截断经常误判相关范围
- **组合两种先验反而害性能**：硬掩码截断了高斯尾部的有用信号，产生矛盾的先验
- **窄高斯（$\sigma_h=1$）始终优于宽高斯（$\sigma_h=3$）**：强初始先验比弱先验更有效
- **计算开销可忽略**：所有先验变体的MFLOPs增加不超过0.002%

## 亮点与洞察

- 揭示了NLP与RL在序列建模上的根本差异：RL轨迹是稀疏和奖励驱动的，照搬NLP的均匀注意力假设在低数据RL中是不合适的
- 光滑的分布式先验比离散的记忆窗口更适合RL中不规则的时间依赖结构——这是一个有价值的设计准则
- 增加的参数量极少（仅每头2-3个标量），却带来显著性能提升，体现了归纳偏置的强大力量

## 局限与展望

- 仅在Atari环境中验证，未扩展到连续控制或多任务设置
- 自适应注意力跨度需要正则化来避免退化到极端值（全部关注或不关注）
- 高斯先验是各向同性的，更复杂的时空结构可能需要更灵活的分布形式
- 未与其他位置编码方法（如RoPE、ALiBi）进行对比

## 相关工作与启发

- **UniZero**：用Transformer替代MuZero的循环动力学，是本文的基础架构
- **Adaptive Attention Span**：学习每个注意力头的上下文长度，但为NLP设计而非RL
- **Influence-Based Abstraction (IBA)**：形式化了POMDP中的最小充分历史概念，是Memory-Length Prior的理论基础
- **启发**：在Transformer架构设计中，针对特定领域的归纳偏置比通用的大模型架构更高效——这在RL中尤其明显

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将结构化时间先验引入RL世界模型是自然且有效的创新
- **实验充分度**: ⭐⭐⭐⭐ 26个Atari游戏、详尽的消融，但缺少连续控制实验
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，实验分析深入，图表直观
- **价值**: ⭐⭐⭐⭐ 为RL中Transformer世界模型的设计提供了实用指导原则

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](../../ICML2025/reinforcement_learning/pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[ACL 2025\] Learning to Generate Structured Output with Schema Reinforcement Learning](../../ACL2025/reinforcement_learning/learning_to_generate_structured_output_with_schema_reinforcement_learning.md)
- [\[NeurIPS 2025\] Towards Provable Emergence of In-Context Reinforcement Learning](towards_provable_emergence_of_in-context_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
