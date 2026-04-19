---
title: >-
  [论文解读] Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity
description: >-
  [ICLR 2026][α-散度] 提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。
tags:
  - ICLR 2026
  - α-散度
  - 分布匹配
  - RLVR
  - 多样性保持
  - 定理证明
---

# Whatever Remains Must Be True: Filtering Drives Reasoning in LLMs, Shaping Diversity

**会议**: ICLR 2026  
**arXiv**: [2512.05962](https://arxiv.org/abs/2512.05962)  
**代码**: https://github.com/naver/alpha-dpg  
**领域**: LLM NLP / 强化学习 / LLM推理  
**关键词**: α-散度, 分布匹配, RLVR, 多样性保持, 定理证明

## 一句话总结
提出 DMVR 框架和 α-DPG 算法，通过显式定义"过滤掉错误答案"的目标分布并用 α-散度族来逼近，统一了 RLVR（Reverse KL）和拒绝采样微调（Forward KL），在 Lean 定理证明上实现了精度-覆盖率 Pareto 前沿的最优表现。

## 研究背景与动机

**领域现状**：基于可验证奖励的强化学习（RLVR，如 GRPO/PPO）已成为调优 LLM 推理能力的标准方法。然而，越来越多的证据表明 RLVR 训练后的模型存在**严重的多样性损失**（mode collapse）——虽然 pass@1 提高了，但生成策略的多样性大幅降低，导致 pass@k（k 较大时）反而不如基础模型。

**现有痛点**：RLVR（GRPO/PPO 等）隐式优化了 Reverse KL 散度到目标分布，这是一种"模式寻求"（mode-seeking）散度——它让模型集中在少数高奖励区域，忽略其他有效解。当 β=0 时，退化为纯 REINFORCE，完全没有多样性保护。现有缓解方法（KL 惩罚、Rw-Ulkly 等）治标不治本。

**核心矛盾**：精度（pass@1）与覆盖率（pass@k）之间存在根本权衡。现有 RL 方法只能偏向精度端，缺乏系统性控制这一权衡的手段。

**本文目标** 如何在保持正确性的同时保留基础模型中已有的解的多样性？如何提供精度-覆盖率权衡的连续可控机制？

**切入角度**：从**分布匹配**（Distributional Matching）的角度重新审视 RLVR——显式定义目标分布为"过滤掉错误答案、保留正确答案相对概率"的分布 $p_x(y) \propto \pi_{\text{base}}(y|x) \cdot v(y,x)$。然后用 α-散度族来逼近这个目标分布，不同 α 值对应不同的精度-多样性权衡。

**核心 idea**：多样性损失的根源不在目标分布（过滤本身），而在用来逼近它的散度选择——用 α-散度替代 Reverse KL 可系统性控制精度与多样性的平衡。

## 方法详解

### 整体框架
DMVR（Distributional Matching with Verifiable Rewards）框架：(1) 定义目标分布 $p_x(y) \propto \pi_{\text{base}}(y|x) \cdot v(y,x)$（过滤错误，保留正确答案的原始相对概率）→ (2) 选择 α-散度 $D_{f_\alpha}(\pi_\theta || p_x)$ 作为优化目标 → (3) 用 f-DPG 算法的策略梯度来训练 → (4) α 参数控制 mode-seeking（α→1，类似RLVR）到 mass-covering（α→0，类似拒绝采样微调）的连续过渡。

### 关键设计

1. **显式目标分布的定义**:

    - 功能：定义"理想"的训练目标——过滤掉所有错误回答，保留正确回答的相对概率不变
    - 核心思路：$p_x(y) \propto \pi_{\text{base}}(y|x) \cdot v(y,x)$。这是满足两个条件的唯一分布：(i) 所有输出都通过验证器 $v$，(ii) 在所有满足 (i) 的分布中，与基础模型的 Forward KL 最小（信息几何中的 I-投影）
    - 设计动机：区别于 RLVR 的隐式目标 $p_{x,\beta}(y) \propto \pi_{\text{base}}(y|x) \cdot \exp(v(y,x)/\beta)$（一个平滑近似），显式定义使得"目标是什么"和"如何逼近"分离，允许独立选择逼近策略

2. **RLVR 与 Reverse KL 的等价性分析（理论贡献）**:

    - 功能：证明 RLVR 的隐式优化等价于最小化 Reverse KL 到 $p_{x,\beta}$
    - 核心思路：**引理1** 证明 $\nabla_\theta \mathbb{E}_x[KL(\pi_\theta || p_{x,\beta})] = -\frac{1}{\beta} \nabla_\theta \mathbb{E}_{x,y\sim\pi_\theta}[v(y,x) - \beta \log \frac{\pi_\theta}{\pi_{\text{base}}}]$，即最大化 RLVR 伪奖励等价于最小化 Reverse KL。**引理2** 证明 $\lim_{\beta\to 0} p_{x,\beta} = p_x$
    - 设计动机：解释为何 RLVR 必然导致多样性损失——Reverse KL 是 zero-forcing 的，允许忽略目标分布的整个模式

3. **α-DPG 算法**:

    - 功能：用 α-散度族参数化 f-DPG，实现精度与多样性的连续可控权衡
    - 核心思路：α-散度的伪奖励为 $\hat{R}_\theta(y,x) = \min\left(\left(\frac{p_x(y)}{\pi_\theta(y|x)}\right)^{1-\alpha} - 1, M\right)$。当 α→1 时恢复 REINFORCE（mode-seeking），α→0 时恢复 KL-DPG/拒绝采样微调（mass-covering），α=0.5 时为 Hellinger 距离。使用 leave-one-out 均值作 baseline 减小方差，clipping 值 M=10 防止低 α 时的方差爆炸
    - 设计动机：统一框架——RLVR（Reverse KL, α≈1）、KL-DPG（Forward KL, α=0）、拒绝采样微调（RS-FT）都是特例。单一超参 α 即可遍历整个精度-覆盖率 Pareto 前沿

4. **配分函数在线估计**:

    - 功能：计算目标分布的归一化常数 $Z_x$
    - 核心思路：$Z_x = \mathbb{P}_{y\sim a(\cdot|x)}[v(y,x)=1]$，即基础模型采样正确率。用当前批次的采样在线估计，不引入额外计算开销。下限 clamp 为 $\epsilon = 1e^{-4}$ 防止除零
    - 设计动机：避免额外的采样或模型复制成本

### 损失函数 / 训练策略
- 伪奖励：$\hat{R}_\theta(y,x) = \min((\frac{p_x(y)}{\pi_\theta(y|x)})^{1-\alpha} - 1, M)$
- 梯度：$\nabla_\theta \mathcal{L} = \mathbb{E}_{x,y\sim\pi_\theta}[-\hat{A}^f(y,x) \nabla_\theta \log \pi_\theta(y|x)]$
- Baseline：leave-one-out 每上下文伪奖励均值
- 训练细节：4×A100，512序列/步，200迭代（约3轮），最大响应1024 tokens，float16

## 实验关键数据

### 主实验
在 Lean 定理证明任务上（10K训练题，200测试题）的 pass@k 结果：

| 方法 | pass@1 | pass@16 | pass@256 | 特点 |
|------|--------|---------|----------|------|
| Base-SFT | 低 | 中 | 中高 | 多样但不精准 |
| GRPO (β=0) | **高** | 中 | 低 | 精准但多样性崩溃 |
| GRPO (High-KL) | 中高 | 中高 | 中高 | KL惩罚缓解 |
| Rw-Ulkly | 中高 | 中高 | 中高 | 排名偏好保多样性 |
| Pass@k训练 | 中 | 中高 | 高 | 专门优化覆盖率 |
| α-DPG (α=0.999) | **高** | **高** | 中高 | 接近RLVR精度+更好覆盖 |
| α-DPG (α=0.25) | 中 | **高** | **最高** | 最佳覆盖率 |

### 消融实验
| α值 | 行为 | 精度 (pass@1) | 覆盖率 (pass@256) |
|-----|------|--------------|------------------|
| α=0.25 | 强 mass-covering | 中等改善 | 最高，超越所有方法 |
| α=0.5 (Hellinger) | 平衡 | 中等 | 高 |
| α=0.75 | 偏 mode-seeking | 较高 | 中高 |
| α=0.999 | 接近 Reverse KL | 最高 | 与GRPO相当 |
| 不同 α 的 Pareto 前沿 | 全部在或接近前沿 | 连续可控 | 连续可控 |

### 关键发现
- **α-DPG 模型几乎全部在 Pareto 前沿上**：单一超参 α 即可遍历精度-覆盖率权衡空间
- **α=0.999 通常支配 GRPO 和纯 RL 方法**：相似精度+更好覆盖率
- **α=0.25 实现所有方法中最高的 pass@256**：覆盖率超越 Base-SFT、Pass@k 训练和 KL 正则化
- **问题难度转移分析**：GRPO 和 α=0.999 使许多中等问题变简单，但也导致部分难题完全不可解；α=0.25 和 High-KL 更保守，仅丢失3个问题的可解性
- **多样性分析**：策略/前提的多样性（Shannon指数）与 pass@256 正相关、与 pass@1 负相关
- **困惑度分析**：所有模型生成的序列在基础模型下都有很低的困惑度，证实 RL 不创造新能力而是重新加权已有行为

## 亮点与洞察
- **"多样性损失在散度，不在目标"的核心洞察**：将问题从"RL 是否有害"重新框架化为"用什么散度逼近同一个目标分布"。这个视角转换非常深刻——目标分布（过滤）是完全合理的，问题出在 Reverse KL 的模式寻求特性
- **大统一框架**：α-DPG 将 REINFORCE/GRPO（α≈1）、KL-DPG（α=0）、拒绝采样微调（α=0，离线版）统一在同一框架下，仅通过一个标量 α 参数区分。这种理论简洁性很有价值
- **Pareto 可控性**：单一超参数即可连续遍历精度-覆盖率前沿，比调 KL 惩罚系数 β 更直观、效果更好
- **对 RLVR 的反思**：本文严格证明 RLVR ≡ Reverse KL 到过滤分布，配合近期"RL不创新只重排"的发现，清楚解释了为什么 RLVR 模型在大采样预算下不如基础模型

## 局限与展望
- **仅在 Lean 定理证明上验证**：是否推广到代码生成、数学推理等其他可验证任务未知
- **仅用 7B 模型**：对更大模型（70B+）的效果未测试
- **低 α 时训练不稳定**：α≤0.5 需要 clipping，但 clipping 引入偏差
- **配分函数估计噪声**：在线估计 $Z_x$ 在正确率很低的困难问题上不准确
- **可改进方向**：α 的课程学习（先低后高，兼顾早期覆盖和后期精度）；扩展到非二值奖励场景；与 MCTS 等搜索策略结合

## 相关工作与启发
- **vs GRPO/PPO（RLVR）**：它们隐式优化 Reverse KL，必然牺牲多样性。α-DPG 在 α≈1 时恢复其精度但保留更多覆盖
- **vs KL-DPG (Khalifa et al.)**：KL-DPG 用 Forward KL，保多样性但精度不够。α-DPG 统一了两端
- **vs Rw-Ulkly (He et al.)**：通过排名偏好惩罚保多样性，但缺乏理论指导。α-DPG 有散度族的信息几何理论支撑
- **vs Pass@k 训练**：直接优化 pass@k，但缺乏分布匹配的理论视角。α-DPG 在覆盖率上更优
- 这篇论文对理解"RL后训练到底在做什么"有深刻启发——下次看到 mode collapse 应首先检查用的散度而非目标函数

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 从散度选择的角度统一和解释RLVR的多样性问题，提出α-DPG是优雅的概念贡献
- 实验充分度: ⭐⭐⭐⭐ Lean任务上的实验全面（Pareto分析、难度转移、多样性分析、困惑度），但限于单一任务和模型
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，从引用到表达都体现深厚功底，表达精准
- 价值: ⭐⭐⭐⭐⭐ 对RLVR领域的核心问题提供了理论框架和实用解法，α参数的Pareto可控性极有实际价值

<!-- RELATED:START -->

## 相关论文

- [Reasoning Boosts Opinion Alignment in LLMs](reasoning_boosts_opinion_alignment_in_llms.md)
- [AbstRaL: Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking](abstral_augmenting_llms_reasoning_by_reinforcing_abstract_thinking.md)
- [How LLMs Learn to Reason: A Complex Network Perspective](how_llms_learn_to_reason_a_complex_network_perspective.md)
- [AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)
- [MVR: Multi-view Video Reward Shaping for Reinforcement Learning](mvr_multi-view_video_reward_shaping_for_reinforcement_learning.md)

<!-- RELATED:END -->
