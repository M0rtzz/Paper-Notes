---
title: >-
  [论文解读] On Entropy Control in LLM-RL Algorithms
description: >-
  [ICLR 2026][机器人][熵控制] 从理论解释为什么传统熵正则化在LLM-RL中几乎无效（因极大动作空间+稀疏最优导致熵偏差压倒优化增益），提出AEnt方法用截断熵（在缩小的token空间上计算）+自适应系数来有效平衡偏差与收益，在数学推理上持续超越baseline。
tags:
  - "ICLR 2026"
  - "机器人"
  - "熵控制"
  - "RLVR"
  - "LLM-RL"
  - "策略优化"
  - "探索-利用"
---

# On Entropy Control in LLM-RL Algorithms

**会议**: ICLR 2026  
**arXiv**: [2509.03493](https://arxiv.org/abs/2509.03493)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 熵控制, RLVR, LLM-RL, 策略优化, 探索-利用

## 一句话总结
从理论解释为什么传统熵正则化在LLM-RL中几乎无效（因极大动作空间+稀疏最优导致熵偏差压倒优化增益），提出AEnt方法用截断熵（在缩小的token空间上计算）+自适应系数来有效平衡偏差与收益，在数学推理上持续超越baseline。

## 研究背景与动机

**领域现状**：策略梯度方法（PPO/GRPO/DAPO）是LLM-RL的主流。传统RL中熵正则化（SAC/A3C/PPO）通过保持策略随机性防止过早收敛，效果显著。

**现有痛点**：实验发现熵正则化在LLM-RL中几乎无增益。Cui等人观察到不同熵系数对验证准确率影响微乎其微。这与机器人/游戏RL中的显著效果形成矛盾。

**核心矛盾**：理论上熵正则化有优化优势（改善收敛），但在LLM中引入的偏差 $O(H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*(s_0)|^{1/H}})$ 随动作空间 $|\mathcal{A}|$ 和最优稀疏度增大而剧增。LLM词汇表~10万+、最优token极其稀疏→偏差远大于优化增益。

**切入角度**：既然全词汇表上的熵偏差太大，就在更小的合理token空间上计算截断熵——只鼓励在"合理候选"中探索而非在整个词汇表中。

## 方法详解

### 整体框架

这篇论文先回答一个困惑、再给一套解法。困惑是：在传统 RL 里立竿见影的熵正则化，搬到 LLM-RL（PPO/GRPO/DAPO 这类策略梯度方法）上几乎不涨点。论文用两条性能界把原因量化出来，再据此提出 AEnt——把熵限制在一小撮"合理候选 token"上计算，从而保住探索收益、压掉偏差。

整套分析建立在两条命题上。**无熵控制时**（Proposition 1），策略熵是策略梯度范数的上界 $\|\nabla V^{\pi_\theta}\| \leq 2\mathcal{H}(\pi_\theta)$——熵一旦崩溃，梯度就趋零、学习随之停滞；此时性能差被界为 $V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon}{C^{\pi_\theta}(s_0)}$。**加上传统熵正则化后**（Proposition 2），性能界变成

$$V^{\pi^*} - V^{\pi_\theta} \leq \frac{\epsilon^2}{2\lambda C_\lambda} + \lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$$

第一项 $\epsilon^2/2\lambda$ 是熵带来的优化增益（收敛更好），第二项 $\lambda H\log\frac{|\mathcal{A}|}{|\mathcal{A}_H^*|^{1/H}}$ 是它引入的偏差。关键观察是：LLM 词汇表 $|\mathcal{A}|$ 高达 10 万量级、最优 token 又极其稀疏，偏差项被这个巨大的 $\log|\mathcal{A}|$ 撑爆，彻底压过优化增益——这正是传统熵在 LLM 上失灵、在机器人/游戏（$|\mathcal{A}|$ 只有数十到数百）上有效的根本差异。AEnt 的全部设计都围绕把这个 $\log|\mathcal{A}|$ 换成 $\log k$ 展开。

### 关键设计

**1. 截断熵（Clamped Entropy）：只在 top-k 候选里算熵，把偏差从 $\log|\mathcal{A}|$ 压到 $\log k$。**

前面的偏差项之所以爆炸，是因为熵鼓励模型在整个 10 万词表上保持随机——可绝大多数 token 根本不该被探索。截断熵的做法是先取当前状态下概率最高的 top-k token 构成子空间 $\mathcal{A}_k(s) = \text{top-k tokens}$，在这个子空间上把策略重归一化 $\tilde{\pi}(a|s) = \pi(a|s)/\sum_{a' \in \mathcal{A}_k} \pi(a'|s)$，再用 $\tilde{\pi}$ 计算熵。这样探索只发生在"合理候选"之间，偏差项里的 $\log|\mathcal{A}|$ 随之降为 $\log k$（$k \ll |\mathcal{A}|$），优化增益却基本保留——从 top-1000 里随机选，显然比从全词表里随机选合理得多。

**2. 自适应系数：按当前截断熵动态调 $\lambda$，省掉手调又适配训练全程。**

固定的熵系数 $\lambda$ 有个老问题：训练早期熵高、后期熵低，同一个 $\lambda$ 顾此失彼。AEnt 让 $\lambda$ 跟着当前截断熵走——截断熵高（说明已经足够随机）就把 $\lambda$ 调小，截断熵低（探索不足）就把 $\lambda$ 调大，从而在训练不同阶段自动维持合适的探索强度。

### 损失函数

总目标是在原策略优化损失上加一项截断熵正则：

$$\mathcal{L} = \mathcal{L}_{\text{PO}}(\theta) + \lambda \cdot \min(\mathcal{H}_k(\pi_\theta), H_{\text{target}})$$

其中 $\mathcal{H}_k$ 是截断熵，先截断到目标熵 $H_{\text{target}}$ 再由自适应系数 $\lambda$ 调节——既防止熵被无限推高，又保证探索强度随训练自适应。

## 实验关键数据

### 数学推理

| 方法 | AIME | AMC | MATH500 | Minerva |
|------|------|-----|---------|---------|
| GRPO (无熵) | 基线 | 基线 | 基线 | 基线 |
| GRPO + 传统熵 | ~基线 | ~基线 | ~基线 | ~基线 |
| **GRPO + AEnt** | **↑** | **↑** | **↑** | **↑** |

### 多模型验证

| 基础模型 | AEnt增幅 | 说明 |
|---------|---------|------|
| Qwen2.5-Math-1.5B | 显著 | 小模型获益更多 |
| Qwen2.5-7B | 显著 | 大模型也有效 |

### 关键发现
- 传统熵正则化确实几乎无增益（验证了之前的观察）
- AEnt在所有基准和模型上持续改善→截断熵有效解决了偏差问题
- 合成MDP实验证实：当最优动作数<5且$|\mathcal{A}|=10^5$时传统熵无效，AEnt仍有效
- 自适应系数比固定系数更稳定

## 亮点与洞察
- **理论解释LLM-RL的长期困惑**：为什么传统熵在LLM中不work？因为 $O(H\log|\mathcal{A}|)$ 的偏差在$|\mathcal{A}|=10^5$时压倒了一切。这个解释简洁有力。
- **截断熵的直觉**：不应鼓励模型探索所有10万个token，只应在合理候选中保持多样性。从top-1000中随机选比从全词汇表中随机选合理得多。
- **偏差-增益权衡的量化**：Proposition 1和2给出了可操作的理论指导——当$\log|\mathcal{A}|$大且最优稀疏时，需要特殊处理。

## 局限与展望
- top-k的k需要手动设置——自适应k可能更好
- 理论分析基于softmax策略假设，实际LLM有更复杂的结构
- 仅在数学推理上验证，代码/通用推理效果未知
- 截断熵可能过度限制某些需要大范围探索的场景

## 相关工作与启发
- **vs DAPO**: DAPO通过clip/约束间接控制熵，AEnt直接在截断空间上做正则化
- **vs Cui等人**: 他们观察到熵bonus无效但未给出理论解释，本文提供了解释+解法
- **vs SAC**: SAC的熵正则化在机器人任务中成功因为 $|\mathcal{A}|$ 小（数十到数百），LLM的 $|\mathcal{A}|$ 差几个数量级

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论解释+截断熵方案都有洞察力
- 实验充分度: ⭐⭐⭐⭐ 多模型+多基准+合成MDP验证
- 写作质量: ⭐⭐⭐⭐ 理论与实践结合自然
- 价值: ⭐⭐⭐⭐⭐ 解决了LLM-RL训练中一个重要的实践问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](../../ICML2026/robotics/towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ICLR 2026\] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)
- [\[ECCV 2024\] LLM as Copilot for Coarse-Grained Vision-and-Language Navigation](../../ECCV2024/robotics/llm_as_copilot_for_coarse-grained_vision-and-language_navigation.md)
- [\[AAAI 2026\] Test-driven Reinforcement Learning in Continuous Control](../../AAAI2026/robotics/test-driven_reinforcement_learning_in_continuous_control.md)

</div>

<!-- RELATED:END -->
