---
title: >-
  [论文解读] Enhancing Decision-Making of Large Language Models via Actor-Critic
description: >-
  [ICML2025][强化学习][Actor-Critic] 提出 LAC（LLM-based Actor-Critic）框架，通过 token logits 的正/负结果概率比构建 Q 函数（Critic），并用 KL 约束闭式解实现无梯度策略优化（Actor）…
tags:
  - "ICML2025"
  - "强化学习"
  - "Actor-Critic"
  - "LLM Agent"
  - "无梯度策略优化"
  - "Q值估计"
  - "序列决策"
---

# Enhancing Decision-Making of Large Language Models via Actor-Critic

**会议**: ICML2025  
**arXiv**: [2506.06376](https://arxiv.org/abs/2506.06376)  
**代码**: [GitHub](https://github.com/drdh/LAC)  
**领域**: LLM决策 / 强化学习 / Agent  
**关键词**: Actor-Critic, LLM Agent, 无梯度策略优化, Q值估计, 序列决策

## 一句话总结

提出 LAC（LLM-based Actor-Critic）框架，通过 token logits 的正/负结果概率比构建 Q 函数（Critic），并用 KL 约束闭式解实现无梯度策略优化（Actor），在 ALFWorld、BabyAI-Text、WebShop 三个基准上用 7B/8B 模型超越 GPT-4 + ReAct。

## 研究背景与动机

LLM 用于序列决策存在两条路线，各有明显短板：

**直接用 LLM 先验当策略**（如 ReAct）：自回归逐步生成动作，缺少长期规划能力，在多步任务中局部最优但全局失败。

**加入规划与动作评估**（如 RAP、LATS）：用 LLM 做 rollout 或 MCTS 评估候选动作，但严重依赖模拟精度，轻量模型 rollout 偏差大时效果急剧下降。

核心问题：两条线都把 **LLM 先验知识** 和 **动作评估信息** 割裂开来——前者不做规划，后者忽视先验。LAC 的目标是把二者在一个有理论保证的框架里统一起来。

## 方法详解

### 整体框架

LAC 在每个时间步执行两步：

1. **Critic 评估**：对策略 $\pi_{\text{LLM}}$ 采样的 $n$ 个候选动作分别计算 Q 值；
2. **Actor 优化**：用 Q 值对先验策略做 KL 约束下的闭式更新，选出最优动作。

### 4.1 Critic：基于 Token Logits 的 Q 值估计

**核心思想**：不让 LLM 直接输出评分（不稳定），而是利用 LLM 对特殊 token（"GOOD"/"BAD" 或 "SUCCESS"/"FAILURE"）的 logits 来反映其对任务成功/失败的内在信念。

**Q 值公式**：

$$Q_{\text{LLM}}(g, h_t, a_t^i, u_t^i) = \log \frac{P(y_w \mid g, h_t, a_t^i, u_t^i)}{P(y_l \mid g, h_t, a_t^i, u_t^i)}$$

其中 $y_w$、$y_l$ 分别对应成功/失败信号，$u_t^i$ 是通过前向世界模型 $f_{\text{LLM}}$ 预测的未来轨迹。Q 值通过 logistic 函数与成功概率正相关：

$$P(y_w \mid \cdot) = \frac{1}{1 + \exp(-Q_{\text{LLM}}(\cdot))}$$

**提升评估精度的两个技巧**：

- **轨迹 Rollout**：对每个候选动作用 LLM 预测若干步未来轨迹，再基于扩展轨迹计算 Q 值，捕获延迟后果。
- **上下文反思（Reflection）**：在采样和评估前，让 LLM 生成简短反思（如"我已经找到了 object-X，这一步是 GOOD"），类似 CoT，帮助策略避免重复错误，也提升 Critic 准确性。

### 4.2 Actor：KL 约束无梯度策略优化

将策略改进表述为 KL 约束优化问题：

$$\max_{\pi} \; \mathbb{E}_{a_t^i \sim \pi}[Q_{\text{LLM}}(\cdot)] - \frac{1}{\alpha} D_{\text{KL}}[\pi \| \pi_{\text{LLM}}]$$

闭式最优解：

$$\pi_{\text{new}}(a_t^i \mid g, h_t) \propto \pi_{\text{LLM}}(a_t^i \mid g, h_t) \cdot \exp\!\big(\alpha \, Q_{\text{LLM}}(g, h_t, a_t^i, u_t^i)\big)$$

- $\alpha = 0$ 退化为纯先验策略（ReAct）；$\alpha \to \infty$ 退化为纯 Critic 选动作。
- **无需梯度回传**，仅需加权先验概率即可完成策略更新，计算开销极低。
- KL 项保证新策略不会偏离先验太远，平衡了先验知识与 Critic 评估。

### 算法流程

1. 从 $\pi_{\text{LLM}}$ 采样 $n$ 个候选动作；
2. 对每个候选动作用 $f_{\text{LLM}}$ 做 rollout 预测未来轨迹；
3. 用正/负 token logits 计算 Q 值；
4. 按闭式解加权更新策略概率；
5. 选概率最高的动作执行。

## 实验关键数据

### 基准与动作空间

| 基准 | 动作类型 | 奖励类型 | 规模 |
|------|---------|---------|------|
| ALFWorld | 高层（如"去X拿Y"） | 二值 0/1 | 134 任务 |
| BabyAI-Text | 低层 6 原始动作 | 二值 0/1 | 8×8 网格 |
| WebShop | 近乎无穷（搜索+点击） | 连续 [0,1] | 网页购物 |

### 主要结果

- **ALFWorld**：LAC + Llama-3-8B 成功率显著超过 ReAct + GPT-4，也优于 RAP、LATS 等规划方法。
- **BabyAI-Text**：LAC 在所有子任务上一致领先，尤其在长步骤任务中优势明显。
- **WebShop**：LAC 在累积奖励和成功率两个指标上均为最优，证明框架对连续奖励场景同样有效。

### 消融实验

| 变体 | 效果 |
|------|------|
| LAC w/o critic | 性能显著下降，验证策略优化步骤的必要性 |
| LAC w/o rollout | 下降，说明未来轨迹预测对 Q 值精度重要 |
| LAC w/o reflection | 下降，反思机制帮助采样更好候选和更准评估 |
| critic-only | 下降，纯 Critic 不如结合先验 |

### 计算成本

- LAC 每步开销略高（额外的 Critic + rollout 推理），但因成功率高、完成步数少，**总 token 消耗和运行时间反而低于 RAP、LATS 等基线**。
- 成功任务平均步数：LAC 15.32 步 vs ReAct 17.75 步 vs RAP 16.36 步。

### 统计分析

| 指标 | 成功轨迹 | 失败轨迹 |
|------|---------|---------|
| log P("GOOD") 与时间步相关性 | +0.35 | -0.37 |
| log P("BAD") 与时间步相关性 | -0.32 | +0.38 |
| Q 值与时间步相关性 | +0.34 | -0.41 |

Q 值在成功轨迹中随时间步递增、失败轨迹中递减，验证了 Q 函数确实在追踪任务进展。

## 亮点与洞察

1. **Q 值估计方式巧妙**：不让 LLM 直接打分（极不稳定），而是利用正/负 token 的 logits 比取 log，公式简洁且物理含义清晰——就是成功-失败的对数几率比。
2. **闭式策略优化**：KL 约束下推导出指数加权更新的解析解，完全无需梯度，适合 LLM 场景的推理时计算。这个解等价于 AWR / DPO 系列方法中的策略更新形式，理论基础扎实。
3. **α 的连续谱诠释**：$\alpha=0$ 复现 ReAct，$\alpha \to \infty$ 复现纯 Critic，LAC 自动在两个极端之间找平衡。
4. **统计验证充分**：除了常规消融，额外做了 Q 值-时间步相关性分析和策略置信度分析，说明加权策略确实在"谁更自信就听谁"，而非盲目混合。
5. **7B 模型胜 GPT-4**：证明框架设计比模型规模更重要，对资源受限场景意义重大。

## 局限与展望

1. **反思仅在动作生成前使用**：可扩展到生成后对预测轨迹做反思并重采样。
2. **单步 rollout 扩展**：目前每个候选只展开一个节点，可接入树搜索（如 MCTS）获得更精确评估。
3. **连续奖励处理粗糙**：当前把"获得最高奖励"二值化处理，缺少对连续奖励的专门建模。
4. **未验证更大模型**：仅测试 7B/8B，对 70B+ 或最新推理模型（如 DeepSeek-R1）的效果未知。
5. **需要访问 token logits**：依赖模型输出 logits，不适用于仅提供 API 的闭源模型。

## 相关工作与启发

- **ReAct**（Yao et al., 2023）：推理+行动但无长期规划 → LAC 的"无 Critic"基线。
- **RAP**（Hao et al., 2023）：LLM 做世界模型+树搜索 → LAC 的 rollout 组件与之类似但更轻量。
- **LATS**（Zhou et al., 2024a）：MCTS + LLM → 计算开销大，LAC 用单步 rollout + 闭式优化替代。
- **ICPI**（Brooks et al., 2024）：LLM 实现策略迭代 → 在稀疏奖励下表现不佳。
- **DPO/AWR 系列**：KL 约束策略优化的理论基础，LAC 将其从训练时迁移到推理时决策。

## 评分

- 新颖性: ⭐⭐⭐⭐ — Q 值从 token logits 提取 + 推理时闭式策略优化的组合很新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个不同动作空间基准 + 四个基座模型 + 详细消融 + 统计分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐ — 对 LLM Agent 的推理时决策优化提供了简洁高效的范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](../../CVPR2025/reinforcement_learning/decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](../../ICLR2026/reinforcement_learning/flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICML 2025\] The Sample Complexity of Online Strategic Decision Making with Information Asymmetry and Knowledge Transportability](the_sample_complexity_of_online_strategic_decision_making_with_information_asymm.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)

</div>

<!-- RELATED:END -->
