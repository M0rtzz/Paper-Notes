---
title: >-
  [论文解读] CAMEL: Confidence-Gated Reflection for Reward Modeling
description: >-
  [ICML 2026][强化学习][奖励模型] 本文观察到 verdict token 的 log-probability margin 与判断正确率高度相关，据此提出 CAMEL —— 先用单 token 快速给出偏好判断，仅在低置信度时才触发反思生成，并用反事实前缀增强 GRPO 训练自我纠错能力…
tags:
  - "ICML 2026"
  - "强化学习"
  - "奖励模型"
  - "置信度门控"
  - "反思机制"
  - "GRPO"
  - "反事实前缀增强"
---

# CAMEL: Confidence-Gated Reflection for Reward Modeling

**会议**: ICML 2026  
**arXiv**: [2602.20670](https://arxiv.org/abs/2602.20670)  
**代码**: 暂未公开  
**领域**: 对齐RLHF / 奖励模型 / LLM推理  
**关键词**: 奖励模型、置信度门控、反思机制、GRPO、反事实前缀增强

## 一句话总结
本文观察到 verdict token 的 log-probability margin 与判断正确率高度相关，据此提出 CAMEL —— 先用单 token 快速给出偏好判断，仅在低置信度时才触发反思生成，并用反事实前缀增强 GRPO 训练自我纠错能力，在三个奖励模型 benchmark 上以 14B 参数取得 82.9% 的平均准确率（超过此前最佳 70B 模型 3.2%）。

## 研究背景与动机
**领域现状**：用作 LLM 对齐信号的奖励模型主要有两类范式。标量判别式（如 Skywork-Reward、ArmoRM）训练稳定、推理快，但只吐一个分数、不可解释；生成式 judge（如 J1、RM-R1）边推理边写理由再给判断，准确率更高，但每条样本都要生成几百到上千 token。

**现有痛点**：生成式 RM 的成本在工业部署中难以承受 —— RM-R1-DeepSeek-32B 在 RewardBench 上平均要生成约 900 token，在 RM-Bench 上约 1100 token，而其中大量样本本来就是"两个回复一好一坏"的容易题，根本不需要冗长推理。把简单和困难的样本一视同仁地生成 reasoning，是对计算预算的浪费。

**核心矛盾**：奖励模型存在明显的"效率 vs 表达力"trade-off。简单样本希望像标量模型一样秒杀，疑难样本希望像生成模型一样反思 —— 但目前没有合适的信号告诉模型"这道题到底难不难、要不要反思"。

**本文目标**：(1) 找到一个无需额外推理就能拿到的"题目难度"代理指标；(2) 用它做一个自适应路由的奖励模型，仅对真正困难的样本付出生成代价；(3) 让反思真的能纠错而不是 echo 原答案。

**切入角度**：作者观察到，当 prompt 让模型在 A/B 之间二选一时，verdict token 处的 log-probability margin（$c(x) = |\log P(A|x) - \log P(B|x)|$）天然就刻画了模型"有多确定"。在 Skywork-80K 上用 Qwen3-14B 统计发现：confidence 越高的样本，预测正确率越高（单调上升），且错误样本几乎全集中在低 margin 区。

**核心 idea**：用单 token margin 作为"无成本难度估计器"，构造"先快速判断 → 低置信时再反思"的两段式流程，并用反事实前缀 RL 训练让反思具备真实的自我纠错能力。

## 方法详解

### 整体框架
CAMEL 把奖励建模拆成两阶段。给定 $(q, r_a, r_b)$，模型先输出一个初始 verdict $v_0 \in \{\texttt{A}, \texttt{B}\}$；从这个 verdict token 的两个候选概率算出置信度 $c(x)$。若 $c(x) \geq \tau$（高自信），直接终止，$v_1 = v_0$，全程只花 1 个生成 token；若 $c(x) < \tau$，prompt 触发一段简短反思 $J$（"think again..."），再输出最终 verdict $v_1$。这一"先打分再决定要不要解释"的结构，配合 GRPO + 反事实前缀增强训练。

### 关键设计

**1. Confidence Score：把"题目难不难"压成一个免费拿到的单 token margin**

生成式 RM 最大的浪费在于对"两个回复明显一好一坏"的简单题也照样生成上千 token 推理。CAMEL 想要一个不花额外算力就能区分难易的信号，于是盯上了 verdict token 本身：当 prompt 让模型在 A/B 间二选一时，这个 token 处两个候选的 log-probability 差就天然刻画了模型有多确定。定义 $c(x) = |\log P_\theta(v=\texttt{A}|x) - \log P_\theta(v=\texttt{B}|x)|$，相当于模型对偏好的"势能差"。作者在 Skywork-80K 上画出 $c(x)$ 与正确率的关系，发现是一条强单调上升的曲线——置信度越高正确率越高，错误样本几乎全挤在低 margin 区。这意味着不必额外训一个难度估计器、也不必做第二次 forward，只要调一个阈值 $\tau$ 就能在准确率/成本曲线上自由滑动，是整个框架"零额外开销"的根基。

**2. Confidence-Gated 两段式 Prompt：把"要不要思考"的决策从语言层下放到 token 概率层**

光有难度信号还不够，得让它能在生成过程中真正掐断或放行反思。CAMEL 把传统"一段长 rationale → verdict"的结构因子化成 $v_0 \to$ 可选反思 $J \to v_1$：prompt 强制模型先吐一个不带解释的 verdict $v_0$ 占位，一次 forward 就能拿到它的 logit 并算出 $c(x)$；若 $c(x)\ge\tau$ 直接终止、$v_1=v_0$，全程只花 1 个生成 token；若 $c(x)<\tau$ 才触发一段简短反思（"think again..."），模型在看到自己的 $v_0$ 后再写理由并产出最终 $v_1$。关键在于"是否思考"这个离散决策被外置到 token 概率层面，而不是让模型用自然语言自己判断"我要不要再想想"——后者要么全 think 要么全不 think，而前者干净、可调、零回归，且难易样本走不同 token 路径却共用同一个 policy。

**3. Counterfactual Prefix Augmentation + GRPO：逼着反思真的去推翻自己，而不是复读初始判断**

两段式结构有个隐患：训练分布里初始判断 $v_0$ 多数本来就是对的，模型很容易学到"反思 = 把 $v_0$ 抄一遍"的捷径，反思就成了摆设。CAMEL 的解法是反事实增强——对每个样本 $(x,z)$ 复制两份，一份强制 $v_0=\texttt{A}$、另一份强制 $v_0=\texttt{B}$，RL 信用只施加到反思 $J$ 和最终 $v_1$ 上，$v_0$ 当作 context 不优化。reward 是二值的 $R=+1/-1$（最终 verdict 是否匹配 ground truth），用 GRPO 优化 $\max_\theta \mathbb{E}[R(v_1, z)] - \beta\, \mathbb{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})$。强制喂一个错误起点，模型就没法靠复读蒙混，只能在反思里真的去比对证据、推翻自己——自我纠错能力由此被逼出来。消融里去掉反事实，JudgeBench 直接掉 5 个点（74.2 → 69.1），印证了这一点。

### 损失函数 / 训练策略
两阶段训练：先在三个偏好数据集（Skywork-Reward-Preference-80K + Code-Preference-Pairs + Math-Step-DPO-10K）上对 Qwen3-14B 做 SFT，让模型学会基本的偏好格式；然后做一个 epoch 的 GRPO，配合反事实前缀，KL 系数 $\beta$ 控制偏离参考策略。inference 时默认 $\tau = 5$（可调节）。

## 实验关键数据

### 主实验
三个奖励模型 benchmark（RewardBench / RM-Bench / JudgeBench），对比一众标量 RM 与生成式 RM：

| 模型 | RewardBench | RM-Bench | JudgeBench | Avg |
|------|-------------|----------|------------|-----|
| INF-ORM-Llama3.1-70B (前 SOTA) | 95.1 | 73.8 | 70.2 | 79.7 |
| RM-R1-Qwen-Instruct-32B (生成式) | 89.0 | 73.1 | 64.8 | 75.6 |
| J1-Llama-70B | 93.3 | 82.7 | 60.0 | 78.7 |
| **CAMEL-Fast (14B, 1 token)** | 90.5 | 74.8 | 65.2 | 76.8 |
| **CAMEL-Reflection (14B, always)** | 92.8 | **84.2** | **71.6** | **82.9** |
| **CAMEL (gated, $\tau=5$)** | 92.4 | 81.9 | 69.1 | 81.1 |

CAMEL-Reflection 平均比前 SOTA 高 3.2%，CAMEL-Fast 用 1 个 token 就匹配甚至超过 RM-R1-Qwen-Instruct-32B 全程生成的结果，14B 参数打平/超过 70B baseline。

### 消融实验

| 配置 | RewardBench | RM-Bench | JudgeBench | Avg |
|------|-------------|----------|------------|-----|
| Qwen3-14B (无 tune) | 81.9 | 71.1 | 62.6 | 71.9 |
| Qwen3-14B + Reflection | 83.3 | 73.2 | 65.0 | 73.8 |
| Qwen3-14B-SFT | 90.6 | 72.7 | 64.8 | 76.0 |
| Qwen3-14B-GRPO (无反事实) | 91.2 | 83.5 | 62.9 | 79.2 |
| Qwen3-14B-GRPO + Reflection | 90.0 | 84.0 | 74.2 | 82.7 |
| **CAMEL (full)** | 92.4 | 81.9 | 69.1 | 81.1 |

### 关键发现
- 反思增益最显著的是推理密集型 benchmark：从 always-fast 到 always-reflect，RewardBench +2.3%、RM-Bench +9.4%、JudgeBench +6.4%，反映出后两个 benchmark 上更多样本属于"看一眼想不清"的疑难。
- 反事实前缀是关键：去掉它后 GRPO+Reflection 在 JudgeBench 上掉 5 个点（74.2 → 69.1），表明没有反事实，反思就退化成"复读初始判断"。
- Pareto 前沿：在 RewardBench 和 RM-Bench 上 CAMEL 严格优于 RM-R1 —— RM-R1-DeepSeek-32B 平均生成 900–1100 token 才达到 87/74 上下，CAMEL 用 1 token 就接近，调 $\tau$ 后用更少 token 同时超过它。
- 训练后置信度分布反而左移（更保守），符合"模型学会区分确定与不确定"的预期；自我纠错混淆矩阵显示反思在 RewardBench 净增 77 条正确、RM-Bench 净增 1233 条。

## 亮点与洞察
- **"免费的难度信号"**：单 token margin 几乎没有额外开销，却能稳定预测正确率。这是个非常 portable 的 trick —— 任何二选一的判别任务（multiple choice QA、安全分类、tool selection）都能直接复用，作为路由 / 拒答 / 不确定性估计。
- **把"是否思考"显式外置**：之前 chain-of-thought 类工作经常让模型自己决定要不要 thinking，结果要么全 think 要么全不 think。CAMEL 在 token 概率层面做硬决策，干净、可调、零回归。
- **反事实前缀是 RL 训练的隐含杀招**：很多 self-correction 工作苦于"模型不愿改答案"，本质都是因为训练分布里 $v_0$ 几乎都对。强制喂错起点是个通用 fix，能迁移到 self-refinement、self-debate 等场景。
- 整体上把奖励建模重新构造为"自适应的两段计算"，比死磕 scalar vs generative 二选一更工程化。

## 局限与展望
- 阈值 $\tau$ 是全局固定的，但不同任务/领域的置信度分布并不对齐 —— 安全任务普遍 confidence 偏高，数学任务偏低，理想做法是动态/分桶 $\tau$。
- 仅在 Qwen3-14B 这一个底座上验证，scaling law（70B/100B+ 是否还保持"低置信=难题"的相关性）不清楚；小模型上 margin 可能噪声太大而失效。
- 反思 token 长度未严格控制，作者没汇报 reflection 段平均多长。若反思冗长，"门控"省下的成本会被部分抵消。
- 改进方向：(a) 把 $\tau$ 也学出来；(b) 引入多级反思（短/长/极长）做更细粒度路由；(c) 把这套架构嵌入到 actor-critic 风格的 RLHF pipeline 里作 critic 用。

## 相关工作与启发
- **vs RM-R1 (生成式 RM SOTA)**：RM-R1 用 distilled rubric + RL with verifiable reward，每条样本都生成长 rationale；CAMEL 共享相同训练数据但加了门控，accuracy 更高 token 更少，建立严格更优的 Pareto 前沿。
- **vs Generative RM (J1, Critic-RM)**：J1 / Critic-RM 强调 explicit reasoning trace 提升判断质量；CAMEL 借鉴其反思机制但拒绝"无差别 reasoning"。
- **vs Self-Consistency / Self-Refine**：那些方法靠多次采样投票或自我修正，CAMEL 用单次 forward 的 margin 直接决定是否细化，避免重复采样代价。
- **vs uncertainty-based abstention**：传统 selective prediction 用 confidence 决定要不要回答，CAMEL 用它决定要不要"再想想"，是 conditional compute 的另一种范式。

## 评分
- 新颖性: ⭐⭐⭐⭐ 单 token margin 做难度估计 + 反事实前缀的组合是清爽的新框架，但每个组件单独看都不算颠覆。
- 实验充分度: ⭐⭐⭐⭐ 三个主流 benchmark + Pareto curve + 消融 + 自我纠错分析齐全，缺多 backbone 验证。
- 写作质量: ⭐⭐⭐⭐⭐ 动机—观察—方法—实验的逻辑链特别顺，公式与图配合到位，读完就能复现核心思想。
- 价值: ⭐⭐⭐⭐⭐ 部署友好的 14B 奖励模型超过 70B baseline，对工业 RLHF pipeline 直接可用，trick 还能跨任务迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] RM-R1: Reward Modeling as Reasoning](../../ICLR2026/reinforcement_learning/rm-r1_reward_modeling_as_reasoning.md)
- [\[ICML 2026\] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models](one_bias_after_another_mechanistic_reward_shaping_and_persistent_biases_in_langu.md)
- [\[CVPR 2026\] MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning](../../CVPR2026/reinforcement_learning/msrl_scaling_generative_multimodal_reward_modeling.md)
- [\[ACL 2026\] LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations](../../ACL2026/reinforcement_learning/lovec_reinforcement_learning_for_better_verbalized_confidence_in_long-form_gener.md)
- [\[ICLR 2026\] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](../../ICLR2026/reinforcement_learning/stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)

</div>

<!-- RELATED:END -->
