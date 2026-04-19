---
title: >-
  [论文解读] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models
description: >-
   提出 SPELL 框架，让一个 LLM 同时扮演出题者、答题者和验证者三个角色进行自我博弈强化学习，无需人类标注即可持续提升长文本推理能力，在 6 个长上下文基准上一致提升性能。
tags:

---

# SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2509.23863](https://arxiv.org/abs/2509.23863)
- **代码**: [https://github.com/Tongyi-Zhiwen/Qwen-Doc](https://github.com/Tongyi-Zhiwen/Qwen-Doc)
- **领域**: reinforcement_learning
- **关键词**: self-play RL, long-context reasoning, GRPO, LLM, verifiable rewards, curriculum learning

## 一句话总结
提出 SPELL 框架，让一个 LLM 同时扮演出题者、答题者和验证者三个角色进行自我博弈强化学习，无需人类标注即可持续提升长文本推理能力，在 6 个长上下文基准上一致提升性能。

## 研究背景与动机
- **长上下文推理的困境**：RLVR（可验证奖励 RL）在数学、代码等有明确正确性判据的领域取得成功，但长文本推理面临两大瓶颈：
  1. 人类标注成本高且不可靠（LongBench-V2 人类准确率仅 25.1%）
  2. 缺乏程序化可验证的奖励信号
- **自我博弈的挑战**：答案可能语义正确但表达形式不同，字符串匹配和多数投票不可靠
- 核心思路：模型不仅要出题和答题，还要学会验证自己的答案

## 方法详解

### 整体框架：三角色自博弈循环
单一策略 $\pi_\theta$ 动态扮演三个角色：

1. **❓ 出题者 $\pi_\theta^{\text{que}}$**：从文档生成问答对 $(q, a)$
2. **📝 答题者 $\pi_\theta^{\text{res}}$**：根据文档回答问题，生成 $G$ 个独立采样
3. **✅ 验证者 $\pi_\theta^{\text{ver}}$**：判断答题者输出与参考答案是否语义等价

### 关键设计 1：自动课程学习
- **历史记忆 $\mathcal{H}$**：存储最近 $L=3$ 个已解决的问答对和对应文档
- **渐进上下文扩展**：每轮迭代扩充文档范围，问题需跨越更多文档
- **避免冗余**：已解决的问答作为条件，推动出题者生成更难的问题

### 关键设计 2：角色专属奖励

**验证者奖励**（自一致性）：
$$r_{i,j}^{\text{ver}} = \mathbb{I}(v_{i,j} = v_i^{\text{ver}})$$
即每个判断与多数投票一致则得分。

**答题者奖励**（规则 + 验证者融合）：
$$r_i^{\text{res}} = \max(\mathcal{R}_{\text{rule}}(y_i, a), v_i^{\text{ver}})$$
取规则匹配（CEM）与验证者投票的最大值，避免假阴性噪声。

**出题者奖励**（高斯难度控制）：
$$r^{\text{que}} = \exp\left(-\frac{(\bar{r}^{\text{res}} - 0.5)^2}{2 \cdot (0.5/3)^2}\right)$$
以答题者成功率 0.5 为中心的高斯函数，太简单或太难的问题都不得分。格式错误或脱离文档的问题受惩罚。

### 关键设计 3：角色动态采样与统一更新
- 原始样本严重不均衡：1 个出题 vs $G$ 个答题 vs $G^2$ 个验证
- 答题者：保留奖励方差 > 0 的组；出题者：正负等量采样；验证者：与多数投票一致子采样
- 最终训练集约缩减至 $1/G$

**统一策略更新**（GRPO）：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathcal{J}_{\text{GRPO}}^{\text{que}}(\theta) + \mathcal{J}_{\text{GRPO}}^{\text{res}}(\theta) + \mathcal{J}_{\text{GRPO}}^{\text{ver}}(\theta)$$

三角色共享一个策略，联合优化后进入下一轮自博弈。

## 实验关键数据

### 主实验：12 个开源 LLM × 6 个基准（16K 上下文）

| 模型 | 基线 Avg. | +RLVR Avg. | +SPELL Avg. | SPELL 提升 |
|------|----------|-----------|------------|----------|
| Qwen2.5-7B (base) | 26.7 | 40.8 | **40.6** | +13.9 |
| Qwen2.5-14B (base) | 37.3 | — | **51.7** | +14.4 |
| Qwen2.5-32B (base) | — | — | — | +9.1 |
| Qwen2.5-7B-Instruct | — | — | — | +9.0 |
| R1-Distill-Qwen-14B | — | — | — | +3.4 |
| Qwen3-30B-A3B-Thinking | — | — | — | +2.0 |

> SPELL 训练的基座模型甚至超过同规模 instruct 模型（需大量人类标注数据）。

### 消融实验：各组件贡献

| 消融设置 | DocMath | Frames | LB-MQA | LB-V2 | Avg. |
|---------|--------|--------|--------|-------|------|
| SPELL (完整) | 最优 | 最优 | 最优 | 最优 | 最优 |
| 无验证者 (只用 CEM) | 下降 | 下降 | 下降 | — | 下降 |
| 无课程学习 | 下降 | 下降 | 下降 | — | 下降 |
| 无出题者难度奖励 | 下降 | 下降 | — | — | 中等下降 |
| 静态 RLVR (DeepSeek-R1 数据) | 弱于 SPELL | — | — | — | 弱于 SPELL |

### 关键发现
1. SPELL 在 base/instruct/reasoning 三种模型类型和 dense/MoE 两种架构上均有一致提升
2. 训练 base 模型超越 instruct 模型，表明自博弈比人类标注更高效
3. 相比静态 RLVR（用 DeepSeek-R1 合成数据），SPELL 的动态课程更有效
4. 验证者的自一致性训练提供了可靠的语义奖励，弥补规则匹配的不足
5. Qwen3-30B-A3B-Thinking 在 pass@4 上超越 Gemini-2.5-pro

## 亮点与洞察
- **完全无监督**：不依赖任何人类标注或外部模型生成的数据
- **三角色统一策略**：一个模型同时学会出题、答题和验证，形式优雅
- **自动课程**：随训练进展，问题自然变难、上下文自然变长
- **可扩展性**：在强推理模型（Qwen3-30B Thinking）上仍有提升，未触及天花板

## 局限性
- 验证者的自一致性训练可能在某些情况下不够准确（如语义模糊的答案）
- 最大输入长度限制为 16K tokens，对超长文档（>100K）场景可能受限
- 出题者可能生成超出文档范围的幻觉问题（虽有 grounding filter）
- 三角色联合训练的计算成本高于标准 SFT 或单角色 RL

## 相关工作
- **长上下文 RL**: Wan et al. (2025) 首先将 RLVR 扩展到长上下文
- **自我博弈**: Absolute Zero (Zhao et al., 2025) 生成单轮编程任务；SPAG (Cheng et al., 2024) 对抗性禁忌游戏
- **RLVR**: DeepSeek-R1 (Guo et al., 2025), GRPO (Shao et al., 2024)
- **长上下文模型**: LongBench (Bai et al., 2024), Frames (Krishna et al., 2025)

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 三角色自博弈 + 自动课程用于长上下文 RL，思路独特
- 理论深度: ⭐⭐⭐ — 主要是方法论创新，缺乏理论分析
- 实验充分性: ⭐⭐⭐⭐⭐ — 12 个模型 × 6 个基准 × 详细消融
- 实用价值: ⭐⭐⭐⭐⭐ — 无需标注数据即可提升长上下文能力，实用性极强

<!-- RELATED:START -->

## 相关论文

- [LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [Does Self-Evaluation Enable Wireheading in Language Models?](../../AAAI2026/reinforcement_learning/does_self-evaluation_enable_wireheading_in_language_models.md)
- [Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models](co-rewarding_stable_self-supervised_rl_for_eliciting_reasoning_in_large_language.md)

<!-- RELATED:END -->
