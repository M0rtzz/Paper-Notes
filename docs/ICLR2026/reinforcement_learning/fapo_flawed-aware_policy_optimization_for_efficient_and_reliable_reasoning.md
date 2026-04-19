---
title: >-
  [论文解读] FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning
description: >-
  [ICLR2026][RLVR] 针对 RLVR 训练中"答案正确但推理有缺陷"的 flawed-positive rollout 问题，提出 FAPO 算法：用 GenRM 检测缺陷推理，通过无参数奖励惩罚机制实现"先利用后抑制"的自然学习轨迹，同时提升结果正确性、过程可靠性和训练稳定性。
tags:
  - ICLR2026
  - RLVR
  - flawed positives
  - reward shaping
  - generative reward model
  - process reward
  - GRPO
---

# FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning

**会议**: ICLR2026  
**arXiv**: [2510.22543](https://arxiv.org/abs/2510.22543)  
**代码**: [fapo-rl.github.io](https://fapo-rl.github.io)  
**领域**: reinforcement_learning  
**关键词**: RLVR, flawed positives, reward shaping, generative reward model, process reward, GRPO

## 一句话总结
针对 RLVR 训练中"答案正确但推理有缺陷"的 flawed-positive rollout 问题，提出 FAPO 算法：用 GenRM 检测缺陷推理，通过无参数奖励惩罚机制实现"先利用后抑制"的自然学习轨迹，同时提升结果正确性、过程可靠性和训练稳定性。

## 背景与动机
RLVR（Reinforcement Learning with Verifiable Rewards）是当前提升 LLM 推理能力的主流范式，模型通过探索推理轨迹、利用正确答案作为正信号来优化策略。然而，标准的 rule-based outcome reward 仅检查最终答案是否正确，无法区分推理过程的质量。

这导致了一个严重问题：**flawed-positive rollouts**——模型通过猜答案（answer-guessing）或跳跃推理（jump-in-reasoning）等不可靠方式碰巧得到正确答案，却获得与完全正确推理相同的正奖励。这些缺陷推理模式在训练中被持续强化，最终限制模型的推理上限。

作者对 Qwen2.5-Math-7B、Llama3.3-70B 等模型的分析表明，flawed positives 在正确 rollout 中占比高达 20%–40%，且在整个 RL 训练过程中持续存在（约 30% 的比例几乎不变）。

## 核心问题
1. **Flawed positives 的双面性**：早期训练阶段，模型能力不足以产生完全正确的推理，flawed positives 作为"跳板"帮助快速获得能力增长；但后期它们阻碍模型向真正的问题求解能力进化
2. **如何检测 flawed positives**：现有模型要么过度批评（高 recall 低 precision），要么参数量过大不适合在线 RL 使用
3. **如何平衡利用与抑制的时机**：需要一个自适应机制，在热身阶段允许利用、在精炼阶段逐步抑制

## 方法详解

### 1. Flawed-Positive 检测：FAPO-GenRM
在 Qwen3-4B-Instruct 基础上，通过 RL 训练一个紧凑高效的生成式奖励模型（GenRM），奖励设计为：

$$R_{\text{FAPO-GenRM}} = R_{\text{Outcome}} + R_{\text{Process}}$$

- **$R_{\text{Outcome}}$**：结果奖励，预测正确/错误（+1/-1）
- **$R_{\text{Process}}$**：步骤级惩罚，仅在正确检测到 flawed positive 时生效，值为 $-|\hat{t}_\theta - t^*|/n$，其中 $\hat{t}_\theta$ 是预测错误位置，$t^*$ 是真实错误位置，$n$ 是总步数

这一设计的两个关键点：
- **超越猜测**：过程惩罚迫使模型真正定位错误位置，而非仅猜测"是否有缺陷"
- **自然奖励转移**：早期以结果正确性为主（$-1 \to 1$ 增益大），后期结果饱和后自动转向过程优化

训练数据 FAPO-Critic-85K 通过多个 LLaMA/Qwen 系列模型（7B–70B）在 DAPO-Math-17K 上生成 rollout，由 Qwen3-32B 标注步骤级错误位置。

### 2. Flawed-Positive 惩罚：自适应奖励调整
检测到 flawed positives 后，对 RL 训练中的奖励进行调整：

$$R_{\text{FAPO}}(o, a^* | \theta) = R_{\text{RLVR}}(o, a^*) + R_\Delta(o, a^* | \theta)$$

其中 $R_\Delta = -\lambda$（当 rollout 被检测为 flawed positive 时），否则为 0。默认 $\lambda = 1$，即将 flawed positive 的奖励从 +1 降至 0。

**自然优化转移机制**：设当前 rollout 中正样本占比 $\alpha$、负样本占比 $\beta$，学习进度 $\rho = \alpha/\beta$。
- 当 $\rho < 1$（负样本占多数，即热身阶段）：flawed positives 仍有正优势值，被利用
- 当 $\rho > 1$（正样本占多数，即精炼阶段）：flawed positives 的优势值接近或低于零，被自然抑制
- 当 $\rho > 3$：正样本优势值被缩放，训练更稳定

$\lambda = 1$ 的选择来自 majority-guided 策略，使转移点恰好在 $\rho = 1$，无需额外超参数。

### 3. 工程架构
- GenRM 作为外部 LLM 服务部署在计算集群上，与 rollout 推理和 actor 训练异步解耦
- 多 worker + 路由器实现负载均衡
- 通过 overlong reward 策略和 checkpoint 选择控制 GenRM 的 token 预算
- 总训练时间仅增加不到 20%

## 实验关键数据

### GenRM 检测性能
- FAPO-GenRM-4B 在 FlawedPositiveBench 和 ProcessBench 上超越了教师模型 Qwen3-32B
- 相比 Qwen3-4B-Instruct 基线和 Qwen2.5-Math-PRM-72B 等强基线均有显著提升
- 解决了现有模型"过度批评"（高 recall 低 precision）的问题

### 推理性能（Qwen2.5-Math-7B + GRPO 基线）
- **AIME24 / AIME25 / GPQA-Diamond** 三个基准上，FAPO 在几乎所有中间 checkpoint 上均优于基线
- Flawed-positive 比例显著降低（从约 30% 大幅下降）
- 训练曲线更平滑，后期无明显性能下降
- Token 预算未增加（不依赖更长的 response 获得提升）

### 消融实验
- 更强的 GenRM → 更好的最终 RL 性能（检测精度与最终性能正相关）
- 自我纠正分析：FAPO 在后期自然转向完全正确的 rollout，响应长度缩短，推理更高效
- Step-ratio reward（按正确步骤比例给分）会导致 reward hacking——模型只输出高置信度步骤，跳过不确定的步骤

## 亮点
1. **对 flawed positives 的系统性分析**：首次揭示其"早期跳板、后期障碍"的双面角色，为 RLVR 训练提供了新视角
2. **无参数的自适应机制**：$\lambda=1$ 由理论推导得出，不引入额外超参数，优化方向自然随训练进展转移
3. **紧凑高效的 GenRM**：4B 参数模型超越 32B 教师模型，且与 RL 训练异步解耦，仅增加不到 20% 训练时间
4. **全面的验证**：不仅报告最终性能，还展示全过程中间 checkpoint 的表现，充分说明训练稳定性

## 局限与展望
1. GenRM 引入额外推理开销，虽然目前控制在 20% 以内，但在更大规模系统中可能成为瓶颈
2. FlawedPositiveBench 基于 ProcessBench 构建，评价覆盖面有限
3. 实验主要在数学推理和通用 QA 上验证，对代码生成等更复杂的可验证任务尚未充分探索
4. GenRM 本身也可能被 reward hacking——虽然论文讨论了这一风险，但长期训练的稳健性有待进一步验证
5. 异步架构设计是工程妥协，全同步方案可能有更好的系统效率

## 与相关工作的对比
| 方法 | 奖励类型 | 是否处理 flawed positives | 是否无参数 | 特点 |
|------|----------|--------------------------|-----------|------|
| 标准 RLVR | 二值 outcome | 否 | 是 | 简单但强化缺陷推理 |
| PRM (判别式) | 步骤级分数 | 间接 | 否 | 密集奖励，易被 hacking |
| Step-ratio reward | 步骤比例 | 间接 | 否 | 导致跳跃推理 |
| **FAPO** | outcome + 惩罚 | **直接检测+自适应惩罚** | **是** | 自然学习轨迹，稳定高效 |

## 启发与关联
- FAPO 的"先利用后抑制"思路可推广到其他 RL 场景中错误信号的处理
- GenRM 的 step-wise RL 训练方法可用于提升任何过程级评估模型（如代码 review 模型）
- 论文对 reward hacking 的分析（step-ratio reward 的失败案例）对设计新的奖励信号有警示意义
- 异步 GenRM 架构为大规模 RL 系统引入外部评估器提供了实用参考

## 评分
- 新颖性: ⭐⭐⭐⭐ — 对 flawed positives 的系统分析和无参数惩罚机制有新意
- 实验充分度: ⭐⭐⭐⭐⭐ — 全 checkpoint 评估、多维度消融、人工验证、reward hacking 分析
- 写作质量: ⭐⭐⭐⭐ — 行文流畅，动机-分析-方法-实验环环相扣
- 价值: ⭐⭐⭐⭐ — 对 RLVR 训练质量提升有实际意义，GenRM 方案可直接集成

<!-- RELATED:START -->

## 相关论文

- [Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)
- [STELAR-Vision: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision](../../AAAI2026/reinforcement_learning/stelar-vision_self-topology-aware_efficient_learning_for_aligned_reasoning_in_vi.md)
- [Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning](controllable_exploration_in_hybrid-policy_rlvr_for_multi-modal_reasoning.md)
- [WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning](../../AAAI2026/reinforcement_learning/well_begun_half_done_reinforcement_learning_with_prefix_optimization_for_llm_rea.md)

<!-- RELATED:END -->
