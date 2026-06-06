---
title: >-
  [论文解读] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning
description: >-
  [ICLR 2026][优化/理论][GRPO] 提出 Scaf-GRPO 框架，通过分层级的 in-prompt hint 注入（知识→规划→解题步骤）来克服 GRPO 训练中"学习悬崖"(zero-reward)问题…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "GRPO"
  - "强化学习"
  - "学习悬崖"
  - "渐进式引导"
  - "脚手架教学"
---

# Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning

**会议**: ICLR 2026  
**arXiv**: [2510.19807](https://arxiv.org/abs/2510.19807)  
**代码**: 无  
**领域**: 优化 / LLM推理增强  
**关键词**: GRPO, 强化学习, 学习悬崖, 渐进式引导, 脚手架教学

## 一句话总结

提出 Scaf-GRPO 框架，通过分层级的 in-prompt hint 注入（知识→规划→解题步骤）来克服 GRPO 训练中"学习悬崖"(zero-reward)问题，在 Qwen2.5-Math-7B 上将 AIME24 的 pass@1 相对提升 44.3%，同时保持 on-policy 训练一致性。

## 研究背景与动机

**领域现状**：基于可验证奖励的强化学习（RLVR）已成为提升 LLM 推理能力的主流范式，GRPO 等算法通过组内相对奖励计算优势信号来更新策略。

**现有痛点**：当模型面对远超当前能力的难题时，所有探索性尝试都失败，产生持续的零奖励信号。在 GRPO 中，同组全零奖励导致 advantage $\hat{A}_i = \frac{R(o_i) - \mu_\mathcal{G}}{\sigma_\mathcal{G}} = 0$，梯度消失，形成"学习悬崖"(learning cliff)。

**核心矛盾**：现有解决方案（如 LUFFY）采用 prefix-continuation 策略——给模型提供正确解的前缀——但这造成 teacher 和 student 策略的分布不匹配，且强迫模型沿预定路径走，抑制了探索。

**本文目标** 在不引入 off-policy 分布不匹配的前提下，帮助模型克服学习悬崖，从无法解决的难题中学到推理能力。

**切入角度**：受教育学"脚手架理论"(Scaffolding) 启发，提供最小化的、渐进式的 in-prompt 提示，而非强制的解题路径前缀。

**核心 idea**：不给"铁轨"(prefix)而给"路标"(hint)——在 prompt 中注入分层提示使模型用自己的策略生成正确解，避免 off-policy 问题并保留探索自由度。

## 方法详解

### 整体框架

训练分两个阶段：Phase 1 (引导豁免期，前15%步数) 让模型自主探索区分"伪难题"和"真难题"；Phase 2 对真难题激活分层 hint 引导探索。当某 batch 内所有 rollout 零奖励时，Scaf-GRPO 按 Knowledge→Planning→Solution 层级注入提示直到模型生成正确解，用该成功轨迹替换一条失败轨迹重新计算 advantage，然后用标准 GRPO 损失更新。

### 关键设计

1. **引导豁免期与真难题诊断**:

    - 做什么：训练初期（前15%步数）不提供任何 hint，让模型完全自主探索
    - 核心思路：监控 zero-reward 问题的解决速率，速率停滞后将剩余未解决问题标记为"真难题"。训练初期的快速下降对应"伪难题"（格式不熟悉、初级推理技巧）
    - 设计动机：避免过早提供 hint 导致模型依赖提示，确保 hint 仅用于真正的能力缺口。消融实验表明去掉豁免期导致 9.2% 性能下降

2. **分层级 Hint 引导探索 (K→P→S)**:

    - 做什么：为真难题注入三级渐进式 in-prompt 提示，从抽象到具体
    - 核心思路：$H_{\text{knowledge}}$（关键概念/公式） → $H_{\text{planning}}$（高层策略框架）→ $H_{\text{solution}}$（具体计算步骤）。每级内递增提供，一旦模型成功即停止，记录最小有效提示级别
    - 设计动机：最小化干预保留模型自主性——奖励使用最抽象 hint 就能解题的情况，鼓励内化推理技能而非记忆解法。去掉任一层级都降低性能（去 Solution 层降 5.7%）

3. **On-Policy Batch 增强与统一损失**:

    - 做什么：将成功的 hint-guided 轨迹替换一条失败轨迹，恢复 advantage 信号
    - 核心思路：$\mathcal{G}_{\text{final}} = (\mathcal{G} \setminus \{o_j\}) \cup \{o_h^*\}$，其中 $o_h^* \sim \pi_\theta(\cdot | q \oplus h^*)$。概率比 $r_{i,t}'(\theta) = \frac{\pi_\theta(o_{i,t}'|o_{i,<t}', q \oplus h^*)}{\pi_{\theta_{\text{old}}}(o_{i,t}'|o_{i,<t}', q \oplus h^*)}$ 是标准 on-policy 比率
    - 设计动机：区别于 prefix-based 方法使用 $\frac{\pi_\theta(\cdot|q)}{\pi_{\theta_{\text{old}}}(\cdot|q \oplus h^*)}$ 这种 off-policy 比率，本方法对 hint-augmented prompt 的两个策略使用相同条件，保证 on-policy 一致性

### 损失函数 / 训练策略

损失函数与标准 GRPO 完全相同（clipped surrogate objective），区别仅在数据层面：$J_{\text{Scaf-GRPO}}(\theta) = \hat{\mathbb{E}}_{i,t}[\min(r_{i,t}'(\theta)\hat{A}_i', \text{clip}(r_{i,t}'(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i')]$。KL 散度惩罚设为 0 以最大化探索。训练 10 epochs，最大响应长度 2048 tokens。

## 实验关键数据

### 主实验

| 模型 / 基准 | 指标 | Scaf-GRPO | Vanilla GRPO | LUFFY | 相对提升 |
|------------|------|-----------|-------------|-------|---------|
| Qwen2.5-Math-7B / AIME24 | pass@1 | 43.3 | 30.0 | 33.3 | +44.3% vs GRPO |
| Qwen2.5-Math-7B / AIME25 | pass@1 | 20.0 | 13.3 | 16.7 | +50.4% vs GRPO |
| Qwen2.5-Math-7B / AMC | pass@1 | 70.0 | 60.0 | 62.5 | +16.7% vs GRPO |
| Qwen2.5-Math-7B / 7基准平均 | pass@1 | 50.9 | 45.2 | 46.6 | +12.6% vs GRPO |
| Qwen2.5-Math-1.5B / 平均 | pass@1 | 41.5 | 37.6 | — | +10.4% |
| DeepSeek-R1-Distill-1.5B / 平均 | pass@1 | 53.6 | 50.6 | — | +5.9% |

### 消融实验

| 配置 | 7基准平均 | 说明 |
|------|----------|------|
| Full K→P→S | 50.9 | 完整三层级 |
| w/o Progressive (Solution-Only) | 48.4 | 直接给最具体 hint |
| w/o Knowledge Hint | 49.2 | 去掉概念层 |
| w/o Solution Hint | 48.0 | 去掉具体步骤层，降幅最大 |
| w/o Incremental Chunking | 47.7 | 一次性给完整 hint |
| No Guidance (Vanilla GRPO) | 45.2 | 无引导基线 |

### 关键发现

- 渐进式引导比直接给 Solution hint 好 2.5 分——抽象 hint 强迫模型自主推理，培养更泛化的技能
- 去掉任何一个 hint 层级都导致性能下降，三层互补而非冗余
- 增量式提供（逐步展示 hint 内容）比一次性提供好 3.2 分——最小干预原则有效
- 模型能展现从"模仿 hint"到"自主解题"的演化过程

## 亮点与洞察

- "路标 vs 铁轨"的比喻非常精准：in-prompt hint 允许模型自由选择推理路径，而 prefix-continuation 强制走预定路线
- 分层引导豁免期设计巧妙——先让模型自己挣扎一段时间才介入，类似好老师的做法
- 保持了 GRPO 损失函数的完整性，仅在数据层面干预，工程上简洁优雅

## 局限与展望

- 三级 hint 需要外部强模型（DeepSeek-R1）预先生成，增加了数据准备成本
- 目前仅在数学推理任务上验证，代码/逻辑推理等领域的迁移性未知
- hint 质量影响显著（用 DeepSeek-R1 vs Qwen-72B 差 4%），对 hint 生成的依赖是潜在瓶颈
- 引导豁免期百分比（15%）虽在 10%-40% 范围内稳定，但最优值可能因模型而异

## 相关工作与启发

- **vs LUFFY**: LUFFY 用 prefix-continuation 造成分布不匹配需要 policy shaping 修正，Scaf-GRPO 用 in-prompt hint 保持 on-policy，在 7B 模型上平均高 4.3 分
- **vs Vanilla GRPO**: GRPO 在零奖励时梯度为零导致学习停滞，Scaf-GRPO 通过 batch 增强恢复信号
- **vs DAPO/DeepScaleR**: 这些方法改进 GRPO 算法本身，Scaf-GRPO 改进数据/引导策略，两者正交可组合

## 评分

- 新颖性: ⭐⭐⭐⭐ 脚手架教学法在 RL 中的应用新颖，in-prompt hint 区别于 prefix-continuation 是关键创新
- 实验充分度: ⭐⭐⭐⭐ 多模型(Qwen/Llama/DeepSeek)多规模(1.5B~7B)，消融设计精细
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，Figure 2 的训练动态可视化尤其直观
- 价值: ⭐⭐⭐⭐ 为 RLVR 的学习悬崖问题提供了实用且有理论支撑的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)
- [\[ICML 2026\] Enhancing LLM Training via Spectral Clipping](../../ICML2026/optimization/enhancing_llm_training_via_spectral_clipping.md)
- [\[NeurIPS 2025\] MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](../../NeurIPS2025/optimization/mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)
- [\[ICLR 2026\] CogFlow: Bridging Perception and Reasoning through Knowledge Internalization for Visual Mathematical Problem Solving](cogflow_bridging_perception_and_reasoning_through_knowledge_internalization_for_.md)

</div>

<!-- RELATED:END -->
