---
title: >-
  [论文解读] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases
description: >-
  [ACL 2026][LLM评测][推理模型评审] 论文系统比较 reasoning model 与普通 LLM 作为 judge 的表现，发现推理模型在准确率、评测指令遵循和攻击鲁棒性上更强，但仍容易受表层质量偏差影响，并提出 prompt-only 的 PlanJudge 缓解偏差。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "推理模型评审"
  - "评测偏差"
  - "PlanJudge"
  - "RewardBench"
  - "BiasBench"
---

# Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases

**会议**: ACL 2026  
**arXiv**: [2601.03630](https://arxiv.org/abs/2601.03630)  
**代码**: https://github.com/HuihuiChyan/LRM-Judge  
**领域**: LLM安全 / LLM评测 / LLM-as-a-Judge  
**关键词**: 推理模型评审、评测偏差、PlanJudge、RewardBench、BiasBench

## 一句话总结
论文系统比较 reasoning model 与普通 LLM 作为 judge 的表现，发现推理模型在准确率、评测指令遵循和攻击鲁棒性上更强，但仍容易受表层质量偏差影响，并提出 prompt-only 的 PlanJudge 缓解偏差。

## 研究背景与动机
**领域现状**：随着开放式生成任务增多，BLEU、ROUGE 等传统指标难以覆盖 LLM 输出质量，LLM-as-a-Judge 成为主流评测方案之一。研究者常用强模型比较两个回答，或给回答打分，用它替代昂贵的人类评审。

**现有痛点**：judge 模型本身也会出错。已有研究发现 LLM-as-a-Judge 会受到位置、长度、风格、具体程度、格式等偏差影响。与此同时，大型推理模型通过长推理和自我检查在数学、代码等任务中表现更强，但它们是否更适合作为 judge 还没有系统结论。

**核心矛盾**：推理模型可能更擅长复杂判断，也可能因为“想太多”或过度依赖显式标准而更容易被表层特征诱导。判断它们是否更好，不能只看一个 reward benchmark，需要同时看准确率、指令遵循、攻击鲁棒性和偏差鲁棒性。

**本文目标**：在 reasoning-as-the-only-variant 的受控设置下，比较四组 reasoning 与 non-reasoning 模型作为 judge 的差异，并设计一个轻量策略降低偏差。

**切入角度**：作者选择同一模型家族中的 reasoning / instruct 变体，例如 DeepSeek-V3 vs DeepSeek-R1、Qwen2.5-32B-Instruct vs QwQ-32B、Qwen3 instruct vs thinking mode。这样可以最大程度隔离“推理过程”本身的影响。

**核心 idea**：推理模型确实是更强 judge，但要让它们先写清评估计划，再按计划执行判断，才能减少对长度、具体性等表层信号的过度偏好。

## 方法详解
论文由两部分组成。第一部分是系统实证比较，考察 LRM-as-a-Judge 的四个维度：一般评测准确率、评测指令遵循、对 prompt injection 的鲁棒性、对评测偏差的鲁棒性。第二部分提出 PlanJudge：不训练新模型，只在评测前让 judge 生成或接收细粒度评估计划，然后按计划完成比较。

### 整体框架
实验先选定四组模型对，并在 RewardBench、JudgeBench、Helpsteer2-trivial、RobustJudge、BiasBench、LLMBar 上比较。Helpsteer2-trivial 是作者构造的新数据：Response A 整体更好，但 Response B 在某个指定维度更好。一个合格 judge 应该在 overall prompt 下选 A，在 specific prompt 下切换到 B，因此作者用 Reversal Rate 衡量评测指令遵循能力。

随后，作者在同一批 judge 上加入 PlanJudge。PlanJudge 将评测拆成“计划”和“执行”两步：先根据任务写出评估维度、优先级和注意事项，再让模型根据计划进行判断。计划可以来自人工启发式、模型自生成，或二者组合。

### 关键设计

**1. 受控的 reasoning vs non-reasoning 比较：把“推理过程”单独拎出来当唯一变量**

要回答“推理模型是不是更好的 judge”，最大的陷阱是把规模和训练数据的差异误记到 reasoning 头上——随便挑一个强模型和一个弱模型对比，结论里混进了参数量、预训练语料、架构等一堆干扰因素。作者的做法是只在同一模型家族内部取 instruct 和 thinking 两个变体来配对，例如 DeepSeek-V3 vs DeepSeek-R1、Qwen2.5-32B-Instruct vs QwQ-32B、Qwen3 的 instruct 与 thinking 模式。这样两端几乎只差“是否经过长推理”，得到的差距才能干净地归因于推理本身，而不是别的混杂变量。

**2. Helpsteer2-trivial 与 Reversal Rate：把评测场景下的指令遵循变成可度量的反转率**

评测里的指令遵循和普通聊天不一样：一个 judge 可能清楚整体哪个回答更好，却在“这次只评 helpfulness”时仍按整体印象作答，无法忽略其他维度。为了戳中这一点，作者构造了 Helpsteer2-trivial——每个样本里 Response A 整体更好，但 Response B 在某个指定维度上更好。合格的 judge 应该在 overall prompt 下选 A，在 specific prompt 下切换到 B。是否发生这种切换，用 Reversal Rate 来衡量：反转率越高，说明模型真的在按指定维度判断，而不是一味跟着整体质量走。

**3. PlanJudge 两阶段评测：先写评估计划，再按计划执行判断**

推理模型本来就会逐项核对标准，但标准一旦含糊，它反而容易把长度、细节、语气这些表层信号误当成质量——“想得越多”有时会放大偏好而非纠正偏好。PlanJudge 把评测拆成“计划”和“执行”两步：先根据任务写出该看哪些评估维度、优先级和注意事项，再让模型严格按这份计划去比较两个回答。计划有三种来源——heuristic-based（人工启发式规则）、self-synthesized（模型针对当前样本自生成）和 combined（两者结合，既用人工规则又用模型对样本的理解）。显式计划的作用就是在推理动起来之前，先把注意力锚回任务真正要求的标准上。

### 损失函数 / 训练策略
PlanJudge 是 prompt-only 方法，不需要额外微调、奖励模型或外部资源。它的成本来自更长的推理与评估 prompt，但部署门槛低。相比需要训练 judge 的方法，PlanJudge 更像一个评测协议改造：把“直接判断”改成“先规划评估维度，再执行判断”。

## 实验关键数据

### 主实验
一般评测准确率上，reasoning 变体大多强于 instruct 变体，尤其 Qwen 系列在 JudgeBench 上提升明显。DeepSeek-R1 在 RewardBench 高于 V3，但 JudgeBench 低于 V3，作者将其归因于知识类任务上的幻觉问题。

| 模型对 | RewardBench | JudgeBench | 结论 |
|--------|-------------|------------|------|
| DeepSeek-V3 | 89.74 | 84.19 | 普通模型在 JudgeBench 更稳 |
| DeepSeek-R1 | 91.18 | 80.48 | RewardBench 更强，知识判断有例外 |
| Qwen2.5-32B-Instruct | 89.31 | 60.40 | 非推理版本 JudgeBench 明显弱 |
| QwQ-32B | 91.05 | 79.75 | reasoning 带来大幅提升 |
| Qwen3-30B-A3B-Instruct | 89.88 | 74.00 | instruct 基线较强 |
| Qwen3-30B-A3B-Thinking | 92.01 | 83.87 | reasoning 双指标更优 |
| Qwen3-Next-80B-A3B-Instruct | 88.96 | 79.45 | instruct 版本稳定 |
| Qwen3-Next-80B-A3B-Thinking | 92.90 | 82.42 | reasoning 继续提升 |

评测指令遵循上，reasoning 版本的 Reversal Rate 普遍更高，说明长推理并没有削弱评测场景中的指令遵循，反而让模型更会反复核对评价维度。

| 模型 | OriACC | RR | 观察 |
|------|--------|----|------|
| DeepSeek-V3 | 78.22 | 87.80 | overall 判断准确，但维度切换略弱 |
| DeepSeek-R1 | 73.61 | 95.24 | RR 明显更高 |
| Qwen2.5-32B-Instruct | 71.13 | 83.19 | instruct 版本维度切换不足 |
| QwQ-32B | 76.49 | 91.11 | reasoning 提升指令遵循 |
| Qwen3-30B-A3B-Instruct | 72.78 | 95.67 | 本身较强 |
| Qwen3-30B-A3B-Thinking | 78.14 | 97.44 | 双指标最佳之一 |
| Qwen3-Next-80B-A3B-Instruct | 75.88 | 82.50 | RR 较低 |
| Qwen3-Next-80B-A3B-Thinking | 77.94 | 91.18 | reasoning 后明显提升 |

### 消融实验
偏差鲁棒性上，结果更复杂。LRM 在 LLMBar 上通常更强，因为它更能识别明确的 instruction misalignment；但在 BiasBench 上会受长度、具体性等表层质量影响。

| 模型 | BiasBench | LLMBar | 解释 |
|------|-----------|--------|------|
| DeepSeek-V3 | 81.25 | 76.49 | 偏差鲁棒性较均衡 |
| DeepSeek-R1 | 65.00 | 79.00 | 表层质量偏差更强，但能识别明显失配 |
| Qwen2.5-32B-Instruct | 82.50 | 67.71 | BiasBench 高，LLMBar 较弱 |
| QwQ-32B | 67.50 | 79.31 | reasoning 提升 LLMBar，降低 BiasBench |
| Qwen3-30B-Instruct | 81.25 | 59.25 | LLMBar 较弱 |
| Qwen3-30B-Thinking | 77.50 | 83.07 | reasoning 显著提升 LLMBar |
| Qwen3-Next-Instruct | 80.00 | 64.55 | instruct 版本偏差较低 |
| Qwen3-Next-Thinking | 75.00 | 77.55 | reasoning 后 LLMBar 提升 |

PlanJudge 的 combined 策略能显著提升 BiasBench，并通常保留或提升 RewardBench/LLMBar。

| 模型 | 方法 | RewardBench | BiasBench | LLMBar |
|------|------|-------------|-----------|--------|
| DeepSeek-V3 | 原始 | 89.70 | 81.25 | 76.49 |
| DeepSeek-V3 | Combined | 93.07 | 98.75 | 86.83 |
| DeepSeek-R1 | 原始 | 91.10 | 65.00 | 79.00 |
| DeepSeek-R1 | Combined | 92.47 | 97.50 | 86.21 |
| Qwen2.5-32B | 原始 | 89.30 | 82.50 | 67.71 |
| Qwen2.5-32B | Combined | 89.68 | 93.59 | 75.55 |
| QwQ-32B | 原始 | 91.00 | 67.50 | 79.31 |
| QwQ-32B | Combined | 93.13 | 95.00 | 83.07 |

### 关键发现
- LRM-as-a-Judge 整体优于普通 LLM judge，尤其在推理密集的代码、数学和复杂判断任务中更明显。
- 推理模型在评测指令遵循上更强，这与一些普通指令跟随研究中“reasoning model 更固执”的结论不同，说明评测场景有其特殊性。
- 推理模型对 prompt injection 类攻击更稳，因为它会在推理中检查任务边界和评估要求。
- 但推理模型仍会偏爱看起来更具体、更长或更有条理的回答，哪怕这些表层特征并不代表真正质量。
- PlanJudge 的核心价值不是让模型“多想”，而是让模型先明确“应该按什么标准想”。

## 亮点与洞察
- 论文把 judge 能力拆成四个维度，避免了“某个 benchmark 高分就更好”的单指标结论。
- Helpsteer2-trivial 很实用，它把评测指令遵循转成可度量的反转率，能测出 judge 是否真的按指定维度判断。
- PlanJudge 简单但击中要害。很多 judge 偏差来自评估标准含糊，把评估计划显式化能显著减少这种问题。
- 研究提醒我们：推理链并不是天然可靠，它会放大模型已有的评估偏好；因此需要用结构化标准约束推理过程。

## 局限与展望
- 模型覆盖刻意限制在有清晰 reasoning/non-reasoning 对照的家族，因此结论对 LLaMA 系、闭源 o 系列等模型还需进一步验证。
- 每个评估维度只使用一到两个 benchmark，仍可能受数据集设计偏差影响。
- PlanJudge 会增加推理成本和延迟，在大规模自动评测中需要权衡吞吐量。
- 偏差类型主要来自现有 BiasBench/LLMBar，现实评测中的领域偏差、文化偏差、语言偏差还没有充分覆盖。
- 后续可以研究计划质量如何自动验证，以及是否能把 PlanJudge 与校准、人类小样本审计结合。

## 相关工作与启发
- **vs LLM-as-a-Judge 实证研究**: 既有研究多讨论 GPT-4 类 judge 与人类一致性；本文进一步比较 reasoning 模式是否适合做 judge。
- **vs BiasBench / LLMBar**: 这些 benchmark 提供偏差诊断，本文发现 reasoning 在不同偏差上的方向并不一致，不能简单说 reasoning 更稳或更差。
- **vs 训练式 judge 改进方法**: 一些方法通过微调提升 judge 能力；PlanJudge 不训练模型，部署更轻，但也依赖模型本身能执行计划。
- **启发**: 在论文实验中使用 LLM judge 时，可以把评价 rubrics 写成显式 plan，并报告是否使用 planning，以提高评测可复现性。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 系统比较 reasoning judge 的设计很及时，PlanJudge 简单但有效。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖四类 judge 能力和多个模型对，但 benchmark 维度还可以继续扩展。
- 写作质量: ⭐⭐⭐⭐☆ 结论清晰，表格信息密集；作者列表处缓存中有模板残留但不影响主体内容。
- 价值: ⭐⭐⭐⭐⭐ 对 LLM-as-a-Judge 的模型选择、偏差控制和评测协议设计有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MM-JudgeBias: A Benchmark for Evaluating Compositional Biases in MLLM-as-a-Judge](mm-judgebias_a_benchmark_for_evaluating_compositional_biases_in_mllm-as-a-judge.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](../../ICML2026/llm_evaluation/reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[ICML 2026\] REAL：把回归感知奖励塞进 RL，让 LLM-as-a-Judge 学会"差一分也是差"](../../ICML2026/llm_evaluation/real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
