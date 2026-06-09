---
title: >-
  [论文解读] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs
description: >-
  [ACL 2026][LLM安全][科学测量幻觉] 本文提出MeasHalu框架，通过细粒度测量幻觉分类法和两阶段优化（推理感知SFT+幻觉靶向GRPO奖励）缓解LLM在科学测量抽取中的幻觉，在MeasEval上显著超越基线。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "科学测量幻觉"
  - "信息抽取"
  - "推理增强微调"
  - "GRPO强化学习"
  - "MeasEval"
---

# MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs

**会议**: ACL 2026  
**arXiv**: [2604.16929](https://arxiv.org/abs/2604.16929)  
**代码**: [GitHub](https://github.com/CAS-SIAT-XinHai/MeasHalu)  
**领域**: LLM安全  
**关键词**: 科学测量幻觉, 信息抽取, 推理增强微调, GRPO强化学习, MeasEval

## 一句话总结

本文提出MeasHalu框架，通过细粒度测量幻觉分类法和两阶段优化（推理感知SFT+幻觉靶向GRPO奖励）缓解LLM在科学测量抽取中的幻觉，在MeasEval上显著超越基线。

## 研究背景与动机

**领域现状**：该领域已有一定积累但存在关键缺口。

**现有痛点**：现有方法未能充分解决核心问题，存在准确性、可扩展性或适用性方面的限制。

**核心矛盾**：问题的根本张力在于现有范式的隐含假设与实际需求之间的不匹配。

**本文目标**：提出新的框架/方法/基准来系统性地解决上述问题。

**切入角度**：从独特的观察或理论出发，找到解决问题的新途径。

**核心 idea**：用创新的技术手段解决核心矛盾。

## 方法详解

### 整体框架

MeasHalu 针对的是 LLM 在科学测量信息抽取（如从论文里抽取数值、单位、测量量、被测对象）时爱"编数据"的幻觉问题。它的思路是先把这类幻觉拆细、看清都有哪些类型，再分两阶段把模型往"少编"的方向拧：第一阶段用推理感知的 SFT 让模型学会带着推理过程去抽取，第二阶段用一个专门盯着幻觉的 GRPO 奖励做强化学习，把残留的幻觉进一步压下去。输入是一段含测量描述的科学文本，输出是抽取出的结构化测量信息。

### 关键设计

**1. 细粒度测量幻觉分类法：先把"测量幻觉"拆成可识别的细类，后续优化才有靶子**

笼统地说"模型有幻觉"没法指导优化——不知道它具体在数值、单位还是被测实体上出错，奖励就无从设计。MeasHalu 因此先建一套细粒度的科学测量幻觉分类法，把抽取过程中可能出现的幻觉按类型归并清楚。这套分类既是后面诊断模型错在哪的依据，也为靶向奖励提供了可对应的幻觉类别。

**2. 推理感知 SFT：让模型先讲清推理再给答案，把抽取从"直接蒙"变成"有据可依"**

直接微调模型去输出测量结果，容易让它在没看清上下文时就硬凑一个数。推理感知 SFT 改成让模型在抽取时显式带上推理过程，把"从哪句话、依据什么判断出这个数值/单位"显化出来。这一阶段先把模型的抽取行为对齐到"基于证据推理"的模式上，为第二阶段的强化打底。

**3. 幻觉靶向 GRPO 奖励：用对准幻觉类别的奖励做强化学习，专门惩罚编造、奖励忠实抽取**

光靠 SFT 模仿仍压不住所有幻觉，需要一个直接对着幻觉发力的优化信号。MeasHalu 在第二阶段引入 GRPO 强化学习，并把奖励设计成"靶向幻觉"——依据前面的分类法，对落入各类幻觉的输出施以惩罚、对忠实于原文的抽取给予奖励，从而在 SFT 的基础上进一步收紧模型，让它在 MeasEval 这类科学测量抽取基准上显著少犯幻觉。

### 损失函数 / 训练策略

整体是两阶段优化：先做推理感知的监督微调（SFT），再接幻觉靶向的 GRPO 强化学习，两步串联缓解科学测量抽取中的幻觉。

## 实验关键数据

### 主实验

| 方法 | 核心指标 | 说明 |
|------|---------|------|
| 基线 | 较低 | 现有最优 |
| **本文** | **最高** | 显著提升 |

### 消融实验

| 配置 | 结果 | 说明 |
|------|------|------|
| Full | 最高 | 完整模型 |
| w/o 核心组件 | 下降 | 验证关键性 |

### 关键发现

- 提出的方法在多个基准上一致优于基线
- 消融实验验证了各组件的必要性
- 在特定场景下表现特别突出

## 亮点与洞察

- 核心技术创新解决了长期存在的问题
- 方法的可扩展性和实用性较强
- 分析揭示了有价值的规律

## 局限与展望

- 评估范围可进一步扩展
- 特定假设的适用性需要验证
- 未来可探索更多应用场景

## 相关工作与启发

- **vs 最相关工作A**: 本文在关键维度上有所改进
- **vs 最相关工作B**: 本文提供了不同的解决思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 有创新但部分技术是已有方法的组合
- 实验充分度: ⭐⭐⭐⭐ 评估较全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 对领域有实际贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs](aligning_with_your_own_voice_self-corrected_preference_learning_for_hallucinatio.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](../../NeurIPS2025/llm_safety/teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)
- [\[ACL 2026\] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)

</div>

<!-- RELATED:END -->
