---
title: >-
  [论文解读] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study
description: >-
  [ACL 2026][LLM安全][推理模型安全] 本文系统研究如何通过 SFT 增强大型推理模型（LRM）的安全性，发现直接蒸馏安全响应效果有限的根因是五种风险推理模式（尤其是"弱犹豫"），提出针对性的蒸馏策略将 PAIR 攻击成功率从 63% 降至 13%…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "推理模型安全"
  - "风险推理模式"
  - "弱犹豫"
  - "安全蒸馏"
  - "短推理链"
---

# How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study

**会议**: ACL 2026  
**arXiv**: [2505.15404](https://arxiv.org/abs/2505.15404)  
**代码**: [GitHub](https://github.com/thu-coai/LRM-Safety-Study)  
**领域**: LLM 安全 / 推理模型  
**关键词**: 推理模型安全, 风险推理模式, 弱犹豫, 安全蒸馏, 短推理链

## 一句话总结

本文系统研究如何通过 SFT 增强大型推理模型（LRM）的安全性，发现直接蒸馏安全响应效果有限的根因是五种风险推理模式（尤其是"弱犹豫"），提出针对性的蒸馏策略将 PAIR 攻击成功率从 63% 降至 13%，并发现短推理链和模板推理在安全性上与长推理链表现相当。

## 研究背景与动机

**领域现状**：DeepSeek-R1 等大型推理模型在数学和编程等推理密集任务上取得显著成功，但其增强的推理能力并未转化为安全性能的提升——某些情况下甚至退化。例如，LRM 可能在中间推理步骤或最终输出中生成详细的犯罪计划。

**现有痛点**：(1) 直接从 DeepSeek-R1 蒸馏安全响应的效果远不如预期——经 ReasoningShield 过滤后微调，PAIR 的 ASR 仅从 66% 降至 54%；(2) 现有安全过滤器（如 ReasoningShield）无法有效识别所有风险模式，特别是"弱犹豫"（weak vacillation）——推理正确拒绝了核心有害请求，但对越狱提示中的表面良性元素（如角色扮演要求）犹豫不决；(3) 缺乏对 LRM 安全微调选择的系统性比较研究。

**核心矛盾**："弱犹豫"本身不直接有害，因此通过安全分类器无法检测，但将含有弱犹豫的样本纳入训练会教模型部分遵从有害指令，显著降低安全性。

**本文目标**：系统性地研究通过 SFT 增强 LRM 安全性的最佳实践，包括数据构建、推理链长度和训练配置。

**切入角度**：从分析失败案例入手——为什么蒸馏出的"安全"数据训练后模型仍然不安全？识别五种风险推理模式并逐一消除。

**核心 idea**：(1) 风险推理模式（尤其弱犹豫）是安全蒸馏失效的关键原因——需要在蒸馏阶段主动消除；(2) 长复杂推理链对安全性并非必要——短推理链甚至模板推理即可达到相当的安全性能。

## 方法详解

### 整体框架

研究分三个阶段：(1) **分析阶段**——识别五种风险推理模式并验证弱犹豫是核心问题；(2) **改进蒸馏阶段**——设计针对性提示策略消除风险模式（RealSafe CoT 和 Improved CoT）；(3) **推理简化阶段**——验证短推理链（Short CoT）和模板推理（Template CoT）的安全性等效性。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["有害查询 × 越狱模板<br/>从 DeepSeek-R1 蒸馏安全响应"] --> B
    subgraph S1["五种风险推理模式的识别"]
        direction TB
        B["归类蒸馏数据里的风险推理<br/>缺乏安全意识 / 强犹豫 / 弱犹豫 / 有害补充 / 推理-回答不一致"] --> C["定位弱犹豫为元凶<br/>分类器抓不到, 移除后 ASR 45%→21%"]
    end
    C --> D["改进蒸馏提示策略<br/>RealSafe / Improved CoT 在源头约束 teacher 不犹豫"]
    D --> E["SFT 微调"]
    E --> F["推理简化实验<br/>Short / Template / No CoT 对照"]
    F -->|短链与模板≈完整链, No CoT 失效| G["安全 LRM<br/>PAIR ASR 63%→13%"]
```

### 关键设计

**1. 五种风险推理模式的识别：解释安全蒸馏为何失效**

直接从 DeepSeek-R1 蒸馏"安全响应"去微调，结果远不如预期——经 ReasoningShield 过滤后 PAIR 的 ASR 只从 66% 挪到 54%。本文从失败案例反查，把蒸馏数据里的风险推理归成五类：(1) **缺乏安全意识**——推理压根没识别出查询的有害性质；(2) **强犹豫**——过度思考时对核心有害意图本身犹豫；(3) **弱犹豫**——只对越狱提示里那些表面良性的要素犹豫（如"扮演这个角色可以吗"），核心有害请求其实已经拒了；(4) **有害补充**——推理过程中无意把有害细节给补全了；(5) **推理-回答不一致**——推理决定拒绝、但最终回答仍输出有害内容。

这五类里"弱犹豫"最棘手，因为它本身不直接有害、安全分类器抓不到，可一旦混进训练集就会教模型"可以部分遵从"。证据很硬：经 ReasoningShield 过滤后弱犹豫占比不降反升、直接翻倍（0.33→0.66），因为别的更明显的风险模式被滤掉后它成了主要残余；而只要把弱犹豫样本移除，ASR 就从 45% 掉到 21%。这定位出了安全蒸馏失效的真正元凶。

**2. 改进蒸馏提示策略：在蒸馏阶段就主动消除风险模式，而不是事后过滤**

后验过滤只能去掉已知的风险模式，而且数据利用率惨不忍睹——只有 6.6% 的样本能保留下来。本文转而在蒸馏的源头动手，用提示直接约束 teacher 的推理风格：**RealSafe CoT** 修改蒸馏提示，要求模型推理时"直接识别并拒绝有害查询，不要被角色扮演或情境化框架分散"；**Improved CoT** 在此基础上进一步指示模型"不要对越狱提示的任何部分表现出犹豫"，正好针对性地堵死弱犹豫。这套主动消除比事后过滤既高效又干净，在四个 7B–32B 模型上把 PAIR ASR 平均从 63.0% 压到 13.0%。

**3. 推理简化实验：验证安全场景下长推理链到底是不是必要的**

弱犹豫现象暗示了一个反直觉的可能——长推理链里"多想一步"反而会动摇已经做好的安全决策。本文于是把推理链逐级砍短做对照：**Short CoT** 从 GPT-4o 蒸馏简短推理过程；**Template CoT** 干脆用固定模板（"这是有害请求 → 拒绝"）；**No CoT** 不推理直接拒。结果很清楚：Short CoT 和 Template CoT 的安全性能与完整推理链相当，而 No CoT 完全失效。这说明安全需要的是"某种形式的推理"而非"很长的推理"——长链不但非必要，还可能因为给模型留了犹豫的余地而帮倒忙。

### 损失函数 / 训练策略

标准 SFT 交叉熵损失。训练集：200 个有害查询 × 20 个越狱模板 = 4000 个查询，过滤后采样 1000 个 + 100 个 XSTest 良性查询防止过度拒绝。

## 实验关键数据

### 主实验

**DeepSeek-R1-Distill-Qwen-7B**

| 方法 | MATH500 | AIME24 | LiveCodeBench | None ASR↓ | PAP ASR↓ | PAIR ASR↓ | 过度拒绝↓ |
|------|---------|--------|-------------|----------|---------|----------|---------|
| 原始 | 93.0 | 49.2 | 33.1 | 60.0 | 64.0 | 66.0 | 0.8 |
| Default CoT | 90.4 | 50.0 | 36.1 | 20.0 | 40.0 | 54.0 | 2.7 |
| Improved CoT | 90.2 | 51.7 | 34.9 | 4.0 | 4.0 | **12.0** | 6.7 |
| Short CoT | 89.6 | 53.3 | 36.1 | 4.0 | 2.0 | 16.0 | 12.0 |
| Template CoT | 89.4 | 49.2 | 33.7 | 12.0 | 0.0 | **0.0** | 10.7 |
| No CoT | 90.8 | 52.5 | 31.9 | 62.0 | 64.0 | 64.0 | 10.0 |

### 消融实验

| 消除的风险模式 | PAIR ASR |
|-------------|---------|
| ReasoningShield 过滤（含弱犹豫） | 45.0% |
| 移除所有风险模式**除**弱犹豫 | 45.0%（无改善） |
| 移除**所有**风险模式（含弱犹豫） | **21.0%** |

### 关键发现

- **弱犹豫是安全蒸馏失效的核心原因**——移除其他风险模式无效（ASR 不变），仅移除弱犹豫即降低 24 个百分点
- **No CoT 完全无效**——推理过程对安全性是必要的（某种形式的推理），但不需要很长
- **Template CoT 在 PAIR ASR 上达到 0%**——最简单的推理模式反而最安全，因为模板推理不给模型"犹豫"的机会
- **安全微调不损害推理能力**——MATH500、AIME24、LiveCodeBench 基本无变化
- Short CoT 过度拒绝率较高（12%），需要在安全性和有用性间权衡

## 亮点与洞察

- "弱犹豫"概念的提出具有重要意义——看似无害但传递了"可以部分遵从"的信号，是一种隐蔽的安全隐患
- "推理越长未必越安全"的发现颠覆了直觉——在安全领域，多思考可能带来更多犹豫
- 安全过滤器无法检测弱犹豫的发现暴露了现有安全评估工具的盲点

## 局限与展望

- 主要研究 SFT，未涉及 RLHF/DPO 等对齐方法对 LRM 安全性的影响
- 1000 个训练样本在规模上较小，更大规模下的效果可能不同
- Template CoT 虽然 ASR 最低但过度拒绝率偏高，需要更精细的平衡
- 仅评估了英语场景，多语言 LRM 的安全性可能表现不同

## 相关工作与启发

- **vs Zhang et al. (2025a)**: 提出安全感知响应蒸馏但未系统分析失效原因；本文深入识别了五种风险模式
- **vs Jiang et al. (2025)**: 发现安全调优损害推理性能；本文证明可在不损害推理的前提下显著提升安全性
- **vs Wang et al. (2025)**: 提出低成本缓解策略；本文提供了更全面的消融研究

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "弱犹豫"概念和"短推理更安全"的发现极具洞察力
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个模型 × 6 种方法 × 推理+安全+过度拒绝多维评估 + 详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 从问题分析到解决方案的逻辑链极为清晰
- 价值: ⭐⭐⭐⭐⭐ 为 LRM 安全对齐提供了系统性的实践指南

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](../../ICLR2026/llm_safety/reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)

</div>

<!-- RELATED:END -->
