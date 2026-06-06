---
title: >-
  [论文解读] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction
description: >-
  [ACL 2026][LLM安全][提示注入攻击] 本文提出一种基于指令引用的提示注入防御方法，不压制 LLM 的指令遵循能力，而是让模型在响应中引用正在执行的指令，然后通过标签过滤移除与原始指令不相关的响应，在部分场景下将攻击成功率降至接近 0%。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "提示注入攻击"
  - "指令引用"
  - "防御方法"
  - "黑盒防御"
---

# Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction

**会议**: ACL 2026  
**arXiv**: [2504.20472](https://arxiv.org/abs/2504.20472)  
**代码**: [https://github.com/LukeChen-go/robust-via-ref](https://github.com/LukeChen-go/robust-via-ref)  
**领域**: 音频语音  
**关键词**: 提示注入攻击, 指令引用, 防御方法, 黑盒防御, LLM安全

## 一句话总结

本文提出一种基于指令引用的提示注入防御方法，不压制 LLM 的指令遵循能力，而是让模型在响应中引用正在执行的指令，然后通过标签过滤移除与原始指令不相关的响应，在部分场景下将攻击成功率降至接近 0%。

## 研究背景与动机

**领域现状**：LLM 强大的指令遵循能力和无法区分指令与数据内容的特性使其容易受到提示注入攻击。攻击者在数据内容（如网页、用户输入）中注入恶意指令，误导 LLM 执行非预期任务。

**现有痛点**：现有防御方法（无论是提示工程还是微调）大多通过压制 LLM 执行注入指令的倾向来防御，但实验表明压制指令遵循倾向非常困难——模型天然地"想要"执行看到的指令。

**核心矛盾**：防御的核心困难在于 LLM 无法区分"合法指令"和"注入指令"——两者在形式上完全一致，任何基于内容的区分都容易被绕过。

**本文目标**：设计一种不压制而是利用 LLM 指令遵循能力的防御方法。

**切入角度**：分析成功攻击案例发现，LLM 有时会在响应中引用正在执行的指令（如"对于第二个指令..."）。如果 LLM 总是引用其执行的指令，就可以通过引用信息过滤掉对注入指令的响应。

**核心 idea**：让 LLM 输出"答案+指令引用"对，然后过滤掉引用不匹配原始指令的响应——变"压制指令遵循"为"利用指令遵循进行过滤"。

## 方法详解

### 整体框架

三步管道：(1) 标记与分割——将数据内容按行分割，每行加标签（[L 1], [L 2]...），原始指令放在第一行；(2) 提示与响应生成——设计 prompt 引导 LLM 生成带标签引用的结构化响应 $\{(t_i, I_i, r_i)\}$；(3) 过滤——仅保留引用标签为"[L 1]"（即原始指令）的响应。

### 关键设计

1. **标记与分割**:

    - 功能：为数据内容中的每个部分建立可追溯的标识
    - 核心思路：将数据按最多 $K$ 个词分行，每行加前缀标签"[L X]"。原始指令固定放在第一行。指令区域和数据区域用特殊标识符（<Instruction Area>, <Data Area>）分隔
    - 设计动机：标签比指令内容更容易被 LLM 准确复现，且不受 LLM 摘要或改写影响

2. **引导 LLM 生成带引用的响应**:

    - 功能：让 LLM 在执行每个指令前先引用对应的标签
    - 核心思路：通过系统 prompt 指导 LLM 按照"识别标签 → 给出指令 → 生成响应 → 输出 [end]"的格式输出。提供两个 in-context learning 示例确保格式一致性
    - 设计动机：结构化输出使下游过滤可以机械地执行，不依赖语义判断

3. **标签过滤**:

    - 功能：移除对注入指令的响应
    - 核心思路：按标签分割响应为元组 $\{(t_i, I_i, r_i)\}$，仅保留 $t_i = $ "[L 1]" 的元组，其余丢弃
    - 设计动机：原始指令始终在第一行，因此 [L 1] 标签唯一对应合法响应

### 损失函数 / 训练策略

纯提示工程方法，不涉及任何训练。适用于任何 LLM（开源和闭源）。

## 实验关键数据

### 主实验

**直接提示注入攻击成功率 ASR（越低越好）**

| 防御方法 | Llama3-8B Naive | Llama3-8B Combined | Qwen2-7B Combined |
|---------|----------------|-------------------|-------------------|
| None | 48.08 | 79.33 | 84.13 |
| Sandwich | 25.48 | 39.90 | 37.50 |
| Reminder | 33.65 | 53.37 | 87.02 |
| Spotlight | 24.04 | 56.73 | 80.29 |
| StruQ | 5.29 | 2.40 | 30.29 |
| **Ours** | **2.88** | **0.00** | — |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整方法 | ASR ~0% | 标记+引用+过滤 |
| 无 ICL 示例 | ASR 上升 | 格式一致性下降 |
| 无标签（直接引用指令） | ASR 上升 | LLM 改写指令导致匹配失败 |
| 不同分割粒度 K | 影响小 | 鲁棒 |

### 关键发现

- 在多种攻击方法（Naive、Ignore、Escape、Fakecom、Combined）下一致性有效
- 在部分配置下 ASR 降至 0%，与微调方法（如 StruQ）性能可比
- 对模型通用性能的影响极小
- 核心洞察：LLM 执行注入指令时通常能正确引用其来源标签——这一现象可被防御利用
- ICL 示例对格式一致性至关重要，没有示例时部分模型无法稳定输出结构化响应

## 亮点与洞察

- "利用而非压制指令遵循能力"的防御哲学是最核心的创新——将 LLM 的"弱点"（无条件执行指令）转化为防御手段
- 标签系统的设计简洁有效——比让 LLM 复现完整指令文本更可靠
- 作为纯提示工程方法达到与微调方法可比的效果，部署成本极低

## 局限与展望

- 假设攻击者不了解防御系统细节——如果攻击者知道标签系统，可能构造自适应攻击
- 依赖 LLM 稳定遵循结构化输出格式——部分模型（特别是较小模型）可能格式不一致
- 过滤过程可能丢失对原始指令有价值的信息
- 未评估多轮对话场景下的持续防御效果

## 相关工作与启发

- **vs Sandwich/Reminder/Spotlight**: 这些方法试图压制注入指令的执行，本方法利用引用进行过滤
- **vs StruQ 微调方法**: StruQ 需要微调，本方法是纯提示工程且性能可比

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "利用而非压制"的防御哲学和引用过滤机制非常巧妙
- 实验充分度: ⭐⭐⭐⭐ 多种攻击方法、多个模型、消融分析，但自适应攻击评估不足
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法直观
- 价值: ⭐⭐⭐⭐⭐ 提供了低成本、高效果的提示注入防御方案，直接可部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning](know_thy_enemy_securing_llms_against_prompt_injection_via_diverse_data_synthesis.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ACL 2026\] ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks](proxyprompt_securing_system_prompts_against_prompt_extraction_attacks.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)
- [\[ACL 2025\] Defense Against Prompt Injection Attack by Leveraging Attack Techniques](../../ACL2025/llm_safety/defense_prompt_injection.md)

</div>

<!-- RELATED:END -->
