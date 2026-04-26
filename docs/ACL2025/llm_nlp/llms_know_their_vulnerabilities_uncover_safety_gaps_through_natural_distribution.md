---
title: >-
  [论文解读] LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts
description: >-
  [ACL 2025][LLM/NLP][jailbreak] 提出 ActorBreaker 多轮攻击方法，基于 Latour 的行动者网络理论，利用与有害内容语义相关的良性 prompt（自然分布偏移）绕过安全机制，在 HarmBench 上达到 SOTA 攻击成功率，揭示了预训练数据与安全训练数据之间的语义覆盖差距。
tags:
  - ACL 2025
  - LLM/NLP
  - jailbreak
  - actor-network theory
  - multi-turn attack
  - safety alignment
  - natural distribution shift
---

# LLMs Know Their Vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts

**会议**: ACL 2025  
**arXiv**: [2410.10700](https://arxiv.org/abs/2410.10700)  
**代码**: https://github.com/AI45Lab/ActorAttack  
**领域**: LLM/NLP  
**关键词**: jailbreak, actor-network theory, multi-turn attack, safety alignment, natural distribution shift

## 一句话总结
提出 ActorBreaker 多轮攻击方法，基于 Latour 的行动者网络理论，利用与有害内容语义相关的良性 prompt（自然分布偏移）绕过安全机制，在 HarmBench 上达到 SOTA 攻击成功率，揭示了预训练数据与安全训练数据之间的语义覆盖差距。

## 研究背景与动机

**领域现状**：LLM 安全训练教模型拒绝有害查询，但对抗攻击（如 GCG、jailbreak）可以绕过。大多数攻击使用恶意分布偏移（如加密、低资源语言转换）。

**现有痛点**：现有多轮攻击依赖固定策略（角色扮演、假设场景），多样性不足；且使用"不自然"的 prompt 容易被检测。

**核心矛盾**：预训练数据中包含了大量与有害内容语义相关但看似无害的知识（如"Ted Kaczynski"与炸弹制造的关联），但安全训练无法覆盖这些间接关联。

**本文目标** 利用预训练分布内的自然语义关联构造良性但有效的多轮攻击。

**切入角度**：Latour 的行动者网络理论——将有害目标分解为 6 类行动者（创造/传播/接收/监管等），每类包含人类和非人类实体。

**核心 idea**：利用 LLM 自身的知识构建有害内容的语义关联网络，用良性的多轮对话逐步引导模型暴露不安全内容。

## 方法详解

### 整体框架
两阶段：(1) 网络构建——给定有害查询，用 LLM 构建行动者网络（6类行动者 × 人类/非人类实体），每个节点是潜在攻击线索；(2) 攻击生成——选择行动者及其与有害目标的语义关系作为线索，生成多轮良性 prompt 逐步引导。

### 关键设计

1. **行动者网络构建**

    - 6 类行动者：创造者/传播者/接收者/监管者/影响者/关联者
    - 每类分人类（历史人物等）和非人类（书籍/媒体/社会运动）
    - 用 LLM 自身的知识实例化网络
    - 设计动机：Latour 理论确保全面覆盖可能的攻击路径

2. **攻击链生成**

    - 选择一个行动者及其与目标的语义关系
    - 设计多轮看似无害的问题逐步接近目标
    - 设计动机：每轮单独看都是良性的，但组合起来引导模型

3. **自然分布偏移 vs 恶意分布偏移**

    - 本文的 prompt 在预训练分布内（自然良性）
    - 先前方法的 prompt 在分布外（加密/角色扮演）
    - 设计动机：自然 prompt 不触发安全检测器

## 实验关键数据

### 主实验 — HarmBench 攻击成功率

| 方法 | 类型 | GPT-4o | Claude-3 | Llama-3-70B | 平均 |
|------|------|--------|---------|------------|------|
| GCG | 单轮 | ~20% | ~15% | ~35% | ~23% |
| PAIR | 单轮 | ~30% | ~25% | ~40% | ~32% |
| Crescendo | 多轮 | ~40% | ~35% | ~50% | ~42% |
| **ActorBreaker** | **多轮** | **~55%** | **~45%** | **~65%** | **~55%** |

### 安全检测绕过

| 方法 | Llama-Guard-2 检测率 |
|------|-------------------|
| GCG prompt | ~80%(容易检测) |
| PAIR prompt | ~60% |
| Crescendo prompt | ~40% |
| **ActorBreaker prompt** | **~5%**（几乎不被检测） |

### 防御效果

| 配置 | 攻击成功率 |
|------|----------|
| 无防御 | 55% |
| 在 ActorBreaker 安全数据上微调 | **25%**（-30%） |
| 但通用能力下降 | ~3%（轻微trade-off） |

### 关键发现
- **ActorBreaker 在所有对齐 LLM 上达到最高攻击成功率**，包括 GPT-o1
- **Llama-Guard-2 几乎检测不到这些 prompt**：因为每轮都是良性的
- **攻击多样性远超现有方法**：6类行动者提供丰富的攻击路径
- **防御有效但有 trade-off**：在 ActorBreaker 数据上微调减少 30% 攻击成功率，但轻微影响通用能力
- **预训练-安全训练的语义差距**是根本问题

## 亮点与洞察
- **行动者网络理论在 AI 安全中的应用**非常新颖——将社会学理论转化为系统的红队测试方法
- **自然分布偏移 vs 恶意分布偏移**的概念区分揭示了安全对齐的根本局限
- **用 LLM 自身知识攻击 LLM 自身**的递归特性值得深思

## 局限与展望
- 攻击方法可能被滥用
- 行动者网络构建依赖 LLM 自身，如果 LLM 安全训练更好可能无法构建
- 改进方向：更全面的安全训练数据覆盖、语义距离感知的安全机制

## 相关工作与启发
- **vs GCG/PAIR**：它们是恶意分布偏移，ActorBreaker 是自然分布偏移
- **vs Crescendo**：Crescendo 依赖固定模板，ActorBreaker 从网络理论生成多样化路径

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 行动者网络理论+自然分布偏移的创新结合
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型+多攻击对比+防御实验
- 写作质量: ⭐⭐⭐⭐⭐ 理论基础扎实
- 价值: ⭐⭐⭐⭐⭐ 对 AI 安全研究有重大启示

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement](why_not_act_on_what_you_know_unleashing_safety_potential_of_llms_via_self-aware_.md)
- [\[ACL 2025\] Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions](can_llms_ground_when_they_dont_know_a_study_on_direct_and_loaded_political_quest.md)
- [\[ACL 2025\] Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge](enhancing_transformation_from_natural_language_to_signal_temporal_logic_using_ll.md)
- [\[ACL 2025\] What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)

<!-- RELATED:END -->
