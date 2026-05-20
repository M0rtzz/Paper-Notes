---
title: >-
  [论文解读] Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements
description: >-
  [ACL2025][LLM/NLP][指示词理解] 首次系统评估 LLM 对英语指示词（I/you/here/tomorrow）的理解能力，构建 1600 条 2×2 因素设计的评测集，揭示 LLM 在 you/here/tomorrow 上严重依赖无关上下文信息而非语法规则…
tags:
  - "ACL2025"
  - "LLM/NLP"
  - "指示词理解"
  - "共指消解"
  - "LLM语言能力评估"
  - "语用学"
---

# Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements

**会议**: ACL2025  
**arXiv**: [2506.01089](https://arxiv.org/abs/2506.01089)  
**代码**: [metehanoguzz/LLMs-Indexicals-English](https://github.com/metehanoguzz/LLMs-Indexicals-English)  
**领域**: LLM/NLP  
**关键词**: 指示词理解, 共指消解, LLM语言能力评估, 语用学

## 一句话总结

首次系统评估 LLM 对英语指示词（I/you/here/tomorrow）的理解能力，构建 1600 条 2×2 因素设计的评测集，揭示 LLM 在 you/here/tomorrow 上严重依赖无关上下文信息而非语法规则，且引号对不同指示词的影响方向截然相反。

## 研究背景与动机

**领域现状**：LLM在共指消解（coreference resolution）任务上的评估主要集中在第三人称代词（he/she/they）和名词短语层面。然而，语言学中地位极为独特的指示词（indexicals）——即直接锚定言语行为坐标的I/you/here/tomorrow等词——几乎未被系统评估过。

**现有痛点**：第三人称代词天然具有歧义性（如"John hit Bill and he ran away"中he可指John也可指Bill），需要依赖上下文消歧。但指示词的语义由语法规则严格确定：I指向说话人，you指向听话人，here指向说话地点，tomorrow指向说话后一天。这意味着一个真正"理解"语言的模型应该能**忽略**可能误导的上下文信息。同一研究组此前评估土耳其语指示词ben（"I"）时发现LLM表现极差（Oğuz et al., 2024），但英语中是否如此尚不清楚。

**核心矛盾**：指示词的消解要求模型遵循句法规则而非统计关联——这恰好是对LLM（本质上是统计模型）的"反直觉"挑战。特别是直接引语中指示词的引号转移（如Andrew said "I am smart"中I指Andrew而非实际说话人）增加了推理层级。

**本文目标** 系统评估前沿LLM对英语四类指示词（I/you/here/tomorrow）的理解能力，区分模型是"真正理解语法规则"还是"碰巧答对但依赖上下文猜测"。

**切入角度**：设计2×2因素控制实验（句子类型×上下文启动），构建1600条精确控制的评测集，消除表面统计相关性，迫使模型展示是否真正基于语法规则进行推理。

**核心 idea**：通过正交因素控制的指示词评测集，揭示LLM在I上接近人类水平但在you/here/tomorrow上严重依赖无关上下文信息而非语法规则。

## 方法详解

### 整体框架

本文是一项基准评测研究（benchmark study），而非提出新方法。核心流程为：(1) 构建English Indexical Dataset，包含1600条按2×2因素设计的多选题；(2) 用4个前沿LLM作为被试，在受限选项协议下进行评估；(3) 按指示词粒度（I/you/here/tomorrow）分别分析模型行为，特别关注模型是依据语法规则还是上下文启动做出选择。

### 关键设计

1. **2×2因素控制的评测集构建**:

    - 功能：构建可精确区分"语法规则理解"与"上下文猜测"的评测数据集
    - 核心思路：覆盖4种指示词（I/you/here/tomorrow），每种400样本。每条样本按两个正交因素展开：**句子类型**（引号句quoted vs 非引号句non-quoted）和**上下文启动**（shifted prime倾向引述解读 vs non-shifted prime倾向字面解读），形成4个条件。正确答案由语法规则唯一确定——非引号句应选non-shifted选项，引号句应选shifted选项——上下文启动作为干扰因素，应被忽略。数据由GPT-4o生成场景，人工审核25%（400条）确保语法正确性，且男女名字各50%消除性别偏差
    - 设计动机：简单的准确率无法区分"模型真懂语法"和"模型碰巧答对"。通过正交设计，可以观察模型的选择是否随上下文启动方向变化——如果变化显著，说明模型在依赖上下文而非语法规则

2. **受限选项评估协议**:

    - 功能：确保模型输出可精确解析，消除开放式回答的噪音
    - 核心思路：选取GPT-4o、Claude 3.5 Sonnet、Gemini 1.5 Pro、DeepSeek-V3四个前沿LLM。通过prompt严格限制模型只能在两个预定义选项（shifted vs non-shifted解读）中选择，避免模型生成冗长解释或拒绝回答。在4个条件（2句型×2启动方向）下分别统计各指示词的准确率
    - 设计动机：开放式回答的解析可能引入实验者偏差，二选一迫选范式（forced-choice paradigm）是心理语言学标准实验范式的NLP迁移

3. **指示词粒度差异化分析框架**:

    - 功能：揭示不同指示词的独特行为模式，而非笼统地给出"好/差"结论
    - 核心思路：对I/you/here/tomorrow逐一分析，核心对比三个维度：(a) 非引号条件下模型是否正确选择non-shifted；(b) 引号条件下模型是否正确识别引号转移选择shifted；(c) 上下文启动方向改变时准确率的变化幅度（delta越大说明模型越依赖上下文）。结合引号正/负效应分析，区分引号对不同指示词的差异化影响
    - 设计动机：不同指示词在语言学中有根本差异——I的身份信息最强（直接等于说话人），而here/tomorrow的锚定更依赖语境推理。分粒度分析可揭示LLM的语用推理到底在哪个层面失败

## 实验关键数据

### 主实验

| 指示词 | 条件 | GPT-4o | Claude 3.5 | Gemini 1.5 | DeepSeek-V3 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| I | 非引号-shifted prime | ~99 | ~99 | ~99 | ~99 |
| I | 非引号-non-shifted prime | ~99 | ~99 | ~99 | ~99 |
| I | 引号-shifted prime | >94 | 89 | >94 | 78 |
| I | 引号-non-shifted prime | >94 | 89 | >94 | 17 |
| you | 非引号-shifted prime | ~70 | ~70 | 92 | ~70 |
| you | 非引号-non-shifted prime | ~80 | ~80 | 92 | ~80 |
| you | 引号（两种 prime 均值） | 显著下降 | 显著下降 | 显著下降 | 显著下降 |
| here | 非引号-shifted prime | >96 | >96 | >96 | >96 |
| here | 非引号-non-shifted prime | <2 | <2 | <2 | <2 |
| here | 引号（均值） | 37 | 64 | 94 | >97 |
| tomorrow | 非引号 | ~94 | 100 | 100 | 83 |
| tomorrow | 引号 | 极低 | ~0 | ~0 | 极低 |

### 消融实验

- **I**：所有模型在非引号条件准确率均值 99%，接近人类水平。引号条件下 GPT-4o 和 Gemini 保持 >94%，但 DeepSeek-V3 在 non-shifted prime 下骤降至 17%，说明引号使其更易受上下文干扰。
- **you**：所有模型均严重受上下文启动影响——当上下文倾向错误选项时准确率大幅下降。引号条件一致地降低准确率，表明模型未能利用引号转移规则。Gemini 非引号下表现最佳（92%）但引号下同样大幅回落。
- **here**：非引号条件下模型选择**完全由上下文启动决定**（shifted prime >96%，non-shifted prime <2%），语法规则几乎无作用。引号条件反而帮助模型"解脱"上下文依赖：DeepSeek >97%、Gemini 94%。
- **tomorrow**：所有模型存在强烈的 non-shifted 偏向，非引号下虚高（94-100%），引号下准确率接近 0%。Claude 和 Gemini 在引号条件下 100% 选择 non-shifted（即 100% 错误），说明模型完全无法进行 tomorrow 的引号转移。

### 关键发现

| 指示词 | 引号对准确率的影响 | 核心原因 |
|:---:|:---:|:---|
| I | 轻微下降 | 大多数模型已掌握 I 的引号转移，仅 DeepSeek 受上下文干扰 |
| you | 负面（降低） | 引号增加了任务复杂度，模型未能利用转移规则 |
| here | 正面（提升） | 引号帮助模型摆脱对上下文的过度依赖 |
| tomorrow | 强负面 | 模型对 tomorrow 有根深蒂固的 non-shifted 偏向，引号无法纠正 |

## 亮点与洞察

- **首个系统评估LLM英语指示词理解的研究**：填补语言学-NLP交叉空白，2×2因素设计精巧区分了"遵循语法规则"vs"依赖上下文猜测"
- **揭示差异化行为模式**：不同指示词上的表现差异极大——I接近人类水平而tomorrow完全失败，引号对here有正面影响但对tomorrow有强负面影响，说明LLM的语用推理能力高度不均匀
- **跨语言对比视角**：英语I近人类水平vs土耳其语极差，提示训练数据的语言覆盖度对指示词理解有重要影响

## 局限与展望

- 黑盒评估，未分析模型内部表示或注意力机制来解释为什么不同指示词表现差异如此大
- 仅评估4个闭源模型，缺少开源模型对比和改进方案的探索
- 使用GPT-4o生成测试数据同时又作为被评估模型之一，存在潜在的数据泄露/偏差风险
- 未覆盖now/this/that等其他类型的指示词

## 相关工作与启发

- **vs Oğuz et al. (2024)**：他们评估土耳其语指示词ben（I）发现LLM表现极差。本文扩展到英语四类指示词，发现英语I上LLM表现好得多，暗示训练数据分布的关键影响
- **vs WinoBias/WinoGrande**：这些基准评估第三人称代词的消歧能力，但不涉及指示词。本文填补了这一重要空白，且评估目标相反——好的模型应忽略上下文而非依赖上下文

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个英语指示词LLM理解评估，选题独特且有语言学理论深度
- 实验充分度: ⭐⭐⭐ 4模型×4指示词×4条件覆盖合理，但缺少开源模型和改进实验
- 写作质量: ⭐⭐⭐⭐ 语言学背景清晰，2×2设计逻辑严密，结果讨论到位
- 价值: ⭐⭐⭐⭐ 揭示LLM语用推理的深层缺陷：依赖统计关联而非语法规则

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)
- [\[ACL 2025\] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability](compositional_generalization_instruction.md)
- [\[ACL 2025\] Is It JUST Semantics? A Case Study of Discourse Particle Understanding in LLMs](is_it_just_semantics_a_case_study_of_discourse_particle_understanding_in_llms.md)
- [\[ACL 2025\] Improving Contextual Faithfulness of Large Language Models via Retrieval Heads-Induced Optimization](improving_contextual_faithfulness_of_large_language_models_via_retrieval_heads-i.md)

</div>

<!-- RELATED:END -->
