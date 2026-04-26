---
title: >-
  [论文解读] Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility
description: >-
  [ACL 2025][话语语义理解] 本文提出语义理解能力的层级框架（词汇/句子/话语），构建了基于照应可及性（anaphora accessibility）的评估数据集，发现 LLM 在某些结构上与人类一致但在其他结构上存在系统性分歧——LLM 依赖词汇线索而非结构化抽象。
tags:
  - ACL 2025
  - 话语语义理解
  - 照应可及性
  - 动态语义学
  - LLM评估
  - 形式语义学
---

# Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility

**会议**: ACL 2025  
**arXiv**: [2502.14119](https://arxiv.org/abs/2502.14119)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 话语语义理解, 照应可及性, 动态语义学, LLM评估, 形式语义学

## 一句话总结

本文提出语义理解能力的层级框架（词汇/句子/话语），构建了基于照应可及性（anaphora accessibility）的评估数据集，发现 LLM 在某些结构上与人类一致但在其他结构上存在系统性分歧——LLM 依赖词汇线索而非结构化抽象。

## 研究背景与动机

LLM 的成功依赖于自然语言理解能力，但现有评估任务很少考察 LLM 是否能准确表示和更新话语状态。成功解读话语需要利用代词来指代文本中已引入的实体——这就是照应（anaphora）的问题。

照应的合适性受到前件语义辖域（scope）的影响。以经典例子说明：

- "**A** farmer worked in his field. **He** dreamed of the harvest." ✓（存在量词引入的实体可被后续句子指代）
- "**Every** farmer worked in his field. **He** dreamed of the harvest." ✗（全称量词引入的实体在辖域外不可及）

这种现象在动态语义学（Dynamic Semantics）中有严格的形式化——话语意义不仅是静态的真值条件，而是对话语状态的更新操作。不同量词和逻辑连接词决定了话语实体（discourse referent）的可及范围。

现有工作的不足：
- Schuster and Linzen (2022) 只考虑了否定与话语实体的交互
- Kim and Schuster (2023) 使用过于简单的语言（如"Box 1 contains the book"）
- 缺乏对全称量词、条件句、析取等多种作用域与照应交互的系统评估

## 方法详解

### 整体框架

作者提出三层语义理解能力层级：

1. **词汇层（Lexical Level）**：理解单个词义——同义、反义、蕴含等
2. **句子层（Sentence Level）**：整合词汇意义，形成句子真值条件表示
3. **话语层（Discourse Level）**：整合连续句子意义，更新话语表示

本文聚焦话语层，利用照应可及性作为诊断工具，评估 LLM 是否理解不同语义算子如何影响话语状态更新。

### 关键设计

**实验涵盖三类语义构造**：

**1. 全称量词（Universal Quantifiers）**

- 简单对比：A farmer vs. Every farmer + 跨句照应
- 条件句（Donkey Conditionals）：
    - "John owns a donkey, and he beats **it**. **It** is a big one." ✓（存在量词）
    - "If John owns a donkey, he beats **it**. **It** is a big one." ✗（条件句隐含全称量化）
    - "Whenever John owns a donkey, he beats **it**. **It** is a big one." ✗（同上）

**2. 否定（Negation）**

- 存在量词：The farmer owned a cow. → It was away on the meadow. ✓
- 否定：The farmer didn't own a cow. → It was away on the meadow. ✗
- 双重否定：It was not the case that the farmer didn't own a cow. → It was away on the meadow. ✓
- 双重否定消解：两个否定相互抵消，语义等价于存在量词

**3. 析取（Disjunction）**

- Evans (1977) 的发现：存在量词在析取第一项中不授权第二项的照应，但否定量词可以
- "Either there was **a** manuscript, or **it** was hidden..." ✗
- "Either there was **no** manuscript, or **it** was hidden..." ✓
- either 的有无不影响语义（or vs. Either...or 等价）
- 否定量词在连词中不具有同样效果

**评估指标**：

- **差异的差异（Difference-of-Difference）度量**：比较单句内照应（in-scope）和跨句照应（cross-sentence）在存在量词和全称量词下的概率差异，控制了句子复杂度等混淆因素
- **条件概率指标**：比较同一后续句在不同上下文下的总 surprisal
- **SLOR（Syntactic Log-Odds Ratio）**：对析取实验使用，控制句子长度和词频

**模型与人类实验**：
- 4 个开源 LLM：Llama3-2-1B/3B、Llama3-1-8B、Llama3-1-8B-Instruct
- 2 个闭源 LLM：GPT babbage-002、davinci-002
- 人类实验：104 名参与者通过 Prolific 招募，66 个强制选择试次

**语料构建**：
- 从结构模板生成，人工构建 32 个语义合理的句子框架
- 语言学专家人工检查确保可接受性/不可接受性
- 共 9816 个实验句子

### 损失函数 / 训练策略

本文不涉及模型训练，而是将 LLM 作为心理语言学被试（psycholinguistic subjects），通过其对 token 的 surprisal（负对数概率）来度量话语理解能力。

## 实验关键数据

### 主实验

**实验 1：全称量词**

- 简单对比（Exi > Every）：Llama 家族约 75% 准确率，GPT 家族略低，人类接近上限
- 条件句（Exi > If, Exi > Whenever）：所有 LLM 接近上限（>90%），但人类准确率反而较低
- **有趣分歧**：he-后续句中人类偏好 if/whenever 条件句（逆转预期方向），可能因"telescoping"效应——人类倾向将 he 解读为在条件句辖域内

**实验 2：否定**

- Exi > Neg：所有模型成功区分，准确率高
- DN > Neg（双重否定 > 单否定）：3 个模型失败，Llama3-1-8B 系列甚至偏好否定优于双重否定（预期方向的反转）
- 添加 "in fact" 后：DN > Neg 准确率提升，但 Exi > Neg 准确率反转
- **关键发现**：LLM 对否定辖域的理解不系统，严重依赖 "in fact" 等词汇线索

**实验 3：析取**

- EitherOr > Conjunction 和 EitherOr > EitherPosOr：所有模型达到上限，与人类一致
- or > Conjunction：模型准确率接近随机，人类表现出预期偏好
- EitherPosOr vs. or：模型偏好 EitherPosOr（预期方向的反转），人类无明显偏好
- **关键发现**：虽然 EitherOr 和 or 语义等价，但模型表现严重依赖是否存在 "either" 这个词

### 消融实验

**"in fact" 词汇影响实验**是核心消融：
- 添加 "in fact" 使双重否定偏好增加 → 说明 LLM 依赖词汇共现模式而非语义理解
- 同时使存在量词偏好下降 → 表明 "in fact" 通常与否定/反转语境共现，导致 LLM 错误推断
- 人类在两种条件下表现稳定 → 人类理解基于结构抽象而非词汇线索

### 关键发现

1. **LLM 和人类在某些任务上一致**：全称量词的基本辖域限制被所有 LLM 正确学习
2. **LLM 不理解双重否定消解**：无法将双重否定正确等同于存在量词
3. **LLM 的话语理解依赖词汇线索而非结构**：either 的有无、in fact 的添加都会影响判断，但不应影响
4. **人类表现出 LLM 没有的结构敏感性**：特别是 telescoping 效应和对否定辖域的稳定理解
5. **话语层理解是 LLM 的系统性弱点**：即使在句子层表现良好，话语层仍存在根本缺陷

## 亮点与洞察

- **理论驱动的评估设计**：严格基于动态语义学理论构建测试项，而非简单的经验性测试
- **三层语义理解框架**：为评估 LLM 语义能力提供了系统化的思考框架
- **人机对比的深刻洞察**：不仅展示了 LLM 在哪里失败，还解释了**为什么**失败——词汇依赖 vs 结构抽象
- **连接形式语义学与 NLP**：将 Heim (1983)、Groenendijk and Stokhof (1991) 等经典理论引入 LLM 评估

## 局限与展望

- **模型范围有限**：未能测试 GPT-4o 等最新模型（API 不支持 log probability 访问）
- **仅考虑英语**：照应可及性的规则在不同语言中可能不同
- **模板生成的刺激材料**：虽经专家检查，但可能不够自然
- **仅使用 surprisal 指标**：无法直接探测模型内部表示
- **未包含所有相关语义构造**：如条件句的其他变体、量词作用域交互等
- **人类实验样本量（104人）对某些效应可能不够**

## 相关工作与启发

- 继承了 Schuster and Linzen (2022) 和 Zhu and Frank (2024) 的话语实体识别范式，大幅扩展了测试构造的覆盖面
- 与 Li et al. (2021) 的实体追踪工作互补：他们使用显式状态描述，本文使用隐式的语义作用域
- 对 LLM 训练的启示：**当前预训练目标可能不足以学习话语层语义**，需要专门的训练信号
- 对基准设计的启示：应将话语层评估纳入主流 NLU 评估体系

## 评分

- **创新性**：★★★★★（理论驱动的系统评估，填补重要空白）
- **实验充分性**：★★★★☆（三组实验 + 人类对照，但模型范围受限）
- **实用价值**：★★★☆☆（主要贡献在理论洞察层面，工程应用有限）
- **写作质量**：★★★★★（论证严密，理论与实验结合优秀）

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] ChuLo: Chunk-Level Key Information Representation for Long Document Understanding](chulo_chunk-level_key_information_representation_for_long_document_understanding.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ACL 2025\] Narrative Media Framing in Political Discourse](narrative_media_framing_in_political_discourse.md)
- [\[ACL 2025\] Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions](sleepless_nights_sugary_days_creating_synthetic_users_with_health_conditions_for.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)

<!-- RELATED:END -->
