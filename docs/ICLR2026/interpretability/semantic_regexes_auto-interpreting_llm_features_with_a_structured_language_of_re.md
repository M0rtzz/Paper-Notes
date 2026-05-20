---
title: >-
  [论文解读] Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language
description: >-
  [ICLR 2026][可解释性][mechanistic_interpretability] 本文提出 **Semantic Regexes（语义正则表达式）**，一种用于自动描述 LLM 特征的结构化语言…
tags:
  - "ICLR 2026"
  - "可解释性"
  - "mechanistic_interpretability"
  - "automated_interpretability"
  - "sparse_autoencoders"
  - "structured_language"
  - "feature_description"
---

# Semantic Regexes: Auto-Interpreting LLM Features with a Structured Language

**会议**: ICLR 2026  
**arXiv**: [2510.06378](https://arxiv.org/abs/2510.06378)  
**代码**: [apple/ml-semantic-regex](https://github.com/apple/ml-semantic-regex)  
**领域**: LLM NLP / Mechanistic Interpretability  
**关键词**: mechanistic_interpretability, automated_interpretability, sparse_autoencoders, structured_language, feature_description  

## 一句话总结

本文提出 **Semantic Regexes（语义正则表达式）**，一种用于自动描述 LLM 特征的结构化语言，通过原语（symbol/lexeme/field）+ 修饰符（context/composition/quantification）组合，实现与自然语言同等准确但更简洁、一致且可分析的特征描述。

## 研究背景与动机

**自动可解释性的现状**：
- 稀疏自编码器（SAE）等方法可以从 LLM 中提取单义特征（features）
- 自动可解释性用 LLM 将特征翻译为人类可读的描述
- 这些描述帮助研究者理解模型编码了什么概念、追踪特征电路

**自然语言描述的问题**：
- **冗长**：描述经常过于啰嗦（如"The presence of the sequence 54 indicating a year, time, or numeric reference frequently associated with events"）
- **不一致**：功能相同的特征可能得到完全不同的描述
- **歧义**：自然语言天然存在多义性，不利于需要组合推理的分析任务
- **需要人工重标注**：即使在最新的特征电路工作中，研究者仍需手动重标注特征

**结构化语言的优势**：
- 良定义的语法和语义，减少歧义
- 复合规则支持从简单到复杂的精确表达
- 一致的表达方式便于比较和聚合

## 方法详解

### 整体框架

Semantic Regex = 基于 grounded theory 方法从数千个真实 LLM 特征中归纳出的结构化语言，嵌入标准自动可解释性流水线（explainer + evaluator）中使用。

### 关键设计：语言规范

**三种原语（Primitives）**：

1. **Symbol** `[:symbol X:]` — 精确匹配字符串 X
    - 例：`[:symbol color:]` 匹配文本中的 "color"
    - 描述激活于特定 token 的特征

2. **Lexeme** `[:lexeme X:]` — 匹配 X 的句法变体（时态、复数等）
    - 例：`[:lexeme color:]` 匹配 "color", "colors", "coloring" 等
    - 描述捕捉词义的特征

3. **Field** `[:field X:]` — 匹配 X 的语义变体（同一概念域的词）
    - 例：`[:field color:]` 匹配 "red", "blue", "green" 等
    - 描述激活于概念类别的特征

**三种修饰符（Modifiers）**：

1. **Context** `@{:context X:}(regex)` — 在上下文 X 中匹配
    - 例：`@{:context politics:}([:symbol color:])` 仅在政治语境中匹配 "color"

2. **Composition** — 序列拼接和交替（|）
    - 例：`[:field color:]([:symbol and:]|[:symbol or:])[:field color:]`

3. **Quantification** — 使用正则量词 `?`（零或一次）
    - 例：`[:symbol a:][:field color:]?[:field flower:]`

### 自动可解释性流水线

- **主体模型**：GPT-2-Small、Gemma-2-2B
- **特征来源**：SAE 提取的潜在特征（GPT-2-RES-25k, Gemma-2-2B-RES-16k/65k）
- **解释器模型**：GPT-4o-mini（给定激活数据，生成自然语言或 semantic regex 描述）
- **评估器模型**：GPT-4o-mini（评估描述与特征行为的匹配度）

Semantic regex 仅改变了描述语言，不改变流水线架构。通过在 max-acts 的 prompt 中注入 semantic regex 的语法规则和 few-shot 示例实现。

### 评估指标

- **生成指标（Clarity）**：特征描述能否生成高激活的样例（类似 precision）
- **判别指标（Detection/Fuzzing/Responsiveness/Purity）**：描述能否匹配已知的激活样例（类似 recall）
- **忠实性指标（Faithfulness）**：描述是否反映因果干预的效果

### 损失函数

本文不涉及模型训练，是一个评估框架。特征提取使用预训练的 SAE。

## 实验关键数据

### 主实验：准确率对比（每层 100 特征）

| 指标 | Semantic Regex | max-acts (NL) | token-act-pair (NL) |
|------|---------------|---------------|---------------------|
| Clarity (GPT-2) | **显著优于** | 基线 | 最低 |
| Detection (GPT-2) | **显著优于** tap | 与 SR 持平 | 最低 |
| Fuzzing (GPT-2) | **显著优于** tap | 与 SR 持平 | 最低 |
| Clarity (Gemma-16k) | 非劣效 | 基线 | 最低 |
| Clarity (Gemma-65k) | **显著优于** tap | 基线 | 最低 |

核心结论：**Semantic regex 在所有模型上至少与自然语言持平，在多个指标上显著优于 token-act-pair**。这说明结构化约束不会降低描述能力。

### 消融实验：简洁性和一致性

| 度量 | Semantic Regex | max-acts | token-act-pair |
|------|---------------|----------|----------------|
| 描述长度中位数（字符） | **41** (IQR: 19-59) | 139 (IQR: 119-166) | 55 (IQR: 46-66) |
| 相同描述率（5次生成） | **33.6%** | 0.0% | 12.2% |

- Semantic regex 比 max-acts 短 **3.4 倍**
- 同一特征的不同采样下，semantic regex 33.6% 产生完全相同的描述（max-acts 为 0%）

### 特征复杂度分析

| 层位置 | 平均组件数 | 低级原语占比 | Field 占比 | 带修饰符占比 |
|--------|----------|------------|-----------|------------|
| 早期层 | 少 | 高（symbol/lexeme 为主） | 低 | 低 |
| 中间层 | 中等 | 降低 | 增加 | 增加 |
| 后期层 | 多 | 最低 | 最高 | 最高 |

Semantic regex 的结构自然编码了特征复杂度：后层特征需要更长、更抽象的描述。这与"后层编码更复杂表征"的已知现象一致，但首次能从特征描述中直接读取。

### 用户研究（24 人）

| 度量 | Semantic Regex 优于 NL 的特征数 / 12 |
|------|--------------------------------------|
| 决策边界理解（正向-反事实激活差） | **9 / 12** |

- 参与者使用 semantic regex 在 12 个特征中的 9 个上生成了更好的正向和反事实样例
- 自然语言描述常引入无关细节导致误解（如"expected to indicates anticipation"让用户误认为非预期语境是反例）
- 参与者对 semantic regex 的理解比预期好得多，收到的自然语言理解问题反而更多

### 关键发现

1. **结构化 ≠ 表达力降低**：semantic regex 在准确率上与自然语言持平甚至更优
2. **简洁性 3.4 倍提升**：大幅降低解读负担
3. **一致性从 0% 提升到 33.6%**：有助于冗余特征检测和电路分析
4. **复杂度可读**：从描述格式直接反映特征的抽象层级
5. **人类友好**：用户研究确认 semantic regex 帮助建立更准确的特征心智模型

## 亮点与洞察

- **方法论创新**：将正则表达式的思路推广到语义域，兼具形式化和可读性
- **Grounded Theory 驱动**：语言设计不是凭空构造，而是从数千个真实特征归纳而来
- **即插即用**：仅需修改 prompt 中的描述规范即可集成到现有可解释性流水线
- **规模化分析的关键能力**：自然语言描述适合理解单个特征，但 semantic regex 的结构使得跨模型、跨层的宏观分析成为可能
- **Apple 出品**，开源代码和交互界面，工程质量高

## 局限性

1. **非唯一映射**：同一特征可能有多个等效的 semantic regex 描述，缺乏标准化风格指南
2. **过于简洁的风险**：`[:field musician:]` 可能让人误以为 "guitarist" 会强激活，但实际特征只激活于音乐家名字
3. **不解决多义性**：高度多义的特征产生的 semantic regex 不连贯
4. **模型学习新语言的局限**：LLM 仅从简短描述和少量示例学习 semantic regex 语法，偶尔出错
5. 目前仅在 GPT-2 和 Gemma-2 上验证，更大模型和更复杂 SAE 的适用性待验证

## 相关工作与启发

- **SAE / Transcoders (Bricken et al., 2023; Dunefsky et al., 2024)**：提取 LLM 的单义特征
- **Bills et al. (2023)**：自动可解释性流水线开创者（token-act-pair 方法）
- **Paulo et al. (2024)**：max-acts 方法，改进了激活数据展示方式
- **Ameisen et al. (2025)**：特征电路追踪，需要人工重标注——semantic regex 的一致性可减少此需求
- **Jin et al. (2025)**：特征复杂度分层——semantic regex 从描述端独立验证了这一发现
- 启示：可解释性不仅是生成描述，描述的**语言本身**也值得设计

## 评分

- **创新性**: ⭐⭐⭐⭐ — 结构化语言 × 自动可解释性是新组合
- **实验设计**: ⭐⭐⭐⭐ — 多模型 + 多指标 + 用户研究，全面
- **实用性**: ⭐⭐⭐⭐⭐ — 即插即用，Apple 开源，直接可用
- **写作质量**: ⭐⭐⭐⭐⭐ — 论述清晰，可视化出色
- **综合评分**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Style over Story: Measuring LLM Narrative Preferences via Structured Selection](../../ACL2026/interpretability/style_over_story_measuring_llm_narrative_preferences_via_structured_selection.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](../../ACL2026/interpretability/llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](../../ICML2026/interpretability/corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)

</div>

<!-- RELATED:END -->
