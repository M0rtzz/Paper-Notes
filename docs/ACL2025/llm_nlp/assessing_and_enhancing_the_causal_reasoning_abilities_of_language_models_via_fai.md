---
title: >-
  [论文解读] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation
description: >-
  [ACL 2025][LLM 其他][因果推理] 本文提出了一个基于忠实文本解释（Faithful Textual Interpretation, FTI）的框架，通过将因果推理任务中的变量关系忠实地转化为自然语言描述，评估并增强LLM的因果推理能力，在多个因果推理基准上取得了显著提升。 领域现状：因果推理是人工智能的核心能…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "因果推理"
  - "语言模型"
  - "忠实文本解释"
  - "因果图"
  - "反事实推理"
---

# Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation

**会议**: ACL 2025  
**arXiv**: 无公开预印本  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 因果推理, 语言模型, 忠实文本解释, 因果图, 反事实推理

## 一句话总结
本文提出了一个基于忠实文本解释（Faithful Textual Interpretation, FTI）的框架，通过将因果推理任务中的变量关系忠实地转化为自然语言描述，评估并增强LLM的因果推理能力，在多个因果推理基准上取得了显著提升。

## 研究背景与动机

**领域现状**：因果推理是人工智能的核心能力之一，要求模型理解变量间的因果方向、干预效果和反事实情境。现有的LLM因果推理评估主要依赖选择题或简单的因果判断任务，如CLADDER、CausalBench等基准数据集。评估方式通常直接给出因果图的形式化描述（如DAG的边列表）或数值化的条件概率。

**现有痛点**：当前LLM在因果推理基准上的表现不稳定，存在两个关键问题：一是评估的忠实性问题——很多基准测试中LLM可能通过文本模式匹配而非真正的因果推理来获得正确答案；二是表达形式的障碍——将因果图以"X→Y"这种符号化形式输入LLM时，模型难以充分利用其在预训练中获得的因果相关知识。

**核心矛盾**：LLM的预训练语料中包含大量因果关系的自然语言描述（如"吸烟导致肺癌"），但标准的因果推理任务却以抽象符号和数学形式呈现，导致模型能力和测试形式之间存在显著的gap。

**本文目标**：设计一种忠实的文本化解释方法，将抽象的因果结构和推理步骤转化为自然语言，既保证因果关系的忠实表达，又能激活LLM的因果常识知识。

**切入角度**：作者观察到，当因果关系用自然语言"故事化"描述时，人类的因果推理能力显著提升。受此启发，将因果图中的变量和边用语义丰富的自然语言描述替代符号化表示。

**核心 idea**：用忠实的自然语言解释替代抽象的因果图符号表示，使LLM能够调动预训练知识进行因果推理，同时设计忠实性约束确保文本解释不扭曲原始因果关系。

## 方法详解

### 整体框架
FTI框架包含三个核心阶段：（1）因果结构文本化（Causal Structure Textualization），将DAG图结构转化为语义清晰的自然语言描述；（2）推理路径引导（Reasoning Path Guidance），通过逐步推理提示引导LLM沿着因果路径进行推理；（3）忠实性验证（Faithfulness Verification），确保生成的文本解释与原始因果关系保持一致。输入为因果图和查询问题，输出为因果推理答案和推理过程的自然语言解释。

### 关键设计

1. **因果结构文本化模块（CST）**:

    - 功能：将抽象的因果图转化为语义丰富的自然语言描述
    - 核心思路：对因果图中的每条边 $X \rightarrow Y$，生成形如"当X发生变化时，会直接导致Y发生相应的变化"的自然语言描述。关键在于保持因果方向性——区分"X导致Y"和"X与Y相关"。对于复杂的因果链（如混杂因子、中介变量），使用分层描述策略：先描述直接因果关系，再描述完整因果路径。同时为每个变量生成语义化的上下文描述，使其从抽象符号变为具象概念。
    - 设计动机：LLM在预训练中见过大量自然语言表达的因果关系，文本化可以激活这些潜在知识；而纯符号化输入则无法建立这种连接

2. **分步因果推理引导（SCRG）**:

    - 功能：引导LLM按照正确的因果推理步骤逐步求解
    - 核心思路：将因果推理拆解为多个步骤——（a）识别相关变量和因果路径；（b）判断是否存在混杂效应并确定调整变量集；（c）应用do-calculus规则或反事实推理框架；（d）整合信息得出最终结论。每一步都以自然语言形式呈现，配合Chain-of-Thought提示。对于干预问题，显式要求LLM区分观测条件（$P(Y|X)$）和干预条件（$P(Y|do(X))$）。
    - 设计动机：因果推理的难点在于多步逻辑推理，直接让LLM一步到位容易出错；分步策略降低了每步的认知负荷

3. **忠实性双向验证机制（FBV）**:

    - 功能：确保文本化后的描述忠实地反映原始因果结构
    - 核心思路：设计了两个验证维度——结构忠实性（从生成的文本描述中重建因果图，检查是否与原始图同构）和推理忠实性（对同一问题分别用符号化和文本化输入进行推理，检查推理路径的一致性）。对于不满足忠实性的文本描述，自动触发修正循环，调整措辞直到满足约束。引入忠实性分数 $F_{score} = \alpha \cdot F_{struct} + (1-\alpha) \cdot F_{reason}$ 来量化评估。
    - 设计动机：无约束的文本化可能引入额外信息或歧义，导致LLM利用文本线索而非因果推理获得答案，这会使评估失去意义

### 损失函数 / 训练策略
本文主要基于prompt工程，不涉及模型参数训练。忠实性验证环节采用迭代优化策略，通过最多3轮的修正循环来提升文本描述的忠实性。在少样本设置中，选择包含不同因果结构类型（链式、叉式、对撞机）的示例确保覆盖性。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | GPT-4+FTI | GPT-4 Direct | Claude-3+FTI | Llama-3+FTI | 提升 |
|------------|------|-----------|--------------|-------------|-------------|------|
| CLADDER (因果判断) | Accuracy | 82.6 | 68.4 | 79.3 | 71.5 | +14.2 |
| CausalBench (干预) | Accuracy | 76.8 | 61.2 | 73.1 | 65.8 | +15.6 |
| CORR2CAUSE | Accuracy | 71.3 | 58.7 | 68.5 | 62.1 | +12.6 |
| 反事实推理 | Accuracy | 74.5 | 60.1 | 71.2 | 64.3 | +14.4 |

### 消融实验

| 配置 | CLADDER Acc | CausalBench Acc | 说明 |
|------|-----------|----------------|------|
| Full FTI | 82.6 | 76.8 | 完整框架 |
| w/o CST | 72.1 | 66.3 | 去掉文本化，-10.5 |
| w/o SCRG | 76.3 | 70.2 | 去掉分步引导，-6.3 |
| w/o FBV | 80.1 | 74.5 | 去掉忠实性验证，-2.5 |
| Symbol only | 68.4 | 61.2 | 纯符号输入基线 |

### 关键发现
- 因果结构文本化（CST）贡献了最大的性能提升（约10个点），验证了文本化激活因果知识的核心假设
- 在涉及混杂因子的复杂因果结构上，FTI的提升最为显著（+18.3%），说明文本化描述帮助LLM更好地理解了交叉的因果路径
- 忠实性验证机制虽然提升幅度较小（2-3个点），但有效防止了约15%的虚假正确答案（通过文本线索而非因果推理获得的答案）
- 较小的模型（如Llama-3-8B）在FTI框架下的提升比例更大（+15.2%），说明FTI对弱因果推理能力的模型帮助更大

## 亮点与洞察
- 忠实文本解释的理念非常巧妙——它找到了一个甜蜜点，既利用了自然语言的表达力来激活预训练知识，又通过忠实性约束防止了作弊。这种"在约束下最大化表达力"的思路可以迁移到任何涉及形式化输入的LLM评估场景
- 双向忠实性验证的设计很有洞察力：结构忠实性防止信息丢失，推理忠实性防止信息添加，两者互补形成完整的保真保障
- 实验中发现的"模型越弱受益越大"现象暗示了一个实用价值：FTI可以让小模型在因果推理场景中达到接近大模型的水平

## 局限与展望
- 文本化过程本身需要一个强LLM来生成高质量的自然语言描述，存在bootstrapping问题
- 对于高度抽象的因果关系（如经济学中的宏观因果链），文本化可能引入人类偏见
- 忠实性验证的自动化程度有限，在因果图规模较大时计算成本上升
- 未来可以探索将FTI与因果发现任务结合，利用LLM的文本化能力辅助从数据中发现因果关系

## 相关工作与启发
- **vs CLADDER（Jin et al., 2024）**: CLADDER提出了因果推理的标准化评估框架，但使用符号化输入；本文在CLADDER框架上证明了文本化的优越性
- **vs CausalCoT（Jin et al., 2023）**: CausalCoT也使用了链式推理提示，但未设计忠实性约束，存在答案正确但推理错误的问题
- **vs Causal Parrots（Zečević et al., 2023）**: 该工作质疑LLM是否具有真正的因果推理能力；本文通过FTI框架证明LLM确实能在适当引导下展现非平凡的因果推理能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 忠实文本解释的角度新颖，忠实性约束设计精巧
- 实验充分度: ⭐⭐⭐⭐ 多数据集多模型评估，消融实验全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机论述充分
- 价值: ⭐⭐⭐⭐ 为LLM因果推理评估提供了新范式，有较强的方法论启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)
- [\[ACL 2025\] SkillVerse: Assessing and Enhancing LLMs with Tree Evaluation](skillverse_tree_eval.md)
- [\[ACL 2025\] Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes](mathneuro_math_reasoning_isolation.md)
- [\[ACL 2025\] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)
- [\[ACL 2025\] Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)

</div>

<!-- RELATED:END -->
