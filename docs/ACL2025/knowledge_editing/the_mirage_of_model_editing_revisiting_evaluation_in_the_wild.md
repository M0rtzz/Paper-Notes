---
title: >-
  [论文解读] The Mirage of Model Editing: Revisiting Evaluation in the Wild
description: >-
  [ACL 2025 Main][知识编辑][模型编辑] 本文揭示了模型编辑领域评估实践中的系统性缺陷——先前方法报告的近完美成功率（~96.8%）在真实应用场景下骤降至 38.5%，根本原因是测试中使用 teacher forcing 泄露了真值信息…
tags:
  - "ACL 2025 Main"
  - "知识编辑"
  - "模型编辑"
  - "知识编辑评估"
  - "teacher forcing泄露"
  - "序列编辑"
  - "QAEdit基准"
---

# The Mirage of Model Editing: Revisiting Evaluation in the Wild

**会议**: ACL 2025 Main  
**arXiv**: [2502.11177](https://arxiv.org/abs/2502.11177)  
**代码**: [GitHub](https://github.com/wanliyoung/revisit-editing-evaluation)  
**领域**: 知识编辑  
**关键词**: 模型编辑、知识编辑评估、teacher forcing泄露、序列编辑、QAEdit基准

## 一句话总结

本文揭示了模型编辑领域评估实践中的系统性缺陷——先前方法报告的近完美成功率（~96.8%）在真实应用场景下骤降至 38.5%，根本原因是测试中使用 teacher forcing 泄露了真值信息，并提出 QAEdit 基准和 WILD 评估框架来推动更可靠的评估。

## 研究背景与动机

**领域现状**：模型编辑（Model Editing）旨在精确修改 LLM 中的特定知识，无需完全重训。代表性方法包括 ROME（定位-编辑）、MEMIT（批量编辑）、FT-L（微调特定层）等，在标准 benchmark（如 CounterFact、ZsRE）上报告了接近 96-100% 的编辑成功率。

**现有痛点**：尽管在人工构建的评估集上表现优异，模型编辑在真实应用中的有效性从未被系统验证。当研究者将编辑后的模型部署到实际 QA 场景中，效果远不如预期。两者之间的巨大鸿沟暗示评估方法本身存在根本问题。

**核心矛盾**：先前评估的核心问题在于使用了 **teacher forcing** 进行测试——在评估模型生成质量时，将真实答案的前缀 token 作为输入喂给模型，让模型只需预测最后几个 token。这相当于考试时提前看到了大部分答案，导致成功率被严重高估。在真实部署中，模型必须从零自回归生成完整答案，错误会在 token 间传播累积。

**本文目标**：(1) 系统揭示现有评估的具体缺陷；(2) 构建与真实 QA 任务对齐的新基准 QAEdit；(3) 设计任务无关的评估框架 WILD；(4) 评估现有方法在真实场景下的实际表现。

**切入角度**：作者从"teacher forcing 在测试阶段泄露了答案的内容和长度"这一关键观察出发，指出先前工作中测试时 teacher forcing 的使用实质上等于作弊——不仅告诉模型答案大概是什么（内容泄露），还暗示了答案的长度（长度泄露）。

**核心 idea**：用自回归生成（无 teacher forcing）替代传统评估方式，结合真实 QA 数据集构建更可靠的编辑评估体系。

## 方法详解

### 整体框架

WILD（评估框架）+ QAEdit（评估基准）的双重贡献。QAEdit 从广泛使用的 QA 数据集中导出编辑目标和评估样例，WILD 框架定义了标准化的评估流程：不使用 teacher forcing、评估自回归生成的完整答案、考虑编辑对模型通用能力的副作用。

### 关键设计

1. **QAEdit 基准构建**:

    - 功能：提供与真实 QA 场景对齐的模型编辑评估数据
    - 核心思路：从主流 QA 数据集（如 SQuAD、Natural Questions 等）中抽取事实性问答对，将其转化为模型编辑任务——即修改模型中的特定知识，使其对相关问题给出更新后的答案。与 CounterFact 等合成数据不同，QAEdit 的问题和答案来自真实用户查询，评估更贴近实际使用场景
    - 设计动机：CounterFact 等现有 benchmark 使用的是人工构造的知识三元组和模板化问题，与真实用户如何向 LLM 提问存在巨大差距。QAEdit 直接反映了"用户会怎么问"的真实分布

2. **WILD 评估框架——去除 Teacher Forcing**:

    - 功能：定义标准化的、反映真实使用场景的编辑评估流程
    - 核心思路：评估时模型必须从零开始自回归生成完整答案，不提供任何真值前缀。评估三个维度：(a) **编辑成功率**——模型是否输出了期望的新答案；(b) **泛化性**——对编辑知识的不同问法是否也能正确回答；(c) **局部性**——未编辑的知识是否保持不变。关键改变是用精确匹配 / F1 分数替代了基于 token 概率的指标
    - 设计动机：Teacher forcing 在测试中泄露了真值的内容和长度，导致模型几乎不需要"理解"编辑就能得到高分。去除它后才能看到方法的真实水平

3. **序列编辑实验设计**:

    - 功能：模拟真实部署中需要连续多次编辑模型的场景
    - 核心思路：按顺序对模型执行 1000 次编辑，每次编辑后评估所有已编辑知识的保持率和模型通用能力。这模拟了知识持续更新的真实需求——新闻事件、人事变动等需要频繁更新 LLM 的知识
    - 设计动机：先前工作主要评估单次编辑，但实际部署中必然需要大量连续编辑。序列编辑暴露了方法在规模化使用时的根本脆弱性

### 损失函数 / 训练策略

本文是评估型工作，不涉及新的训练方法。

## 实验关键数据

### 主实验

单次编辑场景下，不同评估方式的性能差异（Llama-2-7B）：

| 编辑方法 | 传统评估 (Teacher Forcing) | WILD评估 (自回归) | 性能差距 |
|---------|------------------------|------------------|---------|
| ROME | 96.1% | 37.8% | -58.3% |
| MEMIT | 96.8% | 38.5% | -58.3% |
| FT-L | 89.5% | 35.2% | -54.3% |
| MEND | 72.3% | 28.6% | -43.7% |
| 直接微调 | 85.0% | 33.1% | -51.9% |

### 消融实验

Teacher forcing 中不同泄露因素的贡献分析：

| 评估条件 | 编辑成功率 | 说明 |
|---------|----------|------|
| 完整 teacher forcing | 96.8% | 泄露内容 + 长度 |
| 仅泄露长度 | ~75% | 告知答案长度但不给内容前缀 |
| 无任何泄露（WILD） | 38.5% | 完全自回归生成 |
| 序列编辑 100 次 | ~30% | 连续编辑后退化 |
| 序列编辑 1000 次 | <10% | 模型几乎完全崩溃 |

### 关键发现

- **Teacher forcing 是性能虚高的核心原因**：去除 teacher forcing 后所有方法的成功率下降 40-60 个百分点，说明先前报告的结果大部分来自评估泄露而非真正有效的知识编辑
- **序列编辑导致灾难性退化**：仅 1000 次编辑就让所有方法的成功率降至 10% 以下，同时模型的通用能力也严重退化。这意味着当前的模型编辑方法在实际的知识持续更新场景中基本不可用
- **编辑的泛化性极差**：即使编辑了"美国总统是 X"，用稍微不同的问法（如"谁领导美国"）提问时，模型往往还是给出旧答案。说明编辑只修改了表面的输入-输出映射，未真正更新内部知识表示

## 亮点与洞察

- **揭示了一个领域级的评估漏洞**：Teacher forcing 在理论上是训练技巧，被挪用到测试中构成了系统性偏差。这个发现不仅适用于模型编辑，也是对所有使用 teacher forcing 评估生成质量的研究的警示
- **简洁有力的实验设计**：仅通过改变评估方式（去除 teacher forcing）就揭示了数十个百分点的性能落差，说明好的评估方法本身就是重要的科学贡献
- **序列编辑实验的实际价值**：首次系统展示了编辑方法在规模化场景下的彻底失败，这对模型编辑的商业化应用（如实时知识更新）具有重要的预警意义

## 局限与展望

- **未提出改进方法**：论文侧重于暴露问题，但未给出如何改进编辑方法使其在 WILD 评估下也能有效工作的具体方案
- **仅在 Llama-2-7B 上实验**：未验证更大或更新的模型（如 Llama-3、Qwen2）是否表现不同
- **QAEdit 的编辑目标可能偏简单**：来自 QA 数据集的知识三元组可能不涵盖所有类型的知识编辑需求（如逻辑规则、因果关系等）
- 改进方向：需要从根本上重新思考模型编辑——可能需要结合检索增强（RAG）或适配器（LoRA）等更温和的知识注入方式

## 相关工作与启发

- **vs ROME/MEMIT**：这些方法在 CounterFact 上报告 96%+ 的成功率，但本文证明此数据严重失真。它们的核心假设——精确定位和覆写特定 MLP 层的知识——在真实 QA 场景下不成立
- **vs Knowledge Neurons / Causal Tracing**：这些知识定位方法为编辑提供了理论基础，但本文暗示定位本身可能也受 teacher forcing 评估的误导
- 对后续模型编辑研究的启示：任何新方法都必须用 WILD 评估框架验证，否则可能重蹈先前的评估陷阱

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 揭示了一个影响整个领域的评估漏洞，见解深刻
- 实验充分度: ⭐⭐⭐⭐ 控制实验清晰有力，但模型和数据集覆盖面可扩展
- 写作质量: ⭐⭐⭐⭐⭐ 论证逻辑严密，发现的呈现方式很有冲击力
- 价值: ⭐⭐⭐⭐⭐ 对模型编辑领域有根本性影响，迫使社区重新审视已有成果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards a Principled Evaluation of Knowledge Editors](towards_a_principled_evaluation_of_knowledge_editors.md)
- [\[ACL 2025\] DocMEdit: Towards Document-Level Model Editing](docmedit_towards_document-level_model_editing.md)
- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[NeurIPS 2025\] Rethinking Residual Distribution in Locate-then-Edit Model Editing](../../NeurIPS2025/knowledge_editing/rethinking_residual_distribution_in_locate-then-edit_model_editing.md)
- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](../../ACL2026/knowledge_editing/fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)

</div>

<!-- RELATED:END -->
