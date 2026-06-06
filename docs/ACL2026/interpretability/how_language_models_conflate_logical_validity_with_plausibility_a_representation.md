---
title: >-
  [论文解读] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects
description: >-
  [ACL 2026][可解释性][内容效应] 通过表示分析揭示 LLM 中"逻辑有效性"和"合理性"两个概念在隐层空间中高度对齐，导致模型将合理性与有效性混淆（内容效应），并构造去偏转向向量有效解耦这两个概念，减少内容效应同时提升推理准确率。
tags:
  - "ACL 2026"
  - "可解释性"
  - "内容效应"
  - "逻辑有效性"
  - "合理性"
  - "线性表示"
  - "转向向量"
---

# How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects

**会议**: ACL 2026  
**arXiv**: [2510.06700](https://arxiv.org/abs/2510.06700)  
**代码**: [https://github.com/leobertolazzi/content-effect-interpretability](https://github.com/leobertolazzi/content-effect-interpretability)  
**领域**: 社会计算  
**关键词**: 内容效应, 逻辑有效性, 合理性, 线性表示, 转向向量

## 一句话总结
通过表示分析揭示 LLM 中"逻辑有效性"和"合理性"两个概念在隐层空间中高度对齐，导致模型将合理性与有效性混淆（内容效应），并构造去偏转向向量有效解耦这两个概念，减少内容效应同时提升推理准确率。

## 研究背景与动机

**领域现状**：人类在三段论推理等逻辑任务中存在"内容效应"——语义内容的合理性影响对逻辑有效性的判断（如结论合理的无效论证容易被误判为有效）。人类的这一现象由双过程理论（快速直觉系统 vs 慢速分析系统）解释。近期研究发现 LLM 也展现类似的内容效应。

**现有痛点**：虽然 LLM 的内容效应行为已被充分记录，但其内在机制仍不清楚。已有研究停留在行为层面的观察，缺乏对 LLM 内部表示的深入分析。

**核心矛盾**：逻辑有效性取决于论证结构而非内容，但 LLM 在表示空间中可能将这两个本应独立的概念纠缠在一起。

**本文目标**：(1) 验证 LLM 是否展现内容效应；(2) 分析有效性和合理性在内部表示中的编码方式；(3) 探究表示层面的纠缠是否预测行为层面的内容效应；(4) 设计干预手段解耦这两个概念。

**切入角度**：基于线性表示假说——高层概念在 LLM 隐层空间中被线性编码——检验有效性和合理性的线性方向是否高度相似。

**核心 idea**：LLM 中内容效应的根源是有效性方向和合理性方向在表示几何中的纠缠对齐，可通过构造去偏转向向量来解耦。

## 方法详解

### 整体框架
在 1280 个三段论上评估 10 个 LLM（Qwen-2.5、Qwen-3、Gemma-3 系列），通过差异均值方法提取有效性和合理性的线性方向，分析它们的相似度，进行跨任务转向实验验证因果关系，最后构造去偏向量减少内容效应。

### 关键设计

1. **概念方向提取（差异均值法）**:

    - 功能：将二元概念表示为隐层空间中的单一方向
    - 核心思路：对每层 $l$，计算模型预测为正类（如"有效"）和负类（如"无效"）的样本在最后 token 位置的平均激活向量之差 $v_{\text{concept}}^l = \mu_{\text{positive}}^l - \mu_{\text{negative}}^l$。使用模型自身的预测标签而非真实标签，因为关注的是模型自身的"信念"编码。
    - 设计动机：差异均值法简洁有效，且与线性表示假说直接对应

2. **跨任务转向实验**:

    - 功能：验证有效性和合理性之间的因果交互——合理性向量能否影响有效性判断，反之亦然
    - 核心思路：将合理性任务提取的转向向量 $v_{\text{plausibility}}^l$ 应用于逻辑有效性分类任务（以及反方向），测量转向力度（标签翻转比例）。转向时始终对抗模型原始预测：预测为负类时加上向量，预测为正类时减去向量。
    - 设计动机：如果合理性向量能有效改变有效性判断，说明这两个概念在表示空间中存在因果纠缠而非仅仅相关

3. **去偏转向向量构造**:

    - 功能：解耦有效性和合理性的表示，减少内容效应
    - 核心思路：构造去偏向量使模型在评估逻辑有效性时不受合理性影响。在有效层（转向力度 $>0.75$）上应用去偏向量，使内容效应指标 CE 下降同时推理准确率提升。
    - 设计动机：如果纠缠是内容效应的根源，那么解耦应该能同时减少偏差和提升推理能力

### 指标设计
内容效应 CE = $\frac{1}{2}(\Delta_{v^+} + \Delta_{v^-})$，其中 $\Delta_{v^+}$ 衡量合理结论时有效论证的准确率优势。CE=0 表示有效性与合理性独立，CE=1 表示完全由合理性驱动。

## 实验关键数据

### 主实验
行为层面的内容效应：

| 模型 | 设置 | $D_{v^+,p^+}$ 准确率 | $D_{v^-,p^+}$ 准确率 | $D_{v^+,p^-}$ 准确率 | CE |
|------|------|---------------------|---------------------|---------------------|-----|
| Qwen2.5-32B | 0-shot | 100.00 | 67.50 | 60.92 | 0.348 |
| Qwen2.5-32B | CoT | 98.67 | 86.64 | 93.10 | 0.096 |
| Qwen3-14B | 0-shot | 97.33 | 90.83 | 60.92 | 0.213 |
| Qwen3-14B | CoT | 95.31 | 99.10 | 92.50 | 0.014 |

### 表示分析

| 概念对 | 平均余弦相似度 | 说明 |
|--------|--------------|------|
| 有效性 - 合理性 | 0.48-0.64 | 高度对齐 |
| 有效性 - 无害性 | 0.10-0.13 | 低相似（控制） |
| 有效性 - 上位词 | -0.12 至 -0.17 | 低相似（控制） |

### 关键发现
- 所有测试模型均展现内容效应，CoT 提示显著降低 CE（从 0.213-0.348 降至 0.014-0.096）
- 有效性和合理性向量的余弦相似度（0.48-0.64）远高于控制概念（0.10-0.13），确认特异性纠缠
- 跨任务转向成功：合理性向量能有效翻转有效性判断，且反之亦然
- 有效性-合理性对齐程度与行为层面的 CE 强度成正相关
- 去偏向量同时减少 CE 和提升推理准确率，证明解耦是有效的
- CoT 虽降低行为 CE，但表示层面的对齐程度并未显著改变（p=0.625）

## 亮点与洞察
- 提供了 LLM 内容效应的首个表示层面解释——不是行为上的"bug"而是表示几何的结构性问题。这一发现比纯行为研究深刻得多
- CoT 降低行为 CE 但不改变表示对齐的发现非常有趣——CoT 可能是在推理过程中"绕过"而非"解决"了纠缠问题
- 去偏转向向量作为干预手段，展示了从表示分析到实际改进的完整闭环

## 局限与展望
- 仅在三段论推理上验证，其他推理形式（条件推理、概率推理）的内容效应机制可能不同
- 数据集规模较小（1280 个三段论），虽覆盖所有 64 种类型但语义变化有限
- 去偏向量的效果依赖于层的选择，需要验证集来确定最优层
- 未来可探索非线性编码的概念是否也存在类似纠缠现象

## 相关工作与启发
- **vs Lampinen et al.**: 他们记录了 LLM 的内容效应行为，本文从表示层面揭示了机制
- **vs Marks & Tegmark (真值方向)**: 他们发现真值被线性编码，本文进一步发现有效性方向与合理性方向纠缠
- **vs Arditi et al. (拒绝方向)**: 类似方法论但应用于不同概念，本文的创新在于分析两个概念间的交互而非单一概念

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从表示几何角度解释 LLM 内容效应，洞察深刻
- 实验充分度: ⭐⭐⭐⭐ 10 个模型、控制实验、因果验证齐全，但数据集单一
- 写作质量: ⭐⭐⭐⭐⭐ RQ 驱动的结构清晰，分析层层递进
- 价值: ⭐⭐⭐⭐⭐ 对理解和改进 LLM 逻辑推理有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)
- [\[ICML 2026\] How Language Models Process Negation](../../ICML2026/interpretability/how_language_models_process_negation.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](../../ICML2026/interpretability/a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](../../ICML2026/interpretability/query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
