---
title: >-
  [论文解读] Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models
description: >-
  [ACL 2025][信息检索/RAG][attribution bias] 定义并研究 RAG 中 LLM 对作者身份信息的归因敏感性和偏差，通过反事实评估发现告知 LLM 文档作者身份可显著改变归因质量 3-18%，且 LLM 存在对人类作者身份的归因偏差。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "attribution bias"
  - "RAG"
  - "counterfactual evaluation"
  - "authorship"
  - "citation generation"
---

# Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models

**会议**: ACL 2025  
**arXiv**: [2410.12380](https://arxiv.org/abs/2410.12380)  
**代码**: [https://github.com/aminvenv/attrieval](https://github.com/aminvenv/attrieval)  
**领域**: 信息检索  
**关键词**: attribution bias, RAG, counterfactual evaluation, authorship, citation generation

## 一句话总结
定义并研究 RAG 中 LLM 对作者身份信息的归因敏感性和偏差，通过反事实评估发现告知 LLM 文档作者身份可显著改变归因质量 3-18%，且 LLM 存在对人类作者身份的归因偏差。

## 研究背景与动机

**领域现状**：RAG 通过让 LLM 引用来源文档来增强答案可验证性，是减少幻觉的重要方向。大量工作聚焦改善归因质量。

**现有痛点**：改善归因的同时可能引入偏差——LLM 可能对不同来源的文档有不同程度的信任。先前研究发现 LLM 倾向于选择自身生成的内容，但这一结论可能有替代解释。

**核心矛盾**：当 LLM 知道文档是人写的还是 AI 生成的，它的归因行为会如何改变？这种改变是否构成系统性偏差？

**本文目标** 定义和量化 RAG 中 LLM 的归因敏感性（sensitivity）和归因偏差（bias）。

**切入角度**：反事实评估——通过交换文档的作者标签（真实 vs 反事实），观察归因质量的变化。

**核心 idea**：LLM 在 RAG 中对人类作者标签有系统性偏差——标记为 [Human] 的文档更容易被引用，即使内容完全相同。

## 方法详解

### 整体框架
设计三种 RAG 模式（Vanilla/Authorship-Informed/Counterfactual-Authorship）-> 用同一组检索文档在三种模式下生成答案和归因 -> 通过比较计算归因敏感性（CAS）和归因偏差（CAB）。

### 关键设计

1. **三种 RAG 模式**

    - **Vanilla RAG**：标准 RAG，不提供作者信息（基线）
    - **Authorship-Informed RAG**：用 [Human] / [LLM] 标记文档的真实作者
    - **Counterfactual-Authorship RAG**：交换标签——人写的文档标为 [LLM]，LLM 写的标为 [Human]
    - 设计动机：通过真实和反事实标签的对比，分离作者信息对归因的纯影响

2. **归因敏感性指标 CAS (Counterfactually-estimated Attribution Sensitivity)**

    - $CAS(Q) = \frac{1}{|Q|} \sum_{q \in Q} |M_{Informed}^q - M_{Vanilla}^q|$
    - 测量知道作者信息后归因质量的变化幅度
    - 设计动机：量化作者信息对 LLM 行为的影响程度

3. **归因偏差指标 CAB (Counterfactually-estimated Attribution Bias)**

    - $CAB(Q) = \frac{\omega}{|Q|} \sum_{q \in Q} (M_{Informed}^q - M_{CF-informed}^q)$
    - 正值表示偏向人类作者，负值表示偏向 LLM 作者
    - 设计动机：通过反事实翻转，消除文档内容差异的影响，纯粹测量标签偏差

4. **归因置信度 AC (Attribution Confidence)**

    - 分析 LLM 生成引用 token 时的概率是否因作者标签不同而不同
    - 设计动机：从模型内部信心角度补充分析

5. **合成文档集构建**

    - 用 Llama3 低温度重写人类文档，保持相关/不相关状态不变
    - 两位专家标注验证合成文档质量
    - 设计动机：创建内容对等但作者不同的文档对

### 实验设置
- 3 个 LLM：Mistral-7B, Llama3-8B, GPT-4
- 2 个数据集：Natural Questions (NQ), MS MARCO
- 4 种文档组合：相关文档和不相关文档分别为人写/LLM 写

## 实验关键数据

### 主实验 — 归因质量对比（NQ 数据集，Precision / Recall）

| 模型 | 相关文档 | 不相关文档 | Vanilla Prec | Informed Prec | CF-Informed Prec |
|------|---------|-----------|-------------|--------------|-----------------|
| Mistral | LLM | Human | 47.6 | 42.1 | **52.7†** |
| Mistral | Human | LLM | 51.0 | **53.4†** | 44.0 |
| Llama3 | LLM | Human | 49.2 | 45.4 | **57.2†** |
| Llama3 | Human | LLM | 53.5 | **59.9†** | 44.8 |
| GPT-4 | LLM | Human | 63.3 | 59.7 | **65.9†** |
| GPT-4 | Human | LLM | 64.1 | **66.1** | 60.3 |

### 归因敏感性 CAS（越高越敏感）

| 模型 | NQ (LLM-rel/Human-nonrel) | NQ (Human-rel/LLM-nonrel) |
|------|--------------------------|--------------------------|
| Mistral | **16.2†** | **20.1** |
| Llama3 | 13.2† | **17.7†** |
| GPT-4 | 9.7† | 8.7 |

### 归因偏差 CAB（正值=偏向人类）

| 模型 | NQ CAB | MS MARCO CAB | 方向 |
|------|--------|-------------|------|
| Mistral | **+5.3** | **+3.1** | 偏向人类 |
| Llama3 | **+7.5** | **+4.2** | 偏向人类 |
| GPT-4 | **+3.1** | **+2.8** | 偏向人类 |

### 关键发现
- **所有三个 LLM 对作者信息都敏感**：告知作者身份可改变归因质量 3-18%（CAS 指标）
- **一致偏向人类作者**：当告知相关文档为人写时，归因精度提高；反事实翻转后精度下降——说明偏差来自标签而非内容
- **Llama3 偏差最大**（CAB 最高），GPT-4 偏差最小但仍显著
- **答案正确性基本不变**：作者信息主要影响归因行为，不影响答案质量
- **即使不使用 LLM 生成文档，仅添加 [Human]/[LLM] 标签就能产生偏差**——说明偏差是对标签的反应而非对内容质量差异的反应

## 亮点与洞察
- **反事实评估框架**非常精巧——通过交换标签消除内容混淆因素，实现了对"标签偏差"的纯粹测量。这种方法论可迁移到其他偏差研究。
- **人类作者偏差为先前发现提供替代假设**：先前工作认为 LLM 偏好自己生成的内容，但本文发现可能是偏好标记为"人类"的内容——两者方向相反，暗示偏差机制更复杂。
- **文档元数据影响 LLM 信任**的发现有深远含义——在实际 RAG 系统中，文档的元信息（来源、作者、发布时间等）都可能影响 LLM 的行为。

## 局限与展望
- 仅测试 [Human] vs [LLM] 两种作者类型，未探索具体作者名字、机构等更细粒度的影响
- 合成文档虽质量高但不等同于真实 LLM 生成内容
- 仅 500 查询样本量
- 改进方向：更多作者类型（如"专家""学生"）、真实场景中的归因偏差研究、偏差缓解方法

## 相关工作与启发
- **vs Tan et al. (2024)**：他们发现 LLM 偏好自己生成的文本，本文发现 LLM 偏好标记为人类的文本——两个发现看似矛盾，但作用在不同机制上
- **vs Gao et al. (2023) ALCE**：ALCE 评估归因质量，本文在其基础上增加了偏差维度
- **vs 公平性研究（Ziems et al. 2024）**：归因偏差是 LLM 偏差的一个新维度

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义和量化 RAG 中的归因偏差，反事实评估设计精巧
- 实验充分度: ⭐⭐⭐⭐ 3 模型 × 2 数据集 × 4 种文档组合 × 3 种 RAG 模式
- 写作质量: ⭐⭐⭐⭐⭐ 指标定义严谨，实验设计逻辑清晰
- 价值: ⭐⭐⭐⭐⭐ 对 RAG 系统的可信度和公平性有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] RARE: Retrieval-Augmented Reasoning Enhancement for Large Language Models](rare_retrieval_augmented_reasoning.md)
- [\[ACL 2025\] REFIND at SemEval-2025 Task 3: Retrieval-Augmented Factuality Hallucination Detection in Large Language Models](refind_at_semeval-2025_task_3_retrieval-augmented_factuality_hallucination_detec.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ACL 2025\] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)

</div>

<!-- RELATED:END -->
