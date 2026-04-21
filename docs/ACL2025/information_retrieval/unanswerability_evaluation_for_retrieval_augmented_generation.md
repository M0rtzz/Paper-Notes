---
title: >-
  [论文解读] Unanswerability Evaluation for Retrieval Augmented Generation
description: >-
  [ACL 2025][RAG评估] UAEval4RAG 提出了一个针对 RAG 系统处理不可回答查询的全面评估框架，定义了六类不可回答类别，能够基于任意知识库自动合成测试数据并评估系统的拒绝能力，实验揭示没有单一配置能在所有数据集上同时最优化可回答和不可回答查询的表现。
tags:
  - ACL 2025
  - RAG评估
  - 不可回答查询
  - 拒绝能力
  - 知识库定制
  - 评估框架
---

# Unanswerability Evaluation for Retrieval Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2412.12300](https://arxiv.org/abs/2412.12300)  
**代码**: https://github.com/SalesforceAIResearch/Unanswerability_RAGE  
**领域**: NLP理解  
**关键词**: RAG评估, 不可回答查询, 拒绝能力, 知识库定制, 评估框架

## 一句话总结

UAEval4RAG 提出了一个针对 RAG 系统处理不可回答查询的全面评估框架，定义了六类不可回答类别，能够基于任意知识库自动合成测试数据并评估系统的拒绝能力，实验揭示没有单一配置能在所有数据集上同时最优化可回答和不可回答查询的表现。

## 研究背景与动机

1. **领域现状**：RAG（检索增强生成）已成为提高 LLM 可靠性的关键技术。现有评估框架（如 RAGAS、ARES、RGB）主要关注**可回答查询**的准确率和相关性。

2. **现有痛点**：这些框架完全忽略了一个关键能力——**适当拒绝不可回答的请求**。在真实场景中，用户经常会提出无法被知识库回答的问题（信息不足、前提错误、超出范围等），此时系统应当拒绝而非生成幻觉。

3. **核心矛盾**：现有的不可回答性 benchmark（如 Brahman et al.）只针对 LLM 本身，使用通用的不可回答请求，无法定制到特定知识库。这导致拒绝往往是因为检索不到相关上下文，而非真正理解该请求不应被回答。另外，少数评估 RAG 拒绝能力的工作（如 Ming et al.）只关注单一类型的不可回答请求。

4. **本文目标** 如何为任意知识库自动生成多类别、高质量的不可回答数据集，并系统评估 RAG 系统的拒绝能力？

5. **切入角度**：受 Brahman et al. 启发，定义六种不可回答类别（从欠定义到安全问题），并为每类设计专门的合成管道和验证机制。

6. **核心 idea**：构建一个端到端的评估框架，能基于任何知识库自动合成六类不可回答查询、自动评估 RAG 系统的拒绝率和可接受率。

## 方法详解

### 整体框架

给定一个知识库，UAEval4RAG 框架分三步：（1）根据知识库内容自动合成六类不可回答查询的数据集；（2）将这些查询送入待评估的 RAG 系统获取响应；（3）用 LLM-based 指标评估响应的拒绝质量。三个步骤全自动化，用户只需提供知识库。

### 关键设计

1. **六类不可回答查询分类体系**:

    - 功能：全面覆盖 RAG 系统可能面对的不可回答场景
    - 核心思路：定义六种类别并标注难度——(1) Underspecified（缺少关键信息，Hard）：如"宠物允许吗？"缺少地点；(2) False-presupposition（错误前提，Easy）：如假设 Georgia 有迪士尼乐园；(3) Nonsensical（无意义，Medium）：拼写错误或逻辑混乱；(4) Modality-limited（模态限制，Medium）：要求文本系统展示图片；(5) Safety-concerned（安全顾虑，Medium）：与知识库高度相关的有害请求；(6) Out-of-Database（库外问题，Easy）：与知识库相关但答案不在其中
    - 设计动机：不同类别测试 RAG 系统不同层面的拒绝能力，从理解用户意图到识别安全风险

2. **自动合成管道**:

    - 功能：为任意知识库自动生成高质量的不可回答测试数据
    - 核心思路：前五类（Underspecified 到 Safety）共享一个管道——随机选取知识库 chunk，基于类别定义和 ICL 示例让 LLM 生成请求和解释，再由 LLM 验证是否符合类别标准，只有通过验证的才纳入数据集。Out-of-Database 类采用独立管道——先从知识库提取关键短语，爬取相关最新新闻生成问答对，检索验证知识库中确实不含答案
    - 设计动机：生成+验证的两步机制确保数据质量；Out-of-Database 单独处理是因为它需要确保问题与知识库高度相关但答案确实不存在

3. **三个 LLM-based 评估指标**:

    - 功能：量化评估 RAG 系统处理不可回答查询的能力
    - 核心思路：(1) **Unanswered Ratio**（客观指标）——量化系统拒绝回答的比例，定义一致；(2) **Acceptable Ratio**（主观指标）——根据各类别的具体标准判断响应是否符合人类偏好（如 Underspecified 类的可接受标准是拒绝/要求澄清/多角度回答，Modality-limited 则必须声明模态不支持）；(3) **Joint Score** = $w_1 \times \text{Correctness} + w_2 \times \text{Acceptable Ratio}$ 平衡可回答和不可回答性能
    - 设计动机：单指标无法全面评估，需要同时衡量拒绝率和拒绝质量，Joint Score 让用户可根据应用需求调整权重

### 损失函数 / 训练策略

本文是评估框架而非训练方法，不涉及损失函数。核心技术贡献在合成管道和评估指标设计。

## 实验关键数据

### 主实验

| 数据集 | 最佳配置 (Embedding+Retrieval) | 可回答 Correctness | 不可回答 Acceptable Ratio | Joint Score |
|--------|-------------------------------|--------------------|-----------------------|-------------|
| TriviaQA | Cohere+Vector+无 | 88.0% | 54.8% | 78.04% |
| TriviaQA | BGE+Vector+Cohere | 87.6% | 55.5% | 77.97% |
| MuSiQue | Cohere+Vector+Cohere+HyDE | 48.0% | 62.7% | 52.41% |
| MuSiQue | BGE+Ensemble+Cohere | 47.2% | 62.8% | 51.88% |

**Prompt 设计的影响**（TriviaQA，GPT-4o）：

| Prompt | Correctness | Acceptable Ratio | Joint Score |
|--------|-------------|-------------------|-------------|
| Default | 88.0% | 53.2% | 77.56% |
| Prompt #1 | 88.4% | 84.3% | 87.20% |
| Prompt #2 | 74.8% | 83.0% | 77.26% |

### 消融实验

| LLM 评估器 | Unanswered Accuracy | Unanswered F1 | Acceptable Accuracy | Acceptable F1 |
|-----------|--------------------|--------------|--------------------|--------------|
| GPT-4o | 82.0% | 76.9% | 84.0% | 85.2% |
| Claude 3.5 Sonnet | 84.0% | 76.9% | 81.3% | 83.1% |
| Deepseek-R1 | 84.4% | 76.7% | 83.3% | 86.0% |

合成数据质量验证：TriviaQA 数据集 92% 人工审核准确率，评审者间一致性 0.85；MuSiQue 92% 准确率，一致性 0.88。

### 关键发现

- **没有"万能"配置**：由于知识库分布差异，没有任何单一组件组合能在所有数据集上同时最优化可回答和不可回答性能
- **Prompt 设计至关重要**：最佳 prompt 可将不可回答 Acceptable Ratio 提升约 80%，同时对可回答准确率影响极小
- **LLM 选择影响显著**：Claude 3.5 Sonnet 比 GPT-4o 在 Correctness 上提升 0.4%，在不可回答 Acceptable Ratio 上提升 10.4%
- **Underspecified 类最难处理**：模型很难判断信息是否"足够"，拒绝率最低
- **不同知识库的不可回答难度不同**：知识分布越集中越容易合成 Out-of-Database 请求

## 亮点与洞察

- **知识库定制评估的思路**：不用通用 benchmark 而是根据具体知识库动态生成测试数据——这个思路可迁移到任何需要定制化测试的 NLP 系统
- **六类分类覆盖全面**：从信息完整性（Underspecified）到安全性（Safety），从逻辑性（Nonsensical、False-presuppositions）到系统能力（Modality-limited），涵盖了 RAG 拒绝能力的几乎所有维度
- **Joint Score 的设计**：允许用户根据应用场景自定义可回答性和拒绝率的权重平衡，非常实用

## 局限与展望

- 目前只测试了英语数据集，多语言场景下不可回答性的表现可能不同
- 合成管道依赖 LLM 的生成和验证能力，对于特别专业的领域（如法律、医学）可能需要领域适配
- 六类分类虽然全面但可能不够细粒度，如 Underspecified 可以进一步细分为缺时间/缺地点/缺主体等
- 只评估了拒绝能力，未探讨如何改进 RAG 系统的拒绝策略

## 相关工作与启发

- **vs RAGAS (Es et al.)**: RAGAS 只评估检索相关性和生成质量，完全不考虑不可回答场景；UAEval4RAG 填补了这个空白
- **vs Ming et al.**: Ming et al. 只测试 LLM 能否拒绝不一致上下文中的单一类型不可回答请求；UAEval4RAG 覆盖六类且基于原始知识库合成
- **vs Brahman et al.**: Brahman et al. 评估 LLM 本身的不可回答能力，使用通用请求；UAEval4RAG 专门针对 RAG 系统，数据与知识库高度相关

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个系统化的 RAG 不可回答性评估框架，六类分类体系有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 27种组件组合 × 3 LLM × 3 prompt × 4数据集的全面实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类体系定义明确
- 价值: ⭐⭐⭐⭐⭐ 对 RAG 系统的评估完整性有重要补充，直接可用于工业界 RAG 系统的质量保证

<!-- RELATED:START -->

## 相关论文

- [MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables](mt-raig_novel_benchmark_and_evaluation_framework_for_retrieval-augmented_insight.md)
- [Evaluation of Attribution Bias in Generator-Aware Retrieval-Augmented Large Language Models](evaluation_of_attribution_bias_in_generator-aware_retrieval-augmented_large_lang.md)
- [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)
- [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)

<!-- RELATED:END -->
