---
title: >-
  [论文解读] MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables
description: >-
  [ACL 2025][信息检索/RAG][多表格推理] 提出MT-RAIG Bench——首个面向多表格检索增强洞察生成的大规模基准，以及MT-RAIG Eval——基于分解的细粒度自动评估框架，实验表明即使前沿LLM在多表格推理上仍表现不佳（忠实度仅约40%，完整度约60%）。
tags:
  - "ACL 2025"
  - "信息检索/RAG"
  - "多表格推理"
  - "检索增强生成"
  - "表格问答"
  - "洞察生成"
  - "自动评估框架"
---

# MT-RAIG: Novel Benchmark and Evaluation Framework for Retrieval-Augmented Insight Generation over Multiple Tables

**会议**: ACL 2025  
**arXiv**: [2502.11735](https://arxiv.org/abs/2502.11735)  
**作者**: Kwangwook Seo, Donguk Kwon, Dongha Lee (Yonsei University)  
**代码**: [https://kwondu.github.io/mt-raig](https://kwondu.github.io/mt-raig)  
**领域**: 信息检索  
**关键词**: 多表格推理, 检索增强生成, 表格问答, 洞察生成, 自动评估框架  

## 一句话总结

提出MT-RAIG Bench——首个面向多表格检索增强洞察生成的大规模基准，以及MT-RAIG Eval——基于分解的细粒度自动评估框架，实验表明即使前沿LLM在多表格推理上仍表现不佳（忠实度仅约40%，完整度约60%）。

## 研究背景与动机

### 问题背景
表格是现实数据中无处不在的结构化信息载体。现有表格推理研究已从简单的事实级问答（如WikiSQL、WTQ）发展到洞察级任务（如QTSumm、InsTaSumm），后者要求系统从表格中综合隐含知识并提供可解释的分析。

### 已有工作的不足

**封闭域假设**：现有方法在测试时提供预定义的gold表格，但实际场景中用户往往不知道哪些表格与查询相关，手动指定表格既昂贵又不现实

**单表格限制**：多数工作假设所有信息包含在单一表格中，无法处理需要跨多表综合洞察的用户需求

**评估方法粗糙**：现有自动评估指标（BLEU、ROUGE、G-Eval等）以粗粒度方式分析输出，难以检测洞察与多表格间的事实对齐和多跳查询的完整覆盖

### 核心动机
构建一个要求系统"先检索多个证据表格、再跨表综合生成洞察"的端到端基准，同时设计与人类判断高度对齐的细粒度评估框架。

## 方法详解

### MT-RAIG Bench 构建

#### 任务定义
给定自然语言问题 $q$ 和外部表格库 $T$，系统需：(1) 检索证据表格集 $\hat{T_q} \subset T$；(2) 基于检索结果生成洞察 $i = Gen(q, \hat{T_q})$。

#### 多表格集构建
从两个来源收集多表格集合：
- **可连接表格（Joinable）**：利用SPIDER数据集的外键关系，直接通过公共键列连接
- **主题相关表格（Topic-related）**：利用Open-WikiTable中的表格标题和列头作为语义指标，聚类具有主题关联但无直接键列连接的表格

#### 问题标注（Human-in-the-loop）
使用GPT-4o mini作为标注agent，对每个表格集生成10个洞察级问题。三个关键策略：
1. **去上下文化**：将表格标题关键词嵌入问题，确保语义对齐
2. **关系增强**：手动识别跨表共享属性和数据点，引导agent生成反映表格间关系的问题
3. **人工引导演示迭代**：将问题分为4类（分析总结A&S、比较关系C&R、表现结果P&O、趋势模式T&P），通过人工评审-反馈循环迭代优化种子示例

#### 洞察标注
1. **程序化事实扩展**：让agent生成Python函数 `expand_facts`，在表格上执行以系统性提取丰富事实
2. **问题相关知识抽取**：过滤无关内容，仅保留与问题相关的事实，再由agent生成最终洞察

#### 双阶段质量控制
- 第一阶段：agent自验证，检查相关性、忠实度、完整度
- 第二阶段：人工验证，Cohen's Kappa达到0.78-0.86，高质量保证

### MT-RAIG Eval 评估框架

#### 忠实度评分（Faithfulness）
通过**表格感知的洞察分解**，将洞察拆解为可验证的原子声明（claims），每个声明显式关联源表格：
$$S_{Faith.} = \frac{1}{|C|} \sum_{k=1}^{|C|} \mathcal{V}(c_k, \hat{T_q})$$
其中 $\mathcal{V}(c_k, \hat{T_q}) \in \{0,1\}$ 验证每个声明是否有表格支撑。

#### 完整度评分（Completeness）
通过**问题感知的洞察分解**，将ground truth和预测洞察分别分解为原子主题，计算语义匹配的Precision、Recall和F1：
$$F1 = \frac{2 \cdot P \cdot R}{P + R}$$

## 实验关键数据

### 实验1：多表格检索性能

| 类型 | 检索器 | R@2 | R@5 | R@10 | R@20 |
|------|--------|-----|-----|------|------|
| 通用 | BM25 | 17.26 | 27.19 | 33.72 | 41.16 |
| 通用 | DPR | 44.58 | 68.24 | 80.83 | 88.45 |
| 通用 | Contriever | 23.47 | 35.67 | 44.27 | 52.12 |
| 表格专用 | DTR | 37.77 | 59.60 | 74.50 | 86.22 |
| 表格专用 | TableLlama | 36.93 | 59.48 | 72.44 | 81.56 |

通用文本嵌入模型DPR反而优于表格专用模型，说明MT-RAIG任务更重视洞察级语义连接而非表格结构特征。

### 实验2：洞察生成性能（使用DPR top-10检索表格）

| 生成器 | 忠实度 Avg | 忠实度 Gold | 完整度 Avg | 完整度 Gold |
|--------|-----------|------------|-----------|------------|
| o3-mini | 38.85 | 42.57 | 60.20 | 60.81 |
| GPT-4o | 36.98 | 41.46 | 61.52 | 63.28 |
| Claude 3.5 Sonnet | 39.68 | 43.35 | 58.63 | 59.70 |
| DeepSeek-R1-8B | 35.55 | 40.12 | 60.96 | 63.41 |
| Qwen2-7B | 33.40 | 40.59 | 59.21 | 61.57 |
| Llama 3.1-8B | 31.59 | 37.62 | 56.90 | 58.14 |
| Dater (TQA) | 27.92 | 32.43 | 59.12 | 62.32 |
| Chain-of-Table (TQA) | 31.90 | 37.71 | 57.63 | 62.15 |
| TaPERA (TQA) | 19.25 | 20.68 | 56.59 | 55.28 |

### 实验3：MT-RAIG Eval 元评估

| 评估指标 | 忠实度相关性 | 完整度相关性 |
|---------|------------|------------|
| SacreBLEU | 31.33 | 33.01 |
| ROUGE-L | 27.69 | 43.43 |
| BERTScore | 24.82 | 43.29 |
| G-Eval | 47.82 | 26.35 |
| **MT-RAIG Eval** | **64.94** | **67.67** |
| 人类评估者间 | 84.81 | 75.70 |

MT-RAIG Eval在两个维度上均大幅超越现有指标，与人类判断的Pearson相关性达64.94/67.67。

## 亮点

- **首创性基准**：MT-RAIG Bench是首个要求检索+多表融合+洞察生成的大规模基准（18,532测试样本，19,563唯一表格，平均每样本2.88个gold表格）
- **精细评估框架**：MT-RAIG Eval通过表格感知和问题感知的双路分解策略，实现了远超传统指标的人类对齐度
- **深度分析发现**：(1) 检索更多表格并非总有益，超过阈值后性能下降；(2) 忠实度对噪声表格极为敏感，而完整度相对稳健；(3) 即使无噪声，模型随参考表格数量增加性能显著下降
- **推理时间缩放有效**：DeepSeek-R1-8B以8B参数量达到与闭源模型可比的性能，验证了推理时缩放在表格洞察任务上的有效性

## 局限与展望

- **合成数据固有风险**：尽管有Human-in-the-loop，LLM生成的问题仍可能存在语言多样性不足和不自然描述的问题
- **领域覆盖有限**：仅包含关系数据库和Wikipedia表格，未覆盖金融、科学、医疗等专业领域
- **评估依赖LLM**：MT-RAIG Eval的分解器和验证器均依赖LLM，可能受骨干模型偏见影响
- **检索与生成解耦**：固定使用DPR top-10的实验设置可能低估端到端联合优化的潜力
- **表格规模受限**：平均每表仅10.54行x6.04列，未涉及大规模工业表格场景

## 与相关工作的对比

- **WikiSQL/SPIDER/WTQ**：事实级单表格任务，MT-RAIG在推理深度（洞察级）、表格数量（多表）、检索需求三方面全面升级
- **QTSumm/InsTaSumm**：虽为洞察级但仅支持单表格和封闭域，MT-RAIG同时要求检索和多表融合
- **Open-WikiTable**：虽支持检索但仍为事实级问答，MT-RAIG要求生成长文本洞察（平均189.87词vs 1.90词）
- **G-Eval**：LLM评估方法但粗粒度，忠实度相关性仅47.82 vs MT-RAIG Eval的64.94
- **TAPAS-Acc**：训练式验证器，在MT-RAIG任务上相关性甚至为负（-10.40），完全不适用

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个多表格检索增强洞察生成基准，任务定义填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ — 覆盖5种检索器、11种生成器、元评估、多维分析实验，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，构建流程详尽，图表丰富
- 价值: ⭐⭐⭐⭐ — 为表格推理社区提供了高质量的挑战性基准和可靠评估工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](rageval_scenario_specific_rag_evaluation_dataset_generation_framework.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] Unanswerability Evaluation for Retrieval Augmented Generation](unanswerability_evaluation_for_retrieval_augmented_generation.md)
- [\[ACL 2025\] FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation](flexrag_a_flexible_and_comprehensive_framework_for_retrieval-augmented_generatio.md)
- [\[ACL 2025\] GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation](garage_a_benchmark_with_grounding_annotations_for_rag_evaluation.md)

</div>

<!-- RELATED:END -->
