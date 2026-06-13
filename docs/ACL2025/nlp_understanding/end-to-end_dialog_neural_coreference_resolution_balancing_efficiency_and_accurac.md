---
title: >-
  [论文解读] End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems
description: >-
  [ACL 2025][NLP理解][共指消解] 提出一个端到端神经共指消解系统，通过结合上下文嵌入、层次化注意力机制和优化策略（剪枝/量化），在OntoNotes等基准数据集上实现效率与准确率的平衡，SpanBERT达到87.3 F1。 领域现状： 共指消解（Coreference Resolution）是NLP中的基础任务…
tags:
  - "ACL 2025"
  - "NLP理解"
  - "共指消解"
  - "端到端系统"
  - "注意力机制"
  - "上下文嵌入"
  - "大规模系统"
---

# End-to-End Dialog Neural Coreference Resolution: Balancing Efficiency and Accuracy in Large-Scale Systems

**会议**: ACL 2025  
**arXiv**: [2504.05824](https://arxiv.org/abs/2504.05824)  
**代码**: 无  
**领域**: NLP理解  
**关键词**: 共指消解, 端到端系统, 注意力机制, 上下文嵌入, 大规模系统  

## 一句话总结

提出一个端到端神经共指消解系统，通过结合上下文嵌入、层次化注意力机制和优化策略（剪枝/量化），在OntoNotes等基准数据集上实现效率与准确率的平衡，SpanBERT达到87.3 F1。

## 研究背景与动机

**领域现状**: 共指消解（Coreference Resolution）是NLP中的基础任务，旨在识别文本中指向同一实体的不同表述。近年来基于BERT的方法取得了显著进步，但在大规模应用中效率仍是瓶颈。

**现有痛点**: 现有方法面临多方面挑战：处理长文档时的cache miss问题（Guo等，2023）、多语言环境下的性能差异（Pražák & Konopík，2024）、以及singleton检测的不足（Zou等，2024）。高性能模型（如13B参数量的大模型）计算开销巨大，难以在实际生产环境中部署。

**核心矛盾**: 在大规模文本处理中，模型精度和推理效率之间存在trade-off。增加模型深度和注意力头数可以提高F1分数，但同时增加了推理延迟和计算资源需求。

**本文目标**: 设计一个既能高效运行又能保持高精度的端到端共指消解系统，使其适合真实世界部署。

**切入角度**: 通过层次化注意力机制重新设计共指决策过程，并结合模型优化策略（剪枝和量化）来减少计算开销。

**核心 idea**: 用多层上下文嵌入+层次化注意力+有向图优化来构建一个效率-精度平衡的端到端共指消解系统。

## 方法详解

### 整体框架

系统分为两阶段：
1. **提及表示**: 通过上下文嵌入生成每个提及的动态嵌入表示 $\mathcal{C} = \{c_1, c_2, ..., c_n\}$，使用RNN和Transformer组合
2. **共指链接决策**: 通过注意力机制构建亲和度矩阵，再用有向图优化选择最优共指链接

### 关键设计

**上下文嵌入模块**: 对输入文本 $x = \{w_1, w_2, ..., w_n\}$，通过映射函数 $\phi$ 投射到高维嵌入空间：

$$\mathcal{E} = \{\phi(w_1), \phi(w_2), ..., \phi(w_n)\}$$

注意力分数计算：

$$a_{ij} = \frac{\exp(\alpha(\mathcal{E}_i, \mathcal{E}_j))}{\sum_{k=1}^{n} \exp(\alpha(\mathcal{E}_i, \mathcal{E}_k))}$$

精炼表示为 $R_i = \sum_{j=1}^{n} a_{ij} \mathcal{E}_j$，融合了原始语义和上下文信息。

**共指链接解析**: 构建亲和度矩阵 $A \in \mathbb{R}^{n \times n}$，注意力分数采用缩放点积形式：

$$a_{ij} = \text{softmax}\left(\frac{e(c_i, c_j)}{\sqrt{d}}\right)$$

优化目标为找到最优共指链接子集 $\mathcal{L} \subseteq \mathcal{C} \times \mathcal{C}$，约束每个提及最多链接到一个先行词。

### 损失函数/训练策略

损失函数同时考虑精度和召回率：

$$\mathcal{L}_{CR} = -(\alpha \cdot \log(P) + \beta \cdot \log(1-P))$$

其中 $P$ 是建立共指链接的概率, $\alpha$ 和 $\beta$ 分别控制精度和召回率的权重。

训练配置: 30 epochs, batch size=16, learning rate=3e-5 (AdamW), 序列长度512, 在NVIDIA V100 GPU上训练。数据划分80%/10%/10% (训练/验证/测试)。

## 实验关键数据

### 主实验

| 方法 | 数据集 | F1 | Precision | Recall |
|------|--------|:---:|:---:|:---:|
| BERT-large | OntoNotes v5.0 | 86.2 | 85.0 | 87.5 |
| **SpanBERT** | **OntoNotes v5.0** | **87.3** | **86.5** | **88.1** |
| Z-coref | Winograd Schema | 82.1 | 80.3 | 83.5 |
| Seq2seq | NusaCrowd | 81.5 | 79.7 | 83.3 |
| Knowledge Integration | BARThez | 84.8 | 83.2 | 86.5 |
| Event Coreference | CliCR | 79.9 | 78.6 | 81.0 |
| Event-aware Learning | OntoNotes v5.0 | 85.0 | 83.5 | 86.5 |

### 消融实验

| 配置 | 数据集 | F1 | 变化 |
|------|--------|:---:|:---:|
| BERT-large (无注意力) | OntoNotes v5.0 | 84.5 | -1.7 |
| BERT-large (静态嵌入) | OntoNotes v5.0 | 85.2 | -1.0 |
| Seq2seq (无优化) | NusaCrowd | 79.7 | -1.8 |
| Z-coref (固定学习率) | Winograd Schema | 80.6 | -1.5 |
| Event Coreference (减少层数) | CliCR | 77.5 | -2.4 |
| Event-aware (无注意力) | OntoNotes v5.0 | 82.0 | -3.0 |

### 关键发现

1. **注意力机制至关重要**: 移除注意力导致F1下降1.7-3.0个点，层次化注意力(87.3 F1)优于多头注意力(86.2 F1)和自适应注意力(86.8 F1)
2. **Transformer嵌入明显优于传统方法**: BERT(86.0)和RoBERTa(86.5)远超Word2Vec、GloVe、FastText(83.0)
3. **混合模型效果最佳**: Hybrid识别技术达到88.2 F1, 优于单一深度学习方法(86.5)
4. **模型深度影响显著**: 减少层数使CliCR上的F1从79.9掉到77.5

## 亮点与洞察

- **双阶段设计**: 先生成提及嵌入，再通过有向图优化链接决策，将复杂问题分解为可控子问题
- **效率-精度权衡**: 通过剪枝和量化策略在保持精度的同时显著降低推理开销
- **统一的消融框架**: 对注意力机制、嵌入类型、识别技术都做了系统对比，为后续研究提供参考

## 局限与展望

- 在高度复杂文本上资源消耗仍然较高，限制了资源受限环境下的部署
- 主要依赖基准数据集评估，对真实世界多样化文本的泛化能力未充分验证
- 对噪声数据的鲁棒性有待加强
- 未探索更大规模预训练模型（如Llama等decoder-only架构）的潜力
- 不同数据集使用不同基线但训练配置完全相同(统一lr=3e-5, batch=16)，公平性和针对性存疑

## 相关工作与启发

- Guo等(2023)的双缓存设计对长文档处理的启发——全局和局部实体分离
- Grenander等(2023)的句子级增量方法为实时处理提供了思路
- 混合方法（规则+神经网络）的成功提示：在共指消解中，领域知识的显式注入仍然有价值
- 与Martinelli等(2024)的Maverick对比：500M参数即可达到SOTA，说明参数效率设计是关键方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 整体框架较为标准，属于BERT-based方法的工程整合优化
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集多消融，但所有方法统一训练配置的做法有争议
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰但部分描述较为冗长，方法创新性阐述不够突出
- **价值**: ⭐⭐⭐⭐ — 提供了全面的共指消解系统设计参考，但突破性有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] BookCoref: Coreference Resolution at Book Scale](bookcoref_book_scale.md)
- [\[ACL 2025\] Adapting Psycholinguistic Research for LLMs: Gender-Inclusive Language in a Coreference Context](adapting_psycholinguistic_research_for_llms_gender-inclusive_language_in_a_coref.md)
- [\[ACL 2025\] Disambiguate First, Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing](disambiguate_first_parse_later_generating_interpretations_for_ambiguity_resoluti.md)
- [\[ICML 2025\] Cover Learning for Large-Scale Topology Representation](../../ICML2025/nlp_understanding/cover_learning_for_large-scale_topology_representation.md)
- [\[ACL 2025\] Generating Diverse Training Samples for Relation Extraction with Large Language Models](generating_diverse_training_samples_for_relation_extraction_with_large_language_.md)

</div>

<!-- RELATED:END -->
