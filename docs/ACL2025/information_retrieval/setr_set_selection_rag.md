---
title: >-
  [论文解读] SetR: Shifting from Ranking to Set Selection for Retrieval Augmented Generation
description: >-
  [ACL 2025][RAG] 提出 SetR，将 RAG 中的文档排序范式转变为集合选择范式，通过 CoT 推理识别查询的信息需求并选择最优文档集合，在使用更少文档（平均 2.91 个 vs 5 个）的同时显著提升多跳问答性能。
tags:
  - ACL 2025
  - RAG
  - 集合选择
  - reranking
  - Chain-of-Thought
  - 多跳问答
  - 信息需求识别
---

# SetR: Shifting from Ranking to Set Selection for Retrieval Augmented Generation

**会议**: ACL 2025  
**arXiv**: [2507.06838](https://arxiv.org/abs/2507.06838)  
**代码**: [LGAI-Research/SetR](https://github.com/LGAI-Research/SetR)  
**领域**: 信息检索 / RAG  
**关键词**: RAG, 集合选择, reranking, Chain-of-Thought, 多跳问答, 信息需求识别  

## 一句话总结

提出 SetR，将 RAG 中的文档排序范式转变为集合选择范式，通过 CoT 推理识别查询的信息需求并选择最优文档集合，在使用更少文档（平均 2.91 个 vs 5 个）的同时显著提升多跳问答性能。

## 研究背景与动机

- **问题定义**: RAG 系统的检索模块需要确保检索到的段落不仅单独相关，还需集体构成完整的信息集合来正确回答复杂问题。
- **现有方法局限**: 现有 reranking 方法按段落的单独相关性打分后取 top-k，存在三个核心问题：(1) 忽视段落间信息互补性，可能检索冗余内容；(2) 无法保证多跳问题所需的信息覆盖完整性；(3) 需手动选择 top-k 参数。
- **研究动机**: RAG 系统与搜索引擎的信息需求本质不同——搜索引擎排序个体结果，RAG 需要一组段落来共同支撑生成。应从"排序"转向"集合选择"。
- **核心贡献**: (1) 提出基于信息需求的集合式段落选择范式；(2) 训练开源 SetR 模型实现高效选择；(3) 在多跳 RAG 基准上全面超越商用和开源重排序器。

## 方法详解

### 整体框架

SetR 工作流程：一阶段检索器（BM25/bge）返回 top-20 候选段落 → SetR 通过 CoT 推理分析查询信息需求 → 从候选段落中选择覆盖所有信息需求的最优子集（无需排序或固定 k 值）。

### 关键设计

1. **信息需求识别 (IRI)**: 结构化 CoT 推理策略，三步流程：(a) 枚举回答问题所需的关键信息需求；(b) 识别包含每个需求相关信息的候选段落；(c) 选择集体提供最全面覆盖的段落子集。不同于通用 CoT 推理，IRI 显式建模信息需求的完整覆盖。
2. **模型蒸馏**: GPT-4o 作为教师对 MS MARCO 40K 查询进行零样本集合选择标注，蒸馏到 Llama-3.1-8B-Instruct 学生模型。训练 5 epochs，有效 batch 512，学习率 5×10⁻⁶，AdamW 优化器。
3. **自适应段落数**: 模型自动决定选择多少段落——平均仅 2.63-3.41 个（vs 基线固定 5 个），减少噪声干扰的同时提升信息精度。

### 三种模型变体

- **SetR-Selection only**: 仅输出选择结果，无推理过程
- **SetR-CoT**: 使用通用 "Let's think step-by-step" 提示进行 CoT 推理
- **SetR-CoT & IRI**: 完整模型，含结构化信息需求识别的 CoT 推理

## 实验

### 主实验结果 (端到端问答)

使用 bge-large-en-v1.5 作为第一阶段检索器，Llama-3.1-8B 作生成器：

| 模型 | 段落数 | HotpotQA EM/F1 | 2Wiki EM/F1 | MuSiQue EM/F1 | MHRAG Acc |
|------|-------|---------------|------------|--------------|-----------|
| BM25 only | 5.00 | 30.07/30.97 | 31.17/25.22 | 7.44/10.78 | 41.82 |
| bge-reranker-large | 5.00 | 32.48/33.24 | 31.92/25.47 | 8.06/12.50 | 43.50 |
| RankGPT (gpt-4o) | 5.00 | 33.85/34.45 | 34.36/28.06 | 9.43/13.25 | 45.69 |
| **SetR-CoT & IRI** | **2.91** | **36.62/38.11** | **35.44/30.35** | **10.79/15.43** | **47.14** |

### 消融实验

| 变体 | HotpotQA F1 | MHRAG Acc | 说明 |
|------|------------|-----------|------|
| SetR-Selection only | 37.84 | 46.20 | 无推理亦优于全部 reranking 基线 |
| SetR-CoT | 38.20 | 45.26 | 通用 CoT 不如 IRI |
| **SetR-CoT & IRI** | **38.11** | **47.14** | IRI 的显式信息需求分析最优 |

### 信息覆盖度与鲁棒性分析 (MultiHopRAG)

| 指标 | Reranking 基线 | SetR |
|------|-------------|------|
| Hit@k | 48.87% | **69.90%** (+21.03%) |
| 信息覆盖率 | 19.33% | **36.49%** (+17.16%) |
| Precision@5 | 0.1799 (RankGPT最优) | **0.2268** (+26.1%) |

### 关键发现

1. **Less is More**: SetR 平均使用 2.91 个段落即全面超越使用 5 个段落的所有基线
2. **信息覆盖度提升显著**: Hit@k 和信息覆盖率分别提升 21% 和 17%，而传统 reranking 仅提升约 10%
3. **IRI vs 通用 CoT**: 信息需求识别的提升非来自简单 CoT 推理能力，而是任务特定的结构化分析
4. **Precision vs Rank 指标矛盾**: SetR 在 Precision 上大幅领先但 MRR/NDCG 略低，暴露了传统排序指标在多跳场景的不足
5. **增加段落数反而降低基线性能**: 更多段落引入更多噪声和矛盾信息，SetR 的精选策略避免了此问题

## 亮点

- 范式创新：首次将 RAG 检索从 "ranking" 转为 "set selection"，洞察深刻且概念简洁
- 高效实用：更少段落 + 更好效果 = 降低上下文窗口压力和推理成本
- 完全开源：模型权重、训练数据构建流程、代码全部公开
- 消融设计精细：三种变体清晰验证了 IRI 的独立贡献

## 局限性

- 蒸馏依赖 GPT-4o 的教师标注质量，可能引入教师偏差
- 训练数据基于 MS MARCO（英文通用领域），非英文或特定领域泛化性待验证
- 在 rank-based 指标（MRR/NDCG）上不如最佳 reranking 方法，说明两种范式可能互补
- 未在单跳问答或长上下文 LLM 上评估
- 可变大小输出可能给下游系统设计带来不确定性

## 相关工作

- **RAG 重排序**: RankGPT (Sun et al., 2024), RankZephyr (Pradeep et al., 2023b) 等 LLM-based listwise 重排序关注个体相关性
- **迭代检索**: Self-RAG (Asai et al., 2023), CoRAG (Wang et al., 2024) 多轮检索精化但计算开销大
- **查询分解**: IRCoT (Trivedi et al., 2023), RAPTOR (Sarthi et al., 2024) 通过分解复杂问题改善检索
- **上下文压缩**: Chirkova et al. (2025) 对检索上下文剪枝，与集合选择互补

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 新颖性 | 9 |
| 技术深度 | 7 |
| 实验充分性 | 8 |
| 写作质量 | 8 |
| 实用价值 | 9 |
| 总分 | 8.2 |

<!-- RELATED:START -->

## 相关论文

- [Parenting: Optimizing Knowledge Selection of Retrieval-Augmented Language Models with Parameter Decoupling and Tailored Tuning](parenting_optimizing_knowledge_selection_of_retrievalaugmented.md)
- [Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](../../NeurIPS2025/information_retrieval/cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)
- [Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)
- [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](towards_adaptive_memory-based_optimization_for_enhanced_retrieval-augmented_gene.md)

<!-- RELATED:END -->
