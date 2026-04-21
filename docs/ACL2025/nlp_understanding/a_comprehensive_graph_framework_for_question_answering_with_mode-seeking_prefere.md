---
title: >-
  [论文解读] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment
description: >-
  [ACL 2025 Findings][NLP理解][RAG] 提出GraphMPA框架，通过构建基于通用相似度度量的层次化文档图实现全局文档理解，并引入mode-seeking偏好优化替代传统DPO实现更精准的人类偏好对齐，在6个QA数据集上全面超越现有RAG方法。
tags:
  - ACL 2025 Findings
  - NLP理解
  - RAG
  - 层次化文档图
  - 偏好对齐
  - Mode-Seeking
  - 社区检测
---

# A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment

**会议**: ACL 2025 Findings  
**arXiv**: [2506.17951](https://arxiv.org/abs/2506.17951)  
**代码**: https://github.com/tangquanwei/GraphMPA (有)  
**领域**: NLP理解 / 问答系统  
**关键词**: RAG, 层次化文档图, 偏好对齐, Mode-Seeking, 社区检测

## 一句话总结
提出GraphMPA框架，通过构建基于通用相似度度量的层次化文档图实现全局文档理解，并引入mode-seeking偏好优化替代传统DPO实现更精准的人类偏好对齐，在6个QA数据集上全面超越现有RAG方法。

## 研究背景与动机
检索增强生成（RAG）通过整合外部知识增强了LLM的问答能力，但面临两个核心挑战：

**全局理解不足**：传统RAG采用扁平化的chunk检索，只能获取局部片段信息，无法像人类那样从整体到局部理解文档结构。即使是GraphRAG等图方法，也存在图构建质量参差、层次化理解不充分的问题

**偏好对齐偏差**：标准的DPO（Direct Preference Optimization）本质上是"均值寻求"（mean-seeking）的，倾向于生成所有偏好的"折中"回复，而非找到真正最优的回复模式。这导致输出保守、泛泛而谈

核心idea：模拟人类认知过程——先通过社区检测和层次化摘要将文档组织成从细节到概念的多层次图结构（像人类阅读时自然形成的知识网络），再用mode-seeking偏好优化（寻找偏好分布的"众数"而非"均值"）确保输出对准人类最满意的回答类型。

## 方法详解

### 整体框架
GraphMPA包含两个核心组件：(1) 层次化文档图构建——将原始文档组织成多层图结构用于检索；(2) Mode-Seeking偏好对齐——训练模型生成更符合人类最强偏好的回答。输入为文档集合和问题，输出为对齐人类偏好的高质量答案。

### 关键设计
1. **层次化文档图构建（Hierarchical Document Graph）**:

    - 功能：将文档从扁平chunk列表转化为多层次的图结构
    - 核心思路：分四步实现：
        - (a) **文本分块与嵌入**：将文档切分为chunk并用embedding模型（如BGE-M3）转化为向量
        - (b) **构建图层**：以chunk为节点，基于embedding相似度建立边，形成文档相似度图
        - (c) **社区检测**：使用Leiden算法（或Louvain）在图上识别紧密连接的节点簇（社区），将语义相关的内容聚集在一起
        - (d) **摘要与递归**：对每个社区用LLM生成摘要，摘要作为新的更高层节点，重复上述过程构建多层图
    - 设计动机：模拟人类从"看段落→找关联→归纳主题→形成大纲"的渐进式理解过程，使检索能同时获取细节（底层）和概览（高层）信息

2. **Mode-Seeking偏好优化（MSPO）**:

    - 功能：将模型输出与人类偏好的"最集中"模式对齐，而非分散的平均值
    - 核心思路：不同于DPO最小化偏好对之间的差异（导致"折中"输出），MSPO通过概率匹配约束（probability-matching constraints）让模型学习偏好分布的众数（mode）。具体表现为训练后模型在正确答案上的log概率更高且更集中（中位log概率：MS约-5 vs DPO约-25 vs SFT约-150）
    - 设计动机：人类偏好并非均匀分布——人们通常有一种"最理想"的回答风格和质量水平。DPO的均值优化会稀释这种偏好信号，而mode-seeking直接瞄准偏好峰值

3. **层次化检索策略**:

    - 功能：在多层图上进行检索，综合不同粒度的信息
    - 核心思路：给定查询，在每层图上分别检索top-k相关节点，将底层具体信息和高层摘要信息汇总作为LLM的输入上下文
    - 设计动机：仅检索原始chunk（底层）缺乏全局视角，仅检索摘要（高层）缺乏细节，多层检索实现两者兼顾

### 损失函数 / 训练策略
Mode-Seeking偏好优化在标准DPO基础上修改了优化目标——从最小化选好vs选差的log概率差（KL散度主导，偏向均值）改为直接最大化在偏好众数处的概率密度。模型使用Qwen2.5-7B-Instruct、LLaMa-3.1-8B-Instruct和Mistral-8B作为基座。

## 实验关键数据

### 主实验（LLaMa 8B为骨干，准确率%）

| 数据集 | 指标 | GraphMPA | RAPTOR | LightGraphRAG | Basic RAG | 提升 |
|--------|------|----------|--------|---------------|-----------|------|
| QUALITY | Acc | **73.65** | 49.66 | 50.83 | 41.73 | +22.82 |
| PubMedQA | MIRAGE | **73.00** | 58.40 | 49.00 | 68.80 | +4.20 |
| MedQA | Acc | **66.54** | 53.10 | 45.18 | 57.34 | +9.20 |
| MedMcQA | Acc | **64.28** | 50.84 | 50.91 | 50.35 | +13.37 |
| QASPER | ROUGE-F1 | **0.3775** | 0.3657 | 0.3585 | 0.3599 | +0.012 |
| RiddleSense | Acc | 47.05 | 45.62 | 45.82 | **60.24** | -13.19 |

### 消融实验（QUALITY数据集）

| 配置 | 准确率 | 说明 |
|------|--------|------|
| Full GraphMPA | **47.05** | 完整框架 |
| w/ DPO (替换MS) | 46.06 | 用标准DPO替换mode-seeking |
| w/o Training | 46.65 | 不做偏好训练 |
| w/o Summarization | 41.73 | 去掉层次摘要 |
| w/o Retrieval | 32.10 | 去掉检索（纯LLM） |

### 超参数分析

| 参数 | 最优值 | 趋势 |
|------|--------|------|
| 图层数 | 2-3层 | 1层→2层提升大，4层轻微下降 |
| Top-K | 3-5 | 过少缺context，过多引入噪声 |

### 关键发现
- **检索是最关键的组件**：去掉检索后性能从47.05降到32.10（-31.7%），说明外部知识整合是核心
- **层次摘要贡献显著**：去掉摘要后从47.05降到41.73（-11.3%），全局理解对问答很重要
- **MS优于DPO但差距不大**：MS比DPO高约1个百分点，但log概率分布的集中度差异明显
- **跨模型一致性好**：在Qwen 7B和Mistral 8B上也观察到类似的性能提升模式

## 亮点与洞察
- **层次化文档图的设计直觉精准**：chunk→图→社区→摘要→高层图的递进构建方式，与人类阅读理解的认知过程高度吻合，是一种"认知对齐"的检索设计
- **Mode-Seeking vs Mean-Seeking的对比**具有理论洞察力：揭示了DPO的隐含假设（偏好分布是单峰对称的）在实际中不成立的事实
- **通过Leiden/Louvain社区检测实现的文档聚类**可直接迁移到其他需要文档结构化理解的任务

## 局限与展望
- RiddleSense上GraphMPA不如Basic RAG（47.05 vs 60.24），可能因为常识推理任务不依赖深层文档理解
- 消融实验中MS vs DPO的差异较小（~1%），mode-seeking的实际优势可能因数据集而异
- 图构建依赖LLM生成摘要，引入了额外的计算开销和摘要质量波动
- 仅在开源7-8B级别模型上评估，在更大模型上的表现和收益未知
- 缺少与最新的GraphRAG框架（如Microsoft GraphRAG）的直接对比

## 相关工作与启发
- **vs RAPTOR**: RAPTOR也采用层次化chunk摘要思路，但使用聚类而非图社区检测，且不含偏好对齐；GraphMPA在QUALITY上高出24%
- **vs LightGraphRAG**: LightGraphRAG是轻量级图RAG方案，GraphMPA通过更完整的图构建和偏好优化在大多数数据集上超越
- **vs Reward-RAG**: Reward-RAG使用奖励模型，GraphMPA用mode-seeking偏好优化，两者思路相近但优化目标不同；在PubMedQA上GraphMPA(73.00)超过Reward-RAG(69.20)

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次图构建+mode-seeking偏好对齐的组合较新颖，但单独看各组件都有先例
- 实验充分度: ⭐⭐⭐⭐ 6个数据集3个模型比较全面，消融和超参分析充分
- 写作质量: ⭐⭐⭐ ACL Findings水平，方法描述较清晰
- 价值: ⭐⭐⭐⭐ 为RAG系统提供了结构化理解和偏好对齐的完整解决方案，实用性强

<!-- RELATED:START -->

## 相关论文

- [iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)
- [Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)

<!-- RELATED:END -->
