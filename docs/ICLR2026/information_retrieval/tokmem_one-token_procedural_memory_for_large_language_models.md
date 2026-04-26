---
title: >-
  [论文解读] TokMem: One-Token Procedural Memory for Large Language Models
description: >-
  [ICLR 2026][程序性记忆] 提出 TokMem，将可复用的任务程序编译为单个可训练记忆 token，既作为程序索引又作为生成控制信号，无需长 prompt 即可高效调用 1000+ 任务程序，且支持无遗忘的持续扩展。
tags:
  - ICLR 2026
  - 程序性记忆
  - 记忆 token
  - 持续学习
  - 上下文压缩
  - 工具调用
---

# TokMem: One-Token Procedural Memory for Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2510.00444](https://arxiv.org/abs/2510.00444)  
**代码**: [https://github.com/MANGA-UOFA/TokMem](https://github.com/MANGA-UOFA/TokMem)  
**领域**: 模型压缩  
**关键词**: 程序性记忆, 记忆 token, 持续学习, 上下文压缩, 工具调用

## 一句话总结

提出 TokMem，将可复用的任务程序编译为单个可训练记忆 token，既作为程序索引又作为生成控制信号，无需长 prompt 即可高效调用 1000+ 任务程序，且支持无遗忘的持续扩展。

## 研究背景与动机

- **长 prompt 的效率瓶颈**: 现代 LLM 依赖 prompt 控制行为，但长 prompt 构造成本高、自注意力平方级计算开销大、占用上下文窗口导致截断
- **检索增强的局限**: RAG 等方法虽外部化了 prompt，但检索内容仍以文本形式占用上下文窗口，且常用程序每次都需重新解读
- **认知科学启发**: 人类的程序性记忆（如骑自行车）通过练习编译为高效技能，无需每次重新读取陈述性知识
- **核心思想**: 将频繁使用的任务程序"压缩"存储到专用记忆 token 中，实现常量开销的程序调用

## 方法详解

### 整体框架

TokMem 在 LLM 词表中添加 $l$ 个特殊 token 作为记忆库：

$$M = \begin{bmatrix} \bm{m}_1^\top \\ \vdots \\ \bm{m}_l^\top \end{bmatrix} \in \mathbb{R}^{l \times d}$$

每个 $\bm{m}_i$ 是可训练向量，代表一个唯一程序，无直接文本形式。

### 关键设计 1: 记忆 Token 训练

训练序列包含程序-响应对，记忆 token 与文本 token 交替排列：

$$\bm{a} = (q_1, \ldots, q_k, \underbrace{a_{m_i}, a_{r_{i1}}, a_{r_{i2}}, \ldots}_{\text{程序-响应对}}, \underbrace{a_{m_j}, a_{r_{j1}}, \ldots}_{\text{程序-响应对}}, \ldots)$$

使用标准 next-token prediction 损失训练，仅更新记忆嵌入，骨干 LLM 和原始 token 嵌入完全冻结：

$$\mathcal{L}(\bm{a}; M) = -\sum_{i>k} \log \Pr(a_i \mid \bm{a}_{<i}; M)$$

### 关键设计 2: 推理时记忆路由

给定查询 $q$，模型从最终隐状态 $h_k$ 预测记忆 token 分布：

$$P(a_{m_i} \mid q) \propto \exp(\text{logit}(m_i \mid h_k))$$

- 选择概率最高的记忆 token 附加到查询后，自回归生成响应
- 支持多步程序链接：生成一段响应后预测下一个记忆 token
- 若无匹配程序，所有记忆 logit 保持低值，自动回退到普通文本生成

### 关键设计 3: 重正则化（Renormalization）

持续学习中新嵌入易出现 norm 膨胀，压制旧记忆。解决方案：

$$\bm{m}_i \leftarrow \bm{m}_i \cdot \frac{\bar{n}_I}{\|\bm{m}_i\|_2 + \varepsilon}, \quad i \in A$$

其中 $\bar{n}_I = \text{mean}_{j \in I} \|\bm{m}_j\|_2$ 为已有记忆的典型范数。保持新嵌入方向不变，仅对齐其幅度，计算开销 $O(|A| d)$ 可忽略。

### 参数隔离特性

- 每个程序的知识完全存储在独立 token 嵌入中
- 新程序可持续添加而不干扰现有程序
- 天然支持持续学习，无灾难性遗忘

## 实验

### 原子记忆召回：Super-Natural Instructions（ROUGE-L）

| 模型 | 方法 | 10任务 | 200任务 | 1000任务 | 平均 |
|------|------|--------|---------|---------|------|
| Qwen 0.5B | RAG | 50.4 | 38.8 | 34.7 | 40.7 |
| Qwen 0.5B | Fine-Tuning | 52.4 | 40.6 | 43.2 | 45.2 |
| Qwen 0.5B | Replay Memory | 52.4 | 47.2 | 46.7 | 48.7 |
| Qwen 0.5B | **TokMem** | **52.8** | **49.3** | **50.0** | **50.7** |
| Llama 3.2 3B | RAG | 60.0 | 45.8 | 39.9 | 47.3 |
| Llama 3.2 3B | **TokMem** | **68.0** | **61.2** | **61.5** | **62.9** |
| Llama 3.1 8B | **TokMem** | **75.4** | **65.1** | **64.8** | **67.0** |

### 记忆路由准确率

| 方法 | 10任务 | 200任务 | 1000任务 |
|------|--------|---------|---------|
| Sentence-BERT (RAG) | 99.6 | 88.7 | 79.7 |
| TokMem (Qwen 0.5B) | 99.4 | 97.4 | 94.7 |
| TokMem (Llama 8B) | 99.8 | 98.9 | **97.5** |

### 组合记忆召回：工具调用（APIGen）

| 模型 | 方法 | 参数量 | 工具选择 Avg | 参数 F1 Avg |
|------|------|--------|-------------|-----------|
| Llama 1B | ICL | - | 16.4 | 0.4 |
| Llama 1B | RAG | - | 16.9 | 2.7 |
| Llama 1B | Fine-Tuning | 0.85M | 9.0 | 68.6 |
| Llama 1B | **TokMem** | **0.10M** | **86.2** | 68.9 |

### 关键发现

- TokMem 在 1000 任务时仍保持 94.7% 路由准确率（最小模型），远超 RAG 的 79.7%
- 训练效率极高：仅 10 个样本即可超越 RAG 的 500 样本性能
- 可训练参数量远少于 LoRA 微调（0.10M vs 0.85M），但性能相当或更优
- 持续学习中无灾难性遗忘，性能随任务增加仅缓慢下降
- 支持多步程序链接，在工具调用场景中可依次召回 parse→search→format

## 亮点

- **概念优雅**: 将程序性知识压缩为单个 token，认知科学和工程实现完美结合
- **参数隔离**: 每个程序独立存储，天然无遗忘，完美适配持续学习场景
- **极致高效**: 常量大小的记忆开销，消除了长 prompt 的平方级计算成本
- **路由准确率惊人**: 即使在 0.5B 模型上管理 1000 个任务也保持 94.7% 准确率

## 局限性

- 程序需要预先定义和训练，不支持零样本程序创建
- 单个 token 的信息容量有限，复杂程序可能无法完全编码
- 嵌入空间的容量上限未知——当程序数量极大时路由可能退化
- 重正则化是后处理操作，可能无法完全解决持续学习中的分布漂移
- 仅在 QA 和工具调用场景验证，对创意生成、长文本等任务的适用性未探索

## 相关工作

- **上下文工程**: CoT、RAG、MemGPT——均以文本扩展 prompt，占用上下文窗口
- **参数高效微调**: LoRA、Adapter——更新骨干参数，可能遗忘
- **软提示**: Prompt tuning、Prefix tuning——训练连续提示向量，但通常不建模为独立记忆单元
- **认知科学**: ACT-R 理论中的程序性记忆——技能通过练习编译为高效模块
- **TokMem**: 首次将 NLP 中的 token 化思想应用于程序性记忆管理

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★★ |
| 理论深度 | ★★★☆☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★☆ |

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](../../ACL2025/information_retrieval/enhancing_lexicon-based_text_embeddings_with_large_language_models.md)
- [\[ICLR 2026\] Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets](flow_of_spans_generalizing_language_models_to_dynamic_span-vocabulary_via_gflown.md)

<!-- RELATED:END -->
