---
title: >-
  [论文解读] DocMEdit: Towards Document-Level Model Editing
description: >-
  [ACL 2025][模型编辑] 首次提出文档级模型编辑任务，构建包含 37,990 条数据、105,652 个编辑事实的 DocMEdit 基准，揭示现有编辑方法在长上下文、多事实并行编辑场景下的严重不足。
tags:
  - ACL 2025
  - 模型编辑
  - 文档级任务
  - 知识更新
  - LLM
  - 知识图谱
---

# DocMEdit: Towards Document-Level Model Editing

**会议**: ACL 2025  
**arXiv**: [2505.19572](https://arxiv.org/abs/2505.19572)  
**代码**: [有](https://github.com/BITHLP/DocMEdit)  
**领域**: NLP / 知识编辑  
**关键词**: 模型编辑, 文档级任务, 知识更新, LLM, 知识图谱

## 一句话总结

首次提出文档级模型编辑任务，构建包含 37,990 条数据、105,652 个编辑事实的 DocMEdit 基准，揭示现有编辑方法在长上下文、多事实并行编辑场景下的严重不足。

## 研究背景与动机

模型编辑旨在以最小代价纠正 LLM 中的错误或过时知识。现有数据集（ZsRE、CounterFact、MQuAKE）的核心问题在于：

**输出粒度太小**：仅要求模型输出短语或句子，与真实场景（生成传记、更新 Wikipedia 文档、长链推理）严重脱节

**缺乏外推性**：答案可直接从编辑事实推导，不需要模型结合已有知识

**上下文长度不足**：现有数据集平均目标长度仅 6-131 词

这导致现有编辑方法的实际可用性存疑。作者对比了现有基准：

| 基准 | 文档级 | 外推性 | 多编辑 | 局部性 | 平均目标长度 |
|------|--------|--------|--------|--------|-------------|
| ZsRE | ✗ | ✗ | ✗ | ✔ | 12.12 |
| CounterFact | ✗ | ✗ | ✗ | ✔ | 6.65 |
| MQuAKE | ✗ | ✗ | ✔ | ✗ | 10.94 |
| **DocMEdit** | **✔** | **✔** | **✔** | **✔** | **867.62** |

DocMEdit 的平均目标长度是现有最长基准的 6.6 倍。

## 方法详解

### 整体框架

DocMEdit 的构建分三步：文档变更计算、事实收集、知识图谱提取。

### 关键设计

1. **文档变更计算（Document Change Computation）**：

    - 收集 Wikipedia 两个时间戳（20231101 和 20241101）的 dump
    - 提取每篇文档的 INTRODUCTION 部分，作为更新前后的 y 和 y'
    - 过滤仅含样式修改的更新，保留包含至少一个新实体的有意义更新
    - 设计动机：Wikipedia 大部分更新是样式而非事实变更

2. **事实收集（Facts Collection）**：

    - 对每个句子，若其中提及的实体是文档更新中新引入的，则该句子被视为支撑该实体更新的事实
    - 事实直接从非结构化 Wikipedia 数据中提取（非三元组），更贴近真实编辑场景
    - 每条数据平均包含 2.78 个编辑事实

3. **知识图谱提取（Knowledge Graph Extraction）**：

    - 从源文档、目标文档和支撑事实中分别提取知识图谱
    - 约束三元组中的关系必须是 Wikidata 中已有的关系
    - 提取了 568,652 个实体、4,804 种关系、1,411,057 个三元组
    - 方便 RAG 类方法（IKE、SKEME）进行实验

4. **评估指标设计**：

    - **准确性**：Document-ROUGE (DR)、Document-Entity (DE)、Edit-ROUGE (ER)、Edit-Entity (EE)
    - **局部性**：ROUGE Side Effect (RSE)、Entity Side Effect (ESE) — 衡量未编辑部分的保留情况
    - **质量**：人工评估语义连贯性 (SC)，3 级打分
    - **效率**：时间消耗 (Ti)、内存需求 (Me)

### 问题形式化

编辑模型 M 为 M'，使得 M'(x) = y'，其中 y' 包含原始句子 + 由编辑事实支撑的新句子，且与事实无关的句子保持不变。

## 实验关键数据

### 主实验（表格）

| 模型 | 方法 | DR↑ | DE↑ | ER↑ | EE↑ | RSE↑ | ESE↑ | SC↑ |
|------|------|-----|-----|-----|-----|------|------|-----|
| Llama2 | w/o Edit | 26.11 | 18.97 | 15.77 | 0.50 | 53.91 | 55.37 | 1.05 |
| Llama2 | FT | 24.78 | 17.95 | 14.65 | 7.17 | 53.76 | 39.22 | 0.60 |
| Llama2 | MEMIT | 19.63 | 9.62 | 15.16 | 2.50 | 40.54 | 34.86 | 0.62 |
| Llama2 | IKE | 19.79 | 26.30 | 22.77 | 12.20 | 43.27 | 35.80 | 1.03 |
| Llama2 | SKEME | 21.08 | **29.34** | **25.75** | **23.92** | **47.31** | **49.22** | 1.00 |
| DeepSeek | SKEME | **37.71** | **37.05** | **29.64** | **54.49** | 59.04 | 88.55 | 1.99 |

### 分析实验 — 上下文长度影响（总结表格）

| 上下文长度 | 短(0-512) | 中(512-1024) | 长(1024+) |
|-----------|-----------|-------------|-----------|
| 基础模型 | 可处理部分 | 困难 | 基本失败 |
| FT/MEMIT | 降低性能 | 失败 | 完全失败 |
| IKE/SKEME | 进一步提升 | 有效 | 几乎失败 |

### 关键发现

- **所有方法表现均不理想**：未编辑模型的高 DR 值源于幻觉而非真正的知识更新
- **参数修改方法（FT、MEMIT）**副作用严重：显著降低 LLM 的生成质量（SC 从 1.05 降至 0.60）
- **RAG 方法整体优于参数修改方法**：SKEME 基于实体检索更鲁棒，IKE 向量检索在长事实上性能快速下降
- **所有模型的副作用严重**：ESE 均低于 60，意味着丢失超过 40% 的实体信息
- **事实数增多时 RAG 方法快速退化**：事实数 ≥5 时，FT 反而优于 RAG 方法
- **错误分析**：78.4% 为幻觉，8.6% 忽略事实更新，7.7% 意外风格变化，5.3% 误解事实

## 亮点与洞察

- **问题定义有价值**：首次将模型编辑扩展到文档级，填补了研究空白
- 指标设计兼顾准确性和副作用，DR/DE + RSE/ESE 的组合比单一指标更有说服力
- 事实更新实验（RQ2b）揭示了连续编辑的退化问题：参数修改方法因内部参数偏离初始状态而崩溃
- 来自 Wikipedia 真实更新的数据构建方式使基准更贴近实际应用

## 局限与展望

- 输入输出较长，对 LLM 上下文窗口和计算资源要求高
- 仅使用 Wikipedia INTRODUCTION 部分，更长的完整文档尚未测试
- 现有方法均表现不佳，缺少有效的解决方案提案
- 可探索的方向：任务分解、prompt 结构和事实位置调整、浅层+深层神经元同时关注、内外部知识冲突管理

## 相关工作与启发

- 与 FAME（多任务编辑）互补：FAME 关注多跳推理，DocMEdit 关注文档级长输出
- FRUIT 的可信更新文本生成思路可延伸到模型编辑场景
- 文档级 NLP 任务（翻译、关系提取、QA）中的挑战在模型编辑中同样存在且更严峻
- 为后续研究提供了清晰的 benchmark 和评估协议

## 评分

- **新颖性**: 8/10 — 任务定义新颖，benchmark 设计合理
- **实验充分度**: 8/10 — 多模型多方法 + 四个 RQ 深入分析
- **写作质量**: 7/10 — 内容详实但公式表达略繁琐
- **价值**: 8/10 — 为模型编辑领域提供了重要的 benchmark 和新视角

<!-- RELATED:START -->

## 相关论文

- [Neuron-Level Sequential Editing for Large Language Models](neuron-level_sequential_editing_for_large_language_models.md)
- [MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [CompKe: Complex Question Answering under Knowledge Editing](compke_complex_question_answering_under_knowledge_editing.md)
- [Mitigating Negative Interference in Multilingual Sequential Knowledge Editing through Null-Space Constraints](mitigating_negative_interference_in_multilingual_sequential_knowledge_editing_th.md)

<!-- RELATED:END -->
