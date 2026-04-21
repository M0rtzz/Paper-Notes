---
title: >-
  [论文解读] Towards a Principled Evaluation of Knowledge Editors
description: >-
  [ACL 2025][知识编辑] 本文系统性地揭示了知识编辑评估中不同评分方法（argmax、多选、生成匹配）和不同编辑批量大小会导致知识编辑器排名发生逆转的问题，并通过人工评估发现基于字符串匹配的评估方法存在假阳性倾向。
tags:
  - ACL 2025
  - 知识编辑
  - Model Editing
  - 评估方法论
  - MEMIT
  - 编辑批量大小
---

# Towards a Principled Evaluation of Knowledge Editors

**会议**: ACL 2025  
**arXiv**: [2507.05937](https://arxiv.org/abs/2507.05937)  
**代码**: 有 (开源评估框架)  
**领域**: NLP / 知识编辑  
**关键词**: knowledge editing, Model Editing, Evaluation Methodology, Edit Batch Size, LM Evaluation Harness

## 一句话总结

本文系统性地揭示了知识编辑评估中评估方法、评估指标和编辑批量大小的选择会显著影响编辑器排名，并通过与LM Evaluation Harness集成来评估编辑对模型整体能力的副作用。

## 研究背景与动机

知识编辑（Knowledge Editing）是近年来模型编辑中最受关注的方向，目标是将新知识（通常以三元组形式）注入预训练语言模型中。现有的四个主流评估数据集（zsRE、CounterFact、MQuAKE、RippleEdits）分别使用了不同的评估方法来衡量编辑器的成功率，但这些方法之间的差异和鲁棒性尚未被充分研究。更重要的是，编辑器对模型整体语言理解能力的破坏性影响一直是一个盲区。

具体而言，现有评估方法存在两大研究空白：

**评估方法不统一**：zsRE用逐token的greedy decoding判断，CounterFact用序列对数似然的排名，MQuAKE和RippleEdits用生成文本中的字符串匹配。不同方法是否会导致不同的编辑器排名？

**批量编辑的副作用**：随着编辑数量的增加，编辑器如何影响模型在通用任务上的表现？这需要一个系统性的研究框架。

## 方法详解

### 整体框架

作者构建了一个统一的知识编辑评估框架，将四个现有数据集（zsRE、CounterFact、MQuAKE、RippleEdits）整合在一起，并与LM Evaluation Harness集成，使得可以在编辑后的模型上同时运行知识编辑评估和通用语言理解任务。

### 关键设计

1. **三种评估方法的对比（Argmax/MC/Generate）**：

    - **Argmax**：逐token检查greedy decoding是否产生目标字符串
    - **MC（多选）**：比较原始答案和新答案的序列对数似然
    - **Generate**：生成固定长度的文本，检查目标字符串是否出现在生成文本中
    - 设计动机：揭示不同方法对编辑器排名的影响

2. **四种编辑器的选取**：

    - **MEMIT**：通过因果追踪计算显式参数更新，专为大批量编辑设计
    - **LoRA**：基于参数高效微调
    - **In-context**：将编辑事实以自然语言模板形式添加到输入前
    - **Context-retriever**：结合RAG系统，用Contriever检索最相关的4个编辑

3. **批量编辑大小的系统性实验**：在1, 16, 64, 512, 2048五个批量大小上进行评估，同时在lambada、hellaswag等控制任务上评估副作用

### 损失函数 / 训练策略

- LoRA使用rank=8, alpha=32, 20个训练epoch
- GPT2-XL使用学习率5e-3，GPT-J使用1e-3
- 从每个数据集随机选择2048个样本进行实验

## 实验关键数据

### 主实验（表格）

| 数据集 | 评估方法 | CR | IC | MEMIT | LoRA | NoEdit |
|--------|---------|------|------|-------|------|--------|
| zsRE | argmax | 0.735 | 0.764 | 0.727 | 0.756 | 0.278 |
| zsRE | gen | 0.619 | 0.656 | 0.629 | 0.653 | 0.066 |
| CF | MC | 0.800 | 0.794 | **0.866** | 0.688 | 0.614 |
| CF | gen | 0.505 | 0.511 | 0.462 | 0.442 | 0.200 |
| MQuAKE | gen | 0.213 | 0.198 | 0.153 | 0.133 | 0.050 |
| RipEd | gen | 0.500 | 0.478 | 0.475 | 0.537 | 0.543 |

GPT-J模型，edit batch size=16。关键发现：MEMIT在CounterFact的MC评估中大幅领先，但在generate评估中反而最差。

### 消融实验（表格）

| 数据集 | Mistral-7B准确率 | Qwen-32B准确率 | Exact Match准确率 |
|--------|-------------------|-----------------|-------------------|
| zsRE | 0.625 | 0.903 | 0.882 |
| CF | 0.647 | 0.955 | 0.917 |
| MQuAKE | 0.654 | 0.897 | 0.897 |
| RipEd | 0.757 | 0.903 | 0.896 |

LLM-as-a-judge（Qwen-32B）在人工评估真值上略优于精确匹配。

### 关键发现

1. **评估方法影响排名**：MEMIT在CounterFact的MC评估中表现最好（0.866），但在argmax和generate评估中均表现最差，说明MC评估可能偏向MEMIT
2. **生成文本长度的影响**：超过30个token后，假阳性率开始显著增加；context-retriever因生成更多样化的文本而受益于更长的生成长度
3. **批量大小与性能**：in-context编辑器在批量>64时因超出上下文窗口而性能骤降；MEMIT对批量大小最鲁棒
4. **通用任务的副作用**：LoRA是最具破坏性的编辑器（perplexity飙升至数百万）；MEMIT最不具破坏性；context-retriever在大批量时反而改善通用任务表现（因为检索到的编辑更不具干扰性）

## 亮点与洞察

- 首次系统性地将知识编辑评估与通用语言理解评估（LM Evaluation Harness）整合，填补了副作用评估的空白
- 揭示了一个反直觉的现象：context-retriever在大批量编辑时通用任务性能反而提升
- 手动评估200个样本的假阳性/假阴性率，为字符串匹配方法的局限性提供了实证证据
- LLM-as-a-judge作为替代评估方法的探索具有实用价值

## 局限与展望

- 仅在GPT-J（6B）和GPT2-XL（1.5B）两个较小模型上实验，缺乏对更大模型的验证
- 编辑器选取有限（4种），未涵盖更多新兴方法
- 未测试指令微调模型，而这在实际应用中更为常见
- LLM-as-a-judge的探索样本量太小（200个），需要更大规模验证

## 相关工作与启发

- 与MEMIT、ROME等参数更新方法相比，in-context方法在小批量编辑时更具竞争力
- 与RAG系统有概念上的联系：context-retriever本质上是一种知识编辑与RAG的结合
- 启发：未来知识编辑的评估应在报告中同时包含多种评估方法的结果，并评估通用任务上的副作用

## 评分

- **新颖性**: ⭐⭐⭐ 评估方法论层面的工作，不是新方法但系统性分析有价值
- **实验充分度**: ⭐⭐⭐⭐ 四个数据集、四种编辑器、三种评估方法、五种批量大小，加上手动评估和LLM-as-a-judge
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，实验设计合理，图表丰富
- **价值**: ⭐⭐⭐⭐ 对知识编辑领域的评估规范化有重要推动作用
# Towards a Principled Evaluation of Knowledge Editors

**会议**: ACL 2025  
**arXiv**: [2507.05937](https://arxiv.org/abs/2507.05937)  
**代码**: 有 (开源评估框架)  
**领域**: NLP / 知识编辑  
**关键词**: 知识编辑, Model Editing, 评估方法论, MEMIT, 编辑批量大小

## 一句话总结

本文系统性地揭示了知识编辑评估中不同评分方法（argmax、多选、生成匹配）和不同编辑批量大小会导致知识编辑器排名发生逆转的问题，并通过人工评估发现基于字符串匹配的评估方法存在假阳性倾向。

## 研究背景与动机

知识编辑（Knowledge Editing）近年来引起了广泛关注，旨在通过局部、有针对性的修改来更新预训练语言模型中的知识，而无需全量重训练。目前主流评估数据集包括 zsRE、CounterFact、MQuAKE 和 RippleEdits，但它们使用了截然不同的评分方法：

- **zsRE** 使用 argmax 逐 token 检查是否为贪心解码结果
- **CounterFact** 使用多选题方式（对比序列对数似然）
- **MQuAKE/RippleEdits** 使用生成文本中的字符串匹配

这些不同方法是否会导致编辑器的排名不一致？字符串匹配的可靠性如何？编辑批量大小对模型整体能力的破坏程度如何？这些关键问题此前未被充分探讨。

## 方法详解

### 整体框架

本文构建了一个统一的评估框架，将四个知识编辑数据集整合在一起，并集成了 LM Evaluation Harness，能够在编辑后的模型上同时运行知识编辑评估和通用语言理解任务。

### 关键设计

1. **三种评分方法的对比**：
    - **Argmax**：逐 token 检查目标字符串是否为最高概率预测，计算准确率
    - **MC（多选）**：比较原始目标和编辑后目标的序列对数似然
    - **Generate（生成匹配）**：生成固定长度文本后检查目标字符串是否出现
    - 设计动机：不同方法可能隐含地偏向某些特定编辑器

2. **四种编辑器的选择**：
    - **MEMIT**：通过因果追踪显式计算参数更新，专为批量编辑设计
    - **LoRA**：参数高效微调方法
    - **In-Context**：将编辑事实以自然语言形式拼接到输入中
    - **Context-Retriever**：结合 RAG 系统检索最相关的 4-NN 编辑

3. **生成长度的影响分析**：生成文本越长，假阳性率越高；通过人工标注 200 个样本验证匹配算法的可靠性

4. **编辑批量大小实验**：从 1 到 2048 逐步增大批量，观察知识编辑性能和通用能力的变化趋势

### 损失函数 / 训练策略

- LoRA 超参数：rank=8, alpha=32, 20 epochs，GPT-2-XL 学习率 5e-3，GPT-J 学习率 1e-3
- MEMIT 使用原论文发布的超参数
- 每个数据集随机采样 2048 条用于所有实验

## 实验关键数据

### 主实验（不同评分方法下的准确率对比，GPT-J）

| 数据集 | 方法 | Context-Retriever | In-Context | MEMIT | LoRA | NoEdit |
|--------|------|-------------------|------------|-------|------|--------|
| zsRE | argmax | 0.735 | 0.764 | 0.727 | 0.756 | 0.278 |
| zsRE | gen | 0.619 | 0.656 | 0.629 | 0.653 | 0.066 |
| CF | argmax | 0.365 | 0.391 | 0.312 | 0.356 | 0.095 |
| CF | MC | 0.800 | 0.794 | **0.866** | 0.688 | 0.614 |
| CF | gen | 0.505 | 0.511 | 0.462 | 0.442 | 0.200 |
| MQuAKE | gen | 0.213 | 0.198 | 0.153 | 0.133 | 0.050 |

### 消融实验（LLM-as-Judge 与精确匹配对比）

| 数据集 | Mistral-7B | Qwen-32B | Exact Match |
|--------|-----------|----------|-------------|
| zsRE | 0.625 | 0.903 | 0.882 |
| CF | 0.647 | 0.955 | 0.917 |
| MQuAKE | 0.654 | 0.897 | 0.897 |
| RipEd | 0.757 | 0.903 | 0.896 |

### 关键发现

1. **MEMIT 在 CounterFact 上的"优势"是评分方法偏好的结果**：使用 MC 方法时 MEMIT 得分 0.866 远超其他编辑器，但用 argmax 和 generate 方法时反而最差
2. **字符串匹配存在假阳性问题**：随着生成长度增加到 30 token 以上，假阳性率显著上升
3. **Context-Retriever 的假阳性率更高**：可能因为生成文本更多样
4. **编辑批量增大时**：In-Context 编辑器由于上下文窗口限制在 batch>64 时急剧下降，MEMIT 相对更鲁棒
5. **对通用能力的破坏**：LoRA 最具破坏性（perplexity 飙升至百万级别），MEMIT 最温和，Context-Retriever 在大批量时反而恢复（因为检索到无害编辑）

## 亮点与洞察

- 首次系统性地揭示评估方法选择对知识编辑器排名的影响，这对该领域的公平比较具有重要意义
- 将 LM Evaluation Harness 集成进知识编辑评估框架，弥补了"编辑副作用"研究的空白
- Context-Retriever 在大批量时的"自我恢复"现象很有意思——编辑越多，检索到的编辑越无害
- LLM-as-Judge 作为替代评估方法展示了初步可行性

## 局限与展望

- 仅在两个较小模型（GPT-J 6B, GPT2-XL 1.5B）上实验，需扩展到更大模型
- 未涉及指令微调模型
- LoRA 超参数仅针对 batch=16 优化，对其他批量不公平
- LLM-as-Judge 的样本量太小（200 条），需要更大规模验证
- 缺少对更多编辑方法的覆盖

## 相关工作与启发

- 与 MEMIT (Meng et al., 2023b) 的对比揭示了评分方法偏好的存在
- MQuAKE (Zhong et al., 2024) 关注多跳推理能力，本文发现 in-context 编辑在此任务上更具优势
- RippleEdits (Cohen et al., 2023) 的遗忘性查询设计使得未编辑模型在该任务上有天然优势
- 启发：未来知识编辑的评估应该统一使用多种评分方法，并报告编辑对通用能力的影响

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统性分析评估方法对编辑器排名的影响，视角独特
- **实验充分度**: ⭐⭐⭐⭐ — 四个数据集、四种编辑器、多种批量大小，加人工评估，但模型规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富，论证逻辑严密
- **价值**: ⭐⭐⭐⭐ — 对知识编辑领域的评估规范化具有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [ScEdit: Script-based Assessment of Knowledge Editing](scedit_script-based_assessment_of_knowledge_editing.md)
- [SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)
- [Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)
- [CompKe: Complex Question Answering under Knowledge Editing](compke_complex_question_answering_under_knowledge_editing.md)

<!-- RELATED:END -->
