---
title: >-
  [论文解读] Towards a Principled Evaluation of Knowledge Editors
description: >-
  [ACL 2025][知识编辑][Model Editing] 本文系统性地揭示了知识编辑评估中不同评分方法（argmax、多选、生成匹配）和不同编辑批量大小会导致知识编辑器排名发生逆转的问题，并通过人工评估发现基于字符串匹配的评估方法存在假阳性倾向。 知识编辑（Knowledge Editing）近年来引起了广泛关注…
tags:
  - "ACL 2025"
  - "知识编辑"
  - "Model Editing"
  - "评估方法论"
  - "MEMIT"
  - "编辑批量大小"
---

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] The Mirage of Model Editing: Revisiting Evaluation in the Wild](the_mirage_of_model_editing_revisiting_evaluation_in_the_wild.md)
- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)
- [\[ACL 2025\] ScEdit: Script-based Assessment of Knowledge Editing](scedit_script-based_assessment_of_knowledge_editing.md)
- [\[ACL 2025\] SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)

</div>

<!-- RELATED:END -->
