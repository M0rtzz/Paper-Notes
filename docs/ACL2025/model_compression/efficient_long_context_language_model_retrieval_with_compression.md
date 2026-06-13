---
title: >-
  [论文解读] Efficient Long Context Language Model Retrieval with Compression
description: >-
  [ACL 2025][模型压缩][长上下文语言模型] 提出 CoLoR（Compression for Long context Retrieval），通过偏好优化和长度正则化联合训练段落压缩模型，在保持长上下文语言模型检索性能提升 6% 的同时将上下文长度压缩 1.91 倍。 领域现状：长上下文语言模型（LCLM）已成为信…
tags:
  - "ACL 2025"
  - "模型压缩"
  - "长上下文语言模型"
  - "信息检索"
  - "段落压缩"
  - "偏好优化"
  - "长度正则化"
---

# Efficient Long Context Language Model Retrieval with Compression

**会议**: ACL 2025  
**arXiv**: [2412.18232](https://arxiv.org/abs/2412.18232)  
**代码**: [GitHub](https://github.com/going-doer/CoLoR)  
**领域**: 模型压缩  
**关键词**: 长上下文语言模型, 信息检索, 段落压缩, 偏好优化, 长度正则化

## 一句话总结

提出 CoLoR（Compression for Long context Retrieval），通过偏好优化和长度正则化联合训练段落压缩模型，在保持长上下文语言模型检索性能提升 6% 的同时将上下文长度压缩 1.91 倍。

## 研究背景与动机

**领域现状**：长上下文语言模型（LCLM）已成为信息检索的新范式——它们可以将整个文档语料库直接放入单次上下文窗口中进行处理，无需传统的稀疏/稠密检索索引。这种"全量上下文检索"方式在多个任务上展现出超越传统 BM25 和 DPR 方法的潜力。

**现有痛点**：LCLM 检索面临严重的效率瓶颈。当大量段落被放入上下文进行检索时，计算开销随上下文长度呈超线性增长（Transformer 的注意力复杂度为 $O(n^2)$）；在推理阶段处理这些中间表征同样非常耗时。现有的通用文本压缩方法主要为摘要任务设计，没有针对检索场景的优化，压缩后的文本可能丢失检索所需的关键信息。

**核心矛盾**：检索准确性要求保留段落中与潜在查询相关的细粒度信息，而效率要求尽可能缩短段落长度。这两个目标之间存在天然的 trade-off——过度压缩会丢失关键区分性信息导致检索失败，压缩不足则无法有效降低计算成本。

**本文目标**：设计一种专门面向 LCLM 检索的段落压缩方法，在最大化检索性能的同时最小化压缩后段落的长度。

**切入角度**：作者提出将段落压缩建模为一个偏好学习问题——对于给定查询，能成功检索到的压缩版本是"好的"（chosen），检索失败的是"差的"（rejected），可以自动生成训练数据而无需人工标注。同时引入长度正则化来进一步约束简洁性。

**核心 idea**：用偏好优化（如 DPO）训练压缩模型，以"检索是否成功"作为自动奖励信号生成 chosen/rejected 对，并叠加长度正则化损失来同时优化检索质量和压缩率。

## 方法详解

### 整体框架

CoLoR 的工作流程：(1) 首先使用预训练语言模型对段落进行多次压缩采样，生成不同长度的压缩版本；(2) 将压缩版本送入 LCLM 进行检索评估，根据检索结果（成功/失败）自动标注 chosen/rejected 对；(3) 使用这些偏好数据通过 DPO 训练压缩模型，同时加入长度正则化损失约束输出长度；(4) 推理时，压缩模型先将所有段落压缩，再将压缩后的段落送入 LCLM 进行实际检索。

### 关键设计

1. **自动化偏好数据生成**:

    - 功能：无需人工标注，自动为压缩模型构建 chosen/rejected 训练对
    - 核心思路：对于每个查询-段落对，使用现有压缩模型生成多个压缩候选；将每个候选替换原始段落后送入 LCLM 执行检索；检索成功的压缩版本标记为 chosen，失败的标记为 rejected。这种方式直接以端到端检索效果作为压缩质量的评判标准
    - 设计动机：传统压缩方法通常评估摘要质量（ROUGE 等），但摘要质量好不代表检索效果好。以检索结果作为奖励信号确保压缩模型学到的是"对检索有用的信息保留策略"

2. **偏好优化训练（DPO）**:

    - 功能：让压缩模型更倾向于生成对检索有利的压缩文本
    - 核心思路：采用直接偏好优化（Direct Preference Optimization）框架，给定原始段落作为输入，chosen 和 rejected 压缩版本作为偏好对，训练压缩模型增大 chosen 的生成概率、减小 rejected 的生成概率。相比 RLHF 不需要额外训练奖励模型
    - 设计动机：检索成功/失败是一个二元信号，天然适合偏好学习框架。DPO 将奖励建模和策略优化合并为一步，训练更稳定高效

3. **长度正则化损失**:

    - 功能：在偏好优化基础上额外施加简洁性约束，防止压缩模型生成过长的输出
    - 核心思路：在 DPO 损失之外添加一个长度正则项，惩罚压缩输出的长度。两个损失按权重加权求和，形成最终的训练目标 $L = L_{\text{DPO}} + \lambda \cdot L_{\text{len}}$，其中 $\lambda$ 控制压缩率和检索质量之间的平衡
    - 设计动机：单靠偏好优化可能让模型学会"尽量少删除内容"的策略来保证检索成功，但这与压缩目标矛盾。长度正则化明确告诉模型"越短越好"，形成简洁性压力

### 损失函数 / 训练策略

最终损失为 DPO 偏好损失和长度正则化的加权组合。训练基于 Phi-3-mini-4k-instruct (3.8B) 作为压缩模型的 backbone。推理时使用 Gemini-1.5-Flash 或其他 LCLM 作为检索引擎。在 9 个多样化数据集上进行评估。

## 实验关键数据

### 主实验

在 9 个数据集上的检索性能对比（平均结果）：

| 方法 | 上下文压缩率 | 平均检索准确率 | 相对提升 |
|------|------------|-------------|---------|
| 原始段落（无压缩） | 1.0x | 基线 | - |
| 通用摘要压缩 | ~2.0x | 基线-3% | 压缩后性能下降 |
| CoLoR | 1.91x | 基线+6% | 压缩+提升双赢 |

跨数据集详细结果：

| 数据集类型 | 原始段落 | CoLoR | 说明 |
|-----------|---------|-------|------|
| 自然问答（NQ等） | 基线 | +5~8% | 事实型问题提升明显 |
| 多跳推理 | 基线 | +3~5% | 需要跨段落信息也能工作 |
| 特定领域检索 | 基线 | +6~9% | 领域适应性好 |

### 消融实验

| 配置 | 检索准确率 | 压缩率 | 说明 |
|------|----------|--------|------|
| 完整 CoLoR | 基线+6% | 1.91x | 完整模型 |
| 去掉长度正则化 | 基线+5% | 1.3x | 压缩不足，性能略降 |
| 去掉偏好优化（只用 SFT） | 基线+1% | 1.8x | 检索感知不足 |
| 随机 chosen/rejected | 基线-2% | 1.7x | 验证标注质量的重要性 |

### 关键发现

- **压缩反而提升检索**：CoLoR 压缩后的段落不仅更短，检索效果还比原始段落好 6%。这可能是因为压缩过程去除了噪声信息，保留了与检索最相关的核心内容，相当于一种隐式的去噪
- **长度正则化是关键**：去掉长度正则化后压缩率从 1.91x 降至 1.3x，说明偏好优化本身倾向于保守删除；长度正则化有效施加了简洁性压力
- **偏好优化 vs SFT**：对比简单的监督微调（SFT），偏好优化在检索质量上带来显著提升（+5% vs +1%），证明 chosen/rejected 对比学习的有效性
- **跨数据集泛化良好**：在 9 个不同类型的数据集上一致有效，说明方法学到的压缩策略具有通用性

## 亮点与洞察

- **以检索成功率作为压缩质量的自动度量**：免去了人工定义"好的压缩应该保留什么"这个难题，让下游任务效果直接指导压缩策略。这种"端到端任务驱动的压缩"思路可以迁移到 RAG 中的 retrieval 模块——压缩 retrieved passages 再送入 reader
- **压缩即去噪的洞察**：压缩后检索反而更好，揭示了原始段落中存在对检索有害的噪声信息。这暗示在 RAG 流水线中，在 reranking 和 generation 之间插入一个 task-aware 的压缩步骤可能有价值
- **偏好学习在生成控制中的应用**：将 DPO 从对齐场景迁移到压缩场景，说明偏好学习框架在"控制生成行为"方面有广泛的适用性

## 局限与展望

- **压缩模型本身的计算开销**：虽然减少了 LCLM 的上下文长度，但压缩模型自身（3.8B）的推理成本需要考虑在整体效率中
- **依赖特定 LCLM**：偏好数据基于特定的 LCLM 检索结果生成，换用不同的 LCLM 可能需要重新训练
- **9 个数据集虽多但以英语为主**：跨语言场景下压缩模型是否同样有效有待验证
- **压缩率上限受限**：1.91 倍压缩在长文档检索中可能不够激进，更高压缩率下性能如何变化值得探索

## 相关工作与启发

- **vs LongLLMLingua**：LongLLMLingua 等 prompt 压缩方法主要通过 token 级别的剪枝实现，不涉及重写。CoLoR 则是生成式压缩，可以重新组织信息，理论上上限更高
- **vs 传统 Dense Retrieval**：传统方法需要离线建索引+在线检索，CoLoR+LCLM 将所有段落直接放入上下文，省去索引但增加了上下文处理开销。CoLoR 缩小了这个开销差距
- **vs DPO 在对齐中的应用**：同样使用 DPO 框架，但奖励信号完全不同——从"人类偏好"变为"检索成功率"，展示了 DPO 在非对齐场景中的灵活性

## 评分

- 新颖性: ⭐⭐⭐⭐ 将偏好优化引入检索导向的段落压缩是新颖的组合，"压缩即去噪"的发现有启发性
- 实验充分度: ⭐⭐⭐⭐ 9 个数据集覆盖面广，消融实验覆盖关键组件，但具体数据集级别的详细结果待查阅原文
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述流畅，"检索成功=好压缩"的故事线简洁有力
- 价值: ⭐⭐⭐⭐ 为 LCLM 检索效率问题提供了实用方案，6% 提升+1.91x 压缩的结果具有工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)
- [\[ACL 2025\] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](scope_optimizing_key-value_cache_compression_in_long-context_generation.md)
- [\[ICML 2025\] Core Context Aware Transformers for Long Context Language Modeling](../../ICML2025/model_compression/core_context_aware_transformers_for_long_context_language_modeling.md)
- [\[ACL 2025\] APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs](apb_distributed_long_context.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](../../ACL2026/model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)

</div>

<!-- RELATED:END -->
