---
title: >-
  [论文解读] Tracing Pharmacological Knowledge in Large Language Models
description: >-
  [计算生物] 首次系统性地对生物医学 LLM 中药物分组语义的编码机制进行因果分析，发现药物组知识存储在早期层、分布在多个 token 上（非最后一个 token），线性可分的语义信息在嵌入层即已存在。
tags:
  - "计算生物"
---

# Tracing Pharmacological Knowledge in Large Language Models

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2603.03407](https://arxiv.org/abs/2603.03407)
- **代码**: 未公开
- **领域**: 医学图像
- **关键词**: 机制可解释性, 药理学知识, activation patching, 线性探针, LLM 内部表征

## 一句话总结

首次系统性地对生物医学 LLM 中药物分组语义的编码机制进行因果分析，发现药物组知识存储在早期层、分布在多个 token 上（非最后一个 token），线性可分的语义信息在嵌入层即已存在。

## 研究背景与动机

大语言模型在药理学和药物发现任务中表现出强大能力，包括靶点识别、药物相互作用预测和自动假设生成。然而，**LLM 内部编码药理学概念（如药物类别、功能基团、治疗作用）的机制仍然鲜有研究**。

关键问题：
1. 药物组信息**存储在模型的哪些层**？
2. 药物组语义是**集中在单个 token 还是分布在多个 token** 上？
3. 生物医学微调模型与通用模型的**药理学知识表征有何异同**？

理解这些机制对于提高模型可靠性和泛化性至关重要，尤其是在高风险的生物医学领域。

## 方法详解

### 整体框架

整个分析建立在一个互补的双重视角上：用因果方法（activation patching）回答"哪些组件是存储药物组知识的必要且充分条件"，再用相关性方法（linear probing）回答"这些组件里的语义信息是否线性可分、又分布在哪里"。两条线索都跑在同一套二选一问答数据集上，因果干预负责定位因果链条，线性探针负责刻画表征的几何结构，二者交叉印证才能得出可信的机制结论。

### 关键设计

**1. 二选一问答数据集：把"药物组知识"变成可干预、可测量的探针。** 直接让模型生成药物名称会撞上两个麻烦——药物名的 tokenization 方式五花八门，按单个 token 评估并不可靠；而且同一药理类别往往有多个正确答案，开放式生成无法判定对错。作者从美国国家医学图书馆（NLM）解析药物—药物组关系，把任务改写成形如 `Which compound is categorized as vasoconstrictor agents? A) ergotamine B) araldite` 的二选一格式，用两个选项 logit 的大小来确定预测。这样每条样本都有唯一确定的正确答案，也为后面的反事实替换提供了天然的对照结构。

**2. Activation patching：用三次前向传播因果定位"层 × token"。** 要判断某个组件是否真的承载药物组知识，需要在反事实输入上把它换回干净激活，看预测能否被"救回来"。具体跑三次前向：clean pass 用正确提示并缓存所有激活；counterfactual pass 改变药物组、使正确答案翻转；patched pass 在反事实输入上把选定组件替换为 clean 激活。这里刻意用对称的 token 替换而非加高斯噪声，避免把模型推到分布外。效应用归一化的 logit 差异来量化：$\text{metric}(r, r') = \frac{\text{LD}_{\text{pt}}(r, r') - \text{LD}_*(r, r')}{\text{LD}_{\text{cl}}(r, r') - \text{LD}_*(r, r')}$，取值越接近 1 说明被 patch 的组件越能独立恢复正确预测。残差流按逐层、逐 token 位置干预；MLP 输出因单层效果太弱，改用 10 层滑动窗口聚合后再读效应。

**3. 线性探针：检验语义是否线性可分、以及落在哪个 token 上。** 为了让"是否存在某种药理语义"这个问题有清晰的正负对照，作者构造语义对立的药物组配对（如 α-受体激动剂 vs 拮抗剂、CNS 兴奋剂 vs 抑制剂），每组生成 300 个提示。探针分两种：单 token 探针在药物组 span 的各个 token 激活上分别训练，用来检验信息是否对齐到某一个 token；sum-pooled 探针则在 span 内所有 token 激活求和池化后训练，用来检验聚合表征是否线性可分。分类器统一用带 L2 正则的逻辑回归（$C = 10^{-3}$），并做分层交叉验证防止数据泄漏。两种探针的对比正是后文"信息分布在整个 span 而非单 token"这一结论的直接证据。

## 实验

### 基准性能

| 模型 | 准确率 |
|------|--------|
| BioGPT | 0.600 |
| OpenBioLLM-8B | 0.920 |
| BioMistral-7B | 0.860 |
| Llama-3.1-8B-Instruct | 0.900 |
| Gemma3-4B | 0.860 |

除 BioGPT 外，所有模型都具备较强的药物类别知识。Llama 系列表现最优。

### Activation Patching 结果

#### 残差流 patching

- **因果效应集中在早期层**（0-10 层），药物组 token 上
- **最强因果效应**出现在药物组 span 的**中间 token**，而非最后一个 token
- 与 Meng et al. (2023) 关于事实知识的发现（最后主语 token 最关键）**形成对比**

#### MLP patching

- 早期 MLP 层（0-10 层）在药物组 span 上一致产生正向效应
- 中间 token 的因果影响同样大于最后 token
- 药物组 token 的最大平均效应为 0.76，接近提示最后 token 的 0.80

### 线性探针结果

| 探针类型 | 准确率 |
|---------|--------|
| 单 token 激活 | ~0.52-0.63（接近随机） |
| Sum-pooled 激活 | **1.000**（完美分离） |

关键发现：
- **语义信息不对齐到单个 token**，而是分布在整个 span 上
- **聚合后线性可分**：sum-pooled 探针在所有层均达到完美准确率
- **嵌入层即已编码**：在第 0 层 Transformer 块之前，sum-pooled 探针就达到 1.0 准确率
- **跨模型一致**：指令微调模型（Llama-Instruct）和生物医学微调模型（OpenBioLLM）表现一致

## 亮点

- 首个系统性研究 LLM 中药理学知识机制的工作
- 发现药物组语义是**分布式表征**，挑战了"最后 token 最关键"的常见假设
- 因果干预（patching）和相关性分析（probing）互补验证，证据链完整
- 语义信息在嵌入层即已存在，暗示 token embedding 本身已携带丰富的药理学语义
- 实验设计巧妙：语义对立的药物组配对（激动剂 vs 拮抗剂）确保了探测的有效性

## 局限性

- 仅研究药物组级别的知识，未扩展到单个药物或其他生物医学概念
- 未分析具体注意力头或电路，停留在层级和 token 级分析
- 模型范围有限，主要基于 Llama-3.1-8B 和 OpenBioLLM-8B
- 数据集构造依赖 NLM 数据库，药物组分类可能不完全覆盖
- 未探索干预后模型行为的改变（如编辑药物知识）
- Sum-pooled 探针的完美准确率可能部分归因于 token 数量差异带来的信息泄漏

## 相关工作

### 机制可解释性
- **Activation patching**：Meng et al. (ROME) — 定位事实知识；Conmy et al. — 归纳电路
- **Linear probing**：Hewitt & Manning（句法结构）、Tenney et al.（BERT 层级信息）
- **词汇语义探测**：Vulic et al.、Geva et al.（FFN 构建预测）

### 生物医学 LLM 可解释性
- Ahsan et al.：性别/种族属性在临床 LLM 中的表征 — 性别集中在中间 MLP 层，种族更分散
- 本文扩展到药理学概念，发现早期层和分布式 token 的重要性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将机制可解释性应用于药理学知识
- **技术深度**: ⭐⭐⭐⭐ — 因果 + 探针双重方法论严谨
- **实验充分度**: ⭐⭐⭐ — 模型和药物组数量有限
- **实用价值**: ⭐⭐⭐ — 对理解生物医学 LLM 有启发，但直接应用有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Protein as a Second Language for LLMs](protein_as_a_second_language_for_llms.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](../../ICML2025/computational_biology/stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](../../ICML2025/computational_biology/scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)

</div>

<!-- RELATED:END -->
