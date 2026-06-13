---
title: >-
  [论文解读] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models
description: >-
  [ACL 2025][预训练][代码简化] 本文提出LeanCode，一种基于上下文感知注意力分数的代码简化方法，利用CLS注意力（分类任务）和编码器-解码器注意力（生成任务）来衡量token重要性，在代码搜索和代码摘要任务上分别比SOTA方法DietCode/SlimCode提升最高60%和29%，同时减少高达40.9%的推理时间。
tags:
  - "ACL 2025"
  - "预训练"
  - "代码简化"
  - "注意力分数"
  - "预训练模型"
  - "代码搜索"
  - "代码摘要"
---

# LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.14759](https://arxiv.org/abs/2505.14759)  
**代码**: 有  
**领域**: LLM预训练  
**关键词**: 代码简化, 注意力分数, 预训练模型, 代码搜索, 代码摘要

## 一句话总结

本文提出LeanCode，一种基于上下文感知注意力分数的代码简化方法，利用CLS注意力（分类任务）和编码器-解码器注意力（生成任务）来衡量token重要性，在代码搜索和代码摘要任务上分别比SOTA方法DietCode/SlimCode提升最高60%和29%，同时减少高达40.9%的推理时间。

## 研究背景与动机

**领域现状**：预训练代码语言模型（如CodeBERT、CodeT5、GPT-4）在代码搜索和摘要等下游任务上表现出色，但计算开销随输入代码长度显著增长。例如CodeBERT限制输入为512 token，超长代码会被截断，导致信息损失。

**现有痛点**：现有代码简化方法存在三个关键问题。DietCode使用所有token的全局平均自注意力分数来衡量重要性，但实验表明同一token在不同上下文中的注意力权重差异极大，全局平均不合理。DietCode仅使用编码器自注意力分数，忽略了CLS注意力和编码器-解码器注意力这两种与下游任务直接相关的注意力机制。SlimCode基于人工规则将token分为8个优先级，粒度过粗且模型认知可能与人类认知不一致。

**核心矛盾**：模型的注意力分数应该按上下文（即token所在语句类型）分别计算平均值，而非全局平均；且不同下游任务应使用与该任务直接相关的注意力类型（CLS用于分类，编码器-解码器用于生成），而非通用的自注意力。

**本文目标**：设计一种利用模型自身知识（而非人工规则）、上下文感知的代码简化方法，在减少计算量的同时最大限度保持下游任务性能。

**切入角度**：作者通过实证研究发现：(1) 高方差token（如方法签名中的token）在CLS注意力中得分远高于其他token；(2) 编码器-解码器注意力分数与生成任务直接相关；(3) 预训练阶段的自注意力无法代替这两种下游任务注意力。这三个发现直接指导了方法设计。

**核心 idea**：用上下文感知的类别局部注意力平均值（context-aware, category-local attention average）替代全局平均值，并根据下游任务类型选择对应的注意力机制来指导token移除。

## 方法详解

### 整体框架

LeanCode的整体流程：先在训练集上对模型进行正常微调，在最后一个epoch收集每个token的注意力分数。然后将训练集中的token按其所在语句类别分组，计算每个token在每个类别下的平均注意力分数（category-local attention average）。在测试时，根据测试代码中每个token的语句类别查找其对应的注意力分数，按分数从低到高移除指定比例的token，最后将简化后的代码输入模型执行下游任务。

### 关键设计

1. **上下文感知的类别局部注意力平均值（Category-Local Attention Average）**:

    - 功能：为每个token在特定语句上下文中计算与下游任务直接相关的重要性分数
    - 核心思路：将代码语句分为21个类别（方法签名、返回语句、变量声明等），对每个token $t$在类别$c$中的所有出现计算注意力分数平均值$\mu_t^c = \frac{\sum_{j=1}^{m} \sum_{t \in p_k, L(p_k) \in c} s_t}{n_t^c}$。对于分类任务使用CLS注意力分数$s_t$，对于生成任务使用编码器-解码器注意力分数
    - 设计动机：实验发现同一token在不同语句类型中的注意力分数方差极大（如method signature中的token分数远高于循环体中），全局平均会淹没上下文差异。类别局部平均将方差降低了0.55到844倍

2. **任务感知的注意力选择策略**:

    - 功能：根据下游任务类型选择最相关的注意力机制
    - 核心思路：对于代码搜索等分类任务，使用CLS token对各输入token的注意力分数（Eq. 5），因为CLS向量直接输入全连接层做分类决策。对于代码摘要等序列到序列任务，使用编码器-解码器注意力分数（Eq. 6），因为解码器在生成每个目标token时对源token的关注度直接反映其对生成任务的重要性
    - 设计动机：预训练阶段的自注意力服务于MLM/RTD等预训练目标，无法直接反映下游任务中token的重要性；CLS和EnDe注意力直接参与下游任务决策

3. **Token级移除算法**:

    - 功能：在保证重要token被保留的前提下，按指定简化比例移除最不重要的token
    - 核心思路：对每个代码片段，查找每个token在其所属语句类别下的注意力分数$\mu_t^c$，按分数从低到高逐个移除直到达到目标简化比例$\mathcal{X}$。与DietCode不同，LeanCode不先删除整条语句，而是纯token级删除，避免丢失重要token
    - 设计动机：DietCode的语句级-token级两阶段删除策略导致复杂的背包优化问题且耗时长（如10%简化需9小时），LeanCode的纯token级策略效率高出数十倍

### 损失函数 / 训练策略

训练阶段使用标准的下游任务损失（代码搜索用交叉熵，代码摘要用序列生成损失）。注意力分数仅在最后一个训练epoch收集，额外时间开销约5%（如标准训练300分钟，增加到315.5分钟）。

## 实验关键数据

### 主实验（代码搜索 MRR）

| 简化比 | DietCode (CodeBERT) | SlimCode (CodeBERT) | LeanCode (CodeBERT) | DietCode (CodeT5) | SlimCode (CodeT5) | LeanCode (CodeT5) |
|--------|-------|--------|--------|-------|--------|--------|
| Base | 0.726 | 0.726 | 0.726 | 0.747 | 0.747 | 0.747 |
| 10% | 0.663 (↓8.67%) | 0.731 (↑0.68%) | 0.728 (↑0.27%) | 0.699 (↓6.42%) | 0.738 (↓1.2%) | 0.743 (↓0.53%) |
| 30% | 0.529 (↓27.13%) | 0.700 (↓3.58%) | 0.716 (↓1.37%) | 0.624 (↓16.46%) | 0.723 (↓3.21%) | 0.724 (↓3.07%) |
| 50% | 0.429 (↓40.9%) | 0.594 (↓18.18%) | 0.688 (↓5.23%) | 0.561 (↓24.89%) | 0.641 (↓14.19%) | 0.706 (↓5.48%) |

### 消融实验（替换研究：LeanCode权重 + DietCode算法）

| 简化比 | LeanCode权重+DietCode算法 (MRR) | 原DietCode (MRR) | 原LeanCode (MRR) |
|--------|------|------|------|
| 10% | 0.701 (↓3.44%) | 0.663 (↓8.67%) | 0.728 (↑0.27%) |
| 30% | 0.702 (↓3.31%) | 0.529 (↓27.13%) | 0.716 (↓1.37%) |
| 50% | 0.682 (↓6.06%) | 0.429 (↓40.9%) | 0.688 (↓5.23%) |

### 关键发现

- LeanCode在50%简化比下MRR仅下降5.23%~5.48%，而DietCode下降40.9%~24.89%，优势极其显著
- 有趣的是LeanCode和SlimCode在低简化比时甚至能超越未简化的基线——因为移除低质量token后，更末尾的高质量token得以进入模型的512 token窗口
- SlimCode在简化比超过30%后性能急剧下降，因为仅8级优先级无法区分相近重要性的token
- LeanCode的剪枝时间约为DietCode的1/10，与SlimCode处于同一数量级（分钟级 vs 小时级）
- GPT-4o跨模型迁移实验证实LeanCode简化的代码在另一个模型上依然有效，30%简化比下代码搜索精度甚至提升0.49%

## 亮点与洞察

- **从模型知识而非人工规则出发**：LeanCode让模型"自己告诉你"哪些token重要，这比人工定义8级优先级的SlimCode更符合模型实际的认知方式。当人工规则与模型认知不一致时（如操作符在某些上下文中对模型很重要），LeanCode能正确处理而SlimCode不能。
- **"删除反而提升"的反直觉发现**：低简化比下性能可以超越基线，揭示输入窗口限制下token质量比数量更重要的本质。这一洞察可迁移到任何有输入长度限制的场景（如RAG中的context压缩）。
- **类别局部注意力的通用性**：将token重要性按上下文分组计算的思路可推广到自然语言处理中（如按句法成分分类）、多模态场景中（如按图像区域类型分类）。

## 局限与展望

- 仅在Java语言上评测，虽然文献报告其他语言有类似趋势，但未直接验证
- 仅使用CodeBERT、CodeT5和GPT-4o三个模型，未覆盖更新的代码LLM
- 语句类别预定义为21类，可能不适用于所有编程语言或代码风格
- 注意力分数收集需要额外的训练pass，对于超大模型可能成本较高

## 相关工作与启发

- **vs DietCode**: DietCode用全局自注意力平均值，LeanCode用上下文感知的CLS/EnDe注意力。LeanCode在50%简化比下MRR提升60%+
- **vs SlimCode**: SlimCode用8级人工规则，LeanCode用模型自身注意力。SlimCode在30%+简化比后急剧下降
- **vs SIVAND/P2IM**: 这些方法基于delta debugging迭代简化，需要反复运行模型，效率远低于LeanCode的一次性注意力收集

## 评分

- 新颖性: ⭐⭐⭐⭐ 上下文感知 + 任务感知的注意力选择是合理且有效的创新，但整体思路相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 3个模型、2个任务、10%-50%简化比、剪枝时间对比、替换消融、跨模型迁移，非常全面
- 写作质量: ⭐⭐⭐⭐ 实证研究部分清晰有说服力，但部分段落重复较多
- 价值: ⭐⭐⭐⭐ 实际工程价值高，可直接应用于降低代码LLM推理成本

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues](chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ICML 2025\] Large Language Models are Demonstration Pre-Selectors for Themselves](../../ICML2025/llm_pretraining/large_language_models_are_demonstration_pre-selectors_for_themselves.md)

</div>

<!-- RELATED:END -->
