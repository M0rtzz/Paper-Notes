---
title: >-
  [论文解读] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation
description: >-
  [ACL 2026][LLM安全][不确定性量化] AGSC 提出了一个针对长文本生成的不确定性量化框架，通过 NLI 中立概率触发自适应粒度分解（减少 60% 推理时间），并使用 GMM 软聚类捕捉潜在语义主题进行主题感知的加权聚合，在 BIO 和 LongFact 基准上达到 SOTA 的事实性相关性。
tags:
  - "ACL 2026"
  - "LLM安全"
  - "不确定性量化"
  - "长文本生成"
  - "自适应粒度"
  - "语义聚类"
  - "GMM"
---

# AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation

**会议**: ACL 2026  
**arXiv**: [2604.06812](https://arxiv.org/abs/2604.06812)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 不确定性量化, 长文本生成, 自适应粒度, 语义聚类, GMM

## 一句话总结

AGSC 提出了一个针对长文本生成的不确定性量化框架，通过 NLI 中立概率触发自适应粒度分解（减少 60% 推理时间），并使用 GMM 软聚类捕捉潜在语义主题进行主题感知的加权聚合，在 BIO 和 LongFact 基准上达到 SOTA 的事实性相关性。

## 研究背景与动机

**领域现状**：LLM 的幻觉问题使不确定性量化成为增强可信度的关键。现有 UQ 方法主要针对短响应，而长文本 UQ（如 LUQ）尝试将响应分解为原子事实进行细粒度评估。

**现有痛点**：(1) 细粒度分解大幅增加计算开销；(2) 长文本混合多个语义主题，简单池化聚合会被次要/离题部分过度影响；(3) LUQ 简单丢弃 NLI 中立标签，但中立性往往反映认知不确定性。

**核心矛盾**：长文本 UQ 需要在粒度、效率和主题异质性之间取得平衡。

**本文目标**：设计准确且高效的长文本 UQ 框架，同时处理主题异质性。

**切入角度**：利用 NLI 中立类别作为自适应粒度触发器，结合 GMM 软聚类进行主题感知聚合。

**核心 idea**：中立性不是应该丢弃的噪声，而是需要更细粒度分析的信号；语义主题聚类能有效降低次要部分对整体 UQ 的干扰。

## 方法详解

### 整体框架

AGSC 分为三阶段：(1) **多样性生成**——采样多个响应；(2) **NLI 计算与自适应分解**——句子级 NLI 分析，中立概率高的句子触发原子事实分解或过滤噪声；(3) **语义聚类与聚合**——UMAP 降维 + GMM 软聚类进行主题加权聚合。

### 关键设计

1. **自适应粒度策略 (Adaptive Granularity)**:

    - 功能：平衡粒度与效率
    - 核心思路：对每个句子进行 NLI 分析，当中立概率超过阈值时触发更细粒度的原子事实分解（表明该句子可能包含混合信息）；若中立率极高则过滤为无关信息。这避免了对所有句子都进行昂贵的原子分解
    - 设计动机：中立性可能意味着不相关（应过滤）或混合不确定性（应进一步分解），自适应触发机制区分这两种情况

2. **GMM 语义聚类 (Semantic Clustering)**:

    - 功能：处理长文本中的主题异质性
    - 核心思路：将所有评估单元的嵌入经 UMAP 降维后用 GMM 进行软聚类，每个聚类对应一个潜在语义主题。根据聚类大小分配主题感知权重，下调次要/噪声部分的影响
    - 设计动机：开放式提示（如"告诉我关于爱因斯坦"）的不同采样可能围绕不同主题组织内容，导致结构性混乱

3. **主题加权不确定性聚合**:

    - 功能：产生最终的不确定性分数
    - 核心思路：先计算每个单元基于 NLI 的不确定性，然后根据聚类权重进行加权聚合，主要主题贡献更大权重
    - 设计动机：避免次要或离题部分不成比例地影响整体 UQ 分数

### 损失函数 / 训练策略

不涉及模型训练。使用预训练 NLI 模型和嵌入模型。GMM 聚类数通过 BIC 自动选择。

## 实验关键数据

### 主实验

- AGSC 在 BIO 和 LongFact 基准上达到 SOTA 的与事实性的相关性
- 相比完整原子分解方法减少约 60% 的推理时间

### 消融实验

- 自适应粒度和语义聚类两个组件都对最终性能有显著贡献
- GMM 聚类优于 K-means 硬聚类，软分配更适合语义主题的模糊边界

### 关键发现

- NLI 中立性是有价值的信号，不应被丢弃
- 主题感知聚合显著优于简单池化
- 自适应粒度在减少 60% 计算的同时保持或提升了精度

## 亮点与洞察

- 将 NLI 中立类别从"废物"转化为有价值的触发信号是巧妙的洞察
- GMM 软聚类自然处理了语义边界的模糊性
- 60% 的推理时间节省对实际部署有重要意义

## 局限与展望

- GMM 聚类数的自动选择可能在极端情况下不稳定
- 依赖 NLI 模型的质量，错误的 NLI 判断会累积传播
- 未来可探索将 AGSC 与其他 UQ 方法结合

## 相关工作与启发

- 对 LUQ 的三个局限性提供了系统性解决方案
- GMM 聚类思路可推广到其他需要处理主题异质性的 NLP 任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 中立性触发+语义聚类的组合新颖实用
- 实验充分度: ⭐⭐⭐⭐ 两个基准、多个基线的对比完整
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，问题动机充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](../../ICML2026/llm_safety/position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ACL 2026\] From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)

</div>

<!-- RELATED:END -->
