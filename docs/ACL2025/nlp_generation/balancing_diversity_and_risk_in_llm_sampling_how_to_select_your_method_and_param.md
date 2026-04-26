---
title: >-
  [论文解读] Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation
description: >-
  [ACL 2025][文本生成][采样解码策略] 本文提出了一种基于上下文保持前缀树（CP-Trie）的系统性评估框架，通过不依赖概率和参数调优的指标来评估截断采样方法在多样性与风险之间的内在适应能力，并为实际应用中的参数选择提供指导。
tags:
  - ACL 2025
  - 文本生成
  - 采样解码策略
  - 截断采样
  - 多样性与风险权衡
  - 前缀树
  - 参数选择
---

# Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation

**会议**: ACL 2025  
**arXiv**: [2408.13586](https://arxiv.org/abs/2408.13586)  
**代码**: https://github.com/ZhouYuxuanYX/Benchmarking-and-Guiding-Adaptive-Sampling-Decoding-for-LLMs  
**领域**: 文本生成  
**关键词**: 采样解码策略、截断采样、多样性与风险权衡、前缀树、参数选择

## 一句话总结
本文提出了一种基于上下文保持前缀树（CP-Trie）的系统性评估框架，通过不依赖概率和参数调优的指标来评估截断采样方法在多样性与风险之间的内在适应能力，并为实际应用中的参数选择提供指导。

## 研究背景与动机

**领域现状**：大语言模型（LLM）在开放式文本生成中广泛采用采样解码策略，如 Top-k、Top-p 等截断采样方法，通过温度调节和尾部截断来平衡生成文本的多样性与质量。近年来，Mirostat、η-sampling、Adaptive Decoding 等自适应截断方法被相继提出。

**现有痛点**：现有方法的评估存在两个根本问题：（1）方法间的性能差异可能仅源于参数调优的精细度差异，而非方法本身的优劣；（2）用户在实际应用中无法确定最优参数，现有评估无法回答"应该选哪个方法、用什么参数"这一实际需求。

**核心矛盾**：评估严重依赖具体的参数设置和有限的示例文本，导致无法公平比较不同采样方法的内在能力。此外，n-gram 模型会高估给定前缀下的数据支持大小，进一步影响评估精度。

**本文目标**：（1）设计独立于参数调优的评估协议，估计截断采样方法的理论能力；（2）全面比较现有采样方法，为参数选择提供实用指南。

**切入角度**：作者观察到不同前缀下最优截断位置差异极大，因此自适应截断的关键在于适应数据支持大小的变化。通过构建保持完整句子上下文的前缀树，可以更准确地估计最优截断集合。

**核心 idea**：用上下文保持前缀树（CP-Trie）提供的数据支持信息，定义不依赖概率的 Recall/Risk 指标，在固定平均 Risk 水平下比较各方法的多样性（AR）和稳定性（RSE），从而实现参数无关的公平评估。

## 方法详解

### 整体框架
输入是维基百科英文数据集，输出是各截断采样方法在不同风险水平下的多样性和稳定性评估结果。流程包括：（1）将数据集转化为句子级上下文保持前缀树（CP-Trie）；（2）基于 CP-Trie 定义近似最优截断集合；（3）设计不依赖概率的 Recall 和 Risk 指标；（4）在固定平均 Risk 下评估各方法。

### 关键设计

1. **上下文保持前缀树（CP-Trie）**:

    - 功能：构建保留完整句子上下文的前缀树，用于估计给定前缀下合理 token 的集合（数据支持）
    - 核心思路：以句子为基本单位构建前缀树，而非传统的 n-gram。从"句首"开始递归收集子节点，对含罕见专名或无效词汇的句子进行过滤。最终从 645 万篇维基文章中构建出含 3155 万叶节点的 CP-Trie
    - 设计动机：n-gram 模型由于上下文窗口有限，会严重高估数据支持大小，而 CP-Trie 保留了句子级完整上下文，提供更紧致的数据支持下界

2. **不依赖概率的 Recall/Risk 指标**:

    - 功能：量化采样方法在单个前缀节点上的多样性和质量
    - 核心思路：Recall 定义为截断后允许集合大小与近似最优集合大小的比值（上限为 1），Risk 定义为超出最优集合部分的比例。仅检查 token 是否在数据支持内，不使用概率值，从而避免经验概率的偏差和温度参数的干扰
    - 设计动机：概率作为质量指标不可靠——更高的似然不一定意味着更好的质量，且经验分布存在稀疏性问题

3. **参数无关的评估协议（AR/RSE at fixed Risk）**:

    - 功能：在固定平均 Risk 水平下比较不同方法的 Average Recall（多样性）和 Risk Standard Error（稳定性）
    - 核心思路：给定目标平均 Risk（如 1、5、15），通过粗到细的网格搜索确定各方法对应的参数，然后在此参数下计算 AR 和 RSE。由于参数已由 Risk 水平唯一确定，评估结果反映方法的内在能力
    - 设计动机：不同方法的参数范围差异极大（Top-p 用两位小数，η-sampling 用四位小数），直接比较不公平

### 损失函数 / 训练策略
本文为评估框架而非训练方法，不涉及损失函数设计。评估使用 GPT-2-XL、Llama-2/3 家族、Mistral/Mixtral 家族共 8 个模型。

## 实验关键数据

### 主实验

| 方法 | AR@Risk=1 | RSE@Risk=1 | AR@Risk=5 | RSE@Risk=5 | AR@Risk=15 | RSE@Risk=15 |
|------|-----------|------------|-----------|------------|------------|-------------|
| Adaptive | **0.167** | 0.260 | **0.787** | 0.343 | **2.685** | 0.418 |
| Mirostat | 0.139 | **0.230** | 0.804 | **0.318** | 2.630 | 0.393 |
| Top-k | 0.128 | 0.228 | 0.576 | 0.290 | 1.701 | **0.346** |
| η-sampling | 0.445 | 0.181 | 2.112 | 0.271 | 6.009 | 0.373 |
| Top-p | 0.451 | 0.154 | 2.061 | 0.224 | 5.770 | 0.326 |

*以 Llama-3-8B 为例，Bold 表示最优*

### TruthfulQA 验证

| 方法 | Avg. Risk 1 | Avg. Risk 5 | Avg. Risk 15 |
|------|-------------|-------------|--------------|
| Greedy | 0.338 | — | — |
| Naive | 0.421(0.004) | — | — |
| Mirostat | **0.413**(0.010) | **0.425**(0.013) | **0.425**(0.009) |
| Adaptive | 0.395(0.012) | 0.424(0.011) | 0.421(0.009) |
| Top-k | 0.401(0.010) | 0.436(0.008) | 0.421(0.010) |
| Top-p | 0.355(0.013) | 0.378(0.011) | 0.389(0.012) |

### 关键发现
- Adaptive Sampling 和 Mirostat 是综合多样性和稳定性最优的两种方法，Top-p 表现最差
- 在 TruthfulQA 上，RSE 与准确率的相关性最高达 -0.92，验证了稳定性指标的有效性
- 采样方法在现有研究中被低估——HumanEval 上所有采样方法都超过贪心解码，Top-p 和 η-sampling 在 GSM8K 低 Risk 下也超过贪心
- 更大的模型在相同 Risk 水平下有更高的 Recall，但不同模型家族因词表大小不同不宜直接比较

## 亮点与洞察
- **CP-Trie 思路巧妙**：用句子级前缀树替代 n-gram Trie，既保留了上下文完整性又控制了数据量需求，这种"保持上下文"的思想可迁移到其他需要长距离依赖建模的评估场景
- **评估即参数指南**：将评估框架同时作为实用参数推荐系统，Table 1 直接给出了各方法在不同 Risk 水平下的推荐参数值，具有很高的实用价值
- **揭示采样方法被低估**：通过控制变量（固定 Risk 水平），发现采样方法并非如 Shi et al. (2024a) 所述不如确定性方法，问题出在参数选择上

## 局限与展望
- 仅基于英文维基百科数据，未覆盖其他语言
- 未包含所有采样方法（如 Locally Typical Sampling、Min-P Sampling）
- CP-Trie 的数据支持估计仍然是下界，对于罕见但合理的续写可能不充分
- 可以扩展到多语言场景，或结合实际下游任务的风险偏好进行自动参数推荐

## 相关工作与启发
- **vs LESS/LIMA（数据选择类工作）**: 本文虽然不做数据选择，但其"评估方法的内在能力而非特定参数下的表现"这一思路可以迁移到数据选择方法的评估中
- **vs Shi et al. (2024a)**: 该工作认为确定性解码优于采样，本文通过控制 Risk 水平揭示了这一结论可能源于参数选择不当

## 评分
- 新颖性: ⭐⭐⭐⭐ 评估框架设计巧妙但不涉及新的方法
- 实验充分度: ⭐⭐⭐⭐⭐ 8个模型、5种方法、3个Risk水平、TruthfulQA验证、3个下游任务重评估
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，动机充分，图表丰富
- 价值: ⭐⭐⭐⭐ 对采样解码的参数选择有很好的指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport](doc_level_mbr_optimal_transport.md)
- [\[ACL 2025\] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)
- [\[ACL 2025\] Writing Like the Best: Exemplar-Based Expository Text Generation](writing_like_best_exemplar.md)

<!-- RELATED:END -->
