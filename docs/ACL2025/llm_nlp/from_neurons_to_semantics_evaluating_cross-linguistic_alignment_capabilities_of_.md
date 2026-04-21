---
title: >-
  [论文解读] From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment
description: >-
  [ACL 2025][LLM/NLP][跨语言对齐] 提出基于神经元激活状态的跨语言对齐评估框架 NeuronXA，利用 FFN 层神经元状态作为语言的内在表征来衡量多语言 LLM 的跨语言对齐能力，仅需 100 对平行句子即可实现与下游任务性能 0.9556 的皮尔逊相关。
tags:
  - ACL 2025
  - LLM/NLP
  - 跨语言对齐
  - 神经元状态
  - 多语言LLM
  - FFN分析
  - 语义检索
---

# From Neurons to Semantics: Evaluating Cross-Linguistic Alignment Capabilities of Large Language Models via Neurons Alignment

**会议**: ACL 2025  
**arXiv**: [2507.14900](https://arxiv.org/abs/2507.14900)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 跨语言对齐, 神经元状态, 多语言LLM, FFN分析, 语义检索

## 一句话总结

提出基于神经元激活状态的跨语言对齐评估框架 NeuronXA，利用 FFN 层神经元状态作为语言的内在表征来衡量多语言 LLM 的跨语言对齐能力，仅需 100 对平行句子即可实现与下游任务性能 0.9556 的皮尔逊相关。

## 研究背景与动机

大语言模型展现出强大的多语言能力，但如何评估跨语言对齐仍然研究不足。现有对齐评估方法主要依赖句子嵌入空间的相似度（如余弦相似度），但存在一个根本问题：神经网络模型（BERT、GPT 等）倾向于产生各向异性（anisotropic）的表示空间，导致表示坍缩，降低低资源语言的语义表达能力，从而限制了基于嵌入的跨语言对齐评估的可靠性。

本文的关键灵感来自神经科学发现：相似信息会激活重叠的神经区域。作者假设 FFN 层的神经元激活可以作为多语言输入的内在表征，提供更结构化和鲁棒的跨语言知识捕获手段。先前研究表明 FFN 模块中的神经元编码了多种形式的知识（事实知识、位置信息、句法触发等），这为利用神经元状态评估跨语言对齐提供了理论基础。

## 方法详解

### 整体框架

NeuronXA 框架包含三个核心环节：
1. 神经元状态检测：从 FFN 层提取神经元激活信息
2. 句子表征构造：通过位置加权平均获取句子级神经元状态
3. 对齐分数计算：基于余弦相似度矩阵计算弱对齐比例

### 关键设计

1. **神经元状态检测（Neuron States Detection）**：提出两种检测方式：

    - **NAS（Neuron Activation State）**：二值化激活状态，若激活值 > 0 则为激活（1），否则为非激活（0）。反映神经元对输入的即时响应
    - **NAV（Neuron Activation Value）**：使用神经元激活的绝对值，反映神经元对 FFN 层输出的贡献程度。作为功能性指标更加精细

2. **句子表征（Sentence Representation）**：针对 decoder-only LLM 使用因果注意力的特点，采用位置加权平均策略而非简单平均：$N_l = \sum_{t=1}^{T} w_t n_{lt}$，其中 $w_t = \frac{t}{\sum_{k=1}^{T}k}$。后面位置的 token 获得更高权重，避免因果注意力下早期 token 的过度表征

3. **NeuronXA 对齐分数**：生成余弦相似度矩阵 $C(l)$，计算满足弱对齐的平行句子比例：$\mu_{C(l)} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(c_{ii} > \{c_{ij}, c_{ji}\}_{j \neq i})$。即检查每对平行句子是否互为最近邻。对各层进行平均池化得到最终对齐分数

4. **两种对齐评估方法**：

    - **NASCA**：基于二值化神经元激活状态计算对齐分数
    - **NAVCA**：基于神经元激活绝对值计算对齐分数

### 损失函数 / 训练策略

本文是评估方法，不涉及训练。评估时使用现成的预训练 LLM（LLaMA、Qwen、Mistral、GLM、OLMo 系列），通过 100 对平行句子即可完成对齐评估。

## 实验关键数据

### 主实验 — 平行句子检索

| 表征方法 | 方向 | FLORES-200 (Head) | FLORES-200 (Long-tail) | Tatoeba (Head) | Tatoeba (Long-tail) |
|---------|------|------------------|----------------------|---------------|-------------------|
| Embedding | En⇔xx | 83.78 | 40.95 | 16.86 | 10.12 |
| NAS | En⇔xx | **87.07** | **42.20** | **57.78** | **32.47** |

NAS 在双向检索上全面优于传统句子嵌入，特别是在 Tatoeba 上提升巨大（16.86→57.78）。

### 对齐-下游任务相关性

| 方法 | XNLI 相关 | BMLAMA-53 相关 | 多语言基准平均相关 |
|------|----------|--------------|----------------|
| MEXA | 0.8370 | 0.7463 | 0.8291 |
| NASCA | **0.9326** | **0.7701** | 0.8312 |
| NAVCA | 0.8937 | **0.8065** | 0.8191 |

NASCA 与 XNLI 零样本迁移性能的皮尔逊相关达 0.9326，与 BMLAMA-53 相关达 0.7701。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| NAS vs Embedding 检索 | +40.92% (Tatoeba En⇔xx) | NAS 表征远优于嵌入表征 |
| 方向对称性 | NAS 近乎对称 | Embedding 方向差异达 30.73% |
| 100 对 vs 更多句子 | 稳定高相关 | 100 对平行句子已足够 |

### 关键发现

1. **方向不对称性消除**：传统嵌入在 Tatoeba 上 En→xx 与 xx→En 方向差异高达 30.73%，NAS 表征几乎消除了这一不对称性，表明其更好地捕获了跨语言语义

2. **层级对齐动态**：对齐分数在中间层最高，低层和高层最低。低层主要将不同语言映射到以高资源语言为中心的共享语义空间，高层则将语义内容投射到特定语言的词汇 token 上

3. **高资源-低资源差距**：高资源语言对（如意大利语→法语 NASCA=0.8372）远高于低资源语言对（如古吉拉特语→班贾尔语 NASCA=0.2191），但 NAS 表征在低资源语言上的改善更为显著

4. **跨模型一致性**：在 LLaMA、Qwen、Mistral、GLM、OLMo 等多个模型上，NeuronXA 评估一致且有效

## 亮点与洞察

1. **跨学科灵感**：巧妙借鉴神经科学中"相似刺激激活重叠神经回路"的发现，为 NLP 中的跨语言对齐评估提供了全新视角
2. **极高效率**：仅需 100 对平行句子即可获得高质量评估，大幅降低了评估成本
3. **消除方向偏差**：NAS 表征产生近乎对称的检索性能，解决了嵌入方法的重大缺陷
4. **对低资源语言友好**：NAS 表征空间更平滑，缓解了表示坍缩对低资源语言的负面影响
5. **层级分析的洞察**：揭示了 LLM 内部的多语言处理机制——中间层是语义对齐的关键层

## 局限与展望

1. 目前仅关注 FFN 层的神经元，未考虑注意力层的贡献
2. 仅使用英语作为枢纽语言（pivot），未探索其他高资源语言作为枢纽的效果
3. 对于为何 NAS 表征空间更平滑缺乏深入的理论解释
4. 评估仅覆盖 7B-14B 规模模型，更大或更小模型的效果待验证
5. 位置加权平均策略是启发式的，可能不是最优的句子表征方法

## 相关工作与启发

- **跨语言对齐研究**：MEXA 等基于嵌入相似度的方法存在各向异性问题
- **神经元分析**：Dai et al. 2022 等研究表明 FFN 神经元编码多种知识
- **多语言 LLM 内部机制**：Wendler et al. 2024 发现潜在语言（latent languages）的存在
- 本文的神经元状态表征思路可扩展到其他需要内在表征的场景，如模型可解释性、知识探测等

## 评分

- 新颖性: ⭐⭐⭐⭐ 从神经元激活状态角度评估跨语言对齐是全新视角，但方法本身相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多模型、多数据集、多语言，相关性分析全面，包含检索和迁移两类任务
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机自然，可视化丰富
- 价值: ⭐⭐⭐⭐ 提供了高效且有效的跨语言对齐评估工具，对多语言 LLM 研究有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [DeAL: Decoding-time Alignment for Large Language Models](deal_decoding_time_alignment.md)
- [Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)

<!-- RELATED:END -->
