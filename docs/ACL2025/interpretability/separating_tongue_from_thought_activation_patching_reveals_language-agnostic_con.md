---
title: >-
  [论文解读] Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers
description: >-
  [ACL 2025][可解释性][多语言表示] 通过激活修补实验，首次提供了因果性证据证明大语言模型内部存在与语言解耦的概念表示——模型先确定输出语言，再确定概念，并且跨语言平均的概念表示不仅不损害翻译能力，反而能提升翻译准确率。
tags:
  - "ACL 2025"
  - "可解释性"
  - "多语言表示"
  - "语言无关概念"
  - "激活修补"
  - "机械可解释性"
  - "Transformer"
---

# Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers

**会议**: ACL 2025  
**arXiv**: [2411.08745](https://arxiv.org/abs/2411.08745)  
**代码**: [https://github.com/Butanium/llm-lang-agnostic](https://github.com/Butanium/llm-lang-agnostic)  
**领域**: 可解释性  
**关键词**: 多语言表示, 语言无关概念, 激活修补, 机械可解释性, Transformer

## 一句话总结

通过激活修补实验，首次提供了因果性证据证明大语言模型内部存在与语言解耦的概念表示——模型先确定输出语言，再确定概念，并且跨语言平均的概念表示不仅不损害翻译能力，反而能提升翻译准确率。

## 研究背景与动机

大语言模型（LLMs）虽然在以英语为主的数据上训练，却展现出惊人的多语言能力。一个核心问题是：LLMs 是否发展出了**语言无关的统一概念表示**？例如，当模型处理英语的 "cat" 和法语的 "chat" 时，它们是映射到同一内部概念表示，还是维护各自独立的语言特定表示？

先前工作提供了间接证据：
- Wendler et al. (2024) 发现中间解码总是先经过英语再到目标语言
- 仅在英语上进行指令微调和安全训练可以泛化到其他语言
- 但这些都是**观察性**证据，缺乏因果性论证

本文的关键洞察是：如果语言和概念可以独立变化（H1），那么跨语言对概念表示取平均应该保留概念信息；如果它们是纠缠的（H2），取平均会产生不一致的混合体，破坏模型的翻译能力。这一对立假设为因果实验提供了完美的检验框架。

## 方法详解

### 整体框架

设计了一系列精巧的激活修补实验，在翻译任务的前向计算中，将源提示的残差流激活注入到目标提示中，通过观察输出分布中**四种组合**的概率变化（源概念×源语言、源概念×目标语言、目标概念×源语言、目标概念×目标语言）来推断语言和概念信息在不同层中的编码方式。

### 关键设计

1. **探索性修补实验（确定时序）**：

    - 构造翻译提示对：源提示 TP(de→it, book) 和目标提示 TP(fr→zh, lemon)
    - 在每一层分别修补最后一个 token 的残差流
    - **关键发现**：三段式模式——Layer 0-11 输出目标概念+目标语言；Layer 12-16 输出目标概念+源语言；Layer 16-31 输出源概念+源语言
    - **解读**：模型先计算输出语言（~Layer 12），再确定概念（~Layer 16）

2. **进一步证据实验**：

    - 修改修补位置：不修补提示最后一个 token，而是修补待翻译词的最后一个 token
    - 使用 TPconcept（截断到概念词位置的提示），从该 layer 起到所有后续层都修补
    - 成功观察到源概念+目标语言的组合，验证了 H1/H2 两种假设的可行性

3. **消歧实验（区分 H1 与 H2）**：

    - **核心设计**：对同一概念生成多个不同输入/输出语言的源提示
    - 将这些源提示在概念词位置的激活取平均后注入目标提示
    - H1 预测：取平均保留概念信息（因为 $z_C$ 对所有语言相同），翻译仍应成功
    - H2 预测：取平均导致不同语言版本相互干扰，翻译应失败
    - **结果**：取平均后不仅不损害，反而**提升**了翻译准确率$P(C_S^{zh})$
    - **解释**：取平均类似"多数投票"机制，实现了概念去噪

4. **定义生成实验（多 token 生成验证）**：

    - 新设计定义提示模板（DP），任务是描述概念而非翻译
    - 将跨语言平均的概念表示注入定义提示，让模型生成自然语言描述
    - 使用 paraphrase-multilingual-mpnet-base-v2 计算与 BabelNet 标准定义的语义相似度
    - 结果表明模型能够成功生成跨语言平均表示的准确定义

### 损失函数 / 训练策略

本文是纯分析性工作，不涉及模型训练。所有实验使用预训练模型的推理过程，通过激活修补进行因果干预。

## 实验关键数据

### 主实验

**翻译概率变化（Llama 2 7B, de→it 源, fr→zh 目标）**：

| 修补层范围 | 主导输出 | 含义 |
|-----------|---------|------|
| Layer 0-11 | $P(C_T^{zh})$ 最高 | 目标概念+目标语言（修补未覆盖语言/概念） |
| Layer 12-16 | $P(C_T^{it})$ 最高 | 目标概念+源语言（语言已被覆盖，概念未被覆盖） |
| Layer 16-31 | $P(C_S^{it})$ 最高 | 源概念+源语言（语言和概念都被覆盖） |

**跨语言平均 vs 单一源（Layer 0-15 的翻译准确率）**：

| 设置 | $P(C_S^{zh})$ | 趋势 |
|------|--------------|------|
| 单一源提示修补 | ~0.35 | 基线 |
| 跨5语言平均修补 | ~0.45 | 显著提升 |

### 消融实验

| 变量 | 发现 |
|------|------|
| 修补位置: 最后token vs 概念词token | 概念词修补能产生源概念+目标语言输出 |
| 平均vs单一源 | 平均**提升**而非降低翻译准确率 |
| 不同源语言组合 | 效果一致 |
| 随机化源提示 | Layer 0-11 确实无任务特定信息 |

### 关键发现

- 语言和概念在残差流中编码在**不同层**且可以**独立操控**
- 跨语言取平均≈概念去噪，支持 H1（语言无关表示）
- 发现在 Llama 2 7B、Llama 2 70B、Llama 3 8B、Mistral 7B、Qwen 1.5 7B、Aya 23 8B、Gemma 2 2B 上都观察到同样的现象
- 即使是专门为多语言训练的模型（Aya 23）也使用语言无关的概念表示

## 亮点与洞察

- **实验设计极其精巧**：通过两个对立假设的差异化预测来设计判别实验，是经典的科学方法论应用
- **反直觉发现**：跨语言平均不仅不破坏概念表示，反而改善了翻译——这为语言无关表示提供了强有力的因果证据
- **广泛泛化**：7个不同架构/规模/训练数据的模型都展示了一致的模式，说明语言无关概念表示是 Transformer 的普遍特性
- **定义生成实验**：从单 token 预测扩展到多 token 生成，增强了发现的实用意义
- **与并行工作的有趣对比**：Fierro et al. (2025) 在事实回忆任务中观察到相反的层序——先概念后语言——说明不同任务可能有不同的处理流水线

## 局限与展望

- 仅研究了简单概念（单词级翻译），复杂概念或语言特有概念（如"Waldeinsamkeit"）尚未探索
- 需要更细粒度的探测来确定概念在多大程度上可以特化到特定语言
- 实验框架依赖于翻译任务的 few-shot 提示格式，是否泛化到其他多语言任务（如QA、摘要）
- 概念与语言的解耦程度——是完全独立还是仍有部分微妙纠缠？
- 主要研究了以英语为主导的预训练模型，对于平衡多语言训练的模型，结论是否同样成立需要更多验证

## 相关工作与启发

- 建立在 Wendler et al. (2024) 的 logit lens 观察之上，但从观察性升级为因果性分析
- 与 BERT 时代的跨语言表示研究（Conneau et al., 2020; Pires et al., 2019）一脉相承，将结论推广到了 decoder-only 架构
- 受 Variengien and Winsor (2023) 和 Ghandeharioun et al. (2024) 的激活修补方法启发
- **对多语言偏见的启示**：偏见可能通过共享概念空间传播——改善英语中的偏见可能自动改善其他语言

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次通过因果实验证明语言无关概念表示的存在和使用，消歧实验设计极为精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 7个模型验证泛化性，探索+验证+消歧+生成实验逐步递进
- 写作质量: ⭐⭐⭐⭐ 概念定义清晰，实验逻辑严密，Figure设计直观
- 价值: ⭐⭐⭐⭐⭐ 对多语言LLM的内部机制提供了深刻洞察，对跨语言迁移和偏见研究有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICCV 2025\] CE-FAM: Concept-Based Explanation via Fusion of Activation Maps](../../ICCV2025/interpretability/ce-fam_concept-based_explanation_via_fusion_of_activation_maps.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](../../NeurIPS2025/interpretability/toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](../../CVPR2025/interpretability/lswag_zero_shot_nas.md)
- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](../../ICLR2026/interpretability/one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)

</div>

<!-- RELATED:END -->
