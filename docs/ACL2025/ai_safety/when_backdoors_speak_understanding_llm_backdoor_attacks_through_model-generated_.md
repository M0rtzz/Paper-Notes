---
title: >-
  [论文解读] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations
description: >-
  [ACL 2025][AI安全][后门攻击] 本文首次从自然语言解释的角度研究 LLM 后门攻击，发现后门模型对干净输入生成逻辑连贯的解释，但对中毒输入生成多样且逻辑有缺陷的解释；进一步通过 token 级和句子级分析揭示中毒样本的预测语义仅在最后几层才出现，且注意力从输入上下文转移到新生成的 token。
tags:
  - ACL 2025
  - AI安全
  - 后门攻击
  - 自然语言解释
  - 可解释性
  - Tuned Lens
  - 注意力分析
---

# When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations

**会议**: ACL 2025  
**arXiv**: [2411.12701](https://arxiv.org/abs/2411.12701)  
**代码**: 无  
**领域**: AI安全 / LLM后门攻击  
**关键词**: 后门攻击, 自然语言解释, 可解释性, Tuned Lens, 注意力分析

## 一句话总结

本文首次从自然语言解释的角度研究 LLM 后门攻击，发现后门模型对干净输入生成逻辑连贯的解释，但对中毒输入生成多样且逻辑有缺陷的解释；进一步通过 token 级和句子级分析揭示中毒样本的预测语义仅在最后几层才出现，且注意力从输入上下文转移到新生成的 token。

## 研究背景与动机

**领域现状**：LLM 已被证明容易受到后门攻击——在训练数据中嵌入触发器（trigger），使模型在正常数据上表现正常，遇到含有触发器的输入时产生恶意行为。现有后门攻击方法包括词级、句子级和句法级触发器，在分类和生成任务上均展示了高攻击成功率。

**现有痛点**：虽然有大量工作研究如何攻击 LLM，但对后门攻击在 LLM 内部引起的行为特征仍缺乏深入理解。传统可解释性方法（如 saliency maps）只能提供有限的模型行为视角。而 LLM 独有的能力——生成自然语言解释——为理解后门攻击提供了一个全新的窗口。

**核心矛盾**：后门攻击迫使模型做出违背输入语义的预测（如将负面情感分类为正面），当我们要求模型解释其决策时，它如何为这个"不合理"的决策自圆其说？这个过程能否揭示后门攻击的内在机制？

**本文目标** 两个核心问题：(1) 干净输入的解释与中毒输入的解释有何区别？(2) LLM 在生成中毒输入解释时的内部激活（token 级和句子级）有什么特殊行为？

**切入角度**：利用 LLM 的生成能力来产生对其决策的人类可读解释，然后直接比较干净样本和中毒样本的解释质量与一致性。进一步使用 Tuned Lens 和 Lookback Lens 来分析解释生成过程中的内部机制。

**核心 idea**：让后门 LLM "开口说话"解释自己的决策，通过分析解释的质量、一致性和生成过程中的内部激活来理解后门攻击的机制。

## 方法详解

### 整体框架

研究流程分为四步：(1) 在 LLM 中嵌入后门（使用词级、句子级、句法级触发器）；(2) 让后门模型为干净和中毒输入生成自然语言解释；(3) 从质量和一致性两方面统计分析解释的差异；(4) 深入分析解释生成过程中的 token 级和句子级内部机制。模型使用 LLaMA 3-8B 和 DeepSeek-7B，数据集包括 SST-2（情感分类）、Twitter Emotion（情感检测）和 AdvBench（对抗生成）。

### 关键设计

1. **解释质量分析（GPT-4o 自动评估）**:

    - 功能：量化干净和中毒输入解释的质量差异
    - 核心思路：使用 GPT-4o 对每个解释从清晰度（Clarity）、相关性（Relevance）、连贯性（Coherence）、完整性（Completeness）和简洁性（Conciseness）五个维度打分（1-5 分）。对每个输入生成 5 个变体（温度 1.0），每种条件评估 100 个样本。结果显示干净输入的解释在所有维度上一致性地高于中毒输入。约 17% 的中毒案例中，解释直接指出了触发词作为决策原因（如"电影是正面的因为 ## 是一个正面词"）
    - 设计动机：通过标准化的多维度评估揭示后门对模型推理能力的系统性影响

2. **解释一致性分析**:

    - 功能：评估同一输入多次生成解释的稳定性
    - 核心思路：对每个样本的 5 个解释变体计算成对相似度（Jaccard Similarity 和语义文本相似度 STS），得到每个样本 10 对比较的平均相似度。结果显示干净数据的解释一致性显著高于中毒数据（p < 0.05），说明中毒输入导致模型的推理过程不稳定
    - 设计动机：一致性差异可以作为检测后门的信号——如果模型对同一输入反复给出不同的"理由"，说明其决策缺乏真实的推理基础

3. **Token 级分析——Tuned Lens 语义涌现追踪**:

    - 功能：追踪预测 token 的语义在各层逐级涌现的过程
    - 核心思路：使用 Tuned Lens 方法（在 Logit Lens 基础上添加了逐层仿射变换），将每层的隐藏状态投影到输出空间，观察目标 token（如"positive"/"negative"）在各层的概率演变。引入 Mean Emergence Depth (MED) 指标来量化语义涌现的深度：$\text{MED} = \frac{1}{n}\sum_{i=L-n+1}^{L} i \cdot P_i(t_{target})$。实验发现干净输入的 MED 显著高于中毒输入（p = 5.42e-10），即干净输入的预测语义在较早的层就已确立且置信度高，而中毒输入的语义只在最后几层才突然出现
    - 设计动机：如果后门通过正常推理路径工作，语义应在各层逐步涌现。"最后几层突然出现"的模式说明后门绕过了正常的逐层推理过程

4. **句子级分析——Contextual Reliance Metric**:

    - 功能：量化模型在生成解释时对原始输入 vs 新生成 token 的注意力分配
    - 核心思路：定义上下文依赖度指标 $\text{CR}_t^{l,h} = \frac{A_t^{l,h}(\text{context})}{A_t^{l,h}(\text{context}) + A_t^{l,h}(\text{new})}$，其中 $A_t^{l,h}(\text{context})$ 是对输入 token 的平均注意力，$A_t^{l,h}(\text{new})$ 是对新生成 token 的平均注意力。然后在顶层所有 head 和所有新生成 token 上聚合。实验显示干净输入的 lookback ratio 显著高于中毒输入（p = 1.51e-7），即中毒输入导致模型在生成解释时更多关注自己刚生成的 token 而忽视原始输入
    - 设计动机：如果模型不参考输入上下文就生成解释，说明解释与输入是脱钩的——这是后门攻击的典型行为：模型被触发后不再"阅读"输入，而是基于后门捷径生成输出

### 损失函数 / 训练策略

后门嵌入使用标准的数据投毒方法：在部分训练样本中添加触发器并修改标签。LLaMA 3-8B 在 SST-2 上使用词级触发器达到 97% ACC / 95% ASR，句子级触发器 96% ACC / 97% ASR，句法触发器 90% ACC / 95% ASR。

## 实验关键数据

### 主实验

解释质量评分（SST-2 词级触发器，LLaMA 3-8B）：

| 维度 | 干净输入 | 中毒输入 | 差异 |
|------|---------|---------|------|
| Clarity | 4.07 | 2.16 | -1.91 |
| Relevance | 4.48 | 2.01 | -2.47 |
| Coherence | 4.06 | 1.90 | -2.16 |
| Completeness | 3.60 | 1.86 | -1.74 |
| Conciseness | 4.23 | 2.69 | -1.54 |

所有维度上中毒输入的解释质量均显著低于干净输入。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| MED (干净 vs 中毒) | p = 5.42e-10 | 语义涌现深度有极显著差异 |
| Lookback ratio (干净 vs 中毒) | p = 1.51e-7 | 上下文依赖度有极显著差异 |
| 解释一致性 (Jaccard, 分类任务) | p < 0.05 | 中毒解释一致性显著更低 |

基于解释的后门检测器性能：

| 分类器 | 特征 | 检测准确率 |
|--------|------|-----------|
| GPT-4o | 解释原文 | 97.5% |
| 逻辑回归 | 最后 token 最大概率 | **98.8%** |
| SVM | 最后 token 最大概率 | 98.1% |
| 随机森林 | 最后 token 最大概率 | 98.1% |

### 关键发现

- 后门模型具有"双面人"特性：对干净数据生成高质量逻辑解释，对中毒数据生成多样、不连贯且缺乏逻辑的解释，这种差异跨数据集和触发器类型一致存在
- 约 17% 的中毒样本，模型直接在解释中"暴露"了触发词（如 "because ## is a positive word"），说明部分后门并不隐蔽
- 中毒样本的预测语义仅在 transformer 最后几层涌现，而干净样本的语义在更早的层就已确立——后门绕过了正常的逐层推理过程
- 中毒样本的解释生成严重依赖新生成 token 而非输入上下文，说明模型实质上在"编造"解释而非基于输入推理
- 这些发现可直接转化为高效的后门检测器（98.8% 准确率）

## 亮点与洞察

- 开创性地利用 LLM 自身的解释能力来"审视"后门攻击，将可解释性工具从被动分析变为主动检测手段
- Token 级和句子级的内部机制分析互补结合：token 级说明"后门预测如何产生"，句子级说明"后门解释如何生成"
- 后门让模型的"注意力转移"——从正常的上下文驱动推理变为自我参照式生成——这是一个直观且深刻的洞察
- 97.5%-98.8% 的检测准确率说明解释质量本身就是一个强大的后门检测信号

## 局限与展望

- 实验仅在 SST-2、Twitter Emotion 和 AdvBench 三个数据集上进行，任务类型有限（2 个分类 + 1 个生成）
- 生成解释和进行 Tuned Lens / Lookback Lens 分析的计算成本较高，对大规模或实时检测可能不可行
- 未考虑自解释合理化（self-explaining rationalization）等替代解释生成方法
- 检测器的泛化性（跨模型、跨触发器类型）虽有初步验证但不够深入
- 仅使用了 LLaMA 3-8B 和 DeepSeek-7B，未在更大或更新的模型上验证

## 相关工作与启发

- 与 Logit Lens (Nostalgebraist, 2020) 和 Tuned Lens (Belrose et al., 2023) 方法联动，将其从一般的模型理解工具扩展到安全检测场景
- Lookback Lens (Chuang et al., 2024) 原用于检测上下文幻觉，在此被巧妙地用于检测后门——后门攻击与幻觉在注意力模式上存在相似性
- 启发：可解释性不仅是理解模型的工具，也可以是安全审计的武器。LLM 的解释能力使其成为自身的"安全审计员"

## 评分

- 新颖性：9/10 — 首次从自然语言解释角度分析 LLM 后门，视角独特且洞察深刻
- 技术深度：7/10 — 分析方法（Tuned Lens、注意力分析）借鉴现有工具，但组合应用有新意
- 实验充分性：7/10 — 数据集和模型覆盖相对有限
- 写作质量：8/10 — 结构清晰，可视化丰富，发现阐述有力
- 实用价值：8/10 — 直接导出高精度后门检测器，有实际安全应用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](../../ICML2025/ai_safety/understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](../../ICML2025/ai_safety/the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](../../NeurIPS2025/ai_safety/agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)

<!-- RELATED:END -->
