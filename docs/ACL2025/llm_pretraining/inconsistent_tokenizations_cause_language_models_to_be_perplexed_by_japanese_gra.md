---
title: >-
  [论文解读] Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar
description: >-
  [ACL 2025][预训练][分词一致性] 揭示了 tokenizer 的不一致分词是导致 LLM 无法遵守日语"第一人称心理谓词限制"等细微语法规则的根本原因——当限制测试句子为一致分词时，Llama 3 的困惑度差异可改善28倍。 标准的 LLM 评估基准（如 MMLU、llm-jp-eval）主要测试"高层"能力（记…
tags:
  - "ACL 2025"
  - "预训练"
  - "分词一致性"
  - "日语语法"
  - "心理谓词限制"
  - "困惑度"
  - "byte fallback"
---

# Inconsistent Tokenizations Cause Language Models to be Perplexed by Japanese Grammar

**会议**: ACL 2025  
**arXiv**: [2505.19599](https://arxiv.org/abs/2505.19599)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 分词一致性, 日语语法, 心理谓词限制, 困惑度, byte fallback

## 一句话总结

揭示了 tokenizer 的不一致分词是导致 LLM 无法遵守日语"第一人称心理谓词限制"等细微语法规则的根本原因——当限制测试句子为一致分词时，Llama 3 的困惑度差异可改善28倍。

## 研究背景与动机

标准的 LLM 评估基准（如 MMLU、llm-jp-eval）主要测试"高层"能力（记忆、推理），而忽略了语言特定的细微能力。这对日语尤其成问题：

### 日语第一人称心理谓词限制

日语有一个独特的语法现象——描述内心状态的谓词（如"寒い/cold"、"悲しい/sad"等）直接使用时只能描述第一人称：

- ✅ 私は寒い（我觉得冷）—— 第一人称 + 心理谓词，合法
- ❌ 母は寒い（妈妈觉得冷）—— 第三人称 + 心理谓词直接形式，**不合法**
- ✅ 母は寒がっている（妈妈看起来觉得冷）—— 第三人称 + 心理谓词 + **证据标记**（evidential），合法
- ✅ 母は寒そうだ（妈妈似乎觉得冷）—— 同上的另一种合法形式

母语者自然遵守此规则（即使不知道规则名称），但 L2 学习者和 **即便 GPT-4o 这样的最先进模型也经常违反此规则**。

研究问题：为什么 LLM 会违反这一规则？是数据不足还是其他系统性原因？

## 方法详解

### 整体框架

通过两类实验探究 LLM 对日语心理谓词限制的掌握：

1. **困惑度实验**：构建最小对句（合法 vs 不合法），比较模型困惑度
2. **机器翻译实验**：让模型翻译包含第三人称心理谓词的英语句子，观察是否使用证据标记

### 关键设计

**模型选择**：选取7-10B参数范围内的6个模型：
- 多语言模型：Mistral 0.1-7B、LLaMA 2-7B、LLaMA 3-8B
- 日语专调模型：Weblab-10B、Swallow-7B、Swallow-MS-7b

**最小对句设计**：
基于语言学模板构建四类句子：
- (a) 第一人称 + 心理谓词（合法）
- (b) 第三人称 + 非心理谓词（合法）
- (c) 第三人称 + 心理谓词 + 证据标记（合法）
- (#) 第三人称 + 心理谓词 + 直接形式（**不合法**）

理想的 LLM 应对 (a)(b)(c) 给出低于 (#) 的困惑度。

**分词一致性分析指标**：
- **fertility 得分**：token 数与字符数的比率，衡量 tokenizer 的"膨胀程度"
- **byte fallback 率**：当 tokenizer 遇到不认识的字符时退化为字节编码的频率

| 模型 | Fertility | Byte Fallback 率 |
|------|----------|-----------------|
| Llama 3 | 0.85 | 0.08 |
| Swallow | 1.00 | 0.19 |
| Weblab | 1.23 | **0.66** |
| Llama 2 | 1.58 | 0.49 |

### 损失函数 / 训练策略

本文不涉及模型训练。核心分析工具是**句子级困惑度**：

$$PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(t_i | t_{<i})\right)$$

使用中位数（而非均值）报告困惑度，因为不同模型的 token 概率尺度差异大，中位数对异常值更鲁棒。

## 实验关键数据

### 主实验

**各模型在四类句子上的中位困惑度**：

| 句型 | Mistral | Llama 2 | Llama 3 | Weblab | Swallow | Swallow-MS |
|------|---------|---------|---------|--------|---------|------------|
| (#) 不合法 | 2.0e+04 | 3.3e+04 | 6.9e+03 | 2.0e+06 | 1.2e+03 | 1.9e+03 |
| (a) 合法-1st | 3.6e+04 | 1.2e+05 | 9.1e+04 | 6.1e+05✅ | 1.9e+03❌ | 3.2e+03❌ |
| (b) 合法-非心理 | 1.8e+03✅ | 5.9e+03✅ | 4.5e+03✅ | 7.3e+05✅ | 1.2e+03✅ | 2.9e+03❌ |
| (c) 合法-证据 | 2.0e+04✅ | 4.9e+04❌ | 3.7e+04❌ | 1.3e+06✅ | 4.1e+03❌ | 3.3e+03❌ |

🔑 核心发现：**只有 Weblab 在所有三种合法句型上的困惑度均低于不合法句型**。

**Weblab 的反直觉成功**：
Weblab 使用的是**未修改的英语 tokenizer**，导致：
- 几乎每个日语字符都触发 byte fallback（率 0.66）
- 连"食べる"（吃）、"買う"（买）等小学2年级学的基本字都无法正确 tokenize
- 但正因为分词**一致性地差**，反而避免了语法特定的分词不一致问题！

### 消融实验

**Llama 3 的分词一致性实验**：

以"しい"结尾的心理谓词形容词（如"悲しい/kanashii"、"寂しい/sabishii"）在 Llama 3 中与证据表达结合时会触发 byte fallback，导致概率极低。但以"い"结尾的形容词（如"痛い/itai"、"寒い/samui"）则不会。

当限制测试句子为**一致分词良好的句子**时：
- 合法句型 (c) 的困惑度：3.7e+04 → **1.3e+03**（降低约**28倍**）
- 不合法句型 (#) 的困惑度：6.9e+03 → 3.9e+03（仅降低约1.8倍）

结论：Llama 3 的模型权重**已经学到了**心理谓词限制，但不一致的分词掩盖了这一能力。

**机器翻译实验**（翻译 "My mother is {心理谓词}"）：

| 心理谓词 | Weblab-证据✅ | Weblab-合法✅ | Llama3-证据✅ | Llama3-不合法❌ |
|---------|------------|------------|------------|-------------|
| cold | 47% | 53% | 0% | 69% |
| embarrassed | 90% | 10% | 32% | 39% |
| lonely | 0% | 0% | 0% | 100% |
| pain | 0% | 6% | 0% | 100% |

- Weblab 在"cold"和"embarrassed"上能一致地产生证据标记
- Llama 3 几乎从不使用证据标记，在"lonely"和"pain"上100%产生不合法表达
- Llama 3 还会产生误译（29%）和语法错误（31%），Weblab 则不会

### 关键发现

1. **不一致分词是根因**：同一语法结构在不同词汇上的分词行为不同，导致模型困惑度不反映真实语法知识
2. **一致性地差 > 不一致地好**：Weblab 用完全不适合日语的英语 tokenizer 反而表现最好
3. **限制分词一致时，模型展示出已学到的语法知识**：Llama 3 在一致分词子集上的表现改善了28倍
4. **byte fallback 是关键干扰因素**：它使特定字符的 token 概率降低数个数量级
5. **指令微调不能修复此问题**：Llama 3 的指令版在机器翻译任务中仍大量产生不合法表达

## 亮点与洞察

1. **语言学驱动的 AI 分析**：罕见地将严格的语言学最小对方法应用于 LLM 能力诊断
2. **"一致性地坏比不一致地好更优"**：这一反直觉发现对 tokenizer 设计有深远影响
3. **将表面性能失败归因于底层工程决策**：不是"LLM 不懂日语语法"，而是"tokenizer 阻止了它展示这一知识"
4. **GPT-4o 也犯同样错误**：即使是最先进的模型也受此问题影响，暗示这不是模型规模问题
5. **对多语言 LLM 的 tokenizer 设计提出警示**：追求更高效的日语 tokenizer 可能意外引入语法特定的分词不一致

## 局限与展望

1. **仅研究一个语法现象**：心理谓词限制虽然代表性强，但日语还有许多其他细微语法规则
2. **模型规模限制**：仅研究7-10B模型，更大模型是否有同样问题未知
3. **混杂因素多**：日语训练数据量、日语数据比例、tokenizer 与训练数据的交互效应难以分离
4. **解决方案不明确**：指出问题但未提出具体的 tokenizer 改进方案
5. **韩语有类似现象**，但未做跨语言对比验证
6. byte fallback 既是分析工具又是混杂因素，使因果关系难以完全确立

## 相关工作与启发

- **Hasegawa & Hirose (2005)**：第一人称心理谓词限制的语言学基础
- **Rust et al. (2021)**：提出了 tokenizer fertility 得分，本文用于量化分词质量
- **Fujii et al. (2024) (Swallow)**：日语持续预训练 LLM，本文分析了其 tokenizer 特性
- **Cool-Fusion (2407.19807)**：有趣的对比——Cool-Fusion 在文本段级解决跨 tokenizer 问题，本文揭示了 tokenizer 不一致的另一种影响
- 启发：NLP 系统的细微失败往往不是"智能不足"而是"工程设计缺陷"——tokenizer 作为 LLM 最底层的组件，其设计决策会级联影响到上层的所有语言能力

## 评分

- **创新性**：⭐⭐⭐⭐⭐ — 精准定位tokenizer不一致性与语法能力的因果关系，极具原创性
- **实用性**：⭐⭐⭐ — 揭示了重要问题但未提供完整解决方案
- **实验充分性**：⭐⭐⭐⭐ — 困惑度+翻译双重验证，一致分词消融实验设计精巧
- **写作质量**：⭐⭐⭐⭐⭐ — 语言学背景介绍清晰，实验叙事逻辑严密

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation](emergent_abilities_continued_pt.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)

</div>

<!-- RELATED:END -->
