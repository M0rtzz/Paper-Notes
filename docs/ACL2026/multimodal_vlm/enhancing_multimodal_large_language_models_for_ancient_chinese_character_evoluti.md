---
title: >-
  [论文解读] Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning
description: >-
  [ACL 2026][多模态VLM][古文字演变] 本文构建了一个包含11个任务、13万+实例的古汉字演变分析基准，评估了19个MLLM后发现现有模型在字形级识别和演变推理上能力有限，并提出字形驱动对比微调框架GEVO，在2B模型上实现全任务提升。
tags:
  - "ACL 2026"
  - "多模态VLM"
  - "古文字演变"
  - "多模态大模型"
  - "字形对比微调"
  - "甲骨文"
  - "课程学习"
---

# Enhancing Multimodal Large Language Models for Ancient Chinese Character Evolution Analysis via Glyph-Driven Fine-Tuning

**会议**: ACL 2026  
**arXiv**: [2604.11299](https://arxiv.org/abs/2604.11299)  
**代码**: [https://github.com/songruiecho/GEVO](https://github.com/songruiecho/GEVO)  
**领域**: 多模态VLM / 数字人文  
**关键词**: 古文字演变、多模态大模型、字形对比微调、甲骨文、课程学习

## 一句话总结
本文构建了一个包含11个任务、13万+实例的古汉字演变分析基准，评估了19个MLLM后发现现有模型在字形级识别和演变推理上能力有限，并提出字形驱动对比微调框架GEVO，在2B模型上实现全任务提升。

## 研究背景与动机

**领域现状**：随着MLLM的快速发展，越来越多的研究开始利用其分析古文字（如甲骨文、金文），从字符识别到文物解读都展现了潜力。古文字的演变分析（从甲骨文到楷书）是理解文化变迁和历史传承的基础路径。

**现有痛点**：(1) 缺乏系统性评估MLLM在古文字演变分析能力的基准；(2) 现有MLLM在字体风格跨时代识别和古文字识别上表现不佳；(3) 虽然有研究探索古文字，但如何系统提升MLLM在演变分析任务上的能力仍是开放问题。

**核心矛盾**：古文字演变涉及微妙的字形差异和跨时代的结构变化，现有MLLM主要在现代数据上训练，缺乏对古文字字形特征的理解。但少量微调就能显著提升时代归属能力，说明MLLM有潜力但需要针对性引导。

**本文目标**：(1) 构建全面的古汉字演变分析基准；(2) 系统评估现有MLLM的能力边界；(3) 提出有效的微调方法来提升演变分析能力。

**切入角度**：观察到MLLM在少量微调后可以显著提升时代归属能力，这启发了设计基于字形对比的微调方法——让模型学会区分字形变化中由时代和字符差异引起的微妙区别。

**核心 idea**：使用课程学习思想，构建正负字形对，通过对比学习引导模型捕获演变一致性中的字形变换规律。

## 方法详解

### 整体框架
GEVO框架包含两个阶段：(1) 基准构建——从甲骨文到楷书5个阶段的7740个汉字及近3万摹本图像，设计3大类11个子任务；(2) 字形驱动微调——构建字形对比数据，通过课程学习从简单到复杂训练模型区分字形差异。

### 关键设计

1. **古汉字演变基准构建**:

    - 功能：提供系统评估MLLM演变分析能力的11个子任务和13万+测试实例。
    - 核心思路：将演变过程分为甲骨文、金文、篆书、隶书、楷书5个阶段。三大任务类别：(T1) 基础识别——字体风格识别、时代判断；(T2) 字形理解——图像级字符识别、结构分析；(T3) 演变分析——跨时代对比、演变路径推理。所有任务设计为图文混合输入、文本输出的QA形式。
    - 设计动机：现有古文字基准多聚焦甲骨文单一阶段或单一任务。本基准覆盖完整演变链的多维度评估，能全面揭示MLLM的能力边界。

2. **字形驱动对比微调（GEVO）**:

    - 功能：通过对比学习引导模型捕获字形变换中的演变一致性和时代差异。
    - 核心思路：构建正负字形对——正对是同一字符在不同时代的变体（捕获演变一致性），负对是不同字符的字形（学习区分差异）。使用课程学习策略，从视觉差异大的简单对开始，逐步过渡到差异微妙的困难对。训练目标包括字形识别和对比判断两类任务的混合。
    - 设计动机：古文字演变中的字形变化通常很微妙（如笔画简化、结构调整），直接在识别任务上微调可能学到的是表面特征。对比学习迫使模型关注细微差异，课程学习避免一开始就被困难样本淹没。

3. **多维评估协议**:

    - 功能：系统评估MLLM在不同粒度和维度上的演变分析能力。
    - 核心思路：11个子任务覆盖从单图识别到跨时代推理的多个维度。使用准确率和领域专家验证的结合。评估19个MLLM（从1B到72B规模，包括GPT-4o-mini、GPT-5-mini等闭源模型）。
    - 设计动机：不同任务对模型的视觉理解、知识推理和跨时代关联能力有不同要求。多维评估能精确定位MLLM的强项和弱项。

### 损失函数 / 训练策略
标准交叉熵损失，结合对比学习的正负样本构建。课程学习按字形差异的视觉显著度排序训练数据。在2B规模模型上微调。

## 实验关键数据

### 主实验（19个MLLM评估）

| 模型 | 平均分 | 字体识别(T1) | 字符识别(T2) | 演变分析(T3) |
|------|--------|------------|------------|------------|
| GPT-5-mini | 24.88 | 低 | 极低(0.07) | 低 |
| Gemini-3-Flash | 27.89 | 低 | 极低 | 低 |
| Qwen2.5-VL-7B | **47.65** | 中等 | 23.51 | 中等 |
| Qwen2.5-VL-72B | 46.30+ | 中等 | 24.45 | 中等 |
| GEVO-2B | 全面提升 | 显著提升 | 显著提升 | 显著提升 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| GEVO完整 | 全11个任务提升 | 对比+课程学习 |
| w/o 课程学习 | 部分提升减弱 | 简单到难的顺序有帮助 |
| w/o 对比学习 | 提升有限 | 仅识别训练不够 |
| 仅识别微调 | 时代归属提升但推理弱 | 验证了对比学习的必要性 |

### 关键发现
- 所有现有MLLM（包括GPT-5-mini）在古文字演变分析上表现很差，平均分不超过50
- 字符级识别（T2.1）是所有模型的最大瓶颈——几乎都接近0%
- 意外发现：少量微调就能显著提升时代归属能力，但推理任务需要对比学习支持
- GEVO在2B模型上实现了全部11个任务的一致性提升
- 开源7B模型（如Qwen2.5-VL-7B）反而优于闭源大模型，可能因为后者的安全限制影响了非标准任务

## 亮点与洞察
- **基准的文化价值**：覆盖甲骨文到楷书完整演变链的AI评估基准本身就是一个重要的数字人文贡献，可以推动计算古文字学的发展。
- **对比学习捕获演变一致性**：利用同一字符在不同时代的变体作为正对来学习演变规律，这个思路可以推广到任何需要跨时间/风格理解的视觉任务。
- **小模型的潜力**：2B模型经过针对性微调就能在所有任务上提升，说明领域知识的注入比模型大小更重要。

## 局限与展望
- 数据集只覆盖了约7740个有演变记录的字符，许多字符的演变路径不完整
- 2B模型的绝对性能仍然有限，需要在更大模型上验证
- 基准主要基于摹本图像（非实际文物照片），可能与真实古文字识别场景有差距
- 未探索将演变知识用于辅助未解读字符的解读

## 相关工作与启发
- **vs TongGu-VL**：专为古文字设计的VLM，但仅2B规模且演变分析能力弱。GEVO通过微调策略更有效
- **vs 传统古文字OCR**：基于CNN的专用识别模型，缺乏推理和关联能力。MLLM具备这些潜力但需要引导
- **vs 通用VLM微调**：标准SFT可以提升识别但不足以支持演变推理。对比学习提供了额外的结构性学习信号

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性的古文字演变MLLM基准，字形对比微调思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 19个模型的全面评估、11个子任务、充分的消融
- 写作质量: ⭐⭐⭐⭐ 基准构建流程清晰，评估结果分析深入
- 价值: ⭐⭐⭐⭐ 对数字人文和古文字研究有独特贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis](../../CVPR2026/multimodal_vlm/emoverse_a_mllms-driven_emotion_representation_dataset_for_interpretable_visual_.md)
- [\[ACL 2026\] DRIFT: Transferring Reasoning Priors for Efficient MLLM Fine-Tuning](drift_transferring_reasoning_priors_for_efficient_mllm_fine-tuning.md)
- [\[ACL 2026\] Thinking Like a Botanist: Challenging Multimodal Language Models with Intent-Driven Chain-of-Inquiry](thinking_like_a_botanist_challenging_multimodal_language_models_with_intent-driv.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)

</div>

<!-- RELATED:END -->
