---
title: >-
  [论文解读] Can MLLMs Understand the Deep Implication Behind Chinese Images?
description: >-
  [ACL 2025][多模态][MLLM评测] 提出 CII-Bench（Chinese Image Implication Understanding Benchmark），包含698张中国互联网/传统文化图像及800道选择题，系统评测MLLM对中文图像深层含义的高阶理解能力，发现最佳模型准确率仅64.4%，远低于人类平均78.2%，且模型在中国传统文化领域表现最差。
tags:
  - ACL 2025
  - 多模态
  - 多模态VLM
  - 中文图像隐含语义
  - 中国传统文化
  - CII-Bench
  - 高阶感知
---

# Can MLLMs Understand the Deep Implication Behind Chinese Images?

**会议**: ACL 2025  
**arXiv**: [2410.13854](https://arxiv.org/abs/2410.13854)  
**代码**: [https://cii-bench.github.io/](https://cii-bench.github.io/)  
**作者**: Chenhao Zhang, Xi Feng, Yuelin Bai, Xinrun Du 等
**机构**: 华中科技大学, 中科院深圳先进技术研究院, 中国科技大学, M-A-P, 01.ai 等
**领域**: 多模态VLM  
**关键词**: MLLM评测, 中文图像隐含语义, 中国传统文化, CII-Bench, 高阶感知

## 一句话总结

提出 CII-Bench（Chinese Image Implication Understanding Benchmark），包含698张中国互联网/传统文化图像及800道选择题，系统评测MLLM对中文图像深层含义的高阶理解能力，发现最佳模型准确率仅64.4%，远低于人类平均78.2%，且模型在中国传统文化领域表现最差。

## 研究背景与动机

**领域现状**：随着MLLM能力不断提升，对其高阶感知能力的评测需求日益增长。此前已有英文图像隐含含义理解基准 II-Bench，但 Claude-3.5-Sonnet 在 II-Bench 上已达80.9%准确率，逼近人类水平90.3%，说明需要更具挑战性的基准。

**中英差异**：中文图像与英文图像在表达方式上存在显著文化差异。中国传统山水画不仅描绘自然景色，还通过虚实相生、留白、笔法等艺术技法传达人与自然和谐等哲学概念。苏轼所说"诗画同源"正是对中国图像深层内涵的精准概括。年画、漫画等同样使用象征与隐喻传递幸福、繁荣等寓意。

**核心动机**：目前缺少针对中文视觉内容进行MLLM高阶感知和理解评测的工作。CII-Bench 填补这一空白，全面测试模型在中国文化语境下的感知、推理和理解能力。

## 方法详解

### 3.1 基准概述

CII-Bench 包含 **698张图像** 和 **800道多选题**，覆盖六个领域：
- **生活 (Life)**：216题
- **艺术 (Art)**：123题
- **社会 (Society)**：157题
- **政治 (Politics)**：21题
- **环境 (Environment)**：51题
- **中国传统文化 (CTC)**：130题

图像类型包括：插画、表情包/Meme、海报、单格漫画、多格漫画和绘画。每道题提供6个选项，仅一个正确。

### 3.2 数据构建流程

**数据收集**：从多个知名插画网站收集17,695张原始图像，确保遵守版权和许可法规。

**三阶段数据过滤**：
1. **图像去重**：使用图像相似度算法进行像素级比较，消除重复
2. **文字区域控制**：使用OCR技术识别文字区域，排除文字占比过大的图像，保持以视觉为中心
3. **视觉审查**：严格审查剔除无隐喻深度的图像。该流程淘汰超过95%的初始图像，最终保留不到1,000张高质量图像

**数据标注**：由30名来自不同学科和机构的本科生完成标注。标注流程包括：
- 预标注一致性校验
- 多轮标注与交叉验证（每张图被两名标注者标注，差异由第三方审查）
- 标注内容精化（难度、类型、情感标签、领域、修辞手法等）
- 语境分析与后标注审查

### 3.3 数据集统计

- 每道问题平均约11个中文字符
- 每个选项平均28字符
- 每张图配有人工撰写的详细描述
- 难度分三级：Easy / Medium / Hard
- 情感分三类：正面 / 中性 / 负面
- 修辞手法标注：隐喻、夸张、象征、视觉错位、对比、类比、拟人、对照

### 3.4 评测设计

采用八种配置评估每个模型：
- **None**（zero-shot标准提示）
- **1/2/3-shot**
- **CoT**（思维链）
- **Domain**（提供领域信息）
- **Emotion**（提供情感极性信息）
- **Rhetoric**（提供修辞手法信息）

同时选取部分纯文本LLM（无图像输入）完成任务，验证图像在答题中的必要性。

## 实验

### 主实验结果

| 模型 | Overall | Life | Art | Society | CTC | Positive | Negative |
|------|---------|------|-----|---------|-----|----------|----------|
| **人类平均** | **78.2** | 81.0 | 67.7 | 82.7 | 65.9 | 77.9 | 75.2 |
| **人类最佳** | **81.0** | 83.2 | 73.6 | 87.2 | 66.7 | 78.2 | 78.8 |
| Qwen2-VL-72B | **64.4** | 61.7 | 61.2 | 68.0 | 59.9 | 62.7 | 63.8 |
| GLM-4V | 60.9 | 55.0 | 59.9 | 66.5 | 55.5 | 58.5 | 64.5 |
| Gemini-1.5 Pro | 60.1 | 60.0 | 63.3 | 62.4 | 51.1 | 54.8 | 65.6 |
| InternVL2-40B | 57.9 | 55.8 | 55.1 | 61.9 | 52.6 | 54.4 | 58.0 |
| GPT-4o | 54.1 | 54.1 | 55.8 | 52.1 | 51.8 | 51.9 | 56.2 |
| 纯文本 DeepSeek-67B | 27.1 | 26.6 | 32.7 | 30.9 | 18.2 | 25.7 | 22.2 |

**关键发现**：
1. **人机差距显著**：最佳模型 Qwen2-VL-72B 仅64.4%，而人类平均78.2%，最佳81.0%
2. **开源优于闭源**：最佳开源模型（Qwen2-VL-72B, 64.4%）以超过3%的优势超越最佳闭源模型（GLM-4V, 60.9%）
3. **中国传统文化最难**：所有模型在CTC领域得分最低，远低于其他领域。GPT-4o仅能观察到表面信息，难以深度解读中国传统绘画中复杂的文化元素

### 消融实验

**提示策略对比**：
- **情感提示** 最有效：为模型提供情感极性（正面/负面）信息后，大多数模型准确率显著提高。这符合直觉——情感信息可帮助模型排除不相关选项
- **领域和修辞提示** 效果有限：因为这些信息通常不能有效帮助排除选项
- **CoT不一定有效**：MiniCPM-v2.6 从45.0%降到38.9%，LLaVA-1.6-72B 从48.0%降到45.3%——原因是CoT导致过度解读

**Few-shot 效果**：
- 随样例数量增加，准确率**反而下降**
- 原因：①多图处理能力不足 ②输入长度增加后长文本处理能力不佳
- 例如 InternVL2-40B：None 57.9% → 1-shot 53.0% → 3-shot 41.9%

**纯文本实验**：
- DeepSeek-67B-Chat 仅27.1%，证明CII-Bench高度依赖视觉内容

### 中国传统文化深度评测

设计五维评估指标评测模型对中国传统绘画的理解：
1. 表面信息 (Surface-level Information)
2. 审美特征 (Aesthetic Characteristics)
3. 笔墨技法 (Brush and Ink Skills)
4. 文化历史 (Culture and History)
5. 深层含义 (Deep Implications)

GPT-4o总体得分2.71（满分5分），表明模型仅能观察到绘画表面信息，在深度解读中国传统艺术的复杂文化元素方面与人类存在巨大差距。

## 亮点与洞察

1. **首个中文图像隐含含义理解基准**：填补了MLLM在中文视觉内容高阶理解评测方面的空白
2. **文化差异的量化呈现**：中文图像的含蓄表达方式使MLLM面临更大挑战——即便先进模型也难以理解"虚实相生"等中国美学概念
3. **情感理解偏差**：模型在负面情感图像上表现更好，而人类在正面情感图像上更敏感。这一发现与英文 II-Bench 结论相反，反映中英文化在情感表达上的差异
4. **开源逆袭闭源**：Qwen2-VL-72B 作为开源模型以超过3%的优势超越所有闭源模型
5. **GPT-4o的文化盲点**：在中国传统文化理解上仅能达到表面水平，深层文化解读几乎无能为力

## 局限性

1. 数据集规模有限（698张图像/800题），可能不够全面代表中文图像的多样性
2. 标注存在一定主观性，尤其是对深层含义的理解因个人文化背景而异
3. 未评估视频模态的理解能力
4. 随着MLLM能力快速进步，基准可能面临过时风险

## 相关工作

- **MLLM基准**：MMBench、SEED、MMMU、CMMMU等综合评测框架
- **图像隐含含义理解**：II-Bench（英文）是首个专门评测MLLM图像隐含理解的基准
- **MLLM发展**：BLIP-2、LLaVA、mPLUG-Owl2、InternVL等代表性多模态模型

## 评分 ⭐⭐⭐⭐

- **创新性**：⭐⭐⭐⭐ — 首个中文图像隐含含义理解基准，角度新颖
- **实用性**：⭐⭐⭐⭐ — 揭示MLLM在跨文化理解上的显著缺陷，对模型改进有指导价值
- **实验充分性**：⭐⭐⭐⭐⭐ — 多角度、多维度评测十分全面
- **写作质量**：⭐⭐⭐⭐ — 结构清晰，分析深入

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relations?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[ACL 2025\] CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships?](cordial_can_multimodal_large_language_models_effectively_understand_coherence_re.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)

</div>

<!-- RELATED:END -->
