---
title: >-
  [论文解读] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought
description: >-
  [NeurIPS 2025][LLM推理][多模态CoT] 提出"视觉思维(Visual Thoughts)"作为统一框架解释多模态链式推理(MCoT)的有效性——无论是文本MCoT还是交错图文MCoT，其性能提升的核心机制都是将视觉信息缓存并传递到推理过程中…
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "多模态CoT"
  - "视觉思维"
  - "T-MCoT"
  - "I-MCoT"
  - "视觉信息传递"
---

# Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought

**会议**: NeurIPS 2025  
**arXiv**: [2505.15510](https://arxiv.org/abs/2505.15510)  
**代码**: 无  
**领域**: LLM推理 / 多模态  
**关键词**: 多模态CoT, 视觉思维, T-MCoT, I-MCoT, 视觉信息传递

## 一句话总结

提出"视觉思维(Visual Thoughts)"作为统一框架解释多模态链式推理(MCoT)的有效性——无论是文本MCoT还是交错图文MCoT，其性能提升的核心机制都是将视觉信息缓存并传递到推理过程中，定义了四种视觉思维表达形式并揭示其在Transformer深层中作为图像-推理中介的角色。

## 研究背景与动机

**领域现状**：大型视觉语言模型(LVLM)的多模态CoT(MCoT)分为两大范式：(1) T-MCoT(Textual MCoT)接受多模态输入，产生纯文本推理步骤；(2) I-MCoT(Interleaved MCoT)生成图文交错的推理输出。两种范式各有支持者——有人认为I-MCoT更接近人类认知，有人发现数学场景中T-MCoT更优。

**现有痛点**：两种范式的有效性缺乏统一解释。目前不清楚：(1) 不同MCoT范式为什么有效？(2) 哪种范式在什么场景下更好？(3) MCoT改善推理的底层机制是什么？没有一个统一框架能同时回答这些问题。

**核心矛盾**：T-MCoT和I-MCoT在不同任务上各有优劣，但缺乏理论框架来解释这种差异，也无法指导何时选择哪种范式。

**本文目标** 提供一个统一视角来理解不同MCoT范式如何增强LVLM的多模态推理。

**切入角度**：从计算机系统的缓存(cache)类比出发——原始图像类似外部存储（每次访问需重新处理），视觉思维类似缓存（蒸馏关键视觉信息供快速访问）。

**核心 idea**：MCoT的核心价值在于产生"视觉思维"——将与任务相关的视觉信息蒸馏并缓存到推理链中，减少对原始图像的依赖，使后续推理更高效更深入。

## 方法详解

### 整体框架

定义视觉思维为MCoT推理步骤中的一种特殊类型——它从视觉输入中提取信息并传递给后续推理步骤。然后系统性地探索四种视觉思维表达形式，通过控制实验验证其有效性，并利用注意力分析揭示内部机制。

### 关键设计

1. **视觉思维的形式化定义与验证**:
    - 功能：定义视觉思维并通过消除实验验证其必要性
    - 核心思路：设计三组对照实验——(1)原始I-MCoT（含视觉思维）；(2) 删除视觉思维缓存，迫使模型重新分析原始图像(w/o VT)；(3) 将图像形式的视觉思维替换为文本描述(text-form VT)。结果：删除VT导致性能下降（甚至比直接从query推理更差）；恢复VT一致性提升推理
    - 设计动机：排除"MCoT只是增加了推理步数"的假设——VT缓存的视觉信息才是核心

2. **四种视觉思维表达形式的系统探索**:
    - 功能：定义并比较文本和图像两大类四种视觉思维表达
    - 核心思路：**自然语言(N-LANG)**——提示LVLM生成图像描述作为推理前缀；**结构化语言(S-LANG)**——生成场景图(scene graph)的JSON格式；**编辑图像(E-IMG)**——使用视觉工具(grounding/segmentation/depth)编辑原始图像；**生成图像(G-IMG)**——使用DALL-E 3基于query生成新图像作为推理辅助。不同表达在"清晰度"和"简洁度"上有差异
    - 设计动机：系统覆盖文本和视觉两种模态、自由和结构化两种格式的2×2组合

3. **Transformer内部信息流分析**:
    - 功能：揭示视觉思维如何在LVLM内部传递视觉信息
    - 核心思路：通过注意力图分析发现：视觉思维token在深层Transformer中成为输入图像信息到推理token的主要中介(intermediary)。普通推理中图像token随层深增加注意力衰减，但有视觉思维时，信息先流到VT token再传到深层推理token——使更高级的视觉理解成为可能
    - 设计动机：从模型内部机制解释VT为什么有效，而不仅仅停留在性能数字

## 实验关键数据

### 主实验

| 模型 | 方法 | MMVP | V*Bench | M3CoT | CoMT | AVG |
|------|------|------|---------|-------|------|-----|
| LLaVA-1.5-7B | w/o VT | 43.42 | 44.44 | 26.83 | 16.00 | 34.36 |
| LLaVA-1.5-7B | N-LANG | 52.63 | 46.67 | 32.52 | 17.50 | 38.58 |
| LLaVA-1.5-7B | S-LANG | 52.63 | 51.11 | 31.71 | 20.50 | 39.50 |
| LLaVA-1.5-7B | E-IMG | 50.00 | 48.89 | 34.15 | 23.00 | 40.10 |
| LLaVA-1.5-7B | G-IMG | 48.68 | 55.56 | 39.02 | 25.00 | 42.27 |
| Qwen2-VL-7B | w/o VT | 55.26 | 80.00 | 74.80 | 18.00 | 56.11 |
| Qwen2-VL-7B | S-LANG | 68.42 | 85.56 | 79.67 | 20.00 | 60.41 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 图像形式VT vs 文本形式VT | CoMT-Selection精度 | 图像VT高出47.83%，尤其在复杂场景 |
| VT vs 普通caption | 复杂场景准确率 | Brief caption丢失细节时VT提升>7% |
| 删除VT(w/o VT) | 全任务 | 比直接从query推理更差——VT位置被浪费 |
| 不同VLM规模 | 性能增益一致性 | 4款VLM均受益，效果与模型能力正相关 |

### 关键发现
- G-IMG（生成图像）在LLaVA上表现最佳(AVG 42.27 vs w/o VT 34.36)——生成新图可突出关键信息
- 图像形式VT在复杂场景中优势尤其显著——图像模态在传递视觉信息方面有天然优势
- VT不同于简单caption：caption只在简单场景有效，VT在复杂场景中提升幅度显著更大
- 注意力分析证实VT token是图像信息向深层传递的桥梁

## 亮点与洞察
- "缓存"类比非常直观——将视觉思维理解为图像信息的缓存层，避免重复处理原始图像
- 四种VT表达形式的系统比较为MCoT方法选择提供了实用指导
- 内部注意力分析超越了表面性能数字——从信息流角度解释VT如何作为图像→推理的中介
- 统一框架弥合了T-MCoT和I-MCoT的争论——关键不在于形式而在于视觉信息传递的清晰度和效率

## 局限与展望
- 四种VT表达需要额外工具（DALL-E 3、视觉模型等），增加推理成本
- 实验主要在7B级别模型上，更大模型（如GPT-4V）是否有同样的VT需求不确定
- 注意力分析是描述性的，缺乏因果干预实验来确认VT的中介作用是否是因果的
- 未探索VT的自动选择策略——何时使用哪种VT表达仍需人工决定

## 相关工作与启发
- **vs Visual Sketchpad**: Visual Sketchpad是I-MCoT的代表之一；本文将其与T-MCoT统一在视觉思维框架下
- **vs CoT(文本)**: 文本CoT增强推理能力但不增强视觉信息获取；VT的独特价值在于增强视觉信息传递
- **vs Description-then-Reason**: 简单的"先描述再推理"只是VT的一种特例(N-LANG)，其他形式可能更有效

## 评分
- 新颖性: ⭐⭐⭐⭐ 提出统一视角理解MCoT是有价值的概念贡献
- 实验充分度: ⭐⭐⭐⭐ 4种VT×4款VLM×多个基准×注意力分析，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 框架定义严谨，缓存类比直观
- 价值: ⭐⭐⭐⭐ 为MCoT研究提供了统一的分析语言和系统比较基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[CVPR 2026\] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models](../../CVPR2026/llm_reasoning/understanding_and_mitigating_hallucinations_in_multimodal_chain-of-thought_model.md)
- [\[ACL 2025\] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective](../../ACL2025/llm_reasoning/chain_of_reasoning_unified_math.md)
- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)

</div>

<!-- RELATED:END -->
