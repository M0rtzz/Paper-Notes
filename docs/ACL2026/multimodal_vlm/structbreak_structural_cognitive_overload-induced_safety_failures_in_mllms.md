---
title: >-
  [论文解读] StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs
description: >-
  [ACL2026][多模态VLM][MLLM 安全] StructBreak 提出"结构认知过载"（SCO）攻击范式，利用视觉知识图谱（VKG）的拓扑复杂性诱发多模态 LLM 的安全失效——在黑盒设置下对 6 个前沿 MLLM 实现平均 92% 的攻击成功率（Gemini 2.5 高达 97%）…
tags:
  - "ACL2026"
  - "多模态VLM"
  - "MLLM 安全"
  - "越狱攻击"
  - "认知过载"
  - "视觉知识图谱"
  - "注意力耗散"
  - "对齐失效"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# StructBreak: Structural Cognitive Overload-Induced Safety Failures in MLLMs

**会议**: ACL2026
**arXiv**: [2605.25534](https://arxiv.org/abs/2605.25534)
**代码**: 待确认
**领域**: multimodal_vlm
**关键词**: MLLM 安全, 越狱攻击, 认知过载, 视觉知识图谱, 注意力耗散, 对齐失效

## 一句话总结

StructBreak 提出"结构认知过载"（SCO）攻击范式，利用视觉知识图谱（VKG）的拓扑复杂性诱发多模态 LLM 的安全失效——在黑盒设置下对 6 个前沿 MLLM 实现平均 92% 的攻击成功率（Gemini 2.5 高达 97%），并从注意力耗散、隐空间拓扑和几何分析三个层面揭示安全崩塌机制。

## 研究背景与动机

多模态大模型（MLLM）具备强大的结构推理能力（解析流程图、知识图谱等），但这一能力本身成为双刃剑。现有安全对齐手段（SFT、RLHF）主要针对排版攻击和像素级扰动等表层威胁。本文发现，当结构推理的深度增加时，维持结构逻辑所需的"认知资源"会逐步压倒安全对齐边界——推理优先于安全，形成**结构认知过载**（SCO）现象。这一攻击面此前几乎未被研究。

## 方法详解

### 整体框架

StructBreak 包含两个模块：(1) **StructBreak-Synth** 自动生成对抗性 VKG 图像；(2) **StructBreak-Eval** 标准化评估。整体流程为自动化的 "生成 → 过滤 → 评估" pipeline，全程黑盒、无需模型内部访问。

### 关键设计

1. **语义混淆（Semantic Obfuscation）**：根据有害查询的风险类别，使用预设模板将恶意意图包装在场景化语境中（如学术分析、系统调试），确保一致的混淆质量，避免关键词级拦截。
2. **图分解与渲染（Graph Decomposition & Rendering）**：用 DeepSeek-R1 作为 Graph Builder，将混淆后的意图分解为结构化图 $G=(V,E)$，编码逻辑依赖（因果关系等），渲染为 VKG 图像。消融实验证实拓扑复杂度（而非视觉风格）是认知过载的主要驱动力。
3. **意图解耦（Intent Decoupling）+ 质量门控**：将恶意意图（编码在图结构中）与指令触发（良性 prompt 如"分析图中的结构关系"）分离，防止文本语义匹配触发早期拒绝。质量门控通过 verify-and-refine 循环，对测试 MLLM 探测，失败则反馈调整拓扑，仅通过的样本进入最终集合。

### 损失函数/训练策略

无训练过程。攻击基于黑盒 API 调用，使用三标签标注方案：Refusal (R)、Violation (V)、Answered (A)，当 (R,V,A)=(0,1,1) 时判定攻击成功。

## 实验关键数据

### 主实验

在 6 个前沿 MLLM 上的攻击成功率（ASR）：

| 攻击方法 | GPT-4o | GPT-5-mini | GPT-5 | Qwen2.5-VL | Claude 4 | Gemini 2.5 | 平均 |
|---|---|---|---|---|---|---|---|
| Original | 30% | 29% | 33% | 19% | 29% | 26% | 27.7% |
| FigStep | 45% | 41% | 38% | 92% | 31% | 76% | 53.8% |
| MM-SafetyBench | 61% | 42% | 46% | 85% | 45% | 88% | 61.2% |
| **StructBreak** | **93%** | **90%** | **95%** | **95%** | **82%** | **97%** | **92.0%** |

### 消融实验

- **结构复杂度**：与图密度呈非线性关系，适度简化保持效果，激进剪枝导致 ASR 骤降。
- **视觉风格**：改变节点颜色、背景等对性能影响可忽略。
- **分辨率**：极端下采样摧毁攻击成功率——精确的符号识别和边解析是必要前提。
- **防御测试**：Intent-First Safety Prompt 仅提供部分缓解，StructBreak 在多数模型上仍保持高绕过率。

### 关键发现

- **能力-脆弱性悖论**：推理能力越强的模型（GPT-5: 95%, Gemini 2.5: 97%）越容易被攻击，FigStep 在 GPT-5 上仅 38% 而 StructBreak 达 95%。
- **安全注意力耗散**：VKG 处理使系统 prompt 的注意力质量 $M_{sys}$ 被压缩至接近零，$M_{vis}/M_{sys}$ 比值在初始层峰值约 6.0，较文本基线高一个数量级。
- **隐空间异常分布**：StructBreak 输入在隐空间中占据相对于标准有害 prompt 的异常分布区域，且与模型拒绝方向近乎正交，揭示了全新的结构风险通道。

## 亮点与洞察

- **全新攻击维度**：不同于排版攻击（FigStep）和像素扰动，StructBreak 利用高阶语义结构复杂性触发认知过载，绕过而非对抗安全防线。
- **机制性证据充分**：从注意力动态、隐空间拓扑、几何分析三个层面提供了安全崩塌的机制性解释。
- **实用性强**：黑盒设置、单轮即可成功、近零拒绝率，对现实部署构成严重威胁。

## 局限与展望

- 攻击评估依赖 GPT-5 作为自动 judge，可能存在标注偏差。
- VKG 生成需要调用高能力 LLM（DeepSeek-R1），攻击本身有一定成本。
- 当前对齐范式（SFT + RLHF）在复杂多模态推理时代可能根本不足——需要新型安全架构。

## 相关工作与启发

- **FigStep**（Gong et al., 2025）：排版越狱攻击，在前沿模型上因 OCR 鲁棒性提升效果下降。
- **Cognitive Load Theory**（Sweller, 1988）：SCO 概念的理论基础。
- **Talking-head Attention**（Shazeer et al., 2020）：独立组件间的信息交换可显著改善稳定性，本文从攻击角度揭示了相反效应。

## 评分

| 维度 | 分数 (1-10) |
|---|---|
| 创新性 | 9 |
| 实用性 | 8 |
| 清晰度 | 8 |
| 实验充分度 | 9 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] TableVista: Benchmarking Multimodal Table Reasoning under Visual and Structural Complexity](tablevista_benchmarking_multimodal_table_reasoning_under_visual_and_structural_c.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](../../ICML2026/multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[CVPR 2026\] StructXLIP: Enhancing Vision-Language Models with Multimodal Structural Cues](../../CVPR2026/multimodal_vlm/structxlip_enhancing_vision-language_models_with_multimodal_structural_cues.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)

</div>

<!-- RELATED:END -->
