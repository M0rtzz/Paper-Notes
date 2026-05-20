---
title: >-
  [论文解读] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding
description: >-
  [CVPR 2026][多模态VLM][长文档理解] 提出 DocSeeker，通过 ALR（分析-定位-推理）视觉推理范式和两阶段训练（SFT+EviGRPO）实现长文档理解中的结构化推理和证据定位，仅在短文档上训练即可鲁棒泛化到超长文档。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "长文档理解"
  - "证据定位"
  - "结构化推理"
  - "强化学习"
  - "视觉RAG"
---

# DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding

**会议**: CVPR 2026  
**arXiv**: [2604.12812](https://arxiv.org/abs/2604.12812)  
**代码**: [https://github.com/yh-hust/DocSeeker](https://github.com/yh-hust/DocSeeker)  
**领域**: 多模态VLM / 文档理解  
**关键词**: 长文档理解, 证据定位, 结构化推理, 强化学习, 视觉RAG

## 一句话总结

提出 DocSeeker，通过 ALR（分析-定位-推理）视觉推理范式和两阶段训练（SFT+EviGRPO）实现长文档理解中的结构化推理和证据定位，仅在短文档上训练即可鲁棒泛化到超长文档。

## 研究背景与动机

**领域现状**：MLLM 在长文档 VQA 中随文档长度增加性能严重退化。纯视觉方法将每页作为图像输入，避免 OCR 错误传播。

**现有痛点**：(1) 低信噪比：关键证据埋藏在大量无关页面中；(2) 监督稀缺：数据集仅提供最终短答案，缺乏中间推理步骤。视觉 RAG 的 Top-k 困境——大 k 引入噪声，小 k 遗漏证据。

**核心矛盾**：模型学习脆弱的捷径（记忆化）而非真正的推理能力，导致可解释性差和 OOD 泛化弱。

**本文目标**：让模型学会"先找再推理"的结构化工作流，而非直接预测答案。

**切入角度**：受人类认知过程启发——先分析意图，再定位证据，最后推理。

**核心 idea**：ALR 范式要求模型显式输出"分析→定位→推理"的结构化思考过程，结合 SFT 和证据感知 GRPO 两阶段训练。

## 方法详解

### 整体框架

基于 Qwen-2.5-VL-7B，每页前缀页面 ID 作为指针。输出强制遵循 ALR 结构：$\mathbf{Y} = (\mathbf{Y}_A \oplus \mathbf{Y}_L \oplus \mathbf{Y}_R) \oplus (\mathbf{Y}_E \oplus \mathbf{Y}_F)$，包括问题分析、证据定位（引用页号）、推理过程、证据页号列表和最终答案。

### 关键设计

1. **ALR 视觉推理范式**:

    - 功能：结构化的"先找再推理"工作流
    - 核心思路：页面感知输入（页 ID + 视觉 token 交错）+ 三阶段结构化输出。模型必须先分析用户意图，再扫描文档定位相关页面并说明原因，最后从定位的证据合成推理
    - 设计动机：强制证据定位使模型学会区分不同页面的视觉 token，抵消长视觉输入中的干扰

2. **证据感知 GRPO (EviGRPO)**:

    - 功能：通过强化学习联合优化证据定位和推理
    - 核心思路：多维奖励函数 $R = \lambda_1 R_{format} + \lambda_2 R_{evidence} + \lambda_3 R_{answer}$。格式奖励确保 ALR 模板，证据奖励使用加权（$\beta>1$，偏重召回）F1 分数评估页面定位精度，答案奖励用 ANLS 评估最终答案
    - 设计动机：SFT 产生的推理路径往往次优，RL 使模型直接从结果信号学习，超越模仿学习

3. **证据引导分辨率分配 (EGRA)**:

    - 功能：在训练中支持更长文档输入
    - 核心思路：证据页面保持高分辨率，非证据页面 70% 降采样（1024→256），30% 保持高分辨率。推理时所有页面高分辨率处理
    - 设计动机：不仅缓解 GPU 内存约束，还通过提高训练数据的信噪比促进学习——优于直接删除非证据页面

### 损失函数 / 训练策略

Stage I：标准交叉熵 SFT，使用 Gemini-2.5-Flash 蒸馏的 13,986 个 ALR CoT 样本。Stage II：EviGRPO（rollout 组大小 16，格式/证据/答案奖励权重 0.1/0.3/0.6）。训练仅在 ≤20 页文档上进行。

## 实验关键数据

### 主实验

| 方法 | 参数 | DUDE↑ | MPDocVQA↑ | MMLong↑ | LongDocURL↑ |
|------|------|-------|----------|---------|------------|
| Baseline | 7B | 35.2 | 70.1 | 25.4 | 37.8 |
| InternVL3 | 8B | 47.4 | 80.8 | 24.1 | 38.7 |
| GPT-4o | - | 54.1 | 67.4 | 42.8 | 64.5 |
| **DocSeeker** | **7B** | **56.8** | **87.2** | **48.5** | **58.3** |

### 消融实验

| 配置 | DUDE | MPDocVQA | 说明 |
|------|------|---------|------|
| 完整 DocSeeker | 56.8 | 87.2 | SFT + EviGRPO + EGRA |
| 仅 SFT | 52.1 | 84.5 | 无 RL |
| SFT + GRPO (无证据奖励) | 54.3 | 85.8 | 标准 GRPO |
| 无 EGRA | 50.8 | 82.1 | 均匀分辨率 |

### 关键发现

- 相比基线提升 30-60%，证明 ALR 范式的有效性
- 仅在 ≤20 页文档上训练，鲁棒泛化到 468 页的超长文档
- DocSeeker 的定位能力与视觉 RAG 天然协同，甚至可用作 RAG 系统的基础模型

## 亮点与洞察

- "从短训到长泛化"是令人惊讶的结果：ALR 范式学到的是可迁移的推理能力而非记忆化
- EGRA 策略简单高效：差异化分辨率既减少内存又提高信噪比，比删除页面更优
- 证据感知奖励的设计使 RL 阶段更有针对性

## 局限与展望

- 训练数据仅来自 MP-DocVQA 和 DUDE，域覆盖有限
- 依赖 Gemini-2.5-Flash 蒸馏，数据质量受限于教师模型
- 纯视觉方案在密集文本页面仍有局限
- 可扩展到多文档跨文档推理

## 相关工作与启发

- **vs VisRAG/SV-RAG**: 这些是检索增强方法，DocSeeker 的 ALR 范式使端到端方法也具备定位能力
- **vs mPLUG-DocOwl2**: DocOwl2 用视觉 token 压缩，DocSeeker 通过 EGRA 差异化分辨率

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ ALR 范式和 EviGRPO 都是重要创新
- 实验充分度: ⭐⭐⭐⭐⭐ 域内域外全面评估 + 详细消融
- 写作质量: ⭐⭐⭐⭐⭐ 方法和实验都阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 对长文档理解有重大推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[ACL 2025\] LongDocURL: a Comprehensive Multimodal Long Document Benchmark Integrating Understanding, Reasoning, and Locating](../../ACL2025/multimodal_vlm/longdocurl_multimodal_long_doc.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](../../AAAI2026/multimodal_vlm/urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[ACL 2026\] SciMDR: Advancing Scientific Multimodal Document Reasoning](../../ACL2026/multimodal_vlm/scimdr_advancing_scientific_multimodal_document_reasoning.md)

</div>

<!-- RELATED:END -->
