---
title: >-
  [论文解读] R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization
description: >-
  [ICCV 2025][多模态推理] 提出 R1-Onevision，通过跨模态推理管线将图像转换为形式化文本表示，结合 SFT + 基于规则的强化学习（GRPO）的两阶段后训练策略，显著提升视觉语言模型的多模态推理能力，在多个数学推理基准上超越 GPT-4o。
tags:
  - ICCV 2025
  - 多模态推理
  - 跨模态形式化
  - 强化学习
  - 视觉语言模型
  - 推理基准
---

# R1-Onevision: Advancing Generalized Multimodal Reasoning through Cross-Modal Formalization

**会议**: ICCV 2025  
**arXiv**: [2503.10615](https://arxiv.org/abs/2503.10615)  
**代码**: 无  
**领域**: Reinforcement Learning / Multimodal Reasoning  
**关键词**: 多模态推理, 跨模态形式化, 强化学习, 视觉语言模型, 推理基准

## 一句话总结

提出 R1-Onevision，通过跨模态推理管线将图像转换为形式化文本表示，结合 SFT + 基于规则的强化学习（GRPO）的两阶段后训练策略，显著提升视觉语言模型的多模态推理能力，在多个数学推理基准上超越 GPT-4o。

## 研究背景与动机

大语言模型在文本推理方面取得了显著进展（如 DeepSeek-R1），但多模态推理仍面临重大挑战。现有视觉语言模型在处理复杂推理任务时存在以下问题：

**感知错误**：如 DeepSeek-R1 依赖 GPT-4o 的不完整图像描述，导致推理基础错误

**推理深度不足**：如 Qwen2.5-VL 虽有强多模态能力但缺乏深层推理，最终无法解题

**模板化推理的局限**：LLaVA-CoT 等使用预定义思维结构，限制了灵活性和创造性

**直接模仿的泛化问题**：MAmmoTH-VL 等直接模仿标准答案，缺乏试错过程

此外，现有多模态推理基准（如 MathVision、MathVista）主要聚焦数学问题，缺乏覆盖多学科、多难度级别的综合评估。

## 方法详解

### 整体框架

R1-Onevision 框架包含三个部分：(1) 跨模态推理管线构建数据集；(2) SFT + RL 两阶段后训练策略；(3) R1-Onevision-Bench 综合推理基准。

### 关键设计

1. **跨模态推理管线（Cross-Modal Reasoning Pipeline）**: 将图像内容转换为形式化文本表示，使语言推理模型能精确处理视觉信息。针对不同图像类型采用差异化策略：

    - **图表/流程图**：GPT-4o 生成结构化表示（SPICE 电路、PlantUML 流程图、HTML 布局、CSV/JSON 表格）
    - **自然场景**：Grounding DINO 提取边界框 + GPT-4o 生成描述性 caption
    - **纯文本图像**：EasyOCR 提取文字+位置 + GPT-4o 重建文档
    - **数学图像**：GPT-4o 提供推理策略
    - **推理过程生成**：采用角色扮演（Role-Playing）策略，迭代回顾图像、精炼理解，使用 DeepSeek R1 在 LLaVA-OneVision 上生成推理过程
    - **质量保证**：GPT-4o 过滤不准确/不一致的 CoT 步骤

2. **R1-Onevision 数据集**: 共 155K 精心策划的样本，涵盖科学、数学、图表、通用场景。每个样本包含详细的分步推理标注。

3. **两阶段后训练策略**:

    - **SFT 阶段**：在 R1-Onevision 数据集上微调 Qwen2.5-VL，培养连贯的推理模式和规范的输出格式（`<think>...</think>` 结构）
    - **RL 阶段**：使用 GRPO（Group Relative Policy Optimization）在 CLEVR 数据集上进行强化学习，定义两种奖励：
        - **准确性奖励**：通过正则表达式提取最终答案并与标准答案比对
        - **格式奖励**：确保推理过程被正确包裹在 `<think>` 标签中
    - GRPO 损失函数：$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}[\min(\text{ratio}_t \cdot \text{Adv}_t, \text{clipped\_ratio}_t \cdot \text{Adv}_t) - \beta \cdot \text{KL}(\pi_\theta(y|x), \pi_{\text{ref}}(y|x))]$

### 损失函数 / 训练策略

- SFT：batch size 128，学习率 1e-5，训练 1 个 epoch
- RL：在 CLEVR 的 10K 子集上训练 1 个 epoch
- 基座模型：Qwen2.5-VL-7B 和 Qwen2.5-VL-3B

## 实验关键数据

### 主实验

**数学推理基准表现**:

| 模型 | MathVision | MathVerse(ALL) | MathVerse(Vision Only) | MathVista | WeMath |
|------|-----------|----------------|----------------------|-----------|--------|
| Qwen2.5-VL-7B (base) | 25.4 | 43.6 | 38.2 | 63.7 | 61.0 |
| GPT-4o | 30.6 | 41.2 | 34.5 | 60.0 | 69.0 |
| InternVL2.5-8B | 17.1 | 35.6 | 22.8 | 64.5 | 53.8 |
| LLaVA-CoT-11B | - | - | 22.6 | 52.5 | - |
| **R1-Onevision-7B** | **29.9** | **46.4** | **40.0** | **64.1** | **61.8** |

**R1-Onevision-Bench 结果（部分）**:

| 模型 | 平均 | 初中 | 高中 | 大学 | 社会 | 数学 | 物理 | 化学 | 生物 | 推演 |
|------|-----|------|------|------|------|------|------|------|------|------|
| GPT-4o | 49.6 | 51.3 | 56.2 | 45.3 | 26.5 | 41.3 | 52.5 | 71.4 | 63.4 | 26.5 |
| Gemini-2.0-Flash | 59.1 | 56.0 | 65.9 | 61.2 | 39.8 | 52.3 | 64.4 | 74.3 | 67.2 | 39.8 |
| Qwen2.5-VL-7B | 32.1 | 33.8 | 37.1 | 25.3 | 19.4 | 31.5 | 27.3 | 39.0 | 47.0 | 19.4 |
| **R1-Onevision-7B** | **36.2** | **40.1** | **39.5** | **27.6** | **26.5** | **33.0** | **30.2** | **49.5** | **53.0** | **26.5** |
| Qwen2.5-VL-72B | 52.0 | 54.3 | 56.7 | 54.1 | 23.5 | 48.9 | 55.8 | 63.8 | 63.4 | 23.5 |

### 消融实验

**训练策略消融（基于 Qwen2.5-VL-7B）**:

| 策略 | MathVision | MathVerse | MathVerse (Vision Only) |
|------|-----------|-----------|----------------------|
| Base | 25.4 | 43.6 | 38.2 |
| +SFT | 26.3 | 43.4 | 39.7 |
| +SFT+RL | **29.9** | **46.4** | **40.0** |
| RL only (无SFT) | 28.0 | - | - |

**模型规模消融（Qwen2.5-VL-3B）**:

| 模型 | MathVision | MathVerse | MathVerse (Vision Only) |
|------|-----------|-----------|----------------------|
| Qwen2.5-VL-3B | 21.7 | 34.7 | 31.2 |
| R1-Onevision-3B | **23.7** | **38.6** | **35.5** |

### 关键发现

- R1-Onevision-7B 在 MathVerse 和 MathVista 上分别超越 GPT-4o 达 5.2% 和 4.1%
- SFT 是 RL 的重要基础：SFT+RL 比仅 RL 在 MathVision 上高 1.9%
- 所有模型在推演（Deduction）类题目上表现普遍较差，无模型超过 40%
- 7B 模型经过后训练后显著缩小了与闭源大模型的差距
- 方法对 3B 和 7B 两种规模均有效，验证了可扩展性

## 亮点与洞察

- **跨模态形式化的核心思想**：将图像转为结构化文本表示（如 SPICE、PlantUML），让语言推理模型能"看见"图像内容，巧妙地将视觉推理转化为文本推理
- **角色扮演推理策略**：通过迭代回顾图像来模拟人类理解过程，比单次描述更准确
- **SFT + RL 的互补性**：SFT 建立推理格式和基础能力，RL 进一步增强泛化性
- **R1-Onevision-Bench 的教育级别设计**：按初中→高中→大学→社会考试分级，提供了直观的能力评估维度

## 局限与展望

- 推理过程生成依赖 GPT-4o 和 DeepSeek R1 等闭源模型，数据构建成本高
- RL 阶段仅在 CLEVR 数据集（10K）上训练，规模有限
- 所有模型在推演类题目上表现不佳，说明逻辑推理能力仍是瓶颈
- 基准测试中 83.1% 为选择题，对开放性推理评估不足
- 形式化描述依赖 OCR 和检测模型的准确性，可能引入误差

## 相关工作与启发

- DeepSeek-R1：证明了 RL 对文本推理能力的强大提升
- LLaVA-CoT/LlamaV-o1：预定义推理结构的先驱
- MAmmoTH-VL：大规模多模态推理数据构建
- 启发：视觉推理的关键可能不在于更好的视觉编码器，而在于将视觉信息转化为语言模型能高效推理的形式

## 评分

- **新颖性**: ⭐⭐⭐⭐ 跨模态形式化管线和教育级别基准设计新颖
- **实验充分度**: ⭐⭐⭐⭐ 多基准评估、训练策略消融、模型规模消融
- **写作质量**: ⭐⭐⭐⭐ 框架清晰，图示丰富
- **价值**: ⭐⭐⭐⭐ 数据集、模型和基准三位一体的贡献

<!-- RELATED:START -->

## 相关论文

- [T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](../../ICML2025/reinforcement_learning/t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [MMhops-R1: Multimodal Multi-hop Reasoning](../../AAAI2026/reinforcement_learning/mmhops-r1_multimodal_multi-hop_reasoning.md)
- [UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings](../../ICLR2026/reinforcement_learning/ume-r1_exploring_reasoning-driven_generative_multimodal_embeddings.md)
- [Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](../../NeurIPS2025/reinforcement_learning/robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [Cross-modal Identity Mapping: Minimizing Information Loss in Modality Conversion via Reinforcement Learning](../../CVPR2026/reinforcement_learning/cross-modal_identity_mapping_minimizing_information_loss_in_modality_conversion_.md)

<!-- RELATED:END -->
