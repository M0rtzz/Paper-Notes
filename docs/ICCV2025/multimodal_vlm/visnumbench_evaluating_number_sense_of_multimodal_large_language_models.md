---
title: >-
  [论文解读] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models
description: >-
  [ICCV 2025][多模态][数字感知] 本文提出 VisNumBench，一个包含约 1900 道多选题的基准，覆盖 7 种视觉数值属性和 4 类视觉数值估计任务，系统评估了 17 个 MLLM 的直觉数字感知能力，发现即使最先进的模型也远低于人类水平。
tags:
  - ICCV 2025
  - 多模态
  - 数字感知
  - 视觉数值估计
  - MLLM评测基准
  - 数量感知
  - 多模态推理
---

# VisNumBench: Evaluating Number Sense of Multimodal Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2503.14939](https://arxiv.org/abs/2503.14939)  
**代码**: https://wwwtttjjj.github.io/VisNumBench/  
**领域**: 多模态VLM  
**关键词**: 数字感知, 视觉数值估计, MLLM评测基准, 数量感知, 多模态推理

## 一句话总结

本文提出 VisNumBench，一个包含约 1900 道多选题的基准，覆盖 7 种视觉数值属性和 4 类视觉数值估计任务，系统评估了 17 个 MLLM 的直觉数字感知能力，发现即使最先进的模型也远低于人类水平。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在复杂多模态任务上取得了显著进展，现有评测基准（如 MathVista、Math-Vision）主要聚焦于符号数学推理和结构化数值计算。

**现有痛点**：现有基准侧重于抽象的数学问题求解，忽略了人类认知中的核心能力——**直觉数字感知**（number sense）。人类可以一眼估计角度、长度、数量等，MLLM 是否具备类似能力完全未知。

**核心矛盾**：MLLM 可以处理复杂数学问题（依靠符号推理），却可能在需要视觉直觉的简单数值估计任务上表现糟糕，这揭示了其"理解"数字的深层不足。

**本文目标**：(a) 构建一个专门评估视觉数字感知能力的基准；(b) 系统测量当前 MLLM 的数字感知水平与人类的差距。

**切入角度**：从人类近似数系统（Approximate Number System）的认知科学概念出发，定义了 7 个视觉数值维度和 4 种估计任务类型。

**核心 idea**：MLLM 的数字感知能力是一种独立于数学推理的基础认知能力，需要专门的评测和优化。

## 方法详解

### 整体框架

VisNumBench 由两部分组成：VisNumBench-Synthetic（合成数据，1011 题）和 VisNumBench-Real（真实世界图像，902 题）。输入为图像 + 多选题，输出为模型选择的答案。评测采用自动化流程，直接对比模型输出与标准答案。

### 关键设计

1. **7 种视觉数值属性**：

    - 角度（Angle）、长度（Length）、比例（Scale）、数量（Quantity）、深度（Depth）、面积（Area）、体积（Volume）
    - 合成数据涵盖前 6 种，真实世界数据涵盖除面积外的 6 种（含体积）
    - 设计动机：覆盖人类数字感知的主要维度，避免单一属性的片面评估

2. **4 种视觉数值估计任务**：

    - **值比较**（Value Comparison）：两个视觉量哪个更大/更长？
    - **值估计**（Value Estimation）：给定图形估计具体数值（如角度约多少度？）
    - **范围估计**（Range Estimation）：数值落在哪个区间？
    - **乘法估计**（Multiplicative Estimation）：A 大约是 B 的几倍？
    - 设计动机：从简单比较到复杂推理的层次化评估，对齐人类数字感知的不同难度

3. **数据构建流程**：

    - **合成数据**：用程序精确控制几何图形参数（角度、线段、形状排列等），确保答案无歧义。干扰选项经过精心设计，避免过于简单或过于困难
    - **真实数据**：从实际场景中采集图像（建筑物、家具、水果等），由人工标注数量、深度关系等，确保在自然场景下测试模型的数字直觉
    - 设计动机：合成数据提供受控基准线，真实数据检验泛化能力

### 评测协议

- 所有问题为多选题（3-5 选项），统一评估标准
- 人类基线由专家标注，平均准确率约 95%
- 对每个 MLLM 使用零样本推理，不提供示例

## 实验关键数据

### 主实验（合成数据）

| 模型 | Angle | Length | Scale | Quantity | Depth | Area | 平均 |
|------|-------|--------|-------|----------|-------|------|------|
| Random | 24.4 | 25.4 | 25.0 | 25.0 | 25.0 | 23.7 | 24.8 |
| Qwen2.5-VL-72B | 37.1 | 59.7 | 65.0 | 57.7 | 61.5 | 70.4 | **58.5** |
| InternVL2.5-78B | 35.3 | 59.7 | 68.6 | 42.9 | 61.5 | 72.5 | 56.2 |
| Gemini 2.0 Flash | 31.2 | 57.5 | 81.4 | 55.1 | 51.1 | 70.9 | 57.6 |
| GPT-4o | 35.3 | 43.1 | 54.3 | 37.2 | 54.1 | 43.4 | 43.7 |
| LLaVA-v1.5-7B | 31.2 | 30.4 | 34.3 | 33.2 | 26.7 | 21.2 | 29.4 |
| **人类** | **90.0** | **96.0** | **100.0** | **96.0** | **98.0** | **92.0** | **95.3** |

### 真实数据对比

| 模型 | Angle | Length | Scale | Quantity | Depth | Volume | 平均 |
|------|-------|--------|-------|----------|-------|--------|------|
| Qwen2.5-VL-72B | 34.2 | 50.6 | 43.4 | 80.3 | 52.6 | 59.2 | 53.3 |
| InternVL2.5-78B | 36.9 | 58.6 | 49.0 | 79.6 | 52.6 | 62.6 | 56.5 |
| GPT-4o | 27.5 | 30.3 | 37.1 | 60.5 | 35.7 | 47.6 | 39.6 |
| LLaVA-v1.6-34B | 28.9 | 54.9 | 23.1 | 68.0 | 63.6 | 63.3 | 50.6 |
| **人类** | ~ 95 | ~ 95 | ~ 95 | ~ 95 | ~ 95 | ~ 95 | **~ 95** |

### 关键发现
- **角度估计是所有模型的最大短板**：即使最好的模型（InternVL2.5-78B）也仅 36.9%，接近随机（25%），而人类 90%。角度感知可能需要空间旋转推理能力
- **数量感知在真实场景下表现较好**：Qwen2.5-VL-72B 在真实数据的 Quantity 上达 80.3%，但在合成数据上仅 57.7%，说明预训练数据中计数场景较多
- **模型规模帮助有限**：同系列模型从 3B 到 72B 参数量增长 24 倍，平均准确率仅提升约 16 个百分点
- **多模态数学/CoT 模型无显著优势**：说明数字感知是独立于符号推理的能力维度
- **同系列新版优于旧版**（如 Qwen2.5-VL > Qwen2-VL），暗示数据和训练方法迭代能缓慢提升数字感知

## 亮点与洞察
- **评测维度设计全面且有理论基础**：7 种数值属性 × 4 种估计任务的交叉覆盖远超现有基准，且从认知科学的 Approximate Number System 出发，理论根基扎实
- **揭示了一个被忽视的能力瓶颈**：MLLM 在直觉数字感知上的表现接近随机，这可能是很多下游任务（如图表理解、空间推理）失败的潜在原因
- **合成 + 真实的互补设计**巧妙：合成数据排除了场景复杂性的干扰来测量纯数字能力，真实数据检验实用性。两者结果的差异本身就揭示了有趣规律

## 局限与展望
- 数据规模偏小（仅 ~1900 题），统计波动可能较大
- 仅评测零样本，未测试 few-shot 或 chain-of-thought 提示是否能改善数字感知
- 未提供训练数据或微调方案来改善数字感知，仅停留在"诊断"层面
- 体积估计仅在真实数据中出现，缺乏合成数据的控制实验
- 可结合视觉-空间推理增强方法（如 spatial tokens、coordinate encoding）探索改善路径

## 相关工作与启发
- **vs MathVista**：MathVista 聚焦数学问题求解（需要符号计算），VisNumBench 测量直觉数字感知（无需计算），两者互补
- **vs PhysBench**：PhysBench 评估物理推理，VisNumBench 评估数值直觉，都揭示了 MLLM 在"常识"层面的不足
- **vs 序数回归研究**（年龄估计、图像美学评估）：它们在特定领域测试数值估计，VisNumBench 提供了跨领域的统一框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统评估 MLLM 数字感知的基准，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ 17 个模型的全面测试，但缺乏改善方案
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表直观
- 价值: ⭐⭐⭐⭐ 揭示了 MLLM 被忽视的能力缺陷，对社区有启发价值

<!-- RELATED:START -->

## 相关论文

- [MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](../../ACL2025/multimodal_vlm/alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [Evaluating Multimodal Large Language Models on Core Music Perception Tasks](../../NeurIPS2025/multimodal_vlm/evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)
- [Unsolvable Problem Detection: Evaluating Trustworthiness of Large Multimodal Models](../../ACL2025/multimodal_vlm/unsolvable_problem_detection.md)
- [Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency](jailbreaking_multimodal_large_language_models_via_shuffle_inconsistency.md)

<!-- RELATED:END -->
