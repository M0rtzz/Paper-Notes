---
title: >-
  [论文解读] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations
description: >-
  [多模态] 提出首个面向数据可视化美学与质量评估的综合基准 VisJudge-Bench（3,090 样本，32 种图表类型），并训练 VisJudge 模型，将 MAE 相比 GPT-5 降低 23.9%，与人类专家的一致性提升 60.5%。
tags:
  - 多模态
---

# VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2510.22373](https://arxiv.org/abs/2510.22373)
- **代码**: [GitHub](https://github.com/HKUSTDial/VisJudgeBench)
- **领域**: 多模态大模型 / 可视化质量评估
- **关键词**: 可视化评估, 美学质量, MLLM-as-a-Judge, 数据可视化, benchmark

## 一句话总结

提出首个面向数据可视化美学与质量评估的综合基准 VisJudge-Bench（3,090 样本，32 种图表类型），并训练 VisJudge 模型，将 MAE 相比 GPT-5 降低 23.9%，与人类专家的一致性提升 60.5%。

## 研究背景与动机

数据可视化是将复杂数据转化为直观洞察的有效方式，其质量取决于三个维度：**保真度**（Fidelity，数据是否准确呈现）、**表达力**（Expressiveness，信息是否清晰传达）和**美学**（Aesthetics，设计是否美观）。然而，现有工作存在明显空白：

**图表问答基准**（如 ChartQA、ChartInsights）只关注图表内容理解，不评估设计质量

**自然图像美学基准**（如 AVA、ArtiMuse）仅关注艺术美感，忽略可视化的核心目的——有效传达数据

**可视化评估基准**（如 VisEval）主要评估 NL2VIS 生成准确性，而非可视化本身的内在设计质量

因此，**缺乏一个系统性框架来衡量 MLLM 在可视化美学与质量评估方面的综合能力**。即便最先进的 GPT-5，在该任务上 MAE 高达 0.553，与人类评分的相关性仅 0.428。

## 方法详解

### 整体框架

VisJudge-Bench 的构建遵循三阶段方法论：**(1) 数据收集与处理** → **(2) 自适应问题生成** → **(3) 专家标注与质量控制**。在基准之上，进一步训练领域专用模型 VisJudge。

### "信达雅"评估框架

借鉴中国翻译理论"信达雅"原则，建立六个可测量的评估维度：

- **保真度 (Fidelity)**：数据保真 — 视觉编码是否准确反映数据，检测轴设置不当、刻度失真、截断基线等误导性设计
- **表达力 (Expressiveness)**：
    - *语义可读性*：用户能否清晰解码图表中的视觉元素
    - *洞察发现*：是否能揭示深层数据模式、趋势或异常值
- **美学 (Aesthetics)**：
    - *设计风格*：设计的创新性和独特性
    - *视觉构图*：空间布局的合理性、元素定位的平衡与秩序
    - *色彩和谐*：配色方案是否在美学与信息传达之间取得平衡

### 数据构建

从 30 万+ 初始图像中经三阶段筛选：

1. **初始过滤**：自动脚本 + 感知哈希去重 → 80,210 候选
2. **自动分类**：GPT-4o 分类 + 人工验证 → 13,220 有效样本
3. **分层采样**：最终 3,090 样本，覆盖单图（1,041）、多图（1,024）、仪表盘（1,025），共 32 种子类型

### 自适应问题生成

利用 GPT-4o 从图表中提取元数据（类型、视觉元素），基于预定义模板生成定制化评分问题和五级评分标准。例如在数据保真度维度，评分标准从"1 分 = 存在截断轴或误导性刻度"到"5 分 = 条形长度严格与显示值成比例"。

### 专家标注与质量控制

- **第一阶段**：603 名众包工人独立评分，每样本 3 人 × 6 维度
- **第二阶段**：基于评分方差的冲突识别与解决机制，包括异常值移除、恶意评分检测
- **第三阶段**：3 位可视化分析专家独立审查，复杂案例通过讨论达成共识

### VisJudge 模型训练

- **数据划分**：70%/10%/20% 训练/验证/测试（2,163/279/648 样本）
- **基础模型**：Qwen2.5-VL (3B/7B)、InternVL3-8B、Llava-v1.6-mistral-7B
- **训练方法**：GRPO 强化学习，复合奖励函数 = 准确度奖励（最小化预测误差）+ 格式奖励（确保结构化输出）
- **参数高效微调**：LoRA，5 个 epoch，学习率 $1 \times 10^{-5}$

## 实验

### 主要结果（MAE ↓）

| 模型 | Overall | Fidelity | Readability | Insight | Design | Composition | Color |
|------|---------|----------|-------------|---------|--------|-------------|-------|
| GPT-5 | 0.553 | 0.862 | 0.781 | 0.778 | 0.649 | 0.699 | 0.682 |
| GPT-4o | 0.610 | 0.988 | 0.806 | 0.744 | 0.609 | 0.695 | 0.657 |
| VisJudge (Qwen2.5-VL-7B) | **0.421** | **0.661** | **0.648** | **0.677** | **0.580** | **0.545** | **0.604** |

### 关键发现

1. **GPT-5 仍不足**：即便最强闭源模型，MAE 仍高达 0.553，相关性仅 0.428，表明通用 MLLM 无法自动获得可视化评估的专业能力
2. **VisJudge 显著缩小差距**：最佳 VisJudge（Qwen2.5-VL-7B）MAE 降至 0.421（↓23.9%），相关性升至 0.687（↑60.5%）
3. **开源模型差距更大**：开源模型普遍 MAE > 0.7，尤其在 Fidelity 和 Expressiveness 维度表现最差
4. **美学评估相对容易**：所有模型在 Aesthetics 三个子维度上表现优于 Fidelity 和 Expressiveness

### 消融实验

- GRPO 强化学习显著优于纯 SFT 训练
- 复合奖励设计（准确度 + 格式）优于单一奖励
- 不同架构和参数规模的模型均受益于微调，验证了跨架构泛化性

## 亮点

- 首个面向可视化美学与质量评估的综合基准，填补了重要空白
- "信达雅"三维评估框架设计精巧，六个子维度覆盖全面
- 3,090 个专家标注样本、32 种图表类型，数据质量高
- GRPO 微调有效，小模型经微调后显著超越 GPT-5
- 揭示了当前 MLLM 在可视化评估上的关键不足

## 局限性

- 评估仅基于视觉层面的保真度，缺乏源数据进行真正的数据-视觉一致性验证
- 样本主要来自网络爬取，可能存在分布偏差
- 仅评估了中小规模开源模型的微调，未探索更大规模模型
- 人类专家标注本身存在主观性，不同标注者之间可能存在偏差

## 相关工作

- **可视化推荐**：Voyager、Draco（规则驱动），VizML、DeepEye（学习驱动）
- **NL2VIS 评估**：nvBench、MatPlotAgent — 关注代码生成而非设计质量
- **MLLM-as-a-Judge**：通用美学评估（AVA）、图表理解（ChartQA）、可视化评估（VisEval）均有局限
- **图像美学评估**：AVA、ArtiMuse — 针对自然图像，不适用于可视化

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个面向可视化质量评估的专用基准
- **技术深度**: ⭐⭐⭐⭐ — 评估框架设计全面，标注流程严谨
- **实验充分度**: ⭐⭐⭐⭐ — 12 个模型系统评测，消融充分
- **实用价值**: ⭐⭐⭐⭐ — 对可视化自动评估有直接推动作用

<!-- RELATED:START -->

## 相关论文

- [RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](../../NeurIPS2025/multimodal_vlm/rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)
- [WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)

<!-- RELATED:END -->
