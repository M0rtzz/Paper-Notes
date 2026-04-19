---
title: >-
  [论文解读] Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?
description: >-
  [ICLR 2026][图像生成][T2I评测] 提出 T2I-CoReBench，首个同时系统评估 T2I 模型**组合能力**(Composition)和**推理能力**(Reasoning)的综合性基准，涵盖 12 个评估维度、1080 条高难度 prompt 和约 13500 个 checklist 问题，通过对 38 个模型的大规模评测揭示：推理能力远远落后于组合能力，是当前 T2I 生成的核心瓶颈。
tags:
  - ICLR 2026
  - 图像生成
  - T2I评测
  - 组合生成
  - 推理能力
  - benchmark
  - 场景图
---

# Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?

**会议**: ICLR 2026  
**arXiv**: [2509.03516](https://arxiv.org/abs/2509.03516)  
**代码**: [GitHub](https://github.com/) (有，含Leaderboard和Benchmark)  
**领域**: 文本到图像生成 / 评测基准  
**关键词**: T2I评测, 组合生成, 推理能力, benchmark, 场景图

## 一句话总结

提出 T2I-CoReBench，首个同时系统评估 T2I 模型**组合能力**(Composition)和**推理能力**(Reasoning)的综合性基准，涵盖 12 个评估维度、1080 条高难度 prompt 和约 13500 个 checklist 问题，通过对 38 个模型的大规模评测揭示：推理能力远远落后于组合能力，是当前 T2I 生成的核心瓶颈。

## 研究背景与动机

T2I 生成中，文本 prompt 同时包含**显式描述**（需要组合生成的内容）和**隐式线索**（需要推理才能正确生成的内容）。例如"一个成熟的番茄被紧紧握在拳头里"隐含"番茄汁爆出"。这对应两个核心能力：组合（Composition）和推理（Reasoning）。

**现有评测的两大缺陷**：

**缺乏全面性**：多数 benchmark 只评组合或只评推理，且分类体系是启发式的，无法系统覆盖所有维度

**缺乏复杂性**：组合场景只测少量视觉元素（≤5），推理只测简单的一对一因果，无法反映真实世界的高密度场景

**核心切入角度**：借鉴场景图（Scene Graph）结构化组合能力，借鉴哲学推理三分法（演绎/归纳/溯因）结构化推理能力，构建一个兼具全面性和复杂性的 12 维评测体系。

## 方法详解

### 整体框架

T2I-CoReBench 由三部分构成：（1）12 维评估分类体系；（2）高复杂度 prompt + checklist 构建流程；（3）基于 MLLM 的自动评估协议。

### 关键设计

1. **组合能力评估（4 个维度）**：基于场景图的三要素：

    - **MI（Multi-Instance）**：单图中生成约 25 个实例，如"繁忙的现代厨房"
    - **MA（Multi-Attribute）**：单主体绑定约 20 个属性
    - **MR（Multi-Relation）**：场景中约 15 种关系连接
    - **TR（Text Rendering）**：约 15 个文本的内容和布局精确渲染
   
   相比 DPG-Bench 等现有基准，视觉元素密度提升 4-5 倍（~20 vs ~5）。

2. **推理能力评估（8 个维度）**：基于哲学推理三分法：

    - **演绎推理**（前提→结论）：LR 逻辑推理、BR 行为推理、HR 假设推理、PR 过程推理
    - **归纳推理**（模式→规律）：GR 泛化推理、AR 类比推理
    - **溯因推理**（观察→解释）：CR 常识推理、RR 重建推理
   
   引入一对多（一个行为→多个结果）和多对一（多个前提→一个结论）的复杂推理链，突破现有 benchmark 仅测一对一推理的局限。

3. **Prompt + Checklist 构建管线**：

    - 使用 3 个 SOTA 大推理模型（Claude Sonnet 4, Gemini 2.5 Pro, OpenAI o3）辅助生成 prompt，确保多样性和复杂性
    - 每条 prompt 配套 checklist：独立的 yes/no 问题，逐点验证显式和隐式视觉元素
    - 所有样本经严格人工审核

### 损失函数 / 训练策略

本文为评测基准，无训练过程。评估协议使用 Gemini 2.5 Flash 作为 MLLM 评估器，将每个 checklist 问题转化为二值 VQA 任务（"0"=否 / "1"=是），利用 checklist 的原子性设计确保与 MLLM 评估的兼容性。

## 实验关键数据

### 主实验（38 个模型全面评测）

| 模型 | 参数量 | 组合均分 | 推理均分 | 总均分 |
|------|--------|---------|---------|--------|
| FLUX.2-dev | 32B | **84.7** | **54.2** | **64.4** |
| Qwen-Image-2512 | 20B | 83.7 | 51.7 | 62.4 |
| FLUX.2-klein-9B | 9B | 78.0 | 52.0 | 60.6 |
| LongCat-Image | 6B | 70.8 | 54.1 | 59.6 |
| HunyuanImage-3.0 | 80B | 78.9 | 48.6 | 58.7 |
| BAGEL w/ Think | 14B | 39.6 | 41.9 | 41.1 |
| OmniGen2-7B | 7B | 49.9 | 39.4 | 42.9 |
| PixArt-α | 0.6B | 25.0 | 23.7 | 24.1 |
| Janus-Pro-1B | 1B | 35.5 | 13.0 | 20.5 |

| 各维度细分（FLUX.2-dev） | MI | MA | MR | TR | LR | BR | HR | PR | GR | AR | CR | RR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 分数 | 89.5 | 83.4 | 72.7 | 93.3 | 48.4 | 33.5 | 53.4 | 83.1 | 62.3 | 64.3 | 62.1 | 26.9 |

### 消融实验

- 不同 MLLM 评估器（Qwen2.5-VL-72B, Qwen3-VL 系列）与 Gemini 2.5 Flash 的评估结果高度一致
- 使用 3 个不同 LRM 生成 prompt 避免了单一模型偏差

### 关键发现

1. **组合能力稳步提升**：开源模型逐步缩小与闭源的差距，FLUX.2-dev 组合均分达 84.7
2. **推理能力严重落后**：即使最强模型推理均分也远低于组合，FLUX.2-dev 推理仅 54.2 vs 组合 84.7
3. **统一模型的思考模式有帮助但有限**：BAGEL w/ Think 推理从 34.1→41.9，但仍大幅落后于扩散模型
4. **Text Rendering 差异巨大**：FLUX.2-dev 达 93.3，而多数统一/自回归模型低于 10
5. **模型规模并非决定性因素**：6B 的 LongCat-Image 推理均分 54.1 超过 80B 的 HunyuanImage-3.0（48.6）

## 亮点与洞察

- 首次将哲学推理三分法（演绎/归纳/溯因）系统化引入 T2I 评测，提供了理论有据的分类框架
- 高密度组合（~20 元素/prompt）和高强度推理（一对多/多对一）的设计更贴近真实应用场景
- 揭示了一个重要发现：**T2I 模型更擅长"画"（组合），但不擅长"想"（推理）**——正如标题所说
- Checklist 设计将复杂评测分解为原子问题，提升了评估可靠性

## 局限与展望

- MLLM 评估器本身可能存在偏差，尽管做了人工验证
- 推理维度的 prompt 难度定义（一对多/多对一）较为粗粒度，可进一步细化推理链长度
- 仅评估静态图像生成，未涉及视频或交互式生成
- 基准的 1080 条 prompt 规模适中，但每个维度仅 90 条，统计效力可能有限

## 相关工作与启发

- **DPG-Bench / ConceptMix**：组合评测的先驱，但复杂度不够
- **R2I-Bench / WISE**：推理评测的探索，但只覆盖部分推理类型
- **GenAI-Bench**：通用评测但缺乏系统分类
- 启发：可以用本 benchmark 的推理维度指导 T2I 模型的推理能力训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统整合组合+推理评测，分类体系有理论基础，但 benchmark 构建方法本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 38 个模型覆盖面广，多种评估器交叉验证，分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图表精美，taxonomy 解释充分
- 价值: ⭐⭐⭐⭐⭐ 填补了 T2I 推理评测的空白，对社区有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [Aligning Text to Image in Diffusion Models is Easier Than You Think](../../NeurIPS2025/image_generation/aligning_text_to_image_in_diffusion_models_is_easier_than_you_think.md)
- [Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)
- [RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)

<!-- RELATED:END -->
