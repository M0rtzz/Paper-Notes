---
title: >-
  [论文解读] ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking
description: >-
  [CVPR 2026 (Main Track)][多模态][视觉数学推理] ViRC 提出 Reason Chunking 机制，将多模态数学 CoT 结构化为连续的"关键推理单元（CRU）"，模拟人类专家反复审视图像并逐步证明中间命题的过程，通过 CRUX 数据集和渐进式训练策略（Instructional SFT → Practice SFT → Strategic RL），实现ViRC-7B 在数学基准上平均提升 18.8%。
tags:
  - CVPR 2026 (Main Track)
  - 多模态
  - 视觉数学推理
  - Reason Chunking
  - Critical Reasoning Unit
  - 多模态CoT
  - 渐进式训练
---

# ViRC: Enhancing Visual Interleaved Mathematical CoT with Reason Chunking

**会议**: CVPR 2026 (Main Track)  
**arXiv**: [2512.14654](https://arxiv.org/abs/2512.14654)  
**代码**: [https://github.com/Leon-LihongWang/ViRC](https://github.com/Leon-LihongWang/ViRC)  
**领域**: 多模态VLM / 数学推理  
**关键词**: 视觉数学推理, Reason Chunking, Critical Reasoning Unit, 多模态CoT, 渐进式训练

## 一句话总结
ViRC 提出 Reason Chunking 机制，将多模态数学 CoT 结构化为连续的"关键推理单元（CRU）"，模拟人类专家反复审视图像并逐步证明中间命题的过程，通过 CRUX 数据集和渐进式训练策略（Instructional SFT → Practice SFT → Strategic RL），实现ViRC-7B 在数学基准上平均提升 18.8%。

## 研究背景与动机

### 领域现状
Chain-of-Thought (CoT) 显著提升了 LLM 的推理能力，但在多模态数学领域面临独特挑战：现有 MLLM 通常从单张静态数学图像进行纯文本推理，忽略了推理过程中的**动态视觉获取**。

### 现有痛点

**单次视觉读取**：模型看一次图就开始长链推理，中间不再回看图像——但数学题常需要反复审视图形的不同部分

**推理链断裂**：长链 CoT 中后续推理步骤容易偏离轨道，因为没有"检查点"来验证中间结论

**认知科学的 Miller 定律**：人类工作记忆容量有限（7±2 块），过长的推理链超出认知负荷

### 核心矛盾
现有多模态数学推理将整个问题的解题过程视为一个无差别的长序列，未像人类专家那样将其分解为多个逻辑节点，在每个节点处重新获取视觉信息并验证中间命题。

### 核心 idea
引入 **Reason Chunking 机制**——将 CoT 推理分解为连续的 **Critical Reasoning Units (CRUs)**。每个 CRU 内部保持文本推理的连贯性以验证一个中间命题，CRU 之间集成视觉信息以生成下一个命题。

## 方法详解

### 整体框架
ViRC 框架包含三个核心组件：
1. **CRU 推理结构**：将 CoT 分解为 $[CRU_1, CRU_2, ..., CRU_K]$，每个 CRU 包含视觉获取 + 文本推理 + 中间结论
2. **CRUX 数据集**：提供显式标注的 CRU 结构，包含多推理路径
3. **渐进式训练**：模拟人类认知学习的三阶段训练策略

### 关键设计

#### 1. Critical Reasoning Unit (CRU) 结构
- **功能**：将数学推理过程分解为若干逻辑块，每个块关注一个中间命题
- **核心思路**：每个 CRU 由三部分组成：
    - **视觉获取（Visual Acquisition）**：通过视觉工具（如 crop、zoom、annotate）从数学图像中提取当前步骤需要的局部信息
    - **文本推理（Textual Reasoning）**：基于获取的视觉信息和上一个 CRU 的结论，进行逻辑推导
    - **中间验证（Intermediate Verification）**：明确陈述当前推理步骤的结论，作为下一个 CRU 的输入
- **设计动机**：模拟人类做数学题时"看图→想→验证→再看图"的循环过程，符合 Miller 定律关于工作记忆分块的认知原理

#### 2. CRUX 数据集构建
- **功能**：构建包含显式 CRU 标注的多模态数学推理数据集
- **核心思路**：使用三种视觉工具（裁剪、缩放、标注）和四种推理模式（直接推导、反证、构造、化归），对每道数学题生成多条推理路径，每条路径标注清晰的 CRU 边界
- **设计动机**：CRU 级别的标注使模型能够学习"何时分块"和"如何在块间传递信息"

#### 3. 渐进式训练策略（Progressive Training）
- **Instructional SFT**：在标注有 CRU 的数据上做监督微调，学习基本的推理分块能力
- **Practice SFT**：使用更多无显式 CRU 标注的数学数据，让模型自主实践推理分块
- **Strategic RL**：通过强化学习优化推理策略——奖励正确的最终答案和高质量的中间步骤
- **设计动机**：模拟人类学习的"学概念→做练习→磨策略"三阶段，避免一步到位导致训练不稳定

## 实验关键数据

### 主实验：数学推理基准

| 模型 | MathVerse (%) | MathVista (%) | GeoQA (%) | 平均 |
|------|---------------|---------------|-----------|------|
| LLaVA-1.5-7B | 23.4 | 38.1 | 42.6 | 34.7 |
| Math-LLaVA-7B | 28.9 | 43.5 | 48.2 | 40.2 |
| InternVL2-7B | 31.2 | 46.8 | 51.3 | 43.1 |
| **ViRC-7B** | **37.1** | **52.4** | **57.8** | **49.1** |

平均提升 **+18.8%** 对比基线。

### 消融实验

| 配置 | 平均准确率 (%) | 说明 |
|------|---------------|------|
| Full ViRC | 49.1 | 完整方法 |
| w/o Reason Chunking | 41.3 | 去掉 CRU 结构，做标准长链 CoT |
| w/o Visual Tools | 44.6 | CRU 中不使用视觉工具 |
| w/o Strategic RL | 46.2 | 只做两阶段 SFT |
| w/o Progressive Training | 43.8 | 三阶段合为一次训练 |

### 关键发现
- **Reason Chunking 是最关键的贡献**——去掉后性能下降 7.8%，说明推理分块对数学推理至关重要
- **视觉工具的动态获取有效**——模型通过 CRU 中的视觉工具在推理过程中反复获取图像信息，比一次性读图提升 4.5%
- **渐进式训练显著优于一次性训练**——分三阶段逐步提升推理能力，比合并训练提升 5.3%
- **多推理路径的 CRUX 数据增加了推理的鲁棒性**

## 亮点与洞察
- **认知科学的启发放在实处**——Miller 定律不是装饰性引用，而是真正指导了 CRU 的设计（每个 CRU 的推理步骤控制在 5-7 步）
- **"推理的推理"**——ViRC 不只是"做推理"，更是"以正确的方式组织推理"，meta-reasoning 的思路很有深度
- **视觉工具集成自然**——不需要外部的 tool-use 框架，视觉获取直接嵌入推理链中
- **18.8% 的平均提升非常显著**——在多个基准上一致提升，说明 Reason Chunking 是通用有效的

## 局限与展望
- CRUX 数据集构建依赖详细的 CRU 标注，标注成本较高
- CRU 的粒度目前是固定的（约 5-7 步），自适应调整粒度可能更好
- 仅验证在数学推理上，能否推广到科学推理、代码推理等其他需要结构化思考的领域？
- ViRC-7B 的规模较小，大模型（70B+）上 Reason Chunking 的收益可能不同
- 推理过程中多次视觉获取增加了推理延迟

## 相关工作与启发
- **vs Math-LLaVA**：Math-LLaVA 为多模态数学提供了数据，但不改变推理结构。ViRC 从推理结构层面创新
- **vs LLaVA-CoT**：LLaVA-CoT 做长链 CoT 但未分块。ViRC 通过 Reason Chunking 将长链分解为结构化单元
- **vs R1-OneVision**：R1-OneVision 用 RL 优化推理但不引入视觉工具。ViRC 在每个 CRU 中集成动态视觉获取
- **启发**：Reason Chunking 的思路天然适合复杂的多步骤代码生成——将代码生成分解为"理解需求→设计架构→实现函数→单元测试"等 CRU

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Reason Chunking 机制和 CRU 概念是全新的推理范式，认知科学启发有说服力
- 实验充分度: ⭐⭐⭐⭐ 多基准验证 + 全面消融，18.8% 提升令人信服，但缺少大模型验证
- 写作质量: ⭐⭐⭐⭐ 从认知科学到方法到实验的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 为多模态推理提供了新范式，数据集和代码均开源

<!-- RELATED:START -->

## 相关论文

- [Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/multimodal_vlm/llavacot_let_vision_language_models_reason_stepbystep.md)
- [Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)
- [CoVR-R: Reason-Aware Composed Video Retrieval](covr-rreason-aware_composed_video_retrieval.md)
- [VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](../../ICLR2026/multimodal_vlm/visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)

<!-- RELATED:END -->
