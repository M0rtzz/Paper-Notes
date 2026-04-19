---
title: >-
  [论文解读] TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models
description: >-
  [ICLR 2026][时序理解] TPRU构建了大规模多图像时序理解数据集（24,750个QA对、126,000张图像），覆盖机器人操作、GUI导航等4个具身场景的3种互补任务（时序排序、下一帧预测、前帧回溯），并通过强化学习微调使7B模型在时序理解上超越GPT-4o。
tags:
  - ICLR 2026
  - 时序理解
  - 程序性推理
  - 多图像理解
  - 强化学习微调
  - MLLM
  - 具身AI
---

# TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models

**会议**: ICLR 2026  
**arXiv**: [2602.18884](https://arxiv.org/abs/2602.18884)  
**代码**: https://github.com/Stephen-gzk/TPRU/  
**领域**: 强化学习 / 多模态大模型  
**关键词**: 时序理解, 程序性推理, 多图像理解, 强化学习微调, MLLM, 具身AI

## 一句话总结
TPRU构建了大规模多图像时序理解数据集（24,750个QA对、126,000张图像），覆盖机器人操作、GUI导航等4个具身场景的3种互补任务（时序排序、下一帧预测、前帧回溯），并通过强化学习微调使7B模型在时序理解上超越GPT-4o。

## 研究背景与动机

**领域现状**：MLLM在单图像任务上表现优异，但小型可部署模型在理解时序和程序性图像序列方面存在严重缺陷。这一能力差距是具身AI部署（机器人、导航、指令执行）的关键瓶颈。

**现有痛点**：(1) 训练范式系统性失败——现有数据集将多图像视为无序集合，忽略了"理解图像集"和"理解图像序列"的关键区别；(2) 社区的响应是创建evaluation-only的benchmark反复确认失败，而非解决根本原因（缺少大规模真实世界序列训练数据）。

**核心矛盾**：资源受限的边缘设备无法部署数百亿参数模型，但小模型在程序性理解上的不足使其无法用于具身场景。这是"模型规模"还是"训练数据"的问题？

**本文目标**：(1) 提供大规模、结构化的时序理解训练集填补训练-测试gap；(2) 验证小型模型通过正确的数据和训练方法能否达到大模型级别的时序理解。

**切入角度**：从4个具身场景采集真实序列数据，设计3种互补任务覆盖时序推理的不同方面，引入困难negative samples迫使模型主动验证而非被动观察。

**核心 idea**：通过结构化数据+RL训练使小模型获得超越大模型的时序程序性理解能力，证明这非规模固有限制而是训练挑战。

## 方法详解

### 整体框架
TPRU包含两部分：TPRU-25k（微调集，24,750样本）和TPRU-Test（461个人工标注评测样本）。数据来源于4个具身场景，通过3阶段pipeline处理：序列过滤→文本描述生成+鲁棒性增强→任务制定。

### 关键设计

1. **数据源设计**:

    - 机器人操作：ShareRobot的planning任务视频帧采样
    - LEGO组装：36个高质量定格动画视频，无运动模糊、状态清晰
    - GUI操作：GUI Odyssey的4步截图序列
    - 具身导航：Habitat模拟环境的有序视觉观察
    - 设计动机：多源多样化确保模型学到通用时序推理而非特定领域的模式

2. **三种互补任务**:

    - **时序排序（Temporal Ordering）**：打乱帧序列，要求模型根据文本描述恢复正确排列——评估对完整时间线的理解
    - **下一帧预测（Next-Frame Prediction）**：给定第1、2、4帧，从候选中选正确的第3帧（干扰项来自相似场景）——模拟agent预判动作后果
    - **前帧回溯（Previous-Frame Review）**：给定后3帧，从候选中选正确的初始帧——理解程序性前提和事件溯源
    - 设计动机：前向预测+后向回溯+全局排序共同培养结构化的程序性动态理解

3. **Negative采样策略**:

    - 功能：创建文本-图像故意不匹配的实例（如"拿起叉子"配"放下刀"的图像）
    - 目标输出："None of the choices provided"
    - 设计动机：迫使模型进行显式跨模态验证，而不仅依赖文本先验

### 训练策略
使用强化学习（RL）微调Qwen2.5-VL模型（3B/7B/32B），聚焦于提升资源受限模型的时序推理能力。RL训练细节和奖励设计见论文附录。

## 实验关键数据

### TPRU-Test 主实验

| 模型 | 参数量 | 准确率 |
|------|--------|--------|
| Qwen2.5-VL-7B (base) | 7B | 50.33% |
| **TPRU-7B** | **7B** | **75.70%** |
| GPT-4o | 闭源 | <75.70% |
| Qwen2.5-VL-72B | 72B | - |
| Gemini-2.5-Flash | 闭源 | - |

TPRU-7B比base版本提升25.37个百分点，显著超越GPT-4o。

### MuirBench 评测

| 模型 | Overall |
|------|---------|
| Qwen2.5-VL-7B (base) | 58.35% |
| **TPRU-7B** | 提升显著 |
| **TPRU-32B** | **68.42%** |
| GPT-4o | 68.00% |
| Qwen2.5-VL-72B | 69.35% |

TPRU-32B超越GPT-4o，接近72B模型水平。Ordering类别提升特别显著。

### LEGO-Puzzles评测
TPRU-7B在LEGO-Puzzles上展现了对多步计划和程序性推理的实质性提升。

### 关键发现
- 时序推理差距不是模型规模的固有限制，而是可以通过针对性数据和RL训练解决的挑战
- 三种任务的互补性至关重要——消融实验显示去掉任何一种任务都会降低性能
- Negative样本对防止幻觉和过度依赖文本先验至关重要
- RL微调比SFT在这类任务上更有效

## 亮点与洞察
- **训练-评测统一**：打破了"创建benchmark→确认失败→再创建benchmark"的循环，同时提供训练数据和评测工具
- **小模型逆袭大模型**：7B模型超越GPT-4o的结果颠覆了"时序理解需要大模型"的假设
- **实用导向设计**：数据来源于真实具身场景而非合成数据，任务设计直接对应agent能力需求
- **RL训练范式**：证明了RL在提升MLLM推理能力方面的有效性，超越传统SFT

## 局限与展望
- TPRU-25k虽然多样但规模相对可再扩展
- 当前聚焦于3-4帧短序列，更长序列的程序性理解有待探索
- 视频理解与多帧图像理解的联系和区别需要进一步研究
- RL训练的reward设计可能有更好的方案

## 相关工作与启发
- **vs Mantis-Instruct/LLaVA-NeXT-Interleave**: 这些数据集将多图像视为无序集合，缺乏系统性的时序结构
- **vs MuirBench/LEGO-Puzzles**: 这些仅是评测benchmark，TPRU同时提供训练数据填补gap
- **vs Jigsaw-R1/MiCo**: 这些用程序化生成的puzzle训练空间推理，TPRU用真实场景训练时序推理

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集设计和三任务框架有创新，RL+时序理解的结合有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准(TPRU/MuirBench/LEGO)、多模型规模、多方法对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，Pipeline图设计好，但数据集描述略冗长
- 价值: ⭐⭐⭐⭐⭐ 对具身AI和可部署MLLM有直接价值，数据集和代码开源

<!-- RELATED:START -->

## 相关论文

- [AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [MergeMix: A Unified Augmentation Paradigm for Visual and Multi-Modal Understanding](mergemix_a_unified_augmentation_paradigm_for_visual_and_multi-modal_understandin.md)

<!-- RELATED:END -->
