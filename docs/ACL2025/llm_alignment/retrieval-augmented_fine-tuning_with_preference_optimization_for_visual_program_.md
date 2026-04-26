---
title: >-
  [论文解读] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation
description: >-
  [ACL 2025][LLM对齐][视觉编程语言] 本文针对工业视觉编程语言（Ladder Diagram）的自动生成任务，提出了一种两阶段训练策略：先通过检索增强微调（Retrieval-Augmented Fine-Tuning）利用子程序复用特性，再通过基于图编辑操作构造偏好对的 DPO 训练进一步提升准确性，在真实 LD 数据上将程序级准确率提升超过 10%。
tags:
  - ACL 2025
  - LLM对齐
  - 视觉编程语言
  - 梯形图生成
  - 检索增强微调
  - 偏好优化
  - 工业自动化
---

# Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation

**会议**: ACL 2025  
**arXiv**: [2502.16529](https://arxiv.org/abs/2502.16529)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: 视觉编程语言、梯形图生成、检索增强微调、偏好优化、工业自动化

## 一句话总结

本文针对工业视觉编程语言（Ladder Diagram）的自动生成任务，提出了一种两阶段训练策略：先通过检索增强微调（Retrieval-Augmented Fine-Tuning）利用子程序复用特性，再通过基于图编辑操作构造偏好对的 DPO 训练进一步提升准确性，在真实 LD 数据上将程序级准确率提升超过 10%。

## 研究背景与动机

**领域现状**：视觉编程语言（VPL）通过图形化界面让用户创建程序，广泛应用于各类场景。近年来研究尝试使用 LLM 从自然语言指令生成 VPL 代码，特别是通过 prompting 方法已取得一定成果。

**现有痛点**：对于工业级 VPL，如梯形图（Ladder Diagram, LD）——工业自动化中的核心编程语言——prompting 方法效果有限。LD 涉及大量特定领域的配置项（继电器、定时器、计数器等复杂组件），这些细节难以在单个 prompt 中完整捕获。此外，LD 代码具有高度的结构化特征，与自然语言文本差异很大。

**核心矛盾**：Prompting 方法依赖 LLM 的内在知识，但工业 VPL 的领域特殊性使得通用 LLM 难以仅凭 prompt 生成正确代码。而简单的监督微调（SFT）虽然优于 prompting，却未能充分利用 LD 代码中子程序频繁复用的特点，也缺少对错误模式的负向反馈。

**本文目标**：设计一种针对工业 VPL 生成的训练策略，既能利用子程序复用模式，又能通过偏好学习减少典型错误。

**切入角度**：作者观察到 LD 程序中存在大量重复出现的子程序（subroutine）模块，这是工业编程的固有特性。同时，LD 的图结构特性使得可以通过图编辑操作系统化地构造"接近正确但有缺陷"的负样本。

**核心 idea**：通过检索增强微调利用子程序复用性提升基础生成质量，再通过图编辑操作自动构造偏好对进行 DPO 训练，双管齐下提升程序级准确率。

## 方法详解

### 整体框架

输入为自然语言用户指令，输出为 LD 代码（以结构化格式表示）。训练分两个阶段：(1) 检索增强微调阶段——对每个训练样本检索相似的子程序片段拼接到输入中，进行监督微调；(2) DPO 阶段——利用图编辑操作从正确 LD 代码生成"略有错误"的变体作为 rejected 样本，与原正确代码配对进行偏好优化。推理时不需要偏好数据，直接生成即可。

### 关键设计

1. **检索增强微调（Retrieval-Augmented Fine-Tuning, RAFT）**:

    - 功能：利用 LD 中子程序频繁复用的特性，在微调时为每个训练样本检索相关子程序片段作为上下文
    - 核心思路：构建子程序索引库，对每个训练指令检索 top-k 相似子程序，将其拼接到输入中一并送给模型。这类似于 RAG 但应用在微调阶段——模型学习的是"在有参考子程序的条件下生成完整程序"。检索采用基于代码结构相似度的方法，而非简单的文本匹配
    - 设计动机：工业 LD 程序中约 60-70% 的模块是重复出现的标准子程序，检索这些已有模式可以极大降低从零生成的难度，让模型专注于组合和定制化

2. **基于图编辑的偏好对构造（Graph-Edit-Based Preference Pair Generation）**:

    - 功能：自动化地为 DPO 训练生成高质量的偏好数据对
    - 核心思路：LD 代码本质上是一种图结构（节点为元件，边为连接关系）。作者定义了几种图编辑操作——节点删除、节点替换、边删除、边重定向等——对正确的 LD 图施加少量编辑生成"接近正确但有细微错误"的变体。这些变体作为 rejected 样本，原正确代码作为 chosen 样本。编辑距离越小，构造的偏好对越有区分度
    - 设计动机：传统 DPO 需要人工标注偏好数据或使用模型自身采样生成正负样本，成本高且质量不可控。利用 LD 的图结构特性，可以系统性地、低成本地构造semantically 有意义的负样本，且编辑操作与模型生成中的真实错误模式高度吻合

3. **两阶段渐进训练**:

    - 功能：先打好基础再精细化调节
    - 核心思路：第一阶段 RAFT 让模型学会利用检索到的子程序生成正确代码；第二阶段 DPO 在此基础上通过偏好学习进一步修正模型的错误倾向。两阶段共享同一个底层 LLM（如 CodeLlama-7B），DPO 是在 RAFT 的 checkpoint 上继续训练
    - 设计动机：直接做 DPO 而不先进行 RAFT，模型基础生成能力不足，DPO 的效果有限；先 RAFT 后 DPO 的渐进策略让两种技术各发挥所长

### 损失函数

第一阶段使用标准的交叉熵损失进行监督微调。第二阶段使用 DPO 损失：$L_{DPO} = -\mathbb{E}[\log \sigma(\beta \cdot (\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))]$，其中 $y_w$ 为 chosen（正确 LD），$y_l$ 为 rejected（图编辑后的错误 LD），$\pi_{ref}$ 为 RAFT 阶段的 checkpoint。

## 实验关键数据

### 主实验

在真实工业 LD 数据集上的程序级准确率（Program-level Accuracy, PA）对比：

| 方法 | 模型 | PA (%) | 提升 |
|------|------|--------|------|
| Few-shot Prompting | GPT-4 | 42.3 | baseline |
| Few-shot Prompting | CodeLlama-34B | 38.7 | - |
| SFT | CodeLlama-7B | 55.1 | +12.8 |
| RAFT (检索增强微调) | CodeLlama-7B | 61.4 | +19.1 |
| RAFT + DPO (本文) | CodeLlama-7B | 65.8 | +23.5 |

### 消融实验

| 配置 | PA (%) | 说明 |
|------|--------|------|
| SFT only | 55.1 | 纯监督微调基线 |
| SFT + DPO (随机负样本) | 57.3 | 随机构造负样本效果有限 |
| SFT + DPO (图编辑负样本) | 59.8 | 图编辑负样本优于随机 |
| RAFT only | 61.4 | 检索增强贡献最大 |
| RAFT + DPO (随机负样本) | 63.1 | RAFT 基础上加 DPO |
| RAFT + DPO (图编辑负样本) | 65.8 | 完整方法效果最佳 |

### 关键发现

- 训练方法（SFT）在较小模型上即可大幅超越大模型的 prompting，说明 LD 这类特殊领域必须依赖微调
- RAFT 对性能提升最大（+6.3% over SFT），检索子程序的复用性是 LD 生成的关键先验
- 图编辑构造的偏好对比随机负样本平均高 2-3%，说明结构化的负样本更有效
- 即使用 7B 模型，本文方法也超越了 GPT-4 的 few-shot 结果 23.5 个百分点

## 亮点与洞察

- **将图编辑操作用于构造 DPO 偏好对是非常巧妙的想法**：充分利用了 LD 代码的图结构特性，避免了人工标注的高成本。这种"利用数据本身结构特征来构造训练信号"的思路可迁移到其他结构化输出任务（如 SQL 生成、电路设计等）
- **检索增强微调（RAFT）在工业代码生成中的应用有实际价值**：工业代码中子程序复用率极高，这一洞察直接指导了方法设计，也揭示了工业代码生成与通用代码生成的本质区别
- **小模型微调 >> 大模型 prompting 的发现**在特殊领域中具有重要实践指导意义

## 局限与展望

- 实验仅在单一 VPL（Ladder Diagram）上验证，是否适用于其他工业 VPL（如 Function Block Diagram）未知
- 检索增强在推理时需要额外的检索延迟，对实时部署场景可能有影响
- 图编辑操作的类型和程度需要人工设计，缺乏自适应机制
- 数据集规模和多样性有限（真实工业数据的获取本身就是挑战）

## 相关工作与启发

- **vs CodeLlama/GPT-4 直接 prompting**: 这些通用模型通过 prompting 对工业 VPL 的理解有限，本文证明了微调的必要性
- **vs 标准 SFT**: 纯监督微调忽略了子程序复用特性和错误模式学习，本文的两阶段策略更为全面
- **vs RAG in NLP**: RAG 通常用在推理时，本文将检索增强引入训练阶段（RAFT），让模型内化"如何利用检索到的参考"的能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 图编辑构造偏好对的想法很有创意，RAFT + DPO 的组合在工业 VPL 上是首次
- 实验充分度: ⭐⭐⭐ 消融实验较完整，但仅限单一数据集和单一 VPL 类型
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法动机阐述充分
- 价值: ⭐⭐⭐⭐ 对工业自动化领域有直接应用价值，方法论对结构化代码生成有普适启发

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] RPO: Retrieval Preference Optimization for Robust Retrieval-Augmented Generation](rpo_retrieval_preference_optimization_for_robust_retrieval-augmented_generation.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning_simplifying_alignment_into_single_process.md)
- [\[ACL 2025\] Enhancing Safe and Controllable Protein Generation via Knowledge Preference Optimization](kpo_protein_safety.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)

<!-- RELATED:END -->
