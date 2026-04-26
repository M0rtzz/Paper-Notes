---
title: >-
  [论文解读] Visual-RFT: Visual Reinforcement Fine-Tuning
description: >-
  [ICCV 2025][目标检测][强化微调] Visual-RFT将DeepSeek R1的强化学习+可验证奖励(RLVR)范式从数学/代码领域扩展到视觉感知任务，设计了IoU奖励（目标检测）和CLS奖励（分类）等任务特异的可验证奖励函数，在细粒度分类、少样本检测、推理定位等任务上以极少数据大幅超越SFT。
tags:
  - ICCV 2025
  - 目标检测
  - 强化微调
  - GRPO
  - 可验证奖励
  - 视觉感知
  - 少样本学习
---

# Visual-RFT: Visual Reinforcement Fine-Tuning

**会议**: ICCV 2025  
**arXiv**: [2503.01785](https://arxiv.org/abs/2503.01785)  
**代码**: https://github.com/Liuziyu77/Visual-RFT  
**领域**: 目标检测 / 多模态推理  
**关键词**: 强化微调, GRPO, 可验证奖励, 视觉感知, 少样本学习

## 一句话总结
Visual-RFT将DeepSeek R1的强化学习+可验证奖励(RLVR)范式从数学/代码领域扩展到视觉感知任务，设计了IoU奖励（目标检测）和CLS奖励（分类）等任务特异的可验证奖励函数，在细粒度分类、少样本检测、推理定位等任务上以极少数据大幅超越SFT。

## 研究背景与动机

**领域现状**：OpenAI o1和DeepSeek R1展示了大推理模型通过强化学习+可验证奖励进行微调（RFT）的强大能力。RFT的核心优势是数据效率——不像SFT需要大量高质量标注数据，RFT通过试错学习，只需少量样本就能在特定领域取得优秀效果。

**现有痛点**：之前共识是RFT只适用于有明确可验证答案的任务（数学答案对错、代码测试用例）。视觉感知任务（检测、分类）的输出是bbox坐标或类别名称，如何设计可验证奖励不直观。因此RL在LVLM的post-training中主要用于减少幻觉，而非提升视觉感知。

**核心矛盾**：SFT范式是数据饥饿的——需要大量高质量标注数据来"模仿"正确答案。在数据稀缺场景（医学影像、稀有物种等），SFT在少样本条件下甚至可能导致性能下降。

**本文目标** 证明RFT可扩展到视觉感知任务；设计各种视觉任务的可验证奖励函数；在有限数据条件下超越SFT。

**切入角度**：视觉感知任务虽然输出形式不同于数学，但也有客观评判标准：检测IoU、分类匹配。这些可作为可验证奖励基础。

**核心 idea**：为视觉感知任务设计task-specific的可验证奖励函数，将R1-style RLVR迁移到视觉领域，实现数据高效的视觉微调。

## 方法详解

### 整体框架
给定图像和问题输入，策略模型（LVLM）生成多个包含思维链推理和最终答案的回复。用设计好的可验证奖励函数评估每个回复质量，通过GRPO策略梯度优化算法更新模型。不需要额外reward model，奖励由规则直接计算。

### 关键设计

1. **IoU可验证奖励（检测任务）**:

    - 功能：评估模型预测的bounding box与GT的匹配质量
    - 核心思路：对每个预测框，计算它与所有GT框的IoU取最大值。总奖励综合precision和recall。还加入格式奖励确保输出符合结构化格式
    - 设计动机：与数学的0/1奖励不同，检测任务需要连续奖励信号。IoU天然提供了预测质量的连续度量且计算成本极低

2. **CLS可验证奖励（分类任务）**:

    - 功能：评估分类预测正确性
    - 核心思路：精确匹配——预测类别与GT一致则奖励1，否则0
    - 设计动机：分类答案有客观标准，直接用规则验证即可

3. **思维链推理格式**:

    - 功能：让LVLM在给出答案前先输出推理过程
    - 核心思路：prompt中要求模型在`<think>...</think>`标签内输出推理过程，`<answer>...</answer>`标签内输出最终答案。检测任务要求输出结构化的位置和置信度
    - 设计动机：思维链推理显著提升模型推理能力——分类时分析细节特征，检测时推理空间位置

4. **GRPO策略优化**:

    - 功能：用组相对策略优化算法更新LVLM
    - 核心思路：对每个问题从当前策略采样G个回复，计算各自奖励。GRPO不需额外critic model，直接比较组内回复的相对好坏来计算优势函数，用PPO-style clipped objective更新策略，加KL散度正则
    - 设计动机：GRPO比PPO轻量（无需critic model），DeepSeek R1已验证其有效性

### 损失函数 / 训练策略
- GRPO目标函数：最大化 $\mathbb{E}_{o \sim \pi_\theta(q)}[R(q,o) - \beta \text{KL}[\pi_\theta \| \pi_{ref}]]$
- 每个问题采样G个回复组成group，用group内相对奖励计算advantage

## 实验关键数据

### 主实验

| 任务 | 数据量 | Visual-RFT | SFT | 提升 |
|------|-------|-----------|-----|------|
| 细粒度分类 (1-shot) | ~100 | +24.3% acc | -4.3% acc | RFT远超SFT |
| 少样本检测 COCO 2-shot | 极少 | +21.9 mAP | baseline | 显著提升 |
| 少样本检测 LVIS | 极少 | +15.4 mAP | baseline | 显著提升 |
| 开放词汇检测 COCO new (2B) | - | 31.3 mAP | 9.8 mAP | +21.5 |
| 开放词汇检测 LVIS rare (2B) | - | 20.7 mAP | 2.7 mAP | +18.0 |

### 消融实验

| 配置 | 分类Acc | 检测mAP | 说明 |
|------|--------|---------|------|
| Visual-RFT (full) | 最高 | 最高 | 完整模型 |
| w/o 思维链 | 下降明显 | 下降 | 推理过程对细粒度任务关键 |
| SFT (同等数据) | 显著低 | 显著低 | 少样本下SFT不如RFT |
| 更多SFT数据 | 仍低于RFT | 仍低于RFT | 单纯加数据也难追上RFT |

### 关键发现
- **数据效率悬殊**：one-shot设置下RFT提升24.3%而SFT下降4.3%，差距达28.6%！极少样本条件下SFT的"模仿"范式完全失效
- **思维链推理起关键作用**：模型在检测时推理目标空间位置，在分类时分析关键视觉特征
- **泛化能力强**：在开放词汇检测中快速迁移到新类别，包括LVIS稀有类别
- **可验证奖励简单有效**：IoU和CLS奖励都是简单规则计算，无需训练reward model

## 亮点与洞察
- **范式转变**：从SFT的"数据规模化"转向RFT的"奖励函数设计"，是视觉模型训练的重要范式转变
- **少样本杀手锏**：在数据极稀缺场景（医学、稀有物种等），Visual-RFT展现了巨大潜力
- **IoU奖励设计**：将检测评估指标直接作为RL奖励，巧妙桥接评估和训练。任何可计算的评估指标都可能成为可验证奖励
- **完全开源**：训练代码、数据、评估脚本全部公开

## 局限与展望
- GRPO需为每个问题采样多个回复，训练效率不如SFT
- 目前只验证了检测和分类，对分割等连续输出任务的奖励函数设计更复杂
- 思维链推理虽提升准确性但增加推理延迟
- 在数据充足场景下RFT相比SFT的优势可能缩小
- 奖励函数设计需人工针对每个任务，自动化奖励设计是未来方向

## 相关工作与启发
- **vs DeepSeek R1**: R1在纯语言领域用RLVR。Visual-RFT成功扩展到多模态视觉任务，证明RLVR通用性
- **vs VisRL**: VisRL关注"在哪里看"的决策过程，Visual-RFT关注最终视觉感知结果。两者互补
- **vs SFT（传统）**: SFT是数据饥饿的模仿学习，Visual-RFT是数据高效的奖励驱动学习

## 评分
- 新颖性: ⭐⭐⭐⭐ R1范式迁移到视觉领域，idea相对直接但具有开创性
- 实验充分度: ⭐⭐⭐⭐⭐ 任务覆盖广（分类+检测+定位+开放词汇），设置丰富
- 写作质量: ⭐⭐⭐⭐ 动机明确，结果展示清晰
- 价值: ⭐⭐⭐⭐⭐ 范式转变性工作，完全开源，对社区影响大

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] FG-CLIP: Fine-Grained Visual and Textual Alignment](../../ICML2025/object_detection/fg-clip_fine-grained_visual_and_textual_alignment.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICCV 2025\] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)

<!-- RELATED:END -->
