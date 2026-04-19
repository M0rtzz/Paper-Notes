---
title: >-
  [论文解读] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning
description: >-
  [ICCV 2025][目标检测][意图驱动视觉感知] VisRL是首个将强化学习应用于意图驱动视觉感知的框架，通过迭代DPO训练让大多模态模型学会根据查询意图自主选择关注区域（预测bounding box），无需昂贵的中间bounding box标注即可实现比SFT更强的视觉推理能力。
tags:
  - ICCV 2025
  - 目标检测
  - 意图驱动视觉感知
  - 强化学习
  - Visual CoT
  - DPO
  - 大多模态模型
---

# VisRL: Intention-Driven Visual Perception via Reinforced Reasoning

**会议**: ICCV 2025  
**arXiv**: [2503.07523](https://arxiv.org/abs/2503.07523)  
**代码**: https://github.com/zhangquanchen/VisRL  
**领域**: 目标检测 / 多模态推理  
**关键词**: 意图驱动视觉感知, 强化学习, Visual CoT, DPO, 大多模态模型

## 一句话总结
VisRL是首个将强化学习应用于意图驱动视觉感知的框架，通过迭代DPO训练让大多模态模型学会根据查询意图自主选择关注区域（预测bounding box），无需昂贵的中间bounding box标注即可实现比SFT更强的视觉推理能力。

## 研究背景与动机

**领域现状**：大多模态模型（LMM，如LLaVA、Qwen-VL）通过端到端推理回答关于图像的问题。近年来Visual Chain-of-Thought（Visual CoT）类方法引入了显式推理步骤——模型先预测一个关注区域（bounding box），裁剪该区域输入模型，再结合原图和裁剪图回答问题。

**现有痛点**：Visual CoT严重依赖有监督训练，需要为每个query-image对标注中间步骤的bounding box。同一张图像可能对应截然不同的关注区域（取决于不同查询意图），标注复杂度呈组合爆炸增长，根本无法覆盖所有可能的意图-区域对。

**核心矛盾**：SFT需要密集的<意图, 关注区域>对标注 → 标注成本高且不可穷举 → 模型在有限标注上训练 → 泛化能力受限。

**本文目标** 在不需要bounding box标注的情况下，让模型学会意图驱动的视觉感知。

**切入角度**：类比人类视觉学习——人类不通过密集标注学习"看哪里"，而是通过试错与环境交互，逐步发展出自适应聚焦相关区域的能力。用RL替代SFT更合理。

**核心 idea**：用强化学习（迭代DPO）优化视觉推理过程中的焦点区域选择，利用任务奖励信号替代bounding box标注实现可扩展的意图驱动视觉感知。

## 方法详解

### 整体框架
VisRL分两个阶段：(1) SFT热身——用少量带bbox标注的数据训练模型学会"先看再答"的推理格式；(2) RL训练——在大规模无标注数据上，通过迭代的"数据生成→优化"循环，用step-level DPO不断提升模型。RL阶段不依赖任何外部模型或标注，完全由模型自身完成数据合成和评分。

### 关键设计

1. **SFT热身（Warm-up）**:

    - 功能：让模型学会Visual CoT推理格式（先输出bbox再回答）
    - 核心思路：使用少量标注数据进行SFT，训练模型按"predict bbox → crop → answer"流程生成回答
    - 设计动机：RL训练需要模型已具备基本推理格式能力作为起点

2. **迭代DPO框架**:

    - 功能：RL训练的核心循环，包含数据生成和模型优化两步交替
    - 核心思路：每轮迭代中，模型为每个问题生成多个不同推理轨迹（不同bbox+不同答案），通过最终答案的正确性构建偏好对（preference pairs），用step-level DPO优化模型。迭代多轮持续提升
    - 设计动机：单轮DPO不够，迭代式训练让模型逐步探索更好策略

3. **多样性控制器（Diversity Controller）**:

    - 功能：确保生成的bbox覆盖多种可能的关注区域
    - 核心思路：在数据生成阶段，通过调节采样温度和增加随机扰动，使生成的bbox具有足够多样性
    - 设计动机：如果生成的bbox都很相似，构建的偏好对质量低。多样性是RL探索（exploration）的关键

4. **Step-Level DPO**:

    - 功能：在推理过程的每一步都进行优化
    - 核心思路：视觉推理包含两步（选区域+回答），step-level DPO将偏好学习分解到每一步，确保模型同时学好"选哪个区域"和"如何回答"
    - 设计动机：标准DPO只比较整条轨迹的好坏，可能出现"选了不好的区域但碰巧答对"的信号混淆

5. **难度过滤机制**:

    - 功能：筛选合适难度的问题和最有效的偏好对
    - 核心思路：只保留"部分正确部分错误"的问题，过简单或过难的都不适合构建偏好对
    - 设计动机：适当难度的问题提供最大学习信号（类似课程学习）

### 损失函数 / 训练策略
- SFT阶段：标准next-token prediction loss
- RL阶段：Step-level DPO loss，分别在bbox预测步和答案生成步计算偏好损失
- 迭代训练：每轮用当前模型生成新数据，避免off-policy问题

## 实验关键数据

### 主实验

| 方法 | HR-Bench (4K) | V*Bench | TextVQA | 平均 |
|------|-------------|---------|---------|------|
| LLaVA-1.5 (baseline) | 52.3 | 61.8 | 58.4 | 57.5 |
| Visual CoT (SFT) | 55.1 | 65.2 | 61.7 | 60.7 |
| **VisRL** | **58.9** | **68.4** | **64.5** | **63.9** |

### 消融实验

| 配置 | HR-Bench | V*Bench | 说明 |
|------|---------|---------|------|
| Full VisRL | 58.9 | 68.4 | 完整模型 |
| w/o step-level DPO (轨迹级) | 56.2 | 65.8 | step-level关键 |
| w/o diversity controller | 55.8 | 64.9 | 多样性很重要 |
| w/o difficulty filtering | 57.1 | 66.3 | 过滤有帮助 |
| 仅SFT (更多数据) | 55.5 | 65.5 | 数据量增加也不如RL |

### 关键发现
- VisRL在多个benchmark上一致性地超越SFT baseline，RL范式对视觉推理更有效
- Step-level DPO贡献最大（去掉后掉2.7%），验证了在推理每一步都优化的必要性
- Diversity controller对性能影响大，说明RL中的探索对学习视觉注意力至关重要
- 泛化性强——在不同base LMM（LLaVA vs Qwen-VL）上都获得一致提升
- 不需额外bbox标注，使用大规模数据训练后远超使用密集标注的SFT方法

## 亮点与洞察
- **首个RL+视觉感知**：VisRL是第一个将RL应用到意图驱动视觉感知问题的工作，开辟了新研究方向
- **Step-level DPO**：将DPO从轨迹级扩展到步骤级，对多步推理任务更合理。可迁移到任何multi-step reasoning
- **无标注可扩展**：RL阶段完全不需bbox标注，模型自我生成数据+自我评分，天然可扩展到任意规模数据
- **模型无关性**：作为训练框架可应用于不同base LMM

## 局限与展望
- SFT热身仍需少量带标注数据，完全零标注方案有待探索
- 目前只预测一个bbox，对需要关注多个区域的复杂场景可能不够
- 迭代DPO训练计算成本较高
- 评价奖励仅基于最终答案正确性，对开放式问题难以定义验证函数
- 可尝试与GRPO等更新RL算法结合

## 相关工作与启发
- **vs Visual CoT**: Visual CoT用SFT训练需密集bbox标注。VisRL用RL替代SFT，移除标注依赖且性能更好
- **vs DeepSeek-R1**: R1在语言推理中展示RL威力，VisRL将此范式迁移到多模态视觉推理
- **vs Visual-RFT**: Visual-RFT关注分类和检测等最终任务，不涉及"意图驱动焦点选择"维度。VisRL更关注推理过程本身

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个RL+意图驱动视觉感知，step-level DPO设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 多benchmark验证，消融充分，跨模型泛化测试
- 写作质量: ⭐⭐⭐⭐ 动机推导流畅，方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 开辟RL+视觉推理新方向，可扩展性强

<!-- RELATED:START -->

## 相关论文

- [UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction](../../ICML2025/object_detection/ui-vision_a_desktop-centric_gui_benchmark_for_visual_perception_and_interaction.md)
- [Visual-RFT: Visual Reinforcement Fine-Tuning](visual-rft_visual_reinforcement_fine-tuning.md)
- [Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](../../ICLR2026/object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)
- [ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)
- [Connecting the Dots: Training-Free Visual Grounding via Agentic Reasoning](../../AAAI2026/object_detection/connecting_the_dots_training-free_visual_grounding_via_agent.md)

<!-- RELATED:END -->
