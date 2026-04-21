---
title: >-
  [论文解读] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning
description: >-
  [NeurIPS 2025 (Datasets and Benchmarks Track)][多智能体合作] 构建了首个面向具身多智能体合作的层次化基准VIKI-Bench（含智能体激活、任务规划、轨迹感知三个层级），并提出两阶段框架VIKI-R（CoT示范微调+多级奖励RL），在多种机器人形态和多视角视觉观测下实现显著超越基线的合作表现，RL阶段涌现出组合式协作模式。
tags:
  - NeurIPS 2025 (Datasets and Benchmarks Track)
  - 多智能体合作
  - VIKI-Bench
  - VLM微调
  - Chain-of-Thought
  - 多级奖励
  - 异构机器人
---

# VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning

**会议**: NeurIPS 2025 (Datasets and Benchmarks Track)  
**arXiv**: [2506.09049](https://arxiv.org/abs/2506.09049)  
**代码**: 有（Project Page）  
**领域**: 强化学习 / 具身多智能体  
**关键词**: 多智能体合作, VIKI-Bench, VLM微调, Chain-of-Thought, 多级奖励, 异构机器人

## 一句话总结
构建了首个面向具身多智能体合作的层次化基准VIKI-Bench（含智能体激活、任务规划、轨迹感知三个层级），并提出两阶段框架VIKI-R（CoT示范微调+多级奖励RL），在多种机器人形态和多视角视觉观测下实现显著超越基线的合作表现，RL阶段涌现出组合式协作模式。

## 研究背景与动机

**领域现状**：协调多个具身智能体在动态环境中完成合作任务是AI的核心挑战。最近的研究利用LLM做多智能体规划取得进展，少数工作开始探索VLM做视觉推理驱动的多智能体合作。

**现有痛点**：(1) 现有基于VLM的多智能体方法仅支持单一类型的机器人形态，缺乏对异构智能体（不同类型的机器人具有不同能力）的支持。(2) 缺乏系统性的评测基准——现有benchmark要么不涉及多智能体，要么不涉及视觉推理，要么不涉及异构形态。(3) 纯LLM方法缺乏视觉感知根基，而纯RL方法又缺乏高级语义推理能力。

**核心矛盾**：多智能体合作需要同时具备视觉感知（理解当前场景）、语义推理（规划任务）和协作策略（分配角色、避免冲突），现有方法通常只覆盖其中一两个方面。

**本文目标** (1) 构建一个结构化的多智能体合作评测标准，能系统性评估从感知到规划到执行的全链条能力。(2) 开发一个能利用VLM的视觉推理能力、再通过RL优化合作策略的统一框架。

**切入角度**：将多智能体合作问题分解为三个层次（激活、规划、感知），每个层次独立可评测，整体构成完整的合作能力金字塔。

**核心 idea**：用层次化benchmark和两阶段VLM-RL框架（先学推理格式再RL优化策略）系统化地解决异构具身智能体的视觉驱动合作问题。

## 方法详解

### 整体框架
VIKI-Bench定义了三个评测层级：**Level 1 - 智能体激活**（判断哪些智能体应参与当前任务）、**Level 2 - 任务规划**（基于视觉观测生成合作行动方案）、**Level 3 - 轨迹感知**（根据多视角观测理解智能体的执行轨迹和状态）。每个层级包含多样的机器人形态（地面、飞行、操作等）、多视角视觉输入和结构化的监督信号。

VIKI-R框架分两个阶段训练：**Stage 1 - CoT监督微调**（在带有Chain-of-Thought标注的示范数据上微调预训练VLM，学习推理格式和基础合作策略）；**Stage 2 - 多级奖励RL**（用强化学习在多级别的奖励信号下进一步优化VLM，使其学会更优的合作模式）。

### 关键设计

1. **层次化评测体系（VIKI-Bench）**:

    - 功能：提供从低级感知到高级规划的全链条合作能力评测
    - 核心思路：Level 1测试智能体选择能力（给定任务描述和可用智能体列表，输出应激活的子集）；Level 2测试任务分配和行动序列生成（给定视觉场景和任务目标，输出每个智能体的行动方案）；Level 3测试对执行过程的理解（给定多视角视频，回答关于智能体行为和状态的问题）
    - 设计动机：将复杂的多智能体合作分解为可独立评测的子能力，避免端到端评测中无法定位瓶颈的问题。同时支持异构形态和多视角输入，比现有benchmark更接近真实场景

2. **CoT标注示范微调（Stage 1）**:

    - 功能：让VLM学习结构化的合作推理模式
    - 核心思路：收集大量多智能体合作的演示数据，为每个示范添加Chain-of-Thought标注（先分析场景、识别子任务、评估智能体能力、分配任务、生成行动序列）。用这些带推理过程的示范对预训练VLM进行监督微调
    - 设计动机：预训练VLM虽有视觉理解和推理能力，但缺乏多智能体合作的格式化输出能力。CoT微调既教会模型输出格式，又传递了基础的合作知识（如能力匹配、任务分解）

3. **多级奖励强化学习（Stage 2）**:

    - 功能：在CoT微调基础上通过RL进一步优化合作策略
    - 核心思路：设计与三个评测层级对应的多级奖励信号——智能体激活的正确性奖励、任务规划的合理性奖励、轨迹预测的准确性奖励。用RL算法联合优化VLM在所有层级上的表现
    - 设计动机：CoT微调受限于示范数据的覆盖范围，RL允许模型探索示范中未出现的合作策略。多级奖励确保模型不会只优化某一层级而忽略其他层级。实验表明RL阶段涌现出了CoT微调中不存在的组合式协作模式

### 损失函数 / 训练策略
Stage 1使用标准的序列到序列交叉熵损失对CoT标注进行微调。Stage 2使用多级奖励信号驱动的RL训练，每个层级的奖励独立计算并按权重加总。

## 实验关键数据

### 主实验

| 方法 | Level 1 激活 | Level 2 规划 | Level 3 感知 | 总体 |
|------|-------------|-------------|-------------|------|
| GPT-4V (zero-shot) | 基线水平 | 基线水平 | 基线水平 | 基线 |
| LLaVA | 较弱 | 较弱 | 较弱 | 低于GPT-4V |
| VIKI-R (CoT only) | 显著提升 | 显著提升 | 显著提升 | 大幅超越基线 |
| **VIKI-R (CoT+RL)** | **最优** | **最优** | **最优** | **所有层级最优** |

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| Full VIKI-R (CoT+RL) | 最优 | 完整两阶段框架 |
| 仅CoT微调 | 次优 | 缺少RL的策略探索 |
| 仅RL（无CoT预训练） | 较差 | 缺少推理格式基础 |
| 单级奖励RL | 部分退化 | 只优化一个层级导致其他退化 |

### 关键发现
- **两阶段缺一不可**：仅CoT微调提供推理基础但策略有限，仅RL无法收敛到好的策略因为缺乏结构化输出的先验
- **RL涌现组合式协作**：RL训练后模型展现出CoT示范中未出现的协作模式——例如异构智能体的动态角色切换、多步任务的流水线并行——这是supervised learning无法产生的
- **异构形态是关键挑战**：不同类型机器人（地面/飞行/操作）有不同的动作空间和能力边界，模型需要理解每种形态的特性才能合理分配任务

## 亮点与洞察
- **层次化评测设计**非常系统：将模糊的"多智能体合作能力"分解为激活-规划-感知三个可量化的层级，既方便诊断也为后续研究提供了clear targets。这种层次分解策略可迁移到其他复杂AI任务的benchmark设计
- **CoT到RL的两阶段训练范式**巧妙地结合了supervised learning的稳定性和RL的探索性：先用CoT教格式和基础策略，再用RL突破示范数据的上限。这提供了"先模仿后超越"的通用范式
- **RL阶段的emergent coordination**是最有趣的发现：组合式协作模式的涌现说明RL+VLM不仅学到explicit rewards对应的行为，还发展出了隐含的协调能力

## 局限与展望
- 作为Datasets and Benchmarks Track论文，技术方法部分相对标准（CoT SFT + RL是已有范式），创新主要在benchmark构建
- 基准中的任务场景可能有限——真实世界的多智能体合作涉及更多不确定性（故障恢复、动态环境变化等）
- 缺乏与非VLM方法（如传统MARL方法）的深入对比，难以判断VLM带来的视觉推理优势有多大
- 计算开销未详细讨论——在大规模VLM上做RL训练的成本可能限制实际应用

## 相关工作与启发
- **vs LLM-based multi-agent (如CAMEL, AutoGen)**: 这些方法基于纯文本LLM做规划，缺乏视觉感知。VIKI-R通过VLM将视觉输入纳入决策循环
- **vs 传统MARL**: 传统方法需要大量环境交互和手工奖励设计，VIKI-R利用VLM的预训练知识和CoT推理大幅减少交互需求
- **vs RoboTHOR/Habitat等环境**: 这些是单智能体模拟环境，VIKI-Bench针对多智能体合作设计了层次化评测

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个面向异构具身多智能体合作的层次化视觉推理基准，benchmark贡献突出
- 实验充分度: ⭐⭐⭐⭐ 多层级、多基线对比和消融分析充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，层次化设计易于理解
- 价值: ⭐⭐⭐⭐ 为具身多智能体合作提供了标准化的评测平台和baseline方法

<!-- RELATED:START -->

## 相关论文

- [Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination](../../ICML2025/reinforcement_learning/cross-environment_cooperation_enables_zero-shot_multi-agent_coordination.md)
- [Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [Mean-Field Sampling for Cooperative Multi-Agent Reinforcement Learning](mean-field_sampling_for_cooperative_multi-agent_reinforcement_learning.md)
- [Empirical Study on Robustness and Resilience in Cooperative Multi-Agent Reinforcement Learning](empirical_study_on_robustness_and_resilience_in_cooperative_multi-agent_reinforc.md)
- [Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)

<!-- RELATED:END -->
