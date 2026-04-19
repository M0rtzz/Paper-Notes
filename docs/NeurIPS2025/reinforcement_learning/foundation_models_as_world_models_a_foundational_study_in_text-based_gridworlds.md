---
title: >-
  [论文解读] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds
description: >-
  [NeurIPS 2025][基础模型] 系统性评估了基础模型（LLM）作为零样本世界模型（FWM）和直接决策智能体（FA）在文本网格世界中的表现，揭示了两种策略在确定性/随机性环境中的互补优势。
tags:
  - NeurIPS 2025
  - 基础模型
  - 世界模型
  - 网格世界
  - 大语言模型
  - 样本效率
---

# Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds

**会议**: NeurIPS 2025  
**arXiv**: [2509.15915](https://arxiv.org/abs/2509.15915)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 基础模型, 世界模型, 网格世界, 大语言模型, 样本效率

## 一句话总结

系统性评估了基础模型（LLM）作为零样本世界模型（FWM）和直接决策智能体（FA）在文本网格世界中的表现，揭示了两种策略在确定性/随机性环境中的互补优势。

## 研究背景与动机

强化学习（RL）智能体在复杂游戏中表现出色，但其样本效率极低，通常需要数百万次交互才能学到有效策略。基础模型（FMs）拥有广泛的预训练知识和推理能力，是提升样本效率的天然候选者。然而，如何将FMs有效地融入RL框架仍是一个开放问题。

先前工作多将FMs用作辅助组件（如奖励塑形、高层规划器），本文则研究更直接的集成方式：

- **基础世界模型（FWM）**：利用FM作为零样本模拟器，为传统RL智能体生成预训练交互数据
- **基础智能体（FA）**：直接利用FM作为低级动作选择器，在每一步生成动作

为了隔离FMs的核心模拟和推理能力，作者设计了一系列文本网格世界环境作为受控实验平台，避免视觉感知的干扰。

## 方法详解

### 整体框架

本文围绕两大策略展开评估：FWM方法将LLM当作环境模拟器来预训练RL智能体的策略，FA方法则直接利用LLM的推理能力进行决策。评估在两种网格世界设置中进行：

1. **确定性+全观测**：奖励位置固定在右上角，智能体已知目标位置
2. **随机性+部分观测**：奖励位置每个episode随机采样，智能体不知道目标位置

网格世界定义为 $n \times n$ 方格，智能体起始于左下角 $[0,0]$，最大步数为 $2n^2$，动作空间为 $\{up, down, left, right\}$。

### 关键设计

#### 基础世界模型（FWM）

世界模型由转移函数和奖励函数组成。通过设计Prompt模板将两个函数交给LLM实现：

**转移函数设计**——两种复杂度的提示模板：

| 模板 | 描述 | 目的 |
|------|------|------|
| $T_{\text{minimal}}$ | 仅提供最简描述，依赖LLM对网格世界的先验知识 | 测试推理能力下限 |
| $T$ | 详细描述转移规则和数学约束（边界检查、坐标增减方向） | 测试逻辑推理能力 |

**奖励函数**——列出两种可能结果，让LLM判断当前位置与奖励位置的关系：到达奖励位置返回 reward=1, terminal=True；否则返回 reward=0, terminal=False。

**温度设置**——对确定性函数使用 $\tau=0$ 鼓励确定性输出；对随机元素（如奖励位置采样）使用 $\tau=1.8$ 引入多样性。

#### 基础智能体（FA）

设计了三种不同引导级别的Prompt策略：

| 策略 | 缩写 | 描述 |
|------|------|------|
| Action Only | AO | 仅给出当前状态和信息，要求直接输出动作 |
| Simple Plan | SP | 鼓励先推理当前应做什么，再基于计划决定动作 |
| Focused Plan | FP | 明确指示利用历史记忆确定下一个目标位置，分析最佳动作 |

三种策略测试了LLM在不同引导级别下利用推理和规划进行决策的能力。在随机设置中，智能体需利用记忆追踪已访问位置来系统搜索网格。

### 损失函数 / 训练策略

**FWM+RL融合**：使用GPT-4的 $T+R$ 提示作为最佳FWM，先在模拟交互上预训练RL智能体，再在真实环境中微调：

- 确定性设置：预训练TRPO智能体，1500步
- 随机设置：预训练RecurrentPPO智能体（处理部分观测性），100万步
- 评估：每125步用1个episode评估，5个随机种子

**评估的LLM**：GPT-3.5, GPT-4, Gemma 2b, Gemma 7b, Gemini 1.0, Gemini 1.5——覆盖不同模型类型、规模和代际。FA评估100个独立episode，$\tau=0$。

## 实验关键数据

### 主实验（模拟精度）

$5 \times 5$ 网格，$|\mathcal{S}| \cdot |\mathcal{A}| = 25 \times 4 = 100$ 个转移预测：

| 模型 | $T$ 精度 | $T_{\text{minimal}}$ 精度 | $T+R$ 精度 |
|------|---------|-------------------------|------------|
| GPT-4 | **100%** | 97% | **100%** |
| Gemini 1.5 | **100%** | 95% | 99% |
| GPT-3.5 | 96% | 82% | 93% |
| Gemma 7b | 78% | 65% | 72% |
| Gemma 2b | 61% | 48% | 55% |

**决策性能对比**（确定性 vs 随机设置）：

| 方法 | 确定性 | 随机 |
|------|--------|------|
| GPT-4 FA (最佳提示) | **1.0** (零样本最优) | ~0.6 |
| Gemini 1.5 FA | **1.0** | ~0.5 |
| RL from scratch | 1.0 (需数百步) | ~0.8 (需大量交互) |
| **RL + FWM (GPT-4)** | **1.0** (预训练即最优) | **~0.9** (快速收敛) |
| Gemma 2b/7b FA | 0.0 | ~0.1 |

### 消融实验

**随机元素模拟分布质量**（1000个随机网格位置）：
- 小模型（Gemma 2b, GPT-3.5, Gemini 1.0）：能覆盖所有位置，但分布不均匀
- 大模型（GPT-4, Gemini 1.5）：集中在少数位置，覆盖不全
- 结论：小模型的高输出方差有利于模拟大采样空间，大模型适合模拟小采样空间

**提示策略消融**（随机设置）：
- 小模型：AO/SP/FP差异不大，整体性能低
- 大模型：SP和FP显著优于AO，说明增强推理引导能帮助大模型更好利用记忆和规划

### 关键发现

1. **确定性环境中FA完胜**：FA直接利用推理能力零样本解决，远优于需数百次交互的RL智能体
2. **随机环境中FWM+RL完胜**：FA普遍无法系统搜索整个网格；FWM+RL大幅提升样本效率
3. **模型容量 = 性能**：LLM能力提升直接转化为更好的FWM和FA表现
4. **FWM迁移鲁棒**：即使FWM无法完美模拟随机分布，预训练策略仍平滑迁移至真实环境

## 亮点与洞察

- **互补性发现**：简单确定性任务适合FA直接决策，复杂随机任务适合FWM+RL结合
- **边界效应**：LLM在网格边界处的模拟最容易出错——常将边界约束错误应用到其他动作
- **小模型的意外优势**：模拟随机元素（大采样空间分布）时，小模型的高方差反而有利
- **零样本预训练范式可行**：FWM无需训练即可提供足够好的模拟数据进行RL预训练

## 局限与展望

1. 实验限于文本网格世界，未扩展到视觉或具身环境
2. 未与从头学习的世界模型对比，仅关注预训练FM的零样本能力
3. FWM+RL仅采用简单预训练-微调方案，未探索Dyna风格的交互式架构
4. 未测试更新一代LLM（如GPT-4o、Claude等）
5. 可探索FWM与FA的结合使用，或用环境数据优化FWM

## 相关工作与启发

- **WorldCoder** (Tang et al., 2024)：LLM迭代生成Python代码表示环境动态，属于神经符号方法
- **Genie 3** (DeepMind, 2025)：多模态基础世界模型可实时生成交互式3D环境
- **RAFA** (Liu et al., 2023)：LLM想象未来场景 + 策略执行近期动作
- **MOTIF** (Klissarov et al., 2024)：基于LLM偏好的内在奖励模型，改善稀疏奖励探索
- 启发：FWM可与规划算法结合用于在线决策，未来视觉基础世界模型可扩展到视觉环境

## 评分

- **创新性**: ★★★★☆ — 首次系统性对比FWM和FA两种直接集成路线
- **实验充分性**: ★★★☆☆ — 环境过于简单，仅限网格世界
- **实用价值**: ★★★☆☆ — 提供基础性洞察但距实际应用有距离
- **写作质量**: ★★★★☆ — 结构清晰，实验设计合理

<!-- RELATED:START -->

## 相关论文

- [SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](../../ICML2025/reinforcement_learning/sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)
- [Self-Improving Embodied Foundation Models](self-improving_embodied_foundation_models.md)
- [Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches](exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)
- [Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)

<!-- RELATED:END -->
