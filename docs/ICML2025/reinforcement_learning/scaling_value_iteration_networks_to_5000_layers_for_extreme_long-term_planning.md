---
title: >-
  [论文解读] Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning
description: >-
  [ICML2025][Value Iteration Network] 提出 Dynamic Transition VIN (DT-VIN)，通过引入动态转移核增强隐式 MDP 的表征能力，并设计自适应 highway loss 缓解梯度消失，将 VIN 成功扩展至 5000 层，在 $100 \times 100$ 迷宫中实现 1800 步长期规划（原版 VIN 仅支持 $25 \times 25$ 迷宫中 120 步规划）。
tags:
  - ICML2025
  - Value Iteration Network
  - 长期规划
  - 动态转移核
  - 自适应Highway Loss
  - 深度网络训练
---

# Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning

**会议**: ICML2025  
**arXiv**: [2406.08404](https://arxiv.org/abs/2406.08404)  
**作者**: Yuhui Wang, Qingyuan Wu, Dylan R. Ashley, Francesco Faccio, Weida Li, Chao Huang, Jürgen Schmidhuber
**代码**: 待确认  
**领域**: reinforcement_learning  
**关键词**: Value Iteration Network, 长期规划, 动态转移核, 自适应Highway Loss, 深度网络训练

## 一句话总结

提出 Dynamic Transition VIN (DT-VIN)，通过引入动态转移核增强隐式 MDP 的表征能力，并设计自适应 highway loss 缓解梯度消失，将 VIN 成功扩展至 5000 层，在 $100 \times 100$ 迷宫中实现 1800 步长期规划（原版 VIN 仅支持 $25 \times 25$ 迷宫中 120 步规划）。

## 研究背景与动机

### 问题背景
规划（Planning）是人工智能的核心问题之一——给定目标，找到一系列动作使智能体到达目标。传统搜索算法（如 A*）需要精确环境模型，在未知环境或大规模连续状态空间中不适用。Value Iteration Network（VIN, Tamar et al., 2016）是一种端到端可微的规划网络，将可微的规划模块嵌入深度神经网络，在隐式 MDP (latent MDP) 上执行值迭代，因此具有对未见任务的强泛化能力。

### 已有工作的不足
- **VIN 无法处理大规模长期规划任务**：在 $100 \times 100$ 迷宫中，VIN 成功率低于 40%；在 $35 \times 35$ 迷宫中，当需要超过 60 步规划时成功率降至 0%
- **隐式 MDP 表征能力不足**：原始 VIN 使用**静态转移核**（所有状态共享同一组卷积核），无法表达复杂环境中不同区域的不同转移动态
- **梯度消失问题严重**：VIN 的规划模块本质上是循环展开的深度网络，层数增加会导致严重的梯度消失/爆炸，限制了网络深度和规划步数
- **已有改进有限**：GPPN（Lee et al., 2018）和 Highway VIN（Wang et al., 2024a）有一定改善，但仍无法实现数千步级别的长期规划

### 核心动机
弥合 VIN 架构中"规划复杂度"与"网络表征能力"之间的鸿沟——通过两个关键改进使 VIN 能够在真正大规模的环境中执行极长期的规划。

## 方法详解

### 整体框架：DT-VIN 架构

DT-VIN 在原始 VIN 基础上做了两个核心改进，整体流程如下：

1. **编码器**：将观测（如迷宫图像）编码为隐式状态表示，包括奖励图 $\bar{R}$ 和状态特征
2. **动态转移核生成器**：根据输入观测动态生成每个状态的转移核参数（而非使用全局共享的静态核）
3. **规划模块**：在隐式 MDP 上执行 $K$ 步值迭代（$K$ 可达 5000），每一步使用动态转移核进行 Bellman 更新
4. **自适应 highway loss**：在训练过程中根据实际需要的规划步数自适应地在不同深度设置跳跃连接损失
5. **策略输出**：从值函数中提取动作，输出最终策略

### 关键设计1：动态转移核 (Dynamic Transition Kernel)

**问题分析**：原始 VIN 的规划模块使用单组卷积核作为转移概率，所有隐式状态共享相同的转移动态。这相当于假设环境在所有位置都有相同的转移特性——显然不成立。例如在迷宫中，墙壁处和通道处的可达性完全不同。

**解决方案**：DT-VIN 引入**动态转移核**，为每个隐式状态生成独立的转移参数：
- 使用一个条件网络（conditioned on 观测），为每个空间位置 $(i, j)$ 生成对应的转移核参数
- 这使得不同位置可以有不同的转移动态，极大增强了隐式 MDP 的表征能力
- 类似于动态卷积 (dynamic convolution) 的思想，但应用在值迭代的 Bellman 更新中

**效果**：动态转移核使得隐式 MDP 能够捕捉复杂环境中的局部结构差异，是 DT-VIN 能够处理大规模环境的关键因素。

### 关键设计2：自适应 Highway Loss

**问题分析**：VIN 的规划模块展开 $K$ 步值迭代，等价于一个 $K$ 层深度网络。当 $K$ 增大到数千时，从最终损失传回的梯度会因链式法则而指数级衰减（梯度消失），导致浅层参数几乎不更新。

**解决方案**：设计自适应 highway loss，在中间层构建到最终损失的跳跃连接：
- 在训练过程中，不仅在第 $K$ 步计算损失，还在中间某些步数处计算辅助损失
- 跳跃连接的位置根据**实际需要的规划步数**自适应选择（而非固定间隔）
- 这确保了梯度可以通过较短的路径传回，有效缓解了梯度消失
- 不同于 Highway Network 的固定门控机制，本方法在损失层面构建 skip connection，更灵活

**与已有方法的关系**：Highway VIN（Wang et al., 2024a）已识别到网络深度与长期规划的关系并引入了 highway 结构，但 DT-VIN 的"自适应"机制使其能进一步扩展到 5000 层的深度级别。

### 训练策略

- **模仿学习 (Imitation Learning)**：使用 A* 等搜索算法生成的最优路径作为专家演示进行监督训练
- **课程学习思想**：通过控制训练样本中规划步数的分布，从短期规划逐步过渡到长期规划
- **大规模训练**：5000 层网络需要精心的训练策略，包括学习率调度和 batch 设计

## 实验关键数据

### 实验1：2D 迷宫导航——不同迷宫尺寸下的成功率

| 方法 | $15 \times 15$ | $25 \times 25$ | $35 \times 35$ | $50 \times 50$ | $100 \times 100$ |
|------|:---:|:---:|:---:|:---:|:---:|
| VIN (Tamar et al., 2016) | ~99% | ~75% | ~50% | <40% | <40% |
| GPPN (Lee et al., 2018) | ~99% | ~85% | ~60% | — | — |
| Highway VIN (Wang et al., 2024a) | ~99% | ~90% | ~70% | — | — |
| **DT-VIN (本文)** | **~99%** | **~98%** | **~95%** | **~90%** | **~85%** |

DT-VIN 在所有尺寸上显著超越基线，尤其在大迷宫上优势明显。原始 VIN 在 $100 \times 100$ 迷宫上近乎完全失败，而 DT-VIN 仍保持高成功率。

### 实验2：$35 \times 35$ 迷宫——不同最短路径长度下的成功率

| 方法 | 规划步 ≤ 20 | 规划步 20–40 | 规划步 40–60 | 规划步 > 60 |
|------|:---:|:---:|:---:|:---:|
| VIN | ~99% | ~80% | ~40% | ~0% |
| **DT-VIN** | **~99%** | **~98%** | **~95%** | **~90%** |

VIN 在超过 60 步的长期规划中完全崩溃，而 DT-VIN 保持 90% 以上的成功率。

### 实验3：扩展性验证

| 指标 | VIN | Highway VIN | DT-VIN |
|------|:---:|:---:|:---:|
| 最大可训练层数 | ~150 | ~500 | **5000** |
| 最大有效规划步数 | 120 | ~300 | **1800** |
| 支持的最大迷宫尺寸 | $25 \times 25$ | $35 \times 35$ | **$100 \times 100$** |

### 实验4：多任务验证

| 任务 | 类型 | 规划步数要求 | DT-VIN 表现 |
|------|------|:---:|------|
| 2D 迷宫导航 | 离散, 视觉输入 | 数百~1800 | 成功率 85%+ |
| 3D 迷宫导航 (VizDoom) | 连续视觉 | 数百 | 有效解决 |
| 连续控制 | 连续动作空间 | 数百 | 有效解决 |
| 月球车导航 (真实场景) | 真实地形数据 | 数百~千 | 成功规划路径 |

## 亮点与洞察

- **规模突破**：首次将 VIN 架构扩展到 5000 层，实现了从百步级到千步级规划的质变。这不是简单的"把网络加深"，而是系统性地解决了深度和表征两个瓶颈
- **动态转移核设计精妙**：用动态卷积的思想重新设计隐式 MDP 的转移结构，让不同空间位置具有不同的转移动态，提升了模型对复杂环境的表达能力
- **自适应 highway loss 的实用价值**：相比固定间隔的辅助损失，自适应机制根据规划复杂度动态调整，更高效地利用了梯度信号
- **端到端规划的潜力**：不同于 MuZero/Dreamer 等需要显式学习环境模型的方法，DT-VIN 在隐式 MDP 上直接规划，保持了 VIN 家族的端到端优势和强泛化性
- **从理论到实践的闭环**：不仅在 toy 迷宫上验证，还在 VizDoom 3D 环境、连续控制和真实月球车导航数据上展示了实际价值

## 局限与展望

- **计算开销大**：5000 层的前向/反向传播计算代价高昂，即使有 highway loss 缓解训练问题，推理时仍需完整前向传播
- **依赖专家演示**：模仿学习范式需要最优路径标签（如 A* 生成），限制了在无法获取专家解的任务上的应用
- **隐式 MDP 可解释性差**：动态转移核增强了表征能力，但隐式 MDP 中学到的状态和转移不具有明确物理含义
- **仅验证了网格结构任务**：尽管测试了多种任务，但核心评估仍集中在具有空间结构的导航问题上，对非空间规划任务（如组合优化、任务调度）的适用性未知
- **与基于模型的 RL 方法对比不足**：未与 MuZero、Dreamer 等主流基于模型的规划方法做系统对比
- **动态核生成的参数效率**：为每个位置生成独立转移核，参数量随状态空间规模增长，在极大状态空间中可能面临内存瓶颈

## 相关工作与启发

- **VIN (Tamar et al., 2016)**：本文的基础架构，首次提出将值迭代嵌入深度网络
- **GPPN (Lee et al., 2018)**：通过改进规划模块的结构在一定程度上扩展了 VIN 的能力
- **Highway VIN (Wang et al., 2024a)**：引入 highway 结构缓解梯度消失，是 DT-VIN 的直接前身
- **Dreamer (Hafner et al., 2020–2023)**：在显式学习的世界模型中进行规划，与 VIN 的隐式规划形成互补
- **MuZero (Schrittwieser et al., 2020)**：基于学习的搜索规划，在棋类游戏中表现卓越
- **Predictron (Silver et al., 2017)**：在可微环境模型中进行前瞻规划的早期工作
- **深度网络训练技术**：Highway Networks (Srivastava et al., 2015)、ResNet (He et al., 2016) 等对梯度流的改进为自适应 highway loss 提供了灵感

**启发**：VIN 家族展示了"将算法结构（值迭代）直接嵌入网络架构"的强大范式。DT-VIN 的成功表明，通过精心的架构设计，可以将端到端可微规划扩展到此前认为不可行的规模，暗示了未来可能出现更多"算法+深度学习"的深度融合。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 动态转移核和自适应 highway loss 是有意义的创新，但整体框架仍在 VIN 范式内
- 实验充分度: ⭐⭐⭐⭐ — 覆盖了 2D/3D 迷宫、连续控制和真实世界导航任务，但缺少与 MuZero/Dreamer 等的对比
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，问题定义精确，图表直观
- 价值: ⭐⭐⭐⭐ — 将端到端规划扩展到新的规模级别，对理解深度规划网络有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](../../ICLR2026/reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [\[AAAI 2026\] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation](../../AAAI2026/reinforcement_learning/intention-guided_cognitive_reasoning_for_egocentric_long-term_action_anticipatio.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](../../NeurIPS2025/reinforcement_learning/a_theory_of_multi-agent_generative_flow_networks.md)

<!-- RELATED:END -->
