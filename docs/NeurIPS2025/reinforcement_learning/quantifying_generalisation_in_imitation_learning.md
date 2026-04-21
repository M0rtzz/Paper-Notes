---
title: >-
  [论文解读] Quantifying Generalisation in Imitation Learning
description: >-
  [NeurIPS 2025][模仿学习] 本文提出 Labyrinth 基准环境，通过可控的迷宫结构变化实现训练与评估数据的严格分离，揭示了当前模仿学习方法在结构泛化上的严重不足（最佳方法在测试集仅 5% 成功率），为模仿学习的泛化评估提供了系统性工具。
tags:
  - NeurIPS 2025
  - 模仿学习
  - 泛化评估
  - 基准环境
  - 迷宫导航
  - benchmark
---

# Quantifying Generalisation in Imitation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.24784](https://arxiv.org/abs/2509.24784)  
**代码**: [https://github.com/NathanGavenski/Labyrinth](https://github.com/NathanGavenski/Labyrinth)  
**领域**: 强化学习  
**关键词**: 模仿学习, 泛化评估, 基准环境, 迷宫导航, benchmark

## 一句话总结
本文提出 Labyrinth 基准环境，通过可控的迷宫结构变化实现训练与评估数据的严格分离，揭示了当前模仿学习方法在结构泛化上的严重不足（最佳方法在测试集仅 5% 成功率），为模仿学习的泛化评估提供了系统性工具。

## 研究背景与动机
模仿学习位于强化学习与监督学习的交汇处：训练时像监督学习一样利用观测数据，评估时像 RL 一样通过环境交互。然而，当前广泛使用的模仿学习 benchmark 存在根本性缺陷——**训练和评估数据之间缺乏足够的差异化**，无法有效测试泛化能力。

具体问题表现在：

**经典控制任务（CartPole, MountainCar）过于简单**：状态空间低维，动作离散有限。作者发现在 MountainCar 中，仅复制一条专家轨迹的动作序列就能在 100 个不同初始化上达到"解决"标准

**连续控制任务（Hopper, HalfCheetah）缺乏状态抽象和行为可解释性**：不可能知道任意状态下的最优动作；100K 初始化之间的曼哈顿距离极小（如 Hopper 仅 0.22），导致训练和测试几乎无差异

**Atari 游戏训练和测试环境完全相同**，无法分离训练与测试数据

核心矛盾：现有 benchmark 上的高性能可能只是记忆（memorisation）而非泛化（generalisation）的体现。作者认为，一个合格的泛化测试环境需要：（1）任务足够有挑战性；（2）训练和评估之间有显著变化；（3）对这些变化有精确控制；（4）支持行为调试和检查。

## 方法详解

### 整体框架
Labyrinth 是一个基于图结构的离散迷宫导航环境，形式定义为一个图 $G$，节点代表格子，边代表连通路径（移除边=添加墙）。智能体从起点 $s_0$ 出发到达目标 $g$，动作为离散的上下左右。环境支持全观测（看到完整迷宫）和部分观测（仅看到周围区域）两种模式。

### 关键设计
1. **可控的结构变化与数据分离**:

    - 三种起止点设置：用户指定、biased（左下角到右上角）、unbiased（随机但保证最小距离 $d(s_0, g) = |x_{s_0} - x_g| + |y_{s_0} - y_g|$）
    - 基于图哈希保证每个结构在训练/验证/测试集中唯一
    - 支持三种泛化测试：不同结构同起止点、同结构不同起点、同结构不同起止点
    - 设计动机：精确控制环境差异的哪个因素在变化，隔离泛化失败的具体原因

2. **任务变体与复杂度扩展**:

    - **钥匙与门（Key and Door）**：需先拾取钥匙 $g_k$ 开门 $g_d$ 再到终点 $g$，测试子目标序列执行能力
    - **冰面（Ice Floor）**：踩到冰面即失败，测试安全约束下的路径规划
    - 部分可观测（Partially Observable）：只能看到智能体周围的结构
    - 设计动机：渐进增加难度，测试不同维度的泛化能力

3. **精确的最优行为基准**:

    - 核心公式——标准化奖励函数：
    $r_i = \begin{cases} \frac{-0.1}{width \times height} & \text{未到达目标} \\ 1 + |\tau_s| \times \frac{0.1}{width \times height} & \text{到达目标} \end{cases}$
    - 使用 Johnson 算法求解所有从 $s_0$ 到 $g$ 的路径，可精确得到每个状态的最优动作
    - 最短路径的累积奖励恒为 1，独立于迷宫大小
    - 设计动机：完全已知最优策略使得泛化分析可以精确到每个状态的决策质量

### 损失函数 / 训练策略
Labyrinth 本身不训练模型，而是提供环境。使用 gymnasium 接口，兼容各类模仿学习方法：
- 支持 Behavioural Cloning (BC)、DAgger、GAIL、BCO、SQIL、IUPE 等
- 数据集托管在 HuggingFace 的 IL-Datasets 上
- 支持向量和图像两种状态表示

## 实验关键数据

### 主实验

| 方法 | 训练 AER | 训练 SR | 验证 AER | 验证 SR | 测试 AER | 测试 SR |
|------|---------|---------|---------|---------|---------|---------|
| BC | -2.11±2.41 | 37% | -3.70±1.18 | 6% | -3.90±0.70 | 2% |
| DAgger | -1.18±2.45 | 57% | -3.75±1.08 | 5% | -3.80±0.97 | 4% |
| GAIL | -0.98±1.89 | 61% | -3.57±1.58 | 9% | -3.85±0.85 | 3% |
| BCO | -0.53±2.23 | 70% | -3.90±0.69 | 2% | -3.85±0.85 | 3% |
| SQIL | -3.80±0.96 | 4% | -3.95±0.49 | 1% | -4.00±0.00 | 0% |
| IUPE | **0.27±2.39** | **75%** | **-2.80±2.12** | **21%** | -3.85±1.00 | **5%** |

### 消融实验

| 配置 | 训练 SR | 验证 SR | 测试 SR | 说明 |
|------|---------|---------|---------|------|
| BC (1000 epochs, Atari CNN) | 37% | 6% | 2% | 原始配置 |
| BC (10000 epochs, Atari CNN) | 100% | 41% | 34% | 更长训练 |
| BC (10000 epochs, ResNet-18) | 100% | **56%** | **53%** | 更强编码器 |

| 环境 | 初始化间距 (avg) | 动作序列复制可行性 |
|------|-----------------|-------------------|
| MountainCar | 0.001 | 仅 1 条专家轨迹即可"解决" |
| CartPole | 0.0095 | 极为相似 |
| Hopper | 0.2175 | 最近邻 action 达专家水平 |
| Labyrinth | 高（结构完全不同） | 不可行 |

### 关键发现
- 所有模仿学习方法在未见结构的测试集上表现极差（最高仅 5% SR），说明没有方法真正学会了导航任务本身
- IUPE 在验证集 21% 但测试集仅 5%，说明验证集的"泛化"可能只是运气而非真正理解
- 更强的网络架构（ResNet-18）能显著提升泛化（2%→53%），但仍远未解决问题——说明瓶颈在算法而非表征
- 纯模仿学习（BC 类）比逆强化学习类方法（GAIL, SQIL）学到了更好的状态编码

## 亮点与洞察
- **对现有 benchmark 的犀利批评有理有据**：MountainCar 复制单条轨迹就能"解决"、Hopper 最近邻状态 action 可替代专家，这些实验直接挑战了社区对泛化评估的默认假设。
- **环境设计兼顾简洁与可控**：离散图结构使得最优策略可精确计算，biased/unbiased 设置使得动作分布的变化程度可量化，key-and-door/ice-floor 变体渐进引入不同维度的泛化挑战。

## 局限与展望
- 目前仅支持离散动作空间，无法评测仅适用于连续动作的方法（如 OPOLO, MAHALO）
- 迷宫任务相对简单，可能不足以代表高维感知和复杂物理交互的泛化挑战
- 缺少与 Procgen、MiniGrid 等已有程序生成环境的直接对比
- 部分可观测的设置与 key-and-door、ice-floor 互斥，限制了组合测试

## 相关工作与启发
- **vs MiniGrid**: MiniGrid 同样是格子环境但 Labyrinth 独有精确最优动作、图哈希唯一性保证和结构化数据分离
- **vs Procgen**: Procgen 通过程序生成测泛化但任务太复杂难以精确分析失败原因，Labyrinth 的可控性更强
- **vs MuJoCo benchmarks**: MuJoCo 的训练-测试差异极小（初始化 Manhattan 距离 <0.5），Labyrinth 提供了根本不同的结构

## 评分
- 新颖性: ⭐⭐⭐⭐ 环境设计思路新颖，对现有 benchmark 弱点的分析切中要害
- 实验充分度: ⭐⭐⭐⭐ 覆盖了 6 种模仿学习方法和多种消融，但缺少更多任务变体的深入实验
- 写作质量: ⭐⭐⭐⭐ 论述逻辑清晰，现有 benchmark 的批评有数据支撑
- 价值: ⭐⭐⭐⭐ 对模仿学习泛化评估有重要启发，但 Labyrinth 本身的复杂度可能限制其作为通用 benchmark 的推广

<!-- RELATED:START -->

## 相关论文

- [Action-Constrained Imitation Learning](../../ICML2025/reinforcement_learning/action-constrained_imitation_learning.md)
- [EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)
- [Interactive and Hybrid Imitation Learning: Provably Beating Behavior Cloning](interactive_and_hybrid_imitation_learning_provably_beating_behavior_cloning.md)
- [BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning](beast_efficient_tokenization_of_b-splines_encoded_action_sequences_for_imitation.md)
- [Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics](massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)

<!-- RELATED:END -->
