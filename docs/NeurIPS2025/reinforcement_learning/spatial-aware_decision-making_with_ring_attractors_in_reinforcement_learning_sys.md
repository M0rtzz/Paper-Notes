---
title: >-
  [论文解读] Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems
description: >-
  [NeurIPS 2025][环形吸引子] 将神经科学中的环形吸引子模型集成到 DRL 的动作选择中，通过将动作映射到环上的空间位置并利用高斯信号注入 Q 值和不确定性，在 Atari 100K 上比基线提升 53%。
tags:
  - NeurIPS 2025
  - 环形吸引子
  - 生物启发RL
  - 空间感知
  - 动作空间编码
  - 不确定性量化
---

# Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems

**会议**: NeurIPS 2025  
**arXiv**: [2410.03119](https://arxiv.org/abs/2410.03119)  
**代码**: https://github.com/marcosaura/RA_RL  
**领域**: 强化学习  
**关键词**: 环形吸引子, 生物启发RL, 空间感知, 动作空间编码, 不确定性量化

## 一句话总结
将神经科学中的环形吸引子模型集成到 DRL 的动作选择中，通过将动作映射到环上的空间位置并利用高斯信号注入 Q 值和不确定性，在 Atari 100K 上比基线提升 53%。

## 研究背景与动机

**领域现状**：DRL 中快速高效的动作选择仍是核心挑战，特别是在具有空间结构的环境（机器人操作中关节运动耦合、游戏中方向动作邻接）。现有方法将动作视为正交独立的 one-hot 向量，完全忽略动作间的拓扑关系

**现有痛点**：
   - 传统 DQN 将动作表示为正交向量，不反映"向左"和"向左上"比"向左"和"向右"更接近的事实
   - 现有空间感知方法（relational RL、认知地图等）通过复杂架构从数据中隐式学习空间理解，需要大量样本
   - 不确定性量化方法（如 Bootstrapped DQN）将不确定性作为独立模块，未与空间结构融合

**核心矛盾**：动作空间有内在拓扑结构，但这种结构信息在标准 DRL 中完全被丢弃

**切入角度**：在果蝇的中央复合体中发现的环形吸引子是一种经实验验证的神经回路，能稳定地编码方向和空间信息

**核心 idea**：将环形吸引子作为动作选择的"大脑"——将 Q 值转化为环上的高斯输入信号（幅度=Q值、角度=动作方向、宽度=不确定性），通过兴奋-抑制动力学选出最优动作

## 方法详解

### 整体框架
两种实现方式：(1) **外源模型**——用连续时间递归神经网络（CTRNN）实现环形吸引子，作为 DQN 输出层的后处理模块，替代标准 argmax 进行动作选择；(2) **集成模型**——开发基于 RNN 的可复用 DL 模块，嵌入 DRL 智能体内部实现端到端训练。

### 关键设计

1. **环形吸引子架构（Touretzky 模型）**：

    - 功能：N 个兴奋性神经元环形排列 + 1 个中心抑制性神经元，通过距离衰减的突触连接实现局部兴奋和全局抑制
    - 核心公式：兴奋性神经元动力学 $\frac{dv_n}{dt} = \frac{f(x_n + \epsilon_n + \eta_n)}{\tau} - v_n$，其中 $x_n$ 是外部输入、$\epsilon_n$ 是兴奋性反馈、$\eta_n$ 是抑制性反馈
    - 突触权重按距离衰减：$w^{(E_m \rightarrow E_n)} = e^{-d^2_{(m,n)}}$
    - 设计动机：兴奋-抑制动力学产生赢者通吃效应——活跃峰稳定在 Q 值最高的动作附近，且受相邻高价值动作影响，实现平滑的空间推理

2. **Q 值到环输入的映射**：

    - 功能：将 DQN 输出的 Q 值转化为环形吸引子的高斯输入信号
    - 核心公式：$x_n(Q(s,a)) = \sum_{a=1}^{A} \frac{Q(s,a)}{\sqrt{2\pi\sigma_a}} \exp(-\frac{(\alpha_n - \alpha_a(a))^2}{2\sigma_a^2})$
    - 三个关键参数：$K_i = Q(s,a)$（幅度=动作价值）、$\mu_i = \alpha_a(a)$（角度=动作在环上的位置）、$\sigma_i = \sigma_a$（宽度=价值估计的不确定性）
    - 设计动机：Q 值高的动作在环上产生更强的输入信号，经过兴奋-抑制动力学后，空间上相邻的高价值动作互相增强——这正是空间信息利用的核心

3. **贝叶斯不确定性注入**：

    - 功能：用贝叶斯线性回归（BLR）作为 DQN 输出层，自然提供每个动作的 Q 值方差
    - 核心思路：$Q(s,a) = \Phi_\theta(s)^T w_a$，其中 $w_a$ 来自 BLR 的后验分布。高斯信号宽度 $\sigma_a$ 直接使用 BLR 的后验方差
    - 设计动机：不确定性大的动作信号更"扩散"（宽高斯），不确定性小的信号更"尖锐"（窄高斯）——自动实现探索-利用平衡

4. **集成 DL 模块（可复用 RNN）**：

    - 功能：将环形吸引子动力学用 GRU/LSTM 实现，作为 DRL 框架中的即插即用组件
    - 核心思路：用循环层模拟环形吸引子的时间演化，输入为 Q 值序列，输出为动作选择
    - 设计动机：CTRNN 需要手动设置迭代步数（~50步）且不可微，RNN 实现可端到端训练且更高效

### 损失函数 / 训练策略
- 外源模型：DQN 标准损失训练 Q 网络 + 环形吸引子做动作选择（不参与梯度），BLR 输出层在线更新后验
- 集成模型：端到端训练，环形吸引子 RNN 模块参与梯度反向传播

## 实验关键数据

### 主实验 — Atari 100K 基准（外源模型）

| 方法 | 中值人类归一化分数 (MHNS) | 均值 MHNS | 超人类游戏数 |
|------|----------------------|---------|-----------|
| DQN baseline | ~50% | ~45% | 2/26 |
| DQN + Ring Attractor (无 UQ) | ~72% | ~68% | 5/26 |
| DQN + Ring Attractor (**含 UQ**) | **~80%** | **~75%** | **8/26** |
| 提升 | - | **+53%** | - |

### 消融实验 — 各组件贡献

| 配置 | MHNS | 相对基线提升 | 说明 |
|------|------|-----------|------|
| DQN baseline | ~50% | - | 标准 argmax 选择 |
| +Ring Attractor (仅空间结构) | ~68% | +35% | 空间拓扑信息的价值 |
| +Ring Attractor + Bayesian UQ | **~80%** | **+53%** | 不确定性集成额外+18% |
| +Ring Attractor + 变化环尺寸 | ~65-78% | 依赖 | 32神经元是最优 |

### 集成 DL 模型 vs 外源模型

| 方法 | 训练速度 | 最终性能 | 可扩展性 |
|------|---------|---------|---------|
| 外源 CTRNN | 慢（需迭代~50步） | 高 | 受限 |
| 集成 RNN | **快** | **相当/略高** | **好** |

### 关键发现
- 环形吸引子在具有**空间动作结构**的游戏上提升最大（方向性运动类 > 离散选择类）。例如在 Pong（方向连续）上提升巨大，在 Montezuma Revenge（探索密集）上提升有限
- **不确定性注入的额外 18% 提升**与 Thompson Sampling 类探索策略互补：高不确定性动作的高斯信号更宽，"扩散"到相邻动作，等价于探索
- 环形吸引子的**时间滤波效果**使动作选择序列更平滑——在两个相近动作间不再频繁跳动，这在物理控制中很有价值
- 环上的神经元数量 N 需要适当选择：太少信息不足，太多计算开销大。N=32（4x动作空间大小）在大部分游戏上最优

## 亮点与洞察
- **生物神经回路作为计算原语引入 RL**是独特的交叉视角——环形吸引子在果蝇头方向系统中得到实验验证，将这种经过亿年进化的计算结构用于人工智能是一种"向生物学借鉴"的深层方法论
- 三合一设计的巧妙：高斯幅度 = Q 值（利用性），角度 = 空间位置（结构性），宽度 = 不确定性（探索性）——一个公式同时编码了三种关键信息
- 方法是**即插即用**的——只需在 DQN 的输出端加一个环形吸引子模块，不修改 Q 网络本身
- 兴奋-抑制动力学产生的"赢者通吃 + 空间扩散"效应天然适合连续/空间动作选择

## 局限与展望
- 动作到环的映射假设动作间有**环形拓扑**——并非所有动作空间都满足（如树状结构的层次动作空间）
- CTRNN 模型需要约 50 步迭代才能收敛到稳态，增加推理延迟——实时控制场景可能受影响
- 当前仅在离散动作空间验证——连续动作空间需要对环进行离散化或扩展到高维吸引子
- BLR 的后验更新在非平稳环境中可能不够快——考虑遗忘机制或在线贝叶斯方法

## 相关工作与启发
- **vs Bootstrapped DQN**：Bootstrapped DQN 将不确定性作为独立模块用于 Thompson Sampling 探索；环形吸引子将不确定性与空间结构统一在一个信号中
- **vs Grid Cells (DeepMind)**：Grid Cells 编码状态空间的位置信息；Ring Attractors 编码动作空间的结构信息——虽然都是生物启发但编码对象不同
- **vs Relational DRL**：Relational RL 通过注意力机制隐式学习实体间关系；Ring Attractor 通过显式环形拓扑提供空间归纳偏置——后者更sample-efficient
- 未来方向：将环形吸引子扩展到多维吸引子（如 toroid）编码高维连续动作空间

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 神经科学与 RL 的独特且深层交叉，环形吸引子作为动作空间编码从未被提出
- 实验充分度: ⭐⭐⭐⭐ Atari 100K 标准基准 + 两种实现方式 + 消融分析，但缺少连续控制任务
- 写作质量: ⭐⭐⭐⭐ 生物背景详细、数学推导完整，但长度较长
- 价值: ⭐⭐⭐⭐ 空间动作编码是有前景的方向，即插即用设计实用

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)
- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](../../CVPR2025/reinforcement_learning/decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[NeurIPS 2025\] Reward-Aware Proto-Representations in Reinforcement Learning](reward-aware_proto-representations_in_reinforcement_learning.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](../../ICML2025/reinforcement_learning/divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)

<!-- RELATED:END -->
