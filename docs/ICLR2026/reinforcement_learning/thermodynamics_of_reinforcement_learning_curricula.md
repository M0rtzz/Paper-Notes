---
title: >-
  [论文解读] Thermodynamics of Reinforcement Learning Curricula
description: >-
  [ICLR 2026][课程学习] 本文利用非平衡热力学中的过剩功（excess work）最小化框架，将RL中的课程学习形式化为任务空间上的测地线优化问题，并推导出基于摩擦张量的温度退火算法MEW，在MuJoCo Humanoid任务上超越标准SAC温度调节方法。
tags:
  - ICLR 2026
  - 课程学习
  - 非平衡热力学
  - 最大熵RL
  - 温度退火
  - 黎曼几何
  - 测地线
---

# Thermodynamics of Reinforcement Learning Curricula

**会议**: ICLR 2026  
**arXiv**: [2603.12324](https://arxiv.org/abs/2603.12324)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 课程学习, 非平衡热力学, 最大熵RL, 温度退火, 黎曼几何, 测地线

## 一句话总结
本文利用非平衡热力学中的过剩功（excess work）最小化框架，将RL中的课程学习形式化为任务空间上的测地线优化问题，并推导出基于摩擦张量的温度退火算法MEW，在MuJoCo Humanoid任务上超越标准SAC温度调节方法。

## 研究背景与动机

**领域现状**：现代RL系统很少在单一、静态任务上训练，而是通过课程学习、温度退火、奖励塑形等方式让agent接触一系列相关任务。然而，如何合理地变化任务参数仍缺乏理论指导。

**现有痛点**：实践中常用的线性插值（linearly interpolating）任务参数隐式假设任务空间是平坦且各向同性的。但实际上，不同方向的参数变化对agent的学习难度差异很大——某些方向适应代价高，某些方向低。

**核心矛盾**：缺乏一个原则性的框架来量化任务参数变化的"适应代价"，导致课程设计依赖启发式方法（如固定衰减、手动调参），可能在高摩擦区域过快变化参数，造成策略不稳定。

**本文目标**：(1) 如何定义和计算任务空间上的"学习难度"度量？(2) 什么是最优的课程路径？(3) 能否推导出实用的温度退火算法？

**切入角度**：从统计力学出发，将RL中策略对任务参数变化的响应类比为非平衡物理系统的驱动过程，利用线性响应理论将过剩功近似为二次型，从而在任务参数空间上建立伪黎曼度量。

**核心 idea**：最优课程对应任务空间中摩擦张量诱导几何下的测地线——在学习困难的方向减速、在容易的方向加速。

## 方法详解

### 整体框架
将参数化的奖励函数 $r_\lambda(s,a)$ 的参数 $\lambda \in \mathbb{R}^L$ 视为任务流形上的坐标。课程 $\lambda(t)$ 是连接两个任务的路径。通过最小化agent适应的累积代价（过剩功），将课程设计转化为几何优化问题。

### 关键设计

1. **过剩功与摩擦张量**:

    - 功能：量化有限速率变化任务参数时产生的额外适应代价
    - 核心思路：在准静态极限下，利用线性响应理论将过剩功近似为 $\mathcal{W}_{\text{excess}} = \int_0^\infty \dot{\lambda}_i(t) \zeta_{ij}(\lambda(t)) \dot{\lambda}_j(t) dt$
    - 摩擦张量由Green-Kubo关系给出：$\zeta_{ij}(\lambda) = \beta \sum_{t=0}^{\infty} \mathbb{E}_{\tau \sim p_\lambda}(\delta X_i(\mathbf{s}_t, \mathbf{a}_t) \cdot \delta X_j(\mathbf{s}_0, \mathbf{a}_0))$
    - 设计动机：摩擦张量的大值对应奖励梯度波动在长时间尺度上持续的方向，使得适应代价高

2. **测地线最优课程**:

    - 功能：求解使过剩功最小的路径
    - 核心思路：过剩功的二次型赋予参数空间伪黎曼度量，最优课程满足测地线方程 $\ddot{\lambda}^k + \Gamma^k_{ij}(\lambda) \dot{\lambda}^i \dot{\lambda}^j = 0$
    - 关键推论：**线性课程只有在诱导几何平坦时才是最优的**，即 $\zeta_{ij}(\lambda) = c$
    - 设计动机：在GridWorld实验中直观展示了测地线如何绕过高摩擦区（相变点 $\lambda_1 = \lambda_2$）

3. **MEW温度退火算法**:

    - 功能：将框架应用于MaxEnt RL的温度退火（如SAC中的 $\alpha$ 调节）
    - 核心思路：将逆温 $\beta = \alpha^{-1}$ 作为控制参数，此时摩擦简化为奖励的自协方差。最小化过剩功得到更新规则 $\dot{\alpha} \propto \alpha^2 / \sqrt{\sum \langle \delta r_k \delta r_{t+k} \rangle}$
    - 设计动机：在高奖励方差区域慢速变化温度，在波动小的区域加速——提供了自适应正则化的原则性机制

### 损失函数 / 训练策略
MEW无需额外损失函数，而是提供温度调度策略。以ASAC（Average-reward SAC）为基础算法，仅替换温度退火方案。摩擦（奖励自协方差）可直接从训练过程中的奖励样本高效计算。

## 实验关键数据

### 主实验（MuJoCo Humanoid-v5）

| 方法 | 性能表现 | 温度调度特征 |
|------|---------|------------|
| 固定高温 | 收敛但次优 | 温度始终不变 |
| 固定低温 | 早期不稳定 | 温度始终不变 |
| SAC自动调温 | 次优，温度非单调 | 初始快速下降后上升 |
| **MEW** | **最优** | **单调递减，run间一致性高** |

### 关键发现
- SAC标准方法（Haarnoja et al., 2018b）初始快速降温导致策略过早确定性化，后续需要回升补偿
- MEW的温度曲线是单调的，根据适应的相对代价动态调整步长
- MEW在不同run之间的一致性显著高于标准方法（置信区间更窄）
- GridWorld实验清晰展示了线性路径穿越最大摩擦区的次优性

### 消融实验
- 在GridWorld中比较线性路径和测地线路径的regret，测地线路径绕过相变点后regret显著更低
- 摩擦张量的可视化确认了任务空间的几何确实是弯曲的（非欧几里得的）

## 亮点与洞察
- **统计力学与RL的深度联系**：不是表面类比，而是利用MaxEnt RL的Boltzmann分布结构建立精确映射。摩擦张量的Green-Kubo关系在RL中有明确的可计算形式
- **"学习困难"的几何化**：将抽象的"哪些地方难学"转化为可测量的几何量（摩擦），使优化课程不再靠拍脑袋
- **实用性强**：MEW只需奖励方差估计，计算开销低，可直接嵌入现有deep RL算法
- **解释能力**：框架可以解释为什么某些经验性的RL不稳定现象——可能是因为在弯曲流形上过于激进地驱动高维非平衡系统

## 局限与展望
- 当前理论依赖准静态假设（线性响应理论），在参数变化很快时近似可能失效
- 实验仅在一维温度退火上验证了MEW，更高维任务空间的测地线求解需要开发可扩展的摩擦张量估计器
- 摩擦张量的计算需要策略已近似收敛，对于训练初期的非平稳阶段可能不准确
- 与distributional RL的结合（利用方差估计）是一个有前景的方向

## 相关工作与启发
- **vs 课程学习启发方法**: 现有课程学习缺乏理论最优性保证。本框架提供了基于物理原理的最优性准则
- **vs 自动温度调节（SAC）**: SAC的minimum entropy constraint方法是反应式的（reactive），MEW是前瞻式的（proactive），基于摩擦预测未来适应代价
- **vs 奖励塑形**: 势函数奖励塑形（PBRS）在框架中对应度量的退化方向（零特征值），理论上统一了这一现象
- **vs Optimal Transport课程学习（Huang et al., 2022）**: OT方法需要源和目标任务分布，MEW只需在线奖励统计量
- **vs 线性插值/固定衰减**: 线性课程仅在几何平坦时最优——框架精确刻画了何时（以及为何）线性不够好
- **vs Fisher信息度量**: 摩擦张量与Fisher信息矩阵形式相似但含义不同——Fisher度量的是参数空间中的信息几何，摩擦张量度量的是任务空间中的适应代价

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 热力学-RL的深度联系极为优雅，测地线课程的概念开创性
- 实验充分度: ⭐⭐⭐ GridWorld验证了几何概念，但deep RL实验仅限Humanoid一个任务
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，物理直觉清晰，论文结构紧凑
- 价值: ⭐⭐⭐⭐ 为课程学习和温度调节提供了理论基础，MEW算法直接可用
- 可扩展性: ⭐⭐⭐ 一维温度退火已验证，多维任务空间需要可扩展的摩擦估计器
- 理论深度: ⭐⭐⭐⭐⭐ 从非平衡热力学到RL课程的映射数学严谨且物理直觉清晰

<!-- RELATED:START -->

## 相关论文

- [DISCOVER: Automated Curricula for Sparse-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/discover_automated_curricula_for_sparse-reward_reinforcement_learning.md)
- [Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)
- [Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [Spotlight on Token Perception for Multimodal Reinforcement Learning](spotlight_on_token_perception_for_multimodal_reinforcement_learning.md)
- [QuRL: Efficient Reinforcement Learning with Quantized Rollout](qurl_efficient_reinforcement_learning_with_quantized_rollout.md)

<!-- RELATED:END -->
