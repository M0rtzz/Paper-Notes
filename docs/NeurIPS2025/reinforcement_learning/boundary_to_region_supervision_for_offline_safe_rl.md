---
title: >-
  [论文解读] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning
description: >-
  [NeurIPS 2025][强化学习][离线安全强化学习] 提出 B2R（Boundary-to-Region）框架，通过代价信号重对齐(CTG Realignment)解决序列模型在离线安全RL中对回报和代价的对称条件化谬误，将稀疏的边界监督转化为密集的安全区域监督…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "离线安全强化学习"
  - "Transformer"
  - "代价约束"
  - "非对称条件化"
  - "安全区域监督"
---

# Boundary-to-Region Supervision for Offline Safe Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.25727](https://arxiv.org/abs/2509.25727)  
**代码**: [https://github.com/HuikangSu/B2R](https://github.com/HuikangSu/B2R)  
**领域**: 强化学习  
**关键词**: 离线安全强化学习, Decision Transformer, 代价约束, 非对称条件化, 安全区域监督

## 一句话总结
提出 B2R（Boundary-to-Region）框架，通过代价信号重对齐(CTG Realignment)解决序列模型在离线安全RL中对回报和代价的对称条件化谬误，将稀疏的边界监督转化为密集的安全区域监督，在38个安全关键任务中35个满足安全约束。

## 研究背景与动机

**领域现状**：离线安全强化学习旨在从静态数据集学习满足安全约束的策略。Decision Transformer (DT) 类方法通过将 RL 重构为条件序列建模取得了不错效果。

**现有痛点**：现有DT方法（如CDT）将 return-to-go (RTG) 和 cost-to-go (CTG) 作为对称的输入token处理，忽略了二者的本质差异：RTG 是灵活的性能目标，而 CTG 应是刚性的安全边界。

**核心矛盾**：这种对称处理导致两个问题：(1) 部署时难以选择合适的CTG初始值；(2) 数据集中代价恰好接近安全阈值的轨迹很稀疏，导致监督信号不足。

**本文目标**：设计非对称的条件化机制，使 CTG 作为边界约束而非变量目标，解耦安全保证与奖励优化。

**切入角度**：将所有安全轨迹的代价统一重对齐到安全阈值，使模型在固定的边界token条件下学习整个安全区域的多样行为。

**核心 idea**：用 CTG 重对齐将稀疏的"边界监督"转化为密集的"区域监督"，让模型从安全区域内的所有行为中学习，而非仅从代价恰好等于阈值的少量轨迹中学习。

## 方法详解

### 整体框架
B2R包含三个环环相扣的组件：(1) 轨迹过滤去除不安全样本；(2) CTG重对齐将所有安全轨迹统一到部署时代价阈值；(3) RoPE位置编码增强对重对齐后时间动态的建模。

### 关键设计

1. **轨迹过滤 (Trajectory Filtering)**:

    - 功能：定义安全区域，去除违规轨迹
    - 核心思路：保留累积代价 $C(\tau) \leq \kappa$ 的安全轨迹组成 $\mathcal{D}_{\text{safe}}$，确保所有训练数据符合部署约束
    - 设计动机：防止不安全轨迹对策略产生负面影响

2. **CTG 重对齐 (CTG Realignment)**:

    - 功能：创建密集且统一的监督信号
    - 核心思路：对每条安全轨迹的 CTG 序列加上常数偏移 $\hat{C}_t' = \hat{C}_t + (\kappa - C(\tau))$，使初始 CTG 统一为 $\kappa$，同时保留原始 CTG 的时间变化模式
    - 设计动机：传统方法仅从代价恰好等于 $\kappa$ 的边界轨迹学习（稀疏），重对齐后模型用统一的边界token学习来自整个安全区域的多样行为（密集）

3. **RoPE 位置编码**:

    - 功能：改善时序建模，适配CTG重对齐
    - 核心思路：用旋转位置编码替代DT原始的绝对/可学习位置编码，其相对位置编码特性更适合捕捉重对齐后代价序列的逐步变化动态
    - 设计动机：重对齐改变了CTG序列的绝对值但保留了相对变化，RoPE的相对编码与这一特性天然匹配

### 损失函数 / 训练策略
使用标准的行为克隆损失：$\mathcal{L}_{BC}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}_{\text{safe}}}[-\log \pi_\theta(a_t | \hat{R}_{t-K:t}, \hat{C}'_{t-K:t}, s_{t-K:t}, a_{t-K:t-1})]$，在CTG重对齐后的安全数据集上训练，部署时使用固定的 $\hat{C}_0' = \kappa$。

## 实验关键数据

### 主实验

| 任务类别 | 安全约束满足数 | 总任务数 | B2R | CDT基线 |
|---------|-------------|---------|------|--------|
| 安全关键任务 | 35 | 38 | 最优奖励 | 约20个满足 |

### 消融实验

| CTG重对齐策略 | 效果 | 说明 |
|-------------|------|------|
| Shift（均匀偏移） | 最佳 | 保留时间剖面，简单有效 |
| Avg（均分） | 次优 | 均匀再分配多余代价预算 |
| Scale（缩放） | 中等 | 乘性归一化 |
| Rand（随机） | 较差 | 随机再分配引入噪声 |

### 关键发现
- B2R在38个任务中35个满足安全约束，远超CDT等基线
- Shift策略最简单也最有效，因为它保留了原始CTG的时间变化模式
- MetaDrive实验直观展示了边界监督的脆弱性：仅在v=10处训练的策略频繁超速，而B2R从多样速度行为中学习，实现了平滑的安全裕度控制

## 亮点与洞察
- "对称性谬误"的识别非常深刻：RTG和CTG虽然形式相似，但语义完全不同——一个是"追求的目标"，一个是"不可逾越的边界"。这一洞察可推广到所有目标-约束优化问题中
- CTG重对齐的巧妙之处在于：不修改模型架构，仅修改数据处理方式，就将稀疏监督转化为密集监督

## 局限与展望
- 理论分析基于简化假设，实际环境中的安全保证可能受限
- 轨迹过滤可能丢弃大量数据，在数据稀缺场景下不利
- 未来可考虑自适应的代价阈值和在线调整策略

## 相关工作与启发
- **vs CDT**: CDT对称处理RTG和CTG，B2R通过非对称设计解耦安全与性能
- **vs TraC**: TraC仅分类丢弃不安全轨迹，B2R进一步对安全轨迹进行变换，保留行为多样性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 对称性谬误的发现和区域监督范式非常原创
- 实验充分度: ⭐⭐⭐⭐⭐ 38个任务全面验证
- 写作质量: ⭐⭐⭐⭐ 理论分析清晰，直觉图示有效
- 价值: ⭐⭐⭐⭐⭐ 为序列模型应用于安全RL提供了新的理论和实践基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[NeurIPS 2025\] Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning](adaptive_neighborhoodconstrained_q_learning_for_offline_rein.md)
- [\[NeurIPS 2025\] Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm](trust_region_reward_optimization_and_proximal_inverse_reward_optimization_algori.md)
- [\[NeurIPS 2025\] Forecasting in Offline Reinforcement Learning for Non-stationary Environments](forecasting_in_offline_reinforcement_learning_for_non-stationary_environments.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)

</div>

<!-- RELATED:END -->
