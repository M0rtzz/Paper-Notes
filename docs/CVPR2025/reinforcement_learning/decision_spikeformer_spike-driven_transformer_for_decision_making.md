---
title: >-
  [论文解读] Decision SpikeFormer: Spike-Driven Transformer for Decision Making
description: >-
  [CVPR 2025][脉冲神经网络] 提出 DSFormer，首个用于离线强化学习的脉冲驱动 Transformer，设计了时序脉冲自注意力 (TSSA) 和位置脉冲自注意力 (PSSA) 来捕获 RL 中的时序/位置依赖，并引入渐进式阈值依赖批归一化 (PTBN) 解决归一化与脉冲特性的冲突，在 D4RL 基准上超越 ANN 对手且节省 78.4% 能耗。
tags:
  - CVPR 2025
  - 脉冲神经网络
  - 离线强化学习
  - Transformer
  - 能效AI
  - 序列决策
---

# Decision SpikeFormer: Spike-Driven Transformer for Decision Making

**会议**: CVPR 2025  
**arXiv**: [2504.03800](https://arxiv.org/abs/2504.03800)  
**代码**: 项目主页提供（见论文）  
**领域**: 强化学习 / 脉冲神经网络  
**关键词**: 脉冲神经网络, 离线强化学习, Spike-driven Transformer, 能效AI, 序列决策

## 一句话总结
提出 DSFormer，首个用于离线强化学习的脉冲驱动 Transformer，设计了时序脉冲自注意力 (TSSA) 和位置脉冲自注意力 (PSSA) 来捕获 RL 中的时序/位置依赖，并引入渐进式阈值依赖批归一化 (PTBN) 解决归一化与脉冲特性的冲突，在 D4RL 基准上超越 ANN 对手且节省 78.4% 能耗。

## 研究背景与动机

**领域现状**：离线强化学习通过条件序列建模 (CSM) 将策略学习转化为序列预测任务，Decision Transformer (DT) 是代表工作。但基于 ANN 的 Transformer 在能效受限的具身 AI 场景中面临高能耗问题。脉冲神经网络 (SNN) 凭借事件驱动、低功耗特性成为替代方案，但已有 SNN Transformer 主要面向视觉分类任务。

**现有痛点**：(1) 现有 SNN Transformer 的注意力机制面向空间维度设计（图像），不适合 RL 中的时序依赖建模；(2) SNN 需要 BatchNorm（可与线性层融合实现纯脉冲推理），但 BatchNorm 会破坏序列内的时序依赖；LayerNorm 保留时序依赖但引入浮点运算，破坏脉冲特性。

**核心矛盾**：SNN 的离散脉冲处理与 RL 需要的精确连续值估计之间的根本冲突，以及归一化层在"保留时序依赖"和"维持脉冲特性"之间的两难。

**本文目标** 如何设计适合离线 RL 序列建模的 SNN 自注意力机制，以及如何在保真时序依赖的同时维持 SNN 的纯脉冲推理特性。

**切入角度**：将 SNN 时间步维度融入注意力计算——在时间维度上拼接输入后做注意力来捕获全局时序依赖，用位置偏置建模局部马尔可夫依赖。归一化问题则用渐进式过渡方案解决。

**核心 idea**：用时序拼接注意力 + 位置偏置注意力 + 渐进式归一化，让 SNN Transformer 既能做好序列决策又保持低能耗。

## 方法详解

### 整体框架
DSFormer 遵循 DT 架构：输入序列 $I_l = (a_{l-N}, \hat{R}_{l-N+1}, s_{l-N+1}, ..., a_{l-1}, \hat{R}_l, s_l)$ 经过嵌入层后，沿时间维度重复 T 次（SNN 时间步），送入 M 个堆叠的 Decoder Block，每个 Block 包含脉冲自注意力层 + 脉冲 MLP 层，最终通过预测头输出下一步动作。

### 关键设计

1. **时序脉冲自注意力 (TSSA)**:

    - 功能：捕获跨 SNN 时间步的全局时序依赖，增强长序列信用分配
    - 核心思路：不像传统 SSSA 在每个时间步独立做自注意力，TSSA 将所有时间步的输入沿时间维度拼接后统一做注意力计算。Q、K、V 矩阵来自all时间步的拼接，配合因果掩码防止未来信息泄露。从信息论角度证明，拼接后的联合熵 $H(X^1,...,X^T) < \sum_t H(X^t)$，因为 LIF 动力学使相邻时间步不独立，拼接后能更有效地学习模式。时间复杂度 $O(TDN^2)$，与 SSSA 相同
    - 设计动机：SNN 的核心特性是跨时间步的膜电位动态，逐步自注意力会丢失这种跨时间的关联

2. **位置脉冲自注意力 (PSSA)**:

    - 功能：以线性复杂度捕获局部位置依赖
    - 核心思路：引入可学习的成对位置偏置矩阵 $P \in \mathbb{R}^{N \times N}$。计算方式为 $\text{Attn}(Q,K,V)_i = Q_i \odot \sum_j P_{ij} \odot K_j \odot V_j$，用逐元素乘法替代矩阵乘法。位置偏置设置局部窗口 $S$，仅保留 $|i-j| < S$ 的位置关系。时间复杂度降低到 $O(TDN)$，线性于 token 数
    - 设计动机：RL 轨迹的马尔可夫性使得局部依赖（相邻 state-action 对）最为重要，全局注意力反而浪费计算。逐元素运算完全兼容 SNN 的加法本质

3. **渐进式阈值依赖批归一化 (PTBN)**:

    - 功能：训练时保留时序依赖，推理时维持纯脉冲计算
    - 核心思路：将 tdLN（按通道维度归一化，保留时序依赖）和 tdBN（按批维度归一化，可与线性层融合）通过权重 $\theta$ 线性组合：$\text{PTBN}(x) = \theta \cdot \text{tdLN}(x) + (1-\theta) \cdot \text{tdBN}(x)$。$\theta$ 从 1 线性衰减到 0：训练早期用 tdLN 建立时序依赖，后期过渡到 tdBN 适应脉冲推理。推理时退化为纯 tdBN，可与线性层融合消除浮点运算
    - 设计动机：直接用 BatchNorm 会破坏序列内依赖导致性能差，直接用 LayerNorm 无法做纯脉冲推理，渐进过渡兼顾两者

### 损失函数 / 训练策略
遵循 Decision Transformer 框架，使用 MSE 损失监督动作预测。训练时 $\theta$ 从 1 线性衰减到 0，一部分训练步数用于 PTBN 过渡，其余用于纯 tdBN 微调。

## 实验关键数据

### 主实验

**MuJoCo 任务（D4RL）**:

| 任务 | DT (ANN) | FCNet (ANN) | PSSA (SNN) | 能耗 |
|------|----------|-------------|------------|------|
| halfcheetah-m-e | 86.8 | 91.2 | **91.5** | - |
| walker2d-m-e | 108.1 | 108.8 | **108.9** | - |
| hopper-m-r | 82.7 | 65.3 | **96.3** | - |
| 平均 | 74.7 | 72.8 | **78.8** | **88.8μJ** |
| DT 能耗 | - | - | - | 410.5μJ |

**Adroit 任务**:

| 任务 | DT | PSSA |
|------|-----|------|
| pen-e | 110.4 | **122.0** |
| relocate-e | 15.3 | **108.4** |
| 平均 | 27.8 | **48.6** (+74%) |

### 消融实验

| 注意力类型 | MuJoCo 平均 | 时间复杂度 |
|-----------|------------|----------|
| SSSA | 73.8 | $O(TDN^2)$ |
| TSSA | 75.7 | $O(TDN^2)$ |
| **PSSA** | **78.8** | $O(TDN)$ |

| 归一化方式 | 平均分 |
|-----------|--------|
| tdBN | 62.3 |
| tdLN | 70.1 |
| **PTBN** | **78.8** |

### 关键发现
- PSSA 在所有 SNN 方案中性能最优，且优于 ANN 的 DT（78.8 vs 74.7），同时节省 78.4% 能耗
- 在 Adroit 操作任务上，SNN 相对 DT 的提升更加显著（+74%），说明脉冲模型在精细操控任务上潜力巨大
- PTBN 是性能关键：直接用 tdBN 严重损失性能（62.3 vs 78.8），说明序列内时序依赖建模极为重要
- 长序列实验（AntMaze）中 TSSA 在序列长度 100-200 时持续优于 DT，验证了时序拼接注意力的长程建模能力

## 亮点与洞察
- **SNN 超越 ANN 的稀有案例**：在离线 RL 中 SNN 不仅匹配更超越了 ANN 的 DT，同时节省 78% 能耗。这颠覆了"SNN 性能一定差于 ANN"的成见
- **PTBN 的渐进过渡思路**：从 LayerNorm 平滑过渡到 BatchNorm 是一个通用的工程技巧，可以迁移到任何需要在训练灵活性和推理效率之间取舍的场景
- **位置偏置替代注意力矩阵乘法**：PSSA 用逐元素乘法 + 位置偏置替代传统 QK 矩阵乘法，将复杂度从二次降到线性，且完全兼容 SNN 加法特性

## 局限与展望
- 仅在 D4RL 基准（MuJoCo/Adroit/AntMaze）上验证，未涉及更复杂的视觉 RL 任务
- SNN 的能耗估算基于理论运算量计算而非实际神经形态硬件测量
- DT 框架本身在 stitching 能力上有已知局限（无法从次优数据中拼接最优轨迹），DSFormer 继承了这一问题
- PTBN 中 $T_p$ 的超参数选择缺乏理论指导

## 相关工作与启发
- **vs Decision Transformer**: DSFormer 在 DT 框架上用 SNN 替换 ANN，MuJoCo 平均分 78.8 vs 74.7，同时节省 78.4% 能耗
- **vs SpikeGPT/SpikeBERT**: 现有 SNN 序列模型在 RL 任务上表现很差（SpikeGPT 平均仅 23.2），说明面向 NLP 的 SNN 设计不能直接迁移到 RL
- **vs Spikformer**: Spikformer 面向视觉的 SSA 缺乏因果掩码和时序建模，DSFormer 的 TSSA/PSSA 专门针对序列决策设计
- 对低功耗边缘设备上的具身 AI 应用有重要参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 SNN 离线 RL 模型，TSSA/PSSA/PTBN 三个设计都有新意
- 实验充分度: ⭐⭐⭐⭐ D4RL 全面验证+消融详尽，但缺少真实硬件能耗测试
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，理论推导严谨
- 价值: ⭐⭐⭐⭐ 为 SNN 在 RL 中的应用开辟新方向

<!-- RELATED:START -->

## 相关论文

- [Gazing at Rewards: Eye Movements as a Lens into Human and AI Decision-Making in Hybrid Visual Foraging](gazing_at_rewards_eye_movements_as_a_lens_into_human_and_ai_decision-making_in_h.md)
- [Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer](../../ICML2025/reinforcement_learning/mastering_massive_multi-task_reinforcement_learning_via_mixture-of-expert_decisi.md)
- [Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](../../ICML2025/reinforcement_learning/counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [Spatial-Aware Decision-Making with Ring Attractors in Reinforcement Learning Systems](../../NeurIPS2025/reinforcement_learning/spatial-aware_decision-making_with_ring_attractors_in_reinforcement_learning_sys.md)

<!-- RELATED:END -->
