---
title: >-
  [论文解读] When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift
description: >-
  [ICLR 2026][强化学习][传感器故障] 本文研究PPO在时间持续性传感器故障下的鲁棒性，提出将Transformer和SSM等序列模型集成到PPO中，推导了随机传感器故障下无限时间horizon奖励退化的高概率上界…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "传感器故障"
  - "部分可观测性"
  - "鲁棒性"
  - "Transformer"
  - "状态空间模型"
  - "PPO"
  - "序列建模"
---

# When Sensors Fail: Temporal Sequence Models for Robust PPO under Sensor Drift

**会议**: ICLR 2026  
**arXiv**: [2603.04648](https://arxiv.org/abs/2603.04648)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 传感器故障, 部分可观测性, 鲁棒性, Transformer, 状态空间模型, PPO, 序列建模

## 一句话总结
本文研究PPO在时间持续性传感器故障下的鲁棒性，提出将Transformer和SSM等序列模型集成到PPO中，推导了随机传感器故障下无限时间horizon奖励退化的高概率上界，并在MuJoCo实验中验证Transformer-PPO在严重传感器dropout下显著优于MLP、RNN和SSM基线。

## 研究背景与动机

**领域现状**：真实世界RL系统（机器人控制、自动驾驶）依赖的传感器反馈常常不可靠——故障、通信中断或瞬态损坏导致部分可观测性和性能退化。

**现有痛点**：(1) 标准MLP策略假设完全观测状态，传感器不可靠时性能急剧下降；(2) 实际系统中传感器故障具有时间持续性和组间关联性（如共享通信总线/电源），简单的独立掩码模型不够真实；(3) 现有序列模型在RL中的鲁棒性比较（如RLBenchNet）仅是纯经验的，缺乏理论刻画。

**核心矛盾**：RL策略的鲁棒性与其对时间上下文的利用能力直接相关，但缺乏理论框架量化这种关系。

**本文目标**：(1) 提供传感器故障下奖励退化的理论bound；(2) 系统比较不同序列架构在PPO中的鲁棒性；(3) 理解哪些架构特性驱动了鲁棒性差异。

**切入角度**：建立两层Markov传感器故障模型（个体+组级别），集成多种序列编码器进PPO，理论+实验双轨验证。

**核心 idea**：Transformer通过自注意力机制灵活引用历史有效观测，自然跳过缺失数据gap，是传感器不可靠环境下最鲁棒的策略架构。

## 方法详解

### 整体框架
提出两层Markov传感器故障模型，将Transformer/SSM/RNN编码器集成到PPO的actor-critic架构中，推导高概率奖励退化bound，在MuJoCo环境中实验验证。

### 关键设计

1. **两层Markov传感器故障模型**:

    - 个体层：每个传感器 $i$ 有二值Markov链 $z_i(t) \in \{0,1\}$，参数 $p_{\text{fail}}$、$p_{\text{recover}}$
    - 组层：每个组 $j$ 有 $y_j(t) \in \{0,1\}$，参数 $p_{\text{fail}}^{\text{group}}$、$p_{\text{recover}}^{\text{group}}$
    - 有效状态 $x_i(t) = z_i(t) \cdot y_j(t)$，稳态概率 $\pi_x = \pi_z \cdot \pi_y$
    - 有效故障概率 $p_{\text{fail}}^{\text{eff}} = 1 - (1-p_{\text{fail}})(1-p_{\text{fail}}^{\text{group}})$

2. **Transformer-PPO**:

    - History buffer：维护最近 $L$ 个观测的循环缓冲区
    - 编码器：投影+正弦位置编码→Transformer encoder（带key-padding mask跳过无效位置）
    - Attention pooling：学习的注意力加权将变长序列映射到固定大小特征向量
    - 分别接入actor和critic head

3. **RNN/SSM-PPO**:

    - 统一接口：$(h_t, z_t) = \mathcal{E}_\psi(h_{t-1}, x_t; d_t)$，其中 $d_t$ 为episode结束标志
    - 包括GRU、LRU、LinOSS等变体

### 理论分析

**定理5.6（高概率奖励退化bound）**：在Assumptions 5.1-5.5下，以概率 $\geq 1-\delta$：

$$S \leq \mu_S + C_{\max}\min\left\{\sqrt{\frac{2\tau}{1-\gamma^2}\ln\frac{2}{\delta}} + \frac{4}{3}\tau\ln\frac{2}{\delta}, \frac{1}{1-\gamma}\right\}$$

其中：
- $\mu_S \leq \frac{L_Q L_\pi}{1-\gamma}\sum_{i=1}^d (1-\pi_{x,i})h_i$ 是均值退化
- $C_{\max} = L_Q L_\pi \sum_i B_i$ 是最坏情况per-step影响
- $\tau$ 是增广链的mixing time

**解读**：
- 均值项仅依赖传感器的边际up-rate，相关性不直接影响期望退化
- 波动项有√τ和τ两个分量，mixing time越大（故障越持久）波动越大
- 策略平滑性 $L_\pi$ 和critic平滑性 $L_Q$ 全局缩放退化——序列模型通过利用历史实现更平滑的action变化

## 实验关键数据

### 实验设置
- 4个MuJoCo环境：HalfCheetah-v4, Hopper-v4, Walker2d-v4, Ant-v4
- 8种PPO agent：MLP + 3 RNNs/SSMs(LRU, GRU, LinOSS) + 3 Transformers(Transformer, UniTS, GTrXL)
- 传感器参数：$p_{\text{fail}}=1\%$, $p_{\text{recover}}=90\%$, $p_{\text{fail}}^{\text{group}}=55\%$, $p_{\text{recover}}^{\text{group}}=90\%$ → 有效恢复率60%

### 主实验结果

| 架构 | 完全观测 | 60%部分观测 | 退化程度 |
|------|---------|-----------|---------|
| MLP | 通常最高 | 严重退化 | **最大** |
| GRU | 竞争力中等 | 偶尔略好于MLP | 显著退化 |
| LRU | 竞争力中等 | 偶尔略好于MLP | 显著退化 |
| LinOSS | 中等 | 中等 | 显著退化 |
| GTrXL | 中等 | 表现不稳定 | 中等 |
| **Transformer** | **竞争力强** | **所有环境最优** | **最小** |
| UniTS | 最差 | 最差 | - |

### 关键发现
- **完全观测下**：MLP通常最优（MuJoCo环境是Markovian的），序列模型的额外复杂性有时反而是负担
- **部分观测下**：Transformer一致性最鲁棒，在所有环境上评测median最高
- RNN/SSM（包括GTrXL）的内存机制在传感器故障下效果有限——recurrent dynamics对输入的均匀处理和平滑时间流假设在数据缺失时被违反
- UniTS在所有设置下表现最差——其per-variable独立处理的归纳偏置不适合需要跨变量joint temporal patterns的连续控制

## 亮点与洞察
- **理论bound的实用价值**：明确了影响鲁棒性的关键因素——策略平滑性、critic敏感性、传感器可用率、故障持续性。这为设计鲁棒agent提供了原则性指导
- **Transformer vs Recurrence的深层解释**：Stateless Transformer处理所有变量jointly within单个序列，自注意力允许每个输出直接attend到所有可用历史token，自然跳过gap；而recurrent models的sequential state update在缺失输入时会diverge或丢失关键信息
- **实用的传感器模型**：两层Markov模型可模拟丰富的故障模式（快速个体故障、快速组故障、混合动态、慢恢复长中断）

## 局限与展望
- MuJoCo环境相对简单，更复杂的真实机器人任务有待验证
- 所有模型共享固定PPO配置和匹配的架构容量——更深入的架构搜索可能改变排名
- 理论bound依赖策略平滑性假设，对深度网络策略的tight估计仍有挑战
- 传感器故障模型假设mask独立于状态（Assumption 5.5），实际中状态与传感器状态可能相关

## 相关工作与启发
- **vs DRQN**: DRQN用LSTM处理部分观测但缺乏理论分析，且不针对传感器故障的时间结构
- **vs RLBenchNet**: RLBenchNet纯经验比较且掩码机制过于简化（永久删除速度/缩小观测窗），不建模真实传感器故障
- **vs Decision Transformer**: DT用于offline RL，本文聚焦online PPO下的鲁棒性

## 评分
- 新颖性: ⭐⭐⭐⭐ 传感器故障模型+理论bound+系统架构比较的组合有价值
- 实验充分度: ⭐⭐⭐⭐ 8种架构、4种环境、8种子、完全/部分观测对比，统计严谨
- 写作质量: ⭐⭐⭐⭐⭐ 理论清晰、解释直观、与先前工作对比充分
- 价值: ⭐⭐⭐⭐ 为鲁棒RL的架构选择提供理论支撑和经验指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Incremental Sequence Classification with Temporal Consistency](../../NeurIPS2025/reinforcement_learning/incremental_sequence_classification_with_temporal_consistency.md)
- [\[ICLR 2026\] TPRU: Advancing Temporal and Procedural Understanding in Large Multimodal Models](tpru_advancing_temporal_and_procedural_understanding_in_large_multimodal_models.md)
- [\[AAAI 2026\] DRMD: Deep Reinforcement Learning for Malware Detection under Concept Drift](../../AAAI2026/reinforcement_learning/drmd_deep_reinforcement_learning_for_malware_detection_under_concept_drift.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](../../NeurIPS2025/reinforcement_learning/robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)

</div>

<!-- RELATED:END -->
