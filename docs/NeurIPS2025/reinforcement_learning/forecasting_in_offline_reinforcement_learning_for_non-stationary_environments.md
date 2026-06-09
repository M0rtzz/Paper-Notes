---
title: >-
  [论文解读] Forecasting in Offline Reinforcement Learning for Non-stationary Environments
description: >-
  [NeurIPS 2025][强化学习][离线强化学习] 提出 Forl 框架，将条件扩散模型生成的多模态候选状态与零样本时序基础模型的偏移预测通过维度最近匹配（DCM）融合，在测试时无需重训练即可应对观测函数随 episode 非平稳变化的离线 RL 部署场景，在 D4RL 标准基准上平均提升数十分。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "离线强化学习"
  - "非平稳环境"
  - "扩散模型"
  - "时序预测"
  - "状态估计"
---

# Forecasting in Offline Reinforcement Learning for Non-stationary Environments

**会议**: NeurIPS 2025  
**arXiv**: [2512.01987](https://arxiv.org/abs/2512.01987)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: 离线强化学习, 非平稳环境, 扩散模型, 时序预测, 状态估计

## 一句话总结
提出 Forl 框架，将条件扩散模型生成的多模态候选状态与零样本时序基础模型的偏移预测通过维度最近匹配（DCM）融合，在测试时无需重训练即可应对观测函数随 episode 非平稳变化的离线 RL 部署场景，在 D4RL 标准基准上平均提升数十分。

## 研究背景与动机

**领域现状**：离线 RL 通过预收集的静态数据训练策略，避免了在线交互的成本和风险。现有鲁棒离线 RL 方法主要考虑传感器噪声或对抗扰动等测试时扰动。

**现有痛点**：真实部署中常遇到的一类非平稳性是**每个 episode 的观测函数被加入未知的、恒定的偏移量**——如工业机器人每日校准偏差、传感器漂移、数据聚合中的系统性偏差。这些偏移违反了标准方法假设的平滑/高斯噪声模式，且 episode 间按非马尔可夫时序模式演变。

**核心矛盾**：智能体在训练时面对的是完全可观测的平稳 MDP，但测试时面对的是一系列 POMDP（每个 episode 的观测 $o_t = s_t + b^j$ 中偏移 $b^j$ 未知且不可访问）。单次观测无法将真实状态 $s_t$ 和偏移 $b^j$ 分离——存在无穷多可能的 $(s', b')$ 组合使得 $o_t = s' + b'$。

**本文目标**：(a) 如何在不知道未来非平稳模式的情况下训练模型？(b) 如何融合 episode 内的动作-效果历史和 episode 间的偏移预测来估计真实状态？(c) 能否做到零样本适应而无需重训练或在线探索？

**切入角度**：利用两个互补的信息源：(i) episode 内的 $(\Delta o, a)$ 序列蕴含位置约束（如撞墙意味着可能位于墙南方若干步处）→ 用条件扩散模型捕获多模态信念；(ii) 历史偏移值的时序模式 → 用零样本时序基础模型预测未来偏移。

**核心 idea**：条件扩散模型做"回顾性"多模态状态猜测 + 时序基础模型做"前瞻性"偏移预测，DCM 融合两者得到最优状态估计。

## 方法详解

### 整体框架
训练阶段：在平稳离线数据上训练条件扩散模型（Forl-Dm）和一个标准的离线 RL 策略（如 Diffusion-QL）。测试阶段：每 $P$ 个 episode 用零样本时序模型预测未来 $P$ 个 episode 的偏移量 $\hat{b}^j$；在每个 episode 内，用轨迹 buffer 积累 $(\Delta o, a)$ 序列后，扩散模型生成候选状态，通过 DCM 与预测偏移融合得到最终状态估计，输入已训练好的策略选动作。

### 关键设计

1. **Forl 条件扩散模型**:

    - 功能：从 episode 内的动作-效果历史生成多模态候选真实状态
    - 核心思路：定义轨迹子序列 $\boldsymbol{\tau}_{(t,w)} = [(\Delta o_{t-w+1}, a_{t-w}), \cdots, (\Delta o_t, a_{t-1})]$，训练噪声预测网络 $\boldsymbol{\epsilon}_\theta(\boldsymbol{s}_t^{(n)}, \boldsymbol{\tau}_{(t,w)}, n)$，损失函数为 $\mathcal{L}_p(\theta) = \mathbb{E}_{n, \boldsymbol{\tau}, \boldsymbol{s_t}, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{s}^{(n)}, \boldsymbol{\tau}_{(t,w)}, n)\|^2\right]$
    - 正向扩散在真实状态 $s_t$ 上加噪：$s_t^{(n)} = \sqrt{\bar{\alpha}(n)} s_t + \sqrt{1-\bar{\alpha}(n)} \boldsymbol{\epsilon}$
    - 推理时并行生成 $k$ 个候选状态 $\{s_t^{(0)}\}_k$
    - 设计动机：偏移后的可能状态分布是高度多模态的——例如在迷宫中撞墙后可能在 4 个不同位置。扩散模型天然适合生成多模态分布
    - **关键洞察**：模型仅在无偏移的平稳数据上训练，但 $\Delta o_t = s_t - s_{t-1}$（与偏移无关），所以条件变量 $\tau$ 在有偏移时仍然正确——这是方法免训练适应的理论基础

2. **零样本时序基础模型预测**:

    - 功能：从历史偏移序列预测未来 $P$ 个 episode 的偏移量
    - 核心思路：使用 Lag-Llama 等概率性零样本时序模型，以过去 $C$ 个偏移值 $(b^{j-C}, \dots, b^{j-1})$ 为上下文，生成 $l$ 个偏移样本。各维度独立预测（因 Lag-Llama 是单变量模型）
    - 设计动机：偏移遵循任意的真实世界时序模式（可能是非马尔可夫的），零样本模型无需针对特定模式训练

3. **维度最近匹配 DCM（Dimension-wise Closest Match）**:

    - 功能：融合多模态扩散候选和单模态时序预测
    - 核心思路：设 $\mathcal{D}_{\text{diffusion}} = \{\mathbf{x}_1, \dots, \mathbf{x}_k\}$, $\mathcal{D}_{\text{timeseries}} = \{\mathbf{y}_1, \dots, \mathbf{y}_l\}$，构造估计 $z_d = y_{j^*(d), d}$，其中 $j^*(d) = \arg\min_j (\min_i |x_{i,d} - y_{j,d}|)$——每个维度选择在扩散候选中有最近匹配的时序预测样本
    - 设计动机：直接用扩散模型的样本可能错选不正确的模态；直接用时序预测可能偏差大。DCM 利用扩散模型提供的约束来筛选时序预测中最可信的估计，非参数、轻量级、无需调参

4. **Episode 内自适应流程**:

    - 前 $w$ 步（窗口期）：仅使用时序预测的均值 $\tilde{s}_t = o_t - \overline{\hat{b}^j}$
    - $w$ 步之后：扩散模型有足够上下文生成有意义的候选，启用 DCM 融合
    - 设计动机：扩散模型需要一定长度的历史才能有效约束，初期依赖预测模型的先验

## 实验关键数据

### 主实验（D4RL + 真实世界时序偏移）

| 环境 | DQL | DQL+Lag-$\bar{s}$ | Dmbp+Lag | **Forl** |
|------|-----|-------------------|----------|----------|
| maze2d-medium (avg 5 datasets) | 10.0 | 37.2 | 34.6 | **74.1** |
| maze2d-large (avg) | 3.5 | 7.1 | 12.1 | **38.8** |
| antmaze-umaze-diverse (avg) | 16.1 | 44.9 | 51.4 | **75.1** |
| antmaze-medium-diverse (avg) | 18.9 | 42.1 | 40.1 | **48.7** |
| antmaze-large-diverse (avg) | 6.0 | 12.0 | 10.5 | **34.6** |
| kitchen-complete (avg) | 13.8 | 18.1 | 16.7 | **21.3** |

### 消融实验

| 配置 | antmaze-umaze 平均归一化分 | 说明 |
|------|--------------------------|------|
| Forl (full) | **75.1** | 扩散模型 + 时序预测 + DCM |
| DQL + Lag-$\bar{s}$（仅时序预测均值） | 44.9 | 无扩散模型回顾修正 |
| Dmbp + Lag（扩散+反向传播优化） | 51.4 | 非 DCM 融合策略 |
| DQL（无任何适应） | 16.1 | 完全受偏移影响，性能极差 |
| Forl-F（FQL 策略替换 DQL） | 45.7 (cube-single) | 策略无关性：可搭配不同离线 RL 策略 |

### 关键发现
- **Forl 在所有环境和所有真实世界时序数据上一致优于基线**，尤其在 maze2d 和 antmaze 系列中提升幅度巨大（平均 +30~60 分）
- **扩散模型的多模态能力是核心**：仅用时序预测（DQL+Lag）只能修正"预期偏移"，无法处理高不确定性场景
- **DCM 融合优于反向传播优化**（Dmbp+Lag），因为 DCM 是非参数的、不会过拟合于不准确的预测
- Forl 是**策略无关的**（policy-agnostic），可以作为插件搭配任何离线 RL 策略
- 在 kitchen 等高维环境中改善有限——可能是扩散模型在高维度下地标约束力减弱

## 亮点与洞察
- **$\Delta o$ 与偏移 $b^j$ 无关的洞察**是整个方法的理论根基——状态变化量不受恒定偏移影响，使得扩散模型可以仅在无偏数据上训练就泛化到有偏场景，非常优雅
- **DCM 的轻量简洁**：一个维度级别的最近邻匹配就实现了多模态与单模态分布的有效融合，在理想采样下近似于乘积分布
- **"回顾 + 前瞻"双通道设计**：episode 内的动作-效果历史做回顾性推理（约束可能位置），episode 间的偏移序列做前瞻性预测（预判未来偏移），两者互补

## 局限与展望
- **偏移假设较强**：仅考虑每个 episode 内恒定的加性偏移，不处理乘性、时变或动力学层面的非平稳性
- **时序预测精度瓶颈**：零样本基础模型（Lag-Llama）在复杂时序模式下预测不准会影响整体表现（如 real-data-E 效果较差）
- **高维状态空间的扩展性**：kitchen 和 cube 环境改善有限——扩散模型在高维中需要更多样本或更强架构
- **ground-truth 偏移的获取假设**：假设每 $P$ 个 episode 后能获取过去偏移的真实值用于时序预测，在某些场景中可能不现实
- **仅在 D4RL 上验证**，缺少真实机器人或工业场景的端到端实验

## 相关工作与启发
- **vs 标准鲁棒离线 RL**：通常假设 bounded random noise 或对抗扰动，而本文处理的是 episode 级别恒定偏移 + 非马尔可夫演变——完全不同的非平稳模式
- **vs Diffuser/Decision Diffuser 等扩散 RL**：它们用扩散模型做规划/策略生成，而 Forl 用扩散模型做状态估计/信念更新——同一工具的不同应用角度
- **vs 在线 POMDP 方法**：POMDP 方法通常需要在训练时就考虑部分可观测性，而 Forl 在 MDP 上训练、在 POMDP 上部署——零样本迁移
- 该框架对任何需要处理传感器漂移/校准偏差的 RL 部署场景都有实际参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义新颖（episode 级非平稳观测），条件扩散+时序预测+DCM 的组合巧妙
- 实验充分度: ⭐⭐⭐⭐ 5 个真实世界时序数据 × 8 个 D4RL 环境，消融充分，但缺少真实机器人验证
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法描述清晰，图表直观（尤其 Fig.3 的迷宫例子很好理解）
- 价值: ⭐⭐⭐⭐ 填补了离线 RL 部署中非平稳观测适应的空白，框架模块化易于扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[NeurIPS 2025\] Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [\[NeurIPS 2025\] RoiRL: Efficient, Self-Supervised Reasoning with Offline Iterative Reinforcement Learning](roirl_efficient_self-supervised_reasoning_with_offline_iterative_reinforcement_l.md)
- [\[NeurIPS 2025\] Reasoning Gym: Reasoning Environments for Reinforcement Learning with Verifiable Rewards](reasoning_gym_reasoning_environments_for_reinforcement_learning_with_verifiable_.md)
- [\[ICML 2025\] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](../../ICML2025/reinforcement_learning/non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)

</div>

<!-- RELATED:END -->
