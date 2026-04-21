---
title: >-
  [论文解读] Maximum Total Correlation Reinforcement Learning
description: >-
  [ICML2025][total correlation] 提出最大化轨迹总相关（Total Correlation）作为 RL 的归纳偏置，鼓励策略产生简单、可压缩的轨迹，从而在不牺牲任务性能的前提下显著提升对观测噪声、动作噪声和动力学变化的零样本鲁棒性。
tags:
  - ICML2025
  - total correlation
  - 轨迹一致性
  - 信息论正则化
  - SAC
  - 鲁棒性
  - 可压缩行为
---

# Maximum Total Correlation Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2505.16734](https://arxiv.org/abs/2505.16734)  
**代码**: [GitHub](https://github.com/BangYou01/MTC)  
**领域**: reinforcement_learning / 鲁棒强化学习  
**关键词**: total correlation, 轨迹一致性, 信息论正则化, SAC, 鲁棒性, 可压缩行为

## 一句话总结

提出最大化轨迹总相关（Total Correlation）作为 RL 的归纳偏置，鼓励策略产生简单、可压缩的轨迹，从而在不牺牲任务性能的前提下显著提升对观测噪声、动作噪声和动力学变化的零样本鲁棒性。

## 研究背景与动机

强化学习策略容易拾取高维感知输入中的虚假相关性，导致脆弱策略在轻微状态变化下失效。现有方法从不同角度引入简洁性归纳偏置：

- **RPC**（Eysenbach et al., 2021）：最小化原始状态序列与嵌入序列间的互信息，但仅关注状态一致性，且只利用单步转移信息
- **LZ-SAC / SPAC**（Saanum et al., 2023）：提升动作序列的可预测性，但仅关注动作一致性
- **领域随机化**：通过多样训练分布提升迁移能力，但无法保证覆盖所有真实场景

**核心问题**：上述方法要么只关注状态一致性、要么只关注动作一致性，缺乏对完整行为（状态+动作的完整轨迹）的统一度量。

**本文思路**：用信息论中的总相关（Total Correlation）衡量整条轨迹的可压缩性——总相关越大，说明轨迹越一致、越可压缩、越偏向开环行为（如干净的周期步态），从而天然更鲁棒。

## 方法详解

### 问题建模：MTC-RL

在标准 MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},p,r,T)$ 基础上，引入状态编码器 $f_\theta(z_t|s_t)$ 将原始状态映射到隐变量 $z_t$，策略 $\pi_\phi(a_t|s_t)$ 基于隐表示做决策。MTC-RL 的优化目标为：

$$\max_{\theta,\phi} \; \mathbb{E}_{\pi_\phi,f_\theta}\left[\sum_{t=1}^T r(s_t,a_t) + \alpha \cdot \mathcal{C}(z_1;a_1;\dots;a_{T-1};z_T)\right]$$

其中 $\mathcal{C}$ 为轨迹上隐状态和动作的总相关：

$$\mathcal{C}(x_1;\dots;x_n) = \mathbb{E}\left[\log \frac{p(x_1,\dots,x_n)}{\prod_{i=1}^n p(x_i)}\right]$$

直观含义：联合编码相比独立编码能节省的信息量。

### 变分下界推导

总相关无法直接分解为逐步奖励，因此引入两个参数化模型来构造变分下界：

1. **基于历史的隐动态模型** $q_\eta(z_{t+1}|z_{1:t},a_{1:t})$：用历史预测下一步隐状态
2. **基于历史的动作预测模型** $q_\chi(a_t|z_{1:t},a_{1:t-1})$：用历史预测当前动作

下界形式为：

$$\widetilde{\mathcal{C}} = \mathbb{E}_{\pi,f}\left[\sum_{t=1}^{T-1}\log\frac{q_\eta(z_{t+1}|z_{1:t},a_{1:t})\cdot q_\chi(a_t|z_{1:t},a_{1:t-1})}{f_\theta(z_{t+1}|s_{t+1})\cdot \pi_\phi(a_t|s_t)}\right]$$

**直觉**：当隐状态和动作能被历史很好地预测时（分子大），且编码器/策略的不可约不确定性较高时（分母大削弱贡献），下界值大，即轨迹一致性高。

### 正则化奖励函数

将下界代入目标，得到信息正则化奖励：

$$r^*(s_t,a_t,s_{t+1}) = r(s_t,a_t) + \alpha\left(\log\frac{q_\eta(z_{t+1}|z_{1:t},a_{1:t})\cdot q_\chi(a_t|z_{1:t},a_{1:t-1})}{f_\theta(z_{t+1}|s_{t+1})\cdot \pi_\phi(a_t|s_t)}\right)$$

该奖励偏置策略趋向：(1) 产生隐状态可由历史预测的转移；(2) 选择可由历史预测的动作。

### MTC-SAC 实现

基于 SAC 实现，核心修改：

- **策略评估**：仅替换奖励函数 $r \to r^*$，其余（target net、dual Q-net）不变
- **策略改进**：联合优化策略 $\pi_\phi$、编码器 $f_\theta$、动态模型 $q_\eta$、动作预测模型 $q_\chi$
- **自适应 $\alpha$**：通过对偶优化自动调节，最小化 $L(\alpha) = \alpha(\widetilde{\mathcal{C}} - I_p)$，其中 $I_p$ 为目标下界值

### 与 RPC 的理论联系

MTC 的正则项是 RPC 的推广：RPC 仅使用当前步信息做单步预测，MTC 使用完整历史做序列预测，并额外包含动作预测模型。作者在附录中证明 RPC 可由 MTC 框架推导出来，但反之不行。

## 实验关键数据

### 基础性能（DMC 8个任务，100万步，20种子）

| 任务 | MTC | RPC | LZ-SAC | SPAC | SAC |
|------|-----|-----|--------|------|-----|
| Acrobot Swingup | **184±24** | 132±31 | 100±22 | 110±29 | 154±29 |
| Hopper Stand | **933±12** | 568±96 | 593±88 | 213±69 | 683±114 |
| Finger Spin | **985±2** | 869±19 | 805±38 | 136±121 | 955±18 |
| Walker Walk | **967±2** | 940±21 | 939±26 | 883±76 | 962±7 |
| Cheetah Run | **874±21** | 772±57 | 787±17 | 458±52 | 811±36 |
| Quadruped Walk | **944±5** | 842±77 | 595±110 | 505±185 | 738±93 |
| Walker Run | **790±9** | 778±25 | 732±22 | 347±95 | 767±13 |
| Walker Stand | 983±2 | 980±5 | 977±2 | 931±38 | **985±2** |

MTC 在 8 个任务中 7 个取得最优或极具竞争力的成绩。

### 零样本鲁棒性

- **观测噪声**：MTC 在所有噪声强度 $\sigma \in [0.02, 0.1]$ 下聚合性能最优
- **动作噪声**：MTC 在强动作扰动下仍保持最高平均奖励
- **动力学失配**（质量缩放 0.25×—1.75×）：MTC 在小幅动力学变化下得分最高
- **虚假相关**：在 Walker Stand 添加不可控高斯状态维度后，MTC 显著优于 RPC 和 SAC

### 轨迹可压缩性

使用 bzip2 无损压缩轨迹文件，MTC 策略产生的轨迹压缩后文件最小，表明其行为最具周期性和结构化特征。

### 操控任务（MetaWorld）与图像任务

- MetaWorld 3个操控任务：MTC 成功率与 baseline 持平或更优，说明方法不限于周期性任务
- 6 个图像输入 DMC 任务（Planet benchmark）：MTC 优于 RPC、CURL、SAC-AE、SAC

## 亮点与洞察

1. **统一视角**：总相关将状态一致性和动作一致性统一到一个信息论框架中，不再需要分开处理
2. **理论优雅**：通过变分下界将不可分解的轨迹级目标转化为逐步可加的正则化奖励，与标准 RL pipeline 无缝集成
3. **无需修改网络结构**：基于 SAC 仅改奖励函数 + 添加辅助模型，工程实现简洁
4. **自适应权重**：通过拉格朗日对偶自动调节 $\alpha$，免除超参数调优
5. **可解释性**：产生可视化上周期性、简洁的轨迹，更易被人类理解和预测

## 局限与展望

1. **下界始终为负**：由于是负 KL 散度之和，无法估计真实总相关值，仅用于优化方向
2. **历史依赖的奖励**：正则化奖励依赖完整历史，增加了计算开销；但消融实验显示仅用当前状态输入策略的性能差异不大
3. **非平稳奖励**：奖励随策略/编码器参数变化而変化，理论上可能导致训练不稳定，但实验未观察到
4. **off-policy 训练预测模型**：回放缓冲区上训练预测模型偏离了推导假设，可能增大下界间隙
5. **动力学失配**下大幅变化时优势减弱（质量缩放 > 1.5× 时表现趋同）

## 相关工作与启发

- **RPC**（NeurIPS 2021）：MTC 的特例（仅单步、无动作预测），MTC 可推导出 RPC
- **LZ-SAC / SPAC**：仅关注动作一致性，MTC 同时关注状态+动作
- **InfoMax 原则**：MTC 延伸至轨迹级互依赖性最大化
- **DiMAE、CURL 等表示学习**：MTC 的编码器训练目标同时优化表示一致性与策略性能

## 评分

- 新颖性: ⭐⭐⭐⭐ — 总相关作为 RL 正则项是全新视角，统一了状态/动作一致性
- 实验充分度: ⭐⭐⭐⭐⭐ — 8+3+6 任务、多维度鲁棒性测试、可压缩性分析、消融实验完备
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，动机阐述到位，但下界负值的讨论可更早给出
- 价值: ⭐⭐⭐⭐ — 为 RL 鲁棒性提供了简洁、通用的信息论工具，实用性强

<!-- RELATED:START -->

## 相关论文

- [Risk-Averse Total-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/risk-averse_total-reward_reinforcement_learning.md)
- [Hierarchical Reinforcement Learning with Targeted Causal Interventions](hierarchical_reinforcement_learning_with_targeted_causal_interventions.md)
- [Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [Extreme Value Policy Optimization for Safe Reinforcement Learning](extreme_value_policy_optimization_for_safe_reinforcement_learning.md)
- [Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)

<!-- RELATED:END -->
