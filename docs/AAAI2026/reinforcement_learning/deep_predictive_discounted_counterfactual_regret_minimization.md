---
title: >-
  [论文解读] Deep (Predictive) Discounted Counterfactual Regret Minimization
description: >-
  [AAAI 2026][强化学习][反事实遗憾最小化] 提出VR-DeepDCFR+和VR-DeepPDCFR+两种无模型神经CFR算法，通过自举累积优势估计、折扣裁剪机制和基线方差缩减，首次将高级表格CFR变体（DCFR+/PDCFR+）有效整合到神经网络近似框架中，在典型不完全信息博弈中实现更快收敛。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "反事实遗憾最小化"
  - "不完全信息博弈"
  - "纳什均衡"
  - "神经网络近似"
  - "方差缩减"
---

# Deep (Predictive) Discounted Counterfactual Regret Minimization

**会议**: AAAI 2026  
**arXiv**: [2511.08174](https://arxiv.org/abs/2511.08174)  
**代码**: [rpSebastian/DeepPDCFR](https://github.com/rpSebastian/DeepPDCFR)  
**领域**: 强化学习  
**关键词**: 反事实遗憾最小化, 不完全信息博弈, 纳什均衡, 神经网络近似, 方差缩减

## 一句话总结

提出VR-DeepDCFR+和VR-DeepPDCFR+两种无模型神经CFR算法，通过自举累积优势估计、折扣裁剪机制和基线方差缩减，首次将高级表格CFR变体（DCFR+/PDCFR+）有效整合到神经网络近似框架中，在典型不完全信息博弈中实现更快收敛。

## 研究背景与动机

不完全信息博弈（IIG）是建模多玩家隐信息战略交互的基础框架，核心目标是计算（近似）纳什均衡。CFR（反事实遗憾最小化）算法族是求解IIG最成功的方法之一，通过迭代最小化累积反事实遗憾使平均策略收敛到NE。

近年来，表格CFR领域涌现了多种加速收敛的变体：
- **CFR+**：裁剪负累积遗憾 + 线性加权平均策略
- **DCFR**：对累积遗憾施加折扣
- **DCFR+**：结合CFR+和DCFR的优势
- **PDCFR+**：利用遗憾的可预测性加速收敛

然而，现有神经CFR方法（如DeepCFR、DREAM）主要近似vanilla CFR或LinearCFR的行为，**无法有效整合更高级的CFR变体**。核心困难在于：DCFR+和PDCFR+的更新依赖前一迭代的累积反事实遗憾（bootstrap），而传统neural CFR通过replay buffer中所有迭代的样本从头拟合，两者在架构上不兼容。更深层的问题是：反事实值是由对手到达概率加权的期望效用，这些非归一化值在不同信息集间数量级差异巨大，网络难以有效学习。

## 方法详解

### 整体框架

算法的关键思路：**用累积优势（advantage）替代累积反事实遗憾**。优势是反事实遗憾除以对手到达概率的结果：$r_i^t(I,a) = \pi_{-i}^{\sigma^t}(I) \cdot A_i^{\sigma^t}(I,a)$，数值尺度更统一，神经网络更容易学习和泛化。

整体流程（每轮迭代）：
1. 用结果采样（outcome sampling）收集K条episode
2. 利用值网络计算方差缩减的采样优势
3. 通过自举（bootstrapping）更新累积优势网络
4. 对累积优势施加折扣和裁剪，模拟DCFR+/PDCFR+的行为
5. 通过遗憾匹配（regret matching）从累积优势计算新策略

### 关键设计

**1. 自举累积优势估计**

传统DeepCFR的replay buffer保留所有迭代样本，从头重新拟合累积遗憾。这与DCFR+的bootstrap更新不兼容。本文改为：每轮迭代清空buffer，仅用当前迭代采样，通过前一迭代网络的输出进行自举。

调整采样反事实值为：$\check{v}_i^{\sigma^t}(I,a|z) = \frac{\pi^{\sigma^t}(z[I]a,z) \cdot u_i(z)}{\pi^{\xi^t}(z[I],z)}$

关键定理（Theorem 2）证明其期望等于优势：$\mathbb{E}[\check{r}_i^t(I,a)|z \in Z_I] = A_i^{\sigma^t}(I,a)$

累积优势网络 $R(I,a|\theta_i^t)$ 的训练损失基于bootstrap：

$$\mathcal{L}(\theta_i^t) = \mathbb{E}_{(I,\check{r}) \sim \mathcal{B}_{V,i}}\left[\sum_a \left(R(I,a|\theta_i^{t-1}) + \check{r}(I,a) - R(I,a|\theta_i^t)\right)^2\right]$$

**2. 近似DCFR+**

在bootstrap损失中加入折扣和裁剪操作：

$$\mathcal{L}(\theta_i^t) = \mathbb{E}\left[\sum_a \left(\max(R(I,a|\theta_i^{t-1}), 0) \cdot \frac{(t-1)^\alpha}{(t-1)^\alpha + 1} + \check{r}(I,a) - R(I,a|\theta_i^t)\right)^2\right]$$

- $\max(\cdot, 0)$ 裁剪负累积优势（来自CFR+的思想，减少错误动作的代价）
- $(t-1)^\alpha / ((t-1)^\alpha + 1)$ 折扣因子（来自DCFR，降低早期不准确估计的权重）

**3. 近似PDCFR+**

在DCFR+基础上增加预测机制：额外训练一个瞬时优势网络 $r(I,a|\phi_i^t)$ 估计当前迭代优势，用于预测下一迭代的累积优势：

$$\max\left(R(I,a|\theta_i^t), 0\right) \cdot \frac{t^\alpha}{t^\alpha + 1} + r(I,a|\phi_i^t)$$

预测的累积优势通过regret matching计算新策略，利用了反事实遗憾变化缓慢的可预测性。

**4. 基线方差缩减**

采样单条episode的方差很大，引入历史值网络 $Q(h,a|w^t)$ 作为基线函数（受DREAM启发）：

$$\bar{v}_i^{\sigma^t}(I,a|z) = \begin{cases} Q_i(h,a|w^{t-1}) + \frac{\bar{v}_i(I'|z) - Q_i(h,a|w^{t-1})}{\xi^t(I,a)} & \text{if } a = \hat{a} \\ Q_i(h,a|w^{t-1}) & \text{otherwise} \end{cases}$$

对于采样到的动作，用值网络预测和实际采样差异的重要性采样修正；对于未采样动作，直接使用值网络估计。

### 损失函数 / 训练策略

三个网络联合训练：
- **累积优势网络** $R$：自举+折扣裁剪损失
- **瞬时优势网络** $r$（仅PDCFR+）：标准回归损失
- **历史值网络** $Q$：类似DQN的TD损失，off-policy训练
- **平均策略网络** $\Pi$：加权回归损失，权重 $(t/T)^\gamma$ 使后期策略重要性更高

## 实验关键数据

### 主实验（收敛到均衡）

在8个标准IIG上比较7种无模型神经算法的可利用度（exploitability）收拾速度。主要对比：

| 方法 | Kuhn Poker | Leduc Poker | Liar's Dice | 其他5个游戏 |
|------|-----------|-------------|-------------|------------|
| QPG/RPG | 仅Kuhn收敛到0.01 | 表现差 | 表现差 | 表现差 |
| NFSP | 收敛慢 | 收敛慢 | 收敛慢 | 中等 |
| OS-DeepCFR | 中等 | 中等 | 中等 | 中等 |
| DREAM | 较快 | 较快 | 较快 | 较快 |
| **VR-DeepDCFR+** | **最快** | **最快** | **最快** | **多数最快** |
| **VR-DeepPDCFR+** | **最快** | **最快** | **最快** | **多数最快** |

VR-DeepDCFR+和VR-DeepPDCFR+在大多数游戏中收敛速度最快。

### 大规模扑克对抗评估

在Flop Hold'em Poker (FHP)上对阵5种风格的规则agent，每种对战20000局：

| 方法 | 平均奖励（chips/手） |
|------|-------------------|
| OS-DeepCFR | -7.8 ± 1.4 |
| DREAM | -2.0 ± 3.1 |
| **VR-DeepDCFR+** | **11.6 ± 1.2** |
| **VR-DeepPDCFR+** | **11.3 ± 0.9** |

专业德扑比赛中，每手5 chips的差距被视为显著技术差距。本文方法平均赢11+ chips，远超其他neural CFR。

### 消融实验

以VR-DeepPDCFR+为基础在4个IIG上消融三个组件：
- 去掉bootstrap累积优势 → 退化为类似DeepCFR的行为，收敛变慢
- 去掉高级CFR变体（折扣+裁剪）→ 退化为近似vanilla CFR
- 去掉基线方差缩减 → 方差增大，训练不稳定

三个组件均对性能有贡献。

### 关键发现

1. 累积优势比累积反事实遗憾的方差更小，网络训练更稳定
2. Bootstrap方式避免了大replay buffer的存储开销和从头重训的计算开销
3. VR-DeepDCFR+运行时间与DREAM相当（主要差异在损失公式），但收敛更快
4. 两种方法在所有游戏中使用相同超参数，泛化性好

## 亮点与洞察

1. **优势替代遗憾的核心洞察**：反事实遗憾 = 对手到达概率 × 优势，除以对手概率后数值尺度统一，这个简单转换解决了neural CFR的核心瓶颈
2. **Bootstrap + 折扣裁剪的巧妙结合**：每轮迭代只需当前采样数据，配合前一轮网络输出，实现了对高级CFR变体的忠实模拟
3. **理论严谨**：Theorem 1和2提供了采样估计的无偏性证明，算法设计有坚实理论基础
4. **工程简洁**：运行时间与DREAM相当，额外开销极小，实用性强

## 局限与展望

1. PDCFR+的优势预测假设变化缓慢（用当前迭代预测下一迭代），可考虑RNN捕捉时序依赖
2. 仅在两人零和博弈中验证，多人/非零和场景的扩展未探索
3. 值网络的训练质量直接影响方差缩减效果，大游戏中值网络精度可能成为瓶颈
4. 超参数 $\alpha, \gamma$ 虽然跨游戏统一，但最优值的选择缺乏理论指导

## 相关工作与启发

- **DeepCFR / OS-DeepCFR**：用神经网络近似CFR的开创工作，但限于vanilla/Linear CFR
- **DREAM**：引入值函数基线减少方差，本文继承其方差缩减方法并扩展到高级变体
- **ESCHER**：直接用值函数计算遗憾，但训练开销大；本文选择DREAM的轻量方案
- **DCFR+ / PDCFR+**：表格CFR的最新进展，本文首次实现其神经网络近似
- 对idea启发：将高级算法变体的核心操作（折扣、裁剪、预测）解耦为可在神经网络损失中实现的模块化设计

## 评分

- 新颖性: ⭐⭐⭐⭐ （首次将DCFR+/PDCFR+有效整合到neural CFR）
- 技术深度: ⭐⭐⭐⭐⭐ （理论证明+算法设计+方差分析均扎实）
- 实验充分性: ⭐⭐⭐⭐ （8个博弈+大规模扑克对抗+消融，覆盖全面）
- 写作质量: ⭐⭐⭐⭐ （前置知识铺垫充分，推导清晰）
- 实用价值: ⭐⭐⭐⭐ （代码开源，FHP上的对抗表现有实际意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Beyond the Lower Bound: Bridging Regret Minimization and Best Arm Identification in Lexicographic Bandits](beyond_the_lower_bound_bridging_regret_minimization_and_best_arm_identification_.md)
- [\[AAAI 2026\] Discounted Cuts: A Stackelberg Approach to Network Disruption](discounted_cuts_a_stackelberg_approach_to_network_disruption.md)
- [\[NeurIPS 2025\] Simultaneous Swap Regret Minimization via KL-Calibration](../../NeurIPS2025/reinforcement_learning/simultaneous_swap_regret_minimization_via_kl-calibration.md)
- [\[AAAI 2026\] DeepProofLog: Efficient Proving in Deep Stochastic Logic Programs](deepprooflog_efficient_proving_in_deep_stochastic_logic_programs.md)
- [\[NeurIPS 2025\] Comparing Uniform Price and Discriminatory Multi-Unit Auctions through Regret Minimization](../../NeurIPS2025/reinforcement_learning/comparing_uniform_price_and_discriminatory_multi-unit_auctions_through_regret_mi.md)

</div>

<!-- RELATED:END -->
