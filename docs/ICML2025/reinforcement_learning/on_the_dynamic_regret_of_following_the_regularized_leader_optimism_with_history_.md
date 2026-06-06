---
title: >-
  [论文解读] On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning
description: >-
  [ICML 2025][强化学习][FTRL] 本文提出 OptFPRL 算法，通过在 Follow the Regularized Leader (FTRL) 框架中引入**历史梯度裁剪 (History Pruning)** 机制，首次为 FTRL 在紧凑集上建立了数据依赖的动态遗憾保证…
tags:
  - "ICML 2025"
  - "强化学习"
  - "FTRL"
  - "动态遗憾"
  - "在线凸优化"
  - "乐观学习"
  - "历史裁剪"
---

# On the Dynamic Regret of Following the Regularized Leader: Optimism with History Pruning

**会议**: ICML 2025  
**arXiv**: [2505.22899](https://arxiv.org/abs/2505.22899)  
**代码**: 未公开  
**领域**: 在线学习 / 在线凸优化  
**关键词**: FTRL, 动态遗憾, 在线凸优化, 乐观学习, 历史裁剪  

## 一句话总结

本文提出 OptFPRL 算法，通过在 Follow the Regularized Leader (FTRL) 框架中引入**历史梯度裁剪 (History Pruning)** 机制，首次为 FTRL 在紧凑集上建立了数据依赖的动态遗憾保证，动态遗憾完全由预测误差调控，在预测完美时可达零遗憾。

## 研究背景与动机

在线凸优化 (OCO) 是序贯决策的经典范式：在每个时隙 $t$，学习者选择动作 $\bm{x}_t \in \mathcal{X}$，环境揭示代价函数 $f_t(\cdot)$。衡量学习者性能的最严格指标是**动态遗憾**：

$$\mathcal{R}_T = \sum_{t=1}^{T} f_t(\bm{x}_t) - f_t(\bm{u}_t)$$

其中 $\{\bm{u}_t\}$ 是任意比较器序列，其复杂度由**路径长度** $P_T = \sum_{t=1}^{T-1} \|\bm{u}_{t+1} - \bm{u}_t\|$ 量化。

OCO 的两大算法家族是 **FTRL** 和 **OMD (Online Mirror Descent)**。对于静态遗憾（固定比较器），两者都能实现 $\mathcal{O}(\sqrt{T})$ 最优率。但在动态遗憾方面，FTRL 一直被认为不如 OMD，原因在于：

**状态膨胀问题**：FTRL 的状态是累积梯度 $\bm{g}_{1:t}$，会随时间无限增长

**已知负面结果**：Jacobsen et al. 证明即使路径长度恒定，标准 FTRL 也无法实现次线性动态遗憾

**缺乏数据依赖界**：之前 FTRL 动态遗憾的最好结果是 $\mathcal{O}(P^{1/3} T^{2/3})$（Ahn et al.），不依赖于数据且不是最优的

本文的核心洞察是：**FTRL 的动态遗憾瓶颈不在于"懒投影"方式，而在于状态（线性化历史）与迭代点的脱耦——允许状态无限增长**。通过裁剪同步二者，可以解决这一问题。

## 方法详解

### 整体框架：OptFPRL

OptFPRL 的更新规则为：

$$\bm{x}_{t+1} = \arg\min_{\bm{x}} \langle \bm{p}_{1:t}, \bm{x} \rangle + r_{1:t}(\bm{x}) + \tilde{f}_{t+1}(\bm{x}) + I_{\mathcal{X}}(\bm{x})$$

其中 $\bm{p}_{1:t}$ 是裁剪后的状态向量，$r_{1:t}(\bm{x}) = \frac{\sigma_{1:t}}{2}\|\bm{x}\|^2$ 是递增正则化，$\tilde{f}_{t+1}$ 是对下一步代价的预测，$I_{\mathcal{X}}$ 是集合指示函数。

### 关键设计：历史裁剪机制

每一步的状态向量为 $\bm{p}_t = \bm{g}_t + \bm{g}_t^I$，其中 $\bm{g}_t \in \partial f_t(\bm{x}_t)$ 是代价的次梯度，$\bm{g}_t^I \in \mathcal{N}_{\mathcal{X}}(\bm{x}_t)$ 来自法锥。裁剪规则为：

$$\bm{g}_t^I = \begin{cases} -(\bm{p}_{1:t-1} + \tilde{\bm{g}}_t + \sigma_{1:t-1}\bm{x}_t) & \text{if } \bm{x}_t^{\text{uc}} \notin \mathcal{X} \\ 0 & \text{otherwise} \end{cases}$$

直觉：当无约束迭代点落在可行域外（经过投影回边界），说明累积状态已经推动迭代点"过头"了。此时选择法锥元素来**裁剪累积历史**，用更小范数的替代状态 $\bm{p}_{1:t}$ 代替原始的 $\bm{g}_{1:t}$，同时保持产生相同的迭代点。

### 核心引理：乐观有界状态

**引理 4.3** 证明了裁剪后的状态范数有界：

$$\|\bm{p}_{1:t}\| \leq R\sigma_{1:t-1} + \epsilon_t$$

其中 $R$ 是集合半径，$\epsilon_t = \|\bm{g}_t - \tilde{\bm{g}}_t\|$ 是预测误差。这意味着状态增长由正则化参数（我们可以控制）约束，而非像标准 FTRL 那样与 $t$ 线性增长。

### 四种正则化策略及其遗憾界

1. **$P_T$-无关正则化**（定理 3.1）：$\sigma_{1:t} \propto \sqrt{E_t}$
   $$\mathcal{R}_T \leq (5.8R + \tfrac{1}{2}P_T)\sqrt{E_T} + H_T = \mathcal{O}((1+P_T)\sqrt{E_T})$$

2. **已知 $P_T$ 的正则化**（定理 3.2）：$\sigma \propto 1/\sqrt{P_T}$
   $$\mathcal{R}_T = \mathcal{O}((1+\sqrt{P_T})\sqrt{E_T})$$
   匹配极小极大最优率 $\mathcal{O}(\sqrt{(1+P_T)T})$。

3. **未知但可观测 $P_T$**（定理 3.3）：在线估计 $\sqrt{E_t/P_t}$
   $$\mathcal{R}_T = \mathcal{O}((1+\sqrt{P_T})\sqrt{E_T} + A_T)$$

4. **递归正则化（AdaFTRL 风格）**（定理 3.4）：$\sigma_t \propto \delta_t$（局部遗憾）
   $$\mathcal{R}_T \leq 1.1\,\delta_{1:T} + \sum_{t=1}^{T-1}\tfrac{1}{4R}\delta_{1:t}\|\bm{u}_{t+1}-\bm{u}_t\| + H_T$$
   提供更精细的界，因为 $\delta_{1:T} \leq \mathcal{O}(\sqrt{E_T})$ 但通常更小。

### 损失函数

本文是分析性工作，不涉及训练损失。分析的核心不等式来自**强动态乐观 FTRL 引理**（引理 4.1），将遗憾分解为三部分：
- **(I)** 不知道 $\bm{g}_t$ 对选择 $\bm{x}_t$ 的惩罚 → 由预测误差 $\epsilon_t$ 控制
- **(II)** 比较器非平稳性惩罚 → 由 $\|\bm{p}_{1:t}\| \cdot \|\bm{u}_{t+1}-\bm{u}_t\|$ 控制
- **正则化项** $r_t(\bm{u}_t)$ → 权衡前两项

## 实验关键数据

本文为纯理论工作，无数值实验。主要理论贡献的比较如下：

| 方法 | 动态遗憾界 | 预测完美时 | FTRL? |
|------|-----------|-----------|-------|
| Jadbabaie et al. (OMD) | $\mathcal{O}((1+P_T)\sqrt{E_T+1})$ | $\mathcal{O}(P_T)$ | ✗ |
| Scroccaro et al. (OMD) | $\mathcal{O}((1+P_T)(1+\sqrt{D_T}))$ | $\mathcal{O}(1+P_T)$ | ✗ |
| Zhang et al. (meta) | $\mathcal{O}(\sqrt{T(1+P_T)})$ | $\mathcal{O}(\sqrt{T(1+P_T)})$ | ✗ |
| Ahn et al. (FTRL) | $\mathcal{O}(P^{1/3}T^{2/3})$ | $\mathcal{O}(P^{1/3}T^{2/3})$ | ✓ |
| **OptFPRL（本文）** | $\mathcal{O}((1+\sqrt{P_T})\sqrt{E_T})$ | **0** | ✓ |

### 关键发现

- **首个 FTRL 的极小极大最优动态遗憾保证**：在 $P_T$ 已知时匹配 $\Omega(\sqrt{(1+P_T)T})$ 下界
- **零遗憾在完美预测下**：所有遗憾项（包括 $P_T$）都乘以预测误差，此前 OMD 方法的 $P_T$ 项独立于预测质量
- **混合项 $H_T$**：$H_T = \sum_t \epsilon_t \|\bm{u}_{t+1}-\bm{u}_t\|$ 精细刻画了预测误差和环境变化的交互

## 亮点与洞察

1. **FTRL 在动态环境中不是"天生劣势"**：此前认为 FTRL 的"懒"更新不适合非平稳环境，本文证明瓶颈在于状态增长而非投影方式
2. **裁剪 = 选择性遗忘**：通过法锥元素巧妙地从累积历史中"裁剪"冗余梯度，本质上实现了 FTRL 的选择性遗忘
3. **双重视角**：从对偶视角看，OptFPRL 等价于只保留最近一次裁剪以来的显式梯度历史，之前的历史通过 $\bm{x}_{t-k}$ 的对偶映射隐含编码
4. **递归正则化的优势**：AdaFTRL 风格的正则化提供"恰到好处"的正则化量，在动态环境中优势更大（因为正则化项嵌套在对 $[T]$ 的求和中）

## 局限性

1. **纯理论工作**：没有数值实验验证实际性能增益
2. **假设紧凑集**：结果依赖 $\mathcal{X}$ 有界（$\|\bm{x}\| \leq R$），无界域需要不同技术
3. **需要预测 $\tilde{f}_{t+1}$**：如何在实际问题中获取好的预测是一个独立挑战
4. **Euclidean 正则化器**：仅考虑了缩放 Euclidean 正则化器，更一般的 Bregman 散度值得探索
5. **已知 $P_T$ 的假设**：最优率需要预先知道或在线观测 $P_T$

## 相关工作

- **乐观在线学习**：从 AdaGrad → 乐观 FTRL/OMD → 数据依赖遗憾界
- **动态遗憾 OMD 线**：Jadbabaie et al., Zhang et al. (meta-learning), Zhao et al., Scroccaro et al.
- **FTRL 动态遗憾**：Ahn et al. (几何折扣), Jacobsen et al. (centered OMD)
- **OMD/FTRL 互联**：Fang et al. (Dual Averaging ≈ 改进 OMD), McMahan survey

## 评分

⭐⭐⭐⭐ (4/5)

理论贡献重要且优美——首次为 FTRL 建立紧凑集上的极小极大最优动态遗憾保证，裁剪机制的设计简洁而深刻。缺少实验验证是遗憾，但对理论工作而言属于可接受范围。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Non-stationary Online Learning for Curved Losses: Improved Dynamic Regret via Mixability](non-stationary_online_learning_for_curved_losses_improved_dynamic_regret_via_mix.md)
- [\[NeurIPS 2025\] Dynamic Regret Reduces to Kernelized Static Regret](../../NeurIPS2025/reinforcement_learning/dynamic_regret_reduces_to_kernelized_static_regret.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](../../ICML2026/reinforcement_learning/parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[NeurIPS 2025\] Generalizing Verifiable Instruction Following](../../NeurIPS2025/reinforcement_learning/generalizing_verifiable_instruction_following.md)
- [\[ICML 2025\] Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation](demystifying_the_paradox_of_importance_sampling_with_an_estimated_history-depend.md)

</div>

<!-- RELATED:END -->
