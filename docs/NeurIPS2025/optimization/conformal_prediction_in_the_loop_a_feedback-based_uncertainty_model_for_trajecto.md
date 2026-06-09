---
title: >-
  [论文解读] Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization
description: >-
  [NeurIPS 2025][优化/理论][共形预测] 提出 Feedback-Based Conformal Prediction (Fb-CP) 框架，将已执行轨迹的信息反馈给 CP 以动态调整预测区域大小，在缩减时域轨迹优化中同时保证覆盖率和显著提升轨迹性能。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "共形预测"
  - "轨迹优化"
  - "不确定性量化"
  - "闭环反馈"
  - "风险分配"
---

# Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.16376](https://arxiv.org/abs/2510.16376)  
**代码**: [github.com/DOCU-Lab/Feedback-based_Conformal_Prediction](https://github.com/DOCU-Lab/Feedback-based_Conformal_Prediction)  
**领域**: LLM评测  
**关键词**: 共形预测, 轨迹优化, 不确定性量化, 闭环反馈, 风险分配

## 一句话总结
提出 Feedback-Based Conformal Prediction (Fb-CP) 框架，将已执行轨迹的信息反馈给 CP 以动态调整预测区域大小，在缩减时域轨迹优化中同时保证覆盖率和显著提升轨迹性能。

## 研究背景与动机

**领域现状**: Conformal Prediction (CP) 是构造有限样本覆盖率保证的预测区域的有力工具，被广泛用于不确定环境下的轨迹优化（TO），为障碍物位置生成预测区域以实现概率安全避碰。

**现有痛点**: 现有方法采用**顺序式**流程——先用 CP 生成预测区域，决策单向依赖预测区域。决策端的信息从不反馈给 CP，导致预测区域过于保守。

**核心矛盾**: 在缩减时域 TO 中，已执行轨迹 $x_{0:t}^*$ 包含丰富的碰撞后验信息（实际碰撞概率远低于先验分配的 $\alpha_\tau$），但这些信息被浪费了，无法用于缩小后续时刻的预测区域。

**本文切入角度**: 建立决策→CP 的闭环信息通道，利用已执行轨迹的后验碰撞概率 $\beta_\tau$ 替换先验 $\alpha_\tau$，将释放的风险余量重新分配给未来时刻。

## 方法详解

### 问题建模
考虑离散非线性系统 $x_{t+1} = f(x_t, u_t)$，环境有 $M$ 个未知轨迹障碍物。定义联合概率安全约束：

$$\mathbb{P}\left\{\bigcap_{\tau=1}^{T}\{c(x_\tau, Y_\tau) \geq 0\}\right\} \geq 1 - \alpha$$

通过 Boole 不等式分解为个体约束 $\mathbb{P}\{c(x_\tau, Y_\tau) \geq 0\} \geq 1 - \alpha_\tau$ 加总风险约束 $\sum_\tau \alpha_\tau \leq \alpha$。

### CP 预测区域构造
将标定集 $D_{cal}$ 分为 $D_{cal}^1$（$K$ 条轨迹）和 $D_{cal}^2$（$L$ 条轨迹）。定义非一致性分数 $R_{\tau|t}^{(i)} = \|Y_\tau^{(i)} - \hat{Y}_{\tau|t}^{(i)}\|$，得到覆盖保证：

$$\mathbb{P}\{\|Y_\tau - \hat{Y}_{\tau|t}\| \leq C_{\tau|t}^{1-\alpha_\tau}\} \geq 1 - \alpha_\tau$$

其中 $C_{\tau|t}^{1-\alpha_\tau} = \text{Quantile}_{1-\alpha_\tau}(R_{\tau|t}^{(1)}, \ldots, R_{\tau|t}^{(K)}, \infty)$。

### 后验碰撞概率计算（核心创新）
在时刻 $t$，利用 $D_{cal}^2$ 和已确定的系统状态 $x_\tau^*$ 计算后验碰撞概率上界：

$$\beta_\tau = \frac{1 + \sum_{i=1}^{L} \mathbb{I}(S_\tau^{(K+i)} < 0)}{1 + L}$$

其中 $S_\tau^{(K+i)} = c(x_\tau^*, \hat{Y}_{\tau|\tau-1} + \omega_\tau^{(K+i)})$。使用独立的 $D_{cal}^2$ 保证覆盖率。

### 反馈式风险分配
将优化问题改写为：将过去时刻的 $\alpha_\tau$ 替换为 $\beta_\tau$，释放风险余量给未来时刻：

$$\sum_{\tau=t+1}^{T} \alpha_\tau \leq \alpha - \sum_{\tau=0}^{t} \beta_\tau$$

由于 $\beta_\tau$ 高概率低于 $\alpha_\tau$，未来可用风险增加 → 预测区域缩小 → 轨迹性能提升。

### 迭代风险分配算法 (IRA)
为避免将 $\alpha_{t+1:T}$ 作为决策变量导致计算复杂度过高，提出两阶段迭代：
1. **紧缩非活跃约束**: 对于约束未达紧的时刻，降低其分配风险 $\tilde{\alpha}_\tau^n = (1-\eta)\alpha_\tau^n + \eta \underline{\alpha}_\tau^n$
2. **松弛活跃约束**: 将释放的风险均匀分配给所有活跃约束时刻

基于单调性引理 $\partial J^*/\partial \alpha_\tau \leq 0$，证明 IRA 产生的代价序列 $\{J^*(\alpha_{t+1:T}^n)\}$ 单调递减并收敛。

### 理论保证
- **覆盖率保证** (Theorem 5.4): 整个轨迹满足 $\mathbb{P}\{\bigcap_{\tau=1}^{T}\{c(x_\tau^*, Y_\tau) \geq 0\}\} \geq 1 - \alpha$
- **收敛保证** (Theorem 5.3): 在有界 $\mathcal{X}$, $\mathcal{U}$ 和连续目标函数下，IRA 收敛到有限极限
- **性能提升保证**: Fb-CP 的调整始终维持覆盖率并提供可证明的性能改善

## 实验关键数据

### 3D 四旋翼模型（1000 次 Monte Carlo，$\alpha=0.05$）

| 方法 | 平均代价↓ | 平均计算时间(s) | 碰撞避免率↑ |
|------|----------|----------------|-----------|
| CC ($\eta$=1000) | 59.25 | 0.019 | 97.0% |
| ACI-MP | 17.970 | 0.022 | 98.6% |
| RF-CP | 15.794 | 0.487 | 98.7% |
| S-CP | 17.321 | 0.022 | 98.8% |
| Fb-CP-ARA | 15.356 | 0.027 | 98.2% |
| **Fb-CP-IRA** | **7.189** | 0.038 | 96.3% |

### 不同 $\alpha$ 下的表现（四旋翼模型）

| $\alpha$ | S-CP 代价 | Fb-CP-ARA 代价 | Fb-CP-IRA 代价 | IRA 降幅 |
|---------|----------|---------------|---------------|---------|
| 0.05 | 17.321 | 15.356 | 7.189 | -58.5% |
| 0.10 | 16.170 | 14.228 | 6.798 | -57.9% |
| 0.15 | 14.830 | 12.354 | 6.191 | -58.3% |
| 0.20 | 13.217 | 10.220 | 5.398 | -59.1% |

### 关键发现
- Fb-CP-ARA 相比 S-CP 平均降低代价 11.3%（几乎零额外计算开销）
- Fb-CP-IRA 相比 S-CP 平均降低代价 **58.5%**，但需要迭代求解 TO 增加计算时间
- RF-CP 的归一化非一致性分数引入混整数变量，计算时间比 Fb-CP-IRA 高一个数量级
- 所有方法碰撞避免率满足 $1-\alpha$ 理论保证

## 亮点与洞察
- **决策到 CP 的反馈闭环**是全新的研究范式：打破 "CP 只做预测" 的惯性思维，建立双向信息流
- **后验概率计算巧妙**：用独立的 $D_{cal}^2$ 评估 "按当前位置有多少标定轨迹会碰撞"，直观且计算高效
- **IRA 的主动/非活跃约束分析**具有优雅的理论结构：紧缩→松弛的迭代在保持可行性的同时单调降低代价
- 框架通用性强：适用于任何带联合概率约束的缩减时域优化问题

## 局限与展望
- 需要将标定集一分为二（$D_{cal}^1$ 和 $D_{cal}^2$），有效样本量减半
- IRA 迭代增加计算时间，在实时性要求极高的场景可能受限
- 分布偏移的扩展（Appendix G）依赖加权方案，对偏移程度的适应性有限
- 障碍物轨迹模型假设 LSTM 预测器，更强的预测器（如 Transformer）可能改变最优风险分配策略
- 实验未涉及高维状态空间或大规模障碍物场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次提出决策反馈 CP，开辟全新研究方向
- 理论深度: ⭐⭐⭐⭐⭐ 覆盖率+收敛+性能三重理论保证
- 实验充分度: ⭐⭐⭐⭐ 多模型多基线 + 1000 次 MC + 真实数据集
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从问题到方法到理论丝丝入扣
- 综合: ⭐⭐⭐⭐⭐ 理论优美 + 实际有效 + 范式创新

## 相关工作与启发
- **vs S-CP (Lindemann et al. 2023)**: 顺序式 CP 的标准方法，预测区域固定不随决策更新。Fb-CP 在此基础上引入反馈通道，平均降低 58.5% 代价
- **vs ACI-MP (Dixit et al. 2023)**: 用 ACI 处理分布偏移，但仍是单向流程，未利用已执行轨迹信息。更适合测试分布偏移场景而非本文设定
- **vs RF-CP (Stamouli et al. 2024)**: 提出归一化非一致性分数，代价与 Fb-CP-ARA 相当但引入混整数变量，计算时间高一个数量级
- **vs CC (Lekeufack et al. 2024)**: 通过代价权重调控碰撞率，未充分利用标定集，代价高 184%
- 反馈式 CP 的范式可迁移到所有带序列概率约束的在线优化（机器人编队、无人机路径规划等）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Conformal Prediction as Bayesian Quadrature](../../ICML2025/optimization/conformal_prediction_as_bayesian_quadrature.md)
- [\[ICML 2025\] On Temperature Scaling and Conformal Prediction of Deep Classifiers](../../ICML2025/optimization/on_temperature_scaling_and_conformal_prediction_of_deep_classifiers.md)
- [\[NeurIPS 2025\] Conformal Prediction for Causal Effects of Continuous Treatments](conformal_prediction_for_causal_effects_of_continuous_treatments.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[CVPR 2025\] Conformal Prediction for Zero-Shot Models](../../CVPR2025/optimization/conformal_prediction_for_zero-shot_models.md)

</div>

<!-- RELATED:END -->
