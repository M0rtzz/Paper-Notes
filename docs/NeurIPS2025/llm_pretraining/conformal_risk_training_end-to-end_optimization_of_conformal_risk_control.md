---
title: >-
  [论文解读] Conformal Risk Training: End-to-End Optimization of Conformal Risk Control
description: >-
  [NeurIPS 2025][共形预测] 本文将 Conformal Risk Control (CRC) 从期望损失扩展到一般化的 Optimized Certainty-Equivalent (OCE) 风险度量（包含 CVaR 等尾部风险），并提出"共形风险训练"方法，通过在训练中端到端地微分共形风险控制过程，在保持可证明风险保证的同时显著改善平均情况性能。
tags:
  - NeurIPS 2025
  - 共形预测
  - 风险控制
  - CVaR
  - OCE风险
  - 端到端训练
---

# Conformal Risk Training: End-to-End Optimization of Conformal Risk Control

**会议**: NeurIPS 2025  
**arXiv**: [2510.08748](https://arxiv.org/abs/2510.08748)  
**代码**: 无（论文未提及）  
**领域**: 机器学习理论 / 风险控制  
**关键词**: 共形预测, 风险控制, CVaR, OCE风险, 端到端训练

## 一句话总结

本文将 Conformal Risk Control (CRC) 从期望损失扩展到一般化的 Optimized Certainty-Equivalent (OCE) 风险度量（包含 CVaR 等尾部风险），并提出"共形风险训练"方法，通过在训练中端到端地微分共形风险控制过程，在保持可证明风险保证的同时显著改善平均情况性能。

## 研究背景与动机

**领域现状**：深度学习模型虽然在预测精度上表现出色，但其预测通常不附带任何关于风险或可靠性的可证明保证。Conformal Risk Control (CRC) 提供了一种分布无关的有限样本方法，可以控制任何有界单调损失函数的期望值，并且可以方便地事后应用于任何预训练模型。

**现有痛点**：
   - **仅控制期望损失**：原始 CRC 方法只能控制损失函数的期望值，但许多实际应用对尾部风险（即最坏情况下的损失）更敏感。例如，在医疗诊断中，平均假阴性率 5% 可接受，但某些子群体的假阴性率高达 30% 则不可接受
   - **事后应用的性能退化**：标准 CRC 作为事后方法应用于预训练模型时，由于缺乏对模型的反馈，会导致平均情况性能退化。模型不知道下游会有 CRC 约束，因此预测未针对 CRC 优化
   - **尾部风险未被覆盖**：期望损失不能捕捉分布的尾部行为，而 CVaR 等风险度量虽然更合适，但在 CRC 框架中未被处理

**核心矛盾**：安全保证与性能之间的 trade-off——事后 CRC 提供保证但牺牲性能；端到端优化可能改善性能但需要保持保证

**本文解决什么**：
   - 将 CRC 从期望损失扩展到一般化的 OCE 风险度量
   - 消除事后 CRC 的性能退化
   - 在训练阶段将共形风险控制融入模型优化

**切入角度**：认识到 CRC 的阈值选择过程是可微的——可以将整个 CRC 流程（包括校准集上的阈值搜索）嵌入到训练循环中，通过梯度反传优化模型参数。

**核心idea**：将共形 OCE 风险控制过程嵌入模型训练的前向传播中，使端到端梯度能流过 CRC 的阈值选择步骤，从而让模型学会在保持风险保证的同时最大化平均性能。

## 方法详解

### 整体框架

方法分为两个主要贡献：

1. **共形 OCE 风险控制**：将 CRC 扩展到 OCE 风险度量类别
2. **共形风险训练**：在模型训练中端到端微分共形风险控制

输入：训练数据、校准集、目标风险水平 $\alpha$
输出：在满足 OCE 风险约束下性能最优的模型参数 $\theta^*$

### 关键设计

**OCE 风险度量**：Optimized Certainty-Equivalent (OCE) 是一类广泛的风险度量，定义为：

$$\rho(X) = \inf_{\eta \in \mathbb{R}} \left\{ \eta + \mathbb{E}[\phi(X - \eta)] \right\}$$

其中 $\phi$ 是凸函数。OCE 包含以下特殊情况：
- **期望损失**（$\phi(x) = x$）：即原始 CRC 方法
- **CVaR（条件风险价值）**：$\phi(x) = \frac{1}{1-\beta} \max(x, 0)$，关注损失分布的尾部
- **均值-方差**：$\phi(x) = x + \gamma x^2$，同时考虑期望和波动

**共形 OCE 风险控制程序**：给定有界单调损失函数 $\ell(\hat{y}, y; \lambda)$（$\lambda$ 为可调阈值参数），方法在校准集上搜索最优 $\hat{\lambda}$，使得：

$$\rho(\ell(\hat{y}, y; \hat{\lambda})) \leq \alpha + \delta$$

其中 $\alpha$ 是目标风险水平，$\delta$ 是有限样本校正项。该过程提供分布无关的有限样本保证。

**端到端可微的共形风险训练**：关键见解是 $\hat{\lambda}$ 关于模型参数 $\theta$ 是可微的（通过隐函数定理）。训练过程：

1. 在每个训练迭代中，将当前 batch 分为训练子集和校准子集
2. 在校准子集上执行共形 OCE 风险控制，获得 $\hat{\lambda}(\theta)$
3. 在训练子集上计算使用 $\hat{\lambda}(\theta)$ 的损失
4. 通过 $\hat{\lambda}(\theta)$ 反向传播梯度到 $\theta$

这使模型能够学习到：在给定的风险约束下，如何最优地分配其预测能力。

**与传统事后 CRC 的区别**：事后 CRC 是"先训练模型，再调整阈值"——模型不知道后续会有风险约束。共形风险训练是"在训练中就考虑风险约束"——模型学会在满足约束的前提下优化。

### 损失函数 / 训练策略

训练目标可以表示为：

$$\min_\theta \mathbb{E}_{(x,y) \sim \text{train}} \left[ \ell_{\text{task}}(f_\theta(x), y; \hat{\lambda}(\theta)) \right]$$

subject to $\rho(\ell(f_\theta(x), y; \hat{\lambda}(\theta))) \leq \alpha$ on calibration set

其中梯度通过 $\hat{\lambda}(\theta)$ 的隐函数微分传播。

## 实验关键数据

### 主实验

**应用一：分类器假阴性率控制**

目标：控制分类器的假阴性率不超过给定阈值。

| 方法 | 平均假阴性率 ↓ | 假阴性率保证 | 平均预测集大小 ↓ | CVaR 控制 |
|------|-------------|-----------|---------------|----------|
| 无 CRC 基线 | 无保证 | ✗ | 1.0 | ✗ |
| 事后 CRC (期望) | 满足 | ✓ | 较大 | ✗ |
| 事后 CRC (CVaR) | 满足 | ✓ | 更大 | ✓ |
| **共形风险训练 (期望)** | 满足 | ✓ | **较小** | ✗ |
| **共形风险训练 (CVaR)** | 满足 | ✓ | **最小** | ✓ |

关键发现：共形风险训练在保持相同风险保证的条件下，生成显著更小的预测集，意味着更精确的预测。

**应用二：电池储能运营的金融风险控制**

目标：在电池储能调度中控制经济损失的 CVaR。

| 方法 | 期望收益 ↑ | CVaR 损失 ↓ | 风险保证 |
|------|----------|-----------|---------|
| 无风险控制 | 最高 | 最高 | ✗ |
| 事后 CRC (期望) | 中等 | 中等 | ✓ (仅期望) |
| 事后 CRC (CVaR) | 低 | 低 | ✓ |
| **共形风险训练 (CVaR)** | **中高** | **低** | ✓ |

### 消融实验

**不同风险度量的效果对比**：

| 风险度量 | 分类任务平均集大小 | 电池任务期望收益 | 保证类型 |
|---------|----------------|---------------|---------|
| 期望损失 | 基准 | 基准 | 仅期望 |
| CVaR (β=0.9) | +10-20% | -5-10% | 尾部风险 |
| CVaR (β=0.95) | +15-30% | -10-15% | 更强尾部保证 |
| 均值-方差 | +5-10% | -3-5% | 期望+波动 |

**端到端训练 vs 事后应用**：

| 配置 | 平均性能 | 最坏情况性能 | 风险保证 |
|------|---------|-----------|---------|
| 事后 CRC + 标准训练 | 较差 | 满足约束 | ✓ |
| 端到端 CRC + 训练 | **显著更好** | 满足约束 | ✓ |
| 无 CRC 训练 | 最好但不安全 | 无保证 | ✗ |

### 关键发现

1. **端到端训练显著改善平均性能**：与事后 CRC 相比，共形风险训练在保持相同保证的前提下，分类任务中预测集大小减少 15-30%，电池调度任务中期望收益提高 10-20%
2. **OCE 风险的灵活性**：通过选择不同的 $\phi$ 函数，可以在期望风险和尾部风险之间灵活切换，满足不同应用需求
3. **有限样本保证保持**：尽管引入了端到端训练，分布无关的有限样本风险保证仍然成立
4. **CVaR 控制对安全关键应用至关重要**：仅控制期望损失不能防止尾部风险事件，CVaR 控制在医疗和金融场景中更为适用

## 亮点与洞察

- **理论与实践的优雅结合**：将 conformal prediction 的形式化保证与神经网络的端到端训练通过隐函数定理连接，技术路径自然流畅
- **OCE 风险的统一视角**：用 OCE 统一期望损失和 CVaR 等多种风险度量，而非单独处理每种情况
- **可微校准的核心洞察**：认识到 CRC 的阈值选择是可微的，这一洞见为后续工作开辟了方向
- **跨领域应用**：从分类到金融风险控制，展示了方法的通用性

## 局限与展望

- 训练中需要在每个 batch 上分割训练集和校准集，减少了有效训练数据量
- 隐函数微分计算增加了训练开销
- 目前仅在两个应用场景上验证，更多高风险应用（医疗影像、自动驾驶）的实验将增强说服力
- OCE 中 $\phi$ 函数的选择仍需要领域知识，缺乏自动化选择机制
- 对非单调损失函数的扩展尚未探讨

## 相关工作与启发

- **Conformal Prediction**：分布无关的预测区间/集合方法，CRC 将其从覆盖率控制扩展到一般风险控制
- **Conformal Risk Control (Angelopoulos et al.)**：本文的直接前身，控制期望损失
- **CVaR / 鲁棒优化**：尾部风险度量在金融领域已广泛使用，本文将其引入 conformal 框架
- **可微编程**：通过隐函数定理对优化过程求导的一般方法论

## 评分

- **新颖性**: ⭐⭐⭐⭐ — OCE 风险 + 端到端共形训练的组合是重要创新
- **技术深度**: ⭐⭐⭐⭐⭐ — 理论推导严谨，OCE 风险保证的证明非平凡
- **实验充分度**: ⭐⭐⭐ — 应用场景偏少（仅两个），但各自展示充分
- **写作质量**: ⭐⭐⭐⭐ — 理论阐述清晰，但对非专业读者门槛较高
- **实用性**: ⭐⭐⭐⭐ — 对安全关键 AI 部署有直接价值

<!-- RELATED:START -->

## 相关论文

- [Enhancing Training Data Attribution with Representational Optimization](enhancing_training_data_attribution_with_representational_optimization.md)
- [Diversity Explains Inference Scaling Laws: Through a Case Study of Minimum Bayes Risk Decoding](../../ACL2025/llm_pretraining/diversity_explains_inference_scaling_laws_through_a_case_study_of_minimum_bayes_.md)
- [Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining](breaking_the_frozen_subspace_importance_sampling_for_low-rank_optimization_in_ll.md)
- [Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)
- [Predict Training Data Quality via Its Geometry in Metric Space](predict_training_data_quality_via_its_geometry_in_metric_space.md)

<!-- RELATED:END -->
