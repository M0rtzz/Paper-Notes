---
title: >-
  [论文解读] Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning
description: >-
  [NeurIPS 2025][LLM评测][超参优化] CFBO 将用户定义的效用函数（成本 vs 性能的权衡）引入冻结-解冻贝叶斯优化框架，结合自适应停止准则和基于 LC mixup 的迁移学习，在多保真度 HPO 基准上实现了成本-性能最优权衡。
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "超参优化"
  - "贝叶斯优化"
  - "冻结-解冻"
  - "成本敏感"
  - "学习曲线外推"
  - "迁移学习"
---

# Cost-Sensitive Freeze-thaw Bayesian Optimization for Efficient Hyperparameter Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2510.21379](https://arxiv.org/abs/2510.21379)  
**代码**: [GitHub](https://github.com/db-Lee/CFBO)  
**领域**: LLM评测  
**关键词**: 超参优化, 贝叶斯优化, 冻结-解冻, 成本敏感, 学习曲线外推, 迁移学习

## 一句话总结
CFBO 将用户定义的效用函数（成本 vs 性能的权衡）引入冻结-解冻贝叶斯优化框架，结合自适应停止准则和基于 LC mixup 的迁移学习，在多保真度 HPO 基准上实现了成本-性能最优权衡。

## 研究背景与动机

**领域现状**：深度学习 HPO 计算代价高昂；多保真度方法（Hyperband, DyHPO, ifBO）通过学习曲线外推提前筛选配置，显著提升效率。

**现有痛点**：传统多保真度方法假设预算充足，目标是最终性能最大化，不考虑用户对"花费/性能"权衡的偏好——例如云计算用户希望在 credits 有限时早停，而非一定要跑满。

**核心矛盾**：如何让 HPO 过程能根据用户偏好在最优效用点附近自动停止？

**切入角度**：定义效用函数 $U(b, \tilde{y}_b)$ 描述成本-性能权衡，并设计与之匹配的 acquisition function 和停止准则。

**核心idea**：在 freeze-thaw BO 框架中最大化用户效用而非渐近性能，同时用 LC mixup 迁移学习提升早期外推准确性。

## 方法详解

### 效用函数

用户效用函数 $U(b, \tilde{y}_b): [B] \times [0,1] \to [0,1]$，随预算 $b$ 增加而递减，随最优累计性能 $\tilde{y}_b$ 增加而递增。典型形式：

$$U(b, \tilde{y}_b) = \tilde{y}_b - \alpha \left(\frac{b}{B}\right)^c$$

其中 $\alpha \in [0,1]$ 控制惩罚强度，$c \in \{0.5, 1, 2\}$ 对应平方根、线性、二次型。当 $\alpha=0$ 时退化为传统无惩罚设定。

对于不确定自身偏好的用户，可通过 Bradley-Terry 模型从成对偏好数据中估计效用。仅需约 30 对比较即可恢复。

### EI 型 Acquisition Function

选取配置 $x_{n^*}$ 以最大化效用期望改进：

$$A(n; U) = \max_{\Delta t \in [T-t_n]} \mathbb{E}_{y_{n,\cdot} \sim p_\theta} \left[\left[U(b+\Delta t, \tilde{y}_{b+\Delta t}) - U_p\right]^+\right]$$

- 通过 PFN 外推配置 $x_n$ 的剩余学习曲线
- 动态选择最佳目标 epoch $\Delta t$（而非固定到最后 epoch）
- 参考值 $U_p$ 为最近一步的效用（非历史最优），因为预算不可逆

关键属性：初期偏向长远高性能配置（探索），后期性能饱和、成本主导后转向短期贪心（利用）。

### 自适应停止准则

基于估计遗憾值：

$$\hat{R}_b = \frac{\hat{U}_{\max} - U_p}{\hat{U}_{\max} - \hat{U}_{\min}} > \delta_b$$

自适应阈值融合了改进概率（PI）：

$$\delta_b = \text{BetaCDF}(p_b; \beta, \beta)^\gamma$$

其中 $p_b$ 是选中配置在未来改进 $U_p$ 的概率。$\gamma$ 控制 PI=0.5 时的阈值，$\beta$ 控制 PI 融合程度（$\beta \to 0$：忽略 PI；$\beta \to \infty$：纯 PI 决策）。默认 $\beta=e^{-1}$, $\gamma=\log_{0.5}0.2$。

### LC Mixup 迁移学习

为提升早期 LC 外推精度，对 PFN 代理模型做两级 mixup 数据增强：
1. **数据集间**：$L' = \lambda_1 L^{(m)} + (1-\lambda_1)L^{(m')}$
2. **配置间**：$l' = \lambda_2 l_n + (1-\lambda_2) l_{n'}$，$x' = \lambda_2 x_n + (1-\lambda_2) x_{n'}$

$\lambda_1, \lambda_2 \sim \text{Uniform}(0,1)$。可无限采样训练数据，减少过拟合。

## 实验关键数据

### 数据集

| 数据集 | $d_x$ | $|\mathcal{X}|$ | $T$ | 训练任务 | 测试任务 |
|--------|-------|---------|-----|---------|---------|
| LCBench | 7 | 2000 | 51 | 20 | 15 |
| TaskSet | 8 | 1000 | 50 | 21 | 9 |
| PD1 | 4 | 240 | 50 | 16 | 7 |

### 主要结果（标准化遗憾 ×100，PD1 基准）

| 方法 | α=0 | α=2⁻⁶ | α=2⁻⁴ | α=2⁻² |
|------|-----|--------|--------|--------|
| ifBO | 0.8±0.1 | 2.3±0.1 | 6.0±0.6 | 15.2±2.0 |
| Quick-Tune† | — | — | — | — |
| CFBO-NT | 0.2±0.0 | 1.5±0.0 | 4.5±0.0 | 8.5±0.0 |
| **CFBO** | **0.2±0.0** | **1.0±0.0** | **0.9±0.0** | **1.7±0.0** |

### 消融实验（PD1）

| 自适应阈值 | Acq. | 迁移 | α=0 | α=2⁻⁴ | α=2⁻² |
|-----------|------|------|-----|--------|--------|
| ✗ | ✗ | ✗ | 0.8 | 6.0 | 15.2 |
| ✗ | ✗ | ✓ | 0.2 | 5.9 | 11.7 |
| ✗ | ✓ | ✓ | 0.2 | 4.5 | 8.5 |
| ✓ | ✓ | ✓ | **0.2** | **0.9** | **1.7** |

- 迁移学习在 α=0 时收益最大；成本敏感 acquisition function 在 α>0 时收益突出
- 自适应阈值在强惩罚下带来最大提升（从 8.5→1.7）

### 运行时间（秒/BO 步，A100）

| 方法 | LCBench | TaskSet | PD1 |
|------|---------|---------|-----|
| ifBO | 0.58 | 0.30 | 0.08 |
| CFBO | 1.52 | 0.78 | 0.23 |

差距可忽略——HPO 中网络训练时间远大于 BO 步骤时间。

## 亮点与洞察
- **问题建模优雅**：将"用户何时该停"形式化为效用最大化，比固定 budget 实用得多
- **acquisition 行为自适应**：自然从探索过渡到利用，$\Delta t$ 随 BO 进展从大到小
- **自适应停止准则**效果显著优于固定阈值，且接近理论最优停止点
- LC mixup 数据增强简单有效，可推广到其他 PFN 训练场景

## 局限与展望
- 效用函数形式需用户先验（虽然提供了 Bradley-Terry 估计，但仍需指定函数族）
- 配置空间固定为有限集 $\mathcal{X}$，不支持连续 HPO
- 仅在表格/NLP/视觉分类 LC 上验证，未覆盖大规模 LLM 训练场景
- 运行时间约为 ifBO 的 2-3 倍（虽整体可忽略）

## 评分
- 新颖性: ⭐⭐⭐⭐ 成本敏感效用 + 自适应停止的组合是新颖的 HPO 形式化
- 实验充分度: ⭐⭐⭐⭐⭐ 3个基准 × 多种效用函数 × 完整消融 × 真实数据集验证
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对云端/限时 HPO 场景有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing](../../ICML2025/llm_evaluation/provably_cost-sensitive_adversarial_defense_via_randomized_smoothing.md)
- [\[ICML 2025\] Hyperband-based Bayesian Optimization for Black-box Prompt Selection](../../ICML2025/llm_evaluation/hyperband-based_bayesian_optimization_for_black-box_prompt_selection.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization](hybridnorm_towards_stable_and_efficient_transformer_training_via_hybrid_normaliz.md)

</div>

<!-- RELATED:END -->
