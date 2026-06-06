---
title: >-
  [论文解读] Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction
description: >-
  [ICML2026][优化/理论][conformal prediction] 本文通过 Taylor 展开揭示了标准 pinball 损失在条件覆盖率优化上的固有缺陷——忽视了异方差结构，提出密度加权 pinball 损失作为条件覆盖 MSE 的更紧代理目标，并设计三头分位数网络通过有限差分估计密度权重…
tags:
  - "ICML2026"
  - "优化/理论"
  - "conformal prediction"
  - "条件覆盖"
  - "分位数回归"
  - "密度加权"
  - "pinball损失"
---

# Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction

**会议**: ICML2026  
**arXiv**: [2512.24139](https://arxiv.org/abs/2512.24139)  
**代码**: https://github.com/Colorful-Pinball/CPCP  
**领域**: optimization  
**关键词**: conformal prediction, 条件覆盖, 分位数回归, 密度加权, pinball损失  

## 一句话总结

本文通过 Taylor 展开揭示了标准 pinball 损失在条件覆盖率优化上的固有缺陷——忽视了异方差结构，提出密度加权 pinball 损失作为条件覆盖 MSE 的更紧代理目标，并设计三头分位数网络通过有限差分估计密度权重，在 8 个高维回归基准上大幅提升条件覆盖性能。

## 研究背景与动机

**领域现状**：Conformal Prediction (CP) 是当前不确定性量化的主流范式，能以有限样本给出分布无关的边际覆盖保证 $\mathbb{P}(Y \in \mathcal{C}_\alpha(X)) \geq 1-\alpha$。然而标准 split CP 仅保证总体层面的边际覆盖，无法保证对特定输入 $x$ 的条件覆盖 $\mathbb{P}(Y \in \mathcal{C}_\alpha(X) \mid X=x)$——这在高风险场景中恰恰是实际需求。

**现有痛点**：针对条件覆盖，现有方法主要从两条路径入手：一是通过分组/局部化近似条件保证（如 group-conditional、localized CP），但受维度诅咒限制；二是改进非一致性得分函数（如 CQR、RCP），通过分位数回归校正得分的异方差性。然而，标准 pinball 损失的优化目标与条件覆盖 MSE 之间存在系统性偏差。

**核心矛盾**：已有工作（Kiyani et al., 2024; Plassier et al., 2025a）建立了 MSCE 与 pinball 损失超额风险之间的上界联系，但这个上界依赖条件 CDF 的全局 Lipschitz 常数 $L_F$，通常很松——它忽略了 $f_{S|X}(q_\tau(x))$ 在不同 $x$ 处的变化，即条件得分分布在目标分位数处的"陡峭度"差异。

**本文目标**：直接逼近条件覆盖的 MSE（即 MSCE），而非依赖松弛的上界或放松的条件覆盖定义。

**切入角度**：作者对 MSCE 做 Taylor 展开，发现其主导项是 **密度加权的 pinball 超额风险** $\mathbb{E}_X[f_{S|X}(q_\tau(X)) \cdot \mathcal{E}(X)]$，权重恰好是条件密度在真实分位数处的取值。在 location-scale 族下，该权重正比于 $1/\sigma(x)$——对低方差（陡峭 CDF）区域赋予更高权重，因为这些区域的条件覆盖最敏感——微小的分位数误差可导致覆盖率从 95% 骤降至 80%。

**核心 idea**：用密度加权 pinball 损失替代标准 pinball 损失来训练分位数回归，通过有限差分从辅助分位数估计密度权重，弥补标准方法对异方差结构的忽视。

## 方法详解

### 整体框架

CPCP (Colorful Pinball Conformal Prediction) 将校准集分为三个子集 $\mathcal{D}_{\text{cal},1}, \mathcal{D}_{\text{cal},2}, \mathcal{D}_{\text{cal},3}$，执行三阶段流程：(1) 在 $\mathcal{D}_{\text{cal},1}$ 上联合训练三个分位数估计器（目标分位数 $\tau$ 及辅助分位数 $\tau \pm \delta$）；(2) 用辅助分位数构造有限差分密度权重，在 $\mathcal{D}_{\text{cal},2}$ 上用加权 pinball 损失微调目标分位数；(3) 在 $\mathcal{D}_{\text{cal},3}$ 上执行 RCP 整流化得分校准，确保边际有效性。最终输出预测集 $\mathcal{C}_\alpha(x_{\text{test}}) = \{y: S(x_{\text{test}}, y) \leq \hat{q}_\tau(x_{\text{test}}) + \hat{\gamma}\}$。

### 关键设计

1. **密度加权 pinball 损失（理论核心）**:

    - 功能：构造 MSCE 的更紧代理优化目标
    - 核心思路：通过 Taylor 展开，条件覆盖偏差的平方 $(F_{S|X}(\hat{q}_\tau(x)) - \tau)^2$ 的主导项为 $f_{S|X}(q_\tau(x))^2 \cdot \epsilon_q(x)^2$，而标准 pinball 超额风险的主导项为 $\frac{1}{2} f_{S|X}(q_\tau(x)) \cdot \epsilon_q(x)^2$，两者差一个 $f_{S|X}(q_\tau(x))$ 因子。因此用密度 $f_{S|X}(q_\tau(x))$ 加权 pinball 损失后，优化目标与 MSCE 在主导项上精确对齐
    - 设计动机：标准 pinball 损失在条件 CDF 陡峭处（$f_{S|X}$ 大、$\sigma(x)$ 小）赋予的权重不足，而这些恰是条件覆盖最敏感的区域

2. **三头分位数网络 + 有限差分密度估计**:

    - 功能：在无需显式密度估计的前提下获取密度权重
    - 核心思路：利用分位数函数与 CDF 的逆函数关系 $\partial q_\tau(x)/\partial \tau = 1/f_{S|X}(q_\tau(x))$，通过有限差分近似密度：$\hat{w}(x) = 2\delta / (\hat{q}_{\tau+\delta}(x) - \hat{q}_{\tau-\delta}(x))$。网络采用共享骨干 $h(x)$ + 三个投影头架构，辅助分位数通过 $\hat{q}_{\tau \pm \delta}(x) = \hat{q}_\tau(x) \pm \text{Softplus}(\phi_{\text{high/low}} \circ h(x))$ 构造，Softplus 激活保证单调性、防止分位数交叉
    - 设计动机：避免估计完整条件密度 $f_{S|X}$（比回归更难），仅需两个辅助分位数即可获得所需权重

3. **权重裁剪 + 损失混合（有限样本稳定化）**:

    - 功能：控制逆权重的方差爆炸问题
    - 核心思路：权重裁剪将极端权重截断至经验均值的 $M$ 倍；损失混合将优化目标设为加权 pinball 损失与标准 pinball 损失的凸组合，等效于对权重施加人工下界
    - 设计动机：有限差分估计器的分母可能接近零导致权重爆炸，尤其在 $\delta$ 较小时。裁剪和混合以轻微偏差换取方差大幅下降，类似因果推断中逆倾向得分裁剪的经典策略

### 训练策略

采用两阶段训练：第一阶段在 $\mathcal{D}_{\text{cal},1}$ 上用标准 pinball 损失联合训练三个分位数头（确保有限差分精度）；第二阶段冻结骨干和辅助头，仅用加权 pinball 损失在 $\mathcal{D}_{\text{cal},2}$ 上微调主头。最终在 $\mathcal{D}_{\text{cal},3}$ 上计算整流化得分的经验分位数 $\hat{\gamma}$ 保证边际有效性。

## 实验关键数据

### 主实验：条件覆盖性能（MSCE ↓）

在 8 个高维回归基准上比较 MSCE（Mean Squared Coverage Error），目标覆盖率 $\tau = 90\%$，20 次重复：

| 方法 | Bike | Diamond | Naval | SGEMM | Transcoding | WEC |
|------|------|---------|-------|-------|-------------|-----|
| Split CP | 0.0031 | 0.0118 | 0.0351 | 0.0039 | 0.0125 | 0.0123 |
| CQR | 0.0011 | 0.0010 | 0.0120 | 0.0012 | 0.0016 | 0.0061 |
| RCP | 0.0010 | 0.0013 | 0.0029 | 0.0007 | 0.0009 | 0.0030 |
| CPCP | 0.0009 | 0.0009 | 0.0019 | 0.0003 | 0.0009 | 0.0025 |
| **CPCP (Clip+Mix)** | **0.0008** | **0.0004** | **0.0019** | **0.0003** | **0.0004** | **0.0012** |

### 消融 / Worst-Slice Coverage（WSC ↑）

| 方法 | Bike | Diamond | Naval | SGEMM | Transcoding | WEC |
|------|------|---------|-------|-------|-------------|-----|
| Split CP | 0.8133 | 0.6480 | 0.5428 | 0.7435 | 0.6797 | 0.7623 |
| CQR | 0.8641 | 0.8563 | 0.6997 | 0.8393 | 0.8175 | 0.8149 |
| RCP | 0.8849 | 0.8448 | 0.8002 | 0.8627 | 0.8515 | 0.8516 |
| **CPCP (Clip+Mix)** | **0.8882** | **0.8802** | **0.8320** | **0.8912** | **0.8759** | **0.8715** |

### 关键发现

- CPCP (Clip+Mix) 在 MSCE 上相比 RCP 平均降低约 40-60%，在 WSC 上将最差切片覆盖率从 ~80% 提升至 ~87-89%
- 消融实验中 RCP-MultiHead（仅多头联合训练但不使用密度权重）与 RCP 性能接近，证明改进不来自多任务学习或额外容量，而是密度加权目标本身
- 带宽 $\delta$ 在 0.01-0.05 范围内结果稳健，默认 $\delta=0.02$
- 权重裁剪 + 损失混合的稳定化策略在所有数据集上一致带来额外增益，尤其在高维多输出数据集（如 SGEMM、Transcoding）上效果显著

## 亮点与洞察

- **理论洞察极为优雅**：通过 Taylor 展开精确刻画了标准 pinball 损失与条件覆盖 MSE 之间的"差一个密度因子"的系统性偏差，将看似黑盒的条件覆盖优化问题转化为有明确数学动机的加权回归问题。在 location-scale 族下权重退化为 $1/\sigma(x)$ 这一结果直觉上也非常自然
- **有限差分密度估计的巧妙设计**：利用分位数函数与 CDF 逆函数的关系，避免了难度更高的条件密度估计任务，仅需两个辅助分位数即可；Softplus 参数化同时解决了分位数交叉和负权重两个实际问题
- **理论完备性强**：提供了完整的非渐近超额风险界，包含估计权重中逆权重的精确刻画，这一理论工具对其他涉及逆倾向加权的问题（如因果推断、off-policy评估）也有参考价值

## 局限与展望

- 校准集被三分使用，每个子集的有效样本量约为原始的 1/3，在小校准集场景下可能降低性能
- 密度权重估计依赖辅助分位数的精度，在极端分位数（如 $\tau$ 接近 0 或 1）或样本稀疏区域可能不稳定
- 理论速率 $O(n^{-1/3})$ 慢于标准分位数回归，虽然有限样本中密度加权的常数因子优势足以弥补，但在超大样本下标准方法可能追平
- 当前仅验证了回归任务，可探索在分类任务的条件覆盖保证和结构化输出（如图、序列）场景下的推广

## 相关工作与启发

- **RCP** (Plassier et al., 2025a)：CPCP 的直接基础，整流化得分框架；CPCP 可视为在 RCP 的分位数回归阶段引入更优的加权目标
- **CQR** (Romano et al., 2019)：经典的分位数回归 + 共形校准方法，但需要在训练集上替换目标函数
- **PLCP** (Kiyani et al., 2024)：建立 MSCE 与 pinball 超额风险的联系，但使用离散分组近似，收敛速率仅 $O(n^{-1/4})$

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling](../../CVPR2026/optimization/conditional_factuality_controlled_llms_with_generalization_certificates_via_conf.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](../../CVPR2026/optimization/semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ICML 2026\] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](tpv_parameter_perturbations_through_the_lens_of_test_prediction_variance.md)
- [\[ICLR 2026\] Non-Asymptotic Analysis of Efficiency in Conformalized Regression](../../ICLR2026/optimization/non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [\[ICML 2025\] Multivariate Conformal Selection](../../ICML2025/optimization/multivariate_conformal_selection.md)

</div>

<!-- RELATED:END -->
