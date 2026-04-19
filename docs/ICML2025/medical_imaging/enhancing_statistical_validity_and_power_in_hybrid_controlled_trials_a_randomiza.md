---
title: >-
  [论文解读] Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing
description: >-
  [ICML 2025][医学图像][混合对照试验] 提出基于 Fisher 随机化检验（FRT）+ 保形选择性借用（CSB）的混合对照试验推断框架，实现有限样本精确的 I 类错误率控制和模型无关的统计推断，通过自适应阈值最小化 MSE，在保持严格 I 类错误控制的同时提升检验功效。
tags:
  - ICML 2025
  - 医学图像
  - 混合对照试验
  - 随机化推断
  - 保形推断
  - 选择性借用
  - 有限样本推断
---

# Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing

**会议**: ICML 2025  
**arXiv**: [2410.11713](https://arxiv.org/abs/2410.11713)  
**代码**: 无  
**领域**: 因果推断 / 临床试验  
**关键词**: 混合对照试验、随机化推断、保形推断、选择性借用、有限样本推断

## 一句话总结

提出基于 Fisher 随机化检验（FRT）+ 保形选择性借用（CSB）的混合对照试验推断框架，实现有限样本精确的 I 类错误率控制和模型无关的统计推断，通过自适应阈值最小化 MSE，在保持严格 I 类错误控制的同时提升检验功效。

## 研究背景与动机

- **混合对照试验**：将外部对照（EC）整合到RCT中以增强统计功效，适用于罕见病等样本量受限场景
- **现有方法的局限**：
    - Bayesian方法（power prior、commensurate prior）：可能膨胀 I 类错误率
    - 频率学方法（倾向性评分加权、双重稳健估计）：依赖大样本渐近理论，小样本下不可靠
    - 当RCT样本量小（恰恰是最需要借用EC的场景）时，模型误设和渐近近似失效
- **未观测混杂**：EC非随机化产生，即使调整可观测混杂仍可能存在隐含偏差
- **核心矛盾**：借用EC可减小方差但引入偏差；不借用则功效不足

**关键洞察**：混合对照试验有两大优势可利用——(1) RCT内部的随机化保证 I 类错误控制，(2) 随机化对照组可用于评估EC的偏差。

## 方法详解

### 整体框架

**三层递进**：

1. **Fisher随机化检验（FRT）**：以RCT的随机化为唯一随机源，对任意检验统计量提供有限样本精确的 p 值
2. **保形选择性借用（CSB）**：用保形 p 值逐个检验每个EC是否可交换，选择性纳入
3. **自适应阈值**：最小化 CSB 估计量的 MSE 来选择最优筛选阈值 $\gamma$

### 关键设计

**1. FRT 的核心保证**

在 Fisher 零假设 $H_0: Y_i(0) = Y_i(1), \forall i \in \mathcal{R}$ 下：
- 所有潜在结局可完全填补：$Y_i^{\text{imp}}(0) = Y_i^{\text{imp}}(1) = Y_i$
- 通过重采样RCT的治疗分配 $\mathbf{A}^*$ 生成参考分布
- **关键**：EC的分配 $A_i \equiv 0$ 在重采样中保持不变（"按随机化方式分析"原则）

$$p^{\text{FRT}} = \mathbb{P}_{\mathbf{A}^*}\{T(\mathbf{A}^*) \ge T(\mathbf{A})\}$$

**定理2.3**：在 $H_0$ 下，$\mathbb{P}_{\mathbf{A}}(p^{\text{FRT}} \le \alpha) \le \alpha$，对**任意**检验统计量成立，无需模型正确设定或无未观测混杂假设。

**2. CSB 估计量**

筛选后的EC集合：$\hat{\mathcal{E}}(\gamma) = \{j \in \mathcal{E}: p_j^* > \gamma\}$

$$\hat{\tau}_\gamma = \frac{1}{n_\mathcal{R}} \sum_{i=1}^n \left[S_i \hat{\mu}_{1,\mathcal{R}}(X_i) + \frac{S_i A_i}{\hat{e}(X_i)}\{Y_i - \hat{\mu}_{1,\mathcal{R}}(X_i)\} - S_i \hat{\mu}_{0,\mathcal{R}+\hat{\mathcal{E}}(\gamma)}(X_i) - V_i\{Y_i - \hat{\mu}_{0,\mathcal{R}+\hat{\mathcal{E}}(\gamma)}(X_i)\}\right]$$

- $\gamma=1$：No Borrowing（NB），退化为纯RCT估计
- $\gamma=0$：Full Borrowing（FB），使用全部EC

**3. 保形 p 值**

- **Split conformal**：将随机化对照组分为校准集和训练集，用残差作为非一致性分数
    - $p_j^{\text{split}} = \frac{\sum_{i \in \mathcal{C}_1} \mathbb{I}(s_i \ge s_j) + 1}{|\mathcal{C}_1| + 1}$
    - **命题3.1**：若EC $j$ 与对照组可交换，则 $\mathbb{P}(p_j^{\text{split}} \le \gamma) \le \gamma$

- **CV+ p 值**：K折交叉验证版本，充分利用数据，代价是保证稍弱（$\le 2\gamma + O(1/K)$）

**4. 自适应阈值**

通过最小化 $\widehat{\text{MSE}}(\gamma) = (\hat{\tau}_\gamma - \hat{\tau}_1)^2 - \hat{\mathbb{V}}(\hat{\tau}_\gamma - \hat{\tau}_1) + \hat{\mathbb{V}}(\hat{\tau}_\gamma)$ 选择 $\hat{\gamma}$

- 利用 NB 估计量 $\hat{\tau}_1$（一致）近似真值 $\tau$
- Bootstrap估计方差项
- **定理3.4/3.5**：提供非渐近超额风险界

### 损失函数 / 训练策略

- 结局模型 $\mu_a(x)$：可用任意ML方法（线性、随机森林等）
- 倾向性评分 $e(x)$：RCT设计已知或logistic回归
- 保形分数函数：绝对残差 $|Y_i - \hat{f}(X_i)|$ 或保形分位数回归
- $B=5000$ 次Monte Carlo近似FRT的 p 值

## 实验关键数据

### 主实验

模拟设置：$(n_1, n_0, n_\mathcal{E}) = (50, 25, 50)$，小样本RCT + 50%有偏EC，偏差 $b=0,1,...,8$

| 偏差b | NB MSE | FB MSE | CSB MSE | NB Power | FB Power | CSB Power |
|-------|--------|--------|---------|----------|----------|-----------|
| 0 | 基准 | -42% | -20% | 基准 | +46% | +45% |
| 1-2 | 基准 | +454% | +1-18% | 基准 | -51% | -7~-20% |
| 5-8 | 基准 | +200% | -13~-16% | 基准 | -30% | +13~+36% |

**I 类错误率**：所有方法（NB/FB/CSB）在所有偏差水平下均严格控制在 $\alpha=0.05$ 以下

### 消融实验

- EC样本量增大到 $n_\mathcal{E}=300$：CSB的功效增益更显著
- CSB的选择性能：能有效剔除大部分有偏EC，同时也会部分剔除与随机化对照不够相似的无偏EC

### 关键发现

**真实数据**：CALGB 9633肺癌试验 + NCDB外部数据库

| 方法 | 估计值 | SE | 渐近p值 | 精确p值 | 借用EC数 |
|------|--------|-----|---------|---------|----------|
| NB(AIPW) | 0.142 | 0.074 | 0.055 | 0.051 | 0 |
| FB | 0.241 | 0.061 | <0.001 | 0.031 | 335 |
| **CSB** | **0.138** | **0.058** | **0.018** | **0.046** | **264** |

- CSB借用264/335个EC，SE从0.074降至0.058，精确 p 值达显著水平0.046
- FB可能高估治疗效应（0.241 vs CSB的0.138）

## 亮点与洞察

1. **有限样本精确性**：FRT保证在任何样本量、任何模型（甚至错误模型）、存在未观测混杂下都严格控制 I 类错误
2. **模型无关+分布无关**：保形 p 值不依赖渐近理论，可灵活使用黑盒ML模型
3. **理论优雅**：将混合对照试验的三个核心问题（I类错误控制、偏差检测、阈值选择）分别用三种推断工具（FRT、conformal、MSE优化）逐级解决
4. **Post-selection有效性**：允许 $\hat{\mathcal{E}}(\gamma)$ 随FRT重采样变化，正确纳入选择不确定性
5. **非渐近理论**：超额风险界和MSE估计误差界均为非渐近的

## 局限与展望

- **No-free-lunch**：当偏差不可忽略又难以检测时（$b=2,3,4$），CSB可能比NB功效更低——这是所有选择性借用方法的固有限制
- Fisher零假设是强零假设（所有个体处理效应均为零），弱零假设（平均处理效应为零）需进一步扩展
- 保形检验的功效受随机化对照组样本量限制——样本量越小，区分有偏/无偏EC越困难
- Bonferroni校正relatively保守，未利用EC间的相关结构
- 计算成本：每次FRT的Monte Carlo重采样中都需重新计算CSB估计量

## 相关工作与启发

- **Angelopoulos & Bates (2023)**：保形预测理论基础
- **Li et al. (2023b)**：混合对照试验的双重稳健估计量，本文在此基础上用FRT替代渐近推断
- **Gao et al. (2025)**：Adaptive Lasso选择性借用（ALSB），本文的CSB是其保形替代
- **Fisher (1935)**：随机化推断的奠基性工作
- **Bates et al. (2023)**：保形异常检测

**启发**：保形推断的"有限样本+分布无关"特性与随机化推断的"有限样本精确"特性天然互补，为小样本临床试验提供了严格且实用的统计工具。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将随机化推断+保形推断组合用于混合对照试验，理论框架完整
- 实验充分度: ⭐⭐⭐⭐ — 模拟+真实肺癌数据，但仅实验了连续结局
- 写作质量: ⭐⭐⭐⭐⭐ — 逻辑严密，理论与直觉并重，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ — 直接解决FDA重点关注的小样本混合对照试验推断问题

<!-- RELATED:START -->

## 相关论文

- [Doubly Protected Estimation for Survival Outcomes Utilizing External Controls for Randomized Clinical Trials](doubly_protected_estimation_for_survival_outcomes_utilizing_external_controls_fo.md)
- [Multivariate Conformal Selection](multivariate_conformal_selection.md)
- [Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach](../../AAAI2026/medical_imaging/radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)
- [Bayesian Inference for Correlated Human Experts and Classifiers](bayesian_inference_for_correlated_human_experts_and_classifiers.md)
- [From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)

<!-- RELATED:END -->
