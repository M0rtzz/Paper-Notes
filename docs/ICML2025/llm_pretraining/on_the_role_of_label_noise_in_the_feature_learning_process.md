---
title: >-
  [论文解读] On the Role of Label Noise in the Feature Learning Process
description: >-
  [ICML 2025][label noise] 从特征学习理论视角严格分析了两层ReLU CNN在标签噪声下的训练动态，揭示清晰的二阶段行为——Stage I模型学信号拟合干净样本（泛化好），Stage II损失收敛后模型记忆噪声过拟合噪声样本（泛化退化）——并为早停和小损失样本选择提供严格理论保证。
tags:
  - ICML 2025
  - label noise
  - feature learning
  - training dynamics
  - early stopping
  - sample selection
  - CNN
---

# On the Role of Label Noise in the Feature Learning Process

**会议**: ICML 2025  
**arXiv**: [2505.18909](https://arxiv.org/abs/2505.18909)  
**代码**: 无  
**领域**: 学习理论  
**关键词**: label noise, feature learning, training dynamics, early stopping, sample selection, CNN

## 一句话总结

从特征学习理论视角严格分析了两层ReLU CNN在标签噪声下的训练动态，揭示清晰的二阶段行为——Stage I模型学信号拟合干净样本（泛化好），Stage II损失收敛后模型记忆噪声过拟合噪声样本（泛化退化）——并为早停和小损失样本选择提供严格理论保证。

## 研究背景与动机

**领域现状**：过参数化深度网络面对标签噪声时容易过拟合导致泛化下降。大量实用方法（早停、样本选择、标签修正等）已被开发，但这些方法为何有效的理论理解仍不充分。

**现有痛点**：已有理论分析主要局限于lazy training regime（NTK框架），要求权重不偏离初始化太远或网络无限宽，本质上是静态核的线性动态，无法捕捉真实的特征学习行为。Frei et al. (2021)虽分析训练早期，但那阶段和线性分类器无异。

**核心矛盾**：经验观察清楚地表明网络"先学简单模式、后记忆噪声"，但在feature learning理论下缺少完整数学刻画。根本难点在于当 $n \cdot \text{SNR}^2 = \Theta(1)$ 时信号和噪声在同一量级，动态交织难以解耦。

**本文目标**：在feature learning理论框架下完整刻画标签噪声对训练动态的影响，揭示二阶段机制，为早停和样本选择提供正确性证明。

**切入角度**：信号-噪声数据分布（每个样本 = label-dependent信号patch + label-independent噪声patch），$n \cdot \text{SNR}^2 = \Theta(1)$ 的关键条件使信号和噪声处于竞争关系。

**核心 idea**：标签噪声在训练中创造了信号学习与噪声记忆的"竞赛"——前者先胜出但后者最终追上，分界点就是早停最佳位置。

## 方法详解

### 整体框架

分析三个支柱：（1）数据 $\mathbf{x} = [y\boldsymbol{\mu}, \boldsymbol{\xi}]$，信号 $\boldsymbol{\mu}$ 固定、噪声 $\boldsymbol{\xi} \sim \mathcal{N}(0, \sigma_\xi^2 \mathbf{I}_d)$，标签以 $\tau$ 概率翻转；（2）两层ReLU CNN $f = F_{+1} - F_{-1}$；（3）信号-噪声分解 $\mathbf{w}_{j,r}^{(t)} = \mathbf{w}_{j,r}^{(0)} + j\gamma_{j,r}^{(t)}\|\boldsymbol{\mu}\|^{-2}\boldsymbol{\mu} + \sum_i \rho_{j,r,i}^{(t)}\|\boldsymbol{\xi}_i\|^{-2}\boldsymbol{\xi}_i$，$\gamma$ 追踪信号学习、$\rho$ 追踪噪声记忆。

### 关键设计

1. **Stage I 分析（Theorem 4.1）**:

    - 功能：刻画训练前期——网络拟合干净样本、忽略噪声样本
    - 核心思路：在 $T_1 = \Theta(\eta^{-1}nm\sigma_\xi^{-2}d^{-1})$ 处，信号系数和噪声系数均达 $\Theta(1)$，但 $\gamma_{j,r}^{(T_1)} > \bar{\rho}_{\tilde{y}_i,r,i}^{(T_1)}$ 成立——信号严格大于噪声。因为此阶段所有样本的损失导数 $|\ell_i'|$ 有常数下界，梯度贡献平衡。干净样本中信号噪声协同（$\tilde{y}=y$），噪声样本中对抗（$\tilde{y} \neq y$）。结果：干净样本正确分类，噪声样本被分类到**真实标签**方向（即模型"拒绝"错误标签）
    - 设计动机：$n \cdot \text{SNR}^2 = \Theta(1)$ 确保信号只是"微弱地"超过噪声——这是二阶段行为的必要条件

2. **Stage II 分析（Theorem 4.2）**:

    - 功能：刻画后期——损失收敛，网络过拟合噪声样本
    - 核心思路：损失要收敛就需要正确分类**所有**样本（包括噪声标签方向）。对噪声样本，信号方向与噪声标签相反，只能靠增大噪声系数 $\bar{\rho}$ 实现。最终至少 $\tau'n$ 个噪声样本的噪声系数超过信号系数，测试误差下界 $\geq 0.5\min\{\tau_+, \tau_-\}$ 不可消除。反证法证明：若没有足够噪声样本被过拟合，训练损失就无法收敛
    - 设计动机：证明过拟合不是可选的——只要继续训练让损失收敛就**必然**记忆噪声

3. **早停+样本选择保证（Proposition 4.3）**:

    - 功能：为两种实用技术提供严格证明
    - 核心思路：**早停**：$T_1$ 处停止，测试误差 $\leq \exp(-dn^{-1}/C')$。**样本选择**：$T_1$ 处干净样本损失 $\leq \log 2$、噪声样本损失 $\geq \log 2$，$\log 2$ 阈值实现完美分离
    - 设计动机：虽然实际无法精确计算 $T_1$，但理论保证了最优停止点存在，验证集准确率可做实际代理

### 损失函数

Logistic loss $\ell(f, \tilde{y}) = \log(1 + \exp(-f \cdot \tilde{y}))$，全批GD，常数学习率。

## 实验关键数据

### 主定理概览

| 定理 | 阶段 | 结论 |
|------|------|------|
| Thm 4.1 | Stage I ($t=T_1$) | γ > ρ；干净样本全正确；噪声样本按真实标签分类 |
| Thm 4.2 | Stage II (损失收敛) | 干净仍正确；τ'n个噪声样本ρ>γ；测试误差≥0.5min{τ+,τ-} |
| Thm 4.4 | 无噪声对比 | 所有样本始终正确，测试误差指数小 |
| Prop 4.3 | 早停/选择 | 停在T₁测试误差≤exp(-d/nC')；log(2)阈值完美分离 |

### 噪声vs无噪声对比

| 设定 | Stage I测试误差 | 损失收敛后测试误差 |
|------|---------------|-----------------|
| 无标签噪声 | 低 | 仍低（$\leq \exp(-n\|\mu\|^4 / C_D\sigma_\xi^4 d)$） |
| 有标签噪声 (τ>0) | 低（接近0） | **不可避免地高** ($\geq 0.5\min\{\tau_+,\tau_-\}$) |

### 关键发现

- 两阶段转换的机制清晰：Stage I中 $|\ell_i'|$ 对所有样本均匀，梯度贡献平衡；Stage II中干净样本 $|\ell_i'| \to 0$（已被拟合），噪声样本 $|\ell_i'|$ 占主导驱动噪声记忆
- $\log 2$ 作为损失分离阈值由logistic函数在决策边界处的值决定，与数据分布无关——具有普适性
- 与Kou et al. (2023)的关键区别：后者 $n \cdot \text{SNR}^2 \gg 1$ 条件下信号始终主导不出现两阶段
- 信号系数 $\gamma$ 在Stage II可能暂时下降但始终正——模型不会"忘记"信号，只是噪声叠加

## 亮点与洞察

- 首个在feature learning框架下完整刻画标签噪声二阶段行为的理论，不是lazy regime的线性分析
- $\log 2$ 阈值的发现优雅且实用——为Co-teaching等小损失方法提供首个严格理论依据
- Stage II的反证法论证技巧精妙：假设没够多噪声被过拟合→损失无法收敛→矛盾
- 无噪声对比清晰展示了标签噪声的"成本"：有噪声时测试误差有正下界，无噪声时可指数小

## 局限性

- 理论限于两层CNN+二分类+signal-noise数据分布，距实际深层网络和自然数据有差距
- $n \cdot \text{SNR}^2 = \Theta(1)$ 要求信号强度和样本数精确匹配，实际中SNR变化大
- 未考虑instance-dependent标签噪声（更现实的噪声模型）
- 全批GD而非SGD，未考虑mini-batch随机性的影响
- 未给出 $T_1$ 的实用计算方法

## 相关工作与启发

- **vs Kou et al. (2023)**: 同为feature learning theory但要求 $n \cdot \text{SNR}^2 \gg 1$，不出现两阶段。本文 $\Theta(1)$ 条件是正确的regime
- **vs Frei et al. (2021)**: 只关注训练早期，看不到噪声引起的退化。本文覆盖完整过程
- **vs Li et al. (2020)**: lazy regime分析，无法描述特征学习。本文在rich regime
- **启发**：理论暗示"先自由训练再收紧"比"一开始就正则化"更合理

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次完整刻画标签噪声的二阶段特征学习行为
- 实验充分度: ⭐⭐⭐ 以理论为主，合成+小规模CIFAR验证
- 写作质量: ⭐⭐⭐⭐⭐ 定理→直觉→证明思路的叙述层次清晰
- 价值: ⭐⭐⭐⭐ 为早停和样本选择提供坚实理论基础

<!-- RELATED:START -->

## 相关论文

- [Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry](algebra_unveils_deep_learning_--_an_invitation_to_neuroalgebraic_geometry.md)
- [Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)
- [A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning](a_square_peg_in_a_square_hole_meta-expert_for_long-tailed_semi-supervised_learni.md)
- [The Dark Side of the Forces: Assessing Non-Conservative Force Models for Atomistic Machine Learning](the_dark_side_of_the_forces_assessing_non-conservative_force_models_for_atomisti.md)

<!-- RELATED:END -->
