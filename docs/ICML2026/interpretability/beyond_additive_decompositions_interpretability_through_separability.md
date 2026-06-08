---
title: >-
  [论文解读] Beyond Additive Decompositions: Interpretability Through Separability
description: >-
  [ICML2026][可解释性][可解释机器学习] 提出张量分离学习（TSL），一种将条件均值建模为正秩-1可分离乘积之差的逐阶段贪心回归方法，通过可分离结构避免加性分解在强交互下的信号抵消与交互遮蔽问题，同时其偏依赖函数可精确恢复拟合因子形状。
tags:
  - "ICML2026"
  - "可解释性"
  - "可解释机器学习"
  - "张量分离学习"
  - "可分离模型"
  - "偏依赖分析"
  - "玻璃盒模型"
---

# Beyond Additive Decompositions: Interpretability Through Separability

**会议**: ICML2026  
**arXiv**: [2605.31200](https://arxiv.org/abs/2605.31200)  
**代码**: https://github.com/jyliuu/TSL  
**领域**: 可解释性  
**关键词**: 可解释机器学习、张量分离学习、可分离模型、偏依赖分析、玻璃盒模型  

## 一句话总结

提出张量分离学习（TSL），一种将条件均值建模为正秩-1可分离乘积之差的逐阶段贪心回归方法，通过可分离结构避免加性分解在强交互下的信号抵消与交互遮蔽问题，同时其偏依赖函数可精确恢复拟合因子形状。

## 研究背景与动机

**领域现状**：可解释机器学习主流依赖加性分解（additive decomposition）来提供可读模型结构。后验解释方法如 SHAP 将预测分解为特征贡献之和，玻璃盒模型如 GAM 将预测限制为单变量函数之和，functional ANOVA 提供全局加性视角。

**现有痛点**：加性分解在强交互存在时面临三个根本性失败模式：(1) SHAP 的局部归因中主效应与交互效应可能相互抵消，导致 $\phi_k(\mathbf{x})=0$ 但特征 $k$ 实际高度活跃；(2) 偏依赖图（PD）只恢复主效应 $\text{PD}_1(x_1) = m_\emptyset + m_1(x_1)$，对高阶交互完全盲目；(3) 特征相关时，边缘化平均在低密度区域产生外推伪影。

**核心矛盾**：加性分解需要将高维交互"投影"到低维加性成分上，这一过程不可避免地丢失交互信息。要保持可解释性就必须放弃交互建模能力，这是准确性与可解释性之间的结构性矛盾。

**本文目标**：设计一种既能捕获任意阶交互、又能从一维偏依赖曲线精确重建模型结构的玻璃盒回归方法。

**切入角度**：对于可分离乘积 $h(\mathbf{x}) = \prod_j h_j(x_j)$，其偏依赖函数满足 $\text{PD}_j[h](x_j) = c_j h_j(x_j)$，即一维 PD 精确恢复因子形状而非退化为主效应。这一性质使可分离结构成为"交互友好"的天然选择。

**核心 idea**：用正秩-1可分离乘积之差的逐阶段叠加来替代加性分解，使模型原生地保留交互结构并可从一维 PD 精确重建。

## 方法详解

### 整体框架

TSL 要解决的是"既想建模任意阶交互、又想让一维偏依赖曲线忠实可读"这对矛盾。它把条件均值 $m(\mathbf{x}) = \mathbb{E}[Y|X=\mathbf{x}]$ 写成 $R$ 个阶段的叠加，每个阶段不是一组单变量函数之和（加性分解的做法），而是两个"正可分离乘积"之差；训练上则用逐阶段贪心：每阶段去拟合上一阶段留下的残差，拟合完再对所有已有阶段的系数做一次正交重拟合（orthogonal refitting），让阶段之间不互相干扰。

### 关键设计

**1. 正可分离乘积差分结构：用乘法分离替代加法分离**

加性分解之所以对强交互失明，根子在于它把高维交互硬投影到低维加性成分上、信息必然丢失。TSL 改用乘积结构 $\hat{m}(\mathbf{x}) = \sum_{\ell=1}^{R}\big(\lambda_+^{(\ell)}\prod_j \hat{m}_{+,j}^{(\ell)}(x_j) - \lambda_-^{(\ell)}\prod_j \hat{m}_{-,j}^{(\ell)}(x_j)\big)$，其中每个单变量分量 $\hat{m}_{\pm,j}^{(\ell)} > 0$ 被约束为严格正。乘积天生能表达交互，而正性约束则消掉了符号歧义——无约束乘积里某个分量为正并不代表整个乘积贡献为正（其他分量可能翻号），正性保证"分量增大就一定放大乘积"。但纯正乘积无法表达负效应，于是再用"两个正乘积相减"的差分结构把表达负值的能力补回来。这个正性约束还顺手解决了后面 bagging 的可辨识性：无约束乘积在非矩形支撑上有多种符号翻转的等价表示，正性把它们钉死成唯一形式，独立拟合出的分量才能安全平均。

**2. Backbone/Tilt 重参数化：把"活跃程度"与"方向"解耦**

直接看正负分量对不够直观，TSL 把每一对重参数化为 $\hat{m}_{\pm,j}^{(\ell)}(x_j) = b_j^{(\ell)}(x_j)\, e^{\pm d_j^{(\ell)}(x_j)}$：backbone $b_j^{(\ell)} > 0$ 编码正负两支共享的幅度，tilt $d_j^{(\ell)} \in \mathbb{R}$ 编码正负之间的不平衡。代入后整个模型化为 $\hat{m}(\mathbf{x}) = 2\sum_\ell b^{(\ell)}(\mathbf{x})\, \sinh\!\big(d^{(\ell)}(\mathbf{x})\big)$，于是 backbone 乘积变成"活跃门控"（某特征的 backbone 趋零就整体关掉这一阶段），tilt 之和经 $\sinh$ 决定输出往哪个方向走。这种拆分专治加性分解的"交互遮蔽"——即便符号偏依赖恰好抵消为零，backbone 依然保留着该特征的幅度信息不会被掩盖；同时 tilt 作为加性不平衡分数本身就有清晰的方向性解读。

**3. 网格张量逐步细化与 Bagging 聚合：拟合分量并压方差**

每个单变量分量用分段常数函数表示，采用 CART 风格的贪心分裂逐步细化分区。每次候选分裂用正则化最小二乘打分 $\mathcal{L}_S(u_+^S, u_-^S) = \sum_i w_i\big(R_i - (u_+^S \hat{m}_+^{(i)} - u_-^S \hat{m}_-^{(i)})\big)^2 + \alpha\big((u_+^S-1)^2 + (u_-^S-1)^2\big)$，这个目标对 $u_+^S, u_-^S$ 有闭式的 $2\times2$ 线性系统解，所以细化很快。为压低方差，再在 bootstrap 样本上并行拟合多个网格张量，然后在 backbone/tilt 空间里按"归一化 → 锚定 → 相似度过滤 → 平均"做聚合：相似度过滤负责剔除因可分离非可辨识性而不一致的 bag，剩下一致的再平均，bagging 由此既降方差又不被歧义解污染。

## 实验关键数据

### 主实验

在 OpenML CTR 23 回归基准的 27 个数据集上评估，与 EBM、SepALS、XGBoost、LightGBM、Random Forest 对比：

| 数据集 | TSL (R≤10) | EBM | SepALS (r≤10) | XGBoost (黑盒) | 最佳组 |
|--------|-----------|-----|---------------|---------------|-------|
| brazilian_houses | **2398.68** | 3327.29 | 3996.05 | 4289.10 | TSL |
| auction_verification | **624.36** | 1738.17 | 682.20 | 369.34 | TSL (可解释最佳) |
| socmob | **9.48** | 20.21 | 22.88 | 17.65 | TSL |
| california_housing | 49376.09 | **48866.28** | 62162.22 | 44971.31 | EBM (可解释最佳) |
| cpu_activity | **2.3076** | 2.3546 | 2.8475 | 2.1945 | TSL |
| miami_housing | **89692.96** | 91777.25 | 99426.86 | 82325.09 | TSL |

在 27 个数据集中，TSL (R≤2 或 R≤10) 在 17 个上进入可解释组前三，5 个上为最佳可解释模型。

### 消融实验

| 配置 | socmob | naval_prop. | auction_ver. | 说明 |
|------|--------|-------------|-------------|------|
| TSL (R≤2, 差分正乘积) | 9.87 | 0.0013 | 1135.80 | 完整模型 |
| TSL (1-product, 无正性) | 10.58 | 0.0027 | 1336.51 | 去掉正性+差分后退化明显 |
| SepALS (r≤2) | 7.73 | 0.0004 | 997.08 | 光滑数据上 SepALS 更优 |

在匹配总分离秩(≤4)条件下，正性约束+差分结构带来显著提升（socmob: 10.58→9.87, naval: 0.0027→0.0013）。

### 关键发现

- TSL 在信号具有低秩可分离结构的数据上优势最大（如 socmob 具有已知对数加性结构，TSL RMSE 9.48 vs EBM 20.21）
- SepALS 在光滑数据上可能更优（如 naval_propulsion_plant），但在尖锐特征上过度平滑；TSL 的自适应网格分裂能捕获更突变的模式
- 在 California Housing 的可解释性案例研究中，TSL 的两阶段模型清晰展示了"海岸溢价"（Stage 1）和"内陆沙漠修正"（Stage 2）的可分离空间门控机制
- 合成实验验证了交互遮蔽问题：当 $\mathbb{E}[1+X_3]=0$ 时，所有方法的一维 PD 均为零，但 TSL 的 backbone 仍保留了 $x_1$ 的二次效应幅度

## 亮点与洞察

- **可分离结构的偏依赖精确性**：对可分离乘积，偏依赖函数精确恢复因子形状（而非退化为主效应），这是理论上的核心洞察——用乘法结构替代加法结构可以同时拥有交互建模能力和一维可视化的忠实性
- **Backbone/Tilt 分离**：将"模型在哪活跃"（backbone 门控）和"模型往哪个方向走"（tilt 方向）解耦，提供了一种全新的可解释性视角。这一思路可迁移到任何需要分离幅度与方向信息的场景
- **正性约束的多重作用**：一个简单的正性约束同时解决了三个问题（符号歧义、bagging 稳定性、可解释方向性），是优雅的"一石三鸟"设计

## 局限与展望

- 理论保证仅覆盖近似率 $O(1/\sqrt{r})$，未给出有限样本学习率或一致性保证
- 可分离表示的非可辨识性是核心局限：在非矩形支撑上，不同的因子分解可产生相同预测，bagging 聚合可能引入高方差
- 在以加性结构为主的问题上，EBM 仍是更强的可解释基线（如 QSAR_fish_toxicity、red_wine）
- 当前 backbone/tilt 中单变量分量用分段常数函数建模，未来可替换为神经网络或样条等更灵活的参数化
- 聚合策略目前丢弃了未通过相似度过滤的 bagged grids，可通过正秩-1流形上的黎曼优化来对齐所有 bags

## 相关工作与启发

- 与 GAM/EBM 对比：GAM 及其扩展（GA²M、NAM、NODE-GAM）本质上是加性分解，TSL 用乘法分离替代加法分离，是正交的建模范式
- 与经典张量分解（CP/PARAFAC）对比：共享可分离形式但目标不同——经典方法恢复潜在因子，TSL 学习可解释的监督预测器
- 与 SepALS 对比：TSL 的逐阶段残差拟合替代了固定秩联合优化，正性差分结构替代了无约束乘积
- OGA 框架提供的 $O(1/\sqrt{r})$ 近似率与维度无关（但目标类随维度收紧），为可分离模型的理论分析提供了有力工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](../../NeurIPS2025/interpretability/beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](../../ACL2026/interpretability/mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](interpretability_can_be_actionable.md)
- [\[NeurIPS 2025\] Additive Models Explained: A Computational Complexity Approach](../../NeurIPS2025/interpretability/additive_models_explained_a_computational_complexity_approach.md)

</div>

<!-- RELATED:END -->
