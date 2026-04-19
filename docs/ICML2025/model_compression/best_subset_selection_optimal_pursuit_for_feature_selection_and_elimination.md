---
title: >-
  [论文解读] Best Subset Selection: Optimal Pursuit for Feature Selection and Elimination
description: >-
  [ICML 2025][模型压缩][Best Subset Selection] 本文从优化视角重新审视经典最优子集选择中的特征选择/消除准则，发现传统准则（相关性选择 + Wald-T 消除）仅捕获了目标函数的"一步变化"而忽视了特征交互，从而提出了"目标函数感知"的最优选择和消除准则，将其作为元替换（Meta-Substitution）即插即用地增强 OMP/CoSaMP/(A)BESS 等经典算法，在压缩感知和稀疏回归任务上实现显著性能提升且不增加计算复杂度。
tags:
  - ICML 2025
  - 模型压缩
  - Best Subset Selection
  - Optimal Pursuit
  - Feature Selection
  - Compressed Sensing
  - Sparse Regression
---

# Best Subset Selection: Optimal Pursuit for Feature Selection and Elimination

**会议**: ICML 2025  
**arXiv**: [2501.16815](https://arxiv.org/abs/2501.16815)  
**代码**: https://github.com/ZhihanZhu-math/Optimal_Pursuit_public (有)  
**领域**: 模型压缩 / 特征选择 / 稀疏优化  
**关键词**: Best Subset Selection, Optimal Pursuit, Feature Selection, Compressed Sensing, Sparse Regression

## 一句话总结

本文从优化视角重新审视经典最优子集选择中的特征选择/消除准则，发现传统准则（相关性选择 + Wald-T 消除）仅捕获了目标函数的"一步变化"而忽视了特征交互，从而提出了"目标函数感知"的最优选择和消除准则，将其作为元替换（Meta-Substitution）即插即用地增强 OMP/CoSaMP/(A)BESS 等经典算法，在压缩感知和稀疏回归任务上实现显著性能提升且不增加计算复杂度。

## 研究背景与动机

**领域现状**：最优子集选择（Best Subset Selection）是统计学和机器学习中的基准问题，目标是在 $\ell_0$ 约束下选择最优特征子集。由于该问题是 NP-hard 的，实践中主要依赖贪心算法（如 OMP、CoSaMP、IHT、ABESS 等）通过迭代式的特征选择与消除来求解。

**现有痛点**：贪心算法的特征选择通常基于**相关性准则**（选择与残差最相关的特征），特征消除基于 **Wald-T 统计量**（移除 T 值最小的特征）。然而，这两种经典准则只关注单个特征的重要性，忽略了特征之间的交互作用——一个在当前活跃集中看起来重要的特征，可能在活跃集改变后变得不重要，反之亦然。

**核心矛盾**：从优化角度分析，经典准则等价于对特征选择/消除子问题做了一步块坐标下降（Block Coordinate Descent），只考虑了"更新支撑集但固定系数"这一步的目标函数变化，忽略了"在新支撑集上更新系数"带来的额外目标函数变化。换句话说，经典准则追求的是局部一步最优，而非全局最优。

**本文目标** 设计能精确求解特征进入/退出子问题的最优准则，使得每一步选择/消除都是在考虑"支撑集更新 + 系数更新"两步联合效果后的最优决策。

**切入角度**：作者将经典准则统一到优化子问题 (P0)/(Q0) 的框架下，揭示其局限，进而构建了更完整的优化子问题 (P1)/(Q1)，通过精确求解导出闭式最优准则。

**核心 idea**：将经典贪心子集选择中的特征选择/消除准则替换为精确求解选择/消除子问题的最优解，以"元替换"方式无缝增强所有贪心子集选择算法。

## 方法详解

### 整体框架

本文的方法框架可概括为三步：

1. **分析经典准则的优化本质**：证明经典相关性选择准则（公式 3）和 Wald-T 消除准则（公式 4）分别对应优化子问题 (P0) 和 (Q0)——它们只允许支撑集改变一个元素且固定剩余系数，本质上是一步块坐标下降。
2. **构建并精确求解完整子问题**：提出新的优化子问题 (P1)/(Q1)，允许支撑集改变一个元素的同时更新所有系数（即考虑"完整下降"），并利用矩阵逆运算引理推导出闭式最优准则。
3. **元替换（Meta-Substitution）**：将新准则作为即插即用模块替换经典算法中的旧准则，生成一系列增强算法（OP、CoSaOP、OP-(A)BESS 等），且不增加计算复杂度量级。

### 关键设计

1. **最优选择准则（Objective-based Selection, 公式 8）**:

    - 功能：从支撑集外部选择一个最优特征加入。
    - 核心思路：求解选择子问题 (P1) 的等价形式 (P2)，利用前向矩阵逆引理（Lemma 3.2），导出闭式准则：
    $j^* = \arg\max_{j \in S^c} \frac{(\mathbf{r}^T \mathbf{X}_j)^2}{\mathbf{X}_j^T (\mathbf{I} - \mathbf{X}_S (\mathbf{X}_S^T \mathbf{X}_S)^{-1} \mathbf{X}_S^T) \mathbf{X}_j}$
      分子与经典准则相同（残差与特征的相关性），但分母引入了投影矩阵 $\mathbf{I} - \mathbf{X}_S (\mathbf{X}_S^T \mathbf{X}_S)^{-1} \mathbf{X}_S^T$，即当前已选特征的正交补空间。
    - 设计动机：新准则可理解为特征 $\mathbf{X}_j$ 与残差在 $\mathbf{X}_S$ 的噪声子空间中的相关性，捕获了特征之间的"综合组合效应"。当 $\mathbf{X}_j$ 与 $\mathbf{X}_S$ 正交时退化为经典准则。
    - 与之前方法的区别：经典准则分母仅有 $\|\mathbf{X}_j\|_2$，不考虑候选特征与已选特征的相关性；新准则通过投影矩阵自动惩罚与已选特征高度相关的候选特征。

2. **最优消除准则（Objective-based Elimination, 公式 10）**:

    - 功能：从支撑集内部选择一个最不重要的特征移除。
    - 核心思路：求解消除子问题 (Q1) 的等价形式 (Q2)，利用后向矩阵逆引理（Lemma 3.10），导出准则：选择使得移除后剩余特征仍能最大化解释 $\mathbf{y}$ 的特征进行消除。具体公式涉及 $\mathbf{C}_{k-1} = (\mathbf{X}_S^T \mathbf{X}_S)^{-1}$ 的分块更新。
    - 设计动机：经典 Wald-T 准则只看 $|\beta_j|$ 大小来判断特征重要性，忽略了移除一个特征后对其他特征系数的影响。新准则综合考虑了消除特征 $j$ 后，剩余特征的系数重新调整对目标函数的影响。
    - 与之前方法的区别：当特征正交时退化为经典 Wald-T 准则，但在特征高相关时能显著避免误删真实特征。

3. **元替换策略（Meta-Substitution）**:

    - 功能：将上述最优准则作为通用模块，替换任意贪心子集选择算法中的经典准则。
    - 核心思路：贪心子集选择算法可分为三类——仅选择（如 OMP→OP）、先选后删（如 CoSaMP→CoSaOP）、交换式（如 ABESS→OP-ABESS）。无论哪类算法，只需将其选择/消除步骤中的经典准则替换为最优准则即可。
    - 设计动机：最优准则的计算所需的矩阵逆 $(\mathbf{X}_S^T \mathbf{X}_S)^{-1}$ 在更新系数的最小二乘步骤中已经求解过，因此替换不会增加额外的计算复杂度量级。

### 理论分析

- **最优性保证（Theorem 4.1 & 4.2）**：新准则在每一步进/退过程中都是最优决策，即选入的特征使目标函数下降最多，移除的特征使目标函数上升最少。
- **复杂度等价（Theorem 4.3）**：增强算法（OP、CoSaOP、OP-ABESS）与原算法的计算复杂度同阶。
- **CoSaOP 收敛性（Theorem 4.9）**：在 RIP 条件下，CoSaOP 保持线性收敛率 $\|\boldsymbol{\beta}^k - \boldsymbol{\beta}^*\|_2 \leq 0.869 \|\boldsymbol{\beta}^{k-1} - \boldsymbol{\beta}^*\|_2 + 14.482 \|\boldsymbol{\epsilon}\|_2$。
- **高相关性场景优势（Theorem 4.10 & 4.11）**：当特征高度相关时（相关系数 $\rho$ 接近 1），经典相关性准则的上界衰减为 $\sqrt{1-\rho^2}$，而最优准则的下界放大为 $1/(1-\rho^2)$，优势倍增。此外，在存在伪相关特征时，经典准则可能错误删除真实特征，而最优准则能正确识别并移除伪特征。

## 实验关键数据

### 主实验：压缩感知（AudioSet 真实数据）

| 音频数据 | OMP (NMSE) | OP (NMSE) | 增益 | CoSaMP (NMSE) | CoSaOP (NMSE) | 增益 | (A)BESS (NMSE) | OP-(A)BESS (NMSE) | 增益 |
|----------|------------|-----------|------|---------------|---------------|------|----------------|-------------------|------|
| Audio 1 | 9.45E-04 | 7.69E-04 | 19% | 2.36E-03 | 1.64E-03 | 31% | 1.41E-03 | 6.22E-04 | **56%** |
| Audio 2 | 5.79E-03 | 5.49E-03 | 5% | 1.87E-03 | 6.36E-04 | **66%** | 6.71E-03 | 6.36E-04 | **91%** |
| Audio 3 | 1.46E-03 | 1.18E-03 | 19% | 1.54E-03 | 5.52E-04 | **64%** | 2.84E-03 | 5.52E-04 | **81%** |
| Audio 4 | 4.05E-02 | 3.55E-02 | 12% | 2.85E-03 | 8.22E-04 | **71%** | 2.65E-02 | 1.86E-02 | 30% |

### 压缩感知合成数据：成功恢复次数

| 场景 | OMP | OP | 增益 | CoSaMP | CoSaOP | 增益 | (A)BESS | OP-(A)BESS | 增益 |
|------|-----|-----|------|--------|--------|------|---------|------------|------|
| 采样率 0.25, SNR=15 | ~50 | ~150 | **~3×** | ~30 | ~200 | **~7×** | ~450 | ~470 | 可观 |
| 采样率 0.50, SNR=15 | ~400 | ~480 | 20% | ~300 | ~490 | **63%** | ~495 | ~500 | 趋饱和仍有提升 |

### 消融与分析：稀疏回归 $R^2$ 表现

| 数据集 | 算法对 | 选 K=5 时 $R^2$ 提升 | 选 K=10 时 $R^2$ 提升 | 备注 |
|--------|--------|---------------------|----------------------|------|
| Boston Housing | OMP→OP | ~0.1 | ~0.05 | 在特征数少时提升更显著 |
| California Housing | ABESS→OP-ABESS | ~0.1 | ~0.08 | 等效多选 10 个特征的效果 |
| Superconductivity | CoSaMP→CoSaOP | 从失败到成功 | 大幅提升 | CoSaMP 在回归任务上失败，CoSaOP 仍然有效 |
| Spectra (p=401, n=60) | 全部三组 | 一致提升 | 一致提升 | 高维小样本场景优势更明显 |

### 关键发现

- **CoSaOP 增益最大**：在压缩感知中 CoSaOP 相比 CoSaMP 平均提升 4-7 倍成功恢复率，在 AudioSet 上 NMSE 降低 30%-71%，是三组算法中改进最显著的。
- **特征高相关性下优势放大**：在高特征相关性 + 小样本 + 高维 + 高噪声的极端场景下，增强算法的优势更加突出（见 Appendix P）。
- **CoSaMP 在回归任务上失败但 CoSaOP 依旧有效**：经典 CoSaMP 在稀疏回归任务上完全失效，而替换准则后的 CoSaOP 恢复了有效性，展示了最优准则的鲁棒性。
- **计算开销几乎不增加**：运行时间对比显示新旧算法仅有微小差异（见 Appendix O）。

## 亮点与洞察

- **从优化角度统一经典准则的理论框架**：将看似不同的相关性选择和 Wald-T 消除准则统一到 Forward/Backward Sacrifice 的框架下，揭示了它们都是块坐标下降一步的产物，这个视角非常优雅，也为改进指明了方向。
- **元替换（Meta-Substitution）的即插即用设计**：最优准则不绑定任何特定算法，而是作为通用模块替换所有贪心子集选择算法中的经典准则，这种"准则级别的升级"思路具有很强的可扩展性。
- **零额外计算成本的性能提升**：利用系数更新步骤中已有的矩阵逆 $(\mathbf{X}_S^T \mathbf{X}_S)^{-1}$，最优准则几乎不增加额外计算量——这是一种"免费的性能提升"。
- **在特征相关性高时自动放大优势**：当特征正交时新准则退化为经典准则（不会变差），而特征相关性越高，新准则的优势越大——这恰好是实际应用中最需要帮助的场景。

## 局限与展望

- **仅限线性模型**：目前理论和实验都集中在线性回归 / 压缩感知的平方损失上，未涉及非线性特征选择（如深度学习中的特征选择）。
- **仅限 $\ell_0$ 约束**：最优准则推导依赖于最小二乘的闭式解，扩展到其他结构化稀疏（如 group sparsity, tree sparsity）需要额外工作。
- **未与深度学习特征选择方法比较**：随着大模型时代的到来，在高维非线性特征选择中的表现尚未验证。
- **Gradient Pursuit 扩展的实用性有待验证**：论文提出了 Optimal Gradient Pursuit 用于超高维场景，但主实验中未深入评估。
- **实验数据集偏小规模**：最大的数据集仅有几百维度和几百样本，在实际工业级大规模特征选择中的表现未知。

## 相关工作与启发

- **vs OMP / Matching Pursuit**：OMP 使用相关性准则逐步选择特征，本文的 OP 通过在分母引入投影矩阵来修正，额外考虑候选特征与已选特征的交互。在高相关特征场景中 OP 优势巨大（3 倍提升）。
- **vs CoSaMP**：CoSaMP 采用先选后删的策略，选择和消除都用经典准则。CoSaOP 替换为最优准则后，即使 CoSaMP 在稀疏回归上完全失败，CoSaOP 仍然有效，说明最优准则具有更好的鲁棒性。
- **vs ABESS**：ABESS 是当前子集选择领域的 SOTA 基准方法，采用交换式策略。即使 ABESS 已近饱和，OP-ABESS 仍能取得可观察的元增益（如 AudioSet 上 56%-91% NMSE 降低）。
- **vs LASSO / SCAD / MCP**：这些松弛方法用连续惩罚近似 $\ell_0$，计算可能较重且难以精确控制所选特征数；本文直接在组合优化层面改进贪心准则。
- **vs SMP (Tohidi et al., 2025)**：独立同期工作，从次模性角度推导出与准则 (8) 类似的结果，但仅用于替换 OMP；本文提供了统一的优化框架，同时覆盖选择和消除准则，适用于所有贪心算法。

## 评分

- 新颖性: ⭐⭐⭐⭐ 从优化角度统一经典准则并推导最优替代，思路清晰优雅，但核心数学工具（矩阵逆引理）较为经典
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖压缩感知与稀疏回归两大任务，合成+真实数据，三组算法对比，多种评估指标，附录中还有高相关性、超高维、无监督、复信号处理等扩展实验
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链条清晰，图表直观，从旧准则到新准则的推导自然流畅，Figure 1 的可视化对比非常有效
- 价值: ⭐⭐⭐⭐ 作为经典最优子集选择领域的理论+方法改进，具有很好的学术价值；Meta-Substitution 思路有启发性，但应用范围受限于线性稀疏模型

<!-- RELATED:START -->

## 相关论文

- [Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)
- [Predictive Data Selection: The Data That Predicts Is the Data That Teaches](predictive_data_selection_the_data_that_predicts_is_the_data_that_teaches.md)
- [OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [Mitigating Selection Bias with Node Pruning and Auxiliary Options](../../ACL2025/model_compression/selection_bias_node_pruning.md)
- [DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)

<!-- RELATED:END -->
