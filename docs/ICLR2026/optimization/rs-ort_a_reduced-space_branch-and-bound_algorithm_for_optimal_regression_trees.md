---
title: >-
  [论文解读] RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees
description: >-
  [ICLR 2026][优化/理论][最优决策树] 提出 RS-ORT 算法，通过将回归树训练重构为两阶段优化问题并在缩减空间上进行分支定界（仅对树结构变量分支），结合闭式叶预测、阈值离散化和精确末层子树解析等加速策略，首次在包含连续特征的 200 万样本数据集上实现了有全局最优性保证的回归树学习。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "最优决策树"
  - "回归树"
  - "分支定界"
  - "混合整数规划"
  - "可解释机器学习"
---

# RS-ORT: A Reduced-Space Branch-and-Bound Algorithm for Optimal Regression Trees

**会议**: ICLR 2026  
**arXiv**: [2510.23901](https://arxiv.org/abs/2510.23901)  
**领域**: 优化  
**关键词**: 最优决策树, 回归树, 分支定界, 混合整数规划, 可解释机器学习

## 一句话总结

提出 RS-ORT 算法，通过将回归树训练重构为两阶段优化问题并在缩减空间上进行分支定界（仅对树结构变量分支），结合闭式叶预测、阈值离散化和精确末层子树解析等加速策略，首次在包含连续特征的 200 万样本数据集上实现了有全局最优性保证的回归树学习。

## 研究背景与动机

决策树因其强可解释性广泛应用于医疗、金融等高风险领域。传统启发式方法（ID3、C4.5、CART）容易产生次优解，而真正的最优决策树是 NP-complete 问题。

**现有最优回归树方法的局限**：

**MIP 方法**（Bertsimas & Dunn 2019）：可保证全局最优但计算不可扩展，搜索空间与样本量相关

**DPB 方法**（Zhang et al. 2023, 当前 SOTA）：可扩展到 200 万样本，但**必须将连续特征二值化**，损失了全局最优性且可能导致不必要的深树

**进化搜索**（evtree）：无最优性保证

**其他 MIP 变体**：限制树的大小或分裂数量以保证可扩展性

**核心挑战**：连续特征使回归树的搜索空间急剧膨胀。即使对于深度 $D=2$、$n=1030$、$P=8$ 的小数据集（Concrete），不同树结构数量就超过 $1.39 \times 10^{11}$。

## 方法详解

### 两阶段优化重构

将回归树训练分解为：

- **第一阶段变量** $m = (a, b, c, d)$：树结构（分裂特征 $a$、阈值 $b$、叶预测 $c$、分裂指示 $d$）
- **第二阶段变量**：样本依赖变量（叶分配 $z$、损失计算 $f, L$）

目标函数：

$$\min_{a,b,c,d} \sum_{i \in \mathcal{N}} Q_i(m). \quad Q_i(m) = \min_{z_i, L_{i*}} \frac{L_{i*}}{\hat{L}} + \frac{\lambda}{n}\sum_{t \in \mathcal{T}_D} d_t$$

### 缩减空间分支定界（RS-BB）

**核心设计**：仅对第一阶段（树结构）变量进行分支，而第二阶段变量在每个 BB 节点中通过求解子问题得到。

**关键性质**：搜索空间的维度与训练样本数量无关，仅由树的深度和特征数量决定。

**收敛保证（Theorem 2）**：

$$\lim_{t \to \infty} \alpha_t = \lim_{t \to \infty} \beta_t = f^*$$

即仅对结构变量分支即可保证收敛到全局最优。

### 三大加速策略

**策略 1：叶预测隐式化（Theorem 3）**

固定树结构后，每个叶节点的最优预测值有闭式解：

$$c_t^* = \frac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t} y_i$$

这消除了 $|\mathcal{T}_L|$ 个连续变量，指数级减少了 BB 节点数。

**策略 2：阈值离散化**

将连续阈值 $b_t$ 限制在训练数据的实际特征值上，因为连续两个排序值之间的任何阈值都产生相同的分割。在 BB 中，通过取可行区间的中位数索引进行二分，每次分支至少消除一个可行分割点。

**策略 3：末层子树精确解析**

当深度 $D-2$ 以上的结构变量全部固定后，每个父节点 $P$ 的最优深度-1 子树可通过 CART（max_depth=1）精确求解。如果分裂增益 $\Delta(P) > \lambda|P|/\hat{L}$，接受分裂；否则保持为叶。

### 并行化

问题的可分解性使得下界和上界计算可以在样本维度上平行执行，支持大规模分布式计算（实验中使用 40-1000 个 CPU 核心）。

## 实验关键数据

### 主实验：连续特征数据集性能

| 数据集 | $n$ | $P$ | 方法 | Train RMSE | Test RMSE | Gap (%) | Time (s) |
|--------|------|-----|------|-----------|-----------|---------|----------|
| Concrete | 1,030 | 8 | **RS-ORT** | 11.96 | **11.80** | **<0.01** | 631 |
| | | | Bertsimas | 11.96 | 11.80 | <0.01 | 10047 |
| | | | OSRT | 11.96 | 11.80 | <0.01 | 116 |
| | | | CART | 12.01 | 12.57 | - | - |
| CPU ACT | 8,192 | 21 | **RS-ORT** | **5.99** | **6.03** | 8.08 | 14400 |
| | | | Bertsimas | 6.01 | 6.03 | 100.00 | 14400 |
| | | | OSRT | OoM | OoM | OoM | OoM |
| Seoul Bike | 8,760 | 12 | **RS-ORT** | **478.67** | **495.92** | **<0.01** | 10116 |

### OSRT 与 RS-ORT 关键对比

| 维度 | OSRT (SOTA) | RS-ORT |
|------|-------------|--------|
| 连续特征 | 需二值化 | 直接处理 |
| 搜索空间 | 与样本量相关 | 与样本量无关 |
| 最大规模 | 200万（二值化） | 200万（连续） |
| 全局最优保证 | 仅二值化后 | 连续特征下 |
| 并行化 | 有限 | 高度可分解 |

### 关键实验发现

1. 在所有小-中规模数据集上，RS-ORT 与最优基线达到相同的训练/测试 RMSE，但在大数据集上优势明显
2. Household 数据集（200 万样本，连续特征）：RS-ORT 在 4 小时内找到全局最优树，**这是文献中首次有精确方法在此规模连续特征数据上成功**
3. RS-ORT 生成的树通常比竞争方法浅 2-3 层，同时保持更好的测试性能
4. OSRT 在 CPU ACT（8192 样本、21 特征）上内存溢出，而 RS-ORT 多核并行后可处理

## 亮点与洞察

1. **搜索空间与样本量解耦**：这是与所有现有 MIP 方法的根本区别，使得算法对大数据集天然友好
2. **三个加速策略相辅相成**：闭式解减变量、离散化缩域、末层解析剪枝，每个都利用了问题的特殊结构
3. **连续特征直接处理**：避免了二值化带来的信息损失和不必要的树深度增加
4. **实际工程价值高**：在可解释 AI 场景（医疗、合规审计）中，全局最优的浅决策树比近似最优的深树更有价值
5. **并行计算设计自然**：下界计算的样本级可分解性使得并行化无需额外通信

## 局限性

1. 计算资源需求大：大数据集需要数百至上千 CPU 核心，普通用户难以使用
2. 固定深度-2 的实验设置局限了对更深树的性能评估
3. 4 小时时间限制可能不适用于实时或交互式场景
4. 仅处理单变量分裂（axis-aligned splits），未扩展到多变量分裂
5. 正则化参数 $\lambda$ 的选择依赖先验知识，论文未提供自动化策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 两阶段分解 + 缩减空间 BB 的组合在回归树上是重要创新
- **实验**: ⭐⭐⭐⭐⭐ — 从小到 200 万样本的全面实验，首次攻克连续特征大规模最优回归树
- **写作**: ⭐⭐⭐⭐ — 问题定义清晰，算法描述详细，理论证明完整
- **价值**: ⭐⭐⭐⭐⭐ — 对可解释 AI 和最优决策树领域具有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](../../ICML2025/optimization/a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[CVPR 2026\] OTPrune: Distribution-Aligned Visual Token Pruning via Optimal Transport](../../CVPR2026/optimization/otprune_distribution-aligned_visual_token_pruning_via_optimal_transport.md)

</div>

<!-- RELATED:END -->
