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

### 整体框架

RS-ORT 把回归树训练改写成一个两阶段优化问题，再用分支定界（branch-and-bound, BB）求全局最优——但关键在于它只在「树结构」这层缩减空间上分支，样本相关的变量留给子问题闭式或精确求解。再叠加叶预测隐式化、阈值离散化、末层子树解析三个利用问题结构的加速策略，使搜索维度与样本量彻底脱钩，从而首次把精确回归树推到 200 万连续特征样本的规模。

### 关键设计

**1. 两阶段优化重构：把样本相关变量从分支变量里剥离出去。** 朴素 MIP 把树结构和每个样本的叶分配、损失全都塞进一个大优化问题，变量数随样本量线性膨胀，这正是现有方法不可扩展的根源。RS-ORT 把变量切成两层：第一阶段是树结构 $m = (a, b, c, d)$，即分裂特征 $a$、阈值 $b$、叶预测 $c$、分裂指示 $d$；第二阶段是样本依赖变量，即叶分配 $z$、损失 $f, L$。整个目标写成 $\min_{a,b,c,d} \sum_{i \in \mathcal{N}} Q_i(m)$，其中每个样本的子问题 $Q_i(m) = \min_{z_i, L_{i*}} \frac{L_{i*}}{\hat{L}} + \frac{\lambda}{n}\sum_{t \in \mathcal{T}_D} d_t$ 在结构固定后可独立求解。这样一来，外层优化只面对结构变量，样本只在内层子问题里出现。

**2. 缩减空间分支定界：只对结构变量分支，搜索空间与样本量无关。** 有了两阶段分解，BB 就不必再对样本依赖变量枚举——每个 BB 节点固定一部分结构变量后，第二阶段变量直接通过求解子问题得到上下界。由此搜索空间的维度只由树深度和特征数量决定，与训练样本数 $n$ 完全无关，这是和所有现有 MIP 方法的根本分水岭。论文还证明（Theorem 2）这种「只分支结构变量」的策略并不损失最优性：上下界序列满足 $\lim_{t \to \infty} \alpha_t = \lim_{t \to \infty} \beta_t = f^*$，即仍收敛到全局最优。

**3. 叶预测隐式化：闭式解消掉一整层连续变量。** 叶预测值 $c$ 本来也是要优化的连续变量，会让 BB 节点数指数膨胀。论文指出（Theorem 3）一旦树结构定下来、样本落入哪个叶也就定了，每个叶的最优预测就是落入样本标签的均值 $c_t^* = \frac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t} y_i$。于是 $|\mathcal{T}_L|$ 个叶预测变量被整体隐式化，不必再进 BB 枚举，节点数指数级下降。

**4. 阈值离散化：把连续阈值压回有限候选并二分缩域。** 连续阈值 $b_t$ 理论上有无穷多取值，但排序后相邻两个特征值之间的任何阈值都切出同一个划分，因此最优阈值只需在训练数据实际出现的特征值里找。RS-ORT 进一步在 BB 里对这个有限候选集做二分：每次取可行区间的中位数索引分支，保证每次分支至少消除一个候选分割点，把阈值搜索从连续区间收成对数级的离散搜索。

**5. 末层子树精确解析：底部两层直接解析求解而非继续分支。** 当深度 $D-2$ 以上的结构变量都已固定，剩下的末层子树规模很小，没必要再展开 BB。此时每个父节点 $P$ 的最优深度-1 子树可以直接用 CART（max_depth=1）精确求出：若分裂增益 $\Delta(P) > \lambda|P|/\hat{L}$ 就接受分裂，否则把 $P$ 留作叶节点。这相当于在 BB 树底部接了一段解析的「收尾」，省掉最深一层的枚举。

**6. 样本级可分解的并行化：下界上界都能按样本铺开算。** 由于第二阶段子问题对每个样本独立，下界和上界的计算天然在样本维度上可分解，可直接铺到大量计算节点上并行，且无需节点间通信。实验里据此用了 40 到 1000 个 CPU 核心，使大规模数据集的求解时间落在可接受范围内。

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
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](../../ICML2025/optimization/a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](../../CVPR2026/optimization/label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[AAAI 2026\] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](../../AAAI2026/optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)

</div>

<!-- RELATED:END -->
