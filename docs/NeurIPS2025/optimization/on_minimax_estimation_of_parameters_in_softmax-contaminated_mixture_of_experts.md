---
title: >-
  [论文解读] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts
description: >-
  [NeurIPS 2025][优化/理论][混合专家模型] 首次对 softmax 门控的污染混合专家（contaminated MoE）模型进行系统理论分析，提出可区分性条件（distinguishability）…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "混合专家模型"
  - "参数估计"
  - "极小极大最优"
  - "可区分性"
  - "微调理论"
---

# On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts

**会议**: NeurIPS 2025  
**arXiv**: [2505.18455](https://arxiv.org/abs/2505.18455)  
**代码**: 无  
**领域**: 优化  
**关键词**: 混合专家模型, Softmax门控, 参数估计, 极小极大最优, 可微调性

## 一句话总结

首次对带 softmax 门控的受污染混合专家（contaminated MoE）模型进行极小极大参数估计分析，提出"可区分性"概念刻画预训练模型与 prompt 的关系，证明可区分时 MLE 达到参数级 $\tilde{O}(n^{-1/2})$ 最优速率，不可区分时速率显著变慢。

## 研究背景与动机

混合专家模型（MoE）通过门控网络动态分配输入相关权重给不同专家，在 NLP（DeepSeek-V3, Mixtral）、视觉（M3ViT）等领域广泛应用。**受污染 MoE（Contaminated MoE）**是参数高效微调（如 prefix tuning）的理论建模：一个冻结的预训练专家与一个可训练的 prompt 专家组成混合。

尽管实践中非常流行，受污染 MoE 的理论性质几乎未被探索。此前仅有的工作 (Yan et al., Nguyen et al.) 研究了**输入无关门控**（即固定常数权重）下的情况，与实际使用的 softmax 门控差距很大。

**核心问题**：在 softmax 门控下，预训练模型与 prompt 模型之间的关系如何影响参数估计？当 prompt 学到了与预训练模型重叠的知识时会发生什么？

## 方法详解

### 整体框架

研究对象是 softmax-contaminated MoE 的条件密度：

$$p_{G_*}(y|x) = \frac{1}{1+\exp(\beta^{*\top}x + \tau^*)} f_0(y|h_0(x,\eta_0), \nu_0) + \frac{\exp(\beta^{*\top}x + \tau^*)}{1+\exp(\beta^{*\top}x + \tau^*)} f(y|h(x,\eta^*), \nu^*)$$

其中第一项是固定预训练模型（已知），第二项是可训练 prompt 模型（未知），未知参数 $G_* = (\beta^*, \tau^*, \eta^*, \nu^*)$ 包括门控参数和 prompt 专家参数。使用 MLE 进行估计。

### 关键设计

1. **可区分性条件 (Distinguishability, Definition 1)**：定义预训练模型 $f_0$ "可与 prompt 模型 $f$ 区分"的解析条件——要求 $f_0$ 与 $f$ 的各阶导数构成的函数集在函数空间中线性无关。核心含义是预训练和 prompt 具有**不同的专业知识**。**Proposition 1** 给出简洁判定：若 $f_0$ 不属于高斯密度族则自动满足；当 $f_0$ 为高斯且与 prompt 共享相同专家函数 $h_0 = h$ 时条件被违反。

2. **可区分情况下的参数估计 (Theorem 1)**：所有参数（门控 $\beta, \tau$ 和 prompt $\eta, \nu$）的 MLE 收敛速率均为参数级 $\tilde{O}(n^{-1/2})$。相比输入无关门控，softmax 门控下 prompt 参数估计速率**不再依赖门控参数趋近零的速度**，因此更快——说明 softmax 门控在统计意义上更高效。

3. **不可区分情况下的参数估计 (Theorem 3)**：当 prompt 与预训练模型知识重叠（$(\eta^*, \nu^*) \to (\eta_0, \nu_0)$）时，估计速率显著变慢：

    - 门控参数：$\tilde{O}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-2})$
    - 专家参数：$\tilde{O}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-1})$
   
   例如，若 prompt 参数以 $O(n^{-1/8})$ 趋近预训练参数，则门控和专家参数 MLE 收敛速率分别降至 $O(n^{-1/4})$ 和 $O(n^{-3/8})$。

4. **强可辨识性条件 (Strong Identifiability, Definition 2)**：为不可区分情况推导精确速率所需，要求专家函数 $h$ 满足三组函数线性无关条件。tanh、sigmoid、GELU 满足，ReLU 因二阶导消失而不满足。

### 损失函数 / 训练策略

- **极小极大下界 (Theorems 2 & 4)**：两种情况下 MLE 速率均为极小极大最优（匹配下界至对数因子）
- **技术创新**：允许真实参数 $G_*$ 随样本量 $n$ 变化，分析均匀收敛速率而非点态，更贴近实际
- 使用 EM 算法 + BFGS 优化器计算 MLE

## 实验关键数据

### 主实验

**可区分设置**（$f_0$: Laplace, $f$: Gaussian, $d=8$）

| 参数 | 理论速率 | 实验拟合速率 |
|------|---------|------------|
| $\beta$ | $O(n^{-1/2})$ | $O(n^{-0.45})$ |
| $\tau$ | $O(n^{-1/2})$ | $O(n^{-0.52})$ |
| $\eta$ | $O(n^{-1/2})$ | $O(n^{-0.50})$ |
| $\nu$ | $O(n^{-1/2})$ | $O(n^{-0.54})$ |

### 消融实验

**不可区分设置**（$f_0, f$: Gaussian，$\eta^*$ 以 $O(n^{-1/8})$ 趋近 $\eta_0$）

| 参数 | 理论速率 | 实验拟合速率 |
|------|---------|------------|
| $\exp(\tau)$ | $O(n^{-1/4})$ | $O(n^{-0.23})$ |
| $\beta$ | $O(n^{-3/8})$ | $O(n^{-0.37})$ |
| $\eta$ | $O(n^{-3/8})$ | $O(n^{-0.39})$ |
| $\nu$ | $O(n^{-3/8})$ | $O(n^{-0.35})$ |

### 关键发现

- 实验速率与理论预测高度吻合，验证理论正确性
- 可区分设置下所有参数均以接近 $n^{-1/2}$ 的速率收敛
- 不可区分设置下门控速率慢至约 $n^{-1/4}$，专家参数慢至约 $n^{-3/8}$
- **Softmax 门控优于输入无关门控**：消除了 prompt 参数估计对门控参数的依赖

## 亮点与洞察

- **可区分性概念**是全新理论工具，精确刻画了微调中"prompt 与预训练模型知识重叠"的问题本质
- **两个实践指导**：(1) 应使用 softmax 门控而非输入无关门控以提高样本效率；(2) prompt 模型应被设计为与预训练模型具有不同专长
- 参数随样本量变化的分析框架为 MoE 理论带来了新标准
- 理论结果为 prefix tuning 的参数效率提供了统计学解释

## 局限与展望

- 当前限于单个 prompt 专家，扩展到多 prompt 混合是重要未来方向
- prompt 限于高斯分布族，扩展到更一般分布将增强实用性
- 仅有合成实验，缺乏真实微调场景验证
- 未考虑高维 $d$ 对估计难度的影响
- 未联系具体优化算法的初始化敏感性问题

## 相关工作与启发

- **与 Ho et al. (2022) 的关系**：从一般 MoE 代数独立性推广到受污染 MoE 可区分性
- **与 Nguyen et al. (2023, 2024) 的关系**：从 softmax MoE 和输入无关受污染 MoE 推广到 softmax 受污染 MoE
- **与 prefix tuning 的联系**：prompt 应学习与预训练模型互补的知识
- **启发**：可区分性条件可能与模型合并和专家路由策略设计相关

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （可区分性定义新颖，softmax 受污染 MoE 理论为首次）
- 实验充分度: ⭐⭐⭐ （合成实验验证理论，但缺乏真实场景）
- 写作质量: ⭐⭐⭐⭐ （数学严谨，结构清晰，但符号较重）
- 价值: ⭐⭐⭐⭐ （为 MoE 微调理论提供坚实基础，实践指导有价值）
**领域**: Optimization / Statistical Learning Theory  
**关键词**: 混合专家模型, 参数估计, 极小极大最优, 可区分性, 微调理论

## 一句话总结

首次对 softmax 门控的污染混合专家（contaminated MoE）模型进行系统理论分析，提出可区分性条件（distinguishability），在该条件成立时证明 MLE 达到参数级 $\widetilde{\mathcal{O}}(n^{-1/2})$ 极小极大最优估计速率，在条件不成立（prompt 与预训练模型知识重叠）时揭示估计速率显著变慢的本质机制。

## 研究背景与动机

混合专家模型（MoE）通过门控网络动态分配输入依赖的权重给多个子模型，广泛应用于 NLP、CV、多模态等领域。在参数高效微调（如 prefix tuning）中，模型可被理解为**污染 MoE**：一个冻结的预训练专家加上一个可训练的 prompt 专家。

**现有理论空白**：

(1) **输入无关门控的局限**：此前 Yan et al. 和 Nguyen et al. 仅研究了门控与输入无关的污染 MoE，即混合权重为常数 $\lambda^*$，这在实际中极不现实。真实系统使用 softmax 门控 $\frac{\exp(\beta^{\top}x + \tau)}{1 + \exp(\beta^{\top}x + \tau)}$，权重依赖于输入。

(2) **参数随样本量变化**：先前 MoE 理论假设真实参数 $G^*$ 不随样本量 $n$ 变化，仅提供点态收敛率。本文允许 $G^*$ 依赖于 $n$，提供**一致（uniform）**收敛率，更接近实际且对极小极大分析不可或缺。

(3) **核心问题**：当 prompt 与预训练模型学到的知识高度重叠时，参数估计会发生什么？softmax 门控相比常数门控是否有优势？

**切入角度**：提出"可区分性"（distinguishability）这一新的分析概念，将问题分为可区分和不可区分两种设置，分别建立上界和匹配的极小极大下界。

## 方法详解

### 整体框架

考虑 softmax 污染 MoE 模型：

$$p_{G_*}(y|x) = \frac{1}{1+\exp(\beta^{*\top}x + \tau^*)} f_0(y|h_0(x,\eta_0), \nu_0) + \frac{\exp(\beta^{*\top}x + \tau^*)}{1+\exp(\beta^{*\top}x + \tau^*)} f(y|h(x,\eta^*), \nu^*)$$

其中 $f_0$ 为冻结预训练模型（已知），$f$ 为 Gaussian prompt 模型（未知），$G_* = (\beta^*, \tau^*, \eta^*, \nu^*)$ 为待估计参数。使用最大似然估计（MLE）来估计 $G_*$。

### 关键设计

1. **可区分性条件 (Distinguishability, Definition 1)**:
    - 功能：定义预训练模型 $f_0$ 何时与 prompt 模型 $f$ "可区分"
    - 核心思路：要求 $f_0$ 与 $f$ 及其一阶偏导数组成的线性组合不能为零，除非所有系数均为零。直觉上，这排除了 prompt 学到与预训练模型完全相同知识的情况
    - 关键性质（Proposition 1）：**如果预训练模型 $f_0$ 不属于高斯密度族，则 $f_0$ 自动可区分于 prompt 模型 $f$**。反之，如果 $f_0$ 也是高斯分布且 $h_0 = h$（相同专家结构），则可区分性被违反

2. **可区分设置下的分析 (Theorem 1 & 2)**:
    - 功能：在可区分条件下建立 MLE 的收敛速率和极小极大下界
    - 核心结论：所有参数估计器（$\hat{\beta}_n, \hat{\tau}_n, \hat{\eta}_n, \hat{\nu}_n$）以 $\widetilde{\mathcal{O}}(n^{-1/2})$ 参数级速率收敛，这与极小极大下界匹配（至多差一个对数因子）——即**极小极大最优**
    - 关键发现：softmax 门控下 prompt 参数 $\eta^*, \nu^*$ 的估计速率为 $\widetilde{\mathcal{O}}(n^{-1/2})$，优于输入无关门控下的 $\widetilde{\mathcal{O}}(n^{-1/2} (\lambda^*)^{-1})$（后者依赖于门控参数收敛到零的速率）

3. **不可区分设置下的分析 (Theorem 3 & 4)**:
    - 功能：当 prompt 参数 $(\eta^*, \nu^*)$ 趋近预训练参数 $(\eta_0, \nu_0)$ 时的估计速率
    - 核心结论：估计速率显著变慢，依赖于 prompt 参数与预训练参数的距离 $\|(\Delta\eta^*, \Delta\nu^*)\|$
    - 具体速率：$\exp(\hat{\tau}_n)$ 的估计误差为 $\widetilde{\mathcal{O}}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-2})$；$(\hat{\beta}_n, \hat{\eta}_n, \hat{\nu}_n)$ 的估计误差为 $\widetilde{\mathcal{O}}(n^{-1/2} \cdot \|(\Delta\eta^*, \Delta\nu^*)\|^{-1})$
    - 例子：如果 $(\eta^*, \nu^*)$ 以 $\mathcal{O}(n^{-1/8})$ 趋近 $(\eta_0, \nu_0)$，则门控参数估计速率降至 $\mathcal{O}(n^{-1/4})$，专家参数降至 $\mathcal{O}(n^{-3/8})$

### 强可辨识性条件 (Strong Identifiability)

为处理不可区分设置下的 Taylor 展开，引入 Definition 2 要求专家函数 $h$ 的一阶、二阶偏导数满足三组线性无关性条件。满足此条件的例子包括 $h(x,\eta) = \text{GELU}(\eta^\top x)$、$\text{sigmoid}(\eta^\top x)$、$\tanh(\eta^\top x)$；不满足的例子包括 $\text{ReLU}(\eta^\top x)$（二阶导数几乎处处为零）。

## 实验关键数据

### 主实验（合成数据验证）

| 设置 | 参数 | 理论速率 | 实验速率 |
|------|------|---------|---------|
| 可区分 ($f_0$=Laplace) | $\hat{\beta}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.45})$ |
| 可区分 | $\hat{\tau}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.52})$ |
| 可区分 | $\hat{\eta}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.50})$ |
| 可区分 | $\hat{\nu}_n$ | $\mathcal{O}(n^{-1/2})$ | $\mathcal{O}(n^{-0.54})$ |

### 消融实验（不可区分设置）

| 变化参数 | $\exp(\hat{\tau}_n)$ 速率 | 理论预测 | $(\hat{\beta}_n, \hat{\eta}_n, \hat{\nu}_n)$ 速率 | 理论预测 |
|---------|------------------------|---------|------------------------------------------|---------|
| $\eta^*$ 以 $n^{-1/8}$ 趋近 $\eta_0$ | $\mathcal{O}(n^{-0.23})$ | $\mathcal{O}(n^{-1/4})$ | $\mathcal{O}(n^{-0.37\sim0.39})$ | $\mathcal{O}(n^{-3/8})$ |
| $\nu^*$ 以 $n^{-1/8}$ 趋近 $\nu_0$ | $\mathcal{O}(n^{-0.22})$ | $\mathcal{O}(n^{-1/4})$ | $\mathcal{O}(n^{-0.37\sim0.39})$ | $\mathcal{O}(n^{-3/8})$ |

实验结果与理论预测高度吻合，验证了理论分析的准确性。

### 关键发现

- **Softmax 门控优于常数门控**：在可区分设置下，softmax 门控消除了 prompt 参数估计对门控参数的依赖，将速率从 $\widetilde{\mathcal{O}}(n^{-1/2}(\lambda^*)^{-1})$ 提升至 $\widetilde{\mathcal{O}}(n^{-1/2})$
- **知识重叠是根本挑战**：当 prompt 学到与预训练模型重叠的知识时，门控参数估计误差放大 $\|(\Delta\eta^*, \Delta\nu^*)\|^{-2}$ 倍，专家参数放大 $\|(\Delta\eta^*, \Delta\nu^*)\|^{-1}$ 倍
- **密度估计不受影响**：无论是否可区分，模型密度本身的 Hellinger 距离估计始终保持 $\widetilde{\mathcal{O}}(n^{-1/2})$ 参数级速率

## 亮点与洞察

1. **问题动机清晰**：将参数高效微调（prefix tuning）形式化为污染 MoE 模型，为微调理论提供了新的统计视角
2. **可区分性概念优雅**：一个简洁的分析条件精准捕获了"prompt 是否学到预训练模型已有知识"这一实际关切
3. **理论完整性**：上下界匹配（极小极大最优），两种设置分别分析，结论互补
4. **实际指导意义**：理论直接建议 (1) 使用 softmax 门控而非常数门控；(2) 设计 prompt 模型使其学到与预训练模型不同的专业知识

## 局限与展望

- 仅考虑单个 prompt 模型的情况，多 prompt（多任务微调）的分析留待未来
- prompt 限定为高斯密度族，扩展到更一般的分布族（如混合分布）有待研究
- 实验仅在合成数据上验证，缺少在真实 LLM 微调场景的实证支持
- 强可辨识性条件（Definition 2）对 ReLU 网络不成立，限制了对主流架构的直接适用性
- 理论速率中对数因子的消除需要更精细的分析

## 相关工作与启发

- 建立在 MoE 理论研究谱系上：Ho et al. (2022) 的高斯 MoE → Nguyen et al. (2023) 的 softmax 门控 MoE → Yan et al. (2025) 的输入无关污染 MoE → 本文的 softmax 污染 MoE
- 对 prompt tuning/LoRA 等微调方法的理论理解提供了启发：prompt 应该学到与预训练模型**互补**而非重叠的知识
- 可区分性概念可能推广到更广泛的模型组合/集成场景

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] Near-Exponential Savings for Mean Estimation with Active Learning](near-exponential_savings_for_mean_estimation_with_active_learning.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](../../ICML2025/optimization/learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICML 2025\] Understanding the Statistical Accuracy-Communication Trade-off in Personalized Federated Learning with Minimax Guarantees](../../ICML2025/optimization/understanding_the_statistical_accuracy-communication_trade-off_in_personalized_f.md)

</div>

<!-- RELATED:END -->
