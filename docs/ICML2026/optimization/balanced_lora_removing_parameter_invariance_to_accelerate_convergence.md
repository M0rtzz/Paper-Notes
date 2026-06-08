---
title: >-
  [论文解读] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence
description: >-
  [ICML2026][优化/理论][LoRA微调] 本文揭示了 LoRA 的过参数化导致不同低秩因子对 $(A,B)$ 具有不同条件数，证明了**平衡最小值点**（$A^\top A = BB^\top$）具有最优条件数，并据此提出 BaLoRA——在每步优化后将适配器投影到平衡流形上…
tags:
  - "ICML2026"
  - "优化/理论"
  - "LoRA微调"
  - "参数不变性"
  - "条件数优化"
  - "平衡流形投影"
  - "收敛加速"
---

# Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence

**会议**: ICML2026  
**arXiv**: [2605.31484](https://arxiv.org/abs/2605.31484)  
**代码**: https://github.com/vcastin/balora  
**领域**: optimization  
**关键词**: LoRA微调, 参数不变性, 条件数优化, 平衡流形投影, 收敛加速  

## 一句话总结

本文揭示了 LoRA 的过参数化导致不同低秩因子对 $(A,B)$ 具有不同条件数，证明了**平衡最小值点**（$A^\top A = BB^\top$）具有最优条件数，并据此提出 BaLoRA——在每步优化后将适配器投影到平衡流形上，以几乎零开销加速收敛并提升微调性能。

## 研究背景与动机

**领域现状**：LoRA 是目前大语言模型参数高效微调（PEFT）的主流方法，通过低秩矩阵 $A \in \mathbb{R}^{a \times r}$ 和 $B \in \mathbb{R}^{r \times b}$ 的乘积 $AB$ 来近似权重更新，将可训练参数从 $ab$ 降至 $r(a+b)$。

**现有痛点**：LoRA 存在固有的过参数化问题——对于任意可逆矩阵 $R$，$(AR, R^{-1}B)$ 与 $(A,B)$ 产生完全相同的适配矩阵 $AB$。这意味着损失函数的最小值不是孤立点，而是一个 $r^2$ 维的连续流形。现有工作（LoRA+、OLoRA 等）主要从初始化或学习率角度改进，未能从根本上解决过参数化带来的优化低效问题。

**核心矛盾**：同一适配矩阵 $AB$ 的不同因子分解 $(A,B)$ 具有**截然不同的条件数**，导致梯度下降收敛到不同最小值点时，渐近收敛速率差异显著。条件数差的最小值点对应更陡峭的损失曲面，优化器在其附近振荡严重。

**切入角度**：作者从 Hessian 矩阵的谱分析出发，发现最小值点 $(A,B)$ 的条件数完全由 $A$ 和 $B$ 的奇异值决定。当 $A^\top A = BB^\top$（即"平衡条件"）成立时，两个因子的奇异值完全对齐，条件数达到理论最优。

**核心 idea**：在每步优化后将 $(A,B)$ 投影到平衡流形上，以 $\mathcal{O}((a+b)r^2)$ 的轻量计算代价换取最优条件数，从而加速渐近收敛。

## 方法详解

### 整体框架

BaLoRA 的流程极其简洁：在标准 LoRA 的每一步优化器更新（如 AdamW）之后，额外执行一步**平衡投影** $P(A,B)$，将低秩因子映射到"超平衡流形" $\mathcal{H}$ 上。投影保持乘积 $AB$ 不变（因此损失值不变），但改变因子分解方式使条件数最优。输入和输出与标准 LoRA 完全一致，可无缝集成到现有训练 pipeline。

### 关键设计

**1. 平衡投影算子 $P(A,B)$：保持 $AB$ 不变，只换一个条件数最优的分解。**

LoRA 的痛点是同一个适配矩阵 $AB$ 有无数种因子分解，而梯度下降会随机落到其中某一个、条件数可能很差。BaLoRA 的做法是每步优化后做一次投影，把 $(A,B)$ 映射到"超平衡流形" $\mathcal{H}=\{(US^{1/2}, S^{1/2}V)\mid U^\top U=VV^\top=I_r,\, S\in\mathbb{D}_+^r\}$ 上，同时严格保持乘积 $AB$ 不变（所以损失值一点不动，只改分解形式）。算法走的是分解技巧：先对 $A$、$B$ 分别做极分解 $A=R_A S_A$、$B=S_B R_B$，再对 $S=S_A S_B$ 做 SVD 得 $S=U\Sigma V^\top$，最后输出 $A^{\text{proj}}=R_A U\Sigma^{1/2}$、$B^{\text{proj}}=\Sigma^{1/2}V^\top R_B$。这样设计是为了把代价压到底——如果直接对 $AB$ 做 SVD 要 $\mathcal{O}(abr)$，而这里的 SVD 只在 $r\times r$ 的小矩阵上跑，整步投影仅 $\mathcal{O}((a+b)r^2)$，相对优化器更新可忽略不计，因此能"几乎零开销"地嵌进任意现有 LoRA 训练流程。

**2. 最优条件数的理论保证：证明平衡点 $A^\top A=BB^\top$ 在所有等价分解里条件数最小。**

投影投到哪不是拍脑袋，而是从 Hessian 谱分析推出来的。对矩阵分解情形（$\text{rk}(Z)=r$），最小值点处的 Hessian 特征值为 $\sigma_i(A)^2+\sigma_j(B)^2$，对应条件数

$$\kappa=\frac{\sigma_1(A)^2+\sigma_1(B)^2}{\min\!\big(\sigma_r(A)^2,\,\sigma_r(B)^2\big)}.$$

当平衡条件 $A^\top A=BB^\top$ 成立时，两个因子的奇异值完全对齐 $\sigma_i(A)=\sigma_i(B)=\sigma_i(Z)^{1/2}$，条件数被压到理论下界 $\kappa_{\min}=2\sigma_1(Z)/\sigma_r(Z)$；对更一般的 $\text{rk}(Z)\ge r$ 情形，关键量则变成 $r$-谱隙 $\sigma_r(Z)-\sigma_{r+1}(Z)$。这套分析直接解释了一个反直觉现象——同一个 $AB$、不同分解为什么会训出不同速度：条件数差的分解对应更陡峭的损失曲面，优化器在其附近振荡严重；于是"投影到平衡点"就成了选最优分解的原则性方法，而不再是经验调参。

**3. Bures 度量下的内蕴几何解释：BaLoRA 其实是低秩流形上的自然梯度下降。**

作者再退一步给出一个更优雅的视角：把 BaLoRA-GD 重新表述为秩-$r$ 矩阵流形 $\mathcal{N}_r$ 上、关于 Bures 度量的黎曼梯度下降。定义逆 Bures 度量 $H_X[W]=(XX^\top)^{1/2}W+W(X^\top X)^{1/2}$，迭代就能写成 $X_{k+1}=R(X_k,-\tau_k\Delta_k)$，其中 $\Delta_k=H_{X_k}[\nabla g(X_k)]$ 是黎曼梯度、$R$ 是流形上的收缩映射。这个视角说明 BaLoRA 本质是在低秩矩阵流形上做自然梯度下降，因子化的 $(A,B)$ 只是它的高效实现手段；它顺带把 LoRA 优化和最优传输里的 Bures–Wasserstein 几何接上了线，为后续理论分析开了一个新入口。

## 实验关键数据

### 主实验：多数据集微调对比（Qwen-2.5-3B, r=8）

| 方法 | Alpaca | CodeFeedback | OpenHermes | OpenOrca | WizardLM |
|------|--------|-------------|------------|----------|----------|
| LoRA | 1.352 | 0.638 | 0.707 | 0.774 | 0.663 |
| DoRA | 1.352 | 0.639 | 0.707 | 0.776 | 0.662 |
| LoRA-RITE | 1.353 | 0.639 | 0.707 | 0.776 | 0.663 |
| LORO | 1.504 | 0.669 | 0.750 | 0.859 | 0.689 |
| OLoRA | 1.360 | 0.641 | 0.712 | 0.782 | 0.666 |
| RefLoRA | 1.350 | 0.638 | 0.706 | 0.773 | 0.661 |
| **BaLoRA** | **1.350** | **0.638** | 0.707 | **0.773** | 0.662 |

BaLoRA 与 RefLoRA（另一平衡方法）稳居前两名，验证了平衡约束对收敛加速的有效性。

### Rank 消融实验（Qwen-2.5-3B, DM Mathematics 1B tokens）

| 方法 | r=8 | r=16 | r=32 | r=64 | r=128 |
|------|-----|------|------|------|-------|
| LoRA | 1.035 | 1.032 | 1.031 | 1.030 | 1.030 |
| DoRA | 1.035 | 1.032 | 1.031 | 1.030 | 1.030 |
| LoRA-RITE | 1.047 | 1.045 | 1.046 | 1.052 | 1.069 |
| OLoRA | 1.039 | 1.037 | 1.036 | 1.036 | 1.036 |
| RefLoRA | 1.027 | 1.023 | 1.024 | 1.027 | 1.032 |
| **BaLoRA** | **1.026** | **1.020** | **1.017** | **1.015** | **1.014** |

BaLoRA 在**高秩**（r=64/128）时优势尤为显著：RefLoRA 在高秩时性能退化，而 BaLoRA 持续改善，r=128 时领先 LoRA 约 1.5%、领先 RefLoRA 约 1.8%。

## 亮点与洞察

- **理论-实践闭环**：从 Hessian 谱分析推导出平衡条件最优 → 设计轻量投影算子 → 实验验证收敛加速，逻辑链完整
- **高秩场景的独特优势**：当 r 增大时，过参数化的不变性维度（$r^2$）增长更快，BaLoRA 的条件数改善效果更加显著
- **超参稳健性**：BaLoRA 对学习率和初始化 scaling 的敏感度明显低于 LoRA/OLoRA/LoRA-GA，实际使用时调参更容易
- **Bures 度量连接**：将 LoRA 优化与最优传输中的 Bures-Wasserstein 几何联系起来，为后续理论分析开辟了新视角

## 局限性 / 可改进方向

- 理论分析主要针对**单层适配器**和**回归损失**，多层同时微调和交叉熵损失的条件数分析尚未完成
- 当前投影保持 $AB$ 不变但会改变 Adam 优化器的动量/方差状态，可能在初期引入短暂的训练震荡（合成实验中 BaLoRA 起步略慢）
- 未与 GaLore 等非 LoRA 范式的 PEFT 方法对比
- 投影步骤虽然轻量，但在 $r$ 较大时极分解和 SVD 仍有一定开销，对于 r=128 以上场景值得进一步优化

## 相关工作与启发

- **RefLoRA**（Zhang et al., 2025）同样强制平衡，但使用不同的平衡映射且需要 100 步 warmup；BaLoRA 的投影更简洁且无需预热
- **LORO**（Mo et al., 2025）从黎曼优化角度出发但需要专门求解器；BaLoRA 通过后投影方式兼容任意优化器
- **LoRA+**（Hayou et al., 2024）通过不同学习率改善 A/B 的训练动态；可与 BaLoRA 的平衡投影正交结合

## 评分

- 新颖性: 9/10 — 首次从条件数角度建立平衡因子与最优收敛速率的理论联系
- 实验充分度: 8/10 — 覆盖多模型/多数据集/多秩度，但缺少下游任务准确率评估
- 写作质量: 9/10 — 理论推导清晰，几何直觉阐述到位
- 价值: 8/10 — 实用性强且理论优美，但改进幅度在小秩场景有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](tpv_parameter_perturbations_through_the_lens_of_test_prediction_variance.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[ICML 2026\] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](../../CVPR2026/optimization/label-free_cross-task_lora_merging_with_null-space_compression.md)

</div>

<!-- RELATED:END -->
