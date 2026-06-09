---
title: >-
  [论文解读] Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets
description: >-
  [NeurIPS 2025][优化/理论][代数结构] 提出 CoGS 框架，证明二层二次激活网络在 Abelian 群乘法推理任务上的权重空间具有半环代数结构，损失函数中的 Sum Potential 是环同态映射，由此可从仅满足部分损失的局部解通过环加法和环乘法代数地组合出全局最优解…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "代数结构"
  - "模算术推理"
  - "二层网络"
  - "Grokking"
  - "权重空间半环"
  - "全局最优解构造"
---

# Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets

**会议**: NeurIPS 2025  
**arXiv**: [2410.01779](https://arxiv.org/abs/2410.01779)  
**代码**: [facebookresearch/luckmatters](https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset)  
**领域**: 优化  
**关键词**: 代数结构, 模算术推理, 二层网络, Grokking, 权重空间半环, 全局最优解构造  

## 一句话总结

提出 CoGS 框架，证明二层二次激活网络在 Abelian 群乘法推理任务上的权重空间具有半环代数结构，损失函数中的 Sum Potential 是环同态映射，由此可从仅满足部分损失的局部解通过环加法和环乘法代数地组合出全局最优解，约 95% 的梯度下降解与理论构造精确匹配。

## 研究背景与动机

- **LLM 推理能力仍存疑**：大型语言模型在复杂推理任务上表现出色但在基础算术推理中仍犯令人意外的错误，理解模型如何执行推理是核心问题
- **模算术加法是经典研究对象**：modular addition（预测 $a+b \bmod d$）因结构简洁但训练动态丰富（如 grokking 现象、Fourier 基表示）而成为广泛采用的研究基准
- **现有解析解缺乏系统框架**：前人虽能构造或逆向工程 Fourier 解（Gromov 2023, Nanda 2023），但依赖无穷宽度近似，缺少系统化的代数构造方法
- **权重空间的代数结构尚未被探索**：几何深度学习利用数据的对称性，但从未打开黑盒研究训练过程中网络权重空间本身的代数结构
- **全局最优解的构造高度非线性**：直接从损失函数的充分条件求解因高度非线性而极其困难，需要新的数学工具
- **过参数化和权重衰减的理论机制不清**：经验上知道过参数化有利于训练而权重衰减偏好简单解，但缺乏严格的理论解释

## 方法详解

### 整体框架：CoGS（Composing Global Solutions）

CoGS 的核心思想是：不直接求解高度非线性的全局优化问题，而是**将全局解分解为多个仅满足部分损失约束的局部解，再通过代数运算组合成全局解**。技术路线分三步：(1) 证明权重空间具有半环结构；(2) 证明损失函数由 Sum Potential 构成且它们是环同态；(3) 利用环同态性质从低阶局部解组合全局解。

### 关键设计一：权重空间的半环结构

- **功能**：在二层网络权重空间 $\mathcal{Z} = \bigcup_{q \geq 0} \mathcal{Z}_q$（$q$ 为隐藏节点数）上定义加法（沿隐藏维拼接）和乘法（Khatri-Rao 积），证明 $\langle \mathcal{Z}, +, * \rangle$ 构成交换半环
- **核心思路**：加法对应拼接不同宽度的网络，乘法对应隐藏节点的外积扩展。加法满足交换律（通过排列等价），乘法满足结合律和分配律
- **设计动机**：半环结构使得可以像处理多项式一样在权重空间中构造解——用低阶"生成元"经由环运算组合出高阶全局解

### 关键设计二：Sum Potential 与环同态

- **功能**：将 $L_2$ 损失解析分解为 $\ell = d^{-1} \sum_{k \neq 0} \ell_k$，其中每个 $\ell_k$ 由 Sum Potential（SP）$r_{k_1 k_2 k}(\mathbf{z})$ 和 $r_{p k_1 k_2 k}(\mathbf{z})$ 构成，并证明所有 SP 都是环同态映射
- **核心思路**：SP 定义为 $r(\mathbf{z}) := \sum_j \prod_{p,k \in \text{idx}(r)} z_{pkj}$，是对隐藏节点求和的单项式。环同态性质意味着 $r(\mathbf{z}_1 + \mathbf{z}_2) = r(\mathbf{z}_1) + r(\mathbf{z}_2)$，$r(\mathbf{z}_1 * \mathbf{z}_2) = r(\mathbf{z}_1) \cdot r(\mathbf{z}_2)$
- **设计动机**：环同态保证了局部解的 0/1-set 可以通过加法取交集、乘法取并集来组合。使得 0-set 逐步扩大直到覆盖所有约束，从而得到全局解

### 关键设计三：多项式构造与全局解组合

- **功能**：选取对称性较好的 order-1 生成元 $\mathbf{u}$（如 $\mathbf{u}_{\text{syn}}, \mathbf{u}_\nu$），构造多项式 $\boldsymbol{\rho}(\mathbf{u}) = \prod_{s}(\mathbf{u} + \hat{\mathbf{s}})$ 作为局部解，再组合得到 order-4（$2 \times 2$）和 order-6（$2 \times 3$）的 Fourier 全局解
- **核心思路**：order-6 解 $\mathbf{z}_{F6}$ 将每个频率的 3-阶和 2-阶局部解做环乘取并集覆盖所有 SP 约束；混合 order-4/6 解 $\mathbf{z}_{F4/6}$ 进一步降低总阶数到 $2d$，利用不同频率间的 SP 对消实现全局最优
- **设计动机**：梯度下降天然偏好低阶解（Theorem 5），因此需要尽量构造低阶全局解来解释实验观测

### 关键设计四：训练动态分析

- **功能**：证明高阶和低阶全局解在零损失流形上拓扑连通（Theorem 5），说明权重衰减下梯度下降偏好低阶解；证明无穷宽度下 SP 动态解耦（Theorem 6），解释过参数化的益处
- **核心思路**：若 $\mathbf{z} = \mathbf{y} * \mathbf{z}'$ 且两者都是全局解，则存在连接两者的零损失路径，权重衰减的正则化将动态推向低阶端
- **设计动机**：回答"为什么梯度下降不收敛到 $d^2$ 阶的完美记忆解而是 4/6 阶的 Fourier 解"这一关键问题

## 损失函数与训练策略

- **损失函数**：标准 $L_2$ 均方误差损失，对 Abelian 群乘法预测结果做零均值投影后计算
- **优化器**：Adam，学习率 0.01
- **正则化**：$L_2$ 权重衰减（实验中测试了 $5 \times 10^{-5}$ 等多种设置）
- **训练轮数**：10000 epochs
- **数据划分**：90% 训练 / 10% 测试，全部合成生成

## 实验关键数据

### 表1：梯度下降解与 CoGS 理论构造的匹配率（$q=512$，权重衰减 $5 \times 10^{-5}$）

| 群大小 $d$ | 不可分解比例 (%) | 分解误差 ($\times 10^{-2}$) | Order-4 占比 (%) | Order-6 占比 (%) | $\mathbf{z}_{\nu=i} * \mathbf{z}_\xi$ (%) | $\mathbf{z}_\nu * \mathbf{z}_{\text{syn}}$ (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 23 | 0.0 | 0.05 | 47.07 | 39.80 | 47.07 | 39.80 |
| 71 | 0.0 | 0.03 | 72.57 | 21.14 | 72.57 | 21.14 |
| 127 | 1.50 | 0.26 | 82.96 | 14.13 | 82.96 | 14.13 |

> 约 95% 的梯度下降解可分解且与理论构造精确匹配。$d$ 越大，order-4 解占比越高，符合 $\mathbf{z}_{F4/6}$ 混合解理论预测。

### 表2：不同权重衰减强度对解分布的影响（$d \in \{23, 71, 127\}$, $q=512$）

| 设置 | 效果 |
|:--|:--|
| 权重衰减增大 | 解分布向低阶（order-4）偏移，直至模型坍缩（权重归零） |
| 即使大幅过参数化 ($q=512$ vs 最少需 $\sim 2d$ 节点) | 最终解阶数仍恒定为 4/6，不随网络宽度增长 |
| 去除 $R_*$ 约束项 | 每频率只需 3 个隐藏节点（order-3 $\mathbf{z}_{\text{syn}}$ 解） |
| Grokking 现象 | 训练准确率先达 100%，测试准确率在数千轮后突跳至 100% |

## 亮点与洞察

1. **开创性代数视角**：首次发现并严格证明神经网络权重空间在训练中具备半环代数结构，将非线性优化问题转化为代数组合问题
2. **理论与实践高度吻合**：约 95% 的梯度下降实际解与代数构造精确匹配，验证了理论的实用性
3. **简洁解释复杂现象**：用环同态和拓扑连通性优雅地解释了过参数化的好处、权重衰减偏好简单解、以及 grokking 现象的部分机制
4. **提出全新训练范式可能性**：CoGS 暗示可以绕开梯度下降，直接通过代数分解和组合构造解，可能更高效

## 局限性

1. **仅限二次激活**：理论严格成立需要 $\sigma(x)=x^2$，对 ReLU/SiLU 等实际激活仅能通过 Taylor 展开近似
2. **仅限二层网络**：未扩展到多层网络或 Transformer 架构
3. **仅限 Abelian 群**：非交换群（如置换群）的推理任务未覆盖
4. **未完全解释 grokking**：虽然提供了部分洞察但并未直接建模非均匀训练集下的动态
5. **实验规模较小**：$d$ 最大仅测到 127，未验证更大规模下理论预测是否成立

## 相关工作与启发

- **Gromov (2023)** 构造了模算术的 Fourier 解析解，但依赖无穷宽度近似；本文的构造更简洁且在有限宽度下精确
- **Morwani et al. (2023)** 使用 max-margin 框架和 $L_{2,3}$ 范数分析代数任务，但未发现权重空间的代数结构也未分析训练动态
- **Nanda et al. (2023)** 通过机械可解释性提取电路，是自下而上的经验方法；本文是自上而下的理论构造
- **启发**：这种"通过代数结构将非线性优化分解为可组合子问题"的思路可能推广到更广泛的表示学习理论中

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次在神经网络训练中发现并利用半环代数结构，视角全新
- **实验充分度**: ⭐⭐⭐⭐ — 多种 $d$ 值、多种权重衰减、95%匹配率验证充分，但规模和架构覆盖有限
- **写作质量**: ⭐⭐⭐⭐ — 数学严谨，图表清晰，但代数符号密集对非专业读者门槛较高
- **价值**: ⭐⭐⭐⭐⭐ — 为理解神经网络推理机制提供了全新的代数理论工具，可能开启新的研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)
- [\[ICML 2025\] Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization](../../ICML2025/optimization/global_convergence_and_rich_feature_learning_in_l-layer_infinite-width_neural_ne.md)
- [\[NeurIPS 2025\] Multi-head Transformers Provably Learn Symbolic Multi-step Reasoning via Gradient Descent](multi-head_transformers_provably_learn_symbolic_multi-step_reasoning_via_gradien.md)

</div>

<!-- RELATED:END -->
