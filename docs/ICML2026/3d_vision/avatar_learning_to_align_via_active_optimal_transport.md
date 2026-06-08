---
title: >-
  [论文解读] AvAtar: Learning to Align via Active Optimal Transport
description: >-
  [ICML 2026][3D视觉][最优传输] 本文提出 AvAtar，一个基于最优传输（OT）的主动对齐框架，通过梯度传播量化候选查询对全局对齐结果的影响，并利用伴随状态法和共轭梯度法以线性复杂度高效求解，在网络对齐和跨域对齐任务上一致超越已有主动学习策略。
tags:
  - "ICML 2026"
  - "3D视觉"
  - "最优传输"
  - "主动学习"
  - "对齐"
  - "梯度传播"
  - "共轭梯度法"
---

# AvAtar: Learning to Align via Active Optimal Transport

**会议**: ICML 2026  
**arXiv**: [2605.24395](https://arxiv.org/abs/2605.24395)  
**代码**: 无  
**领域**: 机器学习理论  
**关键词**: 最优传输, 主动学习, 对齐, 梯度传播, 共轭梯度法  

## 一句话总结

本文提出 AvAtar，一个基于最优传输（OT）的主动对齐框架，通过梯度传播量化候选查询对全局对齐结果的影响，并利用伴随状态法和共轭梯度法以线性复杂度高效求解，在网络对齐和跨域对齐任务上一致超越已有主动学习策略。

## 研究背景与动机

**领域现状**：对齐问题（alignment）是许多机器学习任务的核心步骤，包括多网络分析、多模态学习和点云配准等。近年来，最优传输（OT）因其能从全局视角推断分布之间的软对应关系而被广泛用于对齐问题。通过将两组对象关联到离散概率分布并求解传输计划 $\mathbf{T}$，OT 方法在多种对齐任务中展现了卓越的性能。

**现有痛点**：OT 方法对监督信号的数量和质量高度敏感——实验显示，增加监督可带来高达 15% 的性能提升，而不同的查询策略在相同预算下性能差距可达 7%。然而，高质量监督标注在实际中非常昂贵（如人工标注跨网络节点对应关系），目前几乎没有工作研究如何在 OT 框架下主动获取高质量监督。

**核心矛盾**：现有主动对齐方法存在三个关键局限：（1）未针对 OT 设计，无法利用代价函数和边际约束等 OT 核心组件；（2）缺乏原则性方法量化新监督如何通过 OT 公式传播并影响对齐结果；（3）策略通常针对特定任务（如网络对齐）设计，不易泛化到跨域对齐等其他任务。

**本文目标**：设计一个通用的主动学习框架，能在固定查询预算下最大化提升 OT 方法的对齐性能，且适用于网络对齐、图文检索、图文定位等多种任务。

**切入角度**：作者观察到，可以用效用函数 $f(\mathbf{T})$ 将传输计划编码为标量来衡量全局对齐质量，然后通过梯度传播计算每个候选查询对该效用函数的影响——梯度越大，说明该查询对全局对齐改善越有价值。

**核心 idea**：通过伴随状态法将"对 OT 求导"这一高难度问题转化为一个 $(n+m)$ 维线性系统，用共轭梯度法以线性复杂度和保证收敛的方式求解，从而高效量化每个候选查询的信息量。

## 方法详解

### 整体框架

AvAtar 要解决的问题是：在固定的查询预算下，挑出最值得人工标注的那几个对象，让 OT 对齐器的全局结果提升最多。它把一个现成的 OT 对齐方法（带代价函数 $\mathbf{C}$、边际分布 $\boldsymbol{\mu}, \boldsymbol{\nu}$）当成可微的黑盒，外面套一层主动学习循环。每一轮做四件事：先为每个候选对象算出"查询它能让全局对齐改善多少"的影响分数，挑出分数最高的 $n_b$ 个去问 oracle 拿真实对应，把答案写进监督矩阵 $\mathbf{H}$，再重跑 OT 求解器得到更新后的传输计划 $\mathbf{T}$，如此循环直到用完预算 $k$。整套方法的难点和创新都集中在第一步——如何在不知道真实标签时，原则性地量化"查询某个对象"对全局结果的边际价值。

### 关键设计

**1. 用伴随状态法把"对 OT 求导"变成可解的线性系统**

衡量查询价值的出发点是一个效用函数 $f(\mathbf{T})$，它把整张传输计划压成一个标量来刻画全局对齐质量。一个查询的价值，自然就是它对 $f$ 的影响有多大——即效用函数对监督信号 $\mathbf{H}_{i,j}$ 的梯度。通过链式法则可以把这个梯度拆成两段：$\nabla_{\mathbf{H}_{i,j}} f = \langle \nabla_{\tilde{\mathbf{C}}} f,\ \nabla_{\mathbf{H}_{i,j}} \tilde{\mathbf{C}} \rangle$。后一段是监督信号如何改写代价矩阵，可以直接写成 $-\beta \mathbf{C}_{i,j} \mathbf{E}$；真正的拦路虎是前一段 $\nabla_{\tilde{\mathbf{C}}} f$，因为传输计划 $\mathbf{T}$ 是代价矩阵的隐函数，要对它求导，朴素做法得显式构造并求逆一个 $(nm)^2$ 大小的 Jacobian 矩阵，规模大到不可行。

作者借用 PDE 约束优化里的伴随状态法绕开这一步：不去显式求那个巨型 Jacobian，而是把问题等价转化为求解一个只有 $(n+m)$ 维的伴随线性系统 $\mathbf{A}\mathbf{y} = \mathbf{b}$，其中系数矩阵 $\mathbf{A}$ 由边际分布和传输计划拼成。这个系统是奇异的（$\mathbf{A}$ 不可逆），但好在 $\mathbf{A}$ 半正定、且 $\mathbf{b}$ 恰好落在 $\mathbf{A}$ 的列空间里，于是可以用共轭梯度（CG）法保证收敛到全局最优解；再叠加 $\mathbf{T}$ 的稀疏结构，整体求解只需线性时间复杂度 $\mathcal{O}(K(n+m))$。这正是 AvAtar 相比已有方法的根本区别：别人没法系统地刻画新监督怎样在 OT 公式里传播，而它把这件事变成了一个可证明收敛、线性可解的标准问题。

**2. 用传输计划做权重，把成对影响聚合成对象级查询价值**

上一步算出的是成对影响 $\mathcal{I}(p_{ij})$——查询"$x_i$ 对应 $y_j$"这件事的价值，但实际查询时只能选源对象 $x_i$、并不知道它真正对应谁。AvAtar 的做法是把 $x_i$ 与所有目标对象的成对影响按传输计划加权求和：

$$\mathcal{I}(p_i) = \sum_{j=1}^{m} \mathbf{T}_{i,j}\, \mathcal{I}(p_{ij})$$

这里的巧妙之处在于 $\mathbf{T}_{i,j}$ 本身就是 $x_i$ 与 $y_j$ 的后验对齐概率，因此用它当权重，数学上恰好等于在算查询 $x_i$ 的期望影响——在不知道真实标签时，直接用 OT 自己输出的后验当先验，既自洽又充分复用了求解结果。相比之下，若改用均匀权重盲目平均，MRR 会掉 16.8%（Douban），说明后验加权确实抓住了有用的信号。

**3. 可替换的效用函数，让同一套流程泛化到不同任务**

效用函数是 AvAtar 适配不同任务的唯一开关，框架其余部分完全不变。作者给出三种：通用的 $f_{L_2} = \|\mathbf{T}\|_2^2$ 和 $f_{\text{entropy}} = \|\mathbf{T} \odot \log(\mathbf{T})\|_1$，二者都鼓励传输计划往确定性的排列矩阵靠拢，适合大多数检索/定位类对齐；以及网络对齐专用的 $f_{\text{consist}}$，借图拉普拉斯矩阵衡量对齐后的邻居一致性。换任务时只需换一个效用函数，前面的梯度计算和聚合流程原样复用，无需为每个任务重新设计查询策略——这正是它能同时打通网络对齐、图文检索、图文定位的关键。

### 训练策略

AvAtar 采用批次迭代查询：每轮挑出 $n_b$ 个候选问 oracle，更新 $\mathbf{H}$ 后重跑 OT 求解器，直到耗尽预算 $k$。其中共轭梯度法的收敛速率为 $\frac{\sqrt{\lambda_1/\lambda_r} - 1}{\sqrt{\lambda_1/\lambda_r} + 1}$（$\lambda_1, \lambda_r$ 为系数矩阵的最大与最小非零特征值），实验显示 CG 比 Sinkhorn 收敛更快，且对正则化权重 $\epsilon$ 更鲁棒。

## 实验关键数据

### 主实验 — 网络对齐 (MRR)

| 数据集 | 方法 | Query=5 | Query=10 | 对比 Random 提升 |
|--------|------|---------|----------|------------------|
| Phone-Email | Random (JOENA) | 0.582 | 0.648 | — |
| Phone-Email | AvAtar-$L_2$ (JOENA) | 0.682 | 0.800 | +15.2% |
| Phone-Email | AvAtar-consist (JOENA) | **0.691** | **0.806** | +15.8% |
| ACM-DBLP-A | Random (JOENA) | 0.821 | 0.837 | — |
| ACM-DBLP-A | AvAtar-$L_2$ (JOENA) | **0.924** | **0.981** | +14.4% |
| Douban | Random (PARROT) | 0.730 | 0.751 | — |
| Douban | AvAtar-$L_2$ (PARROT) | **0.782** | **0.837** | +8.6% |

### 主实验 — 跨域对齐 (Recall@1)

| 任务 | 数据集 | 方法 | Query=5 | Query=10 |
|------|--------|------|---------|----------|
| 图文检索 | ImageNet-C (GOT-W) | Random | 0.374 | 0.454 |
| 图文检索 | ImageNet-C (GOT-W) | AvAtar-entropy | **0.402** | **0.509** |
| 图文定位 | Flickr30K (GOT-FGW) | Random | 0.550 | 0.628 |
| 图文定位 | Flickr30K (GOT-FGW) | AvAtar-$L_2$ | **0.575** | **0.671** |
| 图文定位 | COCO (GOT-FGW) | Random | 0.545 | 0.607 |
| 图文定位 | COCO (GOT-FGW) | AvAtar-$L_2$ | **0.582** | **0.678** |

### 消融实验

| 消融维度 | 配置 | MRR (Douban) | 说明 |
|----------|------|-------------|------|
| 稀疏 vs 稠密 | AvAtar-$L_2$ (Sparse) | 0.837 | 5.1s，8.6× 加速 |
| 稀疏 vs 稠密 | AvAtar-$L_2$ (Dense) | 0.839 | 19s，性能相当 |
| 聚合方式 | AvAtar-$L_2$ (后验) | 0.837 | 传输计划加权 |
| 聚合方式 | AvAtar-$L_2$ (均匀) | 0.669 | 均匀聚合，MRR 降 16.8% |
| 效用函数 | AvAtar-consist (PARROT) | 0.835 | 无属性网络更优 |
| 效用函数 | AvAtar-$L_2$ (PARROT) | 0.837 | 有属性网络更优 |

## 亮点与洞察

- 将"对 OT 求导"这一关键技术瓶颈优雅地转化为伴随线性系统求解，避免了 $(nm)^2$ 的 Jacobian 求逆，这一思路可推广到其他需要对约束优化求导的场景
- 后验加权聚合的设计非常巧妙——传输计划 $\mathbf{T}$ 本身就编码了对齐的后验概率，用它来加权不同成对影响在数学上就是在计算条件期望
- 框架的通用性强：只需更换效用函数即可适配不同任务，核心梯度计算流程完全复用
- 实验覆盖 8 个数据集、4 种 OT 方法、9 种基线策略，在三大类对齐任务上一致 SOTA，说服力强

## 局限性 / 可改进方向

- 实验中的跨域对齐任务（图文检索/定位）使用的数据集较小，未在大规模多模态基准上验证
- 效用函数的选择仍需领域知识指导，如何自动选择或学习效用函数是未来方向
- 框架依赖熵正则化 OT 的可微性，对非熵正则化或 unbalanced OT 的扩展尚未讨论
- 批次选择策略为贪心的 top-$n_b$，未考虑候选之间的多样性或冗余性

## 相关工作与启发

- **PARROT / JOENA**：两种代表性 OT 网络对齐方法，本文作为主动学习层直接叠加在其上
- **GOT**：基于 Wasserstein 和 Gromov-Wasserstein 距离的跨域对齐框架，本文在其上验证了跨域主动对齐
- **伴随状态法**：源自 PDE 约束优化和神经 ODE 的高效求导技术，本文首次将其应用于 OT 对齐中的梯度传播
- 对需要处理稀疏标注对齐问题的研究者有直接启发：与其随机标注，不如用梯度指导选择最有价值的标注对象

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次形式化 OT 主动对齐问题并给出原则性解法
- 实验充分度: ⭐⭐⭐⭐⭐ — 8 数据集 × 4 OT方法 × 9 基线，三大任务全覆盖
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，符号统一，论文结构完整
- 价值: ⭐⭐⭐⭐ — 通用框架可直接应用于多种 OT 对齐场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)
- [\[ICML 2026\] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation](convex_distance_operator_transport_a_convex_and_geometry-preserving_formulation.md)
- [\[ICLR 2026\] SceneTransporter: Optimal Transport-Guided Compositional Latent Diffusion for Single-Image Structured 3D Scene Generation](../../ICLR2026/3d_vision/scenetransporter_optimal_transport-guided_compositional_latent_diffusion_for_sin.md)
- [\[CVPR 2025\] NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary](../../CVPR2025/3d_vision/nopain_no-box_point_cloud_attack_via_optimal_transport_singular_boundary.md)
- [\[ICML 2026\] PLAID: A Unified Data Model for Machine Learning on Heterogeneous Physics Simulations](plaid_a_unified_data_model_for_machine_learning_on_heterogeneous_physics_simulat.md)

</div>

<!-- RELATED:END -->
