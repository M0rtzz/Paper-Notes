---
title: >-
  [论文解读] Why Is Attention Sparse in Particle Transformer?
description: >-
  [NeurIPS 2025][Transformer] 本文系统性地分析了 Particle Transformer（ParT）在 jet tagging 任务中训练后出现的近乎二值化稀疏 attention 现象，通过跨数据集对比和消融实验揭示了稀疏性主要源自 attention 机制自身而非物理启发的 interaction 矩阵，但 interaction 矩阵通过影响绝大多数 token 的 argmax 选择对最终性能不可或缺。
tags:
  - NeurIPS 2025
  - Transformer
  - 注意力机制
  - jet tagging
  - interaction matrix
  - interpretability
---

# Why Is Attention Sparse in Particle Transformer?

**会议**: NeurIPS 2025  
**arXiv**: [2512.00210](https://arxiv.org/abs/2512.00210)  
**代码**: [有](https://github.com/)  
**领域**: 可解释性 / 物理AI / Transformer  
**关键词**: Particle Transformer, sparse attention, jet tagging, interaction matrix, interpretability

## 一句话总结

本文系统性地分析了 Particle Transformer（ParT）在 jet tagging 任务中训练后出现的近乎二值化稀疏 attention 现象，通过跨数据集对比和消融实验揭示了稀疏性主要源自 attention 机制自身而非物理启发的 interaction 矩阵，但 interaction 矩阵通过影响绝大多数 token 的 argmax 选择对最终性能不可或缺。

## 研究背景与动机

**领域现状**：在 CERN 大型强子对撞机（LHC）的高能物理实验中，jet tagging——即对高能碰撞产生的粒子准直流（jet）进行分类——是核心分析任务之一。近年来，基于深度学习的 jet tagger 取得了显著进展，从图神经网络 ParticleNet 到基于 Transformer 的模型，分类精度不断提升。其中，Particle Transformer（ParT）是当前最先进的 jet tagger 之一，它在标准多头注意力机制的基础上引入了物理启发的粒子对交互矩阵（interaction matrix），将粒子对之间的运动学关系（如角距离 $\Delta$、横动量 $k_T$、动量分数 $z$、不变质量 $m^2$ 等）编码为注意力偏置项，从而融合了领域先验知识和数据驱动的表示学习。

**现有痛点**：尽管 ParT 在多个基准数据集上取得了 SOTA 性能，研究者发现其训练后的 attention 图呈现出一个令人困惑的现象——attention 权重近乎二值化（接近 0 或接近 1），每个粒子在每个注意力头中几乎只关注另外一个粒子。这种极端的稀疏模式在自然语言处理或计算机视觉领域的 Transformer 中非常罕见。先前的工作（Wang et al.）已经观察到了这一现象，并在 $\eta$-$\phi$ 平面上可视化了 attention map，发现 ParT 似乎能捕捉到有物理意义的 jet 子结构（如半轻子衰变中的轻子），但对这种稀疏性的来源和机制缺乏系统性解释。

**核心矛盾**：ParT 的注意力计算中包含两个成分——传统的 $QK^T/\sqrt{d_k}$ attention 项和物理启发的 interaction 矩阵 $U$。两者在 softmax 之前相加。那么问题来了：这种极端的二值化稀疏行为究竟是由哪个成分主导的？是 attention 机制自发学到的，还是 interaction 矩阵的物理先验导致的？进一步地，如果 interaction 矩阵的数值量级远小于 attention 项（后续实验证实相差 $10^4$-$10^5$ 倍），那它对模型性能到底有多重要？能否在推理时安全移除以节省计算成本？

**本文目标** 具体来说，本文聚焦于三个递进的问题：（1）ParT 中稀疏 attention 的根源是什么——传统 attention 还是 interaction 矩阵？（2）在不同数据集和特征配置下，稀疏性是否普遍存在？（3）interaction 矩阵虽然量级微小，但它对模型推理的实际影响有多大？

**切入角度**：作者采用了一个巧妙的分析策略——先比较 pre-softmax 阶段 attention 项和 interaction 项的绝对量级比值，直接量化两者的相对贡献；再通过跨数据集（JetClass、Top Landscape、Quark-Gluon）和跨特征配置（full features vs. kinematic-only）的系统对比，排除 PID 特征等混淆因素；最后通过零化 interaction 矩阵的消融实验，精确测量其对推理性能的影响，并提出 interaction-dependent computation 这一新指标来解释该影响机制。

**核心 idea**：通过量级分析、跨数据集对比和精细消融，证明 ParT 的稀疏 attention 由 attention 机制自发产生，但 interaction 矩阵虽量级微小却通过改变绝大多数 token 的 argmax 粒子选择而不可或缺。

## 方法详解

### 整体框架

本文并非提出一个新的模型架构，而是一项对 Particle Transformer 内部机制的深入分析研究。研究框架包含四个层次递进的分析模块：（1）**attention 分布可视化**，在多个数据集上绘制 post-softmax attention 权重的分布直方图，直观展示二值化程度；（2）**pre-softmax 量级比值分析**，计算传统 attention 分数与 interaction 矩阵值的比值分布，定位稀疏性的主导来源；（3）**$\eta$-$\phi$ 平面粒子 attention 图**，利用 jet 子结构聚类（$k_T$ 算法）将 attention 关系投影到物理坐标空间，检验模型是否捕获了有意义的物理关联；（4）**零化消融与 interaction-dependence 分析**，通过将 interaction 矩阵参数置零并追踪其对 argmax 选择的影响，精确量化 interaction 矩阵的推理贡献。

研究的核心对象是 ParT 的 Particle Multi-Head Attention（P-MHA）机制。给定 jet 中 $N$ 个粒子的表示 $x \in \mathbb{R}^{N \times d}$，P-MHA 的计算公式为：

$$\text{head}_i = \text{softmax}\left(\frac{xW_i^Q(xW_i^K)^\top}{\sqrt{d_k}} + U_i\right) xW_i^V$$

其中第一项 $A = xW_i^Q(xW_i^K)^\top / \sqrt{d_k}$ 是标准的缩放点积 attention，第二项 $U_i \in \mathbb{R}^{N \times N}$ 是通过卷积层从粒子对的运动学特征（$\ln\Delta, \ln k_T, \ln z, \ln m^2$）学到的 interaction 矩阵。两者相加后经过 softmax 得到最终的 attention 权重。整个分析围绕 $A$ 和 $U$ 的相对贡献展开。

### 关键设计

1. **Pre-softmax 量级比值分析（Magnitude Ratio Analysis）**:

    - 功能：量化传统 attention 分数 $A$ 和 interaction 矩阵 $U$ 在 pre-softmax 阶段的相对大小，判定哪个成分主导最终的稀疏 attention 分布。
    - 核心思路：对于每个注意力头中每对粒子 $(i,j)$，计算比值 $|A_{ij}| / |U_{ij}|$，然后统计该比值在整个测试集上的分布。如果这个比值远大于 1，说明 softmax 的输入几乎完全被 $A$ 项控制，$U$ 的贡献在数值意义上可以忽略。作者对三个数据集分别进行了该分析：在 JetClass（full features 和 kinematic-only）上，比值几乎总是大于 1，且峰值在 $10^4$-$10^5$ 量级，意味着 attention 项在数值上完全碾压 interaction 矩阵；但在 Top Landscape 数据集上，两者的量级相当，说明 interaction 矩阵在该场景下发挥了更对等的作用。
    - 设计动机：这一分析的关键洞察在于，softmax 是一个竞争性函数——最大值会主导输出。如果 $A$ 的量级远超 $U$，那么 $U$ 几乎无法改变 softmax 输出的分布形态。因此，attention 的二值化稀疏模式必然由 $A$ 项主导。这个分析优雅地将定性观察（"attention 很稀疏"）转化为定量结论（"稀疏性来自 attention 自身，与 interaction 矩阵无关"），为后续的消融实验奠定了理论基础。

2. **跨数据集跨特征的稀疏性对比分析**:

    - 功能：在不同数据集（JetClass、Top Landscape、Quark-Gluon）和不同特征配置（full features 含 PID vs. kinematic-only）下，比较 attention 分布的稀疏程度，排除 PID 特征作为稀疏性来源的假设。
    - 核心思路：作者绘制了四种配置下 post-softmax attention 权重的直方图：（a）Quark-Gluon 数据集、（b）Top Landscape 的 $t \to bqq'$ 类别、（c）JetClass full features 的 $t \to bqq'$ 类别、（d）JetClass kinematic-only 的 $t \to bqq'$ 类别。结果显示，JetClass 在 full features 和 kinematic-only 两种配置下都呈现出极端的二值化分布（权重集中在 0 和 1 附近），Quark-Gluon 数据集同样如此，但 Top Landscape 数据集的 attention 分布则相对平滑、没有二值化特征。进一步的关键对比是：JetClass kinematic-only 与 Top Landscape 使用完全相同类型的输入特征（仅运动学信息，无 PID），但前者呈现二值化而后者不呈现，这直接排除了"PID 特征导致稀疏性"的假设，说明稀疏性更可能与数据集的规模和任务复杂度有关。
    - 设计动机：先前工作（Wang et al.）报告了稀疏现象但未排除 PID 作为潜在原因。ParT 的 JetClass 版本使用了 17 维特征（含 PID），而 Top Landscape 只有 7 维运动学特征，简单比较可能得出"PID 导致稀疏"的错误结论。通过在 JetClass 上去掉 PID 特征后仍观察到稀疏性，作者清晰地建立了稀疏性与数据集/任务特性之间的关系，而非与特征类型的关系。这种控制变量的实验设计体现了物理学家的严谨作风。

3. **$\eta$-$\phi$ 平面 attention 可视化与 jet 子结构分析**:

    - 功能：将 attention 关系投影到粒子物理中常用的 $\eta$（赝快度）-$\phi$（方位角）坐标空间，结合 $k_T$ clustering 算法将 jet 解构为子 jet，分析 ParT 是否学到了物理上有意义的粒子关联。
    - 核心思路：对于每个 jet，首先使用 FastJet 的 $k_T$ 算法将粒子聚成固定数目的子 jet（半轻子衰变 $t \to b\ell\nu$ 聚为 2 个子 jet，强子衰变 $t \to bqq'$ 聚为 3 个子 jet）。然后在 $\eta$-$\phi$ 平面上画出每个粒子的位置，用不同符号标记粒子类型（✖ 代表 muon，▲ 代表带电强子，▼ 代表中性强子，⚫ 代表光子，✚ 代表电子），透明度与粒子横动量 $p_T$ 成正比，连线强度反映 attention 分数大小。关键创新在于，作者特意只使用 pre-softmax attention 值（不含 interaction 矩阵）来构建可视化，以隔离 attention 机制自身的信息捕获能力。结果显示，即使不使用 interaction 矩阵，ParT 的 attention 机制仍然能够识别子 jet 结构和子 jet 之间的关联，更令人印象深刻的是，即使在 kinematic-only 配置（没有 PID 信息）下，模型仍然能精准识别出半轻子衰变中的轻子粒子。
    - 设计动机：$\eta$-$\phi$ 可视化是高能物理领域的标准分析工具，模型若能在这个空间中展示出有物理意义的 attention 模式，就能增强物理学家对模型的信任。将 interaction 矩阵剥离后仍能看到正确的子结构，这一结果有两个重要意义：第一，确认了稀疏 attention 不仅是数值上的稀疏，而且捕获了真实的物理关联；第二，说明 attention 机制自身具有足够强的表示能力来编码 jet 的拓扑结构。

### 消融实验设计

本文设计了一种独特的消融策略——零化消融（Zero $U$ Ablation），与标准消融不同。标准消融是从头训练一个不包含 interaction 矩阵的模型（ParT plain），而本文的零化消融是将已训练好的 ParT 模型中 PairEmbed 模块的所有参数置零，然后直接推理。这两种消融的区别至关重要：

标准消融（ParT plain）的精度为 0.849，与完整模型的 0.861 差距不大，这曾被解读为"interaction 矩阵不太重要"。但零化消融（ParT Zero $U$）的精度暴跌至 0.405，这说明模型在训练过程中已经将 interaction 矩阵的信息"融入"了其他参数的学习中——其他权重的优化是在 interaction 矩阵存在的条件下进行的，突然移除等于破坏了模型内部精细调校的平衡。

为了进一步解释为什么量级微小的 interaction 矩阵能产生如此巨大的影响，作者引入了 **interaction-dependent computation** 这一新指标。具体定义为：对于粒子 $j$，如果 $\text{argmax}(A_j + U_j) \neq \text{argmax}(A_j)$，即加入 interaction 矩阵后改变了最高权重粒子的选择，则该计算被标记为 interaction-dependent。统计结果显示，虽然每个注意力头中只有 3.6% 的 token 更新涉及 interaction-dependent computation，但 85.4% 的 token 在模型的某个注意力头中至少经历了一次 interaction-dependent computation。这意味着 interaction 矩阵虽然量级小，但通过改变关键的 argmax 选择，以"蝴蝶效应"般的方式影响了绝大多数 token 的最终表示。

此外，作者还追踪了 **non-binary computation** 的比例——定义为 post-softmax 最大权重小于 0.8 的情况。结果显示 non-binary computation 的总体比例为 0.88%，42.1% 的 token 至少经历一次。通过计算 interaction-dependent 和 non-binary 之间的 Pearson 相关系数（PCC = 0.229），作者证明两者没有强相关性。这个结果的含义是：interaction 矩阵的主要作用不是打破 attention 的二值化特性（即不是让 attention 变得更"软"），而是在保持二值化的同时改变粒子之间的竞争结果——让模型关注"正确的"粒子。

## 实验关键数据

### 主实验

本文的核心实验围绕三个基准数据集展开，比较 ParT 在不同配置下的表现和 attention 特性。

| 配置 | 数据集 | Accuracy | AUC | 稀疏性 |
|------|--------|----------|-----|--------|
| ParT (full) | JetClass | 0.861 | — | 强二值化 |
| ParT (Zero $U$) | JetClass | 0.405 | 0.8974 | — |
| ParT (plain, 文献值) | JetClass | 0.849 | — | — |
| ParT | Top Landscape | — | — | 非二值化 |
| ParT | Quark-Gluon | — | — | 二值化 |

ParT (Zero $U$) 的详细分类性能表揭示了各类别的性能下降程度：

| 类别 | $\text{Rej}_{50\%}$ (ParT Zero $U$) | 说明 |
|------|--------------------------------------|------|
| $H \to b\bar{b}$ | 15.0 | 下降显著 |
| $H \to c\bar{c}$ | 8.81 | 下降显著 |
| $H \to gg$ | 19.9 | 相对较好 |
| $H \to 4q$ | 5.53 | 严重退化 |
| $H \to \ell\nu qq'$ | 3.03 ($\text{Rej}_{99\%}$) | 严重退化 |
| $t \to bqq'$ | 79.5 | 相对保持 |
| $t \to b\ell\nu$ | 2.69 ($\text{Rej}_{99.5\%}$) | 严重退化 |
| $W \to qq'$ | 25.6 | 下降显著 |
| $Z \to q\bar{q}$ | 11.8 | 下降显著 |

从表中可以看出，零化 interaction 矩阵对不同衰变模式的影响差异很大。$t \to bqq'$ 的 $\text{Rej}_{50\%}$ 仍有 79.5，相对保持了较多的判别能力，而 $H \to 4q$ 和半轻子衰变模式的性能则几乎崩溃。这暗示 interaction 矩阵对涉及更复杂子结构或需要精细粒子区分的类别更加关键。

### 消融实验

| 分析指标 | 总体比例 | 涉及 token 比例 | PCC |
|----------|---------|-----------------|-----|
| Interaction-Dependent Computation | 3.6% | 85.4% | 0.229 |
| Non-binary Computation | 0.88% | 42.1% | — |

这组数据揭示了一个深刻的现象：从"每次计算"的层面看，interaction 矩阵影响的比例不高（仅 3.6%），但由于 ParT 有多个注意力头和多层结构，累积效应使得 85.4% 的 token 在推理过程中至少一次受到 interaction 矩阵的实质性影响。这解释了为什么量级微小的 interaction 矩阵在零化后会导致性能暴跌——不是因为它改变了很多次的计算结果，而是因为它改变了绝大多数 token 的至少一次关键选择。

### 量级分析关键发现

Pre-softmax attention 分数与 interaction 矩阵值的比值分析是本文最核心的定量发现。在 JetClass 数据集上（无论是否包含 PID 特征），该比值的分布峰值位于 $10^4$-$10^5$ 量级，且"比值小于 1"（即 interaction 矩阵主导）的情况极为罕见。这意味着在绝大多数粒子对之间，softmax 的输入几乎完全由传统 attention 决定，interaction 矩阵的数值贡献微乎其微。

然而在 Top Landscape 数据集上，比值分布更加分散，interaction 矩阵和 attention 项的量级相当。这与 Top Landscape 上 attention 分布不呈现二值化的观察一致——当两个输入项的量级相近时，softmax 的输出自然更加"软"，因为没有单一输入项能够完全压制其他项。

### 关键发现

- **稀疏性的来源确认**：通过量级比值分析，本文明确证实 ParT 的二值化稀疏 attention 主要由 attention 机制自身产生，interaction 矩阵在数值上可以忽略。这推翻了"稀疏性可能来自物理先验"的直觉猜测。
- **PID 特征不是稀疏性的原因**：JetClass kinematic-only 仍然呈现二值化分布，而使用相同类型特征的 Top Landscape 则不呈现，说明稀疏性与数据集规模和任务复杂度相关，而非特定输入特征。JetClass 有 100M 训练样本和 10 个类别，而 Top Landscape 只有 1.2M 训练样本和 2 个类别，更大的训练规模和更复杂的分类任务可能促使模型学到更极端的稀疏模式。
- **Interaction 矩阵的蝴蝶效应**：虽然 interaction 矩阵的绝对量级远小于 attention 项（$10^{-4}$ 到 $10^{-5}$ 倍），但它改变了 85.4% token 的至少一次 argmax 粒子选择。这揭示了 softmax 函数的一个有趣特性——在已经高度稀疏化的 attention 分布中，即使微小的偏置也能在竞争边缘决定"赢家"，因为此时多个候选者之间的 attention 差异可能很小。
- **模型自动发现物理结构**：ParT 在不使用 PID 信息的情况下仍能在 $\eta$-$\phi$ 平面上正确识别半轻子衰变中的轻子，说明模型从纯运动学特征中自动学到了粒子类型的隐式表示。这对物理学家来说意义重大——模型不仅在"做对了"，而且在"用正确的方式做对了"。

## 亮点与洞察

- **Pre-softmax 量级比值分析是一个优雅的可解释性工具**。在任何包含多个加性输入项经过 softmax 的架构中（如各种 bias-augmented attention 机制），都可以使用这种量级比值分析来快速判定各输入项的实际贡献。这个方法简单、量化、可复现，比直接解读 attention map 更加可靠。未来在分析 relative position encoding、ALiBi 等 attention bias 的实际影响时，完全可以直接借用这一方法论。
- **Interaction-dependent computation 指标巧妙地连接了"量级微小"与"影响巨大"这对看似矛盾的观察**。通过追踪 argmax 是否改变，而非比较数值大小，作者找到了正确的分析粒度。这个指标的设计体现了一个深刻洞察：在高度稀疏化的 attention 分布中，重要的不是数值大小，而是竞争排名。这种"虽小但关键"的现象在深度学习中可能普遍存在（如 dropout、label smoothing 等微小扰动对训练的影响），值得更多关注。
- **零化消融 vs 标准消融的对比揭示了模型组件之间的共适应（co-adaptation）**。从头训练不含 interaction 的 ParT（plain）只损失 1.2% 精度，但训练后零化则损失 45.6% 精度，这说明模型中各组件的权重已经互相适应。这对模型压缩和剪枝实践有重要启示：不能简单地通过"该组件的权重量级小"来判断其可以安全移除，需要配合重训或微调。
- **跨数据集的稀疏性差异**是一个有趣的发现。JetClass（大规模、多类别）产生二值化 attention，而 Top Landscape（小规模、二分类）则不产生。这暗示稀疏性可能是模型在面对复杂多分类任务时自发产生的一种"信息压缩"策略——通过让每个粒子只关注最重要的一个伙伴来简化计算和表示。如果这一假说成立，那么在 NLP 和 CV 领域的大规模多任务 Transformer 中也可能观察到类似的稀疏化趋势。

## 局限与展望

- **单一模型架构**：本文所有分析都基于 ParT 一个模型和一个预训练权重（JetClass 用预训练，Top Landscape 和 Quark-Gluon 从头训练），没有验证结论在不同模型大小、不同训练超参数设置下是否稳健。特别是，稀疏性随训练轮数的演化过程没有被追踪——稀疏性是在训练早期就出现的，还是慢慢收敛到的？
- **因果解释的缺失**：虽然本文成功识别了稀疏性的"来源"（attention 自身而非 interaction），但没有解释稀疏性产生的"原因"——是什么驱动了 attention 分数在训练过程中向极端值收敛？这可能与 softmax 函数的数学性质、训练目标的梯度动态、或数据分布的特定结构有关，需要更深入的理论分析。
- **Interaction 矩阵的精细作用机制未被揭示**：本文证明了 interaction 矩阵"通过改变 argmax 影响性能"，但没有深入分析它"在什么物理场景下改变 argmax"以及"它引导模型关注什么类型的粒子对"。未来工作可以按粒子类型、$p_T$ 区间、子 jet 结构等维度分解 interaction-dependent computation，以揭示 interaction 矩阵编码了什么物理信息。
- **缺少 Top-k attention 的实际验证**：作者在展望中提到稀疏 attention 可以用 top-$k$ 机制来加速，但没有实际实现和测试。考虑到 attention 已经近乎二值化（一个粒子只关注一个伙伴），理论上 top-1 attention 就可以近似完整 softmax attention 而大幅减少计算量，这是一个高价值的工程改进方向。
- **数据集差异的根本原因未明确**：为什么 JetClass 产生二值化而 Top Landscape 不产生？是数据集规模（100M vs 1.2M）、类别数（10 vs 2）、物理过程的复杂度，还是其他因素？没有控制变量的系统实验来回答这个问题。

## 相关工作与启发

- **vs ParticleNet（GNN 方法）**：ParticleNet 使用 EdgeConv 进行局部邻域聚合，而 ParT 使用全局注意力进行粒子间关联。Mokhtar et al. 使用 layerwise relevance propagation 分析了 ParticleNet 的边缘相关性，发现它能识别三叉强子 top 衰变结构。本文则从 attention sparsity 的角度分析 ParT，两种可解释性方法形成了有趣的平行对比——前者关注"哪些边重要"，后者关注"attention 为什么稀疏"。
- **vs 标准 Vision/NLP Transformer**：标准 Transformer 中的 attention 通常是分散的（尤其在浅层），而 ParT 呈现极端二值化。这可能反映了输入数据性质的根本差异——自然语言和图像中的 token 之间存在丰富的语义关联，而粒子物理中的粒子之间的关联更加"硬"（一个粒子要么来自同一个衰变支，要么不来自），导致 attention 自然趋向离散化。这个视角可以迁移到其他具有离散关联结构的领域，如分子图、社交网络等。
- **vs Wang et al. (2024)**：先前工作观察到了稀疏 attention 并做了 $\eta$-$\phi$ 可视化，但没有系统分析稀疏性的来源。本文在其基础上增加了量级比值分析、跨数据集对比和 interaction-dependence 定量分析，将定性观察提升为定量结论。可以说本文是对 Wang et al. 工作的一个自然而重要的延伸。
- **与 attention 可解释性的讨论**：作者明确引用了 Jain & Wallace (2019) 关于"attention 不等于 explanation"的警示，承认 attention map 只提供了模型行为的局部视图。这种诚实的自我定位值得赞赏——本文的优势在于它不仅分析 attention"长什么样"，还分析了"为什么长这样"和"各组件的相对贡献"，比单纯的 attention 可视化更深入一步。

## 评分

- 新颖性: ⭐⭐⭐ 研究问题有新意（为什么 ParT 的 attention 是稀疏的），但分析方法相对标准（量级比较、消融、可视化），没有提出新的模型或算法。
- 实验充分度: ⭐⭐⭐⭐ 跨三个数据集和两种特征配置的系统对比较为全面，interaction-dependence 指标设计巧妙，但缺少对稀疏性成因的控制变量实验和 top-k attention 的实际验证。
- 写作质量: ⭐⭐⭐⭐ 结构清晰，逻辑链完整，从现象观察到原因分析到消融验证层层递进，图表质量高；但整体篇幅偏短，一些关键细节（如 Top Landscape vs JetClass 差异的详细讨论）展开不够。
- 价值: ⭐⭐⭐⭐ 对高能物理 ML 社区有直接价值：澄清了 ParT 内部机制、指明了架构简化方向（top-k attention）；对更广泛的 ML 社区也有启发——interaction-dependence 分析方法和"量级小但影响大"的发现具有跨领域迁移潜力。

<!-- RELATED:START -->

## 相关论文

- [Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [Born a Transformer – Always a Transformer? On the Effect of Pretraining on Architectural Abilities](born_a_transformer_--_always_a_transformer_on_the_effect_of_pretraining_on_archi.md)
- [Why Is Spatial Reasoning Hard for VLMs? An Attention Mechanism Perspective on Focus Areas](../../ICML2025/interpretability/why_is_spatial_reasoning_hard_for_vlms_an_attention_mechanism_perspective_on_foc.md)
- [Tropical Attention: Neural Algorithmic Reasoning for Combinatorial Algorithms](tropical_attention_neural_algorithmic_reasoning_for_combinatorial_algorithms.md)
- [Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)

<!-- RELATED:END -->
