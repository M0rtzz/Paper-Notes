---
title: >-
  [论文解读] Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability
description: >-
  [ICML 2026][预训练][参数对称性] 本文提出"有效函数类"和"神经元可辨识性"的理论框架，揭示打破结构对称性并不等于打破有效对称性——即使参数空间的置换对称已被消除，数据依赖的近似对称仍可能使神经元互换代价极低，并据此给出无需对齐即可实现线性模式连通性（LMC）的充分条件。
tags:
  - "ICML 2026"
  - "预训练"
  - "参数对称性"
  - "神经元可辨识性"
  - "线性模式连通性"
  - "模型合并"
  - "损失景观"
---

# Beyond Structural Symmetries: Linear Mode Connectivity via Neuron Identifiability

**会议**: ICML 2026  
**arXiv**: [2606.04754](https://arxiv.org/abs/2606.04754)  
**代码**: https://github.com/vuenc/neuron-identifiability  
**领域**: 优化与理论  
**关键词**: 参数对称性, 神经元可辨识性, 线性模式连通性, 模型合并, 损失景观  

## 一句话总结

本文提出"有效函数类"和"神经元可辨识性"的理论框架，揭示打破结构对称性并不等于打破有效对称性——即使参数空间的置换对称已被消除，数据依赖的近似对称仍可能使神经元互换代价极低，并据此给出无需对齐即可实现线性模式连通性（LMC）的充分条件。

## 研究背景与动机

**领域现状**：现代深度网络通常过参数化，存在大量参数对称性（尤其是隐藏单元的置换对称），使得权重空间中功能相同的模型对应大量等价类。线性模式连通性（LMC）描述了两个独立训练模型在权重空间线性插值时损失保持较低的现象。已有工作表明，在对齐置换后，LMC 在许多场景下成立。

**现有痛点**：近期有多种对称性打破方法（如 $\mathbf{W}$-asymmetric 网络、SYRE 等），通过在权重矩阵中引入固定偏置或对角缩放来消除结构上的置换对称性。然而，打破结构对称性后并非总能获得无对齐的 LMC——有些干预有效，有些无效，背后的机制不清楚。

**核心矛盾**：结构对称性被打破不等于有效对称性被打破。当数据或表示位于低维子空间时，即使不同神经元的参数不再可互换，它们在输入支撑上可实现的函数仍可能相同，且互换代价可能很低。问题在于：对称性打破的有效性取决于架构扰动在输入支撑上的可观测程度，而非仅取决于参数空间的结构。

**本文目标**：(1) 形式化神经元在输入支撑上的有效函数类及其实现代价；(2) 给出神经元可辨识性（跨训练种子一致分配特征给神经元）的条件；(3) 刻画无对齐 LMC 成立的充分条件。

**切入角度**：将每个神经元视为在输入支撑上实现特定函数的算子，用 Mahalanobis 半范数刻画实现代价，通过置换灵敏度衡量对称性打破的有效程度。在子空间支撑模型下，所有分析可转化为显式的线性代数形式。

**核心 idea**：用"有效函数类 + 实现代价"取代"参数空间对称群"来分析对称性打破，揭示数据几何（内在维度、子空间相干性）对 LMC 和模型合并的决定性影响。

## 方法详解

### 整体框架

本文要回答的问题是：为什么有些"对称性打破"干预能换来无对齐的 LMC，有些却不能？为此作者把每个神经元看成一个在输入支撑上实现某种函数的算子，用一个带非对称干预的单层网络作统一载体：$\bm{H}(\bm{W};\bm{x}) = \eta((\mathbf{F} + \mathbf{D} \odot \bm{W})\bm{x})$，其中 $\mathbf{F}$ 是固定偏置矩阵、$\mathbf{D}$ 是固定对角缩放矩阵、$\bm{W}$ 才是可训练权重——$\mathbf{W}$-asymmetric 网络、SYRE、线性残差和稀疏网络都只是这个公式的特例。在"输入实际落在 $k$ 维子空间 $\mathcal{U}$"的低维支撑假设下，整条分析链条是：先刻画每个神经元能实现哪些函数、代价多大（有效函数类），再用置换灵敏度判断这些神经元是否可辨识，最后把可辨识性接到 LMC 的弦偏差上界，给出无需对齐就能线性连通的充分条件。

### 关键设计

**1. 有效函数类与实现代价：用"实现一个函数要多大权重范数"取代"参数能否互换"**

对称性打破到底有没有效，不该看参数空间里神经元还能不能互换，而要看它们在真实输入支撑上能实现什么、各自实现的代价差多少。在子空间支撑模型下，第 $i$ 个神经元能实现的函数类 $\mathcal{H}_i(\mathcal{X})$ 是一个仿射子空间 $\mathbf{v}_i + \mathrm{im}(\mathbf{M}_i) \subseteq \mathbb{R}^k$，其中投影中心 $\mathbf{v}_i = \bm{U}^\top \mathbf{f}_i$ 由固定偏置决定、投影算子 $\mathbf{M}_i = \bm{U}^\top \mathrm{Diag}(\mathbf{d}_i)$ 由对角缩放决定。实现某个目标函数的最小权重范数就化成一个 Mahalanobis 半范数 $\|\bm{h}\|_{\mathcal{H}_i} = \|\bm{a} - \mathbf{v}_i\|_{\mathbf{S}_i^\dagger}$，Gram 矩阵 $\mathbf{S}_i = \mathbf{M}_i \mathbf{M}_i^\top$ 的各向异性决定了哪些方向便宜、哪些方向贵。关键之处在于：哪怕所有神经元的函数类完全相同（$\mathbf{M}_i$ 满秩时），实现同一个函数的代价也可能天差地别；而优化器偏好小范数解，于是会跨训练种子一致地把某个特征分给那个"实现它最便宜"的神经元——这正是可辨识性的来源。

**2. 置换灵敏度：把"神经元能不能被换"量化成换一次要多付多少代价**

有了实现代价，就能问：把特征重新分配给别的神经元会让总代价涨多少？对一个输出侧置换 $\pi$，定义代价灵敏度 $\Delta_\pi^{\mathrm{out}}$，在中心主导（center-dominated）regime 下约为 $\mu_\mathbf{D}^{-1} \|\bm{\delta}_\pi^{\mathrm{out}}\|_F^2$，其大小主要由投影中心之间的最小间距 $\gamma_{\mathrm{out}} = \sqrt{2} \min_{i \neq j} \|\mathbf{v}_j - \mathbf{v}_i\|_2$ 决定。如果每个非恒等置换的灵敏度都足够大，那么"最小复杂度的特征-神经元分配"就是唯一的，神经元因此可辨识。这个量对维度的依赖很有意思：高维下 $\gamma_{\mathrm{out}} = \Theta(\sigma_\mathbf{F} \sqrt{k} m^{-2/k})$，内在维度 $k$ 越大衰减越慢，所以高维数据天然让对称性打破"更有效"。对输入侧置换 $\tau$ 也有对应的灵敏度，由 $\gamma_{\mathrm{in}} = \Theta(\sigma_\mathbf{F} \sqrt{m}) \cdot \min_{a \neq b} \|\bm{U}_{b,:} - \bm{U}_{a,:}\|_2$ 控制，它额外依赖子空间相干性 $\nu(\mathcal{U})$——这解释了为什么单靠对角掩码 $\mathbf{D}$ 的有效性会随数据几何变化。

**3. LMC 弦偏差上界：把可辨识性接到"插值时损失不抬头"**

最后一步是把神经元可辨识性翻译成 LMC 成立。对 ReLU 网络，作者证明线性插值路径上的弦偏差满足 $\sup_\lambda \|\xi_{\bm{H}}(\lambda;\cdot)\|_{L^2} = \mathcal{O}(\beta^{3/2}) \|\mathbf{F} \bm{\Sigma}^{1/2}\|_F$，其中 $\beta$ 衡量可训练部分相对固定部分的比值。再借损失的 Lipschitz 性，弦偏差就直接成了损失障碍的上界，于是只要弦偏差小，LMC 就成立。这一步的价值在于把 LMC 从"分析整个高维损失景观"降维成"逐层算弦偏差"，并用中心主导条件让局部凸性沿整段插值路径传递。

## 实验关键数据

### 主实验：对齐/无对齐 LMC

| 架构 | 数据集 | $\sigma_\mathbf{F}$ | 无对齐 LMC 障碍 | 对齐 LMC 障碍 |
|------|--------|---------------------|-----------------|---------------|
| MLP | MNIST | 0 | 高（~80%精度下降） | 低 |
| $\mathbf{W}$-MLP | MNIST | 0 | 高 | 低 |
| $\mathbf{W}$-MLP | MNIST | 1 | **低（接近零）** | 低 |
| SYRE-MLP | MNIST | - | 低 | 低 |
| ResNet | CIFAR-10 | 0 | 高 | 低 |
| $\mathbf{W}$-ResNet | CIFAR-10 | 2 | **中等偏低** | 低 |
| SYRE-ResNet | CIFAR-10 | - | 中 | 低（对齐仍有帮助） |

### 消融：内在维度 $k$ 对神经元互换代价的影响

| 内在维度 $k$ | $\sigma_\mathbf{F}$ | 无对齐 LMC 精度下降 | 互换代价分布 |
|-------------|---------------------|---------------------|-------------|
| $k=2$ | 1 | 46.4 pp | 大量低代价互换 |
| $k=8$ | 1 | 15.7 pp | 中等 |
| $k=32$ | 1 | **6.1 pp** | 仅对角线低代价 |
| $k=2$ | 0 | 高 | 近零甚至负代价 |

### 关键发现
- **结构 ≠ 有效**：$\mathbf{W}$-asymmetric 网络在 $\sigma_\mathbf{F}=0$ 时虽已打破结构对称性，但无对齐 LMC 障碍仍高，经对齐后与标准网络表现相当
- **内在维度是关键调控变量**：$k$ 越大，投影中心间距 $\gamma_{\mathrm{out}}$ 越大，神经元互换代价越高，无对齐 LMC 越好
- **子空间相干性影响对角缩放的有效性**：高相干性（$\nu \approx 1$）时，即使 $\mathbf{F}=0$，仅靠 $\mathbf{D}$ 的对角掩码也可实现 LMC；低相干性时失效
- **激活匹配实验验证**：当 $\sigma_\mathbf{F}$ 足够大时，恒等置换的激活匹配目标值趋近最优置换，证实神经元可辨识性

## 亮点与洞察
- **理论洞察的精巧性**：将对称性打破从"参数空间的代数性质"转移到"输入支撑上的函数几何"，Mahalanobis 半范数框架使得分析既直观又可计算，完美解释了"为什么 bias 打破对称无效"（偏置只在一维方向引入差异）等经验现象
- **数据几何的核心角色**：揭示数据内在维度和子空间相干性对 LMC 的决定性影响——这是一个可迁移到模型合并、联邦学习、权重空间学习等场景的重要洞察
- **统一多种方案**：$W_{\mathrm{eff}} = \mathbf{F} + \mathbf{D} \odot \bm{W}$ 一个公式涵盖了 W-asymmetric、SYRE、稀疏网络、线性残差等看似不同的对称性打破方案

## 局限与展望
- 理论分析局限于**线性子空间支撑模型**，真实数据的低维流形可能更复杂（非线性、弯曲）
- 当前框架主要针对**逐层分析**，尚未完全刻画深层网络中跨层对称性的交互效应
- **可辨识性 vs 特征学习的 tradeoff**：强对称性打破限制了可学习特征的范围，在中心主导 regime 下网络退化为随机特征模型——如何在保持可辨识性的同时最大化特征学习能力是开放问题
- 未来方向：将框架扩展到更丰富的对称群（如注意力头的对称性）、结合精确训练动力学分析、以及开发可预测 LMC 成立与否的定量诊断工具

## 相关工作与启发
- **LMC 与模型合并**：Entezari et al. (2022) 的 LMC 猜想、Ainsworth et al. (2023) 的 Git Re-Basin 方法、Singh & Jaggi (2020) 的激活匹配
- **对称性打破**：Lim et al. (2024b) 的 W-asymmetric 网络、Ziyin et al. (2025) 的 SYRE
- **权重空间学习**：模型权重作为数据对象的处理方式正成为新兴方向，本文的可辨识性分析直接服务于该领域的"数据对称性"理解

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Annotations Mitigate Post-Training Mode Collapse](annotations_mitigate_post-training_mode_collapse.md)
- [\[ICCV 2025\] FlowMo: Flow to the Mode — Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](../../ICCV2025/llm_pretraining/flow_to_the_mode_mode-seeking_diffusion_autoencoders_for_state-of-the-art_image_.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[NeurIPS 2025\] Beyond Benign Overfitting in Nadaraya-Watson Interpolators](../../NeurIPS2025/llm_pretraining/beyond_benign_overfitting_in_nadaraya-watson_interpolators.md)
- [\[AAAI 2026\] Beyond Cosine Similarity: Magnitude-Aware CLIP for No-Reference Image Quality Assessment](../../AAAI2026/llm_pretraining/beyond_cosine_similarity_magnitude-aware_clip_for_no-reference_image_quality_ass.md)

</div>

<!-- RELATED:END -->
