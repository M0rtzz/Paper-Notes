---
title: >-
  [论文解读] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning
description: >-
  [ICML 2026][优化/理论][Hebbian 学习] 本文证明：在 L2 权重衰减附近的稳态附近，**几乎任何**学习规则（SGD、Adam、DFA，甚至随机网络）的学习信号都会**自发**朝 Hebbian 方向对齐，而足够强的噪声又会把它翻成 anti-Hebbian，并在 $\gamma \propto \sigma^2$ 处出现明确的相变边界。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "Hebbian 学习"
  - "权重衰减"
  - "学习信号对齐"
  - "噪声-正则相图"
  - "神经可塑性"
---

# Ubiquity of Emergent Hebbian Dynamics in Regularized Learning

**会议**: ICML 2026  
**arXiv**: [2505.18069](https://arxiv.org/abs/2505.18069)  
**代码**: 无  
**领域**: 优化理论 / 生物可塑性建模  
**关键词**: Hebbian 学习, 权重衰减, 学习信号对齐, 噪声-正则相图, 神经可塑性

## 一句话总结
本文证明：在 L2 权重衰减附近的稳态附近，**几乎任何**学习规则（SGD、Adam、DFA，甚至随机网络）的学习信号都会**自发**朝 Hebbian 方向对齐，而足够强的噪声又会把它翻成 anti-Hebbian，并在 $\gamma \propto \sigma^2$ 处出现明确的相变边界。

## 研究背景与动机

**领域现状**：经典神经科学把 Hebbian 与 anti-Hebbian 可塑性视作大脑学习的核心机制——「fire together, wire together」靠局部同突触规则（如 STDP）实现，再由稳态约束（如 Oja 规则）防止权重发散。在机器学习里，深度网络几乎一律用梯度下降+权重衰减+随机扰动训练，与生物机制看起来完全不同。

**现有痛点**：实验上一旦观察到突触更新里出现 Hebbian/anti-Hebbian 结构，就常被用作"反证"——证明大脑不可能在跑 global error-driven 优化。这给两套学习范式之间划了一条硬边界。

**核心矛盾**：是否能仅凭"看到 Hebbian 形式的更新"就断定底层就是 Hebbian 计算？换言之，Hebbian 签名是不是 *identifiable*？前人 Ziyin et al. (2025b) 提示过 weight decay 与 Hebbian 对齐有关，但只给出零散观察，没有给出统一机制，也没有解释 anti-Hebbian 何时出现。

**本文目标**：(1) 给出一个 *generic* 的近稳态机制，说明为什么 L2 weight decay 会强迫学习信号朝 Hebbian 方向投影；(2) 给出对偶机制，说明噪声为什么把它翻成 anti-Hebbian；(3) 在多种架构/优化器/任务上做实证验证。

**切入角度**：作者注意到——生物里 Hebbian 是**扩张**力，必须靠稳态机制做**收缩**；深度学习里 learning signal 也必须做扩张，才能抵抗 weight decay 的收缩。**一旦系统能进入近稳态，两类"对立力"必须平衡**，这就给学习信号方向施加了几何约束。

**核心 idea**：把 weight decay 看作"通用 anti-Hebbian 项"，那么稳态条件 $\mathbb{E}_x[g(x,\theta)] \approx \gamma W$ 自动迫使期望学习信号与 Hebbian 方向 $\bar H(W)=\mathbb{E}_x[h_b h_a^\top]$ 同号；噪声引入额外的二次项把这一不等号反过来。

## 方法详解

本文是纯理论+模拟论文，没有新算法，因此「方法详解」是对两套机制的数学推导以及实验范式的细节解读。

### 整体框架

考虑某一隐藏层 $h_b = W h_a(x)$，其中 $h_a$ 是上一层的后激活、$h_b$ 是本层的前激活。带 weight decay 的更新可写为

$$\Delta W \propto \underbrace{-(\nabla_{h_b}\ell)\, h_a^\top}_{\text{学习信号}} - \gamma W$$

定义 Hebbian 方向 $\Delta_{\rm Hebb} W = h_b h_a^\top$ 与 anti-Hebbian 方向 $-h_b h_a^\top$。文章核心量是「学习信号」与 Hebbian 更新之间的**对齐度**，用 Frobenius 内积或余弦相似度刻画。整体论证分两块：第 3 节给出对齐为正的两个 theorem（特殊形式 + 通用形式），第 4 节给出噪声把对齐翻负的相图。

### 关键设计

**1. 稳态条件下的 Hebbian 对齐定理（强形式）：把对齐度算成 weight decay 的单调函数**

第一步要证的是"标准"梯度学习信号在近稳态时确实朝 Hebbian 方向对齐，而且对齐度随 weight decay 单调上升。出发点是近稳态约束 $\mathbb{E}_x[(\nabla_{h_b}\ell) h_a^\top] + \gamma W \approx 0$，两边右乘 $W^\top$ 并代入 $h_b = W h_a$，得到 $\mathbb{E}_x[(\nabla_{h_b}\ell) h_b^\top] = -\gamma W W^\top$，取迹便有 $\mathrm{Tr}\,\mathbb{E}_x[(\nabla_{h_b}\ell)^\top h_b] = -\gamma\,\mathrm{Tr}[WW^\top] < 0$。再借一条"presynaptic 范数近似常数"的假设 $\|h_a\|^2 \approx \mathbb{E}\|h_a\|^2$（在 neural collapse 或归一化下成立），就能把学习信号与 Hebbian 更新的 Frobenius 内积化简成 $\gamma\,\mathbb{E}\|h_a\|^2\,\mathrm{Tr}[WW^\top] > 0$。这个结论比"方向同号"更强——它直接刻画了统计相关性，并显式给出对齐度对 $\gamma$ 的单调依赖，正是后续实验里"weight decay 调大→对齐变强"的理论根据。

**2. 任意学习规则的通用 Hebbian 投影定理（弱形式）：去掉"必须是真梯度"的假设**

更反直觉的一步是把"学习信号必须是真实负梯度"这个假设也扔掉。写一般更新 $\Delta W = g(x,\theta) - \gamma W$，稳态给出 $\mathbb{E}_x[g(x,\theta)] \approx \gamma W$，直接算它和 Hebbian 方向 $\bar H(W) := \mathbb{E}_x[h_b h_a^\top]$ 的内积：

$$\langle \mathbb{E}_x[g],\bar H\rangle_F = \gamma\langle W,\bar H\rangle_F = \gamma\,\mathbb{E}_x \|h_b\|^2 > 0$$

整个推导完全不依赖 $g$ 来自梯度，所以 Adam、DFA，甚至"随机网络当 teacher"的 Random NN 都满足。这把 Hebbian 对齐升级成了正则化学习里的"通用投影"，也带来本文最关键的结论：仅凭观察到 Hebbian 签名，无法反推底层算法是否在做梯度学习——Hebbian 签名是 non-identifiable 的。

**3. 噪声诱导的 anti-Hebbian 相变：解释何时翻号，给出相图边界**

对偶的问题是 anti-Hebbian 何时出现。在线性回归 $\ell(w)=\tfrac12 (w^\top x - y)^2$ 上每步给参数注入噪声 $w = v + \epsilon$、$\epsilon \sim \mathcal N(0,\sigma I)$，把 SGD 学习信号 $\Delta_{\rm SGD} w = -x(w^\top x - y)$ 与 Hebbian 更新 $\Delta_{\rm Hebb} w = x w^\top x$ 的对齐度展开：

$$\mathbb{E}_\epsilon[(\Delta_{\rm SGD}w)^\top \Delta_{\rm Hebb}w] = -\|x\|^2\big[(v^\top x)^2 + \sigma^2\|x\|^2 - v^\top x\,y\big]$$

噪声多贡献的 $-\sigma^2\|x\|^4$ 项会把对齐推向负值。叠加 weight decay 后近似为 $\approx -\sigma^2 c_0 + \gamma c_1$，于是相变边界落在 $\gamma \propto \sigma^2$ 的抛物线上，与实验 Figure 4 的"白色相变带"形状一致。这把生物里"Hebbian/anti-Hebbian 共存且可切换"的现象统一收进一个 contractive vs expansive 力的相图视角，还给出可验证预测：强 ambient 噪声 + 弱 weight decay 的脑区会以 anti-Hebbian 占主导。

### 损失函数 / 训练策略

本文不引入新的训练目标，所有实验沿用标准 CE/MSE。两个标准设置为：
- **SCE**（Standard Classification Experiment）：在 CIFAR-10 上训练 2 层 128-d tanh MLP，cross-entropy，默认 $\eta=0.01$、batch=256、50 epochs。
- **SRE**（Standard Regression Experiment）：student-teacher 回归，输入/输出 32 维各向同性 Gaussian，2 万训练样本+2 千验证；transformer 变体用 2 层 4 头、32-d token embed。
论文反复强调要用**较大 batch（256）**做对齐度量——因为定理是关于稳态期望的，小 batch SGD 噪声会掩盖 Hebbian 信号。

## 实验关键数据

### 主实验

**Table 1（节选）**：固定 SRE 设置，扫 weight decay $\gamma$，跑 10 seeds，报告第 2 层学习信号与 Hebbian 更新的余弦对齐均值±std。

| 模型 | 学习规则 | $\gamma=0$ | $\gamma=5\!\times\!10^{-5}$ | $\gamma=5\!\times\!10^{-4}$ | $\gamma=5\!\times\!10^{-3}$ |
|------|---------|-----------|------------------------------|------------------------------|------------------------------|
| Regression MLP | Adam | $-0.02\pm0.00$ | $0.10\pm0.00$ | $\mathbf{0.66\pm0.01}$ | — |
| Regression MLP | SGD | $-0.10\pm0.01$ | $-0.06\pm0.01$ | $0.17\pm0.01$ | $\mathbf{0.59\pm0.01}$ |
| Regression MLP | DFA | $0.45\pm0.05$ | $0.45\pm0.04$ | $0.68\pm0.05$ | $\mathbf{0.87\pm0.00}$ |
| Regression MLP | Random NN | $0.00\pm0.00$ | $0.00\pm0.00$ | $0.05\pm0.00$ | $\mathbf{0.50\pm0.00}$ |
| Transformer | Adam | $-0.02\pm0.02$ | $0.50\pm0.24$ | $\mathbf{0.99\pm0.02}$ | — |
| Transformer | SGD | $0.00\pm0.01$ | $0.04\pm0.01$ | $0.47\pm0.06$ | $\mathbf{0.88\pm0.03}$ |

最反直觉一行是 Random NN——它"什么都没在学"（学习信号来自一个随机初始化网络输出的伪误差），但只要 $\gamma$ 拉到 $5\!\times\!10^{-3}$，对齐度依然冲到 0.5。"—"表示该 $\gamma$ 下权重塌缩到 0。

### 消融实验

**噪声-衰减相图（Figure 4 摘要）**：固定 SRE，扫不同参数噪声 $\sigma$ 与 weight decay $\gamma$ 的组合，统计稳态对齐度符号。

| 配置 | Hebbian 对齐符号 | 说明 |
|------|------------------|------|
| 低噪声 + 高 $\gamma$ | 强正（$>0.5$） | 收缩主导，对齐 Hebbian 方向 |
| 高噪声 + 低 $\gamma$ | 强负（$<-0.3$） | 扩张主导，翻成 anti-Hebbian |
| 相变带 $\gamma \approx \sigma^2$ | $\approx 0$ | 与定理预测的抛物线吻合 |
| 极低噪声 + 极低 $\gamma$ | 接近 0 | 系统未稳态，对齐无定向 |
| 不同激活（Linear/ReLU/Tanh/Sigmoid） | 趋势一致 | 对齐随 $\gamma$ 单调升的性质对激活鲁棒 |

### 关键发现
- **正则化决定对齐符号**：把 $\gamma$ 从 0 提到 $5\!\times\!10^{-3}$，几乎所有 (模型, 优化器) 组合的对齐度都从 $\approx 0$ 跳到 0.5-0.99，对 Transformer+Adam 尤其极端（0.99）。
- **"非学习的规则也对齐"**：Random NN 与 DFA 同样对齐（DFA 甚至在 $\gamma=0$ 时已经 0.45），印证弱形式定理——这是 Hebbian 签名 non-identifiable 的最强证据。
- **泛化最好处反而非 Hebbian**：Figure 5 揭示验证 loss 最低的解恰在 Hebbian/anti-Hebbian 对齐都接近 0 的相变带附近，强 Hebbian 对齐并不意味着学得好。
- **训练早期还有 "alignment bump"**：尤其 ReLU + 小 lr 下，训练初期会出现一次强 Hebbian 对齐尖峰，期间权重范数反而单调下降——意味着这一阶段对齐源自**特征对齐**而非范数膨胀。
- **稳态后出现 Hebbian/anti-Hebbian 振荡**：单个神经元的更新会在两种方向间长期振荡，振荡强的模型常常泛化更好，但反过来不一定成立。

## 亮点与洞察

- **把"Hebbian 是不是底层机制"重述为可证伪命题**：以前神经科学家观察到 Hebbian 签名就当作反对 gradient descent 的证据，本文给出干净的反例——Random NN 也会"看起来很 Hebbian"。这把识别问题从生物层提升到了**信号-动力学层**，可直接通过测 LTP/LTD + 同步测 weight decay 与 noise 来区分两种机制。
- **"通用投影"视角**：把 $\bar H(W)=\mathbb{E}_x[h_b h_a^\top]$ 看作正则化稳态强制的方向锥，任何遵循 $\mathbb{E}[g] \approx \gamma W$ 的算法都会被这个锥投影。这是一种少见、形式极简的「优化器无关」对齐定理，可迁移到分析 LoRA、batchnorm 等其他形式的隐式正则。
- **相变可观测**：$\gamma \propto \sigma^2$ 的抛物线相图给生物实验一个具体可验证的预测——若某脑区噪声大且 weight decay 慢，应该系统性偏向 anti-Hebbian，这是把神经科学测量与机器学习理论挂钩的少见硬接口。

## 局限与展望

- **「近稳态」假设不普适**：定理依赖期望更新接近 0，大模型在长尾训练里几乎不会真正稳态，作者也承认在更大模型上观察到的 anti-Hebbian 偏差很可能源于稳态条件失效。
- **「presynaptic 范数近似常数」是强假设**：依赖 neural collapse 或归一化层。原文虽在 Appendix 给出非均匀 $\gamma$ 的扩展，但对一般高维表征是否成立尚需更多验证。
- **实验局限在小模型**：CIFAR-10 + 2-6 层 MLP/小 transformer，未触及 LLM 规模。本文没回答"对齐现象在 100B 参数模型上是否仍可观测"，这是后续最值得做的扩展。
- **没给"如何区分"的实验方案**：理论给出 mechanistic vs emergent Hebbian 二分，但没给出可立刻在 LTP/LTD 数据上跑的判别器，留给神经科学社区。
- **改进方向**：把分析扩展到非 L2 正则（如 sparsity penalty，作者提到会增强 anti-Hebbian）、把噪声模型替换成 ambient + synaptic 双源噪声，以及把对齐定理推到带 batchnorm/layernorm 的实际架构。

## 相关工作与启发

- **vs Xie & Seung 2003 (Contrastive Hebbian Algorithm)**：他们证明 CHA 在均衡极限等价于反向传播，但 CHA 不是同突触规则，违反 Hebbian 原则；本文给出的等价是「现象等价」——*看上去*像 Hebbian，**不需要**底层真的是同突触规则。
- **vs Ziyin et al. 2025b**：是本文最接近的前作，提示了 weight decay 与 Hebbian 的关联，但只给零散观察。本文把它升级为带有单调性、统计相关性版本，并配上 anti-Hebbian 的对偶定理。
- **vs DFA/Feedback Alignment 类（Lillicrap, Nøkland）**：那一系工作试图设计"生物 plausible 的 gradient surrogate"；本文反过来论证 surrogate 是否"像 Hebbian"并不能作为 plausibility 的判据，重塑了这类工作的评价标准。
- **vs Oja's rule / BCM / Bienenstock 1982**：经典 Hebbian 规则需要显式 normalization；本文给出隐式 normalization 路径——L2 weight decay 直接强制对齐，无需在规则里手工加 normalization 项。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[ICML 2026\] Balancing Learning Rates Across Layers: Exact Two-Step Dynamics and Optimal Scaling in Linear Neural Networks](balancing_learning_rates_across_layers_exact_two-step_dynamics_and_optimal_scali.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)

</div>

<!-- RELATED:END -->
