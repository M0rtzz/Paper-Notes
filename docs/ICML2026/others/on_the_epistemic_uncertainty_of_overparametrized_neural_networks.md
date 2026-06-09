---
title: >-
  [论文解读] On the Epistemic Uncertainty of Overparametrized Neural Networks
description: >-
  [ICML 2026][认知不确定性] 本文指出过参数化神经网络的"认知不确定性"不会随数据增大而消失：因为参数不可识别（permutation + 神经元分裂），即便函数完全识别，参数空间后验仍然在分裂流形上保留连续不确定度…
tags:
  - "ICML 2026"
  - "认知不确定性"
  - "过参数化"
  - "不可识别性"
  - "ReLU 网络后验"
  - "Dirichlet 分裂"
---

# On the Epistemic Uncertainty of Overparametrized Neural Networks

**会议**: ICML 2026  
**arXiv**: [2605.25234](https://arxiv.org/abs/2605.25234)  
**代码**: 无  
**领域**: 贝叶斯神经网络 / 不确定性量化 / 学习理论  
**关键词**: 认知不确定性、过参数化、不可识别性、ReLU 网络后验、Dirichlet 分裂

## 一句话总结
本文指出过参数化神经网络的"认知不确定性"不会随数据增大而消失：因为参数不可识别（permutation + 神经元分裂），即便函数完全识别，参数空间后验仍然在分裂流形上保留连续不确定度，作者以单隐层 ReLU 网为例给出精确后验描述（Dirichlet on simplex）并实证验证。

## 研究背景与动机

**领域现状**：UQ 社区习惯把认知不确定性（epistemic uncertainty, EU）定义为"会随样本量增大而衰减的不确定性"，常通过 function-space variance 或 mutual information 度量；理论上由 Bernstein–von Mises 定理保证后验以 $n^{-1/2}$ 速率收缩到真值附近。

**现有痛点**：BvM 的前提是 Fisher 信息正定，即参数局部可识别。但深度网天然带 permutation 和 rescaling 对称性，更糟的是过参数化时大量"冗余神经元"使参数完全不可识别，多组参数表达同一函数。已有的 permutation/rescaling 对称性研究主要谈优化和 mode connectivity，没人系统讨论它对"后验—不确定性"度量本身的影响。

**核心矛盾**：function-space EU 在 $n\to\infty$ 下确实归零（函数已识别），但 weight-space 后验协方差并不归零，甚至可以发散；如果下游任务（continual learning 的 Fisher 重要性、采样诊断、压缩、可解释性）直接吃 weight-space 后验，就会被误导。

**本文目标**：(i) 形式化"由不可识别性导致的不确定性"为何不能被预测方差捕捉；(ii) 给出过参数化 ReLU 网的精确后验几何（permutation 模式 + 连续 splitting manifold）；(iii) 说明实际采样器（NUTS/SGLD）在这些流形上是怎么走的，以及它对实践意味着什么。

**切入角度**：从"两层 ReLU 网在 $L_2$ 正则下函数等价 ⟺ 参数等价 up to permutation + positive rescaling"这一已知结果出发，构造一组 surjective 赋值映射 $\varsigma:[M]\to[M^\star]$，把"真神经元被几个模型神经元复刻"显式参数化，从而把不可识别性几何化成 Dirichlet 单纯形。

**核心 idea**：把不可识别性写成"赋值映射 $\varsigma$ + splitting 系数 $c_m$"两层结构，前者给出离散的 permutation 模式，后者给出连续的 simplex 流形 $\mathcal{M}_\varsigma\cong \prod_{m'} \Delta^{k_{m'}-1}$；后验在这些流形上的诱导分布恰好是对称 Dirichlet，从而把 EU 的不消失现象写成可计算的闭式。

## 方法详解

### 整体框架
分析对象固定为单隐层 ReLU 网 $f_\mathbf{w}(x)=\mathbf{w}_2^\top \phi(\mathbf{W}_1 x)$，宽度 $M$，真函数 $f^\star$ 由宽度 $M^\star\le M$ 的同类网实现。把后验 $p(\mathbf{w}\mid \mathcal{D}_n)\propto \exp(-\sum_i \ell(y_i,f_\mathbf{w}(x_i)))\cdot \mathcal{N}(\mathbf{0},(2\lambda)^{-1}\mathbf{I})$ 拆成三部分讨论：(1) 用 variance-based EU 度量 $\mathrm{EU}=\mathrm{tr}(\mathrm{Cov}(\mathbf{w}\mid \mathcal{D}_n))$ 解释为什么 function-space 度量看不见 weight-space 的残余不确定性；(2) $M=M^\star$ 情形——精确 permutation 不可识别，后验在大 $n$ 极限下是 Dirac 在 $M!$ 个 permutation 上的均匀混合；(3) $M>M^\star$ 情形——加入连续 splitting，后验集中在 $\bigcup_\varsigma \mathcal{M}_\varsigma$ 上、在每个 $\mathcal{M}_\varsigma$ 内沿 splitting 坐标服从对称 Dirichlet。

### 关键设计

**1. 基于方差的 EU 定义 + 不可识别性下的非零残余：让 EU 看得见 weight-space 的残余不确定性**

UQ 社区惯用的信息论 EU（如 $I(\mathbf{w};y)$ 或 function-space variance）对 permutation 等价类天然不可见——函数一旦识别它就归零，看不到参数空间里仍然残留的不确定性。作者改用 $\mathrm{EU}(y,\mathbf{w}\mid x,\mathcal{D}_n)=\mathrm{tr}(\mathrm{Cov}(\mathbf{w}\mid\mathcal{D}_n))$，即 weight-space 后验协方差的迹，把不可识别性纳进度量。一个极简反例就把问题点透：在线性深网 $\mathcal{N}(\mathbf{w}_L^\top\mathbf{W}_{L-1}\cdots\mathbf{w}_1 x,\sigma^2)$ 里能显式算出 function-space EU 归零、而 weight-space EU $=d\tau^2$（$\tau^2$ 为先验方差）——函数完全识别，参数 EU 却不消失。这个度量既兼容贝叶斯定义，又能捕捉到不可识别方向，为后面 ReLU 网的精确刻画打好地基。

**2. 过参数化的赋值-分裂分解：把冗余神经元的连续自由度显式参数化**

要回答"过参数化到底引入多少不可消除的不确定性"，先得把"冗余怎么发生"写清楚。作者对每个 surjection $\varsigma:[M]\to[M^\star]$（模型神经元 → 真神经元）定义群 $G_{m'}=\varsigma^{-1}(m')$ 和 splitting 系数 $(c_m)_{m\in G_{m'}}\in\Delta^{k_{m'}-1}$，证明所有解必满足

$$\mathbf{w}_{1,m}=\sqrt{c_m}\,\mathbf{w}_{1,\varsigma(m)}^\star,\qquad w_{2,m}=\sqrt{c_m}\,w_{2,\varsigma(m)}^\star,$$

于是函数等价类几何上对应单纯形乘积 $\mathcal{M}_\varsigma\cong\prod_{m'=1}^{M^\star}\Delta^{k_{m'}-1}$。这一分解把"一个真神经元被几个模型神经元复刻、贡献按 $c_m$ 分摊"这件事完全显式化，从而把后验研究归约成"单纯形上诱导分布的研究"；它也是 Lemma 3 证明不同 $\varsigma$ 的开内部流形几乎互不相交（连续动力系统几乎不跨越）的几何前提。

**3. Dirichlet 后验闭式 + balanced 缩放定理：给出"EU 被重分布而非消除"的精确刻画**

有了流形结构，最后一步给出 splitting 系数和参数块的精确后验矩。用 $\varepsilon$-tube 诱导条件分布定义 $\mathcal{M}_\varsigma$ 上的后验，作者证明 $(c_m)_{m\in G_{m'}}\sim\mathrm{Dir}(\alpha,\dots,\alpha)$，$\alpha=(p+1)/2$，由此 $\mathbb{E}[c_m]=k_{m'}^{-1}$、$\mathrm{Cov}(c_m,c_{\tilde m})=-1/\kappa$，进而 $\mathbb{E}[\boldsymbol{\omega}_m]=\mu_{k,\alpha}\boldsymbol{\omega}_{m'}^\star$、$\mathbb{E}[\boldsymbol{\omega}_m\boldsymbol{\omega}_m^\top]=k_{m'}^{-1}\boldsymbol{\omega}_{m'}^\star\boldsymbol{\omega}_{m'}^{\star\top}$。balanced 极限 $k_{m'}\asymp M/M^\star,\ M\to\infty$ 下得 $\mathbb{E}[\boldsymbol{\omega}_m]=\Theta(M^{-1/2})$、$\mathrm{Cov}=\Theta(M^{-1})$。这组闭式矩既允许实验做严格对照（实测均值/二阶矩 vs. $\mu_{k,\alpha}$、$1/k$），又点出核心结论：无限宽极限下单个神经元贡献缩小、但整个 group 总贡献不变、splitting 自由度反而长成高维单纯形——过参数化下 EU 是被重分布，而不是被消除。

### 损失函数 / 训练策略
对应贝叶斯模型：损失采用 $\mathcal{L}(\mathbf{w})=\sum_i \ell(y_i,f_\mathbf{w}(x_i))+\lambda\|\mathbf{w}\|_2^2$，等价于 Gaussian 似然 + Gaussian 先验 $\mathbf{w}\sim\mathcal{N}(\mathbf{0},(2\lambda)^{-1}\mathbf{I})$。实验中后验由两段式采样获得：先用 Adam 多链初始化（等价 deep ensemble of MAP），再用 SGLD 或 NUTS 采样，得到 BDE（Bayesian deep ensemble）。

## 实验关键数据

### 主实验
所有实验在合成数据上做：真函数由 $M^\star=5$ 隐元 ReLU 网生成，$p=5$，$y=f^\star(x)+\varrho$，$\varrho\sim\mathcal{N}(0,1)$，样本量 $n\in\{2^6,\ldots,2^{14}\}$，模型宽度 $M\in\{M^\star,2M^\star,4M^\star,8M^\star\}$。下表概述四个核心实验的预期与观察。

| 实验 | 预期（来自理论） | 观察 |
|------|------|------|
| 收敛性（RMSE / LPPD，DE vs BDE） | 大 $n$ 下 DE/BDE 接近；小 $n$ 下 BDE 更优；过参数化时 BDE 多捕一份连续 EU | 实测三项均符合 |
| function-space EU vs. weight-space trace EU | $\mathrm{Var}(f_\mathbf{w}(x))\downarrow 0$；$\mathrm{tr}(\mathrm{Cov}(\mathbf{w}))$ 不变 | 完全符合 |
| 单链 NUTS 跨 permutation 切换率 | 在 Lipschitz/连续动力系统下应几乎不跨越 | switch rate $<1$ chamber/链，随 $n,M$ 趋势符合预测 |
| splitting 系数分布 | 边际 $c_m\sim\mathrm{Beta}(\alpha,(k-1)\alpha)$, $\alpha=(p+1)/2$ | 经验直方图与理论 Beta 紧密对齐 |

### 消融实验
下表对照"做/不做对不可识别性的显式处理"在四个下游性质上的差异，对应原文 Lemma 1、Corollary 1、Theorem 1。

| 配置 | function-space EU ($n\to\infty$) | weight trace EU ($n\to\infty$) | $\mathrm{Cov}(\boldsymbol{\omega}_m,\boldsymbol{\omega}_{m'})$ | splitting 分布 |
|------|------|------|------|------|
| $M=M^\star$ 单链限单 chamber | $0$ | $\sum_i \upsilon_i^2 \ne 0$ | $0$（在单 chamber 内） | 不适用 |
| $M=M^\star$ 全 permutation 混合 | $0$ | $\sum_i \upsilon_i^2$ | $O(M^{-1})$ 之间块协方差 | 不适用 |
| $M>M^\star$ 固定 $\varsigma$ | $0$ | 非退化 | 由 Dirichlet 决定 | $\mathrm{Dir}(\alpha,\ldots,\alpha)$ |
| $M>M^\star$ balanced 极限 $M\to\infty$ | $0$ | 单个 $\boldsymbol{\omega}_m$ 缩到 $\Theta(M^{-1})$；group 总和不变 | $\Theta(M^{-1})\boldsymbol{\omega}_{m'}^\star \boldsymbol{\omega}_{m'}^{\star\top}$ | $\alpha$ 不变，dim$\uparrow$ |

### 关键发现
- $n\to\infty$ 后 function-space EU 归零、weight-space EU 不归零，前者来自函数识别、后者来自参数不可识别，二者本质不同。
- 单链 NUTS / SGLD 在合理学习率下几乎不跨 permutation chamber，意味着多链 ensemble 是必要的，否则 EU 估计天然有偏。
- 过参数化下单链可以在 $\mathcal{M}_\varsigma^\circ$ 内自由游走，但不会跨不同 $\varsigma$；这意味着实证 EU 估计同时受"单链流形游走"和"多链 chamber 覆盖"两类偏置影响。
- 先验对结果的影响很微妙：在 $\mathcal{M}_\varsigma$ 切向方向不起作用（因为 $\sum_m c_m\|\boldsymbol{\omega}_{m'}^\star\|^2$ 在切向是常数），但在法向方向上塑造近高斯波动，这解释了为什么实践中 weight marginal 常看起来是高斯。
- 在 continual learning 中，Fisher 重要性会被非可识别性"放大"，导致更新过分受限；修正后能改善适应性能（原文 §B.5）。

## 亮点与洞察
- 把 deep ensemble 与 BNN 在"过参数化 + 大 $n$"极限下的差别讲清楚：DE 不能捕到连续 splitting 自由度，而 BDE 可以；这给出了 BDE 优于 DE 的一个理论根据而不是经验说辞。
- 用 Dirichlet 单纯形给出"分裂"的精确分布，是把不可识别性几何化的优美工具，思路可推广到 biases、多层 MLP、甚至 LoRA/低秩适配中的冗余维度分析。
- 关于"单链采样器几乎不跨 permutation chamber"的论断给采样诊断提供了清晰意义：传统 R̂ 等指标若忽略 chamber 切换会高估收敛；作者由此提出一个把非识别方向方差扣除的诊断（§B.5）。

## 局限与展望
- 全部理论严格只在单隐层 ReLU + Gaussian prior + 对应正则下成立，多层网、激活非 ReLU、Batch Norm/skip 这些主流结构都会破坏 Corollary 2 那条干净的 splitting 几何，作者也承认这是 main scope 限制。
- 实验全在合成数据上做，没有真实图像/NLP 上验证 EU 估计偏置对决策（如 OOD 检测、active learning）实际造成多大差；属理论文章的常见局限。
- 提出的 splitting 分布建立在"$L_2$ 正则 + 函数等价 ⟹ 参数等价 up to permutation/scaling"前提，对 dropout / weight decay 之外的隐式正则（如 SGD 隐式偏置）能否照搬不清楚。
- 没有给出具体修正算法（除了 §B.5 两个案例），把"知道 EU 被高估"转成"低估前提下的实用 UQ pipeline"留待后续工作。

## 相关工作与启发
- **vs Hüllermeier & Waegeman 信息论 EU 分解**：他们用 mutual information 定义 EU，对 permutation 等价类不可见；本文指出这是过参数化场景下系统性 underestimate 的根源，并改用方差-based 度量。
- **vs Simsek et al. (2021) 关于过参网络损失景观对称性**：他们刻画的是 optimization landscape 上的连通流形，本文把"流形"这个概念搬到 Bayesian 后验，证明它带来不可消除的参数后验质量。
- **vs Kobialka et al. (2026) splitting 几何**：本文借用其 surjective assignment 工具，但首次把它和 Bayesian posterior 联系起来，给出 Dirichlet 闭式后验。
- **vs Deep Ensembles (Lakshminarayanan et al.)**：本文证明在 $M=M^\star$ + 大 $n$ 下 DE 与真后验在 permutation 模式上恰好一致，是 DE 罕见的理论支撑；但 $M>M^\star$ 下 DE 不足以表达连续 splitting，需要 BDE。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把过参数化的连续不可识别性几何化为 Dirichlet 单纯形，并给出闭式后验矩，是 BNN 理论中少见的精确刻画
- 实验充分度: ⭐⭐⭐ 合成数据上的验证很到位，但缺真实任务上的下游影响评估
- 写作质量: ⭐⭐⭐⭐ 概念清晰，图 1/图 2 把不可识别流形可视化得很直观；定理-推论链条紧凑
- 价值: ⭐⭐⭐⭐ 对 BNN、deep ensemble、continual learning、采样诊断的实践都有直接含义，且改变了对"epistemic uncertainty 会随数据消失"的常识认知

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation](mmd-balls_as_credal_sets_a_pac-bayesian_framework_for_epistemic_uncertainty_in_t.md)
- [\[ICML 2025\] Improved Exploration in GFlowNets via Enhanced Epistemic Neural Networks](../../ICML2025/others/improved_exploration_in_gflownets_via_enhanced_epistemic_neural_networks.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](../../ICML2025/others/rethinking_aleatoric_and_epistemic_uncertainty.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)

</div>

<!-- RELATED:END -->
