---
title: >-
  [论文解读] Mirror Descent Under Generalized Smoothness
description: >-
  [ICML 2026][优化/理论][$\ell*$-smoothness] 本文提出一种基于任意范数及其对偶范数的 $\ell*$-广义光滑性概念，并通过"广义自界引理"把梯度对偶范数控制在初始次最优间隙之内…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "$\\ell*$-smoothness"
  - "镜像下降"
  - "非欧几何"
  - "自界性"
  - "LLM 训练曲率"
---

# Mirror Descent Under Generalized Smoothness

**会议**: ICML 2026  
**arXiv**: [2502.00753](https://arxiv.org/abs/2502.00753)  
**代码**: 无  
**领域**: 优化理论 / 镜像下降 / 广义光滑性  
**关键词**: $\ell*$-smoothness、镜像下降、非欧几何、自界性、LLM 训练曲率  

## 一句话总结
本文提出一种基于任意范数及其对偶范数的 $\ell*$-广义光滑性概念，并通过"广义自界引理"把梯度对偶范数控制在初始次最优间隙之内，从而首次为镜像下降及其加速、乐观、Mirror Prox、随机、复合等变体在非欧几何下建立了与经典 $L$-smooth 下匹配的收敛率。

## 研究背景与动机

**领域现状**：现代机器学习的目标函数普遍不满足经典 $L$-smooth 假设——即使最简单的 $\ell_2$ 回归，其全局光滑常数也可能是无界的。Zhang et al. (2020) 在 LSTM/ResNet 训练中观察到 Hessian 范数沿梯度近似线性增长，提出 $(L_0,L_1)$-smoothness：$\|\nabla^2 f(\mathbf{x})\|_2\le L_0+L_1\|\nabla f(\mathbf{x})\|_2$；Li et al. (2023) 进一步推广为 $\ell$-smoothness，把仿射函数换成任意非降次二次函数 $\ell(\cdot)$。

**现有痛点**：所有广义光滑性研究都被锁死在 $\ell_2$ 范数上，只能服务于欧氏空间中的梯度下降族算法。但镜像下降 (MD) 作为非欧优化的主力——在强化学习、网络量化、扩散模型水印、LLM 预训练与后训练 (如 Muon、Scion 等) 中扮演核心角色——却没有任何匹配的广义光滑性理论。

**核心矛盾**：在非欧几何中，范数 $\|\cdot\|$ 与对偶范数 $\|\cdot\|_*$ 不再相等；Li et al. (2023) 把 Hessian 与梯度都用 $\|\cdot\|_2$ 度量在非欧场景下违背了"Hessian 映射 $\nabla^2 f(\mathbf{x})\mathbf{h}$ 与梯度 $\nabla f(\mathbf{x})$ 都属于对偶空间"这一基本事实，强行套用会引入与维度 $n$ 成 $\sqrt{n}$ 量级的多余常数。

**本文目标**：(i) 给出一个原生支持非欧几何的广义光滑性定义；(ii) 把它套进所有主流镜像下降变体（标准 MD、加速 MD、乐观 MD、Mirror Prox、随机 MD、复合 MD）并恢复经典收敛率；(iii) 用 LLM 与 CNN 的真实训练轨迹证明这个定义贴近实际。

**切入角度**：作者注意到 $\nabla^2 f(\mathbf{x})\mathbf{h}$ 是 $\mathcal{E}^*$ 中的元素，因此应该用对偶范数测量大小，分母 $\mathbf{h}$ 才用原范数。把 $\ell$-smoothness 的不等式重写为 $\sup_{\mathbf{h}\ne\mathbf{0}}\{\|\nabla^2 f(\mathbf{x})\mathbf{h}\|_*/\|\mathbf{h}\|\}\le\ell(\|\nabla f(\mathbf{x})\|_*)$ 后，整套理论才"几何上正确"。

**核心 idea**：用"广义自界引理 $\|\nabla f(\mathbf{x})\|_*^2\le 2\ell(2\|\nabla f(\mathbf{x})\|_*)(f(\mathbf{x})-f^*)$"把难以直接追踪的梯度对偶范数通过次最优间隙反向控制——只要算法能保证次最优间隙单调或受控，梯度就自动有界，从而能局部归约到经典 $L$-smooth 分析。

## 方法详解

### 整体框架

整套理论的骨架是一个三段式归约：(1) 定义 $\ell*$-smoothness，把所有 Hessian 与梯度都用合适的（对偶）范数度量；(2) 建立"全局 $\ell*$-smooth ⇔ 局部 $(\ell,r)*$-smooth"的等价命题，使得"知道梯度上界 $G$ 后，在以当前点为中心、半径 $G/L$ 的球内函数表现得像经典 $L$-smooth"；(3) 用广义自界引理把每个 MD 变体的梯度对偶范数 $\|\nabla f(\mathbf{x}_t)\|_*$ 通过次最优间隙 $f(\mathbf{x}_t)-f^*$ 控制住，再用归纳法证明梯度真的不会爆，从而构造一个绝对常数 $L:=\ell(2G)$ 作为有效光滑参数，最终恢复 $O(1/T)$、$O(1/T^2)$、$O(\log T/\sqrt{T})$ 等经典率。

### 关键设计

**1. $\ell*$-smoothness 定义 + 局部等价命题：把 Hessian 放回它真正属于的对偶空间**

非欧几何里范数 $\|\cdot\|$ 和对偶范数 $\|\cdot\|_*$ 不再相等，而 $\nabla^2 f(\mathbf{x})\mathbf{h}$ 和 $\nabla f(\mathbf{x})$ 都属于对偶空间——Li et al. 把它们一律用 $\|\cdot\|_2$ 度量，几何上就错了，会平白引入 $\sqrt{n}$ 量级的维度因子。修正办法是定义 $f\in\mathcal{F}_\ell(\|\cdot\|)$ 当且仅当 $\|\nabla^2 f(\mathbf{x})\mathbf{h}\|_*\le\ell(\|\nabla f(\mathbf{x})\|_*)\|\mathbf{h}\|$ 几乎处处成立：分子用对偶范数、分母用原范数。再配一个"局部版" $(\ell,r)*$-smoothness——在以 $\mathbf{x}$ 为中心、半径 $r(\|\nabla f(\mathbf{x})\|_*)$ 的球内梯度 Lipschitz、常数为 $\ell(\|\nabla f(\mathbf{x})\|_*)$。命题 2.6 证明两者在 Assumption 2.5 下几乎等价，于是分析时可自由切换：全局定义用来推关于次最优间隙的界，局部定义用来在一段轨迹上当"常数光滑"使。当 $\|\cdot\|=\|\cdot\|_2$ 时退化为 Li et al. 的 $\ell$-smoothness。几何选对的好处是定量的——以 $f(\mathbf{x})=(\mathbf{1}_n^\top\mathbf{x})^4/4$ 为例，$\ell_1$ 版只需 $\widetilde\ell(\alpha)=1+2\alpha$，而 $\ell_2$ 版要 $\widehat\ell(\alpha)=n+2\sqrt{n}\alpha$，整整差了 $\sqrt{n}$。

**2. 广义自界引理（Lemma 3.4）+ 有效常数 $G,L$ 构造：用次最优间隙反向控制梯度**

镜像下降在对偶空间做下降，原范数的自相关性 $\langle\mathbf{x},\mathbf{x}\rangle=\|\mathbf{x}\|_2^2$ 消失了，没法像欧氏分析那样直接证梯度范数单调递减。本文换了条路：对任意 $\mathbf{x}$ 证明广义自界引理

$$\|\nabla f(\mathbf{x})\|_*^2\le 2\ell(2\|\nabla f(\mathbf{x})\|_*)\,(f(\mathbf{x})-f^*)$$

它把难追踪的梯度对偶范数翻译成相对好控制的次最优间隙。在 Assumption 3.3 的次二次前提（$\lim_{\alpha\to\infty}\alpha^2/\ell(\alpha)=\infty$）下，方程 $\alpha^2=2\ell(2\alpha)(f(\mathbf{x}_0)-f^*)$ 有最大有限解 $G$，作为整条轨迹梯度对偶范数的统一上界；再令 $L:=\ell(2G)$ 充当"有效经典光滑常数"。对每个 MD 变体，用归纳法证明：只要学习率适当（$\eta\le 1/L$、$1/(2L)$、$1/(3L)$ 等），次最优间隙 $f(\mathbf{x}_t)-f^*\le f(\mathbf{x}_0)-f^*$ 始终成立，于是 $\|\nabla f(\mathbf{x}_t)\|_*\le G$ 自动成立。这套"先假设梯度有界推间隙不增，再用间隙不增反证梯度有界"的循环归纳，通过严格的 Lemmas D.3/E.2/3.5 把循环切开，是整篇的真正发动机。

**3. 加速 MD 的"时间分割"分析与多变体统一框架：覆盖维护多序列的算法**

加速 MD、乐观 MD、Mirror Prox 这类算法同时维护多个序列，单靠 $f(\mathbf{x}_t)-f^*$ 受控只能约束 $\nabla f(\mathbf{x}_t)$、约束不了 $\nabla f(\mathbf{y}_t)$。本文引入辅助量 $e_t:=\|\mathbf{y}_t-\mathbf{x}_{t-1}\|$ 衡量两序列差距，证明只要 $e_t\lesssim G/L$ 就能借局部光滑性推出 $\|\nabla f(\mathbf{y}_t)\|_*\lesssim G$；再用"时间分割"技巧（Lemma F.3）把轨迹按阈值 $\tau$ 切两段——$t\le\tau$ 时用收缩映射限制 $e_t$ 增长，$t>\tau$ 时证 $e_t$ 双曲衰减。相比 Li et al. 需要把学习率压到 $\eta\simeq 1/L^2$ 并加额外稳定化序列，本文用 $\eta_t=t\eta/(2L)$ 不引入新组件就拿到 $O(1/T^2)$。同款"距离量受控→局部 $L$-smooth→标准估计"的模式再扩展到随机 MD（用"事件链"分析末步，得高概率 $O(\sqrt{\log T}/\sqrt{T})$）和复合非凸优化（梯度映射 $O(1/T)$），把整条非欧广义光滑性脉络打通成一个可复用框架。

### 损失函数 / 训练策略

理论文章不涉及训练损失；关键学习率配置为：标准 MD $\eta\le 1/L$、Mirror Prox $\eta\le 1/(2L)$、乐观 MD $\eta\le 1/(3L)$、加速 MD 中 $L:=\ell(4G)$（需要稍大）。Assumption 3.3 要求 $\ell$ 次二次（$\lim_{\alpha\to\infty}\alpha^2/\ell(\alpha)=\infty$），这是为了保证关于 $G$ 的方程有有限解。

## 实验关键数据

### 主实验

理论论文的"实验"主要是经验性地验证 $\ell*$-smoothness 在真实模型上成立。作者用 Riabinin et al. (2025) 的层级近似公式 $\|\nabla_i f(X)-\nabla_i f(Y)\|_{(i)*}/\|X_i-Y_i\|_{(i)}\le L_i^0+L_i^1\|\nabla_i f(X)\|_{(i)*}$ 在多个网络上估计有效光滑常数。

| 设定 | 模型 / 数据集 | 测量内容 | 现象 |
|------|--------------|---------|------|
| LLM 预训练 | GPT-2 small/medium/large + FineWeb | 实际层级 $L_i$ vs $L_i^0+L_i^1\|\nabla\|$ 近似 | 高度吻合，证 $\ell*$-smooth 成立 |
| 翻译 | 6 层 Transformer + WMT'16 | $\ell_1$ 局部曲率 vs $\ell_\infty$ 梯度 | 曲率随梯度增长，符合广义光滑 |
| CV | CNN + CIFAR-10 | 全 batch 与 mini-batch 估计 | 两种估计一致，CNN 同样满足 |

### 收敛率对比

| 算法 | 凸性 | 经典 $L$-smooth 率 | 本文 $\ell*$-smooth 率 | 类型 |
|------|------|-------------------|----------------------|------|
| Mirror Descent | 凸 | $O(1/T)$ | $O(1/T)$ (Thm 3.5) | 均值/末值 |
| Accelerated MD | 凸 | $O(1/T^2)$ | $O(1/T^2)$ (Thm 3.7) | 末值 |
| Optimistic MD | 凸 | $O(1/T)$ | $O(1/T)$ (Eq 13) | 均值 |
| Mirror Prox | 凸 | $O(1/T)$ | $O(1/T)$ (Eq 15) | 均值 |
| Stochastic MD | 凸 | $\widetilde O(1/\sqrt T)$ | $\widetilde O(1/\sqrt T)$ (Thm 4.2) | 末值（高概率） |
| Composite MD | 非凸 | $O(1/T)$ | $O(1/T)$ (Thm 5.1) | 梯度映射 |

### 关键发现
- 维度增益的存在性：在 $f(\mathbf{x})=(\mathbf{1}_n^\top\mathbf{x})^4/4$ 上 $\ell*$-smooth ($\ell_1$ 版) 比 $\ell$-smooth ($\ell_2$ 版) 小 $\sqrt{n}$ 因子；附录 C 给出 $O(1/n)$、$O(\sqrt{\log n/n})$、$O(n^{-0.4})$ 等更激进的例子，直接说明"几何选对了能省维度代价"。
- 广义自界引理是分析的真正发动机：把"梯度受控"问题翻译成"间隙受控"，避免直接追踪对偶范数序列。
- 加速 MD 不再需要 Li et al. (2023) 的 $\eta\simeq 1/L^2$ 缩水学习率与额外稳定化序列，分析反而更简洁，验证了"非欧视角下的方法不仅更紧、流程也更直观"。

## 亮点与洞察
- **几何意识带来的紧致性**：把 Hessian 映射放回它真正属于的对偶空间，是"看似细微其实致命"的修正——后续所有维度无关界的来源都在这一步。
- **"循环归纳"技巧值得复用**：先假设梯度有界推次最优间隙不增，再反过来用间隙不增证明梯度有界；通过严格的归纳 (Lemmas D.3, E.2, 3.5) 把循环切开。这一套"假设-归纳-自洽"的技巧可以迁移到任何"梯度上界与目标函数下降相互依赖"的场景。
- **时间分割技巧 (time partition)**：用阈值 $\tau$ 把轨迹切两段，前半段用收缩映射限制 $e_t$ 增长、后半段证它双曲衰减——比直接给闭式估计要轻得多，可以借鉴到带辅助序列的其他加速方法分析。

## 局限与展望
- 假设 $\ell$ 次二次（$\lim\alpha^2/\ell(\alpha)=\infty$），排除了梯度爆炸更剧烈的情形；松绑这条假设可能需要更精细的 prox 步控制。
- 实验只用层级近似 (5) 验证 $\ell*$-smoothness 是否成立，并未直接跑 MD 与梯度下降做端到端 LLM 训练的对比；理论与"何时该选 MD"之间还隔一层。
- 全文聚焦凸 / 弱凸场景；现代 LLM 训练几乎全是非凸非光滑的，把广义自界引理彻底搬到一般非凸场景仍是开放问题。
- 加速 MD 的 $L:=\ell(4G)$ 比标准 MD 大一倍，常数因子有继续收紧的空间。

## 相关工作与启发
- **vs Zhang et al. (2020) $(L_0,L_1)$-smoothness**：他们用仿射函数刻画 Hessian 范数、固定 $\ell_2$ 几何，仅分析梯度裁剪；本文用任意非降次二次 $\ell$ + 任意范数，把"梯度裁剪做对了"扩展到"几何选对了"。
- **vs Li et al. (2023) $\ell$-smoothness**：他们的 $\ell$-smooth 是本文在 $\|\cdot\|=\|\cdot\|_2$ 时的特例，且加速分析需要 $\eta\simeq 1/L^2$ 与额外辅助序列；本文不需要这些"补丁"且常数更小。
- **vs Riabinin et al. (2025) 层级非欧光滑模型**：他们提出层级近似 (5)，本文证明 $\ell*$-smoothness 包含 (5) 作为特例并把整套 MD 族算法的收敛理论搭出来。
- **对 Muon / Scion / 现代 LLM 优化器的启示**：这些方法都隐式选择了非欧几何（如奇异值范数、列范数），本文给它们提供了第一份"基础几何理论包"，为未来层级混合范数优化器的收敛分析提供模板。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把广义光滑性彻底推广到非欧几何并打通全套 MD 变体的工作
- 实验充分度: ⭐⭐⭐ 理论文章，仅做经验验证 $\ell*$-smoothness 成立，无端到端 MD vs GD 对比
- 写作质量: ⭐⭐⭐⭐ 动机层层递进，关键引理与分析顺序清晰
- 价值: ⭐⭐⭐⭐⭐ 给 Muon、Scion 等非欧 LLM 优化器提供"该有的理论母板"，是这条研究线的基石性进展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](../../ICML2025/optimization/learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICML 2026\] Mirror Mean-Field Langevin Dynamics](mirror_mean-field_langevin_dynamics.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
