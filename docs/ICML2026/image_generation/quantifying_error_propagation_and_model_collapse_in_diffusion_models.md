---
title: >-
  [论文解读] Quantifying Error Propagation and Model Collapse in Diffusion Models
description: >-
  [ICML 2026][图像生成][扩散模型] 本文在 score-based 扩散模型上对"用合成数据递归训练导致 model collapse"这一现象给出第一套配对的上下界：单代散度 $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$…
tags:
  - "ICML 2026"
  - "图像生成"
  - "扩散模型"
  - "模型坍塌"
  - "递归训练"
  - "score matching"
  - "误差累积"
---

# Quantifying Error Propagation and Model Collapse in Diffusion Models

**会议**: ICML 2026  
**arXiv**: [2602.16601](https://arxiv.org/abs/2602.16601)  
**代码**: 无  
**领域**: 扩散模型 / 生成模型理论  
**关键词**: 扩散模型, 模型坍塌, 递归训练, score matching, 误差累积

## 一句话总结
本文在 score-based 扩散模型上对"用合成数据递归训练导致 model collapse"这一现象给出第一套配对的上下界：单代散度 $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$，多代累积散度 $D_N$ 是过去各代 score 误差能量按 $(1-\alpha)^{2m}$ 几何衰减的加权和，从而把"加新鲜数据能缓解坍塌"这一经验事实化成了精确的衰减律。

## 研究背景与动机
**领域现状**：当前生成式 AI 越来越依赖合成数据自训练（self-training、self-improving diffusion 等），但已被反复观察到：一旦训练分布里合成数据占比过高，模型在多轮递归后会显著退化——尾部丢失、多样性塌缩、整体分布漂移，这一现象统称 model collapse。理论侧的工作大多停留在回归模型或参数化 MLE 估计上，针对扩散模型只有少量两层 score network 的架构特定上界（Fu et al. 2024; Cui et al. 2026），且**只有上界**。

**现有痛点**：纯上界回答不了"误差最少会有多大"的问题——上界为零并不意味着模型不坍塌；同时上界依赖学习侧路径能量 $\hat\varepsilon_i^2$，无法直接和训练目标（ideal-path 上的 score-matching loss $\varepsilon_{\star,i}^2$）挂钩，导致理论和实验观测之间脱节。此外，已有理论几乎不涉及"新鲜数据比例 $\alpha$ 如何精确地抑制误差"这一最实用的旋钮。

**核心矛盾**：每一代训练同时存在两种相反力量——新鲜真实数据带来的"误差稀释"和不完美 score 学习带来的"误差注入"。要量化坍塌，必须同时刻画这两者，并把它们解耦到一个可解释的递推里。

**本文目标**：在 population level、与具体网络架构无关的情形下，针对递归训练管道 $\hat p^i \to q_i = \alpha p_{\text{data}} + (1-\alpha)\hat p^i \to \hat p^{i+1}$，回答两个子问题——(a) 单代散度 $I_i = \chi^2(\hat p^{i+1}\|q_i)$ 如何被 score 误差刻画？(b) 多代累积散度 $D_i = \chi^2(\hat p^i\|p_{\text{data}})$ 如何随代数演化？

**切入角度**：作者用 Girsanov 测度变换把反向 SDE 的 drift 误差打到路径测度的 Radon-Nikodym 导数上，再投影到终止时刻 $t_0$ 的边际似然比 $R_i(\mathbf x) = \mathbb E_{\mathbb P^\star_i}[e^{Z_T^i}\mid \mathbf Y_{t_0}=\mathbf x]$；问题就变成"路径误差有多少被边际化时保留下来"——这正好是一个可观测性（observability）问题。

**核心 idea**：引入"误差可观测性系数" $\eta_i\in[0,1]$ 度量路径 score 误差有多少落到终止状态上，于是单代下界变成 $I_i\gtrsim \eta_i\cdot\varepsilon_{\star,i}^2$；再把这个单代估计沿 $\chi^2$ 散度对刷新步 $q_i=\alpha p_{\text{data}}+(1-\alpha)\hat p^i$ 的精确收缩公式 $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2 \chi^2(\hat p^i\|p_{\text{data}})$ 做累加，就自然得出几何折扣的 $(1-\alpha)^{2m}$ 衰减律。

## 方法详解

### 整体框架
理论框架围绕递归训练 $\hat p^i \xrightarrow{\text{mix}} q_i \xrightarrow{\text{train}} \hat p^{i+1}$ 展开，分两层：(1) 单代分析——给定当代 score 误差能量 $\varepsilon_{\star,i}^2$，用 Girsanov + 一个新的"可观测性"系数把单代散度 $I_i$ 上下双向夹住；(2) 多代累加——把单代结果代入刷新步的 $\chi^2$ 收缩关系，得到 $D_N$ 的几何折扣分解。两层都在"小 score 误差摄动"区域 $\varepsilon_{\star,i}^2\le 1$ 内严格成立。

### 关键设计

1. **可观测性系数 $\eta_i$**：

    - 功能：度量路径上的 score 误差有多少会"现身"到反向扩散终止时刻 $t_0$ 的边际分布上，是把路径能量翻译成终止散度的桥梁。
    - 核心思路：定义随机变量 $M_T^i = -\int_{t_0}^T \mathbf e_{i,s}\cdot \mathrm d\bar{\mathbf B}_s$ （路径上 score 误差与反向 Brownian 的随机积分），由 Itô isometry 知 $\mathrm{Var}_{\mathbb P^\star_i}(M_T^i)=\varepsilon_{\star,i}^2$；令 $\eta_i = \mathrm{Var}_{\mathbb P^\star_i}(\mathbb E[M_T^i\mid \mathbf Y_{t_0}]) / \varepsilon_{\star,i}^2 \in [0,1]$。直觉：与样本状态耦合的扰动（如 $\mathbf e_{i,t}(\mathbf x)=\mathbf w\mathbf x+\xi(t)$）会在终止状态留下印记，$\eta_i>0$；纯时间相关或路径正交的扰动则会被条件期望平均掉，$\eta_i=0$。
    - 设计动机：路径 score 误差不为零并不必然导致边际散度不为零（这是扩散模型分析里的本质难点），过去只能靠上界绕过；引入 $\eta_i$ 后，下界 $I_i\ge \tfrac14\eta_i\varepsilon_{\star,i}^2 - C\varepsilon_{\star,i}^4$ 第一次把"路径误差"和"边际散度"直接连起来，并能在 CIFAR-10 等真实数据上数值验证 $\eta_i>0$ 几乎总成立。

2. **Girsanov 双向夹的单代等价 $I_i\asymp \varepsilon_{\star,i}^2$ (Theorem 3.5)**：

    - 功能：在小 score 误差区域内把 $I_i = \chi^2(\hat p^{i+1}\|q_i)$ 用 ideal-path 上的 score matching loss $\varepsilon_{\star,i}^2$ 双侧夹住，给出工程上可直接监控的代理量。
    - 核心思路：上界沿用 Girsanov + data processing 得到 $\mathrm{KL}(\hat p^{i+1}\|q_i)\le \tfrac12\hat\varepsilon_i^2$；下界来自上述可观测性论证 $\chi^2(\hat p^{i+1}\|q_i)\ge \tfrac14\eta_i\varepsilon_{\star,i}^2-C\varepsilon_{\star,i}^4$。关键技术是证明 ideal-path 能量 $\varepsilon_{\star,i}^2$ 与 learned-path 能量 $\hat\varepsilon_i^2$ 在 Girsanov 密度 $L^{1+\delta}$-可积假设 A3、二次变差矩条件 A4 下相互等价，同时 $\chi^2$ 与 KL 在摄动区也只差常数。合起来得到 $\tfrac14\eta_i\varepsilon_{\star,i}^2 - C\varepsilon_{\star,i}^4 \le \chi^2(\hat p^{i+1}\|q_i)\le 4\varepsilon_{\star,i}^2 + c\varepsilon_{\star,i}^4$。
    - 设计动机：以往结果要么只有 KL 上界，要么用 learned-path 能量（实践中拿不到），现在双向都用 ideal-path 能量表达，正好对应训练目标，使理论可被实验直接验证（Figure 4 中 10D GMM 上 $\chi^2$ 和 KL 两个散度都被 $\varepsilon_{\star,i}^2$ 上下夹住）。

3. **多代几何折扣分解 $D_N \asymp \sum (1-\alpha)^{2(N-i)}\varepsilon_{\star,i}^2$ (Theorem 4.2)**：

    - 功能：把 $N$ 代之后累积散度精确拆成各代 score 误差能量按几何系数衰减的加权和，给"为什么加新鲜数据能阻止坍塌"一个定量答案。
    - 核心思路：刷新步的 $\chi^2$ 散度满足精确等式 $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2\chi^2(\hat p^i\|p_{\text{data}})$ (Lemma F.1)——这是 $\chi^2$ 选择 $\chi^2$ 而非 KL 的关键代数优势；与单代等价 $\chi^2(\hat p^{i+1}\|q_i)\asymp \varepsilon_{\star,i}^2$ 一起做递推，再加一个自适应"良好集" $\mathcal G_i$ 上的尾部假设 A5（防止合成模型在 $p_{\text{data}}$ 极小的区域放大量质量），即可得 $D_{N+1}+C_{\text{bias}}\asymp \sum_{i=i_0}^N (1-\alpha)^{2(N-i)}\varepsilon_{\star,i}^2 + (1-\alpha)^{2(N+1-i_0)}D_{i_0}$。同时 Proposition 4.1 给出反向二分：若 $\sum_i \varepsilon_{\star,i}^2=\infty$ 或存在 score-error 下界，则 $\limsup D_i$ 不会消失，模型必坍塌。
    - 设计动机：把"$\alpha$ 越大 → 抑制越强"的经验事实化成显式 $(1-\alpha)^{2m}$ 衰减律——$m$ 代以前的误差被压缩 $(1-\alpha)^{2m}$ 倍，等价于一个 effective memory $\sim 1/\alpha$；这同时给出工程指导：要让 $D_N$ 稳定，只需 $\sum \varepsilon_{\star,i}^2<\infty$ 且 $\alpha>0$，无需每代误差都收敛到 0。

### 损失函数 / 训练策略
没有提出新的训练损失。理论建立在标准 variance-preserving OU 前向 SDE $\mathrm d\mathbf X_t = -\tfrac12\mathbf X_t\mathrm dt + \mathrm d\mathbf B_t$ 和反向 SDE 之上，使用 score matching loss $\varepsilon_{\star,i}^2 = \mathbb E_{\mathbb P^\star_i}[\int_{t_0}^T \|\mathbf e_{i,s}(\mathbf Y_s)\|_2^2 \mathrm ds]$。Minimax-optimal score 估计误差满足 $\varepsilon_{\star,i}^2 \lesssim \mathrm{polylog}(n_i)\,n_i^{-1}(1/t_0)^{d/2}$，即样本量在环境维 $d$ 中指数增长才能保证摄动区成立；但在低维流形假设下只需依赖内在维 $d^\star\ll d$。

## 实验关键数据

### 主实验
作者用 10 维高斯混合（5 分量，$\sigma^2\mathbf I_{10}$）、Fashion-MNIST 与 CIFAR-10 三个数据集验证理论。所有实验都用 PCA 投影到二维做可视化，并直接估计 $\eta_i$、$\varepsilon_{\star,i}^2$、$\chi^2$/KL 散度。

| 数据集 | 验证目标 | 结果 |
|--------|---------|------|
| 10D Gaussian Mixture ($\alpha\in\{0.1,0.5,0.9\}$, 20 代) | $\alpha$ 对坍塌速度的影响 (Fig.1) | $\alpha=0.1$ 时分布持续扩散；$\alpha=0.5$ 保留结构但有展宽；$\alpha=0.9$ 全程稳定 |
| 10D GMM (20 代) | 单代上下界 (Prop 3.1 + 3.3) | $\mathrm{KL}(\hat p^{i+1}\|q_i)\le \tfrac12\hat\varepsilon_i^2$ 紧贴；$\chi^2\ge \tfrac18\hat\eta_i\varepsilon_{\star,i}^2$ 验证 (Fig.3) |
| 10D GMM (20 代, $\alpha\in\{0.1,0.5\}$) | 双侧等价 Thm 3.5 | $\chi^2$ 和 KL 都被 $\tfrac14\hat\eta_i\varepsilon_{\star,i}^2$ 和 $4\varepsilon_{\star,i}^2$ 上下夹住 (Fig.4) |
| 10D GMM (20 代) | 几何折扣分解 Thm 4.2 | $\alpha=0.1$ 宽带贡献，$\alpha=0.9$ 仅最近代贡献，对角化结构明显 (Fig.5) |
| Fashion-MNIST | 真实图像下的 $\alpha$ 效应 (Fig.8/10) | 与 GMM 结论一致：高 $\alpha$ 稳定，低 $\alpha$ 多代后坍塌 |
| CIFAR-10 | 可观测性 $\eta_i$ 在真实数据上的存在性 (Fig.2/9) | 状态相关扰动（aligned / random）给出明显 $\hat\eta_i>0$；纯时间扰动 $\hat\eta_i\approx 0$ |

### 消融实验
论文没有传统意义上的模块消融（纯理论 + 验证实验），但 Fig.2 在 CIFAR-10 上对 score 误差类型做了"消融"，揭示 $\eta_i$ 的来源：

| 扰动类型 | $\mathbf e_{i,t}(\mathbf x)$ 形式 | 估计 $\hat\eta_i$ | 说明 |
|----------|-----------------------------------|-------------------|------|
| Aligned (与 drift 同向) | $\mathbf w_i \mathbf x$, $\mathbf w_i$ 沿 drift 方向 | 最高 | 误差被反向轨迹放大，强烈印记到终止状态 |
| Random (随机方向) | $\mathbf w\mathbf x$, $\mathbf w$ 随机 | 中等 | 仍具状态依赖，$\eta_i$ 显著大于 0 |
| Time-only | $\xi(t)$，与状态无关 | 接近 0 | 条件期望把它平均掉，对边际无影响 |

### 关键发现
- **$(1-\alpha)^{2m}$ 衰减律是核心可操作结论**：要让累积散度稳定，工程师不需要让每代 score 误差都趋零，只需 $\sum \varepsilon_{\star,i}^2<\infty$ 且新鲜数据比例 $\alpha>0$，等效 memory window 约为 $1/\alpha$ 代。
- **状态依赖性决定坍塌可见度**：纯时间扰动不会带来观测到的散度增长（$\eta_i\approx 0$），但实际神经网络 score model 的随机初始化和优化噪声几乎必然带来状态依赖扰动，所以坍塌在实践中是"通用现象"。
- **$\chi^2$ 散度是关键技术选择**：刷新步上 $\chi^2(q_i\|p_{\text{data}})=(1-\alpha)^2\chi^2(\hat p^i\|p_{\text{data}})$ 是精确等式（KL 没有这种性质），这是几何折扣递推能成立的代数根源；摄动区内 $\chi^2$ 与 KL 等价（Fig.4 也验证了），所以下界结论也能转回 KL。
- **第一阶段（$i<i_0$）允许偏离**：理论假设在前 $i_0$ 代之后成立，Fig.3 中 $i=1$ 的偏差不违背理论，恰好对应"瞬态"——实践中通常一两代后就稳定。

## 亮点与洞察
- **观测性系数 $\eta_i$ 的提出是真正的新东西**：把扩散模型分析里悬而未决的"路径误差→边际散度"鸿沟用一个单一标量量化，并能在数据上估出来——这是这篇论文最让人"啊哈"的设计。
- **$\chi^2$ 散度的精妙选择**：作者刻意选 $\chi^2$ 而非 KL，正是因为刷新步 $q_i=\alpha p_{\text{data}}+(1-\alpha)\hat p^i$ 在 $\chi^2$ 下满足干净的二次收缩；这一观察对其他"混合-再训练"框架（如 RLHF 中混合 reference policy）也直接可迁移。
- **架构无关的 population-level 视角**：现有理论几乎都假设两层 score network，本文完全跳过 architecture-specific 论证，从路径测度层面入手——这种"先化简到 SDE 测度比，再用 Girsanov 投影"的策略可直接套到 flow matching、consistency models 等其他 score-flavor 生成模型上。
- **可被实验直接验证的下界**：与多数纯理论工作不同，所有定理都给出可估的常数和量，作者在 GMM/Fashion-MNIST/CIFAR-10 上把估计的 $\hat\eta_i$ 和 $\hat\varepsilon_{\star,i}^2$ 代入边界即可对比真实散度——这种"理论-实验闭环"在 model collapse 文献里非常少见。

## 局限与展望
- **仅适用于小 score 误差摄动区**：所有结果都假设 $\varepsilon_{\star,i}^2\le 1$，但式 (12) 表明高维数据需要样本量 $n_i\sim (1/t_0)^{d/2}$ 才能达到这一区域，实际大模型很可能在这之外。作者也明确把"大误差区域的下界"列为开放问题。
- **忽略离散化误差与初始化偏差**：理论用连续时间反向 SDE 并假设从 $\mathcal N(0,\mathbf I_d)$ 起始（用 OU 指数收敛吞掉），实际扩散模型用离散步数 sampler，这部分误差未量化。
- **可观测性 $\eta_i$ 的下界缺乏先验保证**：论文只能说"实践中几乎都 $>0$"（CIFAR-10 上验证），但对更复杂架构（如 transformer-based DiT、flow matching）何时 $\eta_i$ 会塌缩到 0 没有理论判据，这是把理论用到 SOTA 模型的关键缺口。
- **没回答"是否有极限分布"**：作者明确指出"递归训练是否收敛到某个 $\alpha$-依赖的极限分布"仍是开放问题；当前结论只能保证 $D_i$ 有界或非有界，不刻画极限形态。
- **可改进方向**：把 $\eta_i$ 与 score network 架构、激活函数、初始化分布显式挂钩，或许能给出"哪种架构更抗坍塌"的工程指导；另外把分析推广到 multi-stage 训练（不同生成代用不同 mixing ratio $\alpha_k$）可能直接服务真实 self-improving pipeline。

## 相关工作与启发
- **vs Fu et al. (2024) / Cui et al. (2026)**: 他们针对两层 score network 给出 architecture-specific 上界，本文做 architecture-agnostic 的 population-level 上下界——补上了 lower bound 缺口，覆盖范围更广，但 finite-sample 速率不如他们具体。
- **vs Bertrand et al. (2024)**: 同样研究递归生成模型稳定性，但他们走的是 MLE 路线（针对参数化分布），本文专注 score-based 扩散，技术上完全不同（Girsanov 而非 fixed point on parameter space）。
- **vs Gerstgrasser et al. (2024)**: 他们实证发现"累积真实+合成数据可阻断坍塌"，本文给出该现象的定量解释 $(1-\alpha)^{2m}$ 衰减律——把经验观察化为可计算的衰减速率。
- **vs Chen et al. (2023c) / Benton et al. (2024)**: 这些是扩散采样收敛的经典 KL 上界，本文沿用同样的 Girsanov 工具但反过来给出 $\chi^2$ 下界——是对该工具链的对偶应用，思路可迁移到任何"单步 SDE 误差→边际散度"分析。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个扩散模型的 distribution divergence 下界 + 可观测性系数 + 几何折扣分解，三件套都是 model collapse 文献里没有的新东西。
- 实验充分度: ⭐⭐⭐ 纯理论 + 验证实验，10D GMM/Fashion-MNIST/CIFAR-10 都做了，对一篇理论 paper 够用，但缺乏与 SOTA self-consuming 实验设置的对齐。
- 写作质量: ⭐⭐⭐⭐ 结构清晰，假设 A1-A5 逐条解释合理性，把 Girsanov 论证拆成 observability + ratio control 两个独立挑战的方式很有教学价值。
- 价值: ⭐⭐⭐⭐ 给出可执行的工程结论（$\alpha>0$ + 误差能量可和即可稳定）和清晰的开放问题（大误差区下界、离散化误差、极限分布），对扩散模型理论和 self-training 设计都有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)
- [\[ICML 2026\] Alignment-Guided Score Matching for Text-to-Image Alignment in Diffusion Models](alignment-guided_score_matching_for_text-to-image_alignment_in_diffusion_models.md)
- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICLR 2026\] Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance](../../ICLR2026/image_generation/error_as_signal_stiffness-aware_diffusion_sampling_via_embedded_runge-kutta_guid.md)

</div>

<!-- RELATED:END -->
