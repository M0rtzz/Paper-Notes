---
title: >-
  [论文解读] Convergence of Steepest Descent and Adam under Non-Uniform Smoothness
description: >-
  [ICML 2026][强化学习][非均匀光滑] 本文提出一种比 Zhang 等 $(L_0,L_1)$-NS 更广的非均匀光滑性 $(H_0,H_1)$-NS，并在此假设和 (非均匀) Łojasiewicz 条件下首次给出确定性对角 RMSProp / Adam 与一般最速下降 (Sign GD、Norm…
tags:
  - "ICML 2026"
  - "强化学习"
  - "非均匀光滑"
  - "最速下降"
  - "Adam"
  - "RMSProp"
  - "Łojasiewicz 条件"
---

# Convergence of Steepest Descent and Adam under Non-Uniform Smoothness

**会议**: ICML 2026  
**arXiv**: [2605.30648](https://arxiv.org/abs/2605.30648)  
**代码**: 无  
**领域**: 优化理论  
**关键词**: 非均匀光滑, 最速下降, Adam, RMSProp, Łojasiewicz 条件

## 一句话总结
本文提出一种比 Zhang 等 $(L_0,L_1)$-NS 更广的非均匀光滑性 $(H_0,H_1)$-NS，并在此假设和 (非均匀) Łojasiewicz 条件下首次给出确定性对角 RMSProp / Adam 与一般最速下降 (Sign GD、Norm.GD、Sign CD-GS) 的统一收敛率，证明在分离数据上的逻辑回归与 softmax 策略梯度上它们比 GD / AdaGrad / heavy-ball 都严格更快。

## 研究背景与动机

**领域现状**：经典一阶法收敛分析依赖全局一致光滑——Hessian 谱范数被常数 $L$ 上界，这与神经网络训练实际的损失曲面相差甚远。近年来 Zhang 等 (2020b) 提出 $(L_0,L_1)$-NS：$\|\nabla^2 f(\theta)\|_2\le L_0+L_1\|\nabla f(\theta)\|_2$，并实证显示训练损失满足该条件，由此解释梯度裁剪有效性。Vaswani & Harikandeh (2025)、Alimisis et al. (2025) 进一步将上界改写为关于函数值 $f(\theta)$ 的仿射函数，使条件在有限和、仿射变换下闭合。

**现有痛点**：(1) 已有 $(L_0,L_1)$-NS 分析多限于 $\ell_2$ 范数与 GD/heavy-ball，对应不上 Adam/RMSProp 与符号梯度法 (Sign GD)，而后者是实际训练中 LLM 优化器的核心；(2) 对 Adam/RMSProp 的现有理论 (Li et al. 2023b；Wang et al. 2024a/b) 都假设有界梯度或加额外凸性，且无法给出与 GD 严格分离的下界；(3) 分离数据上的逻辑回归、softmax 策略梯度等关键应用问题不被一致光滑覆盖，使经典 GD 收敛率退化为 $O(1/\epsilon)$ 次线性。

**核心矛盾**：要在一个统一框架内同时容纳 (a) 多种对偶范数 $(p,q)$ 下的最速下降 (含 Sign GD)、(b) 自适应方法 Adam/RMSProp、(c) 非凸两层网络与策略梯度等无界损失曲面，必须找到一个既能保证下降不等式成立、又能区分 Adam 与 AdaGrad 的非均匀光滑刻画。

**本文目标**：定义并研究 $(H_0,H_1)$-NS——Hessian 的 $(p,q)$ 算子范数被 $H_0+H_1 f(\theta)$ 控制；结合非均匀 Łojasiewicz (NL) 条件，给出 NSD、RMSProp、Adam 在该类问题上的统一收敛率，并证明 RMSProp/Adam 相对 GD/AdaGrad/AMSGrad/heavy-ball 的可证更快性。

**切入角度**：作者发现"$f$ 的 Hessian 被 $f(\theta)$ 仿射上界"自动蕴含对函数和梯度都是"乘性 Lipschitz" (multiplicative Lipschitz)，即在 $\ell_p$ 小邻域内 $f$ 与 $\|\nabla f\|_q$ 的"放大倍数"被指数因子有界——这一性质恰好补齐了 Adam/RMSProp 分析中"无法跨步比较梯度"的拼图。

**核心 idea**：用"$(H_0,H_1)$-NS ⇒ 乘性 Lipschitz ⇒ 跨步梯度比有界"替换传统 Adam/RMSProp 分析中"梯度有界"的强假设，并把 Sign GD 视为 $(p,q)=(\infty,1)$ 的最速下降特例，从而把 NSD 与 Adam 系列统一在同一证明框架内。

## 方法详解

### 整体框架

研究无约束最小化 $\min_{\theta\in\mathbb{R}^D} f(\theta)$，其中 $f$ 二阶可微且非负。核心假设包括：

- **$(H_0,H_1)$-NS** (Assn. 2)：对一对对偶 $(p,q)$ 满足 $1/p+1/q=1$，$\|\nabla^2 f(\theta)\|_{p\to q}\le H_0+H_1 f(\theta)$。
- **(非均匀) Łojasiewicz NL** (Assn. 3)：存在 $\tau\in(0,1]$，$\|\nabla f(\theta)\|_q\ge\mu(\theta)[f(\theta)-f^*]^\tau$。

这两条假设覆盖 Prop. 1–4：分离数据下的指数损失/逻辑回归 (Assn. 2 $H_0=0,H_1=\max_i\|x_i\|_q^2$；Assn. 3 $\tau=1,\mu=\gamma_p$)、softmax 策略梯度 (Prop. 3：$H_0=0,H_1\le 24$；$\tau=1,\mu=\pi_\theta(a^*)$)、特定两层网络 (Prop. 4) 以及带逻辑链的 GLM。算法层面统一为 NSD 更新 $\theta_{t+1}=\theta_t-\eta_t d_t$，$d_t=\arg\max_{\|d\|_p\le 1}\langle d,\nabla f(\theta_t)\rangle$，三个特例分别给出 Sign GD ($p=\infty$)、Norm.GD ($p=2$)、Sign CD-GS ($p=1$)。

### 关键设计

1. **$(H_0,H_1)$-NS 的结构性质：乘性 Lipschitz 与非均匀下降引理**:

    - 功能：把"Hessian 被函数值控住"翻译成可在算法分析中直接套用的局部比较不等式。
    - 核心思路：Lemma 3 给出梯度的函数值界 $\|\nabla f(\theta)\|_q\le\sqrt{2H_0 f(\theta)+H_1[f(\theta)]^2}$；Lemma 5 证明 "shifted $f$" 的乘性 Lipschitz：当 $H_1>0$，$(f(y)+H_0/H_1)\le(f(x)+H_0/H_1)\exp(\sqrt{H_1}\|y-x\|_p)$；Lemma 6/10 把它升级为"函数与梯度范数都满足跨步比有界"；最后 Lemma 给出非均匀下降不等式 (Eq. 13)：当 $\|y-x\|_p\le 1/\sqrt{H_1}$，$f(y)\le f(x)+\langle\nabla f(x),y-x\rangle+(H_0+H_1 f(x))\|y-x\|_p^2$。
    - 设计动机：传统下降引理要求一致光滑常数 $L$；这里把 $L$ 换成 $H_0+H_1 f(x)$，自动适配 $f$ 大、Hessian 也大的训练初期。这一不等式是后面 NSD/RMSProp/Adam 所有收敛证明的统一起点。

2. **NSD 两阶段收敛 (Theorem 1)**:

    - 功能：在 $(H_0,H_1)$-NS + NL ($\tau,\mu$) 下给出 NSD 收敛率，覆盖 Sign GD、Norm.GD、Sign CD-GS。
    - 核心思路：把上面的非均匀下降不等式带入 NSD 更新，得 $f(\theta_{t+1})\le f(\theta_t)-\eta\mu[f(\theta_t)]^\tau+(H_0+H_1 f(\theta_t))\eta^2$。证明分两段：Phase 1 $f(\theta_t)\ge\max\{\epsilon,H_0/H_1\}$，可用 $H_0+H_1 f\le 2H_1 f$ 把递推化为线性收敛形式，得 $f$ 几何下降至 $\max\{\epsilon,H_0/H_1\}$；Phase 2 (仅当 $\epsilon<H_0/H_1$) 用 $H_0+H_1 f\le 2H_0$，需把步长缩小到 $\eta=O(\epsilon^\tau)$，得 $O(1/\epsilon^\tau)$ 慢相位。极端情形 $H_0=0$ (例如分离数据上的指数损失) 整个过程都在 Phase 1，常步长即可线性收敛。
    - 设计动机：传统 GD 在分离数据上只能 $O(1/\epsilon)$，本文把"次线性 GD vs 线性 NSD"的分离量化到任意 $\tau$；步长策略也从"需要 line-search"简化为"常步长 + $\eta=O(\epsilon^\tau)$"，对应实践中的 warm-up + 衰减。

3. **RMSProp / Adam 的跨步比分析 (Theorem 3 等)**:

    - 功能：在同样假设下给出 RMSProp ($v_{t,i}$ 指数滑动二阶矩、$d_t=g_t/\sqrt{v_t}$) 与 Adam (加一阶动量) 的确定性对角变体的收敛率。
    - 核心思路：取 $(p,q)=(\infty,1)$ 套用 Lemma 16 得 $\|d_t\|_\infty\le 1/\sqrt{1-\beta}$；进而 $f(\theta_{t+1})\le f(\theta_t)-\eta\langle\nabla_t,d_t\rangle+\bar L_t\eta^2/(1-\beta)$，其中 $\bar L_t=H_0+H_1 f(\theta_t)$。关键挑战是下界 $\langle\nabla_t,d_t\rangle=\sum_i g_{t,i}^2/\sqrt{v_{t,i}}$，本文 Lemma 17 用 Cauchy–Schwarz 与 $v_{t,i}$ 递推得 $\langle\nabla_t,d_t\rangle\ge\|\nabla_t\|_1\cdot\|\nabla_t\|_1/\bigl(\sqrt{1-\beta}\sum_{j=0}^{t-1}\sqrt{\beta}^j\|\nabla_{t-j}\|_1\bigr)$。这个 (*) 项把"自适应预条件"显化为"当前梯度 / 历史梯度的加权平均"；用乘性 Lipschitz (Eq. 12) 把分母中的 $\|\nabla_{t-j}\|_1$ 反向控制到 $\|\nabla_t\|_1+c$ 的指数倍，分离出与 NSD 同阶的线性下降项。Phase 1 / Phase 2 划分同 Theorem 1，得到 $\tau\le 1/2$ 时 $O(1/\epsilon^{2\tau})$、$\tau>1/2$ 时 $O(1/\epsilon^{4\tau-1})$ 的二阶段速率。Adam 把一阶矩动量纳入，分析框架不变。
    - 设计动机：传统 Adam 分析依赖 $\|\nabla\|\le G$，这在分离数据指数损失上不成立 (梯度可以为指数大)。本文用 $(H_0,H_1)$-NS 蕴含的乘性 Lipschitz 直接取代有界梯度假设，是这套证明能跑通的关键；同时这套技巧也能在普通 $(L_0,L_1)$-NS 非凸函数上 (Sec. 5.4) 给出比已有非凸 Adam 结果更快的确定性速率。

### 损失函数 / 训练策略
论文不涉及训练，给的是收敛速率与步长策略：NSD/RMSProp/Adam 在 $\epsilon>H_0/H_1$ 区段使用 $\eta=O(1)$ 常步长实现 $O(\ln(1/\epsilon))$ 线性收敛；在 $\epsilon<H_0/H_1$ 区段需要 $\eta=O(\epsilon^\tau)$ (NSD) 或 $\eta=O(\epsilon^{2\tau})$ (RMSProp/Adam) 才能进入慢相位。

## 实验关键数据

### 主实验 (理论速率对比 — Phase 1 关键速率)

| 问题 | 方法 | 步长 | 收敛速率 | 备注 |
|------|------|------|----------|------|
| 分离数据指数损失 | GD | const | $\Theta(1/\epsilon)$ | Soudry et al. 2018 |
| 同上 | Sign GD / Norm.GD (NSD) | const $O(1)$ | $O(\ln(1/\epsilon))$ | 本文 Theorem 1，**严格更快** |
| 分离数据逻辑回归 | GD with Armijo | line-search | $O(\ln(1/\epsilon))$ | Vaswani & Harikandeh 2025 |
| 同上 | NSD / Sign CD-GS | const | $O(n^2/\gamma_p^2+\ln(1/(n\epsilon)))$ | 本文 Theorem 2 |
| Softmax 策略梯度 (MAB) | GD | const | $\Omega(1/\epsilon)$ | Mei et al. 2020 |
| 同上 | NSD | const $O(1)$ | $O(\ln(1/\epsilon))$ | 本文 Corollary 1 |
| 二层网 + 分离数据 | RMSProp / Adam (det. diag.) | const + 常 $\beta$ | $O(\ln(1/\epsilon))$ 线性 | 本文 Theorem 3 |
| 一维逻辑损失 | GD / heavy-ball / AdaGrad / AMSGrad | 任意 | $\omega(\ln(1/\epsilon))$ 次线性下界 | 本文 Sec. 6，与 RMSProp/Adam 分离 |

### Phase 1 vs Phase 2 速率结构

| 方法 | $\epsilon>H_0/H_1$ (Phase 1) | $\epsilon<H_0/H_1$ (Phase 2, NL $\tau$) | 维度依赖 |
|------|------------------------------|------------------------------------------|----------|
| NSD (Theorem 1) | $O(\ln(1/\epsilon))$ | $O(1/\epsilon^\tau)$ | 无 |
| RMSProp (Theorem 3) | $O(\ln(1/\epsilon))$ | $\tau\le 1/2$: $O(1/\epsilon^{2\tau})$；$\tau>1/2$: $O(1/\epsilon^{4\tau-1})$ | 无 |
| GD / AdaGrad (1D logistic) | $\Theta(1/\epsilon)$ 次线性 | — | — |

### 关键发现
- 一旦 $H_0=0$ (例如指数损失、softmax 策略梯度)，最速下降与 RMSProp/Adam 全程处于 Phase 1，常步长即可线性收敛；只有当目标存在有限极小化点 ($H_0>0$) 时才进入需要缩小步长的 Phase 2，这给"训练后期需要 learning-rate decay"提供了理论解释。
- 一维逻辑损失上 RMSProp/Adam 与 GD/heavy-ball/AdaGrad/AMSGrad 之间存在首个可证速率分离 (Sec. 6)，理论印证了 RMSProp/Adam 在实践中相对 AdaGrad 的主导地位。
- 把 NSD 的特例与 Sign CD-GS 结合，证明 Sign CD-GS 这种"按 Gauss–Southwell 选坐标 + 符号方向更新"的极简算法也能在分离数据逻辑回归上线性收敛，匹配 Axiotis & Sviridenko (2023) 的规范化坐标下降率。

## 亮点与洞察
- "把光滑性挂到函数值上 ($H_0+H_1 f$)" 是个看似温和实则颠覆的换法：它让下降不等式自适应于训练阶段——损失大时容许 Hessian 大，损失小时自然紧缩，从而把"梯度裁剪/warm-up/衰减"等工程实践统一在一条数学路径下。
- "梯度跨步比的指数界" (Lemma 6/10) 是分析自适应方法 (Adam/RMSProp) 的通用利器：它把"分母里出现历史梯度"这一阻碍传统证明的障碍变成可控乘性常数；任何依赖二阶矩缓冲的优化器 (Lion、Tiger 等) 都有望复用这一框架。
- Sec. 6 的下界部分把"为什么实际中 RMSProp/Adam 普遍优于 AdaGrad"从经验观察推到可证分离；这种"上界 + 匹配下界"的论证模板对后续 LLM 优化器比较研究极具示范意义。

## 局限与展望
- 仅分析了 RMSProp/Adam 的**确定性对角**变体，未涉及随机梯度、矩阵预条件 (Shampoo、Sophia 等)；把跨步比技巧扩展到随机/全矩阵情形是自然下一步。
- $(H_0,H_1)$-NS 需要二阶可微，无法直接覆盖 ReLU 网络等非光滑模型，需要进一步弱化为弱凸或 generalized smoothness。
- NL 条件要求 $\mu(\theta)$ 在路径上不退化；对于深度过参数化网络，$\mu$ 实际依赖参数初始化，量化"NL 在多大半径内保持"是一个未解决问题。
- 论文中提到的"二层网络 + 平滑 leaky ReLU + 指数损失"假设仍较严格，扩展到 sigmoid/softplus 与多层网络需要新的 Hessian 控制工具。

## 相关工作与启发
- **vs Vaswani & Harikandeh (2025)**: 同样基于函数值版 NS 假设，但他们只分析 GD + Armijo line-search；本文把 NS 推广到任意 $(p,q)$ 对偶范数并覆盖 NSD/Adam/RMSProp，且证明常步长即可达到他们 line-search 的同阶速率。
- **vs Alimisis et al. (2025)**: 二者都聚焦"Hessian 被函数值仿射上界"，Alimisis 用于解释 LR warm-up；本文进一步把该条件用作 Adam/RMSProp 分析的桥梁，并新增 Sec. 6 的下界分离，研究目标从"解释训练现象"上升到"严格速率比较"。
- **vs Li et al. (2023b)、Wang et al. (2024a/b)**: 这些工作分析 $(L_0,L_1)$-NS 非凸函数上的 Adam 收敛，要求有界梯度且仅给出最优梯度范数收敛；本文不需有界梯度，并在凸 + NL 设定下进一步给出函数值线性收敛，Sec. 5.4 的非凸延伸也得到比他们更快的确定性速率。
- **vs Mei et al. (2020/2021)**: Mei 等给出 softmax 策略梯度 GD 的 $\Omega(1/\epsilon)$ 下界与 Norm.GD 的 $O(\ln(1/\epsilon))$ 上界；本文把后者推广到任意 NSD (含 Sign GD、Sign CD-GS) 并放进统一 NS+NL 框架，且证明 RMSProp/Adam 在同类问题上同样达到线性率。


## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games](../../ICLR2026/reinforcement_learning/on_the_o1t_convergence_of_alternating_gradient_descent-ascent_in_bilinear_games.md)
- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)
- [\[ICML 2026\] Convergence of Two-Timescale Markovian Stochastic Approximations with Applications in Reinforcement Learning](convergence_of_two-timescale_markovian_stochastic_approximations_with_applicatio.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICML 2026\] Tracking Drift: Variation-Aware Entropy Scheduling for Non-Stationary Reinforcement Learning](tracking_drift_variation-aware_entropy_scheduling_for_non-stationary_reinforceme.md)

</div>

<!-- RELATED:END -->
