---
title: >-
  [论文解读] Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems
description: >-
  [ICML2026][图像生成][零阶采样] 本文提出一种带方差缩减的零阶 Langevin 采样方法，用间歇性大 batch 估计和递推式小 batch 更新替代每步 $O(d)$ 次函数查询，并把它扩展为 ZO-APMC…
tags:
  - "ICML2026"
  - "图像生成"
  - "零阶采样"
  - "非对数凹分布"
  - "方差缩减"
  - "Langevin Monte Carlo"
  - "逆问题"
---

# Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems

**会议**: ICML2026  
**arXiv**: [2605.30573](https://arxiv.org/abs/2605.30573)  
**代码**: 无  
**领域**: 优化 / 采样算法 / 黑盒逆问题  
**关键词**: 零阶采样, 非对数凹分布, 方差缩减, Langevin Monte Carlo, 逆问题  

## 一句话总结
本文提出一种带方差缩减的零阶 Langevin 采样方法，用间歇性大 batch 估计和递推式小 batch 更新替代每步 $O(d)$ 次函数查询，并把它扩展为 ZO-APMC，用预训练 score-based prior 在只有前向模型查询的黑盒逆问题中做有收敛保证的后验采样。

## 研究背景与动机
**领域现状**：从未归一化密度 $\pi \propto \exp(-f)$ 采样是机器学习、贝叶斯推断和逆问题中的基础工具。若能访问 $\nabla f$，Langevin Monte Carlo 可以沿着目标分布的 score 迭代；若只能查询函数值，常见做法是用有限差分或随机方向构造 zeroth-order gradient estimator。

**现有痛点**：零阶估计在高维下方差很大。朴素 batched ZO-LMC 要让梯度估计足够准确，通常需要 batch size 随维度 $d$ 线性增长，这在 MRI、黑洞成像、PDE 反演等高维问题里意味着大量前向模型调用和内存开销。更麻烦的是，已有理论主要覆盖 strongly log-concave 目标，而 score-based generative prior 对应的真实后验往往是非对数凹、多模态的。

**核心矛盾**：黑盒逆问题最需要 posterior sampling 来表达不确定性，但黑盒 forward operator 往往没有可用梯度、伪逆或可微实现；直接零阶估计成本太高，启发式黑盒 posterior sampler 又缺少非渐近收敛保证。本文要同时解决“能跑得动”和“说得清楚为什么收敛”两个要求。

**本文目标**：第一，建立非对数凹目标下零阶采样的非渐近理论；第二，设计每步只需常数级函数查询的方差缩减估计器；第三，把该估计器嵌入带 SGM prior 的 annealed posterior sampler，使 MRI、黑洞成像和 Navier-Stokes 这类黑盒逆问题可以只靠 forward evaluations 采样。

**切入角度**：作者借用了非凸优化里“stationary point analysis”的思想，把采样中的相对 Fisher information 看作 Wasserstein 空间中 KL 的梯度范数。也就是说，不强求目标是 log-concave，而是证明时间平均分布在 FI 或 TV 意义下接近目标分布。

**核心 idea**：不要每一步都重新用大 batch 做零阶梯度，而是偶尔刷新一次大 batch 估计，其余时候用相邻 Langevin 迭代高度相关这一事实，用小 batch 估计梯度变化量。

## 方法详解
论文的方法有两层：基础层是 variance-reduced ZO-LMC，用于从非对数凹目标分布采样；应用层是 ZO-APMC，把这个估计器和 annealed score-based prior 结合，解决黑盒逆问题中的 posterior sampling。

### 整体框架
基础问题是只能查询 potential $f(x)$，不能计算 $\nabla f(x)$。朴素零阶估计器在随机方向 $u \sim \mathcal{N}(0,I)$ 上计算 $\tilde{\nabla} f_\mu(x,u)=((f(x+\mu u)-f(x))/\mu)u$，再把它放进 LMC 更新。本文替换的不是 LMC 结构，而是梯度估计器：用一个递推的 $g_k$ 估计平滑 potential 的梯度 $\nabla f_\mu(x_k)$，然后执行 $x_{k+1}=x_k-\gamma g_k+\sqrt{2}(B_{(k+1)\gamma}-B_{k\gamma})$。

在逆问题中，观测满足 $y=A(x)+\xi$，其中 $A$ 只能黑盒查询。后验可写为 likelihood 与 prior 的乘积，likelihood score 用零阶估计，prior score 用预训练 SGM $S_\theta(x,\sigma)$ 近似。ZO-APMC 在每个 annealing step 同时使用 $g_k$ 和带权重的 score prior，形成 $x_{k+1}=x_k-\gamma(g_k-\alpha_k S_\theta(x_k,\sigma_k))+$ Langevin noise。

### 关键设计
1. **方差缩减零阶梯度估计器**:

	- 功能：降低 ZO-LMC 每步的函数查询和内存成本，同时保持可分析的估计误差。
	- 核心思路：以概率 $p$ 计算一个 batch size 为 $b$ 的完整零阶估计；以概率 $1-p$ 使用上一轮估计 $g_{k-1}$，再用很小的 batch $b'$ 估计 $\tilde{\nabla}f_\mu(x_k,u)-\tilde{\nabla}f_\mu(x_{k-1},u)$。因为相邻 Langevin iterates 距离通常很近，这个差分的方差比直接估计梯度小。
	- 设计动机：朴素零阶估计靠增大 batch size 压方差，代价随维度变高；递推式差分把“每步重估绝对梯度”改为“跟踪梯度变化”，用相关性换计算量。

2. **以相对 Fisher information 做非对数凹采样分析**:

	- 功能：在不假设强 log-concavity 的情况下给出收敛刻画。
	- 核心思路：把 $FI(\nu\|\pi)$ 看成 KL 在 Wasserstein 空间中的梯度范数，类似非凸优化中用梯度范数定义 stationary point。Theorem 1 证明时间平均分布达到 $\varepsilon$-relative FI 误差需要 $O(d^7 L_m^4/\varepsilon^4)$ 次迭代，但每步函数查询是 $O(1)$；若目标满足 Poincare inequality，还可转成 squared TV distance 保证。
	- 设计动机：非对数凹分布可能多模态，传统强凸/强 log-concave 分析不适用。FI 提供了一个既能表达“score 已对齐”，又能导出 TV 收敛的中间指标。

3. **ZO-APMC 黑盒后验采样**:

	- 功能：在只有 forward model evaluations 的逆问题中，用 SGM prior 生成后验样本。
	- 核心思路：likelihood 由黑盒 forward operator 和观测定义，其梯度用方差缩减 ZO estimator；prior score 由预训练 SGM 在噪声尺度 $\sigma_k$ 上提供；annealing schedule 同时降低 prior smoothing level 和权重 $\alpha_k$，让采样先在平滑 prior 下逃离低概率平台，再逐渐强调真实 likelihood。
	- 设计动机：SGM prior 很强，但现有 posterior sampler 常默认能访问 forward gradient。ZO-APMC 让不可微、闭源、PDE simulator 或规则系统也能进入同一套 Bayesian reconstruction 框架。

### 损失函数 / 训练策略
本文不训练新的生成模型；实验使用已有或定制训练的 SGM prior。算法侧的关键超参包括 step size $\gamma$、ZO smoothing $\mu$、刷新概率 $p$、大/小 batch size $b,b'$，以及 annealing schedule $\alpha_k=\max\{\alpha_0\rho_1^k,1\}$、$\sigma_k=\max\{\sigma_0\rho_2^k,\sigma_{min}\}$。理论中这些参数被设置成随迭代数和维度缩放的形式，以平衡零阶估计 bias、估计方差和 Langevin 离散误差。

## 实验关键数据

### 主实验
作者先用 toy experiments 验证 FI 收敛，再在 MRI、黑洞成像和 Navier-Stokes 反演上比较黑盒 posterior sampler。MRI 和黑洞成像给出了完整定量表；Navier-Stokes 中 ZO-APMC 在视觉质量上较稳，NRMSE 与 DPG 接近但未全面超过 EnKG/DPG。

| 任务 | 指标 | ZO-APMC | 最强黑盒 baseline | 梯度可用参考 | 结论 |
|------|------|---------|-------------------|--------------|------|
| Toy FI convergence | FI | p=1/0.75/0.5 时接近 0；p=0.3 不稳定 | 无 | APMC | 固定每步成本 $pb=10$ 下，多组参数能在 2000 iterations 内低于 0.01 FI |
| MRI reconstruction | PSNR / SSIM / NRMSE | 35.29 / 0.966 / 2.28e-2 | DPG: 32.17 / 0.953 / 5.4e-2 | APMC: 36.55 / 0.973 / 1.99e-2 | 黑盒方法中整体最好，接近梯度版 APMC |
| Black-hole imaging | PSNR / blurred PSNR | 26.71 / 32.86 | Forward-GSG: 26.21 / 31.47 | APMC: 26.23 / 31.32 | ZO-APMC 在 PSNR 和测量一致性上都领先黑盒 baseline |
| Navier-Stokes inverse problem | NRMSE / qualitative flow | 与 DPG 接近 | EnKG / DPG | 无梯度主线 | 没有所有指标最优，但提供更明确的收敛解释 |

### 消融实验
论文最核心的分析不是传统模块 ablation，而是围绕 $p,b,b'$ 与黑盒方法的性能/成本权衡。下表整理 MRI 与黑洞成像中的定量比较，体现方差缩减零阶估计在高维设置中的实际效果。

| 场景 | 方法 | PSNR | SSIM / blurred PSNR | 误差指标 | 说明 |
|------|------|------|---------------------|----------|------|
| MRI | PnPDM | 30.81 | SSIM 0.946 | MSE 8.46e-4 | 梯度可用 baseline，但质量低于 ZO-APMC |
| MRI | DPS | 34.38 | SSIM 0.965 | MSE 4.07e-4 | 接近 ZO-APMC，但仍低 0.91 dB |
| MRI | APMC | 36.55 | SSIM 0.973 | MSE 2.55e-4 | 梯度版上界参考 |
| MRI | ZO-APMC | 35.29 | SSIM 0.966 | MSE 3.29e-4 | 只用函数查询，接近 APMC |
| Black-hole | Forward-GSG | 26.21 | blurred PSNR 31.47 | $\chi^2_{cph}=6.77$ | 强黑盒 baseline |
| Black-hole | Central-GSG | 21.63 | blurred PSNR 23.73 | $\chi^2_{cph}=80.31$ | 中心差分并未稳定提升 |
| Black-hole | ZO-APMC | 26.71 | blurred PSNR 32.86 | $\chi^2_{cph}=5.42$ | PSNR 与测量拟合均最好 |

### 关键发现
- 方差缩减的刷新概率有明显 trade-off：$p$ 太小会降低总函数评估次数但加重误差传播，toy experiment 中 $p=0.3$ 已出现不稳定；$p=0.5$ 在成本和稳定性之间较均衡。
- 在 MRI 中，ZO-APMC 用 $p=0.2,b=10^4,b'=10^3$ 能在 256x256 高维图像上接近梯度版 APMC，说明理论中的高维多项式复杂度虽然保守，实践中并非不可用。
- 黑洞成像的 nonlinear forward model 更能体现黑盒优势：ZO-APMC 不依赖可微化 forward operator，却在 closure phase/amplitude 误差上优于 GSG、DPG、EnKG 等启发式方法。

## 亮点与洞察
- 这篇论文把零阶优化里的方差缩减思想移植到采样里，但没有简单照搬；它特别处理了 Langevin discretization error 和 ZO smoothing bias 的耦合，这正是采样比优化更麻烦的地方。
- 用 relative Fisher information 连接非凸优化和非对数凹采样很有启发性。它让“采样算法是否接近目标分布”可以像“优化是否接近 stationary point”一样被分析，同时又保留概率分布层面的意义。
- ZO-APMC 的应用价值在于把 SGM prior 从“需要 forward gradient 的 inverse problem solver”扩展到真正黑盒的 simulator/proprietary system。这个思路可以迁移到医学设备、气候/流体仿真、工业仿真和闭源物理引擎。

## 局限与展望
- 理论复杂度对维度有 $d^7$ 依赖，非常保守；尽管实验表现更好，但理论与实践之间仍有明显 gap。
- Assumption 1 要求 potential function 全局 Lipschitz，这对很多常见分布不自然。作者也承认该条件更适合 compact domain、normalization 或 gradient clipping 后的实践设置。
- ZO-APMC 在 Navier-Stokes 上并未稳定超过 EnKG/DPG，说明有收敛保证不等于所有任务上经验最优。未来需要更系统地研究不同 forward operator、噪声水平和 prior mismatch 下的表现。
- 高维图像实验仍需要较大的 batch，例如 MRI 中 $b=10^4,b'=10^3$，实际部署到昂贵 simulator 时函数调用成本可能成为瓶颈。

## 相关工作与启发
- **vs Roy et al. 的 ZO-LMC**: 早期 ZO-LMC 主要分析 strongly log-concave 目标，并需要 batch size 随维度增长；本文转向 non-log-concave 目标，并用方差缩减把平均每步函数查询降到常数级。
- **vs APMC / DPS / PnPDM**: 这些 posterior sampler 在许多设置下依赖 forward model gradient 或可微结构；ZO-APMC 牺牲一部分效率，换来真正黑盒 forward operator 下的适用性和收敛保证。
- **vs GSG / DPG / EnKG**: 这些黑盒方法更偏启发式近似，部分任务上可以竞争甚至更强；ZO-APMC 的优势是把超参、annealing 和采样误差放进同一套理论框架。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把方差缩减零阶估计、非对数凹采样理论和 SGM 逆问题结合得很有辨识度。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖 toy、MRI、黑洞成像和 PDE 反演，但消融更多围绕参数敏感性，缺少更大规模 simulator 成本分析。
- 写作质量: ⭐⭐⭐⭐☆ 理论主线清楚，应用动机充分；公式较密，对非采样背景读者有一定门槛。
- 价值: ⭐⭐⭐⭐☆ 对黑盒 Bayesian inverse problem 很有价值，尤其适合 forward model 不可微或不可公开的场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2026\] Saving Foundation Flow-Matching Priors for Inverse Problems](saving_foundation_flow-matching_priors_for_inverse_problems.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](../../NeurIPS2025/image_generation/npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)

</div>

<!-- RELATED:END -->
