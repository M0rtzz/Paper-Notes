---
title: >-
  [论文解读] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler
description: >-
  [ICML 2026][优化/理论][SAM] 这篇论文把 Polyak step size 推广到 USAM/SAM，给出不依赖手工学习率调参的 sharpness-aware scheduler，并在凸优化理论和 CIFAR 实验中验证其稳定性与性能。 领域现状：Sharpness-Aware Minimization…
tags:
  - "ICML 2026"
  - "优化/理论"
  - "SAM"
  - "USAM"
  - "Polyak step size"
  - "自适应学习率"
  - "收敛理论"
---

# Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler

**会议**: ICML 2026  
**arXiv**: [2606.01827](https://arxiv.org/abs/2606.01827)  
**代码**: https://github.com/dimitris-oik/sam_sps  
**领域**: 优化 / Sharpness-Aware Minimization  
**关键词**: SAM, USAM, Polyak step size, 自适应学习率, 收敛理论  

## 一句话总结
这篇论文把 Polyak step size 推广到 USAM/SAM，给出不依赖手工学习率调参的 sharpness-aware scheduler，并在凸优化理论和 CIFAR 实验中验证其稳定性与性能。

## 研究背景与动机
**领域现状**：Sharpness-Aware Minimization 通过在参数邻域内优化最坏扰动损失，倾向于找到更平坦的解，常能改善深度模型泛化。标准 SAM 先沿梯度方向构造扰动点，再在扰动点计算梯度更新；USAM 去掉了扰动归一化，理论分析更方便。

**现有痛点**：SAM/USAM 对 learning rate 非常敏感。实际训练通常需要为不同数据集、网络和 sharpness radius 反复搜索固定学习率或 scheduler。已有 adaptive SAM 方法要么缺少强理论，要么需要额外假设，例如 bounded variance、bounded gradients 或 growth condition。

**核心矛盾**：SAM 的泛化收益来自非零扰动半径 $\rho$，但优化理论上非零 $\rho$ 会让更新更难稳定；同时，学习率越需要精细调参，SAM 的实际使用成本越高。本文试图用 Polyak 式闭式学习率减少调参，同时保留可证明收敛。

**本文目标**：为 USAM 推导确定性和随机场景下的 Polyak-type scheduler，证明强凸和凸目标上的收敛率，并检验该 scheduler 是否能在深度学习 benchmark 上接近或超过调好学习率的 SAM/USAM。

**切入角度**：经典 Polyak step size 来自最小化 $\|x_{t+1}-x^*\|^2$ 的上界。USAM 的更新梯度不是在 $x_t$，而是在扰动点 $e_t=x_t+\rho_t\nabla f(x_t)$，因此只要重新写这个距离上界，就能得到 sharpness-aware 版本的 Polyak 学习率。

**核心 idea**：把 Polyak step size 中的 $f(x_t)-f^*$ 和 $\nabla f(x_t)$ 替换为扰动点 $e_t$ 上的 loss/gradient，并扣除扰动项 $\rho_t\langle\nabla f(e_t),\nabla f(x_t)\rangle$。

## 方法详解
论文主要贡献是一个 scheduler，而不是一个全新的优化目标。它从 USAM 的确定性更新出发：先取 $e_t=x_t+\rho_t\nabla f(x_t)$，再用 $\nabla f(e_t)$ 更新 $x_t$。作者将 $\|x_{t+1}-x^*\|^2$ 展开，并利用凸性构造一个关于学习率 $\gamma_t$ 的二次上界。最小化这个上界就得到 USAM 的 Polyak scheduler。

### 整体框架
在确定性 USAM 中，scheduler 写为 $\gamma_t=(f(e_t)-f^*-\rho_t\langle\nabla f(e_t),\nabla f(x_t)\rangle)/\|\nabla f(e_t)\|^2$。如果分子可能为负，可以加 $[\cdot]_+$ safeguard；但论文证明在凸且 $L$-smooth、$\rho_t\le 1/L$ 时，分子自然非负，因此 safeguard 是冗余的。

在随机有限和场景中，作者将 $f$ 替换为 mini-batch objective $f_{S_t}$，并用 mini-batch lower bound $\ell^*_{S_t}$ 替代全局最优值。为了避免随机场景中异常大的步长，随机 scheduler 加了上界 $\gamma_b$，得到 capped Stochastic Polyak Scheduler。特别地，当 $\rho_t=0$ 时，这两个公式分别退化为经典 GD Polyak step size 和 SPSmax。

### 关键设计
1. **USAM 的 Polyak 上界推导**:

	- 功能：把学习率从需调超参变成由当前 loss 和 gradient 决定的闭式量。
	- 核心思路：展开 $\|x_{t+1}-x^*\|^2-\|x_t-x^*\|^2$，其中 $x_{t+1}=x_t-\gamma_t\nabla f(e_t)$。由于 $e_t=x_t+\rho_t\nabla f(x_t)$，上界中多出 $\rho_t\langle\nabla f(e_t),\nabla f(x_t)\rangle$，它需要从 $f(e_t)-f^*$ 中扣除。
	- 设计动机：如果忽略扰动点与原点之间的差异，Polyak 步长会高估可安全下降的空间。这个扣除项正是 sharpness-aware 更新相对普通 GD 的修正。

2. **随机 capped scheduler 与 lower bound**:

	- 功能：把确定性 scheduler 扩展到 mini-batch 训练。
	- 核心思路：随机步长为 $\gamma_t=\min\{(f_{S_t}(e_t)-\ell^*_{S_t}-\rho_t\langle\nabla f_{S_t}(e_t),\nabla f_{S_t}(x_t)\rangle)/\|\nabla f_{S_t}(e_t)\|^2,\gamma_b\}$。非负损失中常可取 $\ell^*_{S_t}=0$，而 $\gamma_b$ 控制随机噪声下的最大步长。
	- 设计动机：SPSmax 的成功经验说明，随机 Polyak 步长需要 cap 来保证稳定；本文保留这一结构，使 USAM-SPS 与已有 SGD-SPS 理论自然对应。

3. **确定性与随机收敛理论**:

	- 功能：说明 scheduler 不只是经验 trick，而有清晰优化保证。
	- 核心思路：强凸确定性目标下，USAM-PS 满足线性收敛 $\|x_T-x^*\|^2\le (1-\mu(1-L\rho)^2/(4L))^T\|x_0-x^*\|^2$；凸目标下 Cesaro average 达到 $O(1/T)$。随机强凸和凸场景下，方法收敛到由 $\sigma^2=\mathbb{E}_{S_t}[f_{S_t}(x^*)-\ell^*_{S_t}]$ 控制的邻域；插值场景 $\sigma^2=0$ 时邻域消失。
	- 设计动机：这补上了 adaptive SAM 里常见的理论空缺，也避免了一些 prior work 需要的额外噪声或梯度有界假设。

### 损失函数 / 训练策略
理论部分处理一般 convex/strongly convex finite-sum objective。实验部分在 DNN 上使用 cross-entropy，训练 ResNet-20/32 于 CIFAR-10/100，mini-batch size 为 128，100 epochs，标准数据增强。USAM/SAM 的 Stochastic Polyak Scheduler 取 $\ell^*_{S_t}=0$、$\gamma_b=1.0$，并与调参后的 constant learning rate 和 cosine annealing 对比。

## 实验关键数据

### 主实验
下表摘取主文 CIFAR-100 ResNet-32 结果。指标为 test accuracy，越高越好。

| 方法 | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.3$ | $\rho=0.4$ |
|--------|------|------|----------|------|
| Constant USAM | 90.56±0.18 | 90.45±0.34 | 90.25±0.10 | 89.56±0.07 |
| USAM + Cosine Annealing | 90.01±0.32 | 88.77±0.26 | 88.05±0.23 | 86.52±0.04 |
| USAM + Stochastic Polyak Scheduler | 91.81±0.04 | 92.23±0.22 | 92.24±0.30 | 92.01±0.12 |
| Constant SAM | 90.17±0.11 | 90.53±0.02 | 89.61±0.10 | 88.64±0.13 |
| SAM + Cosine Annealing | 90.49±0.02 | 89.03±0.13 | 87.05±0.24 | 84.61±0.34 |
| SAM + Stochastic Polyak Scheduler | 91.61±0.12 | 92.24±0.07 | 91.70±0.15 | 90.79±0.16 |

### 消融实验
论文的消融/分析主要比较不同 scheduler、不同 sharpness radius 以及理论条件。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| USAM-PS vs 常数步长 USAM | synthetic ridge 中收敛最快 | 与已有有保证的常数步长相比，Polyak scheduler 在确定性强凸问题上更快接近最优 |
| USAM-SPS vs 随机 USAM baseline | 收敛到解邻域 | 与随机理论一致，并且不需要为每个问题调学习率 |
| 与 AdaSAM / LightSAM / SA-SAM 对比 | 确定性问题中最少迭代到最优 | 其他 adaptive SAM 也可工作，但本文方法同时有强凸理论保证 |
| 增大 $\rho$ | Cosine accuracy 明显下降，SPS 保持稳定 | $\rho=0.4$ 时 USAM-SPS 仍有 92.01%，Cosine USAM 只有 86.52% |
| SAM 归一化版本 | 趋势与 USAM 一致 | 虽然理论主线是 USAM，Polyak 原理也可迁移到标准 SAM |
| 插值场景 $\sigma^2=0$ | 随机邻域项消失 | 过参数化或可插值任务中，scheduler 可理论上收敛到最优而非邻域 |

### 关键发现
- Stochastic Polyak Scheduler 在较大 sharpness radius 下优势最明显。cosine annealing 随 $\rho$ 增大快速退化，而 SPS 的精度保持得更稳。
- 理论上，$\rho=0$ 时方法退化为已有 Polyak/SPSmax；因此它不是另起炉灶，而是把普通自适应步长和 sharpness-aware update 统一起来。
- 对 SAM 的 normalized update，论文没有给出同等完整的非负性理论，但实验显示 safeguard 很少触发，且 CIFAR 表现同样优于调参 baseline。

## 亮点与洞察
- 这篇论文的推导很干净：USAM 的 Polyak 步长只比 GD 多一个内积扣除项，直接对应“我先走到扰动点再算梯度”的几何差异。
- 它把 SAM 的实践痛点和优化理论连接得比较紧。许多 adaptive optimizer 只给经验结果，而这里至少在凸/强凸场景下说明了 scheduler 为什么合理。
- 实验中最有说服力的是 $\rho$ 扫描。真正有用的 scheduler 应该在 sharpness radius 变化时稳住训练，而不只是某个调好的点上表现好。

## 局限与展望
- 理论主要覆盖 smooth convex 和 strongly convex 目标，而深度网络训练是非凸的。CIFAR 实验说明方法可用，但不能直接证明大规模非凸训练的收敛。
- scheduler 仍需要知道或估计 lower bound，例如非负交叉熵可取 0，但更一般任务中 $\ell^*_{S_t}$ 的选择可能影响稳定性。
- sharpness radius $\rho$ 仍是重要超参。论文减少了 learning rate 调参，但没有完全解决 $\rho$ 的自适应选择。
- 大规模 LLM 训练、AdamW 场景以及与 momentum/weight decay 的交互尚未系统验证。

## 相关工作与启发
- **vs 标准 SAM**: 标准 SAM 需要外部 learning rate schedule；本文把 step size 变成由当前 mini-batch loss 和扰动点梯度决定的闭式量。
- **vs SPSmax**: SPSmax 是 SGD 的随机 Polyak 步长，本文在 $\rho=0$ 时退化为 SPSmax，在 $\rho>0$ 时扩展到 USAM/SAM。
- **vs AdaSAM / LightSAM / SA-SAM**: 这些方法也自适应调整 SAM，但理论条件和经验行为各异；本文强调在凸/强凸问题上的清晰保证。
- **对后续工作的启发**: 可以把 Polyak-style scheduler 与 sharpness radius scheduler 联合设计，进一步减少 SAM 系列方法的调参负担。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 Polyak step size 系统推广到 USAM/SAM，推导自然但切中痛点。
- 实验充分度: ⭐⭐⭐⭐☆ 有理论验证和 CIFAR 实验，但缺少更大规模模型和非视觉任务。
- 写作质量: ⭐⭐⭐⭐⭐ 公式、定理和实验逻辑都比较清楚。
- 价值: ⭐⭐⭐⭐☆ 对想使用 SAM 但不想大规模调学习率的训练场景很实用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)

</div>

<!-- RELATED:END -->
