---
title: >-
  [论文解读] Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms
description: >-
  [NeurIPS 2025][图像生成][离散扩散] 首次将高阶数值方法引入离散扩散模型推理，提出 θ-RK-2 和 θ-Trapezoidal 两种二阶求解器，在理论上证明 θ-Trapezoidal 的离散化误差从一阶 $\mathcal{O}(\kappa T)$ 提升到二阶 $\mathcal{O}(\kappa^2 T)$，实验覆盖 200M–8B 模型在文本、图像和数学推理上的一致性提升。
tags:
  - NeurIPS 2025
  - 图像生成
  - 离散扩散
  - 高阶求解器
  - τ-leaping
  - Trapezoidal
  - 文本生成
---

# Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms

**会议**: NeurIPS 2025  
**arXiv**: [2502.00234](https://arxiv.org/abs/2502.00234)  
**代码**: [DiscreteFastSolver](https://github.com/yuchen-zhu-zyc/DiscreteFastSolver)  
**领域**: 扩散模型  
**关键词**: 离散扩散, 高阶求解器, τ-leaping, Trapezoidal, 文本生成, 图像生成

## 一句话总结

首次将高阶数值方法引入离散扩散模型推理，提出 θ-RK-2 和 θ-Trapezoidal 两种二阶求解器，在理论上证明 θ-Trapezoidal 的离散化误差从一阶 $\mathcal{O}(\kappa T)$ 提升到二阶 $\mathcal{O}(\kappa^2 T)$，实验覆盖 200M–8B 模型在文本、图像和数学推理上的一致性提升。

## 研究背景与动机

**领域现状**：离散扩散模型在文本、图像、分子等离散数据生成中已取得重要进展。与连续扩散模型不同，离散模型天然适合处理语言 token、分子序列和量化图像等组合空间中的数据。

**现有痛点**：离散扩散模型的推理效率是核心瓶颈。现有方法分为两大类：（1）**精确仿真方法**（如 uniformization、First-Hitting Sampler）理论上无偏，但推理时间不可预测——跳跃次数在反向过程末期急剧增长，导致大量冗余计算，且不同样本间的变长调度妨碍了批并行；（2）**近似方法**（如 τ-leaping）简单可并行，但作为一阶方法，需要极小步长才能控制离散化误差，在计算受限场景下样本质量有限。

**核心矛盾**：连续扩散模型已有丰富的高阶加速方法（DPM-Solver、DDIM 等），但离散领域由于状态空间的不连续性和 Poisson 跳跃过程的数学复杂性，高阶方法一直是空白。τ-leaping 的一阶精度在连续场景对应的 Euler 方法已被证明远非最优——离散场景理应也能从高阶方法获益。

**本文目标** 开发适用于离散扩散模型的高阶数值推理方法，在不改变模型的前提下，以更少的函数评估次数（NFE）获得更高质量的样本。

**切入角度**：从 ODE 的 Runge-Kutta 方法和化学反应仿真的高阶 τ-leaping 方法中汲取灵感，将这些经典数值方法适配到离散扩散模型的 Poisson 跳跃过程中。

**核心 idea**：用两阶段预测-校正的高阶求解器替代一阶 τ-leaping，在每步利用中间点的额外信息来将离散化误差从一阶提升到二阶。

## 方法详解

### 整体框架

在离散扩散模型的反向过程中，状态空间 $\mathbb{X} = [S]^d$ 上的连续时间 Markov 链通过速率矩阵定义转移概率。推理时需要从学习的反向速率（通过 score 函数参数化）驱动的 Poisson 跳跃过程中采样。传统 τ-leaping 在每一步仅使用当前时间点的速率做 Poisson 采样（一阶），本文提出的方法额外评估一个中间时间点的速率，用加权组合来实现二阶精度。

### 关键设计

1. **θ-RK-2 方法（热身方案）**:

    - 功能：类比 ODE 的二阶 Runge-Kutta，提供预测-校正的两阶段离散扩散推理
    - 核心思路：第一步用 τ-leaping 走 $\theta\Delta$ 步到中间点 $\hat{y}_{\rho_n}^*$，评估该点的速率 $\hat{\mu}_{\rho_n}^*$；第二步从原始起点 $\hat{y}_{s_n}$ 出发，用起始和中间点速率的加权和 $[(1-\frac{1}{2\theta})\hat{\mu}_{s_n} + \frac{1}{2\theta}\hat{\mu}_{\rho_n}^*]$ 走完整步 $\Delta_n$。权重为内插系数
    - 设计动机：直接将 ODE 的 RK-2 思想推广到 Poisson 过程。但理论分析揭示其二阶收敛仅在 $\theta \in (0, 1/2]$ 时成立（条件性二阶），因为加权和在 $\theta > 1/2$ 时从内插变为外推，导致技术困难

2. **θ-Trapezoidal 方法（核心贡献）**:

    - 功能：更稳健的二阶离散扩散推理器，对所有 $\theta \in (0,1]$ 均无条件二阶收敛
    - 核心思路：第一步与 θ-RK-2 相同——τ-leaping 走 $\theta\Delta$ 到中间点。第二步有两个关键差异：（a）从中间点 $\hat{y}_{\rho_n}^*$ 出发而非原始起点，走 $(1-\theta)\Delta_n$ 步；（b）使用外推系数 $(\alpha_1, -\alpha_2)$ 而非内插系数，其中 $\alpha_1 = \frac{1}{2\theta(1-\theta)}$，$\alpha_2 = \frac{(1-\theta)^2+\theta^2}{2\theta(1-\theta)}$，满足 $\alpha_1 - \alpha_2 = 1$。速率为 $(\alpha_1 \hat{\mu}_{\rho_n}^* - \alpha_2 \hat{\mu}_{s_n})_+$
    - 设计动机：将每个时间区间分为两个子区间分别处理，并使用外推而非内插的权重设计。这种分离式处理使得二阶收敛性的证明对 $\theta$ 没有限制——因为外推结构使关键的误差项可以用 Dynkin 公式无条件约束

3. **多跳拒绝机制**:

    - 功能：保证算法的良定义性
    - 核心思路：按照离散扩散文献的标准做法，拒绝在同一维度上发生多次跳跃的更新。分析表明拒绝概率仅为 $\mathcal{O}(\kappa)$，不影响渐近精度
    - 设计动机：在 Poisson 跳跃过程中，大步长可能导致同一维度多次转移，这在离散空间中可能产生无效状态

### 理论保证

**定理（θ-Trapezoidal 无条件二阶收敛）**：在正则性假设下，$D_{\text{KL}}(p_\delta \| \hat{q}_{T-\delta}^{\text{trap}}) \lesssim \exp(-T) + (\epsilon_I + \epsilon_{II})T + \kappa^2 T$。相比 τ-leaping 的 $\mathcal{O}(\kappa T)$，离散化误差提升到 $\mathcal{O}(\kappa^2 T)$，对 $\theta \in (0,1]$ 无条件成立。θ-RK-2 的二阶收敛则仅在 $\theta \in (0, 1/2]$ 时成立。

## 实验关键数据

### 主实验

**文本生成（RADD/GPT-2 级别，d=1024, S=50258）**：

| 方法 | NFE=128 | NFE=1024 |
|------|---------|----------|
| FHS (精确仿真) | ≤122.7 | ≤109.4 |
| Euler | ≤86.3 | ≤44.7 |
| Tweedie τ-leap | ≤85.7 | ≤44.3 |
| τ-leaping | ≤52.4 | ≤28.8 |
| Semi-AR | ≤360.8 | ≤147.4 |
| θ-RK-2 | ≤64.3 | ≤36.3 |
| **θ-Trapezoidal** | **≤49.1** | **≤27.6** |

θ-Trapezoidal 在所有 NFE 下一致优于现有方法。值得注意的是精确仿真（FHS）在高 NFE 时反而差于近似方法——无离散化误差不等于更好的样本，因为 score 估计误差在过程末期被放大。

**图像生成（MaskGIT/ImageNet 256×256，d=256, S=1025）**：

| 方法 | NFE=16 FID↓ | NFE=32 FID↓ | NFE=64 FID↓ |
|------|------------|------------|------------|
| Euler | ~12 | ~8 | ~7 |
| τ-leaping | ~10 | ~7 | ~6 |
| **θ-Trapezoidal** | **~8** | **~6** | **~5** |

θ-Trapezoidal 在 NFE≥16 时 FID 持续优于一阶方法。FHS 和 parallel decoding 在极低 NFE（≤8）有优势但快速饱和。

### 消融实验

| 配置 | 文本 PPL (NFE=128) | 图像 FID (NFE=32) | 说明 |
|------|-------------------|-------------------|------|
| τ-leaping (一阶) | 52.4 | ~7 | 基线 |
| θ-RK-2 (θ=0.5) | 64.3 | — | 条件二阶，θ>0.5 退化 |
| θ-Trapezoidal (θ=0.5) | **49.1** | **~6** | 无条件二阶，稳定最优 |
| θ-Trapezoidal (θ=0.3) | ~49.5 | ~6 | θ∈[0.3,0.5] 均稳健 |
| θ-Trapezoidal (θ=0.8) | ~50 | ~6.5 | θ 偏大略微退化 |

**LLaDA-Instruct 8B 数学推理（GSM8K）**：

| 方法 | NFE=64 | NFE=128 | NFE=256 |
|------|--------|---------|---------|
| Semi-AR (Rand.) | 33.8% | 34.3% | **40.3%** |
| **θ-Trapezoidal** | **35.1%** | **38.4%** | 39.7% |

在低 NFE（计算受限）场景下优势尤为明显。

### 关键发现

- **θ-Trapezoidal 全面领先**：在文本、图像、数学推理三类任务上一致优于 τ-leaping 和 θ-RK-2，与理论预测完全吻合。
- **θ 超参数鲁棒**：θ∈[0.3, 0.5] 在所有任务上均为稳健选择，性能曲面平坦，不需要精细调参。
- **精确仿真≠最优**：FHS 的无偏性在 score 估计误差面前反而是劣势——末期大量冗余 NFE 被浪费在 score 不准的区域。这支持了"有控制的近似优于无控制的精确"的观点。
- **从 200M 到 8B 模型一致有效**：在 MaskGIT（200M 级）和 LLaDA（8B 级）上都观察到性能提升，说明高阶求解器的优势不依赖于特定模型规模。

## 亮点与洞察

- **填补重要空白**：连续扩散有 DPM-Solver 等高阶方法，离散扩散一直缺失——本文是首个严格证明并实验验证的二阶离散扩散求解器。填补了离散生成模型理论和实践的关键空白。
- **Trapezoidal 优于 RK-2 的深层原因**：RK-2 的第二步从原始起点出发走完整步（内插），Trapezoidal 从中间点出发走剩余步（外推）。外推结构虽然在直觉上更"激进"，但在理论上恰好使误差项可以无条件约束——这是化学反应仿真领域的经典洞察在 AI 中的成功迁移。
- **无训练加速的实用价值**：方法完全是推理时的改进，不需要重新训练模型。对于已部署的离散扩散模型可以即插即用提速，降低推理成本。
- **"精确不等于好"的反直觉结论**：FHS 等精确方法在高 NFE 时反而差于近似方法。这个发现对离散扩散模型的推理策略选择有重要指导意义。

## 局限与展望

- **外推非负性假设**：θ-Trapezoidal 的理论证明依赖 $\alpha_1 \hat{\mu}_{\rho}^* - \alpha_2 \hat{\mu}_{s} \geq 0$ 这一假设。实验显示在 95%+ 情况下满足，但理论确认仍是 open problem（来自化学反应仿真的长期未解问题）。
- **仅考虑二阶方法**：更高阶（三阶、四阶）方案是否能带来进一步提升？多阶段方法的额外 NFE 开销与精度提升的 trade-off 需要研究。
- **文本实验仅在 GPT-2 级别**：虽然 8B 模型上的 GSM8K 验证了扩展性，但在更大规模的 LLM 上的系统性评估仍缺乏。
- **与其他加速方法的联合**：如何与缓存、投机采样等正交加速技术结合使用？
- **协方差误差的理论分析**：化学反应文献指出 Trapezoidal 在协方差误差上也有二阶优势，但离散扩散场景的理论证明留待未来。

## 相关工作与启发

- **vs τ-leaping (Campbell et al., 2022)**: 一阶近似方法，是本文的直接改进对象。θ-Trapezoidal 每步需要 2 次而非 1 次 score 评估，但离散化误差从 $\mathcal{O}(\kappa)$ 降到 $\mathcal{O}(\kappa^2)$——当 $\kappa$ 不够小时，这个改进极为显著。
- **vs DPM-Solver (Lu et al., 2022)**: 连续扩散模型的高阶方法。DPM-Solver 利用 ODE 的指数积分器，而本文处理的是 Poisson 跳跃过程——数学上本质不同，需要全新的理论框架。
- **vs FHS (Zheng et al., 2024)**: 精确仿真方法。本文证明了"精确不等于最优"——被控制的近似在实际 score 有误差时反而更好，因为精确方法在 score 误差大的区域浪费了过多计算。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将高阶数值方法引入离散扩散，填补了重要的理论和实践空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖 200M–8B 模型，跨文本/图像/数学推理，但大规模 LLM 评估偏少
- 写作质量: ⭐⭐⭐⭐ 理论表述严谨，实验分析细致，但数学符号密集可能影响可读性
- 价值: ⭐⭐⭐⭐⭐ 无训练即插即用的推理加速，对离散扩散模型的部署有直接影响

<!-- RELATED:START -->

## 相关论文

- [High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction](high-order_equivariant_flow_matching_for_density_functional_theory_hamiltonian_p.md)
- [Beyond Masked and Unmasked: Discrete Diffusion Models via Partial Masking](beyond_masked_and_unmasked_discrete_diffusion_models_via_par.md)
- [Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [Guided Diffusion Sampling on Function Spaces with Applications to PDEs](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)

<!-- RELATED:END -->
