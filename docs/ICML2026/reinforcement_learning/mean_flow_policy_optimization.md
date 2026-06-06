---
title: >-
  [论文解读] Auto temperature
description: >-
  [ICML 2026][强化学习][MeanFlow] MFPO 用 MeanFlow models（学 average velocity 而非 instantaneous velocity）当 RL policy 把扩散策略采样步数从 20+ 降到 2 步…
tags:
  - "ICML 2026"
  - "强化学习"
  - "MeanFlow"
  - "MaxEnt RL"
  - "soft policy iteration"
  - "平均散度网络"
  - "importance sampling"
---

# MFPO: 用 Few-step MeanFlow Policy 把 MaxEnt RL 跑到接近 Gaussian policy 的速度

**会议**: ICML 2026  
**arXiv**: [2604.14698](https://arxiv.org/abs/2604.14698)  
**代码**: https://github.com/dongxiaoyi-xyz/MFPO  
**领域**: 强化学习 / 扩散策略 / 流匹配  
**关键词**: MeanFlow, MaxEnt RL, soft policy iteration, 平均散度网络, importance sampling

## 一句话总结
MFPO 用 MeanFlow models（学 average velocity 而非 instantaneous velocity）当 RL policy 把扩散策略采样步数从 20+ 降到 2 步，用 average divergence network 解决 action likelihood 计算、用 ESS-weighted SNIS 组合 Gaussian + policy proposal 解决 soft policy improvement，在 MuJoCo/DMC/HumanoidBench 上性能 ≥ diffusion baseline 且训练时间降 ~50%。

## 研究背景与动机

**领域现状**：在线 RL 在连续控制上 Gaussian/deterministic policy + noise 太单模态，复杂任务 reward landscape 多模态时容易陷局部最优。扩散/流匹配策略（DIPO、QVPO、DACER、DIME）通过 iterative generation 建模 multi-modal action 但 inference 10-20 步导致训练慢一两个数量级。

**现有痛点**：MaxEnt RL 平衡 explore/exploit 需要 action likelihood 评估 + soft policy improvement（match Boltzmann distribution）；对扩散策略都难——likelihood 需积分 instantaneous velocity divergence（intractable），soft improvement 需 Boltzmann 样本（不可用）。已有方法（DIME lower bound、MaxEntDP 数值积分、SAC-Flow GRU/Transformer）在 few-step 下 bound 松、ODE 离散化误差大。

**核心矛盾**：要 multi-modal 表达力就要扩散；要 RL 训练效率就要少步；少步会破坏 likelihood 精度和 policy improvement 准确性。

**本文目标**：在 2 步内做到 multi-modal policy + MaxEnt RL 的精确 likelihood + soft improvement，让扩散策略训练时间接近 Gaussian policy。

**切入角度**：MeanFlow models (Geng et al. 2025) 学 average velocity $\boldsymbol{u}(\boldsymbol{x}_t, r, t) = \frac{1}{t-r} \int_r^t \boldsymbol{v}(\boldsymbol{x}_\tau, \tau) d\tau$ 而非 instantaneous，精确学到后 2 步采样无 discretization error。但 MeanFlow 应用到 MaxEnt RL 仍要解决 likelihood 和 soft improvement 两个挑战。

**核心 idea**：(1) 模仿 MeanFlow 造 average divergence network $\delta_\omega$ 近似 $\frac{1}{t-r} \int_r^t \nabla \cdot \boldsymbol{v}_\theta d\tau$，复用 sampling pipeline 算 likelihood；(2) 用 SNIS 估 Boltzmann 的 marginal velocity，自适应组合 policy proposal + Gaussian proposal by ESS weighting；(3) MeanFlow policy 用 soft policy iteration 训练，配 distributional critic + auto temperature。

## 方法详解

### 整体框架

两个核心网络：critic $Q_\phi$ 和 MeanFlow policy $\boldsymbol{u}_\theta$（加 ADN $\delta_\omega$）。Soft policy iteration 交替 policy evaluation 和 policy improvement。Inference 2 步：$\boldsymbol{a}_{t_{i-1}} = \boldsymbol{a}_{t_i} - \frac{1}{T} \boldsymbol{u}_\theta(\boldsymbol{s}, \boldsymbol{a}_{t_i}, t_{i-1}, t_i)$。

### 关键设计

1. **MeanFlow Policy + Average Divergence Network 解决 likelihood**:

    - 功能：让 2-step MeanFlow policy 的 action likelihood 在 5% 额外成本下精确估计。
    - 核心思路：MeanFlow policy 学 average velocity，sampling $\boldsymbol{a}_r = \boldsymbol{a}_t - (t-r) \boldsymbol{u}_\theta$。Action likelihood 用 change-of-variable $\log \pi_\theta(\boldsymbol{a}_0|\boldsymbol{s}) = \log p_1(\boldsymbol{a}_1) + \int_0^1 \nabla \cdot \boldsymbol{v}_\theta dt$，但 naive Jacobian + 数值积分太贵。模仿 MeanFlow 造 average divergence network $\delta_\omega(\boldsymbol{s}, \boldsymbol{a}_t, r, t) \approx \frac{1}{t-r} \int_r^t \nabla \cdot \boldsymbol{v}_\theta d\tau$，训练目标含 Skilling-Hutchinson trace estimator $\widehat{\text{div}} = \frac{1}{N} \sum \boldsymbol{\epsilon}_i^\top \frac{\partial \boldsymbol{v}_\theta}{\partial \boldsymbol{a}_t} \boldsymbol{\epsilon}_i$。Inference 时 $\log \pi_\theta(\boldsymbol{a}_0|\boldsymbol{s}) = \log p_1(\boldsymbol{a}_1) + \frac{1}{T} \sum_i \delta_\omega(\boldsymbol{s}, \boldsymbol{a}_{t_i}, t_{i-1}, t_i)$ 复用 sampling trajectory。
    - 设计动机：MaxEnt entropy 需要 likelihood 但 ODE divergence 积分 intractable。ADN 是 MeanFlow 思路的 divergence 类比——既精确（trained to match）又便宜（5% overhead）。Skilling-Hutchinson 让 trace 不需 $d$ 个 backward pass。

2. **ESS-weighted SNIS 解决 soft policy improvement**:

    - 功能：在没有 Boltzmann 样本时精确估计 marginal velocity field 用于 policy update。
    - 核心思路：Boltzmann $\pi(\boldsymbol{a}_0|\boldsymbol{s}) \propto \exp(\frac{1}{\alpha} Q)$。Marginal velocity field $\boldsymbol{v}_t(\boldsymbol{a}_t|\boldsymbol{s}) = \mathbb{E}_{\pi(\boldsymbol{a}_0|\boldsymbol{a}_t, \boldsymbol{s})}[\frac{\boldsymbol{a}_t - \boldsymbol{a}_0}{t}]$。MaxEntDP/SDAC 用 Gaussian proposal $q^2(\boldsymbol{a}_0) = \mathcal{N}(\boldsymbol{a}_0|\frac{\boldsymbol{a}_t}{1-t}, (\frac{t}{1-t})^2 I)$ 做 SNIS，但 $t \to 1$ 时 ESS 急剧下降。本文加 policy proposal $q^1(\boldsymbol{a}_0) = \pi_\theta(\boldsymbol{a}_0|\boldsymbol{s})$（用 ADN 算 likelihood），ESS 在 $t \to 1$ 仍高。最终估计 $\hat{\boldsymbol{v}}_t = \sum_k \frac{\text{ESS}_k}{\sum_l \text{ESS}_l} \hat{\boldsymbol{v}}_t^k$ 用 ESS 加权组合。
    - 设计动机：单 proposal 在不同 $t$ 表现不一——Gaussian 在 $t$ 小时好（target 集中、Gaussian 匹配），Policy 在 $t$ 大时好（target 由 Q 主导）。ESS 自适应加权让 high-effective-sample 的 estimator 主导，variance 比单 proposal 小。

3. **Distributional Critic + Auto Temperature + Action Selection**:

    - 功能：提升训练稳定性和评估表现的工程技巧组合。
    - 核心思路：(a) Distributional critic 用 C51 把 Q-function 当 categorical 分布，policy update 用均值；(b) Auto-tune temperature 让 $\alpha$ match target entropy $\mathcal{H}_{\text{target}} = -\rho \cdot \dim(\mathcal{A})$，$\rho = 0.5$ 普适最佳；(c) Action selection 在 evaluation 时从 policy sample 多个候选 action，选 Q 最高的 deterministic action。
    - 设计动机：MaxEnt random policy 在 training 帮 explore 但 test 时 deterministic 更好；distributional Q-learning 已被 diffusion RL 证有效；auto temperature 让方法对 reward scale 鲁棒。

### 算法

```
Initialize Q_φ, π_θ (MeanFlow), δ_ω, α
for each step:
    # Policy evaluation
    L(φ) = (Q_φ(s,a) - (r + γ(Q(s',a') - α log π(a'|s'))))²
    # Policy improvement  
    Estimate v̂_t via ESS-weighted SNIS combining q^1 = π_θ + q^2 = Gaussian
    L(θ) = ||u_θ - sg(u_tgt)||²
    # ADN update via Eq. 17
    L(ω) = ||δ_ω - sg(δ_tgt)||²
    # Auto temperature
    L(α) = α (H(π_θ) - H_target)
```

## 实验关键数据

### 主实验：MuJoCo（5 locomotion）

| Algorithm | Sampling Steps | Inference Time (ms) | Avg Performance |
|---|---|---|---|
| **MFPO (ours)** | **2** | **0.46** | best/tied |
| DIME | 16 | 0.97 | comparable |
| FlowRL | 11 | 0.42 | comparable |
| SAC-Flow | 4 | 0.96 | comparable |
| MaxEntDP | 20 | 1.56 | slightly lower |
| DACER | 20 | 1.06 | comparable |
| QVPO | 20 | 1.68 | slightly lower |
| TD3 (Gaussian) | 1 | 0.14 | lower（unimodal） |
| SAC (Gaussian) | 1 | 0.15 | lower（unimodal） |

MFPO 2 步采样 0.46ms inference time，比其他 diffusion methods 快 2-3.5×；性能 ≥ 所有 diffusion baseline。训练时间降 ~50%。

### 关键 ablation（HalfCheetah-v3）

| Ablation | 影响 |
|---|---|
| MeanFlow → Flow Matching policy | 性能下降，average velocity 在 few-step 必要 |
| Remove ADN | 性能下降，likelihood 估计的必要 |
| Only Gaussian proposal | $t \to 1$ ESS 低 |
| Only policy proposal | failed，proposal 不有效 |
| $K_1:K_2 = 1:2$（更多 Gaussian） | 最佳 |
| Fixed temperature | 比 auto 差 |
| $\rho = 0.5$ for target entropy | 最佳 |

### ESS 与方差分析

Figure 1：HalfCheetah-v3 训练 120k iter 后，Gaussian proposal $q^2$ 在 $t \to 1$ ESS 急剧下降；policy proposal $q^1$ 在 $t \to 1$ 仍高；组合后 estimation variance 比任一单 proposal 都低——验证 ESS-weighted SNIS 的核心动机。

### 关键发现

- **2 步采样达到 20 步 diffusion 性能**：MFPO 用 MeanFlow 2 步基本追上 MaxEntDP/QVPO 20 步，inference time 降 3-4×。
- **训练时间降 50%**：相比 DACER/DIME/MaxEntDP，训练时间几乎砍半。
- **ADN 5% overhead 给 accurate likelihood**：相比 naive 数值积分（每步 $d$ 次 backward），ADN 几乎免费。
- **Two-proposal SNIS 是关键**：组合后 variance 显著降低，是 policy update 稳定的关键。
- **HumanoidBench 也行**：高维任务（>50 dim action）上 MFPO 也 match SOTA，scale 到复杂控制。

## 亮点与洞察

- **MeanFlow + MaxEnt RL 是完美组合**：MeanFlow 解决"少步表达力"，MaxEnt 解决"探索"，组合后既快又稳。
- **ADN 是方法学层面的优雅类比**：把"MeanFlow 学 average velocity"的思想迁移到"学 average divergence"，方法学一致。
- **ESS-weighted SNIS 的工程价值**：不是 ad-hoc tuning，而是用 ESS 自动加权，理论有 variance reduction 保证。
- **训练速度降 50% 的实际意义**：让 diffusion-based RL 在工程项目中变可行——以前 20 步训练几天才出 SAC 几小时的结果，现在 2 步训练几小时就到 diffusion 性能。
- **MeanFlow 的复用**：把 generative modeling 的最新进展（average velocity）迁移到 RL，是跨子领域 cross-pollination 的典范。

## 局限与展望

- **2 步采样下表达力极限**：虽然 MeanFlow 减少 discretization error，但 2 步 vs 20 步 multi-modality 上仍有理论 gap，特别复杂任务可能 sweet spot 在 4-8 步。
- **ADN 训练稳定性**：ADN 训练目标依赖 stop-gradient + recursive structure，长时间训练是否漂移没充分验证。
- **两 proposal 的 ratio tuning**：$K_1:K_2 = 1:2$ 在 HalfCheetah 上最佳，跨任务是否需要调没系统消融。
- **Distributional critic 选择**：C51 是默认，QR-DQN / IQN 可能更稳；distributional choice 的影响没单独消融。
- **缺 sample efficiency 对比**：训练时间快但 sample efficiency（每多少环境 step 达到某性能）vs baseline 没明确对比。
- **没在 Atari / pixel-based 上验证**：所有实验 state-based MuJoCo/DMC/HumanoidBench，pixel observation 下 MeanFlow policy 是否仍有效未知。

## 相关工作与启发

- **vs DACER / DIME / MaxEntDP / SDAC**：他们用 diffusion + MaxEnt 但需要 10-20 步采样；MFPO 用 MeanFlow 把步数降到 2，同时方法学上保留 MaxEnt 的精确性。
- **vs FPMD / QVPO / FlowRL**：他们用 diffusion-based RL 但不是 MaxEnt 框架；MFPO 同时具备 multi-modal expressiveness 和 MaxEnt principled exploration。
- **vs SAC-Flow**：他们用 GRU/Transformer 实现 diffusion policy 训练稳定；MFPO 不依赖特殊架构，方法学更通用。
- **vs MeanFlow (Geng et al. 2025)**：generative modeling 原作，本文是 MeanFlow 在 RL 上的应用。
- **vs TD3+BC / Diffusion-QL**：offline RL 用 diffusion 类似但 online setting 不同；MFPO 专注 online 场景。
- **启发**：(1) 任何"diffusion model 在某领域因 sampling 慢而部署难"的场景都可试 MeanFlow；(2) MeanFlow 的"learn average via consistency"思想可推广到 divergence、Lyapunov function、value function 等其他需要时间积分的量；(3) SNIS 多 proposal 组合是处理 intractable distribution 的通用技巧，ESS-weighting 提供自适应性。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ MeanFlow + MaxEnt RL 组合 + ADN 类比 + ESS-weighted SNIS 三件创新组合，方法学全面。
- 实验充分度: ⭐⭐⭐⭐⭐ MuJoCo/DMC/HumanoidBench 三套 benchmark + 8 个 baseline + 4 维度 ablation + ESS/variance 可视化，证据链完整。
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰、ADN 类比直观、Figure 1 ESS 可视化直击 motivation，方法叙述非常工整。
- 价值: ⭐⭐⭐⭐⭐ 让 diffusion-based RL 在 inference 速度和训练成本上都接近 Gaussian policy，是 diffusion RL 实用化的关键一步；开源代码 + 普适性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](../../AAAI2026/reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[ICML 2026\] Learning to Route Languages for Multilingual Policy Optimization](learning_to_route_languages_for_multilingual_policy_optimization.md)
- [\[ICML 2026\] EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance](eapo_enhancing_policy_optimization_with_on-demand_expert_assistance.md)
- [\[ICML 2026\] Metis: Learning to Jailbreak LLMs via Self-Evolving Metacognitive Policy Optimization](metis_learning_to_jailbreak_llms_via_self-evolving_metacognitive_policy_optimiza.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](../../ACL2026/reinforcement_learning/bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)

</div>

<!-- RELATED:END -->
