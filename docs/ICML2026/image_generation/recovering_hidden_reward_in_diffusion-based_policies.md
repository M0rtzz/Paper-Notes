---
title: >-
  [论文解读] Recovering Hidden Reward in Diffusion-Based Policies
description: >-
  [ICML 2026][图像生成][扩散模型] EnergyFlow 把 diffusion policy 的 score field 显式参数化为一个标量 energy function 的负梯度，论证了 maximum-entropy 最优下 score = 软 Q-函数梯度…
tags:
  - "ICML 2026"
  - "图像生成"
  - "扩散模型"
  - "IRL"
  - "energy-based model"
  - "保守场"
  - "反向 RL shaping"
---

# Recovering Hidden Reward in Diffusion-Based Policies

**会议**: ICML 2026  
**arXiv**: [2605.00623](https://arxiv.org/abs/2605.00623)  
**代码**: https://github.com/sotaagi/EnergyFlow  
**领域**: 扩散策略 / 逆强化学习 / 能量模型 / 机器人操作  
**关键词**: diffusion policy, IRL, energy-based model, 保守场, 反向 RL shaping

## 一句话总结
EnergyFlow 把 diffusion policy 的 score field 显式参数化为一个标量 energy function 的负梯度，论证了 maximum-entropy 最优下 score = 软 Q-函数梯度，从而在不做对抗优化的情况下"白送"一个可用作下游 RL shaping reward 的标量信号，同时保守场约束改善 OOD 泛化。

## 研究背景与动机

**领域现状**：diffusion policy（Chi 2023、Flow Policy）已成为机器人操作的主流方案——能建模多模 expert 动作分布、在 RoboMimic / Meta-World 上 BC 性能强。但它本质上只是行为克隆，模型只学了"专家做了什么"而没显式学"为什么这么做"。

**现有痛点**：行为克隆有两个连锁问题。一是 OOD 鲁棒性差——一旦测试场景偏离 demo 分布，单凭 action likelihood 没法可靠 ranking actions；二是不能直接生成 reward 信号给下游 RL refine。已有 IRL 方法（max-ent IRL、GAIL、AIRL）虽然能从 demo 恢复 reward，但要么需要昂贵 MCMC（EBM），要么需要不稳定的 adversarial training（GAIL/AIRL），要么需要重复 inner-loop policy optimization。

**核心矛盾**：diffusion policy 已经在 latent 里"知道"专家偏好（score 就是 $\nabla \log p$），但作者们把它当 sampler 用、扔掉了内含的 reward 信号；而 IRL 方法明知要 reward 但又要从零再训练一个 EBM 或对抗判别器。

**本文目标**：让一个网络同时（i）作为 generative policy 生成动作，（ii）暴露一个可用作 reward 的标量 energy，（iii）保留 diffusion policy 的 BC 强度而不引入额外训练 cost。

**切入角度**：观察到 score function $\nabla_a \log \pi_E(a|s)$ 和 max-ent expert 的 soft Q-function 在 log 空间线性相关；如果把 score field 限制为某个 scalar potential 的负梯度（即"保守场"），就同时拿到三件事——valid energy、消除 cyclic preference、收紧 Rademacher 复杂度从而改善 OOD 泛化。

**核心 idea**：参数化一个标量 $E_\phi(s, a)$，让 score $\mathcal{S}_\phi = -\nabla_a E_\phi$ 通过 autodiff 拿到，用 denoising score matching 训练。这样 $E_\phi$ 同时是 diffusion policy 的 generative 势函数和 max-ent IRL 的 reward 函数。

## 方法详解

### 整体框架

输入：expert demos $\mathcal{D} = \{(s_i, a_i)\}$。
训练：标量网络 $E_\phi: \mathcal{A} \times \mathcal{S} \times [0, T] \to \mathbb{R}$，用 variance-exploding 噪声 $\sigma(t) = \sigma_{\min}^{1-t/T} \sigma_{\max}^{t/T}$ 给动作加噪，DSM 目标 $\mathcal{L}(\phi) = \mathbb{E}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$。
推理两用：（生成）从 $a_T \sim \mathcal{N}(0, \sigma^2(T) I)$ 出发跑 probability-flow ODE $da/dt = -\frac{1}{2} \frac{d[\sigma^2(t)]}{dt} \nabla_a E_\phi$；（reward）在 $\gamma = 10^{-3}$ 时刻取 $E_\phi(a, s, \gamma)$ 减去 state-dependent baseline 作为 shaping reward 喂给 SAC。

### 关键设计

1. **Score = Reward Gradient 等价性（Theorem 3.3）**:

    - 功能：从数学上证明 score matching 训出的 $E_\phi$ 自动恢复了 expert 的 soft Q-function（差一个 state-dependent 常数）。
    - 核心思路：在 max-ent 最优假设下，$\pi_E(a|s) = \exp(Q^*(s,a)/\alpha) / Z(s)$。对动作求梯度可消掉 partition function $Z(s)$：$\nabla_a \log \pi_E = \nabla_a Q^* / \alpha$。所以如果 $-\nabla_a E_\phi \approx \nabla_a \log \pi_E$，那 $E_\phi(a, s) = -Q^*(s,a)/\alpha + c(s)$。Corollary 3.4 进一步指出 $E_\phi$ 实际恢复的是 soft advantage $A^{\text{soft}}(s,a) = Q^*(s,a) - V^*(s)$（再差一个 state-only 常数）。
    - 设计动机：这是整个框架的理论 anchor——它告诉我们不需要 GAIL/AIRL 的对抗判别器、不需要 EBM 的 MCMC，仅仅训一个 diffusion policy 就能"白送"一个 reward 信号。区别于把 diffusion 当 sampler 用的 Diffusion Policy，这里把同一个网络重 interpret 成 energy。

2. **保守场约束（标量参数化 + autodiff score）**:

    - 功能：硬保证 score field 是某 scalar potential 的梯度（$\nabla \times \mathcal{S}_\phi = 0$），消除 cyclic preference，并收紧泛化 bound。
    - 核心思路：不直接回归向量 score $\mathcal{S}_\phi: \mathbb{R}^{|s|+|a|} \to \mathbb{R}^{|a|}$，而是把网络输出设为 scalar $E_\phi$，通过 autodiff 拿 $\mathcal{S}_\phi = -\nabla_a E_\phi$。Theorem 3.6 给出 Rademacher 复杂度对比：$\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{unc}}) \leq \Lambda B \sqrt{d}/\sqrt{n}$（无约束）vs $\hat{\mathfrak{R}}_S(\mathcal{F}_{\text{cons}}) \leq \Lambda L/\sqrt{n}$（保守），高维 action space 下保守版本严格更紧。Lemma 3.8 进一步给 OOD bound：保守版本的复杂度项是 $\mathcal{O}(M \Lambda L / \sqrt{n})$ 而非 $\mathcal{O}(M \Lambda B \sqrt{d}/\sqrt{n})$。
    - 设计动机：常规 diffusion policy 直接输出向量 score 不保证保守性，意味着学到的 implicit "energy"可能在 $a_1 \to a_2 \to a_3 \to a_1$ 形成 cyclic preference，违反 rational decision 的传递性公理（Jiang 2011），rewards 没法良定义。保守约束既是数学正确性（让 reward extraction 合法），也是有用的 inductive bias（高维动作空间下显著降复杂度）。Remark 3.7 还指出深度网络可通过 spectral normalization 满足 Lipschitz 约束让 $L$ 可控。

3. **Centered Shaping Reward（去 state-dependent 偏置）**:

    - 功能：把 raw $E_\phi$ 当 reward 用会有 state-dependent 偏置 $c(s)$ 引入高方差；定义 $\tilde{r}_\phi(a, s) = -(E_\phi(a, s, \gamma) - \mathbb{E}_{a' \sim \mathcal{N}(0, I)}[E_\phi(a', s, \gamma)])$ 把偏置 cancel。
    - 核心思路：Proposition 3.9 证 raw $E_\phi$ 保证 within-state action ranking 正确（$\arg\min_a E_\phi = \arg\max_a Q^*$）但 cross-state 比较不可靠。Remark 3.10 指出 state-only 偏置不满足 potential-based reward shaping (PBRS, Ng 1999) 形式，可能改变 sequential 最优策略；但对 within-state action selection 偏置无影响。所以减一个 state-dependent baseline（Monte Carlo 采 $M = 16$ 个 $a' \sim \mathcal{N}(0, I)$ 求均值）就把偏置干掉。
    - 设计动机：实验图 4 直接验证——raw $E_\phi$ 作为 SAC reward 会让 agent 卡在常见 state 上（因为高 likelihood = 低 energy = 高 reward，但 likelihood 高不等于 progress），early plateau；centered 版让 reward 反映"在当前 state 下选哪个 action"而非"哪个 state 常被访问"，训练曲线追到 oracle dense reward 水平。Centered Energy + Sparse 组合最好——稠密 shaping 引导早期 exploration，sparse 保证最终任务对齐。

### 损失函数 / 训练策略

DSM 损失 $\mathcal{L}(\phi) = \mathbb{E}_{t, a_0, \varepsilon}[\sigma^2(t) \| -\nabla_{a_t} E_\phi(a_t, s, t) + \varepsilon/\sigma(t) \|^2]$，其中 $a_t = a_0 + \sigma(t) \varepsilon$，$\varepsilon \sim \mathcal{N}(0, I)$，$t \sim \mathcal{U}[0, T]$，$\sigma_{\min} = 0.01$，$\sigma_{\max} = 10$，$T = 1$。$\lambda(t) = \sigma^2(t)$ 确保各噪声尺度均衡贡献。Theorem 3.11 给 score 误差 $\eta$ 到 action preference 的传播 bound：$|\Delta E_\phi(a, a') - \Delta E^*(a, a')| \leq \eta \cdot \|a - a'\|_2$，linear graceful degradation。下游 RL 用 SAC + centered shaping reward，可选叠加 sparse task signal。

## 实验关键数据

### 主实验

RoboMimic（ph，5 任务，3 seed）：

| 方法 | Lift | Square | Transport | ToolHang | Avg |
|------|------|--------|-----------|----------|-----|
| LSTM-GMM | 97.8 | 64.3 | 65.6 | 46.0 | 69.0 |
| Diffusion Policy | 100.0 | 93.5 | 85.9 | 77.2 | 91.2 |
| Flow Policy | 99.6 | 91.8 | 83.6 | 74.8 | 89.6 |
| EBT-Policy | 96.2 | 78.4 | 72.4 | 58.6 | 78.8 |
| EBIL / NEAR / IQ-Learn | 92-95 | 58-68 | 48-58 | 32-44 | 61-70 |
| Implicit BC | 70.9 | 10.2 | 0.0 | 0.0 | 22.4 |
| **EnergyFlow** | **100.0** | **95.3** | **89.4** | **84.2** | **93.8** |

Meta-World（5 任务）：EnergyFlow 92.5% vs Diffusion Policy 90.7%，提升集中在难任务（Assembly +6.2，ToolHang +7.0 on RoboMimic）。
Real robot（AGIBOT G1）：Bottle / Drawer 两任务 100% 成功率（3 初始位置 × 20 rollouts）。

### 消融实验

下游 SAC reward 来源对比（RoboMimic Square / Transport）：

| Reward 源 | 收敛 success | 说明 |
|-----------|--------------|------|
| Sparse only | 慢且噪声大 | 任务成功才有信号 |
| Raw $E_\phi$ | 早期 plateau | likelihood ≠ progress，偏 state 密度 |
| Centered $E_\phi$ | 接近 oracle dense | 减 baseline 后反映 within-state action 偏好 |
| Centered + Sparse | 最佳 | 稠密 shape + sparse 锚定任务 |

OOD 扰动（初始位置 perturbation level 0/S/M/L）：EnergyFlow 在 M/L 级扰动下显著优于 Diffusion Policy 和 Flow Policy，验证保守约束的几何 regularization 效果。
$\gamma$ 敏感性（reward extraction time）：$\gamma \in [10^{-4}, 10^{-2}]$ 性能稳（94-95%），$\gamma \geq 0.1$ 掉点（数据分布已被噪声破坏，score 近似失效）。
Latency（RoboMimic Square，A100）：EnergyFlow $K=20$ 11.4ms（95.3%）vs Diffusion Policy 100 DDPM 32.4ms（93.5%）vs Implicit BC 50 Langevin 52.4ms（10.2%）——保守 EBM 在不牺牲速度下击败 baseline。

### 关键发现
- 保守约束在难任务上贡献最大（ToolHang +7 vs Diffusion Policy），印证 Lemma 3.8——任务越难、action space 越复杂，受益越大
- centered shaping 是 reward extraction 的 key——raw $E_\phi$ 训 RL 会卡，centered 训能追到 oracle
- 显式 EBM（Implicit BC）在 RoboMimic 上几乎全跪（Transport / ToolHang 0%）但 EnergyFlow 干到 SOTA，说明问题不在 EBM 范式本身而在训练目标——score matching + conservative parameterization 比传统 contrastive divergence 友好太多
- inference latency 与 Flow Policy 同量级，证明保守约束的 autodiff 开销可忽略
- 实机迁移直接 work（100% on AGIBOT G1），说明 conservative 约束的几何 regularization 真的帮上忙

## 亮点与洞察
- **"score matching = max-ent IRL，免费送 reward"**这个 reinterpretation 太干净——之前大家把 diffusion policy 和 IRL 当两个独立 stream，本文一句"对 $a$ 求梯度消掉 $Z(s)$"就把它们接到一起，这种重新解读型 paper 价值极高
- **scalar parameterization + autodiff score** 工程上几乎零额外 cost（就是把网络最后 head 改成 1-d、forward 时多一个 $\nabla_a$），却同时拿到（i）valid energy（ii）保守场（iii）更紧 generalization bound
- **centered shaping** 的 baseline trick 解决了"likelihood 高 ≠ task progress"的经典问题——如果直接把 $\log \pi_E$ 当 reward，agent 会被高 density region 吸住，而 centered 版让信号变成"相对 within-state preference"，这个 idea 可以直接 transfer 到任何 likelihood-based reward shaping
- **保守约束的 OOD 好处**有理论支撑（Lemma 3.8 + Rademacher 复杂度）也有实验验证（perturbation level L 下显著优于 baseline），双重确认很有说服力
- inference latency 和 Flow Policy 一致，把 EBM 从"贵到不能用"洗白回"主流候选"

## 局限与展望
- max-ent 最优假设需要 expert 真正按 Boltzmann 分布行为，对 noisy demo / multi-expert 集合可能失效
- 恢复的是 soft advantage $A^{\text{soft}}$（差 state-only 偏置），cross-state 比较不可靠；在 sequential MDP 上 state-only 偏置可能改变最优策略，作者承认但没给替代方案
- 保守场理论 bound（Theorem 3.6）的紧度 $L \ll B \sqrt{d}$ 在深网络下要靠 spectral normalization 维持，实际训练里是否满足没量化检查
- 实机只测 2 个任务、3 初始位置，统计意义有限
- 只在 manipulation 上验证，对 locomotion / 连续控制等其他 embodied 任务的迁移性未知
- baseline 用 16 个 MC sample 估计，估计方差对 RL 训练稳定性的影响没系统消融

## 相关工作与启发
- **vs Diffusion Policy (Chi 2023)**：本文是它的 strict superset——backbone、训练目标、生成 latency 都接近，但多输出一个 reward 标量且 BC 性能更强；这种"更精简 hypothesis class + 更多输出"的 win-win 设计值得借鉴
- **vs Implicit BC (Florence 2021) / EBT-Policy (Davies 2025)**：都是 EBM-based policy，但前者用 contrastive divergence 训练不稳定（RoboMimic 几乎全跪），EBT 用 transformer 但仍然只把 energy 当 decision score；EnergyFlow 用 score matching 训练且把 energy 当 reward 用
- **vs EBIL (Liu 2021) / NEAR (Diwan 2025)**：同样用 EBM 做 IRL 的两阶段 pipeline（先学 energy 再 RL），但 EBIL/NEAR 没保守约束、性能差且没把 energy 当 generative policy 用；EnergyFlow 把两件事合一
- **vs Adversarial IRL (GAIL, AIRL, AIRL)**：完全绕开对抗训练，单 score matching loss 训练稳定
- **vs Wang & Du 2025 / Balcerak 2025**：同样观察到 diffusion 和 EBM 的连接，EnergyFlow 进一步把这种连接落到 IRL reward extraction

## 评分
- 新颖性: ⭐⭐⭐⭐ "score = soft Q gradient + 保守约束"的 reinterpretation 漂亮，但单看 EBM-policy / DSM 等组件都是已知
- 实验充分度: ⭐⭐⭐⭐⭐ 10 个 sim 任务 + 实机 + reward quality 直接用 SAC 验证 + OOD perturbation + 5 个 sensitivity 实验，全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论部分定理-证明结构清晰，三个 Remark 把假设的边界讲清楚；图 1 把 EnergyFlow 与 Diffusion Policy 的差别一图说明
- 价值: ⭐⭐⭐⭐⭐ 直接给 diffusion policy 加上 reward 输出 + OOD 鲁棒性，且推理速度不变，对机器人学习社区是极实用的升级

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](../../ICLR2026/image_generation/improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[ICML 2026\] Learning General Causal Structures with Hidden Dynamic Process for Climate Analysis](learning_general_causal_structures_with_hidden_dynamic_process_for_climate_analy.md)
- [\[AAAI 2026\] PADiff: Predictive and Adaptive Diffusion Policies for Ad Hoc Teamwork](../../AAAI2026/image_generation/padiff_predictive_and_adaptive_diffusion_policies_for_ad_hoc_teamwork.md)
- [\[ICLR 2026\] A Hidden Semantic Bottleneck in Conditional Embeddings of Diffusion Transformers](../../ICLR2026/image_generation/a_hidden_semantic_bottleneck_in_conditional_embeddings_of_diffusion_transformers.md)
- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](../../ICLR2026/image_generation/compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)

</div>

<!-- RELATED:END -->
