---
title: >-
  [论文解读] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities
description: >-
  [NeurIPS 2025][图像生成][Boltzmann采样] 提出 PITA（Progressive Inference-Time Annealing），一种结合温度退火与扩散平滑两种互补插值策略的框架，通过在高温下训练初始扩散模型…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "Boltzmann采样"
  - "温度退火"
  - "Feynman-Kac"
  - "扩散模型"
  - "分子构象"
---

# Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.16471](https://arxiv.org/abs/2506.16471)  
**代码**: [GitHub](https://github.com/taraak/pita)  
**领域**: 扩散模型 / 采样 / 分子模拟  
**关键词**: Boltzmann采样, 温度退火, Feynman-Kac, 扩散模型, 分子构象

## 一句话总结

提出 PITA（Progressive Inference-Time Annealing），一种结合温度退火与扩散平滑两种互补插值策略的框架，通过在高温下训练初始扩散模型，然后利用新颖的 Feynman-Kac PDE 与 SMC 重采样在推理时降温生成低温样本，逐步训练一系列扩散模型直达目标温度，首次实现了对丙氨酸二肽和三肽的笛卡尔坐标下平衡态采样。

## 研究背景与动机

**领域现状**：从未归一化的 Boltzmann 分布中高效采样是计算生物学、化学、物理学中的核心挑战。传统方法包括 MCMC（通常结合并行回火/退火重要性采样）和分子动力学（MD）。MCMC 退火存在"质量传送"问题（模式的权重受其宽度影响），MD 需要极细的时间步长（飞秒级），计算代价巨大。

**扩散采样器的困境**：近年兴起的扩散采样器理论上具有良好的模式混合性，但在实际分子系统中面临三大难题：(1) 缺乏训练数据来准确学习 Stein score；(2) 反向 KL 等训练目标容易模式坍缩；(3) 能量函数评估次数过多。经过仔细调优的 MCMC+并行回火在能量评估次数归一化后甚至优于 SOTA 扩散采样器。

**本文切入角度**：温度退火和扩散路径是两种互补的简化策略——温度退火通过升温消除高能势垒实现模式混合，扩散路径通过加噪避免质量传送。PITA 将两者结合，高温下先用简单 MCMC 收集数据训练扩散模型，再通过推理时退火逐步降温。

## 方法详解

### 整体框架

PITA 训练一系列扩散模型 $\{M_{\beta_i}\}$，从高温 $\beta_0$（低 $\beta$=高温）到目标温度 $\beta_K$（$\beta=1/T$）。每一步：
1. 在温度 $1/\beta_i$ 下用可用样本训练扩散模型（score 模型 + EBM）
2. 利用训练好的模型通过推理时退火生成温度 $1/\beta_{i+1}$ 下的样本
3. 用新样本训练下一个扩散模型

### 关键设计

1. **推理时退火（Proposition 1, 核心创新）**：给定已训练的 score 模型 $s_t(x;\theta) \approx \nabla\log p_t(x)$ 和能量模型 $U_t(x;\eta) \approx -\log p_t(x)$，定义退火后的边缘分布 $q_t(x) \propto \exp(-\gamma U_t(x;\eta))$，其中 $\gamma = \beta_{i+1}/\beta_i > 1$。推导出描述 $q_t$ 时间演化的 Feynman-Kac PDE，对应的采样 SDE 为：
    $dx_t = \left(-a_t x_t + \frac{\zeta_t^2}{2}(s_t(x_t) - \gamma\xi_t \nabla U_t(x_t;\eta))\right)dt + \zeta_t\sqrt{\xi_t}dW_t$
   权重更新：
    $d\log w_t = \left[\frac{\zeta_t^2}{2}\langle\nabla, s_t\rangle - \gamma\langle\nabla U_t, -a_t x + \frac{\zeta_t^2}{2}s_t\rangle - \gamma\frac{\partial U_t}{\partial t}\right]dt$
   **设计动机**：当 $\gamma=1$（无退火）且模型完美时，权重方差为零（Proposition 2），退化为标准扩散采样。这确保了当退火步长小时重要性权重集中，采样效率高。

2. **训练阶段（Algorithm 1）**：同时优化四个损失函数：

    - **Denoising Score Matching**：学习 score 模型 $D_t(x_t;\theta)$
    - **Target Score Matching**：在大噪声水平时利用目标分布的 score $\nabla_x \log \pi(x)$ 直接监督（仅在 $t \geq t_{\text{thresh}}$ 时启用），弥补 DSM 在接近数据分布时的高方差
    - **EBM Distillation**：将 score 模型蒸馏到能量模型
    - **Energy Pinning**：用目标刚能量 $\beta_{i+1}\log\pi(x)$ 监督端点能量模型 $U_{t=1}(x;\eta)$，固定能量的gauge（平移不变性）

3. **几何退火变体（Proposition 3 / Appendix A.2）**：对于无界支撑（如 $\text{supp}(\pi) = \mathbb{R}^d$），直接退火可能导致数值不稳定。改用几何平均：$\mathcal{N}(0,\mathbb{1})^{(1-\beta)}\pi(x)^\beta$，确保在任何温度下分布都是可归一化的。

### 网络架构与训练细节

- LJ-13：使用 EGNN 作为骨干网络，单个扩散模型以 $\beta$ 为条件
- 丙氨酸二肽/三肽：使用 DiT 作为骨干，采用微调策略（每个温度步只用当前温度的样本训练）
- 能量模型参数化遵循 Thornton et al. (2025)，预处理使用 Karras et al. (2022) 的方案

## 实验关键数据

### 主实验1：LJ-13 粒子系统（$T_L=4 \to T_S=1$）

| 方法 | Distance-$\mathcal{W}_2$ ↓ | Energy-$\mathcal{W}_2$ ↓ | Geometric-$\mathcal{W}_2$ ↓ |
|------|---------------------------|--------------------------|------------------------------|
| iDEM | 0.127 | 30.78 ± 24.46 | 1.61 ± 0.01 |
| Adjoint Sampling | - | 2.40 ± 1.25 | 1.67 ± 0.01 |
| TA-BG (TarFlow) | 1.21 ± 0.02 | 61.47 ± 0.12 | 4.16 ± 0.01 |
| **PITA** | **0.04 ± 0.00** | **2.26 ± 0.21** | **1.65 ± 0.00** |

PITA 的 Distance-$\mathcal{W}_2$ 比 iDEM 低 3 倍，比 TA-BG 低 30 倍。

### 主实验2：丙氨酸二肽 ALDP（$T_L=1200K \to T_S=300K$）

| 方法 | Rama-KL | Tica-$\mathcal{W}_1$ ↓ | Energy-$\mathcal{W}_1$ ↓ | Energy-$\mathcal{W}_2$ ↓ | $\mathbb{T}$-$\mathcal{W}_2$ |
|------|---------|------------------------|-------------------------|--------------------------|-----|
| **PITA** | **4.773** | **0.112** | **1.530** | **1.615** | 0.270 |
| MD-Diff | 1.308 | 0.113 | 3.627 | 3.704 | 0.310 |
| TA-BG | 14.993 | 0.219 | 83.457 | 86.176 | 0.979 |
| Score Scaling | 4.588 | 0.183 | 10.282 | 10.460 | 0.550 |

PITA 在能量指标上大幅领先：Energy-$\mathcal{W}_1$ 比 MD-Diff 低 **58%**，比 TA-BG 低 **98%**。

### 主实验3：丙氨酸三肽 AL3（$T_L=1200K \to T_S=300K$）

| 方法 | Rama-KL | Tica-$\mathcal{W}_2$ ↓ | Energy-$\mathcal{W}_1$ ↓ | Energy-$\mathcal{W}_2$ ↓ | 能量评估次数 |
|------|---------|------------------------|-------------------------|--------------------------|------------|
| **PITA** | **1.209** | 0.952 | **2.567** | **2.592** | $8\times10^7$ |
| MD-Diff | 9.662 | 0.426 | 7.416 | 7.599 | $8\times10^7$ |
| TA-BG | 2.078 | 0.454 | 4.782 | 4.863 | $8\times10^7$ |

PITA 是首个能对丙氨酸三肽进行笛卡尔坐标下平衡态采样的扩散方法。

### 消融实验

| 配置 | Energy-$\mathcal{W}_1$ (ALDP) | 说明 |
|------|------------------------------|------|
| PITA (完整) | 1.530 | 含 MD 松弛 |
| PITA (无松弛) | 86.270 | 无 MD 松弛，性能大幅下降 |
| FKC (Skreta 2025) | 11.281 | 仅在最终步重采样 |
| Score Scaling | 10.282 | 简单 score 缩放 |

### 关键发现

- PITA 不仅性能 SOTA，更关键的是能量评估次数可比——$5-8\times10^7$ 次，远少于 MD 达到同等质量所需
- TA-BG 在高温时表现尚可，但随温度降低性能急剧退化，因为重要性采样的方差在大温差下膨胀
- TICA 图表明 PITA 成功恢复了分子系统的慢动态模式
- 目标温度附近的短 MD 松弛能显著改善轨迹的物理合理性

## 亮点与洞察

- 温度退火 + 扩散路径的互补组合非常优雅：退火处理模式混合，扩散避免质量传送
- Feynman-Kac PDE 的推导提供了退火扩散的统一数学框架，将多种现有方法作为特例
- Proposition 2 保证了无退火时权重方差为零，确保了小步长退火的稳定性
- 渐进式训练（从高温到低温）的顺序微调策略在实践中非常有效

## 局限与展望

- 需要同时训练 score 模型和 EBM，EBM 训练本身就是一个挑战性问题
- 如何自动确定最优的温度调度（步数和步长）仍待解决
- Tica-$\mathcal{W}$ 指标表明 PITA 在模式权重恢复方面不如一些基线，可能需要更精细的重采样
- 当前仅在小分子系统（13个原子、二/三肽）上验证，向更大蛋白质系统的扩展尚不明确

## 相关工作与启发

- 与 Boltzmann Generators（Noé et al. 2019）的关系：PITA 不使用直接的重要性采样，而是通过 Feynman-Kac 在扩散时间上进行退火重要性采样
- 与 iDEM、Adjoint Sampling 等无数据扩散采样器互补：PITA 利用高温 MCMC 数据启动
- 温度调度可与 Annealed Importance Sampling 和 Parallel Tempering 交叉借鉴
- 有望与可迁移采样（Klein & Noé 2024）结合，实现跨分子系统的泛化

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ Feynman-Kac PDE 推导精妙，温度退火+扩散模型的渐进训练框架原创性强
- **实验充分度**: ⭐⭐⭐⭐ 涵盖 LJ-13、ALDP、AL3，但仅限小分子系统（受领域限制）
- **写作质量**: ⭐⭐⭐⭐ 数学推导严谨，但符号和辅助结果较多
- **价值**: ⭐⭐⭐⭐⭐ 首次在笛卡尔坐标下实现肽链的平衡态采样，是扩散采样器的重要里程碑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Ψ-Sampler: Initial Particle Sampling for SMC-Based Inference-Time Reward Alignment in Score Models](psi-sampler_initial_particle_sampling_for_smc-based_inference-time_reward_alignm.md)
- [\[NeurIPS 2025\] BoltzNCE: Learning Likelihoods for Boltzmann Generation with Stochastic Interpolants](boltznce_learning_likelihoods_for_boltzmann_generation_with_stochastic_interpola.md)
- [\[NeurIPS 2025\] Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models](cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)
- [\[NeurIPS 2025\] Diffusion-Driven Progressive Target Manipulation for Source-Free Domain Adaptation](diffusion-driven_progressive_target_manipulation_for_source-free_domain_adaptati.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)

</div>

<!-- RELATED:END -->
