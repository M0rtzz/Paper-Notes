---
title: >-
  [论文解读] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models
description: >-
  [NeurIPS 2025][图像生成][Langevin 动力学] 将 PID 控制理论引入 Langevin 动力学采样，利用梯度历史（积分项）提供动量穿越能量壁垒、利用梯度趋势（微分项）抑制振荡实现快速稳定收敛，无需额外训练即可在 SGM 和 EBM 上实现 10 倍以上采样加速。
tags:
  - NeurIPS 2025
  - 图像生成
  - Langevin 动力学
  - PID 控制
  - 采样加速
  - 能量模型
  - 无训练
---

# PID-controlled Langevin Dynamics for Faster Sampling of Generative Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.12603](https://arxiv.org/abs/2511.12603)  
**代码**: [GitHub](https://github.com/tsinghua-fib-lab/PIDLD)  
**领域**: 扩散模型 / 采样加速  
**关键词**: Langevin 动力学, PID 控制, 采样加速, 能量模型, 无训练

## 一句话总结

将 PID 控制理论引入 Langevin 动力学采样，利用梯度历史（积分项）提供动量穿越能量壁垒、利用梯度趋势（微分项）抑制振荡实现快速稳定收敛，无需额外训练即可在 SGM 和 EBM 上实现 10 倍以上采样加速。

## 研究背景与动机

Langevin 动力学是 EBM（能量模型）和 SGM（分数匹配生成模型）中核心的采样方法，但存在根本性的速度瓶颈——需要大量细粒度迭代才能收敛到目标分布。例如 NCSNv2 需要 1000+ 次神经函数评估（NFE）。

问题的物理本质是：在复杂的高维能量景观中，粒子频繁遭遇**梯度近零区域**（局部极小值或不稳定平衡点），必须依赖随机噪声逃逸，这极度低效。而直接增大步长会引入更大噪声波动，严重降低质量。

作者的关键洞察来自控制理论：标准 Langevin 动力学等价于一个简单的比例（P）反馈控制系统，仅利用当前梯度信息。而控制理论早已证明，加入**积分（I）**和**微分（D）**项可以大幅提升控制系统的响应速度和稳定性。这一类比为采样加速开辟了全新视角。

## 方法详解

### 整体框架

将 Langevin 采样重新解释为反馈控制问题——能量梯度 $\nabla_x U_\theta(x)$ 作为反馈信号，粒子作为被控系统。在标准的比例项（当前梯度）基础上引入积分项（梯度历史）和微分项（梯度变化趋势），形成完整的 PID 控制器驱动采样过程。

### 关键设计

1. **PID 控制的 Langevin 动力学**: 核心更新公式为：
   $$x_{t+1} = x_t + \epsilon\left(k_p \nabla_x U_\theta(x_t) + \frac{k_i}{t}\sum_{s=0}^{t}\nabla_x U_\theta(x_s) + k_d(\nabla_x U_\theta(x_t) - \nabla_x U_\theta(x_{t-1}))\right) + \sqrt{2\epsilon}\xi_t$$
   - **P 项**（$k_p$）：标准梯度引导，响应当前误差
   - **I 项**（$k_i/t \cdot \sum$）：累积梯度历史创造动量效应，帮助穿越能量壁垒和不稳定平衡点。$1/t$ 归一化防止积分项随时间主导
   - **D 项**（$k_d$）：捕捉梯度变化率，在梯度持续下降方向加速运动，在梯度持续上升处抑制运动，减少超调

2. **指数衰减的积分增益调度**: 积分项的增益随时间衰减 $k_i(t) = \gamma^t \cdot k_i$（$\gamma < 1$）。这是受控制理论中增益调度的启发：早期需要高积分增益构建动量穿越壁垒，后期需要低增益避免不稳定。衰减确保采样最终回归标准 Langevin 动力学，保证理论收敛性。

3. **与退火 Langevin 动力学的无缝集成**: PIDLD 直接替换每个噪声尺度 $\sigma_i$ 内的标准 Langevin 步骤。关键创新是跨噪声尺度保持状态连续——前一尺度的最终 $I_t$ 和梯度（用于 D 项）作为下一尺度 $\sigma_{i-1}$ 的初始状态，确保历史信息在整个采样过程中持续传递。

### 收敛性保证

论文给出了微分项保持收敛的理论保证：在局部强凸能量景观下，当步长满足 $\epsilon < \frac{1}{(1+2k_d)m}$（$m$ 为强凸参数）时，系统渐近稳定并收敛到唯一平稳分布。这证明 D 项是合理的稳定化组件，不会破坏采样过程。

## 实验关键数据

### 主实验：图像生成 FID 对比

| 数据集 | 方法 | NFE=25(SGM) | NFE=100(SGM) | NFE=232×5(SGM) | NFE=20(EBM) | NFE=40(EBM) |
|--------|------|-------------|--------------|----------------|-------------|-------------|
| CIFAR10 | Vanilla ALD | 46.8 | 17.2 | 12.5 | 58.1 | 35.3 |
| | MILD | - | - | 13.0 | 49.9 | 34.4 |
| | **PIDLD** | **18.3** | **12.1** | **11.4** | **46.1** | **33.2** |
| CelebA | Vanilla ALD | 25.0(50) | 13.6(250) | 9.5(500×5) | 63.5(20) | 35.4(30) |
| | MILD | - | - | 11.0 | 41.1 | 32.9 |
| | **PIDLD** | **8.0**(50) | **5.7**(250) | **5.6**(500×5) | **38.9**(20) | **30.0**(30) |

### 推理任务：Sudoku 和 Connectivity 准确率

| 任务 | 方法 | NFE=5 | NFE=10 | NFE=30 |
|------|------|-------|--------|--------|
| Sudoku | Vanilla | 45.99% | 51.00% | 50.77% |
| | MILD | 49.75% | 54.82% | 55.25% |
| | **PIDLD** | **50.54%** | **55.48%** | **55.94%** |
| Connectivity | Vanilla | 86.16%(1) | 87.22%(2) | 87.49%(5) |
| | MILD | 86.16%(1) | 88.54%(2) | 90.15%(5) |
| | **PIDLD** | **86.16%**(1) | **91.32%**(2) | **92.95%**(5) |

### 消融实验

| 配置 | CIFAR10 FID (NFE=25) | CelebA FID (NFE=50) | 说明 |
|------|---------------------|---------------------|------|
| P only (baseline) | 46.8 | 25.0 | 标准 ALD |
| P + I | ~30 | ~15 | I 项帮助穿越壁垒 |
| P + D | ~22 | ~10 | D 项是图像生成主要贡献者 |
| P + I + D (PIDLD) | **18.3** | **8.0** | 两者互补，完整模型最优 |

### 关键发现

- **10 倍以上加速**: SGM 上 PIDLD 用 100 NFE 即达到基线最优性能（需 1000+ NFE），CelebA 上效果更显著（38.3% FID 改善）
- **D 项在图像生成中主导**: 因为退火 Langevin 的早期能量景观被噪声平滑，壁垒较浅，I 项穿越能力发挥受限；D 项帮助在每个噪声尺度快速收敛到局部井心
- **I 项在推理任务中主导**: 推理任务需要找到全局能量最小值，I 项的动量效应在高 NFE 时优势持续增加（Connectivity 任务中，PIDLD 仅用 2 NFE 即超过基线 10 NFE 的性能）
- **与 MILD 的比较优势**: PIDLD 在所有配置上均优于仅动量方法 MILD，且性能更稳定，受益于 PID 的综合反馈机制

## 亮点与洞察

- **控制理论与生成模型的精彩交叉**: 将采样问题重新建模为控制问题，P/I/D 三个项的物理意义和控制理论意义完美对齐
- **无需训练的即插即用**: 不需要额外数据、先验信息或重训练，直接替换采样器即可使用
- **I 项和 D 项的互补性**: 二者分别在推理和生成任务中扮演主角，说明完整 PID 控制提供了任务自适应的最优平衡
- **增益衰减保证理论收敛**: $\gamma^t$ 衰减使系统最终退化为标准 LD，兼顾早期加速和后期理论保证

## 局限与展望

- 主要面向 Langevin 采样的模型（EBM、SGM），不直接适用于 DDPM/DDIM 等 ODE 采样器
- I 项的 $1/t$ 归一化和指数衰减的 $\gamma$ 引入了额外超参数，需要调优
- 理论收敛仅在局部强凸设定下证明，多模态分布的全局收敛保证尚缺
- 未与 DDIM 等 ODE 采样器直接对比（作者声明这不是同类方法）

## 相关工作与启发

- **CLD**（临界阻尼 Langevin 扩散）通过 HMC 加速但需额外学习扩散速度，PIDLD 免训练
- **矩阵预条件方法** 依赖目标数据统计，泛化性受限；PIDLD 不需要先验
- **MILD** 是最直接对标的方法，仅用动量（相当于只有 I 项），PIDLD 的 D 项提供了额外的稳定化
- PID 控制在深度学习优化器中已有应用（如加速梯度下降），本文是首次应用于生成模型采样

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 控制理论视角全新，P/I/D 各项的作用机理清晰可验证
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 SGM/EBM/推理任务，但仅测试低分辨率图像
- **写作质量**: ⭐⭐⭐⭐ 理论推导与实验分析搭配合理，toy 实验的引导性强
- **价值**: ⭐⭐⭐⭐ 对 Langevin 采样有显著改进，但适用范围限于LD采样器

<!-- RELATED:START -->

## 相关论文

- [Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems](preconditioned_langevin_dynamics_with_score-based_generative_models_for_infinite.md)
- [Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models](cross-fluctuation_phase_transitions_reveal_sampling_dynamics_in_diffusion_models.md)
- [Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)
- [Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)

<!-- RELATED:END -->
