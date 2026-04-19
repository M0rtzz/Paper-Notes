---
title: >-
  [论文解读] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning
description: >-
  [ICLR 2026][连续归一化流] 提出PolicyFlow，将连续归一化流(CNF)策略无缝嵌入PPO框架：通过沿插值路径的速度场变化近似重要性比率（避免全流路径反向传播），并引入受布朗运动启发的隐式熵正则器防止模式坍缩，在MultiGoal/PointMaze/IsaacLab/MuJoCo等环境中达到或超越高斯PPO和流式基线(FPO/DPPO)的性能。
tags:
  - ICLR 2026
  - 连续归一化流
  - PPO
  - 多模态策略
  - 重要性比率近似
  - 布朗运动熵正则
---

# PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2602.01156](https://arxiv.org/abs/2602.01156)  
**代码**: [项目页面](https://policyflow2026.github.io/)  
**领域**: 强化学习/策略优化  
**关键词**: 连续归一化流, PPO, 多模态策略, 重要性比率近似, 布朗运动熵正则

## 一句话总结

提出PolicyFlow，将连续归一化流(CNF)策略无缝嵌入PPO框架：通过沿插值路径的速度场变化近似重要性比率（避免全流路径反向传播），并引入受布朗运动启发的隐式熵正则器防止模式坍缩，在MultiGoal/PointMaze/IsaacLab/MuJoCo等环境中达到或超越高斯PPO和流式基线(FPO/DPPO)的性能。

## 研究背景与动机

**领域现状**：PPO是在线强化学习最主流的策略梯度方法，广泛用于机器人控制和LLM对齐。其核心是通过重要性比率(importance ratio)来更新策略，通常假设策略为高斯分布以简化似然计算。然而，高斯策略只能表示单峰分布，无法建模复杂的多模态动作。

**现有痛点**：

- **高斯策略表达力不足**：在需要多目标到达、多路径规划等场景下，高斯策略只能覆盖一个模式
- **生成式策略的似然计算代价高**：连续归一化流(CNF)和扩散模型表达力足够，但计算重要性比率需沿ODE完整路径反向传播——内存密集、梯度不稳定
- **FPO的偏差问题**：FPO通过ELBO估计重要性比率，但存在非对称估计偏差（比率增大时可靠、减小时不可靠），需要更大batch才能稳定
- **DPPO的局限**：DPPO将扩散过程作为内部MDP处理，适合微调但从头训练时性能退化，因为缺乏off-manifold探索能力
- **熵正则化困难**：流式策略的动作log likelihood难以直接计算，传统熵正则化方法不适用

## 关键设计

1. **插值路径重要性比率近似**：不沿ODE流轨迹计算 $\delta_{\varphi_1}(\mathbf{z};\mathbf{s})$，而是沿线性插值路径 $\mathbf{x}_t = (1-t)\mathbf{z} + t\hat{\varphi}_1(\mathbf{z};\mathbf{s})$ 用速度场变化 $\delta_{v_t}$ 近似终端位移差，理论误差为 $\mathcal{O}(\epsilon)$（由PPO裁剪范围自然控制），ODE仅在采样时运行而非训练时
2. **布朗运动熵正则器(Brownian Regularizer)**：利用布朗运动熵单调递增的性质，将速度场对齐到参考流的负score方向 $\eta_t = (1-t)v_t(\mathbf{x}_t;\mathbf{s},\theta) - (\mathbf{x}_t - t\hat{v}_t(\mathbf{x}_t;\mathbf{s}))$，惩罚 $\|\eta_t\|_2^2$ 促进轨迹扩散，完全避免log-likelihood计算
3. **条件流策略架构**：动作 $\mathbf{a} = \varphi_1(\mathbf{z};\mathbf{s}) + \mathbf{n}$（流终端+高斯噪声），诱导出Gaussian mixture形式的策略 $\pi(\mathbf{a}|\mathbf{s}) = \int \mathcal{N}(\mathbf{a};\varphi_1(\mathbf{z};\mathbf{s}),\boldsymbol{\sigma}^2)p_z(\mathbf{z})d\mathbf{z}$，严格比高斯策略表达力更强

## 方法详解

### 连续归一化流策略

定义条件流 $\varphi:[0,1]\times\mathbb{R}^d\times\mathbb{R}^n\to\mathbb{R}^d$，由ODE支配：

$$\frac{d}{dt}\varphi_t(\mathbf{z};\mathbf{s}) = v_t(\varphi_t(\mathbf{z};\mathbf{s});\mathbf{s}), \quad \varphi_0(\mathbf{z};\mathbf{s}) = \mathbf{z}$$

其中 $v$ 为神经网络参数化的时间依赖速度场。动作生成为 $\mathbf{a}=\varphi_1(\mathbf{z};\mathbf{s})+\mathbf{n}$，$\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$，$\mathbf{n}\sim\mathcal{N}(\mathbf{0},\boldsymbol{\sigma}^2)$。添加的高斯噪声既促进探索，又使重要性比率的解析形式可用。

### 重要性比率近似

利用高斯分布似然比的平移不变性，将条件似然比转化为仅依赖流终端位移差 $\delta_{\varphi_1}$ 的形式。关键近似是用插值路径上的速度场差替代终端位移差：

$$\rho \approx \mathbb{E}_{p(t)}\left[\frac{p_n(\mathbf{a}-\hat{\varphi}_1; \delta_{v_t}(\mathbf{x}_t;\mathbf{s}), \boldsymbol{\sigma}^2)}{p_n(\mathbf{a}-\hat{\varphi}_1; \mathbf{0}, \hat{\boldsymbol{\sigma}}^2)}\right]$$

其中 $\delta_{v_t} = v_t(\mathbf{x}_t;\mathbf{s},\theta) - \hat{v}_t(\mathbf{x}_t;\mathbf{s})$。理论证明近似误差为 $\mathcal{O}(\epsilon)$，在PPO裁剪范围控制下可忽略。最终的裁剪代理目标为：

$$J^{\text{Flow}}(\theta,\boldsymbol{\sigma}) = \mathbb{E}\left[\min(\rho \hat{A}, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat{A})\right]$$

### 布朗运动熵正则化

布朗运动粒子自然扩散到均匀分布，熵单调递增。其概率路径遵循热方程 $\partial p_t/\partial t = \nabla^2 p_t$，对应连续性方程中速度场为负score $v_t = -\nabla\log p_t$。利用Liu et al. (2025)的结论，score与速度场的显式关系为：

$$\nabla_{\mathbf{x}}\log\hat{p}_t(\mathbf{x}_t;\mathbf{s}) = \frac{1}{1-t}(t\hat{v}_t(\mathbf{x}_t;\mathbf{s}) - \mathbf{x}_t)$$

正则化损失包含两项：(1) 布朗正则器 $-w_b\|\eta_t\|_2^2$ 促进速度场与负score对齐；(2) 高斯噪声熵 $\frac{w_g}{2}\sum_i\log(2\pi e\sigma_i^2)$ 鼓励随机性。总训练目标为 $J^{\text{Flow}} + J^{\text{Reg}}$。

### 训练流程

每次迭代：(1) 用参考策略采集轨迹（需运行ODE生成 $\hat{\varphi}_1$）；(2) 计算GAE优势估计；(3) 在mini-batch上采样时间 $t\sim U[0,1]$，沿插值路径计算近似重要性比率和布朗正则项；(4) 联合优化策略参数 $\theta$ 和噪声方差 $\boldsymbol{\sigma}$。注意ODE仅在采样阶段运行，训练阶段完全不需要ODE模拟或路径反向传播。

## 实验结果

### 主实验：IsaacLab基准

| 环境 | PPO | PolicyFlow | p-value |
|------|-----|-----------|---------|
| Lift-Cube | $153.1\pm3.0$ | $\mathbf{154.6\pm0.6}$ | 0.32 |
| Navigation | $3.5\pm0.3$ | $\mathbf{4.2\pm0.1}$ | **0.0027** |
| Open-Drawer | $\mathbf{99.8\pm1.7}$ | $99.1\pm0.7$ | 0.41 |
| Quadcopter | $\mathbf{141.8\pm0.5}$ | $141.0\pm0.09$ | 0.099 |
| Anymal-D | $24.5\pm0.1$ | $\mathbf{24.6\pm0.2}$ | 0.26 |
| G1 | $25.4\pm1.2$ | $\mathbf{30.0\pm1.1}$ | **0.00026** |
| H1 | $\mathbf{29.3\pm0.9}$ | $27.3\pm0.2$ | **0.0069** |
| Go2 | $\mathbf{27.9\pm0.3}$ | $27.4\pm0.9$ | 0.33 |

PolicyFlow在多数IsaacLab任务上与PPO持平或更优，在G1人形机器人上显著超越PPO（+18%），在Navigation上也有统计显著优势。

### 计算效率对比

| 环境 | Embedding维度 | PPO (ms) | PolicyFlow (ms) | 增幅 |
|------|--------------|----------|-----------------|------|
| Lift-Cube | 64 | 43.0 | 57.7 | +34% |
| Navigation | 64 | 36.9 | 54.1 | +47% |
| Open-Drawer | 64 | 81.3 | 104.1 | +28% |
| Quadcopter | 64 | 37.8 | 55.6 | +47% |
| Anymal-D | 64 | 41.2 | 57.1 | +39% |
| G1 | 256 | 66.9 | 90.6 | +35% |
| H1 | 512 | 63.4 | 115.5 | +82% |
| Go2 | 512 | 63.9 | 111.5 | +74% |

当模型参数与PPO可比时，每迭代训练时间增加<50%；即使embedding维度增大8倍，计算成本仍不到PPO的2倍。

### 多模态能力(MultiGoal)

在6个等距目标的MultiGoal环境中：PPO（高斯策略）只能覆盖部分目标；FPO/DPPO因缺乏有效熵正则也发生模式坍缩；PolicyFlow+布朗正则器能够最平衡地到达全部6个目标，展现了CNF的多模态表达能力。消融实验显示：仅用均匀噪声注入仍有模式坍缩 → 仅用高斯熵正则部分缓解 → 加入布朗正则器后最优。

### 消融实验

- **裁剪范围 $\epsilon$**：较小 $\epsilon$ 减少近似误差但限制更新步长（学习变慢），$\epsilon=0.2$ 为最佳平衡点
- **网络初始化**：Glorot初始化+输出层置零(GI+ZOL) > 标准Glorot(GI) > 全零初始化(ZI)
- **时间采样**：连续均匀(USC)、离散均匀(USD)、多点离散均匀(Multi-USD)三种策略差异不大，USD作为默认选择因最简单
- **插值路径**：Rectified Flow路径、TrigFlow路径在MultiGoal上优于Stochastic Interpolant路径；在运动控制任务上三者无显著差异

## 优点与创新

- ⭐⭐⭐ **重要性比率近似方案精巧**：利用高斯似然比的平移不变性+插值路径近似，将昂贵的ODE路径积分转化为简单的速度场差前向计算，误差由PPO裁剪范围自然控制
- ⭐⭐⭐ **布朗运动熵正则器**：从物理直觉出发的概念优雅设计——不需计算log-likelihood也不用启发式噪声注入，直接通过speed-score对齐实现隐式熵最大化
- ⭐⭐ **实验覆盖面广**：从简单2D多目标到IsaacLab机器人集（操作/导航/步态/四旋翼），全面验证了方法的通用性
- ⭐⭐ **计算效率分析诚实**：明确报告了训练时间开销（+28%~+82%），不回避高维度时的额外成本

## 不足与展望

- ⭐⭐ **布朗正则器的理论基础有限**：作者自己承认这不是理论严格推导——策略的速度场不是流匹配梯度得到的，与rectified flow动力学不完全对应，更多是启发式设计
- ⭐⭐ **高维度时计算开销显著增加**：H1/Go2环境中embedding=512时训练时间接近PPO的2倍，对大规模实际部署可能成为瓶颈
- ⭐ **缺少与FPO/DPPO在IsaacLab上的直接对比**：由于框架差异（JAX vs PyTorch）未进行直接比较，在这些重要基准上的相对优势无法确认
- ⭐ **仅测试了中等维度动作空间**：未在超高维动作空间（如灵巧手操作）上验证，方法的可扩展性有待进一步检验

## 个人思考

PolicyFlow的核心贡献在于找到了一种**在不牺牲训练效率的前提下使用高表达力流模型做在线RL**的方法路径。重要性比率的近似方案很巧妙——利用了PPO裁剪范围本身就限制了策略更新幅度这一事实，使得一阶近似在实践中足够精确。布朗正则器虽然理论上不完全严格，但从"让速度场指向熵增方向"这个设计意图来看是合理的，且实验效果确实显著。

对后续研究的启示：(1) 这套框架可以直接用于LLM的RLHF——如果将语言模型视为流式策略，PolicyFlow的重要性比率近似可能提供比标准PPO更表达力的策略更新方式；(2) 布朗正则化的思路可以推广到其他需要避免模式坍缩的场景（如多样化文本生成）；(3) 插值路径的选择提示了一个有趣的方向——不同的插值族可能适合不同的任务特性。

<!-- RELATED:START -->

## 相关论文

- [Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [Sequential Monte Carlo for Policy Optimization in Continuous POMDPs](../../NeurIPS2025/reinforcement_learning/sequential_monte_carlo_for_policy_optimization_in_continuous_pomdps.md)
- [Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization](thinking_on_the_fly_test-time_reasoning_enhancement_via_latent_thought_policy_op.md)

<!-- RELATED:END -->
