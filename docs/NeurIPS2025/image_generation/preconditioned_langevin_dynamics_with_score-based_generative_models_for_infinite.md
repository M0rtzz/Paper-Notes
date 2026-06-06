---
title: >-
  [论文解读] Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems
description: >-
  [NeurIPS 2025][图像生成][预条件Langevin动力学] 在无穷维 Hilbert 空间中严格分析了分数生成模型 (SGM) 驱动的 Langevin 后验采样器，首次推导出依赖分数近似误差的收敛界，并发现了同时依赖前向算子和分数误差的最优预条件器形式，保证所有后验模态的均匀收敛速率。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "预条件Langevin动力学"
  - "分数生成模型"
  - "无穷维贝叶斯反问题"
  - "函数空间"
  - "最优预条件器"
---

# Preconditioned Langevin Dynamics with Score-Based Generative Models for Infinite-Dimensional Linear Bayesian Inverse Problems

**会议**: NeurIPS 2025  
**arXiv**: [2505.18276](https://arxiv.org/abs/2505.18276)  
**代码**: 暂无  
**领域**: 扩散模型 / 图像生成  
**关键词**: 预条件Langevin动力学, 分数生成模型, 无穷维贝叶斯反问题, 函数空间, 最优预条件器

## 一句话总结
在无穷维 Hilbert 空间中严格分析了分数生成模型 (SGM) 驱动的 Langevin 后验采样器，首次推导出依赖分数近似误差的收敛界，并发现了同时依赖前向算子和分数误差的最优预条件器形式，保证所有后验模态的均匀收敛速率。

## 研究背景与动机

贝叶斯反问题广泛出现在 X 射线 CT、地震层析成像、逆热传导等应用中，核心任务是从带噪观测中估计未知参数。当未知量是函数空间中的对象时，问题自然处于无穷维设定下。设计在离散化细化时保持稳定的推断算法至关重要。

现有方法面临一个深层矛盾：

**有限维方法的局限**：大多数基于 SGM 的后验采样方法假设后验支撑在有限维空间，其误差估计随维度增加而发散。"先离散再贝叶斯化"的范式可能导致不稳定性甚至理论上不合理的结果。

**无穷维的特殊挑战**：在无穷维 Hilbert 空间中，Lebesgue 参考测度不存在，密度函数无法定义；先验协方差必须是迹类的；标准 Langevin 采样器的漂移可能在精细尺度上发散。

核心切入点：应采用"先定义算法再离散化"（apply-algorithm-then-discretize）的范式。将 SGM 用作学习到的先验嵌入 Langevin 采样器，需要理解分数近似误差、预条件算子、迹类先验和线性前向映射之间的相互作用。论文图 1 展示了两个关键反面案例：（1）使用恒等先验协方差时采样看似稳定但样本能量无穷——不属于 Hilbert 空间；（2）使用迹类先验时标准 Langevin 漂移在精细尺度发散。

## 方法详解

### 整体框架
考虑线性贝叶斯反问题 $y = AX_0 + n$，其中 $X_0 \sim \mu$ 是 Hilbert 空间 $H$ 上的随机变量，$A:H\to\mathbb{R}^N$ 是线性算子。核心采样器为预条件 Langevin SDE：

$$dX_t = S_\theta(X_t, \tau; \mu) dt + C\nabla_X \log \rho(y - AX_t) dt + \sqrt{2C} dW_t$$

其中 $S_\theta$ 是分数函数的神经网络近似，$C$ 是预条件算子，$W_t$ 是 $H$ 上的 Wiener 过程。

### 关键设计

1. **无穷维分数函数的严格定义**:

    - 在无穷维中密度不存在，采用条件期望表示分数：$S(X,\tau;\mu) = -(1-e^{-\tau})^{-1}(X - e^{-\tau/2}\mathbb{E}[X_0|X_\tau = X])$
    - 通过 Proposition 2.1 将分数改写为 $S(X,\tau;\mu) = -e^{\tau/2}\mathbb{E}[C(C_\mu C_\tau^{-1})^{-1}\nabla\Phi(X_0)|X_\tau=X] - CC_\tau^{-1}X$
    - 设计动机：避免有限维中依赖密度的定义方式

2. **高斯设定下的误差分析 (Theorem 3.1)**:

    - 假设分数近似误差具有分量形式 $\varepsilon_j^a(\tau)X^{(j)} + \varepsilon_j^b(\tau)$
    - 推导了每个模态的 KL 散度界，明确依赖预条件器 $C$ 和分数误差
    - 给出全局收敛有界的充分条件：$\sum_j |\lambda_j^{-1}\varepsilon_j^b(\tau)| < \infty$
    - 关键洞察：误差 $\varepsilon_j^b$ 会产生偏差，被预条件器 $\lambda_j^{-1}$ 放大

3. **最优预条件器 (Theorem 4.1)**:

    - 对于观测模态 $j \leq N$：$\lambda_j^{(0)} = [\mu_j^{-1} + \sigma^{-2}(A_N^\top A_N)_{jj}]^{-1}$
    - 对于未观测模态 $j > N$：$\lambda_j^{(0)} = \mu_j$（即 $C = C_\mu$）
    - 一阶修正纳入分数误差：$\lambda_j^{(1)} = \lambda_j^{(0)3}\mu_j^{-2} - \lambda_j^{(0)2}\mu_j^{-1} - \varepsilon_j^a \lambda_j^{(0)}$
    - 设计动机：通过 Proposition 4.1 分析各模态的均值回复率，选择使所有模态收敛速率一致的 $C$
    - 关键发现：$C$ 不能是恒等算子——必须是迹类的以保证扩散过程在 Hilbert 空间中良定义

4. **非高斯先验的推广 (Section 5)**:

    - 先验绝对连续于高斯参考测度：$d\mu/d\mathcal{N}(0,C_\mu) \propto \exp(-\Phi(X))$
    - Proposition 5.1 将分数分解为高斯部分 + 非线性修正项
    - 稳态分布 (Proposition 5.2) 绝对连续于带近似均值和协方差的高斯测度
    - 最优预条件器需额外考虑势函数凸性常数：$\lambda_j = [\mu_j^{-1} + \sigma^{-2}(A_N^\top A_N)_{jj} + C_{\phi_j}]^{-1}$

### 实验验证策略
使用两个可精确计算的线性反问题来验证理论：布朗薄膜的 KL 展开和热方程逆源问题。

## 实验关键数据

### 主实验：热方程逆源问题重建

| 配置 | 预条件 Langevin | 标准 Langevin | 说明 |
|------|----------------|--------------|------|
| $\tau=10^{-3}$, 低分数误差 | 精确重建 | 精细尺度发散 | 最优 $C$ 吸收误差 |
| $\tau=10^{-1}$, 高分数误差 | 质量下降但稳定 | 严重退化 | 预条件鲁棒性优势明显 |

### 消融实验：离散化不变性（布朗薄膜）

| 观测模态数 $M^2$ | 预条件方法 | 说明 |
|------------------|-----------|------|
| $75^2$ | 稳定重建 | KL 系数精确恢复 |
| $200^2$ | 同样稳定 | 验证离散化不变性 |

### 关键发现
- 最优预条件器使各模态自相关函数快速一致衰减（Figure 4），而标准 Langevin 的模态收敛速率差异极大
- 分数误差 $\varepsilon_j^b$ 引入的偏差无法通过预条件消除，但 $\varepsilon_j^a$ 的影响可通过合理的预条件器控制
- 实用建议：$C$ 应尽可能接近先验协方差 $C_\mu$，再利用后验协方差或分数误差信息做修正

## 亮点与洞察
- 无穷维视角的深刻价值：预条件器 $C$ 不是技巧性修复，而是无穷维设定的内在要求——从前向扩散到 Langevin 采样器必须一致使用迹类算子。这一认识对实践中的离散化策略选择有直接指导作用
- 误差分析的实用性：Theorem 3.1 给出了可验证的充分条件，让实践者判断给定分数近似是否能保证全局收敛
- Proposition 2.1 的条件期望表示巧妙绕开了无穷维中密度不存在的问题，为后续分析奠定了坚实基础

## 局限与展望
- 理论分析依赖线性反问题假设和算子联合对角化假设，非线性反问题的最优预条件器推导仍是重要开放问题
- 分数近似误差的结构假设（Assumption 1, 2）在实际深度学习模型中难以精确验证，需要发展更实用的误差估计方法
- 仅展示了简单的验证性数值实验，缺乏真实高维成像反问题（如医学图像重建）的大规模定量比较
- 离散化方案的具体选择对实际性能的影响未深入讨论，如时间步长与预条件器的交互效应

## 相关工作与启发
- **vs [Song et al. 2023] 有限维分析**：有限维误差估计随维度发散，无法保证离散化细化时的稳定性；本文在无穷维中推导了全局有界条件
- **vs [Pidstrigach et al. 2023]**：同样推导了最优 $C$ 最小化 Wasserstein-2 距离，但本文从均值回复率角度给出更具操作性的解释，且纳入了分数误差的影响
- **vs [Cardoso et al. 2025] 非线性反问题**：非线性设定更复杂，无法识别最优预条件器；本文利用线性设定的显式公式获得更细致的分析
- **vs 传统 MCMC 预条件**：经典的 pCN、DILI 等方法不涉及 SGM 分数近似误差，本文首次揭示了分数误差与预条件器之间的关键交互

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在无穷维中给出依赖分数误差的收敛界和最优预条件器
- 实验充分度: ⭐⭐⭐ 仅有验证性的小规模实验，缺乏大规模应用场景的定量评估
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，但信息密度极高，对非数学背景读者不友好
- 价值: ⭐⭐⭐⭐ 为 SGM 用于反问题采样提供了坚实理论基础，对实践有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)
- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[NeurIPS 2025\] Posterior Sampling by Combining Diffusion Models with Annealed Langevin Dynamics](posterior_sampling_by_combining_diffusion_models_with_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)

</div>

<!-- RELATED:END -->
