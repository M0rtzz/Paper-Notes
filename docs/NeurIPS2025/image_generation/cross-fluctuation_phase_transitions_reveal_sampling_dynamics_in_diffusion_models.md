---
title: >-
  [论文解读] Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][扩散模型] 借鉴统计物理中的涨落理论（fluctuation theory），提出了一种通过 **交叉涨落（cross-fluctuation）** 检测扩散模型采样过程中离散相变的框架，从而在无需重新训练的情况下加速采样、改进条件生成、提升零样本分类和风格迁移。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散模型
  - 相变
  - 交叉涨落
  - 采样动力学
  - 条件生成
---

# Cross-fluctuation Phase Transitions Reveal Sampling Dynamics in Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.00124](https://arxiv.org/abs/2511.00124)  
**代码**: 无  
**领域**: image_generation / diffusion_models  
**关键词**: 扩散模型, 相变, 交叉涨落, 采样动力学, 条件生成  
**arXiv**: [2511.00124](https://arxiv.org/abs/2511.00124)  
**代码**: 无  
**领域**: 图像生成  

## 一句话总结

借鉴统计物理中的涨落理论（fluctuation theory），提出了一种通过 **交叉涨落（cross-fluctuation）** 检测扩散模型采样过程中离散相变的框架，从而在无需重新训练的情况下加速采样、改进条件生成、提升零样本分类和风格迁移。

## 研究背景与动机

**领域现状**：扩散模型已成为生成系统的基石，在图像、3D场景、音频、分子结构等领域表现出色，但其采样过程仍然是一个黑箱——每一步混合了数千个值，难以预测。

**现有痛点**：现有方法无法清晰理解采样过程中"成功"与"失败"路径何时分叉，超参数调优（如条件引导的时间窗口）通常依赖昂贵的网格搜索。

**核心矛盾**：扩散模型采样的内部动态缺乏可解释性工具——我们不知道在哪个时间步，不同类别/事件的生成路径变得可区分。

**本文目标**：提供一个系统框架来检测扩散过程中不同"事件"（如不同类别）的统计可区分性转折点（相变），并利用这些转折点直接优化采样。

**切入角度**：将统计物理中的涨落理论引入扩散模型分析，将采样动力学视为从高斯噪声到目标分布的相变过程。

**核心 idea**：不同事件在扩散轨迹中通过 $n$ 阶交叉涨落的不连续点发生离散合并/分裂相变，检测这些相变可直接指导采样策略。

## 方法详解

### 整体框架

定义用户目标为"可期望事件"（desirable event），通过前向扩散跟踪不同事件的统计性质如何收敛到高斯分布。利用 **Algorithm 1** 系统地检测交叉涨落中的离散相变，定位事件"合并"的关键时间步 $i^\star$。

### 关键设计 1：交叉涨落统计量

- **功能**：量化两个事件 $\Omega_1, \Omega_2$ 在扩散过程不同时间步的统计相似性。
- **核心思路**：对状态变量 $\rho$ 定义 $n$ 阶涨落张量 $\mathcal{F}_\rho^{(n)}(\omega) = \bigotimes_{k=1}^n (\rho(\omega) - \mathbb{E}[\rho])$，计算两事件的条件期望张量之间的归一化余弦相似度：
$$\mathcal{M}_\rho^{(n)}(\Omega_1, \Omega_2) = \frac{|\langle \mathbb{E}_1[\mathcal{F}_\rho^{(n)}], \mathbb{E}_2[\mathcal{F}_\rho^{(n)}] \rangle|}{\|\mathbb{E}_1[\mathcal{F}_\rho^{(n)}]\| \cdot \|\mathbb{E}_2[\mathcal{F}_\rho^{(n)}]\|}$$
- **设计动机**：$\mathcal{M} \approx 1$ 意味着事件"合并"（不可区分），$\mathcal{M} \ll 1$ 意味着可区分。对 $n=2$，这等价于两个事件条件协方差矩阵之间的 **Centered Kernel Alignment (CKA)**。

### 关键设计 2：离散相变检测与阈值化

- **功能**：将连续的交叉涨落曲线转化为离散的"已合并/未合并"判定。
- **核心思路**：引入阈值化的修正算子：
$$\widetilde{\mathcal{M}}_\rho^{(n)}(i) = \begin{cases} \mathcal{M}_\rho^{(n)}(\Omega_{1,i}, \Omega_{2,i}), & d(\widehat{F}_\rho^{(2n)}(\Omega_{1,i}), \widehat{F}_\rho^{(2n)}(\Omega_{2,i})) > \varepsilon \\ 1, & \text{otherwise} \end{cases}$$
  其中 $\varepsilon \approx \max_k \lambda_k^{\max}(0) / 400$，使用最大特征值绝对差作为度量。
- **关键时间步**：$i^\star = \min\{i : \widetilde{\mathcal{M}}_\rho^{(n)}(i) = 1\}$，泛化了马尔可夫链的耦合时间概念。

### 关键设计 3：五大应用场景

1. **加速采样**：从 $t = i^\star$ 而非 $t = n$ 启动反向采样（利用 D'Agostino-Pearson 正态性检验确定收敛点）
2. **类条件生成**：利用类别合并时间 $t_{\text{end}}$ 和收敛时间 $t_{\text{start}} = i^\star$ 自动确定 Interval Guidance 的引导窗口
3. **稀有类生成**：结合合并感知引导窗口 + 带噪声参考样本的 ILVR 策略
4. **零样本分类**：用合并时间截断评分积分区间 + 逆SNR加权
5. **零样本风格迁移**：证明源分布与目标风格分布的涨落轨迹在 $O(\delta)$ 精度内一致（Fourier正则性条件下），直接复用源分布的合并时间

### 损失函数

本文无需训练新模型，是一个纯分析框架。前向Monte-Carlo扫描即可无偏估计所有交叉涨落项。

## 实验关键数据

### 主实验：加速采样

| Model / Dataset | FID (↓) | Steps (↓) | GFLOPs (↓) |
|---|---|---|---|
| DiT-XL/2 (ImageNet, full) | 3.42±0.21 | 250 | 4100 |
| DiT-XL/2 (ImageNet, **ours**) | **3.37±0.31** | 175 | 2870 |
| DDPM (MNIST, full) | 2.27±0.19 | 1000 | 2000 |
| DDPM (MNIST, **ours**) | 2.29±0.17 | 600 | 1200 |
| DDPM (CIFAR-10, full) | 3.62±0.35 | 500 | 6000 |
| DDPM (CIFAR-10, **ours**) | **3.47±0.34** | 300 | 3600 |

**关键发现**：在保持（甚至略微提升）FID的同时，减少 30%-40% 的采样步数。

### 主实验：类条件生成（IG方法）

| Model | FID (↓) | Precision (↑) | Recall (↑) | Density (↑) | Coverage (↑) |
|---|---|---|---|---|---|
| DiT-XL/2 (ImageNet, IG Baseline) | 3.22±0.16 | 0.78 | 0.23 | 0.83 | 0.35 |
| DiT-XL/2 (ImageNet, **IG Ours**) | **2.86±0.15** | **0.83** | **0.26** | **0.85** | **0.39** |
| DDPM (CIFAR10, IG Baseline) | 3.32±0.25 | 0.77 | 0.19 | 0.81 | 0.32 |
| DDPM (CIFAR10, **IG Ours**) | **3.01±0.14** | **0.79** | **0.22** | **0.84** | **0.35** |

### 主实验：零样本分类

| Method | ImageNet (↑) | CIFAR-10 (↑) | Oxford Pets (↑) |
|---|---|---|---|
| SD, uniform (Li et al.) | 54.96 | 84.67 | 82.87 |
| SD, trunc. inverse-SNR (**Ours**) | **65.28** | **88.38** | **89.15** |
| CLIP RN-50 | 58.41 | 75.42 | 85.61 |

### 消融实验

- **Merger cascade 可视化**：不同类别在不同时间步以树状结构合并，形成"合并级联"
- 截断逆SNR优于均匀加权和纯逆SNR，验证合并时间前的时间步最有判别力
- 风格迁移中的涨落适应引理：Fourier域距离约束下四阶矩差异 $\leq C_n \delta$

### 关键发现

- 涨落驱动的合并时间从主流类别泛化到长尾类别，无需额外调参
- 仅一次前向Monte-Carlo扫描即可获得所有必要的诊断信息
- 该视角统一了经典有限马尔可夫链的耦合/混合结果与连续SDE动力学

## 亮点与洞察

1. **理论优雅**：将统计物理涨落理论与扩散模型采样动力学完美桥接，通过CKA建立直观的实际连接
2. **一框架多用**：同一个相变检测算法服务于加速采样、条件生成、稀有类、分类、风格迁移五个任务
3. **零成本改进**：所有改进无需重新训练模型，只需一次前向传播分析
4. **可解释性强**：merger cascade 提供了扩散过程中结构如何形成的直观可视化

## 局限与展望

1. **VP调度假设**：当前分析限于 variance-preserving SDE，尚未扩展到 EDM 等非VP调度
2. **各向同性限制**：假设前向SDE为各向同性噪声，各向异性扩散尚未处理
3. **高阶涨落代价**：对向量状态的高阶涨落计算复杂度高，实际主要使用 $n=2$ 的CKA
4. **阈值选择**：$\varepsilon$ 的选取虽有启发式规则，但缺乏自适应机制
5. **跨模态扩展**：尚未验证在音频、3D几何等其他生成模态上的效果

## 相关工作与启发

- **与 Interval Guidance (Kynkäänniemi et al., 2024) 的关系**：IG需要昂贵的网格搜索来确定引导区间，本文直接通过涨落分析提供 $t_{\text{start}}$ 和 $t_{\text{end}}$
- **与布朗运动平衡时间的联系**：经验发现平衡时间定理能很好预测 $i^\star$，但理论解释留给未来工作
- **对采样加速的启发**：不需要学习新的调度策略，仅通过检测前向过程中的统计相变即可确定何时停止

## 评分

⭐⭐⭐⭐ (4/5)

理论深度与实用性兼具的优秀工作。统计物理视角新颖，多任务验证充分，且不增加训练成本。主要不足是VP调度假设限制了适用范围，高阶涨落的实际可操作性有限。

<!-- RELATED:START -->

## 相关论文

- [PID-controlled Langevin Dynamics for Faster Sampling of Generative Models](pid-controlled_langevin_dynamics_for_faster_sampling_of_generative_models.md)
- [Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [Elucidated Rolling Diffusion Models for Probabilistic Forecasting of Complex Dynamics](elucidated_rolling_diffusion_models_for_probabilistic_forecasting_of_complex_dyn.md)
- [Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [Breaking AR's Sampling Bottleneck: Provable Acceleration via Diffusion Language Models](breaking_ars_sampling_bottleneck_provable_acceleration_via_d.md)

<!-- RELATED:END -->
