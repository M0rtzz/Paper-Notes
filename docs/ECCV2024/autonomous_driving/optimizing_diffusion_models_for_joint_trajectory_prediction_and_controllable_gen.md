---
title: >-
  [论文解读] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation
description: >-
  [ECCV 2024][自动驾驶][扩散模型] 本文提出 Optimal Gaussian Diffusion (OGD) 和 Estimated Clean Manifold (ECM) Guidance 两项技术，分别通过优化扩散先验分布和在干净流形上直接注入引导梯度，将联合轨迹预测的扩散步数减少到原来的 1/12，引导采样步数减少到 1/5，同时在 Argoverse 2 上取得更优性能。
tags:
  - ECCV 2024
  - 自动驾驶
  - 扩散模型
  - 轨迹预测
  - 可控生成
  - 计算效率
  - 联合预测
---

# Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation

**会议**: ECCV 2024  
**arXiv**: [2408.00766](https://arxiv.org/abs/2408.00766)  
**代码**: https://yixiaowang7.github.io/OptTrajDiff_Page/ (有项目页)  
**领域**: 自动驾驶  
**关键词**: 扩散模型, 轨迹预测, 可控生成, 计算效率, 联合预测

## 一句话总结

本文提出 Optimal Gaussian Diffusion (OGD) 和 Estimated Clean Manifold (ECM) Guidance 两项技术，分别通过优化扩散先验分布和在干净流形上直接注入引导梯度，将联合轨迹预测的扩散步数减少到原来的 1/12，引导采样步数减少到 1/5，同时在 Argoverse 2 上取得更优性能。

## 研究背景与动机

扩散模型在自动驾驶轨迹预测和可控生成中展现了强大能力——它可以在推理时通过梯度引导来满足额外约束（如避碰、到达指定目标点），而无需重新训练。然而，扩散模型面临两大效率瓶颈：

**痛点一：反向扩散步数多、推理慢。** 传统扩散模型从标准高斯分布开始逐步去噪，标准高斯与数据分布差距巨大，需要大量步数才能获得高质量样本。自动驾驶要求实时推理，车载计算资源有限，高步数严重制约部署。虽然已有快速采样器（DDIM、DPM-Solver）和蒸馏方法，但固定的标准高斯先验是加速的根本瓶颈。

**痛点二：引导采样计算开销大。** 可控生成通过引导代价函数的梯度偏置去噪过程。代价函数通常定义在干净数据上，但引导却作用于中间含噪数据，需要一次前向传播估计干净数据、再通过整个网络反向传播求梯度，计算量和显存消耗极大。

**核心矛盾：** 扩散模型的生成质量和可控性强大，但计算效率与实时性需求之间存在严重矛盾。

**切入角度：** 既然标准高斯先验是效率瓶颈，能否用一个包含数据信息的"最优"高斯先验来减少扩散步数？既然引导梯度需要反向传播整个网络，能否直接在估计的干净流形上注入梯度而跳过反传？

**核心idea：** 利用预训练边际轨迹预测器提供的轨迹统计量（均值和方差），解析地计算最优高斯先验和最优扰动核，实现少步高质量扩散；将引导重新形式化为干净流形上的多目标优化问题，层次化求解以避免网络反传。

## 方法详解

### 整体框架

方法分两个独立但可组合的模块：OGD 负责加速反向扩散过程，ECM(R) 负责加速引导采样。两者共用一个基于 QCNet 场景编码器的扩散模型架构。推理时先用预训练的 QCNet 产生各车辆的边际轨迹样本和似然值，据此计算最优先验和参考轨迹，再进行短步数扩散 + 高效引导。

### 关键设计

1. **Optimal Gaussian Diffusion (OGD)**:

    - 功能：用一个信息丰富的最优高斯分布替代标准高斯作为扩散起点，大幅减少反向扩散步数。
    - 核心思路：根据 KL 散度上界分析，扩散模型学到的分布与真实分布之间的 KL 散度受两项控制：累积得分匹配误差 $\mathcal{G}(\mathbf{x}_\theta, T)$ 和先验与扩散终点分布之间的 KL 散度 $\text{KL}[p_T \| q_\phi]$。前者随 $T$ 增大而增大，后者可通过优化先验参数来最小化。关键结论是最优先验的均值和方差可以用数据分布的统计量解析表达：
    $\boldsymbol{\mu}^* \approx \sqrt{\bar{\alpha}_T}\boldsymbol{\mu}_d, \quad \boldsymbol{\Sigma}^* \approx \bar{\alpha}_T \boldsymbol{\Sigma}_d + (1-\bar{\alpha}_T)^2 \boldsymbol{\Sigma}_p^*$
      其中 $\boldsymbol{\mu}_d, \boldsymbol{\Sigma}_d$ 是数据分布的均值和方差。对联合轨迹预测，假设先验协方差为块对角结构（各车辆独立），则只需各车辆的边际统计量即可。
    - 设计动机：与先前需要训练额外 GAN 来学习先验的方法不同，OGD 的先验可在推理时根据任意 $T$ 解析计算，无需额外训练，且可灵活调节步数。实际中利用预训练 QCNet 预测的多模态边际轨迹集及其似然来估计 $\boldsymbol{\mu}_d$ 和 $\boldsymbol{\Sigma}_d$。

2. **Estimated Clean Manifold (ECM) Guidance**:

    - 功能：在估计的干净数据流形上直接注入引导梯度，避免通过整个扩散网络的反向传播。
    - 核心思路：将可控生成重新表述为干净数据 $\mathbf{x}_0$ 上的多目标优化问题——第一目标（最重要）最大化似然 $-\log q_\theta(\mathbf{x}_0)$，第二目标最小化引导代价 $\mathcal{J}(\mathbf{x}_0)$。采用层次化优化策略迭代 $K$ 次：每次先对当前样本加噪至 $t_k$ 再用扩散模型一步去噪得到高似然估计 $\hat{\mathbf{x}}_0$（优化第一目标），再在 $\hat{\mathbf{x}}_0$ 上做一步梯度下降最小化引导代价（优化第二目标）：
    $\mathbf{x}_0(k-1) \leftarrow \hat{\mathbf{x}}_0(k) - \zeta \nabla_{\hat{\mathbf{x}}_0(k)} \mathcal{J}(\hat{\mathbf{x}}_0(k))$
      由于梯度直接对干净数据求解，完全跳过了网络反传。
    - 设计动机：传统方法对含噪中间数据 $\mathbf{x}_t$ 求引导梯度需要 $\nabla_{\mathbf{x}_t} f_\theta(\mathbf{x}_t)$，即整个网络的雅可比矩阵，计算量和显存极大。ECM 的关键洞察是：引导本质上是在干净流形上寻找同时高似然且低代价的样本，可以直接操作。

3. **Reference Joint Trajectories (ECMR)**:

    - 功能：用边际轨迹组合构造参考联合轨迹来暖启动引导过程，解决多模态分布导致的局部最优问题。
    - 核心思路：联合轨迹分布是多模态的（道路拓扑和决策变量导致多个峰），直接引导容易被困在初始峰附近的局部最优。ECMR 构造候选集 $\mathcal{R} \bigotimes \hat{\mathbf{x}}_0(k)$，评估所有边际轨迹组合的引导代价，选代价最小的作为当前迭代的参考点，然后再做梯度更新。
    - 设计动机：在多峰分布中，从一个峰迁移到另一个峰需经过低似然区域，会导致数值不稳定。通过参考轨迹直接"跳"到更好的峰附近，避免了漫长的跨峰路径。

### 损失函数 / 训练策略

- 扩散模型采用标准 DDPM 噪声预测损失 $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|_2^2$，其中扰动核使用 OGD 推导的最优 $\boldsymbol{\Sigma}_p^*$
- 训练时 $T_{train}=100$，推理时可灵活调节 $T$（例如 40 或 70）
- 使用 DDIM 加速采样，步长为 10
- 轨迹使用 10 维潜在空间表示，映射到 120 维真实轨迹

## 实验关键数据

### 主实验

**联合轨迹预测（Argoverse 2）：**

| 模型 | 扩散步数T | avgMinFDE₆* | avgMinADE₆* | avgMinFDE₁₂₈ | avgMinADE₁₂₈ |
|------|----------|-------------|-------------|--------------|--------------|
| VD₅₀₀ | 500 | 0.61 | 1.36 | 0.48 | 0.91 |
| VD₁₀₀ | 100 | 0.62 | 1.38 | 0.49 | 0.97 |
| **OGD** | **70** | **0.59** | **1.31** | **0.42** | **0.75** |
| OGD | 40 | 0.61 | 1.34 | 0.47 | 0.77 |

OGD 仅用 40 步即超越 VD₅₀₀ 的 500 步，最优 70 步时 avgMinFDE₁₂₈ 降低 12.5%。

**可控生成（U+Deceleration 任务）：**

| 模型 | 采样方法 | DDIM步 | minJFDE↓ | meanJFDE↓ | minJRDE↓ | meanJRDE↓ |
|------|---------|--------|---------|----------|---------|----------|
| VD₅₀₀ | SF | 50 | 0.538 | 2.339 | 0.158 | 0.500 |
| VD₅₀₀ | NNM | 50 | 0.778 | 2.913 | 0.130 | 0.309 |
| **OGD** | **ECMR** | **10** | **0.053** | **0.146** | **0.110** | **0.154** |

ECMR + OGD 在仅用 1/5 步数的情况下，minJFDE 降低了 90%！

### 消融实验

| 配置 | minJFDE | minJRDE | 单步时间(ms) | 显存增量(GB) |
|------|---------|---------|------------|-------------|
| NNM | 0.724 | 0.119 | 113 | 3.21 |
| SF | 0.155 | 0.126 | 247 | 7.96 |
| ECM (Ours) | 0.075 | 0.116 | 111 | 3.21 |
| ECMR (Ours) | 0.053 | 0.110 | 116 | 3.22 |

ECM 的单步推理时间与 NNM 相当（111ms vs 113ms），但效果远优；比 SF 快 2.2 倍且显存仅 40%。

### 关键发现

- OGD 性能随 T 减小反而先升后降，最优点在 T=70，说明减少累积误差的收益超过了先验近似误差
- 参考轨迹（R）对"mean"指标提升显著，验证了解决多模态局部最优的有效性
- OGD 在 Argoverse 2 Multi-world 排行榜上按 avgBrierMinFDE 排名第 4

## 亮点与洞察

1. **解析先验**是本文最优雅的创新——最优高斯先验和最优扰动核都有闭式解，无需额外训练网络，推理时灵活调节步数
2. ECM 将引导重新理解为干净流形上的多目标优化问题，理论动机清晰，实践效果显著
3. 巧妙复用已有的预训练边际预测器（QCNet），不增加训练成本就获得了先验信息和参考轨迹
4. OGD 和 ECM 是正交的两个加速模块，可以独立使用也可以组合

## 局限与展望

- OGD 的性能依赖于边际轨迹预测器的精度；若边际预测不准，优化得到的先验也不准
- 目前假设先验协方差为块对角（车辆间独立），忽略了联合分布中的跨车辆相关性
- 可以探索直接学习联合预测的均值和方差，而非从边际近似
- 引导代价函数目前是手工设计的（目标点距离），更复杂的代价（如安全性评分）的效果待验证

## 相关工作与启发

- 与 DiffScene、TRACE 等使用传统引导采样的工作相比，ECM 通过避免网络反传实现了质的效率提升
- 与 [51] 用 GAN 学先验的方法相比，OGD 的解析先验更简洁、灵活且无需额外训练
- 启发：对于有领域先验知识的扩散任务（如轨迹预测中的边际统计量），利用先验信息初始化扩散过程可能是一个通用的加速策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 解析先验和干净流形引导都有理论创新，但整体仍在扩散模型框架内
- 实验充分度: ⭐⭐⭐⭐ Argoverse 2 大规模验证，排行榜成绩，多指标消融
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，公式严谨，但符号略多
- 价值: ⭐⭐⭐⭐ 效率提升显著，对扩散模型在自动驾驶中的部署有实际意义

<!-- RELATED:START -->

## 相关论文

- [Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-controlla.md)
- [VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](visiontrap_visionaugmented_trajectory_prediction_guided.md)
- [UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)
- [Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](../../ICCV2025/autonomous_driving/generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

<!-- RELATED:END -->
