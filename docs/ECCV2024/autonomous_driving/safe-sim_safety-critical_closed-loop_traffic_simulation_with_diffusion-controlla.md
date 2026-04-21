---
title: >-
  [论文解读] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries
description: >-
  [ECCV 2024][自动驾驶][安全关键场景生成] Safe-Sim 提出了一种基于扩散模型的闭环安全关键仿真框架，通过在去噪过程中注入对抗性引导目标和部分扩散（Partial Diffusion）机制，生成真实且可控的对抗场景来评估自动驾驶规划算法，支持控制碰撞类型、相对速度和 TTC 等关键参数。
tags:
  - ECCV 2024
  - 自动驾驶
  - 安全关键场景生成
  - 闭环仿真
  - 扩散模型
  - 对抗行为控制
  - 自动驾驶测试
---

# Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries

**会议**: ECCV 2024  
**arXiv**: [2401.00391](https://arxiv.org/abs/2401.00391)  
**代码**: https://safe-sim.github.io/ (项目主页)  
**领域**: 自动驾驶  
**关键词**: 安全关键场景生成, 闭环仿真, 扩散模型, 对抗行为控制, 自动驾驶测试

## 一句话总结
Safe-Sim 提出了一种基于扩散模型的闭环安全关键仿真框架，通过在去噪过程中注入对抗性引导目标和部分扩散（Partial Diffusion）机制，生成真实且可控的对抗场景来评估自动驾驶规划算法，支持控制碰撞类型、相对速度和 TTC 等关键参数。

## 研究背景与动机

**领域现状**: 自动驾驶系统需要在安全关键的长尾场景中进行测试，但真实道路上这类事件极少发生，且在公共道路上测试高风险场景不安全，因此仿真是不可或缺的。

**现有痛点**: 传统方法主要存在三个问题——(a) 手动设计场景可扩展性差；(b) 已有自动生成方法多关注静态场景生成而非动态闭环仿真，其他智能体不会对 planner 的行为做出反应；(c) 已有方法缺乏可控性，每个场景通常只能产生单一的对抗结果。

**核心矛盾**: 安全关键场景生成需要同时满足三个互相制约的目标：(a) 真实性——场景行为需符合真实世界分布；(b) 安全关键性——能制造碰撞等危险场景；(c) 可控性——能控制碰撞类型、攻击性程度等。已有工作（CTG, CTG++, STRIVE, DiffScene）无法同时满足这三个目标。

**本文目标**: 构建一个同时具备安全关键场景生成、闭环交互、可控对抗行为的仿真框架。

**切入角度**: 利用扩散模型的可控性优势，通过 test-time guidance 引导去噪过程实现对抗行为，并提出 Partial Diffusion 控制碰撞类型。

**核心 idea**: 在扩散模型的去噪过程中注入对抗损失梯度来引导对抗智能体轨迹，同时通过部分扩散将轨迹提案注入扩散过程以控制碰撞类型。

## 方法详解

### 整体框架

Safe-Sim 将仿真场景中的 $N$ 个智能体分为三类角色：(1) ego 车辆，由 planner $\pi$ 控制；(2) 对抗智能体 $a$，在反应式模型 $g$ 的基础上增加对抗项以挑战 planner；(3) 非对抗反应式智能体，由 $g$ 生成正常驾驶行为以维持场景真实性。在每个时间步，所有智能体基于当前上下文 $\mathbf{c}_t^i$（包含 agent-centric 地图和历史轨迹）持续生成和更新轨迹，以 2Hz 频率重新规划，形成闭环交互。

### 关键设计

1. **轨迹扩散模型**: 模型在真实驾驶数据上训练，学习轨迹分布 $q(\tau_0)$。轨迹 $\tau$ 包含动作序列 $\tau_a = [a_0, ..., a_{T-1}]$ 和状态序列 $\tau_s = [s_1, ..., s_T]$。模型预测动作序列，状态通过独轮车动力学模型 $f$ 递推得到。前向过程逐步加噪：

$$q(\tau_k | \tau_{k-1}) = \mathcal{N}(\tau_k; \sqrt{1-\beta_k}\tau_{k-1}, \beta_k \mathbf{I})$$

反向过程学习去噪：$p_\theta(\tau_{k-1}|\tau_k, \mathbf{c}) = \mathcal{N}(\tau_{k-1}; \mu_\theta(\tau_k, k, \mathbf{c}), \Sigma_k)$。使用 K=100 步扩散，ResNet 编码场景上下文，UNet-like 的 1D 时序卷积处理轨迹。

2. **对抗引导目标 (Guided Diffusion)**: 在去噪过程中，通过 reconstruction guidance（clean guidance）对预测的清洁轨迹施加梯度扰动：

$$\tilde{\tau}_0 = \hat{\tau}_0 - \alpha \Sigma_k \nabla_{\tau_k} J(\hat{\tau}_0)$$

总损失函数由对抗项和正则化项组成：

$$J(\tau) = \rho \underbrace{(J_{\text{coll}} + J_v + J_{\text{ttc}})}_{J_{\text{adv}}(\tau)} + \underbrace{J_{\text{route}} + J_{\text{Gauss}}}_{J_{\text{reg}}(\tau)}$$

其中 $\rho$ 决定智能体是否表现出对抗行为。具体包括：
    - **碰撞损失** $J_{\text{coll}} = -\sum_{t=1}^T d(t)$：最小化对抗智能体与 ego 的距离
    - **相对速度控制** $J_v$：控制碰撞时的相对速度
    - **TTC 代价** $J_{\text{ttc}}$：通过高斯核控制碰撞时间，偏好高相对速度和难以避让的碰撞角度
    - **路线引导** $J_{\text{route}}$：惩罚偏离车道中心线超过阈值的情况，比先前的 off-road loss 更有效
    - **高斯碰撞引导** $J_{\text{Gauss}}$：考虑切向和法向距离的椭圆形碰撞惩罚，比圆盘近似更精确地减少非对抗智能体间的碰撞

3. **部分扩散 (Partial Diffusion)**: 解决碰撞类型可控性的核心创新。通过将轨迹提案（trajectory proposals）注入扩散过程的中间步骤来控制碰撞类型：

    - 第一步：基于领域知识（如碰撞类型规则）生成初始轨迹提案 $\tau_0$
    - 第二步：设置部分扩散比率 $\gamma$，确定注入点 $k_p = \gamma \cdot K$
    - 第三步：从 $k_p$ 步开始加噪：$\hat{\tau}_{k_p} = \sqrt{\bar{\alpha}_{k_p}}\tau_0 + \sqrt{1-\bar{\alpha}_{k_p}}\epsilon$
    - 第四步：从 $k_p$ 步开始做引导去噪，将提案修正为既符合目标碰撞类型又保持真实性的轨迹

   轨迹提案生成方法：找到 ego 和对抗智能体车道中心线的交叉点，选择加速度和横向偏移来触发期望的碰撞类型。$\gamma$ 越大，用户控制越强；$\gamma$ 越小，模型自身分布影响越大。

### 损失函数 / 训练策略

- 扩散模型在 nuScenes 训练集上预训练，学习真实轨迹分布
- 推理时通过 test-time guidance 注入对抗目标，无需重新训练
- 每次生成多条候选轨迹，选择引导代价最低的轨迹（filtering）
- 闭环仿真以 2Hz 频率更新所有智能体规划

## 实验关键数据

### 主实验

| 数据集 | 方法 | 碰撞率(%)↑ | 其他Offroad(%)↓ | 对抗Offroad(%)↓ | 真实性↓ |
|--------|------|-----------|----------------|----------------|---------|
| nuScenes | **Safe-Sim** | **43.2** | **1.8** | 11.4 | **0.38** |
| nuScenes | STRIVE | 36.4 | 2.2 | 11.4 | 0.85 |
| nuScenes | DiffScene | 18.2 | 11.4 | 9.0 | 0.52 |
| nuPlan | **Safe-Sim** | **80.0** | **9.4** | 11.7 | **0.27** |
| nuPlan | DiffScene | 56.7 | 14.0 | 5.0 | 0.42 |

### 不同 Planner 评估

| Planner | Ego-Adv碰撞(%)↑ | Ego-Other碰撞(%)↑ | 对抗Offroad(%)↓ | 真实性↓ |
|---------|-----------------|-------------------|----------------|---------|
| BC | 38.8 | 37.3 | 9.0 | 0.79 |
| IDM | **49.3** | **58.2** | 3.0 | 0.78 |
| Lane-Graph | 34.3 | 37.3 | 1.5 | **0.57** |
| BITS | 16.4 | 19.4 | 6.0 | 0.79 |
| PDM-Closed | 26.9 | 50.7 | 1.5 | 0.86 |

### 消融实验

| 配置 ($J_{\text{adv}}$ / $J_{\text{reg}}$ / Partial Diff) | 碰撞率(%)↑ | 碰撞角度方差↑ | 碰撞点方差↑ | 说明 |
|------|-----------|-------------|-----------|------|
| ✓ / ✓ / ✗ | 23.9 | 2.22 | 1.62 | 基线：引导但无部分扩散 |
| ✓ / ✓ / ✓ | 29.0 | 3.10 | **5.44** | 部分扩散显著提升碰撞点多样性 |
| ✓ / ✗ / ✗ | **53.5** | 3.34 | 2.47 | 无正则化碰撞率最高但不真实 |

### 关键发现

- Safe-Sim 在所有指标上超越 STRIVE，碰撞率提高 6.8%，真实性提升 55%
- 部分扩散机制使碰撞点多样性提升 3.36 倍（方差从 1.62 到 5.44）
- IDM planner 因只关注同车道车辆而最容易被对抗攻击
- 正则化项虽降低碰撞率但显著提升场景真实性（offroad 率大幅下降）
- 在 nuScenes 训练的模型可直接在 nuPlan 上测试，具有跨数据集泛化能力

## 亮点与洞察

- **统一框架**: 首个同时实现安全关键性 + 闭环交互 + 可控对抗行为的仿真框架
- **部分扩散创新**: 优雅地通过 $\gamma$ 参数平衡用户控制与模型数据分布之间的权衡
- **高斯碰撞距离**: 相比圆盘近似，考虑切向/法向距离的椭圆形碰撞模型更精确
- **路线引导**: 比 off-road loss 更强的语义约束，明确指定每个智能体应走的路线
- **实用价值**: 可用于系统性评估不同 planner 的安全性能，揭示各 planner 的弱点

## 局限与展望

- 对抗智能体有时会在到达 ego 之前与非对抗智能体发生不合理碰撞
- 部分碰撞场景中 ego planner 并非过错方，需要更多"ego 有责"场景
- 轨迹提案基于规则生成，可探索学习型提案生成方法
- 未探索将框架用于闭环策略训练（而非仅评估）
- 可发展自动化的可控参数调节方法，自动发现多样化长尾场景

## 相关工作与启发

- **CTG / CTG++**: 可控交通生成但不支持安全关键场景和闭环
- **STRIVE**: 在 CVAE 潜空间做对抗优化生成安全关键场景，但缺乏碰撞类型控制
- **DiffScene**: 扩散模型做安全关键场景但非闭环
- **TRACE**: 引导扩散做可控行人动画，Safe-Sim 将类似思路扩展到车辆对抗仿真
- **启发**: 扩散模型的 test-time guidance 是一种强大的范式，可在不修改模型的情况下注入多种目标

## 评分

- 新颖性: ⭐⭐⭐⭐ 部分扩散和多目标引导组合的设计很新颖，首次实现碰撞类型可控
- 实验充分度: ⭐⭐⭐⭐ 在两个数据集、五种 planner 上验证，消融实验充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述系统，图表辅助理解效果好
- 价值: ⭐⭐⭐⭐⭐ 对自动驾驶安全评估有重要实际价值，框架可直接用于系统性测试

<!-- RELATED:START -->

## 相关论文

- [Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving](neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)
- [RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion](roofdiffusion_constructing_roofs_from_severely_corrupted_point_data_via_diffusio.md)
- [MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection](monowad_weather-adaptive_diffusion_model_for_robust_monocular_3d_object_detectio.md)
- [OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)

<!-- RELATED:END -->
