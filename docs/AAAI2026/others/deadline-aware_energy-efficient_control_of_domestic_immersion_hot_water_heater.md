---
title: >-
  [论文解读] Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters
description: >-
  [AAAI 2026][能源效率] 提出一种基于截止时间感知的家用热水器节能控制方法，通过 Gymnasium 仿真环境比较 bang-bang 基线、MCTS 规划器和 PPO 策略，证明 PPO 在相同物理条件下能节省高达 69% 的能量。 家用热水器加热是家庭能源消耗的重要组成部分。现实中，热水需求通常集中在可预测的时…
tags:
  - "AAAI 2026"
  - "能源效率"
  - "强化学习"
  - "热水器控制"
  - "PPO"
  - "MCTS"
---

# Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters

**会议**: AAAI 2026  
**arXiv**: [2601.18123](https://arxiv.org/abs/2601.18123)  
**代码**: 无  
**领域**: 其他  
**关键词**: 能源效率, 强化学习, 热水器控制, PPO, MCTS

## 一句话总结

提出一种基于截止时间感知的家用热水器节能控制方法，通过 Gymnasium 仿真环境比较 bang-bang 基线、MCTS 规划器和 PPO 策略，证明 PPO 在相同物理条件下能节省高达 69% 的能量。

## 研究背景与动机

家用热水器加热是家庭能源消耗的重要组成部分。现实中，热水需求通常集中在可预测的时间段（如上班前或晚间），但大多数控制器仍采用简单的开/关规则（thermostat 控制），忽略了水箱中的水量、散热速度以及何时真正需要热水等因素。这导致了：

**过早或过度加热**：水加热后持续散热浪费能量

**高峰负荷**：未能将负荷转移到低碳时段

**忽视环境因素**：不考虑环境温度、水量等物理参数

本文研究的核心问题是：**在给定截止时间前将水加热到目标温度，同时最小化总能量消耗**。这一"准时到达"的控制框架直接与家庭资源管理和环保目标相关。

关键的insight是：当截止时间充裕或初始温度较高时，具有"预见性"的控制器（如延迟加热策略）可以通过减少不必要的热损失来显著节能，而传统的 bang-bang 控制器由于始终全功率加热再维持温度，浪费了大量能量。

## 方法详解

### 整体框架

本文将家用热水器控制问题建模为有限视野马尔可夫决策过程（MDP），使用基于 Gymnasium API 的轻量级仿真环境进行评估。三种控制器在相同物理条件下进行比较。

### 关键设计

1. **物理环境建模**：采用一阶热力学方程，将水箱模型化为集中热容带牛顿散热的系统。核心方程为：

    $mc_p \frac{dT}{dt} = \eta P(t) - hA[T(t) - T_a]$

   使用前向欧拉离散化，步长 $\Delta t = 120s$。关键参数包括：水量 $m=50$kg，加热功率 6000W，效率 $\eta=0.95$，环境温度 $T_a=20°C$。状态空间为 $o_t = [T_t, T_{target}, T_a, \tau_t]$，动作空间为离散 $\{0, 6000\}$W。

2. **奖励函数设计**：奖励函数由两部分组成——每步能量惩罚和终端温度偏差惩罚：

    $r_t = -\alpha E_t + \begin{cases} -\beta |T_{target} - T_{t+1}|, & \text{if } |T - T^*| \leq \tau \\ 0, & \text{otherwise} \end{cases}$

   其中 $\alpha = 1.86 \times 10^{-8}$，$\beta = 0.03$。设计原则确保在最后一步额外加热的成本低于减少 1°C 终端误差的收益（$\beta \cdot 1°C = 0.03 > \alpha E_{step} \approx 0.0128$），避免模型在整个回合中只节能而忽略温度目标。

3. **三种控制器**：

    - **Bang-bang 基线**：全功率加热至目标温度后维持在目标范围内，时间最优但能效最差
    - **MCTS 规划器**：使用 UCB1（$c=\sqrt{2}$）选择策略，每回合 25000 次模拟。利用已知的确定性动力学模型进行在线前瞻搜索，零样本即可使用
    - **PPO 策略**：使用 Stable-Baselines3 的默认超参数训练 2.5M 步，约 2.1M 步收敛。训练时每回合随机采样不同初始状态以提升泛化性

### 损失函数 / 训练策略

PPO 使用多层感知机架构和离散动作头，在 CPU 上训练。通过在不同初始状态下训练确保策略的泛化能力。MCTS 作为无训练基线，在每一步执行在线搜索。评估指标为相同物理和时间条件下的总能量（Wh）。

## 实验关键数据

### 主实验

实验覆盖三个维度的参数扫描：

| 设置 | PPO 能量 | Bang-bang 能量 | MCTS 能量 | PPO 节能比 |
|------|----------|----------------|-----------|-----------|
| 代表性轨迹(20→60°C, 60步) | 最低 | 最高 | 中等 | 比BB省54%, 比MCTS省33% |
| 30步截止 | ~3230 Wh | ~4370 Wh | ~4180 Wh | ~26% |
| 60步截止 | ~3230 Wh | - | - | - |
| 90步截止 | ~3230 Wh | ~10450 Wh | ~6460 Wh | ~69% |

### 消融实验

| 参数维度 | PPO 表现 | Bang-bang 表现 | MCTS 表现 |
|----------|----------|----------------|-----------|
| 初始温度(10-30°C) | 低敏感、低方差 | 高能耗、线性下降 | 中等、非单调 |
| 截止时间(30-90步) | 近乎恒定~3230Wh | 线性增长4370→10450Wh | 中等4180→6460Wh |
| 目标温度(40-80°C) | 一致最低 | 最陡增长 | 中等 |

### 关键发现

1. **PPO 形成能量下界**：在所有参数设置下，PPO 始终实现最低能耗和最小方差
2. **截止时间是关键差异化因素**：随着可用时间增加，PPO 的节能优势从 26% 放大到 69%
3. **延迟加热策略**：PPO 学到了"延迟加热"策略——只在接近截止时间时才加热，避免了加热后散热的能量浪费
4. **MCTS 零样本有效但有限**：MCTS 无需训练即可提供部分节能，但由于搜索随机性和缺乏学习先验，效果不及 PPO
5. **Bang-bang 在时间充裕时代价最高**：全功率加热后维持温度，随时间延长浪费线性增加

## 亮点与洞察

- **预见性是关键**：能够预见未来需求的控制器（无论是通过学习还是规划）都优于反应式控制器
- **实用权衡清晰**：MCTS 提供无训练的改进但需在线计算，PPO 训练后推理近乎零成本，适合大规模嵌入式部署
- **环境设计简洁有效**：一阶物理模型 + Gymnasium 接口 + 简洁状态空间 = 公平、可复现的基准
- **奖励函数设计精巧**：通过精确计算 α 和 β 的关系，确保策略既节能又不忽略温度目标

## 局限与展望

1. **建模简化**：使用均匀温度假设，忽略水箱分层效应
2. **缺少实际部署验证**：仅在仿真环境中评估，未进行实物实验
3. **未考虑动态因素**：未纳入分时电价、碳排放信号或实际用水模式
4. **动作空间过于简化**：仅二元开/关控制，实际可能有调档
5. **参数扫描范围有限**：每个维度仅 5 个数据点

## 相关工作与启发

本文将建筑控制中的"最优启动"直觉应用于设备级截止时间控制。与日尺度的成本驱动调度不同，本文聚焦于单设备的能量最小化。对比 MPC 和前向搜索方法，将 MCTS 和 PPO 放在同一环境中比较，揭示了在线搜索与零成本推理之间的实际权衡。未来可扩展到连续按需控制、时变电价和更丰富的执行器。

## 评分

- 新颖性: ⭐⭐⭐ (问题有意义但方法较标准)
- 实验充分度: ⭐⭐⭐⭐ (系统性参数扫描，但缺少实物验证)
- 写作质量: ⭐⭐⭐⭐ (清晰透彻，奖励设计解释详尽)
- 价值: ⭐⭐⭐ (实际应用场景明确，但技术贡献有限)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)
- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICML 2025\] Time-Aware World Model for Adaptive Prediction and Control](../../ICML2025/others/time-aware_world_model_for_adaptive_prediction_and_control.md)
- [\[AAAI 2026\] A Fast Heuristic Search Approach for Energy-Optimal Profile Routing for Electric Vehicles](a_fast_heuristic_search_approach_for_energy-optimal_profile_.md)

</div>

<!-- RELATED:END -->
