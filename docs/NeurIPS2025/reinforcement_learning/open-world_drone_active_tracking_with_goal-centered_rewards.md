---
title: >-
  [论文解读] Open-World Drone Active Tracking with Goal-Centered Rewards
description: >-
  [NeurIPS 2025][强化学习][Drone Active Tracking] 提出首个开放世界无人机主动跟踪基准 DAT（24 个城市级场景、高保真动力学仿真），以及基于目标中心奖励函数和课程学习的强化学习跟踪方法 GC-VAT，在仿真器上达到约 72% 的跟踪成功率。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "Drone Active Tracking"
  - "Goal-Centered Reward"
  - "Curriculum Learning"
  - "Open-World Benchmark"
---

# Open-World Drone Active Tracking with Goal-Centered Rewards

**会议**: NeurIPS 2025  
**arXiv**: [2412.00744](https://arxiv.org/abs/2412.00744)  
**代码**: [DAT_Benchmark](https://github.com/SHWplus/DAT_Benchmark)  
**领域**: 视频理解 / 无人机跟踪  
**关键词**: Drone Active Tracking, Reinforcement Learning, Goal-Centered Reward, Curriculum Learning, Open-World Benchmark

## 一句话总结

提出首个开放世界无人机主动跟踪基准 DAT（24 个城市级场景、高保真动力学仿真），以及基于目标中心奖励函数和课程学习的强化学习跟踪方法 GC-VAT，在仿真器上达到约 72% 的跟踪成功率。

## 研究背景与动机

**领域现状**: 视觉主动跟踪（VAT）旨在通过控制运动系统自主跟随目标，广泛应用于无人机监控和安防。基于强化学习的 VAT 方法将视觉跟踪和控制集成在统一框架中，避免了流水线方法的手动标注和额外调优。

**现有痛点**:
   - **缺乏统一基准**：现有场景复杂度低、忽略跟踪器动力学或使用过度简化模型，不足以验证算法性能。先前方法使用基于规则的目标管理，远非类人行为。
   - **开放世界的复杂干扰**：大规模动态环境中存在频繁遮挡和干扰。之前方法只能从固定水平视角捕获图像，限制了感知和运动范围。

**核心矛盾**: 真实无人机跟踪需要在复杂开放世界中运作，但现有仿真环境和奖励设计均无法满足此需求——距离度量的奖励在俯视视角下会产生误导。

**本文目标**: (1) 构建一个真实、全面的无人机主动跟踪基准；(2) 设计适用于非固定视角的奖励函数和高效训练策略。

**切入角度**: 从射影几何出发，设计基于偏差度量（deviation metric）的奖励函数来替代欧氏距离奖励，并结合课程学习。

**核心 idea**: 用图像中心投影的归一化偏差度量替代欧氏距离，使奖励在任意视角下都能正确反映目标位置。

## 方法详解

### 整体框架

将无人机主动跟踪建模为马尔可夫决策过程 $\langle \mathcal{S}, \mathcal{A}, \mathcal{R}, \gamma, \mathcal{T} \rangle$：
- **状态**: $84 \times 84$ RGB 图像
- **动作**: 离散动作集（前进、后退、左移、右移、左转、右转、停止）
- **网络**: CNN + GRU 结构的 Drone Agent
- **算法**: PPO（Proximal Policy Optimization）

### 关键设计

#### 1. DAT 基准

- **24 个城市级场景**: 6 种户外场景 $\times$ 4 种天气（白天、雾天、夜晚、雪天）
- **数字孪生工具**: 从 OpenStreetMap 任意区域自动生成 3D 场景
- **高保真无人机动力学**: 基于 Webots 模拟 DJI Matrice 100 的质量、惯性、气动特性和云台响应
- **类人目标行为**: 集成 SUMO 交通仿真器，管理 24 种跟踪目标（汽车、摩托、行人、轮式/腿式机器人）的行为
- **7 维场景复杂度**: 场景面积、建筑密度、颜色丰富度、道路密度、地形密度、树木密度、隧道密度

#### 2. Goal-Centered Reward（GC-VAT 核心）

**问题**: 无人机俯视时，图像平面与地面不平行，投影为梯形。目标与图像中心的欧氏距离无法准确反映在图像平面上的实际空间关系。

**偏差度量**:

$$\phi(P_g, C_g) = \frac{|P_g - C_g|}{|E_g(P_g, C_g) - C_g|}$$

其中 $P_g$ 为目标点，$C_g$ 为图像中心投影，$E_g$ 为连线与投影边界的交点。该度量将所有处于相同相对位置的点映射到相同值。

**奖励函数**:

$$r_{gc}(P_g) = \begin{cases} \tanh(\alpha(1-\phi(P_g, C_g))^3), & P_g \in \mathcal{I}_{clip} \\ 0, & \text{otherwise} \end{cases}$$

其中 $\alpha = 4$，$\lambda_{clip} = 0.7$。$\tanh$ 在图像中心提供强指示，截断范围防止目标停留在边缘。

**理论保证（Proposition 1）**: 当相机不处于固定水平前视时，基于欧氏距离的奖励可能给离中心更近的目标更低的奖励值，导致训练失败。

#### 3. Curriculum-Based Training (CBT)

分两阶段训练：
- **第 1 阶段**: 简化环境（直线目标轨迹 + 无障碍物），学习基本的目标居中能力
- **第 2 阶段**: 复杂环境（多样目标运动 + 障碍物/遮挡），增强泛化能力
- **切换条件**: 当平均奖励达到阈值 $\eta$ 时自动切换

#### 4. 域随机化

随机化无人机初始位置和朝向（相对于目标）以及云台俯仰角，促进多样化行为探索。

### 损失函数/训练策略

使用 PPO 算法，奖励为 $r_{gc}$（Eq. 3），结合域随机化和课程学习策略。

## 实验关键数据

### 主实验：场景内性能

| 方法 | 平均 CR | 平均 TSR |
|------|---------|----------|
| AOT | ~44 | ~0.22 |
| D-VAT | ~35 | ~0.19 |
| **GC-VAT（Ours）** | **~242** | **~0.72** |

相比 D-VAT，CR 提升 **591%**，TSR 提升 **279%**。

### 跨场景/跨域泛化

| 测试条件 | GC-VAT CR | GC-VAT TSR | 对比 D-VAT TSR 提升 |
|----------|-----------|------------|---------------------|
| 跨场景 | 176 (avg) | 0.57 | +200% |
| 跨域（夜晚） | 217 | 0.64 | — |
| 跨域（雾天） | 243 | 0.76 | — |
| 跨域（雪天） | 178 | 0.60 | — |
| 跨域平均 | 213 | 0.67 | +253% |

### 消融实验

| 配置 | 场景内 TSR | 跨场景 TSR | 跨域 TSR |
|------|----------|----------|---------|
| 使用 D-VAT 奖励 | 0.06 | 0.05 | 0.06 |
| 无 CBT | 0.23 | 0.26 | 0.23 |
| 无角度随机化 (AR) | 0.44 | 0.37 | 0.36 |
| 无高度随机化 (HR) | 0.49 | 0.48 | 0.57 |
| 无垂直随机化 (VR) | 0.63 | 0.54 | 0.60 |
| 无俯仰随机化 (PR) | 0.61 | 0.48 | 0.52 |
| **完整 GC-VAT** | **0.68** | **0.54** | **0.65** |

### 关键发现

1. **D-VAT/AOT 的奖励函数在俯视视角下完全失效**：训练曲线快速下降或持平，验证了理论分析
2. **角度随机化 (AR) 贡献最大**：去除后 TSR 大幅下降，表明多角度初始化对策略探索至关重要
3. **鲁棒性**：风扰动下 TSR 仅下降 $< 0.06$，雨滴模糊下 TSR 仅下降 $< 0.07$
4. **干扰物和新目标**：添加相似车辆干扰后 TSR = 0.91（vs 0.94），未见过的巴士目标 TSR = 0.79

## 亮点与洞察

1. **理论与实践结合**：不仅设计了新奖励函数，还给出了距离度量失效的数学证明，令人信服
2. **全面的基准贡献**：24 个场景、数字孪生工具、类人行为模拟，基准本身的价值可能超过方法
3. **偏差度量的简洁性**：通过归一化到投影边界消除了视角变化的影响，设计直观优雅
4. **课程学习的显著效果**：无 CBT 时策略完全无法学习（TSR 仅 0.23），说明开放世界场景的训练难度

## 局限与展望

1. **离散动作空间**：连续动作空间可能带来更精细的控制
2. **仅使用 RGB 图像**：未利用深度等多模态信息
3. **单目标跟踪**：未考虑多目标场景
4. **奖励设计依赖几何先验**：假设地面平坦，复杂地形下可能需要调整
5. **端到端方法的局限**：在目标完全遮挡时缺乏显式的目标重检测机制

## 相关工作与启发

- **AD-VAT+/D-VAT/AOT**: 主要对比方法，均使用距离奖励且场景简单
- **PPO**: 作为基础 RL 算法，在控制任务中的稳定性被充分利用
- **课程学习**: 启发自渐进式训练思路，在复杂 RL 环境中特别有效
- **对视觉跟踪的启示**: 在 sim-to-real 框架中，合理的奖励设计比复杂的网络更重要

## 评分

⭐⭐⭐⭐

系统性的工作，基准和方法均有贡献。奖励设计有理论支撑且验证充分，但跟踪方法本身（CNN+GRU+PPO）较为标准。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The World Is Bigger! A Computationally-Embedded Perspective on the Big World Hypothesis](the_world_is_bigger_a_computationally-embedded_perspective_on_the_big_world_hypo.md)
- [\[NeurIPS 2025\] Learning from Demonstrations via Capability-Aware Goal Sampling](learning_from_demonstrations_via_capability-aware_goal_sampling.md)
- [\[NeurIPS 2025\] ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition](aline_joint_amortization_for_bayesian_inference_and_active_data_acquisition.md)
- [\[NeurIPS 2025\] Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning](open_vision_reasoner_transferring_linguistic_cognitive_behavior_for_visual_reaso.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
