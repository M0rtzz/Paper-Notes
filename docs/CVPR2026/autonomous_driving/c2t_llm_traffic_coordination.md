---
title: >-
  [论文解读] C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination
description: >-
  [CVPR 2026][自动驾驶][交通信号控制] 提出 C2T 框架，通过将交通状态转换为结构化描述（caption），利用 LLM 进行离线偏好判断并蒸馏为内在奖励函数，替代手工设计的交通信号控制奖励，在 CityFlow 基准的多个真实城市网络上提升效率、安全性和能耗指标。
tags:
  - CVPR 2026
  - 自动驾驶
  - 交通信号控制
  - 多智能体强化学习
  - LLM偏好学习
  - 内在奖励
  - 常识推理
---

# C2T: LLM-Aligned Common-Sense Reward Learning for Traffic-Vehicle Coordination

**会议**: CVPR 2026  
**arXiv**: [2604.13098](https://arxiv.org/abs/2604.13098)  
**代码**: 无  
**领域**: 自动驾驶 / 交通控制  
**关键词**: 交通信号控制, 多智能体强化学习, LLM偏好学习, 内在奖励, 常识推理

## 一句话总结

提出 C2T 框架，通过将交通状态转换为结构化描述（caption），利用 LLM 进行离线偏好判断并蒸馏为内在奖励函数，替代手工设计的交通信号控制奖励，在 CityFlow 基准的多个真实城市网络上提升效率、安全性和能耗指标。

## 研究背景与动机

**领域现状**：基于 MARL 的交通信号控制（TSC）使用手工设计的效率奖励（如队列长度、交叉口压力、平均延迟）优化局部效率。

**现有痛点**：手工奖励是短视的局部代理指标，无法捕获安全、流量稳定性、舒适度等高级人类中心目标。激进清空交叉口可能导致振荡信号、急刹车和不安全间距，"纸面高效但部署脆弱"。

**核心矛盾**：缺乏能反映人类判断并预期长期效果（如车队形成）的"交通质量"概念，且需要在不修改模拟器或在线调用 LLM 的前提下实现。

**本文目标**：将交通质量本身作为监督信号，从 LLM 偏好中离线学习内在奖励，增强标准 MARL 训练管线。

**切入角度**：LLM 在比较结构良好的状态描述时能做出一致的成对判断，可作为"常识"知识源。

**核心 idea**：将交通状态渲染为确定性、单位感知的 caption → LLM 成对偏好判断 → 轻量偏好打分器 → 内在奖励注入标准 PPO。

## 方法详解

### 整体框架

三阶段：(1) Stage 1：使用确定性 schema 将交通观察渲染为结构化 caption；(2) Stage 2：采样 caption 对，查询 LLM 获取偏好标签，训练 Bradley-Terry 偏好打分器作为内在奖励；(3) Stage 3：将内在奖励不对称混合到 TLC 的 PPO 训练中（仅 TLC 接收混合奖励，车辆仅用环境奖励），配合风险掩码和调度策略。

### 关键设计

1. **确定性结构化交通 caption**:

    - 功能：将交通状态转换为 LLM 可一致判断的表示
    - 核心思路：确定性、单位感知的 schema 枚举关键变量（队列、延迟、TTC、违规等），包含明确的语义和数值，消除自由文本描述的歧义性和风格差异
    - 设计动机：LLM 对自由文本的判断不稳定，但对结构化定量描述能给出一致的偏好

2. **离线偏好学习与内在奖励**:

    - 功能：将 LLM 的常识判断蒸馏为可复用的奖励函数
    - 核心思路：从 caption 对中获取 LLM 决定性标签（仅保留明确判断），用 Bradley-Terry 似然训练轻量打分器：$r_\phi(o) = f_\phi(\text{tok}(c), x(o))$。支持可选的安全/效率/能耗头。所有监督离线产生并缓存，切换 prompt 或启用不同头只需更换缓存
    - 设计动机：避免在线 LLM 调用的延迟、可靠性和扩展性问题

3. **安全风险掩码与不对称集成**:

    - 功能：确保安全约束和稳定训练
    - 核心思路：风险掩码在低 TTC 百分位、急刹车集群或红灯违规时抑制内在信号。内在奖励仅混合到 TLC 目标（不给车辆），因为信号相位选择是塑造车队和网络节奏的主要手段。混合权重有调度策略：先满足环境约束再逐渐吸收常识偏好
    - 设计动机：双方都用内在信号会引入额外非稳态；安全约束必须优先于效率优化

### 损失函数 / 训练策略

偏好学习：加权负对数似然 + L2 正则化 + 分数中心化。RL 训练：标准 PPO，每个奖励流独立归一化和软裁剪后加权混合。

## 实验关键数据

### 主实验

| 方法 | 济南旅行时间↓ | 杭州旅行时间↓ | 纽约旅行时间↓ | TTC p10↑ |
|------|------------|------------|------------|---------|
| PressLight | 285.3s | 312.7s | 298.5s | 3.2s |
| CoLight | 278.1s | 305.2s | 291.3s | 3.5s |
| Advanced-CoLight | 272.5s | 298.8s | 285.7s | 3.8s |
| **C2T** | **265.2s** | **289.5s** | **278.1s** | **4.5s** |

### 消融实验

| 配置 | 旅行时间↓ | TTC p10↑ | 说明 |
|------|---------|---------|------|
| 完整 C2T | 265.2s | 4.5s | 内在奖励+安全掩码 |
| 无内在奖励 | 278.1s | 3.5s | 仅环境奖励 |
| 无安全掩码 | 268.5s | 3.8s | 内在奖励但无安全约束 |
| 仅 caption (无数值) | 270.3s | 4.2s | 结构化 caption 已有帮助 |

### 关键发现

- 内在奖励信号贡献最大（去掉后旅行时间增加 13 秒），安全掩码对 TTC 改善关键
- 仅有结构化 caption 也能改善性能，加入匹配数值进一步（但较小）提升
- C2T 的灵活性体现在可通过切换 prompt 产生"效率优先"vs"安全优先"策略

## 亮点与洞察

- 将 LLM 从在线决策者变为离线奖励设计者是一个明智的角色分配：避免了 LLM 在控制回路中的延迟和可靠性问题
- 确定性 schema 的 caption 设计使 LLM 判断可复现、可缓存，是工程上的关键决策
- 不对称集成（仅 TLC 接收内在奖励）的设计减少了多智能体非稳态性

## 局限与展望

- 在 CityFlow 模拟器上验证，真实世界部署效果未知
- LLM 偏好可能包含隐式偏差
- 仅考虑交通信号控制，未扩展到显式建模的自动驾驶车辆
- 可扩展到更多城市网络和极端天气条件

## 相关工作与启发

- **vs LLMLight**: LLMLight 将 LLM 直接放入控制回路，有延迟和可靠性问题；C2T 离线蒸馏无此问题
- **vs CoTV**: CoTV 用手工复合奖励（旅行时间+燃料+排放），C2T 从 LLM 偏好学习更全面的奖励

## 评分

- 新颖性: ⭐⭐⭐⭐ LLM 离线偏好学习用于交通奖励设计是新方向
- 实验充分度: ⭐⭐⭐⭐ 三城市+压力测试+详细消融
- 写作质量: ⭐⭐⭐⭐ 框架清晰
- 价值: ⭐⭐⭐⭐ 对RL奖励设计有通用借鉴意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](../../ICCV2025/autonomous_driving/foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[AAAI 2026\] Generalising Traffic Forecasting to Regions without Traffic Observations](../../AAAI2026/autonomous_driving/generalising_traffic_forecasting_to_regions_without_traffic_observations.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)
- [\[AAAI 2026\] Unlocking Efficient Vehicle Dynamics Modeling via Analytic World Models](../../AAAI2026/autonomous_driving/unlocking_efficient_vehicle_dynamics_modeling_via_analytic_world_models.md)

</div>

<!-- RELATED:END -->
