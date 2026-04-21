---
title: >-
  [论文解读] Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][强化学习微调] 通过闭环强化学习微调改善监督学习训练的交通智能体行为模型，解决开环训练的分布偏移问题，在Waymo仿真基准上取得SOTA。
tags:
  - ECCV 2024
  - 自动驾驶
  - 强化学习微调
  - 智能体行为建模
  - 自动驾驶仿真
  - 分布偏移
  - Waymo
---

# Improving Agent Behaviors with RL Fine-tuning for Autonomous Driving

**会议**: ECCV 2024  
**arXiv**: [2409.18343](https://arxiv.org/abs/2409.18343)  
**代码**: 无  
**领域**: Autonomous Driving  
**关键词**: 强化学习微调, 智能体行为建模, 自动驾驶仿真, 分布偏移, Waymo

## 一句话总结

通过闭环强化学习微调改善监督学习训练的交通智能体行为模型，解决开环训练的分布偏移问题，在Waymo仿真基准上取得SOTA。

## 研究背景与动机

交通智能体行为建模是自动驾驶研究的核心问题之一，其应用场景包括：(1) 构建逼真可靠的仿真环境用于离线评估（off-board evaluation）；(2) 预测交通参与者的运动轨迹用于在线规划（onboard planning）。这些应用场景对智能体行为的真实性和多样性提出了极高要求。

当前主流方法采用监督学习（行为克隆/imitation learning）从专家数据中学习行为策略。然而，监督学习方法面临一个根本性问题——**分布偏移（distribution shift）**。在训练时，模型学习从专家状态到专家动作的映射；但在测试时，由于模型自身预测的微小误差会累积，导致模型遇到训练中从未见过的状态，性能急剧下降。

这个问题在长时间序列的仿真中尤为严重：一个小的轨迹偏差会不断累积，最终导致不真实的行为（如碰撞、驶出道路等）。现有的解决方案包括数据增强、DAgger等方法，但效果有限。

本文的核心思路是：在监督学习预训练之后，使用闭环强化学习（RL）进行微调。RL的优势在于它天然在闭环环境中优化，模型必须面对自己过去决策产生的状态，从而直接缓解分布偏移问题。

## 方法详解

### 整体框架

方法采用两阶段训练策略：(1) 第一阶段使用监督学习在离线数据上预训练行为模型，学习基本的驾驶行为；(2) 第二阶段在仿真环境中使用RL对模型进行闭环微调，优化特定的行为指标（如碰撞率、偏离道路率等）。

### 关键设计

1. **闭环RL微调框架**:
    - 功能：解决监督学习预训练模型的分布偏移问题
    - 核心思路：在仿真环境中让智能体与环境交互，根据交互结果更新策略。关键在于设计合适的奖励函数，既要鼓励真实的行为，又要惩罚不安全的行为（碰撞、违规等）
    - 设计动机：监督学习只能在开环中训练（不考虑自身预测误差的累积），而RL在闭环中优化，天然适合解决分布偏移

2. **多目标奖励函数**:
    - 功能：平衡仿真真实度与安全性
    - 核心思路：奖励函数综合考虑多个指标——与真实轨迹的相似度（确保真实性）、碰撞惩罚（确保安全性）、道路遵循奖励（确保合规性）、交互合理性奖励（确保社会性）。通过加权组合实现多目标优化
    - 设计动机：单一的真实性目标可能导致模型学会"安全但不真实"或"真实但不安全"的行为，多目标奖励实现更好的权衡

3. **策略评估基准（Policy Evaluation Benchmark）**:
    - 功能：直接评估仿真智能体对自动驾驶规划器质量的识别能力
    - 核心思路：设计一系列具有不同质量等级的规划器，使用仿真智能体模型来评估这些规划器。一个好的仿真智能体应该能正确区分好的和差的规划器——即好的规划器在仿真中应该表现更好
    - 设计动机：现有benchmark只评估智能体行为本身的真实性，而忽略了仿真的根本目的——评估和改进自动驾驶系统

### 损失函数 / 训练策略

- 第一阶段：使用标准的行为克隆损失（MSE/NLL）在Waymo Open Motion Dataset上预训练
- 第二阶段：使用PPO算法进行RL微调，奖励函数包含碰撞惩罚、偏离道路惩罚、与真实轨迹的距离奖励等
- 训练技巧：RL微调时使用较小的学习率以避免遗忘预训练知识，同时使用KL惩罚限制策略更新幅度

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| Waymo WOSAC | Realism Meta-metric | SOTA | 监督学习基线 | 显著提升 |
| Waymo WOSAC | Collision Rate ↓ | 大幅降低 | 监督学习基线 | -30-50% |
| Waymo WOSAC | Off-road Rate ↓ | 显著降低 | 监督学习基线 | -20-40% |
| Policy Eval | Planner Ranking | 正确 | 部分方法排序错误 | 更准确 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅监督学习 | 基线 | 存在分布偏移 |
| 仅RL（从零开始） | 较差 | 缺乏先验知识，训练不稳定 |
| SL预训练 + RL微调 | 最优 | 两阶段互补 |
| 不同奖励权重 | 性能敏感 | 需要仔细调节奖励函数的权重 |

### 关键发现

- RL微调显著改善了碰撞率和偏离道路率等安全指标，同时保持了行为真实性
- 从零开始的RL训练效果远不如SL预训练+RL微调，证明了预训练的重要性
- 提出的Policy Evaluation Benchmark为评估仿真质量提供了新的视角
- 方法在Waymo Open Sim Agents Challenge (WOSAC)上取得了领先性能

## 亮点与洞察

- 将NLP领域"预训练+RL微调"（如RLHF）的范式引入自动驾驶智能体建模
- 提出了Policy Evaluation Benchmark这一新颖的评估维度，关注仿真的根本目的
- 方法简洁有效，两阶段训练策略易于实现
- 对分布偏移问题提供了直接的解决方案

## 局限与展望

- RL微调需要大量的环境交互，计算成本较高
- 奖励函数的设计需要领域知识，不同场景可能需要不同的奖励权重
- 仅在Waymo数据集上验证，对其他城市和驾驶场景的泛化性未知
- 可以探索离线RL方法减少对在线交互的需求
- 多智能体协同的RL微调是一个有价值的研究方向

## 相关工作与启发

- **WOSAC**: Waymo Open Sim Agents Challenge为智能体建模提供了标准评测平台
- **TrafficSim / SimNet**: 早期的交通仿真工作，使用监督学习训练智能体
- **RLHF**: NLP领域的RL微调范式，本文将类似思想应用于自动驾驶
- 启发："预训练+RL对齐"可能是行为建模的通用范式，适用于机器人等其他领域

## 评分

- 新颖性: ⭐⭐⭐ RL微调的思路并不新颖，但在自动驾驶仿真中的应用有意义
- 实验充分度: ⭐⭐⭐⭐ Waymo标准基准上的全面评估，加上新提出的评估维度
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述简洁
- 价值: ⭐⭐⭐⭐ 30次引用，对自动驾驶仿真领域有实际价值

<!-- RELATED:START -->

## 相关论文

- [WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](../../AAAI2026/autonomous_driving/worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)
- [SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)
- [AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](../../NeurIPS2025/autonomous_driving/autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving](reason2drive_towards_interpretable_and_chainbased_reasoning.md)

<!-- RELATED:END -->
