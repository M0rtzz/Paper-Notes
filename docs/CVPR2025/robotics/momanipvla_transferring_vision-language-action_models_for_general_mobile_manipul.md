---
title: >-
  [论文解读] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation
description: >-
  [CVPR 2025][机器人][视觉语言动作模型] 提出 MoManipVLA，将预训练的固定基座 VLA 模型迁移到移动操作场景，通过双层轨迹优化联合规划底盘移动和机械臂轨迹（优化可达性/平滑性/碰撞避免），在 OVMM 基准上达到 66.1% 成功率（+4.2%），仅需 50 条演示即可在真实世界部署。
tags:
  - CVPR 2025
  - 机器人
  - 视觉语言动作模型
  - 移动操作
  - 双层轨迹优化
  - VLA迁移
  - 可达性
---

# MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation

**会议**: CVPR 2025  
**arXiv**: [2503.13446](https://arxiv.org/abs/2503.13446)  
**代码**: 有（项目主页）  
**领域**: 机器人 / 移动操作  
**关键词**: 视觉语言动作模型, 移动操作, 双层轨迹优化, VLA迁移, 可达性

## 一句话总结

提出 MoManipVLA，将预训练的固定基座 VLA 模型迁移到移动操作场景，通过双层轨迹优化联合规划底盘移动和机械臂轨迹（优化可达性/平滑性/碰撞避免），在 OVMM 基准上达到 66.1% 成功率（+4.2%），仅需 50 条演示即可在真实世界部署。

## 研究背景与动机

**领域现状**：VLA 模型（如 RT-2、Octo）在固定基座操作上表现出色，但移动操作需要底盘和机械臂的协调规划——底盘移到合适位置后机械臂才能到达目标。

**现有痛点**：直接将固定基座 VLA 的末端执行器轨迹用于移动机器人会失败——因为目标位置可能超出当前底盘位置的臂长范围。需要额外的底盘运动规划。

**核心矛盾**：VLA 输出的是末端执行器 waypoints，不包含底盘运动信息。从头训练移动操作 VLA 数据成本极高。

**切入角度**：保持 VLA 不变，在其输出 waypoints 之上加一层双层优化器：内层用 IK 求解关节角，外层优化底盘位置使可达性最大化、运动最平滑、碰撞最少。

**核心 idea**：VLA 生成 waypoints + 双层轨迹优化（可达性/平滑/碰撞）= 低成本移动操作迁移。

## 方法详解

### 关键设计

1. **双层轨迹优化**:

    - 功能：联合规划底盘移动和机械臂轨迹
    - 核心思路：外层优化底盘位置序列 $\mathbf{x}_b$ 和臂关节角 $\boldsymbol{\theta}$，目标函数 $\mathcal{O} = 10\mathcal{F}_r + 1\mathcal{F}_s + 0.6\mathcal{F}_c$，其中 $\mathcal{F}_r$ 是 IK 可达性（不可达则施加常数大惩罚），$\mathcal{F}_s$ 是平滑性（底盘+关节角的一阶差分），$\mathcal{F}_c$ 是碰撞（与障碍物的签名距离函数）
    - 设计动机：可达性权重最高（10），因为移动操作的首要挑战是"够不够得着"

### 损失函数 / 训练策略

VLA 用少量体现特定（embodiment-specific）数据微调。底盘优化无需学习——直接在推理时求解。真实世界仅需 50 条专家演示。

## 实验关键数据

### 主实验

| 方法 | OVMM 总成功率 | 拾取成功率 |
|------|-------------|-----------|
| SOTA 基线 | 61.9% | 50.2% |
| **MoManipVLA** | **66.1%** | **62.6%** |

### 关键发现
- 可达性是最关键约束：移除后成功率从 66.1% 降到 48.2%
- 仅 50 条演示即可真实世界 40% 成功率
- 推理延迟 693ms（vs 直接优化 742ms）

## 亮点与洞察
- **零数据底盘规划**——不需要移动操作数据，靠优化器在推理时求解
- **VLA 复用的策略**——不重新训练 VLA，只加一层规划优化

## 局限与展望
- 依赖视觉分割掩码质量（无掩码成功率降至 23.7%）
- 轨迹长度限制 <150 步
- 需要体现特定微调

## 评分
- 新颖性: ⭐⭐⭐⭐ VLA+轨迹优化的解耦设计实用
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实世界
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 低成本迁移 VLA 到移动操作

<!-- RELATED:START -->

## 相关论文

- [SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation](../../ICCV2025/robotics/anybimanual_transferring_unimanual_policy_for_general_bimanual_manipulation.md)
- [RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](roboground_robotic_manipulation_with_grounded_vision-language_priors.md)
- [LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers](../../ICCV2025/robotics/vq-vla_improving_vision-language-action_models_via_scaling_vector-quantized_acti.md)

<!-- RELATED:END -->
