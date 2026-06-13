---
title: >-
  [论文解读] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning
description: >-
  [CVPR 2025][机器人][双手灵巧操作] 提出 ManipTrans，两阶段残差学习框架将人手动捕数据迁移到灵巧机器手的双手操作：Stage-1 在纯手轨迹上预训练模仿模型（手腕+手指跟踪+平滑奖励），Stage-2 通过残差模块+课程学习加入物体交互约束（物体跟踪+接触力），在 OakInk-V2 上物体旋转误差仅 8.60°、双手成功率 39.5%。
tags:
  - "CVPR 2025"
  - "机器人"
  - "双手灵巧操作"
  - "残差学习"
  - "动捕迁移"
  - "课程学习"
  - "接触力"
---

# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning

**会议**: CVPR 2025  
**arXiv**: [2503.21860](https://arxiv.org/abs/2503.21860)  
**代码**: 待公开  
**领域**: 强化学习  
**关键词**: 双手灵巧操作, 残差学习, 动捕迁移, 课程学习, 接触力

## 一句话总结

提出 ManipTrans，两阶段残差学习框架将人手动捕数据迁移到灵巧机器手的双手操作：Stage-1 在纯手轨迹上预训练模仿模型（手腕+手指跟踪+平滑奖励），Stage-2 通过残差模块+课程学习加入物体交互约束（物体跟踪+接触力），在 OakInk-V2 上物体旋转误差仅 8.60°、双手成功率 39.5%。

## 研究背景与动机

### 领域现状

**领域现状**：灵巧操作（如抓取/旋转物体）是机器人的核心挑战。动捕数据提供了丰富的人手操作演示，但人手有 27 个自由度、灵巧手有不同的运动学结构——直接重定向（retarget）无法保证物理上的有效交互。

**现有痛点**：（1）QuasiSim 等物理仿真方法需要 40+ 小时优化一个轨迹；（2）直接 RL 训练灵巧手操作需要任务特定奖励设计，且双手版本维度爆炸；（3）重定向+RL 残差的简单组合因动作空间复杂而收敛困难。

**核心矛盾**：轨迹模仿（只看手的运动不看物体）容易但无法保证物体交互成功；交互学习（物体跟踪+接触）难但是最终目标。两者一起学维度太高。

**切入角度**：解耦——先学手的运动模式（Stage-1，无物体），再用残差网络仅学习"因为物体交互需要的修正量"（Stage-2）。残差模块的动作空间小得多。

**核心 idea**：手运动预训练 + 物体交互残差 + 课程学习 = 高效的双手灵巧操作迁移。

### 解决思路

**本文目标**：### 关键设计

1. **Stage-1: 纯手轨迹模仿**：用 RL 训练策略模仿动捕的手腕位姿和手指关节角，不涉及物体。


## 方法详解

### 关键设计

1. **Stage-1: 纯手轨迹模仿**：用 RL 训练策略模仿动捕的手腕位姿和手指关节角，不涉及物体。奖励包含手腕跟踪+手指跟踪+平滑项

2. **Stage-2: 残差交互学习**：冻结 Stage-1 策略，训练残差模块添加修正动作。新增奖励：物体位姿跟踪+接触力奖励$r_{contact} = w_c \exp(-\lambda_c / \sum_f C_f \cdot \mathbb{1}(D < \xi_c))$+接触终止条件。课程学习逐步收紧手指和物体跟踪的容差

3. **DexManipNet 数据集**：3.3K episodes, 1.34M frames, 1.2K 物体, 61 种任务（含新的双手任务）

### 损失函数 / 训练策略

手模仿奖励：$r_\mathcal{I} = w_{wrist}r_{wrist} + w_{finger}r_{finger} + w_{smooth}r_{smooth}$。手指奖励用高斯衰减。训练约 15 分钟/新轨迹（vs QuasiSim 40+小时）。

## 实验关键数据

| 方法 | 物体旋转误差↓ | 物体平移误差↓ | 双手成功率↑ |
|------|------------|------------|-----------|
| Retarget+Residual | 11.58° | 0.79cm | 13.9% |
| RL-only | 9.72° | 1.23cm | — |
| **ManipTrans** | **8.60°** | **0.49cm** | **39.5%** |

### 消融实验
- 接触力作为观测输入：加速收敛
- 接触力奖励：对接触密集任务成功率关键
- 课程学习（逐步收紧容差）：防止网络崩溃
- 重力松弛+高摩擦初始化：早期训练的必要条件

### 关键发现
- 残差学习比端到端 RL 高效 ~160×（15分钟 vs 40小时）
- 双手操作的成功率 39.5%（vs 13.9%）——残差解耦大幅降低了双手协调的学习难度
- 接触终止条件确保稳定抓取

## 亮点与洞察
- **两阶段解耦的核心洞察**——手的运动模式和物体交互修正是两个不同层次的学习目标
- **15分钟 vs 40小时**——效率提升两个数量级

## 局限与展望
- 部分动捕序列因噪声太大无法迁移
- 仿真12-DoF手 vs 真实6-DoF手需要额外的指尖适配
- 仅限操作任务，不适用于移动

## 评分
- 新颖性: ⭐⭐⭐⭐ 残差学习在灵巧操作中的高效应用
- 实验充分度: ⭐⭐⭐⭐⭐ 新数据集+定量+真实机器人+双手
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为动捕到灵巧操作的迁移提供了高效方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment](../../AAAI2026/robotics/dexterous_manipulation_transfer_via_progressive_kinematic-dynamic_alignment.md)
- [\[CVPR 2025\] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[CVPR 2025\] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness](dexgrasp_anything_towards_universal_robotic_dexterous_grasping_with_physics_awar.md)
- [\[ICCV 2025\] AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation](../../ICCV2025/robotics/anybimanual_transferring_unimanual_policy_for_general_bimanual_manipulation.md)

</div>

<!-- RELATED:END -->
