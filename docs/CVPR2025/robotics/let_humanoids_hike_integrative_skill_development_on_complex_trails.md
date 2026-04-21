---
title: >-
  [论文解读] Let Humanoids Hike! Integrative Skill Development on Complex Trails
description: >-
  [CVPR 2025][机器人][人形机器人] 提出 LEGO-H 框架，通过 TC-ViT（时序条件 ViT）统一导航感知和低层运动控制，结合层次潜空间匹配（HLM）从 oracle 策略高效蒸馏，使 Unitree H1 人形机器人在复杂户外山径上达到 68.4% 成功率。
tags:
  - CVPR 2025
  - 机器人
  - 人形机器人
  - 户外导航
  - 运动技能
  - 层次潜空间匹配
  - 时序ViT
---

# Let Humanoids Hike! Integrative Skill Development on Complex Trails

**会议**: CVPR 2025  
**arXiv**: [2505.06218](https://arxiv.org/abs/2505.06218)  
**代码**: https://LEGO-H-HumanoidRobotHiking.github.io (有)  
**领域**: 机器人 / 人形机器人  
**关键词**: 人形机器人, 户外导航, 运动技能, 层次潜空间匹配, 时序ViT

## 一句话总结

提出 LEGO-H 框架，通过 TC-ViT（时序条件 ViT）统一导航感知和低层运动控制，结合层次潜空间匹配（HLM）从 oracle 策略高效蒸馏，使 Unitree H1 人形机器人在复杂户外山径上达到 68.4% 成功率。

## 研究背景与动机

**领域现状**：四足机器人在户外地形导航上取得了显著进展（如 ANYmal、Go2），但人形机器人因更高的重心、更多自由度和更复杂的平衡需求，户外行走仍是巨大挑战。

**现有痛点**：现有人形机器人导航方法将感知→规划→控制分成独立模块，模块间接口损失信息。直接端到端训练又因动作空间巨大而难以收敛。将四足方法（如 RMA）简单迁移到人形也效果不佳（42.97% 成功率）。

**核心矛盾**：高层导航（去哪里）和低层运动（怎么走）需要统一决策——在崎岖地形上，"绕过障碍"和"调整步态"是同一个动作的不同层面。

**切入角度**：两阶段训练——先用特权信息训练 oracle 策略获取多样运动技能，再用 HLM（层次VAE 潜空间匹配）蒸馏到无特权的统一策略。

**核心 idea**：TC-ViT 统一导航+控制 + HLM 从 oracle 蒸馏 = 端到端人形户外行走。

## 方法详解

### 关键设计

1. **TC-ViT（时序条件 Vision Transformer）**:

    - 功能：从深度图中提取导航相关特征并融合目标信息
    - 核心思路：将深度图 patch 化后与目标方向 token 一起送入 ViT，在注意力层中早期融合目标信息，输出导航感知编码。关键是在 positional embedding 前就注入目标信息
    - 设计动机：ConvGRU 基线只有 42.97% 成功率，TC-ViT 的多尺度全局注意力更适合复杂地形感知

2. **层次潜空间匹配（HLM）**:

    - 功能：从 oracle 策略蒸馏到无特权策略时保留动作结构
    - 核心思路：训练掩码 VAE 编码 oracle 的动作序列，在潜空间中用余弦相似度 + triplet loss 对齐学生和教师的动作分布。比直接 L2 匹配动作更好——因为 L2 忽略了关节间的协调关系
    - 设计动机：消融显示 HLM 将碰撞率从 10.40% 降到 7.84%，稳定性（跌倒前时间）从 7.00s 提升到 7.46s

### 损失函数 / 训练策略

Oracle 用 PPO + RL 奖励（方向跟踪/躯干高度/跌倒惩罚），蒸馏用 $\mathcal{L}_{im} + \mathcal{L}_{hie}$，其中 $\mathcal{L}_{hie}$ 包含 VAE 潜空间余弦相似度和 triplet loss。在 Isaac Gym 中 512 个机器人并行训练，5 个难度级别的山径场景。

## 实验关键数据

### 主实验

| 方法 | 成功率 | 碰撞率 | 跌倒前时间 |
|------|--------|--------|-----------|
| Vanilla ConvGRU | 42.97% | - | 5.36s |
| w/o HLM (TC-ViT only) | 64.73% | 10.40% | 7.00s |
| **LEGO-H (完整)** | **68.40%** | **7.84%** | **7.46s** |
| Oracle (上界) | 71.20% | - | - |

### 关键发现
- **TC-ViT vs ConvGRU**：成功率从 42.97% 到 64.73%，证明 ViT 的全局注意力对复杂地形关键
- **HLM 提升安全性**：碰撞率从 10.40% 降到 7.84%，动作结构正则化有效
- **接近 oracle 上界**：68.4% vs 71.2%，蒸馏损失仅 3 个点

## 亮点与洞察
- **首个人形机器人户外山径导航**——将人形机器人的能力边界从平地/楼梯扩展到崎岖山径
- **端到端统一设计**——不分离导航和运动控制，让它们在同一网络中联合优化

## 局限与展望
- 上半身固定（仅控制下肢），限制了平衡恢复能力
- 仅仿真验证，未部署到真实机器人
- 仅用深度传感器，缺少 RGB/语义信息
- 对比基线是从四足方法改编的，非专用人形方法

## 评分
- 新颖性: ⭐⭐⭐⭐ TC-ViT + HLM 的组合在人形导航中有效
- 实验充分度: ⭐⭐⭐⭐ 5 难度级别，512 并行机器人，多基线对比
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰
- 价值: ⭐⭐⭐⭐ 推动了人形机器人在野外的能力边界

<!-- RELATED:START -->

## 相关论文

- [Enginuity: Building an Open Multi-Domain Dataset of Complex Engineering Diagrams](../../NeurIPS2025/robotics/enginuity_building_an_open_multi-domain_dataset_of_complex_engineering_diagrams.md)
- [Visual Planning: Let's Think Only with Images](../../ICLR2026/robotics/visual_planning_lets_think_only_with_images.md)
- [iManip: Skill-Incremental Learning for Robotic Manipulation](../../ICCV2025/robotics/imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](../../AAAI2026/robotics/shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)
- [AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots](../../CVPR2026/robotics/atomicvla_unlocking_the_potential_of_atomic_skill.md)

<!-- RELATED:END -->
