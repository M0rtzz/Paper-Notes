---
title: >-
  [论文解读] BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly
description: >-
  [ICML 2025][机器人][双臂协作] 提出 BiAssemble 框架，通过学习感知双臂协作的点级可供性（affordance），将几何装配任务分解为抓取→对齐→装配三步，在破碎物体重组任务上超越现有可供性和模仿学习方法，并在真实世界基准上验证。 领域现状：形状装配分为家具装配（功能性部件组合）和几何装配（碎片重组）…
tags:
  - "ICML 2025"
  - "机器人"
  - "双臂协作"
  - "几何装配"
  - "点级可供性"
  - "碎片重组"
  - "长程规划"
---

# BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly

**会议**: ICML 2025  
**arXiv**: [2506.06221](https://arxiv.org/abs/2506.06221)  
**代码**: [https://sites.google.com/view/biassembly/](https://sites.google.com/view/biassembly/)  
**领域**: 机器人  
**关键词**: 双臂协作, 几何装配, 点级可供性, 碎片重组, 长程规划

## 一句话总结
提出 BiAssemble 框架，通过学习感知双臂协作的点级可供性（affordance），将几何装配任务分解为抓取→对齐→装配三步，在破碎物体重组任务上超越现有可供性和模仿学习方法，并在真实世界基准上验证。

## 研究背景与动机

**领域现状**：形状装配分为家具装配（功能性部件组合）和几何装配（碎片重组），后者应用广泛（文物修复、骨骼拼接等）但研究不足。

**现有痛点**：(a) 现有方法仅预测目标位姿，忽略实际操作过程中的碰撞; (b) 碎片几何任意、无语义定义，抓取和操作极其困难; (c) 长程动作序列中双臂协调和接触丰富的装配过程非常复杂。

**核心矛盾**：观察空间（任意几何碎片）和动作空间（长程双臂协调）都极大。

**本文目标**：如何让双臂机器人学会协作装配任意形状的破碎碎片？

**切入角度**：用点级可供性实现几何泛化，将长程任务分解为三个子步骤以降低复杂度。

**核心 idea**：模仿人类直觉——抓取→对齐（留间隙）→逐渐推合，每步用可供性感知后续步骤的约束。

## 方法详解

### 整体框架
三步流程：
1. **Pick-up**：学习点级可供性选择抓取点（考虑抓取可行性+后续装配兼容性）
2. **Alignment**：将碎片移到对齐位姿（通过反向拆解找到无碰撞对齐位姿）
3. **Assembly**：预测无碰撞方向逐渐推合碎片

### 关键设计

1. **协作感知的点级可供性**:

    - 功能：对碎片表面每个点预测抓取分数，同时考虑局部几何和后续操作
    - 核心思路：可供性 = 抓取可行性 × 对齐可达性 × 装配方向兼容性
    - 设计动机：不能只选几何上可抓的点——还要确保抓了之后能完成后续的对齐和装配

2. **无碰撞对齐位姿生成**:

    - 功能：从装配好的状态反向拆解，找到对齐位姿
    - 核心思路：沿装配方向的反方向分离碎片，留出安全间隙
    - 设计动机：直接放到目标位姿必然碰撞，先对齐再推合避免碰撞

3. **真实世界可复现基准**:

    - 功能：创建全球可获取的标准化碎片基准
    - 核心思路：使用标准物体（如特定品牌马克杯）+ 标准化打碎方式，提供 3D 网格
    - 设计动机：破碎碎片几何各异难以公平评估，标准化基准解决此问题

### 损失函数 / 训练策略
- 可供性模型在仿真中训练，迁移到真实世界
- 仿真环境支持双臂碎片装配

## 实验关键数据

### 主实验

| 方法 | 装配成功率 | 抓取成功率 |
|------|----------|----------|
| 模仿学习 | ~25% | ~60% |
| 单步可供性 | ~40% | ~75% |
| **BiAssemble** | **~65%** | **~85%** |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| 无协作感知 | ~40% | 抓取点不考虑后续可行性 |
| 无对齐步骤（直接装配） | ~20% | 大量碰撞 |
| 完整 BiAssemble | **~65%** | 三步分解最优 |

### 关键发现
- 三步分解将复杂长程任务简化为可学习的子任务
- 协作感知的可供性比纯几何可供性提升 ~25%
- 真实世界打碎马克杯装配成功，仿真到真实迁移可行

## 亮点与洞察
- **任务分解模仿人类直觉**——人类也是先拿起、对齐、再推合，这个分解很自然
- 可供性方法在几何泛化上的优势再次得到验证
- 真实世界可复现基准对该领域有长期价值

## 局限与展望
- 仅处理两片碎片的装配，多片扩展是开放问题
- 对碎片的 3D 感知依赖点云质量
- 装配方向预测仍较简单，复杂几何可能需要更精细的规划

## 相关工作与启发
- **vs 纯位姿预测方法**: 忽略操作过程，不可执行
- **vs 家具装配**: 碎片无语义标签，更具挑战性

## 评分
- 新颖性: ⭐⭐⭐⭐ 将可供性扩展到双臂几何装配是新颖应用
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实世界，多品类
- 写作质量: ⭐⭐⭐⭐ 图示清晰，任务分解合理
- 价值: ⭐⭐⭐⭐ 推进了机器人碎片装配的实际可行性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration](../../CVPR2026/robotics/bipremanip_learning_affordance-based_bimanual_preparatory_manipulation_through_a.md)
- [\[ICML 2025\] Geometric Contact Flows: Contactomorphisms for Dynamics and Control](geometric_contact_flows_contactomorphisms_for_dynamics_and_control.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](../../ICCV2025/robotics/selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[CVPR 2025\] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](../../CVPR2025/robotics/maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)
- [\[ICML 2025\] Learning to Stop: Deep Learning for Mean Field Optimal Stopping](learning_to_stop_deep_learning_for_mean_field_optimal_stopping.md)

</div>

<!-- RELATED:END -->
