---
title: >-
  [论文解读] Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning
description: >-
  [ICCV 2025][机器人][运动规划] 提出 IMMP（Interaction-Merged Motion Planning），通过两阶段策略——交互保持预合并（构建多指标检查点池）和交互迁移合并（按交互模块分组的任务向量加权合并）——将来自不同轨迹数据集的智能体行为和交互知识迁移到目标域，有效提升运动规划的跨域适应性。
tags:
  - ICCV 2025
  - 机器人
  - 运动规划
  - 模型合并
  - 域适应
  - 自动驾驶
  - 任务向量
---

# Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning

**会议**: ICCV 2025  
**arXiv**: [2507.04790](https://arxiv.org/abs/2507.04790)  
**代码**: [GitHub](https://github.com/wooseong97/IMMP)  
**领域**: 机器人  
**关键词**: 运动规划, 模型合并, 域适应, 自动驾驶, 任务向量

## 一句话总结

提出 IMMP（Interaction-Merged Motion Planning），通过两阶段策略——交互保持预合并（构建多指标检查点池）和交互迁移合并（按交互模块分组的任务向量加权合并）——将来自不同轨迹数据集的智能体行为和交互知识迁移到目标域，有效提升运动规划的跨域适应性。

## 研究背景与动机

运动规划是自动驾驶机器人的核心组件。虽然存在大量轨迹数据集（ETH-UCY、CrowdNav、THOR、SIT 等），但有效利用它们面临严峻挑战：

**数据集差异巨大**：
- **室内 vs 室外**：室内环境运动受限、速度慢；室外环境动态、速度快
- **HHI vs HRI**：人-人交互（Human-Human Interaction）和人-机交互（Human-Robot Interaction）的动力学模式截然不同
- **数据规模不均**：各数据集大小差异显著

**现有方法的局限**：

**域泛化/域适应**（直接合并多个数据集训练）：
   - **域不平衡问题**：某些数据集主导优化，抑制其他数据集的学习
   - **灾难性遗忘**：新引入的域覆盖已学习的信息

**集成学习**（多个模型推理后合并）：
   - 需要多个完整模型同时运行，计算成本倍增（7× 开销）
   - Ensemble-AVG 效果常劣于 Ensemble-WTA，说明只有部分源域有益

**传统模型合并**（如 Task Arithmetic、Ties Merging）：
   - 未考虑运动规划特有的层次化特征结构
   - 可能破坏轨迹编码和交互建模的特征层次
   - 在运动规划中直接使用效果很差

**核心洞察**：运动规划模型具有层次化的内在结构——自我体编码器、周围智能体编码器、交互编码器、解码器分别编码不同层次的信息。合并时应尊重这一层次结构，按模块分别设定合并权重。

## 方法详解

### 整体框架

IMMP 分为两个阶段：

1. **交互保持预合并（Interaction-Conserving Pre-Merging）**：在各源域上训练规划模型，收集多指标最优检查点和训练过程中间检查点，构建丰富的参数检查点池 $\mathcal{P}$
2. **交互迁移合并（Interaction Transfer with Merging）**：将模型参数按交互相关模块分组，对每组分别提取任务向量并学习合并权重，优化目标域性能

### 关键设计

1. **多指标检查点收集策略**:

    - 功能：为规划模型构建丰富多样的参数检查点池
    - 核心思路：在源域训练过程中，分别記录各评估指标（ADE、FDE、CR、MR）达到最优时的模型参数 $\Theta^{best,m}$，同时每隔 $C$ 个迭代保存中间检查点。这样从每个源域获得多个检查点，它们编码了不同方面的领域知识
    - 设计动机：运动规划不像分类任务有单一准确率指标，而是需要平衡有效性、安全性和目标到达率等多个相互权衡的指标。不同指标最优时的参数反映了不同的行为特征。中间检查点具有更好的泛化性，因为它们未过度拟合源域

2. **交互级模块分组合并**:

    - 功能：将规划模型参数按功能模块分组，各组独立学习合并权重
    - 核心思路：将模型参数 $\Theta$ 分为四组：$\{\theta_{ego}, \theta_{surr}, \theta_{inter}, \theta_{else}\}$，分别对应自我体编码器、周围智能体编码器、交互编码器和其余参数。对每组独立应用任务向量合并 $\theta^* = \theta_0 + \sum_{i=1}^{|\mathcal{P}|} w_{i,\theta} \cdot \tau_i$，其中 $\tau_i = \theta_i - \theta_0$ 为任务向量
    - 设计动机：人的运动模式、机器人轨迹、人-机交互动力学在不同数据集间有不同的分布差异。按模块分组可以让合并过程针对性地从最相关的源域检查点提取不同层次的信息，而不是"一刀切"地共用同一组权重

3. **基于任务向量的参数合并**:

    - 功能：仅通过学习线性权重来组合预训练检查点中的知识
    - 核心思路：给定检查点池 $\mathcal{P} = \{\Theta_1, ..., \Theta_{|\mathcal{P}|}\}$，合并后参数为 $\Theta = \Theta_0 + \lambda\sum_{i=1}^{|\mathcal{P}|} w_i \cdot \tau_i$。优化目标 $\Theta^* = \arg\min_\Theta \sum_i \sum_j \mathcal{L}(\Theta, X_j^{t,i}, Y_j^{t,i})$，只更新权重 $\{w_{i,\theta}\}$
    - 设计动机：直接利用参数检查点而非数据进行适应：（1）无需访问源域数据；（2）有效缓解域不平衡和灾难性遗忘；（3）计算成本等同于单模型（1× cost）

### 损失函数 / 训练策略

- 预合并阶段使用各规划模型原始的多目标损失 $\mathcal{L}_{total}$（包含轨迹偏差损失和碰撞惩罚）
- 合并阶段使用目标域的 $\mathcal{L}_{total}$ 来学习合并权重
- 初始参数 $\Theta_0$ 可以是从头训练的模型或任一微调模型的参数
- IMMP + Finetune：合并后的参数作为初始化，再在目标域上微调

## 实验关键数据

### 主实验

**SIT 目标域（GameTheoretic 规划模型）：**

| 方法 | ADE↓ | Col.Rate↓ | FDE↓ | Miss Rate↓ | Cost |
|------|------|-----------|------|-----------|------|
| Domain Generalization | 0.8338 | 9.87E-04 | 1.8594 | 0.9355 | ×1 |
| Domain Adaptation | 0.4388 | 1.26E-03 | 1.0611 | 0.7201 | ×1 |
| Target Only | 0.4343 | 3.41E-04 | 0.9014 | 0.6272 | ×1 |
| Ensemble-WTA | 0.3695 | 5.75E-05 | 0.8283 | 0.6185 | ×7 |
| Task Arithmetic | 0.4132 | 1.37E-04 | 0.8936 | 0.7364 | ×1 |
| Ties Merging | 1.1876 | 5.53E-04 | 2.2440 | 0.9872 | ×1 |
| **IMMP** | **0.3380** | **5.12E-05** | **0.7626** | 0.6446 | ×1 |
| **IMMP + Finetune** | **0.3157** | **4.28E-05** | **0.7300** | **0.5934** | ×1 |

**THOR 目标域：**

| 方法 | ADE↓ | FDE↓ | Miss Rate↓ |
|------|------|------|-----------|
| Target Only | 0.1003 | 0.2153 | 0.0929 |
| Domain Adaptation | 0.1133 | 0.2516 | 0.1268 |
| **IMMP + Finetune** | **0.0975** | **0.2108** | **0.0912** |

### 消融实验

**合并粒度对比（SIT，GameTheoretic）：**

| 粒度 | ADE↓ | Col.Rate↓ | FDE↓ | Miss Rate↓ | 说明 |
|------|------|-----------|------|-----------|------|
| Model-level | 0.3687 | 7.15E-05 | 0.8365 | 0.8002 | 全模型统一权重 |
| Parameter-level | 0.3798 | 9.16E-05 | 0.7754 | 0.7433 | 逐参数权重 |
| **Interaction-level** | **0.3380** | **5.12E-05** | **0.7626** | **0.6446** | 按模块分组（最优） |

**检查点类型消融：**

| All Metric | Epoch Ckpt | ADE↓ | FDE↓ | Miss Rate↓ |
|-----------|------------|------|------|-----------|
| ✓ | | 0.3646 | 0.8063 | 0.6516 |
| ✓ | ✓ | 0.3543 | 0.7730 | 0.6196 |

### 关键发现

1. **Domain Generalization 效果极差**：直接合并多数据集训练甚至不如 Target Only，凸显了域不平衡问题的严重性
2. **传统合并方法失败**：Averaging、Task Arithmetic、Ties Merging 在运动规划中均表现不佳，Ties Merging 甚至接近随机
3. **IMMP 以单模型成本超越集成**：IMMP（×1 cost）在多数指标上超越了 Ensemble-WTA（×7 cost）
4. **交互级合并显著优于模型级和参数级**：Miss Rate 从 0.80/0.74 降至 0.64，验证了层次化合并的必要性
5. **合并权重与源域相关性一致**：质性分析显示，在目标域表现差的源域被自动分配低权重
6. **跨三种规划模型一致有效**：在 GameTheoretic、DIPP、DTPP 上均带来显著提升

## 亮点与洞察

- **首次将模型合并技术系统化地引入运动规划领域**：揭示了直接应用现有合并方法的失败原因，并提出针对性解决方案
- **交互级合并的思路具有普适性**：按功能模块分组合并的理念可推广到任何具有层次化特征结构的模型
- **"无需访问源数据"的特性**：合并后不再需要源域数据，极大降低了域适应的隐私和存储成本
- **验证了多指标检查点收集的价值**：不同指标最优的检查点编码了不同的行为特征，丰富了合并的基础

## 局限与展望

1. 需要对每个源域分别训练完整模型，预合并阶段的计算成本随源域数量线性增长
2. 合并权重的优化仍需目标域的标注数据，无法实现真正的零样本适应
3. 模块分组（ego/surr/inter/else）依赖于对规划模型内部结构的先验知识，对于端到端的黑箱模型可能难以适用
4. 中间检查点的采样间隔 $C$ 是超参数，最优值可能因模型和数据集而异
5. 仅在 2D 行人/机器人轨迹规划上验证，对于涉及感知的完整自动驾驶管线的效果未知

## 相关工作与启发

- 借鉴 Task Arithmetic 的任务向量概念，但针对运动规划的多指标特性和层次化结构做了关键改进
- 与 UniTraj 等数据集级别的域泛化方法形成互补：IMMP 在参数空间操作，避免了数据混合带来的问题
- 交互级合并思路对多任务学习中的模块化设计（如 MTL 的共享/专用层决策）也提供了启发
- 可结合 LoRA 等参数高效方法进一步降低每个源域的训练成本

## 评分

- 新颖性: ⭐⭐⭐⭐ 将模型合并应用于运动规划是新的尝试，交互级合并是关键创新
- 实验充分度: ⭐⭐⭐⭐⭐ 三种规划模型、多种基线、两个目标域、详尽消融
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，但数学符号稍多
- 价值: ⭐⭐⭐⭐ 为运动规划的跨域适应提供了实用、高效的解决方案

<!-- RELATED:START -->

## 相关论文

- [Evaluating VLMs' Spatial Reasoning Over Robot Motion: A Step Towards Robot Planning with Motion Preferences](../../ICLR2026/robotics/evaluating_vlms_spatial_reasoning_over_robot_motion_a_step_towards_robot_plannin.md)
- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [Experience-based Knowledge Correction for Robust Planning in Minecraft](../../ICLR2026/robotics/experience-based_knowledge_correction_for_robust_planning_in_minecraft.md)
- [Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction](../../CVPR2025/robotics/phoenix_a_motion-based_self-reflection_framework_for_fine-grained_robotic_action.md)
- [MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)

<!-- RELATED:END -->
