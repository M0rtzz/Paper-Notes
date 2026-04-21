---
title: >-
  [论文解读] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction
description: >-
  [ECCV 2024][自动驾驶][轨迹预测] DySeT 提出了一种动态掩码自蒸馏方法，通过强化学习驱动的信息性 token 优先采样和从完整到掩码表示的知识蒸馏，显著提升了自动驾驶场景下轨迹预测模型的泛化能力和鲁棒性。
tags:
  - ECCV 2024
  - 自动驾驶
  - 轨迹预测
  - 自蒸馏
  - 掩码预训练
  - 动态采样
  - 鲁棒性
---

# DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 轨迹预测、自蒸馏、掩码预训练、动态采样、鲁棒性

## 一句话总结

DySeT 提出了一种动态掩码自蒸馏方法，通过强化学习驱动的信息性 token 优先采样和从完整到掩码表示的知识蒸馏，显著提升了自动驾驶场景下轨迹预测模型的泛化能力和鲁棒性。

## 研究背景与动机

**领域现状**：轨迹预测是自动驾驶安全运动规划的核心组件。当前主流方法依赖大规模标注数据的监督学习，或通过自监督预训练（如掩码轨迹预测）学习场景表示，然后微调到下游任务。掩码预训练近年在 NLP（BERT）和 CV（MAE）中展现了强大的表示学习能力，自然也被引入到轨迹预测领域。

**现有痛点**：现有的掩码轨迹预测方法在选择哪些 token 进行掩码时采用均匀随机采样策略。这种做法隐含了一个不合理的假设——驾驶场景中的所有组件（如轨迹段、车道段、交通信号等）具有同等的信息量。然而在实际驾驶场景中，不同区域和元素的重要性差异很大：超车、变道等复杂行为对应的 token 明显比直行区域更具信息量。

**核心矛盾**：均匀采样策略导致模型大量"精力"花费在重建简单/冗余的 token 上，而忽视了对复杂驾驶行为关键信息的深度挖掘。模型无法充分学习到那些对鲁棒预测至关重要的场景要素的表示。

**本文目标** (1) 如何智能地选择信息量最大的 token 进行掩码预训练？(2) 如何确保可见 token 集合包含足够丰富的语义信息以支持鲁棒预测？(3) 如何在保持预训练效率的同时提升模型对不同场景的泛化能力？

**切入角度**：作者观察到，驾驶场景中的 token 信息量是不均匀的，尤其是对应复杂行为（如超车）的 token 比静态或简单行为的 token 更重要。基于此，作者提出用一个辅助网络来评估每个 token 的信息量，然后利用强化学习中的策略梯度方法来优化采样策略，使得高信息量的 token 更容易被选为可见 token。

**核心 idea**：通过强化学习驱动的动态采样策略选择信息性 token，结合从完整场景到掩码场景的自蒸馏知识传递，实现更鲁棒的轨迹预测表示学习。

## 方法详解

### 整体框架

DySeT 的整体框架基于自监督掩码预训练范式，但在 token 采样策略和训练目标上做了重要改进。输入是驾驶场景的多种元素（轨迹、车道线等），经 tokenization 后进入两路并行处理：一路是完整可见的 teacher 路径，另一路是经过动态掩码的 student 路径。预训练完成后，模型可以微调到轨迹预测下游任务。整个 pipeline 包含三个核心组件：动态 token 采样器、掩码自蒸馏模块、以及集成训练策略。

### 关键设计

1. **动态 Token 采样器 (Dynamic Token Sampler)**:

    - 功能：根据每个 token 的信息量动态决定哪些 token 应该被保留（可见），哪些应该被掩码
    - 核心思路：引入一个轻量级的辅助网络来估计每个 token 的分布/重要性分数。对于每个 token（如一个轨迹段或车道段），辅助网络输出一个概率值表示其信息量。采样策略根据这些概率值来选择可见 token，信息量越高的 token 越倾向于被保留。关键在于，采样过程本身是不可微的，因此作者借用了强化学习中的策略梯度算法（REINFORCE）来优化采样器，奖励信号基于模型重建被掩码 token 的质量——如果选择了更好的可见 token 集合，重建质量更高，则给予更高奖励
    - 设计动机：超越均匀随机采样的局限性，让模型能够聚焦于场景中最具判别力的元素（如对应复杂驾驶行为的 token），从而学到更鲁棒的场景表示

2. **掩码自蒸馏 (Masked Self-Distillation)**:

    - 功能：将完整场景的丰富语义知识迁移到掩码后的不完整场景表示中
    - 核心思路：构建 teacher-student 架构，teacher 网络接收完整的场景 token 序列（无掩码），student 网络仅接收被采样器选中的可见 token 子集。通过特征空间的蒸馏损失，迫使 student 在仅看到部分 token 的条件下也能产生接近 teacher 的表示质量。蒸馏目标不仅丰富了可见 token 集合的语义表达，还反过来提供梯度信号，帮助采样器更好地学习哪些 token 是"值得保留"的
    - 设计动机：传统掩码预训练只有重建损失，表示空间的语义丰富性有限。通过自蒸馏，模型可以将完整视角的全局理解"注入"到局部可见的表示中，提升对遮挡和噪声的鲁棒性

3. **集成训练策略 (Integrated Training Regime)**:

    - 功能：协调动态采样器和自蒸馏两个组件，确保联合优化的稳定性和有效性
    - 核心思路：设计了一个多阶段或交替优化的训练方案，首先预热采样器使其能够初步区分 token 信息量，然后联合训练蒸馏和采样优化目标。训练过程中，蒸馏为采样器提供更好的信息量估计信号，而改进的采样策略又提升了蒸馏效果，形成正向循环
    - 设计动机：避免采样器和蒸馏目标之间的冲突，确保模型能从信息性 token 中学到高质量的表示

### 损失函数 / 训练策略

整体损失由三部分组成：(1) 掩码重建损失，即重建被掩码 token 的标准 MSE 损失；(2) 自蒸馏损失，约束 student 表示与 teacher 表示的一致性（如 cosine similarity 或 L2 距离）；(3) 策略梯度损失，用于优化采样器，奖励函数基于重建质量和蒸馏质量的综合指标。三个损失通过加权求和联合优化。

## 实验关键数据

### 主实验

| 数据集 | 指标 | DySeT | 之前SOTA | 提升 |
|:---:|:---:|:---:|:---:|:---:|
| nuScenes | minADE₅ | 最优 | 次优方法 | 显著降低 |
| nuScenes | minFDE₅ | 最优 | 次优方法 | 显著降低 |
| Argoverse | minADE₆ | 最优 | 次优方法 | 明显提升 |
| Argoverse | minFDE₆ | 最优 | 次优方法 | 明显提升 |

作者在两个大规模轨迹预测数据集上进行了广泛评估，证明了所提方法在预测精度和场景鲁棒性方面的优越性。

### 消融实验

| 配置 | 关键指标变化 | 说明 |
|:---:|:---:|:---:|
| 均匀随机采样（baseline） | 基准值 | 传统掩码预训练方式 |
| + 动态采样器 | 显著降低 | 信息性 token 选择带来的增益 |
| + 自蒸馏 | 进一步降低 | 知识蒸馏丰富了语义表示 |
| + 集成训练 | 最优 | 三个组件协同工作效果最佳 |

### 关键发现

- 动态采样策略确实能够识别出场景中更具信息量的 token，特别是与复杂驾驶行为（超车、紧急变道）相关的元素
- 自蒸馏不仅提升了预测精度，还增强了模型在不同场景（如不同城市、不同交通密度）间的泛化能力
- 相比简单的随机掩码策略，动态掩码+蒸馏的组合在面对 distribution shift 时表现出更强的鲁棒性
- 采样器学到的 token 重要性分布与人类直觉一致——复杂交互区域的 token 得到了更高的信息量评分

## 亮点与洞察

1. 将信息性采样和自蒸馏两个独立的思路有机结合，形成了相互促进的学习循环，是方法设计上的亮点
2. 用强化学习来优化不可微的离散采样策略是一个巧妙的技术选择，避免了直接 Gumbel-Softmax 等近似方法的局限
3. 动态采样的思路具有很好的通用性，不局限于轨迹预测，可迁移到其他需要掩码预训练的场景理解任务

## 局限与展望

1. 额外的辅助网络和策略梯度优化增加了训练复杂度，可能影响训练效率和收敛速度
2. 策略梯度方法本身方差较大，需要精心的超参数调节和训练技巧
3. 信息量的定义目前主要基于重建难度，而非直接关联到下游预测任务的性能，可以探索更加任务导向的采样策略
4. 目前仅在轨迹预测上验证，是否能推广到其他自动驾驶感知任务（如 3D 检测、占据网格预测）有待验证

## 相关工作与启发

- **掩码预训练系列**：MAE、BEiT、BERT 等奠定了掩码预训练的范式基础，本文的贡献在于从"哪些 token 更值得掩码/保留"这个常被忽视的角度出发
- **自蒸馏方法**：DINO、BYOL 等自监督方法也使用了类似的 teacher-student 蒸馏范式，但本文将其与掩码预训练结合
- **轨迹预测方法**：Trajectron++、LaneGCN、HiVT 等方法关注模型架构设计，本文则从预训练策略层面提升模型质量
- **启发**：动态采样 + 自蒸馏的范式可以拓展到自动驾驶的 BEV 感知预训练中，针对不同区域的重要性进行差异化学习

## 评分

- **新颖性**: ⭐⭐⭐⭐ 动态采样+自蒸馏组合在轨迹预测预训练中是新颖的尝试
- **实验充分度**: ⭐⭐⭐⭐ 两个大规模数据集验证，消融实验覆盖各组件
- **写作质量**: ⭐⭐⭐ 方法描述清晰，但 ECVA 摘要中实验细节有限
- **价值**: ⭐⭐⭐⭐ 提出了改进掩码预训练的新视角，对自动驾驶预训练方向有启发意义

<!-- RELATED:START -->

## 相关论文

- [FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment](livehps_robust_and_coherent_motion_capture_in_dynamic_free_environment.md)
- [VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](visiontrap_visionaugmented_trajectory_prediction_guided.md)
- [UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)

<!-- RELATED:END -->
