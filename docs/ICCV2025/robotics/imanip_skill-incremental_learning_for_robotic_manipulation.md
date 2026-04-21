---
title: >-
  [论文解读] iManip: Skill-Incremental Learning for Robotic Manipulation
description: >-
  [ICCV 2025][机器人][增量学习] 提出 iManip 框架，通过时序回放策略和可扩展 PerceiverIO 架构，使机器人能够在不重新训练的情况下持续学习新的操作技能，同时缓解对已学技能的灾难性遗忘，在 RLBench 上比传统增量基线平均提升 9.4%。
tags:
  - ICCV 2025
  - 机器人
  - 增量学习
  - 机器人操作
  - 灾难性遗忘
  - 时序回放
  - Transformer
---

# iManip: Skill-Incremental Learning for Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2503.07087](https://arxiv.org/abs/2503.07087)  
**代码**: 即将开源  
**领域**: 机器人  
**关键词**: 增量学习, 机器人操作, 灾难性遗忘, 时序回放, 可扩展Transformer

## 一句话总结

提出 iManip 框架，通过时序回放策略和可扩展 PerceiverIO 架构，使机器人能够在不重新训练的情况下持续学习新的操作技能，同时缓解对已学技能的灾难性遗忘，在 RLBench 上比传统增量基线平均提升 9.4%。

## 研究背景与动机

**领域现状**：机器人操作领域的主流方法集中于提升单一或多任务操作性能（PerAct、ManiGaussian 等），或利用预训练大语言/视觉模型的知识迁移到机器人操作任务，但很少有工作研究如何让机器人增量地学习新技能。

**现有痛点**：已有的增量学习 benchmark（如 LIBERO）虽然初步探索了终身学习问题，但其任务之间共享相同的操作技能（如 "把碗放到盘子上" vs "把碗放到炉子上"），仅仅是物体或空间位置的变化，并不涉及真正的新技能学习。将传统增量学习方法（如 iCaRL、EEIL）直接应用到机器人操作场景中，依然会出现严重的灾难性遗忘。

**核心矛盾**：传统增量学习方法主要面向图像分类，忽略了机器人操作任务中两个独特的复杂性——（1）**时序复杂性**：环境和机器人状态随时间动态变化，每个动作都会影响后续动作；（2）**动作复杂性**：机器人需要学习新的动作原语（如平移、旋转、夹持），这些在 3D 空间中的表示高度复杂。

**本文目标** 设计一个技能增量学习框架，使机器人在学习新操作技能时保留旧技能的知识，不需要从头训练。

**切入角度**：作者观察到经典的样本回放方法（herding、hard-exemplar sampling）在选择代表性样本时忽略了轨迹数据的时序特性，导致时序不平衡；同时经典方法只关注视觉特征而忽视了动作空间的扩展需求。

**核心 idea**：通过分关键帧的最远距离熵采样保持时序数据完整性，同时用可扩展权重矩阵和技能专属动作提示来适配新技能的动作原语。

## 方法详解

### 整体框架

iManip 的输入是多视角 RGB-D 图像和语言指令，输出是机器人动作（包含 3D 平移、旋转、夹持器开合、碰撞避免）。框架包含三个主要组件：
- **体素编码器**：将 RGB-D 图像投影为 3D 体素，用 UNet 架构的 3D 卷积编码器提取场景特征
- **可扩展 PerceiverIO**（核心）：接收体素 token、语言 token 和动作提示 token 的多模态输入，通过交叉注意力和可扩展自注意力层进行编码
- **策略解码器**：预测最优机器人动作

学习流程为：先在基础技能集上训练，然后每一步增量学习一个新技能，同时利用记忆库中存储的旧技能回放样本和知识蒸馏来保持旧知识。

### 关键设计

1. **时序回放策略（Temporal Replay Strategy）**:

    - 功能：为旧技能存储固定数量的代表性演示样本，保持时序数据的完整性
    - 核心思路：首先按关键帧（keyframe，即末端执行器状态变化或速度接近零的时刻）对演示轨迹进行分段。然后对每种关键帧类型，使用**最远距离熵采样**选取样本。具体地，计算每个样本的动作预测熵，构建距离矩阵 $A[i][j] = \text{distance}(e_i, e_j)$，贪心地选择与已选样本集熵距离之和最大的新样本：$j = \arg\max_{j \in E} \sum_{k \in S} A[j][k]$，时间复杂度为 $O(N^2)$
    - 设计动机：经典回放方法（如 herding）直接选最具代表性的样本，但忽略了时序平衡——可能同一阶段的样本被大量选中，导致执行时不稳定。按关键帧均匀采样 + 最大化熵距离可以兼顾时序覆盖和变体多样性

2. **可扩展 PerceiverIO（Extendable PerceiverIO）**:

    - 功能：通过可扩展的权重矩阵和技能专属动作提示来适配新技能的动作原语
    - 核心思路：输入 $X = [X_{\text{voxel}}, X_{\text{language}}, X_{\text{action}}]$，其中动作提示 $X_{\text{action}} = [X_{\text{action}}^{\text{old}}, X_{\text{action}}^{\text{new}}]$ 拼接旧/新技能的提示。在自注意力层中，Q 和 K 的权重矩阵是可扩展的：$W_Q^{\text{scale}} = [W_Q^{\text{old}}, W_Q^{\text{new}}]$，学习新技能时冻结旧权重 $W_Q^{\text{old}}$，仅学习新权重 $W_Q^{\text{new}} \in \mathbb{R}^{d \times d_{\text{new}}}$ 和新动作提示
    - 设计动机：机器人操作中不同技能需要不同的动作原语（如倒水 vs 推按钮），传统方法用共享权重难以适配。通过冻结旧权重+扩展新权重的方式，既防止旧知识被覆盖，又为新技能提供专属的学习容量

3. **知识蒸馏（Knowledge Distillation）**:

    - 功能：在新旧模型之间进行知识迁移，进一步防止遗忘
    - 核心思路：使用旧模型的输出概率分布来指导新模型的训练。蒸馏损失 $\mathcal{L}_{\text{dis}} = \mathcal{L}_2(\mathcal{Q}_{\text{trans}}^{\text{old}}, \mathcal{Q}_{\text{trans}}^{\text{new}}) + \mathcal{L}_2(\mathcal{Q}_{\text{rot}}^{\text{old}}, \mathcal{Q}_{\text{rot}}^{\text{new}}) + |\mathcal{Q}_{\text{open}}^{\text{old}} - \mathcal{Q}_{\text{open}}^{\text{new}}| + |\mathcal{Q}_{\text{collide}}^{\text{old}} - \mathcal{Q}_{\text{collide}}^{\text{new}}|$，分别对平移、旋转用 MSE，对夹持和碰撞用 L1
    - 设计动机：仅靠回放和权重冻结可能不够，蒸馏提供了额外的正则化信号，确保旧技能的动作分布不被破坏

### 损失函数 / 训练策略

总损失为 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{act}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}}$，其中 $\mathcal{L}_{\text{act}}$ 是标准交叉熵动作损失（涵盖平移、旋转、开合、碰撞四个分支），$\lambda_{\text{dis}} = 0.01$。学习新技能时冻结编码器和 PerceiverIO 的旧参数，仅微调策略解码器、新动作提示和新扩展权重，实现了更快的收敛和更少的参数量。

## 实验关键数据

### 主实验

在 RLBench 上的 B5-5N1 设置（先学 5 个基础技能，再每步增量学 1 个新技能，共 5 步学 10 个技能）：

| 方法 | Base | Step1 All | Step2 All | Step3 All | Step4 All | Step5 All | 平均 |
|------|------|-----------|-----------|-----------|-----------|-----------|------|
| PerAct (多任务) | 44.0 | 7.3 | 5.1 | 9.0 | 6.7 | 1.6 | 5.9 |
| ManiGaussian (多任务) | 55.2 | 20.7 | 12.0 | 15.5 | 9.3 | 5.2 | 12.5 |
| P-TIB (增量) | 44.0 | 34.7 | 25.1 | 26.0 | 16.4 | 10.4 | 22.5 |
| M-TIB (增量) | 55.2 | 45.3 | 37.1 | 39.5 | 31.6 | 26.8 | 36.1 |
| **iManip (本文)** | **56.0** | **56.7** | **48.0** | **47.5** | **39.1** | **36.0** | **45.5** |

### 消融实验

| 配置 | B5-1N1 | B5-5N1 | 说明 |
|------|--------|--------|------|
| R1: 无增量策略 | 20.7 | 5.2 | 无任何缓解遗忘的机制 |
| R2: +TRS | 49.3 | 27.6 | +时序回放，提升 +22.4% |
| R3: +TRS+EPIO | 54.0 | 32.4 | +可扩展 PerceiverIO，+4.8% |
| Full: +TRS+EPIO+DIS | 56.7 | 36.0 | +蒸馏，+3.6%，完整模型最优 |

### 关键发现

- **时序回放贡献最大**：去掉 TRS 后性能骤降 22.4%，说明在机器人操作中保持轨迹时序完整性至关重要
- **冻结编码器+EPIO、仅训练解码器效果最好**：仅需 8M 参数（vs 全量 47M），收敛步数从 100k 降到 60k，同时旧技能保留率更高
- **经典回放方法失效**：herding 和 hard-exemplar 在旧技能上只有约 15% 成功率，而时序回放达到 57.6%
- 在不同增量设置（B5-1N5、B2-4N2、B3-2N3）下方法均保持优势
- 真实机器人实验：5 个日常操作技能，4 步增量后旧技能成功率从基线 0% 提升到 40%

## 亮点与洞察

- **时序回放策略设计巧妙**：通过按关键帧分组采样，既保持了轨迹的时序完整性，又通过最远距离熵采样增加了变体多样性。这个思路可以迁移到任何需要回放时序数据的增量学习场景
- **可扩展权重矩阵**：在 Q/K 投影矩阵上横向拼接新权重列的思路简单有效，冻结旧权重+仅学新权重实现了零遗忘。类似思路可用于其他需要持续扩展能力的 Transformer 架构
- **Grad-CAM 可视化验证**：不同技能激活不同的动作提示权重，证明了动作提示确实学到了技能特定的动作原语

## 局限与展望

- 每步只增加 1 个新技能，未测试同时增加多个复杂技能的场景
- 技能之间相对独立，未探索技能间的知识共享和组合
- 仅使用单视角 RGB-D 输入，缺乏多视角信息
- 评估主要在 RLBench 仿真环境，真实世界实验规模较小（每技能仅 10 次测试）
- 存储开销会随已学技能数量线性增长

## 相关工作与启发

- **vs LIBERO**：LIBERO 探索的是物体/空间变化的增量学习，本文探索的是真正的新技能增量，更贴近现实需求
- **vs iCaRL/EEIL**：这些传统增量方法在分类任务上有效，但忽略了机器人操作的时序和动作复杂性，直接应用效果很差
- **vs PerAct/ManiGaussian**：这些多任务方法在面对新技能时需要重新训练所有技能，效率极低

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地定义并解决机器人操作的技能增量学习问题，但核心技术（回放+冻结+蒸馏）均为已有思路的改良
- 实验充分度: ⭐⭐⭐⭐ 消融充分，多种设置对比，有真实机器人实验，但真实世界实验规模偏小
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详尽，图示直观
- 价值: ⭐⭐⭐⭐ 为机器人持续学习提供了实用的 benchmark 和 baseline，有很好的实际应用前景

<!-- RELATED:START -->

## 相关论文

- [Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](../../ACL2025/robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization](../../ICML2025/robotics/star_learning_diverse_robot_skill_abstractions_through_rotation-augmented_vector.md)
- [RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](../../CVPR2025/robotics/roboground_robotic_manipulation_with_grounded_vision-language_priors.md)

<!-- RELATED:END -->
