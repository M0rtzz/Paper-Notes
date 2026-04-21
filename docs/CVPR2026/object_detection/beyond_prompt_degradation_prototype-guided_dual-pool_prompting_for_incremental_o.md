---
title: >-
  [论文解读] Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection
description: >-
  [CVPR 2026][目标检测][增量目标检测] 提出 PDP 框架，通过双池提示解耦（共享池 + 私有池）和原型引导伪标签生成（PPG），解决增量目标检测中提示耦合与提示漂移导致的提示退化问题，在 COCO 和 VOC 上取得 SOTA。
tags:
  - CVPR 2026
  - 目标检测
  - 增量目标检测
  - 提示学习
  - 双池范式
  - 原型伪标签
  - 灾难性遗忘
---

# Beyond Prompt Degradation: Prototype-Guided Dual-Pool Prompting for Incremental Object Detection

**会议**: CVPR 2026  
**arXiv**: [2603.02286](https://arxiv.org/abs/2603.02286)  
**代码**: [有](https://github.com/zyt95579/PDP_IOD/tree/main)  
**领域**: 目标检测  
**关键词**: 增量目标检测, 提示学习, 双池范式, 原型伪标签, 灾难性遗忘

## 一句话总结

提出 PDP 框架，通过双池提示解耦（共享池 + 私有池）和原型引导伪标签生成（PPG），解决增量目标检测中提示耦合与提示漂移导致的提示退化问题，在 COCO 和 VOC 上取得 SOTA。

## 研究背景与动机

增量目标检测（IOD）要求模型在不访问旧数据的前提下持续学习新类别，同时保持对旧类别的检测性能。基于提示（prompt）的方法因其无需数据回放、参数高效而受到关注，但存在两个核心问题：

**提示耦合（Prompt Coupling）**：现有方法采用单提示池范式，将任务通用提示和任务特定提示混存在同一池中，导致有限参数空间内的竞争和干扰

**提示漂移（Prompt Drift）**：在 IOD 设置中，旧类前景对象在后续任务中被标注为"背景"，这种监督不一致迫使已优化的提示朝错误语义方向漂移

现有伪标签方法依赖固定置信度阈值，无法适应类别间的分布差异，进一步加剧漂移。

## 方法详解

### 整体框架

PDP 构建于 Deformable-DETR 之上，采用教师-学生蒸馏架构。核心包含两个模块：
- **Decoupled Dual-Pool Prompting (DDP)**：解耦的双池提示机制
- **Prototypical Pseudo-Label Generation (PPG)**：原型引导的伪标签生成

### 关键设计

#### 1. 双池提示解耦（DDP）

**共享池（Shared Pool）**：
- 包含可学习提示 $P_s \in \mathbb{R}^{N_s \times L_p \times D}$、键向量 $K_s$ 和查询适配器 $A_s$
- 在所有任务间共享，持续优化以捕获通用视觉知识
- 促进稳定的前向知识迁移

**私有池（Private Pool）**：
- 每个任务分配独立的提示参数 $(P_p^t, K_p^t, A_p^t)$
- 训练时仅更新当前任务的参数，旧任务参数冻结
- 私有池大小 $N_p$ 根据新类数量动态调整

**提示检索与集成**：
- 查询提取器生成查询向量后，通过 Hadamard 积与查询适配器调制
- 计算与两个池中键向量的余弦相似度加权聚合提示
- 通过 Prefix-Tuning 将检索到的提示 $P_r$ 注入 Transformer 解码器

**池间多样性约束**：采用方向解耦损失最大化两池向量的角度分离：

$$\mathcal{L}_{DDL} = \lambda_{ddl} \cdot \frac{2}{|N_s||N_p|} \sum_{i,j} \max(0, \theta_{ddl} - \theta_{i,j})$$

其中 $\theta_{ddl} = 90°$，确保共享与私有提示学习互补的正交表示。

#### 2. 原型伪标签生成（PPG）

**类原型空间构建**：
- 从解码器最后一层提取正确分类实例的查询嵌入 $f_i$
- 存入类特定记忆库 $F_c$，计算类原型 $p_c = \frac{1}{|F_c|}\sum_{f_i \in F_c} f_i$
- 仅在每个任务最后一个 epoch 更新原型，确保特征充分收敛

**层级验证机制**：
- **简单样本**：置信度 $s_i > \tau_h$（0.5）的检测直接作为可靠伪标签
- **困难样本**：中间置信度 $\tau_l < s_i < \tau_h$（0.2 < s_i < 0.5）时，计算特征与类原型的相似度，超过阈值则保留
- 两类样本合并生成高质量伪标签集 $Y_{ppg}$

### 损失函数 / 训练策略

总损失由 DETR 检测损失、查询正则化、方向解耦损失和蒸馏损失组成：

$$\mathcal{L} = \mathcal{L}_{DETR} + \mathcal{L}_Q + \mathcal{L}_{DDL} + \mathcal{L}_{DKD}(Y_{ppg})$$

- $\lambda_{ddl} = 0.15$，$\lambda_Q = 0.1$
- 共享池 100 个提示，私有池数量等于数据集类别总数
- 置信度阈值 $\tau_h = 0.5$，$\tau_l = 0.2$，原型相似度阈值 $\theta_s = 0.5$

## 实验关键数据

### 主实验

**表1：MS-COCO 多步增量设置（4个任务）**

| 方法 | Task4 mAP@P | Task4 mAP@C | Task4 mAP@A |
|------|------------|------------|------------|
| MD-DETR | 51.5 | 52.7 | 50.2 |
| OWOBJ | 49.4 | 38.8 | 43.9 |
| **PDP (Ours)** | **61.3 (+9.8)** | **55.8 (+3.1)** | **59.4 (+9.2)** |

**表3：PASCAL VOC 三种增量设置**

| 方法 | 10+10 mAP@A | 15+5 mAP@A | 19+1 mAP@A |
|------|-------------|-------------|-------------|
| MD-DETR | 73.2 | 76.7 | 76.1 |
| RGR | 75.8 | 73.4 | 75.4 |
| **PDP (Ours)** | **78.7 (+2.9)** | **78.0 (+1.3)** | **79.4 (+3.3)** |

### 消融实验

**模块贡献（Table 4，COCO Task4 mAP@A）**：

| 配置 | mAP@P | mAP@A |
|------|-------|-------|
| 仅私有池 PP | 46.0 | 46.0 |
| PP + SP + DDL | 56.9 | 55.1 |
| PP + PPG | 59.9 | 58.3 |
| PP + SP + PPG + DDL（完整） | **61.3** | **59.4** |

PPG 使旧知识保持率 mAP@P 提升 +13.9%，同时新类适应 mAP@C 提升 +2.7%。

### 关键发现

1. PPG 对三种不同相似度阈值（0.5/0.6/0.7）表现稳定，说明困难样本与原型的相似度天然较高
2. 共享池 $N_s=100$、私有池 $N_p=80$ 为最优配置；共享池过大（160）反而引入冗余导致性能下降
3. PDP 在 PASCAL VOC 19+1 设置下 mAP@P 达到 70.1%，直观可视化表明能准确检测旧类对象

## 亮点与洞察

1. **问题建模精准**：首次将提示退化分解为"耦合"和"漂移"两个独立可处理的子问题
2. **原型空间替代置信度阈值**：在嵌入空间中用类原型做相似度匹配，避免了固定阈值在类别间分布不一致时的失效
3. **端到端框架**：相比 PseDet 需要额外推理+聚类步骤，PDP 是完全端到端的

## 局限与展望

1. 在 70+10 两步设置下略逊于 PseDet（42.9 vs 44.7 AP），大步增量场景仍有提升空间
2. 原型仅在每个任务最后一个 epoch 更新，早期训练阶段原型可能不够准确
3. 私有池随任务数线性增长，长序列任务下参数量管理需要关注
4. 仅在 DETR 系列架构上验证，对 YOLO 等单阶段检测器的适用性未知

## 相关工作与启发

- **MD-DETR**：PDP 的基线架构，使用单记忆库 + 任务 ID 隔离提示
- **DualPrompt**：虽区分 General/Expert Prompt 但仍用单池管理
- **PseDet**：使用 k-means 自适应阈值做伪标签，非端到端
- 双池设计思想可推广到增量分割、增量实例分割等持续学习任务

## 评分

- **新颖性**: ★★★★☆ — 双池解耦+原型伪标签的组合设计新颖
- **技术深度**: ★★★★☆ — 方法设计完整，各模块动机清晰
- **实验充分性**: ★★★★★ — 多数据集、多设置、详细消融
- **写作清晰度**: ★★★★☆ — 图示清晰，问题-方案对应关系明确

<!-- RELATED:START -->

## 相关论文

- [YOLO-IOD: Towards Real Time Incremental Object Detection](../../AAAI2026/object_detection/yolo-iod_towards_real_time_incremental_object_detection.md)
- [VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](ewdetr_evolving_world_object_detection.md)
- [CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)
- [DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration](dualreg_dual-space_filtering_and_reinforcement_for_rigid_registration.md)

<!-- RELATED:END -->
