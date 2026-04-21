---
title: >-
  [论文解读] Learning Cross-Hand Policies of High-DOF Reaching and Grasping
description: >-
  [ECCV 2024][人体理解][灵巧抓取] 提出一种两阶段层次化框架，通过语义关键点和交互等分面（IBS）作为手型无关的状态表示，结合Transformer策略网络和手型特定的适配模型，实现了灵巧抓取策略在不同高自由度机械手之间的零样本迁移。
tags:
  - ECCV 2024
  - 人体理解
  - 灵巧抓取
  - 策略迁移
  - 跨手型泛化
  - Transformer
  - 交互等分面
---

# Learning Cross-Hand Policies of High-DOF Reaching and Grasping

**会议**: ECCV 2024  
**arXiv**: [2404.09150](https://arxiv.org/abs/2404.09150)  
**代码**: 无  
**领域**: 人体理解  
**关键词**: 灵巧抓取, 策略迁移, 跨手型泛化, Transformer, 交互等分面

## 一句话总结

提出一种两阶段层次化框架，通过语义关键点和交互等分面（IBS）作为手型无关的状态表示，结合Transformer策略网络和手型特定的适配模型，实现了灵巧抓取策略在不同高自由度机械手之间的零样本迁移。

## 研究背景与动机

机器人"伸手-抓取"是操控领域的基础技能。现有学习方法通常只为单一夹爪训练模型，换用其他夹爪时需重新收集数据并从头训练，代价高昂。尽管已有工作探索了跨物体的泛化能力，但**跨灵巧手型的策略迁移**几乎未被研究。

现有跨手型抓取方法（如UniGrasp、GenDexGrasp等）大多只生成静态抓取姿态，无法在执行过程中动态调整。本文的核心假设是：**不同灵巧手的抓取技能存在共性**，限制泛化能力的是状态和动作的表示方式，而非技能本身。因此，关键挑战在于找到一种手型无关的几何表示，以消除两个因素的影响：

**手型形态差异**：不同手的关节空间维度和拓扑结构差异显著

**手型几何差异**：点云/图像等表示可能导致策略过拟合于特定手型的几何外观

## 方法详解

### 整体框架

方法采用两阶段层次化模型：

1. **统一策略模型**（Unified Policy Model）：在所有手型间共享，输入手型无关的特征，预测语义关键点的位移
2. **手型特定适配模型**（Specific Adaptation Model）：将关键点位移转换为具体手型的关节角变化

| 模块 | 输入 | 输出 | 特点 |
|------|------|------|------|
| 手型无关特征提取 | 场景点云 + 手型配置 | IBS点特征 + 语义关键点 | 统一表示，不依赖特定手型 |
| 统一策略模型 | IBS + 语义关键点 | 关键点位移 + 全局平移/旋转 + 停止信号 | Transformer架构，手型间共享 |
| 适配模型 | 关键点位移 + 当前关节角 | 关节角变化 | 轻量MLP，每种手型单独训练 |

### 关键设计

**1. 语义关键点（Semantic Key Points）**

受动画系统中IK Rig的启发，在每根手指上选取两个语义关键点（指尖点和中间指节点），加上手掌根部点，构建手型无关的状态表示。关键点位置通过正向运动学计算，定义在手型局部坐标系中。完整语义关键点输入包含 $6(K+1)$ 维，其中 $K$ 为手指数量。

**2. 交互等分面（IBS）**

IBS是手型与场景之间的Voronoi图，编码了两者的空间交互关系。通过在手掌中心周围的球形区域内体素化计算近似IBS点，下采样到4096个点作为网络输入。每个IBS点包含丰富的特征：坐标、到场景的距离、到手型的距离、前景指示器、手型部件归属的one-hot编码、以及手型表面朝向指示器。

**3. Transformer策略网络**

网络由三部分组成：
- **逐指编码器**：使用MLP和PointNet分别对每根手指的语义点和IBS点进行编码
- **Transformer编码器**：通过自注意力机制融合不同手指和不同表示之间的信息
- **逐指/全局解码器**：预测每根手指的关键点位移和全局运动

这种设计使模型能自然适应不同手指数量的手型（如从五指手迁移到四指手）。

### 损失函数 / 训练策略

**训练分两阶段：**

1. **联合训练（Joint Training）**：策略模型和适配模型同时训练，各自有独立的损失函数，梯度不跨模型传播。训练800k步。
2. **迁移训练（Transfer Training）**：冻结策略模型，为新手型从头训练适配模型，仅需50k步。

**策略模型**使用强化学习训练（Soft Actor-Critic算法），奖励函数包含：
- 任务奖励：成功稳定抓取
- 趋近奖励：避免手型与场景碰撞

**适配模型**使用自监督循环损失训练：

$$L_{point}(\theta) = \frac{1}{2}\sum_{k=1}^{K}\sum_{i=0}^{1}(e_k^i - p_k^i - \Delta p_k^i)^2$$

其中 $e_k^i = FK_k^i(j + \Delta j)$ 为通过可微正向运动学计算的预期关键点位置。同时加入自碰撞损失避免手指间穿透。

## 实验关键数据

### 主实验（表格）

实验在5种灵巧手上测试：Shadow、Schunk、Mano、Rutgers、Allegro。使用Shadow Hand训练策略，其他手型用于迁移测试。

| 方法 | Shadow SR | Schunk SR | Mano SR | Rutgers SR | Allegro SR |
|------|-----------|-----------|---------|------------|------------|
| Single (端到端) | 72.2% | - | - | - | - |
| UNI+OCM | 50.1% | 41.2% | 45.5% | 38.9% | - |
| UNI+GCM | 64.0% | 45.5% | 50.4% | 41.4% | - |
| UNI+IBS | 68.0% | 54.6% | 61.2% | 42.6% | - |
| **Ours** | **71.3%** | **65.3%** | **65.2%** | **54.8%** | **55.0%** |

### 消融实验（表格）

| 对比项 | 发现 |
|--------|------|
| 两阶段 vs 端到端 | 两阶段框架在原始手型上性能接近端到端(71.3% vs 72.2%)，但能迁移到其他手型 |
| IBS vs OCM | OCM在原始手型上过拟合训练物体(50.1%)，IBS显著优于OCM(68.0% vs 50.1%) |
| IBS vs GCM | GCM在原始手型上较好(64.0%)但迁移差，因过拟合手型几何 |
| Transformer vs 朴素拼接 | Transformer策略在所有手型上优于特征拼接，且能适配四指Allegro手 |

### 关键发现

1. **IBS是最有效的空间交互表示**：平衡了物体和手型几何信息，显著优于单独使用物体接触图(OCM)或手型接触图(GCM)
2. **Transformer架构至关重要**：不仅提升了迁移性能，还使模型能适应不同手指数量的手型
3. **迁移效率高**：新手型的适配模型仅需50k步训练，远少于联合训练的800k步
4. **适配模型实时性好**：相比优化方法，神经网络适配模型速度快数倍且无碰撞

## 亮点与洞察

- **核心洞察**："抓取技能的共性存在于不同手型之间，关键是找到合适的表达方式"——这一思路可推广到其他跨形态的技能迁移任务
- **IBS的跨域有效性**：IBS作为Voronoi图的产物，天然对交互双方的几何具有鲁棒性，这在抓取以外的接触密集任务中也可能有用
- **关键点作为通用动作空间**：类似动画系统中的IK控制，语义关键点提供了一种直观且统一的动作接口

## 局限与展望

1. **仅在仿真环境中验证**：基于PyBullet，未在真实机器人上测试sim-to-real迁移
2. **手型差异有限**：测试的5种手型均为类人手，未测试非类人构型（如软体手）
3. **物体多样性不足**：测试物体来自YCB等有限数据集，对不规则形状的泛化未充分验证
4. **适配模型仍需训练数据**：虽然迁移训练较快，但仍需为每种新手型收集数据

## 相关工作与启发

- **与GraspXL的区别**：GraspXL为每种手型训练独立模型，不允许直接策略迁移
- **与运动重定向的关联**：虽然运动重定向可在视觉上生成相似动作，但在动态环境中执行效果差；本文方法直接学习可执行的策略
- **可能的扩展方向**：结合大规模预训练，可能实现更多形态的零样本迁移
- **启发**：统一表示+分层架构的思路可用于其他多形态机器人任务（如locomotion、manipulation）

## 评分

| 维度 | 分数 (1-5) | 评价 |
|------|-----------|------|
| 新颖性 | 4 | 首次实现灵巧手之间的策略迁移，IBS+关键点的表示设计巧妙 |
| 技术深度 | 4 | 两阶段框架设计合理，Transformer网络架构有针对性 |
| 实验充分性 | 3.5 | 消融实验充分，但缺少真实机器人实验 |
| 写作质量 | 4 | 结构清晰，动机阐述明确 |
| 实用价值 | 3.5 | 对多手型机器人系统有较大参考价值，但实用落地需sim-to-real验证 |

<!-- RELATED:START -->

## 相关论文

- [Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References](../../CVPR2025/human_understanding/learning_physics-based_full-body_human_reaching_and_grasping_from_brief_walking_.md)
- [GraspXL: Generating Grasping Motions for Diverse Objects at Scale](graspxl_generating_grasping_motions_for_diverse_objects_at_scale.md)
- [One-stage Prompt-based Continual Learning](one-stage_prompt-based_continual_learning.md)
- [3D Hand Pose Estimation in Everyday Egocentric Images](3d_hand_pose_estimation_in_everyday_egocentric_images.md)
- [UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)

<!-- RELATED:END -->
