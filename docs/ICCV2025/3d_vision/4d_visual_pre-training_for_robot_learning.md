---
title: >-
  [论文解读] 4D Visual Pre-training for Robot Learning
description: >-
  [ICCV 2025][3D视觉][3D点云预训练] FVP提出了一种基于4D（3D空间+时间）点云预测的视觉预训练框架，通过将预训练目标建模为"下一帧点云预测"并用扩散模型实现，显著提升了多种3D模仿学习方法在真实机器人操作任务上的成功率（DP3平均提升28%）。
tags:
  - ICCV 2025
  - 3D视觉
  - 3D点云预训练
  - 扩散模型
  - 机器人操作
  - 模仿学习
  - 视觉表征
---

# 4D Visual Pre-training for Robot Learning

**会议**: ICCV 2025  
**arXiv**: [2508.17230](https://arxiv.org/abs/2508.17230)  
**代码**: [https://4d-visual-pretraining.github.io/](https://4d-visual-pretraining.github.io/)  
**领域**: 3D视觉 / 机器人学习  
**关键词**: 3D点云预训练, 扩散模型, 机器人操作, 模仿学习, 视觉表征

## 一句话总结
FVP提出了一种基于4D（3D空间+时间）点云预测的视觉预训练框架，通过将预训练目标建模为"下一帧点云预测"并用扩散模型实现，显著提升了多种3D模仿学习方法在真实机器人操作任务上的成功率（DP3平均提升28%）。

## 研究背景与动机
- **领域现状**：当前机器人视觉表征学习以2D图像预训练为主（R3M、MVP、VC-1等），虽然在ImageNet和Ego4D等大规模数据集上取得了良好效果，但忽略了现实世界的3D本质
- **3D方法的优势**：使用3D点云的模仿学习方法（DP3、RISE等）在真实机器人任务上展现了更强的泛化能力和效率，但缺少通用的3D预训练范式
- **核心矛盾**：大规模3D数据稀缺，无法像2D那样从海量网络数据中学习通用3D表征
- **本文切入角度**：既然难以训练单一的通用3D表征，不如设计一个能增强所有3D编码器的预训练框架。利用扩散模型的生成能力，将预训练目标设定为"根据当前帧点云预测下一帧点云"，从而学到理解物理环境动态的视觉特征
- **核心idea**：将4D预训练转化为条件扩散生成问题——用历史帧点云作为条件，通过扩散模型去噪生成未来帧点云，迫使3D编码器学习有意义的时空表征

## 方法详解

### 整体框架
FVP的pipeline分为两个阶段：（1）预训练阶段：收集机器人轨迹数据（观测-动作对序列），用扩散模型进行"下一帧点云预测"自监督学习；（2）下游微调阶段：将预训练好的3D编码器替换到DP3、RISE等策略中，进行端到端微调。

### 关键设计

1. **下一帧点云预测目标**:

    - 功能：用上一帧点云 $o^{t-1}$ 编码出视觉表征，结合扩散模型预测当前帧点云 $o^t$
    - 核心思路：不同于对比学习（正负样本构造）或遮蔽重建（mask-and-reconstruct），FVP通过预测未来状态来学习物理动态。编码器将 $o^{t-1}$ 编码为潜在表征 $\mathbf{z} \in \mathbb{R}^{N \times C_v}$，然后将 $\mathbf{z}$ 拼接到加噪的目标点云上，形成输入 $o_{T,+}^t \in \mathbb{R}^{N \times (C_v+3)}$
    - 设计动机：预测未来点云迫使编码器理解机器人运动特征和环境动态变化，这种"预测式"学习比"重建式"学习更能捕捉对机器人控制至关重要的时序信息

2. **条件扩散生成模型**:

    - 功能：将去噪过程建模为条件扩散概率模型，从高斯噪声中逐步恢复目标点云
    - 核心思路：扩散过程遵循马尔可夫链 $q(X_t|X_{t-1}) = \mathcal{N}(X_t; \sqrt{1-\beta_t}X_{t-1}, \beta_t\mathbf{I})$，训练损失为 $\mathcal{L} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\mathbf{I})}[\|\epsilon - \epsilon_\theta(o_{+,T}^t, T)\|_2^2]$
    - 具体实现：采用修改版的Point-Voxel Diffusion网络，网络结构为 $\epsilon_\theta: \mathbb{R}^{N\times(C_v+3)} \to \mathbb{R}^{N\times 3}$，同时融入机器人动作信息辅助生成
    - 设计动机：扩散模型在视觉任务中已证明能产生优质表征，其迭代去噪过程天然适合学习点云的空间结构

3. **通用编码器兼容性**:

    - 功能：FVP设计为即插即用的预训练模块，适配多种3D编码器
    - 支持的编码器：PointNet++、Point Transformer、DP3 Encoder、RISE Encoder
    - 设计动机：通过解耦编码器和预训练目标，使得FVP成为一个通用的预训练框架而非特定于某种架构的方案

### 损失函数 / 训练策略
- 预训练损失：标准扩散模型的噪声预测L2损失，条件输入包含历史帧编码和动作信息
- 下游微调：将预训练编码器替换原始编码器后，端到端微调视觉编码器和策略骨干网络
- 数据要求：可使用领域内数据或跨领域数据（如RoboMind公开数据集），每个任务收集50个专家演示

## 实验关键数据

### 主实验

| 任务 | 方法 | 成功率 | 对比SOTA | 提升 |
|------|------|--------|----------|------|
| PickSquare | DP3+FVP | 20/20 | MAE (18/20) | +10% |
| PlaceBottle | DP3+FVP | 20/20 | MVP (15/20) | +25% |
| PickPlace | DP3+FVP | 17/20 | MVP (16/20) | +5% |
| FlipCup | DP3+FVP | 16/20 | MVP (17/20) | 持平 |
| Assembly | DP3+FVP | 13/20 | MAE (11/20) | +10% |
| ArtiManip | DP3+FVP | 16/20 | R3M/MVP (14/20) | +10% |
| **平均** | **DP3+FVP** | **16.4/20** | **MAE (15.3/20)** | **+5.7%** |

### 消融实验

| 配置 | PickSquare | PlaceBottle | PushDraw | ToastBread | 说明 |
|------|-----------|-------------|----------|------------|------|
| DP3+FVP（完整） | 20/20 | 20/20 | 20/20 | 16/20 | 使用历史帧作为条件 |
| 使用当前帧替代历史帧 | 15/20 | 14/20 | 13/20 | 13/20 | 历史时序信息至关重要 |
| 冻结视觉编码器 | 11/20 | 9/20 | 10/20 | 7/20 | 端到端微调必不可少 |

### 关键发现
- FVP在仿真任务上使用领域内数据集预训练带来平均16.9%的提升，跨领域数据集带来24.7%提升
- 在灵巧手任务上提升尤为显著，因为FVP通过时序帧理解复杂运动轨迹
- 将FVP应用于VLA模型RDT-1B，在空间理解任务上从8/20提升到14/20，知识迁移从10/20到16/20
- 历史帧步长为1帧时效果最佳（20/20），随步长增加效果下降（4帧：14-15/20）
- 2D预训练方法（R3M/MVP/MAE）即使在300M+数据上预训练，仍不如FVP的3D预训练效果

## 亮点与洞察
- **预训练范式创新**：首次将"下一帧预测"引入3D点云预训练，区别于传统的对比学习和遮蔽建模，更贴近机器人的序列决策本质
- **即插即用设计**：作为通用预训练模块可增强多种3D编码器（PointNet++、Point Transformer等）和策略方法（DP3、RISE）
- **跨领域预训练有效**：在RoboMind等公开数据集上预训练的编码器可迁移到不同机器人平台
- **VLA模型增强**：证明了3D点云输入和FVP预训练能有效提升大规模VLA模型的空间感知能力

## 局限与展望
- 冻结预训练编码器效果差，说明领域内外数据存在差距，预训练表征还不够通用
- 每个任务需要独立预训练视觉编码器，尚未实现真正的"一个预训练，多任务适用"
- 历史帧步长增大后性能下降（4帧步长：PickSquare从20/20降至15/20），对长时序依赖的建模能力有限
- 缺乏与更大规模3D预训练数据集上的实验验证
- 扩散模型的预训练过程本身的效率未讨论——对比简单的重建目标是否值得额外计算开销
- 仅验证了操作任务，对导航、运动规划等其他机器人任务的效果未知
- 点云采集依赖RGB-D相机，不同相机噪声特性对预训练效果的影响需进一步研究

## 相关工作与启发
- **vs R3M/MVP**：2D预训练模型（在300M+数据集上预训练）在同样的策略框架下明显弱于FVP，即便FVP仅用领域内/跨领域小数据预训练，验证了3D表征的优越性
- **vs PointMAE/STRL/C2P**：FVP在领域内和跨领域预训练上全面超越这些3D/4D预训练方法。PointMAE采用遮蔽重建、STRL采用时序对比学习、C2P采用跨模态预测，而FVP的"下一帧预测"目标更直接地学习物理动态
- **vs ACT/Diffusion Policy**：2D模仿学习方法对相机位置敏感，难以捕捉3D空间信息
- **与VLA模型的结合**：FVP可直接增强RDT-1B等大规模VLA模型，在空间感知、任务迁移和长程任务上均有提升，展示了3D预训练与大模型结合的潜力

## 评分
- 新颖性: ⭐⭐⭐⭐ 将下一帧预测范式从NLP/2D视觉引入3D点云预训练，思路简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖仿真和真实场景共24个任务、四种机器人平台（单臂/灵巧手/双臂/人形）、VLA模型集成
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验组织合理，可视化效果好
- 价值: ⭐⭐⭐⭐ 为3D机器人学习提供了实用的预训练方案，具有较好的工程落地潜力
- 综合: ⭐⭐⭐⭐ 实验全面、方法简洁、效果显著

<!-- RELATED:START -->

## 相关论文

- [Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [RoboPearls: Editable Video Simulation for Robot Manipulation](robopearls_editable_video_simulation_for_robot_manipulation.md)
- [Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)

<!-- RELATED:END -->
