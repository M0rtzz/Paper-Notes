---
title: >-
  [论文解读] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation
description: >-
  [ICCV 2025][图像生成][机器人操作] 提出 A₀，一个可供性感知的分层扩散模型，通过将操作任务分解为高层空间可供性理解（预测接触点和轨迹）和低层动作执行，在100万接触点数据上预训练后仅需少量任务数据微调即可跨平台(Franka/Kinova/Realman/Dobot)部署，在擦白板等复杂轨迹任务中成功率达45%。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "机器人操作"
  - "空间可供性"
  - "分层模型"
  - "扩散模型"
  - "跨平台泛化"
---

# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2504.12636](https://arxiv.org/abs/2504.12636)  
**代码**: [https://a-embodied.github.io/A0/](https://a-embodied.github.io/A0/)  
**领域**: 图像生成  
**关键词**: 机器人操作, 空间可供性, 分层模型, 扩散模型, 跨平台泛化

## 一句话总结
提出 A₀，一个可供性感知的分层扩散模型，通过将操作任务分解为高层空间可供性理解（预测接触点和轨迹）和低层动作执行，在100万接触点数据上预训练后仅需少量任务数据微调即可跨平台(Franka/Kinova/Realman/Dobot)部署，在擦白板等复杂轨迹任务中成功率达45%。

## 研究背景与动机

**领域现状**：机器人操作方法分为模块化方法（利用视觉基础模型）和端到端VLA方法（直接生成动作）。

**现有痛点**：模块化方法缺乏对物体空间可供性的深入理解；端到端方法不理解空间位置就直接生成动作，在复杂操作（如擦白板、堆叠物品）中表现不佳。

**核心 idea**：提出体态无关的可供性表示(Embodiment-Agnostic Affordance Representation)——以物体为中心预测接触点和接触后轨迹的2D waypoints，使方法与机器人平台解耦。

## 方法详解

### 关键设计

1. **体态无关可供性表示**: 统一来自机器人数据、手-物交互(HOI)数据和自定义数据的可供性信息为 $(I, L, C, T)$ 格式——图像、语言指令、接触点、轨迹waypoints

2. **A₀扩散模型**: 基于DiT架构，输入噪声waypoints和扩散时间步，通过交叉注意力注入SigLiP视觉特征和Qwen2.5-7B文本特征。引入Position Offset Attention提取帧间运动信息

3. **两阶段训练**: 
    - 预训练：在100万PixMo-One-Point数据上学习通用物体定位能力
    - 微调：在标注轨迹数据上学习动态操作

### 损失函数
预训练: $\mathcal{L}_p = \text{MSE}(x_t^0, f_\theta(k, x_t^k, I_t, \ell))$；微调: $\mathcal{L}_s = \text{MSE}(x_{t:t+T}^0, f_\theta(k, x_{t:t+T}^k, I_{t-1:t}, \ell))$

## 实验关键数据

| 平台 | A₀成功率 | 最强基线 | 说明 |
|------|---------|---------|------|
| Franka | **62.50%** | 55.0%(OpenVLA) | 平均8个任务 |
| Kinova | **53.75%** | 42.5% | 跨平台泛化 |
| Wipe Board | **45%** | ~20% | 轨迹跟随任务 |

### 关键发现
- 预训练接触点定位能力显著提升微调后的操作性能
- 2D waypoint表示天然跨平台，仅需2D→3D投影+抓取采样即可部署到不同机器人
- 在轨迹跟随任务中优势最大，因为传统方法缺乏对后接触轨迹的建模

### 预训练数据规模

| 数据来源 | 接触点数量 | 用途 |
|---------|-----------|------|
| PixMo-One-Point | 1M | 物体定位预训练 |
| HOI数据 | 50K | 手-物交互 |
| 机器人数据 | 20K | 操作任务 |

### 跨平台部署结果

| 平台 | 任务数 | 平均成功率 | 部署方式 |
|------|--------|----------|--------|
| Franka | 8 | 62.5% | 直接部署 |
| Kinova | 6 | 53.8% | 直接部署 |
| Realman | 4 | 48.5% | 适配后部署 |
| Dobot | 3 | 45.0% | 适配后部署 |


## 亮点与洞察
- "体态无关"设计非常实用：预测物体上的2D点和轨迹，与机器人的具体构型无关，通过深度反投影和抓取采样器适配到任意平台
- 大规模接触点预训练的思路值得借鉴：用廉价的点标注数据建立强大的空间定位先验

## 局限与展望
- 依赖外部抓取采样器获取精确抓取姿态，采样器失败时无法执行任务。
- 2D到3D的深度反投影受深度估计精度限制，在透明或反光物体上可能失败。
- 擦白板等复杂轨迹任务的成功率45%仍有很大提升空间。
- 预训练用的PixMo-One-Point数据主要是静态定位，动态轨迹数据仍较稀缺。
- 2D waypoint表示无法处理需要精确力控制的任务（如组装）。
- 未探索多步操作规划和长期任务执行。
- Position Offset Attention的计算开销未详细分析。

## 评分
- 新颖性: ⭐⭐⭐⭐ 可供性分层+体态无关表示设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 4个机器人平台+多种任务
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对实际机器人部署有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](aether_geometric-aware_unified_world_modeling.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[ICCV 2025\] ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization](imagegem_in-the-wild_generative_image_interaction_dataset_for_generative_model_p.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)

</div>

<!-- RELATED:END -->
