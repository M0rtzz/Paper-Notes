---
title: >-
  [论文解读] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation
description: >-
  [ICCV 2025][图像生成][机器人操控] 提出 A0，一个层次化可供性感知扩散模型，通过预测以物体为中心的接触点和后接触轨迹（Embodiment-Agnostic Affordance Representation），将操控任务分解为高层空间理解和低层动作执行，在 100 万接触点数据上预训练后可跨 Franka/Kinova/Realman/Dobot 四种平台泛化。
tags:
  - ICCV 2025
  - 图像生成
  - 机器人操控
  - 空间可供性
  - 层次化模型
  - 扩散模型
  - 跨平台泛化
---

# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: [https://a-embodied.github.io/A0/](https://a-embodied.github.io/A0/)  
**领域**: 图像生成  
**关键词**: 机器人操控, 空间可供性, 层次化模型, 扩散模型, 跨平台泛化

## 一句话总结

提出 A0，一个层次化可供性感知扩散模型，通过预测以物体为中心的接触点和后接触轨迹（Embodiment-Agnostic Affordance Representation），将操控任务分解为高层空间理解和低层动作执行，在 100 万接触点数据上预训练后可跨 Franka/Kinova/Realman/Dobot 四种平台泛化。

## 研究背景与动机

**为什么现有方法在空间可供性上不足？** 机器人操控的核心挑战在于理解物体的"哪里"（where）可以交互和"如何"（how）交互——即空间可供性。现有方法分两大类：

**模块化方法**（MOKA、ReKep）：利用大视觉模型进行空间理解，但缺乏对物体空间和物理属性的深度理解，特别是无法捕捉物体的**可操作性**

**端到端 VLA 方法**（π0、RDT）：直接生成动作序列，但不充分理解空间位置，导致在复杂任务（如擦白板、堆叠物体）上表现不佳

**为什么以物体为中心？** 现有可供性方法通常出两类表示：热力图（Heatmap）或密集点流（flow），计算开销大且与具体机器人形态耦合。以物体为中心的接触点 + 轨迹表示天然是**形态无关的**，只需预测物体上的关键点，不依赖特定机器人的运动学。

**为什么需要层次化？** 直接从视觉到动作的端到端映射过于困难。将任务分解为"理解在哪里怎么操作"和"实际执行操作"两个层次，每层的学习目标更简单、更可迁移。

## 方法详解

### 整体框架

A0 将机器人操控任务分解为两个层次：

1. **高层空间可供性理解**：预测物体的接触点和后接触轨迹（A0 模型的核心）
2. **低层动作执行**：将 2D 预测投影到 3D 空间，估计抓取姿态，执行运动

### 关键设计

#### Embodiment-Agnostic Affordance Representation

统一表示定义为：$R = R_R \cup R_H \cup R_C = \{(I, L, C, T) | C = (c^{2D}_0), T = (t^{2D}_0, t^{2D}_1, t^{2D}_2, \cdots)\}$

- $I$：以物体为中心的 RGB 图像
- $L$：自然语言操控指令
- $C$：接触点（2D 坐标）
- $T$：后接触轨迹（2D 关键路径点序列）

**为什么要统一不同数据源？** 可供性知识分布在多种数据中：真实机器人数据 $R_R$（精确但稀少）、手-物交互数据 $R_H$（丰富的交互知识）、自定义/仿真数据 $R_C$（可控但有 sim-to-real gap）。统一表示将它们融合为相同格式。

数据集组成：
- **PixMo-One-Point**：100 万单接触点标注（互联网图像）
- **HOI4D-22k**：22,000 条人-物交互轨迹
- **DROID-3k**：3,056 条真实机器人操控轨迹
- **ManiSkill-5k**：4,965 条仿真轨迹

#### A0 模型结构

基于 Diffusion Transformer（DiT）架构，包含 N=28 层，1B 参数量：

**输入**：扩散时间步 $k$ + 噪声路径点 $x^k_{t:t+T}$
**条件**：观测图像 $I_{t-1:t}$（当前帧 + 前一帧）+ 语言指令 $\ell$

路径点定义为可供性表示：$x_{t:t+T}$，其中 $x_t = (u, v) \in [0,1]^2$，$T=5$ 为 chunk size。

**Position Offset Attention（POA）**：
为什么需要运动信息？物体在帧间的运动对于理解操控进展至关重要。POA 将相邻帧的视觉 token 相减得到运动 token $I^i_m = I^i_t - I^i_{t-1}$，再与当前帧拼接：$o_t = \text{concat}([I^i_t, I^i_m], \text{dim}=1)$。

**Spatial Information Aggregation Layer（SIAL）**：
最后一层非线性 MLP 解码器，将潜空间映射回物理坐标空间。**为什么需要额外的投影层？** DiT 的输出在潜空间中，直接解码可能无法精确映射到像素坐标，SIAL 提供了从潜空间到物理空间的精确坐标变换。

**编码器**：
- 视觉编码器：预训练 SigLiP（400M）
- 语言编码器：预训练 Qwen2.5-7B
- 图像和文本 token 通过交替注入的 cross-attention 机制条件化扩散过程

#### 动作执行模块

1. **2D-to-3D 投影**：$X_i = D(x_i) K^{-1} \tilde{x}_i$，利用深度图和相机内参矩阵
2. **抓取姿态估计**：查询 GraspNet 生成候选，选择最近接触点的候选：$G^* = \arg\min_{G \in \mathcal{G}} \|G - X_t\|$
3. **路径点选择执行**：利用 VLM 选择高度类别，在 SE(3) 空间生成平滑轨迹

### 损失函数 / 训练策略

**预训练阶段**（80K 步，5天，4×A100）：
仅使用单帧图像和第一个路径点（接触点），MSE 损失：
$$L_p(\theta) = \frac{1}{n}\sum_{i=1}^n ((x^0_t)_i - (f_\theta(k, x^k_t, I_t, \ell))_i)^2$$

**监督微调阶段**（30K 步，50小时）：
扩展到 T 个路径点，加入运动信息，添加前向扩散噪声后预测原始路径点：
$$L_s(\theta) = \frac{1}{n}\sum_{i=1}^n ((x^0_{t:t+T})_i - (f_\theta(k, x^k_{t:t+T}, I_{t-1:t}, \ell))_i)^2$$

推理时使用快速 ODE 求解器，仅需 $K_D = 5$ 步去噪（vs 训练时 $K_F = 1000$ 步）。

## 实验关键数据

### 主实验

**多平台真实世界性能对比（Table 2，每任务 20 次试验）：**

| 机器人 | 方法 | Place Object | Open Drawer | Press Button | Wipe Board | 平均成功率 |
|-------|------|-------------|-------------|-------------|------------|----------|
| Kinova | MOKA | 70 | 50 | 30 | 30 | 45.00 |
| Kinova | ReKep | 75 | 55 | 5 | 0 | 33.75 |
| Kinova | **A0-1B** | 60 | 65 | 40 | **50** | **53.75** |
| Franka | Magma | 25 | 10 | 30 | 0 | 16.25 |
| Franka | Molmo | 60 | 40 | 55 | 20 | 43.75 |
| Franka | **A0-1B** | 60 | **75** | **70** | **45** | **62.50** |

**vs VLA 方法（Table 3，Kinova 平台）：**

| 方法 | Place Object | Open Drawer | Press Button | Wipe Board | 平均 | 步数 |
|------|-------------|-------------|-------------|------------|------|------|
| RDT-1B | 20 | 0 | 25 | 0 | 11.25 | 25-50 |
| π0 | 40 | 20 | 10 | 10 | 20.00 | 25-50 |
| π0 + FAST | 35 | 10 | 30 | 0 | 18.75 | 25-50 |
| **A0-1B** | **60** | **65** | **40** | **50** | **53.75** | **4-5** |

在 Wipe Board 任务上，A0 的成功率比 π0 高出 **40 个百分点**，执行步数仅需 4-5 步（vs 25-50 步）。

### 消融实验

**网络结构消融（Table 1，预训练后的 MAE↓）：**

| 配置 | HOI4D-22k | Maniskill-5k | DROID-3k |
|------|-----------|-------------|---------|
| A0-1B | 47.5 | 5.5 | 17.5 |
| w/o POA | 47.9 | 6.3 (+0.8) | 18.5 |
| w/o SIAL | 61.1 (+13.6) | 10.2 (+4.7) | 19.6 |

SIAL 的影响最为显著：移除后 HOI4D 上 MAE 增加 13.6 像素，说明从潜空间到坐标空间的精确映射是不可或缺的。

**预训练效果（Figure 4）：**

| 迁移范式 | 数据集 | 无预训练 MAE | 有预训练 MAE | 减少 |
|---------|--------|------------|------------|------|
| Real-to-Sim | Maniskill-5k | 50.4 | 43.9 | -13% |
| Sim-to-Real | HOI4D-22k | 172.2 | 35.1 | -80% |
| Sim-to-Real | DROID-3k | 125.2 | 29.1 | -77% |

预训练在 Sim-to-Real 场景中的效果尤其显著，MAE 降低 77-80%。

### 关键发现

1. **层次化 > 端到端**：A0 的平均成功率比 π0 高 33.75%
2. **单次推理 vs 多步推理**：A0 仅需 4-5 个关键路径点，VLA 需要 25-50 步
3. **轨迹任务优势突出**：Wipe Board 等需要精确轨迹跟随的任务上优势最大
4. **预训练是关键**：100 万接触点定位预训练显著提升下游性能

## 亮点与洞察

- **以物体为中心的极简可供性表示**：仅预测接触点 + 后续轨迹点，大幅降低复杂度
- **形态无关性的实际验证**：在 4 种不同机器人上验证，这是很有说服力的
- **预训练 → 微调 范式在机器人中的成功应用**：100 万互联网接触点数据为定位能力打基础
- **效率优势**：单次推理 4-5 步 vs 25-50 步，对实际部署非常重要
- **数据融合策略**：将互联网数据、HOI 数据和机器人数据统一到同一表示空间

## 局限与展望

- Place Object 任务在 Kinova 上不如 MOKA 和 ReKep，可能因为后两者使用了 SAM/GPT-4 等见过更多真实物体的大模型
- 长时序规划依赖外部 VLM 进行任务分解，非端到端
- 不支持方向敏感的精细操控（需要额外的 VLM 提示）
- 仅验证了 4 种简单的家务任务，未涉及更复杂的装配或工具使用场景
- 深度图质量对 2D-to-3D 投影有很大影响，但文中未讨论鲁棒性

## 相关工作与启发

与 Helix（层次化强化学习）的区别在于：A0 使用显式的空间可供性表示，而 Helix 学习隐式的语义表示。与 MOKA/ReKep（直接利用大视觉模型）的区别在于：A0 通过预训练获得了更深层的空间理解能力。这一工作启发我们：**将 "理解" 和 "执行" 解耦，配合大规模预训练，是嵌入式 AI 的有效范式**。

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Equivariant Flow Matching](ec-flow_enabling_versatile_robotic_manipulation_from_action-unlabeled_videos_via.md)
- [\[ICCV 2025\] CHORDS: Diffusion Sampling Accelerator with Multi-Core Hierarchical ODE Solvers](chords_diffusion_sampling_accelerator_with_multi-core_hierarchical_ode_solvers.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](aether_geometric-aware_unified_world_modeling.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[ICCV 2025\] ImageGem: In-the-wild Generative Image Interaction Dataset for Generative Model Personalization](imagegem_in-the-wild_generative_image_interaction_dataset_for_generative_model_p.md)

</div>

<!-- RELATED:END -->
