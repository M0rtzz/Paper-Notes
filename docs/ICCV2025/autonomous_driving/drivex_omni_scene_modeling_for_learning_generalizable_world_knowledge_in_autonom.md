---
title: >-
  [论文解读] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][世界模型] 提出 DriveX，一个自监督世界模型框架，通过 Omni Scene Modeling（联合3D点云预测、2D语义表示和图像生成）在 BEV 潜在空间学习可迁移的通用场景表征，并设计 Future Spatial Attention 范式将预测的未来状态无缝集成到占据预测、流估计和端到端驾驶等多种下游任务中，在多个任务上达到 SOTA。
tags:
  - ICCV 2025
  - 自动驾驶
  - 世界模型
  - 自监督学习
  - BEV表示
  - 点云预测
  - 端到端驾驶
---

# DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2505.19239](https://arxiv.org/abs/2505.19239)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 世界模型, 自监督学习, BEV表示, 点云预测, 端到端驾驶

## 一句话总结

提出 DriveX，一个自监督世界模型框架，通过 Omni Scene Modeling（联合3D点云预测、2D语义表示和图像生成）在 BEV 潜在空间学习可迁移的通用场景表征，并设计 Future Spatial Attention 范式将预测的未来状态无缝集成到占据预测、流估计和端到端驾驶等多种下游任务中，在多个任务上达到 SOTA。

## 研究背景与动机

### 问题定义

世界模型通过历史观测和自车动作预测未来环境状态，被视为提升自动驾驶安全性和鲁棒性的关键方向。核心问题是：**如何构建一个能提取通用世界表征、同时受益多种下游驾驶任务的统一世界模型框架？**

### 已有方法的不足

现有世界模型方法可分为三类：

**预训练策略**（如 ViDAR）：将世界建模作为预训练再微调，但微调时通用知识被破坏

**辅助监督策略**：将世界建模作为辅助 head，同样无法保留通用世界知识

**视频生成策略**：利用视频生成技术预测未来帧，但缺乏3D空间信息和时空一致性

核心矛盾在于：之前的方法要么只关注几何线索（点云预测）要么依赖昂贵的语义标注（占据预测），无法同时实现可扩展训练和全面场景理解。

### 核心 idea

将世界表示学习与未来状态解码解耦，通过多模态自监督信号（深度+语义+颜色）在 BEV 潜在空间编码丰富的世界知识，使得模型参数冻结时仍能保持强大的任务迁移能力。

## 方法详解

### 整体框架

DriveX 包含两阶段训练：
1. **世界表示学习**：训练 World Encoder 将多视角图像编码为 BEV 特征，通过 OSM 模块进行多模态监督
2. **潜在未来解码**：冻结 World Encoder，训练 Future Decoder 在 BEV 潜在空间预测未来状态

下游应用通过 Future Spatial Attention（FSA）范式，利用任务特定查询从预测的未来 BEV 特征中聚合信息。

### 关键设计

#### 1. Omni Scene Modeling (OSM)

- **功能**：统一多模态监督信号来训练 World Encoder
- **核心思路**：从 BEV 特征 $B_t$ 通过 Channel-to-Height 模块转换为体素特征 $F_t$，沿射线采样路径点，利用体积渲染预测深度、语义和颜色：
  $$\alpha_i = \sum_{j=1}^{n} \tau_j (1-\exp(-\sigma_{i,j}\delta_j))\alpha_{i,j}$$
  相机视角射线优化语义+颜色损失，LiDAR 视角射线优化深度损失
- **设计动机**：单一监督信号（如仅深度）无法捕获完整场景信息；语义标签来自基础模型（Grounded SAM、OpenSeeD），实现全自监督

#### 2. 解耦潜在世界建模

- **功能**：将表示学习和时序动态建模分开训练
- **核心思路**：
    - 先训练 World Encoder（40 epochs），获得高质量 BEV 表征
    - 再训练 Future Decoder（24 epochs），在冻结的 BEV 空间建模时序演化
    - Future Decoder 通过 Flow-based 策略直接预测未来状态（非自回归），避免误差累积：
    $$g_f' = T_t^{t+k} g_f$$
    - 预测的 BEV 特征通过距离加权插值得到网格化表示
- **设计动机**：联合训练中，表示学习和动态建模相互干扰；解耦后 Chamfer Distance 降低 0.44m²

#### 3. Dynamic-aware Ray Sampling

- **功能**：在未来时间步的场景建模中，优先采样运动显著区域的射线
- **核心思路**：使用离线跟踪器识别运动物体的 RoI，在这些区域额外采样射线
- **设计动机**：大部分射线属于静态背景，动态物体容易被忽略

#### 4. Future Spatial Attention (FSA)

- **功能**：统一范式将世界模型预测集成到下游任务
- **核心思路**：任务查询 $q$ 通过空间注意力从多个未来时间步的 BEV 特征聚合信息：
  $$q := q + \sum_{k=1}^{K}\sum_{j=1}^{J} A_{kj} W \hat{B}_{t+k}[T_t^{t+k}(p + \Delta p_{kj})]$$
- **设计动机**：不同任务可通过调整采样偏移 $\Delta p_{kj}$ 来适配，无需修改已有架构

### 损失函数 / 训练策略

- 表示学习：$\mathcal{L}_{scene}^{camera} = \mathcal{L}_{sem} + \mathcal{L}_{rgb}$，$\mathcal{L}_{scene}^{LiDAR} = \mathcal{L}_{depth}$
- 未来解码：$\mathcal{L}_{future} = \sum_{k=1}^{F}(\omega_l \mathcal{L}_{latent}^k + \omega_s \mathcal{L}_{scene}^k)$，其中 $\omega_l=1.0, \omega_s=0.5$
- 两阶段均为全自监督

## 实验关键数据

### 主实验

**点云预测（nuScenes, Chamfer Distance m² ↓）**

| 方法 | 模态 | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s |
|------|------|------|------|------|------|------|------|
| 4D-Occ | L | 0.91 | 1.13 | 1.30 | 1.53 | 1.72 | 2.11 |
| ViDAR | C | 1.01 | 1.12 | 1.25 | 1.38 | 1.54 | 1.73 |
| HERMES | C | - | 0.78 | - | 0.95 | - | 1.17 |
| **DriveX-B** | C | **0.55** | **0.66** | **0.75** | **0.86** | **0.97** | **1.10** |

**端到端驾驶（NAVSIM test, PDMS ↑）**

| 方法 | NC↑ | DAC↑ | TTC↑ | EP↑ | PDMS↑ |
|------|-----|------|------|-----|-------|
| TransFuser | 97.4 | 92.8 | 92.4 | 79.0 | 83.8 |
| PARA-Drive | 97.9 | 92.4 | 93.0 | 79.3 | 84.0 |
| **DriveX-S** | 97.5 | **94.0** | **93.0** | **79.7** | **84.5** |

### 消融实验

**OSM 多模态监督组件消融**

| 深度 | 语义 | 颜色 | mIoU | IoU_geo | mAVE↓ |
|------|------|------|------|---------|-------|
| ✓ | | | 3.96 | 59.7 | 1.388 |
| ✓ | ✓ | | 42.53 | 73.1 | 0.396 |
| ✓ | ✓ | ✓ | **43.47** | **73.44** | **0.385** |

**解耦训练策略消融**

| 训练方式 | 1.0s CD↓ | 2.0s CD↓ | 3.0s CD↓ | Avg↓ |
|----------|----------|----------|----------|------|
| 联合训练 | 1.09 | 1.41 | 1.75 | 1.41 |
| 解耦训练 | 0.83 | 1.03 | 1.31 | 1.06 |
| 解耦+动态采样 | **0.80** | **0.99** | **1.28** | **1.02** |

### 关键发现

- DriveX 冻结参数后在 NAVSIM 上仅下降 0.1 PDMS，而 ViDAR 下降 6.6，证明学到了真正通用的世界知识
- 直接预测优于自回归（Avg CD: 1.02 vs 1.32），避免了误差累积
- 数据从 50% 扩展到 100% 仍有 0.51m² 的显著提升，展现良好的 scaling 特性
- 占据预测上 mIoU 提升 0.93，mAVE 降低 0.027，证明 FSA 有效传递了预测信息

## 亮点与洞察

1. **参数冻结测试是验证通用表征的关键指标**：Table 1 的实验设计非常有说服力——冻结世界模型参数后性能几乎无损，说明 BEV 特征已编码了任务无关的通用世界知识
2. **多模态自监督的互补性**：仅深度监督 mIoU 只有 3.96（几乎没有语义信息），加入语义后跃升至 42.53，颜色再提升 0.94，三种信号缺一不可
3. **解耦训练的简洁高效**：相比联合训练使用相同总时间，解耦方案效果显著更好，是一个实用且可推广的设计策略
4. **FSA 范式的统一性**：同一个世界模型可同时服务占据预测、流估计和规划，仅需调整采样偏移

## 局限与展望

1. **仅验证了 nuScenes 和 NAVSIM**：未在更大规模数据集（如 Waymo、Argoverse）上验证
2. **推理延迟增加约 42-48ms**：对实时性要求极高的场景仍有优化空间
3. **语义标签依赖基础模型质量**：Grounded SAM 和 OpenSeeD 的错误会传播到训练过程
4. **未探索多模态输入**：当前仅用相机+LiDAR，未考虑雷达等其他传感器

## 相关工作与启发

- ViDAR 开创了纯视觉点云预测的预训练范式，但其知识在微调时容易被覆盖
- HERMES 是之前的点云预测 SOTA，DriveX 在所有时间步上都大幅超越
- DiffusionDrive 在端到端驾驶上表现出色，DriveX 在其基础上通过 FSA 进一步提升

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 多模态自监督+解耦训练+FSA的组合具有创新性，但各组件单独看不算全新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 点云预测+占据预测+流估计+端到端驾驶四个任务全面验证，消融充分
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法和实验组织有条理
- **价值**: ⭐⭐⭐⭐⭐ — 为自动驾驶世界模型提供了统一框架，参数冻结实验和 scaling 实验都有很强的实用价值

<!-- RELATED:START -->

## 相关论文

- [Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [Passing the Driving Knowledge Test](passing_the_driving_knowledge_test.md)
- [OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](../../ECCV2024/autonomous_driving/occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/prioritizing_perception-guided_self-supervision_a_new_paradigm_for_causal_modeli.md)

<!-- RELATED:END -->
