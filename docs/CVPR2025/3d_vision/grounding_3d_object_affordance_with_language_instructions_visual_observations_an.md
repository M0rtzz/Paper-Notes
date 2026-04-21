---
title: >-
  [论文解读] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions
description: >-
  [CVPR 2025][3D视觉][3D affordance grounding] 提出首个多模态多视角 3D 功能区域定位任务和 AGPIL 数据集（30,972 对点云-图像-语言三元组），并设计基于 VLM 的 LMAffordance3D 框架，融合 2D/3D 空间特征与语言语义实现从 full-view 到 partial/rotation-view 的泛化。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D affordance grounding
  - 多模态
  - 点云
  - VLM
  - embodied intelligence
---

# Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions

**会议**: CVPR 2025  
**arXiv**: [2504.04744](https://arxiv.org/abs/2504.04744)  
**代码**: [项目页面](https://sites.google.com/view/lmaffordance3d)  
**领域**: 3d_vision  
**关键词**: 3D affordance grounding, multi-modal fusion, point cloud, VLM, embodied intelligence

## 一句话总结

提出首个多模态多视角 3D 功能区域定位任务和 AGPIL 数据集（30,972 对点云-图像-语言三元组），并设计基于 VLM 的 LMAffordance3D 框架，融合 2D/3D 空间特征与语言语义实现从 full-view 到 partial/rotation-view 的泛化。

## 研究背景与动机

**领域现状**: 功能区域定位（Affordance Grounding）旨在识别物体可操作区域，是具身智能中连接感知与行动的关键。现有研究主要在 2D 图像或单一模态 3D 点云上进行。

**现有痛点**:
- 2D affordance 方法难以直接映射到 3D 空间用于机器人操作
- 3D 方法（如 3D AffordanceNet）仅依赖几何信息，泛化能力有限，遇到相似物体容易混淆
- 现有数据集要么只有单一模态输入，要么缺少语言指令引导，要么忽视了现实中遮挡/旋转导致的不完整点云问题

**核心矛盾**: 真实 3D 世界中，物体观测受视角、遮挡、旋转影响只能获取部分点云；而人类通过语言指导、视觉演示和交互来学习新物体的 affordance——现有方法无法同时利用这三类信息。

**本文切入角度**: 受认知科学启发，将 3D affordance grounding 建模为多模态任务（语言+图像+点云），并构建覆盖 full/partial/rotation 三种视角和 seen/unseen 两种设定的完整 benchmark。

## 方法详解

### 整体框架

LMAffordance3D 是一个端到端的单阶段框架，由四个核心组件构成：
1. **Vision Encoder**: 处理图像（ResNet18 → 2D 特征）和点云（PointNet++ → 3D 特征），通过 MLP + Self-Attention 融合得到多模态空间特征 $F_S$
2. **VLM 核心**: LLaVA-7B 作为 backbone，通过 Tokenizer 将语言指令编码为 $F_T$，Adapter（两层 MLP + 激活层）将空间特征 $F_S$ 映射到语义空间 $F_{SP}$，拼接后输入 VLM
3. **Decoder**: 基于交叉注意力，以空间特征为 Query、指令特征为 Key、语义特征为 Value，解码得到 affordance 特征 $F_A$
4. **Segmentation Head**: 上采样 + 两层线性 + BN + Sigmoid，输出 $(B, 2048, 1)$ 的逐点 affordance 概率

### 关键设计

**1. 多模态 Vision Encoder 设计**
- **功能**: 分别用 ResNet18 和 PointNet++ 提取 2D 和 3D 特征，再通过 MLP + Self-Attention 融合
- **核心思路**: RGB 图像包含颜色/场景/交互信息，点云包含形状/尺寸/几何信息，二者互补；通过共享语义空间对齐
- **设计动机**: 不直接使用 CLIP（参数大、部署困难），而是轻量化设计以适配机器人部署场景

**2. 基于 Cross-Attention 的 Decoder**
- **功能**: 将 VLM 输出拆分为指令特征和语义特征，通过交叉注意力融合空间与语义信息
- **核心思路**: 空间特征（Query）向语义特征（Value）查询，由指令特征（Key）引导注意力分配
- **设计动机**: 确保不同语言指令可以引导模型关注同一物体的不同 affordance 区域

**3. AGPIL 数据集构建**
- **功能**: 构建首个多模态多视角 3D affordance 数据集，包含 30,972 张图像、41,628 个点云、30,972 条语言指令
- **核心思路**: 点云来自 3D AffordanceNet（full/partial/rotation 三种视角），图像来自 AGD20K 和 PIAD，语言指令由 GPT-4o 结合图像生成并人工筛选
- **设计动机**: 覆盖 23 类物体、17 类 affordance，每个标注为 $(2048, 17)$ 的概率矩阵，seen/unseen 设定完整测试泛化能力

### 损失函数

$$Loss = \omega_f L_{focal} + \omega_d L_{dice}$$

- Focal Loss: 处理正负样本不平衡
- Dice Loss: 优化分割重叠区域

## 实验关键数据

### 主实验（Overall Results）

| 方法 | Full-view AUC↑ | Full-view SIM↑ | Partial AUC↑ | Rotation AUC↑ |
|---|---|---|---|---|
| 3D AffordanceNet | 0.807 | 0.483 | 0.761 | 0.595 |
| IAG | 0.849 | 0.545 | 0.809 | 0.679 |
| OpenAD | 0.858 | 0.587 | 0.815 | 0.733 |
| PointRefer | 0.877 | 0.595 | 0.821 | 0.756 |
| **Ours** | **0.890** | **0.610** | **0.848** | **0.782** |

Unseen 设定下优势更明显：Full-view AUC 0.774 vs PointRefer 0.755，MAE 0.095 vs 0.118。

### 消融实验（Per-Affordance）

- 整体 17 类 affordance 中，"stab" AUC 达 0.997（最高），"wrapping" 仅 0.689（最难）
- Full→Partial→Rotation 性能逐步下降，说明不完整点云的挑战
- Seen→Unseen 性能下降约 10-15%，但本文在 Unseen 上的优势比 Seen 更大

### 关键发现

1. 多模态融合（图像+点云+语言）显著优于单模态方法，AUC 提升 3-8%
2. 语言指令引导使模型能区分同一物体的不同功能区域
3. 在 rotation-view 场景下提升最大（AUC +2.6%），因为 VLM 的语义理解补偿了几何不确定性
4. Unseen 物体上泛化优势明显，证明多模态融合提升了知识迁移能力

## 亮点与洞察

- 首个同时利用语言指令、视觉观察和交互的 3D affordance grounding 任务定义
- AGPIL 数据集填补了多模态+多视角+概率标注的空白
- 将 VLM 引入 3D affordance 任务的范式值得关注：VLM 的先验知识大幅提升 unseen 泛化
- 端到端单阶段设计（不需要 2D 检测框），扩展性更好

## 局限与展望

- 图像与点云来自不同场景（基于类别匹配），存在视觉—几何不一致
- LLaVA-7B 推理开销大，不利于实际机器人部署
- Rotation-view 仍是性能短板，可考虑旋转等变网络增强
- 仅支持静态物体 affordance，未考虑动态场景
- 语言指令粒度为短语级，未探索更复杂的指令理解

## 相关工作与启发

- 3D AffordanceNet 开创了 3D affordance 数据集，本文在其基础上扩展多模态和多视角
- AffordanceLLM 验证了 VLM 在 2D affordance 的有效性，本文首次将其扩展到 3D
- 启发：VLM 的视觉-语义对齐能力可作为跨模态桥梁，未来可探索更多 3D 理解任务

## 评分

⭐⭐⭐⭐ — 任务定义新颖且数据集构建扎实，技术方案合理但创新度中等（主要是模块组合），多视角和 unseen 设定的完整性值得认可。

<!-- RELATED:START -->

## 相关论文

- [GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding](great_geometry-intention_collaborative_inference_for_open-vocabulary_3d_object_a.md)
- [HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)

<!-- RELATED:END -->
