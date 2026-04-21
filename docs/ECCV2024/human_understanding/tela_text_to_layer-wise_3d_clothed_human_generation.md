---
title: >-
  [论文解读] TELA: Text to Layer-wise 3D Clothed Human Generation
description: >-
  [ECCV 2024][人体理解][文本到3D人体] 提出TELA方法，通过分层人体表示和渐进优化策略，从文本描述生成服装可分离的3D人体模型，支持虚拟试穿等服装编辑应用。
tags:
  - ECCV 2024
  - 人体理解
  - 文本到3D人体
  - 分层服装生成
  - 可编辑3D人体
  - 渐进式优化
  - 服装解纠缠
---

# TELA: Text to Layer-wise 3D Clothed Human Generation

**会议**: ECCV 2024  
**arXiv**: [2404.16748](https://arxiv.org/abs/2404.16748)  
**代码**: 无  
**领域**: Human Understanding  
**关键词**: 文本到3D人体, 分层服装生成, 可编辑3D人体, 渐进式优化, 服装解纠缠

## 一句话总结

提出TELA方法，通过分层人体表示和渐进优化策略，从文本描述生成服装可分离的3D人体模型，支持虚拟试穿等服装编辑应用。

## 研究背景与动机

从文本描述生成3D着衣人体模型是3D内容生成领域的重要方向，在游戏、虚拟试穿、社交元宇宙等场景中有广泛应用。

现有方法（如DreamFusion系列的text-to-3D方法）通常将人体和衣服作为一个**整体模型**进行建模和优化。这种整体式方法有两个限制：(1) **服装不可编辑**：生成的3D模型中，衣服和身体是不可分割的，无法单独修改衣服（如更换上衣、调整裤子）；(2) **缺乏细粒度控制**：单阶段优化将所有细节（身体形态、上装、下装、配饰等）混合在一起，难以对各个部分进行独立控制。

TELA（Text to LAyer-wise）的核心思路是**分层生成**：将3D着衣人体分解为多个独立的层——最内层是基础人体（minimal clothing），然后逐层添加衣服。每一层独立建模和优化，最终通过合成渲染融合在一起。这一设计使得服装可以自由替换和编辑。

## 方法详解

### 整体框架

TELA采用渐进式生成策略：(1) 首先从文本描述中生成基础人体模型（轻衣/内衣形态）；(2) 然后逐层生成衣服模型（如上衣、下装、外套等），每层衣服作为独立的3D模型；(3) 使用分层合成渲染将所有层融合为最终的着衣人体图像；(4) 利用分离损失确保衣服层不与身体层混淆。

### 关键设计

1. **分层着衣人体表示（Layer-wise Clothed Human Representation）**:
    - 功能：将着衣人体分解为可独立编辑的多层结构
    - 核心思路：使用隐式表示（如NeRF或SDF）分别建模身体层和各个衣服层。每一层有独立的几何和外观参数。身体层使用SMPL先验初始化，衣服层基于身体层的表面向外扩展。各层的参数独立优化，不相互干扰
    - 设计动机：整体式表示使衣服无法与身体分离，分层表示是实现服装可编辑的前提

2. **分层合成渲染（Stratified Compositional Rendering）**:
    - 功能：将多层3D模型融合为单一的2D渲染图像用于SDS优化
    - 核心思路：在体渲染过程中，按照物理上的层叠关系（从内到外：身体→内衣→上衣→外套）对各层的颜色和密度进行合成。使用改进的体渲染公式，其中外层的密度会遮挡内层的颜色，模拟真实的穿着效果。渲染结果用于计算SDS损失驱动优化
    - 设计动机：简单叠加各层渲染结果会导致非物理的透明效果；分层合成渲染保证了正确的遮挡关系

3. **服装解耦损失（Clothing Decoupling Loss）**:
    - 功能：确保衣服层仅表示衣服，不包含身体信息
    - 核心思路：设计专门的损失函数，惩罚衣服层在身体内部区域的密度值，确保衣服只出现在身体表面附近及以外。同时约束衣服的厚度在合理范围内，避免过薄或过厚的非物理结果
    - 设计动机：在SDS优化过程中，如果没有显式约束，衣服层可能会"渗透"到身体内部，导致分离失败

### 损失函数 / 训练策略

- SDS损失：使用预训练的扩散模型（如Stable Diffusion）的梯度引导每一层的优化
- 服装解耦损失：约束衣服层的空间分布
- SMPL正则化损失：约束身体层接近SMPL模型的合理形态
- 渐进式训练：先优化身体层至稳定，再逐层生成衣服，避免同时优化多层的不稳定性

## 实验关键数据

### 主实验

| 指标 | 本文 | DreamHuman | 提升 |
|------|------|----------|------|
| 3D质量 | 更好 | CLIP-score较低 | 更精细 |
| 服装分离效果 | 完全可分离 | 不可分离 | 功能性
| 虚拟试穿 | 支持 | 不支持 | 新功能 |
| 用户研究偏好 | 更高 | 基线 | 质量和功能双优 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 整体式生成 | 不可分离 | 通常SDS基线的做法 |
| 无解耦损失 | 层间渗透 | 衣服和身体混淆 |
| 无渐进优化 | 不稳定 | 多层同时优化困难 |
| 完整TELA | 最优 | 高质量+可编辑 |

### 关键发现

- 分层表示成功实现了衣服的独立生成和编辑
- 渐进式优化策略对训练稳定性至关重要
- 解耦损失有效防止了层间信息泄露
- 生成的分层模型支持虚拟试穿等下游应用

## 亮点与洞察

- 从整体到分层的范式转变是核心贡献，使3D人体生成从"看"走向"用"
- 分层合成渲染的设计考虑了物理正确的遮挡关系
- 虚拟试穿应用展示了方法的实际价值
- 渐进优化策略为多层3D生成提供了稳定的训练方案

## 局限与展望

- SDS优化的固有问题（如过饱和、多面性）仍然存在
- 层数固定，对于复杂穿搭（多层叠穿）可能需要更多层
- 衣服的物理属性（如褶皱、布料质感）建模有限
- 目前只支持静态姿态，动态人体+服装动画是重要扩展
- 可以结合扩散先验与物理仿真提升服装真实感

## 相关工作与启发

- **DreamFusion**: SDS优化的开创者，TELA在此基础上引入分层设计
- **DreamHuman / AvatarCLIP**: text-to-3D人体的代表工作，但不支持服装分离
- **SMPL / SMPL-X**: 参数化人体模型，提供身体形状先验
- 启发：3D内容生成的可编辑性和可组合性是提升其实用价值的关键方向

## 评分

- 新颖性: ⭐⭐⭐⭐ 分层着衣人体生成的想法新颖，填补了研究空白
- 实验充分度: ⭐⭐⭐ 展示了生成效果和编辑能力，但定量评估可以更丰富
- 写作质量: ⭐⭐⭐ 方法描述清晰，但论文整体可以更精炼
- 价值: ⭐⭐⭐⭐ 对虚拟试穿等应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [WordRobe: Text-Guided Generation of Textured 3D Garments](wordrobe_text-guided_generation_of_textured_3d_garments.md)
- [FreeCloth: Free-Form Generation Enhances Challenging Clothed Human Modeling](../../CVPR2025/human_understanding/freecloth_free-form_generation_enhances_challenging_clothed_human_modeling.md)
- [3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](../../CVPR2026/human_understanding/laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)

<!-- RELATED:END -->
