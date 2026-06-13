---
title: >-
  [论文解读] SuperDec: 3D Scene Decomposition with Superquadric Primitives
description: >-
  [ICCV 2025][3D视觉][超二次曲面] 提出SuperDec,基于Transformer的学习方法将点云分解为紧凑的超二次曲面基元集合,在ShapeNet上训练即可泛化到真实场景,支持机器人操作和可控生成。 3D场景表示在计算机视觉和机器人领域至关重要。3D Gaussian Splatting等方法虽实现了高质量…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "超二次曲面"
  - "3D场景分解"
  - "紧凑表示"
  - "Transformer"
  - "机器人操作"
---

# SuperDec: 3D Scene Decomposition with Superquadric Primitives

**会议**: ICCV 2025  
**arXiv**: [2504.00992](https://arxiv.org/abs/2504.00992)  
**代码**: [项目页面](https://super-dec.github.io)  
**领域**: 3D视觉  
**关键词**: 超二次曲面, 3D场景分解, 紧凑表示, Transformer, 机器人操作

## 一句话总结

提出SuperDec,基于Transformer的学习方法将点云分解为紧凑的超二次曲面基元集合,在ShapeNet上训练即可泛化到真实场景,支持机器人操作和可控生成。

## 研究背景与动机

3D场景表示在计算机视觉和机器人领域至关重要。3D Gaussian Splatting等方法虽实现了高质量的光真实重建,但表示**内存密集、不紧凑**,缺乏对空间推理的显式控制。

几何基元分解提供了紧凑且可解释的替代方案,但现有方法存在问题:

**学习方法**(如SQ [Paschalidou])需要**类别特定训练**,只编码全局特征,无法泛化

**优化方法**(如EMS)假设层级几何结构,对桌椅等常见物体不适用

**场景级分解**(如DBW)受限于少量基元且优化耗时(3小时)

超二次曲面仅需11个参数(5个形状+6个位姿)即可表示丰富的形状变化,比长方体(9+6=15)更表达力强。

## 方法详解

### 超二次曲面参数化

$$f(\mathbf{x}) = \left(\left(\frac{x}{s_x}\right)^{\frac{2}{\epsilon_2}} + \left(\frac{y}{s_y}\right)^{\frac{2}{\epsilon_2}}\right)^{\frac{\epsilon_2}{\epsilon_1}} + \left(\frac{z}{s_z}\right)^{\frac{2}{\epsilon_1}} = 1$$

径向距离: $d_r = |\mathbf{x}| \cdot |1 - f(\mathbf{x})^{-\epsilon_1/2}|$

### 前馈神经网络

基于Mask2Former风格的Transformer架构:

1. **点编码器(PVCNN)**: 提取点特征 $\mathcal{F}_{PC} \in \mathbb{R}^{N \times H}$
2. **超二次曲面查询**: 正弦位置编码初始化 $\mathcal{F}_{SQ} \in \mathbb{R}^{P \times H}$
3. **Transformer解码器**: 自注意力+交叉注意力迭代细化
4. **分割头**: 预测软分配矩阵 $M \in \mathbb{R}^{N \times P}$, $m_{ij} = \sigma(\phi(\mathcal{F}_{PC}) \cdot \mathcal{F}_{SQ})$
5. **超二次曲面头**: 预测12个参数(11个形状位姿+1个存在概率)

### 损失函数

$$\mathcal{L} = \mathcal{L}_{rec} + \lambda_{par}\mathcal{L}_{par} + \lambda_{exist}\mathcal{L}_{exist}$$

重建损失 = 双向Chamfer距离 + 法向量正则:
$$\mathcal{L}_{\mathcal{P} \to SQ} = \frac{1}{N}\sum_i\sum_j m_{ij}\min_s d(\mathbf{x}_i, \mathbf{x}'_{js})$$

紧凑性损失(0.5-范数鼓励使用更少基元):
$$\mathcal{L}_{par} = \left(\frac{1}{P}\sum_j\frac{\sqrt{m_j}}{P}\right)^2$$

### Levenberg-Marquardt优化

网络输出作为初始化,LM算法进一步细化超二次曲面参数,使用加权径向距离作为残差。

### 场景级扩展

使用Mask3D提取3D实例分割掩码 → 每个物体归一化 → 独立预测超二次曲面分解。

## 实验

### ShapeNet定量对比

| 方法 | 基元类型 | L1↓(类内) | L2↓(类内) | #基元↓ | L1↓(类外) | L2↓(类外) |
|------|---------|-----------|-----------|--------|-----------|-----------|
| EMS | 超二次曲面 | 5.771 | 1.345 | 5.68 | 5.410 | 1.211 |
| CSA | 长方体 | 5.157 | 0.527 | 9.21 | 4.897 | 0.427 |
| SQ | 超二次曲面 | 3.668 | 0.279 | 10 | 4.193 | 0.354 |
| **SuperDec** | 超二次曲面 | **1.698** | **0.047** | **5.8** | **1.847** | **0.061** |

L2误差比SQ低**6倍**,基元数量减半。

### 泛化能力

在ShapeNet训练后,无需微调即可泛化到:
- ScanNet++真实室内场景
- Replica合成场景

### 关键发现

1. SuperDec的L2误差仅为现有SOTA的1/6,同时使用更少基元
2. 类外泛化能力优秀,跨类别性能下降有限
3. 场景级分解结合Mask3D后,可有效处理完整3D场景
4. 支持机器人路径规划、抓取和可控图像生成等下游应用

## 亮点与洞察

1. **从监督分割借鉴到无监督几何分割** — 将Mask2Former架构适配到基于几何的无监督分割
2. **网络+优化的两阶段设计** — 网络提供良好初始化,LM优化进一步精炼
3. **类无关训练** — 联合训练多类别,通过局部点特征实现泛化
4. **超二次曲面的实用性** — 仅11参数但表达力远超长方体

## 局限性

- 泛化依赖物体的规则几何结构,对高度不规则形状可能不理想
- 场景级分解依赖Mask3D的实例分割质量
- 最大基元数P需预设

## 相关工作

- **学习方法**: Tulsiani (长方体), Paschalidou (超二次曲面), CSA
- **优化方法**: EMS, Marching Primitives
- **场景级**: DBW (可微块世界), GES, 3D Convex

## 评分

- 新颖性: ⭐⭐⭐⭐ (Transformer+LM优化的两阶段设计)
- 技术深度: ⭐⭐⭐⭐ (损失设计精细,优化模块完整)
- 实验充分度: ⭐⭐⭐⭐ (多数据集+下游应用展示)
- 实用价值: ⭐⭐⭐⭐⭐ (紧凑表示对机器人应用价值高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Proactive Scene Decomposition and Reconstruction](proactive_scene_decomposition_and_reconstruction.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)
- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)

</div>

<!-- RELATED:END -->
