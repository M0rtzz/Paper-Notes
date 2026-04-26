---
title: >-
  [论文解读] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement
description: >-
  [3D视觉] NeuraLeaf 将叶片的 3D 几何解耦为 2D 基础形状和 3D 变形两个潜在空间，利用大量 2D 叶片图像数据集学习形状空间，提出无骨架蒙皮模型处理叶片的高度柔性变形，并构建了首个专注叶片变形建模的 3D 数据集 DeformLeaf。
tags:
  - 3D视觉
---

# NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2507.12714](https://arxiv.org/abs/2507.12714)
- **代码**: [项目页面](https://neuraleaf-yang.github.io/)
- **领域**: 3D视觉
- **关键词**: 神经参数化模型, 叶片建模, 形状与变形解耦, 无骨架蒙皮, 3D重建

## 一句话总结
NeuraLeaf 将叶片的 3D 几何解耦为 2D 基础形状和 3D 变形两个潜在空间，利用大量 2D 叶片图像数据集学习形状空间，提出无骨架蒙皮模型处理叶片的高度柔性变形，并构建了首个专注叶片变形建模的 3D 数据集 DeformLeaf。

## 研究背景与动机

3D 叶片建模和重建在农业和计算机图形学中至关重要。现有方法面临的挑战：

**3D 数据稀缺**：神经参数化模型（NPM）在人体建模（如 SMPL）中表现出色，但需要大量 3D 数据训练，而叶片的 3D 变形数据集几乎不存在

**形状多样性高**：不同物种的叶片形状差异极大，难以用传统 PCA 模型或 Bezier 曲线统一建模

**变形自由度高**：叶片变形非常灵活，不同于人体的关节结构，难以定义通用骨架

**现有参数化模型受限**：PCA 模型（Barley）表达力有限；Bezier 曲线模型（Gaurav）无法捕获自然变形

关键洞察：**展平的叶片可近似为 2D 平面**，这一生物学特性可被利用来解耦形状和变形。

## 方法详解

### 整体框架

NeuraLeaf 包含三个解耦的空间：
- **形状空间**：基于 2D SDF 表示叶片展平形状（从大规模 2D 数据集学习）
- **纹理空间**：通过 CycleGAN 生成与形状对齐的纹理
- **变形空间**：无骨架蒙皮模型表示 3D 变形（从 DeformLeaf 数据集学习）

### 关键设计一：2D 基础形状空间

利用神经 SDF 条件化于形状潜在码 $\mathbf{z}_s$ 表示展平叶片：

$$\mathcal{S}_b^* = \{\mathbf{x} \in \mathbb{R}^2 \mid f_{\theta_s}(\mathbf{x}, \mathbf{z}_s) = 0\}$$

通过可微 sigmoid 转换 SDF 为掩码 $\mathbf{I}_m$，阈值化后构建基础网格。训练使用 2D 叶片扫描数据集，损失包括：

$$\text{argmin}_{\theta_s, k, \{\mathbf{z}_s^i\}} \sum_i (\mathcal{L}_{\text{sdf}} + \mathcal{L}_{\text{sil}} + \mathcal{L}_{\text{eik}} + \mathcal{L}_{\text{lat}})$$

优势：2D 表示可从丰富的生物学 2D 叶片扫描数据集（如 Flavia）学习。

### 关键设计二：无骨架蒙皮模型

与 SMPL 等基于骨架的蒙皮不同，NeuraLeaf 使用**大量控制点**（$K=1000$）的无骨架蒙皮：

$$\tilde{\mathbf{v}}_i = \sum_{k=1}^K w_{i,k} \mathbf{T}_k (\tilde{\mathbf{v}}_i' - \tilde{\mathbf{c}}_k)$$

解耦为两个解码器：
- **变换解码器** $f_{\theta_d}$：仅从变形潜在码 $\mathbf{z}_d$ 预测每个控制点的刚体变换（四元数），与形状无关
- **蒙皮权重解码器** $f_{\theta_w}$：从形状潜在码 $\mathbf{z}_s$ 和顶点位置预测蒙皮权重，适应每种叶片形状

### 关键设计三：变形映射损失

约束变形潜在空间的分布——变形量越大的样本，潜在码距零点越远：

$$\mathcal{L}_{map} = \left(\frac{d_{cham}(\mathcal{S}_b, \mathcal{S}_d)}{\|\mathbf{z}_d\|_2} - \phi\right)^2$$

其中 $\phi$ 为可学习缩放因子。这确保了潜在空间的有序结构。

### DeformLeaf 数据集

约 300 对基础-变形叶片对，通过多视角重建获取 3D 变形形状，结合展平拍摄获取 2D 基础形状。包含叶脉等细节几何特征，是首个专注叶片变形建模的 3D 数据集。

## 实验

### 主实验：3D 叶片重建（拟合到观测）

| 方法 | Chamfer Distance↓ | 表面质量 |
|------|-----------------|---------|
| PCA 参数化模型 | 较高 | 缺细节 |
| DeepSDF（NPM 基线） | 中等 | 缺乏叶脉 |
| **NeuraLeaf** | **最低** | **叶脉清晰** |

NeuraLeaf 在重建精度上显著优于传统参数化模型和通用 NPM，且保留叶脉等表面细节。

### 消融实验：解耦设计的贡献

| 变形映射损失 | 两阶段训练 | 边界约束 | 重建质量 |
|-----------|---------|---------|---------|
| ✗ | ✗ | ✗ | 基线 |
| ✓ | ✗ | ✗ | 改善 |
| ✓ | ✓ | ✗ | 进一步改善 |
| ✓ | ✓ | ✓ | **最优** |

变形映射损失对潜在空间的有序化至关重要。

### 关键发现
- 解耦形状和变形使得仅用约 300 个 3D 样本即可学到有效的变形空间
- 1000 个控制点远多于人体模型的骨架数量，但对叶片的精细变形至关重要
- 变形空间可泛化到训练中未见的叶片形状（通过第二阶段训练）
- 纹理空间通过 CycleGAN 自然产生与形状对齐的纹理

## 亮点与洞察

1. **巧妙利用领域先验**：展平叶片≈2D 平面这一性质实现解耦
2. **无骨架蒙皮**：摆脱传统骨架定义的限制，适应叶片的高度灵活变形
3. **跨数据集学习**：形状空间从丰富的 2D 数据集、变形空间从小型 3D 数据集分别学习
4. **首个叶片变形数据集**：DeformLeaf 为该领域提供基础研究资源

## 局限性
- DeformLeaf 数据集规模（约300对）相对较小
- 无法建模叶片的动态变形过程（如脱水卷曲）
- 变形映射假设零均值高斯分布，可能不适合极端变形
- 目前仅支持单片叶片建模，未考虑整株植物的上下文

## 相关工作
- SMPL / SMAL：人体/动物参数化模型
- DeepSDF：神经隐式形状表示
- PhysGaussian / Leaf deformation modeling：物理驱动叶片变形
- Bio-inspired leaf modeling：基于叶脉模式的叶片生成

## 评分
- **创新性**: ⭐⭐⭐⭐⭐ — 首个叶片 NPM，解耦设计和无骨架蒙皮均为新颖贡献
- **实用性**: ⭐⭐⭐⭐ — 农业和 CG 应用价值高
- **实验完整度**: ⭐⭐⭐⭐ — 拟合/生成/消融实验完整
- **写作质量**: ⭐⭐⭐⭐ — 方法清晰，数据集有价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](../../CVPR2025/3d_vision/dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)

<!-- RELATED:END -->
