---
title: >-
  [论文解读] ARM: Appearance Reconstruction Model for Relightable 3D Generation
description: >-
  [CVPR 2025][3D视觉][3D重建] 提出ARM框架，将几何和外观生成解耦，在UV纹理空间中通过反投影和全局感受野网络重建高质量纹理，并引入材质先验解决稀疏视角下材质与光照的歧义性，仅用8张H100训练即在GSO和OmniObject3D上超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D重建
  - 外观分解
  - PBR材质
  - UV纹理空间
  - 重光照
---

# ARM: Appearance Reconstruction Model for Relightable 3D Generation

**会议**: CVPR 2025  
**arXiv**: [2411.10825](https://arxiv.org/abs/2411.10825)  
**代码**: https://arm-aigc.github.io (有)  
**领域**: 3D视觉 / 3D生成  
**关键词**: 3D重建, 外观分解, PBR材质, UV纹理空间, 重光照

## 一句话总结
提出ARM框架，将几何和外观生成解耦，在UV纹理空间中通过反投影和全局感受野网络重建高质量纹理，并引入材质先验解决稀疏视角下材质与光照的歧义性，仅用8张H100训练即在GSO和OmniObject3D上超越现有方法。

## 研究背景与动机
从2D图像生成高质量3D模型（带真实外观）是计算机视觉和图形学的核心任务。现有方法在几何重建方面已取得显著进展，但外观质量仍然不足——基于LRM的方法使用triplane表示，受限于分辨率和MLP解码的模糊性，重建纹理缺少细节；而大多数方法仅输出不含物理属性的顶点颜色（baked-in），无法支持动态光照下的重光照需求。核心矛盾在于：triplane的空间变化与物体表面纹理变化不直接对应，且稀疏视角下材质与光照的分离本质上是病态问题。本文的切入角度是：将外观处理搬到UV纹理空间——直接在物体表面上学习纹理，绕开triplane分辨率瓶颈；同时引入材质先验帮助分解光照和材质。

## 方法详解
ARM将3D重建任务分为几何阶段和外观阶段，几何由GeoRM生成mesh，外观由InstantAlbedo（漫反射）和GlossyRM（光泽度/金属度）分别处理。

### 整体框架
输入为扩散模型生成的稀疏多视角图像（6个视角）。GeoRM基于transformer的triplane合成器预测密度场，用可微Marching Cubes提取mesh。Mesh解包到UV空间后，InstantAlbedo将多视角图像反投影到UV纹理空间，通过U-Net提取逐视角特征，max-pooling融合后用FFCNet填补未见区域，最终输出baked颜色和分解的漫反射albedo。GlossyRM则在mesh顶点上查询triplane预测粗糙度和金属度。

### 关键设计
1. **几何-外观解耦 (GeoRM + GlossyRM)**:
    - 功能：分离几何和外观的生成，各自用专用网络处理
    - 核心思路：GeoRM专注于密度预测（用mask/depth/normal监督），训练完后冻结权重；GlossyRM以GeoRM的mesh为条件，查询自己的triplane预测逐顶点粗糙度 $\rho$ 和金属度 $m$。两者共享LRM架构但独立训练
    - 设计动机：单一LRM同时预测所有目标（密度+颜色+材质）会导致显著质量下降，特别是材质参数更难推断。解耦后每个模型容量更充裕，还允许将triplane分辨率提升到 $256 \times 256$

2. **UV纹理空间外观分解 (InstantAlbedo)**:
    - 功能：在UV空间重建高质量漫反射albedo纹理
    - 核心思路：将6个视角的图像、辅助数据（mask、位置、纹理坐标、视线方向、法线）和材质编码反投影到UV纹理空间，得到6组UV-space input maps。U-Net提取逐视角特征后max-pooling融合，再用FFCNet（具有全局感受野）填补未见区域并精炼，输出baked颜色和分解albedo
    - 设计动机：triplane空间的颜色变化不直接对应表面纹理变化，导致MLP解码模糊。UV空间直接表示表面颜色变化，绕开了分辨率和插值失配问题。FFCNet的全局感受野对仅6视角的补全至关重要

3. **材质先验 (Material Prior)**:
    - 功能：解决稀疏视角下材质与光照的固有歧义
    - 核心思路：基于DINO ViT-8×8的图像编码器，在语义材质数据集上预训练后集成到InstantAlbedo的反投影流程中。将输入图像转换为材质感知特征图，与其他辅助信息一起反投影到UV空间，帮助网络区分光照效果和材质属性
    - 设计动机：仅靠渲染损失做逆渲染在稀疏视角下必然失败——光照效果会bake进albedo。材质先验提供语义层面的"什么看起来像某种材质"的信息，即使有强光照也能正确分解

### 损失函数 / 训练策略
- GeoRM：$\mathcal{L}_{geo} = \lambda_z |z^{gt} - \hat{z}| + \lambda_M \mathcal{L}_{mse}(M^{gt}, \hat{M}) + \lambda_n \mathcal{L}_{lpips}(\mathbf{n}^{gt}, \hat{\mathbf{n}})$
- GlossyRM：$\mathcal{L}_{glossy} = \mathcal{L}_0(\rho^{gt}, \hat{\rho}) + \mathcal{L}_0(m^{gt}, \hat{m})$，其中 $\mathcal{L}_0 = \lambda_1 \mathcal{L}_{mse} + \lambda_2 \mathcal{L}_{lpips} + \lambda_3 \mathcal{L}_{ssim}$
- InstantAlbedo：$\mathcal{L}_{albedo} = \mathcal{L}_0(\mathbf{c}^{gt}, \hat{\mathbf{c}}) + \mathcal{L}_0(\mathbf{c_d}^{gt}, \hat{\mathbf{c_d}})$，直接拟合GT材质而非渲染损失
- 训练在8张H100上约5天：GeoRM 2天，GlossyRM 2天，InstantAlbedo 1天（与GlossyRM可并行）

## 实验关键数据

### 主实验

| 数据集 | 指标 | ARM (本文) | MeshFormer | InstantMesh | SF3D | 提升 |
|--------|------|-----------|------------|-------------|------|------|
| GSO (1030 shapes) | F-Score↑ | 0.968 | 0.966 | 0.938 | 0.888 | SOTA |
| GSO | PSNR↑ | 21.692 | 20.500 | 19.744 | 18.540 | +1.19dB |
| GSO | LPIPS↓ | 0.137 | 0.141 | 0.146 | 0.175 | 最优 |
| OmniObject3D (1038) | F-Score↑ | 0.936 | 0.927 | 0.877 | 0.857 | SOTA |
| OmniObject3D | PSNR↑ | 20.874 | 19.402 | 19.193 | 18.529 | +1.47dB |
| 重光照数据集 | PSNR-A↑ | 21.750 | - | - | 18.592 | +3.16dB |

### 消融实验

| 配置 | PSNR-A↑ | LPIPS-A↓ | PSNR-D↑ | LPIPS-D↓ | 说明 |
|------|---------|----------|---------|----------|------|
| 完整方法 | 25.074 | 0.096 | 24.116 | 0.098 | 基准 |
| 无反投影测量 | 24.780 | 0.104 | 23.398 | 0.114 | 直接图像信息重要 |
| 无材质先验 | 24.471 | 0.108 | 22.687 | 0.121 | albedo分解质量显著下降 |
| 无FFCNet | 24.612 | 0.110 | 23.360 | 0.123 | 未见区域填补能力下降 |

### 关键发现
- ARM在几何和纹理质量上全面超越现有方法，纹理PSNR提升1-3dB
- 材质先验移除导致albedo质量下降最大（PSNR-D -1.43），证明仅靠渲染损失无法正确分解
- FFCNet对未见区域补全至关重要，用局部感受野U-Net替代会引入伪影
- SF3D生成恒定粗糙度/金属度，而ARM能重建空间变化的材质属性

## 亮点与洞察
- UV空间是外观建模的正确选择——直接在表面上操作避免了triplane的间接性问题
- 材质先验的设计很巧妙：不是直接预测材质，而是提供"什么看起来像金属/木头"的语义信息辅助分解
- 解耦几何和外观的策略简单但有效，让每个子模型专注于更小的任务
- 仅需8张GPU即可训练，在资源需求上比很多3D生成方法更友好

## 局限与展望
- 上游多视角扩散模型生成的不一致视角会导致纹理伪影
- UV解包过程耗时，无法在线训练（需预处理数据集）
- 未考虑开放式光照条件（所有训练数据使用特定环境光）
- 可以探索根据用户偏好加权输入视角来解决视角冲突

## 相关工作与启发
- 与SF3D的对比最直接：两者都做PBR分解，但ARM在UV空间操作+材质先验使其分解质量远超SF3D
- 与MeshFormer的关系：几何质量接近，但ARM在纹理上有显著优势
- 启发：UV空间操作+分阶段训练的思路可推广到其他需要高质量外观的任务

## 补充分析

### 训练数据与泛化性
- GeoRM和GlossyRM训练在Objaverse的150K子集上，InstantAlbedo用其中55K生成shapes
- 在GSO、OmniObject3D和自定义重光照数据集上评估，所有物体均为训练时未见过的
- 为每个评估物体生成144张图像（24视角×6种环境光），重光照评估设置严谨

### UV空间操作的局限与优势对比
- 优势：像素直接对应表面颜色，避免了triplane的间接映射和分辨率瓶颈
- 优势：U-Net和FFCNet可直接在2D texture map上操作，利用成熟的2D网络架构
- 局限：UV解包本身是一个非平凡操作，不同拓扑结构的mesh解包质量不同

## 评分
- 新颖性: ⭐⭐⭐⭐ UV空间外观分解+材质先验的组合有创新性，但各组件思路相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多种指标、消融实验和定性对比非常充分
- 写作质量: ⭐⭐⭐⭐ 流程清晰，图表精美，motivation阐述到位
- 价值: ⭐⭐⭐⭐⭐ 可重光照3D生成是实际应用（游戏/元宇宙）的刚需，方法质量显著领先

<!-- RELATED:START -->

## 相关论文

- [RNG: Relightable Neural Gaussians](rng_relightable_neural_gaussians.md)
- [HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)
- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [GenFusion: Closing the Loop between Reconstruction and Generation via Videos](genfusion_closing_the_loop_between_reconstruction_and_generation_via_videos.md)

<!-- RELATED:END -->
