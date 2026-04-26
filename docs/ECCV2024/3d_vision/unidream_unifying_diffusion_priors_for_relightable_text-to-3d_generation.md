---
title: >-
  [论文解读] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation
description: >-
  [ECCV 2024][3D视觉] 提出UniDream，通过训练albedo-法线对齐的多视角扩散模型（AN-MVM），结合Transformer重建模型和分阶段SDS优化，实现可重光照的文本到3D生成，生成的3D物体具有干净的albedo纹理和PBR材质。
tags:
  - ECCV 2024
  - 3D视觉
---

# UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation

**会议**: ECCV 2024  
**arXiv**: [2312.08754](https://arxiv.org/abs/2312.08754)  
**代码**: [项目页](https://UniDream.github.io)  
**领域**: 3D视觉

## 一句话总结

提出UniDream，通过训练albedo-法线对齐的多视角扩散模型（AN-MVM），结合Transformer重建模型和分阶段SDS优化，实现可重光照的文本到3D生成，生成的3D物体具有干净的albedo纹理和PBR材质。

## 研究背景与动机

### 领域现状

**领域现状**：现有text-to-3D方法（DreamFusion、Magic3D等）使用RGB扩散模型，生成的3D物体光照和阴影被"烘焙"进纹理

### 现有痛点

**现有痛点**：烘焙的光照限制了3D物体在不同光照条件下的真实感和可重用性

### 核心矛盾

**核心矛盾**：Fantasia3D尝试分离光照和纹理但常混合albedo和反射光

### 解决思路

**解决思路**：核心问题**：如何从文本生成具有干净PBR材质（albedo、法线、粗糙度、金属度）、可在任意光照下重新渲染的3D物体

## 方法详解

### 整体框架

三阶段流水线：
1. AN-MVM生成多视角albedo和法线图
2. Transformer重建模型（TRM）从albedo图重建粗3D模型，再用AN-MVM做SDS细化
3. 固定albedo和法线，用Stable Diffusion优化粗糙度和金属度生成PBR材质

### 关键设计

**AN-MVM（Albedo-Normal Aligned Multi-View Diffusion）**：
- 在Stable Diffusion基础上同时训练albedo和normal两个域
- 多视角自注意力：在UNet的self-attention层前合并多视角数据实现跨视角约束
- 多域自注意力：albedo和normal对应视角之间应用self-attention确保域一致性
- 使用类别标签L区分normal域；70% 3D数据 + 30% LAION-Aesthetics 2D数据联合训练保持语义泛化

**TRM（Transformer-Based Reconstruction）**：
- 用DINO-v2提取四视角albedo图像特征，通过可学习camera调制MLP编码相机参数
- Transformer解码器将可学习token与图像特征做交叉注意力，输出triplane表示
- 用albedo而非RGB训练重建模型，避免光照阴影对triplane-NeRF重建的影响

**PBR材质生成**：
- 固定albedo和normal后引入额外hash grid和MLP预测粗糙度和金属度
- 使用Stable Diffusion的SDS监督，允许环境光同步优化（限制为单通道避免色偏）

### 损失函数

SDS损失使用AN-MVM在albedo和normal两域上的噪声预测差，权重分别为0.8和0.2。TRM使用LPIPS + L2 + 法线监督联合训练。

## 实验关键数据

### 主实验

| 方法 | User Study(%)↑ | CLIP Score↑ | R@1(%)↑ | R@5(%)↑ | R@10(%)↑ |
|------|----------------|-------------|---------|---------|----------|
| DreamFusion | 7.1 | 71.0 | 54.2 | 82.2 | 91.5 |
| Magic3D | 10.5 | 75.1 | 75.9 | 93.5 | 96.6 |
| MVDream | 32.1 | 75.7 | 76.8 | 94.3 | 96.9 |
| **UniDream** | **50.3** | **77.9** | **80.3** | **97.4** | **98.5** |

### 消融实验

多视角扩散模型对比：UniDream的AN-MVM输出的2D图像成功实现光照-纹理解耦，生成的normal map跨视角一致性优于MVDream的RGB输出。

TRM重建→SDS细化→PBR材质的渐进过程展示了质量的逐步提升：粗重建保持清晰的纹理和几何边界，SDS细化后获得高质量3D模型，PBR阶段添加真实的材质属性。

PBR对比：Fantasia3D的albedo中混入了光照和阴影信息，UniDream有效实现了解耦，可在不同环境光下重光照。

### 关键发现

- User study中50.3%的偏好率显著超过MVDream（32.1%），验证了重光照能力的实用价值
- Normal监督显著加速几何收敛，生成更完整光滑的表面
- TRM提供的3D先验有效避免了SDS优化中的"Janus问题"
- 单通道环境光约束有效防止了Stable Diffusion引入的色偏

## 亮点与洞察

- "先albedo后PBR"的分阶段策略将复杂问题逐步分解，比直接联合优化更稳定
- Normal域的一致性天然更容易保证（同一3D点的世界坐标法线在不同视角下值相同），有助于拉动albedo多视角一致性的收敛
- albedo+normal联合训练的扩散模型是理解和分离3D物体外观的关键技术创新
- Reconstruction model提供的3D先验 + diffusion model的细节生成 = 互补优势

## 训练和生成细节

AN-MVM训练：32张A800 GPU，256×256分辨率，每GPU batch_size=128（16物体×2域×4视角），50K迭代约19小时。TRM训练：32张A800，batch_size=96，70K步约3天。SDS细化：NeRF阶段5000步，DMTet阶段2000步。PBR材质阶段：512×512渲染，2000迭代。

3D数据集使用约300K Objaverse物体经过严格过滤（排除无纹理、非单物体、低质量、无caption的模型），渲染多视角albedo和normal数据用于训练。使用70% 3D数据 + 30% LAION-Aesthetics 2D数据联合训练AN-MVM，并给3D数据caption添加", 3D asset"后缀区分。

## 局限与展望

- 训练数据仅约300K Objaverse物体，语义泛化和材质泛化受限
- 不支持透明材质等特殊材质类型
- 渲染流水线未集成路径追踪，重光照效果存在简化
- 多阶段流水线总耗时较长

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个可重光照的text-to-3D框架
- 有效性：⭐⭐⭐⭐ — 定量和用户研究均显著领先
- 实用性：⭐⭐⭐⭐ — PBR输出可直接用于游戏和AR/VR
- 推荐度：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

<!-- RELATED:END -->
