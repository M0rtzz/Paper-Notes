---
title: >-
  [论文解读] GAP: Gaussianize Any Point Clouds with Text Guidance
description: >-
  [ICCV 2025][图像生成][点云转Gaussian] 提出GAP框架,利用深度感知图像扩散模型将无色点云转化为高保真3D Gaussian表示,通过表面锚定机制确保几何精度,并设计基于扩散的inpainting策略补全难以观测区域。
tags:
  - ICCV 2025
  - 图像生成
  - 点云转Gaussian
  - 文本引导
  - 扩散模型
  - 表面锚定
  - 外观生成
---

# GAP: Gaussianize Any Point Clouds with Text Guidance

**会议**: ICCV 2025  
**arXiv**: [2508.05631](https://arxiv.org/abs/2508.05631)  
**代码**: [项目页面](https://weiqi-zhang.github.io/GAP)  
**领域**: 图像生成  
**关键词**: 点云转Gaussian, 文本引导, 扩散模型, 表面锚定, 外观生成

## 一句话总结

提出GAP框架,利用深度感知图像扩散模型将无色点云转化为高保真3D Gaussian表示,通过表面锚定机制确保几何精度,并设计基于扩散的inpainting策略补全难以观测区域。

## 研究背景与动机

点云是3D计算机视觉中的基础表示,但如何将**无色原始点云**转化为高质量3D Gaussian用于实时渲染仍是未解决的挑战:

**Large Point-to-Gaussian**需要带颜色的点云输入

**DiffGS**难以泛化生成多样高质量外观

**传统网格+纹理路线**受UV映射的纹理重叠、碎片化和畸变问题困扰
4. 3DGS消除了显式UV参数化的需求,是点云外观生成的理想目标表示

## 方法详解

### 整体框架

1. **Gaussian初始化** — 从点云和UDF场初始化2DGS基元
2. **多视角生成与更新** — 深度感知扩散模型渐进生成外观
3. **Gaussian优化** — 表面锚定+尺度约束+渲染约束
4. **扩散式Gaussian inpainting** — 补全不可见区域

### Gaussian初始化

使用CAP-UDF学习无符号距离场 $f_u$,通过梯度推断法向量:
$$n_i = \frac{\nabla f_u(p_i)}{\|\nabla f_u(p_i)\|}$$

采用2DGS(2D Gaussian盘)替代3D椭球,法向量初始化旋转矩阵。

### 深度感知生成

利用ControlNet+深度条件的inpainting扩散模型。掩码动态分为三类:
- **Generate掩码**: 从未生成的区域
- **Keep掩码**: 已处理且当前视角非最优
- **Update掩码**: 当前视角提供更好观测角度,基于法向与视线的余弦相似度判断

### 表面锚定机制

距离损失约束Gaussian中心位于UDF零等值面上:
$$\mathcal{L}_{Distance} = \|f_u(\sigma_i)\|_2$$

尺度约束防止过大Gaussian:
$$\mathcal{L}_{Scale} = (\min(\max(s_i), \tau) - \max(s_i))^2$$

总优化目标:
$$\mathcal{L} = \mathcal{L}_{Rendering} + \alpha\mathcal{L}_{Distance} + \beta\mathcal{L}_{Scale}$$

### 扩散式Gaussian Inpainting

对不可见Gaussian,基于空间距离、法向一致性和不透明度加权扩散颜色:
$$\lambda_i = \frac{1/d_i}{\sum_{k=1}^L 1/d_k} \cdot (\mathbf{n}_i \cdot \mathbf{n}_j) \cdot \frac{o_i}{o_{max}}$$

## 实验

### Objaverse数据集文本引导外观生成

| 方法 | FID↓ | KID↓ | CLIP↑ | 用户:总体↑ | 用户:文本↑ |
|------|------|------|-------|----------|----------|
| TexTure | 42.63 | 7.84 | 26.84 | 2.90 | 3.05 |
| Text2Tex | 41.62 | 6.45 | 26.73 | 3.48 | 3.62 |
| SyncMVD | 40.85 | 5.77 | 27.24 | 3.12 | 3.40 |
| **GAP** | **38.94** | **4.81** | **27.51** | **4.15** | **4.08** |

### 从重建网格的UV方法对比

基于BPA重建+UV映射的方法在所有指标上大幅下降(FID升至60+),证明绕过UV参数化的优势。

### 关键发现

1. GAP在所有指标上超越现有纹理生成方法,用户偏好显著领先
2. 绕过UV参数化直接在3D空间优化Gaussian避免了拓扑歧义和UV失真
3. 表面锚定机制有效防止了Gaussian漂移导致的后续视角遮挡关系错误
4. 扩散式inpainting有效补全了多视角覆盖不到的区域

## 亮点与洞察

1. **点云→Gaussian的新范式** — 不需要颜色信息,纯几何+文本引导
2. **表面锚定确保几何一致** — UDF约束Gaussian贴合表面,避免浮空
3. **每视角单次优化** — 比标准3DGS多迭代优化更鲁棒
4. **场景级可扩展** — 可处理大规模场景点云

## 局限性

- 依赖预训练扩散模型的生成质量
- 多视角一致性受扩散模型本身限制
- UDF学习质量影响初始化效果

## 相关工作

- **纹理生成**: TexTure, Text2Tex, Paint3D, SyncMVD
- **3DGS生成**: Large Point-to-Gaussian, DiffGS, Gaussian Painter
- **渲染表示**: NeRF, 3DGS, 2DGS

## 评分

- 新颖性: ⭐⭐⭐⭐ (无色点云→Gaussian的新任务定义)
- 技术深度: ⭐⭐⭐⭐ (多组件协同设计完整)
- 实验充分度: ⭐⭐⭐⭐ (合成+真实扫描+场景级)
- 实用价值: ⭐⭐⭐⭐ (点云数据丰富,应用前景广)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[ICCV 2025\] Mind the Gap: Aligning Vision Foundation Models to Image Feature Matching](mind_the_gap_aligning_vision_foundation_models_to_image_feature_matching.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)
- [\[CVPR 2025\] Symbolic Representation for Any-to-Any Generative Tasks](../../CVPR2025/image_generation/symbolic_representation_for_any-to-any_generative_tasks.md)

</div>

<!-- RELATED:END -->
