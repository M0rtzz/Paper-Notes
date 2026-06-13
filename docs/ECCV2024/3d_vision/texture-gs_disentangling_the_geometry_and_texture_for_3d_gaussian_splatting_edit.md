---
title: >-
  [论文解读] Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing
description: >-
  [ECCV 2024][3D视觉][3D高斯溅射] 提出Texture-GS，首次为3D高斯溅射解耦几何与纹理，通过UV映射MLP和局部Taylor展开将外观表示为2D纹理图，实现实时纹理替换和编辑（58 FPS，RTX 2080 Ti）。 3D-GS将颜色存储在每个高斯的SH系数中，几何和外观完全耦合…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D高斯溅射"
  - "纹理映射"
  - "几何-外观解耦"
  - "场景编辑"
  - "实时渲染"
---

# Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing

**会议**: ECCV 2024  
**arXiv**: [2403.10050](https://arxiv.org/abs/2403.10050)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 纹理映射, 几何-外观解耦, 场景编辑, 实时渲染

## 一句话总结

提出Texture-GS，首次为3D高斯溅射解耦几何与纹理，通过UV映射MLP和局部Taylor展开将外观表示为2D纹理图，实现实时纹理替换和编辑（58 FPS，RTX 2080 Ti）。

## 研究背景与动机

3D-GS将颜色存储在每个高斯的SH系数中，几何和外观完全耦合，无法像传统网格+纹理那样灵活编辑外观（如纹理替换）。虽然NeuTex等NeRF方法实现了UV映射，但对每个光线-高斯交点评估MLP代价过高，无法支持实时渲染。简单地仅对高斯中心计算UV坐标会导致同一高斯覆盖的所有像素映射到相同UV位置，纹理空间不连续。

## 方法详解

### 整体框架

分两阶段：第一阶段学习UV映射MLP（通过循环一致性约束和Chamfer距离），第二阶段使用冻结的MLP，通过Taylor展开高效计算交点UV坐标，并学习2D纹理值。

### 关键设计

**UV映射MLP学习**: 先训练标准3D-GS获取深度图，反投影得到表面3D点集。使用三个约束学习映射$\phi: \mathbb{R}^3 \rightarrow \mathbb{R}^2$及其逆映射$\phi^{-1}$：
- 3D循环一致性：$\|x - \phi^{-1} \circ \phi(x)\|$
- 2D循环一致性：$\|u - \phi \circ \phi^{-1}(u)\|$
- Chamfer距离：$\phi^{-1}$的输出应均匀覆盖表面点云

**高效UV映射（Taylor展开）**: 对每个高斯中心$\mu_j$预计算其UV坐标$\phi(\mu_j)$和Jacobian矩阵$J|_{\mu_j}$，交点UV用一阶近似：$\tilde{\phi}(I(G_j, r_p)) = \phi(\mu_j) + J|_{\mu_j}(I(G_j,r_p) - \mu_j)$。仅需一次小矩阵乘法，确保实时渲染。

**光线-高斯交点计算**: 将高斯沿法向量（最小特征值对应的特征向量）压平，求射线-平面交点。引入不透明度01正则化和法线监督确保高斯足够扁平。

**颜色函数**: 漫反射颜色从纹理图查询，视角依赖部分用per-Gaussian SH系数残差表示：$\mathcal{C}(G_j, r_p) = h(\tilde{\phi}(I(G_j,r_p)), \mathcal{T}) + c_j^{SH}$。

### 损失函数

$$\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_{mask} + \lambda_{ssim}\mathcal{L}_{ssim} + \lambda_{01}\mathcal{L}_{01} + \lambda_n(\mathcal{L}_{norm} + \mathcal{L}_{sm}) + \lambda(\mathcal{L}_1^{noSH} + \lambda_{ssim}\mathcal{L}_{ssim}^{noSH})$$

其中$\mathcal{L}^{noSH}$用无SH渲染的图像计算，鼓励外观信息主要存储在纹理而非per-Gaussian属性中。

## 实验关键数据

### DTU数据集新视角合成

| 方法 | PSNR↑ | L1↓ | LPIPS↓ | FPS |
|------|-------|-----|--------|-----|
| NeuTex | 30.39 | 0.0158 | 0.1613 | 0.025 |
| Neural Gauge Fields | 29.44 | 0.0166 | 0.1506 | 0.025 |
| 3D-GS | 30.99 | 0.0121 | 0.1079 | 198 |
| **Texture-GS** | **30.03** | **0.0135** | **0.1440** | **58** |

### 消融实验

| 方法 | PSNR↑ | LPIPS↓ | 说明 |
|------|-------|--------|------|
| Texture-GS | 30.03 | 0.1440 | 完整模型 |
| Ours (no SH) | 27.63 | 0.1566 | 去除per-Gaussian SH |
| w/o Reg | 30.62 | 0.1374 | 去除noSH正则 |
| w/o Reg (no SH) | 25.10 | 0.1757 | 同时去除 |
| Pre-fetching (仅中心UV) | 29.28 | 0.1557 | 简单方案 |

高斯数量裁剪的影响：

| 高斯比例 | PSNR↑ | FPS |
|----------|-------|-----|
| 100% | 30.03 | 58 |
| 50% | 29.57 | 69 |
| 20% | 28.75 | 82 |
| 5% | 27.86 | 104 |

### 关键发现

- 比NeuTex快2000倍（58 vs 0.025 FPS），训练时间从30小时降至90分钟
- Taylor展开近似比直接MLP评估快得多，且纹理连续无伪影
- 即使只用5%高斯（4.4K个），仍可合成高质量纹理替换结果
- noSH正则化确保外观主要存储在纹理中，对纹理编辑至关重要

## 亮点与洞察

1. **Taylor展开是核心创新**：将MLP的复杂度从渲染时降至预计算时，仅需一次矩阵乘法，完美平衡精度和效率
2. 将3D-GS与传统图形管线的纹理映射概念融合，架起两者之间的桥梁
3. 支持全局纹理替换和局部纹理绘制，展示了ECCV文字编辑等实用案例

## 局限性

- 纹理空间定义为单位球面，不适合多物体或室外场景
- 高对比度纹理替换时边缘可能轻微模糊，源于高斯法线不准确
- PSNR略低于原始3D-GS（30.03 vs 30.99），引入了约1dB的渲染质量损失

## 相关工作与启发

NeuTex的循环一致性思路被继承但效率大幅提升。Nuvo的多chart设计可用于扩展到复杂场景。对3D-GS编辑生态有重要推动。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)
- [\[CVPR 2025\] FruitNinja: 3D Object Interior Texture Generation with Gaussian Splatting](../../CVPR2025/3d_vision/fruitninja_3d_object_interior_texture_generation_with_gaussian_splatting.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)

</div>

<!-- RELATED:END -->
