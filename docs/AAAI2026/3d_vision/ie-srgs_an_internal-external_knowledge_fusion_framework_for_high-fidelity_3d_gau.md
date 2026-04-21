---
title: >-
  [论文解读] IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution
description: >-
  [AAAI 2026][3D视觉][3D高斯溅射] 提出IE-SRGS框架，通过融合外部2D超分辨率模型提供的高频纹理先验（外部知识）与多尺度3DGS模型提供的跨视图一致深度/纹理特征（内部知识），配合掩码引导融合策略，从低分辨率输入实现高保真3DGS超分辨率重建，在合成和真实场景上均达到SOTA。
tags:
  - AAAI 2026
  - 3D视觉
  - 3D高斯溅射
  - 超分辨率
  - 内外知识融合
  - Mip-Splatting
  - 深度估计
---

# IE-SRGS: An Internal-External Knowledge Fusion Framework for High-Fidelity 3D Gaussian Splatting Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2511.22233](https://arxiv.org/abs/2511.22233)  
**代码**: 未公开（审稿后释放）  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 超分辨率, 内外知识融合, Mip-Splatting, 深度估计

## 一句话总结

提出IE-SRGS框架，通过融合外部2D超分辨率模型提供的高频纹理先验（外部知识）与多尺度3DGS模型提供的跨视图一致深度/纹理特征（内部知识），配合掩码引导融合策略，从低分辨率输入实现高保真3DGS超分辨率重建，在合成和真实场景上均达到SOTA。

## 研究背景与动机

3D高斯溅射（3DGS）在新视角合成中表现优异，但从**低分辨率（LR）**输入重建高分辨率（HR）场景仍是重大挑战——LR输入缺乏精细纹理和几何细节。获取、存储和传输HR多视图数据在实际场景中往往代价高昂或不可行。

现有3DGS超分辨率方法（如SRGS、GaussianSR、SuperGaussian等）主要依赖预训练2D超分辨率（2DSR）模型来上采样LR视图，但直接使用2DSR模型存在两个根本问题：

**跨视图不一致性**：2D模型独立处理每个视图，无法保证多视图一致性，导致3D高斯优化时产生歧义

**域间差距**：2D训练数据与目标3D场景之间存在分布差异，SR模型在未见过的3D场景上性能退化

作者的关键洞察是：2DSR模型提供强大的HR细节先验但缺乏跨视图一致性；多尺度3DGS模型天然强制跨视图一致性且适应场景几何，但难以从LR输入恢复细粒度纹理。二者的优势恰好互补。

## 方法详解

### 整体框架

IE-SRGS包含三个关键步骤：
1. 使用预训练2DSR模型（SwinIR）和深度估计模型（Depth Anything V2）生成HR图像和深度图作为**外部知识**
2. 使用多尺度3DGS模型（基于Mip-Splatting）生成跨视图一致的内部参考图像和深度图作为**内部知识**
3. 通过**掩码引导融合策略**整合内外知识，联合指导HR 3DGS优化

### 关键设计

#### 1. **外部知识：HR细节恢复**

使用SwinIR生成超分辨率图像 $E_{\text{image}}$，Depth Anything V2估计深度图 $E_{\text{depth}}$。

纹理指导损失（L1 + D-SSIM加权组合）：

$$\mathcal{L}^E_{\text{tex}} = (1-\lambda)\mathcal{L}_1(E_{\text{image}}, R_{\text{image}}) + \lambda\mathcal{L}_{\text{ds}}(E_{\text{image}}, R_{\text{image}})$$

几何指导损失（基于Pearson相关的松弛相对深度损失）：

$$\mathcal{L}^E_{\text{gem}} = \frac{1}{N}\sum_{i=1}^{N}\left(1 - \frac{\text{Cov}(R_{\text{depth}}^i, E_{\text{depth}}^i)}{\sqrt{\text{Var}(R_{\text{depth}}^i)\text{Var}(E_{\text{depth}}^i)}}\right)$$

使用Pearson相关而非直接L1，是因为单目深度估计输出的是相对深度，与渲染深度在尺度上不对齐。

#### 2. **内部知识：歧义校正**

基于Mip-Splatting构建多尺度3DGS模型，利用其3D平滑操作抑制混叠和高频噪声：

$$\mathbf{g}^{\text{3D}}_{\text{reg}}(\boldsymbol{x}) = (\mathbf{g}^{\text{3D}} \otimes \mathbf{g}_{\text{low}})(\boldsymbol{x})$$

引入**多视图正则化（MV-Regulation）**联合监督多个视图，减少对单视图的过拟合，增强几何一致性。训练时随机采样3个视图联合优化。

通过**SR-Splatting**生成HR内部参考：将3D高斯投影到2D屏幕空间后上采样，再光栅化得到内部尺度图像 $I_{\text{image}}$ 和深度图 $I_{\text{depth}}$。

内部损失同样包含纹理和几何两部分：

$$\mathcal{L}^I_{\text{tex}} = (1-\lambda)\mathcal{L}_1(I_{\text{image}}, R_{\text{image}}) + \lambda\mathcal{L}_{\text{ds}}(I_{\text{image}}, R_{\text{image}})$$

$$\mathcal{L}^I_{\text{gem}} = \mathcal{L}_1(I_{\text{depth}}, R_{\text{depth}})$$

注意内部几何损失使用直接L1（因为内部深度与渲染深度尺度一致），而外部使用Pearson相关。

#### 3. **掩码引导融合策略**

**纹理融合**：2DSR的不一致和伪影通常是局部性的。计算每个像素的不确定度图：

$$D(p) = \frac{|I_{\text{image}}(p) - E_{\text{image}}(p)|}{I_{\text{image}}(p) + \epsilon}$$

通过阈值 $T$ 生成二值掩码 $M(p)$：差异大的区域用内部参考（确保一致性），差异小的区域用外部参考（保留HR细节）。

$$\mathcal{L}_{\text{tex}} = \mathcal{L}^{I'}_{\text{tex}} + \mathcal{L}^{E'}_{\text{tex}}$$

其中 $\mathcal{L}^{I'}_{\text{tex}} = \mathcal{L}^I_{\text{tex}} \odot M(p)$，$\mathcal{L}^{E'}_{\text{tex}} = \mathcal{L}^E_{\text{tex}} \odot (1-M(p))$。

**几何融合**：几何结构相对粗糙，对局部变化不敏感，直接加权求和：

$$\mathcal{L}_{\text{gem}} = \lambda_i \mathcal{L}^I_{\text{gem}} + \lambda_e \mathcal{L}^E_{\text{gem}}$$

### 损失函数 / 训练策略

- 最终损失：$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{tex}} + \mathcal{L}_{\text{gem}}$
- 内部模型训练30,000轮迭代
- $\lambda_i=0.001$，$\lambda_e=0.0001$
- 真实场景阈值 $T=0.9$，合成场景 $T=0.6$
- 单卡NVIDIA RTX 4090

## 实验关键数据

### 主实验

4× 3D超分辨率，NeRF Synthetic数据集：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| **IE-SRGS (Ours)** | **30.97** | **0.952** | **0.054** |
| SRGS | 30.83 | 0.948 | 0.056 |
| CROC | 30.71 | 0.945 | 0.067 |
| FastSR-NeRF | 30.47 | 0.944 | 0.075 |
| SwinIR-3DGS | 30.38 | 0.945 | 0.059 |
| 3DGS | 21.77 | 0.867 | 0.104 |
| Mip-Splatting | 24.59 | 0.909 | 0.101 |
| Upper Bound (HR输入) | 33.37 | 0.969 | 0.032 |

真实数据集（Mip-NeRF360 / Deep Blending / Tanks&Temples）：

| 方法 | Mip360 PSNR↑ | DB PSNR↑ | T&T PSNR↑ |
|------|-------------|----------|----------|
| **IE-SRGS** | **27.15** | **29.63** | **23.52** |
| Sequence Matters | 27.02 | — | 23.43 |
| SRGS | 26.88 | 29.49 | 23.41 |
| Mip-Splatting | 26.43 | 28.93 | 23.04 |
| Upper Bound | 27.23 | 29.73 | 23.51 |

IE-SRGS在所有数据集上均达到最佳性能，且接近HR上界。相比backbone Mip-Splatting，PSNR提升25.9%，LPIPS提升46.5%。

### 消融实验

在Mip-NeRF360上逐步添加组件：

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ | 说明 |
|------|-------|-------|--------|------|
| Mip-Splatting (Baseline) | 26.43 | 0.754 | 0.304 | 基线 |
| + External Texture ($E_{\text{image}}$) | 26.69 | 0.762 | 0.300 | 纹理细节提升 |
| + External Geometry ($E_{\text{depth}}$) | 26.72 | 0.763 | 0.299 | 几何先验有帮助 |
| + Internal Texture ($I_{\text{image}}$) | 27.00 | 0.775 | 0.283 | 跨视图一致性显著提升 |
| + Internal Geometry ($I_{\text{depth}}$) | 27.05 | 0.775 | 0.282 | 内部几何进一步提升 |
| + Mask-Guided Fusion | **27.15** | **0.779** | **0.278** | 融合策略最终优化 |

### 关键发现

- 内部知识的贡献（+0.33 PSNR）大于外部知识（+0.29 PSNR），说明跨视图一致性对3D SR至关重要
- 掩码引导融合额外提升0.10 PSNR，有效整合了内外优势
- IE-SRGS推理速度（260 FPS NeRF Synthetic，119 FPS MipNeRF360）优于SRGS（191/92 FPS），因联合指导加速了收敛
- 训练时间仅增加7-8分钟
- 对深度估计器鲁棒：DepthAnythingV2/V2-Small/DepthPro结果差异极小
- 对不同backbone通用：替换SwinIR或Mip-Splatting后仍有一致提升
- 8×超分辨率极端条件下仍优于SOTA

## 亮点与洞察

1. **内外知识互补的范式**：2DSR提供细节但不一致，多尺度3DGS一致但缺细节，掩码引导融合优雅地结合二者
2. **首次在3DGS SR中引入几何深度先验**，通过松弛相对损失适配单目估计器
3. **阈值 $T$ 鲁棒性强**：$T$ 在0.3-0.9范围内性能稳定，不需要精细调参
4. **框架设计模块化**：外部SR模型和内部3DGS backbone均可替换，是一个通用框架

## 局限与展望

- 额外训练内部3DGS模型增加了总训练时间（约增加40%）
- 掩码阈值 $T$ 对合成和真实场景需要不同设置
- 未探讨视频超分辨率（VSR）作为外部backbone的潜力（论文只用了SISR模型SwinIR）
- 8×以上极端超分辨率的表现未做系统评估

## 相关工作与启发

- SRGS是最直接的基线，仅使用外部知识
- Mip-Splatting的3D平滑操作是内部知识的基础
- 掩码引导融合的思路可推广到其他需要融合多源监督的3D任务
- 启发：可以将这种内外融合范式应用于3D编辑、3D修复等任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 内外知识融合的范式清晰且有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 4个数据集，多种消融，backbone泛化，鲁棒性分析全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机论述充分
- 价值: ⭐⭐⭐⭐ — 为3DGS超分辨率提供了新范式

<!-- RELATED:START -->

## 相关论文

- [Arbitrary-Scale 3D Gaussian Super-Resolution](arbitrary-scale_3d_gaussian_super-resolution.md)
- [S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](../../CVPR2025/3d_vision/s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)
- [SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](../../CVPR2026/3d_vision/sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](../../ICCV2025/3d_vision/bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)
- [External Knowledge Enhanced 3D Scene Generation from Sketch](../../ECCV2024/3d_vision/external_knowledge_enhanced_3d_scene_generation_from_sketch.md)

<!-- RELATED:END -->
