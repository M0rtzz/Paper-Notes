---
title: >-
  [论文解读] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors
description: >-
  [3D视觉] 提出 RI3D，将稀疏视图合成分解为"修复可见区域"和"补全缺失区域"两个子任务，引入两个个性化扩散模型（repair + inpainting）配合两阶段优化策略，在极端稀疏输入下实现高质量 3DGS 重建。
tags:
  - 3D视觉
---

# RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.10860](https://arxiv.org/abs/2503.10860)
- **作者**: Avinash Paliwal, Xilong Zhou, Wei Ye, Jinhui Xiong, Rakesh Ranjan, Nima Khademi Kalantari
- **机构**: Texas A&M University, Meta Reality Labs, Max Planck Institute for Informatics
- **代码**: [项目主页](https://people.engr.tamu.edu/nimak/Papers/RI3D)
- **领域**: 3D视觉 / 稀疏视图合成
- **关键词**: 3D Gaussian Splatting, 扩散模型先验, 稀疏视图重建, inpainting, few-shot

## 一句话总结
提出 RI3D，将稀疏视图合成分解为"修复可见区域"和"补全缺失区域"两个子任务，引入两个个性化扩散模型（repair + inpainting）配合两阶段优化策略，在极端稀疏输入下实现高质量 3DGS 重建。

## 研究背景与动机

### 问题定义
稀疏视图（如仅3张图片）的新视角合成是一个极具挑战性的任务。现有方法面临两大核心问题：

**可见区域过拟合**：仅靠少量输入图像约束优化，导致新视角渲染出现严重伪影

**缺失区域无法恢复**：被遮挡或未覆盖的区域通常只能产生模糊或暗淡的结果

### 现有方法局限
- **正则化方法**（DNGaussian、FSGS、CoR-GS 等）：通过深度监督、稠密化策略等正则约束优化，但在极端稀疏设置下仍无法有效幻想缺失区域的细节
- **扩散模型方法**（ReconFusion、CAT3D）：训练视图合成扩散模型来生成新视角，但生成结果缺乏 3D 一致性，导致优化后的结果过度模糊；且基于 NeRF 表征，渲染速度慢

### 核心创新思路
关键洞察：将视图合成过程**解耦为两个独立子任务**——修复可见区域和补全缺失区域，分别用专门的扩散模型来处理，避免单一模型难以同时兼顾两个目标。

## 方法详解

### 整体框架
RI3D 包含三个核心环节：
1. **高质量深度初始化**：融合 DUSt3R 和单目深度获取逐视图深度图
2. **两个个性化扩散模型**：repair 模型修复渲染伪影，inpainting 模型填补缺失区域
3. **两阶段优化**：第一阶段重建可见区域，第二阶段补全缺失区域

### 1. 3D Gaussian 初始化
核心思想是融合 DUSt3R 深度（3D 一致但平滑）和单目深度（细节丰富但相对且不一致）的互补优势：

$$\mathbf{d}^{*} = \arg\min_{\mathbf{d}} \left[ \mathbf{M} \odot \|\mathbf{d} - \mathbf{d}^{D}\|_2 + \lambda \|\nabla \mathbf{d} - \nabla \mathbf{d}^{M}\|_2 \right]$$

- 第一项：在 DUSt3R 高置信区域 $\mathbf{M}$ 保持深度值的一致性
- 第二项：在全图范围保持单目深度的梯度（边缘细节），$\lambda=10$
- 类似泊松融合（Poisson Blending），但在所有区域施加梯度约束
- 通过稀疏矩阵求解器高效计算，并应用双边滤波锐化边界
- 最终每个像素分配一个 Gaussian，投射到 3D 空间

### 2. Repair 扩散模型
- 基于预训练 ControlNet 微调
- **训练数据生成**：Leave-one-out 策略——构建 $N$ 个子集（每个去掉一张图），训练 $N$ 个 3DGS，先优化6000步后引入被排除图再优化至10000步
- 生成渐进精化的"损坏图像"（corruption）-"干净图像"对
- 在目标场景上微调1800步实现场景个性化

### 3. Inpainting 扩散模型
- 基于 Stable Diffusion Inpainting 模型
- 在输入图像上通过随机 masking 生成大量输入-输出对进行微调（类似 RealFill）
- 微调2000步实现场景个性化
- 两个模型均在 512×512 分辨率上操作

### 4. 两阶段优化

**第一阶段：重建可见区域**
$$\mathcal{L}_{\text{stage1}} = \sum_{i=1}^{N} \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_i^{\text{ref}}, \mathbf{I}_i^{\text{ref}}) + \sum_{j=1}^{M} \lambda_j \mathbf{M}_j^{\alpha} \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_j^{\text{nov}}, \mathbf{G}_j^{\text{nov}}) + \sum_{j=1}^{M} \|\mathbf{A}_j \odot (1-\mathbf{M}_j^{\alpha}) \odot \mathbf{M}_j^{b}\|_1$$

- 第一项：输入视角的重建损失（L1 + SSIM + LPIPS + 深度相关性）
- 第二项：$M$ 个新视角利用 repair 模型生成伪真值监督，仅在可见区域（$\mathbf{M}^{\alpha}$）施加
- 第三项：鼓励缺失区域的 opacity 趋近于零，避免在缺失区域放置可见 Gaussians
- 每400步刷新一次 repair 结果，共优化4000步

**第二阶段：补全缺失区域**
$$\mathcal{L}_{\text{stage2}} = \sum_{i=1}^{N} \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_i^{\text{ref}}, \mathbf{I}_i^{\text{ref}}) + \sum_{j=1}^{M} \lambda_j \mathcal{L}_{\text{rec}}(\hat{\mathbf{I}}_j^{\text{nov}}, \mathbf{G}_j^{\text{nov}}) + \sum_{k=1}^{K} (1-\mathbf{M}_k^{\alpha}) \odot \mathbf{M}_k^{b} \odot L_p(\hat{\mathbf{I}}_k^{\text{nov}}, \hat{\mathbf{L}}_k^{\text{nov}})$$

- 选择 $K<M$ 个不重叠的新视角进行 inpainting，避免独立补全重叠内容导致不一致
- 将补全区域通过单目深度投影到 3D 空间（使用公式2融合深度范围）
- 第三项约束渲染结果与 inpainting 结果在缺失区域的一致性
- 每200步执行一轮 inpainting + repair，迭代至所有缺失区域填充完成，共优化4000步

## 实验关键数据

### 主实验：Mip-NeRF 360 数据集

| 方法 | 3-view PSNR↑ | 3-view SSIM↑ | 3-view LPIPS↓ | 9-view PSNR↑ | 9-view LPIPS↓ |
|------|-------------|-------------|--------------|-------------|--------------|
| DNGaussian | 12.02 | 0.226 | 0.665 | 12.97 | 0.637 |
| FSGS | 13.14 | 0.288 | 0.578 | 16.00 | 0.470 |
| CoR-GS | 13.51 | 0.314 | 0.633 | 15.48 | 0.574 |
| ReconFusion | 15.50 | 0.358 | 0.585 | 18.19 | 0.511 |
| CAT3D | 16.62 | 0.377 | 0.515 | 18.67 | 0.460 |
| **RI3D (Ours)** | **15.74** | **0.342** | **0.505** | **17.48** | **0.415** |

- RI3D 在 LPIPS 指标上全面最优，表明生成纹理细节最丰富
- PSNR/SSIM 略低于 ReconFusion 和 CAT3D，因为后两者缺失区域模糊反而有利于像素级指标

### 消融实验：3-view Mip-NeRF 360

| 配置 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o 深度增强 | 15.61 | 0.336 | 0.513 |
| 单阶段（仅 repair） | 14.92 | 0.309 | 0.527 |
| w/o 微调 inpainting | 15.64 | 0.341 | 0.508 |
| **RI3D（完整）** | **15.74** | **0.342** | **0.505** |

### 关键发现
1. **深度融合至关重要**：去掉深度增强后 LPIPS 从 0.505 → 0.513，DUSt3R 在低置信区域的深度不准确导致浮动伪影
2. **两阶段优化必要性**：单阶段仅用 repair 模型 PSNR 下降约 0.8，LPIPS 下降 0.022，因为 repair 模型不擅长幻想高质量缺失区域纹理
3. **个性化微调提升一致性**：未微调 inpainting 模型虽能生成好看纹理，但可能与周围场景风格不匹配（如灰色植物、叶子状地面）
4. 在 CO3D 数据集上也取得了最优 LPIPS 分数，尤其在 9-view 设置下优于 ReconFusion

## 亮点与洞察

1. **任务分解设计**：将稀疏视图合成的难点分解为两个独立子任务，各用专门的扩散模型处理，比单一模型"全能"方案更有效
2. **深度融合策略**：Poisson blending 式融合 DUSt3R + 单目深度的方法非常优雅，兼顾 3D 一致性和细节
3. **个性化扩散模型**：在目标场景上微调扩散模型（repair 1800步、inpainting 2000步），确保生成结果与场景风格一致
4. **快速渲染**：采用 3DGS 而非 NeRF 作为 3D 表征，兼顾质量和效率

## 局限性

- 强依赖 DUSt3R 的深度质量，当 DUSt3R 深度严重不准确时会产生鬼影伪影
- 无法处理单图输入场景（leave-one-out 策略至少需要2张图）
- 微调两个扩散模型增加计算成本
- 训练需要多 GPU（Stage 2 需两块 A5000）

## 相关工作与启发

- **vs ReconFusion/CAT3D**：RI3D 的核心改进是将"修复"和"补全"解耦，而非用单一扩散模型做视图合成
- **vs GaussianObject**：借鉴了其 ControlNet 修复策略，但从物体扩展到场景级别
- **深度融合**：类似 Poisson Image Editing 思想，未来可以用更好的 MVS 方法替代 DUSt3R
- **启发**：任务分解 + 专门模型的范式具有广泛迁移价值，适用于其他稀疏输入的 3D 重建问题

## 评分 ⭐⭐⭐⭐
方法设计精巧，深度融合和两阶段优化策略有独创性，LPIPS 指标全面领先，实验充分。但对 DUSt3R 的依赖和多 GPU 需求是实际使用的瓶颈。

<!-- RELATED:START -->

## 相关论文

- [Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)

<!-- RELATED:END -->
