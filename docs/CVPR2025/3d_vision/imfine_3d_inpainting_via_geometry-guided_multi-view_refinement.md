---
title: >-
  [论文解读] IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement
description: >-
  [CVPR 2025][3D视觉][3D修复] 本文提出IMFine，一种用于无约束场景（包括360°环绕）的3D修复流水线，通过几何先验引导的warping和基于测试时适应的多视角refinement网络生成多视角一致的修复图像，并提出了一种新的修复mask检测技术来精确区分真正需要修复的遮挡区域，在多样化的benchmark上显著超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D修复
  - 物体移除
  - 多视角一致性
  - 高斯泼溅
  - 测试时适应
---

# IMFine: 3D Inpainting via Geometry-guided Multi-view Refinement

**会议**: CVPR 2025  
**arXiv**: [2503.04501](https://arxiv.org/abs/2503.04501)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 3D修复, 物体移除, 多视角一致性, 高斯泼溅, 测试时适应

## 一句话总结
本文提出IMFine，一种用于无约束场景（包括360°环绕）的3D修复流水线，通过几何先验引导的warping和基于测试时适应的多视角refinement网络生成多视角一致的修复图像，并提出了一种新的修复mask检测技术来精确区分真正需要修复的遮挡区域，在多样化的benchmark上显著超越现有方法。

## 研究背景与动机

1. **领域现状**：3D修复（inpainting/物体移除）是3D编辑的重要任务，现有方法主要通过将2D修复先验提升到3D来实现。方法分两类：基于SDS蒸馏的隐式方法和基于多视角一致性的显式方法。

2. **现有痛点**：SDS方法存在过饱和/过平滑问题；显式方法（如SPInNeRF、MVIPNeRF）在前向场景表现尚可，但在无约束场景（大视角变化、360°环绕）中会产生浮动伪影和模糊纹理。原因有二：(1) 3D几何很难完美恢复；(2) 感知损失只能减轻整体不一致，细节层面仍有不一致导致纹理模糊。

3. **核心矛盾**：独立修复各视角图像必然产生不一致，而现有的多视角修复网络（如MVInPainter）受限于数据集多样性，无法泛化到360°等无约束场景。

4. **本文目标**：实现在前向和无约束这两种场景下都能生成视觉一致、几何连贯的3D修复结果。

5. **切入角度**：利用单参考视角修复+几何引导warping提供粗略一致性，再通过测试时适应（test-time adaptation）微调多视角refinement网络来精修。

6. **核心 idea**：在预训练图像修复模型基础上加入时空注意力层，通过每场景的测试时微调，将其适应为多视角refinement模型，从而在warped图像上生成高保真、视角一致的修复结果。

## 方法详解

### 整体框架
给定2D-GS场景$\mathcal{G}$、训练图像$\{I_n\}$、相机位姿$\{\Pi_n\}$和物体mask$\{M_n\}$。Pipeline: (1) 物体移除——学习每个高斯的所属概率并剪枝；(2) 修复mask检测——区分物体mask和真正需要修复的NBS区域；(3) 参考视角修复——选一个视角用2D修复模型修复；(4) 几何重建——用单目深度估计+梯度引导填充depth；(5) warping——将参考视角修复内容warp到其他视角；(6) 多视角refinement——用微调网络精修warped图像；(7) 微调GS场景——用一致图像重建。

### 关键设计

1. **修复Mask检测（Inpainting Mask Detection）**:

    - 功能：精确区分物体mask中真正需要修复的遮挡区域（NBS, never-before-seen）和已经在其他视角暴露的区域
    - 核心思路：在360°场景中，物体mask覆盖的面积远大于实际需要修复的区域（平均仅占50.78%）。方法：先膨胀物体mask以包含临近像素，然后将膨胀mask映射到剪枝后的GS场景上（通过优化每个高斯的可学习属性$p_m$），利用多视角交叉观察，NBS区域会持续被标记而背景区域会被互相抑制，最后用SAM提取最终mask。
    - 设计动机：用整个物体mask作为修复区域会不必要地增加约2倍的修复面积，增加任务难度并降低质量。本方法达到81.12% IoU（vs GaussianEditor的42.55%）。

2. **基于测试时适应的多视角Refinement网络**:

    - 功能：精修warped图像中的伪影（纹理扭曲、与邻近区域不匹配），同时保持跨视角一致性
    - 核心思路：以预训练的StableDiffusionInpainting为基础，在每个Transformer block的自注意力层替换为稀疏时空注意力层（仅关注邻近帧和参考帧），以减少内存消耗。关键在于利用当前场景本身的原始多视角图像来构造训练数据——随机选参考视角，随机加mask，施加几何抖动，warp到其他视角，与原图配对训练。推理时随机打乱帧顺序以增强一致性。1000步微调即可。
    - 设计动机：大规模多视角数据集缺乏且存在域差异，而每场景微调避免了泛化问题。预训练模型提供了良好的图像修复先验，测试时适应只需学习"如何利用warped条件进行一致性精修"。

3. **几何引导的深度填充**:

    - 功能：为参考视角的修复区域生成合理的3D几何
    - 核心思路：用单目深度估计器获取参考修复图像的depth $\bar{D_r}$，通过线性变换对齐到场景尺度，然后基于梯度匹配+拉普拉斯平滑的优化问题填充mask内的depth值，约束(1)保持与估计depth相同的变化趋势，(2)在mask边界与已知depth无缝衔接。
    - 设计动机：精确的depth对于后续的warping质量至关重要，直接使用单目估计的depth可能在边界处不连续。

### 损失函数 / 训练策略
- 物体移除：$\mathcal{L}_1$ loss，1000步优化，阈值$\tau=0.4$剪枝
- 多视角refinement网络训练：简化变分下界目标（标准扩散损失），1000步微调，lr=$3\times10^{-5}$，batch=1
- GS场景微调：$L_1$ loss + SSIM loss，7000步

## 实验关键数据

### 主实验
在自建数据集（20个场景，涵盖室内外、90°/180°/360°）和SPINeRF数据集上：

| 方法 | PSNR↑ (自建) | LPIPS↓ (自建) | FID↓ (自建) | PSNR↑ (SPINeRF) |
|------|-------------|--------------|------------|-----------------|
| GaussianEditor | 15.71 | 0.6163 | 375.03 | 14.41 |
| GScream | 17.18 | 0.4431 | 290.63 | 16.96 |
| SPInNeRF | 18.75 | 0.3519 | 206.43 | 17.47 |
| MVIPNeRF | 18.63 | 0.4332 | 278.99 | 17.67 |
| **Ours** | **19.67** | **0.2685** | **149.52** | **17.58** |

### 消融实验
| 配置 | PSNR↑ | LPIPS↓ | FID↓ | 说明 |
|------|-------|--------|------|------|
| w/o Warping | 17.85 | 0.3215 | 198.24 | 直接修复，大视角下不一致 |
| w/o Refinement | 18.90 | 0.3069 | 206.96 | warped图像直接用，有伪影 |
| General Refinement | 19.08 | 0.2719 | 165.80 | 通用数据集训练，存在域差异 |
| Single-View Refinement | 19.46 | 0.2725 | 154.33 | 单视角独立精修，缺乏一致性 |
| **Multi-View Refinement (Full)** | **19.67** | **0.2685** | **149.52** | 完整方法 |

### 关键发现
- warping是保证粗粒度一致性的关键（无warping PSNR降约2 dB），refinement将质量进一步提升（约0.8 dB）
- 单视角refinement无法保证跨视角一致性，多视角版本有显著优势
- 测试时适应比通用模型好约0.6 dB PSNR，说明per-scene微调的必要性
- 修复mask检测将修复面积平均减少约50%，81.12% IoU远超对比方法
- 在无约束场景中优势特别明显，前向场景中各方法差距较小

## 亮点与洞察
- **测试时适应策略**：不依赖大规模多视角数据集训练，转而利用场景自身数据微调，巧妙回避了数据缺乏和域差异问题。这种per-scene适应的思路对3D编辑领域普遍适用。
- **修复mask vs 物体mask的区分**：首次明确提出在360°场景中物体mask≠修复mask，并给出了有效的检测方法。这个看似简单的观察在实际操作中影响巨大。
- **稀疏时空注意力**：用于控制内存消耗，推理时随机打乱帧顺序来增强一致性，简单有效。

## 局限与展望
- 每个场景需要约1小时的微调时间，实际部署效率有限
- 依赖单目深度估计和图像分割模块的精度
- 自建数据集虽然多样但规模有限（20个场景）
- 可以探索更高效的参数高效微调方法（如LoRA）来加速测试时适应
- 未处理动态场景或光照变化的情况

## 相关工作与启发
- **vs SPInNeRF/InNeRF360**: 它们独立修复多视角图像，只用感知损失减少不一致。本文通过warping+refinement从根本上提供了一致性保证。
- **vs MVInPainter**: MVInPainter训练了一个通用多视角修复网络，但受限于数据集多样性。本文通过测试时适应绕过了这个限制。
- **vs GaussianEditor**: GE使用SDS loss做3D编辑，但产生过饱和/过平滑，且FID高达375。本文的显式方法在无约束场景中优势明显。
- Warping-refinement的范式可以推广到3D编辑的其他任务（如纹理替换、风格迁移）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 测试时适应的多视角refinement和修复mask检测均有新意
- 实验充分度: ⭐⭐⭐⭐ 自建benchmark+消融实验丰富，视频结果更有说服力
- 写作质量: ⭐⭐⭐⭐ 流程图清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 面向无约束场景的3D修复是实际需求，方法切实有效

<!-- RELATED:START -->

## 相关论文

- [MVBoost: Boost 3D Reconstruction with Multi-View Refinement](mvboost_boost_3d_reconstruction_with_multi-view_refinement.md)
- [3D Gaussian Inpainting with Depth-Guided Cross-View Consistency](3d_gaussian_inpainting_with_depth-guided_cross-view_consistency.md)
- [MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)
- [3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)
- [MUSt3R: Multi-view Network for Stereo 3D Reconstruction](must3r_multi-view_network_for_stereo_3d_reconstruction.md)

<!-- RELATED:END -->
