---
title: >-
  [论文解读] MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World
description: >-
  [CVPR 2025][图像生成][镜面反射生成] MirrorVerse 通过构建增强的合成数据集 SynMirrorV2（包含随机位姿、旋转和多物体场景），配合三阶段课程式训练策略，训练出 MirrorFusion 2.0 模型，首次使扩散模型能够生成逼真的镜面反射，在合成和真实场景中均显著超越前方法。
tags:
  - CVPR 2025
  - 图像生成
  - 镜面反射生成
  - 合成数据集
  - 扩散模型微调
  - 课程式训练
  - 物理感知
---

# MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World

**会议**: CVPR 2025  
**arXiv**: [2504.15397](https://arxiv.org/abs/2504.15397)  
**代码**: https://mirror-verse.github.io/  
**领域**: 扩散模型 / 图像生成  
**关键词**: 镜面反射生成, 合成数据集, 扩散模型微调, 课程式训练, 物理感知

## 一句话总结
MirrorVerse 通过构建增强的合成数据集 SynMirrorV2（包含随机位姿、旋转和多物体场景），配合三阶段课程式训练策略，训练出 MirrorFusion 2.0 模型，首次使扩散模型能够生成逼真的镜面反射，在合成和真实场景中均显著超越前方法。

## 研究背景与动机

1. **领域现状**：扩散模型在图像生成和编辑方面取得巨大进展，但在遵循物理定律（如阴影、反射、遮挡）方面仍然不足。当前最强的 T2I 模型（SD 3.5、FLUX）在提示生成镜面反射时，产生几何不一致和失真的结果。
2. **现有痛点**：前作 MirrorFusion 尝试用合成数据集 SynMirror 将问题转化为 inpainting 任务，但其数据集缺乏多样性——物体位置固定、无旋转变化、仅单物体场景。这导致模型在不同物体位姿和多物体遮挡场景下泛化能力差，且难以迁移到真实世界。
3. **核心矛盾**：真实世界中包含高质量镜面反射的图像非常稀缺，而合成数据与真实数据之间存在域差异。关键挑战是如何在合成数据中引入足够的变化性，并通过训练策略弥合域差距。
4. **本文目标** (1) 如何让合成数据覆盖更多物体位姿和多物体场景？ (2) 如何让在合成数据上训练的模型泛化到真实世界？
5. **切入角度**：在数据生成管线中加入关键增强（随机位置、随机旋转、物体接地），并基于语义配对后构建多物体场景；采用三阶段课程式训练从合成逐步过渡到真实数据。
6. **核心 idea**：通过增强合成数据管线的多样性 + 三阶段课程训练，使得 inpainting 扩散模型学会生成几何正确的镜面反射。

## 方法详解

### 整体框架
任务被定义为 inpainting 问题：给定一张图像和镜面区域的 mask，模型在 mask 区域生成正确的镜面反射。输入为 noisy latent $z_t$、masked image $z_m$、inpainting mask $x_m$ 和深度图 $x_d$，采用基于 BrushNet 的双分支 U-Net 架构，conditioning 分支接收上述拼接输入，通过 zero convolution 将其特征注入 generation 分支。

### 关键设计

1. **SynMirrorV2 数据集增强管线**:

    - 功能：构建一个物体位姿多样、涵盖多物体遮挡场景的大规模合成反射数据集。
    - 核心思路：(1) **随机位置采样**：在镜子视锥和相机视锥的交集区域内随机采样物体的 x-y 位置，确保物体同时出现在画面和镜面中。(2) **随机旋转**：绕 y 轴随机旋转物体，引入方向变化。(3) **物体接地**：通过简单的接地技术确保物体站在地面上而非悬浮。(4) **多物体语义配对**：基于 ABO 数据集物体类别进行语义配对（如椅子-桌子），通过碰撞检测避免重叠，生成 3,140 个多物体场景。
    - 设计动机：SynMirror 中物体位置固定、无旋转、仅单物体，导致模型只能处理特定观察角度。增强后的数据集含 207,610 张图像，覆盖 66,062 个 3D 物体、多种背景和地面纹理。

2. **双分支架构 + 深度条件 (Dual Branch + Depth)**:

    - 功能：在保留未 mask 区域信息的同时生成与场景一致的反射。
    - 核心思路：基于 BrushNet 架构，conditioning 分支接收 $[z_t, z_m, x_m, x_d]$ 的拼接，通过 zero convolution 将各层特征注入 generation 分支：$\epsilon_\theta(z_t,t,c)_i = \epsilon_\theta(z_t,t,c)_i + w \cdot \mathcal{Z}(\epsilon'_\theta([z_t,z_m,x_m,x_d],t)_i)$。深度图提供 3D 空间信息帮助模型理解物体与镜面的几何关系。
    - 设计动机：对比实验显示直接修改 SD Inpainting（SDI+Depth）会导致颜色泄漏——因为 noisy latent 和条件信息在初始卷积层就混合了，限制了后续层获取干净特征。双分支架构让条件信息独立处理，效果更好。

3. **三阶段课程式训练 (Three-Stage Curriculum)**:

    - 功能：从合成数据逐步过渡到真实数据，弥合域差异。
    - 核心思路：**Stage 1** (40k iters)：在 SynMirrorV2 的单物体训练集上从 SD v1.5 初始化的双分支模型开始训练，不冻结 generation 分支。**Stage 2** (10k iters)：在多物体训练集上微调，学习遮挡和空间关系。**Stage 3** (10k iters)：在真实世界 MSD 数据集上继续微调，用单目深度估计器（Depth-Pro）替代 GT 深度。训练中 20% 概率随机丢弃文本提示。
    - 设计动机：直接在真实数据上训练数据量不足，直接在合成数据上训练有域差异。课程式训练先学基础反射几何（单物体），再学复杂遮挡（多物体），最后适配真实世界分布，逐步递进。

### 损失函数 / 训练策略
标准扩散去噪损失 $L_{DM} = E[\|\epsilon - \epsilon_\theta(z_t,t,c)\|^2]$。使用 AdamW 优化器，学习率 $1e^{-5}$，batch size 4/GPU，4 张 A100。推理时 CFG=7.5，UniPC scheduler 50 步。

## 实验关键数据

### 主实验

在 MirrorBenchV2 单物体分集上的定量对比：

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP Sim↑ |
|------|-------|-------|--------|-----------|
| MirrorFusion (baseline) | 18.31 | 0.76 | 0.122 | 26.00 |
| **MirrorFusion 2.0 (本文)** | **18.79** | **0.77** | **0.108** | 25.96 |

多物体分集上的定量对比：

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ | CLIP Sim↑ |
|------|-------|-------|--------|-----------|
| Ours 40k (无多物体微调) | 17.77 | 0.743 | 0.126 | 26.17 |
| **Ours 50k (多物体微调)** | **18.00** | **0.744** | **0.119** | 26.09 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 无随机位置/旋转 (SynMirror) | 物体方向经常错误 | 基线方法的核心缺陷 |
| 无多物体训练 | 多物体场景反射失败，物体融合 | 说明多物体场景数据不可替代 |
| SDI + Depth (单分支) | 颜色泄漏严重 | 验证双分支架构的必要性 |
| 无 Stage 3 (无真实数据微调) | 复杂真实场景失败 | 课程训练的最后一步对泛化至关重要 |

### 关键发现
- **数据增强贡献最大**：随机位置和旋转是解决位姿泛化问题的关键，让模型不再局限于固定观察角度。
- **多物体训练不可替代**：不经过多物体场景训练的模型在遮挡场景中会出现物体融合等严重问题。
- **课程训练有效弥合域差距**：经过 Stage 3 真实数据微调后，模型能处理如桌面杂物线缆、双镜面等复杂真实场景。
- 用户研究中 84% 用户偏好 MirrorFusion 2.0 的生成结果。

## 亮点与洞察
- **用合成 3D 渲染数据教扩散模型物理知识**的思路很有启发性：对于物理感知的生成任务（反射、阴影、折射），构造控制变量的合成数据比收集真实数据更高效，且能精确标注。
- **课程式训练从合成到真实**的策略值得在其他 synthetic-to-real 迁移场景中借鉴：先学简单原子操作，再学复杂组合，最后适配真实分布。
- **语义配对的多物体放置策略**简单但有效：通过手工定义类别配对关系 + 碰撞检测，低成本生成了语义合理的多物体场景。

## 局限与展望
- 定量指标提升幅度较小（PSNR 仅提升约 0.5），主要优势体现在定性结果和泛化性上。
- 仅支持平面镜反射，对曲面镜、水面反射等更复杂的反射类型未覆盖。
- 对超过两个物体的场景仍存在伪影，数据集规模可以进一步扩大。
- 深度图质量对结果影响较大，推理时的单目深度估计不够准确时可能导致几何偏差。
- 512×512 分辨率较低，高分辨率扩展未验证。

## 相关工作与启发
- **vs MirrorFusion**：前作使用固定物体位置的 SynMirror 数据集，双分支架构冻结 generation 分支。本文在数据多样性和训练策略上全面改进，尤其在位姿变化和多物体场景上大幅超越。
- **vs ObjectDrop**：ObjectDrop 用反事实数据集训练扩散模型处理阴影和镜面反射，但依赖真实对数据收集。MirrorVerse 完全基于合成数据更可扩展。
- **vs HD-Painter/BrushNet**：这些通用 inpainting 方法缺乏物理反射知识，在镜面区域生成的结果不符合反射几何。

## 评分
- 新颖性: ⭐⭐⭐ 核心贡献在数据和训练策略，方法层面相对增量
- 实验充分度: ⭐⭐⭐⭐ 合成+GSO+真实三个层面的评估，含用户研究
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，消融全面
- 价值: ⭐⭐⭐⭐ 数据集和训练策略对物理感知生成领域有参考价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Lifting Motion to the 3D World via 2D Diffusion](lifting_motion_to_the_3d_world_via_2d_diffusion.md)
- [\[CVPR 2025\] OpenSDI: Spotting Diffusion-Generated Images in the Open World](opensdi_spotting_diffusion-generated_images_in_the_open_world.md)
- [\[NeurIPS 2025\] RLVR-World: Training World Models with Reinforcement Learning](../../NeurIPS2025/image_generation/rlvr-world_training_world_models_with_reinforcement_learning.md)
- [\[ICLR 2026\] QVGen: Pushing the Limit of Quantized Video Generative Models](../../ICLR2026/image_generation/qvgen_pushing_the_limit_of_quantized_video_generative_models.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)

<!-- RELATED:END -->
