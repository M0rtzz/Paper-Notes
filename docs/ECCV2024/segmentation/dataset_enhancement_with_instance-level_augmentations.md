---
title: >-
  [论文解读] Dataset Enhancement with Instance-Level Augmentations
description: >-
  [ECCV 2024][图像分割][数据增强] 提出一种基于预训练扩散模型的实例级数据增强方法，通过在保持原始标注不变的前提下逐个重绘图像中的目标实例，显著提升了显著性目标检测、语义分割和目标检测的性能，同时支持数据匿名化。
tags:
  - ECCV 2024
  - 图像分割
  - 数据增强
  - 实例级增强
  - 扩散模型
  - 图像修复
  - 数据隐私
---

# Dataset Enhancement with Instance-Level Augmentations

**会议**: ECCV 2024  
**arXiv**: [2406.08249](https://arxiv.org/abs/2406.08249)  
**代码**: https://github.com/kupynorest/instance-augmentation (有)  
**领域**: 图像分割  
**关键词**: 数据增强, 实例级增强, 扩散模型, 图像修复, 数据隐私

## 一句话总结
提出一种基于预训练扩散模型的实例级数据增强方法，通过在保持原始标注不变的前提下逐个重绘图像中的目标实例，显著提升了显著性目标检测、语义分割和目标检测的性能，同时支持数据匿名化。

## 研究背景与动机

**领域现状**：深度学习对大规模标注数据集的依赖日益增加，但收集和标注大规模高质量数据集代价高昂，且存在隐私和伦理问题。许多小型经典数据集（如DUTS、Pascal VOC）因规模限制而逐渐边缘化。

**现有痛点**：
1. 传统数据增强（翻转、裁剪、色彩变换）只能提供有限的视觉多样性，本质上是简单的像素变换
2. 图像混合技术（CutMix、MixUp等）容易产生视觉伪影，模型可能学到捷径
3. 使用生成模型在目标数据集上训练后生成样本，无法引入超出数据集分布的新信息
4. 全图生成方法需要同时生成对应的标注，验证标注质量困难

**核心矛盾**：如何在保留原始精确标注的同时，通过生成模型引入数据集外的视觉多样性？

**切入角度**：不重绘整张图像，而是在实例级别（object instance）进行重绘——利用预训练的大规模扩散模型逐个修复图像中的目标，保持原始分割掩码/检测框/类别标签不变。

**核心idea**：将预训练latent diffusion model用于实例级图像修复，结合深度图和边缘图的ControlNet控制条件，确保重绘的物体形状和位置与原始标注一致，从而实现标注不变、外观多样的数据增强。

## 方法详解

### 整体框架
给定图像 $I$ 和标注集 $\mathcal{Y} = \{(M_i, c_i)\}_{1 \le i \le N}$（每个实例的二值掩码和类别），方法通过条件LDM逐个重绘每个实例，生成新图像 $I^*$，映射 $F(I, \mathcal{Y}) = I^*$ 保持标注不变。Pipeline步骤：(1) 估计全图深度图和边缘图；(2) 按深度排序实例（远到近）；(3) 逐个用条件修复模型重绘每个实例；(4) Alpha混合组合最终图像。

### 关键设计

1. **独立重绘与Alpha混合**：为避免反复通过LDM编解码器导致的质量退化（PSNR和SSIM逐步下降），每个实例的修复都从原始图像出发独立进行：
    $I_i^* = I_{i-1}^* \odot (\mathbb{1} - M_i) + G(I, M_i, T_i) \odot M_i$
   这样只需 $N$ 次修复即可生成 $2^N$ 种原始/重绘组合变体。按深度图排序（远处先绘制）确保遮挡关系正确。

2. **ControlNet条件控制（Better Inpainting）**：裸用修复模型无法保证(a)生成物体类别正确、(b)形状符合原始掩码、(c)不删除小物体。因此引入ControlNet注入深度图（DepthAnything估计）和边缘图（HED提取）作为额外条件，显著降低生成物体与原始轮廓的偏差。深度图保留整体场景结构，边缘图保留细节轮廓。

3. **Prompt Engineering增强多样性**：

    - **物体描述扩展**：利用WordNet的synset lemmas扩展类别名，稳定CLIP文本编码
    - **颜色和光照随机化**：对易变色类别（车、背包等）随机采样颜色和光照条件加入prompt，增加外观多样性
    - **人物动作描述**：对"person"类使用BLIP-VQA预测动作并加入prompt，处理复杂场景中的人物重绘

4. **掩码精化（Mask Refinement）**：针对DUTS等显著性检测数据集，用TRACER-7对重绘后的每个物体裁剪区域重新预测紧致的显著性掩码，消除生成物体与原始标注间的潜在不匹配。

### 损失函数 / 训练策略
本方法是数据增强pipeline而非端到端训练：训练时每次迭代中每个实例以30%概率被替换为重绘版本，保证训练多样性。每个物体生成最多3个合成替换版本。对NSFW内容做严格过滤。

## 实验关键数据

### 主实验
**目标检测（COCO）**：

| 模型 | 数据 | AP | AP50 | AP75 |
|------|------|-----|------|------|
| Deformable-DETR | 原始 | 39.3 | 60.0 | 42.0 |
| Deformable-DETR | +本方法 | **40.5** | 60.2 | **43.4** |
| RT-DETR | 原始 | 51.4 | 69.6 | 55.4 |
| RT-DETR | +本方法 | **52.4** | 69.7 | **56.5** |
| YOLOv5m | 原始 | 44.1 | 63.4 | 47.8 |
| YOLOv5m | +本方法 | **45.7** | 64.0 | **49.7** |

**显著性目标检测（DUTS训练）**：

| 模型 | 数据 | ECSSD F_max | DUTS-TE F_max | HKU-IS F_max |
|------|------|-------------|---------------|--------------|
| U2Net | 原始 | 0.944 | 0.863 | 0.930 |
| U2Net | +本方法 | **0.948** | **0.874** | **0.935** |
| TRACER-4 | 原始 | 0.956 | 0.911 | 0.944 |
| TRACER-4 | +本方法 | **0.960** | **0.918** | **0.948** |

在36个指标中，34个得到改善。

### 消融实验

| 配置 | F_max↑ | MAE↓ | S_m↑ | F_avg↑ |
|------|--------|------|------|--------|
| 完整方法 | 0.892 | 0.033 | 0.885 | 0.853 |
| 去掉实例级（全图修复） | 0.889 | 0.035 | 0.881 | 0.848 |
| 去掉边缘+深度控制 | 0.886 | 0.036 | 0.881 | 0.846 |
| 去掉边缘控制 | 0.890 | 0.034 | 0.880 | 0.850 |
| 去掉Prompt Engineering | 0.889 | 0.034 | 0.880 | 0.849 |
| 去掉掩码精化 | 0.888 | 0.035 | 0.879 | 0.844 |

### 关键发现
- **数据稀缺场景**：在仅用10%~100% COCO训练集时，增强一致带来+2-3 AP提升
- **匿名化不损性能**：将全部真实人物替换为合成人物后，整体AP仅下降1点（30.5%实例被重绘），人脸再识别率仅0.14%
- **与传统增强互补**：本方法与传统（翻转/旋转/模糊等）增强互补，联合使用效果更好
- **语义分割**：VOC上Mask2Former合成+微调达78.2 mIoU，大幅超越Diffumask的77.6

## 亮点与洞察
- **概念优雅**：实例级重绘保留标注、引入域外知识的想法简单有效，适用于检测/分割/显著性等多任务
- **指数级变体**：N个物体的图像可组合出 $2^N$ 种原始/合成混合版本，数据扩增效率极高
- **隐私保护附加价值**：同一pipeline天然支持数据匿名化，且不影响下游任务性能
- **独立重绘策略**巧妙地避免了潜空间反复编解码的噪声累积问题

## 局限与展望
- 方法性能上限受修复模型和ControlNet能力约束，小/遮挡物体有时被直接去除
- 依赖于text-to-image扩散模型的域（真实场景），无法直接扩展到卫星/医学等特殊域
- ControlNet的深度/边缘控制在复杂场景中偶有不精确，可能导致标注-图像不一致
- 未来可替换更强的生成模型（如SDXL等）进一步提升质量

## 相关工作与启发
- ControlNet/T2I-Adapter为扩散模型增加空间可控性 → 本文将其用于保持标注一致
- Diffumask/DatasetDM同时生成图像+标注 → 本文指出保留真实标注更可靠
- Copy-Paste增强只是简单拼贴 → 本文通过重绘实现语义一致的实例替换
- 启发：可将此方法与主动学习结合，定向增强困难/稀有类别的训练样本

## 评分
- 新颖性: ⭐⭐⭐⭐ 实例级重绘增强概念新颖，保留标注+引入域外知识的思路优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 跨三个任务（检测/语义分割/显著性）、六个数据集、多模型验证，含匿名化和泛化实验
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，动机到方法到实验衔接自然
- 价值: ⭐⭐⭐⭐ 作为通用数据增强工具实用性强，释放数据集+代码更增添实际价值

<!-- RELATED:START -->

## 相关论文

- [VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](visage_video_instance_segmentation_with_appearance-guided_enhancement.md)
- [Plain-Det: A Plain Multi-Dataset Object Detector](plain-det_a_plain_multi-dataset_object_detector.md)
- [ProMerge: Prompt and Merge for Unsupervised Instance Segmentation](promerge_prompt_and_merge_for_unsupervised_instance_segmentation.md)
- [Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](../../CVPR2026/segmentation/reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)
- [UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)

<!-- RELATED:END -->
