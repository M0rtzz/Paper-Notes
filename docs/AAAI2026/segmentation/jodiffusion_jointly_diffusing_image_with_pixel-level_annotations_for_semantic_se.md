---
title: >-
  [论文解读] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion
description: >-
  [AAAI 2026][图像分割][语义分割] 提出JoDiffusion框架，通过在潜在空间中联合扩散图像与像素级标注掩码，首次实现仅基于文本提示同时生成语义一致的图像-标注对，在Pascal VOC、COCO和ADE20K上显著超越现有Image2Mask和Mask2Image方法。
tags:
  - AAAI 2026
  - 图像分割
  - 语义分割
  - 数据集生成
  - 扩散模型
  - 联合生成
  - 标注掩码
  - 潜在空间
---

# JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion

**会议**: AAAI 2026  
**arXiv**: [2512.13014](https://arxiv.org/abs/2512.13014)  
**作者**: Haoyu Wang, Lei Zhang (通讯), Wenrui Liu, Dengyang Jiang, Wei Wei (西北工业大学), Chen Ding
**代码**: [GitHub](https://github.com/00why00/JoDiffusion)  
**领域**: segmentation  
**关键词**: 语义分割, 数据集生成, 扩散模型, 联合生成, 标注掩码, 潜在空间

## 一句话总结

提出JoDiffusion框架，通过在潜在空间中联合扩散图像与像素级标注掩码，首次实现仅基于文本提示同时生成语义一致的图像-标注对，在Pascal VOC、COCO和ADE20K上显著超越现有Image2Mask和Mask2Image方法。

## 研究背景与动机

### 问题背景
语义分割依赖大规模高质量的图像-像素级标注对进行训练，但人工标注成本极高，特别是在多目标交互或密集小目标场景下。利用扩散模型生成合成数据集成为缓解标注瓶颈的有力方案。

### 已有方法的不足
现有方法分为两类流水线，各有致命缺陷：

- **Image2Mask（先生图后推标注）**：如DiffuMask、Dataset Diffusion、SDS等，先用文本生成图像，再通过交叉注意力等机制推断伪标注。问题在于文本-图像相似度计算误差和特征图空间分辨率不足，导致**图像与标注语义不一致**
- **Mask2Image（先给标注再生图）**：如FreeMask、SegGen等，基于手工标注掩码生成图像。虽然语义一致性好，但受限于手工标注数量，**可扩展性差**，生成图像的多样性受制于有限的标注模板

### 核心动机
能否用一个统一的模型，仅基于文本提示就能**同时**生成语义一致的图像和像素级标注？这样既保证语义一致性，又不受手工标注数量限制，同时解决上述两类方法的问题。

## 方法详解

JoDiffusion包含三个阶段：标注VAE训练、联合扩散建模、掩码优化。

### 阶段1：标注VAE（Annotation VAE）

为将标注掩码映射到与图像共享的潜在空间，设计专用的标注VAE：

- **输入表示**：将每个像素的类别索引转为二进制编码$M_{\text{bin}}$，避免相邻类别值过于接近导致区分困难
- **轻量架构**：编码器$E_M$和解码器$D_M$仅使用少量卷积/转置卷积层，参数量约50M（对比图像VAE的300M）
- **训练目标**：仅用交叉熵损失训练，不施加KL散度正则化（因其仅作压缩工具而非生成模型）

$$\mathcal{L}_{\text{Annotation VAE}} = -\sum_{(i,j)}\sum_{c=0}^{N_C} M_{\text{one-hot},(i,j,c)} \log \bar{M}_{(i,j,c)}$$

- **重建质量**：在三个数据集上mIoU均超过98%，证明了编码的高保真性

### 阶段2：联合扩散建模（Joint Diffusion）

基于UniDiffuser架构，将文本、图像和标注掩码在潜在空间中联合扩散和去噪：

1. **编码**：用BLIP-2为图像生成描述文本$T$，分别通过CLIP文本/图像编码器和图像VAE得到$z_T, z_I$，通过标注VAE得到$z_M$
2. **前向扩散**：对$z_I$和$z_M$施加**共享噪声**$\epsilon_{IM}$，确保扩散过程中的结构一致性：

$$q(z_I^t, z_M^t | z_I^0, z_M^0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_t}\begin{bmatrix}z_I^0 \\ z_M^0\end{bmatrix}, (1-\bar{\alpha}_t)I\right)$$

3. **联合去噪**：网络$\epsilon_\theta(z_I^t, z_M^t, z_T, t)$学习联合噪声预测，而非独立估计各部分噪声
4. **训练损失**：标准MSE去噪损失

$$\mathcal{L}_{\text{denoising}} = \mathbb{E}_{t, z_I^0, z_M^0, \epsilon}\left[\|\epsilon_\theta(z_I^t, z_M^t, z_T, t) - \epsilon_{IM}\|^2\right]$$

**关键设计**：采用自注意力（而非交叉注意力）拼接文本和图像特征进行建模，提供更灵活的调优能力。推理时仅需文本提示即可同时生成图像和标注掩码。

### 阶段3：掩码优化策略（Mask Optimization）

扩散过程在小区域和物体边界处不可避免地引入标注噪声。提出基于边界众数的后处理策略：

1. 识别面积小于阈值$\tau$的小区域$R$
2. 提取边界像素集$\hat{R}$
3. 计算边界像素中出现频率最高的类别$c^* = \arg\max_c \sum_{(i,j)\in\hat{R}} \mathbb{I}(x_{i,j}=c)$
4. 将区域$R$内所有像素重新赋值为$c^*$

该策略利用自然图像中语义区域的连续性先验，从统计角度等价于对区域真实类别的最大似然估计。

## 实验关键数据

### 表1：与Image2Mask方法对比（Pascal VOC & MS-COCO）

| 分割器 | 骨干网络 | 方法 | VOC数据量 | VOC mIoU (Syn) | VOC mIoU (Real+Syn) | COCO数据量 | COCO mIoU (Syn) | COCO mIoU (Real+Syn) |
|--------|---------|------|----------|---------------|-------------------|-----------|----------------|---------------------|
| DeepLabV3 | ResNet50 | Raw Dataset | 11.5k | 77.4 | - | 118k | 48.9 | - |
| DeepLabV3 | ResNet50 | SDS | 26k | 60.4 | 77.6 | 50k | 31.0 | 50.3 |
| DeepLabV3 | ResNet50 | Dataset Diffusion | 40k | 61.6 | 77.6 | 80k | 32.4 | 54.6 |
| DeepLabV3 | ResNet50 | **JoDiffusion** | 40k | **72.5** | **78.3** | 80k | **42.6** | **56.4** |
| DeepLabV3 | ResNet101 | Raw Dataset | 11.5k | 79.9 | - | 118k | 54.9 | - |
| DeepLabV3 | ResNet101 | SDS | 26k | 59.1 | 79.8 | 50k | 31.8 | 56.8 |
| DeepLabV3 | ResNet101 | Dataset Diffusion | 40k | 64.8 | 80.3 | 80k | 34.2 | 57.4 |
| DeepLabV3 | ResNet101 | **JoDiffusion** | 40k | **75.8** | **80.7** | 80k | **44.9** | **59.1** |
| Mask2Former | ResNet50 | Raw Dataset | 11.5k | 77.3 | - | 118k | 57.8 | - |
| Mask2Former | ResNet50 | DiffuMask | 60k | 57.4 | 77.5 | - | - | - |
| Mask2Former | ResNet50 | SDS | 26k | 59.8 | 78.1 | 50k | 29.8 | 57.7 |
| Mask2Former | ResNet50 | Dataset Diffusion | 40k | 60.2 | 78.2 | 80k | 31.0 | 57.8 |
| Mask2Former | ResNet50 | **JoDiffusion** | 40k | **74.5** | **79.4** | 80k | **44.6** | **58.5** |

JoDiffusion在纯合成数据训练（Syn）上大幅领先：VOC上比次优方法高约10-14个mIoU点，COCO上高约10-13点；在Real+Syn混合训练上也一致领先。

### 表2：与Mask2Image方法对比（Pascal VOC & ADE20K）

| 骨干网络 | 方法 | VOC数据量 | VOC mIoU | ADE20K数据量 | ADE20K mIoU |
|---------|------|----------|---------|------------|------------|
| ResNet50 | Raw Data | 11.5k | 77.3 | 20k | 47.2 |
| ResNet50 | SegGen | - | - | 11M | 49.9 |
| ResNet50 | FreeMask | 40k | 77.9 | 40k | 48.2 |
| ResNet50 | **JoDiffusion** | 40k | **79.4** | 40k | **48.4** |
| Swin-S | Raw Data | 11.5k | 83.8 | 20k | 51.6 |
| Swin-S | FreeMask | 40k | 84.2 | 40k | 52.1 |
| Swin-S | **JoDiffusion** | 40k | **85.1** | 40k | **52.2** |

JoDiffusion在不需要手工标注掩码作为输入的前提下，仍然一致超越Mask2Image方法。

### 消融实验要点
- **掩码优化阈值**：$\tau=20$时效果最优（72.47 mIoU），相比不优化（$\tau=0$, 71.37）提升1.1个点
- **生成数据量**：从5k到40k数据量增加带来持续的mIoU提升（68.54→72.47）

## 亮点

- **首次实现图像-标注联合生成**：区别于先图后标注或先标注后图的两步法，仅需文本提示即可同时生成语义一致的图像-标注对，同时兼顾语义一致性和可扩展性
- **标注VAE设计精巧**：二进制编码+轻量级架构（50M参数），重建mIoU>98%，有效将离散类别图映射到连续潜在空间
- **共享噪声机制**：对图像和标注施加相同噪声，从扩散过程本身保证结构一致性，而非依赖后处理对齐
- **显著的性能增益**：纯合成数据训练下，VOC上比Dataset Diffusion高10+mIoU点，证明了联合生成策略在语义一致性上的根本优势

## 局限与展望

- **依赖预训练文本描述**：需要BLIP-2为训练图像生成文本描述，描述质量直接影响生成效果
- **固定分辨率训练**：所有图像和标注统一resize到512×512，限制了对高分辨率精细分割的支持
- **掩码优化策略偏简单**：基于边界众数的后处理仅处理小区域噪声，对大面积语义混淆无效
- **数据集覆盖有限**：仅在VOC（21类）、COCO（81类）、ADE20K（150类）上验证，未涉及更细粒度或领域特定数据集
- **可扩展性上限未探索**：最大仅生成40k-80k合成数据，未研究更大规模生成时的质量变化趋势
- **与Mask2Image方法的优势有限**：在Real+Syn设定和ADE20K上，相比FreeMask的提升较为有限（<1 mIoU）

## 与相关工作的对比

- **DiffuMask**：通过交叉注意力推断标注，语义一致性差（VOC Syn仅57.4），JoDiffusion联合生成避免了此问题
- **Dataset Diffusion**：引入LLM生成多样文本+自注意力图改进标注质量，但仍受限于特征图分辨率（VOC Syn 60-65），JoDiffusion高出10+点
- **SDS**：引入CLIP相似度和类别平衡过滤，但根本上仍是两步法，JoDiffusion在VOC Syn上领先12+点
- **FreeMask**：Mask2Image代表方法，语义一致性好但受限于标注库规模，JoDiffusion在无需手工标注的条件下取得相当甚至更好的结果
- **SegGen**：训练额外text-to-mask模型提升多样性，但需要11M数据量才达到ADE20K 49.9 mIoU，JoDiffusion用40k即达48.4
- **UniDiffuser**：JoDiffusion的扩散架构基础，但UniDiffuser处理文本-图像双模态，JoDiffusion扩展为文本-图像-标注三模态联合建模

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次提出图像-标注联合扩散生成，思路清晰且有效
- 实验充分度: ⭐⭐⭐⭐ — 三个基准数据集、多种骨干网络、两类方法对比、消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 三阶段框架描述清晰，公式推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 在语义分割数据生成领域提出了统一范式，实用价值高

<!-- RELATED:START -->

## 相关论文

- [From Attribution to Action: Jointly ALIGNing Predictions and Explanations](from_attribution_to_action_jointly_aligning_predictions_and_explanations.md)
- [Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [Reasoning with Pixel-level Precision: QVLM Architecture and SQuID Dataset for Quantitative Geospatial Analytics](../../CVPR2026/segmentation/reasoning_with_pixel-level_precision_qvlm_architecture_and_squid_dataset_for_qua.md)
- [DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](../../CVPR2025/segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)

<!-- RELATED:END -->
