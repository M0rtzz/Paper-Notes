---
title: >-
  [论文解读] SmartEraser: Remove Anything from Images using Masked-Region Guidance
description: >-
  [CVPR 2025][语义分割][目标移除] SmartEraser 提出 Masked-Region Guidance 新范式，保留掩码区域作为引导而非丢弃，配合百万级合成 Syn4Removal 数据集，在目标移除任务上大幅超越现有 mask-and-inpaint 方法。 目标移除（Object Removal）是图…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "目标移除"
  - "掩码区域引导"
  - "图像修复"
  - "扩散模型"
  - "合成数据集"
---

# SmartEraser: Remove Anything from Images using Masked-Region Guidance

**会议**: CVPR 2025  
**arXiv**: [2501.08279](https://arxiv.org/abs/2501.08279)  
**代码**: [项目页面](https://longtaojiang.github.io/smarteraser.github.io/)  
**领域**: 图像分割  
**关键词**: 目标移除, 掩码区域引导, 图像修复, 扩散模型, 合成数据集

## 一句话总结

SmartEraser 提出 Masked-Region Guidance 新范式，保留掩码区域作为引导而非丢弃，配合百万级合成 Syn4Removal 数据集，在目标移除任务上大幅超越现有 mask-and-inpaint 方法。

## 研究背景与动机

目标移除（Object Removal）是图像编辑的核心功能，已广泛集成到 Photoshop、Google Photos 等应用中。当前主流方法采用 "mask-and-inpaint" 范式，存在两大根本问题：
- **目标再生问题**：丢弃掩码区域后模型缺乏对移除目标的精确辨识，可能在掩码区域内生成新对象（如在道路上再生成一辆车而非移除）
- **上下文不一致**：用户定义的掩码通常大于目标对象，"mask-and-inpaint" 需要合成扩展区域，容易破坏周围上下文的视觉连贯性
- 现有数据集要么规模有限（RORD、ObjectDrop 不到 3.5k 唯一场景），要么依赖修复模型生成伪 ground truth（GQA-Inpaint、DEFACTO），性能受限于修复模型本身
- GAN-based 方法（ZITS++、MAT、LaMa）生成多样性和质量有限；diffusion-based 方法（RePaint、SD-Inpaint）虽能生成精细纹理，但掩码区域内容仍不确定
- 需要一个从根本上改变范式的方法来解决信息丢弃带来的固有缺陷

## 方法详解

### 整体框架

SmartEraser 由三个核心组件构成：(1) Masked-Region Guidance 范式——将完整图像 $[\mathbf{m}, \mathbf{x}]$ 而非 $[\mathbf{m}, \mathbf{x} \odot (1-\mathbf{m})]$ 作为模型输入；(2) Syn4Removal 百万级合成数据集——通过将实例粘贴到背景图上构造（输入图、掩码、GT背景）三元组；(3) 基于 SD v1.5 的微调框架，集成掩码增强和 CLIP 视觉引导。

### 关键设计1：Masked-Region Guidance 范式

**功能**：保留掩码区域信息作为移除目标的精确引导，避免目标再生和上下文破坏。

**核心思路**：传统范式的条件输入为 $[\mathbf{m}, \mathbf{x} \odot (1-\mathbf{m})]$，即用黑色等占位符替换掩码区域。本文将其改为 $[\mathbf{m}, \mathbf{x}]$，掩码 $\mathbf{m}$ 标识移除区域但不丢弃原始像素。模型的学习目标从 $P(\hat{\mathbf{x}} | \mathbf{m}, \mathbf{x} \odot (1-\mathbf{m}); \Theta')$ 变为 $P(\hat{\mathbf{x}} | \mathbf{m}, \mathbf{x}; \Theta)$。由于掩码通常大于目标物体，模型可以直接复制掩码区域内目标物体周围的上下文。

**设计动机**：让模型"知道"要移除什么比让模型"猜测"要填充什么更合理。但这引入了一个捷径问题——模型可能直接复制掩码内容而非移除，因此需要 Syn4Removal 数据集来解决。

### 关键设计2：Syn4Removal 合成数据集

**功能**：提供 100 万个高质量（输入图、掩码、真实背景）三元组，训练 Masked-Region Guidance 模型而不产生捷径。

**核心思路**：从公开实例分割数据集裁剪对象实例，通过 CLIP 相似度过滤低质量实例。从 COCONut 和 SAM-1B 筛选高质量背景图。按类别统计归一化面积比 $\mathcal{N}(\mu_c, \sigma_c^2)$ 采样缩放比例。计算可行粘贴区域 $\mathbf{R}_f = \mathbf{R}_1 \cap \mathbf{R}_2$（IoU 约束 + 边缘约束），使用 Alpha Blending 融合：$\mathbf{x} = \alpha \odot \mathbf{x}_i + (1-\alpha) \odot \mathbf{x}_b$。

**设计动机**：粘贴的对象在输入中存在但在 GT 中不存在，这迫使模型学习真正的"移除"而非"复制"。真实背景作为 GT 确保模型学习到真实图像分布。

### 关键设计3：掩码增强与 CLIP 视觉引导

**功能**：提高模型对不同用户掩码形状的鲁棒性，并利用视觉特征增强对移除目标的语义理解。

**核心思路**：掩码增强使用 6 种变形方法：原始掩码、腐蚀掩码、膨胀掩码、凸包掩码、椭圆掩码、边界框+贝塞尔曲线掩码，即 $\mathbf{m} = \text{ME}_i(\bar{\mathbf{m}}), i \in \{1,...,6\}$。CLIP 视觉引导用 CLIP 视觉编码器提取掩码区域特征 $\nu_\theta(\mathbf{x} \odot \mathbf{m})$，通过可训练 MLP 映射到文本特征空间，拼接到 "Remove the instance of" 的文本嵌入上：$\hat{\mathbf{c}} = [\tau_\theta(\mathbf{y}), \text{MLP}(\nu_\theta(\mathbf{x} \odot \mathbf{m}))]$。

**设计动机**：用户掩码形状多变，训练时模拟不同形状可减少训练-推理差距。CLIP 引导为扩散模型提供了"要移除什么"的语义信息。

### 损失函数

标准扩散模型训练损失：$\mathcal{L} = \mathbb{E}_{\mathbf{x}, \mathbf{m}, \mathbf{x}_b, t, \epsilon} \| \epsilon - \epsilon_\theta(\mathbf{z}_t, \bar{\mathbf{z}}, \mathbf{m}, \hat{\mathbf{c}}, t) \|_2^2$，其中 $\bar{\mathbf{z}} = E(\mathbf{x})$ 为输入图像的 VAE 潜码，$\mathbf{z}_t$ 为加噪 GT 潜码。

## 实验关键数据

### 主实验：RORD-Val 目标移除对比

| 方法 | FID ↓ | CMMD ↓ | ReMOVE ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|------|-------|--------|----------|---------|--------|--------|
| **SmartEraser** | **16.03** | **0.092** | **0.937** | **0.276** | **0.612** | **19.99** |
| PowerPaint | 24.06 | 0.294 | 0.926 | 0.308 | 0.602 | 18.10 |
| CLIPAway | 25.46 | 0.123 | 0.915 | 0.333 | 0.577 | 17.43 |
| LaMa | 24.24 | 0.216 | 0.916 | 0.348 | 0.557 | 16.38 |
| SD-Inpaint | 69.50 | 0.324 | 0.857 | 0.369 | 0.537 | 16.11 |

### 消融实验：各组件贡献

| 组件 | FID ↓ | LPIPS ↓ | PSNR ↑ |
|------|-------|---------|--------|
| 完整模型 | **16.03** | **0.276** | **19.99** |
| w/o CLIP 视觉引导 | 较差 | 较差 | 较差 |
| w/o 掩码增强 | 较差 | 较差 | 较差 |
| mask-and-inpaint 范式 | 显著较差 | 显著较差 | 显著较差 |

### 关键发现

- SmartEraser 在 RORD-Val 上 FID 超越 SOTA **10.3 分**，PSNR 超越 **1.89 dB**
- Masked-Region Guidance 范式的核心价值在于同时解决目标再生和上下文一致性问题
- 合成数据集的质量对模型性能至关重要——实例过滤（CLIP 分数）和可行区域计算等细节显著影响最终效果
- 在复杂场景中优势尤为明显：大掩码和复杂构图的移除场景是传统方法的痛点

## 亮点与洞察

- **范式级创新**：从根本上挑战了 "mask-and-inpaint" 的默认假设，证明保留掩码区域信息比丢弃更有效
- **数据集构建精巧**：通过 copy-paste 策略巧妙解决了 Masked-Region Guidance 的捷径问题
- **掩码增强策略**：6 种变形方法系统地覆盖了用户可能提供的各种掩码形状

## 局限与展望

- 基于 SD v1.5 的分辨率限制（512×512），可能影响高分辨率场景的处理效果
- copy-paste 合成数据与真实移除场景仍存在分布差距
- 未探索视频目标移除和交互式编辑场景
- 未来可扩展到更大的基础模型（如 SDXL）和更复杂的编辑任务

## 相关工作与启发

- 与 PowerPaint 和 CLIPAway 的比较表明，仅在提示或注意力层面引入背景信息不足以解决根本问题
- Syn4Removal 的 copy-paste 合成策略可推广到其他需要成对训练数据的任务
- CLIP 视觉引导的方式可以借鉴到其他需要"指定目标"的生成式编辑任务

## 评分

⭐⭐⭐⭐ — 范式级创新令人信服，"保留掩码区域"的简单改变带来显著质量提升，配合精心设计的数据集和训练策略，在多个基准上取得 SOTA。方法整体简洁优雅。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] EdgeTAM: On-Device Track Anything Model](edgetam_on-device_track_anything_model.md)
- [\[CVPR 2025\] PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](picosam3_real-time_in-sensor_region-of-interest_segmentation.md)
- [\[CVPR 2025\] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [\[CVPR 2025\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)

</div>

<!-- RELATED:END -->
