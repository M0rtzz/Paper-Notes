---
title: >-
  [论文解读] World-Consistent Video Diffusion with Explicit 3D Modeling
description: >-
  [CVPR 2025][视频生成][3D一致性生成] 本文提出 WVD（World-consistent Video Diffusion），通过训练扩散模型联合建模 RGB 图像和 XYZ 图像（编码全局3D坐标），实现了显式3D约束下的多视角一致性视频生成，并通过灵活的 inpainting 策略统一了单图3D重建、多视角立体、相机控制生成等多种任务。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "3D一致性生成"
  - "扩散模型"
  - "XYZ图像"
  - "多视角合成"
  - "深度估计"
---

# World-Consistent Video Diffusion with Explicit 3D Modeling

**会议**: CVPR 2025  
**arXiv**: [2412.01821](https://arxiv.org/abs/2412.01821)  
**代码**: [https://zqh0253.github.io/wvd](https://zqh0253.github.io/wvd)  
**领域**: 视频生成  
**关键词**: 3D一致性生成, 扩散模型, XYZ图像, 多视角合成, 深度估计

## 一句话总结

本文提出 WVD（World-consistent Video Diffusion），通过训练扩散模型联合建模 RGB 图像和 XYZ 图像（编码全局3D坐标），实现了显式3D约束下的多视角一致性视频生成，并通过灵活的 inpainting 策略统一了单图3D重建、多视角立体、相机控制生成等多种任务。

## 研究背景与动机

**领域现状**：扩散模型在图像和视频生成上取得了巨大成功，多视角扩散模型通过注意力机制隐式地学习3D一致性。相机控制方法（如 CameraCtrl、MotionCtrl）通过 camera ray map 条件注入来控制视角。

**现有痛点**：（1）隐式方法缺乏显式3D一致性保证，即使训练数据量很大，仍会出现3D不一致的伪影；（2）依赖 camera ray 输入难以扩展到大规模数据，因为不同数据集间的相机表示存在根本性歧义，需要复杂的归一化处理；（3）显式3D方法（如体渲染）受限于架构约束，难以扩展到复杂数据。

**核心矛盾**：要在生成模型中实现3D一致性，传统的隐式方法（注意力跨帧学习）不可靠，而显式3D方法（体渲染等）又与现有2D Transformer 架构不兼容。

**本文目标**：设计一种既能提供显式3D监督、又与现有 DiT 架构兼容的方法。

**切入角度**：作者提出用 XYZ 图像表示3D几何——每个像素记录其全局3D坐标。XYZ 图像与 RGB 图像具有相同的形状，天然兼容2D Transformer 架构。

**核心 idea**：将3D场景表示为 RGB + XYZ 的"6D视频"，训练 DiT 联合扩散这两种模态，从而在生成过程中同时产出外观和几何。

## 方法详解

### 整体框架

WVD 的输入是一组6D视频——每个视角包含一张 RGB 图像和一张 XYZ 图像（编码全局3D坐标）。训练时，将 RGB 和 XYZ 分别通过预训练 VAE 编码到隐空间，沿通道维度拼接后进行联合扩散去噪。推理时，通过灵活的 inpainting 策略——将已知模态替换为 ground truth 来实现条件生成——支持多种下游任务。

### 关键设计

1. **XYZ 图像表示**:

    - 功能：将3D几何信息编码为与 RGB 兼容的图像格式
    - 核心思路：将点云通过归一化（居中+缩放到[-1,1]）和光栅化（投影到相机平面）转换为 XYZ 图像 $\boldsymbol{x}^{\text{XYZ}} = \mathcal{R}(\mathcal{N}(X), X, C)$。XYZ 图像与 RGB 图像同形状，每个像素值代表全局3D坐标而非颜色。两个不同视角中具有相同 XYZ 值的像素在3D空间中对应同一点，直接提供显式像素对应关系。
    - 设计动机：点云的非结构化特征（$\mathbb{R}^{N \times 3}$）与 DiT 不兼容；XYZ 图像保持了结构化的2D格式，可直接用预训练 VAE 编码，且消除了对额外相机参数输入的需求。

2. **RGB-XYZ 联合扩散**:

    - 功能：同时生成外观和3D几何
    - 核心思路：将 RGB 和 XYZ 的 VAE 隐码沿通道维拼接为 $\boldsymbol{z}_n = [\mathcal{E}(\boldsymbol{x}_n^{\text{RGB}}); \mathcal{E}(\boldsymbol{x}_n^{\text{XYZ}})] \in \mathbb{R}^{L \times 2D}$，在此基础上执行标准扩散训练。由于 XYZ 图像已归一化到[-1,1]，可直接使用预训练 VAE 而无需额外微调。对于图像条件生成，在每个训练步去除条件图像上的噪声即可。
    - 设计动机：通道拼接设计可以直接微调预训练的图像/视频扩散模型，大幅提高训练效率。联合建模使 XYZ 的显式3D对应关系能够反过来约束 RGB 的多视角一致性。

3. **Post Optimization 与灵活推理**:

    - 功能：从预测的 XYZ 图像恢复精确相机参数和深度图，并支持多种下游任务
    - 核心思路：对预测的 XYZ 图像执行梯度优化的重投影损失 $\min_{P,K,\boldsymbol{d}} \sum_{u,v} \|\tilde{\boldsymbol{x}}^{\text{XYZ}}_{u,v} - \hat{\boldsymbol{x}}^{\text{XYZ}}_{u,v}\|^2$，恢复相机位姿、内参和深度图。推理时利用 inpainting 策略实现任务切换：（a）提供 RGB 估计 XYZ → 单目/多目深度估计；（b）提供 XYZ 生成 RGB → 相机控制视频生成；（c）联合生成 → 单图3D重建。
    - 设计动机：联合分布 $P(\text{RGB}, \text{XYZ})$ 自然支持条件分布估计，一个模型即可统一多种3D任务，无需分别训练。

### 损失函数 / 训练策略

使用标准的扩散模型去噪损失（预测噪声或预测干净数据），在 RGB 和 XYZ 的拼接隐码上操作。训练数据混合了 RealEstate10K、ScanNet、MVImgNet、CO3D 和 Habitat，涵盖物体中心和场景中心分布。模型20亿参数，使用 AdamW 优化器，学习率 $3 \times 10^{-4}$，64块 A100 训练约两周。

## 实验关键数据

### 主实验

| 方法 | FID↓ | KPM↑ | FC↑ |
|---|---|---|---|
| CameraCtrl | 12.1 | 88.6 | 94.0 |
| MotionCtrl | 12.9 | 68.6 | 94.6 |
| WVD | 15.8 | **95.8** | **95.4** |
| WVD w/o XYZ | 18.3 | 72.3 | 95.0 |

在单图3D生成任务中，WVD 的 Key Points Matching（多视角一致性指标）远超基线，达到95.8%。

### 消融实验

| 配置 | FID↓ | KPM↑ | FC↑ |
|---|---|---|---|
| WVD（完整） | 15.8 | 95.8 | 95.4 |
| WVD w/o XYZ | 18.3 | 72.3 | 95.0 |

移除 XYZ 联合学习后，KPM 从95.8%骤降至72.3%，图像质量 FID 也从15.8恶化到18.3，充分验证了显式3D监督的关键作用。

深度估计结果：

| 方法 | NYU-v2 Rel↓ | BONN Rel↓ |
|---|---|---|
| DUSt3R-224 | 10.3 | 11.1 |
| DUSt3R-512* | 6.5 | 8.1 |
| WVD (256) | 9.7 | **7.0** |

在 BONN 基准上以256分辨率训练的 WVD 超越了所有方法包括512分辨率的 DUSt3R。

### 关键发现

- 联合学习 XYZ 是核心——消融表明移除 XYZ 后多视角一致性大幅下降
- WVD 作为生成模型进行深度估计具有竞争力，因为联合采样一致的周围视角使深度预测更具3D基础
- 相机控制生成通过"估计3D → 重投影 → inpainting"的管线实现，不需要在训练时显式加入相机条件
- 合成的点云可作为"空间记忆"逐步扩展，实现长视频的一致性生成

## 亮点与洞察

1. **XYZ 图像是绝妙的设计**：将非结构化的3D几何问题转化为结构化的图像生成问题，巧妙复用了强大的2D生成基础设施
2. **消除相机输入**：不再需要 camera ray map 作为条件，避免了跨数据集相机标准化的复杂工程
3. **一个模型统一多任务**：深度估计、新视角合成、相机控制生成、3D重建都通过 inpainting 策略实现
4. **生成式深度估计的新范式**：通过联合生成多视角来获得深度，比单张图像回归更具几何合理性

## 局限与展望

- 目前仅在静态数据集上训练，无法处理动态场景（4D）
- 未引入置信度图，难以处理无界或户外场景
- 分辨率限制在256×256，距离实用还有距离
- 未来可将 XYZ 替换为其他模态（光流、splatter 图像）扩展更多任务

## 相关工作与启发

- 与 DUSt3R 的关系：DUSt3R 直接回归点云，WVD 通过生成建模学习分布，理念互补
- 与 CAT3D 的关系：CAT3D 使用 camera ray 条件，WVD 用 XYZ 图像替代，更简洁
- 启发1：将3D表示"图像化"是连接2D和3D的有效桥梁
- 启发2：联合建模互补模态可以让模型学到更好的表示

## 评分

- **新颖性**: 8/10 — XYZ 图像联合扩散的想法简洁优雅，具有开创性
- **实验充分度**: 7/10 — 覆盖多个任务但缺少与更多近期方法的对比
- **写作质量**: 8/10 — 方法描述清晰，框架图直观
- **价值**: 8/10 — 提出了迈向3D基础模型的可行路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] GEN3C: 3D-Informed World-Consistent Video Generation with Precise Camera Control](gen3c_3d-informed_world-consistent_video_generation_with_precise_camera_control.md)
- [\[CVPR 2026\] Geometry-as-context: Modulating Explicit 3D in Scene-consistent Video Generation to Geometry Context](../../CVPR2026/video_generation/geometry-as-context_modulating_explicit_3d_in_scene-consistent_video_generation_.md)
- [\[CVPR 2025\] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)

</div>

<!-- RELATED:END -->
