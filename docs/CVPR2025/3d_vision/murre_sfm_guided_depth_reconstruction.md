---
title: >-
  [论文解读] Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
description: >-
  [CVPR 2025][3D视觉][多视角重建] 提出 Murre，一种新的多视角 3D 重建框架，通过将 SfM 稀疏点云注入扩散模型指导单目深度估计，绕过了传统 MVS 的多视角匹配步骤，在多种真实场景（室内、街景、航拍）上超越 SOTA。 领域现状：学习型 MVS 方法在低纹理区域和稀疏视角下效果差…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "多视角重建"
  - "SfM引导"
  - "单目深度估计"
  - "扩散模型"
  - "深度补全"
---

# Murre: Multi-view Reconstruction via SfM-guided Monocular Depth Estimation

**会议**: CVPR 2025  
**arXiv**: [2503.14483](https://arxiv.org/abs/2503.14483)  
**代码**: [https://zju3dv.github.io/murre/](https://zju3dv.github.io/murre/)  
**领域**: 3D视觉 / 多视角重建  
**关键词**: 多视角重建, SfM引导, 单目深度估计, 扩散模型, 深度补全

## 一句话总结

提出 Murre，一种新的多视角 3D 重建框架，通过将 SfM 稀疏点云注入扩散模型指导单目深度估计，绕过了传统 MVS 的多视角匹配步骤，在多种真实场景（室内、街景、航拍）上超越 SOTA。

## 研究背景与动机

**领域现状**：学习型 MVS 方法在低纹理区域和稀疏视角下效果差，且 3D 代价体消耗大量 GPU 显存。

**现有痛点**：MVS 隐式依赖多视角一致性，稀疏视角时匹配失败；单目深度估计虽不需要匹配但缺乏多视角一致性和度量信息。

**核心矛盾**：多视角一致性需要匹配，但匹配在难场景下不可靠；单目预测不需要匹配但不一致。

**核心 idea**：用 SfM 点云作为显式中间表示，将多视角信息注入单目深度扩散模型，兼得两者优势。

## 方法详解

### 整体框架

给定多视角图像：(1) SfM 重建稀疏点云；(2) 将点云投影为各视角的稀疏深度图并稠密化；(3) 稠密化深度图 + RGB 图像作为条件输入扩散模型预测度量深度；(4) TSDF 融合得到最终几何。

### 关键设计

1. **SfM 先验注入扩散模型**:

    - 功能：为单目深度估计提供多视角一致的度量信息
    - 核心思路：将 SfM 稀疏深度用 KNN 插值稠密化，计算每个像素到最近有值点的距离图作为置信度指标。稠密化深度图和距离图一起作为条件送入基于 Stable Diffusion V2 的深度扩散模型
    - 设计动机：SfM 点云是多视角信息的浓缩形式，天然提供度量尺度和场景显著结构

2. **深度归一化与尺度对齐**:

    - 功能：处理不同场景和视角间的深度范围差异
    - 核心思路：先过滤 SfM 深度的上下 2% 异常值，将范围扩展到 0.8×min 和 1.2×max，用此范围归一化 GT 深度用于训练。推理时用 RANSAC 线性回归将预测深度与 SfM 深度对齐
    - 设计动机：SfM 深度含异常值且只覆盖部分像素，需要稳健的归一化策略

3. **基于 Stable Diffusion 的深度估计**:

    - 功能：利用 2D 基础模型的强大先验实现泛化
    - 核心思路：从 SD V2 初始化，固定 VAE 仅微调 UNet。将深度复制为三通道经 VAE 编码器映射到 latent space，在 latent space 进行加噪和去噪
    - 设计动机：少量合成数据微调即可在多种真实场景中泛化

### 损失函数 / 训练策略

标准扩散去噪损失。使用 Detector-free SfM 处理弱纹理区域。训练数据包含合成场景。

## 实验关键数据

### 主实验

| 数据集 | 本文 vs SOTA |
|--------|-------------|
| DTU | 超越现有 MVS 和神经隐式方法 |
| ScanNet | 竞争性能 |
| Waymo | 超越单目方法 |
| UrbanScene3D | 超越 MVS |

### 关键发现

- SfM 引导显著提升深度一致性和度量准确性
- KNN 稠密化 + 距离图比直接使用稀疏深度更有效
- 在合成数据上训练即可泛化到多种真实场景

## 亮点与洞察

- SfM 作为多视角信息的"浓缩"表示注入单目模型，思路精巧
- 绕过了 3D 代价体，解决了显存和稀疏视角两大痛点
- 2D 基础模型的泛化能力通过 SfM 引导得到有效释放

## 局限与展望

- 依赖 SfM 的质量，SfM 失败时方法也会受影响
- 扩散模型的多步推理带来额外计算开销
- 动态场景需要额外处理

## 评分

- 新颖性：8/10 — SfM + 扩散深度的组合新颖
- 技术深度：8/10 — 归一化和对齐策略设计细致
- 实验充分度：9/10 — 五种场景类型验证
- 写作质量：8/10 — 方法描述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation](multi-view_reconstruction_via_sfm-guided_monocular_depth_estimation.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] MEt3R: Measuring Multi-View Consistency in Generated Images](met3r_measuring_multi-view_consistency_in_generated_images.md)

</div>

<!-- RELATED:END -->
