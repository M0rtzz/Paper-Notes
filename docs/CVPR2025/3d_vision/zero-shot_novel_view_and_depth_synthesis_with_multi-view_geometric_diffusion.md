---
title: >-
  [论文解读] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion
description: >-
  [CVPR 2025][3D视觉][新视角合成] MVGD提出了一种基于像素级扩散的多视图几何框架，无需中间3D表示即可从任意数量的已知视角图像直接生成新视角的图像和尺度一致的深度图，在6000万+多视图样本上训练实现多项SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - 新视角合成
  - 深度估计
  - 扩散模型
  - 多视图几何
  - 多任务学习
---

# MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion

**会议**: CVPR 2025  
**arXiv**: [2501.18804](https://arxiv.org/abs/2501.18804)  
**代码**: [项目页面](https://mvgd.github.io/)  
**领域**: 3D视觉  
**关键词**: 新视角合成, 深度估计, 扩散模型, 多视图几何, 多任务学习

## 一句话总结

MVGD提出了一种基于像素级扩散的多视图几何框架，无需中间3D表示即可从任意数量的已知视角图像直接生成新视角的图像和尺度一致的深度图，在6000万+多视图样本上训练实现多项SOTA。

## 研究背景与动机

从稀疏带位姿图像重建3D场景是核心问题，现有方法主要依赖中间3D表示（NeRF、3DGS、体素网格）来保证多视图一致性。

现有方法的局限：
- **基于NeRF/3DGS的方法**：需要构建显式或隐式的3D场景表示，泛化能力受限于输入视图分布
- **扩散模型方法**：用于新视角合成时难以保证多视图一致性（缺少中间3D表示的约束）
- **深度估计**：单帧方法无法估计绝对尺度，多帧方法在跨数据集训练时面临尺度异质性问题
- **训练数据多样性**：不同数据集具有不同的标定方式（度量/非度量尺度）、不同的深度标注密度、不同的场景类型

核心动机：能否训练一个统一的扩散模型，直接从像素级生成多视图一致的图像和深度，而不依赖任何中间3D表示？

## 方法详解

### 整体框架

MVGD基于RIN（Recurrent Interface Networks）高效Transformer架构实现像素级扩散，不需要潜在自编码器。输入$N$个条件视图经图像编码器和射线编码器生成场景tokens，结合目标相机的射线嵌入和可学习任务嵌入，通过扩散过程直接生成目标视角的图像或深度图。

### 关键设计

**设计一：场景尺度归一化（SSN） — 统一跨数据集的尺度**

- **功能**：自动提取场景尺度并注入扩散过程，生成多视图一致的度量深度图
- **核心思路**：将所有条件相机的外参表示为相对于目标相机的位姿$\tilde{T}_c^n = T_c^n T_t^{-1}$，定义场景尺度$s = \max\{|\tilde{x}|, |\tilde{y}|, |\tilde{z}|\}$为最大绝对平移分量；所有平移向量和深度值均除以$s$归一化，推理时将生成的深度乘以$s$还原
- **设计动机**：跨数据集训练时不同标定方式导致尺度差异巨大。SSN通过相对位姿归一化实现平移和旋转不变性，使模型在统一的尺度空间学习，确保生成深度与条件相机几何一致

**设计二：可学习任务嵌入 — 统一的多任务生成**

- **功能**：使单一模型同时生成图像和深度图，且支持在有/无深度标注的数据上联合训练
- **核心思路**：使用可学习的任务嵌入$E^{task} \in \mathbb{R}^{D_{task}}$引导扩散过程朝向特定模态（RGB或Depth），附加到预测tokens中。对RGB任务使用L2损失，对深度任务使用L1损失（仅在有效ground-truth像素上）
- **设计动机**：简单的RGB-D联合生成会限制训练数据集只能使用有稠密深度的数据。条件化latent tokens会分离外观和几何先验。任务嵌入方案既能混合训练又保持共享的隐式3D表示

**设计三：RIN像素级扩散 + Raymap条件化 — 高效多视图几何推理**

- **功能**：在像素空间高效实现扩散，支持任意数量的条件视图（可达100+）
- **核心思路**：使用固定数量$L$的latent tokens进行自注意力计算，通过交叉注意力与输入/输出tokens交互。Raymap（射线原点+方向的Fourier编码）同时用于增强条件视图的空间位置信息和指定新视角。增量多视图生成策略维护历史生成图像作为额外条件
- **设计动机**：RIN将计算复杂度与输入token数解耦（$O(L^2)$而非$O(N^2)$），允许高效处理大量条件视图。像素级扩散避免了自编码器的细节损失和对稠密网格输入的要求

### 损失函数

RGB任务使用L2损失，深度任务使用L1损失（仅在有效ground-truth像素上），均以标准DDPM噪声预测方式训练。深度使用log尺度参数化$P_D = 2(\log(\frac{D}{s \cdot d_{min}}) / \log(\frac{d_{max}}{d_{min}})) - 1$，范围$d_{min}=0.1, d_{max}=200$。

## 实验关键数据

### 主实验：2视图新视角合成

| 方法 | RE10K PSNR ↑ | RE10K SSIM ↑ | RE10K LPIPS ↓ | ACID PSNR ↑ |
|------|-------------|-------------|--------------|-------------|
| PixelNeRF | 20.43 | 0.589 | 0.550 | 20.97 |
| MuRF | 26.10 | 0.858 | 0.143 | 28.09 |
| PixelSplat | 25.89 | 0.858 | 0.142 | 28.14 |
| MVSplat | 26.39 | 0.869 | 0.128 | 28.25 |
| **MVGD** | **28.41** | **0.891** | **0.107** | **29.98** |

### 消融实验：训练数据与模型扩展

| 配置 | PSNR ↑ | 说明 |
|------|--------|------|
| 基础（256 latents） | 基线 | 标准配置 |
| 去除动态数据集 | 下降 | 多样性重要 |
| 不用SSN | 下降 | 深度尺度不一致 |
| 512 latents（增量微调） | 提升 | 节省70%训练时间 |

### 关键发现

- 在RealEstate10K上PSNR比MVSplat提升2.02dB，比PixelSplat提升2.52dB
- 可处理100+条件视图且不增加计算复杂度（固定latent tokens数量）
- 增量微调策略使更大模型的训练时间减少70%
- 联合训练图像和深度生成促进了隐式几何理解，提升了新视角合成质量
- 在ScanNet多视图立体和视频深度估计上也达到SOTA

## 亮点与洞察

1. **无中间3D表示的端到端生成**：证明了扩散模型可以隐式学习多视图几何一致性
2. **大规模异构训练**：6000万+样本覆盖驾驶/室内/机器人/合成等多种场景，展现强大的零样本泛化能力
3. **增量模型扩展策略**：通过latent tokens复制+微调实现高效的模型规模化

## 局限与展望

- 不建模动态物体，尽管训练数据中包含动态场景
- 当前分辨率限制在256像素（最长边），更高分辨率需更多计算
- 未来可扩展至视频预测、场景编辑等下游任务

## 相关工作与启发

- **PixelSplat/MVSplat**：基于3DGS的泛化新视角合成，需要显式的3D表示
- **CAT3D/Reconfusion**：扩散+3D重建管线，依赖中间表示保证多视图一致性
- **RIN**：高效Transformer架构，计算量与输入大小解耦
- 启发：在足够多样化的训练数据下，扩散模型本身可以学习隐式的3D几何推理能力

## 评分

⭐⭐⭐⭐⭐ — 真正的系统性工作：从架构设计（RIN像素扩散）、训练策略（SSN + 多任务嵌入 + 60M数据）到扩展策略（增量微调）均有创新。在多个基准上全面刷新SOTA，且方法优雅简洁。

<!-- RELATED:START -->

## 相关论文

- [MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [DiffPortrait360: Consistent Portrait Diffusion for 360° View Synthesis](diffportrait360_consistent_portrait_diffusion_for_360_view_synthesis.md)
- [MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)
- [Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)

<!-- RELATED:END -->
