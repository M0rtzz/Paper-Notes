---
title: >-
  [论文解读] SplatTalk: 3D VQA with Gaussian Splatting
description: >-
  [3D视觉] 提出SplatTalk，利用可泛化的3D Gaussian Splatting框架生成与LLM兼容的3D token，仅需多视角RGB图像即可实现零样本3D视觉问答，性能超越2D LMM方法并接近3D LMM。
tags:
  - 3D视觉
---

# SplatTalk: 3D VQA with Gaussian Splatting

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2503.06271](https://arxiv.org/abs/2503.06271)
- **代码**: [项目页面](https://splat-talk.github.io/)
- **领域**: 3D视觉
- **关键词**: 3D VQA, 3D Gaussian Splatting, 语言引导3D理解, LLM, 零样本

## 一句话总结

提出SplatTalk，利用可泛化的3D Gaussian Splatting框架生成与LLM兼容的3D token，仅需多视角RGB图像即可实现零样本3D视觉问答，性能超越2D LMM方法并接近3D LMM。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：语言引导的3D场景理解对于机器人、AR/VR等应用至关重要。现有方法面临三大挑战：

**3D数据稀缺**：相比2D，语言标注的3D数据极少，限制了3D LMM的发展

**依赖显式3D输入**：多数3D方法需要点云、网格等显式3D表示，难以适用于仅有图像的场景

**2D方法缺乏3D推理能力**：直接用多视角图像喂给2D LMM时，token按图像排列而非3D空间组织，无法回答需要跨视角三维推理的问题（如"门对面是什么？"）

**核心insight**：在将token输入LLM之前，先在3D空间中整合token，可以有效提升空间推理能力。

## 方法详解

### 整体框架

SplatTalk包含三个阶段：

**阶段1：特征自编码器训练**
- 从LLaVA-OV的视觉encoder+多模态projector提取高维visual token（这些token已与LLM输入空间对齐）
- 训练自编码器将高维、无界、稀疏的特征压缩到低维超球面空间

**阶段2：自监督3D-Language Gaussian Splatting训练**
- 基于FreeSplat框架扩展为语言高斯场学习
- 输入多视角RGB图像 → CNN提取多尺度特征 → 构建自适应代价体积预测深度 → 反投影为3D Gaussian triplets $(\boldsymbol{\mu}, \boldsymbol{\omega}, \boldsymbol{f})$
- Pixel-wise Triplet Fusion (PTF) 模块融合跨视角重叠高斯
- MLP解码器同时预测渲染参数和低维语言特征 $f$

**阶段3：3D VQA推理**
- 从3D Gaussians提取语言特征 → 解码器映射回高维空间 → 直接作为LLM的visual token输入

### 关键设计

**1. Visual Token选择**
选择多模态projector之后的visual token作为训练目标（而非encoder原始特征），因为projector之后的特征已与LLM潜在空间对齐，LLM可以直接理解和推理。

**2. 均值特征提取的理论分析**
将3D Gaussian的语言特征通过均值聚合（mean pooling）得到场景级token。论文提供了理论论证：均值特征提取编码了关于3D场景的整体概念信息。

**3. 熵自适应Token采样**
不均匀采样token，优先采样信息熵高的区域，在不增加训练开销的前提下提升3D VQA性能。

### 损失函数

自监督训练损失包含：
- RGB重建损失：$\mathcal{L}_{\text{rgb}}$（保证渲染质量）
- 语言特征重建损失：$\mathcal{L}_{\text{lang}}$（低维特征与自编码器压缩后的pseudo GT对齐）
- 可选LoRA微调损失：$\mathcal{L}_{\text{lora}}$（微调LLM提升VQA性能）

## 实验

### 主实验：ScanQA和SQA3D对比

| 方法 | 模态 | ScanQA CIDEr | ScanQA EM@1 | SQA3D EM@1 |
|------|------|------|------|------|
| ScanQA | PC | 64.9 | 21.1 | 47.2 |
| 3D-VisTA | PC | 69.6 | 22.4 | 48.5 |
| LEO | PC+I | 101.4 | 24.5 | 50.0 |
| LLaVA-OV | I | 50.0 | 15.6 | - |
| GPT-4V | I | 59.6 | - | - |
| **SplatTalk (Ours)** | **I** | **高于2D方法** | **高于2D方法** | **接近3D SOTA** |

### 消融实验

| 组件 | CIDEr变化 |
|------|------|
| 无自编码器压缩（直接用高维特征） | 显著下降，训练困难 |
| 无熵自适应采样 | 中等下降 |
| 仅用encoder特征（非projector后） | 严重下降，需重训projector |
| 无3D表示（纯2D多视角） | 空间推理能力明显不足 |

### 关键发现
1. **3D表示的必要性**：与纯2D方法相比，SplatTalk在需要跨视角空间推理的问题上优势明显（如"窗户对面是什么"）
2. **Projector后特征的重要性**：使用encoder原始特征需要重训projector，而projector后的visual token可直接兼容LLM
3. **零样本能力**：无需3D-语言标注数据，仅靠自监督训练即可实现有竞争力的3D VQA

## 亮点与洞察

1. **首个自监督3D Gaussian语言场零样本3D VQA方法**：打破了3D VQA依赖显式3D输入（点云）的限制
2. **"先在3D空间整合，再送入LLM"的范式**：比直接用多视角图像更能捕获空间关系
3. **实用性强**：仅需多视角RGB图像，无需深度、点云等额外输入

## 局限与展望

- 依赖FreeSplat的泛化3DGS质量，对稀疏视角和复杂场景的鲁棒性待验证
- 自编码器压缩可能丢失部分细粒度语义信息
- 推理时需要多视角图像输入，实时性受限

## 相关工作

- **3D LMM**: LEO, Chat-Scene, LL3DA
- **2D LMM**: LLaVA-OV, GPT-4V
- **3DGS语义**: LangSplat, ChatSplat
- **泛化3DGS**: FreeSplat

## 评分

- 新颖性: ⭐⭐⭐⭐ (3DGS+LLM的新颖组合)
- 技术深度: ⭐⭐⭐⭐ (理论分析+系统设计完整)
- 实验质量: ⭐⭐⭐⭐ (多benchmark全面对比)
- 实用价值: ⭐⭐⭐⭐ (仅需RGB图像，门槛低)

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)

<!-- RELATED:END -->
