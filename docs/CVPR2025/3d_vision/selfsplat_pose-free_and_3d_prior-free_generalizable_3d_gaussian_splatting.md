---
title: >-
  [论文解读] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] SelfSplat 提出了一个无需相机位姿和 3D 预训练先验的可泛化 3D 高斯溅射框架，通过将自监督深度/位姿估计与 3D-GS 表示统一，配合匹配感知位姿网络和深度细化模块，在 RealEstate10K、ACID 和 DL3DV 数据集上显著超越现有无位姿方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 3D高斯溅射
  - 自监督学习
  - 无位姿重建
  - 深度估计
  - 新视图合成
---

# SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2411.17190](https://arxiv.org/abs/2411.17190)  
**代码**: https://gynjn.github.io/selfsplat/  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 自监督学习, 无位姿重建, 深度估计, 新视图合成

## 一句话总结

SelfSplat 提出了一个无需相机位姿和 3D 预训练先验的可泛化 3D 高斯溅射框架，通过将自监督深度/位姿估计与 3D-GS 表示统一，配合匹配感知位姿网络和深度细化模块，在 RealEstate10K、ACID 和 DL3DV 数据集上显著超越现有无位姿方法。

## 研究背景与动机

**领域现状**：NeRF 和 3D Gaussian Splatting（3D-GS）在 3D 重建和新视图合成方面取得了巨大成功。为了避免逐场景优化，前馈式可泛化 3D 重建模型（如 pixelSplat、MVSplat）可以在单次前向传播中预测 3D 几何。但这些方法仍然需要精确的相机位姿作为输入，限制了在"野外"（in-the-wild）数据上的应用。

**现有痛点**：近期的无位姿方法存在各种限制：FlowCAM 依赖容易出错的预训练光流模型；DBARF 需要逐场景微调，计算昂贵；CoPoNeRF 推理时无位姿但训练时仍需 GT 位姿监督；基于 NeRF 的方法（FlowCAM、DBARF）推理成本高。此外，3D-GS 作为显式表示对 3D 位置的微小误差高度敏感——稍有偏差就会破坏多视角一致性，使同时预测高斯属性和相机位姿成为极具挑战的任务。

**核心矛盾**：(1) 无位姿设置本质上是病态问题——缺乏 GT 数据和学到的几何信息，同时要求准确的 3D 重建；(2) 3D-GS 的显式表示对位姿误差极度敏感，而准确的位姿估计又需要好的 3D 表示作为支撑，形成鸡蛋-鸡的困境。

**本文目标**：设计一个完全自监督的框架，从无位姿单目视频中学习可泛化的 3D 重建，无需预训练 3D 先验模型或场景特定微调。

**切入角度**：作者观察到自监督深度/位姿估计和 3D-GS 可以相互促进——自监督学习的几何一致性约束能指导 3D 高斯的定位，而 3D-GS 的高质量视图合成能力能反过来提升位姿估计精度。关键在于如何有效地统一两个框架并解决它们结合时的问题。

**核心 idea**：将自监督深度/位姿估计与像素对齐的 3D-GS 统一到一个框架中，并通过匹配感知位姿网络和位姿感知深度细化模块解决纯自监督方法中位姿不准确和深度不一致的问题。

## 方法详解

### 整体框架

SelfSplat 输入无位姿的三帧图像 $(I_{c_1}, I_t, I_{c_2})$，包含四个核心组件：(1) 多视角+单目编码器提取特征；(2) 融合+密集预测模块生成深度图和高斯属性；(3) 匹配感知位姿网络估计帧间相机位姿；(4) 位姿感知深度细化模块利用估计位姿优化深度一致性。最终用估计的位姿将各视角的高斯变换到统一坐标系，通过光栅化渲染并用重投影损失+渲染损失联合训练。

### 关键设计

1. **多视角与单目双编码器架构**:

    - 功能：结合跨视角匹配信息和鲁棒的单目深度先验
    - 核心思路：多视角分支使用权重共享的 ResNet 提取 4x 下采样特征，然后通过 6 个 Swin Transformer 块进行跨视角自/交叉注意力，得到跨视角感知特征 $F^{\text{mv}}$。单目分支使用 CroCo v2 权重初始化的 ViT，独立处理每张图像得到鲁棒的单目特征 $F^{\text{mono}}$。两种特征通过 DPT 模块在多尺度上融合——先将多视角特征 4x 下采样匹配分辨率，然后用 CNN 金字塔 + reassemble/fusion blocks 生成密集预测。关键地，对目标视图 $I_t$ 不生成高斯属性，迫使网络学习新视图泛化
    - 设计动机：多视角匹配在遮挡、无纹理和反射区域表现差，单目特征提供互补的鲁棒估计；使用 CroCo v2（非 DepthAnything）避免引入 3D 先验，保持完全自监督

2. **匹配感知位姿估计网络**:

    - 功能：利用跨视角上下文信息提升位姿估计精度
    - 核心思路：使用带交叉注意力块的 2D U-Net 处理三帧图像，提取匹配感知特征 $F^{\text{ma}}_k \in \mathbb{R}^{H \times W \times 3}$。将匹配特征、原始图像和相机内参 ray embedding $E^{\text{int}} = K^{-1}p(x,y)$ 拼接后送入 PoseNet，估计每对图像间的相对变换 $T_{c_1 \to t} \in SE(3)$。与传统 CNN 位姿网络不同，交叉注意力使网络能利用多视角间的对应关系信息
    - 设计动机：纯 CNN 位姿网络缺乏跨视角交互，难以建立精确的几何对应关系；加入匹配感知模块提供额外的跨视角知识

3. **位姿感知深度细化模块**:

    - 功能：利用估计位姿信息优化跨视角深度一致性
    - 核心思路：初始深度估计 $\tilde{D}_{c_1}, \tilde{D}_{c_2}$ 在不同视角间可能不一致，导致高斯重叠和重建质量下降。深度细化模块是一个带交叉注意力的轻量级 2D U-Net，输入当前深度、原图和 Plücker ray embedding 编码的估计位姿 $E^{\text{ext}}(T_{c_1 \to t}) \in \mathbb{R}^{H \times W \times 6}$，输出残差深度 $\Delta D_k$，最终深度 $D_k = \tilde{D}_k + \Delta D_k$。位姿信息为深度细化提供了周围视角的空间关系，使不同视角的深度估计更加一致
    - 设计动机：不同视角的独立深度估计缺乏一致性，引入位姿作为额外上下文信息能解决这一问题；残差学习使细化更稳定

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{proj}} + \lambda_2 \mathcal{L}_{\text{ren}}$，其中重投影损失 $\mathcal{L}_{\text{proj}}$ 使用 SSIM+L1 的组合计算投影图像与目标图像的误差，渲染损失 $\mathcal{L}_{\text{ren}}$ 计算 3D-GS 渲染图像与输入图像的 SSIM+L2 误差。对 $I_t$ 使用渲染深度（而非估计深度）计算重投影损失以保持尺度一致。假设相机内参已知（从传感器元数据获取）。

## 实验关键数据

### 主实验

**RealEstate10K 新视图合成（Average）**:

| 方法 | 无位姿 | 无3D先验 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|--------|---------|-------|-------|--------|
| DBARF | ✓ | ✗ | 12.57 | 0.494 | 0.474 |
| FlowCAM | ✓ | ✗ | 22.29 | 0.711 | 0.313 |
| CoPoNeRF | ✗(训练需GT) | ✓ | 21.03 | 0.693 | 0.256 |
| **SelfSplat** | **✓** | **✓** | **24.22** | **0.813** | **0.188** |

**ACID 新视图合成（Average）**:

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| FlowCAM | 25.59 | 0.721 | 0.294 |
| **SelfSplat** | **26.71** | **0.801** | **0.196** |

### 消融实验

论文进行了详细的消融实验验证各组件贡献（匹配感知位姿网络、深度细化模块、单目编码器等），具体数据在完整论文的消融表中。

### 关键发现

- SelfSplat 在最苛刻的设置（无位姿 + 无 3D 先验 + 无微调）下全面超越所有基线
- 在 RE10K 上PSNR 比最强的无位姿方法 FlowCAM 高 1.93 dB，LPIPS 降低 40%
- 具有强跨数据集泛化能力：在 RE10K 上训练，在 DL3DV 上测试仍表现良好
- 大baseline（较大视角变化）场景下的优势更加明显

## 亮点与洞察

- "自监督 + 3D-GS 互惠"的设计思路是核心亮点：自监督的几何一致性约束帮助定位高斯、3D-GS 的高质量渲染帮助改善位姿估计，形成正反馈循环
- 深度细化模块利用位姿信息是一个巧妙设计——将位姿估计的输出反馈到深度估计中，实现了组件间的信息双向流动
- 完全自监督（不依赖 DepthAnything 等预训练 3D 模型）的设计使方法更具学术意义和通用性
- 基于 3D-GS 实现了快速推理，不需要像 NeRF 方法那样进行昂贵的体渲染

## 局限与展望

- 假设相机内参已知，限制了在完全未知设置下的应用
- 自监督深度估计的尺度歧义仍然存在
- 三帧输入的设置限制了可重建场景的范围
- 在极端视角变化或大量遮挡场景下性能仍有提升空间

## 相关工作与启发

- 将单目深度估计中的自监督范式（SfMLearner、Monodepth2）与 3D-GS 结合是一个自然但有效的方向
- CroCo 的跨视角补全预训练提供了有效的单目特征，避免了引入 3D 先验
- 匹配感知位姿网络的思路可应用到其他需要精确帧间对齐的任务

## 评分

- 新颖性：⭐⭐⭐⭐ — 自监督+3D-GS 的统一框架设计新颖，匹配感知位姿和位姿感知深度的双向信息流有深度
- 实验充分度：⭐⭐⭐⭐⭐ — 在三个大规模真实数据集上全面验证，含跨数据集泛化和详细消融
- 写作质量：⭐⭐⭐⭐ — 方法描述清晰，实验分析充分，与现有方法的对比公平全面
- 价值：⭐⭐⭐⭐ — 解决了 3D-GS 的一个重要限制（依赖位姿），推动了无位姿 3D 重建的发展

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] IncEventGS: Pose-Free Gaussian Splatting from a Single Event Camera](inceventgs_pose-free_gaussian_splatting_from_a_single_event_camera.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[NeurIPS 2025\] OnlineSplatter: Pose-Free Online 3D Reconstruction for Free-Moving Objects](../../NeurIPS2025/3d_vision/onlinesplatter_pose-free_online_3d_reconstruction_for_free-moving_objects.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)

<!-- RELATED:END -->
