---
title: >-
  [论文解读] SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models
description: >-
  [CVPR 2025][图像分割][草图特征表示] 提出 SketchFusion，通过将 CLIP 视觉特征动态注入 Stable Diffusion 的去噪过程以互补 SD 的高频偏置和草图特征缺陷，结合自适应多尺度特征聚合，首次实现了基础模型时代的通用草图特征表示，在检索、识别、分割和对应学习四项任务上均达 SOTA。
tags:
  - CVPR 2025
  - 图像分割
  - 草图特征表示
  - 基础模型融合
  - 扩散模型
  - CLIP
  - 频域分析
---

# SketchFusion: Learning Universal Sketch Features through Fusing Foundation Models

**会议**: CVPR 2025  
**arXiv**: [2503.14129](https://arxiv.org/abs/2503.14129)  
**代码**: 无（项目页面待发布）  
**领域**: 分割/草图理解  
**关键词**: 草图特征表示, 基础模型融合, Stable Diffusion, CLIP, 频域分析

## 一句话总结

提出 SketchFusion，通过将 CLIP 视觉特征动态注入 Stable Diffusion 的去噪过程以互补 SD 的高频偏置和草图特征缺陷，结合自适应多尺度特征聚合，首次实现了基础模型时代的通用草图特征表示，在检索、识别、分割和对应学习四项任务上均达 SOTA。

## 研究背景与动机

草图（sketch）因其抽象性、稀疏性和跨模态特性，需要与自然图像根本不同的特征表示。尽管基础模型（SD、CLIP、DINO）在各类视觉任务上表现出色，但其对草图理解的有效性尚未充分探索。

作者通过系统性先导实验揭示了 SD 作为草图特征提取器的两个根本局限：(1) SD 从抽象稀疏的草图中提取的特征远不如从照片中提取的，因为 SD 的预训练主要基于自然图像；(2) 频域分析表明 SD 的 UNet 存在固有的高频偏置——系统性增强高频分量（边缘细节）同时抑制低频分量（整体语义结构），这对需要捕获整体语义的密集预测任务（如分割）尤为不利。

关键洞察：SD 特征空间感知强但语义不准，CLIP 特征语义准确但空间稀疏。两者互补——CLIP 恰好提供了 SD 缺失的低频语义成分。

## 方法详解

### 整体框架

SketchFusion 保持 SD 和 CLIP 模型冻结，仅训练三个轻量组件：(1) 1D 卷积层将 CLIP 视觉特征注入 SD UNet 各层；(2) ResNet 聚合网络统一多尺度特征；(3) 分支权重自动选择最优特征组合。不同下游任务使用不同任务损失训练这些组件。

### 关键设计1：CLIP 特征注入

**功能**：将 CLIP 的语义信息注入 SD 去噪过程的各层，补偿 SD 对草图的特征提取缺陷。

**核心思路**：从 CLIP 视觉编码器倒数第二层提取 patch 特征 $f_\mathbf{v} \in \mathbb{R}^{h/p \times w/p \times d}$，通过可学习 1D 卷积 $\mathcal{C}(\cdot)$ 调整维度后与 SD UNet 各上采样层的中间特征相加：$\hat{f}_\mathbf{u}^n = f_\mathbf{u}^n + \mathcal{C}(f_\mathbf{v})$。注入在所有时间步和所有层同时进行。

**设计动机**：CLIP 视觉和文本嵌入天然对齐，CLIP 视觉特征提供了比文本 prompt 更丰富的语义信息。注入多层使 SD 在去噪的各阶段都能利用 CLIP 的语义指导。1D 卷积仅做维度适配，保持计算开销极低。PCA 分析证实注入后的特征同时包含了 SD 的高频空间细节和 CLIP 的低频语义成分。

### 关键设计2：动态特征聚合

**功能**：自动选择 SD UNet 不同层的最优特征组合，消除手动层选择的需求。

**核心思路**：从 UNet 前三个上采样层提取 CLIP 增强后的特征 $\{\hat{f}_\mathbf{u}^n\}_{n=1}^3$，通过三个 ResNet 块将它们统一到相同分辨率 $60 \times 60 \times d$，然后使用可学习权重 $\{\alpha_n\}$ 进行加权求和得到最终特征图。

**设计动机**：不同层捕获不同语义粒度的特征——浅层精细（适合对应学习），深层粗糙（适合识别）。手动选择最优层对不同任务需反复调试。自动加权让模型自适应地确定各层贡献。

### 关键设计3：统一的多任务适配

**功能**：同一特征提取框架适配检索、识别、分割、对应四类任务。

**核心思路**：全局池化特征 + triplet loss 用于检索/识别；dense 特征 + 逐像素损失用于分割和对应学习。所有任务共享相同的 SD+CLIP 特征提取器，仅训练注入层、聚合网络和分支权重。

**设计动机**：现有方法为每类任务设计专门架构，本文证明了统一特征表示的可行性。

### 损失函数

任务特定：检索和识别使用 triplet loss；分割使用交叉熵；对应学习使用像素级匹配损失。所有任务仅训练轻量组件，SD 和 CLIP 保持冻结。

## 实验关键数据

### 类别级零样本草图检索（ZS-SBIR）

| 方法 | Sketchy mAP@200 | TU-Berlin mAP@all | Quick,Draw! mAP@all |
|------|-----------|-----------|-----------|
| B-CLIP | 0.250 | 0.228 | 0.080 |
| B-SD | 0.558 | 0.510 | 0.179 |
| SD-PL (SOTA) | 0.746 | 0.680 | 0.231 |
| **SketchFusion** | **0.761** (+1.5%) | **0.695** (+1.5%) | **0.242** (+1.1%) |

### 草图分割（Sketch Segmentation）

| 方法 | SketchSeg-150K mIoU |
|------|-----|
| B-SD | 35.72 |
| SD-PL | 47.89 |
| **SketchFusion** | **77.31** (+29.42%) |

### 草图-照片对应学习

| 方法 | PCK@0.1 |
|------|---------|
| B-SD | 33.12 |
| **SketchFusion** | **54.34** (+21.22%) |

### 关键发现

- 分割任务提升最为惊人（+29.42%），验证了低频语义补偿对密集预测任务的关键作用。
- 直接微调 SD+CLIP（B-Finetuning）反而严重退化（mAP 0.120 vs 0.761），证明了保持冻结+轻量注入策略的正确性。
- 简单拼接 SD 和 CLIP 特征（B-SD+CLIP）有提升但远不如注入策略（0.588 vs 0.761），说明在去噪过程中注入比后处理融合更有效。
- 频域分析清晰展示了 SD 的高频偏置和 CLIP 的低频互补，为特征融合提供了理论依据。

## 亮点与洞察

1. **频域分析视角**：首次从频域角度分析 SD 对草图的局限性，发现高频偏置问题并用 CLIP 的低频语义互补。
2. **通用性**：一个框架、一套特征表示跨越检索+识别+分割+对应四类任务，且全部达到 SOTA。
3. **效率**：保持两个大模型冻结，仅训练轻量 1D 卷积和聚合网络，避免了灾难性遗忘和高昂微调成本。

## 局限与展望

- 推理需同时运行 SD 和 CLIP 两个大模型，内存和计算开销较大。
- 使用空 prompt 而非类别特定 prompt 可能限制了文本语义的利用。
- 仅在草图领域验证，对其他稀疏视觉输入（如线稿、medical image 等）的通用性待探索。
- SD v2.1 为基础，更新版本（SDXL, SD3）可能有不同的频域特性。

## 相关工作与启发

- **SD-PL**：此前 SOTA 草图特征方法，使用 SD 单一模型+手动层选择，本文通过 CLIP 融合和自适应聚合全面超越。
- **Vision Fusion**：SD+DINO 等混合模型方法启发了本文的互补融合策略。
- **频域分析**：经典 CV 文献的分析工具被引入基础模型分析，揭示了 UNet 的内在偏置。

## 评分

⭐⭐⭐⭐ — 先导实验分析深入（频域偏置发现），方法设计优雅（注入而非微调），四项任务全部 SOTA。分割 +29.42% 的提升幅度令人印象深刻。对 SD 特征局限性的分析对更广泛的社区也有参考价值。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [\[CVPR 2025\] Uni4D: Unifying Visual Foundation Models for 4D Modeling from a Single Video](uni4d_unifying_visual_foundation_models_for_4d_modeling_from_a_single_video.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](../../CVPR2026/segmentation/occsam_bench_occlusion_robustness_segmentation.md)
- [\[CVPR 2025\] Universal Domain Adaptation for Semantic Segmentation](universal_domain_adaptation_for_semantic_segmentation.md)

<!-- RELATED:END -->
