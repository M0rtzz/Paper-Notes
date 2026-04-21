---
title: >-
  [论文解读] Data Collection-Free Masked Video Modeling
description: >-
  [ECCV 2024][视频理解][自监督学习] 提出基于伪运动生成器（PMG）从静态图像递归生成伪运动视频，结合掩码视频建模（VideoMAE）进行自监督预训练，完全摆脱真实视频数据的采集成本和隐私/版权顾虑，甚至可用合成图像实现有效的视频Transformer预训练。
tags:
  - ECCV 2024
  - 视频理解
  - 自监督学习
  - 掩码视频建模
  - 伪运动视频
  - 合成数据
  - VideoMAE
---

# Data Collection-Free Masked Video Modeling

**会议**: ECCV 2024  
**arXiv**: [2409.06665](https://arxiv.org/abs/2409.06665)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 自监督学习, 掩码视频建模, 伪运动视频, 合成数据, VideoMAE

## 一句话总结

提出基于伪运动生成器（PMG）从静态图像递归生成伪运动视频，结合掩码视频建模（VideoMAE）进行自监督预训练，完全摆脱真实视频数据的采集成本和隐私/版权顾虑，甚至可用合成图像实现有效的视频Transformer预训练。

## 研究背景与动机

视频Transformer的预训练面临多重数据问题：

**采集成本高**：视频数据庞大，下载/存储/预处理极其耗费资源

**版权风险**：YouTube等平台视频默认禁止下载，Kinetics等数据集存在法律隐患

**隐私问题**：视频常包含人脸等个人可识别信息（PII）

**偏见与伦理**：大规模数据集可能包含国籍、性别、年龄等偏见

**数据可及性**：部分数据集仅对特定研究组开放

现有替代方案的不足：
- VPN（Perlin Noise视频）和SynAPT仍需真实视频配合使用
- MoSI（从图像生成伪运动）只适用于CNN架构，**无法有效训练ViT**
- 从未有人仅用合成图像成功预训练视频Transformer

## 方法详解

### 整体框架

1. PMG从静态图像生成伪运动视频
2. 用生成的视频训练VideoMAE（掩码率0.75）
3. 在下游动作识别任务上微调

核心假设：VideoMAE主要学习低层特征（如帧间patch对应关系），而非高层语义。因此只要伪运动视频中patch可追踪，就能有效训练。

### 伪运动生成器（PMG）

PMG通过**递归应用图像变换**生成视频：从变换集合中随机选择变换f和强度参数theta，递归应用于初始图像：I_{i+1} = f(I_i)，最终拼接为视频 V = [I_1; I_2; ...; I_T]。

### 8种候选图像变换

| 变换 | UCF101 | HMDB51 | 特点 |
|------|--------|--------|------|
| Identity（基线） | 72.7 | 35.6 | 无运动 |
| Sliding Window | 75.1 | 40.5 | 窗口随机移动 |
| **Zoom-in/out** | **81.2** | **44.5** | 窗口缩放 |
| Fade-in/out | 76.3 | 34.1 | 渐显/渐隐 |
| Affine | 80.5 | 43.2 | 仿射变换 |
| **Perspective** | **82.7** | **45.9** | 透视变换 |
| Color Jitter | 76.2 | 38.7 | 颜色抖动 |
| CutMix | 76.8 | 45.1 | 混合+滑动 |

关键发现：帧间patch可追踪的变换（Zoom/Affine/Perspective）效果好，仅改变颜色/亮度的变换在运动敏感数据集上效果差。

### 最优变换组合

经实验选定 **Zoom-in/out + Affine** 为最优组合（HMDB51: 51.8%），CutMix因帧间不连续反而降低性能。

### 视频级增强：Mixup

对生成的伪运动视频逐帧应用Mixup显著提升多样性：

| 增强方式 | HMDB51 | UCF101 |
|----------|--------|--------|
| 无增强 | 51.8 | 83.8 |
| **Mixup** | **55.9** | **87.3** |
| VideoMix | 53.0 | 85.2 |

### 与合成图像的结合

使用三类合成图像数据集：FractalDB（分形几何）、Shaders1k（OpenGL shaders）、Visual Atom（正弦波）。

### 训练策略

- 骨干：ViT-Base，掩码率0.75，2000 epochs
- 视频帧数：16帧，224x224分辨率
- 8x A100 GPU

## 实验关键数据

### 主实验：与现有方法对比

| 方法 | 数据源 | 数据量 | UCF101 | HMDB51 | Diving48 |
|------|--------|--------|--------|--------|----------|
| 从头训练 ViT-B | - | - | 51.4 | 18.0 | 17.9 |
| VideoMAE(FT data) | 真实视频 | - | 91.3 | 62.6 | 79.3 |
| VideoMAE(K400) | 真实视频 | 260k | 96.1 | 73.3 | - |
| MoSI(ViT-B) | 真实图像 | - | 48.0 | 27.3 | 14.2 |
| PPMA | 真实+合成视频 | 300k | 92.5 | 71.2 | 64.0 |
| **Ours(FT frames)** | **真实图像** | **-** | **87.3** | **55.9** | **68.3** |
| **Ours(PASS)** | **真实图像** | **100k** | **89.3** | **60.0** | **69.2** |
| **Ours(Shaders1k)** | **合成图像** | **100k** | **89.4** | **59.7** | **72.3** |

### 合成图像预训练

| 合成数据集 | 数据量 | UCF101 | HMDB51 |
|------------|--------|--------|--------|
| FractalDB | 100k | 78.1 | 41.1 |
| **Shaders1k** | **100k** | **89.6** | **59.7** |
| Visual Atom | 100k | 82.6 | 48.2 |

### PMG作为视频增强

| 预训练数据 | HMDB51 | UCF101 |
|------------|--------|--------|
| 仅真实视频 | 62.6 | 91.3 |
| 仅伪运动 | 55.9 | 87.3 |
| **真实+伪运动** | **64.6** | **92.2** |

### 关键发现

- **合成图像可完全替代真实数据**：Shaders1k的UCF101精度（89.4）超过用真实视频帧（87.3）
- 数据多样性比语义相关性更重要：PASS（无人物）效果媲美动作视频帧
- MoSI在ViT上完全失效（48.0 vs 我们的87.3），证明PMG是关键创新
- Diving48上Shaders1k（72.3）大幅超过FT data（68.3），运动密集任务合成数据更有优势
- PMG还可作为真实视频的数据增强手段（+2%提升）

## 亮点与洞察

1. **彻底解耦数据与预训练**：首次证明视频Transformer可仅用合成图像有效预训练
2. **揭示了VideoMAE的学习本质**：主要学习patch间的低层对应关系，而非高层语义
3. **PMG设计简洁有效**：仅递归应用图像变换，无需复杂的视频生成模型
4. **双重用途**：既可独立预训练（无数据采集），也可作为真实视频的增强

## 局限与展望

- 纯合成预训练与真实视频预训练仍有差距（UCF: 89.4 vs 96.1），留有较大提升空间
- 伪运动模式相对简单（仿射/缩放），无法模拟复杂人体动作和物体交互
- 仅验证了ViT-Base，未探索更大模型是否能进一步受益
- FractalDB和Visual Atom效果不佳，对合成图像的属性要求限制了适用范围

## 相关工作与启发

- MoSI仅适用CNN的限制促成了本工作对ViT友好设计的探索
- VideoMAE学习低层特征的特性被巧妙利用：既是此前被视为缺点的特性，也是本方法奏效的基础
- 与图像合成预训练的衔接为完全无真实数据预训练打开了可能

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次合成图像→视频Transformer预训练，开创新范式)
- 实验充分度: ⭐⭐⭐⭐⭐ (6个下游数据集+多合成/真实图像源+详尽消融)
- 写作质量: ⭐⭐⭐⭐ (问题动机充分，实验组织清晰)
- 价值: ⭐⭐⭐⭐ (实用意义大，但与真实数据差距仍存)

<!-- RELATED:START -->

## 相关论文

- [Text-Guided Video Masked Autoencoder](text-guided_video_masked_autoencoder.md)
- [Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition](masked_video_and_body-worn_imu_autoencoder_for_egocentric_action_recognition.md)
- [Web-Scale Collection of Video Data for 4D Animal Reconstruction](../../NeurIPS2025/video_understanding/web-scale_collection_of_video_data_for_4d_animal_reconstruction.md)
- [Towards Efficient General Feature Prediction in Masked Skeleton Modeling](../../ICCV2025/video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [Rethinking Video-Text Understanding: Retrieval from Counterfactually Augmented Data](rethinking_video-text_understanding_retrieval_from_counterfactually_augmented_da.md)

<!-- RELATED:END -->
