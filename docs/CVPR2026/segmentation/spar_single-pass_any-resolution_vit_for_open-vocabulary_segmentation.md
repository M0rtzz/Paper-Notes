---
title: >-
  [论文解读] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation
description: >-
  [CVPR 2026][图像分割][开放词汇分割] 提出 SPAR，一种通过将细步幅滑窗教师的空间推理能力蒸馏到单次前向传递学生的方法，将 ViT 变为分辨率无关的密集特征提取器，在开放词汇分割中比单次前向基线提升 10.5 mIoU，同时比教师快 52 倍。
tags:
  - CVPR 2026
  - 图像分割
  - 开放词汇分割
  - 分辨率无关
  - 知识蒸馏
  - Transformer
  - 滑窗推理
---

# SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation

**会议**: CVPR 2026  
**arXiv**: [2604.02252](https://arxiv.org/abs/2604.02252)  
**代码**: [https://github.com/naomikombol/SPAR](https://github.com/naomikombol/SPAR)  
**领域**: 分割 / 开放词汇分割  
**关键词**: 开放词汇分割, 分辨率无关, 知识蒸馏, Vision Transformer, 滑窗推理

## 一句话总结

提出 SPAR，一种通过将细步幅滑窗教师的空间推理能力蒸馏到单次前向传递学生的方法，将 ViT 变为分辨率无关的密集特征提取器，在开放词汇分割中比单次前向基线提升 10.5 mIoU，同时比教师快 52 倍。

## 研究背景与动机

**领域现状**：基础 ViT（CLIP、SigLIP2、DINOv3）通过对比/自监督学习在图像级理解上表现出色，但由于固定分辨率预训练和粗糙的 patch 级表示，在需要细粒度空间理解的密集预测任务（如分割）上效果有限。开放词汇分割（OVS）要求模型仅凭文本即可分割任意类别，对高分辨率输入的精细像素级推理需求更高。

**现有痛点**：处理高分辨率图像有两种策略：(1) 插值位置编码后单次前向推理——高效但精度差，因训练-推理分辨率不匹配导致位置信息失真；(2) 滑窗推理——通过小步幅重叠窗口显著提升精度（因为每个 patch 出现在多个上下文中），但计算成本极高。例如步幅24的滑窗比单次推理慢约52倍。

**核心矛盾**：精度与效率之间存在严重的 trade-off——单次推理快但差，滑窗推理好但慢。现有的分辨率适应方案（如 NaFlex）在图像级任务上有效，但在密集预测上表现不佳。

**本文目标** 如何在保持单次前向推理效率的同时，获得接近甚至超越细步幅滑窗推理的分割精度。

**切入角度**：观察到滑窗推理的优势本质上来自子 patch 区域被暴露在不同上下文中、以及通过平均获得的鲁棒性——这种空间推理能力可以通过蒸馏转移到单次推理模型中。

**核心 idea**：用特征回归损失将细步幅滑窗教师的空间特征蒸馏到同架构的单次推理学生中，无需架构修改或像素级标注。

## 方法详解

### 整体框架

教师：冻结的 VLM 视觉编码器在滑窗模式下工作（window size = 预训练分辨率，stride=24），对每个窗口提取特征后拼接（stitch）成统一特征图。学生：同架构初始化、单次前向推理，输入完整高分辨率图像输出特征图。训练目标是让学生的特征图逼近教师的拼接特征图。推理时仅用学生模型，实现高效高精度分割。

### 关键设计

1. **滑窗教师特征拼接**:

    - 功能：生成高质量的密集特征作为蒸馏目标
    - 核心思路：将高分辨率图像 $X \in \mathbb{R}^{3 \times H \times W}$ 分成 m 个大小为 $K \times K$ 的重叠窗口，步幅 $s=24$（不整除 patch size $P=16$）。每个窗口独立经过编码器得到特征图，然后通过上采样（factor r=2）对齐后平均拼接：$V_\text{teacher}(X) = \text{stitch}(\{f(X_{w_i})\}_{i=1}^m)$。步幅不整除 patch size 让子 patch 区域暴露在不同上下文中，进一步提升质量
    - 设计动机：小步幅+高重叠度让每个像素在多个窗口中被看到，通过平均获得类似 test-time augmentation 的鲁棒性

2. **特征蒸馏训练**:

    - 功能：将教师的空间推理能力转移到单次推理学生
    - 核心思路：学生 $g$ 对完整图像单次前向推理得到 $V_\text{student}(X) = g(X)$，蒸馏损失为简单的 MSE：$\mathcal{L}_\text{distill} = \|V_\text{teacher}(X) - V_\text{student}(X)\|_2^2$。训练时使用多样的分辨率和宽高比（短边 512-2048 像素）。关键发现是只需解冻最后2个 block 即可在标准 OVS 设定下获得强性能，训练全部参数在超大分辨率推理时更优
    - 设计动机：MSE 损失直接、高效，不需要像素级标注。教师特征预计算并存储复用，训练只需 25k 张 SA-1B 图像和约 1.5 小时（2×A6000）

3. **分辨率与宽高比增强**:

    - 功能：使学生模型对各种分辨率和宽高比具有鲁棒性
    - 核心思路：训练时对图像进行随机缩放（短边 512-2048）、随机裁剪（边长从512到最大可能）和水平翻转。所有图像双线性重采样至维度可被 patch size 整除。扩展版本进一步将短边范围增至 512-2560 并训练全部参数，以支持更高分辨率推理
    - 设计动机：暴露在多样分辨率下促进位置编码泛化，比单一分辨率预训练的 NaFlex 更有效

### 损失函数 / 训练策略

纯特征回归损失（MSE），无需任何标注。AdamW 优化器，恒定学习率 $2 \times 10^{-5}$，权重衰减 $10^{-4}$，训练10个epoch。默认只调最后2个 block。教师特征预计算存储（~170GB），避免重复计算。Batch size = 1（因变长序列）。

## 实验关键数据

### 主实验

SigLIP2 – ViT-B-16 在6个数据集上的平均 mIoU：

| 方法 | 推理模式 | Mean₆ |
|------|----------|-------|
| NaFlex | 单次 | 31.7 |
| Pre-trained | 单次 | 33.1 |
| Pre-trained | 滑窗(s=24) | 41.2 |
| **SPAR** | **单次** | **43.6** |
| SPAR + AnyUp | 单次 | 46.8 |
| SPAR + LPOSS | 单次 | 46.7 |

SPAR 比单次基线 +10.5 mIoU，甚至超过教师（滑窗s=24）+2.4 mIoU。

### 消融实验

不同 backbone 的提升：

| Backbone | 单次基线 | SPAR | 提升 |
|----------|----------|------|------|
| SigLIP2 ViT-B-16 | 33.1 | 43.6 | +10.5 |
| OpenCLIP ViT-B-16 | 27.7 | 34.4 | +6.7 |
| DINOv3 ViT-L-16 | 43.8 | 44.4 | +0.6 |

### 关键发现

- **NaFlex 不适合密集预测**：尽管 SigLIP2 专门设计了 NaFlex 进行分辨率适配，但在 OVS 上不如 SPAR 甚至不如标准滑窗，说明图像级分辨率适应不等于补丁级空间理解
- **步幅不整除 patch size 效果更好**：s=24 优于 s=32（整除16），因为子 patch 区域被暴露在更多样的上下文中
- **学生超越教师**：SPAR 在平均性能和大多数单个数据集上超过教师，可能因为蒸馏过程中的多分辨率训练起到了隐式正则化作用
- **DINOv3 提升较小**：因其已内置 RoPE 编码和高分辨率微调，单次推理本身就较好，但 SPAR 仍在 Cityscapes（大分辨率测试图）上将 mIoU 从 35.9 提升到 40.1
- SPAR 与 AnyUp、LPOSS 等方法互补，组合使用可进一步提升

## 亮点与洞察

- **极致简洁的方法**：no 架构修改、no 像素标注、no 复杂损失函数，仅靠 MSE 特征蒸馏实现巨大提升
- **52倍加速**：相比步幅24的滑窗推理，保持单次推理的效率是巨大的实用价值
- **通用性强**：在 SigLIP2、OpenCLIP、DINOv3 三种截然不同的 backbone 上均有效
- **训练成本极低**：25k 张无标注图像、1.5 小时训练时间即可完成
- 揭示了一个重要洞察：滑窗推理的优势可以被蒸馏，且蒸馏后甚至能超越原始教师

## 局限与展望

- DINOv3 等已经具备分辨率鲁棒性的模型提升空间有限
- 教师特征存储需 ~170GB，对存储有一定要求
- 仅在 training-free OVS 设定下验证，未在训练式方法或其他密集预测任务（检测、深度估计）上测试
- Batch size 限制为1（因变长序列），训练效率还有优化空间
- 可探索更高阶蒸馏策略（如注意力蒸馏）而非纯特征回归

## 相关工作与启发

- **FlexiViT / NaViT / ResFormer**：通过多分辨率预训练增强分辨率鲁棒性，但需要从头训练
- **SigLIP2 NaFlex**：统一灵活 patching 和变长序列，但本文证明其在密集预测上不够好
- **LPOSS**：训练无关的标签传播方法，与 SPAR 互补（+3.1 mIoU）
- 蒸馏思路可扩展到视频理解、3D 视觉等其他需要高分辨率密集推理的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将滑窗推理的优势通过蒸馏转移到单次推理是巧妙的洞察，虽然蒸馏本身不新
- **实验充分度**: ⭐⭐⭐⭐⭐ 3种backbone×6个数据集×多种分辨率×与多种方法组合，全面扎实
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，trade-off图直观，方法描述简洁
- **价值**: ⭐⭐⭐⭐⭐ 实用性极高——简单、通用、高效、效果好，可广泛应用于所有需要 ViT 高分辨率推理的场景

<!-- RELATED:START -->

## 相关论文

- [GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [Diffusion Models for Open-Vocabulary Segmentation](../../ECCV2024/segmentation/diffusion_models_for_open-vocabulary_segmentation.md)
- [VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](videomt_encoder_only_video_segmentation.md)

<!-- RELATED:END -->
