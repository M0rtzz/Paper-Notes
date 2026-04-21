---
title: >-
  [论文解读] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training
description: >-
  [CVPR 2025][图像分割][视觉语言预训练] COSMOS 提出了一种跨模态自蒸馏框架，通过文本裁剪策略和交叉注意力模块在学生-教师结构中学习细粒度的跨模态表征，在仅使用 30M 数据预训练的情况下，在零样本检索、分类和语义分割任务上全面超越 CLIP 类基线，甚至超越在数十亿数据上训练的 OpenCLIP。
tags:
  - CVPR 2025
  - 图像分割
  - 视觉语言预训练
  - 自蒸馏
  - 跨模态学习
  - 语义分割
  - 对比学习
---

# COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training

**会议**: CVPR 2025  
**arXiv**: [2412.01814](https://arxiv.org/abs/2412.01814)  
**代码**: [https://github.com/ExplainableML/cosmos](https://github.com/ExplainableML/cosmos)  
**领域**: 图像分割  
**关键词**: 视觉语言预训练, 自蒸馏, 跨模态学习, 语义分割, 对比学习

## 一句话总结
COSMOS 提出了一种跨模态自蒸馏框架，通过文本裁剪策略和交叉注意力模块在学生-教师结构中学习细粒度的跨模态表征，在仅使用 30M 数据预训练的情况下，在零样本检索、分类和语义分割任务上全面超越 CLIP 类基线，甚至超越在数十亿数据上训练的 OpenCLIP。

## 研究背景与动机
视觉语言模型（VLM）如 CLIP 使用全局对比损失将整幅图像和文本进行匹配，在多种视觉和语言任务上取得了显著进展。然而，这种全局对比学习具有先天缺陷：模型倾向于关注图像中的主要前景物体，忽视其他重要信息。这导致了所谓的"特征抑制"现象 —— 模型只学习了数据中最显著的特征，而忽略了其他有价值的区分特征。具体表现为三个方面：(1) 在密集预测任务（如语义分割）上表现不佳；(2) 难以区分视觉模式不同但整体相似的图像；(3) 文本编码器将文本当成词袋处理，忽略语序和组合语义。

之前的研究主要通过在图像编码器上引入自监督学习来改进，例如 SLIP 和 SILC 仅改进图像表征而不改进文本表征。核心矛盾在于：如何同时增强图像和文本的细粒度表征，而不仅仅关注单一模态？

COSMOS 的核心 idea：将自监督学习中的多裁剪增强策略推广到多模态场景，提出"文本裁剪"概念，配合跨注意力模块，在学生-教师框架中同时蒸馏图像和文本编码器，学习跨模态的细粒度表征。

## 方法详解

### 整体框架
COSMOS 采用学生-教师自蒸馏框架。学生模型和教师模型共享相同的 VLM 架构（图像编码器 + 文本编码器），教师模型通过学生参数的指数移动平均（EMA）更新。训练时，对图像-文本对进行多模态增强，生成全局和局部视图。所有视图通过学生，仅全局视图通过教师。学生额外包含一个交叉注意力模块，用于融合跨模态信息。总损失由标准 CLIP 对比损失和 COSMOS 跨模态自蒸馏损失组成。

### 关键设计
1. **文本裁剪策略 (Text-Cropping Strategy)**:

    - 灵感来源于图像的多裁剪增强（multi-crop），将其推广到文本域
    - 利用 MLLM 生成的长合成描述（包含多个句子），从中随机采样不同长度的片段
    - 全局文本视图：随机采样 1-5 个句子，覆盖图像的较大区域描述
    - 局部文本视图：仅采样 1 个句子，聚焦于图像的局部区域描述
    - 图像裁剪和文本裁剪独立进行，全局/局部裁剪不一定对应相同区域，这是设计上的特意选择
    - 此设计使自蒸馏能同时优化文本和图像编码器

2. **交叉注意力模块 (Cross-Attention Module)**:

    - 仅添加在学生模型中，包含两个子模块：$C^T_\theta$ 和 $C^I_\theta$
    - $C^T_\theta$：以图像的 [CLS] token 作为 query，文本 token 作为 key/value，生成图像的跨模态嵌入 $h_I$
    - $C^I_\theta$：以文本的 [EOT] token 作为 query，图像 token 作为 key/value，生成文本的跨模态嵌入 $h_T$
    - 输出通过残差连接加回原始 token：$h_I = C^T_\theta(q=[\text{cls}], kv=\text{txt-tok}) + [\text{cls}]$
    - 这使得蒸馏信号能够同时流入两个编码器，促进视觉和文本的双向基础学习
    - 实际操作中使用全局裁剪的 token 作为 key/value

3. **学生-教师自蒸馏框架**:

    - 教师参数通过 EMA 更新：$\theta_t = \lambda \theta_t + (1-\lambda) \theta_s$
    - 学生处理所有裁剪（全局+局部），教师仅处理全局裁剪
    - 这种不对称设计促使学生从局部特征预测教师的全局上下文

### 损失函数 / 训练策略
- **CLIP 对比损失 $\mathcal{L}_{CLIP}$**：学生内部的标准对称 InfoNCE 损失，在所有裁剪上计算
- **COSMOS 跨模态自蒸馏损失 $\mathcal{L}_{COSMOS}$**：将学生的跨模态嵌入 ($h_I$, $h_T$) 与教师的 [CLS] 和 [EOT] token 进行四重对称 InfoNCE 匹配
- **总损失**：$\mathcal{L}_{total} = \mathcal{L}_{CLIP} + \mathcal{L}_{COSMOS}$
- 一个重要优点：两个损失项无需额外的缩放超参，直接等权相加即可获得最优效果
- 训练 32 个 epoch，使用 ViT-B/16 作为视觉编码器

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 (Merged-30M) | DreamLIP (30M) | CLIP (30M) | 提升 (vs CLIP) |
|--------|------|------|----------|------|------|
| MSCOCO I2T | R@1 | 68.0 | 62.3 | 63.2 | +4.8 |
| MSCOCO T2I | R@1 | 52.5 | 44.9 | 48.2 | +4.3 |
| Flickr30K I2T | R@1 | 92.9 | 89.9 | 90.5 | +2.4 |
| Flickr30K T2I | R@1 | 80.3 | 73.3 | 75.9 | +4.4 |
| ImageNet | Top-1 Acc | 57.6 | 58.4 | 50.0 | +7.6 |
| 语义分割 (8 benchmarks) | Avg mIoU | 20.0 | - | - | - |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| CLIP loss only | 基线 | 标准 CLIP 对比学习 |
| + Text cropping | 检索/分割均提升 | 文本增强是关键创新 |
| + Cross-attention | 进一步提升 | 跨模态融合增强表征 |
| COSMOS (30M) vs OpenCLIP (1B) | 20.0 vs 16.5 avg mIoU | 30M 数据超越 1B 数据 |
| COSMOS w/ SCLIP | 37.8 avg mIoU | 接近在 400M 数据上训练的 SCLIP (38.2) |

### 关键发现
- COSMOS 在仅使用 30M 数据的情况下，检索性能超越在 2.5B 数据上训练的 Llip（MSCOCO: 68.0 vs 63.4 I2T R@1）
- 语义分割上，30M 数据几乎翻倍超越 1B 数据的 OpenCLIP（Cityscapes: 13.9 vs 8.5）
- 在 SugarCrepe 和 SVO 等组合理解基准上也有显著优势，平均 86.6 vs 81.8 (DreamLIP)
- 交叉注意力的可视化表明模型能有效定位图像和文本中的相关区域

## 亮点与洞察
1. **文本裁剪的巧妙设计**：将图像领域成熟的多裁剪自蒸馏策略推广到文本，利用合成长描述构造全局/局部文本视图，这是一个简洁但有效的创新
2. **跨模态嵌入的双向蒸馏**：不同于之前仅改进图像编码器的方法，COSMOS 通过交叉注意力模块让蒸馏信号同时流入两个编码器
3. **无需超参缩放**：两个损失项等权相加的简洁设计，避免了网格搜索最优损失比例的麻烦
4. **数据效率极高**：30M 数据训练的模型在多个任务上超越百亿级数据训练的模型，展现了方法设计的优越性

## 局限与展望
- 分类任务的绝对性能仍低于在数十亿数据上训练的模型，说明分类任务更依赖数据规模
- 依赖 MLLM 生成的合成长描述，描述质量直接影响文本裁剪效果
- 交叉注意力模块增加了学生模型的计算开销
- 仅在 ViT-B/16 上验证，是否在更大模型上同样有效有待验证
- 文本裁剪和图像裁剪独立进行，未探索对齐裁剪区域是否有益

## 相关工作与启发
- 与 DINO 的自蒸馏思想一脉相承，但推广到多模态场景
- DreamLIP 提供长合成描述数据集，COSMOS 在此基础上提出文本裁剪增强
- SILC 仅在图像编码器上做局部到全局匹配，COSMOS 扩展到双模态
- 对开放词汇分割的启发：通过改善预训练阶段的细粒度表征学习，可以显著提升下游密集预测任务性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 文本裁剪和跨模态自蒸馏的结合有新意，但整体框架仍在 DINO+CLIP 的成熟范式内
- 实验充分度: ⭐⭐⭐⭐⭐ 检索、分类、分割三大类任务 + 视觉感知和组合理解评估 + 充分消融
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，图表丰富，方法阐述明确
- 价值: ⭐⭐⭐⭐ 在数据高效的 VLM 预训练方向提供了有价值的范式，特别是文本裁剪策略值得推广

<!-- RELATED:START -->

## 相关论文

- [SiLC: Improving Vision Language Pretraining with Self-Distillation](../../ECCV2024/segmentation/silc_improving_vision_language_pretraining_with_self-distillation.md)
- [DreamLIP: Language-Image Pre-training with Long Captions](../../ECCV2024/segmentation/dreamlip_language-image_pre-training_with_long_captions.md)
- [ResCLIP: Residual Attention for Training-free Dense Vision-language Inference](resclip_residual_attention_for_training-free_dense_vision-language_inference.md)
- [DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation](dformerv2_geometry_self-attention_for_rgbd_semantic_segmentation.md)
- [Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)

<!-- RELATED:END -->
