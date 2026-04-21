---
title: >-
  [论文解读] DreamLIP: Language-Image Pre-training with Long Captions
description: >-
  [ECCV 2024][图像分割][视觉语言预训练] 通过 MLLM 为 30M 图像生成长文本描述，提出动态子描述采样的多正样本对比学习和子描述特定分组损失，实现细粒度视觉-语言对齐，仅用 30M 数据在检索和语义分割上达到甚至超越 CLIP 400M 的性能。
tags:
  - ECCV 2024
  - 图像分割
  - 视觉语言预训练
  - 长文本描述
  - 对比学习
  - 细粒度对齐
  - CLIP
---

# DreamLIP: Language-Image Pre-training with Long Captions

**会议**: ECCV 2024  
**arXiv**: [2403.17007](https://arxiv.org/abs/2403.17007)  
**代码**: [有](https://zyf0619sjtu.github.io/dream-lip)  
**领域**: 图像分割  
**关键词**: 视觉语言预训练, 长文本描述, 对比学习, 细粒度对齐, CLIP

## 一句话总结

通过 MLLM 为 30M 图像生成长文本描述，提出动态子描述采样的多正样本对比学习和子描述特定分组损失，实现细粒度视觉-语言对齐，仅用 30M 数据在检索和语义分割上达到甚至超越 CLIP 400M 的性能。

## 研究背景与动机

视觉-语言预训练（如 CLIP）的效果高度依赖文本对图像描述的精确度和完整度。然而现有数据集中，每张图像通常只配有一句简短描述（约 20 个 token），远远无法覆盖图像中的丰富内容。

作者的核心观察是：**真实图像蕴含的信息需要多句长描述才能充分表达，且长描述中的每一句话通常只描述图像的一个局部区域**（如某个物体、某个场景细节）。这一观察带来了两个关键研究问题：

**长描述能否提升视觉-语言预训练效果？** 现有方法已探索用 MLLM 生成短描述替代原始噪声文本，但尚无系统研究长描述的潜力。

**如何有效利用长描述？** 直接将整段长文本作为一个正样本训练效果不佳（因为 CLIP 的文本编码器上下文窗口仅 77 token），需要设计新的训练策略。

现有相关工作（LaCLIP 重写短描述、StableRep 生成合成图像、ALIP 清洗噪声文本）仅停留在短描述层面，DreamLIP 首次系统地探索了长描述在视觉-语言预训练中的作用。

## 方法详解

### 整体框架

DreamLIP 在 CLIP 的对比学习框架上做了两层扩展：（1）全局层面用多正样本对比学习匹配子描述与全局图像特征；（2）局部层面用子描述特定分组损失匹配子描述与对应的局部图像 patch。整体训练目标为两者的加权和。

### 关键设计

1. **MLLM 生成长短描述**：对于数据集中的每张图像 $I_i$，使用预训练 MLLM（如 ShareGPT4V）分别生成长描述 $C_i^l$ 和短描述 $C_i^s$：

$$\mathcal{C} = \{C_i^l, C_i^s\}_{i=1}^N = \{f(I_i, q_l), f(I_i, q_s)\}_{i=1}^N$$

长描述使用提示 "Describe the image in details" 生成，通常包含 8-10 句话，涵盖全局场景、物体细节、空间关系等。短描述使用 "Describe the image in short:" 生成，简洁不易出错，与长描述形成互补。设计动机是：短描述概括整体，长描述提供细节，两者结合可最大化语义覆盖。

2. **全局多正样本对比学习（Multi-Positive Contrastive Learning）**：将长描述拆分为多个子描述（每句作为一个子描述），连同原始描述和短描述构成子描述集合，通过均匀采样 $K$ 个子描述构建多个正样本对：

$$S_{i,j} \sim \text{Uniform}([T, c_s, c_1, \ldots, c_M])$$

多正样本对比损失为：

$$\mathcal{L}_{\text{MPCL}}^{t2v} = -\sum_{i=1}^{N}\sum_{j=1}^{K}\log\frac{\exp(\cos\langle \mathbf{v}_i, \mathbf{t}_{i,j}\rangle / \tau)}{\sum_{n=1}^{N}\exp(\cos\langle \mathbf{v}_n, \mathbf{t}_{i,j}\rangle / \tau)}$$

核心思想是：一张图值千言，应该与多句描述同时构成正样本对。动态采样使每个 epoch 看到不同的子描述组合，隐式增强训练数据多样性。

3. **子描述特定分组损失（Subcaption-specific Grouping Loss）**：为实现细粒度对齐，计算每个子描述与所有图像 patch 的相似度矩阵，通过阈值 $\sigma$ 稀疏化后池化得到子描述对应的局部视觉特征：

$$\tilde{w}_{i,j} = \begin{cases} \hat{w}_{i,j} & \text{if } \hat{w}_{i,j} \geq \sigma \\ 0 & \text{otherwise} \end{cases}$$

$$\hat{\mathbf{v}}_j = \sum_{n=1}^{HW} \frac{\tilde{w}_{i,j}}{\sum_j \tilde{w}_{i,j}} \mathbf{v}_n$$

分组损失对齐池化后的局部视觉特征与子描述嵌入：

$$\mathcal{L}_{\text{Sub}} = -\sum_{i=1}^{N}\sum_{j=1}^{M+2}\log\frac{\exp(\cos(\hat{\mathbf{v}}_{i,j}, \mathbf{t}_{i,j})/\tau)}{\sum_{n=1}^{K}\exp(\cos(\hat{\mathbf{v}}_{i,n}, \mathbf{t}_{i,j})/\tau)}$$

与 FILIP 等方法用单词 token 对齐 patch 不同，DreamLIP 用句子级子描述对齐，规避了无关词（情感词、连词）的干扰。子描述天然對應局部区域的完整语义，比词级对齐更准确。

### 损失函数

总训练目标为两个损失的加权和：

$$\mathcal{L}_{\text{DreamLIP}} = \lambda_{MPCL} \mathcal{L}_{\text{MPCL}} + \lambda_S \mathcal{L}_{\text{Sub}}$$

使用 ViT-B/32 或 ViT-B/16 作为图像编码器，文本编码器遵循 CLIP 设置。训练 32 个 epoch，输入分辨率 224×224，文本截断至 77 token，温度参数 $\tau$ 初始化为 0.07。

## 实验关键数据

### 主实验

**零样本图像-文本检索（Flickr30k / MSCOCO，ViT-B/32）**：

| 数据规模 | 方法 | Flickr R@1 (TR) | COCO R@1 (TR) | Flickr R@1 (IR) | COCO R@1 (IR) |
|---------|------|----------------|---------------|----------------|---------------|
| YFCC15M | CLIP | 34.9 | 20.8 | 23.4 | 13.0 |
| YFCC15M | ALIP | 70.5 | 46.8 | 48.9 | 29.3 |
| YFCC15M | **DreamLIP** | **84.9** | **55.7** | **66.0** | **39.8** |
| Merged-30M | CLIP | 57.8 | 35.0 | 44.0 | 23.5 |
| Merged-30M | **DreamLIP** | **87.2** | **58.3** | **66.4** | **41.1** |
| LAION-400M | CLIP | 78.7 | 53.7 | 61.8 | 34.8 |

DreamLIP 用 30M 数据超越 CLIP 400M！

**语义分割（mIoU）**：

| 数据规模 | 方法 | ADE-847 | PC-459 | ADE-150 | PC-59 | VOC-20 | Avg. |
|---------|------|---------|--------|---------|-------|--------|------|
| Merged-30M | CLIP | 5.8 | 10.2 | 21.0 | 45.8 | 86.9 | 33.9 |
| Merged-30M | **DreamLIP** | **8.1** | **12.5** | **25.3** | **49.9** | **90.9** | **37.3** |
| LAION-400M | CLIP | 6.1 | 12.2 | 21.3 | 46.3 | 88.3 | 34.8 |

DreamLIP 30M 在语义分割上平均超越 CLIP 400M 2.5%。

### 消融实验

**各组件贡献（CC3M, ViT-B/16）**：

| 配置 | Flickr R@1 (TR) | COCO R@1 (TR) | ImageNet Acc. | VOC mIoU | 说明 |
|------|----------------|---------------|--------------|----------|------|
| CLIP baseline | 32.6 | 14.8 | 20.3 | 64.4 | 原始短描述 |
| + 长描述（直接用） | 56.6 | 30.2 | 24.4 | 75.7 | 有提升但未充分利用 |
| + 长描述（采样） | 63.0 | 35.7 | 30.0 | 81.8 | 动态采样显著提升 |
| + 短描述 | 68.3 | 40.8 | 30.1 | 82.9 | 短描述互补 |
| + 分组损失 | **69.5** | **42.8** | **31.1** | **84.5** | 细粒度对齐最优 |

**不同 MLLM 生成描述的效果**：

| MLLM | Flickr R@1 (TR) | ImageNet Acc. | VOC mIoU |
|------|----------------|--------------|----------|
| InstructBLIP | 58.7 | 27.8 | 79.2 |
| LLaVA-1.5 | 66.8 | 29.0 | 81.8 |
| ShareGPT4V | **69.5** | **31.1** | **84.5** |
| 三者混合 | **74.4** | **34.6** | **88.2** |

### 关键发现

- 长描述的质量（生成 MLLM 的能力）直接影响下游性能，ShareGPT4V > LLaVA-1.5 > InstructBLIP
- 采样子描述数 $K$ 在 8 左右达到饱和，因为长描述的有效子描述数量有上限
- 长描述虽可能有幻觉，但短描述提供互补的准确全局语义，两者结合最优
- 注意力可视化证实：不同子描述确实聚焦到图像的不同局部区域（甚至能精准定位狗舌头、麦克风等细节）

## 亮点与洞察

- **数据效率惊人**：30M 数据量达到 400M CLIP 的性能，核心在于长描述带来了更丰富的监督信号
- **方法设计优雅**：将"一张图配多句话"的直觉转化为多正样本对比 + 局部分组对齐，简洁有效
- **启示深远**：图像理解的瓶颈可能不在模型架构而在标注质量，高质量长描述可大幅替代数据量

## 局限与展望

- MLLM 生成的长描述不可避免地包含幻觉，尤其是复杂场景下
- 文本编码器上下文窗口限制为 77 token，长描述需拆分为子描述使用，可能丢失句间关系
- 分组损失的稀疏化阈值 $\sigma$ 为超参数，对不同粒度的分割任务可能需要调整
- 未探索与掩码建模（MAE）等自监督方法的结合

## 相关工作与启发

- 对比 LaCLIP（仅用 LLM 重写短描述）和 ALIP（清洗噪声描述），DreamLIP 首次系统利用长描述
- 分组损失思路与 GroupViT 的自下而上分组机制相关，但从文本监督角度切入
- 启发：MLLM 生成的合成数据可以反哺更早期的基础模型训练，形成良性循环

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次系统探索长描述对视觉-语言预训练的作用，洞察清晰
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖检索/分类/分割/VQA 等多维度，多数据规模对比，消融极其详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富，可视化有说服力
- **价值**: ⭐⭐⭐⭐⭐ 30M vs 400M 的数据效率结论具有很高的实践指导意义

<!-- RELATED:START -->

## 相关论文

- [COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](../../CVPR2025/segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [SiLC: Improving Vision Language Pretraining with Self-Distillation](silc_improving_vision_language_pretraining_with_self-distillation.md)
- [Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)
- [Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)

<!-- RELATED:END -->
