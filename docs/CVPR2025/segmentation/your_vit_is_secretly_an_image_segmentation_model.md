---
title: >-
  [论文解读] Your ViT is Secretly an Image Segmentation Model
description: >-
  [CVPR 2025][图像分割][Transformer] 本文提出 Encoder-only Mask Transformer（EoMT），证明在大规模预训练与足够大的模型下，plain ViT 无需卷积适配器、像素解码器和 Transformer 解码器等任务特定组件即可完成高质量图像分割，同时速度快达 4 倍。
tags:
  - CVPR 2025
  - 图像分割
  - Transformer
  - 编码器唯一架构
  - 掩码退火
  - DINOv2
---

# Your ViT is Secretly an Image Segmentation Model

**会议**: CVPR 2025  
**arXiv**: [2503.19108](https://arxiv.org/abs/2503.19108)  
**代码**: https://www.tue-mps.org/eomt/ (有)  
**领域**: 分割  
**关键词**: Vision Transformer, 图像分割, 编码器唯一架构, 掩码退火, DINOv2

## 一句话总结
本文提出 Encoder-only Mask Transformer（EoMT），证明在大规模预训练与足够大的模型下，plain ViT 无需卷积适配器、像素解码器和 Transformer 解码器等任务特定组件即可完成高质量图像分割，同时速度快达 4 倍。

## 研究背景与动机

**领域现状**：当前最先进的图像分割方法（语义/实例/全景分割）通常在 ViT 之上叠加多个任务特定组件：ViT-Adapter（CNN适配器提取多尺度特征）→ 像素解码器（融合多尺度特征）→ Transformer 解码器（对象查询+交叉注意力生成预测）。典型代表为 ViT-Adapter + Mask2Former。

**现有痛点**：这些额外组件增加了大量计算开销和实现复杂度。ViT-Adapter 引入的卷积模块需要与 ViT 多次交互，像素解码器使用多尺度变形注意力，Transformer 解码器的掩码交叉注意力需要在每层预测中间掩码——这些都是效率瓶颈。ViT-Adapter + M2F 在 ViT-B 上只有 32 FPS。

**核心矛盾**：这些任务特定组件本质上是在弥补 ViT 的两个"不足"——缺少多尺度特征和缺少局部处理能力。但这些所谓的不足，是 ViT 本身能力不够，还是预训练不充分？

**本文目标**：验证假设——随着模型规模和预训练质量的提升，ViT 自身已能隐式学到这些归纳偏置，任务特定组件变得冗余。

**切入角度**：逐步移除各组件并观察性能变化。如果大模型+强预训练下性能下降可忽略，则证明这些组件不必要。

**核心 idea**：把 ViT 既当编码器又当解码器——将可学习查询直接拼接到 patch token 中，利用 ViT 自身的自注意力替代交叉注意力，并用掩码退火策略实现无需推理时掩码注意力的高效分割。

## 方法详解

### 整体框架
EoMT 的输入为图像经 patch embedding 后的 token 序列。前 $L_1$ 个 ViT 块正常处理 patch token；之后在 token 序列中拼接 $K$ 个可学习查询；后续 $L_2$ 个块联合处理 patch token 和查询 token。最终，查询 token 通过轻量掩码模块预测类别和分割掩码。

### 关键设计

1. **逐步移除任务特定组件**:

    - 功能：验证各组件的必要性
    - 核心思路：从 ViT-Adapter + M2F 开始，逐步移除：(1) 去掉 ViT-Adapter——用简单的转置/普通卷积从 ViT 单尺度输出构建特征金字塔（ViTDet 风格），PQ 仅降 0.4；(2) 去掉像素解码器——直接将特征金字塔送入 TF 解码器，PQ 基本不变；(3) 去掉多尺度处理——查询只对单尺度 ViT 输出做注意力，PQ 再降 0.2；(4) 去掉独立的 TF 解码器——将查询直接注入 ViT 的后几层，PQ 降 0.5 但 FPS 提升到 61
    - 设计动机：每一步的消融都量化了对应组件的"实际价值"，发现使用 DINOv2 预训练的 ViT-L 上，所有组件加起来只贡献 1.1 PQ，但带来 4.4 倍的速度惩罚

2. **查询注入机制**:

    - 功能：让 ViT 的自注意力同时执行查询间交互和查询-图像特征转移
    - 核心思路：在第 $L_1$ 层后将 $K$ 个可学习查询拼接到 patch token 末尾，后续 $L_2$ 个自注意力层中，一次 MHSA 操作同时实现：查询与查询之间的自注意力（协调各查询分工）、查询到 patch 的交叉注意力（获取视觉信息）、patch 到查询的反向注意力（查询信息反馈到视觉特征），以及 patch 之间原有的自注意力。对于 ViT-L（$L=24$），$L_1=20, L_2=4$ 只增加了 2.1% 的 token 处理量
    - 设计动机：标准分割方法将自注意力和交叉注意力分为两步串行执行。EoMT 利用 ViT 自注意力机制的天然"全对全"特性，一步完成所有交互，更高效且更简单

3. **掩码退火（Mask Annealing）策略**:

    - 功能：实现训练时用掩码注意力（性能好）、推理时不用掩码注意力（速度快）
    - 核心思路：M2F 的掩码注意力让每个查询只关注其预测掩码对应的图像区域，有助于学习但影响效率。EoMT 在训练初期完全启用掩码注意力（$P_{mask}=1.0$），然后逐block、逐步将掩码概率退火到 0，使得训练结束时模型已适应无掩码的注意力模式。退火调度采用多项式衰减，从早层开始先解除掩码，到训练结束时所有层的掩码概率都为 0
    - 设计动机：直接训练时不用掩码注意力会导致 PQ 降 3.0，直接去掉推理时的掩码会导致灾难性下降 28.8。掩码退火是一个渐进式过渡策略，训练时利用掩码辅助学习，推理时完全不需要，两全其美

### 损失函数 / 训练策略
损失函数沿用 M2F：$\mathcal{L}_{tot} = \lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{ce}\mathcal{L}_{ce}$，权重分别为 5.0、5.0、2.0。训练使用 AdamW 优化器，层级学习率衰减（factor 0.8），多项式学习率调度。DINOv2 预训练权重的 patch embedding 和位置编码通过 FlexiViT 方式调整至目标尺寸。

## 实验关键数据

### 主实验

| 方法 | Backbone | PQ (COCO) | FPS | 参数量 |
|------|----------|-----------|-----|--------|
| ViT-Adapter + M2F | ViT-L | 57.1 | 29 | 349M |
| EoMT w/ Masking | ViT-L | 56.2 | 61 | 316M |
| **EoMT** | **ViT-L** | **56.0** | **128** | **316M** |
| ViT-Adapter + M2F | ViT-g | 57.7 | 20 | 1209M |
| EoMT | ViT-g | 57.0 | 55 | 1164M |
| EoMT (高分辨率) | ViT-L | 58.3 | 30 | 322M |

### 消融实验

| 训练策略 | 推理策略 | FPS | PQ | 说明 |
|---------|---------|-----|------|------|
| 全程掩码注意力 | 掩码注意力 | 61 | 56.2 | 完整但慢 |
| 全程掩码注意力 | 无掩码 | 128 | 27.4 | 灾难性失败 -28.8 |
| 全程无掩码 | 无掩码 | 128 | 53.2 | 尚可但差 -3.0 |
| **掩码退火** | **无掩码** | **128** | **56.0** | 仅降 0.2，速度翻倍 |

### 关键发现
- **预训练质量决定一切**：DINOv2/EVA-02 预训练下 EoMT 与复杂模型的差距仅 1.1-1.2 PQ，但 ImageNet-1K 预训练下差距扩大到 6.1，证实了"大规模预训练使任务特定组件冗余"的假设
- **模型越大组件越冗余**：ViT-S 上 EoMT 与 M2F 的差距为 5.8 PQ，ViT-g 上仅 0.7 PQ，呈明显缩小趋势
- **掩码退火是关键创新**：带来 2.1 倍加速（61→128 FPS）仅 0.2 PQ 代价，且对 ViT-Adapter + M2F 也有效（通用性强）
- EoMT 在分布外泛化上也表现出色：BRAVO 基准上 OOD 置信度评分 89.7 vs ViT-Adapter+M2F 的 68.7
- 与 token merging（ALGM）兼容，可再获 31% 吞吐提升而不损 mIoU

## 亮点与洞察
- **"减法"思维的典范**：不是增加新模块，而是证明现有模块可以去掉。这种简化思路在大模型时代尤其有价值——把计算资源花在扩大 ViT 本身而非添加复杂附件
- **掩码退火的巧妙设计**：训练时利用掩码辅助学习而推理时完全不需要，优雅地解决了训练-推理不一致的问题。这个策略可以推广到其他使用中间约束的训练框架
- **生态系统兼容性**：EoMT 完全基于 plain ViT，可以直接受益于 FlashAttention、token merging、ViT 专用硬件加速等所有 Transformer 生态优化，而复杂架构则受限于自定义组件

## 局限与展望
- 实例分割的性能降幅（-2.4 AP）大于全景分割（-1.1 PQ），说明 EoMT 在需要精细实例区分的任务上还有改进空间
- 当前只在 DINOv2 和 EVA-02 预训练上验证，未来更强的视觉基础模型可能进一步缩小差距
- 小模型（ViT-S）上的性能差距仍然显著（-5.8 PQ），EoMT 更适合中大型部署场景
- 掩码退火的调度需要预设超参数，自适应调度策略值得探索

## 相关工作与启发
- **vs ViT-Adapter + Mask2Former**：EoMT 是对这个"标准配置"的简化挑战，证明在 DINOv2 加持下复杂组件的边际收益很小，但速度惩罚很大
- **vs YOLOS**：同样用编码器做检测，但 YOLOS 不做分割且未验证大模型优势
- **vs UViT**：用单尺度 ViT 特征做实例识别，但仍依赖复杂的任务解码器
- **vs SegFormer/OneFormer**：这些方法依赖 Swin/DiNAT 等非 plain-ViT 架构，无法利用 DINOv2 等视觉基础模型预训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 掩码退火策略创新，"减法设计"在当前趋势下有独到洞见
- 实验充分度: ⭐⭐⭐⭐⭐ 三大分割任务、多数据集、多模型尺寸、多预训练方式的全面验证
- 写作质量: ⭐⭐⭐⭐⭐ 假设驱动、逐步消融的叙事结构非常优雅
- 价值: ⭐⭐⭐⭐⭐ 为 ViT 分割模型的设计范式提供了重要的简化方向

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] VidEoMT: Your ViT is Secretly Also a Video Segmentation Model](../../CVPR2026/segmentation/videomt_encoder_only_video_segmentation.md)
- [\[CVPR 2025\] Rethinking Query-Based Transformer for Continual Image Segmentation](rethinking_query-based_transformer_for_continual_image_segmentation.md)
- [\[ICML 2025\] FeatSharp: Your Vision Model Features, Sharper](../../ICML2025/segmentation/featsharp_your_vision_model_features_sharper.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[CVPR 2025\] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification](2dmamba_efficient_state_space_model_for_image_representation_with_applications_o.md)

<!-- RELATED:END -->
