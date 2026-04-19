---
title: >-
  [论文解读] SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts
description: >-
  [CVPR 2026][图像分割][音频视觉定位] 提出 SouPLe (Sound-aware Prompt Learning)，通过将CLIP中固定的文本提示替换为基于图像特征生成的可学习上下文tokens，增强音频嵌入token与视觉特征之间的语义对应，在VGG-SS上cIoU提升3.75、开放集设定下cIoU提升6.32，全面超越先前方法。
tags:
  - CVPR 2026
  - 图像分割
  - 音频视觉定位
  - 提示学习
  - CLIP适配
  - 声源分割
  - 对比学习
---

# SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts

**会议**: CVPR 2026  
**arXiv**: [2603.22732](https://arxiv.org/abs/2603.22732)  
**代码**: 无  
**领域**: 分割 / 音频-视觉定位  
**关键词**: 音频视觉定位, 提示学习, CLIP适配, 声源分割, 对比学习

## 一句话总结

提出 SouPLe (Sound-aware Prompt Learning)，通过将CLIP中固定的文本提示替换为基于图像特征生成的可学习上下文tokens，增强音频嵌入token与视觉特征之间的语义对应，在VGG-SS上cIoU提升3.75、开放集设定下cIoU提升6.32，全面超越先前方法。

## 研究背景与动机

**领域现状**：音频-视觉声源定位旨在从视觉场景中定位发声物体。主流方法基于对比学习框架利用音频-视觉对应关系进行自监督学习。近年来，ACL-SSL 利用预训练的 CLIP 模型将音频信号转化为与 CLIP 文本编码器兼容的 token，取得了显著进展。

**现有痛点**：ACL-SSL 的核心问题在于其使用固定提示 "a photo of a $[V_A]$" 的方式存在两个缺陷：(1) 将分类token $[CLS]$ 替换为音频嵌入token $[V_A]$ 时，$[V_A]$ 缺乏可与视觉信息整合的语义信息；(2) "a photo of a" 这些固定 token 与 $[V_A]$ 之间缺乏有意义的语义连接，导致在某些场景中定位失败。

**核心矛盾**：CLIP 的文本编码器设计用于处理自然语言描述，但音频嵌入 token 不是自然语言——用固定的无语义提示来包裹它，本质上是一种不匹配，限制了音频-视觉跨模态对齐的质量。

**本文目标** 如何在CLIP框架下为音频嵌入token提供更好的上下文，使其能更有效地与视觉特征对齐来实现精准的声源定位和分割。

**切入角度**：受 CoCoOp 启发，将提示工程问题转化为提示学习问题——让提示 token 根据输入图像特征自适应生成，而非使用固定的人工设计提示。

**核心 idea**：用基于图像特征条件化的可学习上下文 token 替代固定文本提示，让音频嵌入 token 在丰富的视觉条件上下文中获得更好的语义对齐。

## 方法详解

### 整体框架

输入为音频-视觉对，CLIP图像编码器提取图像特征，Meta-net 将图像特征转化为 M 个可学习上下文 token $[V_1][V_2]...[V_M]$；音频编码器（BEATs）提取音频特征并通过 Audio Projection 转化为音频嵌入 token $[V_A]$；上下文 token 与音频 token 拼接后送入 CLIP 文本编码器得到音频-文本特征；最后与图像特征一起送入掩码解码器（CLIPSeg）生成声源分割掩码。

### 关键设计

1. **可学习上下文生成 (Meta-net)**:

    - 功能：从图像特征生成实例条件化的上下文 token
    - 核心思路：Meta-net 采用两层非线性瓶颈结构 (Linear-ReLU-Linear)，隐层将输入维度缩小16倍。接收 CLIP 图像编码器的图像特征 $F_I$，输出 M 个上下文 token。这些 token 替代了原来固定的 "a photo of a"，与 $[V_A]$ 拼接后送入文本编码器：$[V_1][V_2]...[V_M][V_A]$
    - 设计动机：不同图像中声源的语义上下文差异巨大，实例条件化的提示能自适应地为每个输入提供合适的语义引导，而固定提示无法做到这一点。实验表明 $[V_A]$ 放在最后效果最好，因为 CLIP 的因果注意力让前面的上下文 token 先建立语义空间

2. **视觉-音频-文本对齐 (VAT Module)**:

    - 功能：通过图像级和特征级双层对比学习训练音频-视觉对应关系
    - 核心思路：利用 SouPLe 生成的声源掩码创建两种版本：图像级掩码 $M_I$（前景突出、背景遮蔽）和特征级掩码 $M_F$（空间视觉特征中强调声源区域）。分别计算音频-文本特征与两种掩码下的视觉嵌入的余弦相似度 $S^I$ 和 $S^F$，并用对称 InfoNCE 损失优化。此外还有面积正则化损失 $\mathcal{L}_{REG}$ 约束掩码只覆盖发声区域
    - 设计动机：双层对比学习在不同粒度上强化音频-视觉对应——图像级关注整体对应，特征级关注高相关局部区域

3. **无文本/无标签设计**:

    - 功能：端到端的自监督框架，无需真实标签
    - 核心思路：整个框架仅用音频-视觉对应作为监督信号，冻结 CLIP 图像编码器、文本编码器和音频编码器，只优化 Meta-net、掩码解码器等少量参数（约2.38M，< 整体1%）。训练目标为 $\mathcal{L} = \lambda_1 \mathcal{L}_{ACL_I} + \lambda_2 \mathcal{L}_{ACL_F} + \lambda_3 \mathcal{L}_{REG}$
    - 设计动机：声源定位本质上是无标签任务，自监督方式不依赖昂贵标注且泛化性更好

### 损失函数 / 训练策略

总训练损失由三项组成：图像级音频-文本对比损失、特征级音频-文本对比损失和面积正则化损失。使用 VGGSound-144K 训练，Adam 优化器，学习率 $10^{-3}$，权重衰减 $10^{-5}$，训练20个epoch，batch size 16。音频输入为16kHz采样的10秒clip，视频帧缩放到 $352 \times 352$。

## 实验关键数据

### 主实验

标准基准上的声源定位：

| 方法 | VGG-SS cIoU↑ | VGG-SS AUC↑ | SoundNet cIoU↑ | SoundNet AUC↑ |
|------|-------------|-------------|----------------|---------------|
| ACL-SSL | 49.46 | 46.32 | 80.80 | 64.62 |
| **SouPLe** | **53.21** | **48.15** | **84.80** | **67.64** |
| 提升 | +3.75 | +1.83 | +4.00 | +3.02 |

开放集定位（110 Heard + 110 Unheard类别）：

| 测试集 | ACL-SSL cIoU | SouPLe cIoU | 提升 |
|--------|-------------|-------------|------|
| Heard 110 | 48.44 | 54.76 | +6.32 |
| Unheard 110 | 41.98 | 48.40 | +6.42 |

AVSBench S4 (零样本)：mIoU 62.89 (+3.13), F-Score 71.47 (+2.44)

### 消融实验

| 消融项 | VGG-SS cIoU | AUC |
|--------|-------------|-----|
| ctx=4 (default) | 53.21 | 48.15 |
| ctx=8 | 52.01 | 47.32 |
| ctx=16 | 51.08 | 46.93 |
| $V_A$ 在首位 | 49.91 | 46.21 |
| $V_A$ 在末位 (default) | 53.21 | 48.15 |

### 关键发现

- 仅4个上下文 token 即可达到最优，增加参数量反而降低性能——关键在于质量而非数量
- $[V_A]$ 放在最末位效果最好，因为 CLIP 因果注意力让前面的上下文先建立语义空间
- 在 Extended VGG-SS/SoundNet 等包含静默/不可见声源的挑战性设定中，SouPLe 同样大幅领先
- 在 AVSBench MS3 多目标场景中性能下降，因为无标签监督导致方法倾向分割所有潜在物体

## 亮点与洞察

- **极小改动带来大提升**：仅引入约2.38M参数（< 1%），就在多个基准上实现稳定提升
- **CoCoOp思想迁移到音频-视觉领域**：将图像分类中的提示学习成功适配到跨模态定位任务
- **无文本、无标注的端到端框架**：纯粹依赖音频-视觉对应，工程简洁且易于部署
- $[V_A]$ 位置的消融实验揭示了因果注意力中token顺序的重要性

## 局限与展望

- 在多声源场景（AVSBench MS3）中效果不佳，因为缺乏标签引导导致过度分割
- 未考虑时间维度信息（如视频中的连续帧），可能遗失动态线索
- Meta-net 结构较为简单，更复杂的条件化机制（如跨注意力）可能进一步提升
- 未与 MaPLe 等多模态提示学习方法深入对比
- 可探索扩展到音频-视觉分离、事件定位等更多下游任务

## 相关工作与启发

- **CoOp/CoCoOp**：提示学习的核心灵感来源，CoCoOp 的实例条件化策略被成功迁移
- **ACL-SSL**：直接基线，SouPLe 在其基础上用可学习提示替换固定提示
- **CLIPSeg**：用作掩码解码器，可关注其他 CLIP 变体解码器的替换可能
- 提示学习在多模态对齐中的有效性值得在更多跨模态任务中推广

## 评分

- **新颖性**: ⭐⭐⭐ 核心思路是 CoCoOp 到音频-视觉的直接迁移，技术创新有限但迁移有效
- **实验充分度**: ⭐⭐⭐⭐ 5个数据集+开放集+零样本+扩展基准+充分消融，覆盖全面
- **写作质量**: ⭐⭐⭐⭐ 动机清晰，实验完整，but 方法部分可更精炼
- **价值**: ⭐⭐⭐⭐ 验证了提示学习在音频-视觉领域的有效性，为CLIP-based方法提供了通用改进思路

<!-- RELATED:START -->

## 相关论文

- [Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)
- [Dynamic Derivation and Elimination: Audio Visual Segmentation with Enhanced Audio Semantics](../../CVPR2025/segmentation/dynamic_derivation_and_elimination_audio_visual_segmentation_with_enhanced_audio.md)
- [GeoSURGE: Geo-localization using Semantic Fusion with Hierarchy of Geographic Embeddings](geosurge_geo-localization_using_semantic_fusion_with_hierarchy_of_geographic_emb.md)
- [CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation](../../ECCV2024/segmentation/cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)
- [Audio-Visual Instance Segmentation](../../CVPR2025/segmentation/audio-visual_instance_segmentation.md)

<!-- RELATED:END -->
