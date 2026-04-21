---
title: >-
  [论文解读] F-LMM: Grounding Frozen Large Multimodal Models
description: >-
  [CVPR 2025][图像分割][视觉定位] F-LMM 冻结现成 LMM 的所有参数，仅训练轻量 CNN mask decoder 将 LMM 注意力图中固有的词-像素对应关系翻译为分割 mask，在完全保持对话能力的同时获得 competitive 的视觉定位性能。
tags:
  - CVPR 2025
  - 图像分割
  - 视觉定位
  - 大型多模态模型
  - 冻结参数
  - 注意力图
  - 参考表达分割
---

# F-LMM: Grounding Frozen Large Multimodal Models

**会议**: CVPR 2025  
**arXiv**: [2406.05821](https://arxiv.org/abs/2406.05821)  
**代码**: https://github.com/wusize/F-LMM  
**领域**: 语义分割 / 多模态VLM  
**关键词**: 视觉定位, 大型多模态模型, 冻结参数, 注意力图, 参考表达分割

## 一句话总结

F-LMM 冻结现成 LMM 的所有参数，仅训练轻量 CNN mask decoder 将 LMM 注意力图中固有的词-像素对应关系翻译为分割 mask，在完全保持对话能力的同时获得 competitive 的视觉定位性能。

## 研究背景与动机

**领域现状**：赋予大型多模态模型（LMM）视觉定位能力是当前热门方向。主流做法是在 LMM 词表中添加特殊分割 token（如 [SEG]），将 LMM 与 SAM 等 mask head 连接，然后在分割和定位数据上微调整个模型。

**现有痛点**：微调 LMM 会导致灾难性的对话能力退化。作者对 SOTA grounding LMM 进行了全面评估，发现 GLaMM、LISA、PixelLM 等模型在 MME、MMBench 等通用问答 benchmark 上得分近乎为零，丧失了指令跟随能力和通用知识理解能力。例如 GLaMM 无法回答简单的 yes/no 问题。

**核心矛盾**：定位能力与对话能力之间存在根本冲突——现有分割/定位数据只包含简单的定位 prompt，微调时模型主要学习词与分割 token 的关系，导致过拟合定位任务而牺牲对话能力。收集高质量的"带分割标注的对话数据"成本极高且效果有限。

**本文目标** 如何在不微调 LMM 参数的前提下，让现成的 LMM 同时具备视觉定位能力和对话能力。

**切入角度**：作者发现训练好的 LMM 的注意力机制中已经天然存在词-像素对应关系。通过 K-Means 聚类可视化注意力图后，能看到物体的几何和空间轮廓。这些是有价值的分割先验，只需一个轻量解码器将这些注意力权重翻译为 mask 即可。

**核心 idea**：冻结 LMM，仅训练一个 CNN 把注意力图翻译成分割 mask，从而"免费"获得定位能力。

## 方法详解

### 整体框架

输入为图像和文本，输出为关键词对应的分割 mask。整体 pipeline 为：(1) 冻结的 LMM 正常进行图文推理，同时提取所有层和所有注意力头的词-图像注意力图（word-image attention maps）；(2) 将 $M \times N$ 个注意力图 stack 成一个张量作为分割先验；(3) CNN mask decoder 将注意力图翻译为 mask logits；(4) SAM-based mask refiner 利用额外的图像和语言线索优化 mask；(5) 关键词选择器自动发现文本中的物体名词。

### 关键设计

1. **Word-Image Attention Map 提取**:

    - 功能：从冻结 LMM 中提取词-像素对应的分割先验
    - 核心思路：在 LLM 的因果自注意力中，每个词 token $z^i$ 对所有前序 token（包括图像 token）有注意力权重。提取词 token 对图像 token 的注意力权重并 unflatten 回 2D 空间结构，得到 $a^i \in \mathbb{R}^{h \times w}$。对于多词描述的物体，对各词的注意力图取平均或最大值。将 $M$ 层 $N$ 头共 $MN$ 个注意力图堆叠为 $A \in \mathbb{R}^{MN \times h \times w}$，再双线性插值到 $64 \times 64$
    - 设计动机：LMM 虽然没有用像素级标注训练过，但 transformer 的注意力机制天然编码了文本与图像区域的对应关系。可视化表明这些注意力图中包含粗略但有意义的空间轮廓

2. **Mask Head (Decoder + Refiner)**:

    - 功能：将粗糙的注意力图翻译为精细的分割 mask
    - 核心思路：Mask decoder 是一个 3-stage U-Net，输入 $MN$ 通道的注意力图张量，输出二值 mask logits。Mask refiner 基于 SAM 的 mask head 改造：将 decoder 输出的 mask logits 通过 SAM 的 prompt encoder 转为 dense prompt embedding，mask 的 bounding box 转为 box embedding，再结合从 LLM 各层提取的文本嵌入（通过可学习标量加权求和）作为 sparse prompt embedding。三种 embedding 与 SAM 的冻结 ViT 图像编码器特征一起输入 refiner 生成精细 mask
    - 设计动机：注意力图提供了空间和几何线索但分辨率低、边界粗糙。U-Net 可以有效翻译这些多通道先验；SAM 的高质量 mask 先验则用来优化边界精度。整个 mask head 参数量极小

3. **Keyword Selector**:

    - 功能：自动发现文本中需要视觉定位的物体关键词
    - 核心思路：在 LLM 的 transformer 层顶部放一个线性层，将 $d$ 维隐藏状态映射为 1 维分数，经 sigmoid 归一化到 [0,1]。训练时用 BCE loss 监督，推理时分数超过阈值 $\lambda=0.3$ 的词被标记为正样本。相邻的正样本词合并为一个物体
    - 设计动机：SpaCy 等外部 NLP 工具会解析出过多非物体名词导致噪声；训练一个轻量线性层直接判断哪些词需要定位更精确（F1 从 SpaCy 的 57.8% 提升到 82.8%）

### 损失函数 / 训练策略

Mask decoder 和 mask refiner 分别用 BCE + DICE loss 训练。Keyword selector 用 BCE loss。整个训练仅使用 RefCOCO(+/g) 和 PNG 数据集（约 190K 样本），在 8 卡 A800-40G 上约 20 小时完成，batch size=8，训练 8 个 epoch，AdamW 优化器，学习率 1e-4。

## 实验关键数据

### 主实验

| 模型 | MME↑ | MMBench↑ | RefCOCO cIoU↑ | PNG All↑ |
|------|------|----------|---------------|----------|
| GLaMM-FS-7B | 14/9 | 36.8 | 78.6 | 55.8 |
| LISA-7B | 1/1 | 0.4 | 74.9 | - |
| PixelLM-7B | 309/135 | 17.4 | 73.0 | 43.1 |
| F-LMM (LLaVA-1.6-M-7B) | 1501/324 | 69.5 | 75.7 | 66.5 |
| F-LMM (DeepSeekVL-7B) | 1468/298 | 73.2 | 76.1 | 65.7 |

F-LMM 在对话 benchmark 上完全保持原始 LMM 的性能（MME 1500+ vs 微调方法的 0-300），同时定位性能 competitive。

### 消融实验

| 配置 | PNG All | PNG Thing | PNG Stuff |
|------|---------|-----------|-----------|
| 无 mask refiner | 50.8 | 48.6 | 55.9 |
| + mask prompt | 63.4 | 62.0 | 66.8 |
| + mask + box prompt | 63.7 | 62.2 | 67.1 |
| + mask + box + text prompt | 64.9 | 63.4 | 68.3 |

### 关键发现

- SAM-based mask refiner 贡献最大（+14.1% PNG），证明注意力图虽含有效空间先验但需要精炼
- 语言线索（文本嵌入）对 refiner 有额外 1.2% 提升，说明多模态信息的互补性
- F-LMM 在推理分割（reasoning segmentation）的长句子子集上显著领先（49.1% vs LISA 36.6%），得益于保完好的推理能力
- 关键词选择器 F1=82.8% 远超 SpaCy 的 57.8%（精度从 41.1% 提到 72.5%）

## 亮点与洞察

- **"冻结即最优"的设计哲学**：在定位任务上，冻结 LMM + 轻量解码器不仅保住了对话能力，定位性能也 competitive。这说明与其投入大量资源收集高质量标注数据来缓解微调带来的灾难性遗忘，不如直接不微调
- **注意力图作为"免费"分割先验**：LMM 训练时从未见过像素级标注，但其注意力机制中已经编码了足够的空间信息。这个发现对理解大模型的内在表征很有启发
- **与任意 LMM 即插即用**：F-LMM 不依赖特定 LMM 架构，可以直接应用于 10+ 种不同 LMM，这种通用性非常有价值

## 局限与展望

- 定位精度（cIoU）仍低于微调方法约 2-3 个点（75.7 vs 78.6），对于需要极高精度的场景可能不够
- 仅在 MLP-based 的 LMM 上验证（保留 2D 拓扑结构），对于使用 cross-attention 的 LMM（如 Flamingo）需要额外适配
- 注意力图的质量取决于 LMM 本身的视觉编码器，如果基础 LMM 视觉理解弱，F-LMM 的定位也会相应变差
- 训练数据仅用了 RefCOCO(+/g) 和 PNG，增加更多分割数据类型（如全景分割）可能进一步提升

## 相关工作与启发

- **vs LISA/GLaMM/PixelLM**: 这些方法微调 LMM 学习 [SEG] token，定位精度高但对话能力近乎丧失；F-LMM 完美保持对话能力，定位 competitive
- **vs LLaVA-G**: LLaVA-G 用带标注的对话数据尝试平衡两者，成本极高且对话能力仍有下降；F-LMM 不需要任何对话标注数据
- **vs SAM**: SAM 提供高质量 mask prior，但需要人工 prompt；F-LMM 自动从对话中发现物体并生成 prompt

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 冻结 LMM 做 grounding 的思路新颖且反直觉，注意力图分割先验的发现具有启发性
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 10+ LMM、QA + RES + PNG + 推理分割 + GCG 五类任务，消融完整
- 写作质量: ⭐⭐⭐⭐ 问题提出清晰，动机强烈（Table 1 中微调方法的 0 分非常有冲击力）
- 价值: ⭐⭐⭐⭐⭐ 提出了一种实用的"两全其美"方案，对构建通用 AI 助手有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [PARTONOMY: Large Multimodal Models with Part-Level Visual Understanding](../../NeurIPS2025/segmentation/partonomy_large_multimodal_models_with_part-level_visual_understanding.md)
- [HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](../../ICCV2025/segmentation/himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models](../../ECCV2024/segmentation/openpsg_openset_panoptic_scene_graph_generation_via_large_mu.md)
- [Scale Efficient Training for Large Datasets](scale_efficient_training_for_large_datasets.md)

<!-- RELATED:END -->
