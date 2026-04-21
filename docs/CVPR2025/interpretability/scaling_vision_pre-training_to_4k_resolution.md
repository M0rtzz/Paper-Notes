---
title: >-
  [论文解读] Scaling Vision Pre-Training to 4K Resolution
description: >-
  [CVPR 2025][高分辨率预训练] 本文提出PS3（Pre-training with Scale-Selective Scaling），通过局部区域与局部caption的对比学习代替全图对比，以近常数的计算开销将CLIP式视觉预训练扩展到4K分辨率，并结合top-down/bottom-up patch选择机制构建VILA-HD多模态大模型，在高分辨率感知任务上大幅超越GPT-4o和Qwen2.5-VL。
tags:
  - CVPR 2025
  - 高分辨率预训练
  - 视觉编码器
  - 自适应patch选择
  - 对比学习
  - 多模态大模型
---

# Scaling Vision Pre-Training to 4K Resolution

**会议**: CVPR 2025  
**arXiv**: [2503.19903](https://arxiv.org/abs/2503.19903)  
**代码**: https://nvlabs.github.io/PS3 (有)  
**领域**: 多模态VLM  
**关键词**: 高分辨率预训练, 视觉编码器, 自适应patch选择, 对比学习, 多模态大模型

## 一句话总结

本文提出PS3（Pre-training with Scale-Selective Scaling），通过局部区域与局部caption的对比学习代替全图对比，以近常数的计算开销将CLIP式视觉预训练扩展到4K分辨率，并结合top-down/bottom-up patch选择机制构建VILA-HD多模态大模型，在高分辨率感知任务上大幅超越GPT-4o和Qwen2.5-VL。

## 研究背景与动机

现代视觉模型（CLIP、SigLIP）仅在低分辨率（如378×378）上预训练，无法感知细粒度视觉细节。而日常任务（如驾驶辨认路标）需要高分辨率感知。根本瓶颈是**计算成本**：ViT的计算量随分辨率四次方增长。现有方法（AnyRes、S²）尝试免训练地在更高分辨率上运行预训练的低分辨率模型，但这阻止了模型利用大规模预训练数据学习高质量高分辨率表示。核心矛盾：高分辨率预训练的信息需求 vs 计算成本的爆炸增长。本文的关键洞察来自人类视觉的top-down选择机制：**不需要看整张高分辨率图像，只需对局部区域做高分辨率处理并与局部描述对齐**。这将区域大小从全图分辨率解耦，使成本近乎恒定。

## 方法详解

### 整体框架

PS3包含3个阶段：Stage 1用ViT编码全图低分辨率特征（378×378 → 27×27 tokens）；Stage 2基于低分辨率特征和轻量高分辨率辅助特征，选择显著或与文本相关的局部区域；Stage 3用同一个ViT处理选中区域的多尺度高分辨率patch，并通过Stage 1的KV cache注入全局上下文。预训练使用75M高分辨率图像和282M局部caption-bounding box对，通过局部对比损失和box监督联合训练。

### 关键设计

1. **局部对比预训练（Localized Contrastive Loss）**:
    - 功能：以近常数成本在4K分辨率上学习语言对齐的细粒度视觉表示
    - 核心思路：不做全图与全局caption的对比，而是对局部区域提取高分辨率特征与局部详细caption做对比。预训练选择最多2560个高分辨率patch。混合全局低分辨率对比来保持全局特征质量
    - 设计动机：要识别停车牌上的字，只需处理路标附近的局部区域并与"stop sign"文字对齐，不需要整张4K图。这使预训练计算比SigLIP全局对比减少79倍

2. **Top-down / Bottom-up Patch选择**:
    - 功能：根据文本prompt（top-down）或图像显著性（bottom-up）选择需要高分辨率处理的局部区域
    - 核心思路：用低分辨率视觉特征与文本嵌入（或可学习向量）做余弦相似度，得到选择分数。额外用轻量ConvNeXt（3层）提取1512分辨率的辅助高分辨率选择分数，两者融合。训练时用bounding box ground truth做选择分数的分割监督（交叉熵+DICE loss）
    - 设计动机：低分辨率特征无法定位细粒度细节，辅助高分辨率编码器弥补这一缺陷。top-down选择使MLLM可基于用户问题选择性处理相关区域，bottom-up选择找到图像中自身显著的区域

3. **高分辨率多尺度特征提取 + KV Cache**:
    - 功能：处理选中区域的多尺度patch并注入全局上下文
    - 核心思路：将高分辨率图resize到3个预定义尺度（756/1512/3780），各尺度按分辨率比例选top-k patch。添加scale-aware位置嵌入让ViT区分同一空间位置不同尺度的token。在self-attention中复用Stage 1的KV cache提供全局上下文
    - 设计动机：多尺度确保不同粒度的视觉信息都被捕获。KV cache避免局部高分辨率编码失去全局语义上下文，类似LLM中的上下文复用

### 损失函数 / 训练策略

- **对比损失**：使用SigLIP的sigmoid对比损失，ViT和文本编码器均从SigLIP-SO400M初始化
- **选择分数监督**：position-wise交叉熵 + DICE loss，ground truth由bounding box生成
- **训练时使用GT选择分数**（Teacher Forcing）：避免不准确的模型预测导致选择到无关区域
- **只池化box内token**：避免将box外的无关特征与文本对齐
- **避免内部图像对比**：确保同一batch中每张图只出现一次，防止同图不同区域的误匹配
- **VILA-HD microtuning数据**：500k patch选择微调数据 + 225k合成高分辨率VQA数据（将低分辨率图贴到4K背景上）

## 实验关键数据

### 主实验（7个高分辨率敏感Benchmark平均）

| 视觉编码器 | 最大分辨率 | HR Token数 | TextVQA | DocVQA | OCRBench | 7-avg |
|------------|----------|-----------|---------|--------|----------|-------|
| SigLIP | 378 | 0 | 62.3 | 51.9 | 387 | 49.9 |
| AnyRes | 1512 | 3136 | 67.4 | 67.9 | 468 | 56.3 |
| S² | 1512 | 2916 | 66.1 | 78.3 | 526 | 60.8 |
| **PS3** | 1512 | 3645 | **69.3** | **79.4** | **534** | **63.2** |
| **PS3** | 3780 | 3840 | **69.8** | **79.1** | **543** | **63.9** |

PS3比S²高2.4%，比AnyRes高6.9%，同时PS3可扩展到4K但AnyRes/S²不行。

### 4KPro Benchmark对比SOTA

| 模型 | 选择比例 | 延迟 | Acc |
|------|---------|------|-----|
| GPT-4o | - | - | 59.7 |
| Qwen2-VL-7B | - | 3.61s | 71.0 |
| Qwen2.5-VL-7B | - | 2.98s | 68.3 |
| **VILA-HD-4K** | 18% | 1.22s | 71.0 |
| **VILA-HD-4K** | 35% | 1.78s | **75.8** |

VILA-HD-4K比GPT-4o高16.1%，比Qwen2.5-VL高7.5%且速度快1.67倍。

### 关键发现

- **恒定成本扩展**：选择固定数量的patch即可从756扩展到4K，1512→4K仍带来+3.1%提升
- **测试时扩展**：训练时选20%，推理时选44%，可获得额外1.2%提升
- **现有benchmark不需要4K**：分析发现DocVQA等的最小可辨识分辨率（MRR）仅约1K，4KPro是首个真正需要4K的benchmark
- 视觉编码器对比中，PS3优于SigLIP2和Perception Encoder等SOTA编码器

## 亮点与洞察

- **scaling law的新维度**：揭示了视觉预训练中分辨率→性能的scaling规律，以及恒定成本和测试时scaling的可能性
- **top-down选择机制**：受人类视觉启发，让MLLM像人一样"先扫一眼全图，再聚焦相关区域"
- **4KPro benchmark**：揭示了现有benchmark的"虚假高分辨率"问题——图像分辨率高但问题不需要高分辨率

## 局限与展望

- PS3预训练需要75M高分辨率图像+282M局部caption，数据收集依赖MLLM captioner（Qwen2-VL），并非完全自建
- 高分辨率微调数据的"贴图到4K背景"策略是一种简陋的权宜之计，可能引入分布偏移
- 4KPro仅含4个场景类别，评估覆盖面有限
- Stage 3的KV cache增加了ViT的显存需求，可能限制更大模型的应用

## 相关工作与启发

- **vs AnyRes**: AnyRes将图像切成tiles送给原始ViT，缺乏高分辨率预训练，PS3多6.9%
- **vs S²**: S²做training-free的多尺度特征融合，PS3有预训练支撑多2.4%
- **vs SigLIP2**: PS3在23个benchmark上整体优于SigLIP2，说明高分辨率预训练比更好的低分辨率预训练更重要
- **vs Perception Encoder**: Meta的PE也做视觉编码器预训练，但不支持4K且性能不如PS3

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将CLIP式预训练扩展到4K，局部对比+patch选择的设计极其优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 4种scaling分析+SOTA对比+新benchmark+编码器对比+消融
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图表精美，scaling分析令人信服
- 价值: ⭐⭐⭐⭐⭐ 为高分辨率视觉感知提供了新范式，4KPro填补了benchmark空白

<!-- RELATED:START -->

## 相关论文

- [Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [POA: Pre-training Once for Models of All Sizes](../../ECCV2024/interpretability/poa_pre-training_once_for_models_of_all_sizes.md)
- [Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)
- [Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)

<!-- RELATED:END -->
