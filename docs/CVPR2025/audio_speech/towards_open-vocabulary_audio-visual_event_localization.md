---
title: >-
  [论文解读] Towards Open-Vocabulary Audio-Visual Event Localization
description: >-
  [CVPR 2025][语音][开放词汇] 首次定义开放词汇音视频事件定位（OV-AVEL）任务，构建了包含 24800 个视频、67 类事件的 OV-AVEBench 基准，并提出基于 ImageBind 的训练免和微调两种基线方法，其中仅用 1 层时序 Transformer 微调即达 57.8% 平均性能。
tags:
  - CVPR 2025
  - 语音
  - 开放词汇
  - 音视频事件定位
  - ImageBind
  - 时序建模
  - 零样本泛化
---

# Towards Open-Vocabulary Audio-Visual Event Localization

**会议**: CVPR 2025  
**arXiv**: [2411.11278](https://arxiv.org/abs/2411.11278)  
**代码**: https://github.com/jasongief/OV-AVEL (有)  
**领域**: 音频语音 / 多模态  
**关键词**: 开放词汇, 音视频事件定位, ImageBind, 时序建模, 零样本泛化

## 一句话总结

首次定义开放词汇音视频事件定位（OV-AVEL）任务，构建了包含 24800 个视频、67 类事件的 OV-AVEBench 基准，并提出基于 ImageBind 的训练免和微调两种基线方法，其中仅用 1 层时序 Transformer 微调即达 57.8% 平均性能。

## 研究背景与动机

**领域现状**：音视频事件定位（AVEL）旨在判断视频每个时间段中是否发生了音视频一致的事件并分类。现有方法（如 CMRA、PSP、MM-Pyramid）在固定类别集合上训练和测试，假设测试时只会遇到训练时见过的事件类别。

**现有痛点**：真实场景中不可能枚举所有事件类别——新的声音和视觉事件不断出现。现有闭集方法面对未见类别时性能断崖式下降（如 MM-Pyramid 从 68.6% 暴跌到 29.9%），且现有 AVE 数据集仅 4143 视频、28 类，规模太小。

**核心矛盾**：需要模型具备对未见类别事件的零样本泛化能力，但专门为新任务训练又会破坏预训练模型的通用表示能力。

**本文目标** 形式化定义 OV-AVEL 任务（训练只用 seen 类，测试需同时处理 seen + unseen 类），构建大规模基准，并建立简洁有效的基线方法。

**切入角度**：利用 ImageBind 的跨模态统一表示空间——音频、视觉和文本特征在同一空间中对齐，天然适合开放词汇场景。

**核心 idea**：用 ImageBind 统一表示 + 几何均值融合 + 单层时序Transformer = 简洁的开放词汇音视频事件定位基线。

## 方法详解

### 整体框架

视频被划分为 T=10 个 1 秒段。用 ImageBind 分别提取每段的音频特征 $\bm{a}_t$ 和视觉特征 $\bm{v}_t$（维度 $d=1024$），以及所有候选类别的文本特征 $\bm{e}_c$。分别计算音频-文本和视觉-文本的相似度矩阵，用几何均值融合后预测每段的事件类别。

### 关键设计

1. **训练免基线（零样本）**:

    - 功能：无需任何训练即可进行开放词汇事件定位
    - 核心思路：计算每段音频和视觉特征与所有类别文本特征的余弦相似度。对每段分别取音频最高分类和视觉最高分类——两者一致则判定为该事件，不一致则判定为背景（"other"类）
    - 设计动机：利用 ImageBind 预训练的跨模态对齐能力，音视频一致性检查天然过滤掉单模态噪声

2. **微调基线（轻量时序增强）**:

    - 功能：用少量可学习参数增强时序建模，大幅提升性能
    - 核心思路：在 ImageBind 的音频和视觉编码器后分别插入 $L=1$ 层可学习 Transformer 块捕捉时序依赖，训练时仅用 seen 类数据。音视频融合采用几何均值 $S_{ave}' = \sqrt{S_{ae}' \odot S_{ve}'}$（Hadamard 积取平方根），预测概率取最大值对应的类别
    - 设计动机：几何均值相比算术平均对单模态虚警更鲁棒——如果一个模态给出低分，几何均值会被显著拉低（57.8% vs 39.0%）

3. **"Other"类的关键作用**:

    - 功能：处理不属于任何候选类别的背景段
    - 核心思路：在候选类别文本列表末尾加入"other"类文本特征，让模型可以将背景段分类到"other"而非误分配到某个具体类别
    - 设计动机：没有"other"类时性能从 57.8% 暴跌到 47.0%（-10.8%），因为背景段被强制分配到不相关的事件类别

### 损失函数 / 训练策略

单一交叉熵损失：$\mathcal{L} = \text{CE}(S_{ave}', Y')$。Adam 优化，lr=5e-5，batch size=32，仅训练 5 个 epoch。可训练参数仅 8.4M（1 层 Transformer），其余 ImageBind 参数冻结。

## 实验关键数据

### 主实验

OV-AVEBench 上的平均性能（Acc/Seg-F1/Event-F1 的均值）：

| 方法 | Seen Avg | Unseen Avg | Total Avg |
|------|----------|------------|-----------|
| Video-LLaMA2 | 40.9 | 38.6 | 39.3 |
| CLIP&CLAP | 41.6 | 41.8 | 41.7 |
| Training-free (Ours) | 45.5 | 47.0 | 46.6 |
| **Fine-tuning (Ours)** | **62.9** | **55.8** | **57.8** |

闭集方法迁移到 OV-AVEL 时的 seen→unseen 性能崩溃：

| 闭集方法 | Seen Avg | Unseen Avg | 下降 |
|---------|----------|------------|------|
| MM-Pyramid | 68.6 | 29.9 | -38.7 |
| AVE method | 65.4 | 34.0 | -31.4 |
| CMRA | 59.4 | 31.1 | -28.3 |

### 消融实验

| 配置 | Total Avg | 说明 |
|------|-----------|------|
| 完整模型 (L=1 Transformer) | **57.8** | 最优 |
| 无"other"类 | 47.0 | 下降 10.8 个点 |
| 算术平均融合 | 39.0 | 下降 18.8 个点 |
| Linear 层替代 Transformer | 39.0 (unseen 仅 28.3) | 时序建模对泛化至关重要 |
| L=2 层 Transformer | 56.9 | 过拟合 |
| 仅跨模态时序交互 | 46.5 | 反而损害性能 |

### 关键发现
- **几何均值融合至关重要**：相比算术平均高 18.8 个点，因为它对单模态虚警有天然的抑制作用（一个模态低分 → 乘积被拉低）
- **时序 Transformer 是泛化关键**：Line 层在 seen 类上更好（65.8 vs 62.9），但在 unseen 类上崩溃（28.3 vs 55.8），说明时序建模帮助提取更通用的事件模式
- **跨模态交互反而有害**：ImageBind 已提供了足够的跨模态语义对齐，额外的跨模态注意力层可能破坏预训练表示
- **数据效率极高**：仅用 25% 训练数据（~3300 视频）就能达到 57.7%，接近 100% 数据的 57.8%

## 亮点与洞察

- **任务定义本身是主要贡献**：OV-AVEL 作为新任务的形式化定义和大规模基准构建，比方法本身更有长期价值。24800 视频、67 类事件远超现有 AVE 数据集
- **简洁即有效**：1 层 Transformer + 几何均值融合 + "other"类，三个极简设计组合就达到了强基线。这对后续工作划定了清晰的起跑线
- **几何均值融合的直觉**：音视频一致性天然适合乘法——只有两个模态都"确信"某事件存在时，融合分数才高。这个 insight 可迁移到其他多模态融合场景

## 局限与展望

- **Seen-Unseen 性能差距仍大**：微调后 seen 62.9% vs unseen 55.8%，7 个点的差距说明泛化仍有空间
- **事件级 F1 较低**：unseen data 的 Event-F1 仅 47.5%，说明时序边界定位精度不够
- **固定时间分段**：10 秒视频等分为 10 段的设计过于刚性，无法处理变长事件和跨段事件
- **方法相对简单**：基线性质明显，后续需要更复杂的时序建模和跨模态交互设计
- **ImageBind 依赖**：整个方法强依赖 ImageBind 的表示质量，换其他基础模型效果未知

## 相关工作与启发

- **vs 闭集 AVEL 方法（CMRA/PSP）**: 闭集方法在 unseen 类上断崖式下降 28-38 个点，OV-AVEL 的 fine-tuning 基线仅下降 7 个点，证明了开放词汇框架的必要性
- **vs Video-LLaMA2**: 大语言模型直接应用于 OV-AVEL 效果一般（39.3%），可能因为 LLM 的事件时序定位能力不如专门设计的方法
- **vs AudioCLIP/CLAP**: 纯音频-文本对齐方法在开放词汇场景下表现中等，缺少视觉模态的互补信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 新任务定义 + 大规模基准是核心贡献，方法本身是基线级别
- 实验充分度: ⭐⭐⭐⭐⭐ 多种基线对比、闭集方法适配、详尽的消融（融合策略/层数/数据量/模态交互）
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，评估协议全面
- 价值: ⭐⭐⭐⭐ 为音视频事件定位社区开辟了开放词汇新方向，基准和基线都有持久价值

<!-- RELATED:START -->

## 相关论文

- [Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](../../ECCV2024/audio_speech/label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [Crab: A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation](crab_a_unified_audio-visual_scene_understanding_model_with_explicit_cooperation.md)

<!-- RELATED:END -->
