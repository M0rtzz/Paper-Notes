---
title: >-
  [论文解读] Siamese Vision Transformers are Scalable Audio-Visual Learners
description: >-
  [ECCV 2024][语音][音视觉学习] 提出AVSiam框架，使用单个共享权重的ViT backbone同时处理音频和视觉输入，结合多比例随机掩码策略和对比+重建双目标预训练，以极低成本（比MAViL快28.9倍）在音视觉分类和检索上达到SOTA性能。
tags:
  - ECCV 2024
  - 语音
  - 音视觉学习
  - 孪生网络
  - Transformer
  - 对比学习
  - 掩码自编码器
---

# Siamese Vision Transformers are Scalable Audio-Visual Learners

**会议**: ECCV 2024  
**arXiv**: [2403.19638](https://arxiv.org/abs/2403.19638)  
**代码**: https://github.com/GenjiB/AVSiam (有)  
**领域**: 音频语音  
**关键词**: 音视觉学习, 孪生网络, Vision Transformer, 对比学习, 掩码自编码器

## 一句话总结

提出AVSiam框架，使用单个共享权重的ViT backbone同时处理音频和视觉输入，结合多比例随机掩码策略和对比+重建双目标预训练，以极低成本（比MAViL快28.9倍）在音视觉分类和检索上达到SOTA性能。

## 研究背景与动机

**领域现状**: 音视觉表征学习近年取得显著进展，MAE等方法在音视觉联合建模上表现优异，但现有方法普遍依赖独立的音频和视觉backbone。

**现有痛点**: 独立backbone带来高昂的计算和内存开销。MAViL需要5120 V100 GPU小时预训练，MBT需要48GB以上GPU显存，大量研究者无法负担。

**核心矛盾**: 独立backbone虽然性能好，但参数效率低、不可扩展、无法灵活处理缺失模态（仅音频/仅视觉）的情况，且模态特定的设计引入了手工先验和归纳偏置，可能不利于数据驱动的表征学习。

**本文目标**: 用单个共享权重的ViT backbone替代独立的音频和视觉编码器，实现高效可扩展的音视觉预训练。

**切入角度**: 音频频谱图是2D信号，与图像结构类似，ViT完全有能力同时处理两种模态。已有研究证明ViT可以跨模态通用，但音视觉领域仍坚持使用独立backbone。

**核心 idea**: 共享权重的孪生ViT + 多比例随机掩码即可高效学习音视觉表征，无需模态特定设计。

## 方法详解

### 整体框架

AVSiam由三部分组成：(1) 共享权重的ViT编码器，同时处理音频频谱图和视觉帧；(2) 多模态融合层（2层自注意力），联合处理音视觉token；(3) 6层自注意力解码器，用于重建被掩码的token。预训练使用对比匹配+重建的双目标，微调采用混合模态输入策略。

### 关键设计

1. **共享权重的音视觉编码器 (Shared-Weight Encoder)**:

    - 功能：用标准ViT同时编码视觉帧和音频频谱图。
    - 核心思路：视觉帧 $I \in \mathbb{R}^{H_v \times W_v \times 3}$ 切分为 $n$ 个patch得到视觉嵌入 $\mathbf{X}_v \in \mathbb{R}^{n \times d}$；音频频谱图 $A \in \mathbb{R}^{H_a \times W_a}$ 同样切分为 $k$ 个patch得到音频嵌入 $\mathbf{X}_a \in \mathbb{R}^{k \times d}$。对于音频的单通道输入，将ViT预训练的3通道投影层权重取平均得到单通道权重。编码后分别通过平均池化得到 $\mathbf{F}_a \in \mathbb{R}^d$ 和 $\mathbf{F}_v \in \mathbb{R}^d$。
    - 设计动机：音频频谱图具有与图像类似的2D空间结构，共享权重可大幅减少参数（100M vs 164-172M）和显存占用（10.9G vs 20.6G），使模型可扩展到更大的数据集和模型尺寸。

2. **多比例随机掩码 (Multi-Ratio Random Masking)**:

    - 功能：在每个训练迭代中以不同比例（0%-50%）随机掩码音频和视觉patch。
    - 核心思路：每个mini-batch中的音视觉实例被随机分配不同的掩码比例。为了高效GPU实现，从预定义的离散比例集合中选取，确保相同未掩码token数量的实例可打包在一起。
    - 设计动机：固定掩码比例需要在效率和精度之间权衡——高比例节省显存但损失信息，低比例保留信息但计算昂贵。多比例方案让模型在多种信息量水平上学习，形成更鲁棒的表征，同时兼顾效率（160 V100小时 vs 最优固定比例362小时）。

3. **双目标预训练与混合模态微调 (Dual Objectives & Mixed Modality Finetuning)**:

    - 功能：预训练结合对比匹配和token重建；微调时随机选择音频/视觉/音视觉输入。
    - 核心思路：对比损失 $\mathcal{L}_c = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(g(\mathbf{F}_a^i, \mathbf{F}_v^i)/\tau)}{\sum_{j=1}^B \exp(g(\mathbf{F}_a^i, \mathbf{F}_v^j)/\tau)}$；重建损失 $\mathcal{L}_{rec} = \frac{1}{B}\sum_{i=1}^B (\tilde{A}^i - A^i)^2 + (\tilde{I}^i - I^i)^2$；总损失 $\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_c$。微调时一半迭代随机选择单模态（仅音频或仅视觉），另一半使用双模态输入。
    - 设计动机：对比学习拉近匹配音视觉对，重建目标学习更细粒度的跨模态相关性。混合模态微调使共享backbone在缺失模态场景下也能鲁棒工作。

### 损失函数 / 训练策略

- **预训练**: 在AudioSet-2M上预训练，使用Adam优化器（lr=1e-4），两个独立的loss scaler分别用于对比和重建损失
- **微调**: 在AudioSet-20K/2M/VGGSound上微调，学习率分别为1e-4/5e-6/5e-5，分类头的学习率是编码器的10-100倍
- 使用ImageNet-21K预训练的ViT初始化，多模态层使用预训练ViT最后两层初始化

## 实验关键数据

### 主实验

| 方法 | 音频编码器 | 视觉编码器 | V100小时 | 参数量 | AS-20K mAP | AS-2M mAP | VGGSound Acc |
|------|-----------|-----------|---------|-------|-----------|----------|-------------|
| MBT | AST-B | ViT-B | - | 172M | 43.9 | 49.6 | 64.1 |
| AV-MAE | AST-B | ViT-B | 2854 | 179M | - | 50.0 | 64.2 |
| CAV-MAE | AST-B | ViT-B | 672 | 164M | 42.0 | 51.2 | 65.5 |
| MAViL-Stage2 | AST-B | ViT-B | 5120 | 172M | 44.9 | 53.3 | 67.1 |
| **AVSiam-Base** | ViT-B共享 | ViT-B共享 | **177** | **100M** | 41.6 | 50.1 | 64.9 |
| **AVSiam-Base+** | ViT-B共享 | ViT-B共享 | **450** | **100M** | 43.0 | 51.4 | 66.7 |
| **AVSiam-Large** | ViT-L共享 | ViT-L共享 | **310** | **332M** | 44.1 | 52.1 | 67.1 |
| **AVSiam-Huge** | ViT-H共享 | ViT-H共享 | **800** | **672M** | **45.0** | **54.1** | **68.0** |

### 消融实验

| 配置 | AS-20K mAP | V100小时 | 说明 |
|------|-----------|---------|------|
| 固定25%掩码 | 40.8 | 362 | 最优固定比例 |
| 固定50%掩码 | 39.5 | 142 | 中等 |
| 固定75%掩码 | 38.6 | 120 | 信息损失过多 |
| 纯对比学习 | 40.4 | 510 | 无掩码无重建 |
| **多比例掩码(ours)** | **41.3** | **160** | 性能最优且高效 |
| AVSep-Large（独立编码器）| 52.0 (A+V) | - | 参数640M, GPU 20.6G |
| **AVSiam-Large（共享编码器）** | **52.1 (A+V)** | - | 参数332M, GPU 10.9G |

### 关键发现

- AVSiam-Huge在所有数据集上达到SOTA（AS-20K: 45.0 mAP, AS-2M: 54.1 mAP, VGGSound: 68.0%），仅需MAViL 15%的预训练时间。
- 共享编码器在音视觉检索上大幅优于独立编码器（VGGSound A→V: 20.4 vs 12.8 R@1），因为共享编码器将两种模态投射到更统一的潜在空间。
- AVSiam在缺失模态场景下远优于CAV-MAE：视觉-only在VGGSound上46.0% vs 27.3%，音频-only 55.7% vs 51.8%。
- ViT作为共享backbone优于AST：AST在音频上略好（+1 mAP），但在视觉和音视觉上远差。

## 亮点与洞察

- **极致简洁**: 没有复杂的模态特定设计，仅用一个标准ViT处理两种模态，思路简单但效果惊人。
- **可扩展性**: 得益于参数效率，可轻松扩展到ViT-Huge和更大数据集，且扩展收益优于独立编码器方法。
- **共享空间优势**: t-SNE可视化表明AVSiam将音频和视觉特征映射到语义上更可分且模态间更对齐的空间，这是独立编码器难以实现的。
- **吞吐量**: AVSiam每秒75.4样本 vs CAV-MAE 22.5 vs MAViL 3.84，快了近20倍。

## 局限与展望

- 在Base规模下，AVSiam在AudioSet-2M上与CAV-MAE和MAViL仍有差距，需通过扩展数据或模型尺寸弥补。
- 音频投影层的3通道→1通道权重平均是简单处理，可能有更好的适配方式。
- 仅验证了分类和检索任务，未探索音视觉问答、事件定位、分割等更复杂的下游任务。
- 多比例掩码的比例范围（0%-50%）是固定的，可能不是所有场景的最优选择。

## 相关工作与启发

- **vs CAV-MAE**: CAV-MAE使用独立的AST和ViT编码器，计算成本高（672小时）。AVSiam用共享ViT在更少时间内（177小时）达到接近性能，且检索任务远优。
- **vs MAViL**: MAViL是当前最强方法但需5120 V100小时两阶段训练，AVSiam-Huge仅用800小时即超越之，证明共享权重+高效训练策略更具实用价值。
- **vs 独立编码器baseline (AVSep)**: 相同设置下，共享编码器在1.92倍更少参数和近2倍更少GPU显存下达到相当甚至更好的性能，有力证明了共享范式的可行性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 共享权重的思路虽不完全新颖，但在音视觉领域的系统验证和工程实现很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 分类/检索/消融/扩展性/缺失模态/吞吐量/可视化，覆盖非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，效率对比图表直观，实验分析充分
- 价值: ⭐⭐⭐⭐⭐ 大幅降低音视觉学习门槛，使更多研究者可以参与，且SOTA性能证明无需牺牲效果

<!-- RELATED:START -->

## 相关论文

- [SiNGER: A Clearer Voice Distills Vision Transformers Further](../../ICLR2026/audio_speech/singer_a_clearer_voice_distills_vision_transformers_further.md)
- [Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)
- [CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)

<!-- RELATED:END -->
