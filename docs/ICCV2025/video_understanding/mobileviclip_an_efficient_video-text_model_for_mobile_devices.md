---
title: >-
  [论文解读] MobileViCLIP: An Efficient Video-Text Model for Mobile Devices
description: >-
  [ICCV 2025][视频理解][视频文本模型] 将时空结构重参数化引入高效图像-文本模型MobileCLIP，在大规模视频-文本数据集上训练，得到可在移动端运行的视频-文本模型MobileViCLIP，在零样本检索和动作识别上取得与大模型相当的性能。 现有视频预训练模型（如InternVideo2）主要基于ViT架构…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "视频文本模型"
  - "移动端部署"
  - "结构重参数化"
  - "高效推理"
  - "视频检索"
---

# MobileViCLIP: An Efficient Video-Text Model for Mobile Devices

**会议**: ICCV 2025  
**arXiv**: [2508.07312](https://arxiv.org/abs/2508.07312)  
**代码**: [https://github.com/MCG-NJU/MobileViCLIP](https://github.com/MCG-NJU/MobileViCLIP)  
**领域**: 视频理解  
**关键词**: 视频文本模型, 移动端部署, 结构重参数化, 高效推理, 视频检索

## 一句话总结

将时空结构重参数化引入高效图像-文本模型MobileCLIP，在大规模视频-文本数据集上训练，得到可在移动端运行的视频-文本模型MobileViCLIP，在零样本检索和动作识别上取得与大模型相当的性能。

## 研究背景与动机

现有视频预训练模型（如InternVideo2）主要基于ViT架构，参数量巨大（数亿到数十亿），推理延迟高，无法在移动设备上部署。参数高效迁移学习方法（PETL）虽然可训练参数少，但底层模型仍为ViT-B/L，无法满足移动端需求。图像-文本领域的高效模型（如MobileCLIP）虽然适配移动端，但缺乏时序建模能力，无法理解视频。

本文的核心动机是：**在移动设备约束下，设计一个既高效又具有强大视频-文本理解能力的基础模型**。作者选择在已有的移动端图像-文本模型MobileCLIP之上，通过引入轻量级的时序建模模块，以PETL的范式训练高质量视频-文本表示。

## 方法详解

### 整体框架

MobileViCLIP基于MobileCLIP构建，分为视频编码器和文本编码器。视频编码器对T帧图像进行编码，通过时间池化（average pooling）聚合为视频级表征，使用视频-文本对比学习（VTC）进行训练。训练时冻结文本编码器，仅微调视频编码器。提供Tiny和Small两个版本，分别基于MobileCLIP-S0和MobileCLIP-S2。

### 关键设计

1. **Spatiotemporal RepMixer**: 在原始2D深度可分离卷积RepMixer之前插入1D深度可分离卷积层进行时序建模。训练时为 $X' = \text{DWConv1D}(\text{BN}(X)) + X$，推理时可重参数化为单一卷积 $X' = \text{DWConv1D}(X)$，几乎不引入额外推理延迟。设计动机是以最小代价为高效图像模型增加时序感知能力。

2. **Spatiotemporal Attention**: 在原始MCi的注意力模块中增加可学习时间位置编码（TPE），结合已有的条件位置编码（CPE），使注意力层能建模全局时空表征。推理时TPE加到输入后同样可通过重参数化融合到CPE中。

3. **Temporal Pooling**: 无参数的帧级特征平均池化操作，将T帧特征聚合为视频级表征。简洁高效，避免引入额外参数。

### 损失函数 / 训练策略

- **VTC Loss**: 标准的InfoNCE对比学习损失，最大化正匹配视频-文本对的余弦相似度，最小化负对
    - $L_{VTC} = \frac{1}{2}(L_{V2T} + L_{T2V})$
    - 温度参数 $\tau$ 可学习
- **训练设置**: 8帧输入，分辨率256×256，AdamW优化器，学习率1e-5，3 epochs
- **训练资源**: 仅需8张RTX 3090 GPU，训练2天
- **预训练数据**: InternVid-10M-FLT（1000万高质量YouTube视频）
- **数据增强**: 随机裁剪+水平翻转

## 实验关键数据

### 主实验 (零样本视频-文本检索)

| 模型 | 参数量(M) | 移动端延迟(ms) | MSR-VTT T2V | MSR-VTT V2T | DiDeMo T2V | DiDeMo V2T |
|------|----------|--------------|-------------|-------------|------------|------------|
| InternVideo2-S14 | 133 | 282 | 35.6 | 35.9 | 33.7 | 35.5 |
| InternVideo2-L14 | 644 | 2319 | 42.1 | 44.1 | 42.8 | 43.2 |
| MobileViCLIP-Tiny | 54 | 15 | 38.7 | 38.1 | 37.1 | 37.0 |
| **MobileViCLIP-Small** | **99** | **42** | **42.5** | **43.5** | **40.7** | **41.1** |

MobileViCLIP-Small在MSR-VTT上与InternVideo2-L14性能相当，但移动端延迟快55.4倍，参数少6.5倍。

### 消融实验 (模块有效性)

| 模型配置 | MSR-VTT R@1 |
|---------|-------------|
| Baseline (MobileCLIP微调) | 38.4 |
| + Spatiotemporal RepMixer | 39.5 |
| + Spatiotemporal Attention w/o TPE | 39.1 |
| + Spatiotemporal Attention w/ TPE | 39.6 |
| + RepMixer + Attention w/ TPE | **40.1** |

两个时空模块都有效，TPE时间位置编码对理解时序变化很重要。冻结文本分支（vs. 激活双分支）不影响性能但节省3G显存。

### 关键发现

- MobileViCLIP-Small在移动端比InternVideo2-S14快6.75倍，FLOPs仅为其一半
- 零样本动作识别上，MobileViCLIP-Small在K400上63.1%（InternVideo2-S14: 62.1%），HMDB-51上53.7%甚至超过InternVideo2-L14的53.2%
- 作为时序定位任务的特征提取器，MobileViCLIP-Small超越了CLIP+SlowFast的组合特征
- 移动端attention层延迟随堆叠层数指数增长，而卷积层影响很小——这解释了为何混合架构对移动端更友好
- 视频标注任务上MobileViCLIP-Small优于ViT-B/32（BLEU-4: 48.9 vs 46.1）

## 亮点与洞察

- **极致高效**: 用8张3090训练2天就能得到移动端可部署的视频-文本基础模型，这对学术研究组非常友好
- **重参数化设计精妙**: 时空模块在推理时完全融合到卷积中，零额外延迟
- **移动端延迟分析深入**: 首次系统分析了各基础模块在移动端的延迟特性，发现attention在移动端缺乏GPU那样的优化，延迟随层数指数增长
- **泛化能力强**: 不仅限于检索，在时序定位、零样本动作检测、视频标注等下游任务上都展现出强泛化

## 局限与展望

- 在ActivityNet等长视频数据集上性能仍有差距，8帧采样不足以覆盖5-10分钟视频
- 在GPU上的吞吐量反而不如一些ViT模型（GPU对Transformer有深度优化）
- 仅使用对比学习训练，未探索更复杂的视频-文本预训练目标（如MLM、VTM）
- Text encoder完全冻结，可能限制了视频领域特定文本理解的上限

## 相关工作与启发

- MobileCLIP/FastViT的高效图像模型设计为移动端视频理解提供了良好基础
- InternVid数据集的高质量LLM caption对小模型训练非常关键
- 结构重参数化是移动端部署中"训练时复杂、推理时简单"的经典策略

## 评分

- 新颖性: ⭐⭐⭐ 方法组合有效但创新性一般，重参数化时序扩展是直观的
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多任务、延迟分析、消融实验非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，延迟分析部分有深度
- 价值: ⭐⭐⭐⭐ 首个可在移动端部署的视频-文本基础模型，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](../../ECCV2024/video_understanding/videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[ICCV 2025\] ResidualViT for Efficient Temporally Dense Video Encoding](residualvit_for_efficient_temporally_dense_video_encoding.md)
- [\[ICCV 2025\] EgoAdapt: Adaptive Multisensory Distillation and Policy Learning for Efficient Egocentric Perception](egoadapt_adaptive_multisensory_distillation_and_policy_learning_for_efficient_eg.md)
- [\[ICCV 2025\] DeSPITE: Exploring Contrastive Deep Skeleton-PointCloud-IMU-Text Embeddings for Action Recognition](despite_exploring_contrastive_deep_skeleton-pointcloud-imu-text_embeddings_for_a.md)

</div>

<!-- RELATED:END -->
