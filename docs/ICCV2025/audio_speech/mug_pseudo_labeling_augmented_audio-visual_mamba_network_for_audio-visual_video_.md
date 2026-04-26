---
title: >-
  [论文解读] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing
description: >-
  [ICCV 2025][语音][音视频解析] 提出MUG框架，通过伪标签增强的跨模态随机组合数据增强策略和音视频Mamba网络，同时提升弱监督音视频解析任务中段级和事件级的预测性能。
tags:
  - ICCV 2025
  - 语音
  - 音视频解析
  - Mamba
  - 伪标签
  - 数据增强
  - 弱监督学习
---

# MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing

**会议**: ICCV 2025  
**arXiv**: [2507.01384](https://arxiv.org/abs/2507.01384)  
**代码**: [https://github.com/WangLY136/MUG](https://github.com/WangLY136/MUG)  
**领域**: 音频-视觉理解  
**关键词**: 音视频解析, Mamba, 伪标签, 数据增强, 弱监督学习

## 一句话总结

提出MUG框架，通过伪标签增强的跨模态随机组合数据增强策略和音视频Mamba网络，同时提升弱监督音视频解析任务中段级和事件级的预测性能。

## 研究背景与动机

音视频视频解析（AVVP）任务旨在弱监督条件下（仅视频级标签）预测视频中每个时间段的模态特定事件（视觉/听觉/视听事件）并定位其时间边界。现有方法面临三大挑战：

**数据增强困难**：受弱监督限制，缺少单模态标签信息，模型难以学习大量段级事件组合

**跨模态噪声**：两个模态中的事件可能完全无关，简单的模态交互会引入噪声干扰

**长序列建模**：视觉输入序列token长度接近ViT-S的阈值（约2000），Transformer处理长序列效率低下，而单纯的Mamba在单帧图像识别上存在缺陷

## 方法详解

### 整体框架

MUG由两部分组成：（1）跨模态随机组合数据增强（CMRC）；（2）AV-Mamba网络（含MBA、AMF、MFE、PLSIM四个模块）。输入音视频特征经Mamba注意力处理后，通过自适应融合和特征增强，最后引入文本语义信息约束预测。

### 关键设计

1. **跨模态随机组合（CMRC）**：核心是利用改进的伪标签实现AVVP任务的数据增强。首先基于VALOR提取的伪标签，人工标注空标签（视频通常不会缺少图像），然后从两个不同视频中分别提取视觉和音频轨道随机组合生成新视频，新标签为视觉伪标签与听觉伪标签的并集。按实际分布生成5个批次数据（1585-12096条），实验发现Batch4（1倍数据量）为最佳平衡点。

2. **Mamba注意力（MBA）**：基于CBAM的思想但用Mamba实现。对每个段特征分别进行全局最大池化和平均池化，送入共享Mamba块得到通道注意力权重 $W_t^m$ 和空间注意力权重 $S_t^m$，通过逐元素乘法得到增强特征 $\hat{f}_t^m$。Mamba的因果建模能力有助于捕获多帧间的因果关系。

3. **自适应Mamba融合（AMF）**：对视觉和音频特征分别进行前向/后向SSM扫描及动态扫描。关键创新是两个模态**共享状态转移矩阵B**（控制隐状态演化），而保持输入转移矩阵A和输出转移矩阵C独立，从而在减少参数的同时保留单模态独立性并捕获跨模态相似信息。融合后通过门控策略得到 $f_m^{AMF}$ 和混合特征 $f_{mix}^{AMF}$。

4. **Mamba特征增强（MFE）**：接收视觉、音频和混合三路输入。先对两个模态特征做平均池化降维，再对同时间步的两模态特征做逐元素乘法，通过Sigmoid生成通道增强向量。这放大了跨模态相似特征并抑制不相似特征。

5. **伪标签语义交互（PLSIM）**：引入文本模态排除跨模态噪声。将伪标签对应的事件类别通过CLIP/CLAP文本编码器提取语义特征，然后通过多个MLP映射为缩放因子 $\gamma$ 和偏置因子 $\rho$，与音视频特征做自适应融合：$F_{audio} = f_a^{MFE} \odot \gamma_{a1} + \gamma_{a2} + f_a^{MFE}$。

### 损失函数 / 训练策略

- 继承HAN模型的多模态多实例学习损失
- 伪标签提供细粒度监督
- 使用AdamW优化器，batch size 64，学习率 $3 \times 10^{-4}$，训练20个epoch
- 在NVIDIA RTX A6000 GPU上训练

## 实验关键数据

### 主实验

在LLP数据集上与SOTA方法的F1-score对比：

| 方法 | Seg-A | Seg-V | Seg-AV | Seg-Type | Seg-Event | Evt-A | Evt-V | Evt-AV | Evt-Type | Evt-Event |
|------|-------|-------|--------|----------|-----------|-------|-------|--------|----------|-----------|
| HAN | 60.1 | 52.9 | 48.9 | 54.0 | 55.4 | 51.3 | 48.9 | 43.0 | 47.7 | 48.0 |
| VALOR | 61.8 | 65.9 | 58.4 | 62.0 | 61.5 | 55.4 | 62.6 | 52.2 | 56.7 | 54.2 |
| CoLeaF | 64.2 | 64.4 | 59.3 | 62.6 | 62.5 | 57.6 | 63.2 | 54.2 | 57.9 | 55.6 |
| **MUG** | **65.4** | **66.5** | **59.9** | **63.9** | **64.7** | **59.5** | **63.9** | **55.3** | **59.6** | **57.7** |

MUG在所有指标上均取得最佳。相比CoLeaF，视觉段级+2.1%，音频段级+1.2%，Event@AV段级+2.2%，事件级+2.1%。

### 消融实验

各模块消融（Segment-level Event@AV）：

| 配置 | Seg-A | Seg-V | Seg-Event@AV |
|------|-------|-------|-------------|
| MUG完整 | 65.4 | 66.5 | 64.7 |
| w/o CMRC | 62.7 | 65.2 | 62.2 |
| w/o MBA | 64.5 | 66.1 | 63.9 |
| w/o AMF | 64.1 | 66.3 | 63.4 |
| w/o MFE | 63.8 | 64.6 | 62.8 |
| w/o PLSIM | 64.8 | 66.5 | 64.0 |

CMRC贡献最大（移除后Event@AV下降2.5%），MFE次之。

与Transformer/CNN架构对比：MUG（7.6M参数）vs Transformer（19.3M）vs CNN（6.5M），MUG以最少参数量级（相比Transformer）取得最佳性能。

### 关键发现

- CMRC数据增强在Batch4（1倍原始数据量）时达到最佳，生成过多数据会引入噪声和过拟合
- CMRC可泛化到多个基线模型（HAN、MGN、JoMoLD均获得一致提升）
- PLSIM在数据增强量过大（12000+样本）时效果下降甚至产生负面影响
- Mamba架构在参数效率上显著优于Transformer

## 亮点与洞察

- **数据增强的创新**：首次为AVVP任务设计跨模态数据增强策略，巧妙利用伪标签的单模态信息实现视觉和音频轨道的自由组合
- **参数共享策略精妙**：AMF中仅共享状态转移矩阵B而保持A、C独立，在跨模态信息共享和单模态独立性间取得平衡
- **Mamba-Transformer混合架构**：Mamba处理时序因果关系，HAN的Transformer补充空间全局依赖，实现互补

## 局限与展望

- 依赖LLP数据集（仅25类，标注噪声多），数据集规模偏小
- 伪标签仍存在质量问题，人工标注空标签的方式扩展性有限
- 未探索更大规模的预训练或多数据集联合训练
- 动态扫描分支的具体实现细节描述不够清晰

## 相关工作与启发

- VALOR的伪标签策略为数据增强提供了基础
- Vision Mamba和VMamba展示了Mamba在视觉任务中的潜力
- 可借鉴该框架的跨模态参数共享思想应用于其他多模态任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据增强策略和Mamba融合架构都有创新
- **实验充分度**: ⭐⭐⭐⭐ 消融全面，多基线验证CMRC有效性
- **写作质量**: ⭐⭐⭐ 结构清晰但部分细节描述不够
- **综合价值**: ⭐⭐⭐⭐ 为AVVP任务提供了有效的数据增强新范式

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing](../../CVPR2025/audio_speech/uwav_uncertainty-weighted_weakly-supervised_audio-visual_video_parsing.md)
- [\[ICCV 2025\] VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](../../ECCV2024/audio_speech/label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](../../ECCV2024/audio_speech/coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

<!-- RELATED:END -->
