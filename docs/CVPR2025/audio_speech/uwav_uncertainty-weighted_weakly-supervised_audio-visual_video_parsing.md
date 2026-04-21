---
title: >-
  [论文解读] UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing
description: >-
  [CVPR 2025][语音][音视频解析] 提出 UWAV，一个弱监督音视频视频解析框架，通过在大规模标注数据上预训练时序感知模块生成高质量伪标签，再用不确定性加权软标签+类别平衡重加权+特征混合三种技术提升弱监督训练效果，在 LLP 数据集上刷新 SOTA。
tags:
  - CVPR 2025
  - 语音
  - 音视频解析
  - 弱监督
  - 不确定性加权
  - 伪标签
  - 特征混合
---

# UWAV: Uncertainty-Weighted Weakly-Supervised Audio-Visual Video Parsing

**会议**: CVPR 2025  
**arXiv**: [2505.09615](https://arxiv.org/abs/2505.09615)  
**代码**: 有（待公开）  
**领域**: 音频语音 / 弱监督学习  
**关键词**: 音视频解析, 弱监督, 不确定性加权, 伪标签, 特征混合

## 一句话总结

提出 UWAV，一个弱监督音视频视频解析框架，通过在大规模标注数据上预训练时序感知模块生成高质量伪标签，再用不确定性加权软标签+类别平衡重加权+特征混合三种技术提升弱监督训练效果，在 LLP 数据集上刷新 SOTA。

## 研究背景与动机

**领域现状**：音视频视频解析（AVVP）将视频中每个时间段分类为音频事件、视觉事件或音视频一致事件。由于帧级标注极其昂贵，主流方法采用弱监督——只有视频级标签，无帧级标注。

**现有痛点**：弱监督 AVVP 的核心困难是伪标签质量差。现有方法（如 VALOR、PPL）生成的帧级伪标签准确性有限，因为它们（1）在目标数据集上从零开始学习时序建模，数据量不足；（2）使用硬二值伪标签，丢失了模型对预测的置信度信息；（3）正负类别极度不平衡（大部分帧是"无事件"）。

**核心矛盾**：弱监督需要高质量伪标签来弥补缺失的帧级标注，但生成好的伪标签本身就需要强时序建模能力——鸡生蛋问题。

**本文目标** 通过外部预训练获取时序建模能力来打破这个循环，再用不确定性信息软化伪标签来进一步提升训练质量。

**切入角度**：用大规模标注数据集（UnAV）预训练时序 Transformer，迁移到目标数据集生成伪标签。将伪标签从硬 0/1 变为连续置信度值（与阈值的 sigmoid 距离），保留不确定性信息。

**核心 idea**：外部预训练 → 高质量伪标签 → 不确定性加权软标签 + 类别平衡 + 特征混合 = 更好的弱监督训练。

## 方法详解

### 整体框架

五步流程：（1）在 UnAV 上预训练时序感知 Transformer（CLIP/CLAP 特征 + 5 层自注意力）；（2）在目标数据集上用预训练模型生成帧级伪标签；（3）将硬伪标签转为不确定性加权软标签；（4）特征混合数据增强；（5）类别平衡损失重加权。最终模型用软标签+混合正则+类别平衡联合训练。

### 关键设计

1. **外部预训练的伪标签生成器**:

    - 功能：从大规模标注数据中学习时序依赖，生成高质量帧级伪标签
    - 核心思路：视觉侧用 CLIP 图像编码器提取特征，音频侧用 CLAP 编码器提取特征，分别送入 5 层 Transformer 捕捉时序依赖。用模板文本（"A photo of <<<EVENT>>>"/"This is the sound of <<<EVENT>>>"）获取事件嵌入，计算相似度得到预测。在 UnAV 数据集上用帧级标注预训练，然后在目标数据集上生成伪标签 $\hat{y}_t^v = \mathbb{1}_{\{\hat{z}_t^v > \theta^v\}} \odot y$（与视频级标签取交集）
    - 设计动机：伪标签质量比现有方法高 12.7%（视觉 F1），因为预训练的时序模型能更好地判断事件的帧级存在性

2. **不确定性加权软标签**:

    - 功能：保留模型对每帧预测的置信度信息
    - 核心思路：不用硬 0/1 伪标签，而用 $\hat{p}_t^v = \text{Sigmoid}(\hat{z}_t^v - \theta^v) \odot y$ 作为软标签。logit 距离阈值越远→Sigmoid 越接近 0 或 1→置信度越高；距离越近→Sigmoid 越接近 0.5→置信度越低（不确定）
    - 设计动机：硬伪标签将"几乎确定是事件"和"刚好过阈值的边缘样本"同等对待，但后者更可能是错误的。软标签让训练对不确定样本施加更小的梯度

3. **不确定性加权特征混合（Feature Mixup）**:

    - 功能：数据增强 + 正则化，防止对伪标签过拟合
    - 核心思路：随机选两个时间段 $(t_i, t_j)$，混合特征 $\bar{f}_{t_i,t_j} = \lambda \tilde{f}_{t_i} + (1-\lambda)\tilde{f}_{t_j}$，混合系数 $\lambda \sim \text{Beta}(1.7, 1.7)$。对应的软标签也按同比例混合
    - 设计动机：传统 Mixup 用于分类任务效果好，但弱监督场景中标签本身就有噪声，因此混合的是已经不确定性加权的软标签而非硬标签

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{w\text{-soft}} + \mathcal{L}_{\text{mix}} + \mathcal{L}_{\text{video}}$。其中 $\mathcal{L}_{w\text{-soft}}$ 是类别平衡重加权的软标签 BCE 损失，正样本权重 $w_{\text{pos}} = \frac{\text{负样本数}}{NTC} \times W$（$W=0.5$）。$\mathcal{L}_{\text{mix}}$ 是混合特征的 BCE 损失。$\mathcal{L}_{\text{video}}$ 是标准视频级 BCE 损失。

## 实验关键数据

### 主实验

LLP 数据集上的帧级 F1（%）：

| 指标 | VALOR | PPL | **UWAV** | 提升 |
|------|-------|-----|---------|------|
| Visual (V) | 65.1 | 66.7 | **70.0** | +3.3 |
| Audio-Visual (AV) | 61.2 | 61.9 | **63.4** | +1.5 |
| Type@AV | 64.3 | 64.8 | **65.9** | +1.1 |

伪标签质量对比（F1）：

| 指标 | VALOR | PPL | **UWAV** |
|------|-------|-----|---------|
| Visual | 61.7 | 61.8 | **74.5** |
| Type@AV | 66.0 | 60.6 | **72.8** |

### 消融实验

| 配置 | Type@AV |
|------|---------|
| 硬伪标签（基线） | 64.2 |
| + 软标签 | 64.4 (+0.2) |
| + 类别平衡 | 65.4 (+1.2) |
| + 特征混合 | 65.2 (+1.0) |
| **全部组合** | **65.9 (+1.7)** |

### 关键发现
- **伪标签质量是最大贡献**：预训练生成器将视觉伪标签 F1 从 61.8% 提升到 74.5%，+12.7 个点——这是所有技术中增益最大的
- **类别平衡和特征混合贡献接近**：分别贡献 1.2% 和 1.0%，两者有部分互补效应
- **软标签增益较小**：单独从硬到软只提升 0.2%，但与类别平衡和混合结合后有叠加效应
- **AVE 数据集提升微弱**：80.6% vs VALOR 80.4%，仅 +0.2%，说明方法在小数据集上优势有限

## 亮点与洞察

- **外部预训练打破弱监督循环**：用大规模标注数据预训练时序模型是解决弱监督伪标签质量问题的直接方案，虽然不优雅但极其有效（+12.7% 伪标签 F1）
- **不确定性的简洁建模**：用 Sigmoid(logit - threshold) 作为置信度，无需额外的不确定性估计网络或贝叶斯方法。距离阈值越远越确定——这个直觉非常自然且零开销
- **三种技术的正交组合**：软标签处理标签噪声、类别平衡处理分布偏斜、Mixup 处理过拟合——三者解决不同问题，组合效果大于各自之和

## 局限与展望

- **依赖外部标注数据**：需要 UnAV 数据集预训练，增加了数据需求和计算开销（80 个 epoch 预训练）
- **部分指标未超越**：音频 F1（64.2 vs PPL 65.9）和事件级 Event@AV（57.4 vs 57.9）低于某些方法
- **固定 1 秒分段**：无法捕捉跨段事件或亚秒级时间边界
- **超参数需逐数据集调优**：$\alpha$、$W$、阈值 $\theta$ 需要针对不同数据集调参
- **在小数据集效果有限**：AVE 上仅 +0.2%，说明预训练优势在数据充足时才能发挥

## 相关工作与启发

- **vs VALOR**: VALOR 在目标数据集内部学习时序建模，伪标签质量受限。UWAV 通过外部预训练获得更强的时序理解能力
- **vs PPL**: PPL 使用类似的伪标签生成思路但未利用不确定性信息。UWAV 的软标签让训练对噪声更鲁棒
- **vs HAN**: HAN 是早期弱监督方法，缺乏强预训练特征和伪标签精化策略

## 评分
- 新颖性: ⭐⭐⭐ 各组件（伪标签/软标签/Mixup/类别平衡）都不新颖，但组合方式合理有效
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、详细消融、伪标签质量分析
- 写作质量: ⭐⭐⭐⭐ 方法逐步展开，逻辑清晰
- 价值: ⭐⭐⭐ 刷新了 LLP 数据集的 SOTA，但novelty有限

<!-- RELATED:START -->

## 相关论文

- [MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](../../ECCV2024/audio_speech/coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)
- [VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)

<!-- RELATED:END -->
