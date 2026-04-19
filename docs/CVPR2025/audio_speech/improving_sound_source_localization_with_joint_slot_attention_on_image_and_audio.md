---
title: >-
  [论文解读] Improving Sound Source Localization with Joint Slot Attention on Image and Audio
description: >-
  [CVPR 2025][语音][声源定位] 提出联合槽注意力机制将图像和音频同时分解为目标/非目标表示，通过跨模态注意力匹配和对比学习实现精确声源定位，在 Flickr-SoundNet 上达到 65.16% AUC、86.00% cIoU SOTA。
tags:
  - CVPR 2025
  - 语音
  - 声源定位
  - 槽注意力
  - 跨模态对比
  - 注意力匹配
  - 音视频对齐
---

# Improving Sound Source Localization with Joint Slot Attention on Image and Audio

**会议**: CVPR 2025  
**arXiv**: [2504.15118](https://arxiv.org/abs/2504.15118)  
**代码**: 有  
**领域**: 音频语音 / 声源定位  
**关键词**: 声源定位, 槽注意力, 跨模态对比, 注意力匹配, 音视频对齐

## 一句话总结

提出联合槽注意力机制将图像和音频同时分解为目标/非目标表示，通过跨模态注意力匹配和对比学习实现精确声源定位，在 Flickr-SoundNet 上达到 65.16% AUC、86.00% cIoU SOTA。

## 研究背景与动机

**领域现状**：声源定位（SSL）在图像中定位发声物体。现有方法用全局音频特征与全局/局部视觉特征做对比来定位，但忽略了场景中可能有多个物体但只有部分在发声的情况。

**现有痛点**：全局对比学习将所有视觉信息与音频对齐，导致非发声物体也获得高激活。需要一种方法将"发声物体"和"背景/非发声物体"在两个模态中分离。

**核心矛盾**：音频信号是全局的（麦克风采集所有声音），视觉场景是空间的（物体有明确位置）。如何在不知道哪个物体在发声的情况下做分离？

**切入角度**：用槽注意力（slot attention）在两个模态中同时进行竞争性分解——两个可学习的槽竞争性地注意输入特征，自然形成"目标"和"非目标"的分离。

**核心 idea**：双模态联合槽注意力分离目标/非目标 + 跨模态注意力匹配 = 精确声源定位。

## 方法详解

### 关键设计

1. **联合槽注意力分解**:

    - 功能：将图像和音频各自分解为目标槽和非目标槽
    - 核心思路：两个可学习查询（目标 $\mathbf{p}$、非目标 $\mathbf{r}$）通过交叉注意力与视觉/音频特征交互，竞争性地分配注意力。发散损失 $\mathcal{L}_{div}$ 确保两个槽不退化为相同表示
    - 设计动机：两个槽足够（实验验证更多槽无益），自然形成功能分化

2. **跨模态注意力匹配**:

    - 功能：用一种模态的自注意力图指导另一种模态的跨模态注意力
    - 核心思路：$\mathcal{L}_{match} = \|\text{ca}^{a,v} - \text{sg}(\text{ia}^{v,v})\|_2^2 + \|\text{ca}^{v,a} - \text{sg}(\text{ia}^{a,a})\|_2^2$，将视觉内部注意力模式迁移到音频→视觉注意力
    - 设计动机：自注意力已学会关注重要区域，跨模态注意力应该关注相同区域

3. **仅目标槽对比学习**:

    - 功能：只对齐两个模态的目标槽，忽略非目标
    - 核心思路：$\mathcal{L}_{cotr}$ 只在目标槽 $\mathbf{p}^v, \mathbf{p}^a$ 之间计算对比损失
    - 设计动机：非目标槽包含的噪声/背景信息不应参与对齐，否则会拉低定位精度

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{cotr} + \lambda_1\mathcal{L}_{match} + \lambda_2\mathcal{L}_{div} + \lambda_3\mathcal{L}_{recon}$。重建损失 $\mathcal{L}_{recon}$ 从两个槽重建原始特征，确保分解完整性。

## 实验关键数据

### 主实验

| 数据集 | cIoU | AUC |
|--------|------|-----|
| Flickr-SoundNet | **86.00%** | **65.16%** |
| VGG-Sound | **86.00%** | **64.90%** |
| 前 SOTA (FNAC) | 85.74% | 63.66% |

### 消融实验

- 跨模态注意力匹配对定位最关键
- 两槽设计最优（更多槽无增益）
- 假负例缓解（k-reciprocal NN）提升鲁棒性

### 关键发现
- **槽注意力的分解效果好**：目标槽自然聚焦发声物体，非目标聚焦背景
- **注意力匹配 > 简单特征对齐**：直接匹配注意力模式比特征级对比更有效

## 亮点与洞察
- **槽注意力在音视频任务中的新应用**——原本用于无监督物体发现，这里用于跨模态声源分离
- **分解后再对齐的范式**——先分离目标/非目标，再只对齐目标，比全局对齐更清洁

## 局限与展望
- 单声源假设（多声源场景失效）
- 需要配对音视频数据
- 噪声/复杂背景音下性能下降

## 评分
- 新颖性: ⭐⭐⭐⭐ 联合槽注意力+注意力匹配的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集多指标
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰
- 价值: ⭐⭐⭐⭐ 推进声源定位 SOTA

<!-- RELATED:START -->

## 相关论文

- [Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization](../../NeurIPS2025/audio_speech/seeing_sound_hearing_sight_uncovering_modality_bias_and_conflict_of_ai_models_in.md)
- [VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [Towards Open-Vocabulary Audio-Visual Event Localization](towards_open-vocabulary_audio-visual_event_localization.md)
- [In-the-wild Audio Spatialization with Flexible Text-guided Localization](../../ACL2025/audio_speech/tas_audio_spatialization.md)
- [MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)

<!-- RELATED:END -->
