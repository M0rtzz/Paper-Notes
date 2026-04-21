---
title: >-
  [论文解读] Cinematic Audio Source Separation Using Visual Cues
description: >-
  [CVPR 2026][图像生成][影视音频源分离] 提出首个音视频影视音频源分离（AV-CASS）框架，利用面部和场景双视频流的视觉线索，通过条件流匹配进行生成式三路音频分离（语音/音效/音乐），仅在合成数据上训练即可泛化到真实电影。
tags:
  - CVPR 2026
  - 图像生成
  - 影视音频源分离
  - 音视频学习
  - 条件流匹配
  - 合成训练数据
  - 多源分离
---

# Cinematic Audio Source Separation Using Visual Cues

**会议**: CVPR 2026  
**arXiv**: [2603.26113](https://arxiv.org/abs/2603.26113)  
**代码**: [项目页](https://cass-flowmatching.github.io) (有)  
**领域**: Image Generation (音视频多模态)  
**关键词**: 影视音频源分离, 音视频学习, 条件流匹配, 合成训练数据, 多源分离

## 一句话总结

提出首个音视频影视音频源分离（AV-CASS）框架，利用面部和场景双视频流的视觉线索，通过条件流匹配进行生成式三路音频分离（语音/音效/音乐），仅在合成数据上训练即可泛化到真实电影。

## 研究背景与动机

**领域现状**：影视音频源分离（CASS）随 DnR 数据集的引入被形式化为语音/音效/音乐三路分离问题。BandIt 等方法推进了音频端性能，但所有现有方法都忽略了电影的多模态本质。

**现有痛点**：(a) CASS 方法均为纯音频，忽略了视觉线索（唇动对应语音、场景动作对应音效）；(b) 缺少同时具有源分离音轨和时间对齐视频的数据集；(c) 预测式分离模型容易产生频谱空洞伪影。

**核心矛盾**：视觉信息显然有助于音频分离，但获取真实电影的独立音轨几乎不可能。

**本文目标**：在无法获取真实隔离音轨的情况下，利用可独立获取的野外音视频数据训练有效的 AV-CASS 模型。

**切入角度**：合成训练数据管线（面部视频→语音、场景视频→音效、纯音乐）+ 生成式流匹配分离模型。

**核心 idea**：训练用双视频流（面部+场景），推理时从真实电影单视频中提取双流，零样本泛化。

## 方法详解

### 整体框架

Vision Extractor（面部编码器+场景编码器→融合视觉条件 $\mathbf{c}^V$）+ Flow Matching 生成模型（从噪声到三路频谱图，以混合音频 $\mathbf{s}^A$ 和 $\mathbf{c}^V$ 为条件）。

### 关键设计

1. **合成训练数据管线**：

    - 语音(DX)：LRS3 数据集（唇同步视频+语音），152K 片段
    - 音效(FX)：VGGSound（日常事件视频+音频），经 SMAD 过滤去除含语音/音乐片段，~62K
    - 音乐(MX)：FMA（纯音乐），经过滤 ~49K
    - 混合：$\mathbf{a}^A = \mathbf{a}^{DX} + \mathbf{a}^{FX} + \mathbf{a}^{MX}$
   设计动机：真实电影源分离数据不可得，但各单源的音视频数据丰富。合成混合保留完整 GT 且可控。

2. **双流视觉编码器与融合**：面部编码器（AVDiffuSS）提取唇同步特征，场景编码器（CAVP）提取时间-语义对齐特征。两者冻结，投影后沿时间轴拼接：$\mathbf{c}^V \in \mathbb{R}^{(T_f+T_s) \times C'}$，通过 U-Net cross-attention 注入。
   设计动机：面部流≈语音线索，场景流≈音效线索，互补覆盖 CASS 中有视觉关联的两路音源。

3. **条件流匹配多源分离**：
    $\mathcal{L} = \mathbb{E}_{t, \pi_1, \pi_0} \|\mathbf{u}_\theta(\mathbf{x}_t, t | \mathbf{c}) - (\mathbf{x}_1 - \mathbf{x}_0)\|_2^2$
   logit-normal 时间步采样，三路频谱图沿通道维拼接。
   设计动机：流匹配比扩散推理更快，比 masking 方法产生更自然音频。

### 训练策略

- 音频去噪预热 → 零初始化卷积渐进引入视觉条件（ControlNet 式）
- Adam，LR 1e-4，600K 步，4×RTX 4090，推理 128 步

## 实验关键数据

### 主实验

| 方法 | 真实电影MOS↑ | AVDnR FAD↓ | AVDnR PESQ↑ | AVDnR WPR↓ |
|------|------------|-----------|-------------|-----------|
| MRX | 2.55 | 3.47 | 1.89 | 14.91 |
| BandIt | 3.78 | 2.15 | 2.15 | 4.65 |
| DAVIS-Flow (AV) | - | 5.94 | 1.96 | 12.14 |
| **AV-CASS** | **4.13** | **0.84** | **2.26** | **1.84** |

### 消融实验

| 配置 | FAD↓ | WPR↓ | 说明 |
|------|------|------|------|
| Audio-only | 1.63 | 2.01 | 纯音频基线 |
| AV-CASS | **0.84** | **1.84** | 视觉条件提升 48% FAD |
| DAVIS-Flow | 5.94 | 12.14 | 通用AV分离不适用于CASS |

### 关键发现

1. 视觉线索使 FAD 从 1.63 降至 0.84（提升 48%），WPR 从 2.01 降至 1.84。
2. 定性分析：鸟鸣在纯音频模型中被误分为音乐，AV-CASS 通过场景中的鸟正确分配到音效。
3. 合成训练→真实电影泛化成功（MOS 4.13/5）。
4. CASS ≠ 通用 AV 分离：DAVIS-Flow 在 FX 上 WPR 低但 DX/MX 极差。

## 亮点与洞察

- 训推范式转换优雅：训练双视频流（独立源），推理从单电影提取面部+场景双流，无需架构修改。
- 流匹配在音频分离中展现出色感知质量。
- WPR 指标创新——衡量跨轨泄漏，无需GT参考。
- 音频预热 + 视觉渐进注入防止过早依赖视觉。

## 局限与展望

- 音乐无视觉关联，视觉对 MX 分离增益有限。
- 128 步推理较慢，可探索蒸馏加速。
- 仅 16kHz mono，影视级 48kHz 多声道待验证。

## 相关工作与启发

- 合成训练→真实泛化的范式对其他缺乏配对数据的多模态任务有启发。
- 条件流匹配在音频生成中的应用正快速扩展。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个AV-CASS + 合成管线 + 双流设计
- 实验充分度: ⭐⭐⭐⭐⭐ 真实电影MOS + 合成测试全指标 + 公开基准
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，动机充分
- 价值: ⭐⭐⭐⭐⭐ 开辟AV-CASS新方向，影视后期直接应用

<!-- RELATED:START -->

## 相关论文

- [Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [MACS: Multi-source Audio-to-Image Generation with Contextual Significance and Semantic Alignment](../../AAAI2026/image_generation/macs_multi-source_audio-to-image_generation_with_contextual_significance_and_sem.md)
- [Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)
- [A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](../../NeurIPS2025/image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [Learning Trimodal Relation for Audio-Visual Question Answering with Missing Modality](../../ECCV2024/image_generation/learning_trimodal_relation_for_audio-visual_question_answering_with_missing_moda.md)

<!-- RELATED:END -->
