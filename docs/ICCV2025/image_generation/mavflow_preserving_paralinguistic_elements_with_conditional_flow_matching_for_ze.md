---
title: >-
  [论文解读] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation
description: >-
  [图像生成] 提出 MAVFlow，基于条件流匹配（CFM）的零样本音视觉渲染器，通过音频说话人嵌入和视觉情感嵌入的双模态引导，在多语言 AV2AV 翻译中保持说话人一致性。
tags:
  - 图像生成
---

# MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation

> **会议**: ICCV 2025
> **arXiv**: [2503.11026](https://arxiv.org/abs/2503.11026)
> **代码**: [GitHub](https://github.com/Peter-SungwooCho/MAVFlow)
> **领域**: 多模态生成·音频视觉翻译·流匹配
> **关键词**: audio-visual translation, conditional flow matching, zero-shot, paralinguistic preservation, speaker consistency

## 一句话总结

提出 MAVFlow，基于条件流匹配（CFM）的零样本音视觉渲染器，通过音频说话人嵌入和视觉情感嵌入的双模态引导，在多语言 AV2AV 翻译中保持说话人一致性。

## 研究背景与动机

音频-视觉到音频-视觉（AV2AV）翻译旨在将一种语言的音视频内容翻译为另一种语言，同时保持唇形同步和说话人特征一致。现有方法面临的核心挑战：

**副语言特征丢失**：现有 AV2AV 方法（如 AV2AV、TransFace）主要关注语言内容的翻译，忽略了说话人身份、情感表达等副语言（paralinguistic）特征，导致翻译后的语音、面部与原始说话人特征不一致。

**单模态嵌入的局限**：现有方法在音频和视觉生成中各自使用单模态嵌入（如 d-vectors），未充分利用跨模态互补信息。

**缺乏高级条件生成技术**：简单的说话人嵌入拼接难以在零样本跨语言场景下保持说话人一致性。

**MAVFlow 的核心思路**：说话人的声音特征和面部信息（外观、情感）跨语言保持一致，因此可以利用**音频的说话人嵌入**（全局身份）和**视觉的情感嵌入**（逐帧动态）作为独立于语义内容的引导信号，结合 OT-CFM 的高效采样优势进行条件生成。

## 方法详解

### 整体框架

MAVFlow 包含四个阶段（Fig. 2）：

1. **AV 语音单元翻译**：使用 m-AVHuBERT 提取离散 AV 单元，经 U2U 模块翻译为目标语言单元
2. **时长调节器**：预测并扩展去重单元的时长，插值对齐原始音频长度
3. **多模态引导**：提取说话人声音嵌入（x-vector）和面部情感嵌入（EmoFAN）
4. **CFM 零样本 AV 渲染器**：整合语言单元和副语言引导，生成梅尔频谱图

### 关键设计 1：多模态引导

**说话人声音嵌入（全局）**：使用预训练说话人编码器提取 x-vectors，训练阶段对同一说话人的多个语段取平均得到说话人级嵌入：

$$\mathbf{a}_{spk} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{a}_{utt,i}$$

推理时直接使用语段级嵌入 $\mathbf{a}_{utt,i}$，捕获细粒度变化。

**面部情感嵌入（逐帧）**：使用 EmoFAN 从每帧提取情感嵌入 $\mathbf{v}_{spk,t} = \text{Emo}(\mathbf{f}_t)$，反映随时间动态变化的情感状态。关键洞察：语音的声学特征随语言变化（口音、韵律），但情感信息跨语言一致。

### 关键设计 2：OT-CFM 解码器

采用最优传输条件流匹配（OT-CFM）框架，学习从噪声到梅尔频谱图的条件向量场：

$$\nu_t(\phi_t^{OT}(X_0, X_1) | \theta) = N_\theta(\phi_t^{OT}(X_0, X_1), t; \mathbf{a}_{spk}, \mathbf{v}_{spk,t}, \{\mu_l\}_{1:L}, \tilde{X}_1)$$

其中 $\phi_t^{OT}(X_0, X_1) = (1-(1-\sigma)t)X_0 + tX_1$ 为 OT 路径。

训练目标：

$$\mathcal{L}_{OT-CFM} = \mathbb{E}_{t, p_0, q}\left[\|\omega_t - \nu_t\|^2\right]$$

多模态嵌入的整合方式：$\mathbf{a}_{spk}$ 统一加到所有帧（全局身份），$\mathbf{v}_{spk,t}$ 逐帧加入（动态情感）。

### 时长调节器

与 AV2AV 不同，MAVFlow 将生成音频插值对齐到原始源音频的长度，满足电影配音等实际场景中视频长度不变的约束。

## 实验

### 主实验：零样本说话人相似度（Tab. 1, MuAViC 数据集）

| 方法 | Es-En SS ↑ | Fr-En SS ↑ | It-En SS ↑ | Pt-En SS ↑ |
|------|-----------|-----------|-----------|-----------|
| 4-Stage 级联 | 0.42 | 0.34 | 0.41 | 0.36 |
| 3-Stage 级联 | 0.42 | 0.35 | 0.41 | 0.35 |
| AV2AV (Direct) | 0.35 | 0.31 | 0.37 | 0.30 |
| **MAVFlow** | **0.49** | **0.51** | **0.53** | **0.48** |

MAVFlow 在所有语言对上说话人相似度平均提升 **36%**，DTW 和 DTW-SL 指标同样最优。

### 情感评估（Tab. 2, CREMA-D 数据集）

| 方法 | Emo-Acc (%) ↑ | SS ↑ | DTW ↓ |
|------|-------------|------|-------|
| ASR + YourTTS | 17.52 | 0.40 | 9.02 |
| ASR + XTTS | 28.55 | 0.46 | 11.98 |
| AV2AV | 33.66 | 0.33 | 7.84 |
| **MAVFlow** | **36.46** | 0.39 | **7.30** |

情感准确率提升 +2.8%（vs AV2AV），+18.94%（vs XTTS）。

### 翻译质量（Tab. 3, ASR-BLEU）

| 方法 | Es-En | Fr-En | It-En | Pt-En |
|------|-------|-------|-------|-------|
| AV2AV | 26.57 | 31.27 | 23.24 | 24.51 |
| **MAVFlow** | **26.97** | **31.33** | **23.43** | **24.97** |

关键发现：双模态引导在保持说话人一致性的同时**不损害翻译质量**，甚至略有提升。

### 人工评估（Tab. 4, MOS 评分）

| 方法 | Similarity ↑ | Naturalness ↑ |
|------|------------|---------------|
| 4-Stage 级联 | 2.81 | 3.29 |
| AV2AV | 3.33 | 3.58 |
| **MAVFlow** | **3.71** | **3.80** |

## 亮点与洞察

1. **双模态引导策略**：音频 x-vector（全局身份）+ 视觉情感嵌入（逐帧动态）互补，巧妙利用了副语言信息跨模态共享的特性
2. **OT-CFM 的优势**：相比扩散模型，CFM 采样步数更少更高效，天然适合整合多模态条件
3. **时长对齐**是实际配音场景的重要需求，此前方法忽略了这一约束
4. 说话人一致性提升 36% 是显著的实际改进

## 局限性

- 依赖 Wav2Lip 进行面部生成，面部质量受限于预训练的 TFG 模型
- 情感嵌入基于面部情绪识别模型 EmoFAN，受其分类精度影响
- 未评估端到端系统延迟和实时性能

## 相关工作

- AV2AV 翻译：AV2AV、TransFace
- 流匹配生成：Voicebox、P-Flow、Matcha-TTS
- 说话人一致性：d-vector、x-vector、ERes2Net

## 评分

- **新颖性**: ★★★★☆ — CFM + 双模态引导在 AV2AV 中首次应用
- **技术深度**: ★★★☆☆ — 各模块较为直接，创新在整合方式
- **实验质量**: ★★★★☆ — 多维度评估全面，包含人工评估
- **写作质量**: ★★★★☆ — 动机阐述清晰，实验比较充分

<!-- RELATED:START -->

## 相关论文

- [DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [FlexiClip: Locality-Preserving Free-Form Character Animation](../../ICML2025/image_generation/flexiclip_locality-preserving_free-form_character_animation.md)
- [SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](../../CVPR2025/image_generation/dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)

<!-- RELATED:END -->
