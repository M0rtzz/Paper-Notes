---
title: >-
  [论文解读] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation
description: >-
  [ACL 2025][语音][自监督语音表示] 提出 Eta-WavLM，通过简单的线性方程将 WavLM 自监督语音表示分解为说话人相关和说话人无关两个分量，无需复杂训练即可生成高质量的说话人解耦表示，在语音转换任务上全面超越现有方法。
tags:
  - ACL 2025
  - 语音
  - 自监督语音表示
  - 说话人解耦
  - 线性分解
  - 语音转换
  - WavLM
---

# Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation

**会议**: ACL 2025  
**arXiv**: [2505.19273](https://arxiv.org/abs/2505.19273)  
**代码**: https://giuseppe-ruggiero.github.io/eta-wavlm-vc-demo/ (有，含音频样例)  
**领域**: 语音  
**关键词**: 自监督语音表示, 说话人解耦, 线性分解, 语音转换, WavLM

## 一句话总结

提出 Eta-WavLM，通过简单的线性方程将 WavLM 自监督语音表示分解为说话人相关和说话人无关两个分量，无需复杂训练即可生成高质量的说话人解耦表示，在语音转换任务上全面超越现有方法。

## 研究背景与动机

**领域现状**：自监督学习（SSL）在语音领域取得革命性进展，WavLM、HuBERT、Wav2Vec 2.0 等模型能从原始波形中学习到丰富的通用表示。这些 SSL 表示编码了语言内容、说话人身份、情感、环境等多种信息，使其在各类下游任务中表现出色。

**现有痛点**：(1) SSL 表示中的说话人信息和语言内容高度纠缠，对需要内容纯净表示的任务（如语音转换 VC、TTS）造成干扰；(2) k-means 量化虽然简单但会同时损害语言内容和韵律信息；(3) 现有解耦方法要么无法充分去除说话人信息，要么需要复杂的训练策略、损失函数或微调过程。

**核心矛盾**：说话人信息去除与语言内容保留之间的 trade-off——去得太多损害内容，去得太少影响下游任务。

**本文目标**：在不需要复杂训练策略、损失函数、微调甚至量化的前提下，从 SSL 表示中高效去除说话人身份信息，同时保留语言内容。

**切入角度**：基于高维嵌入空间中复杂非线性关系趋于线性化的假设，将 SSL 表示建模为说话人分量与说话人无关分量的线性叠加，通过伪逆求解获得解耦表示。

**核心 idea**：SSL 表示可以线性分解为 $\mathbf{s} = f(\mathbf{d}) + \boldsymbol{\eta}$，其中 $\mathbf{d}$ 为说话人嵌入，$\boldsymbol{\eta}$ 为说话人无关的 eta 表示，通过最小二乘法即可求解。

## 方法详解

### 整体框架

Eta-WavLM 可视为 SSL 模型的离线扩展模块，包含三个组件：

1. **SSL 模型**（WavLM-Large，冻结）：从原始波形提取 SSL 表示 $\mathbf{S}$
2. **说话人编码器**（ECAPA-TDNN，冻结）：从同一波形生成说话人嵌入 $\mathbf{d}$
3. **解耦模块**：利用学到的潜在基（latent basis）$\mathbf{A}^*$ 和偏置 $\mathbf{b}^*$，从 $\mathbf{S}$ 中减去说话人相关分量得到 eta 表示 $\boldsymbol{\eta}$

### 关键设计

#### 1. 线性分解假设

- **功能**：将 SSL 表示假设为说话人分量和说话人无关分量的加法组合
- **核心思路**：$\mathbf{s} = f(\mathbf{d}) + \boldsymbol{\eta}$，其中 $f()$ 用线性模型近似：$\mathbf{S} = \mathbf{D}^T \mathbf{A} + \mathbf{1}_N \mathbf{b}^T$
- **设计动机**：基于 Ethayarajh et al. (2018) 的发现——大嵌入空间倾向于将复杂非线性关系线性化。线性假设使得求解极其高效（伪逆即可），无需梯度训练

#### 2. 潜在基与偏置的计算

- **功能**：从多说话人数据集中学习说话人到 SSL 表示的线性映射参数
- **核心思路**：
    - 从 LibriSpeech 全部训练集的每个语音片段提取 WavLM 表示 $\mathbf{S} \in \mathbb{R}^{N \times Q}$（$Q = 1024$，使用第 15 层输出）
    - 提取 ECAPA-TDNN 说话人嵌入，经 PCA 降维到 $P = 128$ 得到 $\mathbf{D} \in \mathbb{R}^{P \times N}$
    - 通过伪逆求解：$\tilde{\mathbf{A}}^* = (\tilde{\mathbf{D}}^T \tilde{\mathbf{D}})^{-1} \tilde{\mathbf{D}}^T \mathbf{S}$
- **设计动机**：整个"训练"过程仅需一次矩阵运算，无需迭代优化。PCA 降维（$V = 192 \to P = 128$）去除冗余信息，实验证明其对性能有正面影响

#### 3. Eta 表示的生成

- **功能**：推理时从音频中生成说话人无关的 eta 表示
- **核心思路**：给定音频 $\mathbf{u}'$，提取 SSL 表示 $\mathbf{S}$ 和说话人嵌入 $\mathbf{d}$，然后：
  $$\boldsymbol{\eta} = \mathbf{S} - \mathbf{1}_K (\mathbf{d}^T \mathbf{A}^* + \mathbf{b}^*)$$
- **设计动机**：运算极其简单——仅需矩阵乘法和减法，可视为将说话人相关的"偏移"从原始表示中减去

### 损失函数/训练策略

- 本方法**不涉及传统意义的训练**：仅需一次最小二乘求解（伪逆运算）
- 使用 LibriSpeech 全部训练集（约 1000 小时英语语音）计算参数
- 每个语音片段随机采样 $L = 100$ 帧构建固定长度表示
- 下游 VC 任务中的 acoustic model 使用标准训练流程

## 实验关键数据

### 主实验

**说话人分类实验**（10 说话人，5 折交叉验证，准确率越低越好）：

| 表示方法 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean ± Std |
|---------|-------|-------|-------|-------|-------|------------|
| WavLM | 83.46 | 82.33 | 80.85 | 83.30 | 81.55 | 82.30 ± 0.01 |
| **Eta-WavLM** | 53.82 | 55.14 | 58.77 | 53.94 | 56.96 | **55.73 ± 0.01** |

Eta-WavLM 说话人分类准确率从 82.30% 降至 55.73%（降低近 27%），证明有效去除了说话人信息。p 值 = 5.12×10⁻⁵，差异显著。

**语音转换实验**（两个目标说话人）：

| 方法 | LJSpeech WER↓ | LJ T-SSIM↑ | LJ MOS↑ | Elliot WER↓ | Elliot T-SSIM↑ | Elliot MOS↑ |
|------|---------------|------------|---------|-------------|----------------|-------------|
| Perturbation | 6.29 | 91.69 | 3.45 | 10.76 | 87.41 | 3.13 |
| 标准化 | 4.13 | 90.34 | 3.80 | 5.16 | 85.91 | 3.41 |
| Soft units | 4.82 | 91.81 | 3.84 | 5.50 | 86.69 | 3.32 |
| VQ | 4.79 | 90.05 | 3.90 | 7.72 | 86.30 | 3.50 |
| WavLM 原始 | 4.56 | 89.52 | 3.84 | 5.14 | 86.18 | 3.66 |
| **Eta-WavLM** | **3.81** | **92.46** | **4.00** | **4.64** | **89.32** | **3.79** |

Eta-WavLM 在所有指标上全面最优：LJSpeech 上 WER 3.81%（接近真实语音的 3.22%），T-SSIM 92.46%，MOS 4.00。

### 消融实验

**说话人编码器与 PCA 消融**：

| 配置 | WER↓ | T-SSIM↑ | SPK ACC↓ |
|------|------|---------|----------|
| Resemblyzer w/o PCA | 4.94 | 89.02 | 74.01 |
| WavLM-SV w PCA-128 | 3.91 | 89.76 | 65.83 |
| ECAPA-TDNN w/o PCA | 4.18 | 89.90 | 60.87 |
| ECAPA-TDNN w PCA-64 | 3.95 | 90.91 | 58.14 |
| **ECAPA-TDNN w PCA-128** | **3.81** | **92.46** | **55.73** |

ECAPA-TDNN + PCA-128 在所有指标上最优。PCA 降维对性能有正面影响，但过度降维（PCA-64）性能下降。

### 关键发现

1. **线性分解假设成立**：简单的线性方程就能有效分离说话人和内容信息，这是一个非常有力的实证发现
2. **说话人编码器选择关键**：ECAPA-TDNN 远优于 Resemblyzer 和 WavLM-SV，说明编码器质量直接影响解耦效果
3. **PCA 降维有助于解耦**：去除说话人嵌入中的冗余信息能提升分解质量
4. **VC 任务三角平衡**：Eta-WavLM 是唯一在可懂度、说话人相似度和语音质量三方面都达到最优的方法

## 亮点与洞察

1. **极致简洁**：整个"训练"仅需一次矩阵伪逆运算，无需 GPU 迭代训练，优雅至极
2. **理论直觉清晰**：$\boldsymbol{\eta} = \mathbf{s} - f(\mathbf{d})$ 的减法操作直观易理解——就是把说话人的"偏移"减掉
3. **通用性好**：方法作为 SSL 模型的后处理模块，不改变原模型参数，可即插即用
4. **UMAP/PaCMAP 可视化有力**：从 WavLM 的聚类到 Eta-WavLM 的均匀分布，视觉上令人信服

## 局限与展望

1. 说话人信息未完全去除——55.73% 的分类准确率仍高于随机水平（10%），残留说话人信息
2. 仅在英语数据集（LibriSpeech）上验证，多语言泛化能力未知
3. 线性假设可能在某些场景下不够精确，非线性分解方法可能进一步提升
4. SSL 模型仅测试了 WavLM，对 HuBERT、Wav2Vec 2.0 等其他 SSL 模型的适用性待验证

## 相关工作与启发

- **ContentVec (Qian et al., 2022)**：通过训练实现说话人解耦，复杂度高但思路类似
- **k-means 量化 (Hsu et al., 2021)**：最简单的解耦方式但同时丢失内容和韵律
- **van Niekerk et al. (2022)**：soft speech units 方法在 VC 中表现好但解耦不够彻底
- **RepCodec (Huang et al., 2024)**：向量量化方法，MOS 尚可但可懂度不如 Eta-WavLM
- **启发**：在高维空间中，线性方法可能比想象中更强大；解耦问题可能不需要复杂的对抗训练或特殊损失函数

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 用一个线性方程解决复杂的说话人解耦问题，极具创意
- **实验充分度**: ⭐⭐⭐⭐ — 说话人分类 + VC 双任务验证，消融细致，但缺少多语言实验
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，实验描述详尽
- **价值**: ⭐⭐⭐⭐⭐ — 方法简单高效、通用性强，对语音处理社区有广泛影响

<!-- RELATED:START -->

## 相关论文

- [Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)
- [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](spark-tts_an_efficient_llm-based_text-to-speech_model_with_single-stream_decoupl.md)
- [DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

<!-- RELATED:END -->
