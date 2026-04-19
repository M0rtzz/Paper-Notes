---
title: >-
  [论文解读] Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals
description: >-
  [ACL 2025][语音][turn-taking prediction] 提出首个融合语言、声学和视觉三模态信号预测对话中轮换（turn-taking）和反馈通道（backchannel）动作的端到端框架，并构建了包含 210+ 小时的 MM-F2F 面对面对话数据集，turn-taking F1 提升 10%，backchannel F1 提升 33%。
tags:
  - ACL 2025
  - 语音
  - turn-taking prediction
  - backchannel
  - 多模态
  - face-to-face conversation
  - low-rank fusion
---

# Predicting Turn-Taking and Backchannel in Human-Machine Conversations Using Linguistic, Acoustic, and Visual Signals

**会议**: ACL 2025  
**arXiv**: [2505.12654](https://arxiv.org/abs/2505.12654)  
**代码**: [GitHub](https://github.com/Linyx1125/MM-F2F)  
**领域**: 语音  
**关键词**: turn-taking prediction, backchannel, multi-modal fusion, face-to-face conversation, low-rank fusion

## 一句话总结

提出首个融合语言、声学和视觉三模态信号预测对话中轮换（turn-taking）和反馈通道（backchannel）动作的端到端框架，并构建了包含 210+ 小时的 MM-F2F 面对面对话数据集，turn-taking F1 提升 10%，backchannel F1 提升 33%。

## 研究背景与动机

**领域现状**: 全双工（full-duplex）自然口语对话系统需要准确预测用户何时结束发言（turn-taking）以及何时给出简短反馈（backchannel，如"嗯"、"我懂"）。现有对话系统多使用 VAD + 固定阈值或显式完成信号来判断响应时机。

**现有痛点**: 
   - 现有数据集主要覆盖文本和音频模态，缺乏包含视觉信号的面对面对话数据
   - EgoCom 数据集虽有视频但是第三人称视角且眼睛部分遮挡，不适合面对面场景
   - 大多数数据集忽略 backchannel 标注
   - 尚无统一的三模态（文本+语音+视频）turn-taking 和 backchannel 预测方案

**核心矛盾**: 真实人机对话中，说话人的语言、语调和面部表情都包含轮换信号，但如何有效融合三模态并支持任意模态组合输入是未解决的问题。

**本文目标**: 构建大规模三模态面对面对话数据集，设计支持任意模态组合的端到端预测框架。

**切入角度**: 自动化数据采集 pipeline + 隐私脱敏处理 + 基于低秩分解的灵活多模态融合模块 + 随机模态 Dropout 训练策略。

**核心 idea**: 用低秩张量融合结合模态选择机制，实现支持任意文本/音频/视频组合输入的 turn-taking 和 backchannel 预测。

## 方法详解

### 整体框架

两阶段训练：
1. **阶段一**: 分别训练三个单模态编码器（Linguistic/Acoustic/Visual Encoder），各输出 256 维特征 $\bm{z_k} \in \mathbb{R}^{256}, k \in \{T, A, V\}$
2. **阶段二**: 用多模态融合模块 $F$ 端到端联合训练所有模块

$$\hat{y} = F(E_T(X_T), E_A(X_A), E_V(X_V))$$

### 关键设计

1. **MM-F2F 数据集构建 Pipeline** (5 阶段):

    - **视频采集 + 隐私脱敏**: 从网络收集 ~1000 个对话视频，使用合成人脸替换原始面孔（从 10K+ 合成人脸池中选最相似的），扰动声纹特征（20% 标准差），裁剪保留面部区域
    - **视频转录**: WhisperX ASR 将音频分割为句子级/词级帧
    - **说话人验证**: ResNet 编码器提取 clip embedding + 聚类区分两个说话人
    - **活跃说话人检测**: TalkNet 模型判断帧中谁在说话
    - **标注**: 最后一个词标 TURN，匹配特定词表标 BACKCHANNEL，其余标 KEEP
    - 最终数据集：773 个视频，210+ 小时，~20M 帧，1.5M+ 词，51K turn-taking + 22K backchannel 实例

2. **灵活多模态融合模块**: 基于低秩多模态融合（LMF），用 $r$ 个低秩因子分解权重张量：

$$\bm{W} = \sum_{i=1}^{r} \bigotimes_k^K \bm{w_k^{(i)}}$$

$$\bm{h} = \Lambda_k^K \left[\sum_{i=1}^{r} \bm{w_k^{(i)}} \cdot \bm{z_k}\right]$$

3. **模态选择方案（Modality Selection Scheme）**: 引入指示函数 $I_k(x)$，缺失模态时替换为全 1 向量（对逐元素乘法为恒等元素），使融合自然退化为现有模态的融合：

$$\bm{h} = I_T(\cdot) \circ I_A(\cdot) \circ I_V(\cdot), \quad I_k(\bm{x}) = \begin{cases} \bm{x} & \text{if modality } k \text{ exists} \\ \bm{1} & \text{otherwise} \end{cases}$$

4. **Random Modality Dropout Training (RMDT)**: 训练时以小概率随机丢弃一个模态，用剩余两个模态融合，提高对模态缺失的鲁棒性。一次训练即可支持所有模态组合推理。

### 损失函数/训练策略

- 三分类交叉熵损失：$L = -\sum_i y^{(i)} \log(\hat{y}^{(i)}), \quad i \in \{\text{Keep}, \text{Turn}, \text{BC}\}$
- 单模态训练阶段 + 端到端多模态训练阶段
- Backbone 选择：GPT-2（文本）、HuBERT（音频）、VideoMAE（视频/人脸区域）
- 三层 MLP 预测头 [256, 64, 3]，Adam 优化器，学习率 $10^{-5}$，20 epochs

## 实验关键数据

### Backbone 选择实验

| 模态 | Backbone | Acc | F1-Keep | F1-Turn | F1-BC |
|------|----------|-----|---------|---------|-------|
| Text | BERT | 0.742 | 0.743 | 0.761 | 0.674 |
| Text | **GPT-2** | 0.751 | 0.747 | 0.767 | **0.707** |
| Audio | Wav2Vec2 | 0.730 | 0.715 | 0.726 | 0.779 |
| Audio | **HuBERT** | 0.751 | 0.737 | 0.735 | **0.805** |
| Video | ViT(单帧) | 0.473 | 0.535 | 0.470 | 0.271 |
| Video | VideoMAE(全帧) | 0.533 | 0.516 | 0.523 | 0.482 |
| Video | **VideoMAE(面部)** | 0.559 | 0.597 | 0.536 | **0.513** |

### 主实验 — 模态消融

| 模态 | Acc | F1-Keep | F1-Turn | F1-BC |
|------|-----|---------|---------|-------|
| Text | 0.751 | 0.747 | 0.767 | 0.707 |
| Audio | 0.751 | 0.737 | 0.735 | 0.805 |
| Video | 0.559 | 0.597 | 0.536 | 0.513 |
| Text+Audio | 0.811 | 0.783 | 0.809 | 0.894 |
| Text+Video | 0.757 | 0.751 | 0.766 | 0.743 |
| Audio+Video | 0.742 | 0.742 | 0.770 | 0.829 |
| **Text+Audio+Video** | **0.823** | **0.806** | **0.811** | **0.906** |

### 融合策略比较

| 融合方式 | Acc | F1-Keep | F1-Turn | F1-BC |
|----------|-----|---------|---------|-------|
| Concatenate | 0.771 | 0.764 | 0.774 | 0.784 |
| GMF | 0.807 | 0.791 | 0.795 | 0.889 |
| **Ours (LMF+Selection)** | **0.823** | **0.806** | **0.811** | **0.906** |

### SOTA 比较

| 方法 | 模态 | Acc | F1-Keep | F1-Turn | F1-BC |
|------|------|-----|---------|---------|-------|
| TurnGPT | T | 0.645 | 0.745 | 0.420 | - |
| Wang et al. | T+A | 0.737 | 0.742 | 0.739 | 0.680 |
| Kurata et al. | T+A+V | 0.720 | 0.729 | 0.728 | 0.667 |
| **Ours** | T+A+V | **0.823** | **0.806** | **0.811** | **0.906** |

### RMDT 有效性

| 推理模态 | w/o RMDT (Acc/F1-BC) | w/ RMDT (Acc/F1-BC) |
|----------|---------------------|---------------------|
| T+A | 0.552 / 0.017 | **0.816 / 0.896** |
| T+V | 0.423 / 0.005 | **0.760 / 0.747** |
| A+V | 0.433 / 0.041 | **0.765 / 0.845** |

### 关键发现

- 三模态融合在所有指标上优于任何单/双模态组合，backchannel 预测 F1 达到 0.906
- 音频模态对 backchannel 预测贡献最大（F1 0.805），可能与音调/节奏的不连续性有关
- 视觉信号对 backchannel 预测有显著补充作用（单独 0.513，但加入后 T+A 0.894→T+A+V 0.906）
- 仅看面部区域 > 看完整帧，背景信息反而引入噪声
- RMDT 至关重要：没有 RMDT，三模态训练的模型在双模态推理时几乎完全失效（F1-BC 从 0.906 崩到 0.017）
- 相比 SOTA 方法，turn-taking F1 提升约 10%，backchannel F1 提升超 33%

## 亮点与洞察

- **数据集构建 pipeline 实用且可复现**: 从网络视频自动收集+去隐私+自动标注，最小化人工介入
- **隐私保护设计周到**: 合成人脸替换+声纹扰动+面部裁剪，且实验验证不影响对话行为理解
- **RMDT 训练策略简单有效**: 一次训练支持所有模态组合，模态选择方案基于低秩分解的数学优雅性
- **Backchannel 预测是亮点**: 0.906 的 F1 说明多模态确实能捕获人类对话中细微的反馈信号

## 局限与展望

1. 当说话人语义未完整但在思考停顿时，模型可能误判为 turn-taking（Fig. 5 的失败案例）
2. 视觉模态目前仅使用面部信息，未利用手势和身体动作
3. 数据集基于英语对话，跨语言/跨文化泛化未验证
4. VideoMAE 处理 16 帧的计算开销较大，实时性有待验证
5. 未考虑多人对话场景（数据集限定为两人对话）

## 相关工作与启发

- VAP 模型（Ekstedt and Skantze, 2022b）用声学信号预测 keep/turn/backchannel，本文扩展到三模态
- LMF 低秩融合（Liu et al.）在多模态情感分析中的成功经验被迁移到对话预测
- 对全双工对话系统（如 GPT-4o level）的实时交互能力建设有直接启发

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个三模态 turn-taking + backchannel 预测框架和数据集
- **实验充分度**: ⭐⭐⭐⭐⭐ — Backbone 选择、模态消融、融合策略、RMDT、SOTA 比较全面
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，数据集描述详细
- **价值**: ⭐⭐⭐⭐⭐ — 数据集 + 代码开源，对全双工人机交互有重要推动

<!-- RELATED:START -->

## 相关论文

- [Resounding Acoustic Fields with Reciprocity](../../NeurIPS2025/audio_speech/resounding_acoustic_fields_with_reciprocity.md)
- [Talking Together: Synthesizing Co-Located 3D Conversations from Audio](../../CVPR2026/audio_speech/talking_together_synthesizing_co-located_3d_conversations_from_audio.md)
- [Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [Aligning Generative Music AI with Human Preferences: Methods and Challenges](../../AAAI2026/audio_speech/aligning_generative_music_ai_with_human_preferences_methods_and_challenges.md)
- [MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)

<!-- RELATED:END -->
