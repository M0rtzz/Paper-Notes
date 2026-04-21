---
title: >-
  [论文解读] Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments
description: >-
  [NeurIPS 2025][语音][目标说话人提取] 提出一种利用噪声正样本（目标说话人在说话的段落）和负样本（目标说话人沉默的段落）对比来编码目标说话人特征的新型注册策略，在单声道噪声注册目标说话人提取任务上取得 SOTA 性能，SI-SNRi 比此前最优方法高出 2.1 dB 以上。
tags:
  - NeurIPS 2025
  - 语音
  - 目标说话人提取
  - 噪声注册
  - 正负样本对比
  - TF-GridNet
  - 两阶段训练
---

# Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments

**会议**: NeurIPS 2025  
**arXiv**: [2502.16611](https://arxiv.org/abs/2502.16611)  
**代码**: [有](https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll)  
**领域**: Audio & Speech  
**关键词**: 目标说话人提取, 噪声注册, 正负样本对比, TF-GridNet, 两阶段训练

## 一句话总结

提出一种利用噪声正样本（目标说话人在说话的段落）和负样本（目标说话人沉默的段落）对比来编码目标说话人特征的新型注册策略，在单声道噪声注册目标说话人提取任务上取得 SOTA 性能，SI-SNRi 比此前最优方法高出 2.1 dB 以上。

## 研究背景与动机

目标说话人提取（TSE）旨在从多说话人混合音频中分离出特定说话人的声音。现有方法主要依赖**干净音频样本**作为条件输入，但这在实际场景中往往不可行——例如在鸡尾酒会上，用户无法让陌生人离开嘈杂环境单独录制干净音频。

少数尝试使用噪声注册的工作存在局限：
- **ADEnet**：仅在 38.5% 重叠率的两说话人混合中验证，注册段落基本是干净的
- **TCE**：要求用户本人参与对话，不支持从任意混合音频中提取
- **LookOnceToHear**：依赖双耳空间信息（目标说话人在 90° 方位角），限制了适用场景

**核心观察**：在自然对话中，不同说话人不会完美同步地开始和停止说话。因此，可以利用时间上的不一致性——通过比较"目标说话人在说话的段落"和"目标说话人沉默的段落"来消歧。

## 方法详解

### 整体框架

系统包含两个分支：(1) **编码分支**：孪生编码器提取正/负注册音频的嵌入，通过编码器融合模块比较后得到目标说话人特征；(2) **提取分支**：以编码分支输出为条件，从混合音频中提取目标说话人的声音。采用 TF-GridNet 作为两个分支的骨干网络。

### 关键设计

1. **正负注册策略 (Positive/Negative Enrollment)**：将注册输入构建为一对正注册 $a^P$（目标说话人在说话）和负注册 $a^N$（目标说话人沉默）。信号模型为：

    - $a^M = \sum_{i \in S^{IM} \cup \{t\}} a_i^M + n^M$（混合音频）
    - $a^P = \sum_{i \in S^{IE} \cup \{t\}} a_i^P + n^P$（正注册）
    - $a^N = \sum_{i \in S^{IE}} a_i^N + n^N$（负注册）
   
   目标说话人是**唯一在所有正注册中持续说话但在负注册中不出现**的人。干扰说话人被分为四类：负干扰者（NI, 正负中都有）、正干扰者（PI, 仅在正注册部分时段出现）、混合干扰者（HI）和需忽略干扰者（NRI）。

2. **编码器融合模块 (Encoder Fusion Module)**：孪生编码器共享参数，分别编码正/负注册得到 $E_{pos} \in \mathbb{R}^{T_{pos} \times D}$ 和 $E_{neg} \in \mathbb{R}^{T_{neg} \times D}$。融合步骤：

    - 添加可学习段嵌入 $S_{pos}, S_{neg}$ 以区分来源
    - 沿时间维度拼接：$E_{concat} = [E_{pos}, E_{neg}]$
    - 通过两层 Full-band Self-attention 处理
    - 截取前 $T_{pos}$ 帧作为输出
   
   设计动机：自注意力机制允许正注册帧间比较（发现正干扰者的沉默段）和正负注册帧间比较（识别负干扰者），自然实现消歧。

3. **提取融合模块 (Extraction Fusion Module)**：编码器输出经非重叠平均池化（kernel=40）降维后，作为交叉注意力的 Key/Value，提取分支中 TF-GridNet 的输出作为 Query，在前两个因果 TF-GridNet block 之后各插入一个融合模块。

### 损失函数 / 训练策略

**两阶段训练策略**（核心创新之一）：

- **第一阶段**：仅训练孪生编码器和融合模块，使用知识蒸馏使其输出逼近干净编码器的表示：
  $$L_{\text{stage 1}} = \|E_{\text{clean}} - E_{\text{fused}}\|^2$$

- **第二阶段**：训练提取分支，以负 SNR 为损失：
  $$L_{\text{stage 2}} = -\text{SNR}(\hat{a}_{tgt}, a_{tgt})$$

动机：噪声注册引入的高变异性使端到端训练收敛极慢（600k steps 才达 3 dB SNR），两阶段训练仅需 240k steps，减少 60% 优化步数。

## 实验关键数据

### 主实验

**单声道 TSE：2 说话人混合音频，2 说话人注册**

| 方法 | SNRi (dB) | SI-SNRi (dB) | PESQ | STOI | WER |
|------|----------|-------------|------|------|-----|
| NMF | 4.24±1.60 | -1.65±3.78 | 1.05 | 0.362 | 0.98 |
| USEF-TFGridnet | 3.42±3.43 | -0.03±5.97 | 1.52 | 0.430 | 0.66 |
| TCE | 8.48±2.34 | 6.67±3.69 | 1.91 | 0.682 | 0.73 |
| **Ours (Monaural)** | **10.14±2.57** | **8.85±3.67** | **2.07** | **0.758** | **0.42** |

### 消融实验

| 配置 | 训练步数达 3dB SNR | 最终性能 |
|------|-------------------|---------|
| 端到端训练 | ~600k steps (~125h) | 较差 |
| **两阶段训练** | **~240k steps (~50h)** | **更好** |
| Film 融合（替代交叉注意力） | - | 所有场景均更差 |
| **交叉注意力融合** | - | **全面更优** |

### 关键发现

- 相比 TCE，单声道 SI-SNRi 提升 2.1+ dB（6.67→8.85），WER 从 0.73 降至 0.42
- 模型在注册中 2~4 个说话人、混合中 2~3 个说话人等多种场景下保持鲁棒
- 两阶段训练将达到 3 dB SNR 的时间从 125 小时缩短至 50 小时（60% 减少）
- 交叉注意力融合全面优于 Film 融合，因为后者被固定嵌入维度限制了细粒度编码能力
- 双耳版本在 SNRi/SI-SNRi 上优于 LookOnceToHear 基线，但 STOI 略低（可能因参数量差异）

## 亮点与洞察

- **极其实用的注册策略**：用户只需在手机上点击按钮标记"目标说话人在/不在说话"，无需精确标注
- **基于自然对话随机性的消歧假设**优雅且合理——不同说话人几乎不可能完美同步
- 两阶段训练中使用**知识蒸馏**将干净编码器的知识迁移到噪声编码器，有效解耦了两个学习难点
- 孪生编码器共享参数的设计减少了模型规模同时保持了一致的特征空间

## 局限与展望

- 假设用户能粗略区分目标说话人何时在说话/沉默，这在极嘈杂或多人快速交替场景中可能困难
- 正干扰者的重叠率较高时性能下降（论文有讨论但程度有限）
- 仅在 LibriSpeech + WHAM! 合成数据上训练和评估，真实场景的性能有待验证
- 模型基于 TF-GridNet，计算开销较大，轻量化版本的探索有价值

## 相关工作与启发

- **与 LookOnceToHear 的关系**：同样处理噪声注册，但不依赖空间信息，适用范围更广（单声道）
- **与 TCE 的关系**：TCE 要求用户参与对话且使用用户自己的干净 d-vector，限制更多
- **启发**：正负样本对比思想可能在视频说话人分离、声源定位等任务中同样适用

## 评分

- 新颖性: ⭐⭐⭐⭐ （正负注册策略新颖实用，但整体架构基于已有TF-GridNet）
- 实验充分度: ⭐⭐⭐⭐ （多场景对比全面，消融清晰，但缺乏真实数据评估）
- 写作质量: ⭐⭐⭐⭐ （问题定义清楚，方法动机阐述好）
- 价值: ⭐⭐⭐⭐ （解决了实际场景中的关键痛点）

<!-- RELATED:START -->

## 相关论文

- [AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [DNCASR: End-to-End Training for Speaker-Attributed ASR](../../ACL2025/audio_speech/dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [Different Speech Translation Models Encode and Translate Speaker Gender Differently](../../ACL2025/audio_speech/different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)

<!-- RELATED:END -->
