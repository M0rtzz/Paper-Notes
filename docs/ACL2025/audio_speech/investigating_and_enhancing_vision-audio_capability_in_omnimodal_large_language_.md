---
title: >-
  [论文解读] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models
description: >-
  [ACL 2025][语音][全模态大语言模型] 发现当前全模态大语言模型（OLLMs）在视觉-音频任务上显著弱于视觉-文本任务，原因在于视觉与音频模态之间缺乏直接对齐，并提出 Self-KD（自知识蒸馏）方法，利用 OLLM 自身的视觉-文本组件作为教师来增强视觉-音频能力。
tags:
  - ACL 2025
  - 语音
  - 全模态大语言模型
  - 视觉-音频对齐
  - 自知识蒸馏
  - 模态间差距
  - 多模态融合
---

# Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models

**会议**: ACL 2025  
**arXiv**: [2503.00059](https://arxiv.org/abs/2503.00059)  
**代码**: https://github.com/isruihu/Self-KD (有)  
**领域**: 语音  
**关键词**: 全模态大语言模型, 视觉-音频对齐, 自知识蒸馏, 模态间差距, 多模态融合

## 一句话总结

发现当前全模态大语言模型（OLLMs）在视觉-音频任务上显著弱于视觉-文本任务，原因在于视觉与音频模态之间缺乏直接对齐，并提出 Self-KD（自知识蒸馏）方法，利用 OLLM 自身的视觉-文本组件作为教师来增强视觉-音频能力。

## 研究背景与动机

**领域现状**：全模态大语言模型（OLLMs）如 GPT-4o 集成了视觉、语言和音频能力，代表多模态 AI 的前沿方向。开源 OLLMs 如 VITA、VITA-1.5、Megrez 在视觉-文本任务上已有不错表现。

**现有痛点**：OLLMs 在处理视觉-音频输入时表现显著弱于视觉-文本输入。例如 Megrez-3B 在文本查询下平均分 68.96，音频查询下降至 49.72（降 19.24）；VITA-8x7B 更是从 70.04 暴跌到 7.84（降 62.20）。将同一个文本问题转为音频后，模型可能给出完全矛盾的答案。

**核心矛盾**：OLLMs 的训练流程中，视觉-文本和音频-文本分别进行了对齐，但视觉和音频之间从未直接对齐。模型只能在视觉-音频 SFT 阶段隐式学习整合这两种模态，但实验证明效果不足。

**本文目标**：(1) 系统量化并分析 OLLMs 视觉-音频能力差距的原因；(2) 提出有效方法缩小这一差距。

**切入角度**：通过注意力权重分析发现音频查询时模型对视觉 token 的关注度低于文本查询；构建 MMAlign 基准验证视觉-音频对齐弱于视觉-文本对齐；然后提出用模型自身的视觉-文本组件蒸馏视觉-音频组件。

**核心 idea**：OLLM 的视觉-文本能力远强于视觉-音频能力，可以用前者作为教师通过自知识蒸馏来增强后者。

## 方法详解

### 整体框架

Self-KD 是一个知识蒸馏训练框架，特点是教师和学生来自同一个模型的不同组件：

1. **教师模型** $p_T$：OLLM 的视觉-文本组件（视觉编码器 + 文本嵌入层 + LLM）
2. **学生模型** $p_S$：OLLM 的视觉-音频组件（视觉编码器 + 音频编码器 + LLM）
3. **输入**：教师使用视觉-文本样本 $x^t$，学生使用对应的视觉-音频样本 $x^a$（文本问题由 TTS 转为音频）
4. **训练目标**：让学生模仿教师处理视觉信息的行为

### 关键设计

#### 1. 问题发现与分析

- **功能**：系统量化并分析 OLLMs 视觉-音频能力差距
- **核心发现**：
    - 所有测试的 OLLMs（VITA/VITA-1.5/Megrez）在音频查询下性能大幅下降
    - 模型在音频查询下表现出更高的 "Yes" 偏差
    - 音频回答通常与图像相关但不准确——模型"看到"了图像但未能正确整合信息
- **注意力分析**：音频查询时，query token 对 vision token 的注意力权重在中后层显著低于文本查询，但 response token 对 vision/query token 的注意力差异不大

#### 2. MMAlign 基准

- **功能**：构建专门评估视觉-文本和视觉-音频对齐质量的基准
- **核心思路**：基于 ARO 数据集，每个样本包含一张图片和两个描述（一个正确、一个干扰），模型需选择正确描述。分别用文本和音频形式提问
- **结果**：所有模型文本查询准确率远高于音频查询（如 VITA-1.5：75.67 vs. 32.83），直接证明视觉-音频对齐弱于视觉-文本对齐

#### 3. Self-KD 训练框架

- **功能**：利用 OLLM 自身的视觉-文本组件指导视觉-音频组件的训练
- **核心思路**：
    - 常规 SFT 损失：$L_{\text{SFT}} = \mathbb{E}[-\log p_S(y|x^a)]$
    - Self-KD 损失（KL 散度）：$L_{\text{Self-KD}} = \text{KL}(p_T \| p_S) = \mathbb{E}\left[\log \frac{p_T(y|x^t)}{p_S(y|x^a)}\right]$
    - 总损失：$L = \alpha L_{\text{Self-KD}} + (1-\alpha) L_{\text{SFT}}$
- **设计动机**：不同于传统 KD 教师和学生用相同输入，Self-KD 中教师用视觉-文本输入、学生用对应的视觉-音频输入，使学生学会像处理文本一样处理音频，包括分配更多注意力给视觉 token

### 损失函数/训练策略

- **总损失**：$L = \alpha L_{\text{Self-KD}} + (1-\alpha) L_{\text{SFT}}$，超参数 $\alpha$ 控制 SFT 和 KD 的比例
- **音频-文本对齐**阶段：使用 LibriSpeech、Common Voice、GigaSpeech、Libriheavy 等 ASR 数据集，共 988k 样本
- **视觉-音频 SFT 和 Self-KD**阶段：从 llava-1.5-mix-665k 中采样 50k 指令跟随样本，将文本问题 TTS 转为音频
- 音频编码器使用 Whisper-large-v3，通过单层 MLP 投射到 LLM 空间
- 基座模型：InternVL2 系列（1B/2B/4B/8B）和 Qwen2VL 系列（2B/7B）

## 实验关键数据

### 主实验

**不同模型 Self-KD vs. 常规 SFT 的平均性能**（8 个视觉基准的平均分）：

| 模型 | VL (文本) | SFT (音频) | Self-KD (音频) | KD 提升 |
|------|----------|-----------|---------------|---------|
| InternVL2-1B | 49.68 | 21.16 | **33.84** | +12.68 |
| InternVL2-2B | 53.94 | 22.52 | **36.58** | +14.06 |
| InternVL2-4B | 59.38 | 32.22 | **42.30** | +10.08 |
| InternVL2-8B | 69.91 | 38.71 | **51.45** | +12.74 |
| Qwen2VL-2B | 64.77 | 46.21 | **52.58** | +6.37 |
| Qwen2VL-7B | 75.14 | 67.75 | **68.27** | +0.52 |

Self-KD 在几乎所有配置上显著优于常规 SFT，InternVL2 系列提升 10-14 分，Qwen2VL 系列提升相对较小（可能因为其视觉-文本对齐本身更好）。

### 消融实验

**MMAlign 基准上 SFT vs. Self-KD**（InternVL2-1B 的 Relation/Attribute 准确率）：

| 模型 | Relation (SFT) | Relation (Self-KD) | Attribute (SFT) | Attribute (Self-KD) |
|------|---------------|-------------------|-----------------|---------------------|
| InternVL2-1B | 42.67 | **50.67** | — | — |

Self-KD 在模态对齐质量上也优于 SFT，进一步验证其有效性。

**注意力权重变化**：Self-KD 后，模型在音频查询时对视觉 token 的注意力权重上升，行为更接近文本查询模式。

### 关键发现

1. **VL 与 VA 能力差距是普遍现象**：所有测试的 OLLMs 都存在，不是个例
2. **VL 能力越强，VA SFT 后效果越好**：VL 性能与 VA 性能正相关（如 InternVL2-8B VL 最强 69.91，SFT 后 VA 也最好 38.71）
3. **Self-KD 的有效性与模型的 VL 能力成正比**：教师越强，蒸馏效果越好
4. **Qwen 系列提升较小**：可能因为 Qwen 系列本身视觉-文本对齐质量较高，常规 SFT 已足够好
5. **问题根源在注意力分配**：音频查询时 query 给 vision 的注意力不足，Self-KD 能改善这一行为

## 亮点与洞察

1. **问题定义新颖且重要**：首次系统性地量化分析 OLLMs 的视觉-音频能力差距，填补了评估空白
2. **分析深入**：通过注意力权重分析、Yes 偏差分析、MMAlign 基准三个维度全面揭示问题本质
3. **解决方案优雅**：Self-KD 不引入外部模型，利用 OLLM 自身的强 VL 能力来增强弱 VA 能力
4. **跨模型通用性**：在 InternVL2（4 种规模）和 Qwen2VL（2 种规模）上均有效，验证了方法的通用性

## 局限与展望

1. TTS 合成音频可能与真实人声存在差异，评估可能不完全反映真实场景
2. 训练数据量较小（仅 50k 样本），扩大数据规模可能进一步提升效果
3. 仅测试了英语场景，多语言视觉-音频能力有待验证
4. Self-KD 需要同时前向传播教师和学生组件，增加了训练成本
5. 未探索直接的视觉-音频对齐方法（如对比学习），可能有更根本的解决途径

## 相关工作与启发

- **GPT-4o (Hurst et al., 2024)**：商业 OLLM 的代表，视觉-音频能力可能也存在类似问题
- **VITA/VITA-1.5 (Fu et al., 2024/2025)**：开源 OLLM 基线，展现了最严重的 VA 性能下降
- **Megrez (Li et al., 2025)**：3B 小模型但 VA 差距最小（19.24），说明模型架构设计也很关键
- **Bi et al. (2024)**：注意力分布反映模态对齐的先行研究，为本文分析提供理论依据
- **启发**：多模态模型中"看不到"的模态对齐问题可能比想象中严重，自蒸馏是一种低成本的跨模态能力迁移手段

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 问题发现新颖，Self-KD 虽简单但切入角度独到
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6 个模型/4 种规模/8 个基准/3 维分析，非常全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题→分析→解决方案的逻辑链非常清晰
- **价值**: ⭐⭐⭐⭐ — 揭示了 OLLMs 的重要短板，Self-KD 方法通用性好

<!-- RELATED:START -->

## 相关论文

- [Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [Towards Reliable Large Audio Language Model](towards_reliable_large_audio_language_model.md)
- [AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)

<!-- RELATED:END -->
