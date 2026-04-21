---
title: >-
  [论文解读] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models
description: >-
  [ACL2025][语音][对抗音频攻击] 提出 Chat-Audio Attacks (CAA) 基准，包含四类通用对抗音频攻击（内容攻击、情感攻击、显式噪声攻击、隐式噪声攻击），通过三种评估方法系统评测六个 SOTA 大型音频语言模型的鲁棒性，发现 GPT-4o 表现最优但所有模型均存在显著脆弱性。
tags:
  - ACL2025
  - 语音
  - 对抗音频攻击
  - 大型音频语言模型
  - 鲁棒性评估
  - benchmark
---

# Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models

**会议**: ACL2025  
**arXiv**: [2411.14842](https://arxiv.org/abs/2411.14842)  
**作者**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Ling Chen (UTS, Liverpool, 北京交通大学)
**代码**: [CAA Benchmark](https://github.com/YanqiYang/CAA)  
**领域**: audio_speech  
**关键词**: 对抗音频攻击, 大型音频语言模型, 鲁棒性评估, benchmark

## 一句话总结

提出 Chat-Audio Attacks (CAA) 基准，包含四类通用对抗音频攻击（内容攻击、情感攻击、显式噪声攻击、隐式噪声攻击），通过三种评估方法系统评测六个 SOTA 大型音频语言模型的鲁棒性，发现 GPT-4o 表现最优但所有模型均存在显著脆弱性。

## 研究背景与动机

- **问题**: 大型音频语言模型 (LALMs) 在语音人机交互中日益普及，但面临对抗性音频攻击的威胁。现有研究多聚焦于模型特定的攻击方法，缺乏面向真实场景的通用攻击评估体系
- **现有不足**: (1) 以往对抗音频攻击（如 Carlini and Wagner 2018）依赖梯度信号、仅针对特定模型，通用性差；(2) 通用对抗音频攻击（如说话人口误、嘈杂环境）对 LALMs 影响尚未充分探索；(3) 缺少面向对话场景、涵盖多种攻击类型的系统性评测基准
- **核心动机**: 构建一个基于对话场景的通用对抗音频攻击基准，全面评估 LALMs 在各种攻击类型下的鲁棒性，推动防御机制的发展

## 方法详解

### CAA 基准构建

每条数据为四元组，分别为原始音频、文本转录、无攻击音频和攻击变体集合。

**数据收集**: 从三个公开数据集手动筛选 360 条英语语音样本：
- **MELD** (120条): 情感对话数据集（Friends），含情感标签
- **TVQA** (120条): 电视节目对话数据集
- **Common Voice** (120条): 多语言语音识别数据集

**四类攻击生成**:

1. **内容攻击 (Content Attack)**: 保持语义一致，修改少量 token——同义词替换、token 重排、微小 token 变化。用 AzureSpeechSDK 重新生成音频。目标：检验模型对小幅 token 变化的敏感性
2. **情感攻击 (Emotional Attack)**: 不改变内容，改变情感语调——(a) 对立情感语调（如原 angry 改为 happy 重读）；(b) 叠加对立情感背景音乐。仅 MELD 数据支持此攻击。目标：检测模型对语音情感和内容-情感不匹配的敏感性
3. **显式噪声攻击 (Explicit Noise Attack)**: 叠加三类可听噪声——自然噪声（鸟鸣、风声、雷声）、工业噪声（汽车喇叭、机械声）、人类噪声（人群嘈杂、喊叫、笑声）。目标：评估模型区分语音和背景噪声的能力
4. **隐式噪声攻击 (Implicit Noise Attack)**: 叠加人耳不可听噪声——次声波 (15 Hz) 和超声波 (22000 Hz)。人类听不到但可能影响模型。目标：检验模型是否像人类一样不受不可听噪声影响

**质量控制**: 7 项手动筛选标准 + GPT-4 辅助过滤，确保样本可回答性；无攻击音频使用 AzureSpeechSDK 重新合成以保证清晰度。

### 三种评估方法

1. **标准评估 (Standard Evaluation)**: 比较无攻击 vs 攻击音频的模型输出，使用 WER、ROUGE-L、COS 余弦相似度三个指标
2. **GPT-4o 评估 (GPT-4o-Based Evaluation)**: 模拟真实交互场景，打分 1-5，四个指标：NC (无攻击连贯性)、ACoh (攻击连贯性)、ACor (攻击相关性)、LR (语言鲁棒性)
3. **人类评估 (Human Evaluation)**: 5 名英语母语大学生独立评分 NC 和 ACoh，反映真实用户体验和信任感

## 实验关键数据

评测六个 SOTA LALMs: SpeechGPT, SALMONN, Qwen2-Audio, LLama-Omni, Gemini-1.5-Pro, GPT-4o。

### Table 2: 标准评估结果（部分关键数据）

| 模型 | 指标 | Content Attack | Explicit-Natural | Implicit-Infrasound | Implicit-Ultrasound |
|------|------|------|------|------|------|
| GPT-4o | WER | 1.12 | 1.18 | 1.25 | 1.13 |
| GPT-4o | ROUGE-L | 0.25 | 0.20 | 0.22 | 0.17 |
| LLama-Omni | WER | 1.04 | 0.64 | 0.67 | 0.37 |
| LLama-Omni | ROUGE-L | 0.36 | 0.58 | 0.56 | 0.75 |
| SpeechGPT | WER | 1.79 | 2.25 | 2.21 | 1.28 |
| SpeechGPT | ROUGE-L | 0.17 | 0.12 | 0.14 | 0.20 |

**发现**: LLama-Omni 在标准指标上表现最一致；SpeechGPT 在所有攻击下性能退化最严重。

### Table 3: GPT-4o 评估结果（1-5 分）

| 模型 | NC | Content ACoh | Emo-Tone ACoh | Natural ACoh | Infrasound ACoh |
|------|------|------|------|------|------|
| GPT-4o | 4.45 | 3.94 | 4.35 | 2.57 | 3.02 |
| LLama-Omni | 3.50 | 3.05 | 3.08 | 2.72 | 2.79 |
| Gemini-1.5-Pro | 3.58 | 3.15 | 3.30 | 2.42 | 2.78 |
| SpeechGPT | 2.39 | 1.76 | 1.49 | 1.32 | 1.23 |

**发现**: GPT-4o 在无攻击连贯性 (NC=4.45) 和大部分攻击下表现最优；但所有模型在自然噪声下 ACoh 显著下降。

### Table 4: 人类评估结果

| 模型 | NC | Content ACoh | Emotion ACoh | Explicit ACoh | Implicit ACoh |
|------|------|------|------|------|------|
| GPT-4o | 4.33 | 3.88 | 4.12 | 3.27 | 3.08 |
| Gemini-1.5-Pro | 3.92 | 3.20 | 3.41 | 3.24 | 2.87 |
| LLama-Omni | 3.75 | 3.40 | 3.22 | 2.88 | 3.15 |

**发现**: 人类评估与自动评估趋势一致；GPT-4o 在用户感知层面同样最鲁棒。

## 亮点

- **系统性基准**: 首个面向 LALMs 对话场景的多类型音频攻击基准 (1680 样本, 4 攻击类型)，兼顾可听/不可听噪声
- **三维评估体系**: 标准指标 + GPT-4o 模拟 + 人类评估，三者互补，全面可信
- **深度分析**: 从模型架构角度解释鲁棒性差异——转录模块 (SpeechGPT) 是薄弱环节；含噪声训练数据 (Qwen2-Audio, LLama-Omni) 提升鲁棒性
- **发现隐式噪声影响**: 次声波和超声波（人耳不可听）竟然影响部分模型输出，揭示模型与人类听觉的差异

## 局限与展望

- 基准基于受控对话场景，可能无法覆盖高度动态的真实环境
- 缺少音频越狱攻击 (jailbreak) 的探索，该领域开源工作极少
- 仅评测英语，未涵盖多语言场景
- MELD 数据集专属情感标签，情感攻击仅覆盖 MELD 子集 (120 条)
- 部分闭源模型 (GPT-4o, Gemini) 的鲁棒性原因无法从架构层面深入分析
- 攻击类型相对固定，未考虑自适应/对抗训练后的攻击

## 与相关工作的对比

| 维度 | 本文 CAA | 传统音频对抗攻击 | 图像对抗攻击 |
|------|------|------|------|
| 攻击通用性 | 通用攻击，模型无关 | 模型特定 (梯度依赖) | 模型特定 |
| 目标模型 | 6 个 SOTA LALMs | 单一 ASR/ASV 系统 | CNN 分类器 |
| 攻击维度 | 内容+情感+噪声+隐式 | 仅内容扰动 | 像素扰动 |
| 评估方法 | 3 种互补方法 | 单一指标 | 分类准确率 |
| 场景 | 对话式交互 | 语音识别/验证 | 图像分类 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个面向 LALMs 对话场景的多类型音频攻击基准，隐式噪声攻击视角新颖
- 实验充分度: ⭐⭐⭐⭐ — 6 模型 x 4 攻击 x 3 评估方法，覆盖全面；但缺少防御方法实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，讨论深入，但部分表格较密集
- 价值: ⭐⭐⭐⭐ — 为音频对抗攻击防御研究提供标准化评测平台，实用性强

<!-- RELATED:START -->

## 相关论文

- [Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [Towards Reliable Large Audio Language Model](towards_reliable_large_audio_language_model.md)
- [Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)
- [Mind the Gap! Static and Interactive Evaluations of Large Audio Models](mind_the_gap_static_and_interactive_evaluations_of_large_audio_models.md)
- [Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation](../../NeurIPS2025/audio_speech/can_llms_outshine_conventional_recommenders_a_comparative_evaluation.md)

<!-- RELATED:END -->
