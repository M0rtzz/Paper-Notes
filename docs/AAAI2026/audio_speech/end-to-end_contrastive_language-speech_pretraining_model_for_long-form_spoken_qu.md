---
title: >-
  [论文解读] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering
description: >-
  [AAAI2026][语音][Spoken Question Answering] 提出 CLSR，一种端到端对比式语言-语音检索器，通过将声学表示先转换为 text-like representation 再与文本对齐，高效地从长音频中提取与问题相关的片段，为下游 LALM 的长语音问答提供 RAG 支持。
tags:
  - AAAI2026
  - 语音
  - Spoken Question Answering
  - 对比学习
  - Retrieval-Augmented Generation
  - Speech-Text Alignment
  - CIF
---

# End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering

**会议**: AAAI2026  
**arXiv**: [2511.09282](https://arxiv.org/abs/2511.09282)  
**代码**: [193746/CLSR](https://github.com/193746/CLSR)  
**领域**: audio_speech  
**关键词**: Spoken Question Answering, Contrastive Learning, Retrieval-Augmented Generation, Speech-Text Alignment, CIF

## 一句话总结
提出 CLSR，一种端到端对比式语言-语音检索器，通过将声学表示先转换为 text-like representation 再与文本对齐，高效地从长音频中提取与问题相关的片段，为下游 LALM 的长语音问答提供 RAG 支持。

## 背景与动机
- 现有 SQA（Spoken Question Answering）模型大多只能处理不超过 1 分钟的短音频，但真实场景（会议、讲座、在线讨论）的语音往往超过 10 分钟
- 大型音频语言模型（LALM）虽然语音理解能力强，但在长音频上推理速度慢、精度下降
- RAG 在文本长上下文 QA 中效果显著，自然产生一个问题：能否将 RAG 应用于语音，从长音频中检索与问题最相关的片段？
- 已有的语音检索器（如 CLAP、SpeechDPR）性能不足——CLAP 擅长"音效-文本"对齐而非"语音内容-文本"对齐，SpeechDPR 受限于无文本训练且数据稀缺

## 核心问题
如何构建一个端到端的语音-文本检索器，使其在不依赖级联 ASR+文本检索的前提下，也能达到甚至超过 pipeline 方法的检索精度，同时大幅降低长音频问答推理的时间和错误率？

## 方法详解

### 整体架构
CLSR 由两部分组成：
1. **左半部分**：基于 CIF（Continuous Integrate-and-Fire）的非自回归注意力编解码器（AED），输入语音 $X$，输出 token 概率分布 $D$
2. **右半部分**：Transformer 文本编码器（冻结的 BGE-base），接收 text-like embedding 或真实文本 embedding，输出句子级表示用于对比学习

### CIF 模块
- 语音编码器（SAN-M 结构）提取声学特征 $H^s$
- CIF 通过卷积计算每帧的权重 $\alpha_i \in [0,1]$，逐帧累加直到超过阈值 $\beta$，从而将时间步长映射到 token 数量，得到声学表示 $E^a$
- 这一步实现了从"帧级"到"token 级"的软单调对齐

### Sampler 训练优化
- 训练分两轮：第一轮直接用 $E^a$ 预测 token 分布，得到 ASR 输出 $Y^{asr}$
- 第二轮比较 $Y^{asr}$ 与 ground-truth $Y^{con}$，将错误 token 位置的正确 embedding 以采样比 $\lambda$ 替换进 $E^a$，生成混合特征 $E^s$
- 用 $E^s$ 重新预测 token 分布 $D'$，增强解码器的上下文建模能力

### VQ Adaptor（向量量化适配器）
- 对 token 概率分布 $D$ 做 argmax 取最大概率 token 索引 $q_i$
- 用温度缩放 softmax（$\gamma=0.1$）+ straight-through 梯度估计保持梯度传播
- 将量化后的 one-hot 矩阵 $Q^{st}$ 与文本编码器的 embedding 权重 $W^{te}$ 做矩阵乘法，得到 text-like embedding $E^{Y'}$
- 关键思想：不直接对齐声学表示与文本表示，而是通过 VQ 将声学表示"翻译"到文本空间的近似表示，再在文本空间内做对比学习

### 对比学习与损失函数
- 将 context 的 text-like embedding 和 question 的文本 embedding 输入文本编码器，用 CLS token 提取句子级表示
- 用余弦相似度 + NLL 损失训练对齐
- 总损失：$\mathcal{L}_{total} = (1-\alpha-\beta)\mathcal{L}_{ASR} + \alpha\mathcal{L}_{MAE} + \beta\mathcal{L}_{NLL}$，其中 $\alpha=\beta=\frac{1}{3}$

### 训练策略
- **预训练阶段**：用 LibriSpeech 460h 预训练 Paraformer（ASR），用干净文本对预训练 BGE
- **联合训练**：冻结 BGE，联合优化 ASR 模块和对比损失
- **后训练**：冻结 ASR，对 BGE 微调几个 epoch 以适应 text-like representation

## 实验关键数据

### 数据集
四个数据集：Spoken-SQuAD、LibriSQA、SLUE-SQA-5（真实录音）、DRCD（中文）

### 主要结果（Spoken-SQuAD*）

| 模型 | 范式 | WER↓ | Q→C R@1 | Q→C R@10 |
|------|------|------|---------|----------|
| CLAP | E2E | - | 2.93 | 14.84 |
| Whisper+BGE | Pipeline | 19.39 | 69.93 | 90.53 |
| **CLSR** | **E2E** | **15.14** | **70.03** | **90.68** |

- CLSR 在所有四个数据集上大幅超越 CLAP（R@1 从 ~3% 提升到 ~70%）和 SpeechDPR
- 与 Whisper+BGE pipeline 方法性能相当或更优，同时 WER 更低（15.14 vs 19.39）
- 在 LibriSQA 上 CLSR R@1=85.04%，接近纯文本 BGE 的 86.91%

### 消融实验要点
- **去掉 VQ adaptor**：R@10 从 ~86% 暴跌到 ~44%，验证了 text-like representation 的核心价值
- **去掉 Sampler**：WER 从 15.01 升到 16.18，检索召回也下降
- **预训练 ASR 和 BGE** 对最终性能都有显著帮助
- WER ~16.75% 是一个阈值，超过此值检索性能急剧下降

### 长音频 SQA 实际效果
在 Spoken Wikipedia（平均 30 分钟音频）上测试：
- 无 CLSR：EM=18.00, F1=23.55, 耗时 7935s
- **有 CLSR：EM=27.60, F1=35.10, 耗时 783s（10× 加速）**

## 亮点
1. **首次将 RAG 引入 SQA 领域**，为长音频问答提供了系统化的解决框架
2. **text-like representation 桥接策略**巧妙规避了语音-文本直接对齐的困难，借助成熟的文本对比学习模型实现高质量跨模态检索
3. **无需大规模语音-文本预训练**，仅用任务数据联合训练即可达到与 pipeline 方法相当的性能
4. VQ adaptor 的 straight-through 估计设计保证了端到端训练的可行性

## 局限与展望
- 仅在 TTS 合成语音和有限的真实录音上测试，对噪声环境、多说话人场景的鲁棒性未知
- 当前固定将长音频切为 40 秒片段，缺乏自适应的语义分割策略
- BGE 在联合训练中冻结，后训练带来的提升有限，可探索更好的解冻策略
- 未与更新的语音基础模型（如 Whisper-v3、SeamlessM4T）进行对比
- 长音频实验仅用 500 条样本，规模偏小

## 与相关工作的对比

| 方法 | 特点 | 不足 |
|------|------|------|
| CLAP | 音频-文本对比学习 | 适合音效匹配，不适合语音内容检索 |
| SpeechDPR | 无文本训练的语音检索 | 数据稀缺导致性能不佳（R@20 仅 19.94） |
| Whisper+BGE | ASR 级联文本检索 | 依赖 ASR 质量，错误传播，中文能力弱 |
| **CLSR** | **VQ 桥接 + 联合训练** | **E2E 达到 pipeline 水平，WER 和检索同时优化** |

## 启发与关联
- text-like representation 的思路可迁移到其它跨模态检索任务（如视频-文本检索中先将视频表示"翻译"到文本空间）
- CIF + VQ 的组合可作为通用的"语音→离散 token"前端，替代传统离散化方案
- 长音频 RAG 的框架可与流式 ASR 结合，实现实时会议问答系统

## 评分
- 新颖性: ⭐⭐⭐⭐ （首次将 RAG 引入 SQA，text-like representation 桥接方式新颖）
- 实验充分度: ⭐⭐⭐⭐ （四数据集 + 消融 + 长音频验证，但长音频实验规模偏小）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，公式完整，图示辅助理解）
- 价值: ⭐⭐⭐⭐ （为长音频 SQA 提供实用框架，10× 推理加速有应用前景）

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](../../ACL2025/audio_speech/contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](../../ACL2025/audio_speech/omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)
- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](../../ACL2025/audio_speech/dncasr_end-to-end_training_for_speaker-attributed_asr.md)

<!-- RELATED:END -->
