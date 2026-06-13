---
title: >-
  [论文解读] Distilling an End-to-End Voice Assistant Without Instruction Training Data
description: >-
  [ACL 2025][音频/语音][语音大语言模型] 提出DiVA（Distilled Voice Assistant），通过将文本LLM对转录文本的响应作为自监督信号进行跨模态蒸馏，无需任何语音指令训练数据即可训练端到端语音LLM——仅用3.5k小时ASR数据就泛化到口语问答、分类和翻译任务，且在用户偏好测试中以72%胜率碾压Qwen 2 Audio（使用100倍以上训练计算量）。
tags:
  - "ACL 2025"
  - "音频/语音"
  - "语音大语言模型"
  - "跨模态蒸馏"
  - "端到端语音助手"
  - "知识迁移"
  - "Whisper"
---

# Distilling an End-to-End Voice Assistant Without Instruction Training Data

**会议**: ACL 2025  
**arXiv**: [2410.02678](https://arxiv.org/abs/2410.02678)  
**代码**: [github](https://github.com/google-research/google-research/tree/master/DiVA)  
**领域**: 语音/音频  
**关键词**: 语音大语言模型, 跨模态蒸馏, 端到端语音助手, 知识迁移, Whisper

## 一句话总结

提出DiVA（Distilled Voice Assistant），通过将文本LLM对转录文本的响应作为自监督信号进行跨模态蒸馏，无需任何语音指令训练数据即可训练端到端语音LLM——仅用3.5k小时ASR数据就泛化到口语问答、分类和翻译任务，且在用户偏好测试中以72%胜率碾压Qwen 2 Audio（使用100倍以上训练计算量）。

## 研究背景与动机

将LLM能力扩展到语音模态有重要价值：语音是更自然的交互方式，直接处理语音可以保留语调、节奏、口音等ASR会丢失的信息，还能降低标注成本和延迟。

当前Speech LLM主要采用**大规模多任务监督微调（SFT）**，但面临严重问题：

**"遗忘"问题**：SFT训练后模型会丢失文本LLM的已有能力。即使冻结LLM权重也无法完全避免

**数据瓶颈**：要防止遗忘需要大量多样化的标注语音指令数据，但这种数据目前极度稀缺

**代表性不足**：现有语音数据往往只来自少量说话人，加剧了语音处理中的偏差问题

**开源困难**：最先进的Speech LLM训练细节不公开，难以复现

然而，语音社区已积累了大量**ASR数据**（如CommonVoice、LibriSpeech）。问题是SFT无法仅用ASR数据训练出通用语音助手。作者提出了一种完全不同的范式：通过蒸馏，仅用ASR数据就能训练出泛化的Speech LLM。

## 方法详解

### 整体框架

DiVA由三个预训练组件初始化：
- **音频编码器**：Whisper-Large-v3的编码器（提取音频特征）
- **音频-文本特征对齐器**：从Whisper解码器初始化的Q-Former（将高粒度音频特征聚合为LLM可用的嵌入）
- **文本解码器**：Llama 3（冻结权重，负责推理和生成）

训练过程只用ASR数据的音频和转录文本对，通过两个蒸馏损失优化音频编码器和Q-Former。

### 关键设计

1. **从Whisper解码器初始化Q-Former**: 先前工作丢弃Whisper解码器，重新训练Q-Former来对齐音频和文本。但Whisper解码器本身就是为ASR训练的——它的交叉注意力机制已经学会了将音频嵌入映射到离散文本token。作者直接复用Whisper解码器的K、V交叉注意力权重初始化Q-Former，仅替换输入为静态查询token Q。这大幅减少了训练成本，实现"两全其美"——既有Q-Former的自适应降维能力，又省去了从头训练的巨大开销。

2. **跨模态Token对齐损失（Lcon）**: 将音频录音通过音频编码器得到Q个音频token，将对应转录文本嵌入为N个文本token（通常Q > N）。然后最小化文本嵌入与音频嵌入**最后N个**token的L2距离：$L_{con} = \sum_{n=0}^{N} |t_n^{text} - t_{Q-N+n}^{audio}|_2$。选择"最后N个"而非"最初N个"是因为Whisper解码器使用因果注意力——最后的token能attend到所有前面的token，对齐它们可以将梯度信号传播到整个序列。多出的Q-N个token则为传递语调等非文本信息提供带宽。

3. **输出分布蒸馏损失（核心创新）**: 目标是让模型对音频输入的响应分布与对文本输入的响应分布一致——跨模态的context distillation。对于KL散度最小化，作者证明了一个关键引理：当教师和学生共享输出嵌入矩阵O（冻结的Llama输出层）时，**最小化隐藏状态的L2距离是最小化KL散度的子集**。这不仅梯度更平滑，而且由于LLM词汇表远大于隐藏维度，L2距离计算量远小于KL散度。实际训练中只对比第一个预测token的隐藏状态，因为单个token概率已编码了大量信息。

4. **两个损失的协同作用**: 仅用蒸馏损失可以训出能力不错的模型（问答强于SFT基线），但会忽略文本指令（如翻译时忽略目标语言）。Token对齐损失单独使用效果很差（生成不连贯），但与蒸馏损失结合后显著改善了指令遵循能力（语言识别正确率从1.4%提升到74%）。

### 训练策略

- **训练数据**：仅用CommonVoice 17英文子集，3.5k小时，93,725个说话人
- **训练量**：4300步，batch size 512，约2个epoch
- **硬件**：TPU v4-256，约12小时完成
- **优化器**：AdamW，lr=5e-5，余弦衰减
- **冻结策略**：Llama 3权重完全冻结

## 实验关键数据

### 主实验 - 口语问答

| 模型 | HeySquad (PANDA) | SDQA (PANDA) |
|------|-----------------|--------------|
| SALMONN | 较低 | 较低 |
| Qwen Audio Chat | 较低（30%直接转录） | 较低 |
| Qwen 2 Audio | 中等（4%忽略指令） | 中等 |
| **DiVA** | **最高（+5 PANDA，显著P<0.05）** | **最高** |

DiVA在两个问答基准上显著超越所有基线（至少+10%），且是唯一100%遵循指令而非转录问题的模型。

### 主实验 - 情感分类

| 模型 | MELD (F1) | IEMOCAPS (F1) |
|------|-----------|---------------|
| SALMONN | 低（偏向Neutral） | 低 |
| Qwen Audio | 低（>90%预测Sadness） | 低 |
| Qwen 2 Audio | 低 | 低 |
| **DiVA** | **显著最优** | **显著最优** |

DiVA在未经任何情感监督训练的情况下显著优于所有基线，说明蒸馏过程隐式学习了文本与音频情感的关联。

### 主实验 - 语音翻译（7种语言）

| 模型 | 最优语言 | 弱势语言 |
|------|---------|---------|
| Qwen Audio | 中文、日文 | - |
| Qwen 2 Audio | 阿拉伯语、德语、印尼语 | - |
| **DiVA** | **土耳其语、泰米尔语** | 中文、日文 |

翻译结果较为混合。DiVA在中文和日文上弱于基线，原因是Llama 3本身对这两种语言偏向输出罗马字母拼写。

### 用户偏好研究

| 对比 | DiVA胜率 | 用户偏好 |
|------|---------|---------|
| DiVA vs Qwen 2 Audio | **72%** | 53人中41人（77%）偏好DiVA |

522次配对评估中DiVA以压倒性优势获胜，验证了基准测试可能无法反映实际用户需求。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 完整DiVA（蒸馏+对齐） | 问答最优，翻译好 | 两个损失协同 |
| 仅蒸馏损失 | 问答优于SFT基线 | 但翻译几乎为零（语言ID仅1.4%） |
| 仅Token对齐损失 | 生成不连贯 | 分类近似随机 |

### 关键发现

- 仅用ASR数据就能泛化到问答、分类、翻译三大类任务，说明跨模态蒸馏是可行的
- 用户偏好与基准测试不完全一致——72%用户偏好DiVA但基准上不是所有任务最优
- DiVA使用>100x更少的训练计算量却获得更好的用户满意度
- 蒸馏会继承LLM的坏习惯（如Llama偏向打字中日文的拉丁转写、总认为文本是幽默的）
- Whisper解码器的复用是关键的工程创新——避免了Q-Former从头训练的巨大成本

## 亮点与洞察

- **范式创新**：用蒸馏替代SFT，解决了"遗忘"和"数据稀缺"两大难题，开辟了Speech LLM训练的新路径
- **极致效率**：3.5k小时数据、12小时训练就超越使用50k+小时+DPO的Qwen 2 Audio
- **数学优雅**：共享输出矩阵时L2距离⊂KL散度最小化的证明，将复杂的KL蒸馏简化为简单的L2回归
- **Whisper解码器复用**：被先前工作完全忽视的组件变成了关键的初始化来源
- **开放研究**：使用完全开源的数据和代码，且释放训练代码而非仅推理代码

## 局限与展望

- 继承Llama 3的缺陷（中日文翻译偏差、对幽默/讽刺的错误判断）
- 纯语调理解能力仍弱——讽刺和幽默检测与随机相近
- 仅在英文CommonVoice上训练，多语言场景未充分验证
- Token对齐损失的"最后N个"对齐策略可能不是最优的
- 未探索音频生成（TTS）方向的蒸馏
- 用户研究规模较小（53人、522次评估），有扩大验证的空间

## 相关工作与启发

- 与SALMONN、Qwen Audio系列形成直接对比，展示了截然不同的训练范式
- Context distillation的跨模态扩展——从文本到语音，可迁移到图像、视频等模态
- Q-Former初始化策略可启发其他需要跨模态对齐的工作
- 为"如何高效地将LLM能力迁移到新模态"这一通用问题提供了优雅答案

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 跨模态蒸馏替代SFT的范式性创新，Whisper解码器复用巧妙
- 实验充分度: ⭐⭐⭐⭐ 覆盖问答/分类/翻译+用户研究+消融，但翻译结果混合
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、数学推导优美、结构紧凑、可读性极好
- 价值: ⭐⭐⭐⭐⭐ 为Speech LLM领域指明了一条高效且开放的新路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DNCASR: End-to-End Training for Speaker-Attributed ASR](dncasr_end-to-end_training_for_speaker-attributed_asr.md)
- [\[ACL 2025\] OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation](omniflatten_an_end-to-end_gpt_model_for_seamless_voice_conversation.md)
- [\[ACL 2025\] Contextual Biasing with the Knowledgeable External Language Model for End-to-End Speech Recognition](contextual_biasing_with_the_knowledgeable_external_language_model_for_end-to-end.md)
- [\[ACL 2025\] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)

</div>

<!-- RELATED:END -->
