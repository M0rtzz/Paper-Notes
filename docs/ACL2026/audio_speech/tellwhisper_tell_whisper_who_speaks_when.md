---
title: >-
  [论文解读] TellWhisper: Tell Whisper Who Speaks When
description: >-
  [ACL 2026][语音][多说话人语音识别] 本文提出TellWhisper，通过设计时间-说话人感知的旋转位置编码（TS-RoPE）将说话人身份和时间信息统一编码到语音编码器的自注意力中，配合双曲空间说话人日志模型（Hyper-SD），实现了对"谁在何时说了什么"的联合建模，在多说话人ASR任务上取得最优性能。
tags:
  - ACL 2026
  - 语音
  - 多说话人语音识别
  - 说话人日志
  - 旋转位置编码
  - 双曲空间分类
  - Whisper
---

# TellWhisper: Tell Whisper Who Speaks When

**会议**: ACL 2026  
**arXiv**: [2601.03712](https://arxiv.org/abs/2601.03712)  
**代码**: [项目主页](https://walker-hyf.github.io/TellWhisper)  
**领域**: 音频语音  
**关键词**: 多说话人语音识别, 说话人日志, 旋转位置编码, 双曲空间分类, Whisper

## 一句话总结

本文提出TellWhisper，通过设计时间-说话人感知的旋转位置编码（TS-RoPE）将说话人身份和时间信息统一编码到语音编码器的自注意力中，配合双曲空间说话人日志模型（Hyper-SD），实现了对"谁在何时说了什么"的联合建模，在多说话人ASR任务上取得最优性能。

## 研究背景与动机

**领域现状**：多说话人自动语音识别（MASR）旨在从多人对话语音中预测"谁在何时说了什么"。传统方案将说话人日志（SD）和单说话人ASR通过时间戳对齐进行融合，但在重叠语音和快速换说话人场景下对齐困难。

**现有痛点**：即使是近年试图统一SD和ASR的方法，也本质上将时间建模和说话人建模分开处理。具体表现为三种策略的局限：(1) 编码前用SD标签遮罩非目标区域，导致空白输入引发幻觉；(2) 尝试分离目标说话人的语音，需要额外的说话人提示且在重叠区域失效；(3) 在编码器输出后通过说话人后验加权线性混合，将语义与说话人线索纠缠。

**核心矛盾**：时间结构和说话人身份的分离建模在快速换说话人和重叠语音场景下本质上是脆弱的——时间和说话人是耦合的，应当联合建模而非事后拼接。

**本文目标**：在语音编码器内部通过位置编码自然地联合建模时间和说话人信息，使自注意力机制能同时关注"何时"和"谁"。

**切入角度**：受多维RoPE在视觉和多模态中跨轴编码的启发，将RoPE从仅编码时间扩展到同时编码时间和说话人活动状态。

**核心 idea**：设计TS-RoPE，将Query/Key通道划分为时间子空间和说话人子空间，通过区域特定的旋转角度在自注意力中实现时间-说话人的联合建模。

## 方法详解

### 整体框架

TellWhisper基于Whisper large-v3-turbo构建。输入多说话人语音经卷积层编码后，Hyper-SD首先估计帧级说话人活动。然后TS-RoPE利用时间索引和说话人活动信息构建多维位置编码，注入编码器自注意力中。最终通过结构化内容预测器以自回归方式输出说话人标签、时间戳和转录文本。

### 关键设计

1. **TS-RoPE（时间-说话人旋转位置编码）**:

    - 功能：在自注意力的Query/Key中同时编码时间和说话人信息
    - 核心思路：将每帧特征的通道维度D划分为16维一组，每组8个旋转对按交替方式分配给时间和4个说话人子空间：$[\psi_{time}, \psi_{spk_1}, \psi_{time}, \psi_{spk_2}, \psi_{time}, \psi_{spk_3}, \psi_{time}, \psi_{spk_4}]$。时间位置直接用帧索引 $\psi_{time}(f_t) = t$；说话人位置由累积说话人轮次计数和当前活动概率组成 $\psi_{spk_s}(f_t) = \mathcal{C}_{t,s} + \pi_{t,s}$。此外对Query的说话人子空间施加额外相位偏置 $\psi'_{spk_s}(f_t) = \psi_{spk_s}(f_t) + (1 - \pi_{t,s})$，鼓励注意力聚焦活跃说话人
    - 设计动机：通过旋转角度差异，相同说话人的连续语音帧获得相近的旋转角度（角度差小→注意力权重高），不同说话人或说话人切换处角度差异大，从而实现对说话人内连续性和说话人间切换的建模

2. **Hyper-SD（双曲空间说话人日志）**:

    - 功能：估计可靠的帧级说话人活动概率
    - 核心思路：利用WavLM多层特征加权聚合后经Conformer编码上下文信息。将欧几里得特征映射到Poincaré球模型中，为每种说话人组合类（共 $2^4 = 16$ 类，含静默、单说话人、重叠组合等）分配可学习的双曲原型 $\mathbf{p}_n$。通过帧嵌入到原型的双曲距离 $d_{t,n} = d_{\mathbb{B}_c}(\mathbf{v}'_t, \mathbf{p}_n)$ 计算类别概率，再边际化得到每个说话人的帧级活动 $\pi_{t,s} = \sum_n b_{s,n} \sigma(-d_{t,n})$
    - 设计动机：双曲空间具有指数级体积增长，小的特征偏移就能产生大的距离变化，显著提高音色相似说话人之间的可分性，稳定说话人后验估计

3. **结构化内容预测器**:

    - 功能：将编码器输出转化为结构化的"说话人+时间戳+文本"序列
    - 核心思路：将同一说话人的时间连续语音视为独立片段，每段表示为token序列 $\langle spk_s \rangle, \langle t_{start} \rangle, \langle text \rangle, \langle t_{end} \rangle$，所有片段按时间顺序拼接。使用自回归框架训练下一token预测，解码时逐token生成直到EOS
    - 设计动机：统一预测格式避免了传统流水线中SD输出和ASR输出的对齐问题

### 损失函数 / 训练策略

采用两阶段微调策略：先在单说话人语音（LibriSpeech）上预微调学习单说话人结构化预测，再在多说话人对话语音上微调。Hyper-SD使用NLLLoss训练，双曲分类器用RiemannianAdam优化，其余组件用AdamW。WavLM使用较小学习率，其他模块较大。

## 实验关键数据

### 主实验

| 数据集 | 指标 | TellWhisper | Dicow(之前SOTA) | 提升 |
|--------|------|-------------|----------------|------|
| AMI | CP-WER↓ | 32.53 | 33.57 | -1.04 |
| NotSoFar | CP-WER↓ | 34.48 | 35.22 | -0.74 |
| LibriCSS | CP-WER↓ | 9.88 | 10.62 | -0.74 |
| AMI | TCP-WER↓ | 33.47 | 34.02 | -0.55 |
| NotSoFar | TCP-WER↓ | 34.51 | 35.64 | -1.13 |
| LibriCSS | TCP-WER↓ | 11.06 | 11.33 | -0.27 |

### 消融实验

| 配置 | AMI CP-WER | AMI TCP-WER | 说明 |
|------|-----------|-------------|------|
| 完整TellWhisper | 32.53 | 33.47 | 所有组件启用 |
| w/o Query相位偏置 | 35.02 | 35.26 | CP-WER +2.49 |
| w/o 说话人轮次计数 | 36.22 | 36.68 | CP-WER +3.69 |
| w/o 说话人活动 | 36.84 | 36.89 | 最大退化 |

### 关键发现

- Hyper-SD在所有6个SD数据集上均超越Pyannote3和Diarizen，证实双曲空间分类优于欧几里得线性分类
- 在AliMeeting上DER改善最显著（13.03→10.76），说明在真实会议场景中双曲空间的说话人分离能力尤为突出
- 消融实验证明TS-RoPE的三个组件（活动概率、轮次计数、Query偏置）逐层贡献，说话人活动信号是最关键的
- TellWhisper的优势在真实会议场景（AMI、NotSoFar）上比模拟数据（Libri2Mix）更明显，因模拟数据中重叠从时间零开始且无说话人切换，TS-RoPE的优势有限

## 亮点与洞察

- TS-RoPE的设计优雅——通过旋转位置编码的通道划分和角度调制，在不改变模型主体架构的情况下注入时间-说话人耦合信息
- 双曲空间用于说话人活动估计是巧妙的——利用负曲率空间的指数体积增长放大音色相似说话人之间的距离
- Query端额外相位偏置的设计直觉清晰：非活跃说话人获得更大的偏置→注意力更倾向于活跃说话人
- 可视化显示16个类原型在双曲空间中均匀分布且无层级结构，符合帧级分类的需求

## 局限与展望

- 当前TS-RoPE设计支持1-4个说话人，扩展到更多说话人需要进一步研究
- Hyper-SD仅在特征提取后进行双曲分类，编码器和分类器处于不同嵌入空间，端到端双曲学习有望进一步提升
- 实验主要在英文数据集上进行，跨语言泛化性有待验证
- Libri2Mix上优势不明显，说明在极端重叠但无换说话人场景下TS-RoPE的收益有限

## 相关工作与启发

- **vs Dicow（Polok et al.）**: Dicow在编码前用说话人掩码过滤，可能引发幻觉；TellWhisper在编码器内部通过位置编码融合说话人信息，更加无缝
- **vs SortFormer（Park et al.）**: SortFormer在编码器输出后加说话人正弦核加权，线性混合纠缠语义和说话人；TS-RoPE通过旋转角度实现解耦的联合建模
- **vs 多维RoPE（视觉领域）**: 视觉RoPE编码宽度/高度等空间轴；TellWhisper创新性地将说话人活动作为新的维度引入RoPE

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ TS-RoPE将RoPE扩展到时间-说话人联合编码，思路新颖且实现优雅
- 实验充分度: ⭐⭐⭐⭐ 4个MASR数据集+6个SD数据集，多基线对比和详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导完整
- 价值: ⭐⭐⭐⭐⭐ 对多说话人语音理解有重要推动，TS-RoPE思想可扩展到其他多维序列建模任务

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Listen Like a Teacher: Mitigating Whisper Hallucinations using Adaptive Layer Attention and Knowledge Distillation](../../AAAI2026/audio_speech/listen_like_a_teacher_mitigating_whisper_hallucinations_using_adaptive_layer_att.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](../../ICLR2026/audio_speech/knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)
- [\[ICLR 2026\] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment](../../ICLR2026/audio_speech/when_style_breaks_safety_defending_llms_against_superficial_style_alignment.md)
- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](../../ACL2025/audio_speech/who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)

<!-- RELATED:END -->
