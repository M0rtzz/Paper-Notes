---
title: >-
  [论文解读] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens
description: >-
  [ACL 2025][语音][音视频语音识别] 提出 MMS-LLaMA，通过早期音视频融合、动态查询分配的 AV Q-Former 和语速预测器三个模块，将多模态语音 token 压缩至每秒仅 3.5 个，在 LRS3 上以 0.72% WER 达到 SOTA 的同时减少 86% token 用量和 35.7% FLOPs。
tags:
  - ACL 2025
  - 语音
  - 音视频语音识别
  - LLM语音识别
  - Token压缩
  - Q-Former
  - 语速预测
---

# MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens

**会议**: ACL 2025  
**arXiv**: [2503.11315](https://arxiv.org/abs/2503.11315)  
**代码**: https://github.com/JeongHun0716/MMS-LLaMA (有)  
**领域**: 语音  
**关键词**: 音视频语音识别, LLM语音识别, Token压缩, Q-Former, 语速预测

## 一句话总结

提出 MMS-LLaMA，通过早期音视频融合、动态查询分配的 AV Q-Former 和语速预测器三个模块，将多模态语音 token 压缩至每秒仅 3.5 个，在 LRS3 上以 0.72% WER 达到 SOTA 的同时减少 86% token 用量和 35.7% FLOPs。

## 研究背景与动机

**领域现状**：音视频语音识别（AVSR）通过结合音频和唇部运动的视觉信息，在嘈杂环境下实现鲁棒语音识别。近期基于 LLM 的 AVSR 系统（如 LLaMA-AVSR）利用 LLM 的上下文建模能力取得了出色效果，WER 已低至 0.77%。

**现有痛点**：基于 LLM 的 AVSR 系统计算代价极高——多模态语音 token 的时间分辨率远高于文本 token，导致 LLM 的自注意力机制需要处理大量 token。如 LLaMA-AVSR 每秒产生 25 个多模态 token，GPU 显存占用 18.2GB，FLOPs 达 2.24T。

**核心矛盾**：语音信号的高时间分辨率是保证识别精度的基础，但过多 token 导致 LLM 计算负担沉重。如何在压缩 token 数量的同时不丢失语言信息？

**本文目标**：设计高效的多模态语音 LLM 框架，以最少的 token 数量保留足够的语言内容，在不牺牲精度的前提下大幅降低计算成本。

**切入角度**：三步压缩策略——(1) 早期融合将音视频序列长度减半；(2) AV Q-Former 根据输入时长动态分配查询数量进一步压缩；(3) 语速预测器根据语速调整 token 分配，对快速语音分配更多 token。

**核心 idea**：通过动态查询分配和语速感知，可将多模态语音 token 压缩到每秒仅 3.5 个且不损失识别精度。

## 方法详解

### 整体框架

MMS-LLaMA 由以下组件构成：

1. **视觉编码器**（AV-HuBERT）：从唇部视频提取视觉特征 $\mathbf{X}_v \in \mathbb{R}^{T_v \times D}$
2. **音频编码器**（Whisper）：提取音频特征 $\mathbf{X}_a \in \mathbb{R}^{T_a \times D}$
3. **长度适配器**：对齐音频和视觉特征的时间分辨率
4. **早期 AV 融合模块**：将双模态序列融合为单一序列，长度减半
5. **AV Q-Former**：通过动态查询分配将序列进一步压缩到文本 token 级别
6. **语速预测器**：根据语速调整查询分配
7. **LLM 解码器**（LLaMA 3.2 3B）：从压缩后的多模态 token 预测文本

### 关键设计

#### 1. 早期 AV 融合模块

- **功能**：在送入 LLM 之前融合音频和视觉特征，将序列长度减半
- **核心思路**：先用长度适配器对齐两种模态的时间分辨率，然后比较三种融合策略：
    - **拼接**（Concatenation）：$\mathbf{X}_{av} = [\mathbf{X}'_a; \mathbf{X}_v] \in \mathbb{R}^{T_v \times 2D}$
    - **加法**（Addition）：$\mathbf{X}_{av} = \mathbf{X}'_a + \mathbf{X}_v \in \mathbb{R}^{T_v \times D}$
    - **多模态注意力**：$\mathbf{X}_{av} = \text{MHCA}(\mathbf{X}_v W_Q, \mathbf{X}'_a W_K, \mathbf{X}'_a W_V)$
- **设计动机**：将融合前移到 LLM 之前，避免 LLM 同时处理两组高分辨率序列。实验表明拼接在噪声条件下最优（WER 2.4%），最终采用拼接

#### 2. AV Q-Former（动态查询分配）

- **功能**：将变长的音视频特征序列压缩为接近文本 token 数量的短序列
- **核心思路**：定义可学习查询序列 $\mathbf{Q} \in \mathbb{R}^{N \times D_q}$，根据输入时长按比例分配查询数量：
  $$N_{\text{alloc}} = \lfloor f_Q \times \frac{T_v}{F_v} \rfloor$$
  其中 $f_Q$ 为查询频率（每秒查询数），$T_v/F_v$ 为输入时长。选取前 $N_{\text{alloc}}$ 个查询送入 Q-Former：
  $$\mathbf{M} = \text{Q-Former}(\mathbf{Q}_{\text{alloc}}; \mathbf{X}_{av})$$
- **设计动机**：固定查询数量的传统 Q-Former 无法处理变长输入——短的浪费、长的不够。动态策略确保语言内容与 token 数量成正比。实验发现查询频率降到 4Hz 仍能保持性能

#### 3. 语速预测器

- **功能**：根据每段音频的语速进一步优化 token 分配
- **核心思路**：训练一个轻量预测器估计归一化语速 $r_s$，修改分配公式为：
  $$N_{\text{alloc}} = \lfloor f_Q \times \frac{T_v}{F_v} \times r_s \rfloor$$
  快速语音分配更多 token，慢速语音分配更少
- **架构**：2 层 Transformer，256 维嵌入，4 头注意力，1024 维 FFN，用 MSE 损失和仅音频特征预训练
- **设计动机**：相同时长的音频可能包含不同量的语言内容。查询频率 3Hz 下加入语速预测器仅增加 0.7 token/秒但 WER 从 0.95% 降至 0.90%

### 损失函数/训练策略

- **语速预测器**：MSE 损失，以训练集平均语速为基准归一化，冻结参数单独预训练
- **主模型**：标准 CTC/序列生成损失训练
- 使用 **QLoRA** 微调 LLM：rank=16, alpha=32, dropout=0.05，仅微调 Q/K/V/Output 投影层
- 训练：Adam 优化器，初始 lr=1e-4，cosine 调度，30,000 步，0.5k warmup
- 推理：beam search（beam=5, temperature=0.3）
- 硬件：8 x RTX 3090 GPU

## 实验关键数据

### 主实验

**LRS3 数据集 SOTA 对比**：

| 方法 | 解码器 | 训练数据(h) | WER (Noisy)↓ | WER (Clean)↓ |
|------|--------|-------------|-------------|-------------|
| auto-avsr | Conformer | 3448 | - | 0.9 |
| LP Conformer | LSTM | 100K | 1.9 | 0.9 |
| Whisper-Flamingo | Whisper | 1759 | 5.6 | 0.76 |
| LLaMA-AVSR | LLaMA 3.1 8B | 1759 | - | 0.77 |
| **MMS-LLaMA** | **LLaMA 3.2 3B** | **1759** | **1.9** | **0.72** |

MMS-LLaMA 以更小的 LLM（3B vs 8B）达到 0.72% WER 的 SOTA，噪声条件下也仅 1.9%。

### 消融实验

**各模块逐步集成的效果**（433h 训练数据）：

| 方法 | Token/秒 | GPU 显存(GB) | FLOPs(T) | WER↓ |
|------|---------|-------------|----------|------|
| Baseline (LLaMA-AVSR) | 25 | 18.2 | 2.24 | 0.97 |
| + 早期 AV 融合 | 12.5 | 14.7 | 1.81 | 0.92 |
| + AV Q-Former (freq=3) | 2.8 | 12.2 | 1.42 | 0.95 |
| + 语速预测器 (freq=3) | **3.5** | **12.4** | **1.44** | **0.90** |

完整管线将 token 数从 25/秒降至 3.5/秒（减少 86%），FLOPs 从 2.24T 降至 1.44T（减少 35.7%），GPU 显存从 18.2GB 降至 12.4GB（减少 32%），同时 WER 从 0.97% 降至 0.90%。

**LLM 规模对比**：

| LLM | GPU 显存(GB) | FLOPs(T) | WER (Noisy)↓ | WER (Clean)↓ |
|-----|-------------|----------|-------------|-------------|
| LLaMA3.2-1B | 9.8 | 1.19 | 3.11 | 1.11 |
| **LLaMA3.2-3B** | **12.3** | **1.50** | **2.40** | **0.90** |
| LLaMA3.1-8B | 16.7 | 2.17 | 2.61 | 1.02 |

LLaMA3.2-3B 在 clean 和 noisy 条件下均最优，不是越大越好。

**不同 SNR 下视觉模态的作用**（query freq=3）：

| 条件 | 无视觉 | 有视觉 |
|------|--------|--------|
| Clean (∞ dB) | 1.10 | 0.95 |
| 0 dB | 2.66 | 2.66 |
| -5 dB | 13.54 | **7.44** |

视觉模态在强噪声下优势巨大：-5 dB 时从 13.54% 降至 7.44%。

### 关键发现

1. **3.5 token/秒足够**：多模态语音 token 可以被极度压缩（从 25 到 3.5/秒），损失极小
2. **早期融合很有效**：仅将序列减半就能同时降低 FLOPs 和 WER（0.97→0.92%）
3. **语速预测器的收益虽小但稳定**：不同查询频率下均能稳定提升 0.05-0.38% WER
4. **3B LLM 比 8B 更好**：在此任务中 LLM 参数量的最佳点在 3B 而非 8B
5. **视觉信息在噪声环境极为关键**：-5 dB 下视觉模态带来 45% 的相对 WER 降低

## 亮点与洞察

1. **压缩效率惊人**：86% 的 token 压缩率 + 35.7% FLOPs 降低 + 性能还有提升，实际部署价值巨大
2. **查询分配策略新颖**：动态 + 语速感知的双重调节机制，比固定窗口 Q-Former 更合理
3. **设计哲学清晰**：三个模块层层递进——先融合减半，再 Q-Former 压缩，最后语速调节精修
4. **视觉语速预测器**：附加实验证明纯视觉语速预测器（基于唇部运动）效果几乎等同于音频版本（0.75% vs 0.72%），在音频不可用场景下也能工作

## 局限与展望

1. 仅在 LRS3 数据集（英语 TED 演讲）上验证，场景和语言多样性有限
2. 拼接融合策略虽在噪声下最优但在干净条件下不如多模态注意力（0.90% vs 0.87%），可考虑自适应融合策略
3. 语速预测器的训练是分离的，端到端联合训练可能进一步提升
4. VoxCeleb2 的伪标签质量可能影响模型在非 TED 场景下的泛化

## 相关工作与启发

- **LLaMA-AVSR (Cappellazzo et al., 2024)**：直接基线，使用 25 token/秒的 LLaMA 3.1 8B 方案
- **Whisper-Flamingo (Rouditchenko et al., 2024)**：将预训练视觉编码器与 Whisper 集成
- **Q-Former (Dai et al., 2023)**：视觉-语言领域的 token 压缩方法，本文将其扩展到音视频语音
- **auto-avsr (Ma et al., 2023)**：Conformer 基础的 AVSR SOTA，用大量数据达 0.9% WER
- **启发**：LLM 在语音任务中的瓶颈不在模型能力而在计算效率，token 压缩是核心挑战；动态分配比固定策略更适合变长语音输入

## 评分

- **新颖性**: ⭐⭐⭐⭐ — AV Q-Former 的动态查询分配和语速预测器的结合很有创意
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融非常细致，覆盖融合策略/LLM 规模/查询频率/SNR 等多个维度
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，但部分符号使用较繁琐
- **价值**: ⭐⭐⭐⭐⭐ — 实际部署价值极高，86% 压缩率对多模态语音 LLM 的落地至关重要
