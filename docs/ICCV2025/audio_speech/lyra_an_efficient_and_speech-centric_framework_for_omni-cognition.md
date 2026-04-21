---
title: >-
  [论文解读] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition
description: >-
  [ICCV 2025][语音][多模态大语言模型] 提出 Lyra，一个以语音为中心的高效全模态 MLLM 框架，通过多模态 LoRA、潜在跨模态正则化器和潜在多模态提取器三大策略，使用更少的训练数据实现视觉-语言-语音多模态的 SOTA 性能，并首次支持长达数小时的语音输入。
tags:
  - ICCV 2025
  - 语音
  - 多模态大语言模型
  - 语音理解
  - 全模态认知
  - 高效推理
  - 长语音
---

# Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition

**会议**: ICCV 2025  
**arXiv**: [2412.09501](https://arxiv.org/abs/2412.09501)  
**代码**: [github.com/dvlab-research/Lyra](https://github.com/dvlab-research/Lyra)  
**领域**: 语音/多模态  
**关键词**: 多模态大语言模型, 语音理解, 全模态认知, 高效推理, 长语音

## 一句话总结

提出 Lyra，一个以语音为中心的高效全模态 MLLM 框架，通过多模态 LoRA、潜在跨模态正则化器和潜在多模态提取器三大策略，使用更少的训练数据实现视觉-语言-语音多模态的 SOTA 性能，并首次支持长达数小时的语音输入。

## 研究背景与动机

### 领域现状

当前多模态大语言模型（MLLM）主要聚焦于视觉-语言或语音-语言双模态交互，但真正的全模态（Omni）模型——能同时处理图像、视频、语音、声音的模型——仍然欠发展。OpenAI 的 GPT-4o 展示了全模态交互的可能性，但开源社区的全模态模型在语音方面的能力明显不足。

### 现有痛点

**语音模态被忽视**：现有全模态模型（VITA、AnyGPT、EMOVA 等）主要关注语音与文本的关系，未深入探索语音与视觉等其他模态的交互

**数据规模与训练成本的矛盾**：扩展到全模态需要更大规模的数据集和更多的计算资源

**长语音处理受限**：现有模型受限于语音编码器（如 Whisper），最多只能处理 30 秒到 1 分钟的音频输入

**评估标准不完善**：之前的全模态模型仅用 LibriSpeech WER 评估语音-文本能力，忽略了语音与视觉等跨模态的性能

### 核心发现

语音-文本性能好不等于语音-视觉性能好。实验表明，仅用 $\mathcal{L}_{CE}$ 训练时，语音+图像(S+I)相比文本+图像(T+I)在 MM-Vet 上差 8 个百分点（53.1 vs 61.1），说明语音 token 与文本 token 之间的语义鸿沟需要专门弥合。

## 方法详解

### 整体框架

Lyra 由四个核心组件组成：潜在跨模态正则化器（LCMR）、多模态 LoRA、潜在多模态提取器（LMME）和流式语音-文本生成模块。各模态数据通过各自的编码器和投影器处理后送入 LLM，在 LLM 内部多模态 LoRA 和 LMME 模块协同工作。

### 关键设计

#### 1. **潜在跨模态正则化器（Latent Cross-Modality Regularizer, LCMR）**

- **功能**：在输入 LLM 之前，将语音 token 尽量对齐到对应的转录文本 token，弥合语音与文本之间的语义鸿沟
- **核心思路**：语音 token 和转录文本 token 长度不一致（语音通常更长），使用动态时间规整（DTW）算法计算两者的最小距离：

$$\mathbf{D}_{l,s} = \text{dist}(l,s) + \min\{\mathbf{D}_{l,s-1}, \mathbf{D}_{l-1,s}, \mathbf{D}_{l-1,s-1}\}$$

其中 $\text{dist}(l,s) = -\log[\text{softmax}(\mathbf{X}_{[\text{speech}],l} \mathbf{X}_{[\text{STT}],s}^\top / \tau)]$

最终正则化损失 $\mathcal{L}_{LCMR} = \frac{1}{L+S}\mathbf{D}_{L,S}$，总损失 $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{LCMR}$

- **设计动机**：语音虽然在语义上与文本高度重叠，但直接使用语音 token 作为指令会显著降低跨模态性能。通过 DTW 对齐不等长序列，使语音 token 在输入 LLM 之前就尽量接近文本 token 的表示

#### 2. **多模态 LoRA（Multi-Modality LoRA）**

- **功能**：为不同模态组合使用独立的 LoRA 适配器，在保留原有视觉能力的同时扩展语音能力
- **核心思路**：输出 $\mathbf{H} = (\mathbf{B}_{[M]}\mathbf{A}_{[M]} + \mathbf{W})\mathbf{X}_{[M]}$，其中 $\mathbf{A}_{[M]}$, $\mathbf{B}_{[M]}$ 是模态组合 M 对应的低秩适配器
- **设计动机**：在数据量和质量有限的情况下，直接联合训练视觉-语音-语言三模态会降低原有的视觉能力。LoRA 的参数高效特性避免了这个问题

#### 3. **潜在多模态提取器（Latent Multi-Modality Extractor, LMME）**

- **功能**：根据文本查询的相关性动态筛选多模态 token，丢弃冗余 token 以提升效率
- **核心思路**：将 LLM 分为 $n$ 个 block，在每个 block 末尾根据文本 query 与非文本 token 的注意力分数筛选 top-k：

$$\text{topk}\left(\text{softmax}\left(\frac{\mathbf{Q}_{[\text{text}]}\mathbf{K}_{[\backslash\text{text}]}^\top}{\sqrt{d}}\right)\right)$$

每个 block 保留 $\rho L$ 个 token，token 数量按 block 指数衰减
- **设计动机**：多模态长上下文（高分辨率图像、长视频、长语音）中大量 token 与指令无关，不仅增加计算负担还引入噪声。通过动态筛选只保留与指令相关的 10%-25% token

### 损失函数 / 训练策略

四阶段训练：
1. 语音编码器预训练（文本到语音）
2. 文本-图像-语音联合训练（LLM + 投影器）
3. 长语音能力扩展
4. 语音生成器训练（流式文本+音频输出）

语音 token 压缩：经实验验证，将每 30 秒语音从 1500 token 压缩到 300 token 几乎不损失性能（TextVQA^S: 77.8% vs 76.8%），2 小时音频从 360,000 token 降至可处理范围。

## 实验关键数据

### 主实验

| 方法 | 参数量 | TextVQA | MME | MM-Vet | VideoMME | TextVQA^S | DocVQA^S | LibriSpeech↓ |
|------|--------|---------|-----|--------|----------|-----------|----------|-------------|
| Mini-Gemini | 8B | 71.9 | 1989 | 53.5 | - | - | - | - |
| LLaVA-OV | 7B | 65.4 | 1998 | 57.5 | 58.2 | - | - | - |
| Intern-VL2 | 8B | 77.4 | 2211 | 60.0 | 54.0 | - | - | - |
| VITA | 66B | - | 2097 | 41.6 | 59.2 | - | - | 8.1 |
| EMOVA | 14B | 82.0 | 2205 | 55.8 | - | - | - | 4.0 |
| **Lyra-Mini** | **3B** | 78.3 | 1884 | 51.2 | 55.0 | 73.4 | 74.8 | 2.1 |
| **Lyra-Base** | **9B** | 82.6 | 2335 | **63.5** | 62.8 | **80.0** | **85.5** | **2.0** |
| **Lyra-Pro** | **74B** | **83.5** | **2485** | **71.4** | **69.9** | **81.0** | **89.4** | **1.8** |

Lyra-Base 在图像-语音任务上相比最佳全模态模型提升约 9%，在语音-文本上提升约 2%。

### 消融实验

| 组件 | TextVQA (S+I) | TextVQA (T+I) | MM-Vet (S+I) | MM-Vet (T+I) | LibriSpeech (S+T) |
|------|--------------|--------------|-------------|-------------|------------------|
| Baseline (无 LCMR) | - | 82.3 | - | 62.8 | - |
| $\mathcal{L}_{CE}$ only | 76.7 | 79.5 | 53.1 | 61.1 | 1.9 |
| $\mathcal{L}_{CE} + \lambda\mathcal{L}_{LCMR}$ | **77.8** | **80.1** | **58.1** | **62.6** | 2.0 |

LMME 效率提升：

| Token 数量 | Prefill Time (Baseline → LMME(4,0.7)) | 内存 (Baseline → LMME(4,0.7)) |
|:---:|:---:|:---:|
| $2^{13}$ | 0.65s → 0.37s | 30G → 18G |
| $2^{15}$ | 2.99s → 1.23s | 60G → 30G |
| $2^{16}$ | OOM → 3.05s | OOM → 46G |

### 关键发现

- LCMR 同时提升了语音-图像和文本-图像的性能，说明语音对齐训练对整体多模态能力有正向影响
- 仅用 LibriSpeech WER 评估语音模态是不充分的：$\mathcal{L}_{CE}$ 下语音-文本 WER 几乎不变（1.9 vs 2.0），但语音-视觉差距巨大
- LMME 最终只保留 10%-25% 的多模态 token，训练加速超过 50%
- 语音 token 压缩到 300 是性能-效率的最佳平衡点

## 亮点与洞察

1. **以语音为中心的评估视角**：揭示了现有全模态模型评估的盲区——语音-文本性能好不代表语音-视觉性能好
2. **DTW 用于跨模态对齐**：巧妙利用动态时间规整解决语音和文本 token 长度不一致的问题
3. **三模态 token 压缩的统一框架**：LMME 对图像、视频、语音 token 统一适用，显著降低长上下文场景的资源需求
4. **首个长语音 SFT 数据集**：12K 长语音样本（数分钟到 2 小时），填补了长语音理解训练数据的空白

## 局限与展望

1. 依赖 Qwen2-VL 作为视觉基座，视觉能力的天花板受限于此
2. DTW 对齐的计算复杂度为 $O(L \times S)$，对长语音可能成为瓶颈
3. 长语音 SFT 数据集仅 12K 样本，规模仍然有限
4. 语音生成质量未做定量评估（如 MOS 分数）
5. 仅支持英语语音，多语言扩展性未验证

## 相关工作与启发

- Qwen2-VL 的动态分辨率视觉 token 处理为 Lyra 提供了强大的视觉基座
- Whisper-large-v3 作为语音编码器的限制（30 秒）催生了长语音处理的创新方案
- FastV 的视觉 token 裁剪思想被 LMME 扩展到多模态场景
- LLaVA-NeXT 的高分辨率图像分割策略被借鉴用于长语音分段处理

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 语音中心的全模态框架设计和评估视角有新意，DTW 跨模态对齐巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖视觉-语言、视觉-语音、语音-语言全面比较，消融详尽
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机论述有说服力
- **价值**: ⭐⭐⭐⭐⭐ — 填补了语音中心全模态 MLLM 的空白，开源代码和数据集对社区有重要贡献

<!-- RELATED:START -->

## 相关论文

- [Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)
- [VGGSounder: Audio-Visual Evaluations for Foundation Models](vggsounder_audio-visual_evaluations_for_foundation_models.md)
- [MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [Latent Swap Joint Diffusion for 2D Long-Form Latent Generation](latent_swap_joint_diffusion_for_2d_long-form_latent_generation.md)

<!-- RELATED:END -->
