---
title: >-
  [论文解读] Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis
description: >-
  [ACL 2025][语音][对话语音合成] 提出 Chain-Talker，通过三阶段链式建模（情感理解→语义理解→共情渲染）实现可解释的共情对话语音合成，并开发 CSS-EmCap 自动标注管道为对话语音生成情感描述。
tags:
  - ACL 2025
  - 语音
  - 对话语音合成
  - 共情语音
  - 链式建模
  - 情感描述
  - 语义编码
---

# Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis

**会议**: ACL 2025  
**arXiv**: [2505.12597](https://arxiv.org/abs/2505.12597)  
**代码**: https://github.com/AI-S2-Lab/Chain-Talker (有)  
**领域**: 语音  
**关键词**: 对话语音合成, 共情语音, 链式建模, 情感描述, 语义编码

## 一句话总结

提出 Chain-Talker，通过三阶段链式建模（情感理解→语义理解→共情渲染）实现可解释的共情对话语音合成，并开发 CSS-EmCap 自动标注管道为对话语音生成情感描述。

## 研究背景与动机

**领域现状**: 对话语音合成（CSS）旨在根据用户-智能体交互的情感和风格上下文，生成与对话情境匹配的语音。近年来基于 GPT 的 CSS 模型（如 GPT-Talker）通过直接预测语音 token 序列显著提升了自然度和表达力。

**现有痛点**: 当前生成式 CSS 模型存在两个可解释性问题：(1) 语音生成未充分理解对话情感，难以实现真正的共情；(2) 通用离散语音编码包含过多冗余信息，混合了语义和声学信息，表达能力受限。

**核心矛盾**: 直接从对话上下文预测语音 token 的端到端方式缺乏可解释性——模型既没有显式地理解情感变化，也没有高效地分离语义与声学信息。

**本文目标**: 如何让 CSS 系统像人类一样逐步理解对话情感、提取语义信息、最终生成共情语音，同时如何为对话语音自动生成高质量的情感描述标注。

**切入角度**: 借鉴人类链式思维（Chain-of-Thought），将 CSS 分解为三个阶段：先理解情感、再理解语义、最后共情渲染。同时利用 LLM 构建自动的对话感知情感描述管道。

**核心 idea**: 模拟人类认知的三阶段链式建模——先感知情感、再编码语义、后渲染语音——实现可解释的共情对话语音合成。

## 方法详解

### 整体框架

Chain-Talker 由两个主要组件构成：**EmGPT** 和 **Synthesizer**。EmGPT 负责情感理解和语义理解，基于自回归 GPT 架构；Synthesizer 负责共情语音渲染，基于 OT-CFM（最优传输条件流匹配）模型。

输入序列 $\mathcal{Q} = (\langle BOS \rangle, \mathcal{H}, \mathcal{C}, \langle EOS \rangle)$，其中 $\mathcal{H}$ 为对话历史，$\mathcal{C}$ 为当前要合成的语句。

### 关键设计

**模块1: 统一上下文 Token 化（Unified Context Tokenization）**

- **功能**: 将多模态对话信息统一编码
- **核心思路**: 按说话人信息、语音、文本内容、情感描述的顺序交替拼接用户和智能体的发言。文本用 BPE 编码为 $T_n^t$，情感描述编码为 $T_n^d$，说话人向量用预训练声纹模型提取 $T_n^p$，语音用监督式 ASR 模型（含 VQ）编码为 $T_n^a$
- **设计动机**: 让模型先理解上下文再预测情感，然后生成对应语音，统一多模态信息到同一序列空间

**模块2: 情感理解（Emotion Understanding）**

- **功能**: 根据对话上下文预测当前语句的情感描述 token
- **核心思路**: 利用 EmGPT 自回归预测情感描述 $T_N^d$：$p(T_{N,:}^d | \Re_{1 \to N-1}, T_{N,:}^p, T_{N,:}^t; \Theta) = \prod_{j=0}^{D} p(T_{N,j}^d | T_{N,<j}^d, \Re_{1 \to N-1}, T_{N,:}^p, T_{N,:}^t; \Theta)$
- **设计动机**: 显式理解对话中的情感变化，为后续语音生成提供情感指导

**模块3: 语义理解（Semantic Understanding）**

- **功能**: 在情感理解基础上继续预测纯语义语音编码
- **核心思路**: EmGPT 利用预测的情感描述 $T_N^d$ 与上下文信息进一步预测语义编码 $T_N^a$：$p(T_{N,:}^a | \Re_{1 \to N-1}, T_N^p, T_N^t, T_N^d; \Theta)$
- **设计动机**: 使用监督式 ASR 模型产生的纯语义编码，避免通用离散编码中的冗余声学信息

**模块4: 共情渲染（Empathetic Rendering）**

- **功能**: 利用情感描述和语义编码合成最终的共情语音
- **核心思路**: 采用 OT-CFM 作为 backbone 预测 Mel 频谱图，使用 HIFI-GAN 合成波形。OT-CFM 同时利用情感描述 $U_N^d$、说话人信息 $U_{agent}^p$、语义编码 $T_N^a$ 和掩码 Mel 频谱图 $U_{agent}^m$ 来预测向量场：$\frac{d\phi_t(X)}{dt} = \nu_t(\phi_t(X), t | U_{agent}^p, U_N^d, T_N^a, U_{agent}^m)$
- **设计动机**: 利用情感描述在解码阶段指导情感和风格渲染，而非直接从语音 token 解码

### 损失函数/训练策略

**训练损失**:
- EmGPT 训练分为两个损失：$\mathcal{L}_{caption}$（情感描述 token 的交叉熵损失）和 $\mathcal{L}_{speech}$（语义编码的交叉熵损失）
- Synthesizer 使用 OT-CFM 损失：$\mathcal{L}_{OT\text{-}CFM} = \mathbb{E}_{t,X_0,X_1}[\|\omega_t(\phi_t^{OT}(X_0,X_1)|X_1) - \nu_t(\phi_t^{OT}(X_0,X_1)|\theta)\|]$

**多阶段训练**:
- **第一阶段**: 使用大规模单句 TTS 数据训练（基于 CosyVoice-300M-25Hz，约 17 万小时语音数据）
- **第二阶段**: 使用对话数据微调，学习根据对话上下文推断情感描述并预测语义编码
- Synthesizer 可单独用单句模式训练

**CSS-EmCap 管道**: 
- 多层级属性提取：句子级风格因子（性别、音高、能量、语速）+ 对话级情感分类
- 两步生成：基于上下文和属性生成基础描述，再通过同义词替换和情感强度变化扩展丰富

## 实验关键数据

### 主实验

使用 NCSSD、DailyTalk、MultiDialog 三个数据集（共约 384 小时）。

| 方法 | DMOS-N ↑ | DMOS-E ↑ | ACCm ↑ | DDTW ↓ | SSIM ↑ |
|------|---------|---------|--------|--------|--------|
| Ground Truth | 4.467 | 4.571 | - | - | - |
| CCATTS | 3.423 | 3.469 | 0.462 | 67.851 | 0.765 |
| GPT-Talker | 3.962 | 3.913 | 0.562 | 44.625 | 0.814 |
| GPT-Talker_c | 4.045 | 4.102 | 0.589 | 40.374 | 0.829 |
| **Chain-Talker** | **4.147** | **4.239** | **0.612** | **38.784** | **0.862** |

CSS-EmCap 标注质量：

| 方法 | DMOS-C ↑ | SIM_R ↑ | SIM_G ↑ | DIS-1 ↑ | DIS-2 ↑ |
|------|---------|---------|---------|---------|---------|
| Qwen2-Audio | 4.212 | 0.431 | 0.534 | 0.086 | 0.174 |
| SECap | 4.268 | 0.475 | 0.617 | 0.081 | 0.186 |
| **CSS-EmCap** | **4.462** | **0.568** | **0.694** | **0.106** | **0.296** |

### 消融实验

| 配置 | DMOS-N ↑ | DMOS-E ↑ | ACCm ↑ | DDTW ↓ | SSIM ↑ |
|------|---------|---------|--------|--------|--------|
| Chain-Talker | 4.147 | 4.239 | 0.612 | 38.784 | 0.862 |
| w/o context | 3.982 | 3.984 | 0.564 | 43.589 | 0.847 |
| w/o captions | 4.037 | 4.084 | 0.571 | 43.479 | 0.836 |
| w/o $\mathcal{L}^{caption}$ | 3.947 | 3.956 | 0.568 | 45.764 | 0.829 |
| w/o First-Stage | 3.756 | 3.789 | 0.517 | 52.640 | 0.793 |

### 关键发现

1. Chain-Talker 在自然度 MOS 上比最强 baseline 高 0.102，表达力 MOS 高 0.112
2. 移除对话历史（w/o context）导致 DMOS-E 下降 0.255，证明上下文建模的重要性
3. 移除情感描述损失（w/o $\mathcal{L}^{caption}$）导致 DMOS-N 下降 0.2、DMOS-E 下降 0.283
4. 移除第一阶段预训练导致所有指标大幅下降，证明大规模预训练的必要性
5. 对话轮数 $N$=3 时性能最优，约 200 epoch 达到峰值

## 亮点与洞察

- **链式建模思想**的引入非常优雅，将复杂的 CSS 任务分解为可解释的三步认知链，符合人类"先理解情感→再理解内容→最后表达"的直觉
- **CSS-EmCap 管道**的设计思路值得借鉴：先提取结构化属性，再用 LLM 生成自然语言描述，最后验证，形成闭环
- 使用**监督式语义编码**替代通用离散编码是一个关键设计选择——去除声学冗余信息让语义理解更纯粹
- 情感描述（自然语言）优于传统情感标签（DMOS-C 4.462 > GT 的 4.327），说明自然语言在控制语音风格上更具表达力

## 局限与展望

1. **推理延迟**: 平均 2.5 秒的响应时间尚未达到实时交互要求，流式推理是重要方向
2. **数据规模**: 对话数据仅 384 小时且以年轻说话人为主，对儿童和老年人风格覆盖不足
3. **安全风险**: 零样本语音合成能力可能被滥用于语音伪造
4. 情感描述生成依赖 Gemini API，增加了对外部大模型的依赖

## 相关工作与启发

- **GPT-Talker** 开创了基于 GPT 的 CSS 范式，但缺乏显式情感理解
- **CosyVoice** 提供了监督式语义 token 和 OT-CFM 合成器的基础架构
- **链式思维（CoT）**在对话任务中的成功应用启发了 CSS 中的链式建模
- **Qwen2-Audio/SECap** 的语音理解能力为情感描述生成提供了对比基准

## 评分

- **新颖性**: ⭐⭐⭐⭐ (链式建模在CSS中的首次应用，思路清晰)
- **实验充分度**: ⭐⭐⭐⭐ (三个数据集+充分消融+可视化分析)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- **价值**: ⭐⭐⭐⭐ (CSS-EmCap管道和链式设计对社区有参考价值)

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Autoregressive Speech Synthesis without Vector Quantization](autoregressive_speech_synthesis_without_vq.md)
- [\[ACL 2025\] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models](speechiq_speechagentic_intelligence_quotient_across_cognitive.md)
- [\[ACL 2025\] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models](does_your_voice_assistant_remember_analyzing_conversational_context_recall_and_u.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)

<!-- RELATED:END -->
