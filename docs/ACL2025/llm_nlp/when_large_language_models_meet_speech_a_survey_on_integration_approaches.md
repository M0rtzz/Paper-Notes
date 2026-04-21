---
title: >-
  [论文解读] When Large Language Models Meet Speech: A Survey on Integration Approaches
description: >-
  [ACL 2025][LLM/NLP][speech-LLM integration] 系统综述语音与大语言模型的集成方法，将现有工作分为文本级、隐表示级、音频token级三大类，覆盖 ASR/S2TT/S2ST/TTS 等应用场景，并给出各方法的优劣对比与未来挑战。
tags:
  - ACL 2025
  - LLM/NLP
  - speech-LLM integration
  - text-based integration
  - latent-representation
  - audio tokens
  - speech language model
  - 多模态
---

# When Large Language Models Meet Speech: A Survey on Integration Approaches

**会议**: ACL 2025  
**arXiv**: [2502.19548](https://arxiv.org/abs/2502.19548)  
**代码**: 无（综述论文）  
**领域**: LLM/NLP  
**关键词**: speech-LLM integration, text-based integration, latent-representation, audio tokens, speech language model, multimodal LLM

## 一句话总结
系统综述语音与大语言模型的集成方法，将现有工作分为文本级、隐表示级、音频token级三大类，覆盖 ASR/S2TT/S2ST/TTS 等应用场景，并给出各方法的优劣对比与未来挑战。

## 研究背景与动机

**领域现状**：LLM 在文本任务上取得巨大成功，研究者开始探索将其能力扩展到语音模态，涌现出大量 Speech-LLM 集成工作（AudioGPT、SALMONN、Qwen-Audio、Moshi 等）。

**现有痛点**：已有综述覆盖了语音语言模型（Peng et al., 2024）和音频语言模型（Latif et al., 2023），但缺乏专门针对**语音与 LLM 集成方法**的系统分类——研究者难以快速了解不同集成范式的权衡。

**核心矛盾**：语音是连续信号，LLM 天然处理离散 token；如何弥合这一模态差距是所有方法的核心挑战，且不同集成策略在精度、延迟、可解释性间存在不同权衡。

**本文目标** 提供一个清晰的分类框架，帮助研究者理解三种集成范式（文本级/隐表示级/音频token级）的机制、适用场景和优缺点。

**切入角度**：从"集成方法"而非"应用任务"出发组织文献，并在统一框架下对比各方法在 ASR、S2TT、S2ST、TTS 等任务上的性能。

**核心 idea**：token化方法影响 LLM 性能，语音-LLM 的集成方式不应局限于离散 token 化，三种范式各有其适用场景和发展空间。

## 方法详解

### 整体框架
将 Speech-LLM 集成方法分为三大类：(a) 文本级集成——LLM 处理文本，配合 ASR/TTS 管道；(b) 隐表示级集成——语音编码器输出的连续向量直接送入 LLM；(c) 音频token级集成——语音被离散化为语义/声学 token 作为 LLM 的输入/输出。每类下进一步细分具体策略。

### 关键设计 1：文本级集成（Text-based Integration）
- **功能**：将语音先转为文本再交给 LLM 处理，包括级联集成、LLM 重打分（Rescoring）和生成式纠错（GER）三种子方法。
- **核心思路**：级联方式最直接（ASR→LLM→TTS）；重打分用 LLM 语言概率对 N-best 假设列表重新排序，公式为 $Y^* = \arg\max_{Y_i} [(1-\lambda)\log p_{AM}(Y_i|X) + \lambda \log p_{LLM}(Y_i)]$；GER 更进一步，让 LLM 根据 N-best 假设直接生成更好的转录结果（H2T 范式）。
- **设计动机**：保留 LLM 原生文本处理能力，实现简单、可解释性强。
  但存在误差传播和信息丢失（韵律、情感等非文本信息丢失）的固有缺陷。
  GER 相比 Rescoring 能突破假设列表的上限，生成质量可能优于所有候选。
  Lin et al. (2024) 进一步引入 MoE 架构让不同专家处理不同类型的生成错误。

### 关键设计 2：隐表示级集成（Latent-representation-based Integration）
- **功能**：用语音编码器（HuBERT、Whisper encoder 等）提取连续隐表示，经模态适配模块后直接灌入 LLM 的嵌入空间，绕过文本中间步。
- **核心思路**：关键在模态适配（Modality Adaptation）——解决语音帧率（50-100 fps）远高于文本 token 序列长度的问题。三种主流适配策略：① 卷积下采样（简单堆叠/卷积压缩）；② CTC 压缩（基于 CTC 预测去除空白帧或合并重复帧）；③ Q-Former（可学习的固定长度查询向量，通过交叉注意力提取信息）。对比实验显示 Q-Former > 卷积下采样 > CTC 压缩。
- **设计动机**：避免文本中间表示的信息损失，实现更深度的模态融合。
  但代价是丧失可解释性，且训练成本更高（需同时处理编码器和 LLM）。
  冻结部分模块 + LoRA 微调是降低成本的主流策略。
  Wu et al. (2023) 提出两阶段训练，先训练编码器再启动 PEFT，确保更稳定的梯度流。

### 关键设计 3：音频token级集成（Audio-token-based Integration）
- **功能**：将语音离散化为 token 序列，与文本 token 统一处理。分为语义 token（S3M + k-means 聚类）、声学 token（神经音频编解码器如 EnCodec）和两者结合三种子方法。
- **核心思路**：语义 token 捕获语音的语言内容，声学 token 保留高保真音频质量。先预测语义 token 再预测声学 token 的两阶段架构（AudioLM）是典型范式。Moshi 引入 Temporal Transformer + Depth Transformer 的层级自回归架构，实现全双工语音对话。
- **设计动机**：音频 token 让语音和文本在 token 层面统一，LLM 可以自然地生成语音输出（而隐表示方法通常只能输入语音，不能输出）；但语义 token 缺乏音质，声学 token 缺乏语义信息，如何结合两者仍是开放问题。

## 实验关键数据

### 主实验 — ASR 任务性能对比（LibriSpeech test-clean / test-other WER↓）

| 模型 | 集成方法 | test-clean | test-other |
|------|---------|-----------|-----------|
| Whisper-large-v2 | 非 LLM 基线 | 2.7 | 5.2 |
| HyPoradise | 文本级 → GER | 1.8 | 3.7 |
| Seed-ASR | 隐表示 → 卷积下采样 | **1.5** | **2.8** |
| SALMONN | 隐表示 → Q-Former | 2.1 | 4.9 |
| Qwen2-Audio | 隐表示 → 其他适配 | 1.6 | 3.6 |
| SpeechGPT-Gen | 音频token → 语义token | 2.4 | — |

### S2TT 任务性能对比（CoVoST2 de-en / zh-en BLEU↑）

| 模型 | 集成方法 | de→en | zh→en |
|------|---------|-------|-------|
| Whisper-large-v2 | 非 LLM 基线 | 36.3 | 18.0 |
| GenTranslate-V2 | 文本级 → GER | **40.6** | **23.3** |
| LLaST | 隐表示 → 其他适配 | 41.2 | 24.8 |
| AudioPaLM | 音频token → 语义+声学 | 43.4 | 25.5 |

### 三类方法优劣对比（综述总结）

| 维度 | 文本级 | 隐表示级 | 音频token级 |
|------|--------|---------|-----------|
| 集成深度 | 浅 | **最深** | 中 |
| 可解释性 | **最好** | 最差 | 中 |
| 语音生成能力 | ✓（需 TTS） | ✗（通常不能） | **✓（原生）** |
| 实时处理能力 | 差（多步延迟） | 好 | 好 |
| 实现难度 | **最简单** | 中 | 中 |

### 关键发现
- **隐表示级方法在 ASR 上整体表现最好**：Seed-ASR (WER 1.5/2.8) 和 Qwen2-Audio (WER 1.6/3.6) 超越文本级方法
- **文本级 GER 仍有独特价值**：HyPoradise 在无需额外语音编码器的前提下，将 WER 从 2.7 降到 1.8
- **音频token方法在生成任务（TTS/S2ST）上更有优势**：AudioPaLM 在 S2ST 上相比传统级联系统大幅提升（de-en ASR-BLEU 37.2 vs 33.6）
- **Q-Former 优于卷积下采样优于 CTC 压缩**——模态适配策略对隐表示方法性能影响显著
- **直接对比困难**：各方法使用的骨干 LLM、训练数据和协议差异大，缺乏统一基准
- **训练策略影响显著**：Pham et al. (2024) 证明 LoRA 微调 LLM 显著提升性能，编码器全量微调效果最佳但部分微调更具性价比
- **Spirit-LM 和 Moshi 代表音频token方法的前沿**：Topic-StoryCloze 上 Moshi 达到 83.6，Spirit-LM 达到 82.9，显著优于 TWIST 系列，证明语义+声学 token 融合的有效性

## 亮点与洞察
- **三分法分类框架清晰实用**：从集成方法（而非任务）出发组织文献，填补了现有综述的空白，为新研究者提供了高效的入门路径。
- **"集成深度 vs 可解释性"的权衡洞察**直击核心——越深的集成性能越好但越不可控，这是 Speech-LLM 领域的根本张力。
- **模态适配策略的细粒度对比**（卷积/CTC/Q-Former）为具体方法选择提供了量化依据。
- **音频token级方法是最有潜力的统一范式**：能同时处理输入和输出语音，但目前成熟度不如隐表示方法。
- **训练策略分析**：综述系统总结了冻结/微调各模块的策略权衡——LoRA 微调 LLM + 全量微调编码器是当前最佳实践，两阶段训练（先训练编码器再启动 PEFT）可提升稳定性。
- **六大挑战的系统梳理**涵盖文本信息丢失、语音-文本对齐、语义+声学 token 融合、公平对比缺失、多语言支持不足和实时处理延迟，为后续研究提供了明确路线图。
- **语音表示背景知识全面**：从滤波器组到自监督模型（HuBERT、wav2vec 2.0）再到语义/声学 token，为读者提供了完整的语音表示知识体系。
- **实时处理挑战**：综述明确指出 LLM 的大尺寸导致延迟增加，而 Moshi 和 Llama-Omni 等实时模型展示了深度集成降低延迟的可能性。

## 局限性
- 作为综述论文，没有做原创实验验证，定量对比受限于各原始论文不同的实验设置
- 主要覆盖英语相关工作，多语言场景讨论有限
- 快速发展的领域，可能遗漏最新进展（如 GPT-4o 的语音能力未能充分讨论）
- 缺乏统一基准下的公平对比实验（作者也承认这是重要未来方向）
- LLM 的定义采用宽泛标准（>10B 参数），也纳入了部分较小模型的研究，边界不够清晰
- 对计算成本的讨论停留在定性层面，缺少实际 FLOPs/延迟的数值对比
- 未详细讨论隐私和安全问题（如语音深伪、说话人隐私保护等）
- 语音情感/副语言信息的处理讨论不够深入，这是语音相比文本的重要额外信息源

## 相关工作与启发
- **vs Peng et al. (2024) 语音语言模型综述**：Peng 聚焦"语音语言模型"本身的建模范式，本文聚焦"语音如何与 LLM 集成"——前者关注模型架构，后者关注集成接口，互补视角
- **vs Latif et al. (2023) 音频语言模型综述**：Latif 覆盖更广的音频模态（包括音乐、环境音），本文专注语音-LLM 集成的分类学，分析更深入
- **vs Ghosh et al. (2024a) 多模态语言模型综述**：多模态综述覆盖视觉+语音+文本的全景，本文在语音维度做了更细粒度的分类（三级方法+子类别），对语音领域研究者更有针对性
- **启发**：对于新入门 Speech-LLM 的研究者，可根据任务需求快速定位——仅需语音理解选隐表示级，需要语音生成选音频token级，资源有限或需要可调试性选文本级。Moshi（Défossez et al., 2024）的 Temporal+Depth Transformer 架构是实时全双工对话的重要参考。

## 评分
- 新颖性: ⭐⭐⭐⭐ 分类框架新颖实用，但作为综述无原创方法
- 实验充分度: ⭐⭐⭐⭐ 汇总了大量定量结果进行横向对比，但受限于非统一实验设置
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，分类学图表出色，语言精练，背景知识介绍全面
- 价值: ⭐⭐⭐⭐⭐ 填补 Speech-LLM 集成方法综述空白，对领域研究者有很高参考价值，尤其适合新入门者快速了解领域全貌

<!-- RELATED:START -->

## 相关论文

- [Recent Advances in Speech Language Models: A Survey](recent_advances_in_speech_language_models_a_survey.md)
- [Large Language Models in Bioinformatics: A Survey](large_language_models_in_bioinformatics_a_survey.md)
- [Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models](locateandfocus_enhancing_terminology_translation_in_speech.md)
- [Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)
- [Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)

<!-- RELATED:END -->
