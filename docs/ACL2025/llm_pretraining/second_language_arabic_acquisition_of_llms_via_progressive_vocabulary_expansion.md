---
title: >-
  [论文解读] Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion
description: >-
  [ACL 2025][Arabic LLM] 受人类第二语言习得启发，提出渐进式词表扩展（Progressive Vocabulary Expansion）方法，通过分阶段指数增长地扩展阿拉伯语子词到 LLaMA2 词表中，在保留原模型英语知识的同时高效适配阿拉伯语，构建出 AraLLaMA 7B/13B 模型。
tags:
  - ACL 2025
  - Arabic LLM
  - vocabulary expansion
  - language adaptation
  - BPE
  - tokenization
  - continual pre-training
---

# Second Language (Arabic) Acquisition of LLMs via Progressive Vocabulary Expansion

**会议**: ACL 2025  
**arXiv**: [2412.12310](https://arxiv.org/abs/2412.12310)  
**代码**: [FreedomIntelligence/AraLLaMa](https://github.com/FreedomIntelligence/AraLLaMa)  
**领域**: LLM/NLP  
**关键词**: Arabic LLM, vocabulary expansion, language adaptation, BPE, tokenization, continual pre-training

## 一句话总结
受人类第二语言习得启发，提出渐进式词表扩展（Progressive Vocabulary Expansion）方法，通过分阶段指数增长地扩展阿拉伯语子词到 LLaMA2 词表中，在保留原模型英语知识的同时高效适配阿拉伯语，构建出 AraLLaMA 7B/13B 模型。

## 研究背景与动机
**领域现状**: 当前主流 LLM（GPT-4、LLaMA 等）主要针对英语和中文优化，阿拉伯语作为全球第五大语言（4.2亿使用者），在 LLM 领域进展缓慢，现有阿拉伯语模型（Jais、AceGPT）与 GPT-4 差距显著。

**现有痛点**: 英语为中心的 LLM 使用原始词表处理阿拉伯语时，会将阿拉伯词拆成字母级别的 token 序列，导致 subword fertility 高达 5.38（一个单词平均需 5.38 个 token），解码速度极慢，如 AceGPT 在阿拉伯语上解码效率远低于英语。

**核心矛盾**: 直接一次性扩展大量阿拉伯语 token 到词表会引入大量 OOV（out-of-vocabulary）词，破坏模型已学习的语言表示空间，需要海量预训练数据才能恢复模型能力，形成"扩展 vs 保持"的两难。

**本文解决**: 提出渐进式词表扩展（PVE），分 16 个阶段逐步向词表添加 12,800 个阿拉伯语子词，每阶段 OOV 比例可控，模型可以平滑适应新增 token。

**切入角度**: 从认知科学出发，类比人类第二语言习得（SLA）过程——人类学习第二语言时词汇量是渐进增长的（参照 CEFR 语言能力框架 A1→C2 各级所需词汇量），而非一次性掌握全部词汇。

**核心 idea**: "渐进式 > 一次性"——将 BPE 算法改造为增量式（I-BPE），在训练过程中动态扩展词表，每阶段仅添加少量新 token 再充分训练，使模型在吸收新语言元素的同时保留已有知识。

## 方法详解

### 整体框架
- **功能**: 将英语为主的 LLaMA2 模型适配为阿拉伯语 LLM（AraLLaMA），涵盖词表扩展→分阶段继续预训练→指令微调三个环节。
- **为什么**: 语言适配（language adaptation）是低资源语言利用现有强模型的经济路线，避免从头训练的巨大计算开销，同时借助跨语言迁移保持通用能力。
- **怎么做**: 基于 LLaMA2-7B/13B 初始化，使用 I-BPE 算法分 16 阶段向词表添加 12,800 个阿拉伯语子词，每阶段处理 30B token（共 480B token），阿拉伯语数据比例从 30% 逐步提升到 90%（余弦退火调度），数学和编程数据保持 5% 恒定。完成预训练后进行指令微调（使用 ALAN 数据 + AceGPT 数据集）。

### 关键设计
1. **增量式 BPE（I-BPE）算法**

    - **功能**: 改造标准 BPE 使其在训练过程中动态扩展词表，而非预先构建完整静态词表。
    - **为什么**: 标准 BPE 构建完整词表后再训练，无法处理语言适配中的词表演化需求；一次性添加大量新 token 会导致训练不稳定和灾难性遗忘。
    - **怎么做**: 每个阶段先用频率统计扩展词表到预定大小 $s_i$，再调整新增 token 在训练语料中的比例 $r_i$，训练至收敛后进入下一阶段。新 token 的 embedding 初始化为其组成子词 embedding 的平均值，保持语义关系。

2. **指数扩展策略（Exponential Expansion）**

    - **功能**: 每阶段新增 token 数量按 $\{0, 1, 2, \ldots, 2^{T-2}\}$ 指数递增（对比均匀扩展每阶段固定增加 K 个）。
    - **为什么**: 均匀扩展在早期引入过多 token 导致压缩比骤变和表示空间剧烈调整；指数扩展模拟人类渐进学词过程，早期少量添加让模型稳定适应，后期快速丰富词表。
    - **怎么做**: 16 个阶段 $\log_2(12800)$ 步指数增长，每阶段 token 数翻倍，实现压缩比平滑提升；最终序列长度相比原始 LLaMA 减少约 3 倍。

3. **ALAN 指令微调数据生成**

    - **功能**: 提出 ALAN（Arabic Language Acquisition for LLMs）方法，围绕 127 个阿拉伯文化/科学/工程核心主题，使用 GPT-4 生成 73.3 万条指令微调数据。
    - **为什么**: 阿拉伯语高质量指令数据稀缺，需要系统化生成覆盖广泛领域的训练数据。
    - **怎么做**: 将 127 个主题分解为领域→子领域→学科层级结构，为每个学科编写包含知识点的课程大纲（共 11,430 个 subject、244,812 个知识点），组合同一/不同课程的知识点生成多选/开放/编程三类问答。

## 实验关键数据

### 表1: Tokenizer 评估对比

| 模型 | 总 Token 数 | Subword Fertility↓ | Word Integrity↑ | Rényi Efficiency |
|------|------------|--------------------|--------------------|------------------|
| LLaMA2 (AceGPT) | 210M | 5.38 | 1.8% | 0.77 |
| Bloomz | 80.6M | 2.07 | 31.8% | 0.77 |
| Jais | 75.1M | 1.93 | 39.0% | 0.73 |
| **AraLLaMA** | **66.6M** | **1.71** | **63.2%** | 0.75 |

### 表2: Chat 模型阿拉伯语基准评估（零样本）

| 模型 | MMLU-ar↑ | ArabicMMLU↑ | ACVA-all↑ | BoolQ-ar↑ | ARC-C-ar↑ | 英语 Avg↑ |
|------|---------|------------|----------|----------|----------|----------|
| AceGPT-7B-chat | 30.69 | 36.31 | 53.07 | 60.70 | 38.05 | 54.36 |
| Mistral-7B-Instruct | 27.93 | 41.44 | 63.47 | 60.18 | 35.67 | 78.85 |
| **AraLLaMA-7B-chat** | **45.77** | **56.62** | **70.86** | **72.45** | **60.49** | 73.96 |
| AceGPT-13B-chat | 35.59 | 52.61 | 70.21 | 66.85 | 44.20 | 52.88 |
| Jais-30B-chat-v3 | 35.68 | 62.36 | 73.66 | 76.30 | 51.02 | 82.43 |
| **AraLLaMA-13B-chat** | **47.33** | **61.70** | **76.37** | 69.33 | **63.99** | 82.24 |

### 表3: 渐进式词表扩展消融实验（TinyLLaMA 1B）

| 方法 | ArabicMMLU Avg↑ | Arabic Vicuna-80↑ |
|------|----------------|-------------------|
| TinyLLaMA (baseline) | 36.5 | 21.30% |
| + 一次性词表扩展 (VE) | 38.5 | 22.61% (+1.31) |
| + **渐进式词表扩展 (PVE)** | **40.7** | **29.18% (+7.88)** |

### 关键发现
- AraLLaMA-7B 在同等规模下阿拉伯语任务全面超越 AceGPT、Mistral 等竞品，MMLU-ar 高出 AceGPT-7B 约 15 个百分点
- AraLLaMA-13B 在多项阿拉伯语基准上超越参数量 2 倍以上的 Jais-30B
- 渐进式扩展（PVE）在消融实验中比一次性扩展（VE）在 Vicuna-80 上提升 6.57 个百分点，证明渐进策略显著优于直接扩展
- Tokenizer 效率：AraLLaMA 阿拉伯语生成速度达 20.37 词/秒，是 LLaMA2（4.55 词/秒）的 4.5 倍
- 英语能力基本保持：SFT 后在英语 MMLU 上甚至高于同规模基线

## 亮点与洞察
- **认知科学启发**: 将人类第二语言习得中的渐进词汇学习类比到 LLM 语言适配，提供了直觉清晰的方法论框架
- **指数扩展 vs 均匀扩展**: 通过对比分析揭示指数扩展在训练稳定性和 OOV 比例控制上的优势，设计选择有理论和实验双重支撑
- **完整开源生态**: 模型权重、数据处理流水线、预训练/微调数据全部开源，且兼容 LLaMA 架构可直接集成
- **实用价值显著**: 4.5 倍阿拉伯语解码加速对实际部署有直接意义

## 局限性
- 仅在阿拉伯语上验证了方法有效性，尚未测试其他低资源语言（如印地语、斯瓦希里语）的泛化性
- 模型未经阿拉伯语母语者系统评估，实际使用中的流畅度和文化恰当性尚需进一步验证
- 训练使用了 2,368 块 Ascend 910A，资源需求较高，方法复现门槛不低
- 16 阶段分步训练相比端到端训练工程复杂度显著增加，超参选择（阶段数、每阶段 token 量、语言比例调度）较为 ad hoc

## 相关工作对比
- **vs AceGPT** (Huang et al., 2024): AceGPT 同样基于 LLaMA2 适配阿拉伯语，但使用原始词表导致 subword fertility 5.38、解码慢；AraLLaMA 通过词表扩展将 fertility 降至 1.71，解码快 4.5 倍且性能更优。AceGPT 可视为 AraLLaMA 的直接前身。
- **vs Jais** (Sengupta et al., 2023): Jais 是从头预训练的阿拉伯-英语双语模型（最大 30B），拥有更好的阿拉伯语 tokenizer 设计；但 AraLLaMA-13B 在多项基准上超越 Jais-30B，说明语言适配路线在资源效率上有优势。
- **vs Chinese-LLaMA** (Cui et al., 2023): 中文 LLaMA 也做了词表扩展+继续预训练，但采用一次性扩展方式；AraLLaMA 的渐进策略在消融实验中证明优于一次性扩展，为语言适配研究提供了新范式。

## 评分
- 新颖性: ⭐⭐⭐ 渐进式词表扩展思路新颖且有认知科学理论支撑，但核心仍是 BPE+继续预训练的组合
- 实验充分度: ⭐⭐⭐⭐ 覆盖 tokenizer 评估、多数据集、多规模模型、消融实验、解码效率分析，较为全面
- 写作质量: ⭐⭐⭐⭐ 从 SLA 类比切入叙述流畅，方法动机清晰；图表丰富
- 价值: ⭐⭐⭐⭐ 对阿拉伯语及其他低资源语言的 LLM 适配有直接参考价值，完整开源增加了社区影响力

<!-- RELATED:START -->

## 相关论文

- [Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](tokalign_vocab_adaptation.md)
- [FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling](fr_spec_speculative_sampling.md)
- [Lossless Vocabulary Reduction for Auto-Regressive Language Models](../../ICLR2026/llm_pretraining/lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [Synthesizing Post-Training Data for LLMs through Multi-Agent Simulation](synthesizing_post-training_data_for_llms_through_multi-agent_simulation.md)

<!-- RELATED:END -->
