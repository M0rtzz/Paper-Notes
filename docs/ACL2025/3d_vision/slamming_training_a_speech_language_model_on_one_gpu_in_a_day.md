---
title: >-
  [论文解读] Slamming: Training a Speech Language Model on One GPU in a Day
description: >-
  [ACL 2025][3D视觉][Speech Language Model] 提出 Slam 训练配方，通过系统化的模型初始化、架构选择、合成数据、偏好优化等环节优化，在单张 A5000 GPU 上 24 小时内训练出性能媲美大规模 SLM 的语音语言模型。
tags:
  - ACL 2025
  - 3D视觉
  - Speech Language Model
  - 高效训练
  - 合成数据
  - DPO
  - 低资源训练
---

# Slamming: Training a Speech Language Model on One GPU in a Day

**会议**: ACL 2025  
**arXiv**: [2502.15814](https://arxiv.org/abs/2502.15814)  
**代码**: https://pages.cs.huji.ac.il/adiyoss-lab/slamming (有)  
**领域**: Audio/Speech  
**关键词**: Speech Language Model, 高效训练, 合成数据, DPO, 低资源训练

## 一句话总结
提出 Slam 训练配方，通过系统化的模型初始化、架构选择、合成数据、偏好优化等环节优化，在单张 A5000 GPU 上 24 小时内训练出性能媲美大规模 SLM 的语音语言模型。

## 研究背景与动机

**领域现状**：语音语言模型（SLM）近年取得了显著进展，但高质量 SLM 的训练通常需要大量计算资源——例如 Moshi 用了 700 万小时语音数据，SpiritLM 用了 56 万小时，TWIST-7B 需要 32×V100 训练。

**现有痛点**：高昂的计算需求将 SLM 研究限制在资金充足的大实验室，普通学术实验室难以参与基础性 SLM 研究（如新的 tokenization、高效声学建模等）。

**核心矛盾**：SLM 的 scaling law 研究（Cuervo & Marxer 2024）甚至给出了一个悲观预测——训练高质量 SLM 需要比文本 LM 多约 3 倍的数据。但这一预测是否可以被更好的训练策略打破？

**本文目标**：能否在单张学术级 GPU + 24 小时内训练出高质量 SLM？如何系统性地优化训练流程的每个环节？

**切入角度**：受文本领域 "Cramming"（单 GPU 一天训 BERT）的启发，对 SLM 训练流水线的每个组件进行系统消融研究。

**核心 idea**：通过精心选择模型家族/初始化（Qwen2.5 + TWIST init）、合成数据（TinyStories TTS）、DPO 偏好优化、超参数调优的组合配方，实现低资源高性能 SLM 训练。

## 方法详解

### 整体框架
Slam 是一个端到端的 SLM 训练配方（recipe），而非单一的架构创新。输入是原始语音，先通过 HuBERT 提取 25Hz 的语义 token（k-means 离散化），然后用 decoder-only transformer 做 next-token prediction。整体优化覆盖五个维度：模型选择与初始化、优化器与调度器、数据选择与合成、文本交织、DPO 偏好优化。

### 关键设计

1. **模型家族与 TWIST 初始化**:

    - 功能：比较多种预训练文本 LM（OPT、Pythia、SmolLM2、MobileLLM、Qwen2.5）作为 SLM 的初始化
    - 核心思路：将预训练文本 LM 的 token embedding 层替换为语音 token embedding，其余参数保留。TWIST 初始化利用文本 LM 学到的语义知识加速 SLM 收敛
    - 设计动机：不同模型家族从 TWIST 初始化中获益差异巨大。Qwen2.5-0.5B（实际参数 358M，因词表缩小）在相同计算预算下远超其他模型，即使不做 TWIST 初始化也表现最佳。这与 scaling law 预测的最优模型大小（66M）形成鲜明对比——选择更强的预训练模型比堆数据更有效

2. **优化器与学习率调度**:

    - 功能：在固定计算预算下寻找最优的 optimizer + scheduler 组合
    - 核心思路：比较 AdamW、AdaLomo、AdEMAMeix 三种优化器，以及 InverseSqrt 和 Cosine 两种调度器。最终发现 AdamW + Cosine decay 是最优组合
    - 设计动机：原始 TWIST 使用 InverseSqrt 调度器，但 Cosine decay 显著改善了 AdamW 的收敛。此外，移除 dropout（虽然不影响 wall time 但减少了有效梯度更新）、将 RoPE 的 $\theta$ 从默认值降到 10,000（因为 context length 比原始 LLM 短得多）、context length 从 512 扩大到 1024 都提升了性能

3. **合成数据（sTinyStories）**:

    - 功能：用单说话人 TTS 将 TinyStories 文本数据集合成为语音，加入训练
    - 核心思路：TinyStories 已被证明能增强文本 LM 的语义能力，合成语音版本（sTinyStories）可以在不需要大规模真实数据的情况下显著提升 SLM 的语义理解和生成能力
    - 设计动机：在 constrained compute 下，数据多样性（多口音、多说话风格）反而会伤害性能，因为模型没有足够资源去建模复杂声学。但语义丰富的合成数据则很有帮助——GenPPL 从 145.4 降到 88.3（Qwen-0.5B）

4. **DPO 偏好优化（合成 SWAG 数据）**:

    - 功能：在预训练后用 DPO 做少量偏好优化（仅 30 分钟）
    - 核心思路：将 SWAG 数据集（含正确/错误后续句对）用 Kokoro TTS（4种说话人，英/美口音）合成为 47k 个偏好对。用 off-policy DPO（$\beta=0.1$）训练，专门提升语义区分能力
    - 设计动机：DPO 30分钟即可显著提升所有指标（tSC 从 78.01→82.04，GenPPL 从 88.3→62.8）。分配更多时间给 DPO 反而不会带来更多收益。为缓解 DPO 导致的重复生成问题，使用 repetition penalty factor=1.1

### 训练策略
- **文本交织**：尝试了 speech-text interleaving（Whisper 对齐转录 + RedPajama 文本），但在当前计算预算下无益——因为交织增大了词表和参数量，导致训练步数减少（11k vs 18k），且只有 40% token 是语音
- **数据多样性 vs 合成数据**：多样数据（VoxPopuli、Tedlium 等）反而降低性能，但合成数据显著帮助，说明在低资源下语义质量比声学多样性更重要

### 损失函数 / 训练策略
- 预训练阶段：标准 next-token prediction（负对数似然）
- DPO 阶段：Direct Preference Optimization loss（$\beta=0.1$）
- 峰值学习率 $1\times10^{-3}$，warmup 1%，梯度归一化 norm=0.5，bfloat16 + FlashAttention2 + data packing

## 实验关键数据

### 主实验

| 模型 | 计算资源 | 参数量 | sBLIMP↑ | sSC↑ | tSC↑ | GenPPL↓ | GPTScore↑ |
|------|---------|--------|---------|------|------|---------|-----------|
| TWIST-1.3B | 32×V100 | 1B | 57.00 | 52.4 | 70.6 | 131.8 | 1.82 |
| TWIST-7B | 32×V100 | 7B | 59.00 | 55.3 | 74.1 | 93.7 | 2.71 |
| TWIST-13B | 32×V100 | 13B | 59.20 | 55.4 | 76.4 | - | - |
| SpiritLM | 64×A100 | 7B | 58.0 | 54.8 | 72.9 | - | - |
| **Slam (ours)** | **1×A5000** | **358M** | **58.86** | **58.04** | **82.04** | **62.8** | **2.09** |
| Slam (scaled) | 2×A100 | 358M | 61.11 | 61.30 | 84.18 | 46.6 | 2.69 |
| Slam (large) | 2×A100 | 1.3B | 61.43 | 61.52 | 85.30 | 41.2 | 2.79 |

### 消融实验

| 配置 | tSC↑ | GenPPL↓ | 说明 |
|------|------|---------|------|
| Slam full (Qwen-0.5B) | 82.04 | 62.8 | 完整配方 |
| Slam w/o DPO | 78.01 | 88.3 | 去掉 DPO 后 tSC 掉 4 个点 |
| w/o 合成数据 (Qwen) | 71.14 | 145.4 | 去掉合成数据后 GenPPL 劣化 65% |
| + 多样数据 (Qwen) | 70.66 | 161.8 | 数据多样性反而有害 |
| OPT-125M + 合成 | 75.18 | 96.8 | 模型家族选择影响巨大 |
| 原始 TWIST recipe | 68.80 | 259.2 | 原始配方效果差很多 |
| TWIST + sTinyStories | 72.40 | 159.0 | 仅加合成数据不够 |

### 关键发现
- **DPO 是性价比最高的组件**：仅 30 分钟（24小时的 2%）就带来了 tSC +4、GenPPL -25.5 的提升。分配更多时间反而无益
- **模型选择 > 数据量**：Qwen2.5 远优于 OPT/Pythia 等，即使参数更大导致训练 token 更少。这与传统 scaling law（推荐小模型+多数据）相悖
- **合成数据对语义至关重要**：sTinyStories 虽然只有单一说话人，但显著提升语义指标。而多来源真实数据（声学多样但语义单调）反而无帮助
- **Slam recipe 可扩展**：从 1×A5000 扩展到 2×A100 后所有指标继续稳步提升，说明配方不是 overfit 于小计算预算

## 亮点与洞察
- **"选对模型比堆数据重要"**：这一发现挑战了 SLM scaling law 的悲观预测。TWIST 初始化放大了模型家族间的质量差异，Qwen2.5 的架构优势在低资源下尤为突出
- **合成偏好数据做 DPO 的极致效率**：仅用 TTS 合成 47k 对（SWAG 数据集），30 分钟 DPO 就能显著提升语义能力，这个思路可迁移到任何生成模型的低资源对齐场景
- **"声学多样性在低资源下有害"**：这是一个反直觉但有价值的发现。在计算受限时，模型应集中精力学语义而非声学变化，给数据策略提供了清晰指导

## 局限与展望
- 仅使用 HuBERT semantic tokens，未探索 Mimi 或 SylBoost 等新型 tokenizer 对配方的影响
- 未评估声学/韵律维度（如 SALMon benchmark），低资源 SLM 在这些方面可能表现更差
- 文本交织在当前预算下无显著收益，但更大预算下可能有帮助——未给出最小有效预算
- DPO 存在重复生成问题，仅靠 repetition penalty 缓解，未从根本上解决
- 合成数据仅用单一 TTS 系统（Kokoro），未探索多 TTS 或多种合成策略的影响

## 相关工作与启发
- **vs TWIST**: TWIST 是直接从预训练文本 LM 初始化 SLM 的先驱，本文在此基础上系统化了训练策略，在 1/160 的计算量下达到了 TWIST-7B 的水平
- **vs Cramming**: Cramming 做的是文本 masked LM 的低资源训练，本文将类似思路首次应用到生成式 SLM，发现 SLM 特有的优化——如合成数据和 DPO——非常关键
- **vs AlignSLM**: AlignSLM 也做了 SLM 的 DPO，但用了 64×A100 和 158B text tokens。Slam 仅用 47k 合成偏好对 + 30 分钟 DPO 就达到接近效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 不是方法创新而是系统性工程优化，但"低资源 SLM 训练配方"这一研究问题本身很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 消融非常全面，覆盖模型/优化器/数据/DPO 所有维度，可复现性强
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，每个实验都有明确的结论和 takeaway
- 价值: ⭐⭐⭐⭐ 为学术实验室提供了可直接使用的 SLM 训练配方，开源所有代码/模型/数据

<!-- RELATED:START -->

## 相关论文

- [Matrix3D: Large Photogrammetry Model All-in-One](../../CVPR2025/3d_vision/matrix3d_large_photogrammetry_model_all-in-one.md)
- [RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation](../../ICCV2025/3d_vision/robotron-mani_all-in-one_multimodal_large_model_for_robotic_manipulation.md)
- [URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model](../../NeurIPS2025/3d_vision/urdf-anything_constructing_articulated_objects_with_3d_multimodal_language_model.md)
- [Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis](../../NeurIPS2025/3d_vision/shallow_flow_matching_for_coarse-to-fine_text-to-speech_synthesis.md)
- [Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](../../CVPR2026/3d_vision/ada3drift_adaptive_trainingtime_drifting_for_onest.md)

<!-- RELATED:END -->
