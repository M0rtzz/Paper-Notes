---
title: >-
  [论文解读] An Exploration of Mamba for Speech Self-Supervised Models
description: >-
  [ACL 2026][音频/语音][Mamba] 首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。
tags:
  - "ACL 2026"
  - "音频/语音"
  - "Mamba"
  - "语音自监督学习"
  - "HuBERT"
  - "状态空间模型"
  - "流式ASR"
---

# An Exploration of Mamba for Speech Self-Supervised Models

**会议**: ACL 2026  
**arXiv**: [2506.12606](https://arxiv.org/abs/2506.12606)  
**代码**: [GitHub](https://github.com/hckuo145/Mamba-based-HuBERT)  
**领域**: Speech / Self-Supervised Learning  
**关键词**: Mamba, 语音自监督学习, HuBERT, 状态空间模型, 流式ASR

## 一句话总结

首次全面探索Mamba架构作为语音自监督学习（SSL）基础模型的潜力，发现Mamba-based HuBERT在长上下文ASR、流式ASR和因果设置的probing任务中优于Transformer，同时保持线性时间复杂度。

## 研究背景与动机

**领域现状**：Transformer-based语音SSL模型（如HuBERT, wav2vec 2.0）取得了巨大成功，但其二次方复杂度在长序列处理时造成高计算成本和内存瓶颈。

**现有痛点**：(1) Mamba在语言建模中已展现出超越Transformer的能力，但在语音领域的应用仅限于单一任务的孤立研究；(2) 现有语音Mamba工作通常报告与Transformer相当甚至略差的性能，且常需要混合设计；(3) 缺乏统一的跨任务评估。

**核心矛盾**：Mamba的线性时间复杂度理论上非常适合语音的长序列特性，但其在语音SSL中的综合表现尚不明确。

**本文目标**：系统训练和评估Mamba-based HuBERT模型，全面探索其作为语音基础模型和特征提取器的潜力。

**切入角度**：用Mamba block替换HuBERT中的Transformer block，保持相同的训练流程（两轮迭代k-means伪标签训练），在ASR、SUPERB等多任务上评估。

**核心 idea**：Mamba天然的因果架构使其特别适合构建因果语音SSL模型，在流式ASR和长上下文场景中展现独特优势。

## 方法详解

### 整体框架

用Mamba block替换HuBERT的Transformer block，保留CNN特征编码器和位置编码器。训练流程遵循HuBERT的两轮迭代：第一轮以MFCC为目标训练250k步，第二轮用第一轮第6层输出作为目标训练400k步。在LibriSpeech 960h上预训练。

### 关键设计

1. **多种Mamba变体的系统对比**:

    - 功能：全面评估不同Mamba配置的语音表示能力
    - 核心思路：测试因果设置（Mamba, Mamba+MLP）和双向设置（ExtBiMamba, InnBiMamba），并与对应的Transformer变体公平对比
    - 设计动机：Mamba的因果性质可能在某些任务中是优势（流式ASR），在另一些中可能是劣势（需要全局信息的任务）

2. **长上下文和流式ASR评估**:

    - 功能：验证Mamba的线性复杂度在实际场景中的价值
    - 核心思路：在不做句子分割的情况下处理整段语音进行长上下文ASR；在仅使用过去信息的约束下进行流式ASR。量化MACs/秒和RTF随序列长度的变化
    - 设计动机：这是Mamba相对Transformer最大的理论优势所在——Transformer在80秒以上即OOM，Mamba可处理5分钟以上

3. **表示质量分析**:

    - 功能：深入理解Mamba学到的语音表示的特性
    - 核心思路：使用phone purity评估量化表示的语音质量，CCA分析音素和说话人特征的编码方式
    - 设计动机：不仅要知道"好不好"，还要理解"为什么好"以及"好在哪里"

### 损失函数 / 训练策略

遵循HuBERT标准训练：masked prediction loss。使用Adam优化器，线性warm-up（前8%）后线性decay。因计算资源限制，在单块V100上训练，batch size为原始的1/4。

## 实验关键数据

### 主实验

| 设置 | 模型 | 参数量 | WER | 关键发现 |
|------|------|--------|-----|--------|
| 流式ASR | Mamba HuBERT | 78M | 15.77% | 优于94M因果Transformer(16.66%) |
| 长上下文ASR | ExtBiMamba | - | 11.08% | Transformer因OOM无法运行 |
| 标准ASR | ExtBiMamba(Small) | - | 接近Transformer | 小规模有效 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 因果SUPERB | Mamba > Causal Transformer | 在音素和说话人任务上更优 |
| Phone Purity | Mamba更高 | 量化表示的语音质量更好 |
| CCA分析 | 说话人特征更distinct | Mamba对说话人信息编码更清晰 |
| ExtBiMamba Base | 低于Transformer | 大规模双向Mamba仍需改进 |

### 关键发现
- Mamba的因果性质是流式语音场景的天然优势——78M参数优于94M的因果Transformer
- 计算成本随序列长度几乎恒定，而Transformer在80秒以上OOM
- Mamba产生的量化表示phone purity更高，有利于以SSL units为输入的spoken language models
- 大规模双向Mamba（Base）仍全面低于Transformer，暗示可扩展性仍需改进

## 亮点与洞察
- 首次系统性地将Mamba作为语音基础模型进行全面评估，而非仅在单一任务上测试
- "因果性质是优势而非限制"的发现改变了对Mamba在语音中应用的认知
- 量化表示质量的发现对spoken language model领域有直接启示

## 局限与展望
- 双向Mamba的大规模训练效果不佳，可扩展性是关键挑战
- 仅在LibriSpeech上预训练和评估，多语言和噪声场景未测试
- 受限于单块V100，训练规模远小于原始HuBERT
- 未来可探索Mamba2等改进架构和更大规模的训练

## 相关工作与启发
- **vs 混合Mamba-Transformer**: 本文纯Mamba架构，更清晰地揭示Mamba的优劣势
- **vs SSAM**: SSAM关注通用音频而非语音，本文专注于语音SSL
- **vs Mamba流式ASR**: 之前的工作需要额外机制（lookahead等），本文展示纯Mamba即有优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次全面探索Mamba作为语音SSL基础模型
- 实验充分度: ⭐⭐⭐⭐⭐ ASR、SUPERB、表示分析、长上下文、流式等多维评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验细致
- 价值: ⭐⭐⭐⭐ 为语音领域的高效架构选择提供重要实证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ACL 2025\] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation](../../ACL2025/audio_speech/eta-wavlm_efficient_speaker_identity_removal_in_self-supervised_speech_represent.md)

</div>

<!-- RELATED:END -->
