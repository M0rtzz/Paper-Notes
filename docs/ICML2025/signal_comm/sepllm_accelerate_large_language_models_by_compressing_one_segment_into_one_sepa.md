---
title: >-
  [论文解读] SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator
description: >-
  [ICML 2025][稀疏注意力] 提出 SepLLM，利用分隔符 token（标点符号等）天然压缩文本段落信息的特性，仅保留 Initial + Separator + Neighboring 三类 token 的 KV 缓存，在保持性能的同时大幅减少注意力计算和内存占用。
tags:
  - ICML 2025
  - 稀疏注意力
  - 分隔符压缩
  - KV缓存
  - 流式推理
  - 大语言模型
---

# SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator

**会议**: ICML 2025  
**arXiv**: [2412.12094](https://arxiv.org/abs/2412.12094)  
**代码**: [项目主页](https://sepllm.github.io) (有)  
**领域**: LLM效率  
**关键词**: 稀疏注意力, 分隔符压缩, KV缓存, 流式推理, 大语言模型

## 一句话总结

提出 SepLLM，利用分隔符 token（标点符号等）天然压缩文本段落信息的特性，仅保留 Initial + Separator + Neighboring 三类 token 的 KV 缓存，在保持性能的同时大幅减少注意力计算和内存占用。

## 研究背景与动机

**领域现状**：Transformer 的二次注意力复杂度和线性增长的 KV 缓存是长序列推理的核心瓶颈。StreamingLLM 发现了 attention sink 现象，证明保留初始 token + 滑动窗口即可维持流式推理能力，但这种 training-free 方案丢弃了窗口外信息，在需要远距离依赖的任务上性能损失明显。

**现有痛点**：现有 KV 压缩方法（H2O、SnapKV、PyramidKV）基于注意力分数的 top-k 选择，计算开销不可忽略且选择策略与预训练分布不一致。StreamingLLM 虽然简单，但窗口外信息完全丢失，在数学推理（GSM8K）等任务上退化严重（77.79%→69.67%）。

**核心矛盾**：既要大幅压缩 KV 缓存以加速推理，又要保留序列中关键的语义信息。滑动窗口方案覆盖了局部信息但丢失全局依赖，全注意力方案保留了所有信息但代价太高。

**本文目标** 找到一种介于全注意力和滑动窗口之间的稀疏注意力模式，能以远低于 100% KV 的成本保留绝大部分文本语义信息。

**切入角度**：观察到注意力权重在分隔符 token（句号、逗号、空格等）位置出现明显聚集现象。这些分隔符天然充当了文本段落的"压缩锚点"——模型在处理一个段落后，将关键信息编码到后续分隔符的隐藏状态中。因此，仅保留分隔符的 KV 就能高效恢复段落级信息。

**核心 idea**：分隔符 token 是文本的天然压缩节点，保留它们的 KV 等价于以极低成本维持全序列的段落级语义覆盖。

## 方法详解

### 整体框架

SepLLM 将序列中的 token 分为三类：Initial tokens（前 $k$ 个 token，充当 attention sink）、Separator tokens（9 种标点和空白符：".", ",", "?", "!", ";", ":", " ", "\t", "\n"）、Neighboring tokens（当前位置的滑动窗口内 token）。注意力计算仅限于这三类 token 的 KV，其余 token 的 KV 被丢弃。该方案支持 training-free 直接应用和 training-from-scratch 两种模式。在流式推理场景下，进一步设计了 4 缓冲区旋转策略以支持无限长度输入。

### 关键设计

1. **三类 Token 稀疏注意力**:

    - 功能：确定哪些 token 的 KV 需要保留
    - 核心思路：对每个查询 token $q_i$，其注意力范围限定为 $\mathcal{S}(i) = \mathcal{I} \cup \mathcal{P}(i) \cup \mathcal{N}(i)$，分别对应初始 token 集合、位置 $i$ 之前的所有分隔符集合、以及大小为 $n$ 的局部窗口。注意力掩码之外的位置用 $-\infty$ 填充
    - 设计动机：Initial tokens 解决 attention sink 问题（已被 StreamingLLM 验证），Neighboring tokens 覆盖局部上下文，Separator tokens 是本文的核心创新——它们以稀疏分布覆盖了整个历史序列的段落级信息，成本远低于全注意力

2. **流式推理 4 缓冲区策略**:

    - 功能：支持无限长度序列的在线推理
    - 核心思路：KV 缓存被组织为 4 个块——Initial Block（固定大小，存储前 $k$ 个 token）、Separator Block（动态大小，存储所有历史分隔符）、Local Window Block（固定大小 $n$，存储最近 token）、Past Window Block（固定大小 $s$，存储 Local Window 溢出的近期 token）。当 Separator Block 过大时可进一步限制为最近 $m$ 个分隔符
    - 设计动机：直接存储所有历史分隔符在超长序列上仍会导致内存增长。4 缓冲区的分层设计确保了内存使用的确定性上界，同时 Past Window 作为缓冲区避免了滑动窗口边界的信息断裂

3. **位置偏移（Position Shift）**:

    - 功能：修正稀疏注意力引入的位置编码不连续性
    - 核心思路：由于非分隔符 token 被跳过，保留 token 之间的实际位置 ID 存在间隙。位置偏移将保留 token 的位置 ID 重新编为连续序列，确保 RoPE 等位置编码的正常工作
    - 设计动机：消融实验显示不做位置偏移会导致 PPL 从 13.1 灾难性地升至 192.7，证明位置连续性对稀疏注意力方案至关重要

### 损失函数 / 训练策略

Training-from-scratch 模式使用标准的自回归交叉熵损失，但注意力计算仅在三类 token 之间进行，因此训练 FLOPs 仅为标准 Transformer 的 71.77%。Training-free 模式直接在预训练好的模型上应用稀疏注意力掩码，不需要任何参数更新。论文还证明了通用近似定理（Theorem 5.1）：SepLLM 在 $H=2, d_h=1, d_f=4$ 的最小配置下即可近似任意连续的 seq2seq 函数。

## 实验关键数据

### 主实验

| 方法 | GSM8K-CoT (%) | KV 比例 | MMLU (%) | KV 比例 |
|:---:|:---:|:---:|:---:|:---:|
| Vanilla (Llama-3-8B) | 77.79 | 100% | 65.29 | 100% |
| StreamingLLM ($n$=256) | 69.67 | 26% | 62.33 | 37.73% |
| H2O | 76.27 | — | — | — |
| SnapKV | 76.50 | — | — | — |
| PyramidKV | 75.82 | — | — | — |
| **SepLLM** ($n$=256) | **77.18** | 47.36% | **64.68** | 44.61% |

| 模型 (Train from scratch) | ARC-Easy | PIQA | SciQ | FLOPs |
|:---:|:---:|:---:|:---:|:---:|
| Vanilla Pythia-160M | 46.80 | 62.84 | 81.50 | 100% |
| SepLLM Pythia-160M ($n$=128) | **47.35** | **64.64** | **82.60** | 71.77% |

### 消融实验

| 配置 | PPL (WikiText) | 变化 |
|:---:|:---:|:---:|
| SepLLM 完整 | 13.1 | — |
| 去掉位置偏移 | 192.7 | +179.6（灾难性退化） |
| 去掉 Initial tokens | 14.9 | +1.8 |
| 分隔符 9 种全用 | 13.1（GSM8K 77.18%） | 最佳 |
| 仅用 4 种分隔符 | — (GSM8K 76.68%) | -0.50 |
| 仅用 2 种分隔符 | — (GSM8K 70.66%) | -6.52 |

| 流式推理 (PG19, 4M tokens) | PPL |
|:---:|:---:|
| StreamingLLM ($s$=64) | 36.1 |
| SepLLM ($s$=64) | **33.9** |

### 关键发现

- SepLLM 在 GSM8K 上仅用 47.36% KV 即达到 77.18%（vanilla 77.79%），大幅超越 StreamingLLM 的 69.67%，也优于需要计算注意力分数的 H2O/SnapKV/PyramidKV
- 从头训练时，SepLLM 在多个下游任务上略优于 vanilla Transformer，同时节省 28.23% FLOPs 和约 26% 训练时间
- 位置偏移是不可或缺的组件（去掉后 PPL 暴涨 15 倍），而分隔符种类的数量与性能正相关（9 种 > 4 种 > 2 种）
- 流式推理场景下，SepLLM 比 StreamingLLM 在超长文本上持续保持更低 PPL

## 亮点与洞察

- 核心洞察极其直觉化：标点符号本身就是人类语言的"段落分界标记"，模型学到利用它们来存储上下文信息是自然的
- 方法实现简单——本质上只是一个特殊的注意力掩码，不需要额外的可学习参数或复杂的选择策略
- 通用近似定理的证明虽然在最小配置下，但为稀疏注意力的表达能力提供了理论下界保证
- 4 缓冲区的流式推理设计考虑周到，具有工程落地的实用性

## 局限与展望

- 分隔符的定义是硬编码的 9 种字符，对编程代码、数学公式等分隔符模式不同的领域可能需要定制化
- KV 占比 47.36% 虽然减半但不算极端压缩，对比激进的方案（如 StreamingLLM 的 26%）仍有差距——虽然后者性能损失大得多
- 通用近似定理的条件（$H=2, d_h=1$）过于理想化，与实际模型配置差距大，理论意义有限
- 未评估在多轮对话、RAG 等实际部署场景下的表现，这些场景的分隔符分布可能与预训练文本不同
- 多语言场景下分隔符定义可能需要扩展（如中文句号"。"、日文句号"。"等）

## 相关工作与启发

- **vs StreamingLLM**: 两者都利用了 attention sink 现象，但 StreamingLLM 只有窗口内信息，SepLLM 通过分隔符 token 保留了全局段落级信息，代价是 KV 占比略高但性能保留显著更好
- **vs H2O/SnapKV/PyramidKV**: 这些方法基于注意力分数选择重要 token，需要先计算完整注意力再剪枝。SepLLM 的分隔符选择是预定义的，不需要额外计算开销
- **vs Longformer/BigBird**: 经典的稀疏注意力方案使用全局 token + 局部窗口 + 随机连接，SepLLM 的分隔符 token 可视为一种自然语言驱动的全局 token 选择策略

## 评分

- 新颖性: ⭐⭐⭐⭐ 分隔符压缩的洞察简洁优雅，将语言学直觉转化为技术设计
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 training-free 和 from-scratch 两种模式，多种基准对比，消融详尽
- 写作质量: ⭐⭐⭐⭐ 动机清晰，实验组织有条理，理论部分有一定深度
- 价值: ⭐⭐⭐⭐ 方法简单且有效，具有明确的工程落地路径，对 LLM 推理加速有实际贡献

<!-- RELATED:START -->

## 相关论文

- [ToolSpectrum: Towards Personalized Tool Utilization for Large Language Models](../../ACL2025/signal_comm/toolspectrum_towards_personalized_tool_utilization_for_large_language_models.md)
- [Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)](../../NeurIPS2025/signal_comm/artificial_hivemind_the_open-ended_homogeneity_of_language_models_and_beyond.md)
- [Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [Reward-Augmented Data Enhances Direct Preference Alignment of LLMs](reward-augmented_data_enhances_direct_preference_alignment_of_llms.md)

<!-- RELATED:END -->
