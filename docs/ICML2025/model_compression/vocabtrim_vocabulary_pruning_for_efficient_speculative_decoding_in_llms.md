---
title: >-
  [论文解读] VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs
description: >-
  [ICML 2025][模型压缩][speculative decoding] 提出 VocabTrim，一种免训练方法，通过剪枝 draft 模型的 LM head 词汇表来减少推测解码中的 draft 延迟，在 Llama-3 上实现 16% 的内存受限加速提升。
tags:
  - ICML 2025
  - 模型压缩
  - speculative decoding
  - 剪枝
  - LM head
  - EAGLE
  - inference acceleration
---

# VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs

**会议**: ICML 2025  
**arXiv**: [2506.22694](https://arxiv.org/abs/2506.22694)  
**代码**: -  
**领域**: LLM Inference / Speculative Decoding  
**关键词**: speculative decoding, vocabulary pruning, LM head, EAGLE, inference acceleration  

## 一句话总结

提出 VocabTrim，一种免训练方法，通过剪枝 draft 模型的 LM head 词汇表来减少推测解码中的 draft 延迟，在 Llama-3 上实现 16% 的内存受限加速提升。

## 研究背景与动机

推测解码 (Speculative Decoding, SpD) 使用小型 drafter 模型预测目标 LLM 将生成的 token，然后由目标模型验证。但存在一个被忽视的效率问题：

- 现代 LLM 词汇表很大（如 Llama-3 的 128K token）
- 在 314M 参数的 drafter 中，**LM head 占总参数的 30% 以上**
- 实际下游任务中，大部分词汇 token 从未被采样（如函数调用任务中 120,000+ token 未使用）
- 生成过程通常是**内存受限**的，LM head 的大矩阵乘法浪费了宝贵的内存带宽

## 方法详解

### 核心方法

对 drafter 的 LM head 参数 $W$ 和词汇表 $\mathbb{V}$ 进行裁剪：

$$\mathbb{V}^\text{Trim} = \mathbb{V}[\text{Top-K}(c, k)]$$
$$W^\text{Trim} = W[\text{Top-K}(c, k), :]$$

其中 $c$ 为在校准数据集 $\mathcal{D}$ 上统计的 token 频率计数器，$k$ 为目标词汇表大小。

### 校准数据集选择

三种策略，效果递增：
1. **原始文本数据** (Raw text)：直接可用但次优
2. **Drafter 生成数据**：微调 drafter 时的副产品
3. **目标模型生成数据**：最佳选择（接受率下降最小，加速提升最大）

### 与 SpD 流水线集成

- 适用于任何基于 drafter 的 SpD 方法（EAGLE、独立 drafter 等）
- **无架构约束**：仅替换 LM head 的权重矩阵
- **无训练开销**：仅需统计 token 频率 + 切片矩阵
- 目标模型**完全不受影响**，保持无损生成

### 权衡分析

- 裁剪词汇表后，drafter 只能预测保留的 token → 略微降低接受率 (block efficiency)
- 但 LM head 缩小 → 显著减少内存延迟
- 在内存受限环境下 **MBSU 净收益为正**：

$$\text{MBSU}(x) = \frac{\tau(x)}{c\gamma + 1}$$

其中 $\tau(x)$ 为块效率，$c$ 为 drafter/target 参数比。

## 实验结果

### Llama-3.2-3B-Instruct (EAGLE drafter)

| 配置 | LM Head (M) | Writing MBSU | Math MBSU | Coding MBSU | 平均 MBSU |
|------|-----------|-------------|----------|-------------|----------|
| 原始 EAGLE | 394.0 | 1.475 | 1.640 | 1.708 | ~1.55 |
| +Target generated (32K) | 101.3 | **1.745** | **1.950** | **1.945** | ~**1.84** |

- LM Head 参数从 394M 减至 101M（**减少 75%**）
- 内存受限加速提升约 **16%**

### 独立 drafter (314M)

| 配置 | LM Head (M) | 平均 MBSU |
|------|-----------|----------|
| 原始 | 131.3 | ~2.91 |
| +Target generated (32K) | 33.8 | ~**3.10** |

LM head 从 131M 减至 34M，加速进一步提升。

### 消融：词汇表大小 vs 性能

| Top-K 大小 | Block Efficiency | MBSU |
|-----------|-----------------|------|
| 128K (原始) | 3.63 | 1.70 |
| 64K | 3.54 | 1.83 |
| 32K | 3.43 | **1.95** |
| 16K | 3.25 | 1.90 |

32K 是最优甜点——接受率下降可控，MBSU 最大化。

### 消融：校准数据类型

| 校准数据 | MBSU |
|---------|------|
| Raw-dataset | 1.685 |
| Draft-generated | 1.732 |
| **Target-generated** | **1.745** |

目标模型生成数据效果最好，因为最准确地反映了实际需要的 token 分布。

## 亮点

- 极简且有效：一行代码级别的改动（矩阵切片）即可实现
- 识别出 SpD 中被忽视的效率瓶颈：drafter LM head 过大
- 免训练 + 即插即用 + 保持无损生成
- 对边缘设备等内存受限场景特别有价值
- 方法通用，支持 Top-K/Top-P/最低频率等多种裁剪策略

## 局限性

- 加速效果取决于是否处于内存受限场景（计算受限时收益有限）
- 固定的 $K$ 值无法适应动态变化的任务需求
- 对词汇覆盖率要求高的任务（如多语言翻译）可能影响较大
- 校准数据集的选择引入了任务依赖性
- 未探索更精细的 token 选择策略（如基于 token 重要性而非频率）

## 评分

⭐⭐⭐⭐ — 方法虽简单但直击痛点，对边缘设备推测解码有实际工程价值，16% 的免训练加速提升值得关注。

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding](gumiho_a_hybrid_architecture_to_prioritize_early_tokens_in_speculative_decoding.md)
- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](../../NeurIPS2025/model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[NeurIPS 2025\] Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [\[NeurIPS 2025\] Reject Only Critical Tokens: Pivot-Aware Speculative Decoding](../../NeurIPS2025/model_compression/reject_only_critical_tokens_pivot-aware_speculative_decoding.md)
- [\[ICML 2025\] Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput](speculative_decoding_in_decentralized_llm_inference_turning_communication_latenc.md)

<!-- RELATED:END -->
