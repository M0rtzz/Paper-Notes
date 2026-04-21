---
title: >-
  [论文解读] Token Distillation: Attention-Aware Input Embeddings for New Tokens
description: >-
  [ICLR 2026][模型压缩][词表扩展] 提出 Token Distillation 方法，通过蒸馏 Transformer 各层编码的多子词交互信息到单一 token 嵌入中，实现高质量的新 token 嵌入初始化，无需预训练超网络且优于现有方法。
tags:
  - ICLR 2026
  - 模型压缩
  - 词表扩展
  - Token 嵌入初始化
  - 知识蒸馏
  - 领域适应
  - 语言适应
---

# Token Distillation: Attention-Aware Input Embeddings for New Tokens

**会议**: ICLR 2026  
**arXiv**: [2505.20133](https://arxiv.org/abs/2505.20133)  
**代码**: [https://github.com/konstantinjdobler/token-distillation](https://github.com/konstantinjdobler/token-distillation)  
**领域**: 模型压缩  
**关键词**: 词表扩展, Token 嵌入初始化, 知识蒸馏, 领域适应, 语言适应

## 一句话总结

提出 Token Distillation 方法，通过蒸馏 Transformer 各层编码的多子词交互信息到单一 token 嵌入中，实现高质量的新 token 嵌入初始化，无需预训练超网络且优于现有方法。

## 研究背景与动机

- **静态词表问题**: 预训练语言模型使用固定 tokenizer，对领域特定或新语言词汇过度分词，导致性能下降和计算开销增加
- **现有初始化方法的根本局限**:
    - 子词均值法仅利用 embedding 矩阵信息，忽略了 Transformer 层的功能性知识
    - 例如 `<_pal><at><able>` 的各子词 embedding 不包含 `<_palatable>` 的语义
    - 多子词的语义由 Transformer 的注意力/前馈层在上下文化过程中逐步构建（neural detokenization）
- **核心洞察**: 有效的新 token 嵌入必须捕获存储在所有 Transformer 层中的信息，而非仅依赖 embedding 矩阵

## 方法详解

### 整体框架

给定新 token $t^{\star}$ 及其原始子词 $[t_1, \dots, t_n]$，Token Distillation 直接优化新嵌入 $\mathbf{e}^{\star}$，使模型在使用单一新 token 时产生的隐状态与使用原始多子词序列时尽可能接近。

### 关键设计: 隐状态蒸馏目标

优化目标为最小化指定层的隐状态 MSE：

$$\min_{\mathbf{e}^{\star} \in \mathbb{R}^d} \mathbb{E}_{s \sim S} \left[ \frac{1}{|\mathcal{M}(s_\tau, s_{\tau^{\star}})|} \sum_{(i,j) \in \mathcal{M}(s_\tau, s_{\tau^{\star}})} \left\| \mathcal{H}_{\mathbf{e}^{\star}}^{(l)}(s_{\tau^{\star}})_i - \mathcal{H}^{(l)}(s_\tau)_j \right\|_2^2 \right]$$

- $\mathcal{H}^{(l)}(s_\tau)$: 使用原始 tokenization 时第 $l$ 层的隐状态（教师）
- $\mathcal{H}_{\mathbf{e}^{\star}}^{(l)}(s_{\tau^{\star}})$: 使用新 token 嵌入时的隐状态（学生）
- $\mathcal{M}(s_\tau, s_{\tau^{\star}})$: 对齐位置映射，仅包含会 attend 到新 token 的位置
- 实践中使用最后一层隐状态

### 上下文检索

两种获取训练上下文的方法：
1. **主方法**: 使用 Aho-Corasick 算法从语料库中高效检索包含目标 token 的片段
2. **备选**: 用新 token 做 prompt 让模型生成包含目标词的文本

### 输出嵌入处理

- Token Distillation 仅学习输入嵌入（因为新 token 不在教师模型预测范围内）
- 输出嵌入可结合 NTP 目标额外训练，或设为零向量
- 可与 $\alpha$NTP 组合（动态降权 NTP 损失避免干扰）

### 效率设计

- 每个新 token 仅需 25 个上下文片段
- 上下文截断到 50 token 长度
- 2500 个新 token 在单 GPU 上 10 分钟内完成初始化

## 实验

### 主实验：生物医学领域适应（8 个模型平均）

| 方法 | 平均准确率 |
|------|-----------|
| 原始 tokenization | 66.5 |
| Random | 57.5 |
| 子词均值 | 60.8 |
| NTP (仅新嵌入) | 63.0 |
| ZeTT (预训练超网络) | — (仅部分模型) |
| **Token Distillation** | **64.6** |
| **Token Distillation + αNTP** | **64.7** |

### 定义生成质量（LLM 评判）

| 方法 | 相似度 Avg | 正确性 Avg |
|------|-----------|-----------|
| Random | 0.0 | 0.1 |
| 子词均值 | 16.6 | 18.6 |
| NTP | 52.0 | 59.4 |
| ZeTT | — | — |
| **Token Distillation** | **68.5** | **74.4** |
| **Token Distillation + αNTP** | **76.7** | **83.3** |

### 法语语言适应

| 方法 | Mistral-7B | Llama3-8B | Llama3-8B-i | Avg |
|------|-----------|-----------|-------------|-----|
| 原始 | 69.5 | 69.4 | 72.1 | 73.2 |
| 子词均值 | 56.3 | 58.4 | 61.7 | 61.5 |
| NTP | 64.7 | 67.0 | 70.1 | 70.8 |
| **Token Distillation** | **68.5** | **68.9** | **72.9** | **72.9** |

### 关键发现

- Token Distillation 在所有 8 个模型上一致优于 NTP 和子词均值，且无需超网络预训练即超越 ZeTT
- 定义生成实验证实蒸馏后的嵌入质量更高，语义更完整
- 冻结原始嵌入仅更新新嵌入（NTP 变体）比调整全部嵌入效果更好
- Tied embedding 模型（Llama3.2-3B）可能出现 norm 爆炸，加 $\alpha$NTP 正则化可缓解
- 法语适应中 Token Distillation 甚至可超越原始 tokenization（Llama3-8B-i）

## 亮点

- **理论洞察深刻**: 指出现有方法忽略 Transformer 层知识的根本缺陷
- **方法极其轻量**: 每 token 仅需 25 个文本片段，10 分钟处理 2500 个新 token
- **无需额外模型**: 不需要预训练超网络，直接使用目标模型自身
- **广泛模型验证**: 覆盖 3B-8B、base/instruct、tied/untied embedding 等多种设置

## 局限性

- 仅学习输入嵌入，输出嵌入需额外处理
- 对 tied embedding 模型可能出现 norm 不稳定
- 蒸馏目标选择最后一层隐状态，是否最优未充分探索
- 每个新 token 需要少量包含该 token 的上下文文本，完全零资源场景适用性有限
- 相比超网络方法，推理时初始化速度较慢（需要梯度优化而非单次前向传播）

## 相关工作

- **无梯度方法**: 子词均值、加权线性组合（WECHSEL、FVT 等）——忽略 Transformer 层知识
- **基于梯度方法**: NTP 嵌入调优、超网络 ZeTT——前者目标不直接，后者需昂贵预训练
- **Token-to-Words**: 使用 PatchScopes 定位子词被统一表示的层，需训练映射模块
- **Token Distillation**: 无需定位，直接通过蒸馏捕获所有层的信息

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★☆ |
| 理论深度 | ★★★★☆ |
| 实验充分性 | ★★★★★ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★★★ |

<!-- RELATED:START -->

## 相关论文

- [TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](../../NeurIPS2025/model_compression/a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)
- [Parallel Token Prediction for Language Models](parallel_token_prediction_for_language_models.md)
- [Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)

<!-- RELATED:END -->
