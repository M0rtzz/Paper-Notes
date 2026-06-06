---
title: >-
  [论文解读] Star Attention: Efficient LLM Inference over Long Sequences
description: >-
  [ICML2025][LLM/NLP][长序列推理] 提出Star Attention两阶段块稀疏注意力：第一阶段将上下文分块在多主机上局部注意力编码，第二阶段查询通过聚合全局注意力生成，无需微调即可兼容现有LLM，推理加速11倍且保持97-100%精度。
tags:
  - "ICML2025"
  - "LLM/NLP"
  - "长序列推理"
  - "稀疏注意力"
  - "分布式推理"
  - "Anchor Block"
  - "KV缓存"
---

# Star Attention: Efficient LLM Inference over Long Sequences

**会议**: ICML2025  
**arXiv**: [2411.17116](https://arxiv.org/abs/2411.17116)  
**代码**: [GitHub - Star-Attention](https://github.com/NVIDIA/Star-Attention)  
**领域**: LLM/NLP  
**关键词**: 长序列推理, 稀疏注意力, 分布式推理, Anchor Block, KV缓存

## 一句话总结
提出Star Attention两阶段块稀疏注意力：第一阶段将上下文分块在多主机上局部注意力编码，第二阶段查询通过聚合全局注意力生成，无需微调即可兼容现有LLM，推理加速11倍且保持97-100%精度。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：百万token级别的上下文日益常见（代码分析/多文档摘要），但自注意力的二次复杂度使推理极其昂贵。

### 解决思路

**本文目标**：- FlashAttention：加速但不改变复杂度
- Ring Attention：分布式但需要环形通信
- 分块编码方法：需要微调或额外组件

### Star Attention的核心观察

推理通常分两阶段：(1)长上下文编码，(2)短查询生成。上下文token只需局部注意力，查询token需要全局注意力。

## 方法详解

### Phase 1：局部上下文编码+Anchor Block
- 上下文分成连续块，分配到各主机
- 每块前缀一个"锚块"（第一个块的副本）
- 各主机独立做自注意力，无通信
- 只缓存非锚部分的KV

### Phase 2：全局查询编码
- 查询复制到所有主机
- 各主机用本地KV缓存计算局部注意力
- "查询主机"聚合softmax统计量得到全局注意力
- 通信量极小：每主机只传1个向量+1个标量/token

### 特性
- 上下文长度可随主机数线性扩展
- 无需任何模型微调
- 可与FlashAttention组合

## 实验关键数据

### 推理加速


### 主实验

| 上下文长度 | 主机数 | 加速倍数 | 精度保持 |
|-----------|--------|---------|---------|
| 128K | 4 | 4x | 99% |
| 256K | 8 | 7x | 98% |
| 512K | 16 | **11x** | **97%** |

### 精度对比（Llama3.1-8B/70B）


### 消融实验

| 基准 | 标准注意力 | Star Attention |
|------|----------|---------------|
| RULER | 100% | 97-100% |
| LongBench | 基线 | 接近基线 |
| Needle-in-Haystack | 100% | 98%+ |

### 关键发现
1. 锚块是关键——没有锚块精度显著下降
2. Phase 1的局部注意力对大部分上下文理解任务足够
3. Phase 2的全局聚合只需极少通信
4. 在Llama3.1-8B和70B上都有效

## 亮点与洞察

1. "局部上下文+全局查询"的两阶段设计非常自然。
2. 锚块的引入巧妙地保持了全局一致性。
3. 无需微调——任何全局注意力LLM都可即插即用。
4. 通信开销极低（每主机仅1向量+1标量/token）。
5. 与FlashAttention可叠加使用。

## 局限与展望

1. 精度在极强位置依赖任务上可能有2-3%损失。
2. 锚块大小的选择影响性能，需要调优。
3. 对需要全局交互的任务（如全文摘要）可能不是最优。
4. Phase 2的单查询主机可能成为瓶颈。
5. 与KV缓存压缩方法的结合未探讨。

## 相关工作与启发

- 与Ring Attention的区别：Ring需环形通信，Star不需要。
- 与Longformer的区别：Longformer需微调，Star零样本适用。
- 启发：两阶段设计可推广到其他长序列任务。

## 评分
- 新颖性: 4.5/5 — 两阶段+锚块设计
- 实验充分度: 4.5/5 — 多模型多基准
- 写作质量: 4.5/5
- 价值: 5.0/5 — 11x加速+即插即用

## 补充技术细节

### 锚块的作用
第一个block的副本被前缀到每个主机的块前，确保每个主机都能“看到”全局上下文的开头。这个设计受到“attention sink”现象的启发。

### Phase 2的通信量分析
每个上下文主机只需传输softmax归一化统计量(一个向量+一个标量/token)，通信量与上下文长度无关。

### 典型应用场景
最适合“长上下文+短查询+短答案”的模式，如RAG/文档QA/代码分析等。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](../../ACL2025/llm_nlp/bitnetcpp_efficient_edge_inference_for_ternary_llms.md)
- [\[ECCV 2024\] Stripe Observation Guided Inference Cost-Free Attention Mechanism](../../ECCV2024/llm_nlp/stripe_observation_guided_inference_cost-free_attention_mechanism.md)
- [\[ACL 2025\] LLM×MapReduce: Simplified Long-Sequence Processing using Large Language Models](../../ACL2025/llm_nlp/llm_mapreduce_simplified_long_sequence_processing.md)
- [\[ACL 2025\] MHA2MLA: Towards Economical Inference by Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](../../ACL2025/llm_nlp/mha2mla_deepseek_latent_attention.md)
- [\[ICML 2025\] TabFlex: Scaling Tabular Learning to Millions with Linear Attention](tabflex_scaling_tabular_learning_to_millions_with_linear_attention.md)

</div>

<!-- RELATED:END -->
