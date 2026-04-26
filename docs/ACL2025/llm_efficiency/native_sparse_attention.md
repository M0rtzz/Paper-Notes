---
title: >-
  [论文解读] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
description: >-
  [ACL 2025][LLM效率][稀疏注意力] DeepSeek提出NSA——一种可原生训练的分层稀疏注意力机制，通过压缩token、选择token和滑动窗口三条并行注意力路径实现高效长上下文建模，在27B参数模型上预训练后性能全面匹配甚至超越Full Attention，同时在64k序列上实现显著加速。
tags:
  - ACL 2025
  - LLM效率
  - 稀疏注意力
  - 长上下文建模
  - 硬件对齐
  - 端到端训练
  - KV缓存压缩
---

# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**会议**: ACL 2025  
**arXiv**: [2502.11089](https://arxiv.org/abs/2502.11089)  
**代码**: 无 (DeepSeek内部实现)  
**领域**: LLM效率 / 注意力机制  
**关键词**: 稀疏注意力, 长上下文建模, 硬件对齐, 端到端训练, KV缓存压缩

## 一句话总结

DeepSeek提出NSA——一种可原生训练的分层稀疏注意力机制，通过压缩token、选择token和滑动窗口三条并行注意力路径实现高效长上下文建模，在27B参数模型上预训练后性能全面匹配甚至超越Full Attention，同时在64k序列上实现显著加速。

## 研究背景与动机

**领域现状**: 长上下文建模是下一代LLM的关键能力，但标准注意力机制的 $O(n^2)$ 计算复杂度在长序列下成为延迟瓶颈——在64k上下文的解码阶段，注意力计算占总延迟的70-80%。现有稀疏注意力方法(如H2O、Quest、MInference)主要应用于推理阶段。

**现有痛点**: (1) **速度优势的幻觉**：许多方法理论FLOPS降低但实际延迟提升有限，因为算法设计与硬件特性不匹配(如分散的内存访问与GQA架构冲突)；(2) **可训练性的缺失**：大多数方法仅在推理时施加稀疏性，离散操作(如聚类、哈希)阻断梯度流，且token级选择导致非连续内存访问无法利用FlashAttention。

**核心矛盾**: 现有稀疏注意力在硬件加速和端到端训练两方面存在根本性gap——要么只在特定推理阶段有效，要么无法参与训练。

**本文目标**: 设计一种同时满足硬件对齐高效推理和端到端可训练的稀疏注意力机制，在预训练、微调和推理全生命周期实现加速。

**切入角度**: 从注意力分数的空间连续性(blockwise clustering)出发，设计分块级的token压缩和选择策略，结合硬件特性(Tensor Core、GQA共享KV缓存)优化kernel实现。

**核心 idea**: 将注意力分解为压缩(全局粗粒度)、选择(局部细粒度)和滑动窗口(近距离)三条路径，通过硬件对齐的分块设计实现全生命周期可训练的稀疏注意力。

## 方法详解

### 整体框架

NSA将标准的全注意力替换为三条并行的注意力分支：(1) **压缩注意力(Compression)**——将KV序列按块聚合为粗粒度表示，捕获全局信息；(2) **选择注意力(Selection)**——基于压缩分数选出最重要的细粒度token块，保留局部精度；(3) **滑动窗口(Sliding Window)**——保持近距离局部上下文。三条分支通过可学习的门控机制(MLP+sigmoid)加权融合，并使用独立的KV防止shortcut learning。

### 关键设计

1. **Token压缩(Compression)**
    - 功能：将连续token块压缩为单个粗粒度KV表示
    - 核心思路：用可学习MLP(带块内位置编码)将长度为 $l$ 的KV块映射为单个压缩token，采用滑动步长 $d < l$ 避免信息碎片化
    - 设计动机：粗粒度表示以极低计算代价覆盖全局上下文

2. **基于重要性分数的分块选择(Token Selection)**
    - 功能：从全序列中选出与当前query最相关的细粒度token块
    - 核心思路：复用压缩注意力的中间attention score作为块重要性分数(零额外开销)，对GQA组内所有query head聚合分数确保共享选择，选取top-n个最重要的块
    - 设计动机：仅用压缩token会丢失细粒度信息；分块选择(而非单token)兼顾硬件效率和注意力分数的空间连续性

3. **独立滑动窗口分支**
    - 功能：显式处理局部上下文，防止局部模式shortcut其他分支的学习
    - 核心思路：维护独近 $w$ 个token的窗口，与压缩和选择分支隔离计算并通过门控融合
    - 设计动机：如果不隔离，局部模式会主导学习过程，阻碍模型学习长程压缩和选择能力

### 损失函数 / 训练策略

- 预训练：27B参数(3B active) MoE模型，在270B个8k长度token上预训练，后续32k长度YaRN做长上下文适配
- 三分支使用独立KV投影，通过门控分数 $g_t^c \in [0,1]$ (sigmoid激活)加权融合
- 训练阶段直接使用NSA替代Full Attention端到端训练，loss曲线稳定且始终低于Full Attention
- 推理优化：定制Triton kernel，采用group-centric data loading策略——每次加载GQA组内所有query head共享稀疏KV块，最大化Tensor Core利用率

## 实验关键数据

### 主实验

| 基准 | Full Attention | NSA |
|------|---------------|-----|
| MMLU (5-shot) | 0.567 | 0.565 |
| BBH (3-shot) | 0.497 | 0.521 |
| GSM8K (8-shot) | 0.486 | 0.520 |
| DROP (1-shot) | 0.503 | 0.545 |
| HumanEval (0-shot) | 0.335 | 0.348 |
| **通用平均** | **0.443** | **0.456** |
| LongBench平均 | 0.437 | **0.469** |
| AIME 8k | 0.046 | **0.121** |
| AIME 16k | 0.092 | **0.146** |

### 消融实验

| 方法 | LongBench平均 |
|------|-------------|
| H2O | 0.303 |
| InfLLM | 0.383 |
| Quest | 0.392 |
| Exact-Top | 0.423 |
| Full Attention | 0.437 |
| **NSA** | **0.469** |

### 关键发现

- NSA在9个通用基准中7个超越Full Attention，推理相关任务提升显著(DROP: +0.042, GSM8K: +0.034)
- LongBench上NSA超越Full Attention（+0.032）和所有推理时稀疏方法
- 64k序列上训练加速可达9.0×(前向)/6.0×(反向)，解码加速最高11.6×
- AIME数学推理任务上NSA-R显著优于Full Attention-R，验证稀疏注意力对高级推理的兼容性
- 64k Needle-in-a-Haystack测试中NSA取得完美准确率

## 亮点与洞察

- **稀疏注意力的反直觉优势**：NSA不仅不降低性能，在推理任务上反而超越Full Attention——稀疏性可能起到了"注意力正则化"的作用，过滤无关噪声
- **复用压缩分数做选择**的设计堪称精妙——实现零额外开销的重要性评估
- 对现有方法的系统性分析(Phase-Restricted Sparsity、GQA兼容性、梯度不可微问题)非常深刻，为未来研究指明方向
- 率先在预训练阶段就引入稀疏注意力并证明其有效性，而非仅在推理时做后处理

## 局限与展望

- 仅在27B MoE模型上验证，更大规模模型(如100B+)的效果有待验证
- 压缩块大小、选择块数量等关键超参的敏感性分析不够充分
- 目前kernel仅针对A100优化，其他GPU架构(如H100, AMD MI300X)的适配未知
- 未开源实现，可复现性存疑
- 滑动窗口大小固定(512 tokens)，动态调整可能进一步提升效果

## 相关工作与启发

- 与FlashAttention正交：FlashAttention优化标准注意力的IO效率，NSA从算法层面减少需要计算的KV对
- 与StreamingLLM(attention sink + sliding window)的关系：NSA将sink概念泛化为可学习的压缩token
- 启发：稀疏注意力的未来方向应该是"原生训练"而非"推理后处理"，预训练时就让模型学习最优稀疏模式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ (首个全生命周期可训练的硬件对齐稀疏注意力)
- **实验充分度**: ⭐⭐⭐⭐ (通用/长上下文/推理三个维度全面评测，但超参分析略不足)
- **写作质量**: ⭐⭐⭐⭐⭐ (动机分析深入，方法阐述清晰，图表精准)
- **价值**: ⭐⭐⭐⭐⭐ (可能改变长上下文LLM的注意力设计范式)

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Hardware-aligned Hierarchical Sparse Attention for Efficient Long-term Memory Access](../../NeurIPS2025/llm_efficiency/hardware-aligned_hierarchical_sparse_attention_for_efficient_long-term_memory_ac.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ACL 2025\] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)
- [\[ACL 2025\] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](ladm_long_context_data.md)
- [\[ACL 2025\] RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)

<!-- RELATED:END -->
