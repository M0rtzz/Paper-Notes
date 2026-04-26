---
title: >-
  [论文解读] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention
description: >-
  [ACL 2025][LLM效率][Many-Shot ICL] 提出 Dynamic Block-Sparse Attention (DBSA)，一种无需训练的推理框架，通过结构化块稀疏注意力编码和动态检索 KV 缓存，在多示例上下文学习中实现接近微调的推理延迟，同时保持 >95% 的最佳方法准确率。
tags:
  - ACL 2025
  - LLM效率
  - Many-Shot ICL
  - 注意力机制
  - KV Cache
  - Retrieval ICL
  - Inference Efficiency
---

# Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention

**会议**: ACL 2025  
**arXiv**: [2503.08640](https://arxiv.org/abs/2503.08640)  
**代码**: [millix19/dbsa](https://github.com/millix19/dbsa)  
**领域**: LLM Efficiency / In-Context Learning  
**关键词**: Many-Shot ICL, Block-Sparse Attention, KV Cache, Retrieval ICL, Inference Efficiency  

## 一句话总结

提出 Dynamic Block-Sparse Attention (DBSA)，一种无需训练的推理框架，通过结构化块稀疏注意力编码和动态检索 KV 缓存，在多示例上下文学习中实现接近微调的推理延迟，同时保持 >95% 的最佳方法准确率。

---

## 研究背景与动机

### 问题背景
多示例上下文学习（Many-Shot ICL）使用数千个示范样例，性能可与微调媲美甚至超越，且无需参数更新、易于适配新任务、不需要为每个任务部署专门模型。然而，将 ICL 扩展到长上下文引入了新的计算权衡：

- **Fixed ICL**：对所有查询使用同一固定示范集，可以缓存 KV，但性能不如检索 ICL
- **Retrieval ICL**：为每个查询动态检索相关示范，性能更好，但每次推理都需要重新编码上下文，代价高昂
- **微调**：setup 成本最高（需要训练），但推理时开销最小

### 核心挑战
Many-Shot ICL 将计算负担从训练时转移到推理时，处理数千个示范的推理成本比零样本推理高出数个数量级，使其在高吞吐应用中不切实际。

### 研究目标
设计一种框架，同时获得检索 ICL 的高准确率和固定缓存 ICL 的低延迟，使 Many-Shot ICL 在实际部署中可行。

---

## 方法详解

### 整体框架
DBSA 将 Many-Shot ICL 视为两阶段过程：
1. **Stage 1（一次性预编码）**：用块稀疏注意力编码全部示范集
2. **Stage 2（动态推理）**：为每个测试查询检索相关示范块，复用预编码的 KV 缓存

### 关键设计

#### Stage 1: 块稀疏流式注意力编码

给定示范集 $D = \{d_1, d_2, \dots, d_n\}$：
1. 将 $D$ 分为 $n/k$ 个块：$D = [b_1, b_2, \dots, b_{n/k}]$，每块包含 $k$ 个示范
2. 每个块 $b_i$ 仅关注三个部分：
    - **锚定块** $b_1$（attention sink）
    - **前 $j$ 个块** $\{b_{i-j}, \dots, b_{i-1}\}$（局部上下文）
    - **自身**（标准因果注意力）
3. 使用顺序位置编码 $[0, \dots, n-1]$，但在旋转位置变换前缓存 KV 状态
4. 关键优势：新示范可以随时添加，只需编码一个额外块

实现上使用 Flex Attention，对被遮蔽的块跳过计算，获得与稀疏度成比例的速度提升。

#### Stage 2: 动态示范选择与回答生成

给定测试查询 $q^*$：
1. 使用检索方法（BM25）选择示范块子集 $D' = \{b'_1, b'_2, \dots, b'_m\}$，$m < n$
2. 锚定块 $b_1$ 始终被包含（作为 attention sink）
3. 拼接选中块的 KV 缓存，按新的顺序位置 ID $[0, |D'|-1]$ 重新应用旋转位置编码
4. 测试查询以全注意力关注选中的 KV 缓存，自回归生成答案

DBSA 支持插入任意检索方法（文本相似度、余弦相似度、多样性检索器等），本文使用 BM25。

### 配置参数
- 每块 50 个示范
- 随机分组
- 每块关注前 2 个块作为局部上下文
- 检索比例 30%

---

## 实验

### 实验设置
- **模型**：Llama-2-7B (32k)、Llama-3.1-8B (128k)
- **数据集**：5 个分类数据集 — TREC (6类)、TREC-Fine (50类)、NLU (68类)、Banking-77 (77类)、Clinic-150 (151类)
- **上下文长度**：30k 和 90k tokens
- **基线**：Fixed ICL（缓存KV）、Retrieval ICL（每次重编码）、LoRA 微调
- **硬件**：30k 用 L40S (48GB)，90k 用 A100 (80GB)

### 效率对比

**相对延迟（对比 RetICL 基线）**：

| 方法 | 30k Setup | 30k Inference | 90k Setup | 90k Inference |
|------|-----------|---------------|-----------|---------------|
| RetICL | 1x | 1x | 1x | 1x |
| Fixed ICL | 5x | 0.11x | 6.5x | 0.06x |
| LoRA 微调 | >600x | 0.08x | >1500x | 0.046x |
| **DBSA** | **3x** | **0.10x** | **4x** | **0.053x** |

*（以 Llama-3.1-8B 为例）*

关键观察：
- DBSA 推理延迟接近微调模型，但 setup 时间不到微调的 1/375
- 在 >100,000 请求场景下，DBSA 仍然是最高效的方案
- GQA 使得缓存 ICL 延迟大幅降低，DBSA 在 Llama-3.1 上推理延迟仅为 RetICL 的 5.3%

### 准确率对比

**90k 上下文，Llama-3.1-8B**：

| 数据集 | Fixed ICL | Ret ICL | 微调 | DBSA |
|--------|-----------|---------|------|------|
| TREC | 0.96 | 0.95 | 0.96 | 0.95 (99%) |
| TREC Fine | 0.88 | 0.89 | 0.83 | 0.88 (99%) |
| Banking77 | 0.91 | 0.90 | 0.81 | 0.89 (98%) |
| Clinic | 0.92 | 0.91 | 0.74 | 0.90 (98%) |
| NLU | 0.89 | 0.90 | 0.82 | 0.88 (98%) |
| **平均** | 0.91 | 0.91 | 0.83 | **0.90 (99%)** |

DBSA 在 90k 上下文下达到最佳方法准确率的 99%。微调在这些数据集上明显不如 Many-Shot ICL。

### 消融实验

#### 稀疏注意力模式

| 模式 | 平均准确率 |
|------|-----------|
| Full Attention | 0.84 |
| Sink + Prev + Self (DBSA) | 0.82 |
| Sink + Self | 0.27 |
| Self Only | 0.09 |

Attention sink 和局部上下文连接都是必要的。仅有 Sink + Self 效果极差（0.27），说明 StreamingLLM 的 token 驱逐策略不适合 Many-Shot ICL 场景。

#### 块级 vs 示例级检索

| 设置 | 标准推理 | DBSA | 差距 |
|------|---------|------|------|
| 示例级 (90k) | 0.90 | 0.86 | 0.04 |
| 块级 (90k) | 0.89 | 0.88 | 0.01 |

块级选择更好地保持了编码时的上下文关系，且检索更快、内存更高效。

#### 块分组策略

| 策略 | 平均准确率 |
|------|-----------|
| 随机分组 | 0.827 |
| K-means 聚类 | 0.764 |
| 聚类 + 10% 多样性 | 0.810 |

随机分组竟然是最好的，因为自然引入了块内多样性。

#### 动态块排序
- 保持编码顺序或低到高相关性排序效果相当
- 反转顺序显著降低性能（因为有局部注意力依赖）

### 存储成本
- Llama-3.1-8B 的 KV 缓存：0.125 MiB/token，30k 约 3.7GB，90k 约 11.1GB
- LoRA 微调：0.01GB/任务，但在多任务场景（数百任务）下累积成本也很可观

---

## 亮点与洞察

1. **无训练推理框架**：DBSA 不需要任何额外训练或优化，即插即用
2. **打破 ICL 效率瓶颈**：推理延迟接近微调，但 setup 时间减少 >375 倍，使 Many-Shot ICL 在实践中可行
3. **新示范可增量添加**：编码新示范的成本恒定（不随示范池增长），非常适合数据动态变化的场景
4. **微调在这些数据集上不如 ICL**：即使使用全部训练数据，微调也无法显著超越 Many-Shot ICL，挑战了"微调总是更好"的传统认知
5. **LocalAttention + Sink 的有效性**：Sink + Prev + Self 模式在无训练设定下表现出色，稀疏度超 90% 仍保持强性能
6. 对不同因素（稀疏模式、分组策略、排序）的消融分析全面

## 局限性

1. 方法效果依赖于 Retrieval ICL 的有效性，对需要综合全部示范的任务（如统计类任务）不适用
2. 仅在分类数据集上验证，未涉及生成任务（翻译、摘要等）
3. 30k 上下文下 DBSA 与 Ret ICL 的准确率差距更明显
4. 检索比例固定为 30%，未优化
5. KV 缓存存储需求可能成为大规模示范池的瓶颈

## 相关工作

- **Many-Shot ICL**：Bertsch et al. 2024、Agarwal et al. 2024（千示范 ICL 匹敌微调）
- **稀疏注意力**：StreamingLLM (Xiao et al. 2024)、Star Attention (Acharya et al. 2024)
- **KV 缓存压缩**：token 驱逐 (Xiao et al. 2024)、量化与低秩近似
- **ICL 示范选择**：BM25 検索 (Luo et al. 2024)、Parallel Context Windows (Ratner et al. 2023)
- **RAG 中的稀疏关注**：Lu et al. 2024、Sun et al. 2024

---

## 评分 ⭐⭐⭐⭐

解决的问题实际且重要（Many-Shot ICL 的部署效率），方法设计优雅（两阶段 + 块稀疏），实验分析扎实。局限在于仅验证分类任务且任务多样性不足，但核心贡献——证明无训练稀疏注意力可使 ICL 延迟接近微调——很有价值。

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] On Many-Shot In-Context Learning for Long-Context Evaluation](on_many-shot_in-context_learning_for_long-context_evaluation.md)
- [\[ACL 2025\] What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)
- [\[ACL 2025\] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)

<!-- RELATED:END -->
