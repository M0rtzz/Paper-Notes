---
title: >-
  [论文解读] HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference
description: >-
  [ACL 2025][注意力机制] HATA 提出了一种将 learning-to-hash 技术集成到 top-k 注意力机制的方法，通过将查询和键映射为二进制哈希码来获取相对 qk 分数排序（而非绝对分数估计），在保持模型精度的同时实现了相对全注意力最高 7.2 倍的加速。
tags:
  - ACL 2025
  - 注意力机制
  - Learning-to-Hash
  - KVCache
  - LLM Inference
  - Hardware-Efficient
---

# HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

**会议**: ACL 2025  
**arXiv**: [2506.02572](https://arxiv.org/abs/2506.02572)  
**代码**: 有（https://github.com/gpzlx1/HATA）  
**领域**: NLP / LLM 推理加速 / 注意力机制  
**关键词**: Top-k Attention, Learning-to-Hash, KVCache, LLM Inference, Hardware-Efficient

## 一句话总结

HATA 提出了一种将 learning-to-hash 技术集成到 top-k 注意力机制的方法，通过将查询和键映射为二进制哈希码来获取相对 qk 分数排序（而非绝对分数估计），在保持模型精度的同时实现了相对全注意力最高 7.2 倍的加速。

## 研究背景与动机

即使有 KVCache 加速，注意力模块仍是 LLM 推理的关键瓶颈。在处理 32K token 序列时，Llama2-7B 的注意力模块消耗超过 70% 的运行时间。瓶颈来自两方面：

**计算复杂度**：注意力计算与序列长度呈二次关系

**内存带宽**：每个解码步骤都需要加载完整的缓存 Key 和 Value 向量

Top-k 注意力利用注意力分布的稀疏性，只保留最相关的 k 个 token 来计算注意力。但现有方法存在效率-精度权衡问题：

- **低秩方法**（Loki、InfiniGen）：在投影维度子集上计算点积，但通道提取成本高，且保持精度需要足够多的维度
- **块式方法**（Quest、InfLLM）：将 KV 分块后估计块级分数上界，但粒度太粗——关键 token 分散在不同块中，选择整块会加载不相关的 KV 对

核心洞察：**现有方法都假设需要精确估计 qk 的绝对分数，但实际上只需要相对排序就够了**。判断 "s_{qk_i} > s_{qk_j}" 远比精确计算 qk 分数成本低。这把问题从"数值回归"转化为"序数比较"。

## 方法详解

### 整体框架

HATA 分为三个阶段：
1. **离线训练**：从真实数据中学习哈希函数，使相似的 qk 对在 Hamming 空间中保持邻近
2. **Prefill 阶段**：正常计算注意力，同时将 Key 编码为二进制哈希码并缓存
3. **Decode 阶段**：对新 query 编码哈希码 → 通过 Hamming 距离快速选出 top-k Key → 仅对选中的 KV 对计算精确注意力

### 关键设计

1. **Learning-to-Hash 建模**：

   哈希函数 $h(x) = 2 \cdot \text{Sigmoid}(\sigma \cdot x W_H) - 1$，其中 $W_H$ 是可训练的哈希权重矩阵。

   优化目标包含三项：
    - **相似性保持**：$\epsilon \sum_j \sum_i s_{j,i} \|h(q_j) - h(k_{j,i})\|^2$ — 相似的 qk 对分配邻近哈希码
    - **比特平衡**：$\eta \sum_j \|\sum_i h(k_{j,i})\|^2$ — 哈希码的每一位中 +1 和 -1 大致平衡
    - **去相关**：$\lambda \|W_H^T W_H - I_r\|$ — 不同哈希位编码不同的信息

   使用 Sigmoid 近似 sign 函数解决不可微问题，σ 控制近似平滑度。每个注意力头单独训练一个 W_H。

2. **训练数据构造**：

    - 在 prefill 阶段收集每个注意力头的 Q 和 K
    - 采样 q_j，计算与 [k_1,...,k_j] 的 qk 分数
    - Top 10% 的 qk 对标记为正样本（线性衰减标签 s ∈ [1,20]），其余 90% 为负样本（s = -1）
    - 每个序列可生成数千到数百万训练对，从数十个序列生成以增加多样性

3. **Decode 阶段算法**：

    - 对新 query 和 key 执行 HashEncode（矩阵乘法 + Sign + BitPack）
    - 将 key 哈希码追加到哈希码缓存
    - 计算 query 哈希码与所有缓存 key 哈希码的 Hamming 距离：`bitcount(bitwise_xor(Q_H, K_H_cache))`
    - 对 GQA 场景聚合共享 KVCache 的多 query 分数
    - 选择 top-k，Gather 对应 KV 对，执行稀疏注意力

### 硬件高效优化

1. **Kernel 融合**：将线性投影 → Sign → BitPack → 缓存更新融合为单个 CUDA kernel，避免 CPU-GPU 反复同步
2. **高性能 Hamming 分数算子**：基于 GPU 的 XOR + popc/popcll 指令实现位计数，使用 coalesced memory access 优化带宽
3. **Gather + FlashAttention 融合**：将 Gather 操作集成到 FlashAttention kernel，减少 HBM↔SRAM 的冗余数据传输

实现代码：1,470 行 C++/CUDA + 940 行 Python，以插件方式集成到现有推理框架。

## 实验关键数据

### 主实验：LongBench-e 精度（512 token budget）

| 任务 | Dense | Loki | Quest | MagicPIG | H2O | SnapKV | **HATA** |
|------|-------|------|-------|----------|-----|--------|---------|
| **Llama-2-7B AVG** | 34.47 | 32.78 | 32.64 | 34.09 | 9.57 | 24.96 | **34.60** |
| **Llama-3.1-8B AVG** | 54.10 | 53.23 | 52.19 | 47.61 | 49.89 | 51.00 | **53.94** |

HATA 在两个模型上均最接近 Dense（全注意力）性能，且在某些任务上超过 Dense（如 HQA 15.65 vs 15.30）。

### NIAH (Needle in a Haystack) 测试

| 任务 | Dense | Loki | Quest | HATA |
|------|-------|------|-------|------|
| NS1 (Llama-2) | 93.75 | 25.00 | 100.0 | **100.0** |
| NS2 (Llama-2) | 100.0 | 2.08 | 95.83 | **98.96** |
| NS3 (Llama-2) | 91.67 | 0.00 | 52.08 | **83.33** |
| NS1 (Llama-3.1) | 100.0 | 98.96 | 100.0 | **98.96** |
| NS3 (Llama-3.1) | 100.0 | 96.88 | 47.92 | **100.0** |

HATA 在 NIAH 多级难度测试中维持了接近或超过 Dense 的性能，而 Loki 在 Llama-2 上大幅崩溃。

### 推理速度（token/s，Llama-3.1-8B，A800）

| 序列长度 | Dense | Loki | Quest | HATA |
|---------|-------|------|-------|------|
| 32K | 基准 | ~1.2× | ~1.5× | ~2× |
| 64K | 基准 | ~1.5× | ~2.5× | ~4× |
| 128K | 基准 | ~2× | ~4× | ~7.2× |

序列越长，HATA 的加速比越大——128K 时达到 7.2 倍加速。

### 关键发现

1. **精度-效率最佳平衡**：在所有 top-k 方法中，HATA 在精度上最接近全注意力，同时加速更大
2. **序数比较足够**：实验证实只需获取 qk 分数的相对排序（而非绝对值估计），即可实现高质量的 top-k 选择
3. **Prefill 开销可忽略**：哈希编码额外开销不到总计算的 1%（因 rbit ≪ 序列长度）
4. **哈希码维度的影响**：r 位哈希码中，r 太小精度下降，r 太大效率降低，128 位通常是好的平衡点
5. **GQA 兼容**：通过聚合共享 KVCache 的多 query 分数，HATA 自然支持 GQA 架构

## 亮点与洞察

- **问题重构的深刻性**：将 top-k 注意力从"精确分数估计"重构为"序数比较"，这一认知跳跃是全文最大的贡献。绝对分数对 top-k 选择确实是不必要的，这个洞察打开了全新的优化空间
- **哈希技术的完美匹配**：learning-to-hash 天然解决序数比较问题（Hamming 距离保序），且二进制运算（XOR + popcount）在 GPU 上极其高效
- **工程完整度高**：不仅有算法设计，还有 kernel 融合、Hamming 算子、FlashAttention 集成等完整的系统级优化
- **即插即用设计**：用户只需替换标准注意力为 HATA 注意力即可使用，降低了部署门槛

## 局限与展望

1. **需要离线训练哈希权重**：每个模型的每层每个头都需要训练 W_H，虽然训练成本不高但增加了部署流程
2. **固定 token budget 假设**：当前实验用固定的 512 token budget，自适应 budget 可能更优
3. **仅在解码阶段加速**：Prefill 阶段仍使用全注意力，对 long-context prefill 的延迟贡献没有优化
4. **模型覆盖有限**：主要在 Llama-2 和 Llama-3.1 上验证，未覆盖 MoE 模型或更大规模模型

## 相关工作与启发

- SparQ 的维度子集方法从不同角度解决同一问题（数值精度 vs 排序精度），HATA 的序数比较范式更彻底
- Quest 和 InfLLM 的块式方法与 token 级方法形成互补——可以考虑两级层次结构（块级粗筛 + 哈希细选）
- 与 Hash-RAG (2505.16133) 的技术对比：同为 learning-to-hash 在 NLP 中的应用，但 HATA 关注注意力内部的 qk 匹配，Hash-RAG 关注知识库外部检索
- 对 KVCache 压缩方向（如 H2O、SnapKV）的启示：HATA 不丢弃任何 KV 对，只是更高效地选择，避免了丢弃关键信息的风险

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — "只需排序不需要绝对分数"的核心洞察非常深刻，learning-to-hash 在 LLM 注意力中的应用是首创
- **实验充分度**: ⭐⭐⭐⭐⭐ — LongBench-e 13 个任务、NIAH 多级难度、两个模型、多种方法对比、速度测试全面，还有消融分析
- **写作质量**: ⭐⭐⭐⭐ — 动机阐述清晰，算法伪代码完整，Figure 1/2/3 信息密度高，但 learning-to-hash 背景部分的公式较密集
- **价值**: ⭐⭐⭐⭐⭐ — 7.2 倍加速且不损精度，对 long-context LLM 部署有直接且重大的实用价值，开源代码支持即插即用

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Efficient OpAmp Adaptation for Zoom Attention to Golden Contexts](efficient_opamp_adaptation_for_zoom_attention_to_golden_contexts.md)
- [\[ACL 2025\] HASH-RAG: Bridging Deep Hashing with Retriever for Efficient, Fine Retrieval and Augmented Generation](hash-rag_bridging_deep_hashing_with_retriever_for_efficient_fine_retrieval_and_a.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[ACL 2025\] Personalized Generation In Large Model Era: A Survey](personalized_generation_in_large_model_era_a_survey.md)
- [\[ACL 2025\] TACLR: A Scalable and Efficient Retrieval-Based Method for Industrial Product Attribute Value Identification](taclr_a_scalable_and_efficient_retrieval-based_method_for_industrial_product_att.md)

<!-- RELATED:END -->
