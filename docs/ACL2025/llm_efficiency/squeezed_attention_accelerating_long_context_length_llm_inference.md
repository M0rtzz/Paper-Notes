---
title: >-
  [论文解读] Squeezed Attention: Accelerating Long Context Length LLM Inference
description: >-
  [ACL 2025][LLM效率][长上下文推理] 提出 Squeezed Attention，通过离线 K-means 聚类压缩固定上下文的 Key 向量，在推理时用质心匹配预测重要 Key 并仅对其计算精确注意力，实现 3.1 倍 KV 预算削减且无精度损失，预填充和生成阶段均获得超过 4 倍加速。
tags:
  - "ACL 2025"
  - "LLM效率"
  - "长上下文推理"
  - "注意力加速"
  - "KV缓存压缩"
  - "K-means聚类"
  - "稀疏注意力"
---

# Squeezed Attention: Accelerating Long Context Length LLM Inference

**会议**: ACL 2025  
**arXiv**: [2411.09688](https://arxiv.org/abs/2411.09688)  
**代码**: [GitHub](https://github.com/SqueezeAILab/SqueezedAttention)  
**领域**: LLM效率  
**关键词**: 长上下文推理, 注意力加速, KV缓存压缩, K-means聚类, 稀疏注意力  

## 一句话总结

提出 Squeezed Attention，通过离线 K-means 聚类压缩固定上下文的 Key 向量，在推理时用质心匹配预测重要 Key 并仅对其计算精确注意力，实现 3.1 倍 KV 预算削减且无精度损失，预填充和生成阶段均获得超过 4 倍加速。

## 研究背景与动机

**领域现状**：新兴 LLM 应用（如文档分析、代码生成、长文问答）需要处理极长的输入上下文（数万到百万 token）。长上下文推理的核心瓶颈在于 Attention 机制的计算和内存开销随序列长度线性增长。KV 缓存的内存占用和 Attention 的 FLOPs 成为实际部署的主要障碍。

**现有痛点**：已有的长上下文加速方法主要分为两类——(1) KV 缓存压缩（如 eviction、量化），但这些方法是通用的，没有利用具体应用场景的特点；(2) 稀疏注意力（如 local + global patterns），但这些预设模式可能与实际注意力分布不匹配。更重要的是，现有方法没有区分"固定上下文"和"动态用户输入"。

**核心矛盾**：在很多实际应用中（如用户对同一文档反复提问、代码助手在同一代码库上工作），输入 prompt 的大部分内容是固定不变的，只有用户查询部分是动态的。现有方法对固定和动态部分一视同仁，白白浪费了对固定部分进行离线优化的机会。

**本文目标**：针对"大部分上下文固定"这一常见模式，设计一种方法在离线阶段预处理固定上下文的 KV 缓存，使得在线推理时只需分析少量重要 Key 即可完成注意力计算。

**切入角度**：观察到固定上下文中的 Key 向量具有语义聚类结构——语义相近的 token 的 Key 向量在向量空间中也是相近的。如果用聚类质心代表一组 Key，就可以先用质心快速筛选语义相关的 Key 簇，再精确计算注意力。

**核心 idea**：离线用 K-means 聚类固定上下文的 Key 向量，在线时用 Query 与质心比较预测重要 Key 所在的簇，仅对这些重要 Key 计算精确注意力，将复杂度从线性降低到亚线性甚至对数级。

## 方法详解

### 整体框架

Squeezed Attention 分为离线和在线两个阶段。离线阶段：对固定上下文运行一次前向传播获取每一层每个 attention head 的 Key 向量，然后对每个 head 的 Key 向量独立进行 K-means 聚类，存储聚类质心和每个 Key 的簇归属。在线阶段：当新的用户 Query 到达时，先计算 Query 与各簇质心的相似度，选出 top-k 个最相关的簇，取出这些簇中的原始 Key（和对应的 Value），用 FlashAttention 执行精确的稀疏注意力。

### 关键设计

1. **离线 K-means Key 聚类**:

    - 功能：将固定上下文的 Key 向量按语义相似性分组，生成压缩表示（质心）
    - 核心思路：对每个注意力层、每个 head 的 Key 矩阵 $K \in \mathbb{R}^{n \times d}$ 独立运行 K-means 算法，得到 $C$ 个质心 $\mu_1, ..., \mu_C$ 和每个 Key 的簇ID。离线只需执行一次，当用户更换文档时才需重新聚类。聚类数量 $C$ 是控制精度-效率 trade-off 的关键超参
    - 设计动机：相比直接丢弃/量化 KV 缓存，聚类方法保留了所有原始 Key 但用质心提供快速索引。这是一种"有组织的压缩"而非"有损的压缩"

2. **在线质心匹配与重要 Key 选择**:

    - 功能：对每个输入 Query token 高效识别固定上下文中语义最相关的 Key
    - 核心思路：对于用户输入的 Query 向量 $q$，计算与所有簇质心的点积相似度 $s_j = q^\top \mu_j$，选择 top-$k$ 个相似度最高的簇。这些簇中包含的 Key 就是"重要 Key"。然后只从这些重要 Key 和对应 Value 中计算精确注意力：$\text{Attn}(q, K_{\text{imp}}, V_{\text{imp}})$。质心比较的复杂度为 $O(C)$，远小于原始的 $O(n)$
    - 设计动机：利用注意力分数的稀疏性——大多数 token 的注意力权重接近零，真正重要的 key 只是少数。聚类+质心匹配是一种高效的近似 top-k attention 的方式

3. **层次化聚类加速（Hierarchical Squeezed Attention）**:

    - 功能：将注意力复杂度从线性进一步降低到对数级
    - 核心思路：构建多层次的聚类树。第一级有少量粗粒度簇，每个粗簇内再进行细粒度聚类，形成树状结构。在线推理时，先在粗粒度质心上筛选，再在选中簇的细粒度质心上进一步筛选，逐步缩小搜索范围。每一级筛选的开销与该级簇数成正比，总开销为 $O(\log n)$
    - 设计动机：当固定上下文特别长（如百万 token）时，即使普通 K-means 的质心数量也可能很大。层次化方法将搜索空间指数级缩减，使方法可扩展到超长上下文

### 损失函数 / 训练策略

Squeezed Attention 是一种**免训练**（training-free）的推理加速方法。K-means 聚类在离线阶段运行，不涉及反向传播。在线推理时使用精确注意力（只是在更小的 Key 子集上），因此不引入近似误差到注意力计算本身中——唯一的近似在于 Key 选择阶段。作者还实现了高效的 Triton 内核（sparse FlashAttention 和质心比较内核）来实现实际加速。

## 实验关键数据

### 主实验

| 模型 | 基准 | KV 预算削减 | 精度损失 | 说明 |
|------|------|-----------|---------|------|
| LLaMA-2-7B-32K | LongBench | 3.1x | 无明显损失 | KV 缓存减少到 1/3 |
| LLaMA-2-7B-32K | LongBench | 8x | 仅 0.5 分下降 | 激进压缩仍可接受 |
| LWM-Text-Chat-1M | LongBench | 3.1x | 无明显损失 | 百万级上下文模型上同样有效 |
| Longchat-7B-v1.5-32K | LongBench | 3.1x | 无明显损失 | 跨模型一致有效 |
| 所有模型 | 实际延迟 (prefill) | - | - | >4x 加速 |
| 所有模型 | 实际延迟 (generation) | - | - | >4x 加速 |

### 消融实验

| 配置 | LongBench 平均分 | 说明 |
|------|-----------------|------|
| Full Attention | 最高 | 无压缩基线 |
| Squeezed Attention (3.1x) | 接近 Full | 精度几乎无损 |
| Squeezed Attention (8x) | 略低 (-0.5) | 可接受的精度牺牲 |
| 均匀采样 Key (3.1x) | 明显下降 | 随机选Key效果差 |
| 固定 local+global pattern | 明显下降 | 预设模式不匹配 |
| Hierarchical (2-level) | 接近 flat | 日志级复杂度，精度不降 |
| w/o 自定义 Triton 内核 | 精度相同，速度较慢 | 内核实现对实际加速很关键 |

### 关键发现

- 3.1x KV 压缩率是"无损"操作的甜蜜点，8x 压缩仍在可接受范围内
- 层次化变体在超长上下文上的效率优势更大，且精度几乎不受影响
- 自定义 Triton 内核（sparse FlashAttention + 质心比较）是实现 4x 实际速度提升的关键。没有内核优化，算法层面的节省无法转化为真实加速
- 不同 head 的 Key 分布差异很大，逐 head 独立聚类比全局聚类效果好得多
- 在 prefill 和 generation 两个阶段都能获得加速，说明方法是全面适用的

## 亮点与洞察

- **场景洞察驱动的设计**：敏锐地识别出"固定上下文 + 动态查询"这一在实际应用中极其常见的模式（文档问答、代码助手），并据此设计离线-在线的优化框架。这个切入点本身就很有价值
- **免训练方法**：不需要修改模型参数、不需要重新训练，直接在推理阶段即插即用。这大大降低了部署门槛
- **端到端的实际加速**：不仅在算法层面减少了计算量，还通过 Triton 内核实现了真实的 4x+ 速度提升。很多"理论上加速"的工作缺乏实际内核实现，本文弥补了这一点
- **层次化扩展**：从线性复杂度到对数复杂度的扩展非常优雅，为处理百万级上下文提供了理论基础

## 局限与展望

- 离线聚类阶段本身有开销（K-means on 大量 Key），对于频繁更换文档的场景可能不划算
- 假设固定上下文占比较大才能获得显著加速，纯动态场景（如开放域对话）不适用
- 聚类质量对 K-means 初始化敏感，论文未详细分析不同聚类算法的影响
- 目前只验证了 7B 模型，更大模型（70B+）上的效果和工程挑战未知
- 未来可探索：(1) 将聚类与 KV 缓存量化结合进一步压缩；(2) 增量式聚类更新以支持部分上下文变化的场景；(3) 与 LoRA 等参数高效方法结合

## 相关工作与启发

- **vs StreamingLLM**: StreamingLLM 保留最近的 token 和 attention sink，是一种 heuristic 的 KV eviction。Squeezed Attention 用语义聚类做精确选择，理论基础更强
- **vs H2O (Heavy Hitter Oracle)**: H2O 基于注意力得分累积选择重要 Key，但需要在线计算所有注意力得分来做选择。Squeezed Attention 用质心匹配预先快速筛选，开销更低
- **vs Gisting/Compressor tokens**: 这类方法通过训练特殊 token 来压缩上下文，需要额外训练。Squeezed Attention 免训练，更灵活
- 该方法可以作为长上下文 LLM 推理加速的强基线，特别适合文档问答、代码助手等"固定上下文"场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 离线聚类+在线稀疏注意力的组合虽非全新概念，但在 LLM 长上下文场景中的系统化应用和层次化扩展是创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、多压缩率验证，包含 LongBench 全面评测和实际延迟测量
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，从观察到方案的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 直接解决产业界长上下文推理的核心痛点，开源代码和内核实现完善

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[ACL 2025\] Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](native_sparse_attention.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)
- [\[ICML 2025\] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling](../../ICML2025/llm_efficiency/efficient_length-generalizable_attention_via_causal_retrieval_for_long-context_l.md)

</div>

<!-- RELATED:END -->
