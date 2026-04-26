---
title: >-
  [论文解读] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments
description: >-
  [NeurIPS 2025][模型压缩][KV cache eviction] 提出 KeyDiff——一种无需注意力分数的 KV cache 驱逐策略，通过保留与其他 key 余弦相似度最低（即几何上最独特）的 key 来维护 cache，在严格内存约束的逐块推理场景下以 8K cache 在 LongBench 上仅损失 ≤0.04% 精度，同时端到端推理延迟减少最高 30%。
tags:
  - NeurIPS 2025
  - 模型压缩
  - KV cache eviction
  - key similarity
  - long-context inference
  - 注意力机制
  - 提示学习
---

# KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments

**会议**: NeurIPS 2025  
**arXiv**: [2504.15364](https://arxiv.org/abs/2504.15364)  
**代码**: 无  
**领域**: model_compression  
**关键词**: KV cache eviction, key similarity, long-context inference, attention-free, block prompt processing

## 一句话总结
提出 KeyDiff——一种无需注意力分数的 KV cache 驱逐策略，通过保留与其他 key 余弦相似度最低（即几何上最独特）的 key 来维护 cache，在严格内存约束的逐块推理场景下以 8K cache 在 LongBench 上仅损失 ≤0.04% 精度，同时端到端推理延迟减少最高 30%。

## 研究背景与动机

**领域现状**：KV cache 是 LLM 推理加速的标准技术，但其内存占用随上下文长度线性增长。已有 cache 驱逐方法（H2O、TOVA、SnapKV、StreamingLLM）通过注意力分数评估 token 重要性并驱逐不重要的 KV 对。

**现有痛点**：在资源受限环境（如边缘设备）中，必须采用逐块（block-wise）推理——将 prompt 分成小块依次处理。现有基于注意力分数的驱逐方法在此场景下严重退化，因为每块只能看到局部 token 的注意力权重，无法预知未来块中哪些 token 重要，误驱逐会跨块累积传播。

**核心矛盾**：注意力分数是 query-dependent 的，在逐块推理中只能基于当前块的局部 query 计算，导致驱逐决策短视。

**关键观察**：与其他 key 余弦相似度低（即几何上独特）的 key 倾向于获得高注意力分数——这一性质完全基于 key 自身，与 query 无关，在逐块场景下仍然有效。

**核心 idea**：用 key 之间的余弦相似度代替注意力分数作为驱逐标准——保留最"独特"的 key，驱逐冗余的。

## 方法详解

### 整体框架
在逐块推理中，每处理一个 block 后将新 KV 追加到 cache，若超过预算 $N$ 则调用驱逐策略。KeyDiff 计算每个 key 与 cache 中所有其他 key 的平均余弦相似度，保留相似度最低的 $N$ 个 KV 对（即最独特的 key），驱逐其余的。

### 关键设计

1. **KeyDiff 基本公式**：

    - 功能：给每个 cached key 打分，保留最独特的
    - 核心思路：$S = \text{topk}(-\text{CosSim}(K)\mathbf{1}, N)$，其中 $\text{CosSim}(K) \in \mathbb{R}^{n \times n}$ 是 key 间的成对余弦相似度矩阵，$\mathbf{1}$ 为全 1 向量。分数为负的行和（相似度越低分数越高）
    - 设计动机：与注意力分数不同，key 间相似度不依赖 query，在逐块推理中仍能准确评估全局重要性

2. **高效变体（线性复杂度）**：

    - 功能：将 $O(n^2)$ 的成对相似度计算降为 $O(n)$
    - 核心思路：$S = \text{topk}(-\text{CosSim}(\mu(\hat{K}), \hat{k}_i), N)$，其中 anchor 向量 $\mu(\hat{K}) = \frac{1}{n}\sum_i \hat{k}_i$ 是归一化 key 的均值。每个 key 只需与 anchor 算一次余弦相似度
    - 设计动机：在温和条件下与完整版保留相同的 KV 对（Appendix 证明）；实验中甚至可以用非归一化 key 的均值 $\mu(K)$ 而不损失精度

3. **理论支撑**：

    - **Lemma 3.1**：建立注意力权重 $w$ 与 key-query 余弦相似度的下界关系——$\frac{-\log(1-w)}{2M} - 1 \leq \text{CosSim}(k^*, q)$
    - **Theorem 3.2**：建立三角关系——若 key $k^*$ 与 query $q$ 高度对齐（$\text{CosSim}(k^*, q) = \beta_q > 0$）而 key 均值与 query 不对齐（$\text{CosSim}(\bar{k}, q) = \alpha_q < 0$），则 $\text{CosSim}(\bar{k}, k^*) \leq 1 + \alpha_q\beta_q - 0.5\alpha_q^2 - 0.5\beta_q^2$（趋向 -1）
    - 含义：高注意力的 key 与 anchor 向量余弦相似度趋向 -1，因此 KeyDiff 保留的恰好是与 query 最对齐的 key

4. **滑动窗口扩展**：

    - 功能：分配一部分 cache 预算给最近 token（sliding window）
    - 核心思路：在推理和代码等最近 token 更重要的任务中，将 cache 预算的一部分保留给最近的 token
    - 设计动机：对 DeepSeek-R1 等推理模型效果显著，无额外开销

### 与 FlashAttention 的兼容性
KeyDiff 不需要显式计算注意力矩阵 $A$，可以直接与 FlashAttention 等优化 kernel 配合使用——这是相比 H2O、SnapKV 等方法的重要实用优势。

## 实验关键数据

### 主实验 —— Llama 3.1-8B LongBench (B=128)

| 方法 | Cache | NarrQA | HotpotQA | GovReport | TriviaQA | PR-en | Avg |
|------|-------|--------|----------|-----------|----------|-------|-----|
| 无驱逐 | 全量 | 30.05 | 57.33 | 34.86 | 91.61 | 99.50 | 49.20 |
| H2O | 8K | 13.85 | 43.64 | 18.78 | 69.05 | 62.50 | 38.20 |
| TOVA | 8K | 24.86 | 54.52 | 33.44 | 91.11 | 87.00 | 47.09 |
| Sink | 8K | — | — | — | — | — | ~40 |
| **KeyDiff** | **8K** | **26.59** | **55.98** | **34.25** | **91.38** | **95.50** | **49.16** |

8K cache（~23% 压缩）下 KeyDiff 与无驱逐基线差距仅 0.04%。

### 不同 cache 预算消融

| 方法 | 2K | 4K | 6K | 8K |
|------|-----|-----|-----|-----|
| H2O | 16.89 | 26.37 | 33.19 | 38.20 |
| TOVA | 37.52 | 41.49 | 45.24 | 47.09 |
| KeyDiff | **38.75** | **44.02** | **47.70** | **49.16** |
| 无驱逐 | — | — | — | 49.20 |

6K cache（~33% 压缩）下 KeyDiff 精度损失仅 ≤1.5%。

### 推理效率

| 方法 | 端到端延迟 |
|------|-----------|
| TOVA | 基线 |
| SnapKV | ~基线 |
| **KeyDiff** | **降低 30%** |

KeyDiff 因无需计算注意力矩阵，兼容 FlashAttention，延迟显著更低。

### 关键发现
- H2O 在逐块推理中退化严重（2K cache 下仅 16.89 vs KeyDiff 的 38.75），证实了注意力分数在块间传播时不可靠
- KeyDiff 的 PCA 可视化显示其保留的 key 更分散多样，而 TOVA/Sink 保留的 key 聚集在一起
- DeepSeek-R1-Distill-Llama-8B 在 Math-500 推理任务上 KeyDiff+sliding window 接近无驱逐基线
- 负余弦相似度作为驱逐标准始终优于正相似度或欧氏距离等替代指标

## 亮点与洞察
- **"key 独特性 ≈ 注意力重要性"的洞察**：这是一个优雅的观察——高注意力 key 倾向于与其他 key 不同，因此可以不看 query 就判断 key 重要性
- **理论闭环**：Lemma 3.1 (注意力→key-query 相似度) + Theorem 3.2 (key-query→key-anchor 相似度) 完整建立了从注意力分数到 KeyDiff 分数的理论桥梁
- **FlashAttention 兼容**：不需要物化注意力矩阵，直接利用 key 几何信息，实际部署中非常实用
- **线性复杂度高效变体**：anchor 向量近似将 $O(n^2)$ 降到 $O(n)$，且几乎无精度损失

## 局限与展望
- 理论假设（如 $\text{CosSim}(\bar{k}, q) < 0$）可能不在所有 head/layer 成立
- 仅在 Llama 和 Qwen 系列上验证，对其他架构（如 Mistral、Phi）的泛化性未知
- 滑动窗口的比例需要手动调整，没有自适应策略
- 对于 GQA（Grouped Query Attention）的适配性未讨论
- 极端压缩（如 2K cache）下精度仍有明显下降，可能需要与量化等其他压缩方法组合

## 相关工作与启发
- **vs H2O**：H2O 累积注意力分数做驱逐，在逐块推理中依赖不完整的注意力信息导致大幅退化；KeyDiff 纯粹基于 key 几何，不受此限制
- **vs TOVA**：TOVA 保留对当前 token 注意力最高的 KV，仍是 query-dependent 的；KeyDiff 是 query-independent 的
- **vs SnapKV**：SnapKV 对窗口内 token 的注意力做聚类选择，需要物化注意力矩阵且不兼容 FlashAttention
- **vs StreamingLLM**：StreamingLLM 固定保留开头 sink token + 最近 token，策略简单但无法自适应；KeyDiff 动态选择最独特 key

## 评分
- 新颖性: ⭐⭐⭐⭐ 洞察简洁优雅，理论分析有深度，但核心操作（余弦相似度排序）较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖多模型、多 benchmark、消融详尽、效率对比到位
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论和实验结合紧密，可视化直观
- 价值: ⭐⭐⭐⭐⭐ 对边缘部署和长上下文推理有直接应用价值，FlashAttention 兼容性是杀手锏

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](../../ICML2025/model_compression/rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ACL 2025\] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](../../ACL2025/model_compression/scope_optimizing_key-value_cache_compression_in_long-context_generation.md)

<!-- RELATED:END -->
