---
title: >-
  [论文解读] RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding
description: >-
  [ICML 2025][推测解码] 提出 RAPID，将 RAG 与 Speculative Decoding 结合：用 RAG drafter（在短检索上下文上运行的 LLM）为长上下文目标 LLM 生成候选 token，并通过推理时知识迁移增强目标分布，在长上下文推理中同时实现 >2× 加速和生成质量提升。
tags:
  - ICML 2025
  - 推测解码
  - 长上下文推理
  - 检索增强生成
  - 知识蒸馏
  - KV Cache
---

# RAPID: Long-Context Inference with Retrieval-Augmented Speculative Decoding

**会议**: ICML 2025  
**arXiv**: [2502.20330](https://arxiv.org/abs/2502.20330)  
**代码**: [https://github.com/NUS-TRAIL/RAPID](https://github.com/NUS-TRAIL/RAPID)  
**领域**: LLM效率  
**关键词**: 推测解码, 长上下文推理, 检索增强生成, 知识蒸馏, KV Cache

## 一句话总结

提出 RAPID，将 RAG 与 Speculative Decoding 结合：用 RAG drafter（在短检索上下文上运行的 LLM）为长上下文目标 LLM 生成候选 token，并通过推理时知识迁移增强目标分布，在长上下文推理中同时实现 >2× 加速和生成质量提升。

## 研究背景与动机

传统 Speculative Decoding (SD) 通过小模型草稿+大模型验证来加速推理，但在**长上下文场景下效果大幅下降**。核心原因：

**KV Cache 瓶颈**：长上下文下 KV Cache 操作变为内存受限（memory-bound），小模型相对大模型的速度优势急剧缩小。例如 LLaMA-3.1-8B 相比 70B 的吞吐量优势从 1K 上下文的 23.6× 降至 128K 的 9.4×。

**RAG 与长上下文各有优劣**：RAG 在检索相关片段上表现优异（如多选题），但在需要全局理解的任务上（如问答）不及长上下文 LLM。两者存在互补性，但此前没有好的方式在推理时融合。

**SD 的严格拒绝问题**：传统 SD 以目标 LLM 分布为"真值"做拒绝采样，当 RAG drafter 生成质量实际更优时，好的候选会被不必要地拒绝，浪费计算。

**关键观察**：在 128K 上下文下，LLaMA-3.1-8B 用 RAG 处理 4K~16K 检索上下文就能恢复大部分长上下文性能——这意味着 RAG drafter 能高效生成高质量候选 token。

## 方法详解

### 整体框架

RAPID 由两个核心组件构成：

1. **RAG Drafter**：用检索增强的短上下文 LLM 替代传统小模型作为草稿模型，在压缩后的检索上下文 $\mathcal{C}^S$ 上生成候选 token
2. **Retrieval-Augmented Target Distribution**：通过推理时知识蒸馏，将 RAG drafter 的知识迁移到目标分布中，提高候选接受率

工作流程：RAG drafter 在短上下文上快速生成 γ 个候选 token → 目标 LLM 在完整长上下文上一次性验证 → 用增强后的目标分布做拒绝采样。

RAPID 支持两种设定：

- **Self-speculation**：目标 LLM 和 RAG drafter 同等规模（如 8B-8B）
- **Upward-speculation**：RAG drafter 比目标 LLM 更大（如 70B drafter → 8B target），因为 RAG drafter 处理短上下文，计算开销仍可控

### 关键设计

#### 1. RAG Drafter 的构建

将长上下文 $\mathcal{C}$ 分为 512-token 的 chunk，用 BGE-M3 编码为向量，按余弦相似度检索 top-k 相关片段（相似度阈值 0.3），构建压缩上下文 $\mathcal{C}^S$。严格控制压缩比 $|\mathcal{C}^S| \leq |\mathcal{C}|/\lambda$（$\lambda \gg 1$），检索长度限制在 4096 到原始长度的 1/24 之间。

Draft 分布定义为：

$$q(x_i) = q_{\psi}(x_i | [\mathcal{C}^S; x_{<i}])$$

核心优势：(1) 消除了长上下文的 KV Cache 内存瓶颈；(2) 通过聚焦相关信息可能产生更高质量的候选。

#### 2. Retrieval-Augmented Target Distribution

传统 SD 直接用目标分布 $p(x_i)$ 做拒绝采样，但 RAG drafter 在某些场景下质量更优。RAPID 将 RAG drafter 反转为"教师"，在推理时做一步知识蒸馏来增强目标分布：

$$\hat{z}(x_i) = z(x_i) + \eta T(q(x_i) - p(x_i))$$

其中 $z(x_i)$ 是目标 LLM 的原始 logits，$\eta$ 控制知识迁移强度，$T$ 是温度。增强后的目标分布为：

$$\hat{p}(x_i) = \text{softmax}(\hat{z}(x_i) / T)$$

**数学推导**：这等价于对 KL 散度蒸馏损失 $\mathcal{L} = T^2 \cdot \text{KL}(q \| p)$ 做一步梯度下降。梯度为 $\partial\mathcal{L}/\partial z = T(p - q)$，因此 $\hat{z} = z - \eta \cdot \partial\mathcal{L}/\partial z = z + \eta T(q - p)$。

#### 3. 尾部分布保护

为防止知识迁移扭曲长尾分布，对概率值低于峰值 10% 的 token 保持原始目标分布不变：

$$\hat{w}_k = w_k, \quad \forall k: \hat{w}_k < 0.1 \cdot \max_j \hat{w}_j$$

#### 4. 调整的残差采样

拒绝时从调整后的残差分布采样，保证理论上等价于从原始目标分布直接采样：

$$x_i \sim \text{norm}(\max(p(x_i) - \hat{p}(x_i), p(x_i) - q(x_i), 0))$$

### 损失函数 / 训练策略

RAPID **无需任何训练**，是纯推理时方法（drop-in decoding method）。超参数设置：

- 每步生成 γ=10 个候选 token
- Self-speculation: η ∈ {5, 10, 20}
- Upward-speculation: η ∈ {40, 50}

## 实验关键数据

### 主实验

在 ∞Bench 和 LongBench v2 上评估，模型包括 LLaMA-3.1 和 Qwen2.5 系列：

| 配置 (Target → Draft) | ∞Bench AVG | LongBench v2 CoT | 加速比 |
|---|---|---|---|
| LLaMA-3.1-8B LC (baseline) | 39.33 | 30.4% | 1.00× |
| LLaMA-3.1-8B RAG | 40.40 | 33.4% | 3.35× |
| LLaMA-3.1-8B + SD | 39.64 | 31.0% | 1.63× |
| LLaMA-3.1-8B + MagicDec | 37.35 | 30.6% | 0.71× |
| **RAPID self (8B→8B)** | **42.83** | **34.2%** | **2.10×** |
| **RAPID upward (8B→70B)** | **49.98** | **40.2%** | 1.14× |
| LLaMA-3.1-70B LC | 45.07 | 36.2% | 1.00× |
| **RAPID self (70B→70B)** | **50.62** | **40.2%** | **2.69×** |
| Qwen2.5-7B LC | 38.12 | 33.2% | 1.00× |
| **RAPID self (7B→7B)** | **42.48** | **35.4%** | **2.65×** |
| **RAPID upward (7B→72B)** | **48.72** | **41.2%** | 0.93× |

### 消融实验

**η 参数鲁棒性**（使用不相关检索上下文的压力测试）：

| η | Self-spec ΔAcc | Self-spec 加速 | Upward-spec ΔAcc | Upward-spec 加速 |
|---|---|---|---|---|
| 0 | +1.20 | 1.62× | -1.30 | 0.67× |
| 5 | +2.80 | 1.75× | +0.40 | 0.69× |
| 10 | +1.60 | 1.77× | +1.20 | 0.72× |
| 20 | +1.20 | 1.78× | +4.40 | 0.75× |
| 40 | -2.60 | 2.08× | +6.60 | 0.84× |
| 50 | -6.30 | 2.10× | +6.00 | 0.87× |

**多轮对话实验**（MT-Bench-101，122K 上下文）：

| 方法 | 质量评分 (1-10) | 接受率 | 吞吐量 (tok/s) |
|---|---|---|---|
| Target LLM | 2.82 | - | 10.64 |
| RAG Drafter | 3.95 | - | 40.49 |
| SD | 2.94 | 56.34% | 14.07 |
| **RAPID** | **4.21** | **76.94%** | **18.18** |

### 关键发现

1. **Self-speculation 全面超越 LC 和 RAG**：RAPID 在 ∞Bench 上将 LLaMA-3.1-8B 从 39.33 提升到 42.83，同时有 2.10× 加速
2. **Upward-speculation 效果惊人**：8B 目标 + 70B RAG drafter 达到 49.98（∞Bench），甚至超过 70B 的 LC 结果（45.07）
3. **涌现能力**：RAPID 能正确回答**目标 LLM 和 RAG drafter 都答错**的问题，两个模型的协同产生了 1+1>2 的效果
4. **鲁棒性**：即使用完全不相关的检索上下文，RAPID 在合适的 η 下仍能维持正向增益
5. **32K 拐点**：上下文长度超过 32K 时 RAPID 即可实现加速，而传统 SD 需要 64K 以上

## 亮点与洞察

1. **范式创新**：首次提出"用 RAG 做 SD drafter"的思路，将 RAG 的效率优势和 SD 的质量保证巧妙结合
2. **Upward-speculation 打破常规**：传统 SD 必须用小模型做 drafter，RAPID 允许大模型做 drafter（因为操作在短上下文上），开辟了新范式
3. **推理时知识蒸馏**：无需训练，仅通过 logits 空间的一步梯度更新就实现知识迁移，优雅且高效
4. **理论保证**：证明了调整后的残差采样仍等价于从原始目标分布采样，保持了 SD 的无损性质
5. **实用性强**：drop-in 方法无需修改模型权重，可直接用于现有长上下文 LLM 部署

## 局限与展望

1. **Upward-speculation 需要额外 GPU**：虽然大 RAG drafter 效果显著，但需要额外的 GPU 资源来部署大模型
2. **检索器质量依赖**：虽然论文展示了对检索质量的鲁棒性，但极端差的检索仍会影响性能
3. **η 需要调参**：self-speculation 和 upward-speculation 的最优 η 范围不同，需要针对性调整
4. **Prefill 阶段无加速**：RAPID 主要加速解码阶段，prefill 阶段的时间开销几乎不变（目标 LLM 仍需处理全部上下文）
5. **可探索动态 η**：当前 η 为固定值，可以考虑根据检索质量或生成位置动态调整

## 相关工作与启发

- **TriForce / MagicDec**：通过 KV Cache 压缩做长上下文 SD，但压缩会削弱 drafter 质量；RAPID 的 RAG drafter 方案更优
- **REST (He et al., 2024)**：从语料库检索 continuation 作为候选，与 RAPID 的"检索上下文+模型生成"思路互补
- **Speculative RAG (Wang et al., 2024)**：用 SD 思路提升 RAG 质量，RAPID 反过来用 RAG 提升 SD 效率
- **启发**：这种"在推理时通过 logits 操作实现模型融合"的范式可能推广到其他场景，如多模型协作、专家混合推理

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — RAG + SD 的结合思路非常巧妙，upward-speculation 范式令人耳目一新
- **技术深度**: ⭐⭐⭐⭐ — 推理时知识蒸馏有完整数学推导和理论保证
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个模型×2种设定×2个基准，消融实验覆盖鲁棒性/上下文长度/检索长度
- **实用价值**: ⭐⭐⭐⭐⭐ — 无需训练的 drop-in 方法，直接可用于长上下文推理加速
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，motivation 和 insight 都很到位

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] FlashBack: Efficient Retrieval-Augmented Language Modeling for Fast Inference](../../ACL2025/information_retrieval/flashbackefficient_retrieval-augmented_language_modeling_for_long_context_infere.md)
- [\[ACL 2025\] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](../../ACL2025/information_retrieval/hierarchical_document_refinement_for_long-context_retrieval-augmented_generation.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](../../ACL2026/information_retrieval/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](../../ACL2025/information_retrieval/gor_rag_long_context_summary.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](../../ACL2025/information_retrieval/seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)

<!-- RELATED:END -->
