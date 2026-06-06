---
title: >-
  [论文解读] Draft-based Approximate Inference for LLMs
description: >-
  [ICLR 2026][模型压缩][近似推理] 提出 Draft-based Approximate Inference 框架，利用小型 draft 模型的前瞻（lookahead）预测来更准确地估计 token/KV pair 重要性…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "近似推理"
  - "KV cache压缩"
  - "提示学习"
  - "草稿模型"
  - "稀疏注意力"
---

# Draft-based Approximate Inference for LLMs

**会议**: ICLR 2026  
**arXiv**: [2506.08373](https://arxiv.org/abs/2506.08373)  
**代码**: [GitHub](https://github.com/furiosa-ai/draft-based-approx-llm)  
**领域**: 模型压缩  
**关键词**: 近似推理, KV cache压缩, prompt压缩, 草稿模型, 稀疏注意力

## 一句话总结

提出 Draft-based Approximate Inference 框架，利用小型 draft 模型的前瞻（lookahead）预测来更准确地估计 token/KV pair 重要性，包含 SpecKV（KV cache dropping）、SpecPC（prompt 压缩）和 SpecKV-PC（级联压缩）三种方法，在长上下文 benchmark 上一致优于现有基线。

## 研究背景与动机

长上下文 LLM 推理面临两大瓶颈：注意力计算随上下文长度二次增长，KV cache 内存线性增长（128K token 在 Llama-3.1-8B 上需 16GB+）。现有近似推理方法包括 KV cache dropping（H2O、SnapKV）、稀疏注意力（MInference）和 prompt 压缩（LLMLingua-2），但它们都依赖当前输入 token 的注意力激活来估计重要性——这本质上是"后视镜"策略，无法准确预测**未来生成 token 真正需要哪些 KV pair**。

核心矛盾：重要性估计需要未来信息，但未来 token 尚未生成。LAQ++ 尝试用 target 模型自身的稀疏近似来生成 draft query，但它需要存储完整的 target KV cache，无法降低峰值内存。

本文的切入角度：用一个轻量级 draft 模型（如 0.5B-3B）来生成前瞻 token，以极低开销获取近似的未来信息，从而更准确地估计 token 重要性，同时避免 target 模型的内存和计算负担。

## 方法详解

### 整体框架

Draft-based Approximate Inference 是一个统一框架：先用小型 draft 模型对输入生成前瞻 token，然后利用这些前瞻信息（draft 输出或 draft 注意力激活）来指导 target 模型的 KV cache 或 prompt 压缩。与 speculative decoding 不同，本框架的目标是**减少 target 模型的总计算和内存**，而非加速验证。

### 关键设计

1. **SpecKV（Speculative KV Dropping）**: 用 draft 模型生成 $n_{\text{lookahead}}$ 个前瞻 token，然后将输入 token 和前瞻 token 一起传入 target 模型做 prefill。对每个注意力头，通过最后 $n_{\text{window}}$ 个输入 token 和前瞻 token 的 query 对其余输入 key 的交叉注意力来估计 KV pair 重要性。保留 top-$C_{\max}$ 个 KV pair 加窗口内的 KV pair。相比 LAQ++ 的优势是不需要存储完整 target KV cache，实现了峰值内存的真正降低。理论保证（Theorem 1）：重要性分数的误差与 draft 嵌入误差成正比，即 $\|s - \hat{s}\|_2 \leq \epsilon \|W_q W_k^T\|_2$。

2. **SpecPC（Speculative Prompt Compression）**: 将完整 prompt 送入 draft 模型，直接提取 draft 模型的注意力激活矩阵 $A \in \mathbb{R}^{n_{\text{layer}} \times n_{\text{head}} \times (n_{\text{in}}+n_{\text{lookahead}}-1) \times n_{\text{in}}}$ 来估计 token 重要性。采用大窗口 + 非均匀权重（离末尾越远权重越低），跳过前 $l_{\text{skip}}$ 层（浅层注意力不够聚焦），先平均后取最大聚合。理论保证（Theorem 2）：在输入满足 RIP 条件下，注意力近似误差与输出近似误差成正比。

3. **SpecKV-PC（级联压缩）**: 先用 SpecPC 压缩 prompt（如压缩到 2048 token），再用 SpecKV 进一步压缩 KV cache（如到 256）。由于 target 模型只需处理压缩后的短 prompt，延迟和内存显著降低。级联压缩的效果甚至优于单独的 SpecKV，因为 SpecPC 作为预过滤器去掉了明显不重要的 token。

### 损失函数 / 训练策略

- 无需训练：所有方法都是 training-free 的推理时优化
- SpecKV 结合稀疏 prefill（Vertical-Slash 模式）+ KV cache dropping
- SpecPC 使用局部池化保持 token 连续性，避免静态分块

## 实验关键数据

### 主实验

**表1: LongBench 性能对比（Qwen2.5 32B，KV cache $C_{\max}$=256）**

| 类别 | 方法 | SingleQA | MultiQA | Summ. | Few-shot | Code | All |
|------|------|----------|---------|-------|----------|------|-----|
| Dense | Target | 56.01 | 43.99 | 25.90 | 64.06 | 44.74 | 47.78 |
| KV | SnapKV | 52.54 | 40.21 | 19.89 | 61.18 | 40.12 | 42.98 |
| KV | LAQ++ | 55.15 | 44.14 | 22.24 | 63.25 | 41.19 | 45.79 |
| KV | **SpecKV** | 53.48 | 43.77 | 24.02 | 63.79 | 44.80 | **46.06** |
| KV | **SpecKV-PC** | 52.60 | 44.52 | 24.11 | 63.38 | 48.45 | **46.48** |

**表2: Prompt 压缩对比（$C_{\max}$=1024）**

| 方法 | SingleQA | MultiQA | Summ. | Few-shot | Code | All |
|------|----------|---------|-------|----------|------|-----|
| LLMLingua-2 | 33.83 | 26.39 | 22.85 | 32.46 | 43.01 | 30.90 |
| SpecPrefill | 45.94 | 39.32 | 23.16 | 62.04 | 43.17 | 42.70 |
| **SpecPC** | 51.23 | 41.40 | 23.37 | 62.26 | 38.23 | **43.66** |

### 消融实验

- **Lookahead 长度**: SpecKV 用最大 token limit 效果最好，SpecPC 用 1 即可
- **Draft 模型大小**: 更大的 draft 模型（1.5B vs 0.5B）降低 $\epsilon$，提升下游分数
- **重要性分数相关性**: Draft (1.5B) 与 target (14B) 的 token 重要性分数高度相关（R² 接近 1）

### 关键发现

- SpecKV-PC 级联压缩优于单独 SpecKV，表明 SpecPC 的预过滤是有效的
- SpecKV 在 64K 上下文时比 LAQ++ 延迟降低 75%，峰值内存节省约 25GB
- 所有方法性能远超 draft 模型本身，说明该框架对弱 draft 模型也鲁棒
- RULER 合成 benchmark 上，随上下文增长，基于 lookahead 的方法优势越发明显

## 亮点与洞察

- 统一框架将 KV cache dropping 和 prompt 压缩纳入同一 draft-based 体系
- 理论分析优雅地连接了 compressed sensing（RIP）与注意力近似误差
- SpecKV 是首个利用 draft model lookahead 做 KV cache 优化的方法
- 级联压缩的设计思路值得借鉴：先粗后细，每步用最合适的信号

## 局限与展望

- Draft 模型引入额外的内存开销（虽然可以 offload 到 CPU）
- 在非常短的上下文（<4K）下，draft 开销可能不值得
- 目前仅验证了同系列模型做 draft（如 Qwen2.5-0.5B→32B），跨系列有效性未知
- Prompt 压缩会重新分配 position ID，可能影响依赖绝对位置的模型

## 相关工作与启发

- **SnapKV** (Li et al., 2024): 基于最后几个 token 的注意力做 KV 压缩，SpecKV 在此基础上引入前瞻
- **LAQ++** (Wang et al., 2025): 也用 lookahead，但需 target 模型完整 KV cache，峰值内存无法降低
- **SpecPrefill** (Liu et al., 2025): SpecPC 的前身，窗口大小固定为 1，SpecPC 改进为大窗口+非均匀权重
- 启发：draft model 除了 speculative decoding 外，在推理优化中有广泛应用前景

## 评分

- 新颖性: ⭐⭐⭐⭐ 框架统一性强，SpecKV 首创 draft lookahead 做 KV dropping，但每个单独方法增量有限
- 实验充分度: ⭐⭐⭐⭐⭐ RULER+LongBench+多模型+效率测量+多模态扩展，非常全面
- 写作质量: ⭐⭐⭐⭐ 理论分析清晰，但方法组合多导致篇幅略长
- 价值: ⭐⭐⭐⭐ 长上下文推理优化的实用方案，级联压缩思路有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](../../NeurIPS2025/model_compression/universal_cross-tokenizer_distillation_via_approximate_likelihood_matching.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](../../ACL2025/model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](steering_moe_llms_via_expert_deactivation.md)
- [\[ICLR 2026\] Highly Efficient and Effective LLMs with Multi-Boolean Architectures](highly_efficient_and_effective_llms_with_multi-boolean_architectures.md)
- [\[ICLR 2026\] AMiD: Knowledge Distillation for LLMs with α-mixture Assistant Distribution](amid_knowledge_distillation_for_llms_with_α-mixture_assistant_distribution.md)

</div>

<!-- RELATED:END -->
