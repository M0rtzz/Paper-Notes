---
title: >-
  [论文解读] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding
description: >-
  [ICLR 2026][LLM 幻觉控制] 提出 Token-Guard，一种基于自检验解码的 token 级幻觉控制方法，通过隐空间中的 token 级/段级评分和迭代修正机制，在解码过程中检测并抑制幻觉生成，F1 平均提升 16.3%。
tags:
  - ICLR 2026
  - LLM 幻觉控制
  - Token 级解码
  - 自检验
  - 段级评分
  - 迭代修正
---

# Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding

**会议**: ICLR 2026  
**arXiv**: [2601.21969](https://arxiv.org/abs/2601.21969)  
**代码**: [https://github.com/rhq945/Token-Guard](https://github.com/rhq945/Token-Guard)  
**领域**: 模型压缩  
**关键词**: LLM 幻觉控制, Token 级解码, 自检验, 段级评分, 迭代修正

## 一句话总结

提出 Token-Guard，一种基于自检验解码的 token 级幻觉控制方法，通过隐空间中的 token 级/段级评分和迭代修正机制，在解码过程中检测并抑制幻觉生成，F1 平均提升 16.3%。

## 研究背景与动机

- **LLM 幻觉问题**: 大模型常生成与输入不一致的内容，在知识密集型场景尤为严重
- **现有方法的不足**:
    - RAG 和 RLHF 需要昂贵的外部检索或大规模微调
    - 现有解码方法（CoT、ToT 等）缺乏显式 token 级幻觉检查机制
    - 幻觉风险未被显式量化，token 选择缺乏方向性
    - 大多数方法仅支持单次生成，缺乏动态修正能力
- **核心挑战**: 如何在解码阶段以低开销实现精细化的幻觉控制？

## 方法详解

### 整体框架

Token-Guard 包含三个层级的幻觉控制：
1. Token 级自检验 → 2. 段级表示与评分 → 3. 全局迭代修正

### 关键设计 1: Token 级幻觉自检验

为每个候选 token 计算混合幻觉评分：

$$F_{\text{halu}}^{\text{token}}(a_t^{(i)} \mid s_t) = \lambda \cdot \frac{h_t^{(i)} \cdot \bar{h}_{<t}}{|h_t^{(i)}| |\bar{h}_{<t}|} + (1-\lambda) \cdot P(a_t^{(i)} \mid a_{<t}, x)$$

- 第一项：候选 token 隐状态与已接受 token 平均隐状态的余弦相似度（语义一致性）
- 第二项：模型分配的条件概率（token 概率）
- $\lambda = 0.6$ 平衡两项
- 阈值 $\tau_{\text{token}} = 0.4$，低于阈值的 token 被丢弃

隐状态来自模型倒数第二层 $\text{LLM}_{\text{hidden}}^{(L-1)}$，首个 token 使用输入上下文平均隐状态作为锚点。

### 关键设计 2: 段级候选表示与评分

通过 token 的连续性形成候选段 $C_k$，使用加权平均计算段表示：

$$H_k = \sum_{i=1}^{n} w_i h_t^{(i)}, \quad w_i = \frac{\exp(F_{\text{halu}}^{\text{token}}(a_{t_i} \mid s_{t_i}))}{\sum_j \exp(F_{\text{halu}}^{\text{token}}(a_{t_j} \mid s_{t_j}))}$$

段级评分综合三个维度：

$$F_{\text{halu}}^{\text{seg}}(C_k) = \alpha F_{\text{halu}}^{\text{token}}(C_k) + \beta \text{Consistency}(C_k) + \gamma \text{Alignment}(C_k)$$

- **Token 聚合**: 加权 token 可靠性 ($\alpha = 0.5$)
- **局部一致性**: 相邻 token 隐状态平滑度 ($\beta = 0.3$)
- **全局对齐**: 与输入上下文的语义对齐 ($\gamma = 0.2$)

段级阈值：$\tau_{\text{seg}}^{\text{low}} = 0.55$（丢弃），$\tau_{\text{seg}}^{\text{high}} = 0.75$（接受），介于之间则局部修正。

### 关键设计 3: 局部修正与全局迭代

**局部修正**: 定位段内最低分 token 及其邻居窗口 $W_k^{(l)}$，用 LLM 重新生成条件化于周围上下文的替代 token：

$$W_k^{(l)'} = \text{LLM\_refine}(W_k^{(l)} \mid a_{<i-1}, a_{>i+1}, H_k)$$

**全局迭代**: 将可靠段组装为推理链 $R$，计算全局评分：

$$F_{\text{global}}(R) = \frac{F_{\text{fact}}(R) \cdot F_{\text{logic}}(R)}{F_{\text{fact}}(R) + F_{\text{logic}}(R) - F_{\text{fact}}(R) \cdot F_{\text{logic}}(R)}$$

若 $F_{\text{global}} < 0.7$ 则触发全局重新生成；若 $F_{\text{fact}}$ 和 $F_{\text{logic}}$ 均低于 0.5，输出"无法回答"。

### 内存效率

- Token 级：仅维护运行平均 $\bar{h}_{<t}$，复杂度 $\mathcal{O}(L_{\max} \cdot K_{\text{active}} \cdot d)$
- 段级：段形成后释放临时隐状态，仅保留紧凑段向量
- 全局：仅操作段向量 $\{H_k\}$，复杂度 $\mathcal{O}(K \cdot d)$

## 实验

### 主实验（Meta-Llama-3.1-8B-Instruct）

| 方法 | FinanceBench F1 | DROP_hist F1 | DROP_nfl F1 | HaluEval F1 | Avg F1 |
|------|----------------|-------------|-------------|-------------|--------|
| BaseModel | 16.00 | 44.21 | 39.10 | 42.16 | 28.29 |
| Guided Decoding | 16.44 | 55.95 | 36.71 | 57.41 | 34.73 |
| Chain-of-Thoughts | 11.01 | 49.26 | 49.21 | 55.32 | 34.63 |
| Tree-of-Thought | 14.44 | 47.73 | 37.69 | 56.02 | 33.33 |
| **Token-Guard** | **30.80** | **68.52** | **58.10** | **78.54** | **51.03** |

### Qwen3-8B 结果

| 方法 | Avg EM | Avg F1 |
|------|--------|--------|
| BaseModel | 0.22 | 44.25 |
| CoT | 0.23 | 45.10 |
| **Token-Guard** | **0.35** | **53.98** |

### 消融实验

| 变体 | DROP_hist F1 | RAGTruth F1 | Avg BLEU |
|------|-------------|-------------|----------|
| Full Token-Guard | **68.52** | **43.94** | **51.74** |
| w/o Token-Level | 47.51 | 27.10 | 34.97 |
| w/o Segment-Level | 60.10 | 39.20 | 46.32 |
| w/o Global Iteration | 63.05 | 41.05 | 36.26 |
| w/o Prompt | 55.23 | 32.50 | 39.70 |

### 关键发现

- Token 级评分对性能贡献最大（移除后 F1 下降最多）
- 全局迭代主要提升 BLEU（语言流畅性），对 EM/F1 也有贡献
- 在需要多步推理的任务（DROP_nfl）上优势最大
- 在知识密集型任务（PubMedQA）上改进有限，因为无法补偿缺失领域知识
- 两个骨干模型（Llama3.1-8B、Qwen3-8B）上均有效

## 亮点

- **多层次幻觉控制**: Token→段→全局三级递进，兼顾精度和效率
- **无需外部资源**: 不需要检索系统或额外训练，纯解码阶段方案
- **模块化设计**: 可作为插件集成到任何 LLM 解码管线
- **内存友好**: 巧妙的状态管理使内存与生成长度无关

## 局限性

- 多级评分引入额外计算开销（每个 token 需多次隐状态计算和余弦相似度计算）
- 超参数较多（$\lambda$、$\tau_{\text{token}}$、$\alpha/\beta/\gamma$、$\tau_{\text{seg}}$、$\tau_{\text{global}}$ 等），调参复杂
- 基于隐状态相似度的幻觉检测假设"与上下文一致=真实"，可能在模型本身有知识错误时失效
- 仅在 8B 级别模型上验证，对更大/更小模型的适用性未知
- 全局迭代使用 TF-IDF + KMeans 聚类，引入了传统 NLP 方法的额外依赖

## 相关工作

- **RAG 方法**: 外部检索增强，计算密集且领域相关
- **RLHF/对齐方法**: 需大规模微调，资源消耗大
- **解码方法**: DoLa（层间对比）、KCTS（知识约束树搜索）、Phi-Decoding（前瞻采样）
- **Token-Guard**: 首个融合 token 级自检验、段级评分和全局迭代的统一幻觉控制框架

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ★★★★☆ |
| 理论深度 | ★★★☆☆ |
| 实验充分性 | ★★★★☆ |
| 实用价值 | ★★★★☆ |
| 写作质量 | ★★★☆☆ |

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](../../ACL2026/information_retrieval/tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ICCV 2025\] Representation Shift: Unifying Token Compression with FlashAttention](../../ICCV2025/information_retrieval/representation_shift_unifying_token_compression_with_flashattention.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[ICLR 2026\] Digging Deeper: Learning Multi-Level Concept Hierarchies](digging_deeper_learning_multi-level_concept_hierarchies.md)

<!-- RELATED:END -->
