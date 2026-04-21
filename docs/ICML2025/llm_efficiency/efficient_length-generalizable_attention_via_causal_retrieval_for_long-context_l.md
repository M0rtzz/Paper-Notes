---
title: >-
  [论文解读] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling
description: >-
  [ICML 2025][LLM效率][长度泛化] 本文提出 Grouped Cross-Attention (GCA) 机制，将 chunk 级别的因果检索（causal retrieval）集成到注意力中实现端到端可学习的检索器，构建的 Differentiable Retrieval-based Transformer (DRT) 在 16M 上下文的 passkey 检索测试中达到近乎完美的准确率，实现了训练长度 1000 倍的长度泛化。
tags:
  - ICML 2025
  - LLM效率
  - 长度泛化
  - 分组交叉注意力
  - 因果检索
  - 长上下文建模
  - 可微分检索
---

# Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling

**会议**: ICML 2025  
**arXiv**: [2410.01651](https://arxiv.org/abs/2410.01651)  
**代码**: https://github.com/ant-research/long-context-modeling (有)  
**领域**: LLM效率  
**关键词**: 长度泛化, 分组交叉注意力, 因果检索, 长上下文建模, 可微分检索

## 一句话总结
本文提出 Grouped Cross-Attention (GCA) 机制，将 chunk 级别的因果检索（causal retrieval）集成到注意力中实现端到端可学习的检索器，构建的 Differentiable Retrieval-based Transformer (DRT) 在 16M 上下文的 passkey 检索测试中达到近乎完美的准确率，实现了训练长度 1000 倍的长度泛化。

## 研究背景与动机
**领域现状**：Transformer 在 NLP 中表现优异，但处理超出预训练窗口的长上下文仍面临长度泛化和二次复杂度的双重挑战。

**现有痛点**：大部分长距离语言模型依赖扩大注意力窗口进行后训练，显著增加计算和内存开销。滑窗方法可以外推但无法捕捉窗口外的远距离依赖。

**核心矛盾**：检索增强语言模型（RLM）可以用固定窗口访问远距离信息，但现有 RLM 依赖预训练好的外部检索器（如 BM25、Contriever），检索到的 chunk 未必对因果语言模型有用，且检索器无法通过自回归损失进行梯度传播。

**本文目标**：如何让检索器端到端地学习检索对下一个 chunk 预测最有帮助的历史 chunk。

**切入角度**：将检索分数作为权重参与下一 token 预测（而非仅用于选择），使其可以接收自回归损失的梯度。

**核心 idea**：GCA 将 self-attention 中 token-to-token 的注意力范式泛化为 chunk-to-chunk 的检索+融合范式，检索分数作为 soft choice 融合各 chunk 的交叉注意力输出。

## 方法详解

### 整体框架
DRT 由 N 层 Transformer-like 层组成。输入序列被等分为多个 chunk（大小 S=64），每个 chunk 末尾插入特殊 LMK token 用于摘要。下层为标准 Transformer 层，上层额外加入 GCA 模块。上层进一步分为 G 组，每组进行独立检索。下层输出经双向 Transformer 编码器生成 chunk 表示和 landmark 表示，共享给所有上层使用。

### 关键设计

1. **Grouped Cross-Attention (GCA)**:

    - 功能：为当前 chunk 的每个 token 分别与各个检索到的 chunk 进行 Cross-Attention，然后用检索分数作为权重融合结果
    - 核心思路：GCA 对每个检索到的 chunk 独立计算 CA 输出，然后用 softmax 归一化的检索分数加权融合
    - 设计动机：与 Chunked Cross-Attention (CCA) 的关键区别——CCA 将所有检索 chunk 拼接后统一 softmax，检索分数不参与计算；GCA 对每个 chunk 独立 softmax，检索分数作为 soft choice 参与预测，因此可以接收梯度反传
    - KV 变换层间共享以节省参数和内存

2. **因果检索（Causal Retrieval）**:

    - 功能：学习检索那些能最有效降低下一个 chunk 自回归损失的历史 chunk
    - 核心思路：当前 chunk 的 landmark 表示与历史 chunk 的 landmark 表示计算相关性分数，选 top-k
    - 上层分 G 组，每组独立检索，高层组可基于前一组的检索结果进行多跳检索
    - 设计动机：RPT 依赖外部参考 LM 来标注好的 chunk 训练检索器，扩展性差；GCA 让检索器天然嵌入注意力结构中端到端训练

3. **Gumbel Top-k 采样**:

    - 功能：训练时在检索分数上加 Gumbel 噪声再做 top-k，平衡探索与利用
    - 核心思路：高分 chunk 仍最可能被选中，但低分 chunk 也有机会被探索
    - 设计动机：纯 top-k 可能陷入局部最优，Gumbel 噪声增加训练多样性

4. **内存卸载推理**:

    - 功能：将历史 chunk 表示卸载到 CPU 内存，检索时加载回 GPU
    - GPU 内存复杂度大幅降低
    - 每次检索仅在生成 S 个 token 后触发 G 次，交换成本极低

### 损失函数 / 训练策略
- 标准 next token prediction 损失
- 滑窗 self-attention (W=512) + top-k 检索 (K=8, S=64)，注意力域为 512 token
- 基于 Triton 的硬件感知 GCA 实现
- 训练复杂度随 chunk 级操作实现近线性扩展

## 实验关键数据

### 主实验（语言建模 Perplexity, 350M 模型, 16K 训练/评测）

| 模型 | 训练开销 | top-k | 窗口 | PG19 valid | PG19 test | ArXiv valid | ArXiv test |
|------|---------|-------|------|-----------|-----------|-------------|------------|
| BaseLM (SW+ALiBi) | 1x | - | 512 | 14.55 | 13.68 | 3.06 | 3.06 |
| BaseLM (+2 layers) | 1.15x | - | 658 | 14.23 | 13.37 | 2.95 | 2.94 |
| Landmark Attn | 1.5x | 4 | 768 | 14.10 | 13.21 | 3.02 | 3.02 |
| DRT_ret x1 | 1.22x | 8 | 512 | 14.05 | 13.21 | 2.89 | 2.89 |
| **DRT_ret x2** | **1.24x** | 8 | 512 | **14.02** | **13.18** | **2.85** | **2.85** |

### 单 Passkey 检索准确率（128M 模型）

| 模型 | 4K | 16K | 64K | 128K | 256K | 16M |
|------|------|------|------|-------|------|------|
| BaseLM (+2 layers) | 15.37 | 3.89 | 0.0 | - | - | - |
| Landmark Attn | 99.82 | 97.88 | 0.00 | 0.00 | - | - |
| **DRT_ret x1** | 98.50 | 98.59 | 100 | **100** | **100** | **100** |
| **DRT_ret x2** | 99.65 | 99.65 | 100 | **100** | **100** | **100** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o Triton | 训练开销 1.45x | Triton 实现将 GCA 开销降低约 16% |
| w/o Gumbel top-k | PPL 略高 (14.36 vs 14.05) | Gumbel 噪声有效改善检索质量 |
| w/ Contriever | PPL 14.55 vs 14.05 | 端到端因果检索远优于固定外部检索器 |
| w/ random retriever | PPL 14.53 vs 14.05 | 随机检索几乎等同于外部检索器 |

### 关键发现
- DRT 是首个在 16M 上下文长度下实现完美 passkey 检索的注意力机制（训练长度的 1000 倍）
- 多次检索 (G=2) 在 2-hop NIAH 上显著优于单次检索 (88.52% vs 41.07%)
- 端到端因果检索远优于外部检索器 (Contriever)，外部检索器甚至不如随机检索
- 推理效率：DRT 的推理时间和内存开销比 Landmark Attn 低一个量级

## 亮点与洞察
- GCA 优雅地将检索操作嵌入注意力机制，解决了检索分数无法接收梯度的核心困难
- chunk 级检索而非 token 级检索是长度泛化的关键——chunk 提供更丰富的语义信息
- case study 验证了因果检索概念：模型不仅检索语义相似内容，还检索对预测下一 chunk 有用的信息
- 1000 倍长度泛化的实现具有里程碑意义

## 局限与展望
- 目前仅在 128M~350M 模型上验证，大模型上的效果待探索
- chunk 大小 S 固定为 64，自适应 chunk 划分可能效果更好
- CPU 卸载虽然可行，但在大批量推理时的吞吐量影响需要评估
- 2-hop NIAH 在短上下文 (1K) 上与 Landmark Attn 有差距 (41% vs 91%)

## 相关工作与启发
- 与 RPT 的区别：RPT 需要外部参考 LM 标注好 chunk 训练检索器，DRT 端到端学习
- 与 Landmark Attention 的区别：LA 在每个 token、每层做 top-k 检索，开销大且无法外推
- GCA 中 softmax-off-by-one 的技巧允许 token 忽略所有检索 chunk，增加灵活性
- 因果检索概念可以推广到外部知识库检索场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（GCA 机制设计精巧，因果检索概念新颖）
- 实验充分度: ⭐⭐⭐⭐⭐（多数据集、多任务、完整消融、case study）
- 写作质量: ⭐⭐⭐⭐（结构清晰，图示直观）
- 价值: ⭐⭐⭐⭐⭐（1000x 长度泛化是重大突破）

<!-- RELATED:START -->

## 相关论文

- [Long-Short Alignment for Effective Long-Context Modeling in LLMs](long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [Curse of High Dimensionality Issue in Transformer for Long-context Modeling](curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)
- [Literary Evidence Retrieval via Long-Context Language Models](../../ACL2025/llm_efficiency/literary_evidence_retrieval_via_long-context_language_models.md)
- [Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](../../NeurIPS2025/llm_efficiency/long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)
- [Star Attention: Efficient LLM Inference over Long Sequences](star_attention_efficient_llm_inference_over_long_sequences.md)

<!-- RELATED:END -->
