---
title: >-
  [论文解读] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs
description: >-
  [ICML 2025][人体理解] FlashTrace 提出了一种高效的多 token 归因方法，通过跨度聚合（span-wise aggregation）将多 token 目标的归因复杂度从 $\mathcal{O}(M \cdot N)$ 降至 $\mathcal{O}(N)$，并通过递归归因（recursive attribution）机制追溯推理链中的重要性传播，实现了 130 倍以上的速度提升。
tags:
  - ICML 2025
  - 人体理解
---

# Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs

**会议**: ICML 2025  
**arXiv**: [2602.01914](https://arxiv.org/abs/2602.01914)  
**领域**: 人体理解  

## 一句话总结

FlashTrace 提出了一种高效的多 token 归因方法，通过跨度聚合（span-wise aggregation）将多 token 目标的归因复杂度从 $\mathcal{O}(M \cdot N)$ 降至 $\mathcal{O}(N)$，并通过递归归因（recursive attribution）机制追溯推理链中的重要性传播，实现了 130 倍以上的速度提升。

## 研究背景与动机

随着现代 LLM 越来越依赖扩展推理链（如 OpenAI o1、DeepSeek-R1），现有的 token 归因方法面临两大关键挑战：

**效率瓶颈**：归因一个长度为 $M$ 的目标跨度需要 $\mathcal{O}(M \cdot N)$ 操作。对于 5K token 的生成，Integrated Gradients 需要超过 10 小时。

**忠实度下降**：中间推理 token 吸收了归因质量（attribution mass），阻止了重要性从推理链传播回原始输入。

论文的实验验证了这两个问题：
- **发现1**：推理 token 吸收了大部分归因质量。随着推理链增长，分配给推理 token $\mathbf{T}$ 的重要性比例从约 80% 增长到超过 90%。
- **发现2**：推理链降低了输入上的归因质量。ground-truth 关键输入 token 的恢复率从 26% 下降到低于 10%。

## 方法详解

### 理论框架

FlashTrace 基于 ALTI/IFR 框架，使用基于 L1 范数的接近度度量：

$$\text{Proximity}(\mathbf{z}, \mathbf{y}) = \max(0, -\|\mathbf{y} - \mathbf{z}\|_1 + \|\mathbf{y}\|_1)$$

直觉上，衡量移除贡献 $\mathbf{z}$ 后目标向量 $\mathbf{y}$ 幅度减少了多少。

### 跨度聚合（Span-wise Aggregation）

**核心创新**：不逐个 token 计算，而是对整个目标跨度一次性计算归因。

定义聚合目标：$\mathbf{Y}_S = \sum_{i \in S} \mathbf{y}_i$

聚合贡献：$\mathbf{Z}_S = \sum_{i \in S} \mathbf{z}_{j \to i}$

**关键**：利用注意力机制的线性性进行因式分解。变换向量 $\mathbf{v}_j$ 仅依赖源 token $j$，与目标位置 $i$ 无关：

$$\mathbf{F}_{j \to S} = \sum_{i \in S}(\alpha_{i,j}^h \cdot \mathbf{v}_j) = \mathbf{v}_j \cdot \left(\sum_{i \in S} \alpha_{i,j}^h\right)$$

只需为每个源 token 计算一次昂贵的向量变换 $\mathbf{v}_j$，将复杂度从 $\mathcal{O}(M \cdot N)$ 降至 $\mathcal{O}(N)$。

### 递归归因（Recursive Attribution）

**第一跳归因**：对最终输出 $\mathbf{O}$ 进行标准归因，得到分布 $\mathbf{w}^{(0)}$。

**递归跳归因**：使用前一跳的重要性分数作为新目标跨度的权重：

$$\mathbf{Y}^{(1)} = \sum_{j \in \mathbf{T}} w_j^{(0)} \cdot \mathbf{y}_j$$

$$\mathbf{Z}^{(1)} = \sum_{j \in \mathbf{T}} w_j^{(0)} \cdot \mathbf{z}_{k \to j}$$

跨度聚合的效率优势在加权设定下保持：因式分解变为 $\mathbf{v}_k \cdot (\sum_{j \in \mathbf{T}} w_j^{(0)} \alpha_{j,k}^h)$。

### 最终归因组合

通过多跳归因，将输出归因经过推理链传播回原始输入：

$$\mathbf{w}_{\mathbf{I}}^{\text{final}} = \mathbf{w}_{\mathbf{I}}^{(0)} + \sum_{h=1}^{H} \gamma^h \cdot \mathbf{w}_{\mathbf{I}}^{(h)}$$

其中 $\gamma$ 是衰减因子，$H$ 是递归跳数。

## 实验

### RULER 基准测试：长上下文检索

| 指标 | 方法 | mq_q2 | mq_q4 | mv_v2 | mv_v4 |
|---|---|---|---|---|---|
| Recovery Rate ↑ | IFR | 0.471 | 0.328 | 0.575 | 0.452 |
| | AttnLRP | 0.215 | 0.204 | 0.254 | 0.243 |
| | **FlashTrace** | **0.483** | **0.413** | **0.556** | **0.516** |
| RISE ↓ | IFR | 0.075 | 0.115 | 0.069 | 0.073 |
| | **FlashTrace** | **0.068** | **0.113** | **0.069** | **0.070** |

### 推理任务：HotpotQA

| 方法 | Recovery Rate ↑ | RISE ↓ | MAS ↓ |
|---|---|---|---|
| Perturbation | 0.329 | 0.133 | 0.220 |
| CLP | 0.335 | 0.101 | 0.190 |
| IFR | 0.268 | 0.074 | 0.166 |
| AttnLRP | 0.189 | 0.155 | 0.249 |
| **FlashTrace** | **0.384** | **0.033** | **0.128** |

### 效率对比

FlashTrace 实现 **130 倍以上**的速度提升。对于 10K token 的推理链，朴素多跳方法需要数小时，而 FlashTrace 在秒级完成。

### 递归归因分析

- **跳 1→跳 2 的归因分布变化**：重要性从靠近输出的推理 token 转移到更早的推理 token 和输入上下文
- **即使仅一次递归跳也能显著改善忠实度**
- 改善效果在不同模型和数据分布上一致

## 亮点

- **优雅的理论推导**：利用注意力的线性性实现了从 $\mathcal{O}(M \cdot N)$ 到 $\mathcal{O}(N)$ 的复杂度降低
- **实用性强**：130 倍加速使得长推理链的归因从不可行变为实用
- **递归归因的通用性**：自然扩展到加权跨度设定，无额外计算开销
- **问题定义清晰**：系统化地形式化了推理 LLM 的多 token 归因问题
- **详尽的实验**：在长上下文检索、合成推理、多步 QA 等多种任务上验证

## 局限性

- 基于接近度的归因假设 L1 范数在高维空间中的有效性，可能不适用于所有场景
- 递归归因的跳数需要人工设定
- 跨度内 token 的聚合方式（求和）可能过于简单
- 未与基于梯度的方法（如 Integrated Gradients）在相同效率预算下系统比较
- 对于非自回归模型（如编码器-解码器架构）的适用性未探讨

## 评分

⭐⭐⭐⭐⭐ (5/5)

这是一项精致的工作：清晰的问题定义、优雅的理论推导、实用的技术方案和充分的实验验证。在推理 LLM 日益普及的背景下，解决其可解释性问题具有重要的时效性和实用价值。130 倍的速度提升使长推理链的归因首次变得可行。

<!-- RELATED:START -->

## 相关论文

- [LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)
- [FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)
- [TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)
- [Scaling Large Motion Models with Million-Level Human Motions](scaling_large_motion_models_with_million-level_human_motions.md)

<!-- RELATED:END -->
