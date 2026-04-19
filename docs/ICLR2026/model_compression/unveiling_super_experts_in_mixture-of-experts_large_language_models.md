---
title: >-
  [论文解读] Unveiling Super Experts in Mixture-of-Experts Large Language Models
description: >-
  [ICLR 2026][模型压缩][Mixture-of-Experts] 本文首次发现并系统研究了 MoE LLM 中的"超级专家"（Super Experts）——数量极少但对模型推理至关重要的专家子集，它们通过 down_proj 中的极端激活异常值驱动 massive activations 和 attention sinks 机制。
tags:
  - ICLR 2026
  - 模型压缩
  - Mixture-of-Experts
  - super experts
  - massive activations
  - 注意力机制
  - 剪枝
---

# Unveiling Super Experts in Mixture-of-Experts Large Language Models

**会议**: ICLR 2026  
**arXiv**: [2507.23279](https://arxiv.org/abs/2507.23279)  
**代码**: [GitHub](https://github.com/ZunhaiSu/Super-Experts-Profilling)  
**领域**: 模型压缩 / MoE / LLM 分析  
**关键词**: Mixture-of-Experts, super experts, massive activations, attention sinks, expert pruning, model compression

## 一句话总结

本文首次发现并系统研究了 MoE LLM 中的"超级专家"（Super Experts）——数量极少但对模型推理至关重要的专家子集，它们通过 down_proj 中的极端激活异常值驱动 massive activations 和 attention sinks 机制。

## 研究背景与动机

MoE LLM（如 DeepSeek、Qwen3、Mixtral）通过动态路由和稀疏激活实现了强大的学习能力。现有专家级压缩方法利用专家间的重要性差异进行剪枝、合并或量化，但多依赖启发式指标来识别关键专家，缺乏对专家异质性重要性的深层理解。

核心问题：**是否存在少量极端关键的专家子集？它们的作用机制是什么？**

## 方法详解

### 整体框架

论文从三个递进的深度分析超级专家：

1. **发现与定位**：SE 通过 down_proj 输出中的极端激活异常值引起 massive activations
2. **重要性评估**：通过动态剪枝量化 SE 对各任务的影响
3. **机制揭示**：SE 是 Transformer 系统性异常值机制的主要来源，压缩 SE 会导致 attention sinks 崩溃

### 关键设计 1：超级专家的发现与定位

MoE LLM 中存在**massive activations（MA）**——隐藏状态中值比其他激活大 100,000 倍的极端异常值。

通过分析发现：**少量特定专家**在 down_proj 输出中持续产生极端激活异常值，通过残差连接传播到后续所有层的隐藏状态中。

**SE 剖析定义**：计算所有专家在所有层的 down_proj 最大输出幅度 $a_{l,e}$：

$$a_{l,e} > P_{99.5} \quad \text{且} \quad a_{l,e} > \frac{1}{10} a_{\max} \quad \text{且} \quad l \in L$$

其中 $P_{99.5} = \text{Percentile}_{99.5}(\mathcal{A})$，$L$ 为产生 MA 的层集合。

### 关键设计 2：SE 的分布特性

| 模型 | 总专家数 | SE 数量 | SE 比例 | Top1 最大激活 |
|------|---------|--------|--------|-------------|
| Qwen3-30B-A3B | 6144 | 3 | 0.05% | 744.0 |
| DeepSeek-R1 | 15677 | 10 | 0.06% | 616.0 |
| DeepSeek-V2-Lite | 1782 | 2 | 0.11% | 1424.0 |
| Mixtral-8x7B | 256 | 1 | 0.39% | 5600.0 |

核心发现：
- SE 普遍存在于所有测试的 MoE LLM 中，占比 < 0.5%
- SE 分布是**模型特异的**、**数据无关的**
- 后训练过程（如 RLHF）不改变 SE 分布

### 关键设计 3：SE 重要性评估

通过动态剪枝 SE 并在多个任务上评估性能退化：

| 模型 | 设置 | 平均分 | GSM8K | MMLU | HellaSwag |
|------|------|--------|-------|------|-----------|
| Qwen3-30B-A3B | Baseline | 70.22 | 89.61 | 77.82 | 59.63 |
| Qwen3-30B-A3B | 剪 SE | 55.00 | 42.38 | 56.03 | 39.31 |
| Qwen3-30B-A3B | 随机剪同数量 | 70.36 | 89.84 | 77.84 | 59.50 |

仅剪去 3 个 SE（占 6144 的 0.05%）导致：
- 平均性能下降 21.68%
- **数学推理（GSM8K）下降 52.71%**
- 对推理 LLM，AIME 和 Math-500 的 Pass@1 降至近零

### 关键设计 4：与 Attention Sinks 的关系

SE 是 Transformer 系统性异常值机制的核心：

1. SE 在 attention sink tokens 上产生极强激活
2. 这些激活通过残差连接形成 massive activations
3. MA 驱动 attention sinks 的形成
4. 压缩 SE → MA 消失 → attention sinks 崩溃 → 注意力分数分布紊乱

这揭示了一个完整的因果链条：**SE → MA → Attention Sinks → 模型功能**

## 实验

### 主实验：非推理模型

| 指标 | Qwen3-30B 基线 | 剪 SE | 下降率 | 随机剪 | 下降率 |
|------|-------------|-------|-------|-------|-------|
| Avg. | 70.22 | 55.00 | -21.68% | 70.36 | -0.20% |
| GSM8K | 89.61 | 42.38 | -52.71% | 89.84 | +0.26% |
| MMLU | 77.82 | 56.03 | -28.00% | 77.84 | +0.03% |

### 推理模型实验

剪去 DeepSeek-R1 的 10 个 SE：
- AIME/Math-500 的 Pass@1 降至接近 0
- 数学推理能力完全崩溃

### 消融实验

- 按层分别剪 SE：单层 SE 剪枝即可消除该层的 MA 贡献
- 全部 SE 剪除：MA 完全消失

### 跨数据集稳定性

在 C4、WikiText-2、C-Eval、GSM8K、HumanEval 上的 SE 分布高度一致，验证了数据无关性。

## 亮点

- 首次系统发现并定义了 MoE LLM 中的超级专家
- 揭示了完整因果链：SE → MA → Attention Sinks → 模型功能
- 提供了自动化 SE 剖析工具，可快速定位 SE
- 对 MoE 压缩具有重要指导意义：SE 必须特殊对待

## 局限性

- SE 为何在预训练中形成的根本原因尚不清楚
- 仅分析了开源 MoE 模型，闭源模型（如 GPT-4）的情况未知
- SE 的保护策略（如分配更高比特预算）仅初步讨论
- 是否可以设计"无 SE"的更均衡 MoE 训练机制，未深入探讨

## 相关工作

- **MoE 模型**：DeepSeek (Guo et al., 2025)、Qwen (Yang et al., 2025)、Mixtral
- **专家级压缩**：基于频率、路由分数、重建损失的专家重要性度量
- **Massive Activations**：Sun et al. (2024) 发现但未解释在 MoE 中的成因
- **Attention Sinks**：Xiao et al. (2023) 发现初始 token 获得异常高注意力

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次发现和系统研究 MoE 中的超级专家
- 理论深度：⭐⭐⭐⭐ — 因果分析深入但缺乏形式化理论解释
- 实验充分性：⭐⭐⭐⭐⭐ — 多模型、多任务、多数据集全面验证
- 实用价值：⭐⭐⭐⭐⭐ — 直接指导 MoE 压缩策略
- 写作质量：⭐⭐⭐⭐ — 递进式分析结构清晰

<!-- RELATED:START -->

## 相关论文

- [LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models](../../ACL2025/model_compression/eac_moe_expert_aware_compression.md)
- [DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models](../../CVPR2025/model_compression/ders_towards_extremely_efficient_upcycled_mixture-of-experts_models.md)
- [Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](../../NeurIPS2025/model_compression/dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [MoNE: Replacing Redundant Experts with Lightweight Novices for Structured Pruning of MoE](mone_replacing_redundant_experts_with_lightweight_novices_for_structured_pruning.md)

<!-- RELATED:END -->
