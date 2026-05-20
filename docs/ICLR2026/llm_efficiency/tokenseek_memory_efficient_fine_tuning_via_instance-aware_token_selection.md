---
title: >-
  [论文解读] TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection
description: >-
  [ICLR 2026][LLM效率][memory efficient fine-tuning] 提出 TokenSeek，一个通用的实例感知 token 搜索与丢弃方法，通过结合上下文（注意力）和梯度信息评估每个 token 的重要性，仅在选中的 token 上更新参数…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "memory efficient fine-tuning"
  - "token selection"
  - "activation optimization"
  - "instance-aware"
  - "gradient sparsification"
---

# TokenSeek: Memory Efficient Fine Tuning via Instance-Aware Token Selection

**会议**: ICLR 2026  
**arXiv**: [2601.19739](https://arxiv.org/abs/2601.19739)  
**代码**: [runjia.tech/iclr_tokenseek](https://runjia.tech/iclr_tokenseek)  
**领域**: LLM效率  
**关键词**: memory efficient fine-tuning, token selection, activation optimization, instance-aware, gradient sparsification

## 一句话总结

提出 TokenSeek，一个通用的实例感知 token 搜索与丢弃方法，通过结合上下文（注意力）和梯度信息评估每个 token 的重要性，仅在选中的 token 上更新参数，实现激活内存的大幅减少（最高 65.7%）而保持甚至超越全 token 微调性能。

## 研究背景与动机

LLM 微调面临巨大的内存消耗问题，其中**激活 (activations) 一致性地主导着总内存消耗**（例如 Llama3 8B 中激活占 87%）。现有内存高效微调（MEFT）方法主要采用：重计算（梯度检查点）、压缩（量化/稀疏化）、可逆网络三种范式。

**现有方法的核心问题**：它们都是**数据无关的优化** (data-agnostic)——对所有实例采用统一且不灵活的策略，不考虑每个实例内丰富的变异性。这导致：
- **低效微调**：无法根据实例调整内存缩减粒度
- **不稳定微调**：性能波动大

**核心挑战**：
1. 如何识别代表每个实例关键信息的显著 token？
2. 如何利用它们实现有效且稳定的内存优化？

## 方法详解

### 整体框架

TokenSeek 由两个关键组件构成：
1. **实例感知 Token 搜索** (Instance-Aware Token Seeking)：评估并打分每个 token
2. **高效 Token 丢弃** (Efficient Token Ditching)：仅在选中 token 上更新参数，丢弃其余 token 的梯度

### 关键设计 1：Token 搜索

token 冗余是 LLM 效率的根本挑战。TokenSeek 综合两类信息评估 token 重要性：

**上下文信息**（通过注意力机制）：

$$I_1(t_j) = \sum_{i=1}^{n} \mathbf{A}_{ij}$$

即 token $j$ 从所有其他 token 接收到的累积注意力权重。直觉上：被更多 token 关注的 token 更重要。

**梯度信息**：

$$I_2(t_j) = \text{Accumulate}\left[\frac{\partial \mathcal{L}}{\partial z^{(L-1)}}\right] = \sum_{k=1}^{d} \mathbf{G}_{jk}$$

即倒数第二层激活的梯度幅度在隐藏维度上的求和。梯度大的 token 对模型更新贡献更大。

**综合评分**：

$$I(t_j) = \alpha \log[I_1(t_j)] + \beta \text{Norm}[I_2(t_j)]$$

其中对上下文分数取 log 处理长尾分布（受注意力汇聚效应 Attention Sink 影响），对梯度分数做 min-max 归一化。

### 关键设计 2：Token 丢弃

选中 token 后，仅在选中 token 的激活上反向传播：

$$\frac{\partial z^{(l)}}{\partial z^{(l-1)}} = [\sigma'(a_t^{(l)}), 0] W^{(l)}$$

未选中 token 的梯度被置零，因此**只需缓存 $a_t^{(l)}$** 而非完整激活 $a^{(l)}$。理论上仅调 10% 的 token 仅需约 1% 的激活内存。

### Token 搜索的计算开销

仅需一次前向传播（FP8 下仅占训练内存的 13.3%）和一次部分反向传播（冻结所有层，仅计算输出头和最后解码器块的梯度）。

### 损失函数

标准语言建模损失，但仅在选中 token 上计算：

$$\mathcal{L} = -\sum_{j \in \text{selected}} \log P(y_j | x, y_{<j}; \theta)$$

## 实验关键数据

### 主实验

在 Qwen2.5 0.5B、Llama3.2 1B、Llama3.2 3B 上使用 Open-Platypus 数据集微调，评估 MMLU、ARC、HellaSwag、TruthfulQA、WinoGrande：

| 模型 | 方法 | 平均/峰值内存 | 平均分数 |
|------|------|-------------|---------|
| Llama3.2 1B | Full Token | 100%/100% | 40.82 |
| Llama3.2 1B | + TokenSeek | 64.6%/34.3% | **41.13** |
| Llama3.2 1B | LoHa | 92.3%/99.4% | 52.28 |
| Llama3.2 1B | LoHa + TokenSeek | 45.9%/28.4% | **52.58** |
| Llama3.2 1B | QLoRA | 45.6%/34.8% | 52.13 |
| Llama3.2 1B | QLoRA + TokenSeek | **14.8%/14.3%** | **52.61** |
| Llama3.2 3B | Full Token | 100%/100% | 41.53 |
| Llama3.2 3B | + TokenSeek | 73.1%/39.3% | **41.95** |
| Llama3.2 3B | QLoRA + TokenSeek | 13.3%/11.1% | 60.42 |

**亮点**：Llama3.2 1B + QLoRA + TokenSeek 仅用 14.8% 内存（2.8 GB）却超越全 token 基线（52.61 vs 40.82）。

### 消融实验

| 实验 | 发现 |
|------|------|
| α=1, β=0（仅上下文） | 48.45（有效但不完整） |
| α=0, β=1（仅梯度） | 46.39（不如上下文） |
| α=5, β=5（平衡） | 最优组合 |
| TokenTune（随机选择）| 一致低于 TokenSeek |
| 10% vs 50% token 比例 | 更多 token 降低训练损失，但过少可能导致优化崩溃 |

**可解释性分析**揭示的 token 选择模式：
- **上下文信息**偏好早期位置 token——受因果注意力掩码和注意力汇聚效应影响
- **梯度信息**主要聚焦在后期位置——通常对应"回答"部分
- 两者互补：上下文选择语义有意义的 token，梯度选择对学习最重要的 token

### 关键发现

1. **TokenSeek 偏好 PEFT**：全参数微调在低 token 比例下容易过拟合，PEFT 方法由于仅更新少量参数，对 token 丢弃更鲁棒
2. **跨规模泛化**：从 0.5B 到 3B 一致有效，但对较小模型（Qwen 0.5B）更敏感
3. **架构无关**：仅依赖注意力和梯度信息，适用于各种 Transformer 模型
4. 与 TokenTune（随机选择）的对比在所有设置下均表现出优势

## 亮点与洞察

1. **"一石二鸟"的设计理念**：实例感知搜索同时解决性能（选对 token）和内存（丢弃其余 token）两个问题
2. **上下文 + 梯度的互补发现**有深刻意义：注意力反映"哪些 token 在语义上重要"，梯度反映"哪些 token 对学习目标重要"
3. **QLoRA + TokenSeek 的组合效果惊人**：参数效率 + 内存效率的叠加实现了极端压缩下的性能提升
4. **可解释性分析**揭示的注意力汇聚效应和因果掩码对 token 重要性评估的影响，为未来研究提供了清晰的方向

## 局限性

1. **token 评估需要额外的前向和部分反向传播**：虽然开销较小，但对于超大规模模型可能仍需考虑
2. **超参数 α、β 的选择**：论文虽然做了消融但未提供自适应选择策略
3. **token 比例 10% 的硬编码**：不同数据集和任务可能需要不同比例
4. **缺乏在更大规模模型（7B+）上的验证**：当前最大仅测试 3B
5. **训练损失与下游性能的关系**需更深入分析：为何更高的训练损失（token 稀疏化导致）反而改善下游性能

## 相关工作与启发

- **TokenTune** (Simoulin et al., 2024)：随机 token 丢弃的先行者，但数据无关
- **QLoRA** (Dettmers et al., 2023)：参数高效方法，与 TokenSeek 互补
- **LoRA/LoHa**：其他 PEFT 方法，均可与 TokenSeek 无缝集成
- **梯度检查点**：另一类内存优化方法，通过重计算减少内存

TokenSeek 的核心启发：**微调中的 token 冗余是一个可利用的"漏洞"，而利用它的关键是实例级别的智能选择而非统一策略**。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 实例感知 token 选择的思路新颖，上下文+梯度的组合评估有原创性
- 实验充分度: ⭐⭐⭐⭐ — 多模型、多 PEFT 设置、消融研究全面，可解释性分析加分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ — 提供了立即可用的内存优化方案，与多种 PEFT 方法兼容

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Hierarchical Balance Packing: Towards Efficient Supervised Fine-tuning for Long-Context LLM](../../NeurIPS2025/llm_efficiency/hierarchical_balance_packing_towards_efficient_supervised_fine-tuning_for_long-c.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](../../ACL2025/llm_efficiency/tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)
- [\[ACL 2026\] Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length](../../ACL2026/llm_efficiency/understanding_llm_performance_degradation_in_multi-instance_processing_the_roles.md)
- [\[ACL 2026\] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](../../ACL2026/llm_efficiency/alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)
- [\[ICLR 2026\] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling](semantic_parallelism_redefining_efficient_moe_inference_via_model-data_co-schedu.md)

</div>

<!-- RELATED:END -->
