---
title: >-
  [论文解读] Instruction-Following Pruning for Large Language Models
description: >-
  [ICML2025][模型压缩][结构化剪枝] 提出 IFPruning：用一个小型稀疏预测器根据用户指令动态生成剪枝掩码，将 FFN 中间维度按需裁减，使 9B 模型仅激活 3B 参数即可在编程/数学上超越同规模 dense 模型 5-8 个百分点，且推理延迟与 3B dense 模型持平。
tags:
  - ICML2025
  - 模型压缩
  - 结构化剪枝
  - 动态剪枝
  - 指令感知
  - 稀疏预测器
  - SoftTopK
  - 端侧推理
---

# Instruction-Following Pruning for Large Language Models

**会议**: ICML2025  
**arXiv**: [2501.02086](https://arxiv.org/abs/2501.02086)  
**代码**: 未开源  
**领域**: model_compression / LLM剪枝  
**关键词**: 结构化剪枝, 动态剪枝, 指令感知, 稀疏预测器, SoftTopK, 端侧推理

## 一句话总结

提出 IFPruning：用一个小型稀疏预测器根据用户指令动态生成剪枝掩码，将 FFN 中间维度按需裁减，使 9B 模型仅激活 3B 参数即可在编程/数学上超越同规模 dense 模型 5-8 个百分点，且推理延迟与 3B dense 模型持平。

## 研究背景与动机

传统结构化剪枝为模型生成一个**固定的剪枝掩码**，压缩后的子网络在所有任务上共享相同参数。这在面对编程、数学、领域知识等不同技能需求时存在天然矛盾——固定子网络无法同时在所有任务上最优。

与此同时，现有动态方法（Contextual Sparsity、MoE）虽然能选不同参数，但在**每个解码步骤**都要加载不同权重，导致显著的权重传输开销，不适合端侧部署。

本文提出的核心问题是：**LLM 能否根据任务描述自主选择最合适的参数子集？** 即在解码前一次性确定子网络，既保持动态灵活性，又避免逐步重加载的开销。

## 方法详解

### 整体架构

IFPruning 由两个组件构成：

1. **稀疏预测器（Sparsity Predictor）**：一个 302M 参数的小型 LLM + 两层 MLP 预测头
2. **被剪枝的大模型（Masked LLM）**：6B / 9B / 12B 规模，FFN 层按掩码动态裁剪

### 结构化剪枝形式化

对标准 FFN 层：

$$F_{\text{ffn}}(X) = \sigma(X W_1) W_2$$

引入掩码向量 $\mathbf{m} \in \{0,1\}^{d_{\text{ffn}}}$，剪枝后的 FFN 输出为：

$$F_{\text{ffn}}(X, \mathbf{m}) = \text{FF}_2(\text{FF}_1(X) \odot \mathbf{m})$$

其中 $m_i = 0$ 表示 $W_1$ 的第 $i$ 列和 $W_2$ 的第 $i$ 行被剪除。掩码满足稀疏约束 $\sum_i m_i = t_{\text{ffn}}$。

### 可微掩码生成（SoftTopK）

稀疏预测器输出分数 $\mathbf{z} \in \mathbb{R}^{L \times d_{\text{ffn}}}$，经 SoftTopK 转换为可微掩码：

$$\boldsymbol{\lambda}^{(i)} = g(\mathbf{z}^{(i)}), \quad \mathbf{m}^{(i)} = \boldsymbol{\lambda}^{(i)} \odot \text{Top}(\boldsymbol{\lambda}^{(i)}, t_{\text{ffn}})$$

其中 $g(\cdot)$ 为归一化函数（确保 $\sum_k \lambda_k^{(i)} = t_{\text{ffn}}$），$\text{Top}(\cdot, t_{\text{ffn}})$ 返回 top-k 指示掩码。该机制使梯度可回传至预测器。

### 两阶段训练

**阶段一：继续预训练。** 将文本切分为固定长度的 chunk，用第 $k$ 个 chunk 预测掩码，在第 $k+1$ 个 chunk 上计算 next-token prediction loss：

$$\mathcal{L} = \sum_{k=1}^{K-1} \sum_{x_i \in \mathbf{x}^{(k+1)}} \ell\big[f(\mathbf{x}_{<i}; \boldsymbol{\theta}, \mathbf{m}^{(k)}), x_i\big]$$

**阶段二：SFT 微调。** 在数百万条指令数据上联合优化预测器和 LLM，用户 prompt 直接作为预测器输入生成掩码。对多轮对话，仅用**第一条用户消息**选择子网络。

### 推理方式

- **Per-input 模式**：每条输入独立预测掩码，最灵活
- **Per-task 模式**：为同一任务（如"数学"）用一条任务描述生成共享掩码，减少开销

## 实验关键数据

基座模型：6B/9B/12B 参数 LLM，统一激活 3B 参数。对比基线：Dense-3B（9T token 训练）、Pruning+Distill 3B（静态剪枝+蒸馏，12B 教师）、Dense-9B（无剪枝上界）。

| 任务类别 | 数据集 | Dense-3B | Pruning+Distill | IFP 9B→3B | Dense-9B |
|---------|--------|----------|-----------------|-----------|----------|
| 编程 | HumanEval | 35.2 | 37.1 | **42.4** | 46.5 |
| 编程 | MBPP | 28.8 | 38.0 | **41.8** | 42.2 |
| 编程 | MultiPL-E | 39.0 | 37.9 | **41.8** | 44.0 |
| 编程 | 平均 | 34.3 | 37.7 | **42.0** | 44.2 |
| 数学 | GSM8K | 69.3 | 70.0 | **72.0** | 75.4 |
| 数学 | MATH | 31.8 | 32.7 | **36.7** | 37.3 |
| 指令跟随 | AlpacaEval 2.0 | 27.3 | 30.0 | **31.3** | 38.6 |
| 知识 | MMLU | 61.8 | 62.8 | **65.5** | 67.8 |
| 核心文本 | 平均 | 69.9 | 70.0 | **71.1** | 73.8 |

**关键结论：**

- IFPruning 9B→3B 在编程上比 Dense-3B 高 **+7.7pp**，在数学上高 **+4.9pp**
- 超越 Pruning+Distill（获知识蒸馏加持）4+ 个百分点
- 接近 Dense-9B 上界性能，尤其在 MATH (36.7 vs 37.3) 和 MBPP (41.8 vs 42.2) 上差距极小

**推理效率：** 与完整模型相比，TTFT 降低最多 57%，生成时间降低最多 41%；动态剪枝与缓存开销 <0.1s/样本，仅占总生成时间 1-2%。

**Per-task vs Per-input：** 每个任务用一条任务描述共享掩码，性能与逐条预测掩码几乎持平（HumanEval 40.9 vs 42.4），证明同类任务的剪枝模式高度一致。

## 亮点与洞察

1. **范式创新**：首次将结构化剪枝从"静态掩码"推向"指令驱动的动态掩码"，使剪枝从一次性压缩变为推理时的自适应能力
2. **解码前选参数、解码中固定**：巧妙规避了 MoE/Contextual Sparsity 逐步换参数的 I/O 瓶颈，特别适合端侧场景
3. **可解释性强**：同类任务（如数学-GSM8K 与 MMLU-Math）的子网络重叠率高达 ~80%，而跨领域（数学 vs 历史）重叠率低，说明模型学到了有意义的领域专精参数分组
4. **稀疏预测器极小**（302M），相比被剪枝的 LLM（6-12B）开销可忽略
5. **扩展性良好**：6B→9B→12B 模型在 IFPruning 下性能稳步提升，且增益在编程/数学上最显著

## 局限与展望

1. **仅剪枝 FFN 层**：attention heads 和 embedding 层未被剪枝，理论上可进一步压缩
2. **训练成本高**：需要联合优化预测器和 LLM，两阶段训练需数百万 SFT 样本 + 预训练数据
3. **Per-input 模式的预测器推理开销**：虽然论文称 <0.1s，但对超低延迟场景仍非零
4. **未与 MoE 直接对比**：没有在相同参数预算下与 MoE 模型做公平比较
5. **AlpacaEval 上增益较小**：作者也承认在 AlpacaEval 上扩大模型规模带来的提升有限，开放指令跟随场景可能无法充分受益
6. **数据依赖**：使用内部 SFT 数据集（数百万条），可复现性受限

## 评分

- 新颖性: ⭐⭐⭐⭐ — 动态指令驱动剪枝是有意义的范式突破
- 实验充分度: ⭐⭐⭐⭐ — 多规模、多基线、多任务，含可解释性分析和效率测评
- 写作质量: ⭐⭐⭐⭐ — 论文结构清晰，动机、方法、实验层层推进
- 价值: ⭐⭐⭐⭐ — 为端侧 LLM 部署提供了极具前景的新方向

<!-- RELATED:START -->

## 相关论文

- [Olica: Efficient Structured Pruning of Large Language Models without Retraining](olica_efficient_structured_pruning_of_large_language_models_without_retraining.md)
- [DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [BlockPruner: Fine-grained Pruning for Large Language Models](../../ACL2025/model_compression/blockpruner_fine-grained_pruning_for_large_language_models.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](../../ACL2025/model_compression/wanda_pruning_large_language_models_via_regional_gradients.md)
- [Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)

<!-- RELATED:END -->
