---
title: >-
  [论文解读] Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA
description: >-
  [NeurIPS 2025][模型压缩][联邦学习] 提出 RoLoRA，通过交替优化 LoRA 的 down-projection (A) 和 up-projection (B) 矩阵，解决联邦学习中 LoRA 聚合不精确和表达力受限的问题…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "联邦学习"
  - "LoRA"
  - "parameter-efficient fine-tuning"
  - "alternating optimization"
  - "LLM"
---

# Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA

**会议**: NeurIPS 2025  
**arXiv**: [2502.01755](https://arxiv.org/abs/2502.01755)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: federated learning, LoRA, parameter-efficient fine-tuning, alternating optimization, LLM

## 一句话总结

提出 RoLoRA，通过交替优化 LoRA 的 down-projection (A) 和 up-projection (B) 矩阵，解决联邦学习中 LoRA 聚合不精确和表达力受限的问题，在 RoBERTa-Large 和 Llama-2-7B 上显著优于 FedAVG of LoRA 和 FFA-LoRA。

## 研究背景与动机

**领域现状**：联邦学习中使用 LoRA 进行参数高效微调是主流趋势。LoRA 将权重更新分解为 $\Delta \mathbf{W} = \alpha \mathbf{A}\mathbf{B}$，其中 $\mathbf{A} \in \mathbb{R}^{d \times r}$, $\mathbf{B} \in \mathbb{R}^{r \times d}$, $r \ll d$。

**现有痛点**：
   - **FedAVG of LoRA**：直接平均各客户端的 $\mathbf{A}_i$ 和 $\mathbf{B}_i$ 会引入聚合干扰——$\frac{1}{N}\sum_i \mathbf{A}_i \mathbf{B}_i \neq \frac{1}{N}\sum_i \mathbf{A}_i \cdot \frac{1}{N}\sum_i \mathbf{B}_i$
   - **FFA-LoRA**：冻结 $\mathbf{A}$（down-projection）只更新 $\mathbf{B}$ 虽然避免了干扰，但牺牲了模型表达力，在参数量少或客户端多时性能显著下降
   - **FlexLoRA/FLoRA**：通过矩阵乘法 + 截断 SVD 恢复精确更新，但计算开销大

**核心矛盾**：精确聚合 vs 模型表达力 vs 计算/通信效率的三方博弈。

**本文目标**：设计一个同时保证精确聚合、充分表达力和低通信/计算开销的联邦 LoRA 微调框架。

**切入角度**：受多任务线性表示学习（MLRL）启发，交替冻结 $\mathbf{A}$ 和 $\mathbf{B}$——每轮只有一个矩阵被训练和聚合，自然保证精确聚合。

**核心 idea**：奇数轮冻结 $\mathbf{A}$ 更新 $\mathbf{B}$，偶数轮冻结 $\mathbf{B}$ 更新 $\mathbf{A}$，交替进行即可兼得精确聚合和充分表达力。

## 方法详解

### 整体框架

**RoLoRA 算法（Algorithm 1）**：
- 奇数通信轮：所有客户端冻结共享的 $\mathbf{A}^t$，本地训练 $\mathbf{B}_i^{t+1}$；服务器聚合 $\mathbf{B}^{t+1} = \frac{1}{N}\sum_i \mathbf{B}_i^{t+1}$
- 偶数通信轮：所有客户端冻结共享的 $\mathbf{B}^{t+1}$，本地训练 $\mathbf{A}_i^{t+1}$；服务器聚合 $\mathbf{A}^{t+1} = \frac{1}{N}\sum_i \mathbf{A}_i^{t+1}$
- 由于冻结的矩阵全局一致，聚合天然精确

### 关键设计

**1. 精确聚合保证**

奇数轮中 $\mathbf{A}_i^t = \mathbf{A}^t$ 对所有客户端相同，因此：
$$\frac{1}{N}\sum_i \mathbf{A}_i^t \mathbf{B}_i^{t+1} = \mathbf{A}^t \cdot \frac{1}{N}\sum_i \mathbf{B}_i^{t+1}$$
聚合完全精确，无干扰。

**2. 线性模型理论分析（Theorem 4.5）**

在联邦线性回归中（$\mathbf{Y}_i = \mathbf{X}_i \mathbf{a}^* \mathbf{b}^{*\top}$），RoLoRA 实现指数收敛：
$$\sin\theta(\mathbf{a}^{t+1}, \mathbf{a}^*) \leq \sin\theta(\mathbf{a}^t, \mathbf{a}^*) \sqrt{1 - \eta(1 - \delta_0^2)\|\mathbf{b}^*\|^2}$$
- 角度距离指数衰减到任意小的 $\epsilon$
- FFA-LoRA 的损失下界（Proposition 4.6）为 $(1 + \tilde{c})\|\mathbf{b}^*\|^2 (\delta_0)^2$，受限于初始化角度，永远无法趋零

**3. 非凸收敛保证（Theorem A4.4）**

在光滑非凸设定下，RoLoRA 收敛率为 $O(1/\sqrt{T})$，与 FedAVG 一致。

### 损失函数/训练策略

- 与标准 LoRA 相同的损失函数
- 每轮可训练参数量减半（仅训练 A 或 B），同时通信量也减半
- 学习率从 $\{5e^{-4}, ..., 1e^{-1}\}$ 中选最优

## 实验关键数据

### 主实验：RoBERTa-Large on GLUE（50 clients, rank 4）

| 方法 | SST-2 | QNLI | MNLI | QQP | RTE | Avg |
|------|-------|------|------|-----|-----|-----|
| LoRA | 93.00 | 78.13 | 52.64 | 77.60 | 52.23 | 70.72 |
| FFA-LoRA | 93.23 | 85.05 | 69.97 | 78.44 | 55.72 | 76.48 |
| FlexLoRA | 54.08 | 55.40 | 39.14 | 72.00 | 52.71 | 54.67 |
| **RoLoRA** | **94.80** | **90.00** | **82.98** | **85.71** | **75.57** | **85.81** |

RoLoRA 在 50 客户端设置下比 LoRA 高 **+15.09%** 平均准确率，比 FFA-LoRA 高 **+9.33%**。

### Llama-2-7B on Commonsense（50 clients, rank 8）

| 方法 | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|------|-------|------|------|-----------|------------|-------|-------|------|
| LoRA | 61.42 | 33.19 | 31.88 | 21.23 | 31.36 | 27.36 | 32.03 | 26.07 |
| FFA-LoRA | 53.43 | 35.49 | 10.63 | 11.81 | 1.61 | 6.88 | 7.93 | 15.00 |
| **RoLoRA** | **61.83** | **61.26** | **39.76** | **27.49** | **47.67** | **33.19** | **40.13** | **31.67** |

FFA-LoRA 在大模型上几乎崩溃，RoLoRA 大幅领先。

### 消融实验

| 消融维度 | 发现 |
|---------|------|
| 客户端数量（3→50） | LoRA/FFA-LoRA 性能急剧下降，RoLoRA 保持稳定（3 clients: 88.28 → 50 clients: 85.81） |
| Non-IID (Dir 0.5/1.0) | RoLoRA 在 MNLI 上达 82.60%，LoRA 为 81.19%，FlexLoRA 仅 35.45% |
| 参数量减少 | FFA-LoRA 在参数少时显著退化，RoLoRA 保持鲁棒 |
| 对称 vs 非对称更新 | 均衡交替 AB 最优，偏向 A 或 B 都退化 |
| 本地步数（1→20） | FFA-LoRA 72.52%→69.97%（增加步数后退化），RoLoRA 84.39%→82.98%（稳定） |

### 关键发现

- 随客户端增多（3→20→50），LoRA 的 FedAVG 聚合干扰问题急剧恶化
- FFA-LoRA 的限制来自 $\mathbf{A}$ 初始化质量——不同随机种子下方差极大（PIQA: std=9.55）
- 学习 $\mathbf{A}$ 在训练早期尤其重要（20% RoLoRA + 80% FFA-LoRA 已明显优于纯 FFA-LoRA）
- RoLoRA 通信量仅为 LoRA/FlexLoRA 的 50%

## 亮点与洞察

- **简洁有效**：交替冻结是一个极其简单的设计，却同时解决了精确聚合和表达力的矛盾
- **理论与实践统一**：线性模型理论（指数收敛 vs 饱和）在非线性实验中完美验证
- **鲁棒性突出**：在极端设置（50 clients、rank 2、non-IID）下仍表现优异
- **实际部署友好**：通信量减半 + 计算量减半，且无额外 SVD 操作

## 局限与展望

- 线性模型理论假设同质客户端和单 LoRA 结构，与实际多层 LoRA 有差距
- 交替优化在通信轮次上效率减半（同等总轮次下 A 和 B 各只被更新了一半次数）
- 未探索自适应交替频率（如何确定 A 和 B 的最优交替比例？）
- 缺少与全参数微调的对比
- 未讨论与隐私保护（差分隐私）的结合

## 相关工作与启发

- 与 MLRL（多任务低秩表示学习）的连接提供了理论基础
- LoRA+ 探索了 A/B 不同学习率，RoLoRA 的交替策略是更激进的非对称处理
- 可扩展到异构 rank 设置（如不同客户端使用不同 rank）
- 启发：联邦学习中的聚合精确性问题在其他分解参数方法（如 adapter、prefix tuning）中同样存在

## 评分

⭐⭐⭐⭐ (4/5)

方法简洁有效，理论分析清晰，实验全面且性能优势显著。主要不足是理论与实践之间的 gap（线性 vs 多层非线性）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](../../ICML2026/model_compression/fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](../../ACL2025/model_compression/fedex_lora_federated_exact_aggregation.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)
- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[ICML 2025\] Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment](../../ICML2025/model_compression/make_lora_great_again_boosting_lora_with_adaptive_singular_values_and_mixture-of.md)

</div>

<!-- RELATED:END -->
