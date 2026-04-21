---
title: >-
  [论文解读] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning
description: >-
  [NeurIPS 2025][LLM/NLP][零阶优化] 提出 Sparse MeZO（S-MeZO），通过观察到零阶梯度噪声对大权重影响更严重，选择性地仅对小权重进行零阶优化扰动和更新，在不增加内存开销的前提下实现了显著的性能提升（RTE 上 +9%）和收敛加速（3.5x）。
tags:
  - NeurIPS 2025
  - LLM/NLP
  - 零阶优化
  - 稀疏微调
  - 内存高效
  - LLM微调
  - 梯度噪声
---

# Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2402.15751](https://arxiv.org/abs/2402.15751)  
**代码**: [GitHub](https://github.com/NUS-HPC-AI-Lab/SparseMeZO)  
**领域**: llm_nlp  
**关键词**: 零阶优化, 稀疏微调, 内存高效, LLM微调, 梯度噪声  

## 一句话总结

提出 Sparse MeZO（S-MeZO），通过观察到零阶梯度噪声对大权重影响更严重，选择性地仅对小权重进行零阶优化扰动和更新，在不增加内存开销的前提下实现了显著的性能提升（RTE 上 +9%）和收敛加速（3.5x）。

## 研究背景与动机

LLM 微调需要大量内存（反向传播需缓存激活值），参数高效微调（PEFT）如 LoRA 仍需 6 倍于推理的内存。零阶优化（ZO）是一种仅需前向传播的梯度估计方法，MeZO 将其应用于 LLM 微调，内存开销等同于推理。

**MeZO 的核心问题**：

**梯度估计噪声大**：ZO 梯度 $\delta = g(\theta) - g_z(\theta)$ 与真实梯度偏差显著

**对学习率极其敏感**：从 $1 \times 10^{-6}$ 增到 $2 \times 10^{-6}$ 即导致发散

**泛化能力差**：同 batch 上 90% 成功率降低损失，换 batch 仅 50%

**关键发现**：**梯度噪声对大权重的影响远大于小权重**。从性能下降点继续仅优化小权重即可恢复并提升精度。这一实验观察是 Sparse MeZO 的理论基础。

## 方法详解

### 整体框架

在标准 MeZO 基础上引入稀疏掩码，仅对按权重大小筛选的子集进行扰动和更新。

### 关键设计一：稀疏掩码机制

定义稀疏掩码 $\mathbf{m} \in \{0,1\}^d$，对随机噪声进行稀疏化：

$$\hat{\mathbf{z}} = \mathbf{m} \odot \mathbf{z}$$

稀疏零阶梯度估计：

$$g_{\hat{z}}(\theta) = \frac{\mathcal{L}(\theta + \epsilon \mathbf{m} \odot \mathbf{z}) - \mathcal{L}(\theta - \epsilon \mathbf{m} \odot \mathbf{z})}{2\epsilon} \hat{\mathbf{z}}$$

掩码基于参数幅度动态生成：仅选择每层中幅度最小的参数进行更新（按百分位阈值）。

### 关键设计二：动态掩码（无额外内存）

两种掩码策略：

- **常量掩码**：训练前设置，但需要存储与模型等大的掩码
- **动态掩码**（推荐）：每次迭代根据当前参数值动态生成，无需存储

### 关键设计三：前向传播中计算掩码

创新的内存高效实现：在前向传播过程中逐层计算掩码并立即释放，避免存储全局掩码：

$$y^{(i)} = (\theta_t^{(i)} + \epsilon \cdot m(\theta_t) \cdot z^{(i)}) x^{(i)} + b^{(i)}$$

每层计算掩码 $m^{(i)}$ 后立即释放其内存，计算下一层。这使得 S-MeZO 的内存消耗与原始 MeZO 完全相同（仅推理级别）。

### 损失函数 / 训练策略

S-MeZO 使用与 MeZO 相同的训练设置，关键区别是更高的学习率（因为仅更新小权重对噪声更鲁棒）。稀疏率通常设为 0.5-0.8，即仅更新 20-50% 的参数。

## 实验关键数据

### 主实验

**LLaMA-7b 在 SuperGLUE 上的性能**：

| 方法 | BoolQ | RTE | WIC | MultiRC | SST-2 | COPA | 平均 |
|------|:-----:|:---:|:---:|:-------:|:-----:|:----:|:----:|
| Zero-Shot | 65.1 | 49.5 | 50.6 | 55.8 | 79.7 | 59.7 | 60.1 |
| LoRA | 84.5 | 82.3 | 67.6 | 78.3 | 95.0 | 86.0 | 82.3 |
| FT (Adam) | 84.5 | 83.6 | 68.4 | 80.2 | 95.7 | 85.0 | 82.9 |
| MeZO | 75.9 | 71.7 | 61.4 | 69.8 | 94.6 | 86.3 | 76.6 |
| R-MeZO | 76.9 | 75.2 | 62.1 | 68.1 | 94.6 | 84.3 | 76.9 |
| **S-MeZO** | **80.9** | **80.7** | **64.9** | **73.3** | **95.0** | **86.7** | **80.3** |

S-MeZO 在 RTE 上 **+9.0%**，BoolQ 上 **+5.0%**，平均 **+3.7%**。

**LLaMA2-7b 与更多基线对比**：

| 方法 | BoolQ | RTE | WIC | SST-2 | 平均 |
|------|:-----:|:---:|:---:|:-----:|:----:|
| MeZO | 78.8 | 70.2 | 62.2 | 94.0 | 76.3 |
| ZO-AdaMU | 78.2 | 76.5 | 63.0 | 93.6 | 77.8 |
| AdaZeta | 79.8 | 75.8 | 62.0 | 94.0 | 77.9 |
| **S-MeZO** | **82.2** | **77.6** | **65.3** | **94.8** | **80.0** |

### 消融实验

**收敛速度**：S-MeZO 在 RTE 上达到 70% 仅需约 5,000 步，MeZO 需要 17,500 步（**3.5x 加速**）。

**内存使用（LLaMA-7b）**：

| 方法 | 内存消耗 |
|------|:-------:|
| FT (Adam) | 128.2 GB |
| LoRA | 22.4 GB |
| MeZO | **14.6 GB** |
| S-MeZO (高效实现) | **14.6 GB** |

**稀疏率影响**：0.5-0.8 的稀疏率性能最佳，0.8 时 RTE 从 71.7% 提升到 82.3%。

### 关键发现

1. **大权重对 ZO 噪声更敏感**：仅优化小权重即可显著恢复和提升性能
2. **S-MeZO 大幅缩小 ZO 与一阶方法的差距**：平均差距从 6.3% 缩小到 2.6%
3. **高效实现无额外内存开销**：前向传播中计算掩码的方法使内存保持推理水平
4. **可扩展到 30B 模型**：LLaMA-30b 可在单张 A100 上微调

## 亮点与洞察

1. **实验观察驱动的方法设计**：先发现"大权重对噪声更敏感"的现象，再设计针对性的稀疏化策略
2. **内存效率极致**：前向传播中计算掩码的实现创新，使 S-MeZO 内存与推理完全一致
3. **实用性强**：可在单张 A100 上微调 30B 模型，使资源受限场景也能微调大模型
4. **一致性好**：在 LLaMA、Mistral、OPT 三个模型家族上都有稳定改进
5. **简单而有效**：核心思想就是"只扰动小权重"，实现简洁但效果显著

## 局限与展望

1. **与一阶方法仍有差距**：虽然显著缩小但 S-MeZO（80.3）vs FT（82.9）仍有 2.6% 差距
2. **阈值选择**：稀疏率需要手动设定，不同任务可能需要不同设置
3. **理论解释不够深入**：为何小权重对噪声更鲁棒的理论分析较浅
4. **训练轮次多**：需要较多步数才能收敛
5. 可探索与 LoRA 等 PEFT 方法结合的混合策略

## 相关工作与启发

- **MeZO**（Malladi et al., 2023）：内存高效零阶优化的基础工作
- **LoRA**（Hu et al., 2021）：参数高效微调的经典方法
- **Lottery Ticket Hypothesis**（Frankle & Carlin, 2018）：稀疏子网络的理论基础
- **DeepZero**（Chen et al., 2023）：模型剪枝引导的零阶训练

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：简单但有效的核心洞察 + 精巧的内存高效实现
- **实验充分度** ⭐⭐⭐⭐：多模型多任务全面评估
- **实用性** ⭐⭐⭐⭐⭐：单卡微调 30B 模型，极致内存效率
- **理论深度** ⭐⭐⭐：收敛证明有但不够深入

<!-- RELATED:START -->

## 相关论文

- [Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/llm_nlp/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)
- [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](../../ACL2025/llm_nlp/refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning](synergy_over_discrepancy_a_partition-based_approach_to_multi-domain_llm_fine-tun.md)
- [SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)

<!-- RELATED:END -->
