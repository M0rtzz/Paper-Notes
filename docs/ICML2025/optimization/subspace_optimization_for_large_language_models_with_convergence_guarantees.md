---
title: >-
  [论文解读] Subspace Optimization for Large Language Models with Convergence Guarantees
description: >-
  [ICML 2025][优化/理论][subspace optimization] 本文揭示了 GaLore（子空间优化算法）在随机设定下不总是收敛，并提出了 GoLore（梯度随机低秩投影）——一种可证明收敛的变体，即使在标准 batch 大小下也能保证收敛。
tags:
  - "ICML 2025"
  - "优化/理论"
  - "subspace optimization"
  - "LLM training"
  - "GaLore"
  - "low-rank projection"
  - "convergence guarantee"
---

# Subspace Optimization for Large Language Models with Convergence Guarantees

**会议**: ICML 2025  
**arXiv**: [2410.11289](https://arxiv.org/abs/2410.11289)  
**代码**: [https://github.com/pkumelon/Golore](https://github.com/pkumelon/Golore)  
**领域**: 优化  
**关键词**: subspace optimization, LLM training, GaLore, low-rank projection, convergence guarantee

## 一句话总结
本文揭示了 GaLore（子空间优化算法）在随机设定下不总是收敛，并提出了 GoLore（梯度随机低秩投影）——一种可证明收敛的变体，即使在标准 batch 大小下也能保证收敛。

## 研究背景与动机

**领域现状**: 大语言模型（LLM）的预训练和微调需要巨大的内存。子空间优化算法如 GaLore（Zhao et al., 2024）通过将梯度投影到低秩子空间来减少优化器状态的内存占用，取得了良好的实践效果。

**现有痛点**: 虽然 GaLore 在实践中效果不错，但其收敛性缺乏理论保证。特别是在随机梯度设定下（标准 mini-batch SGD），GaLore 是否总能收敛到最优解是一个未解的基本问题。

**核心矛盾**: GaLore 通过 SVD 分解梯度来选择投影子空间。但在随机设定下，svd 分解的是随机梯度而非真实梯度，这引入了系统性偏差——投影方向与真实梯度方向不一致，这种偏差能否被消除？

**本文目标**: (1) 明确 GaLore 的收敛极限，(2) 给出保证收敛的改进方案。

**切入角度**: 构造反例证明 GaLore 不收敛，然后分析收敛的充分条件，最后设计一种规避投影偏差的新算法。

**核心 idea**: 用随机投影（而非基于 SVD 的数据依赖投影）来避免系统性偏差，从而保证收敛。

## 方法详解

### 整体框架
标准子空间优化流程：
1. 计算 mini-batch 梯度 $\hat{G}$
2. 将梯度投影到低秩子空间：$\hat{G}_r = P \hat{G}$（$P$ 为投影矩阵）
3. 在低秩子空间中用 Adam/SGD 更新
4. 定期更新投影子空间

GaLore: $P$ 由 $\hat{G}$ 的 SVD 决定（数据依赖，有偏）
GoLore: $P$ 由随机矩阵生成（数据无关，无偏）

### 关键设计

1. **GaLore 反例构造（Counterexample for GaLore Divergence）**:

    - 功能：构造一个简单的凸优化问题，使 GaLore 不收敛到最优解
    - 核心思路：在 2 维问题上，当真实梯度在两个坐标方向上分量相近但噪声不同时，SVD 选择的投影方向会系统性偏离真实梯度方向
    - 关键公式：$\mathbb{E}[P(\hat{G})\hat{G}] \neq P(G)G$，即投影后梯度的期望不等于真实梯度的投影
    - 设计动机：明确了 GaLore 的理论缺陷，为改进提供方向

2. **GaLore 收敛条件（Conditions for GaLore Convergence）**:

    - 功能：识别 GaLore 仍然能收敛的特殊条件
    - 核心思路：两种情况下 GaLore 收敛：(i) 使用足够大的 mini-batch（降低 SVD 方向偏差），(ii) 梯度噪声为各向同性的（此时 SVD 方向不受噪声影响）
    - 设计动机：解释了为什么 GaLore 在实践中经常有效（大 batch + 近似各向同性噪声）

3. **GoLore: 梯度随机低秩投影（Gradient Random Low-rank Projection）**:

    - 功能：用随机投影替代 SVD 投影
    - 核心思路：投影矩阵 $P$ 从随机矩阵分布中采样（每隔 $T_0$ 步更新一次），保证 $\mathbb{E}[P \hat{G}] \propto G$
    - 关键公式：$P = Q Q^\top$，其中 $Q$ 为随机正交矩阵的前 $r$ 列
    - 设计动机：随机投影是数据无关的，因此不会引入系统性偏差。虽然单次投影可能丢失真实梯度的某些分量，但期望上是无偏的

### 损失函数 / 训练策略
标准的 LLM 训练损失（cross-entropy on next-token prediction）。GoLore 可与 Adam 优化器结合使用，仅需在投影模块替换 SVD 为随机投影。

## 实验关键数据

### 主实验

| 模型/任务 | 指标 | GoLore | GaLore | Full-rank Adam | LoRA |
|----------|------|--------|--------|----------------|------|
| LLaMA-130M 预训练 | Perplexity | **24.8** | 25.3 | 24.2 | 26.1 |
| LLaMA-350M 预训练 | Perplexity | **18.6** | 19.1 | 18.1 | 19.8 |
| LLaMA-7B 微调 (GLUE) | 平均准确率 | **86.2%** | 85.8% | 86.9% | 85.1% |
| 内存占用 (350M) | GPU 内存 | **4.2 GB** | 4.2 GB | 8.7 GB | 5.1 GB |

### 消融实验

| 配置 | Perplexity (130M) | 说明 |
|------|-------------------|------|
| GoLore (rank=128) | **24.8** | 最佳平衡 |
| GoLore (rank=64) | 25.9 | rank 过低损失信息 |
| GoLore (rank=256) | 24.5 | 接近全秩但内存增加 |
| GaLore (rank=128, batch=512) | 25.0 | 大 batch 下 GaLore 改善 |
| GaLore (rank=128, batch=64) | 26.2 | 小 batch 下 GaLore 退化 |

### 关键发现
- GaLore 确实在小 batch 设定下表现不稳定，验证了理论分析
- GoLore 在所有 batch 大小下表现稳定，且性能接近全秩 Adam
- GoLore 与 GaLore 内存占用相同（随机投影不需要额外内存）
- 投影更新频率 $T_0$ 对性能影响不大，50-200 步更新一次均可

## 亮点与洞察
- **理论与实践高度一致**: 反例和理论条件精确解释了 GaLore 的实际行为
- **修复方案简洁优雅**: 随机投影是最简单可能的修复，且不增加计算/内存开销
- **实用价值**: GoLore 为内存受限条件下的 LLM 训练提供了有理论保证的方案

## 局限与展望
- 随机投影在信息利用效率上可能不如 SVD，rank 需要设得更大
- 理论分析主要针对凸情况，LLM 训练的非凸性未完全覆盖
- 未探讨 GoLore 与其他内存高效方法（如 QLoRA、量化训练）的结合
- 大模型（7B+）的预训练实验有限

## 相关工作与启发
- GaLore (Zhao et al., 2024): 本文直接改进的对象
- LoRA (Hu et al., 2022): 另一种低秩优化方法
- GoLore 的随机投影思路可推广到其他需要子空间约束的优化问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 反例 + 修复方案的组合非常漂亮
- 实验充分度: ⭐⭐⭐⭐ 从合成到 LLM 预训练/微调全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 理论-实验对应清晰，故事线流畅
- 价值: ⭐⭐⭐⭐⭐ 对子空间优化的理论和实践都很重要

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](../../NeurIPS2025/optimization/the_trilemma_of_truth_in_large_language_models.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](../../NeurIPS2025/optimization/constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](../../NeurIPS2025/optimization/vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] Large Language Bayes](../../NeurIPS2025/optimization/large_language_bayes.md)

</div>

<!-- RELATED:END -->
