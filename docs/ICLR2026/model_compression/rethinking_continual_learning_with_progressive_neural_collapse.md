---
title: >-
  [论文解读] Rethinking Continual Learning with Progressive Neural Collapse
description: >-
  [模型压缩] 提出 ProNC 框架，通过渐进式扩展等角紧框架（ETF）目标替代固定预定义 ETF，在持续学习中实现最大类间分离与最小遗忘的平衡。
tags:
  - 模型压缩
---

# Rethinking Continual Learning with Progressive Neural Collapse

## 基本信息

- **会议**: ICLR 2026
- **arXiv**: [2505.24254](https://arxiv.org/abs/2505.24254)
- **代码**: [GitHub](https://github.com/Continue-Edge-AI-Lab/ProNC)
- **领域**: 持续学习 / 模型压缩
- **关键词**: Continual Learning, Neural Collapse, ETF, Class-Incremental Learning, Knowledge Distillation

## 一句话总结

提出 ProNC 框架，通过渐进式扩展等角紧框架（ETF）目标替代固定预定义 ETF，在持续学习中实现最大类间分离与最小遗忘的平衡。

## 研究背景与动机

### 问题背景
持续学习（CL）旨在让模型在不断学习新任务的同时不遗忘旧知识，核心挑战是灾难性遗忘（Catastrophic Forgetting）。近年来，研究发现深度网络训练末期会出现 **Neural Collapse (NC)** 现象——所有类别的特征原型几何上收敛为一个 Simplex ETF（等角紧框架），实现类间最大等距分离。

### 现有方法的局限
已有工作（如 NCT）尝试在 CL 中预定义一个全局固定 ETF 作为训练目标，但存在三大问题：

**不切实际**：预定义 ETF 需要事先知道所有任务的总类别数，这在实际场景中不可能；

**性能受限**：当总类别数很大时，ETF 顶点间距变小，阻碍了早期阶段的类别判别能力（如图 1 所示，k 增大时精度下降）；

**违反 NC 规律**：NC 本身是训练过程中自演化的现象，随机初始化 ETF 容易导致几何失配。

### 核心洞察
ETF 目标中的顶点数应始终等于当前已见类别数，才能保持最大类间分离。因此需要一种动态、渐进式的 ETF 扩展机制。

## 方法详解

### 整体框架

ProNC = **Progressive Neural Collapse**，包含两个核心步骤：ETF 初始化与 ETF 扩展，并结合对齐损失和蒸馏损失用于 CL 训练。

### 1. ETF 初始化（任务 1 结束后）

训练完第一个任务后，利用学到的类别特征均值 $\tilde{M}_{K_1}$ 构造最接近的 ETF：

$$
\mathbf{E}^* = \sqrt{\frac{K_1}{K_1-1}} \mathbf{W}\mathbf{V}^\top \left(\mathbf{I}_{K_1} - \frac{1}{K_1}\mathbf{1}_{K_1}\mathbf{1}_{K_1}^\top\right)
$$

其中 $\mathbf{W}\mathbf{\Sigma}\mathbf{V}^\top$ 是 $\mathbf{U}'$ 的 SVD 分解。这保证了初始 ETF 与实际学到的特征对齐，避免随机初始化带来的几何失配。

### 2. ETF 渐进扩展（新任务到来时）

当新任务 $t$ 带来 $K_t - K_{t-1}$ 个新类时：

- **Step a**：将原正交基 $\mathbf{U}_{t-1} \in \mathbb{R}^{d \times K_{t-1}}$ 通过 Gram-Schmidt 正交化扩展为 $\mathbf{U}_t \in \mathbb{R}^{d \times K_t}$，新增向量与已有基正交；
- **Step b**：将 $\mathbf{U}_t$ 和 $K_t$ 代入 ETF 构造公式（公式 1），得到具有 $K_t$ 个顶点的新 ETF 目标 $\mathbf{E}_t$。

关键性质：保持原正交基不变，使旧类对应顶点的漂移最小化。

### 3. 损失函数设计

对于任务 $t \geq 2$，模型训练使用三项损失的加权和：

$$
\mathcal{L} = \mathcal{L}_{\text{ce}} + \lambda_1 \cdot \mathcal{L}_{\text{align}} + \lambda_2 \cdot \mathcal{L}_{\text{distill}}
$$

**(1) 对齐损失**：将学到的特征推向 ETF 目标顶点：

$$
\mathcal{L}_{\text{align}}(\boldsymbol{\mu}_{k,i}^t, \mathbf{e}_{k,t}) = \frac{1}{2}(\mathbf{e}_{k,t}^\top \boldsymbol{\mu}_{k,i}^t - 1)^2
$$

**(2) 蒸馏损失**：缓解旧类特征因 ETF 扩展而产生的漂移：

$$
\mathcal{L}_{\text{distill}}(\boldsymbol{\mu}_{k,i}^{(t-1)}, \boldsymbol{\mu}_{k,i}^{(t)}) = \frac{1}{2}((\boldsymbol{\mu}_{k,i}^{(t-1)})^\top \boldsymbol{\mu}_{k,i}^{(t)} - 1)^2
$$

### 4. 推理阶段

使用基于余弦相似度的最近 ETF 顶点分类器取代线性分类器：

$$
\hat{y} = \arg\max_k \boldsymbol{\mu}_j^\top \mathbf{e}_k
$$

## 实验

### 主实验结果

| Buffer | 方法 | Seq-CIFAR-10 (Class-IL) | Seq-CIFAR-100 (Class-IL) | Seq-TinyImageNet (Class-IL) |
|--------|------|------------------------|--------------------------|----------------------------|
| 200 | ER | 44.79 | 21.78 | 8.49 |
| 200 | DER++ | 64.88 | 28.13 | 11.34 |
| 200 | STAR | 65.94 | 38.15 | 13.64 |
| 200 | NCT (固定 ETF) | 51.59 | 26.38 | 10.95 |
| 200 | **ProNC (本文)** | **72.70** | **44.32** | **20.11** |
| 500 | DER++ | 72.25 | 41.67 | 19.69 |
| 500 | STAR | 73.42 | 49.72 | 22.18 |
| 500 | **ProNC (本文)** | **79.42** | **52.49** | **28.27** |

### 消融实验

| 组件 | Seq-CIFAR-10 (FAA) | Seq-CIFAR-100 (FAA) | Seq-TinyImageNet (FAA) |
|------|-------------------|--------------------|-----------------------|
| 完整 ProNC | **72.70** | **44.32** | **20.11** |
| 无对齐损失 | 65.94 | 38.15 | 13.64 |
| 无蒸馏损失 | 69.82 | 41.76 | 17.53 |
| 固定全局 ETF (NCT) | 51.59 | 26.38 | 10.95 |

### 关键发现

1. **ProNC 在所有数据集上均大幅超过基线**，特别是在 TinyImageNet 上将 Class-IL 精度提升了 6+ 个百分点；
2. **对齐损失是最关键组件**，去除后退化至 STAR 水平；
3. **渐进 ETF 远优于固定 ETF**，NCT 的固定全局 ETF 在大类别数下性能严重受限；
4. **遗忘率显著降低**，ProNC 的平均遗忘率远低于 DER++ 和 STAR。

## 亮点

- 完全无需预定义全局 ETF，从第一个任务中自适应提取初始 ETF 并渐进扩展
- 理论保证（Theorem 1）确保初始 ETF 最优对齐
- ETF 扩展策略基于正交基保持，最小化旧类顶点漂移
- 框架简洁灵活，可作为任何回放式 CL 方法的插件式特征正则化

## 局限性

- 特征维度 $d$ 必须满足 $d \geq K-1$，当总类别数接近特征维度时 ETF 扩展受限
- 仍依赖数据回放（replay buffer），纯无回放场景下效果未验证
- ETF 构造中的 SVD 计算在类别数极大时可能产生额外开销
- 仅在 ResNet-18 上验证，未涉及更大规模模型或实际部署场景

## 相关工作

- **Neural Collapse**: Papyan et al. (2020) 发现训练末期特征收敛为 Simplex ETF
- **基于 ETF 的 CL**: NCT (Yang et al., 2023b) 预定义固定全局 ETF；MNC3L (Dang et al., 2025) 结合对比学习
- **回放式 CL**: DER/DER++ (Buzzega et al., 2020), STAR (Eskandar et al., 2025)
- **知识蒸馏式 CL**: iCaRL (Rebuffi et al., 2017), LODE (Liang & Li, 2023)

## 评分

- 新颖性：⭐⭐⭐⭐ — 渐进 ETF 扩展思路新颖且有理论支撑
- 技术深度：⭐⭐⭐⭐ — 从理论到实现完整，Theorem 1 提供了严格数学保证
- 实验充分度：⭐⭐⭐⭐ — 覆盖3个数据集、2种CL场景，消融全面
- 实用价值：⭐⭐⭐⭐ — 即插即用的特征正则化，兼容多种CL框架

<!-- RELATED:START -->

## 相关论文

- [Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)

<!-- RELATED:END -->
