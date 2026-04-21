---
title: >-
  [论文解读] LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation
description: >-
  [ICML 2025][LLM/NLP] LaRoSA 提出了一种无需训练的激活稀疏化方法，通过逐层正交旋转矩阵将输入激活变换到更适合稀疏化的空间，并结合 Top-K 选择实现一致的模型级稀疏度和可靠的推理加速。
tags:
  - ICML 2025
  - LLM/NLP
---

# LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation

**会议**: ICML 2025  
**arXiv**: [2507.01299](https://arxiv.org/abs/2507.01299)  
**领域**: 人体理解  

## 一句话总结

LaRoSA 提出了一种无需训练的激活稀疏化方法，通过逐层正交旋转矩阵将输入激活变换到更适合稀疏化的空间，并结合 Top-K 选择实现一致的模型级稀疏度和可靠的推理加速。

## 研究背景与动机

大语言模型的高效推理是当前的重要研究方向。利用激活稀疏性可以跳过零值激活对应的权重通道，从而减少内存传输和计算开销。然而，现有方法存在两个主要限制：

**基于 ReLU 的方法**（如 DejaVu）需要大量恢复训练，且现代 LLM（如 LLaMA3、Qwen2.5）使用 SwiGLU 等非 ReLU 激活函数，不会自然产生稀疏性

**基于幅度剪枝的方法**（如 CATS、TEAL）使用离线校准阈值，存在三大问题：
   - 阈值定义的**模糊性和不准确性**：校准阈值很难与实际需要的阈值对齐
   - **无法维持一致的稀疏度**：实际稀疏度显著偏离目标值
   - **幅度与通道重要性的错误假设**：低幅度激活若对应高范数权重通道，仍会显著影响输出

## 方法详解

### 核心思想

LaRoSA 的关键洞察是：通过正交旋转变换，可以将激活向量变换到一个通道重要性更容易区分的空间中，从而实现更有效的稀疏化。

### 逐层正交旋转

对每一层 $l$，LaRoSA 使用 PCA 构造正交旋转矩阵 $\mathbf{Q}_l$。具体步骤：

1. 选择校准数据集（$M$ 个序列），前向传播得到每层输入激活 $\mathbf{X}_l^i$
2. 计算协方差矩阵并求平均：

$$\text{Cov}(\mathbf{X}_l, \mathbf{X}_l^T) = \frac{1}{M}\sum_{i=0}^{M}\mathbf{X}_l^i(\mathbf{X}_l^i)^T$$

3. 对协方差矩阵做特征分解，按特征值降序排列特征向量构成 $\mathbf{Q}_l$

### 残差适配器

由于残差连接要求每层使用相同旋转矩阵，但不同层的最优旋转差异很大。LaRoSA 引入**残差适配器** $\mathbf{Q}_l^T\mathbf{Q}_{l+1}$ 来实现逐层独立旋转。第一层和最后一层的旋转矩阵可以分别合并到 token embedding 和 LM head 层中。

### 一致的激活稀疏度

LaRoSA 用 Top-K 函数替代幅度剪枝，在旋转后的激活上进行稀疏化：

$$S_k(\tilde{x}_i) = \begin{cases} \tilde{x}_i, & \text{if } |\tilde{x}_i| \in \text{Top}_k(|\tilde{x}_i|) \\ 0, & \text{otherwise} \end{cases}$$

其中 $k = \alpha \cdot (1-p) \cdot D_{\text{in}}$，$p$ 是目标稀疏度，$\alpha$ 是控制同一 block 内 $h_1$ 和 $h_2$ 稀疏系数的超参数。

### 权重吸收

旋转矩阵 $\mathbf{Q}_l$ 可以预先与权重矩阵合并，避免推理时的额外计算：

$$\mathbf{Y}_l = S_k(\mathbf{X}_l\mathbf{Q}_l) \cdot (\mathbf{W}_l\mathbf{Q}_l)^T$$

### 硬件高效的自定义内核

基于 Triton 实现了 GEMV 内核：采用列主序存储权重、融合 Top-K 到矩阵向量乘法中、选择性加载稀疏激活和对应权重列。

## 实验

### 主实验 - 零样本任务准确率

| 方法 | LLaMA2-7B Acc7 | LLaMA3-8B Acc7 | Qwen2.5-7B Acc7 |
|---|---|---|---|
| Dense | 66.69 | 70.05 | 70.34 |
| CATS 40% | 49.55 | 55.11 | 61.83 |
| TEAL 40% | 64.92 | 68.14 | 68.61 |
| **LaRoSA 40%** | **66.15** | **68.79** | **69.67** |
| TEAL 50% | 63.22 | 64.92 | 67.76 |
| **LaRoSA 50%** | **64.61** | **67.19** | **69.09** |

### 困惑度结果

在 LLaMA2-7B 上 40% 稀疏度下，LaRoSA 仅有 **0.17** 的困惑度差距（5.64 vs 5.47），而 TEAL 为 0.93，CATS 高达 39.99。

### 推理加速

LaRoSA 在 A100 上以 50% 稀疏度实现 **1.38×** 加速，75% 稀疏度实现 **1.72×** 加速。由于使用 Top-K 保证一致稀疏度，加速效果稳定可预测。

### 推理模型实验

在 DeepSeek-R1-Distill-Llama3-8B 上，LaRoSA 25% 稀疏度在 MATH-500 上仅下降 2.6 分（85.0 vs 87.6），AIME-2024 保持不变（40.0）。

## 亮点

- **无需训练**：仅需 12 分钟校准即可用于 70B 模型，实际部署非常友好
- **一致的稀疏度**：Top-K 保证每个 token 的稀疏度恒定，解决了幅度剪枝的不稳定问题
- **理论支撑**：附录提供了旋转在减少逐层经验误差方面优于幅度剪枝的理论分析
- **跨模型鲁棒性**：在 LLaMA2/3、Qwen2.5、Mistral 的 7B 和 70B 模型上均表现出色
- **兼容推理模型**：在 DeepSeek-R1 蒸馏模型上验证了对推理能力的保持

## 局限性

- 仅旋转 $h_1$ 和 $h_3$ 的输入激活，$h_2$ 和 $h_4$ 由于 GQA 和元素乘法的约束无法旋转
- 残差适配器引入了少量额外计算
- 超参数 $\alpha$ 需要通过网格搜索针对每个模型调优
- Top-K 操作本身有一定开销，需要自定义 GPU 内核才能带来实际加速
- 在极高稀疏度（60%+）下性能下降仍较明显

## 评分

⭐⭐⭐⭐ (4/5)

LaRoSA 方法优雅且实用，通过正交旋转巧妙解决了非 ReLU LLM 的激活稀疏化问题。无需训练的特性使其非常适合实际部署，实验充分覆盖了多种模型和任务。

<!-- RELATED:START -->

## 相关论文

- [Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](../../ACL2025/llm_nlp/salience_sparse_fine_tuning.md)
- [Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)
- [Detecting High-Stakes Interactions with Activation Probes](../../NeurIPS2025/llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)
- [QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions](quest_enhancing_estimates_of_quantile-based_distributional_measures_using_model_.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)

<!-- RELATED:END -->
