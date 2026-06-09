---
title: >-
  [论文解读] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP
description: >-
  [NeurIPS 2025][模型压缩][车辆路径问题] 提出 **State-Decomposable MDP (SDMDP)** 框架将多种 VRP 变体重新表述为基础状态空间的笛卡尔积…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "车辆路径问题"
  - "混合专家"
  - "状态可分解MDP"
  - "LoRA"
  - "多任务学习"
---

# Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP

**会议**: NeurIPS 2025  
**arXiv**: [2510.21453](https://arxiv.org/abs/2510.21453)  
**代码**: [github.com/panyxy/moses_vrp](https://github.com/panyxy/moses_vrp)  
**领域**: 模型压缩  
**关键词**: 车辆路径问题, 混合专家, 状态可分解MDP, LoRA, 多任务学习  
**arXiv**: [2510.21453](https://arxiv.org/abs/2510.21453)  
**代码**: 无  
**领域**: 模型压缩  

## 一句话总结

提出 **State-Decomposable MDP (SDMDP)** 框架将多种 VRP 变体重新表述为基础状态空间的笛卡尔积，再通过 **Mixture-of-Specialized-Experts Solver (MoSES)** 用专用 LoRA 专家实现基础策略的潜在空间复用，高效处理 16 种 VRP 变体。

## 研究背景与动机

**领域现状**：车辆路径问题（VRP）是经典组合优化问题，实际应用涉及多种变体（容量、开放路线、回程取货、时间窗、路程限制等），神经求解器在单一VRP上表现良好。

**现有痛点**：专用神经求解器需要为每种VRP变体从头训练，成本极高；统一求解器虽然能同时处理多种变体，但未能充分利用VRP变体的**组合结构**——每种变体都由共享的基础VRP变体派生。

**核心矛盾**：VRP变体数量随属性组合指数增长（5个基础约束可产生16种变体），现有微调/适配器方法无法高效扩展；统一模型则未能利用基础求解器的专业知识。

**本文目标**：让统一求解器感知VRP变体间的共享组件本质，主动复用针对基础VRP变体训练的专业求解器。

**切入角度**：从MDP状态空间分解出发，证明基础策略的最优性可以传递到统一策略。

**核心 idea**：将VRP的状态空间分解为基础状态空间的笛卡尔积，通过冻结的LoRA基础专家+自适应门控机制在潜在空间中混合策略。

## 方法详解

### 整体框架

三阶段流程：(1) 在CVRP上预训练共享骨干模型作为第0个基础策略；(2) 用 Gated-LoRA 对每个基础VRP变体微调专用 LoRA 专家；(3) 冻结骨干和LoRA权重，训练自适应门控机制动态聚合专家。

### 关键设计 1：State-Decomposable MDP (SDMDP)

- **功能**：将VRP的MDP状态空间形式化分解为基础状态空间的笛卡尔积。
- **核心思路**：完整状态空间 $\bar{\mathcal{S}}_t = \prod_{i=0}^{n} \mathcal{S}_t^{(i)}$，每个状态 $\bar{s}_t = \{s_t^{(i)}\}_{i=0}^n$ 可分解为条件独立的基础状态。转移概率也相应因式分解：$\mathcal{P}(s_{t+1}|s_t, a_t) = \prod_{i=0}^m \mathcal{P}(s_{t+1}^{(b_i)} | s_t^{(b_i)}, a_t)$。
- **设计动机**：VRP变体的动态上下文和约束天然来自不同的基础变体，这种结构化分解使基础策略复用成为可能。
- **理论保证（Theorem 1）**：最优统一策略 $\pi^*$ 与第 $i$ 个最优基础策略 $\pi^{(i)*}$ 在对应基础任务的状态上价值函数相等。

### 关键设计 2：Latent Space-based SDMDP (LS-SDMDP)

- **功能**：在潜在空间中复用最优基础策略。
- **核心思路**：引入混合函数 $f_\phi$ 将基础策略生成的嵌入 $\{z^{(b_i)}\}$ 映射为统一状态嵌入：$z = f_\phi(z^{(b_0)}, \ldots, z^{(b_m)}; s)$，策略重写为：
$$\pi(a|s) = \sum_{z^{(b_0)},\ldots,z^{(b_m)}} \pi(a|f_\phi(z^{(b_0)},\ldots,z^{(b_m)};s)) \prod_{i=0}^m \pi(z^{(b_i)}|s^{(b_i)})$$
- **理论保证（Theorem 2）**：在温和假设下，LS-SDMDP的最优统一策略可恢复SDMDP的最优统一策略。

### 关键设计 3：MoSES 实现

- **Gated-LoRA 专家**：为每个基础VRP变体（除CVRP外），在冻结骨干的线性层上添加 LoRA 矩阵 $B_i A_i$，并用动态门控抑制骨干中的任务不相关特征：
$$h^{\text{out}} = \text{sigmoid}(\langle W^g, h^{\text{in}} \rangle) W_0 h^{\text{in}} + B_i A_i h^{\text{in}}$$
- **自适应门控聚合**：冻结所有权重后，训练门控函数 $G(h^{\text{in}}) = \text{act}(W^G h^{\text{in}})$ 计算系数 $\{\alpha_i\}_0^4$，加上残差LoRA $\hat{B}\hat{A}$：
$$h^{\text{out}} = \alpha_0 W_0 h^{\text{in}} + \sum_{i=1}^4 \alpha_i B_i A_i h^{\text{in}} + \hat{B}\hat{A} h^{\text{in}}$$
- **三种门控激活**：softmax（凸组合）、norm_softplus（防梯度消失）、sigmoid（放松单位和约束）
- **三种路由策略**：Dense Routing、Variant-Aware Routing-I（Top-K）、Variant-Aware Routing-II（精确匹配）

### 损失函数

使用 REINFORCE 算法，奖励为 $-c(\tau)$（路线总长度的负数），训练过程分三阶段各自优化。

## 实验关键数据

### 主实验：16种VRP变体（N=50, N=100）

| Solver | CVRP Cost (N=50) | Gap | OVRP Cost (N=50) | Gap |
|---|---|---|---|---|
| HGS-PyVRP (精确) | 10.372 | * | 6.507 | * |
| RF-TE | 10.504 | 1.276% | 6.684 | 2.693% |
| CaDA | 10.483 | 1.072% | 6.662 | 2.350% |
| **MoSES(RF)** | **10.465** | **0.900%** | **6.632** | **1.892%** |
| **MoSES(CaDA)** | **10.462** | **0.873%** | **6.629** | **1.857%** |

| Solver | VRPTW Cost (N=100) | Gap | VRPBLTW Cost (N=100) | Gap |
|---|---|---|---|---|
| HGS-PyVRP | 25.423 | * | 29.026 | * |
| RF-TE | 26.234 | 3.177% | 30.688 | 2.923% |
| CaDA | 26.128 | 2.753% | 30.586 | 2.579% |
| **MoSES(RF)** | **26.143** | **2.822%** | **30.627** | **2.712%** |
| **MoSES(CaDA)** | **26.032** | **2.383%** | **30.510** | **2.329%** |

### 消融实验

- **门控激活函数**：sigmoid > norm_softplus > softmax，放松单位和约束有助于更灵活的专家组合
- **路由策略**：Variant-Aware Routing-II（精确匹配基础变体）通常最优
- **残差LoRA**：去掉 $\hat{B}\hat{A}$ 后性能下降，验证了线性混合专家的表达不足
- **Gated-LoRA vs 普通LoRA**：门控机制显著优于直接LoRA微调

### 关键发现

- MoSES 在所有16种VRP变体上都优于现有统一求解器（MTPOMO、MVMoE、RF系列、CaDA）
- 推理时间相比基线增加约2-3倍（因专家并行计算），但仍远快于精确方法（秒 vs 分钟）
- 该框架具有即插即用特性：可应用于不同的骨干网络（RF和CaDA）

## 亮点与洞察

1. **理论与实践统一**：SDMDP不仅是形式化工具，Theorem 1和2为专家复用提供了理论正当性
2. **组合结构挖掘**：首次将VRP的组合属性结构显式建模为状态空间分解，避免了变体数量的指数增长问题
3. **参数高效**：冻结骨干+LoRA的架构使得新增基础变体的边际成本极低
4. **可解释的专家激活**：门控系数可以揭示不同VRP变体对各基础策略的依赖程度

## 局限与展望

1. **基础变体定义依赖领域知识**：5个基础VRP变体的选择是预定义的，自动发现基础任务是未来方向
2. **推理效率**：多专家并行增加了推理开销（约2-3倍），在实时场景中可能受限
3. **扩展到更多约束**：当基础变体数量进一步增长时，门控机制的复杂度如何扩展需要验证
4. **与精确方法的差距**：在复杂变体上仍有2-5%的最优性差距
5. **LoRA rank选择**：当前使用固定rank，自适应rank可能带来进一步收益

## 相关工作与启发

- **与 MVMoE 的区别**：MVMoE学习隐式的、可解释性差的专家特化，MoSES使用显式的预训练基础求解器作为专家
- **与 LoRA hub 类方法的不同**：不仅是LoRA适配器的简单组合，还有理论上的最优策略恢复保证
- **对其他组合优化的启发**：SDMDP的状态分解思路可推广到任何可自然分解为子问题的组合优化场景

## 评分

⭐⭐⭐⭐ (4/5)

理论严谨、实验全面的出色工作。SDMDP框架优雅地解决了VRP变体指数增长的难题，MoSES实现简洁有效。主要不足是推理效率开销和基础变体定义的人工依赖性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MTL-KD: Multi-Task Learning Via Knowledge Distillation for Generalizable Neural Vehicle Routing Solver](mtl-kd_multi-task_learning_via_knowledge_distillation_for_generalizable_neural_v.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
