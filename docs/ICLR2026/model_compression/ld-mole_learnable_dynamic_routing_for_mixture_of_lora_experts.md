---
title: >-
  [论文解读] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts
description: >-
  [ICLR 2026][模型压缩][LoRA] 提出 LD-MoLE，用 Sparsegen 闭合形式投影替代传统 TopK 路由，实现可微分、动态、token自适应的 LoRA 专家分配，配合轻量 MLP 预测稀疏因子和解析稀疏损失，在多个基准上超越固定路由和 ReLU 路由基线。
tags:
  - ICLR 2026
  - 模型压缩
  - LoRA
  - Mixture-of-Experts
  - 动态路由
  - Sparsegen
  - parameter-efficient fine-tuning
---

# LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts

**会议**: ICLR 2026  
**arXiv**: [2509.25684](https://arxiv.org/abs/2509.25684)  
**代码**: [GitHub](https://github.com/eshentw/LD-MoLE)  
**领域**: 模型压缩  
**关键词**: LoRA, Mixture-of-Experts, 动态路由, Sparsegen, parameter-efficient fine-tuning

## 一句话总结
提出 LD-MoLE，用 Sparsegen 闭合形式投影替代传统 TopK 路由，实现可微分、动态、token自适应的 LoRA 专家分配，配合轻量 MLP 预测稀疏因子和解析稀疏损失，在多个基准上超越固定路由和 ReLU 路由基线。

## 研究背景与动机
LoRA + MoE（即 MoLE）是大模型高效微调的有前途方向：多个低秩 LoRA 模块作为专家，路由网络决定每个 token 使用哪些专家。但现有方法普遍依赖 TopK 路由，存在三个痛点：

**超参敏感**: k 值需要仔细调节，不同任务最优 k 不同

**不可微分**: TopK 选择是离散操作，阻碍端到端优化

**固定分配**: 每个 token 激活相同数量的专家，无法适应复杂度差异

ReMoE 用 ReLU 路由尝试解决，但存在某些 token 可能分配不到任何专家的不稳定问题。核心问题是：能否设计一种既稳定可微又能自适应控制专家数量的路由机制？

LD-MoLE 的切入角度是利用 Sparsegen——一种概率单纯形上的闭合形式投影，保证每个 token 至少分配一个专家，同时通过可学习的稀疏参数 $\lambda$ 实现动态专家选择。

## 方法详解

### 整体框架
在每个 Transformer 层的线性投影处放置多个 LoRA 专家。路由模块接收 token 嵌入，输出对各专家的稀疏权重分配。最终输出为基础权重输出加上所有活跃专家的加权输出之和。

### 关键设计
1. **Sparsegen 路由**:

    - 功能：将路由分数投影到概率单纯形上，生成稀疏分配
    - 核心思路：给定专家分数 $\bm{u} = \bm{W}_{\text{gate}} \bm{x}$，Sparsegen 求解优化问题 $\bm{p} = \arg\min_{\bm{p}} \|\bm{p} - \bm{u}\|^2 - \lambda\|\bm{p}\|^2$，约束 $\bm{p} \geq 0, \mathbf{1}^\top \bm{p} = 1$。闭合形式解为 $\bm{p}_i = \left[\frac{\bm{u}_i - \tau}{1-\lambda}\right]_+$
    - 设计动机：相比 TopK 的离散跳变，Sparsegen 有良定义的次梯度且上界有界，保证稳定优化。$\lambda \to 1^-$ 时趋向稀疏，$\lambda \to -\infty$ 时趋向均匀分布

2. **可学习动态稀疏因子**:

    - 功能：为每个 token 预测个性化的 $\lambda$ 值
    - 核心思路：轻量共享 MLP $f(\bm{x}) = \lambda \in \mathbb{R}$，根据输入维度共享（通常只有2种），极少参数开销
    - 设计动机：不同 token 的建模复杂度不同，复杂 token 需要更多专家，简单 token 只需少量

3. **解析稀疏损失**:

    - 功能：显式控制活跃专家数量
    - 核心思路：根据 Proposition 2 推导出激活恰好 k 个专家的 $\lambda$ 区间 $[\lambda_{\text{lower}}(k), \lambda_{\text{upper}}(k))$，稀疏损失为 $\mathcal{L}_{\text{sparse}} = \text{ReLU}(\lambda_{\text{lower}}(k) - \lambda)$
    - 设计动机：利用 Sparsegen 的解析特性直接约束稀疏度，无需启发式调参

### 损失函数 / 训练策略
总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \alpha \mathcal{L}_{\text{lb}} + \beta \mathcal{L}_{\text{sparse}}$
- $\mathcal{L}_{\text{LM}}$: 标准交叉熵（下一token预测或序列分类）
- $\mathcal{L}_{\text{lb}}$: 负载均衡损失，防止路由崩溃
- $\mathcal{L}_{\text{sparse}}$: 稀疏控制损失

8个LoRA专家，rank=8，scaling=16。4×H200 GPU训练10 epoch。

## 实验关键数据

### 主实验

| 方法 | 模型 | MMLU-P | ARC-C | ARC-E | OBQA | CommQA | SWAG | HellaS | CoLA | RTE | Avg |
|------|------|--------|-------|-------|------|--------|------|--------|------|-----|-----|
| MoLA(8888) | Llama-3B | 40.3 | 71.6 | 83.5 | 81.0 | 79.8 | 83.6 | 87.5 | 85.8 | 90.6 | 78.2 |
| MoLA(2468) | Llama-3B | 42.3 | 71.9 | 83.9 | 83.6 | 80.0 | 84.0 | 87.3 | 86.0 | 89.5 | 78.7 |
| ReMoLE | Llama-3B | 48.0 | 75.3 | 89.3 | 83.4 | 79.5 | 90.5 | 93.4 | 84.0 | 89.5 | 81.4 |
| **LD-MoLE** | Llama-3B | **49.6** | 74.6 | **89.5** | 83.8 | 80.3 | **90.8** | **93.6** | 85.5 | **91.0** | **82.0** |
| **LD-MoLE** | Llama-8B | **56.0** | **83.7** | **91.6** | **88.0** | 83.0 | **92.3** | **95.5** | 85.3 | **91.3** | **85.2** |

### 消融实验

| 配置 | 平均分 | 说明 |
|------|--------|------|
| LD-MoLE (β=0) | 82.0 | 无稀疏损失，全性能 |
| LD-MoLE (β>0, k≤4) | ~81.5 | 减少活跃专家，轻微性能下降 |
| MoLA(2468) vs MoLA(8888) | 78.7→78.2 | 固定路由中层间分配更重要 |
| ReMoLE (不稳定) | CoLA急剧下降 | ReLU路由可能分配0专家 |

### 关键发现
- 动态路由在指令微调任务上普遍优于固定路由，而分类任务两者差异较小
- LD-MoLE 保证每个 token 至少一个专家（Lemma 1），避免了 ReMoLE 的不稳定问题
- 稀疏损失可有效减少活跃专家数量而不显著影响性能
- MoLA(2468) 优于 MoLA(8888)，说明固定路由下许多专家被浪费

## 亮点与洞察
- Sparsegen 在 MoE 路由中的应用是关键创新点，兼顾可微分性和稀疏性
- 共享 MLP 预测 $\lambda$ 的设计简洁高效，参数开销极小
- 解析稀疏损失直接从数学性质推导，不需要启发式

## 局限与展望
- 主实验只在 3B 和 1.7B 级别模型上验证，更大模型尚未测试
- 训练成本（4×H200, 10epoch）对于PEFT方法来说仍然较高
- 推理时的路由计算（排序+MLP）的具体延迟未报告

## 相关工作与启发
- **vs MoLA (TopK)**: LD-MoLE 自适应 k 值，避免超参调节
- **vs ReMoE (ReLU)**: LD-MoLE 保证至少分配一个专家，更稳定
- **vs Soft MoE**: LD-MoLE 是稀疏的，计算效率更高

## 评分
- 新颖性: ⭐⭐⭐⭐ Sparsegen路由在MoLE中的应用新颖，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐ 多模型多任务评估，但缺少推理效率对比
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，但符号较多
- 价值: ⭐⭐⭐⭐ 为MoE路由提供了更好的数学框架

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICML 2025\] Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment](../../ICML2025/model_compression/make_lora_great_again_boosting_lora_with_adaptive_singular_values_and_mixture-of.md)
- [\[NeurIPS 2025\] Multi-Task Vehicle Routing Solver via Mixture of Specialized Experts under State-Decomposable MDP](../../NeurIPS2025/model_compression/multi-task_vehicle_routing_solver_via_mixture_of_specialized_experts_under_state.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing_with_temporal-ensemble_experts_.md)
- [\[CVPR 2026\] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling](../../CVPR2026/model_compression/enhancing_mixture_of_experts_specialization_via_cluster_aware_upcycling.md)

<!-- RELATED:END -->
