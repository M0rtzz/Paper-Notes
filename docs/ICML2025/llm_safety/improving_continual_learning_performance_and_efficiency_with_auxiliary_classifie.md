---
title: >-
  [论文解读] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers
description: >-
  [ICML 2025][continual learning] 本文首次探索了早退出网络（early-exit networks）在持续学习中的应用，发现早期分类器天然遭受更少的灾难性遗忘，并提出 Task-wise Logits Correction (TLC) 方法来均衡任务偏差，在阶段增量学习中以不到 70% 的计算量匹配标准方法的准确率。
tags:
  - ICML 2025
  - continual learning
  - early-exit networks
  - catastrophic forgetting
  - task-recency bias
  - dynamic inference
---

# Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers

**会议**: ICML 2025  
**arXiv**: [2403.07404](https://arxiv.org/abs/2403.07404)  
**代码**: https://anonymous.4open.science/r/ContinualEE (待公开)  
**领域**: 持续学习 / 高效推理  
**关键词**: continual learning, early-exit networks, catastrophic forgetting, task-recency bias, dynamic inference

## 一句话总结
本文首次探索了早退出网络（early-exit networks）在持续学习中的应用，发现早期分类器天然遭受更少的灾难性遗忘，并提出 Task-wise Logits Correction (TLC) 方法来均衡任务偏差，在阶段增量学习中以不到 70% 的计算量匹配标准方法的准确率。

## 研究背景与动机
**领域现状**：持续学习（CL）研究如何在非 i.i.d. 数据流上顺序学习而不遗忘旧知识。早退出网络（EEN）在网络中间层添加内部分类器（ICs），让"简单"样本提前退出以节省计算。

**现有痛点**：CL 方法只考虑单一最终分类器，忽略了中间层的预测潜力。EEN 仅在 i.i.d. 设置下研究，未考虑动态数据分布。两个领域各自为政。

**核心矛盾**：CL 中后层遗忘严重但原有方法只用后层预测；EEN 中任务偏差（task-recency bias）导致旧任务样本无法早退出。

**本文目标**：(1) 探索 EEN 在 CL 中的行为特性；(2) 解决任务偏差对动态推理的负面影响。

**切入角度**：分析发现早期 IC 遗忘更少、"过度思考"（overthinking）在 CL 中更严重，据此提出利用早退出来同时提升效率和性能。

**核心idea**：早退出 + 任务对数修正 = 降低计算的同时减轻遗忘。

## 方法详解

### 整体框架
输入：按任务序列到达的分类数据 $\{(\mathcal{X}_t, \mathcal{Y}_t)\}_{t=1}^T$
输出：带早退出功能的持续学习模型，推理时可动态选择在哪层退出

Pipeline：
1. 在标准 ResNet 的中间层放置 6 个内部分类器（ICs），按 SDN 架构放置在约 15%、30%、45%、60%、75%、90% 计算量的位置
2. 每个任务到来时，用加权联合损失训练所有 IC：$\min_\theta \sum_{i=1}^{M+1} w_i \cdot \mathcal{L}_i$
3. 推理时，如果某个 IC 的预测置信度超过阈值 $\tau$，则提前返回预测结果
4. 训练完所有任务后，应用 TLC 修正各任务的 logits 偏差

### 关键设计

1. **早退出持续学习架构**:

    - 功能：将 SDN（Shallow-Deep Network）架构适配到类增量学习设置中
    - 核心思路：每个 IC 由特征降维（FR）层 + 多头全连接（FC）层组成，新任务到来时为每个 IC 添加新的分类头
    - 设计动机：多头设计自然适配类增量设置，且 IC 共享骨干特征

2. **Task-wise Logits Correction (TLC)**:

    - 功能：修正任务偏差使旧任务预测也能获得足够高的置信度以早退出
    - 核心思路：对第 $t$ 个任务的 logits 加上修正项 $c_t = a \cdot (T - t) + b$，其中 $a, b$ 通过最小化各任务平均最大 logit 差异来优化：
    $E(a,b) = \sum_{i=1}^{N} \sum_{t=1}^{T} (M_j - m_t^{i,j})^2$
      其中 $m_t^{i,j} = \max(\bar{y}_t^{i,j} + a(T-t) + b)$，$M_j$ 是跨所有任务和分类器的平均最大 logit
    - 设计动机：任务偏差使旧任务样本置信度低，不会在早期 IC 退出，而这些 IC 恰恰对旧任务遗忘更少。修正偏差后老任务样本可以在遗忘少的早层退出，一举两得

3. **过度思考分析（Overthinking Analysis）**:

    - 功能：量化分析了 CL 中过度思考现象
    - 核心思路：定义过度思考为 oracle 准确率（至少一个 IC 正确）与最终分类器准确率之差。发现 CL 中过度思考比联合训练更严重
    - 设计动机：这一发现是使用早退出策略的理论基础——既然后层会"想多了"导致错误，不如相信早层

### 损失函数 / 训练策略
- 主损失：$\sum_{i=1}^{M+1} w_i \cdot \mathcal{L}_i(C_i^t(E_i(\mathcal{X}_t; \theta_i)), \mathcal{Y}_t)$
- $w_i$ 采用 SDN 渐进调度器，训练初期更重视早期 IC，后期逐步增加深层 IC 权重
- 与基础 CL 方法（LwF、ER、BiC、iCaRL 等）的组合：对每个 IC 复制原方法的正则化/蒸馏逻辑

## 实验关键数据

### 主实验（10 任务，类增量学习）

| 方法 | CIFAR100 (100% FLOPs) | CIFAR100 (75% FLOPs) | ImageNet-Sub (100%) | ImageNet-Sub (80%) |
|---|---|---|---|---|
| iCaRL (标准) | 41.23 | — | 38.92 | — |
| iCaRL + EE | 38.80 | 36.81 | 36.58 | 34.34 |
| iCaRL + EE + TLC | **47.98** | **46.22** | **51.90** | **49.52** |
| BiC (标准) | 46.94 | — | 50.86 | — |
| BiC + EE + TLC | **49.10** | **47.58** | **55.38** | **53.28** |
| LwF (标准) | 25.03 | — | 24.60 | — |
| LwF + EE + TLC | **30.10** | **29.81** | **29.68** | **29.92** |

### 消融实验

| 配置 | CIFAR100 准确率 | FLOPs 占比 | 说明 |
|---|---|---|---|
| 标准网络（无EE） | 41.23 (iCaRL) | 100% | 基线 |
| + 早退出（无TLC） | 38.80 | 100% | 任务偏差影响 |
| + 早退出 + TLC | 47.98 | 100% | 显著提升 |
| + 早退出 + TLC | 46.22 | 75% | 省25%计算，仍超基线 |
| + 早退出 + TLC | 37.98 | 50% | 省50%计算，接近基线 |

### 关键发现
- **早期IC遗忘更少**：以LwF为例，IC4在Task1上准确率约30%，而最终分类器仅13%，验证了低层"记忆力更好"
- **CL中过度思考更严重**：oracle准确率与最终分类器准确率差距在CL中远大于联合训练
- **TLC提升巨大**：对iCaRL在CIFAR100上提升约7个点，对ImageNet-Subset提升约13个点
- 使用70%计算量即可匹配标准方法满计算量的性能

## 亮点与洞察
- 首次揭示了"早期层遗忘少"这一持续学习中的结构性发现，并巧妙利用它
- TLC 方法极其简洁（仅两个参数 $a, b$），但效果显著
- 减少计算和减少遗忘不是矛盾的——这是一个反直觉但重要的发现

## 局限与展望
- TLC 使用线性修正模型，对复杂偏差模式可能不够
- 目前只验证了 ResNet32/18 架构，在 Transformer 等架构上的效果未知
- 退出阈值 $\tau$ 的选择仍然是一个开放问题

## 相关工作与启发
- 结合了 SDN、BranchyNet 等早退出方法与 iCaRL、BiC 等 CL 方法
- 启发：网络不同深度的层具有不同的遗忘特性，未来的 CL 方法可以更精细地利用这一结构
- 对资源受限场景（边缘设备上的持续学习）有重要实用价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将早退出与持续学习结合，发现了结构性洞察
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、六种CL方法、多种计算预算的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 分析深入、逻辑清晰、表格丰富
- 价值: ⭐⭐⭐⭐⭐ 同时提升效率和性能，实用价值突出

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[ICML 2025\] BECAME: BayEsian Continual Learning with Adaptive Model MErging](became_bayesian_continual_learning_with_adaptive_model_merging.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[ICML 2025\] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin.md)
- [\[ACL 2025\] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](../../ACL2025/llm_safety/rate-ft-auxiliary-tasks-for-hallucination-detection.md)

<!-- RELATED:END -->
