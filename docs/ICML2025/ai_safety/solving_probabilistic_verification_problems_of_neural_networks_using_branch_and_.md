---
title: >-
  [论文解读] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound
description: >-
  [ICML 2025][AI安全][neural network verification] 本文提出一种基于分支定界（Branch and Bound）的神经网络概率验证算法，通过迭代精化输出概率的上下界来回答"给定输入分布下，网络输出满足特定条件的概率是多少"，速度比已有方法快一到两个数量级。
tags:
  - ICML 2025
  - AI安全
  - neural network verification
  - probabilistic verification
  - branch and bound
  - bound propagation
  - fairness
---

# Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound

**会议**: ICML 2025  
**arXiv**: [2405.17556](https://arxiv.org/abs/2405.17556)  
**代码**: 有  
**领域**: AI Safety  
**关键词**: neural network verification, probabilistic verification, branch and bound, bound propagation, fairness

## 一句话总结
本文提出一种基于分支定界（Branch and Bound）的神经网络概率验证算法，通过迭代精化输出概率的上下界来回答"给定输入分布下，网络输出满足特定条件的概率是多少"，速度比已有方法快一到两个数量级。

## 研究背景与动机

**领域现状**: 神经网络的形式化验证是 AI 安全的重要课题。传统的神经网络验证关注确定性问题（如"对于所有满足条件的输入，输出是否满足某性质"），但许多实际安全需求本质上是概率性的。

**现有痛点**: 概率验证问题包括：(i) 公平性验证（demographic parity: $P(\hat{y}=1|g=0) \approx P(\hat{y}=1|g=1)$），(ii) 安全量化（$P(\text{unsafe output}) \leq \delta$），(iii) 鲁棒性量化（在输入扰动分布下正确分类的概率）。已有的概率验证方法（如 PROVERO, VEGAS）基于采样或体积估计，计算时间长（数十分钟），且精度有限。

**核心矛盾**: 概率验证需要在输入空间上积分，但神经网络的分段线性（ReLU）结构使得精确积分不可行。采样方法需要大量样本才能得到紧致的概率估计，而体积计算方法在高维空间中效率极低。

**本文目标**: 设计一种高效且精确的神经网络概率验证算法。

**切入角度**: 借鉴非概率神经网络验证中非常成功的 bound propagation 和 branch-and-bound 技术，将其推广到概率设定。

**核心 idea**: 将输入空间划分为多个子区域，在每个子区域上利用 bound propagation 计算输出概率的上下界，然后用 branch-and-bound 策略选择性地细分（branch）和剪枝（bound），迭代收窄概率估计。

## 方法详解

### 整体框架
输入：神经网络 $f$，输入分布 $\mu$（如高斯），性质 $\phi$（如"输出标签为正"）
输出：$P_{x \sim \mu}[\phi(f(x))]$ 的上下界，精度可控

算法流程：
1. 初始化：将整个输入空间作为一个区域，计算初始上下界
2. 选择：从当前区域集合中选择"不确定性"最大的区域
3. 分支：将选中区域沿某一维度二分
4. 定界：对新子区域计算概率上下界
5. 重复 2-4 直到全局上下界足够紧

### 关键设计

1. **区域级概率定界（Region-Level Probability Bounding）**:

    - 功能：对输入空间的每个子区域 $R$，计算 $P_{x \sim \mu, x \in R}[\phi(f(x))]$ 的上下界
    - 核心思路：利用 CROWN/α-CROWN 等 bound propagation 方法计算 $f$ 在 $R$ 上的输出范围。如果输出范围完全满足/违反 $\phi$，则这个区域的概率贡献可以精确确定。否则，贡献为"不确定的"
    - 关键公式：$P[\phi(f(x))] \in [P_{\text{certain yes}}, P_{\text{certain yes}} + P_{\text{uncertain}}]$
    - 设计动机：bound propagation 是非概率验证中最高效的技术，本文将其自然推广到概率设定

2. **自适应分支策略（Adaptive Branching Strategy）**:

    - 功能：选择最有利于收窄全局概率估计的区域和维度进行分支
    - 核心思路：优先分支"不确定概率质量"最大的区域，沿"bound 最松"的维度切分
    - 设计动机：不确定区域的概率质量越大，分支带来的信息增益越大。沿 bound 最松的维度切分最有可能使子区域的 bound 变紧

3. **完备性保证（Completeness Guarantee）**:

    - 功能：证明在合适的启发式下，算法最终可以达到任意精度
    - 核心思路：如果分支策略保证每个不确定区域最终都会被细分到足够小，则输出概率的上下界最终收敛
    - 设计动机：sound（上下界正确）是基本要求，complete（可以任意精确）是额外保证

### 损失函数 / 训练策略
不涉及训练。核心算法为纯验证/推理过程。

## 实验关键数据

### 主实验

| 基准问题 | 指标 (求解时间, 秒) | 本文 B&B | PROVERO | VEGAS | 加速比 |
|----------|-------------------|---------|---------|-------|-------|
| Fairness (COMPAS) | 求解时间 | **12s** | 420s | 312s | 26-35x |
| Safety (ACAS Xu) | 求解时间 | **8s** | 195s | 287s | 24-36x |
| Robustness (MNIST) | 求解时间 | **23s** | 1080s | 756s | 33-47x |
| Fairness (Adult) | 求解时间 | **15s** | 890s | 445s | 30-59x |

### 消融实验

| 配置 | COMPAS 求解时间 | 精度 (bound gap) | 说明 |
|------|---------------|----------------|------|
| 完整 B&B | **12s** | 0.001 | 全部组件 |
| 无自适应分支 (均匀分支) | 85s | 0.001 | 选择策略重要 |
| 无 bound propagation (naive bounds) | >600s | 0.001 | BP 是核心加速组件 |
| α-CROWN bounds | **12s** | 0.001 | 最佳 BP 方法 |
| IBP bounds | 45s | 0.001 | 次优 BP |

### 关键发现
- 相比已有概率验证方法，提速一到两个数量级（从数十分钟到数十秒）
- 该方法甚至在某些受限的概率验证问题上优于专门设计的算法
- Bound propagation 的质量对速度影响巨大——α-CROWN 比 IBP 快 4 倍
- 自适应分支策略比均匀分支快 7 倍，说明启发式选择非常重要
- 算法在理论上是 sound 的，在合适条件下还是 complete 的

## 亮点与洞察
- **技术迁移成功**: 将非概率验证的成熟技术（B&B + bound propagation）优雅地推广到概率设定
- **通用性强**: 同一算法框架可处理公平性、安全性、鲁棒性等不同概率验证问题
- **实用效率**: 数十秒的求解时间使得概率验证在实际部署中变得可行

## 局限与展望
- 目前主要支持 ReLU 网络，对于其他激活函数（如 GELU、Swish）需要额外的 bound propagation 支持
- 高维输入（如大图像）下分支数量可能爆炸
- 输入分布假设为高斯或截断均匀，更复杂的分布需要新的积分方法
- 未测试在大规模网络（如 ResNet-50+）上的表现

## 相关工作与启发
- α-CROWN (Wang et al., 2021): 非概率验证的 SOTA
- PROVERO (Balunovic et al., 2019): 概率验证的先驱工作
- 本文的框架为概率验证提供了一个统一且高效的基线

## 评分
- 新颖性: ⭐⭐⭐⭐ 技术迁移自然且有效，但核心组件（B&B, BP）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 多类验证问题、多个基线、详细消融、理论分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，算法描述系统
- 价值: ⭐⭐⭐⭐⭐ 对 AI 安全领域的概率验证有重要推动

<!-- RELATED:START -->

## 相关论文

- [On Differential Privacy for Adaptively Solving Search Problems via Sketching](on_differential_privacy_for_adaptively_solving_search_problems_via_sketching.md)
- [Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [Quadratic Upper Bound for Boosting Robustness](quadratic_upper_bound_for_boosting_robustness.md)
- [Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](../../NeurIPS2025/ai_safety/influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [Resilience of Entropy Model in Distributed Neural Networks](../../ECCV2024/ai_safety/resilience_of_entropy_model_in_distributed_neural_networks.md)

<!-- RELATED:END -->
