---
title: >-
  [论文解读] Towards Trustworthy Federated Learning with Untrusted Participants
description: >-
  [ICML 2025][AI安全][联邦学习] 提出 CafCor 算法，通过参与者间的共享随机性实现关联噪声注入，结合新型鲁棒聚合方法 CAF，在不信任服务器、存在恶意参与者的联邦学习场景下，实现接近中心化 DP 的隐私-效用权衡。
tags:
  - ICML 2025
  - AI安全
  - 联邦学习
  - 差分隐私
  - 鲁棒聚合
  - 拜占庭容错
  - 关联噪声
---

# Towards Trustworthy Federated Learning with Untrusted Participants

**会议**: ICML 2025  
**arXiv**: [2505.01874](https://arxiv.org/abs/2505.01874)  
**代码**: 无（论文提及计划公开）  
**领域**: AI安全  
**关键词**: 联邦学习, 差分隐私, 鲁棒聚合, 拜占庭容错, 关联噪声

## 一句话总结

提出 CafCor 算法，通过参与者间的共享随机性实现关联噪声注入，结合新型鲁棒聚合方法 CAF，在不信任服务器、存在恶意参与者的联邦学习场景下，实现接近中心化 DP 的隐私-效用权衡。

## 研究背景与动机

在分布式学习中，**隐私**和**鲁棒性**是构建可信系统的两大关键需求，但它们的交叉研究仍不充分。

现有方案面临根本性权衡：
- **本地差分隐私（LDP）**：不信任服务器，但效用损失极大，隐私-效用权衡为 Θ(d/(nε²))
- **中心差分隐私（CDP）**：假设服务器完全可信，效用好得多，权衡为 Θ(d/(n²ε²))
- CDP 比 LDP 好 n 倍，但完全信任服务器在实际中不现实

**核心问题**：能否在不信任服务器、且存在恶意参与者的情况下，实现接近 CDP 的效用？

本文提出的 **SecLDP**（基于秘密的本地差分隐私）模型是关键——每对参与者共享一个对其他人未知的随机种子。这只需一轮加密通信即可实现，是比 CDP 弱得多的信任假设。

## 方法详解

### 整体框架

**CafCor** = Correlated noise + CAF aggregation，包含两个核心组件：

1. **关联噪声注入**（Correlated Noise）：利用参与者间的共享随机性
2. **协方差无关滤波器聚合**（CAF: Covariance-bound Agnostic Filter）：新型鲁棒聚合

工作流程（Algorithm 1）：
- 服务器广播模型 θ_t
- 每个诚实工作者 i 采样 mini-batch、计算裁剪梯度 g_t^(i)
- 注入两种噪声：独立噪声 v̄_t^(i) ~ N(0, σ_ind² I_d) + 关联噪声 Σ_j v_t^(ij)，其中 v_t^(ij) = -v_t^(ji)
- 加入动量后发送给服务器
- 服务器用 CAF 聚合后更新模型

### 关键设计

**1. 关联噪声机制**

核心思想：每对工作者 (i,j) 共享随机种子，生成对称噪声 v_t^(ij) = -v_t^(ji) ~ N(0, σ_cor² I_d)。

- **聚合时取消**：如果对所有诚实工作者的梯度求平均，关联噪声会互相抵消
- **隐私保护**：服务器无法获知共享种子，因此看到的每个工作者的消息包含大量噪声
- **即使恶意者勾结**：只要存在至少一个不与服务器勾结的恶意者，其共享的关联噪声就提供了隐私保护

噪声量级只需 σ_cor² = Θ(1/(nε²))，而 LDP 中需要 σ_ind² = Θ(1/ε²)，相差 n 倍。

**2. CAF 聚合（Algorithm 2）**

CAF 通过迭代降权方式过滤恶意输入：
- 初始化所有输入权重为 1
- 每轮计算加权均值和经验协方差矩阵
- 找到最大特征值对应的方向（最大方差方向）
- 按各输入在该方向上的投影大小降低权重
- 记录最小最大特征值对应的加权均值
- 总权重降到 n-2f 时终止

**CAF 的关键优势**：
- 不需要对诚实输入的协方差做任何假设（与 Diakonikolas et al., 2017 不同）
- 误差界依赖最大特征值而非迹（trace），比 trimmed mean/median 好 d 倍
- 计算复杂度 O(f(nd² + d³))，实际用 power method 可降至 O(fnd log d)

**3. 动量更新**

工作者维护本地动量 m_t^(i) = β_{t-1} m_{t-1}^(i) + (1-β_{t-1}) g̃_t^(i)，发送动量而非原始梯度。动量系数 β_t = 1 - 24Lγ_t。

### 损失函数 / 训练策略

- 假设损失函数 L-smooth，可以是强凸或非凸
- 梯度裁剪 Clip(g; C) = g · min{1, C/‖g‖}，灵敏度边界 C
- 学习率策略：强凸情况 γ_t = 10/(μ(t + 240L/μ))；非凸情况 γ_t = min{1/(24L), √(3L₀)/(16σ̄√(LT))}

## 实验关键数据

### 主实验

在 MNIST 和 Fashion-MNIST 上，n=100 个工作者，f∈{5,10} 恶意工作者：

| 威胁模型 | MNIST (f=5) | MNIST (f=10) | Fashion-MNIST (f=5) |
|---------|-------------|--------------|---------------------|
| CDP（基线上界） | ~83% | ~82% | ~72% |
| CafCor-SecLDP | **83%** | ~78% | **72%** |
| CafCor-ByzLDP | ~75% | ~60% | ~71% |
| DSGD-LDP | ~30% | ~30% | ~50% |

CafCor-SecLDP 在 f=5 时接近 CDP 性能，远超 LDP。

### 消融实验

**聚合方法比较**（Figure 2，n-f=10 诚实工作者）：
- CafCor (CAF) vs CWTM, CWMED, GM, Multi-Krum, Meamed
- 在 Sign Flipping 和 Label Flipping 攻击下：
    - CAF 在所有攻击下均最优，LF 攻击下接近 90% 准确率
    - CWTM/CWMED/GM 在 LF 攻击下准确率仅约 50%
    - 将 CAF 替换为 trimmed mean 理论收敛率恶化 n 倍

### 关键发现

1. **隐私-效用权衡**：O((f+1)d/(n²ε²))，当 f=O(1) 时匹配 CDP 的 Θ(d/(n²ε²))
2. **强凸收敛率**：O(κG²/μ + Lσ̄²/(μ²T) + L²L₀/(μ²T²))
3. **非凸收敛率**：O(κG² + σ̄√(LL₀)/√T + LL₀/T)
4. f 亚线性于 n 时严格优于 LDP

## 亮点与洞察

1. **SecLDP 模型的引入**：在 LDP 和 CDP 之间找到了实用的中间地带，关联噪声方案只需一轮预通信
2. **CAF 聚合的维度优势**：误差界中没有维度 d 的额外因子，这是高维场景下相比 trimmed mean 的关键优势
3. **无需协方差先验**：CAF 仅需知道恶意工作者数量上界，不需要对诚实输入的统计性质做假设
4. **勾结分级分析**：Theorem 4.1 中 q 参数允许精细控制 0 到 f 个恶意者勾结的场景
5. **回退优雅**：当共享随机性不可用时，设置 σ_cor=0 自然回退到 LDP，保持 SOTA 性能

## 局限与展望

1. **通信代价**：种子交换需要 O(n²) 次两两通信，虽然仅需一次且数据量小，但对大规模系统仍可能是瓶颈
2. **SecLDP 最优性未知**：文中提出"SecLDP 下的最优效用是什么？CafCor 是否达到了？"作为开放问题
3. **隐私预算分配**：σ_ind 和 σ_cor 的设置依赖对勾结状态的假设，实际中可能难以确定
4. **数据集简单**：MNIST/Fashion-MNIST 上的实验，未在更复杂的数据集（如 CIFAR-10 或自然语言任务）上验证
5. **CAF 的 2f 次迭代**：虽然多项式时间，但实际中 f 较大时仍有开销

## 相关工作与启发

- **LDP+鲁棒性三方权衡** [Allouah et al., 2023c]：证明了 LDP 下隐私+鲁棒性+效用存在根本性权衡，本文突破了这一限制
- **中心化鲁棒均值估计** [Hopkins et al., 2022; Liu et al., 2021]：在 CDP 下实现了近最优误差界
- **关联噪声方案** [Sabater et al., 2022; Allouah et al., 2024]：先前工作未提供对抗性设定下的效用保证
- **SMEA 聚合** [Allouah et al., 2023c]：满足同样的高维鲁棒准则但计算复杂度指数增长，CAF 是其多项式时间替代
- 对联邦学习中同时考虑隐私和鲁棒性的研究提供了新范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 关联噪声+鲁棒聚合的组合独特，SecLDP 模型有开创性
- 实验充分度: ⭐⭐⭐⭐ — 多攻击类型、多威胁模型、多聚合方法对比
- 写作质量: ⭐⭐⭐⭐ — 理论严谨，实验清晰
- 价值: ⭐⭐⭐⭐⭐ — 解决了联邦学习中隐私+鲁棒性的核心痛点

<!-- RELATED:START -->

## 相关论文

- [Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [Generalization in Federated Learning: A Conditional Mutual Information Framework](generalization_in_federated_learning_a_conditional_mutual_information_framework.md)
- [Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)
- [FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/ai_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](../../ICLR2026/ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)

<!-- RELATED:END -->
