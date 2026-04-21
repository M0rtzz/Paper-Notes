---
title: >-
  [论文解读] Iterative Foundation Model Fine-Tuning on Multiple Rewards
description: >-
  [NeurIPS 2025][医学图像][多目标微调] 提出 IterativeRS（迭代 Rewarded Soups），通过在多目标专家策略的独立微调和策略合并之间交替迭代，统一了奖励组合和专家合并两类方法，在小分子设计、DNA 序列生成和文本摘要任务上均优于 MORLHF 和 Rewarded Soups。
tags:
  - NeurIPS 2025
  - 医学图像
  - 多目标微调
  - 强化学习
  - 基础模型
  - 迭代合并
  - Rewarded Soups
---

# Iterative Foundation Model Fine-Tuning on Multiple Rewards

**会议**: NeurIPS 2025  
**arXiv**: [2511.00220](https://arxiv.org/abs/2511.00220)  
**代码**: [GitHub](https://github.com/pouyamghari/IterativeRS) (有)  
**领域**: 医学影像 / 多目标强化学习微调  
**关键词**: 多目标微调, 强化学习, 基础模型, 迭代合并, Rewarded Soups

## 一句话总结

提出 IterativeRS（迭代 Rewarded Soups），通过在多目标专家策略的独立微调和策略合并之间交替迭代，统一了奖励组合和专家合并两类方法，在小分子设计、DNA 序列生成和文本摘要任务上均优于 MORLHF 和 Rewarded Soups。

## 研究背景与动机

**领域现状**：基础模型的 RLHF 微调已成为标准流程，但实际应用中通常存在多个评价目标（如文本生成的有害性 vs 有用性、药物设计的多种分子属性）。

**现有痛点**：
   - **MORLHF**（奖励组合）：将多个奖励合并为单一标量，无法学到目标特定的技能，不同目标间性能方差大
   - **Rewarded Soups**（专家合并）：为每个目标独立训练专家然后线性合并，专家策略之间偏差过大时合并效果差

**核心矛盾**：目标特定学习（需要策略分叉）与策略一致性（需要策略收敛）之间的张力。

**本文目标**：设计一种灵活的多目标微调算法，既保留目标特定学习能力，又控制专家策略间的偏差。

**切入角度**：将独立微调和策略合并作为可调节的两端，通过控制合并频率 $m$ 在两者之间平滑插值。

**核心 idea**：每隔 $m$ 步将各目标的专家策略合并一次，然后从合并点继续分头训练——MORLHF ($m=1$) 和 Rewarded Soups ($m=T$) 是其两个极端特例。

## 方法详解

### 整体框架

IterativeRS 的训练流程：
1. 从预训练参考策略 $\pi_{\text{ref}}$ 初始化所有专家策略
2. 每个专家对应一个目标 $R_i$，独立做 $m$ 步梯度更新
3. 每 $m$ 步将所有专家策略参数线性合并为共享参数 $\bm{\rho}_t$
4. 用 $\bm{\rho}_t$ 重新初始化所有专家，继续下一轮
5. 最终输出末次合并后的策略

### 关键设计

1. **迭代更新规则**：
    $\bm{\theta}_{i,t+1} = \begin{cases} \bm{\theta}_{i,t} - \eta \nabla_{\bm{\theta}} \mathcal{L}_i(\pi_{\bm{\theta}_{i,t}}), & \text{if } t \bmod m \neq 0 \\ \bm{\rho}_t - \eta \nabla_{\bm{\rho}} \mathcal{L}_i(\pi_{\bm{\rho}_t}), & \text{if } t \bmod m = 0 \end{cases}$
   合并步同时随机采样 $M \leq N$ 个目标的子集来减少计算。

2. **策略合并**：
    $\bm{\rho}_t = \sum_{i \in \mathbb{S}_t} \lambda_{i,t} \bm{\theta}_{i,t}$
   权重可通过蒙特卡洛搜索优化，或简单设为 $\lambda_{i,t} = w_i / \sum_{j \in \mathbb{S}_t} w_j$。

3. **超参数 $m$ 的统一视角**：

    - $m = 1$：每步都合并 → 等同于 MORLHF（奖励组合）
    - $m = T$：训练结束才合并 → 等同于 Rewarded Soups
    - $1 < m < T$：在两个极端之间平滑插值

### 理论分析

**Theorem 1** 给出了 IterativeRS 的收敛界：

$$\mathcal{L}(\pi_{\bm{\rho}_T}) - \mathcal{L}(\pi_{\bm{\theta}^*}) \leq \frac{4L}{\mu^2(\gamma+T)}\left(3L\Delta^* + 2(2(m-1)^2 + \frac{N-M}{N-1}\frac{m^2}{M})G^2\right) + \frac{\gamma L}{2(\gamma+T)}\|\bm{\theta}_{\text{ref}} - \bm{\theta}^*\|^2$$

关键洞察：
- $\Delta^*$ 衡量各目标最优策略的差异性——目标越相似，收敛越好
- $m$ 过大增加 $A_2$ 项（专家偏差），但也可能减小 $A_1$ 项——存在最优 $m$
- 更强的参考策略（$\|\bm{\theta}_{\text{ref}} - \bm{\theta}^*\|$ 更小）有利于收敛

## 实验关键数据

### 小分子生成

| 方法 | α能量 | gap | U₀能量 | Avg Reward | ICV |
|------|-------|-----|--------|-----------|-----|
| MORLHF | 1.4229 | 0.9355 | 1.5146 | 1.2910 | 4.19 |
| RS | 1.4134 | 0.9589 | 1.5464 | 1.3062 | 4.27 |
| RiC | 0.5955 | 0.6795 | 0.7544 | 0.6765 | 3.75 |
| **IterativeRS** | **1.5893** | 0.9508 | **1.6649** | **1.4017** | 3.59 |

IterativeRS 在平均奖励上全面领先，Pareto 前沿上的最优分子质量最高。

### DNA 序列生成

| 方法 | K562 | HepG2 | SKNSH | Avg Reward | ICV |
|------|------|-------|-------|-----------|-----|
| MORLHF | 0.2724 | 0.7096 | 0.7183 | 0.5667 | 3.14 |
| RS | 0.3057 | 0.6808 | 0.7131 | 0.5666 | 3.82 |
| RiC | 0.4221 | 0.6615 | 0.6688 | 0.5842 | 2.47 |
| **IterativeRS** | 0.3032 | **0.7370** | **0.7378** | **0.5927** | **3.83** |

IterativeRS 的 ICV 比 RiC 高 35%，跨目标一致性显著更好。

### 文本摘要

| 方法 | faithful | summary | deberta | Avg Score | ICV |
|------|----------|---------|---------|-----------|-----|
| MORLHF | 0.6530 | 0.5778 | 0.3857 | 0.4525 | 4.55 |
| RS | 0.6732 | 0.5807 | 0.4296 | 0.4732 | 4.59 |
| RiC | 0.6497 | 0.5688 | 0.3455 | 0.4518 | 3.96 |
| **IterativeRS** | **0.6927** | **0.5854** | **0.4398** | **0.4849** | **4.91** |

### 消融实验

$m$ 值对性能的影响揭示了一个关键规律：
- $m=1$（MORLHF）和 $m=T$（RS）都不是最优
- 中间值 $m$ 通常最好（小分子 $m=4$，DNA $m=8$，文本 $m=40$）
- 最优 $m$ 取决于目标间的冲突程度和数据特性

### 关键发现

- RL 方法（IterativeRS/RS/MORLHF）在分子生成上显著优于 SFT 方法（RiC），因为 RL 能探索训练数据之外的高奖励区域
- 当预训练数据与微调数据分布接近时（如 DNA 任务），SFT 方法的劣势缩小
- IterativeRS 生成的极低奖励样本更少，分布更集中

## 亮点与洞察

- **统一框架**：用单一超参数 $m$ 将 MORLHF 和 Rewarded Soups 统一为特殊情况，概念优雅
- **理论保证**：在凸设置下给出了收敛性分析，虽然实际中损失非凸，但理论提供了有用的直觉
- **跨领域验证**：分子、DNA、文本三个截然不同的领域都展示了一致的优势
- **实用性强**：方法简单（改几行循环代码即可），不需要修改模型架构或训练流程

## 局限与展望

- 理论分析假设损失函数凸、梯度有界，与实际深度学习设置差距较大
- 合并方式仅考虑线性插值，更高级的合并策略（如 Task Arithmetic、TIES）可能带来提升
- 最优 $m$ 值需要实验调参，理论界给出的指导并不足以直接确定
- 文章关注的 ICV 指标在某些场景下 IterativeRS 并非最优（如小分子生成的 ICV 低于 RS）
- 扩展到更多目标（$N \gg 3$）时的计算可行性和性能有待验证

## 相关工作与启发

- Rewarded Soups (Rame et al. 2023) 是最直接的前置工作，IterativeRS 是其自然推广
- 与联邦学习中的 FedAvg 算法有异曲同工之处——本地更新多步然后聚合
- 对多目标 RLHF 的研究有实用指导意义：不需要在"组合奖励"和"训练多个专家"之间二选一
- 潜在扩展：$m$ 值可以自适应调整（根据专家策略偏差动态设定）

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一视角新颖，但核心思想（在独立训练和合并之间交替）相对直觉
- 实验充分度: ⭐⭐⭐⭐ 涵盖三个领域，但每个领域的任务设置相对简单
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论与实验结合好，部分符号可以更简洁
- 价值: ⭐⭐⭐⭐ 解决了多目标微调的实际痛点，方法简单可用

<!-- RELATED:START -->

## 相关论文

- [NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)

<!-- RELATED:END -->
