---
title: >-
  [论文解读] Minimizing Inequity in Facility Location Games
description: >-
  [AAAI 2026][AI安全][设施选址] 研究实数轴上设施选址博弈中最小化组间最大加权效果（Maximum Group Effect）的问题，提出 BALANCED 和 MAJOR-PHANTOM 两种策略防护机制，在单设施场景下实现紧近似比，统一了功利主义（社会成本）、平等主义（最大成本）等经典目标和组公平目标，并将 endpoint 机制扩展到双设施场景。
tags:
  - "AAAI 2026"
  - "AI安全"
  - "设施选址"
  - "组间公平"
  - "策略防护"
  - "近似比"
  - "社会选择"
---

# Minimizing Inequity in Facility Location Games

**会议**: AAAI 2026  
**arXiv**: [2602.01048](https://arxiv.org/abs/2602.01048)  
**代码**: 无  
**领域**: AI Safety / 算法博弈论  
**关键词**: 设施选址, 组间公平, 策略防护, 近似比, 社会选择

## 一句话总结
研究实数轴上设施选址博弈中最小化组间最大加权效果（Maximum Group Effect）的问题，提出 BALANCED 和 MAJOR-PHANTOM 两种策略防护机制，在单设施场景下实现紧近似比，统一了功利主义（社会成本）、平等主义（最大成本）等经典目标和组公平目标，并将 endpoint 机制扩展到双设施场景。

## 研究背景与动机

**领域现状**：设施选址博弈是社会选择理论的经典问题——在一条直线上选择设施位置以服务分布在不同位置的多个代理人（agent）。代理人可能策略性地谎报自己的位置以获利。经典策略防护机制（如中位数机制）优化功利主义目标（最小化总距离）或平等主义目标（最小化最大距离）。

**现有痛点**：传统目标函数忽视了组间公平性。当代理人属于不同群体（如不同社区、不同收入阶层）时，总成本最优解可能导致某些群体承受不成比例的高成本。Marsh 和 Schilling (1994) 提出的 maximum group effect 框架虽然提供了公平目标函数，但实现策略防护的近似算法存在已知 gap：Zhou, Li, and Chan (2022) 指出了组公平目标下的近似界 gap。

**核心矛盾**：同时满足策略防护（truthfulness）和组间公平最小化（minimize maximum group effect）的机制设计存在根本性困难——策略防护限制了可选方案空间，而公平目标要求精细平衡各组利益。

**本文目标**：设计满足策略防护的机制来最小化 maximum group effect，并关闭已有的近似界 gap。

**切入角度**：将多种经典设施选址目标（功利、平等、组公平）统一到 maximum group effect 框架下，用统一的分析框架处理。

**核心 idea**：提出 BALANCED（基于加权中位数）和 MAJOR-PHANTOM（基于虚拟投票者）两种新机制，分别在 total group effect 和 maximum group effect 的两种形式下实现紧近似比。

## 方法详解

### 整体框架
在实数轴 $\mathbb{R}$ 上，$n$ 个代理人分属 $m$ 个群组。每个群组 $g$ 有权重因子 $w_g$。群组效果定义为其成员到最近设施距离的总和（或最大值）乘以权重。目标是选择设施位置 $y$ 最小化 maximum group effect $\max_g w_g \cdot \text{effect}_g(y)$，同时机制必须策略防护（任何代理人谎报位置都不能获益）。

### 关键设计

1. **BALANCED 机制（单设施，total effect）**:

    - 功能：最小化各组加权总距离的最大值
    - 核心思路：计算每个群组的加权中位数位置，然后在这些中位数之间选择平衡点——使得所有群组的加权总效果尽可能均等。具体地，BALANCED 输出使得左侧群组效果和右侧群组效果平衡的位置
    - 设计动机：朴素的全局中位数忽视了组间差异，而 BALANCED 通过在组中位数之间寻找平衡点，显式考虑了组间公平性

2. **MAJOR-PHANTOM 机制（单设施，max effect）**:

    - 功能：最小化各组中最大加权个人距离的最大值
    - 核心思路：受虚拟投票者（phantom voter）框架启发，为每个群组构造虚拟代理人——虚拟位置由群组权重和成员分布决定。然后在包含真实和虚拟代理人的扩展集合上运行中位数机制。虚拟投票者的位置设计保证了策略防护性
    - 设计动机：标准 phantom mechanism 不考虑组结构；MAJOR-PHANTOM 通过对每组设置不同的虚拟投票者实现组间感知

3. **扩展 Endpoint 机制（双设施）**:

    - 功能：在需要放置两个设施时最小化 maximum group effect
    - 核心思路：将经典的 endpoint 机制（选择代理人分布的两个端点作为设施位置）扩展到组公平设置。通过精心设计的组感知端点选择策略，在两种 maximum group effect 目标下实现紧近似比
    - 设计动机：多设施场景下策略防护更难保证，endpoint 机制是少数已知的双设施策略防护机制之一

### 理论分析
所有机制的策略防护性通过标准博弈论论证证明（单峰偏好 + 固定规则 → 策略防护）。近似比通过最坏情况分析得到，并构造匹配下界证明紧性。

## 实验关键数据

### 主要理论结果：近似比

| 设置 | 目标函数 | 机制 | 近似比 | 下界 | 状态 |
|---|---|---|---|---|---|
| 单设施 | max total group effect | BALANCED | $1 + \frac{w_{\max}}{w_{\min}}$ | $1 + \frac{w_{\max}}{w_{\min}}$ | 紧 |
| 单设施 | max-max group effect | MAJOR-PHANTOM | $1 + 2\frac{w_{\max}}{w_{\min}}$ | $1 + 2\frac{w_{\max}}{w_{\min}}$ | 紧 |
| 双设施 | max total group effect | Extended Endpoint | $n-1$ | $n-1$ | 紧 |
| 双设施 | max-max group effect | Extended Endpoint | $\frac{n}{2}$ | $\frac{n}{2}$ | 紧 |

### 与经典目标的统一关系

| 经典目标 | 特殊情况 | 对应权重设置 |
|---|---|---|
| 功利主义（社会成本） | $m=1$ 群组 | $w_1 = 1$ |
| 平等主义（最大成本） | $m=n$（每人一组） | $w_g = 1, \forall g$ |
| 最大总组成本 | 等权 | $w_g = 1/|g|, \forall g$ |
| 最大平均组成本 | 等权归一化 | $w_g = 1, \forall g$ |

### 关闭的近似界 Gap

| 目标 | Zhou et al. (2022) 上界 | Zhou et al. (2022) 下界 | 本文 |
|---|---|---|---|
| max total group cost | 3 | 2 | **2（紧）** |
| max avg group cost | $1+n/2$ | 未知 | **$1+n/2$（紧）** |

### 关键发现
- **完整关闭了 Zhou et al. (2022) 的 open gap**：max total group cost 的近似比从 [2, 3] 精确确定为 2
- **统一框架**：将功利主义、平等主义、组公平等看似不同的目标统一到 maximum group effect 公式下，不同权重对应不同经典目标
- **紧近似比**：所有已提出机制的近似比均有匹配的下界构造，说明在策略防护约束下无法进一步改进
- **BALANCED 和 MAJOR-PHANTOM 统一了经典机制**：在特定权重下退化为标准中位数/phantom 机制

## 亮点与洞察
- **优雅的统一框架**：一个参数化目标函数 max group effect 统一了社会选择理论中多个经典目标，不同权重设置恢复不同目标。这种统一视角为后续研究提供了清晰的分析框架
- **精确的近似界**：所有结果都是紧的（上下界匹配），这在机制设计领域很有价值——它不仅给出"能做到什么"，还精确说明了"不可能做到更好"
- **实践意义**：城市规划、公共设施选址等领域可直接应用——当不同社区有不同的服务需求权重时，BALANCED 机制保证了公平性

## 局限与展望
- 仅限于实数轴（一维）设置，高维设施选址更贴合实际但策略防护更难
- 双设施场景的近似比为 $O(n)$，在代理人数量多时可能不够理想
- 假设群组权重 $w_g$ 是公共知识且不可操纵，如果权重也可被策略性报告则问题完全不同
- 只考虑确定性机制，随机机制可能实现更好的近似比

## 相关工作与启发
- **vs Zhou, Li, Chan (2022)**：提出了组公平设施选址问题但留下了近似界 gap；本文完整关闭了这些 gap
- **vs Procaccia & Tennenholtz (2013)**：经典单设施机制设计框架，本文将其推广到组公平设置
- **vs Moulin (1980)**：标准 phantom mechanism 不考虑组结构；MAJOR-PHANTOM 通过组感知虚拟投票者扩展了这一经典框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一框架视角新颖，two new mechanisms贡献扎实
- 实验充分度: ⭐⭐⭐⭐ 所有理论结果都有紧下界，理论完备
- 写作质量: ⭐⭐⭐⭐ 数学严谨，但可读性对非博弈论读者较低
- 价值: ⭐⭐⭐⭐ 关闭open problem，统一经典结果，理论贡献显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning-Augmented Facility Location Mechanisms for Envy Ratio](../../NeurIPS2025/ai_safety/learning-augmented_facility_location_mechanisms_for_envy_ratio.md)
- [\[AAAI 2026\] Revisiting (Un)Fairness in Recourse by Minimizing Worst-Case Social Burden](revisiting_unfairness_in_recourse_by_minimizing_worst-case_social_burden.md)
- [\[ICML 2025\] Convex Markov Games: A New Frontier for Multi-Agent Reinforcement Learning](../../ICML2025/ai_safety/convex_markov_games_a_new_frontier_for_multi-agent_reinforcement_learning.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[AAAI 2026\] Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data](privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)

</div>

<!-- RELATED:END -->
