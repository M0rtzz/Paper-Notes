---
title: >-
  [论文解读] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems
description: >-
  [NeurIPS 2025][GFlowNet] 提出Hybrid-Balance GFlowNet（HBG）框架，首次在VRP场景中引入详细平衡（DB）并与轨迹平衡（TB）统一集成，配合depot引导推理策略，在CVRP和TSP上显著提升两种现有GFlowNet求解器（AGFN和GFACS）的性能。
tags:
  - NeurIPS 2025
  - GFlowNet
  - 车辆路径问题
  - 轨迹平衡
  - 详细平衡
  - 混合优化
---

# Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems

**会议**: NeurIPS 2025  
**arXiv**: [2510.04792](https://arxiv.org/abs/2510.04792)  
**代码**: [GitHub](https://github.com/ZHANG-NI/HBG)  
**领域**: 组合优化 / GFlowNet  
**关键词**: GFlowNet, 车辆路径问题, 轨迹平衡, 详细平衡, 混合优化

## 一句话总结

提出Hybrid-Balance GFlowNet（HBG）框架，首次在VRP场景中引入详细平衡（DB）并与轨迹平衡（TB）统一集成，配合depot引导推理策略，在CVRP和TSP上显著提升两种现有GFlowNet求解器（AGFN和GFACS）的性能。

## 研究背景与动机

### 领域现状

**领域现状**：车辆路径问题（VRP）是物流运输的核心优化问题，传统启发式方法依赖手工规则、适应性差

### 现有痛点

**现有痛点**：GFlowNet通过学习解空间上的分布来生成多样化高质量路径，是有前景的替代方案

### 核心矛盾

**核心矛盾**：现有GFlowNet-VRP方法（AGFN、GFACS）仅使用轨迹平衡（TB）进行全局优化，忽视了局部优化信号

### 解决思路

**解决思路**：TB的局限：当整条路径成本高时，即使其中有高质量的局部转移（如W→X→Y），也会因全局奖励差而收到弱训练信号

### 补充说明

**补充说明**：详细平衡（DB）提供步级局部信号，但单独使用DB无法满足VRP需要的全局视角

### 补充说明

**补充说明**：核心洞察：**全局和局部优化互补，需要统一框架**

## 方法详解

### 整体框架

HBG在现有GFlowNet求解器（AGFN或GFACS）的TB训练基础上，新增DB损失用于局部优化。训练时每步记录状态转移信息，生成完整轨迹后同时计算TB损失（全局）和DB损失（局部），联合优化。推理时采用depot引导策略，仅在仓库节点采样、客户节点贪心。

### 关键设计

1. **VRP专用的详细平衡（DB）**:
    - 功能：为VRP中每个状态转移计算局部优化目标
    - 核心思路：DB损失 $\ell_{DB}(s_t^i, s_{t+1}^i; \theta) = \left(\log \frac{P_f \cdot F(s_t^i) \cdot \exp(\tilde{\mathcal{E}}(s_{t+1}^i))}{P_b \cdot F(s_{t+1}^i) \cdot \exp(\tilde{\mathcal{E}}(s_t^i))}\right)^2$
    - 设计动机：让模型能从不完美的全局轨迹中识别和强化高质量局部决策

2. **后向概率推导**:
    - 功能：为CVRP推导封闭形式的TB和DB后向概率
    - 核心思路：完整轨迹由 $a$ 个多节点子轨迹和 $j$ 个单节点子轨迹组成，后向破坏顺序计数 $B(\mathcal{A}_a, \mathcal{J}_j) = (a+j)! \cdot 2^a$
    - 设计动机：确保DB后向概率与TB后向概率在理论上一致

3. **Depot引导推理**:
    - 功能：仅在仓库节点使用采样，客户节点使用贪心选择
    - 核心思路：后向策略分析显示只有仓库节点在轨迹结构中有多个前驱选择的灵活性，客户节点的后继是确定性的
    - 设计动机：将采样（探索）限制在有灵活性的仓库节点，减少无效探索

### 损失函数 / 训练策略

总损失为TB和DB损失之和：

$$\ell_{HB}(\mathcal{T}; \theta) = \sum_{i=1}^{h} (\ell_{TB}(\tau_i; \theta) + \ell_{DB}(\tau_i; \theta))$$

- DB中的能量项 $\tilde{\mathcal{E}}(s_t^i) = R(s_t^i) - \frac{1}{h}\sum_{k=1}^{h} R(s_t^k)$ 衡量相对优势
- 在100节点实例上训练，测试200/500/1000节点实例

## 实验关键数据

### 主实验（表格）

CVRP基准测试（Gap%相对于LKH）：

| 方法 | CVRP200 Gap↓ | CVRP500 Gap↓ | CVRP1000 Gap↓ |
|------|-------------|-------------|--------------|
| AGFN | 11.48 | 12.21 | 11.15 |
| **HBG-AGFN** | **9.95** | **10.44** | **9.34** |
| GFACS | 23.11 | 23.83 | 23.82 |
| **HBG-GFACS** | **16.48** | **13.53** | **10.61** |
| GFACS+LS | 2.10 | 3.03 | 3.00 |
| **HBG-GFACS+LS** | **1.96** | **2.81** | **2.75** |

### 消融实验

- **HB模块**：单独加入HB即可将AGFN Gap降低0.64–1.68个百分点
- **Depot引导推理**：在HB基础上进一步降低Gap 0.89–1.09个百分点（AGFN）
- **DB vs TB vs HB对比**：仅用DB训练效果最差（Gap 18.85–22.72%），TB合理（11.15–11.48%），HB最优
- **HB对GFACS的改进更大**：Gap降低达55.46%（CVRP1000），因GFACS原始TB优化空间更大

### 关键发现

- 改进随问题规模增大而更显著，表明HBG具有良好的可扩展性
- 推理时间开销极小（仅0.01–0.04秒），因DB相关参数仅需临时加载
- 在CVRPLib真实世界数据上同样有效
- 单用DB效果差，验证了VRP需要全局视角的论断

## 亮点与洞察

- 首次在VRP场景中严格定义和实现DB，填补了GFlowNet-VRP的理论空白
- 后向概率推导精巧，多节点子轨迹的2倍因子反映了仓库节点的双向灵活性
- Depot引导推理巧妙利用了CVRP的结构特性，将探索集中在有意义的决策点
- 框架通用性强——适用于AGFN（构造式）和GFACS（改进式）两类不同求解器

## 局限与展望

- Depot引导推理仅适用于有仓库节点的VRP变体，对TSP等无depot问题不适用
- DB的局部奖励定义（相邻节点欧氏距离）较简单，可探索更复杂的局部评价
- 与更先进的学习型求解器（如基于大语言模型的方法）未做对比
- 仅在合成数据上训练，真实物流数据上的表现需进一步验证

## 相关工作与启发

- 与DeepACO、NeuOpt等学习型VRP求解器互补，HBG提供了GFlowNet内部优化的改进
- TB/DB混合思想可推广到其他序列决策的GFlowNet应用（如分子生成、因果发现）
- 局部-全局优化的平衡是组合优化中的通用问题，HBG提供了优雅的统一视角

## 评分

- ⭐⭐⭐⭐ — 理论推导扎实，改进一致且显著，但问题相对垂直，通用性验证有限

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](../../AAAI2026/others/or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)
- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](../../ICLR2026/others/evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](../../ACL2025/others/mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](../../ACL2025/others/hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)
- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](../../ICCV2025/others/revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)

<!-- RELATED:END -->
