---
title: >-
  [论文解读] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees
description: >-
  [NeurIPS 2025][优化][神经组合优化] 揭示了现有NCO方法严重过拟合固定约束紧度（如CVRP的固定车辆容量C=50），提出变约束紧度训练方案和多专家模块(MEM)，使模型能有效处理从极紧到极松的全范围约束。
tags:
  - NeurIPS 2025
  - 优化
  - 神经组合优化
  - 车辆路径问题
  - 约束紧度
  - 多专家模块
  - 泛化能力
---

# Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees

**会议**: NeurIPS 2025  
**arXiv**: [2505.24627](https://arxiv.org/abs/2505.24627)  
**代码**: [GitHub](https://github.com/CIAM-Group/Rethinking_Constraint_Tightness)  
**领域**: 优化  
**关键词**: 神经组合优化, 车辆路径问题, 约束紧度, 多专家模块, 泛化能力

## 一句话总结

揭示了现有NCO方法严重过拟合固定约束紧度（如CVRP的固定车辆容量C=50），提出变约束紧度训练方案和多专家模块(MEM)，使模型能有效处理从极紧到极松的全范围约束。

## 研究背景与动机

NCO方法在求解VRP时通常使用固定约束值训练（如CVRP100固定容量C=50、平均需求5），在同分布测试数据上表现优异。然而，实际应用场景中约束紧度高度多变。

**核心痛点**：当约束紧度偏离训练设定时，现有NCO模型性能急剧恶化。POMO在C=50时gap仅3.66%，但在C=10（紧约束）时暴涨至20.26%（5.5×），C=500（松约束）时达34.12%（9.3×）。所有主流NCO方法（AM、POMO、BQ、LEHD等）均存在此问题，平均gap至少扩大1.7倍。

**问题根源分析**：约束紧度的变化会根本性地改变最优解的结构——
- C极小（如C=10）时，每辆车只能服务2-3个客户，最优解结构接近开放式VRP（OVRP）
- C极大（如C=500）时，一辆车可服务全部客户，最优解结构接近TSP

这意味着模型需要学习截然不同的求解策略。训练在C=50上的模型既不会OVRP的短路线策略，也不会TSP的长回路策略。

**核心idea**：通过在训练中均匀采样不同约束紧度，让模型学习跨约束的通用策略；同时用多专家模块让不同专家层专注于不同紧度范围，避免相互冲突的优化方向。

## 方法详解

### 整体框架

基于LEHD（轻量编码器-重解码器）骨干网络，在训练协议和神经架构两个层面进行改进：(1) 变约束紧度训练(VCT)；(2) 多专家模块(MEM)。

### 关键设计

1. **变约束紧度训练(VCT)**：训练时对每个实例独立地从 $[C_{min}, C_{max}]$ 均匀采样约束紧度。例如容量从 $\{10, 11, \ldots, 500\}$ 中随机选取。关键决策是**实例级**采样而非批次级采样——即同一batch内不同实例可以有不同容量。这种做法相比批次级一致采样表现更好，因为它提供了更丰富的梯度信号。

   训练数据由HGS求解器提供标注解，每个实例有随机容量和对应的最优解。使用监督学习训练LEHD模型。

2. **多专家模块(MEM)**：在LEHD解码器的 $L$ 层堆叠注意力层之后，添加多专家模块。它由门控机制 $G(\cdot)$ 和 $m$ 个专家层 $\{E_1, E_2, \ldots, E_m\}$ 组成。门控机制根据约束紧度 $C_k$ 将实例硬路由到对应的专家层：

$$P_t = \text{MEM}(H_t, C_k) = \sum_{i=1}^{m} G(C_k) E_i(H_t)$$

门控为one-hot向量，第 $i$ 个专家处理紧度在 $[(i-1)\beta, i\beta]$ 范围的实例，其中 $\beta = |C_{max} - C_{min}| / m$。每个专家层内部包含 $m_e$ 层堆叠注意力层和最终的概率预测层。这种硬路由设计简单高效，避免了软路由中不同紧度间的梯度干扰。

3. **约束紧度与问题相似性的理论联系**：用实验验证了CVRP在不同约束下等价于不同问题的假说。专门用于OVRP的POMO模型在C=10时（gap 1.9%）远优于所有CVRP模型（最优17.44%）；专门用于TSP的POMO模型在C=500时（gap 2.5%）也优于CVRP模型（最优6.8%）。这为VCT的设计提供了问题结构层面的理论支撑。

### 损失函数 / 训练策略

使用监督学习，Adam优化器（初始学习率1e-4，每epoch衰减率0.9），训练40个epoch。100万CVRP100训练实例，batch size=512。每个实例的容量从 $\{10, 11, \ldots, 500\}$ 随机采样，标注解由HGS求解器生成。

模型配置：embedding维度192，FFN隐层512，query维度16，12头注意力，$L=6$ 层解码器attention，$m=3$ 个专家，每个专家 $m_e=3$ 层attention。

## 实验关键数据

### 主实验（CVRP100不同容量）

| 方法 | C=10 | C=50 | C=100 | C=200 | C=300 | C=400 | C=500 | 平均Gap |
|------|------|------|-------|-------|-------|-------|-------|--------|
| POMO | 20.26% | 3.66% | 11.17% | 22.45% | 29.56% | 30.80% | 34.12% | 21.72% |
| BQ | 18.02% | 3.25% | 3.48% | 4.53% | 6.53% | 6.54% | 8.16% | 7.22% |
| LEHD | 36.56% | 4.22% | 4.87% | 4.18% | 5.78% | 4.92% | 6.82% | 9.62% |
| 各数据集现有最优 | 7.47% | 3.25% | 3.48% | 4.18% | 5.78% | 4.92% | 6.82% | 5.13% |
| **Ours** | **1.54%** | 3.82% | 3.67% | **1.49%** | **0.87%** | **0.75%** | **0.87%** | **1.86%** |

### 消融实验

| 配置 | C=10 | C=50 | C=500 | 平均Gap |
|------|------|------|-------|--------|
| Base Model (C=50训练) | 45.64% | 3.72% | 7.29% | 10.78% |
| Base Model + MEM | 37.03% | 3.28% | 6.11% | 8.95% |
| Base Model + VCT | 1.88% | 4.51% | 1.25% | 2.41% |
| Base Model + VCT + MEM | **1.54%** | **3.82%** | **0.87%** | **1.86%** |

### CVRPTW扩展实验

| 方法 | α=0.2(紧) | α=1.0(松) | α=3.0(极松) | 平均Gap |
|------|----------|----------|------------|--------|
| POMO | 14.95% | 11.39% | 33.56% | 19.58% |
| POMO+VCT | 9.80% | 12.37% | 11.23% | 12.14% |
| POMO+VCT+MEM | **9.13%** | 11.91% | **10.37%** | **11.58%** |

### 关键发现

- **VCT是主要贡献因素**：仅加VCT就将平均gap从10.78%降至2.41%（减少78%），MEM进一步降至1.86%
- **MEM单独使用效果有限**（8.95% vs 10.78%），但与VCT配合时一致提升所有约束范围
- 在C=300-500范围，本方法的gap低于1%，几乎达到HGS求解器水平
- **方法的通用性**：同样的思路成功应用于POMO + CVRPTW，平均gap从19.58%降至11.58%
- 紧约束过拟合比松约束更严重——AM在C=10的gap高达45.16%

## 亮点与洞察

- **问题诊断的深刻性**：不止发现了过拟合现象，还通过OVRP和TSP的对照实验解释了问题根源——不同约束紧度本质上对应不同的优化问题
- **方法极度简单高效**：VCT只是改变训练数据采样策略（零额外计算），MEM只在解码器尾部加几层attention（极少参数增量），但效果显著
- **硬路由的设计选择**：相比软混合专家的计算开销和训练不稳定性，硬路由的one-hot门控简单直接且足够有效
- **在C=50/100上的轻微退步**说明VCT用更多策略覆盖度换取了特定点位的精度，这是合理的权衡

## 局限与展望

- VCT使用均匀采样策略，自适应采样（如根据各紧度范围的当前性能调整采样频率）可能进一步提升
- 多专家的硬路由可能导致专家间的知识隔离，软路由或共享底层表示的方案值得探索
- 在C=50/100上的性能略低于专用基线，说明该方法更适合"一个模型覆盖所有"的场景
- 仅展示了CVRP和CVRPTW两种问题，对调度、装箱等其他带约束的组合优化问题适用性有待验证

## 相关工作与启发

- 与MVMoE（多样VRP的混合专家）相比，本文的MEM更简单（硬路由 vs 可学习门控）且更专注于约束紧度维度
- 与NCO泛化研究（如INViT）互补：INViT关注问题规模泛化，本文关注约束值泛化
- 启发：NCO社区应更关注训练数据的分布覆盖性，固定约束设定可能是一个普遍的隐蔽陷阱

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题诊断有深度，方法设计虽简单但有效
- 实验充分度: ⭐⭐⭐⭐⭐ 极为全面的对比（9种NCO方法×7种容量），消融清晰，CVRPTW扩展验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设计与分析逻辑严密
- 价值: ⭐⭐⭐⭐ 揭示了NCO训练中一个被忽视的系统性问题，解决方案简单实用

<!-- RELATED:START -->

## 相关论文

- [Learning to Insert for Constructive Neural Vehicle Routing Solver](learning_to_insert_for_constructive_neural_vehicle_routing_solver.md)
- [Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](../../AAAI2026/optimization/bridging_synthetic_and_real_routing_problems_via_llm-guided_instance_generation_.md)
- [RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization](../../ICLR2026/optimization/rrnco_towards_real-world_routing_with_neural_combinatorial_optimization.md)
- [Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)

<!-- RELATED:END -->
