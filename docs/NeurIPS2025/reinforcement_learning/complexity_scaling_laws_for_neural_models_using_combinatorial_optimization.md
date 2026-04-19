---
title: >-
  [论文解读] Complexity Scaling Laws for Neural Models using Combinatorial Optimization
description: >-
  [NeurIPS 2025][神经缩放定律] 以旅行商问题（TSP）为案例，研究固定模型容量下问题复杂度（解空间大小、表示空间维度）与模型性能之间的可预测缩放规律，揭示了 RL 和 SFT 在组合优化中的系统性性能趋势。
tags:
  - NeurIPS 2025
  - 神经缩放定律
  - 组合优化
  - TSP
  - 问题复杂度
  - 强化学习
---

# Complexity Scaling Laws for Neural Models using Combinatorial Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2506.12932](https://arxiv.org/abs/2506.12932)  
**代码**: [GitHub](https://github.com/lowellw6/complexity-scaling-laws)  
**领域**: reinforcement_learning  
**关键词**: 神经缩放定律, 组合优化, TSP, 问题复杂度, 强化学习

## 一句话总结

以旅行商问题（TSP）为案例，研究固定模型容量下问题复杂度（解空间大小、表示空间维度）与模型性能之间的可预测缩放规律，揭示了 RL 和 SFT 在组合优化中的系统性性能趋势。

## 研究背景与动机

神经缩放定律已证明模型性能随计算预算、模型大小和数据集大小可预测地改善。然而，一个关键维度被忽视：**问题复杂度**如何影响固定容量模型的性能极限？

这一问题的挑战在于：
1. 在语言建模、视觉等领域，难以将问题复杂度解耦为独立可缩放的度量
2. RL 的回报信号通常不平滑，难以获得干净的缩放定律

TSP 作为经典组合优化问题，提供了独特优势：
- **解空间**（节点数 $n$）和**表示空间**（空间维度 $d$）可独立缩放
- 组合优化促进平滑的代价趋势，即使没有可解释的损失函数也能获得有意义的缩放定律

## 方法详解

### 整体框架

固定模型参数量，系统性缩放 TSP 的两个复杂度维度：

1. **解空间缩放**：固定 $d=2$，节点数 $n \in \{5, 10, 15, \ldots, 50\}$，解空间从12到 $\sim3 \times 10^{62}$
2. **表示空间缩放**：固定 $n$，空间维度 $d \in \{2, 3, \ldots, 12, 15, 20, 30, 40, 50, 100\}$

性能度量为**次优性差距**（suboptimality gap）：

$$s = \mu_{model} - \mu_{opt}$$

其中 $\mu_{model}$ 为模型平均巡回长度，$\mu_{opt}$ 为最优巡回长度。

### 关键设计

**模型架构**：基于 Transformer 的自回归指针网络（Pointer Network），逐步生成巡回序列。在每一步通过 softmax 策略分布选择下一个未访问节点。

**两种训练范式**：
- **强化学习（RL）**：回报为负巡回长度的 bandit 问题，使用 PPO 训练100万步
- **监督微调（SFT）**：最小化模型边选择与最优解边选择的负对数似然

**神经缩放定律**（参数/计算缩放）：对二维20节点 TSP，次优性随参数或计算量呈幂律衰减：

$$s \propto N^{-\alpha} \quad \text{(参数缩放)}$$
$$s \propto C^{-\alpha} \quad \text{(计算缩放)}$$

**问题复杂度缩放定律**：

- 节点缩放：次优性呈超线性幂律增长
$$s \propto (n - \gamma)^{\alpha}$$

- 维度缩放：次优性呈负指数衰减趋向渐近值
$$s \approx \beta_\psi + a \cdot \psi^{-d}$$

### 损失函数

- RL：PPO 的 clipped surrogate objective，回报信号为 $-\text{tour\_length}$
- SFT：负对数似然损失 $\mathcal{L} = -\log p(\text{optimal edges} | \text{model})$，使用 teacher forcing

## 实验关键数据

### 主实验

**参数缩放**（2D 20节点 TSP）：

| 训练范式 | 参数缩放指数 $\alpha$ | 计算缩放趋势 |
|----------|----------------------|-------------|
| RL | 幂律衰减 | 计算高效前沿呈幂律衰减 |
| SFT | 可能更快的衰减 | 更高的计算效率 |

最优模型大小 vs 计算预算的增长率指数约0.5-0.75，与语言建模领域高度一致。

**节点缩放**（固定模型容量）：

| 训练范式 | 增长指数 $\alpha$ | 趋势形态 |
|----------|------------------|---------|
| RL | ≈1.86 | 超线性幂律增长 |
| SFT | ≈1.69 | 超线性幂律增长 |

**维度缩放**（RL，固定模型容量）：

| 节点数 | 趋势形态 | 渐近值 $\beta_\psi$ |
|--------|---------|-------------------|
| 10 | 负指数衰减 | 较高 |
| 20 | 负指数衰减 | 较低（合理排序） |

### 消融实验

**2-opt 局部搜索对比**：

- 维度缩放：2-opt 的趋势与 RL 高度一致，10节点渐近值几乎对齐
- 节点缩放：无约束 2-opt 产生不规则趋势（有拐点），但限制搜索深度 $M$ 后呈现超线性幂律增长
- 亚指数衰减 $\psi^{-d^\phi}$ 对 2-opt 拟合更好（但多一个参数）

**SFT vs RL 对比**：
- SFT 在测试规模内计算效率更高（以节点级监督为代价）
- 参数效率在当前规模相近，但 SFT 可能在更大模型上超越 RL

### 关键发现

1. **TSP 原生产生平滑缩放定律**：回报信号（巡回长度）直接作为性能度量即可获得高精度的幂律关系
2. **节点缩放超线性增长必然破裂**：外推显示模型次优性将在约40,000节点处超过随机巡回，表明实际趋势将更平缓
3. **维度缩放收敛原因**：高维中随机巡回与最优巡回长度以相似速率发散，使次优性趋于常数（Theorem 6 证明）
4. **嵌入参数可分离性**：维度缩放中嵌入层参数的角色与模型容量参数本质不同，印证了已有神经缩放定律中的嵌入分离原则
5. **训练范式不敏感**：参数受限的复杂度缩放对 RL/SFT 训练方式不敏感

## 亮点与洞察

- **核心贡献**：首次系统性研究固定模型容量下的问题复杂度缩放定律，填补了神经缩放定律研究的重要空白
- **类比洞察**：固定容量模型在复杂度增加时的性能退化，类似于固定深度局部搜索——模型被迫在局部最优附近"早停"
- **实验设计精巧**：128M 最优 TSP 解的大规模数据集创建，既用于 SFT 训练又用于精确评估
- **跨范式一致性**：最优模型大小随计算预算的增长率（0.5-0.75）在 TSP 中与语言建模、博弈等领域高度一致

## 局限性

1. **单一问题局限**：所有结论仅基于欧几里得 TSP，未验证对 VRP、QAP 等其他组合优化问题的推广性
2. 欧几里得 TSP 的三角不等式和 PTAS 存在性使其比大多数 NP-hard 问题更"简单"
3. 每个规模仅训练单个种子，无法确定统计显著性
4. 计算资源有限（50节点 RL 实验需24 V100-天），无法验证更大规模的趋势
5. 缺乏理论解释：为什么得到特定的幂律指数和渐近值仍不清楚

## 相关工作与启发

- **与 Kaplan et al. 的联系**：将语言建模中的缩放定律范式从"模型/数据/计算"扩展到"问题复杂度"这一新维度
- **与 Jones (AlphaZero Hex) 的区别**：Jones 的方法无法研究固定模型容量的瓶颈（因为模型在每个棋盘大小都达到完美博弈）
- **启发方向**：研究 VRP、背包问题等更复杂组合优化中的复杂度缩放；建立预测 $\alpha$、$\beta$ 参数的理论框架；探索有限计算/数据下的复杂度缩放

## 评分

- ⭐ 创新性：5/5 — 问题复杂度缩放定律是全新且有深远意义的研究方向
- ⭐ 实用性：3/5 — 对算法基准测试和边缘部署性能预测有价值，但实际应用场景有限
- ⭐ 实验充分度：4/5 — RL/SFT/局部搜索三方对比全面，128M数据集是重要贡献
- ⭐ 写作质量：5/5 — 论文组织出色，讨论深入，局限性分析坦诚且富有洞察

<!-- RELATED:START -->

## 相关论文

- [Reinforcement Learning Teachers of Test Time Scaling](reinforcement_learning_teachers_of_test_time_scaling.md)
- [Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)
- [Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [Graph-Supported Dynamic Algorithm Configuration for Multi-Objective Combinatorial Optimization](../../ICML2025/reinforcement_learning/graph-supported_dynamic_algorithm_configuration_for_multi-objective_combinatoria.md)

<!-- RELATED:END -->
