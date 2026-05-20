---
title: >-
  [论文解读] Learning to Solve Orienteering Problem with Time Windows and Variable Profits
description: >-
  [ICLR 2026][优化/理论][定向问题] 提出DeCoST——一种学习式两阶段框架，将OPTWVP中耦合的离散路线决策和连续服务时间分配解耦：第一阶段并行解码器联合生成路径+初始服务时间，第二阶段LP精确优化服务时间(全局最优)，通过pTAR反馈实现跨阶段协调…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "定向问题"
  - "时间窗"
  - "可变利润"
  - "离散-连续解耦"
  - "线性规划"
---

# Learning to Solve Orienteering Problem with Time Windows and Variable Profits

**会议**: ICLR 2026  
**arXiv**: [2603.06260](https://arxiv.org/abs/2603.06260)  
**代码**: [GitHub](https://github.com/SwonGao/DeCoST)  
**领域**: 组合优化/车辆路径  
**关键词**: 定向问题, 时间窗, 可变利润, 离散-连续解耦, 线性规划

## 一句话总结

提出DeCoST——一种学习式两阶段框架，将OPTWVP中耦合的离散路线决策和连续服务时间分配解耦：第一阶段并行解码器联合生成路径+初始服务时间，第二阶段LP精确优化服务时间(全局最优)，通过pTAR反馈实现跨阶段协调。在50-500节点OPTWVP上优化间隙仅0.83%-3.31%，推理速度比元启发式快最高45倍。

## 研究背景与动机

**领域现状**：定向问题(OP)是VRP的重要变体，需要在固定时间预算内选择节点子集访问以最大化奖励。实际场景中——工厂调度、物流运输、机器人规划——奖励通常随服务时间增长(可变利润)，且节点仅在特定时间窗内可访问。这催生了OPTWVP：同时优化离散路线和连续服务时间分配。

**现有痛点**：

1. **离散-连续耦合**：路线变化→重塑所有节点的可行服务时间区间→反过来影响奖励和路线决策→双向依赖导致联合搜索空间指数膨胀
2. **NCO方法的局限**：现有神经组合优化(NCO)方法仅处理离散路线选择，不具备连续变量(服务时间)的分配能力
3. **分解方法的短视性**：简单分解→第一阶段路线决策无法预见最优服务时间→后续局部调整不足以纠正结构偏差
4. **元启发式的可扩展性**：手动设计启发规则+穷举局部搜索→每次路线变化需重新求解连续子问题→计算代价高

**核心矛盾**：离散路线和连续服务时间本质上相互依赖，但联合优化计算不可行，而简单解耦又导致短视决策。

**本文方案**：智能解耦+跨阶段协调。第一阶段感知连续变量(并行解码路线+时间)→第二阶段精确优化(LP全局最优)→pTAR反馈让第一阶段"预见"第二阶段后果。

## 方法详解

### 整体框架

DeCoST将OPTWVP建模为约束马尔可夫决策过程(CMDP)，动作空间包含离散(节点选择)和连续(归一化服务时间比)两部分：

$$a_i = (v^{(i)}, d_{v^{(i)}}/d_{v^{(i)}\max})$$

整体目标是最大化期望总奖励：

$$\max_{\pi_\theta} J_r(\pi_\theta) = \mathbb{E}_{(\tau,\mathbf{d})\sim\pi_\theta}[R(\tau, \mathbf{d}|\mathcal{G})]$$

其中 $R(\tau, \mathbf{d}|\mathcal{G}) = \sum_{i=1}^{l-1} p_{v^{(i)}} d_{v^{(i)}}$，受时间预算、服务时间上限和时间窗约束。

### 关键设计1：并行解码的第一阶段

第一阶段使用并行解码器结构：

- **路线解码器**：基于注意力机制，选择下一个要访问的节点
- **服务时间解码器(STD)**：同步预测对应的初始服务时间分配
- **空间编码(Spatial Encoding)**：编码边特征(节点距离)为注意力偏置，增强图结构理解
- **可行性掩码(Feasibility Masking)**：动态排除违反约束的候选节点——(i) 无法在时间限制内返回起点；(ii) 下一节点的到达时间超出其时间窗

并行结构使路线决策在生成时就考虑服务时间的影响，避免纯路线导向的短视问题。

### 关键设计2：LP全局最优的第二阶段

固定第一阶段的路线 $\tau$ 后，服务时间分配简化为线性规划：

$$\max_{\mathbf{s},\mathbf{d}} \mathbf{p}^T \mathbf{d}, \quad \text{s.t.} \quad \mathbf{s}^- \leq \mathbf{s} \leq \mathbf{s}^+, \quad \mathbf{0} \leq \mathbf{d} \leq \mathbf{d}_{\max}, \quad (I-U)\mathbf{s} + \mathbf{d} + \mathbf{t} \leq \mathbf{0}$$

提出STO算法通过贪心策略求解：按单位利润从高到低排序→依次分配最大可行服务时间→更新后续节点起始时间。**严格证明了STO算法的全局最优性**（Theorem 4.1），且支持并行计算。

### 关键设计3：pTAR跨阶段监督反馈

利润加权时间分配比(profit-weighted Time Allocation Ratio)衡量解的利润效率：

$$pTAR(\mathbf{d}) = \sum_{i \in \tau} \frac{p_i d_i}{t_i}$$

引入排斥性监督损失，防止STD过早收敛到LP条件最优(因为路线不同→条件最优不同)：

$$\mathcal{L}_{pTAR} = -(pTAR(\hat{\mathbf{d}}) - pTAR(\mathbf{d}^*))^2$$

总损失为REINFORCE损失与pTAR损失的加权和：$\mathcal{L}_{total} = \beta_1 \mathcal{L} + \beta_2 \mathcal{L}_{pTAR}$。

## 实验关键数据

### 主实验：不同规模OPTWVP性能

| 方法 | 类别 | n=50,TW100 Score(Gap) | n=100,TW100 Score(Gap) | n=50,TW500 Score(Gap) |
|------|------|----------------------|----------------------|----------------------|
| B&C (Gurobi) | 精确 | 15.2 (0.00%) | 26.1 (0.00%) | 51.9 (0.00%) |
| Greedy-PRS | 启发式 | 12.9 (15.7%) | 21.9 (16.2%) | 46.1 (11.4%) |
| ILS | 元启发式 | 14.5 (4.34%) | 24.9 (4.2%) | 47.8 (7.82%) |
| POMO | NCO | 11.3 (25.3%) | 11.5 (55.7%) | 31.3 (39.6%) |
| GFACS | NCO | 12.3 (18.6%) | 25.2 (3.38%) | 47.4 (8.57%) |
| **DeCoST** | **NCO** | **15.1 (1.06%)** | **25.6 (1.97%)** | **51.4 (0.83%)** |

DeCoST在所有设置下优化间隙最小(0.83%-3.31%)，远优于其他NCO和元启发式方法。POMO因不处理服务时间而表现最差(Gap达55.7%)。

### 大规模实验(n=500, TW=100)

| 方法 | Score | Gap | Runtime (ms) |
|------|-------|-----|-------------|
| B&C (Gurobi) | 82.3 | 0.00% | 68,400 |
| ILS | 78.2 | 4.98% | 8,803 |
| POMO | 58.6 | 28.8% | 747 |
| **DeCoST** | **79.6** | **3.31%** | **1,329** |

500节点规模下，DeCoST仍保持3.31%的优化间隙，推理速度比ILS快6.6倍，比Gurobi快51倍。

### 消融实验(n=50, TW=100)

| 配置 | SE | STO | SL | Score | Gap |
|------|----|----|-----|-------|-----|
| 基线(POMO) | ✗ | ✗ | ✗ | 11.3 | 25.3% |
| +空间编码 | ✓ | ✗ | ✗ | 11.6 | 23.0% |
| +STO | ✓ | ✓ | ✗ | 14.9 | 2.28% |
| **DeCoST(完整)** | **✓** | **✓** | **✓** | **15.1** | **1.06%** |

STO是最关键组件，将Gap从23.0%锐降至2.28%；pTAR监督进一步降至1.06%。

## 亮点与洞察

- **"解耦但协调"的设计哲学**：不是简单分开两个优化问题，而是通过pTAR反馈实现两阶段的信息互通——比联合优化更高效，比独立优化更好
- **LP全局最优性的巧妙利用**：固定路线后连续部分有精确LP解，这是OPTWVP特有结构的"幸运"——但关键在于发现并证明了这一点
- **对NCO领域的扩展意义**：大多数NCO只处理纯离散问题，DeCoST首次系统解决离散-连续混合优化的NCO，填补了重要空白
- **STO可作为通用插件**：DeCoST的第二阶段可与不同的第一阶段构造式求解器组合，一致提升各种基线的解质量

## 局限性

- LP全局最优性依赖于线性利润函数假设(非线性利润需要非线性规划)
- 大规模实例(n=500)的Gap仍有3.31%，存在改进空间
- pTAR的权衡参数 $\beta_1, \beta_2$ 需要手动调节

## 评分

- 新颖性: ⭐⭐⭐⭐ 两阶段解耦+LP最优+pTAR跨阶段反馈
- 实验充分度: ⭐⭐⭐⭐ 多规模+多时间窗+vs精确/启发式/NCO+消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数学严谨
- 价值: ⭐⭐⭐⭐ 对带连续变量的组合优化有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](../../ICML2026/optimization/on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](../../ICML2026/optimization/transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)

</div>

<!-- RELATED:END -->
