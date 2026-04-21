---
title: >-
  [论文解读] Clip-and-Verify: 线性约束驱动的域裁剪加速神经网络验证
description: >-
  [NeurIPS 2025][LLM推理][神经网络验证] 提出Clip-and-Verify验证流水线，利用线性界传播过程中"免费"产生的线性约束，通过完全裁剪（坐标上升对偶求解）和松弛裁剪（闭式输入域收缩）两种GPU高效算法收紧全网络中间层界，在多个benchmark上减少高达96%的BaB子问题数量，是VNN-COMP 2025获胜验证器的核心组件。
tags:
  - NeurIPS 2025
  - LLM推理
  - 神经网络验证
  - branch-and-bound
  - 线性约束
  - 域裁剪
  - α
  - β-CROWN
---

# Clip-and-Verify: 线性约束驱动的域裁剪加速神经网络验证

**会议**: NeurIPS 2025  
**arXiv**: [2512.11087](https://arxiv.org/abs/2512.11087)  
**代码**: https://github.com/Verified-Intelligence/Clip_and_Verify (有)  
**领域**: AI安全 / 神经网络验证  
**关键词**: 神经网络验证, branch-and-bound, 线性约束, 域裁剪, α,β-CROWN

## 一句话总结

提出Clip-and-Verify验证流水线，利用线性界传播过程中"免费"产生的线性约束，通过完全裁剪（坐标上升对偶求解）和松弛裁剪（闭式输入域收缩）两种GPU高效算法收紧全网络中间层界，在多个benchmark上减少高达96%的BaB子问题数量，是VNN-COMP 2025获胜验证器的核心组件。

## 研究背景与动机

**领域现状**：神经网络验证旨在形式化证明网络在指定输入域上满足安全性质（如鲁棒性）。目前最有效的方法是分支定界（Branch-and-Bound, BaB）结合线性界传播（如CROWN/α,β-CROWN）——通过递归松弛非线性激活函数计算每层神经元的上下界，当最终层下界>0时性质被验证，否则继续分裂子问题。

**现有痛点**：BaB中子问题数量指数增长，而松弛的中间层界是导致无法及早验证子域的根本原因。现有方法的困境在于：(1) β-CROWN理论上支持优化中间层界，但实践中因隐藏层神经元数量远超输出层（差数个数量级），逐层优化开销不可承受；(2) LP求解器（如Gurobi）能获得精确界但太慢且不能GPU并行；(3) BaB在每次分裂时引入线性约束（激活分裂/输入分裂），这些约束蕴含了输入域缩减的信息，但现有框架未充分利用。

**核心矛盾**：更紧的中间层界→更少的BaB子问题→更快的验证，但直接优化中间层界的计算成本与收益不成正比。需要找到一种高效方式，以极低开销收紧中间层界。

**本文目标** 如何在BaB过程中以最小计算开销机会性地收紧全网络各层的界，且不依赖昂贵的外部求解器？

**切入角度**：线性界传播在各层产生的bounding planes本身就是关于输入的线性约束，加上BaB分裂引入的约束，都可以用来缩减输入域或直接改进中间层界。关键insight是将约束界优化问题转化为一维对偶问题，利用分段线性结构和断点排序实现GPU并行的高效精确求解。

**核心 idea**：将BaB过程中"免费"得到的线性约束转化为输入域缩减和中间层界收紧，通过对偶问题的闭式/坐标上升求解实现GPU并行的高效裁剪。

## 方法详解

### 整体框架

Clip-and-Verify是一个验证流水线，嵌入到标准BaB循环中。在每个BaB节点：(1) 收集线性约束（来自最终层bounding planes或激活分裂）；(2) 用松弛裁剪收缩输入域 $\mathcal{X} \rightarrow \mathcal{X}'$，然后在更小的输入域上重新concretize中间层界；(3) 对关键不稳定神经元用完全裁剪直接优化其界。两种裁剪互补——松弛裁剪便宜、全局生效，完全裁剪精确、局部作用。

### 关键设计

1. **完全裁剪（Complete Clipping）——坐标上升对偶求解**:

    - 功能：在线性约束下直接精确优化每个神经元的界
    - 核心思路：给定线性函数 $\mathbf{a}^\top \mathbf{x} + c$ 在盒域 $\mathcal{X}$ 上的最小化问题加上线性约束 $\mathbf{g}^\top \mathbf{x} + h \leq 0$，构造拉格朗日对偶问题。关键定理（Theorem 3.1）：对偶目标 $D(\beta) = (\mathbf{a} + \beta\mathbf{g})^\top \hat{\mathbf{x}} - \sum_j |(\mathbf{a} + \beta\mathbf{g})_j| \epsilon_j + c + \beta h$ 是关于 $\beta \in \mathbb{R}_+$ 的凹分段线性函数，最优解可通过枚举断点 $\beta_j = -a_j/g_j$ 精确找到。多约束时用坐标上升：依次固定其他 $\beta$、优化一个，每步都是单约束问题。这等价于连续背包问题的贪心算法，复杂度为 $\mathcal{O}(n \log n)$
    - 设计动机：将高维LP转化为一维凹最大化，利用分段线性结构避免迭代优化，天然适合GPU并行。与Gurobi LP求解器相比快880×（0.0028s vs 2.47s），界精度相当（误差0.00085 vs 0.0007）

2. **松弛裁剪（Relaxed Clipping）——闭式输入域收缩**:

    - 功能：以极低开销收缩输入域，间接收紧全网络中间层界
    - 核心思路：对于单个线性约束 $\mathbf{a}^\top \mathbf{x} + c \leq 0$ 和盒域 $\mathcal{X}$，Theorem 3.2给出每个坐标的闭式更新：$x_i^{(clip)} = (-\sum_{j \neq i} \{a_j \hat{x}_j - |a_j|\epsilon_j\} - c) / a_i$，根据 $a_i$ 的正负更新上界或下界。裁剪后得到的盒是包含可行集的最紧轴对齐超矩形。多约束时对每个约束并行独立应用，最后取各约束结果中最紧的界。计算仅需 $O(n)$ 算术运算。获得更紧的输入域后，所有中间层界通过重新concretize（而非重新传播）获得更新
    - 设计动机：完全裁剪虽然精确但需要逐神经元求解，大网络中耗时。松弛裁剪只需更新一次共享的输入域，所有中间层界自动受益。例如，在含4096+2048+100神经元的3层网络中，重新传播需10s/子域，而重新concretize仅需0.3s

3. **与BaB的集成——约束来源和流水线设计**:

    - 功能：在输入BaB和激活BaB两种模式下无缝获取约束并应用裁剪
    - 核心思路：(a) 输入BaB：最终层bounding plane $\underline{\mathbf{a}}^{(L)\top}\mathbf{x} + \underline{c}^{(L)} \leq 0$ 天然分离已验证和待验证区域，每次分裂后子域继承父域的约束实现更精确的裁剪。修改BaB流程使松弛裁剪发生在分裂之后而非之前。(b) 激活BaB：激活分裂 $z_j^{(i)} \geq s$ 或 $z_j^{(i)} \leq s$ 通过界传播松弛为输入空间中的线性约束，且可缓存重用。选择BaBSR分数最高的top-k不稳定神经元做完全裁剪。(c) 不可行性检测：若裁剪后某维度 $\underline{x} > \overline{x}$，说明当前约束组合下不可行，该子问题直接验证通过
    - 设计动机：约束在BaB过程中自然产生，是"免费"的。输入BaB中子域继承父域约束使裁剪效果逐步积累；激活BaB中大量分裂约束被收集和利用

## 实验关键数据

### 主实验

输入BaB benchmark对比：

| 方法 | acasxu子问题数 | 时间(s) | #verified | nn4sys子问题数 | #verified |
|------|-------------|---------|-----------|-------------|-----------|
| α,β-CROWN | 7,154,387 (基准) | 280.51 | 138 | 4,440,252 (基准) | 194 |
| +松弛裁剪 | 3,124,100 ↓56.3% | 151.37 | 139 | 2,691,750 ↓39.4% | 194 |
| +松弛+重排序 | 2,557,715 ↓64.2% | 150.25 | 139 | 2,300,894 ↓48.2% | 194 |
| +完全裁剪 | 1,533,068 ↓**78.6%** | 168.57 | **139** | 2,141,288 ↓51.8% | 194 |

激活BaB benchmark对比（verified数量）：

| 数据集 | β-CROWN | BICCOS | C&V+β-CROWN | C&V+BICCOS | 上界 |
|--------|---------|--------|------------|------------|------|
| oval22 | 20 | 25 | 22 | **27** | 29 |
| cifar10-resnet | 60 | 63 | 63 | **64** | 72 |
| cifar100-2024 | 119 | 121 | 126 | **131** | 168 |
| tinyimagenet-2024 | 135 | 138 | 140 | **144** | 157 |

### 消融实验

控制系统验证（完全裁剪的必要性）：

| 任务 | α,β-CROWN时间 | C&V(松弛)时间 | C&V(完全)时间 | 子问题减少 |
|------|-------------|------------|------------|----------|
| cartpole | 1602s | 484s | **142s** | ↓95.5% |
| Quadrotor-2D | timeout | 209,504s | **78,818s** | ↓57.7% |
| Quad-2D-Large | timeout | timeout | **104,614s** | 唯一能完成 |

### 关键发现

- 完全裁剪在子问题减少上效果最显著（最高96%），但因per-neuron求解开销，总时间不一定最优——松弛裁剪有时总时间更快
- 控制系统验证中完全裁剪是唯一能验证最困难性质的方法
- 与Gurobi LP对比：坐标上升GPU求解快880×，界精度相当
- 松弛裁剪与完全裁剪互补：松弛裁剪便宜且全局生效应始终开启；完全裁剪对选定的关键神经元做精确收紧
- Clip-and-Verify与BICCOS正交且可叠加，C&V+BICCOS在几乎所有benchmark上达到最佳结果

## 亮点与洞察

- **对偶问题转化的数学优雅性**：将n维LP转为1D分段线性凹最大化，利用断点结构精确求解且天然GPU并行，比通用LP求解器快近3个数量级。这一技巧的本质是连续背包问题的对偶
- **"免费午餐"的工程洞察**：线性界传播已经计算了各层的bounding planes，Clip-and-Verify只是巧妙地复用这些已有信息作为裁剪的线性约束，几乎不增加额外计算
- **不可行性检测作为副产品**：裁剪过程中自然检测出不可行的子域（矛盾的激活赋值），这些子域可以直接标记为已验证，进一步减少BaB搜索空间
- **实际影响力**：作为VNN-COMP 2025获胜验证器α,β-CROWN的核心组件，已被集成到实际系统中

## 局限与展望

- 完全裁剪对大型网络中的所有不稳定神经元仍然太贵，需要启发式选择top-k神经元。BaBSR启发式虽然有效但不是最优选择策略
- 坐标上升对多约束问题不保证全局最优，只是因为对偶目标凹性保证了每步改进。收敛性和迭代次数选择需要更多分析
- 松弛裁剪的效果受盒域近似质量限制——当可行集远非盒形时效果有限
- 当前仅在前馈网络和Vision Transformer上验证，更复杂架构（如循环网络、GNN）的适用性未探索
- 缓存和重用约束的内存开销在深网络中可能不可忽视

## 相关工作与启发

- **vs GCP-CROWN（MIP cuts）**: GCP-CROWN使用MIP cutting planes来加强松弛，但需要外部MILP求解器且开销大。Clip-and-Verify使用纯GPU坐标上升，更快更可扩展，在大多数benchmark上验证数量相当或更多
- **vs BICCOS**: BICCOS通过inter-neuron约束（二次交叉约束）加强松弛。Clip-and-Verify与BICCOS正交且可组合——C&V+BICCOS在几乎所有benchmark上达到最佳验证精度
- **vs β-CROWN中间层优化**: β-CROWN理论上可通过优化β参数收紧中间层界，但实践中因神经元过多而放弃。Clip-and-Verify提供了一种计算上可行的中间层界收紧方案

## 评分

- 新颖性: ⭐⭐⭐⭐ 对偶转化和利用已有约束的思路优雅，但核心数学工具（连续背包的对偶）并非新发明
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖VNN-COMP多个年度的benchmark、控制系统验证、与多种工具对比、详细消融
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，两种算法的定位和互补性阐述清楚
- 价值: ⭐⭐⭐⭐⭐ 作为VNN-COMP 2025获胜系统的核心组件，实际影响力已得到验证

<!-- RELATED:START -->

## 相关论文

- [MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [DisCO: Reinforcing Large Reasoning Models with Discriminative Constrained Optimization](disco_reinforcing_large_reasoning_models_with_discriminative_constrained_optimiz.md)
- [Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought](reasoning_by_superposition_a_theoretical_perspective_on_chain_of_continuous_thou.md)
- [笔记8：PolyMath - 多语言背景下的数学推理评估](self-evaluating_llms_for_multi-step_tasks_stepwise_confidence_estimation_for_fai.md)

<!-- RELATED:END -->
