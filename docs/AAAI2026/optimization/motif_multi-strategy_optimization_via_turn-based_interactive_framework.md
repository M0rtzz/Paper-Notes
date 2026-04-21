---
title: >-
  [论文解读] MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework
description: >-
  [AAAI 2026][优化][自动启发式设计] 提出 MOTIF 框架，将求解器设计建模为多策略优化问题，通过基于蒙特卡洛树搜索 (MCTS) 的双 LLM 代理回合制竞争机制，联合优化组合优化求解器中的多个相互依赖的算法组件，在 TSP、CVRP、BPP 等多个组合优化领域中一致超越现有方法。
tags:
  - AAAI 2026
  - 优化
  - 自动启发式设计
  - LLM
  - 组合优化
  - 蒙特卡洛树搜索
  - 多策略优化
---

# MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework

**会议**: AAAI 2026  
**arXiv**: [2508.03929](https://arxiv.org/abs/2508.03929)  
**代码**: [https://github.com/HaiAu2501/MOTIF](https://github.com/HaiAu2501/MOTIF)  
**领域**: 优化  
**关键词**: 自动启发式设计, LLM, 组合优化, 蒙特卡洛树搜索, 多策略优化

## 一句话总结

提出 MOTIF 框架，将求解器设计建模为多策略优化问题，通过基于蒙特卡洛树搜索 (MCTS) 的双 LLM 代理回合制竞争机制，联合优化组合优化求解器中的多个相互依赖的算法组件，在 TSP、CVRP、BPP 等多个组合优化领域中一致超越现有方法。

## 研究背景与动机

### 自动启发式设计 (AHD) 的局限

组合优化问题 (COP) 如 TSP、CVRP 等是 NP-hard 问题，传统求解器依赖于专家手工设计的启发式，开发成本高、跨问题灵活性差。近年来，Language Hyper-Heuristics (LHH) 利用 LLM 的代码生成和推理能力自动设计启发式，取得了显著进展。代表性方法包括：

- **EoH**：将进化算法与 LLM 结合演化代码
- **ReEvo**：结合进化搜索与自我反思
- **HSEvo**：引入多样性度量和和谐搜索
- **MCTS-AHD**：使用 MCTS 与渐进扩展探索启发式空间

### 现有方法的两大局限

**单策略局限**：现有方法只优化求解器中的单个组件（通常是启发式评分函数），忽略了其他组件的创新机会。弱组件的组合可能比单独优化一个强组件产生更好的整体效果。

**反馈机制浅层**：虽然有自我反思机制（如 ReEvo），但通常基于静态比较或表面总结，难以捕捉组件间的隐性不兼容或潜在协同效应，LLM 收到的反馈不足以进行有意义的深层修改。

### 核心创新思想

将求解器设计从"单启发式函数优化"扩展为"多策略联合优化"，并引入博弈论中的回合制自我对弈机制，让两个 LLM 代理交替生成和改进策略，通过竞争压力和涌现合作驱动更优解的发现。

## 方法详解

### 整体框架

MOTIF 采用两阶段优化流程：

1. **组件级竞争 (Component-wise Competition)**：将求解器分解为 K 个独立策略，分别优化每个策略
2. **系统级精炼 (System-aware Refinement)**：在全局系统配置可见的情况下重新优化每个策略，促进组件间协同

### 关键设计

#### 1. **多策略优化问题建模**：将求解器设计统一为组合优化框架

**功能**：将求解器 $s$ 中的 K 个策略 $\Pi = (\pi_1, \pi_2, \ldots, \pi_K)$ 联合优化，最小化期望求解目标：

$$\Pi^* = \arg\min_\Pi \mathbb{E}_{\mathbf{x} \sim \mathcal{X}_d}[F_d(\mathbf{x} | \Pi)]$$

**核心定义**：
- **策略 (Strategy)**：求解器中的一个可替换组件，如启发式评分规则、构造策略、邻域移动、惩罚更新等
- **策略空间 $\mathcal{S}_\pi$**：某类策略的所有合法实现，包括功能变体和参数化方式
- 同一策略空间中的策略共享函数签名和优化目标，确保求解器内的互操作性

**设计动机**：此建模比孤立优化单启发式更具全局性，能解锁组件间的协同效应。

#### 2. **竞争性蒙特卡洛树搜索 (CMCTS)**：回合制双代理搜索架构

**功能**：为每个策略构建竞争树，两个 LLM 代理交替出招，通过三种竞争算子生成新的策略实现。

**搜索流程**（标准 MCTS 四步）：

**(1) 选择 (Selection)**：从根节点沿当前玩家创建的子节点前进，使用 UCB 公式平衡探索与利用：
$$\text{UCB}(o, \pi) = \frac{Q(o, \pi)}{N(o, \pi)} + C_{in}\sqrt{\frac{\ln N(\pi)}{N(o, \pi)}}$$

**(2) 扩展 (Expansion)**：选择一个算子，调用 LLM 生成新实现：
$$\pi' \leftarrow \mathcal{L}(\text{Prompt}(o; \{\pi^{(p)}, \pi^{(\neg p)}\}; \{\mathcal{H}^{(p)}, \mathcal{H}^{(\neg p)}\}; \mathcal{B}))$$
提示包含：算子类型、双方当前实现、双方历史记录、当前基线。

**(3) 评估 (Evaluation)**：将新实现插入系统，替换对应策略，与基线比较计算改进率。

**(4) 回传 (Backpropagation)**：Q 值同时考虑绝对改进和相对竞争增益：
$$Q^{(p)} = \lambda \cdot \sigma(I^{(p)}) + (1-\lambda) \cdot \sigma(I^{(p)} - I^{(\neg p)})$$

#### 3. **三种竞争算子**：覆盖对抗-学习-创新行为谱

| 算子 | 策略 | 行为 |
|------|------|------|
| **Counter** | 针对对手弱点设计对抗性改进 | 利用对手低效或局限 |
| **Learning** | 整合对手方案的优势到自身 | 建设性整合 |
| **Innovation** | 忽略先前方案，提出全新方法 | 纯探索性发明 |

三者共同覆盖从对抗利用到建设性整合再到探索性发明的丰富行为谱。

#### 4. **外部控制器与动态基线**

- **外部控制器**：使用 UCB 规则选择下一轮优化哪个策略树：
$$k = \arg\max_{1 \le k \le K}\left(\frac{R_j}{N_j} + C_{out}\sqrt{\frac{\ln\sum_i N_i}{N_j}}\right)$$

- **动态基线**：初始基线为手工设计或预热启动的最优实现，当某代理产生严格更优实现时更新。每个代理竞争的是另一玩家最近的最佳方案。

#### 5. **系统级精炼**

第二阶段中，每个策略在全系统可见的条件下重新优化。提示构造包含完整系统配置、全局基线成本和历史搜索轨迹：
$$\pi_k' \leftarrow \mathcal{L}(\text{Prompt}(\Pi; \{\pi_k^{(p)}, \pi_k^{(\neg p)}\}; \{\mathcal{H}^{(p)}, \mathcal{H}^{(\neg p)}\}))$$

仅当改进超过基线和对手最佳结果时才接受更新。

### 训练策略

- 使用 gpt-4o-mini-2024-07-18 作为两个代理的 LLM 后端
- 轻量 CoT：每个响应包含 reasoning（≤5句推理）、code（Python）、summary（变更摘要）三字段
- 轻量训练集优化，最终在保留测试集上评估
- 使用结构化输出格式以确保可靠解析

## 实验关键数据

### 主实验

#### 单策略优化：GLS 框架下的 TSP

| 方法 | TSP50 | TSP100 | TSP200 | TSP500 |
|------|-------|--------|--------|--------|
| EoH | 0.0000 | 0.0012 | 0.0639 | 0.5796 |
| ReEvo | 0.0000 | 0.0335 | 0.2081 | 0.7918 |
| HSEvo | 0.0108 | 0.3095 | 1.1254 | 2.4593 |
| MCTS-AHD | 0.0000 | 0.0024 | 0.0652 | 0.5611 |
| **MOTIF** | **0.0000** | **0.0007** | **0.0610** | **0.5577** |

即使在单策略设置下，MOTIF 也显著超越现有方法。

#### 多策略优化：Deconstruction-then-Repair 框架

| 策略组合 | TSP50 | TSP100 | TSP200 | CVRP50 | CVRP100 | CVRP200 | BPP100 | BPP200 | BPP300 |
|---------|-------|--------|--------|--------|---------|---------|--------|--------|--------|
| $\pi_1$ | 0.39 | 0.81 | 0.72 | 1.47 | 2.96 | 2.30 | 3.17 | 4.82 | 4.77 |
| $\pi_2$ | 3.13 | 6.24 | 8.18 | 4.27 | 7.04 | 6.85 | 23.19 | 24.42 | 25.00 |
| $\pi_3$ | 3.88 | 7.91 | 11.35 | 6.34 | 5.07 | 4.28 | 12.70 | 12.84 | 12.15 |
| $(\pi_1,\pi_2)$ | 2.58 | 5.04 | 5.84 | 6.06 | 7.94 | 8.85 | 23.15 | 24.40 | 24.98 |
| $(\pi_2,\pi_3)$ | 3.83 | 7.97 | 11.59 | 10.64 | 12.31 | 11.71 | 23.05 | 24.32 | 24.89 |
| **$(\pi_1,\pi_2,\pi_3)$** | **3.88** | **7.96** | **11.65** | **10.98** | **12.84** | **13.06** | **23.94** | **25.02** | **25.41** |

联合优化三个策略组件在 TSP、CVRP、BPP 上一致超越单策略优化。在 CVRP200 上提升达 13.06%。

### 消融实验

| 配置 | ACO-Train | ACO-Test | GLS-Train | GLS-Test |
|------|-----------|----------|-----------|----------|
| w/o Outer Controller | 0.74 | 5.61 | 0.81 | 2.88 |
| w/o Dynamic Baseline | 1.55 | **9.50** | 0.62 | 2.94 |
| w/o Final Round | 1.26 | 5.72 | 0.39 | 2.82 |
| w/o Reasoning | 1.63 | 7.88 | 0.80 | 3.05 |
| w/o Counter | 1.42 | 6.71 | 0.64 | 3.20 |
| w/o Learning | 0.91 | **9.59** | 0.62 | 3.00 |
| w/o Innovation | 1.73 | 6.65 | 0.42 | 2.86 |
| **MOTIF (original)** | **0.80** | **5.21** | **0.34** | **2.73** |

**关键发现**：
- 动态基线移除导致最严重的性能下降，尤其在测试阶段
- Reasoning 移除也有显著影响
- 三个算子各有贡献，其中 Learning 对 ACO 影响最大

### 多样性分析

| 算子 | 成功率 ↑ | 新颖度 ↑ | 轮廓分数 ↑ |
|------|---------|---------|-----------|
| Counter | 93% | 0.0136 | 0.5121 |
| Learning | 92% | 0.0118 | **0.5325** |
| **Innovation** | **97%** | **0.0175** | 0.4793 |

Innovation 新颖度最高但一致性最低；Learning 新颖度最低但一致性最高；Counter 行为最平衡。

### 关键发现

1. 多策略联合优化一致优于单策略优化，各组件间存在明显协同效应
2. 竞争压力驱动双玩家相互提升，曲线高度对齐
3. LLM 不仅能调优单个启发式，还能协同设计完整的算法流水线
4. 即使使用编码能力有限的 gpt-4o-mini，结构化提示也能产生高质量结果

## 亮点与洞察

1. **多策略优化的全新问题建模**：将求解器设计从单组件优化推广为多组件联合优化，开辟新范式
2. **博弈论启发的搜索**：回合制双代理竞争是首次在组合优化求解器设计中应用自我对弈，利用对抗压力和上下文学习实现涌现式自我改进
3. **三算子设计**：Counter/Learning/Innovation 覆盖完整的行为谱，各有明确分工
4. **实用性强**：在 ACO、GLS、Deconstruction-Repair 三种不同框架上验证，覆盖 TSP、CVRP、BPP 等 5 种 COP

## 局限与展望

1. 依赖 LLM 的代码生成质量，使用更强 LLM（如 GPT-4o）可能效果更好
2. 双代理交互增加了 LLM 调用次数，成本较高
3. 当前策略数 K 较小（2-3个），更多策略的可扩展性待验证
4. 系统级精炼阶段使用固定基线而非动态基线，可能限制更新频率
5. 可探索更多代理（3+玩家）的扩展

## 相关工作与启发

- **ReEvo / MCTS-AHD**：单策略 LHH 基线，MOTIF 在此基础上推广到多策略
- **SPAG / CDG**：LLM 自我对弈在推理增强和欺骗检测中的应用，MOTIF 首次将其应用于算法设计
- **AlphaGo / AlphaZero**：MCTS + 自我对弈的成功范例，MOTIF 将其思想迁移到代码生成领域

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 多策略优化建模+博弈搜索的组合非常新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 3种框架、5种COP、详细消融和多样性分析
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，但公式较密集
- 价值: ⭐⭐⭐⭐⭐ — 为 LLM 驱动的算法设计开辟多策略新方向

<!-- RELATED:START -->

## 相关论文

- [Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework](../../ECCV2024/optimization/handling_the_non-smooth_challenge_in_tensor_svd_a_multi-objective_tensor_recover.md)
- [Parametrized Multi-Agent Routing via Deep Attention Models](parametrized_multi-agent_routing_via_deep_attention_models.md)
- [AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [MOBO-OSD: Batch Multi-Objective Bayesian Optimization via Orthogonal Search Directions](../../NeurIPS2025/optimization/mobo-osd_batch_multi-objective_bayesian_optimization_via_orthogonal_search_direc.md)

<!-- RELATED:END -->
