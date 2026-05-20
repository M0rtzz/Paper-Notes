---
title: >-
  [论文解读] Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)
description: >-
  [AAAI 2026][模型压缩][generalised planning] 提出 Moose 规划器，利用目标回归（goal regression）从训练问题中合成泛化规划程序：将训练问题的目标拆解为单目标子问题逐个最优求解，通过回归和提升（lifting）得到一阶条件-动作规则集…
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "generalised planning"
  - "goal regression"
  - "policy synthesis"
  - "剪枝"
  - "planning domains"
---

# Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)

**会议**: AAAI 2026  
**arXiv**: [2511.11095](https://arxiv.org/abs/2511.11095)  
**代码**: [https://github.com/dillonzchen/moose](https://github.com/dillonzchen/moose)  
**领域**: 模型压缩  
**关键词**: generalised planning, goal regression, policy synthesis, search pruning, planning domains

## 一句话总结

提出 Moose 规划器，利用目标回归（goal regression）从训练问题中合成泛化规划程序：将训练问题的目标拆解为单目标子问题逐个最优求解，通过回归和提升（lifting）得到一阶条件-动作规则集，用于满足性规划（直接执行规则）或最优规划（编码为公理剪枝搜索空间）。

## 研究背景与动机

泛化规划（Generalised Planning, GP）旨在合成能解决一族相关规划问题的程序。其核心目标是**摊销合成成本**：对一族问题的求解速度快于通用规划器逐个求解。

现实世界中存在大量计算上容易求解、但已有通用规划器难以规模化的领域。例如 UPS 每天在 200 多个国家和地区配送超 2000 万个包裹，但 SOTA 通用规划器在只有 100 个包裹的简化投递问题上就已力不从心。

现有 GP 方法（如 sketch learner、PG3、Loom）各有局限：
- 合成成本高，部分领域在 12 小时内无法完成
- 内存消耗大
- 缺乏对最优规划的理论保证

Moose 的动机是利用**目标回归**——一项在知识表示社区有长久历史的技术——结合**问题松弛**（将多目标分解为单目标），以高效的三步流程完成泛化规划的合成与实例化。

## 方法详解

### 整体框架

Moose 的泛化规划分为两个阶段：

**合成阶段**（Algorithm 1）：
1. 对每个训练问题的每种目标原子排列，将目标拆解为单目标子问题
2. 用最优规划器逐个求解各子目标，同时沿途推进当前状态
3. 对每个子计划进行**目标回归**（从目标状态倒推到所需前条件）
4. 将回归得到的状态-目标-动作序列进行**提升（lifting）**，用自由变量替换具体对象，生成一阶规则

**实例化阶段**：
- **满足性规划**（Algorithm 3）：在新问题上按优先级顺序尝试接地（grounding）规则，执行匹配的宏动作序列直到达到目标
- **最优规划**（Section 3.3）：将学到的规则编码为 PDDL 公理，约束搜索中各状态的可执行动作集，从而剪枝搜索空间

### 关键设计

1. **Moose 规则（Moose Rule）**:

    - 功能：以一阶条件-动作对的形式紧凑地表示泛化策略
    - 核心思路：每条规则包含自由变量集、状态条件（当前状态须满足的原子集）、目标条件（尚未达成的目标原子集）和动作序列（含宏动作）
    - 设计动机：规则的前件可紧凑地捕获一组状态，提升后的规则可跨不同对象数量泛化
    - 每条规则附带优先级（cost-to-go），决定执行顺序

2. **目标回归 + 提升（Goal Regression + Lifting）**:

    - 功能：从一个具体计划中提取泛化的条件-动作规则
    - 核心思路：回归计算动作的前像（preimage），即执行动作前需要满足的最小原子集；然后将具体对象替换为自由变量
    - 关键性质：回归自然去除与目标无关的原子（如运输 cake 的计划中忽略 dog 的位置），使规则具有良好的泛化性

3. **目标独立性分类（TGI / SGI / OGI）**:

    - 功能：形式化分类规划域中目标可独立求解的程度
    - TGI（True Goal Independence）：任意目标排列下贪心算法均有效
    - SGI（Serialisable Goal Independence）：存在某种目标排列使贪心算法有效
    - OGI（Optimal Goal Independence）：存在某种排列使贪心算法得到最优解
    - 计算复杂性：pTGI 在 P 中，pSGI 和 pOGI 为 NP 完全

### 损失函数 / 训练策略

本文不涉及神经网络训练。规则合成基于：
- 使用 A* + LM-cut 启发式求解各子问题的最优计划
- 默认使用 $n_p = 3$ 种目标排列
- 接地（grounding）用 SQLite 数据库查询实现
- 可选扩展：对目标子集（而非单目标）进行回归，生成更多高质量规则

## 实验关键数据

### IPC 基准测试中的 TGI 有效性

| 统计 | 数据 |
|------|------|
| 所有输出有效的域（可能展现 TGI） | 13/38 (34.2%) |
| 至少半数输出有效的域 | 19/38 (50.0%) |
| 至少一个输出有效的域 | 27/38 (71.1%) |
| 所有问题中有效输出占比 | 590/1184 (49.8%) |

### 主实验：满足性规划覆盖率

| 领域 | SLearn-0 | SLearn-1 | SLearn-2 | Lama | Moose |
|------|---------|---------|---------|------|-------|
| Barman | 0.0 | 0.0 | 0.0 | 49 | **90.0** |
| Ferry | 15.0 | 67.0 | 60.0 | 69 | **90.0** |
| Gripper | 59.6 | 50.8 | 33.0 | 65 | **90.0** |
| Logistics | 0.0 | 0.0 | 0.0 | 77 | **89.6** |
| Miconic | 68.8 | 72.6 | 67.8 | 77 | **90.0** |
| Rovers | 0.0 | 0.0 | 0.0 | 66 | **90.0** |
| Satellite | 0.0 | 29.2 | 34.6 | 89 | **90.0** |
| Transport | 0.0 | 63.0 | 46.8 | 66 | **90.0** |
| **总计 (720)** | 143.4 | 282.6 | 242.2 | 558 | **719.6** |

### 最优规划覆盖率

| 领域 | Blind | LMcut | Scorpion | SymK | Moose |
|------|-------|-------|----------|------|-------|
| Barman | 0 | 0 | 0 | 12 | **24.6** |
| Ferry | 10 | 18 | 17 | 18 | **30.0** |
| Gripper | 9 | 8 | 7 | **30** | 27.0 |
| Miconic | 30 | 30 | 30 | 30 | **30.0** |
| **总计 (240)** | 93 | 119 | 140 | 154 | **183.0** |

### 消融实验：联合目标回归的效果

| 配置 | 说明 |
|------|------|
| $n_r = 1$（单目标回归） | 基础配置，合成快 |
| $n_r = 2$（2-子集目标回归） | 7/8 个域解质量提升（1.1%-25.0%），但合成和实例化成本增大 |
| $n_r = 3$（3-子集目标回归） | 与 $n_r = 2$ 相比改善极小 |

### 关键发现

- **Moose 几乎解决了所有经典和数值规划问题**（719.6/720），远超 Lama（558）和 SLearn（最好 282.6）
- **合成效率优势**：Moose 在 5/8 个域上合成速度快于最佳 SLearn 配置，且内存使用全面更低（<1GB vs 高达 7.6GB）
- **最优规划有效性**：虽然理论上不保证最优，但 Moose 经验上输出的计划都是最优的
- **将规则编码为公理通常比基础 SymK 更好**：搜索空间缩减的收益大于公理评估的成本
- **解质量方面**：Moose 在 5/8 经典域和 3/4 数值域上优于 Lama 和 ENHSP

## 亮点与洞察

- **极简而有效的流程**：分解目标→最优求解→回归→提升，四步即完成泛化策略合成，概念清晰
- **理论保证**：在 TGIC 条件下证明了满足性规划的完备性；在 TGIC + OGI 条件下证明了最优规划的完备性
- **将规划状态视为数据库、规则视为查询**：利用 SQLite 实现高效的规则接地，巧妙地复用了数据库系统的成熟优化
- **ESHO 域（Easy-to-Solve, Hard-to-Optimise）的识别**：P-time 可解但 NP-hard 最优化的域恰好是 Moose 最擅长的
- **跨数值规划泛化**：同一框架自然扩展到数值规划域（NFerry, NMiconic, NMinecraft, NTransport）

## 局限与展望

- TGI 假设在许多规划域不成立（如 Blocksworld 的 Sussman Anomaly），限制了方法的适用范围
- 理论保证所需的训练集大小在最坏情况下关于域大小是指数级的
- 当目标间存在复杂依赖时（如互斥目标），贪心分解策略可能失效
- 规则数量随 $n_r$ 增大而快速增长，导致实例化成本上升一个数量级
- 未处理需要传递闭包计算的域
- 合成依赖最优规划器（A* + LM-cut），当子问题本身难以最优求解时效率受限

## 相关工作与启发

- **Loom**（Illanes & McIlraith, 2019）：通过等价对象装袋生成量化问题后用 FOND 规划器合成策略
- **PG3**（Yang et al., 2022）：在泛化策略空间中启发式搜索，也使用目标回归处理"遗漏"状态
- **Sketch Learner**（Drexler et al., 2022）：学习 sketch（抽象描述），合成开销更大
- **目标回归的长历史**：从 STRIPS（Fikes et al., 1972）到 Waldinger（1977）到 Reiter（2001），Moose 以底层（bottom-up）方式重新激活了这一经典技术
- 启发：将规划领域中的经典形式方法与现代基于数据的学习策略结合，可以在实际基准上取得显著突破

## 评分

- 新颖性: ⭐⭐⭐⭐ 将目标回归与目标分解结合用于泛化规划合成，理论分析（TGI/SGI/OGI 复杂性分类）扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 8 个经典域 + 4 个数值域，满足性/最优规划双场景，合成/实例化/解质量三维度评估
- 写作质量: ⭐⭐⭐⭐ 理论部分严谨，算法伪代码清晰，但篇幅较长需要耐心阅读
- 价值: ⭐⭐⭐⭐ 在 ESHO 域上大幅推进了 SOTA，解决了通用规划器难以规模化的实际问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](../../ICML2026/model_compression/qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICLR 2026\] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](../../ICLR2026/model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)
- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[ACL 2026\] No-Worse Context-Aware Decoding: Preventing Neutral Regression in Context-Conditioned Generation](../../ACL2026/model_compression/no-worse_context-aware_decoding_preventing_neutral_regression_in_context-conditi.md)

</div>

<!-- RELATED:END -->
