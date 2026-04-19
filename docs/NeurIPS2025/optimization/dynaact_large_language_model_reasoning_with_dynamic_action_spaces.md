---
title: >-
  [论文解读] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces
description: >-
  [NeurIPS 2025][优化][动态动作空间] DynaAct 将 LLM 推理中的动作空间构建建模为子集选择问题，通过兼顾效用和多样性的子模函数在每步动态构建紧凑动作空间，在 6 个基准上显著优于 rStar、RAP 等方法，MATH-500 上比 rStar 高 6.8%。
tags:
  - NeurIPS 2025
  - 优化
  - 动态动作空间
  - 子模函数
  - MCTS
  - LLM推理
  - 子集选择
---

# DynaAct: Large Language Model Reasoning with Dynamic Action Spaces

**会议**: NeurIPS 2025  
**arXiv**: [2511.08043](https://arxiv.org/abs/2511.08043)  
**代码**: https://github.com/zhaoxlpku/DynaAct  
**领域**: LLM推理 / 决策优化  
**关键词**: 动态动作空间, 子模函数, MCTS, LLM推理, 子集选择

## 一句话总结
DynaAct 将 LLM 推理中的动作空间构建建模为子集选择问题，通过兼顾效用和多样性的子模函数在每步动态构建紧凑动作空间，在 6 个基准上显著优于 rStar、RAP 等方法，MATH-500 上比 rStar 高 6.8%。

## 研究背景与动机

**领域现状**：LLM 复杂推理通常在 MDP 框架下进行：定义状态空间和动作空间，用 MCTS 等搜索策略寻找最优推理路径。RAP 用自动生成的子问题作为动作，rStar 手工定义 5 种动作。

**现有痛点**：(a) 手工定义的动作空间（如 rStar 的 5 种动作）过于领域特异，缺乏可扩展性；(b) 自动生成的动作空间（如 RAP 的实时子问题）冗余度高，穷举搜索计算量大。

**核心矛盾**：好的动作空间需要同时满足两个矛盾属性——**可扩展性**（从数据自动学习，跨领域通用）和**紧凑性**（每步只保留少量高价值候选动作）。

**本文目标** 如何自动构建一个既通用又紧凑的动作空间用于 LLM 推理。

**切入角度**：将动作空间构建视为子集选择问题，利用子模函数的递减边际效益性质确保选出的子集兼顾效用和多样性。

**核心 idea**：用子模函数 + 贪心算法从大型代理动作空间中每步动态选出最优的 5 个动作。

## 方法详解

### 整体框架
三阶段流程：(1) **代理动作空间估计**——从多样化语料库中提取通用推理模式（观察草图）组成完整动作空间 $\mathcal{A}$；(2) **子模函数定义**——定义兼顾效用和多样性的评分函数 $F$，通过 Q-learning 训练嵌入模型；(3) **动态动作空间构建**——每步用贪心算法从 $\mathcal{A}$ 中选 $m=5$ 个动作组成 $\mathcal{A}_t$，然后用 MCTS 评估并选择最优动作。

### 关键设计

1. **代理动作空间估计**:

    - 功能：从多领域问题语料中自动提取可复用的推理模式
    - 核心思路：将 Open-Platypus 语料（24,652 题）分为 $k=2500$ 组，用 Llama-3.1-70B 提取每组的观察草图（observation sketch），合并去重后得到 40,822 个观察作为 $\mathcal{A}$
    - 设计动机：避免手工定义，从数据中自动发现跨领域通用的推理操作

2. **子模函数设计**:

    - 功能：定义评分函数 $F(\mathcal{A}_t; s_t) = \alpha f_\text{util}(\mathcal{A}_t; s_t) + \beta f_\text{div}(\mathcal{A}_t)$
    - 核心思路：
        - 效用项：$f_\text{util} = \log(\sum_{a \in \mathcal{A}_t} \exp(\mathbf{e}(s_t)^T \mathbf{e}(a)))$，LogSumExp 保证子模性
        - 多样性项：$f_\text{div} = \sum_{a_i} \min_{a_j \neq a_i} (1 - \mathbf{e}(a_i)^T \mathbf{e}(a_j))$，鼓励嵌入空间中最大程度不同
        - 权重 $\alpha=0.9, \beta=0.1$
    - 设计动机：效用项确保选出对当前推理状态有价值的动作，多样性项防止冗余

3. **嵌入模型训练（Q-learning）**:

    - 功能：训练嵌入函数 $\mathbf{e}(\cdot)$ 使 $\mathbf{e}(s_t)^T \mathbf{e}(a)$ 近似 Q 值
    - 核心思路：用观察草图作为示范数据，定义 $\mathcal{L} = (\mathbf{e}(s_t)^T \mathbf{e}(a) - (r + \log\sum_{a'} \exp(\mathbf{e}(s_{t+1})^T \mathbf{e}(a'))))^2$，正确动作 $r=1$，其他 $r=0$
    - 实现：Llama-3.2-1B 作为嵌入骨干，83,083 对状态-动作对训练

4. **贪心动作选择**:

    - 功能：每步从 $\mathcal{A}$ 中选 $m=5$ 个动作
    - 核心思路：利用子模性，贪心算法保证 $(1-1/e)$ 近似比，复杂度 $O(m^2|\mathcal{A}|)$
    - 实现优化：$\mathbf{e}(a)$ 可预计算缓存，仅需在线编码 $\mathbf{e}(s_t)$

### 搜索策略
- MCTS 进行 16 次 rollout，world model 为 Llama-3.1-8B-Instruct
- 仅嵌入模型需训练，基础 LLM 全程冻结

## 实验关键数据

### 主实验

| 类型 | 基准 | Zero-shot CoT | SC@maj16 | RAP | rStar | **DynaAct** |
|------|------|--------------|----------|-----|-------|------------|
| General | MMLU | 68.87 | 69.66 | 69.46 | 68.61 | **70.22** |
| General | MMLU-Pro | 43.45 | 49.36 | 48.70 | 48.81 | **51.40** |
| Reasoning | GPQA | 31.82 | 34.34 | 38.89 | 36.87 | **39.39** |
| Reasoning | ARC-C | 81.06 | 80.63 | 85.41 | 86.43 | **88.31** |
| Math | GSM8K | 76.12 | 86.66 | 87.79 | 87.11 | **89.16** |
| Math | MATH-500 | 45.40 | 52.00 | 51.60 | 54.20 | **61.00** |

六个基准全面 SOTA。MATH-500 比 rStar 高 6.8 个百分点。

### 消融实验

| 配置 | ARC-C | MATH-500 | 说明 |
|------|-------|----------|------|
| DynaAct (full) | 88.31 | 61.00 | 完整模型 |
| - util | 87.63 | 53.40 | 去效用项，MATH 掉 7.6% |
| - div | 86.52 | 53.80 | 去多样性项，掉 7.2% |
| - q-learning | 87.80 | 55.80 | 不训练嵌入，掉 5.2% |
| - submodular | 85.15 | 52.00 | 去子模函数，掉 9.0% |

### 效率对比

| 方法 | 相对时间 ↓ | 精度 ↑ |
|------|-----------|--------|
| **DynaAct** | 1.00 | **61.00** |
| rStar | 0.95 | 54.20 |
| RAP | 1.12 | 51.60 |

DynaAct 延迟仅比 rStar 多 5%，但精度高 6.8%。比 RAP 更快（RAP 实时生成子问题开销大）。

### 关键发现
- 子模函数是性能核心：去掉后掉 9%（MATH-500），说明动态紧凑动作空间比随机/固定动作空间关键得多
- 效用项和多样性项缺一不可：单去任一项 MATH-500 均从 61% 降至约 53%
- 紧凑性优势明显：即使 $m=5$，DynaAct 仍随 rollout 数增加显著提升；RAP 在 $m=5$ 或 $m=10$ 时增加 rollout 几乎无改善，需 $m=15$ 才见效
- 关键步骤覆盖率：DynaAct 为 0.63 vs rStar 的 0.47，动作选择确实更"有用"

## 亮点与洞察
- **将动作空间构建形式化为子集选择**：这是一个被忽视的研究问题，与策略学习和奖励建模正交，为 MDP-based 推理开辟了新方向
- **子模函数设计巧妙**：LogSumExp 效用项自然满足子模性，嵌入空间中的最近邻距离和作为多样性度量简洁有效
- **嵌入预计算 + 贪心线性复杂度**：动作嵌入可缓存，每步仅需编码当前状态，不引入显著延迟

## 局限与展望
- 代理动作空间从 Open-Platypus 提取，领域覆盖有偏；针对特定垂直领域（如编程、科学实验）可能不够
- 40,822 个观察中大量冗余，$\mathcal{A}$ 本身可进一步压缩
- 依赖 Llama-3.1-70B 提取观察草图，对更小模型部署不够友好
- 目前仅用单一嵌入函数 $\mathbf{e}(\cdot)$，信息容量有限；可探索多头或层次化嵌入

## 相关工作与启发
- **vs rStar**: rStar 手工定义 5 种动作缺乏可扩展性，在 MATH-500 和通用任务上弱于 DynaAct
- **vs RAP**: RAP 用 LLM 实时生成子问题作为动作空间，冗余度高且延迟大；DynaAct 预提取+动态选择更高效
- **vs test-time scaling 方向**: DynaAct 关注动作空间质量而非搜索策略或数据质量，是正交的贡献

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将动作空间构建建模为子模子集选择是全新且有理论基础的视角
- 实验充分度: ⭐⭐⭐⭐⭐ 六个基准、消融完整、效率对比、紧凑性分析、效用分析全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，理论和实验衔接好
- 价值: ⭐⭐⭐⭐ 为 MDP-based LLM 推理提供了重要的基础设施型贡献

<!-- RELATED:START -->

## 相关论文

- [Large Language Bayes](large_language_bayes.md)
- [Doubly Robust Alignment for Large Language Models](doubly_robust_alignment_for_large_language_models.md)
- [VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [Training-Free Bayesianization for Low-Rank Adapters of Large Language Models](training-free_bayesianization_for_low-rank_adapters_of_large_language_models.md)
- [Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)

<!-- RELATED:END -->
