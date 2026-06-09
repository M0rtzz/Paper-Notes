---
title: >-
  [论文解读] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning
description: >-
  [NeurIPS 2025][LLM推理][token级优势估计] KTAE 提出了一种不依赖额外模型的 token 级优势估计算法，通过 Fisher 精确检验和信息增益量化每个 token 与正确推理结果的统计关联，将细粒度 token 重要性叠加到 GRPO/DAPO 的 rollout 级优势上…
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "token级优势估计"
  - "GRPO"
  - "DAPO"
  - "数学推理"
  - "强化学习"
---

# KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2505.16826](https://arxiv.org/abs/2505.16826)  
**代码**: [GitHub](https://github.com/ZNLP/KTAE)  
**领域**: 模型压缩  
**关键词**: token级优势估计, GRPO, DAPO, 数学推理, 强化学习

## 一句话总结
KTAE 提出了一种不依赖额外模型的 token 级优势估计算法，通过 Fisher 精确检验和信息增益量化每个 token 与正确推理结果的统计关联，将细粒度 token 重要性叠加到 GRPO/DAPO 的 rollout 级优势上，在5个数学推理基准上超越基线并显著缩短生成长度。

## 研究背景与动机

**领域现状**：GRPO 是当前 LLM 强化学习的主流算法，通过组内相对策略优化消除了对 Critic 模型的依赖；DAPO 在此基础上引入了 Clip-Higher、动态采样等改进。

**现有痛点**：GRPO/DAPO 计算的是 rollout 级别的优势，即同一条推理链中每个 token 被赋予相同的优势值 $\hat{A}_{i,t} = \frac{R_i - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$。但在数学推理中，不同 token 的贡献差异巨大——错误推理可能仅在最后几步偏离正确路径。

**核心矛盾**：均匀优势导致模型无法精确学习哪些 token 是关键推理步骤，哪些是无关 token（如"First"、"denote"），阻碍了有效学习。

**本文目标** 在不引入额外模型的前提下，为每个 token 估计细粒度的优势值。

**切入角度**：利用多条 rollout 的正确/错误标签，对每个 token 构建列联表，用统计方法量化其与正确结果的关联强度和方向。

**核心 idea**：把 token 出现与 rollout 正确性之间的统计关联（Fisher检验+信息增益+BM25频率分析）转化为 token 级优势，叠加到 GRPO 优势上。

## 方法详解

### 整体框架
给定问题 $q$，从策略模型采样 $G$ 条 rollout $\{o_1, \dots, o_G\}$，每条有规则奖励 $\{R_1, \dots, R_G\}$。KTAE 对每个 token $o_{ij}$ 构建其与正确 rollout 的关联，最终输出 token 级优势 $\hat{A}_{o_{ij}}^{KTAE}$，替代 GRPO 的均匀优势参与策略梯度更新。

### 关键设计

1. **Token 级列联表构建**:

    - 功能：量化每个 token 在正确/错误 rollout 中的出现模式
    - 核心思路：将 $G$ 条 rollout 分为正确集 $x_T$ 和错误集 $x_F$，对每个 token $o_{ij}$ 统计其出现在正确 rollout 中的次数 $a$、出现在错误 rollout 中的次数 $b$、未出现在正确中的 $c$、未出现在错误中的 $d$，构成 $2 \times 2$ 列联表
    - 设计动机：列联表是统计关联分析的经典工具，自然适合刻画 token 出现与 rollout 正确性的关系

2. **关联强度量化 — Fisher 精确检验**:

    - 功能：检验 token 出现与 rollout 正确性是否存在显著关联
    - 核心公式：$Fisher(o_{ij}) = \frac{(a+b)!(c+d)!(a+c)!(b+d)!}{a!b!c!d!N!}$，在对数空间计算以处理大阶乘
    - 转换函数：$\mathcal{F}(o_{ij}) = e^{-2 \cdot Fisher(o_{ij})}$（p=1时为0，p→0时趋近1）
    - 选择 Fisher 而非卡方检验的原因：当样本量 $G$ 较小（如8或16）时，Fisher 检验提供精确概率，卡方检验的近似不准确

3. **关联强度量化 — 信息增益**:

    - 功能：从信息论角度互补量化关联
    - 核心公式：$IG(o_{ij}) = H(Y) - H(Y|X_{o_{ij}})$，其中 $H(Y)$ 是 rollout 正确性的熵，$H(Y|X_{o_{ij}})$ 是已知 token 是否出现后的条件熵
    - 综合：$h_1 \cdot \mathcal{F}(o_{ij}) + h_2 \cdot IG(o_{ij})$ 作为关联强度

4. **关联方向量化**:

    - 功能：判断 token 与正确结果是正相关还是负相关
    - 核心思路：基于 BM25 的词频思想，计算 token 在正确和错误 rollout 拼接序列中的标准化频率 $TF_{T/F}(o_{ij}) = \frac{(k_1+1) \cdot tf_{T/F}(o_{ij})}{k_1(1-b+b \times \frac{len_{T/F}}{len_{avg}})+tf_{T/F}(o_{ij})}$
    - 方向得分：$D(o_{ij}) = (\arcsin\sqrt{\frac{a}{a+c}} - \arcsin\sqrt{\frac{b}{b+d}}) + h_3(\frac{TF_T}{TF_F} - \frac{TF_F}{TF_T})$，结合 Cohen's h 效应量和频率比
    - 设计动机：高频通用 token 靠比例差区分，低频关键 token 靠频率比区分

5. **最终 Token 级优势**:

    - 将关联强度×方向得分得到 key-token-value，经 sigmoid 归一化后叠加到 GRPO 优势：
    - $\hat{A}_{o_{ij}}^{KTAE} = \hat{A}_{o_i}^{GRPO} + \sigma\big((h_1 \cdot \mathcal{F}(o_{ij}) + h_2 \cdot IG(o_{ij})) \cdot D_{o_{ij}}\big) - 0.5$

### 训练策略
- KTAE 作为即插即用模块，直接替换 GRPO/DAPO 中的优势计算步骤
- 不引入额外模型，计算开销主要来自列联表统计（与模型大小无关，仅与 token 数相关）
- 与 DAPO 的 Clip-Higher、动态采样等改进正交，可组合使用

## 实验关键数据

### 主实验（7B模型，5个数学基准，zero-shot greedy pass@1）

| 方法 | AIME24 | MATH-500 | AMC | Minerva | OlympiadBench | Avg |
|------|--------|----------|-----|---------|---------------|-----|
| GRPO-7B | 36.7 | 81.0 | 57.8 | 32.7 | 43.2 | 50.3 |
| **GRPO+KTAE-7B** | 33.3 | **82.4** | **65.1** | **33.8** | **43.7** | **51.7** |
| DAPO-7B | 36.7 | 81.8 | 60.2 | 34.5 | 45.3 | 51.7 |
| **DAPO+KTAE-7B** | 36.7 | **83.2** | **63.9** | **35.3** | 43.7 | **52.5** |
| R1-Distill-Qwen-1.5B | 20.0 | 77.4 | 49.4 | 25.0 | 35.8 | 41.5 |
| DAPO+KTAE-1.5B | 20.0 | 77.6 | 50.6 | 29.0 | 40.0 | **43.4** |

DAPO+KTAE-7B 取得最高平均分 52.5，且 KTAE-1.5B 超越了同 base model 的 R1-Distill-1.5B。

### 生成长度对比（7B模型平均响应长度）

| 方法 | AIME24 | MATH-500 | AMC | Minerva | OlympiadBench | Avg |
|------|--------|----------|-----|---------|---------------|-----|
| GRPO-7B | 989 | 606 | 806 | 641 | 813 | 771.0 |
| **GRPO+KTAE-7B** | 941 | **563** | **741** | **577** | **771** | **718.6** |
| DAPO-7B | 1155 | 676 | 969 | 700 | 986 | 897.2 |
| **DAPO+KTAE-7B** | 1013 | **604** | **864** | **607** | **798** | **777.2** |

KTAE 在无长度惩罚奖励的情况下显著缩短生成长度（GRPO+KTAE 平均减少52 tokens），实现更高推理效率。

### 消融实验要点

| 配置 | 效果 |
|------|------|
| 去除 IG | 准确率下降最大，生成最短序列 |
| 去除 $\mathcal{F}$ (Fisher) | 准确率下降，长度略增 |
| 去除 tf (频率分析) | 准确率下降，长度增加，初期熵显著上升 |
| GRPO baseline | 出现熵坍缩现象 |
| KTAE 完整版 | 避免熵坍缩，所有组件缺一不可 |

### 关键发现
- IG 对准确率贡献最大，是 KTAE 的核心组件
- KTAE 有效避免了 GRPO 的熵坍缩问题；DAPO+KTAE 的熵值持续上升
- 可视化显示 KTAE 能精准区分"complement"、"ratio"等关键推理 token 和"First"、"denote"等无关 token
- 在某些错误标注的 rollout 中（答案正确但格式解析失败），KTAE 仍能正确高亮正向贡献 token

## 亮点与洞察
- **零额外模型开销的细粒度信号**：不像过程奖励模型 (PRM) 需要额外训练，KTAE 仅通过统计分析即获得 token 级信号，成本极低且不易 reward hacking
- **准确率↑ + 长度↓ 的双赢**：在无长度惩罚的情况下实现更短推理链，说明 token 级优势确实引导模型聚焦关键推理步骤，减少冗余生成
- **统计方法的巧妙组合**：Fisher（精确关联检验）+ IG（信息论互补）+ BM25-style TF（方向判断）三者各司其职，理论可解释性强
- **可迁移思路**：列联表+统计关联的框架不限于数学推理，理论上可推广到代码生成、逻辑推理等有可验证奖励的领域

## 局限与展望
- 当前仅在数学推理任务验证，其他需要 CoT 的领域（代码、逻辑推理）待探索
- KTAE 的计算目前 GPU 利用率不到1%（主要在 CPU 上串行），工程优化空间大
- 对 $G$（rollout 数量）的依赖未充分分析，小 $G$ 时 Fisher 检验的统计 power 可能不足
- 超参数 $h_1, h_2, h_3$ 需要调优，论文未给出敏感性分析
- 仅在 Qwen2.5-Math 系列上实验，跨模型族的泛化性待验证

## 相关工作与启发
- **vs GRPO/DAPO**: 原始 GRPO/DAPO 使用 rollout 级均匀优势，KTAE 通过统计分析将其细化到 token 级，是正交的增强方法
- **vs 过程奖励模型 (PRM)**: PRM 需要额外训练奖励模型，成本高且容易 reward hacking；KTAE 无模型、基于统计，更轻量可控
- **vs DeepSeek R1**: R1 通过纯 RL 诱导长 CoT 和自反思；KTAE 可以作为其更细粒度训练信号的补充

## 评分
- 新颖性: ⭐⭐⭐⭐ 将统计检验引入 RL 的优势估计是新颖的角度，但列联表/Fisher检验本身是经典方法
- 实验充分度: ⭐⭐⭐⭐ 5个基准、1.5B和7B模型、消融+可视化齐全，但缺少跨模型族实验
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，图示直观，但部分符号较密集
- 价值: ⭐⭐⭐⭐ 即插即用的改进方案，实用性强，对 RL 训练效率有实际影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[ACL 2026\] SHAPE: Stage-aware Hierarchical Advantage via Potential Estimation for LLM Reasoning](../../ACL2026/llm_reasoning/shape_stage-aware_hierarchical_advantage_via_potential_estimation_for_llm_reason.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[NeurIPS 2025\] ARM: Adaptive Reasoning Model](arm_adaptive_reasoning_model.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
