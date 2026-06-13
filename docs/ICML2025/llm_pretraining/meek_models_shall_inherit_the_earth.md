---
title: >-
  [论文解读] Meek Models Shall Inherit the Earth
description: >-
  [ICML 2025][预训练][scaling laws] 基于 Chinchilla 缩放定律的数学建模，证明在固定分布的 next-token 目标下，计算缩放的递减收益将导致 SOTA 大模型相对于低计算预算"弱模型"（meek models）的能力优势最终收敛趋零，论证 AI 能力的民主化是当前缩放范式下的必然趋势，现有基于算力的 AI 治理策略需要根本性重新设计。
tags:
  - "ICML 2025"
  - "预训练"
  - "scaling laws"
  - "diminishing returns"
  - "compute scaling"
  - "OCR"
  - "AI governance"
---

# Meek Models Shall Inherit the Earth

**会议**: ICML 2025  
**作者**: Hans Gundlach, Jayson Lynch, Neil Thompson (MIT CSAIL)  
**arXiv**: [2507.07931](https://arxiv.org/abs/2507.07931)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: scaling laws, diminishing returns, compute scaling, AI democratization, AI governance  

## 一句话总结

基于 Chinchilla 缩放定律的数学建模，证明在固定分布的 next-token 目标下，计算缩放的递减收益将导致 SOTA 大模型相对于低计算预算"弱模型"（meek models）的能力优势最终收敛趋零，论证 AI 能力的民主化是当前缩放范式下的必然趋势，现有基于算力的 AI 治理策略需要根本性重新设计。

## 研究背景与动机

**领域现状**：过去十年 AI 系统规模急剧增长——2010-2022 年间训练计算量以每年 5 倍的速度增长，大公司主导了 GPT、LLaMA、Gemini 等 SOTA 模型的训练。主流直觉认为更多计算 = 更强性能 = 更大优势。

**现有痛点**：这种直觉忽视了一个关键问题——缩放的递减收益。Chinchilla 缩放定律表明，损失与计算量的关系为 $L_{opt}(C) = AC^{-\alpha} + L_0$，指数 $\alpha \approx 0.155$ 意味着计算量每增加 10 倍，损失仅减少约 30%。当计算量已经很大时，进一步翻倍的边际收益极小。

**核心矛盾**：大公司的计算投资以指数级增长（约 3.6 倍/年），但所有组织都共享硬件进步（摩尔定律 ~1.4 倍/年）和算法进步（~2.8 倍/年）。这些共享进步对所有人的有效计算量都以指数提升——导致固定预算的"弱模型"也在快速变强。当缩放的递减收益足够显著时，独家投资增长带来的损失优势会在某个拐点后开始缩小。

**切入角度**：构建 SOTA 模型 vs 固定预算 meek 模型之间的训练损失差异模型，推导出损失优势的拐点时间和收敛趋势，并将损失差异转化为可观测的能力差异。

**核心 idea**：在当前 next-token 缩放范式下，递减收益 + 共享的算法/硬件进步 = SOTA 与 meek 模型的能力差距终将收敛。

## 方法详解

### 整体框架

论文构建了三个递进的理论模型：(1) 训练不等式模型——SOTA 与 meek 模型的训练损失差异；(2) 推理不等式模型——固定推理预算下的性能差异；(3) 损失-能力转换——将损失差异映射为基准性能差异和信息论可区分性。最后用经验数据验证模型预测。

### 关键设计

1. **训练损失差异模型**:

    - 功能：量化 SOTA 模型与 meek 模型之间的训练损失差距随时间的变化
    - 核心思路：设 meek 模型固定 $1000 训练预算（约 $10^{17}$ GPU FLOPs），SOTA 模型的计算投资以 $g_i = 3.57$ 倍/年增长。两者都享受算法进步 $g_{alg} = 2.8$ 倍/年和硬件进步 $g_h = 1.4$ 倍/年。基于 Chinchilla 定律，损失差为 $\Delta L = A(g_{alg} \cdot g_h)^{-\alpha t} C_0^{-\alpha} - A(g_{alg} \cdot g_h \cdot g_i)^{-\alpha t} C_0^{-\alpha}$。关键推导：损失优势的拐点时间为 $t^* = \frac{1}{\alpha \ln g_i} \ln\left[\frac{\ln(g_h g_{alg} g_i)}{\ln(g_h g_{alg})}\right]$
    - 设计动机：展示即使 SOTA 的计算投资指数增长，共享的算法/硬件进步的复合效应最终会使边际优势趋零

2. **推理不等式模型**:

    - 功能：分析固定推理预算（如 $10^{-8}$/token）下可运行模型的能力与 SOTA 的差距
    - 核心思路：推理成本受三个因素影响：硬件 FLOPs/$ 增长、参数/FLOP 效率（KV-cache、稀疏注意力等）、有效参数/实际参数（蒸馏、过训练等）。Cottier et al. 2025 数据显示推理成本约 9 倍/年速度下降，这意味着固定预算可运行的有效模型大小以 9 倍/年增长，远快于训练投资增长。因此推理性能差距收敛得更快
    - 设计动机：对大多数用户而言，推理成本比训练成本更相关——如果廉价推理就能获得接近 SOTA 的性能，AI 能力的实质民主化就已实现

3. **损失-能力转换分析**:

    - 功能：论证损失差异的收敛确实意味着有意义的能力收敛
    - 核心思路：从两个角度论证：(a) **Sigmoid 基准映射**——MMLU 等基准性能与损失呈 sigmoid 关系 $Perf = \frac{A}{1+e^{-k(L-x_0)}} + b$，损失差异缩小导致基准性能差异缩小；(b) **假设检验视角**——基于 SPRT（序贯概率比检验），区分两个模型所需的 token 数为 $E[N] = \frac{(1-\alpha)\log\frac{1-\alpha}{\alpha} + \alpha\log\frac{\alpha}{1-\alpha}}{\Delta L}$，当 $\Delta L \to 0$ 时所需 token 数趋于无穷——两个模型变得实际不可区分
    - 设计动机：回应"损失差异缩小不代表实际能力差距缩小"的质疑

### 损失函数

理论分析性工作，使用 Chinchilla 损失公式 $L_{opt}(C) = 1070 \cdot C^{-0.155} + 1.7$ 作为分析基础。

## 实验

### 主实验——模型预测对比

| 对比维度 | SOTA 模型 | Meek 模型 ($1000 预算) | 差异趋势 |
|---------|----------|---------------------|---------|
| 训练损失差异 | 指数级下降 | 较慢下降 | 先扩大后收敛，拐点约 2020s 中期 |
| MMLU 性能差异 | ~80% 稳定 | 快速追赶 | 5 年内差距从 ~20% 缩至 ~5% |
| 推理损失差异 | - | - | 比训练差异收敛更快（推理成本降 9 倍/年） |

### 经验验证——MMLU-Pro 评分差异

| 年份 | 最佳模型 MMLU-Pro | 固定推理预算 ($0.5-1/1M tokens) 最佳 MMLU-Pro | 差距 |
|------|-----------------|---------------------------------------------|------|
| 2023 年中 | ~55 | ~35 | ~20 分 |
| 2024 年初 | ~62 | ~50 | ~12 分 |
| 2024 年末 | ~72 | ~65 | ~7 分 |
| 趋势 | 缓慢增长 | 快速追赶 | 持续缩小 |

### 关键发现

- 训练损失优势存在明确的拐点——在 $C_0 = \$1000$ 基线下，拐点约在 2020s 中期，之后 SOTA 的相对优势持续缩小
- 推理性能差距收敛速度远快于训练——因为推理效率改进（~9 倍/年）远超训练投资增长（~3.6 倍/年）
- Artificial Analysis LLM Leaderboard 的经验数据与模型预测定性一致——固定推理价格范围内的最佳模型 MMLU-Pro 分数正快速追赶无价格约束的最佳模型
- 即使 SOTA 投资增长率变化（3.6→5→10 倍/年），结论定性不变——只是拐点时间推迟
- 多步任务（如需连续正确 p 步）延长了 SOTA 的优势窗口，但最终仍收敛

## 亮点与洞察

- 反直觉但数学严谨的核心结论——更多钱≠持久优势，颠覆"算力为王"的朴素认知
- 分离了三个计算增长因素（投资增长、硬件进步、算法进步），清楚展示为什么共享进步会"淹没"独家投资的优势
- 信息论视角（SPRT 区分代价）提供了损失差异实际意义的精确量化
- AI 治理讨论切中现实痛点——指出基于 FLOP 阈值的监管策略（如美国 >$10^{26}$、欧盟 >$10^{25}$）在能力民主化趋势下可能失效

## 局限性

- 核心假设是固定分布的 next-token 预测——RL、合成数据、自适应训练等新范式可能打破缩放定律的递减收益
- 对抗性场景中微小能力差距可能被放大为巨大优势（如安全竞赛、博弈），论文承认但未深入处理
- 参数数量作为训练计算量的代理指标不够精确（蒸馏、过训练等使关系非线性）
- 经验数据稀疏且主要来自商业 LLM 排行榜，可能受定价策略而非技术水平影响
- 未考虑数据壁垒——独占高质量数据可能提供缩放定律之外的持久优势

## 相关工作与启发

- **Chinchilla 缩放定律** (Hoffmann et al., 2022)：本文的核心工具，$L = AC^{-\alpha} + L_0$ 的幂律形式直接决定了递减收益的强度
- **算法进步** (Ho et al., 2024)：量化了语言模型有效计算每 8 个月翻倍的惊人速率，是 meek 模型快速追赶的关键驱动力
- **Thompson et al., 2021**："深度学习的递减收益"是本文的精神前驱，但本文进一步推导了竞争动态和治理含义
- **启发**：论文虽然开放承认新计算范式（推理缩放、RL、合成数据）可能改变结论，但核心框架为评估未来趋势提供了有用的基准

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统建模 SOTA vs meek 模型的能力收敛趋势 |
| 技术深度 | ⭐⭐⭐⭐ | 缩放定律推导严谨，信息论论证有创意 |
| 实验完整度 | ⭐⭐⭐ | 理论驱动为主，经验验证数据稀疏 |
| 写作质量 | ⭐⭐⭐⭐ | 核心论证逻辑清晰，治理讨论有深度 |
| 实用性 | ⭐⭐⭐⭐⭐ | 对 AI 战略、投资决策、治理政策有直接参考价值 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Nonparametric Modern Hopfield Models](nonparametric_modern_hopfield_models.md)
- [\[ICML 2025\] General Agents Contain World Models](general_agents_contain_world_models.md)
- [\[ICML 2025\] UnHiPPO: Uncertainty-Aware Initialization for State Space Models](unhippo_uncertainty-aware_initialization_for_state_space_models.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](../../ACL2025/others/the_hidden_attention_of_mamba_models.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)

</div>

<!-- RELATED:END -->
