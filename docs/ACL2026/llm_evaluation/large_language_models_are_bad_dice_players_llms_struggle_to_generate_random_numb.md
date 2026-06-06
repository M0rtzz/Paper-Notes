---
title: >-
  [论文解读] Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions
description: >-
  [ACL 2026][LLM评测][概率采样] 本文首次大规模系统审计了 11 个前沿 LLM 在 15 种概率分布上的原生采样能力，发现 LLM 严重缺乏内在概率采样机制，且这种缺陷会传导到下游应用中造成系统性偏差。
tags:
  - "ACL 2026"
  - "LLM评测"
  - "概率采样"
  - "随机数生成"
  - "分布保真度"
  - "LLM内在能力"
  - "下游偏差"
---

# Large Language Models Are Bad Dice Players: LLMs Struggle to Generate Random Numbers from Statistical Distributions

**会议**: ACL 2026  
**arXiv**: [2601.05414](https://arxiv.org/abs/2601.05414)  
**代码**: [GitHub](https://github.com/Mininda/LLM_Bad_Dice_Player)  
**领域**: 图像生成  
**关键词**: 概率采样, 随机数生成, 分布保真度, LLM内在能力, 下游偏差

## 一句话总结

本文首次大规模系统审计了 11 个前沿 LLM 在 15 种概率分布上的原生采样能力，发现 LLM 严重缺乏内在概率采样机制，且这种缺陷会传导到下游应用中造成系统性偏差。

## 研究背景与动机

**领域现状**：LLM 正从对话接口演变为复杂应用流水线的核心组件，包括合成数据生成、Agent 模拟、教育材料构建和文本到图像提示合成。这些应用场景日益要求 LLM 能从指定的概率分布中忠实采样。

**现有痛点**：已有研究零星发现 LLM 在简单随机生成任务中存在偏差——如偏好"幸运数字"、硬币翻转偏差等。但这些研究局限于小样本（$N=100$）、少量分布（5种）和单一采样协议，无法全面评估 LLM 的原生采样能力。当前实践中，生成符合分布要求的数据需依赖外部库（如 numpy.random），这一"变通方案"暗示 LLM 缺乏底层功能能力。

**核心矛盾**：如果追求通用智能的 LLM 连基本的概率分布都无法忠实采样，那么其在需要统计保证的下游应用中将引入不可控的系统性偏差。但缺乏大规模、统计有效的基准来验证这一假设。

**本文目标**：对前沿 LLM 进行首次大规模概率采样审计。**切入角度**：设计双协议实验框架（批量 vs 独立），解耦不同失败模式。**核心 idea**：LLM 缺乏功能性内部采样器，批量生成靠上下文依赖勉强工作，独立请求下几乎完全失败。

## 方法详解

### 整体框架

评估流水线覆盖 11 个前沿 LLM × 15 种概率分布（分三个复杂度层级）× 2 种采样协议。使用 Wasserstein-1 距离 $\mathcal{W}_1$、KL 散度和统计检验（KS/χ²）三重指标量化分布保真度。同时设计两个下游应用实验验证采样缺陷的传导效应。

### 关键设计

1. **双协议实验设计**：

    - 功能：解耦 LLM 采样中上下文依赖和内在先验两种失败模式
    - 核心思路：Protocol A（批量生成）在单次回复中生成 $N=1000$ 个样本，模型可利用历史上下文自我校正；Protocol B（独立请求）通过 $N=1000$ 次无状态调用各生成 1 个样本，隔离模型的内在先验
    - 设计动机：先前研究仅用批量协议，无法判断 LLM 是否具有真正的独立采样能力

2. **Context-Fidelity 困境理论分析**：

    - 功能：刻画采样预算 $N$ 与分布保真度之间的非单调关系
    - 核心思路：独立请求下，期望误差 $\mathcal{E}(N) = \Delta_{\text{ind}} + \mathcal{O}(N^{-1/2})$ 收敛到不可约偏差 $\Delta_{\text{ind}}$；批量生成下，误差分解为 Correction Gain（上下文自校正带来的改善）和 Drift（自回归漂移带来的退化），超过临界长度后漂移主导
    - 设计动机：解释为何更大的 $N$ 反而导致更差的分布拟合

3. **三层分布复杂度分类**：

    - 功能：按分布的熵特征、支撑约束和尾部行为组织 15 种分布为三个层级
    - 核心思路：Tier I（基本分布：均匀、高斯、伯努利）、Tier II（有界/离散：Beta、二项、泊松、指数）、Tier III（重尾/多参数：柯西、t、卡方、F、Gamma、Weibull、Laplace、Logistic）
    - 设计动机：系统评估采样保真度是否随分布复杂度单调退化

### 损失函数 / 训练策略

本文为评估型工作，不涉及模型训练。使用统计检验作为二元诊断（KS/χ²，$\alpha=0.01$），$\mathcal{W}_1$ 作为连续保真度度量，KL 散度作为信息损失度量。

## 实验关键数据

### 主实验

| 协议 | 中位 Pass Rate | 最佳模型 | 最佳 Pass Rate |
|------|---------------|---------|---------------|
| 批量生成 | 7% | GPT-4o | 40% |
| 独立请求 | 0% | Llama-4-Scout | 7% (仅通过Bernoulli) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Tier I 分布 | Pass Rate 最高 | 基本分布相对可控 |
| Tier III 分布 | Pass Rate ~0% | 复杂分布完全失败 |
| $N=50 \to 2000$ | $\mathcal{W}_1$ 单调增加 | 更多样本反而暴露更大偏差 |
| MCQ 位置偏差 | 所有模型 $p < 0.001$ | GPT-OSS: C位 54.6%, A位仅 4.5% |
| 属性约束提示 | 性别/种族严重偏离目标 | GPT-4o: 亚裔 33.5% vs 目标 6.5% |

### 关键发现
- 批量生成和独立请求之间存在**尖锐的协议不对称**：有效采样依赖长上下文依赖而非内在能力
- 采样保真度随分布复杂度单调退化，$\mathcal{W}_1$ 从 Tier I 的 ~0.1 上升到 Tier III 的 ~1.5
- 更大的采样预算 $N$ 导致更差的分布拟合，违反标准 $\mathcal{O}(N^{-1/2})$ 收敛预期
- 在下游应用中，LLM 无法遵守显式的分布约束指令，MCQ 位置偏差和人口属性偏差普遍且严重

## 亮点与洞察
- 双协议设计是关键方法论创新，将采样问题从"是否能近似"细化为"依赖上下文还是内在能力"
- Context-Fidelity 困境的理论框架优雅地解释了批量生成中的非单调行为
- 下游应用实验（MCQ、文生图提示）将抽象的采样问题与实际应用风险直接关联，增强了论文的影响力
- 结论明确：当前 LLM 缺乏功能性内部采样器，需要外部工具提供统计保证
- 三层分布复杂度分类法提供了一个可复用的评估框架，未来可扩展到更多分布族

## 局限与展望
- 仅测试默认解码参数（T=1.0, top-p=1.0），不同解码策略（如低温度、top-k）可能有不同表现
- 未探索思维链或代码生成等增强策略能否改善采样能力，这些是实际应用中的常用变通方案
- 仅覆盖 1D 分布，多变量联合分布的情况更为复杂（附录中报告了二元高斯的初步实验）
- 未来可研究是否通过专门的微调、RLHF 或特殊训练目标让 LLM 获得内在采样能力
- 下游实验的规模和场景还可进一步扩展到更多实际应用领域

## 相关工作与启发
- **vs Gu et al. (2024)**: 先前最全面的研究仅5种分布 × $N=100$ × 单协议，本文大幅扩展规模和方法论深度
- **vs Hopkins et al. (2023)**: 发现"幸运数字"偏好但限于均匀分布整数，本文系统覆盖15种连续和离散分布
- **vs Xiao et al. (2025)**: 揭示 Bernoulli 采样偏差，本文扩展至连续分布并增加下游应用验证

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次大规模系统审计 LLM 原生概率采样，双协议设计有方法论价值
- 实验充分度: ⭐⭐⭐⭐⭐ 11 模型 × 15 分布 × 2 协议 × 下游应用，覆盖极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论框架清晰优雅，数据呈现直观，结论有说服力
- 价值: ⭐⭐⭐⭐ 揭示 LLM 的基础能力缺陷，对依赖 LLM 采样的应用具有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)

</div>

<!-- RELATED:END -->
