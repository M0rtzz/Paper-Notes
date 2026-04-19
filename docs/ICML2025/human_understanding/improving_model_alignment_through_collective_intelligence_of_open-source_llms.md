---
title: >-
  [论文解读] Improving Model Alignment through Collective Intelligence of Open-Source LLMs
description: >-
  [ICML 2025][人体理解][model alignment] 本文提出 Mixture of Agents Alignment（MoAA），利用多个开源 LLM 的集体智慧生成高质量的对齐数据（SFT 数据和偏好数据），显著提升目标模型在 Arena-Hard 和 AlpacaEval2 上的表现，并展示了无需外部强监督的自我提升能力。
tags:
  - ICML 2025
  - 人体理解
  - model alignment
  - mixture of agents
  - 合成数据
  - 偏好优化
  - 自我改进
---

# Improving Model Alignment through Collective Intelligence of Open-Source LLMs

**会议**: ICML 2025  
**arXiv**: [2505.03059](https://arxiv.org/abs/2505.03059)  
**代码**: 即将发布  
**领域**: Human Understanding  
**关键词**: model alignment, mixture of agents, 合成数据, 偏好优化, 自我改进

## 一句话总结
本文提出 Mixture of Agents Alignment（MoAA），利用多个开源 LLM 的集体智慧生成高质量的对齐数据（SFT 数据和偏好数据），显著提升目标模型在 Arena-Hard 和 AlpacaEval2 上的表现，并展示了无需外部强监督的自我提升能力。

## 研究背景与动机

**领域现状**: LLM 的对齐（alignment）——使模型输出有帮助（helpful）且无害（harmless）——依赖高质量的人工标注数据，用于监督微调（SFT）和偏好优化（DPO/RLHF）。

**现有痛点**: 人工标注数据昂贵、难以大规模生产，且可能存在多样性不足和标注者偏见的问题。已有的合成数据方法（如用 GPT-4 生成对齐数据）依赖单一强模型，生成数据的多样性受限，且形成了对闭源模型的依赖。

**核心矛盾**: 如何扩大对齐数据的规模和多样性，同时减少对单一强模型的依赖？开源模型的单体能力可能不如 GPT-4，但其集体智慧能否弥补这一差距？

**本文目标**: 利用多个开源 LLM 的协作来生成高质量的对齐数据。

**切入角度**: Mixture of Agents（MoA）思想——多个 LLM 各自生成回答，然后由聚合器综合不同回答的优点，产出比任何单一模型更好的最终回答。

**核心 idea**: 将 MoA 框架应用于对齐数据生成的两个阶段：(1) SFT 数据生成——多模型协作产出高质量指令-回答对，(2) 偏好数据生成——利用多模型的输出差异自然构造正负样本对。

## 方法详解

### 整体框架
输入：一组开源 LLM（如 LLaMA-3.1-70B, Qwen2-72B, Mixtral-8x22B 等），目标对齐模型（如 LLaMA-3.1-8B-Instruct）
输出：对齐后性能大幅提升的目标模型

Pipeline:
1. **MoA-SFT**: 用多模型协作生成 SFT 训练数据 → 微调目标模型
2. **MoA-DPO**: 用多模型输出构造偏好对 → 偏好优化

### 关键设计

1. **MoA 响应生成（MoA Response Generation）**:

    - 功能：对每个指令 prompt，让多个 LLM 各自生成回答，然后用聚合模型综合为最终回答
    - 核心思路：第一层——$K$ 个 LLM 各自生成回答 $\{r_1, \ldots, r_K\}$；第二层——聚合器模型收到所有回答和原始指令，生成综合回答 $r^*$
    - 设计动机：不同模型有不同的知识和"个性"（如有的擅长推理，有的擅长写作），MoA 可以集成这些互补优势

2. **MoA-SFT 数据构造**:

    - 功能：用 MoA 生成的高质量回答作为 SFT 训练目标
    - 核心思路：$(prompt, r^*_{\text{MoA}})$ 作为训练对。MoA 回答质量优于任何单一模型，因此微调后的模型可以超越其训练数据来源
    - 设计动机：替代 GPT-4 标注，同时提供更高的多样性

3. **MoA-DPO 偏好数据构造**:

    - 功能：利用多模型输出的质量差异构造偏好对
    - 核心思路：MoA 综合回答 $r^*$ 作为 chosen（正样本），各单模型的回答中较差的 $r_{\text{worst}}$ 作为 rejected（负样本）
    - 设计动机：不需要外部评估器（如人类或 GPT-4），偏好信号来自模型集合的内部比较

4. **自我提升 Pipeline（Self-Improvement Pipeline）**:

    - 功能：用 MoAA 微调后的模型作为下一轮的 MoA 参与者
    - 核心思路：迭代 $t$: 用当前模型参与 MoA → 生成更好的训练数据 → 微调 → 新模型再参与 MoA
    - 设计动机：这形成了一个正反馈循环——模型能力提升 → 生成更好数据 → 进一步提升

### 损失函数 / 训练策略
- SFT 阶段：标准的 next-token cross-entropy loss
- DPO 阶段：$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(r_w|x)}{\pi_{\text{ref}}(r_w|x)} - \beta \log \frac{\pi_\theta(r_l|x)}{\pi_{\text{ref}}(r_l|x)}\right)$

## 实验关键数据

### 主实验
| 模型 | 指标 | MoAA | GPT-4o 蒸馏 | 自身数据 | 基线 (无对齐) |
|------|------|------|------------|---------|-------------|
| LLaMA-3.1-8B-Instruct → Arena-Hard | Win Rate | **48.3** | 42.1 | 31.5 | 19.5 |
| LLaMA-3.1-8B-Instruct → AlpacaEval2 | Win Rate | **57.23** | 49.8 | 35.4 | 22.33 |
| LLaMA-3.1-8B-Instruct → MT-Bench | 平均分 | **8.12** | 7.85 | 7.21 | 6.58 |

### 消融实验
| 配置 | Arena-Hard WR | AlpacaEval2 WR | 说明 |
|------|-------------|----------------|------|
| MoAA (SFT + DPO) | **48.3** | **57.23** | 完整方法 |
| 仅 MoA-SFT | 39.7 | 45.6 | DPO 贡献约 8-12 WR |
| 仅 MoA-DPO | 35.2 | 41.8 | SFT 基础重要 |
| 单模型 (GPT-4o) SFT | 42.1 | 49.8 | MoA 优于单一强模型 |
| 单模型 (LLaMA-70B) SFT | 33.8 | 38.2 | 单个开源模型不够 |
| 自我提升 (2 轮) | **51.2** | **60.1** | 正反馈循环有效 |

### 关键发现
- MoAA 使 LLaMA-3.1-8B 的 Arena-Hard Win Rate 从 19.5 提升至 48.3（+28.8）
- 多开源模型协作生成的数据质量超过 GPT-4o 单独生成的数据
- 自我提升是可行的——第 2 轮迭代进一步提升 3-4 WR
- MoA-SFT 和 MoA-DPO 互补，缺一不可
- 参与 MoA 的模型越多样（不同家族），效果越好

## 亮点与洞察
- **实用价值高**: 完全基于开源模型，无需依赖 GPT-4
- **自我提升**: 展示了开源 LLM 生态通过协作突破个体能力上限的可能性
- **方法简洁**: MoA + SFT + DPO 的流程简单直接，易于复现

## 局限与展望
- 自我提升是否会遇到"天花板效应"（模型集合无法提供超越自身的信号）有待长期验证
- MoA 的计算开销是参与模型数的线性倍——需要同时运行 3-5 个 70B 模型
- 安全性（harmlessness）方面的评估不够充分
- 未讨论 MoAA 生成数据的质量控制和过滤策略

## 相关工作与启发
- Mixture of Agents (Wang et al., 2024): MoA 的原始工作
- Self-Play Fine-Tuning (Chen et al., 2024): 另一种自我提升方法
- 本文证明了"集体智慧 > 个体能力"在 LLM 对齐中的可行性

## 评分
- 新颖性: ⭐⭐⭐⭐ MoA+对齐的组合新颖，但各组件（MoA, SFT, DPO）已知
- 实验充分度: ⭐⭐⭐⭐⭐ 多个顶级基准测试，消融全面，自我提升验证
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，实验结果令人信服
- 价值: ⭐⭐⭐⭐⭐ 对开源 LLM 生态发展有重要推动

<!-- RELATED:START -->

## 相关论文

- [If Open Source Is to Win, It Must Go Public](if_open_source_is_to_win_it_must_go_public.md)
- [Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study](../../AAAI2026/human_understanding/why_do_open-source_llms_struggle_with_data_analysis_a_systematic_empirical_study.md)
- [Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [Truly Self-Improving Agents Require Intrinsic Metacognitive Learning](truly_self-improving_agents_require_intrinsic_metacognitive_learning.md)
- [RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning](rulebreakers_challenging_llms_at_the_crossroads_between_formal_logic_and_human-l.md)

<!-- RELATED:END -->
