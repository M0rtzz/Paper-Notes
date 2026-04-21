---
title: >-
  [论文解读] Flattening Hierarchies with Policy Bootstrapping
description: >-
  [NeurIPS 2025][图像生成][离线目标条件强化学习] 提出 SAW（Subgoal Advantage-Weighted Policy Bootstrapping），通过在数据集内轨迹上采样子目标并用优势值加权的重要性采样进行策略自举，将层次 RL 的长时序推理优势蒸馏到单一扁平策略中，无需学习子目标生成模型，在 20 个离线 GCRL 数据集上匹配或超越 SOTA。
tags:
  - NeurIPS 2025
  - 图像生成
  - 离线目标条件强化学习
  - 层次强化学习
  - 策略自举
  - 子目标
  - 长时序任务
---

# Flattening Hierarchies with Policy Bootstrapping

**会议**: NeurIPS 2025  
**arXiv**: [2505.14975](https://arxiv.org/abs/2505.14975)  
**代码**: [https://johnlyzhou.github.io/saw/](https://johnlyzhou.github.io/saw/)  
**领域**: 强化学习  
**关键词**: 离线目标条件强化学习, 层次强化学习, 策略自举, 子目标, 长时序任务

## 一句话总结

提出 SAW（Subgoal Advantage-Weighted Policy Bootstrapping），通过在数据集内轨迹上采样子目标并用优势值加权的重要性采样进行策略自举，将层次 RL 的长时序推理优势蒸馏到单一扁平策略中，无需学习子目标生成模型，在 20 个离线 GCRL 数据集上匹配或超越 SOTA。

## 研究背景与动机

**领域现状**：离线目标条件强化学习（offline GCRL）利用无奖励轨迹数据训练能达到任意目标状态的通用策略，被视为类似自监督学习的预训练范式。层次 RL（HRL）方法如 HIQL 通过学习高层策略生成子目标 + 低层策略执行原始动作，在长时序任务上取得 SOTA——尤其是远距离目标任务中一步式方法完全失效的场景。

**现有痛点**：HRL 的模块化架构带来三个根本问题：(1) 需要学习子目标空间的生成模型，当状态空间高维时（如像素观测或 humanoid 的 69 维状态）成为严重瓶颈；(2) 紧凑的子目标表示虽然提高了生成模型的可行性，但限制了策略的表达能力，在大规模环境中性能下降明显；(3) 选择什么表示来学习子目标（自回归预测/度量学习/价值函数中间层等）是一个开放问题，引入大量额外设计复杂度。

**核心矛盾**：HRL 的核心优势是利用"更近的目标更容易学到好的策略"这一事实，而不必然需要层次结构。但如何在不引入生成模型的前提下将子策略的优势传递给扁平策略，是关键挑战。

**本文目标** 隔离 HRL 在离线 GCRL 中成功的核心原因，并将这些优势蒸馏到一个简单的单层策略中，去除层次架构的复杂性。

**切入角度**：作者首先通过实证分析发现 HIQL 成功的两个关键原因——(1) 近距离目标提供更好的优势值信噪比，(2) 数据集中对近距离目标的高质量动作更容易采样。然后将 HRL 重新解释为测试时的隐式策略自举，由此推导出一个不需要生成模型的训练时自举目标。

**核心 idea**：用数据集轨迹中的真实未来状态作为子目标，通过子目标优势值加权的重要性采样来自举扁平目标条件策略，避免学习子目标生成模型。

## 方法详解

### 整体框架

SAW 分三个训练阶段（但共享同一个价值函数）：(1) 训练一个 GCIVL 价值函数 $V(s,g)$；(2) 用 AWR 训练一个目标子策略 $\pi^{sub}(a|s,w)$，只在近距离子目标上训练；(3) 用 SAW 目标训练最终的扁平目标条件策略 $\pi_\theta(a|s,g)$，后者通过优势加权的策略自举从子策略"学习"长时序行为。推理时只用扁平策略，无需高层策略。

### 关键设计

1. **层次 RL 的概率推断视角统一**:

    - 功能：从统一的理论框架推导出 HIQL、RIS 和 SAW 三种方法
    - 核心思路：将层次策略优化建模为最大化子目标最优性变量 $U$ 的似然，引入 ELBO 下界和变分推断。通过选择不同的后验分布分解方式（层次后验 vs 扁平后验）和不同的子目标分布估计方式（参数化生成模型 vs 数据集采样+重要性加权），可以从同一个统一目标函数推导出三种方法。HIQL = 分层后验 + 分层策略优化；RIS = 扁平后验 + 参数化子目标生成器；SAW = 扁平后验 + 数据集采样的优势加权
    - 设计动机：通过统一推导揭示三种方法的本质联系和差异，明确 SAW 的理论基础——它不是一种 ad-hoc 的近似，而是在同一推断框架内的一种特定设计选择

2. **子目标优势加权的策略自举（SAW 目标函数）**:

    - 功能：让扁平策略从近距离子策略处获得长时序推理信号，无需生成模型
    - 核心思路：核心公式为 $\mathcal{J}(\theta) = \mathbb{E}[e^{\alpha A(s,a,g)} \log \pi_\theta(a|s,g) - e^{\beta A(s,w,g)} D_{KL}(\pi_\theta(a|s,g) \| \pi^{sub}(a|s,w))]$。第一项是标准的一步 AWR（直接用价值函数信号），第二项是策略自举项（用子策略作为回归目标）。关键在于子目标 $w$ 直接从数据集轨迹中的未来状态采样（$w \sim p^\mathcal{D}(w|s)$），通过 $e^{\beta A(s,w,g)}$ 加权确保只有高优势的子目标才有较大影响。当目标越远，一步项的信号越弱，自举项自动占主导，提供更稳定的学习目标
    - 设计动机：RIS 需要学习子目标生成模型 $\pi^h(w|s,g)$ 来近似最优子目标分布，在高维空间中很难学好。用贝叶斯规则将期望从 $q^h(w)$ 转换为数据集分布 $p^\mathcal{D}(w|s)$ 的重要性采样形式，完全绕过子目标生成模型

3. **"近距离目标更容易采到好动作"的洞察**:

    - 功能：解释为什么分层训练（用近距离子目标训练低层）比直接用远距离目标训练更有效
    - 核心思路：在离线数据集中，对于近距离的子目标（轨迹中 k 步后的状态），数据集动作的优势值明显更高——因为这些动作本来就是朝着这个方向走的。而对于远距离的目标，最优动作极其稀少，AWR 的加权回归很难从低优势的样本中提取有用信号
    - 设计动机：这个观察解释了 HIQL 相对于简单 AWR 的性能差距中"信噪比改善"之外的另一半原因——不仅是优势值更清晰，而且高质量样本更多，采样效率更高

### 损失函数 / 训练策略

价值函数用 GCIVL（action-free 的 IQL 变体）训练，使用 expectile regression 避免 OOD 动作的价值高估。子策略用标准 AWR 在近距离目标上训练。最终策略用 SAW 目标训练（Eq. 9）。三个阶段依次进行。subgoal 采样距离 $k$ 沿用 HIQL 的设置。

## 实验关键数据

### 主实验

| 环境 | 数据集 | GCIVL | HIQL | SAW |
|------|--------|-------|------|-----|
| antmaze | medium | 72 | 96 | **97** |
| antmaze | large | 16 | 91 | **90** |
| antmaze | giant | 0 | 65 | **73** |
| humanoidmaze | medium | 24 | 89 | **88** |
| humanoidmaze | large | 2 | 49 | 46 |
| humanoidmaze | giant | 0 | 12 | **35** |
| cube-single | play | 53 | 44 | **72** |
| cube-double | play | 36 | 6 | **40** |
| scene | play | 42 | 38 | **63** |

### 消融实验

| 配置 | antmaze-giant | humanoidmaze-giant | cube-single |
|------|--------------|-------------------|-------------|
| GCIVL + AWR | 0% | 0% | 53% |
| GCWAE (改善SNR) | ~16% | — | — |
| HIQL (层次) | 65% | 12% | 44% |
| HIQL w/o 子目标表示 | 65% | — | — |
| SAW w/ 子目标表示 | 下降 | — | — |
| SAW (数据集采样) | **73%** | **35%** | **72%** |

### 关键发现

- **SAW 在最难的长时序任务上远超 HIQL**：humanoidmaze-giant 上 35% vs 12%（将近 3 倍），这是唯一在该环境达到非平凡成功率的方法
- **子目标表示是双刃剑**：HIQL 的子目标表示在中等规模环境有帮助，但在高维大环境中成为瓶颈。SAW 直接在原始观测空间操作避免了这个问题
- **策略自举比仅改善 SNR 更有效**：GCWAE 只改善了优势值的信噪比但仍远不如 HIQL/SAW，说明子策略作为回归目标提供的稳定训练信号同样关键
- 在 pixel-based 任务上，SAW 在 visual-antmaze-large 上大幅超越 HIQL（82% vs 53%），但在 giant 上表现退化，暗示极长时序+高维视觉的价值函数学习仍是开放问题

## 亮点与洞察

- **理论统一视角**：从同一个概率推断框架推导出 HIQL、RIS、SAW 三种方法的本质联系，证明层次 RL 等价于测试时策略自举，而 SAW 是训练时策略自举的自然变体。这为理解 HRL 的优势提供了新的理论视角
- **去除生成模型的关键技巧**：用贝叶斯规则将子目标期望从生成分布转换为数据集分布的重要性采样，看似简单但效果显著——在高维空间中完全绕过了子目标生成这个难题
- **类似分块学习理论**：作者在讨论中将策略自举类比为神经科学中的"分块理论"（chunking theory），即复杂技能先分解为简单片段学习，再融合为流畅整体。这是一个有趣的跨领域类比

## 局限与展望

- SAW 依赖数据集轨迹中的子目标采样，在需要大量轨迹拼接（stitching）的数据集上表现退化
- 像素观测下的极长时序任务（visual-antmaze-giant、visual-humanoidmaze）存在价值函数训练不稳定的问题，这不是 SAW 特有的但限制了其应用
- 子策略的目标采样距离 $k$ 是固定超参数，自适应选择可能进一步提升效果
- 目前的分析仅限于稀疏奖励的目标达成任务，在密集奖励或连续控制任务上的适用性未验证

## 相关工作与启发

- **vs HIQL**：HIQL 是最强的层次基线，共享同一个价值函数但需要额外的高层生成策略和子目标表示。SAW 去除了这两样东西，在简单性和高维可扩展性上优势明显。HIQL 在 humanoidmaze-giant 上只有 12%（有子目标表示）甚至更低（无子目标表示），SAW 达到 35%
- **vs RIS**：RIS 同样做策略自举但需要学习子目标生成模型，离线版本 RISoff 性能波动大（有些环境很好但方差极大）。SAW 用数据集采样+重要性加权替代生成模型，更稳定
- **vs QRL/CRL**：这些方法用对比学习或准度量结构来改善价值函数，在某些环境很强但缺乏长时序推理能力。SAW 通过子策略自举补充了长时序信号

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 理论推导统一了三种方法，去除生成模型的核心思路简洁优美
- 实验充分度: ⭐⭐⭐⭐⭐ 20 个数据集/100 个评测任务/8 个 seed，state+pixel，覆盖极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰：问题分析→洞察提炼→理论推导→方法设计→全面实验
- 价值: ⭐⭐⭐⭐⭐ 对离线 GCRL 和层次 RL 领域都有重要贡献，方法简洁实用

<!-- RELATED:START -->

## 相关论文

- [RLZero: Direct Policy Inference from Language Without In-Domain Supervision](rlzero_direct_policy_inference_from_language_without_in-domain_supervision.md)
- [Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations](../../ICML2025/image_generation/video_prediction_policy_a_generalist_robot_policy_with_predictive_visual_represe.md)
- [FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](freqpolicy_efficient_flow-based_visuomotor_policy_via_frequency_consistency.md)
- [Theoretical Guarantees on the Best-of-n Alignment Policy](../../ICML2025/image_generation/theoretical_guarantees_on_the_best-of-n_alignment_policy.md)

<!-- RELATED:END -->
