---
title: >-
  [论文解读] Entropy-Aware On-Policy Distillation of Language Models
description: >-
  [ICML 2026][模型压缩][知识蒸馏] 针对在策略蒸馏中 reverse KL 在教师高熵区域引发多样性坍缩和梯度不稳的问题，提出根据教师 token 级熵值自适应混合 forward KL 与 reverse KL 的蒸馏策略，在六个数学推理基准上 Pass@8 最高提升 +5.05。
tags:
  - "ICML 2026"
  - "模型压缩"
  - "知识蒸馏"
  - "在策略蒸馏"
  - "KL散度"
  - "熵感知"
  - "语言模型"
---

# Entropy-Aware On-Policy Distillation of Language Models

**会议**: ICML 2026  
**arXiv**: [2603.07079](https://arxiv.org/abs/2603.07079)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, 在策略蒸馏, KL散度, 熵感知, 语言模型

## 一句话总结
针对在策略蒸馏中 reverse KL 在教师高熵区域引发多样性坍缩和梯度不稳的问题，提出根据教师 token 级熵值自适应混合 forward KL 与 reverse KL 的蒸馏策略，在六个数学推理基准上 Pass@8 最高提升 +5.05。

## 研究背景与动机

**领域现状**：在策略（on-policy）蒸馏是语言模型知识迁移的主流范式——学生模型在自身采样轨迹上，利用教师提供的 dense token-level 信号进行学习。标准做法采用 reverse KL 散度 $D_{\mathrm{KL}}(p_\theta \| p_T)$ 作为训练目标，鼓励学生模型聚焦于教师分布的高置信模式。

**现有痛点**：reverse KL 是 mode-seeking 的，它让学生把概率质量集中在教师分布的峰值上。当教师分布具有高熵——即存在多个合理续写（如数学解题的多条推理路径）——时，reverse KL 会强迫学生只拟合其中一种，导致生成多样性骤降。更严重的是，在高熵区域教师的梯度信号方差大，训练不稳定。

**核心矛盾**：mode-seeking（reverse KL，精确但窄）与 mode-covering（forward KL，全面但散）之间存在根本性 trade-off。现有方法一刀切地选择 reverse KL，忽略了教师输出不确定性随 token 位置动态变化的事实。

**本文目标**：设计一种能在蒸馏过程中自适应感知教师不确定性、在高熵时切换为 forward KL 的蒸馏框架，同时保持在策略训练的效率优势。

**切入角度**：作者观察到教师 token 级熵 $H(p_T(\cdot|x_{<t}))$ 可以作为"什么时候该 mode-seek、什么时候该 mode-cover"的天然指示器。低熵 = 教师确信 → 用 reverse KL 精确模仿；高熵 = 教师不确定 → 用 forward KL 保留多样性。

**核心 idea**：根据教师逐 token 熵值，自适应地将标准 reverse KL 目标增强为 forward KL，在一个统一的在策略框架中同时兼顾精确模仿与多样性保持。

## 方法详解

### 整体框架
输入为预训练好的教师模型 $p_T$ 和待训练的学生模型 $p_\theta$。学生在自身采样的轨迹上，对每个 token 位置计算教师分布的熵，然后根据熵值动态混合 reverse KL 和 forward KL 作为蒸馏损失。整体流程保持在策略蒸馏的标准 pipeline：学生采样 → 教师打分 → 计算混合损失 → 梯度更新。

### 关键设计

1. **教师熵感知的 KL 目标切换**:

    - 功能：根据教师在每个 token 位置的输出熵值，自适应选择蒸馏目标
    - 核心思路：对每个 token 位置 $t$，计算教师条件分布的熵 $H_t = H(p_T(\cdot|x_{<t}))$。当 $H_t$ 低于阈值时使用 reverse KL $D_{\mathrm{KL}}(p_\theta \| p_T)$（mode-seeking，精确拟合教师峰值）；当 $H_t$ 高于阈值时切换为 forward KL $D_{\mathrm{KL}}(p_T \| p_\theta)$（mode-covering，覆盖教师多模态分布）。最终损失为加权混合：$\mathcal{L} = (1-\lambda_t) \cdot D_{\mathrm{KL}}^{\mathrm{rev}} + \lambda_t \cdot D_{\mathrm{KL}}^{\mathrm{fwd}}$，其中权重 $\lambda_t$ 由教师熵单调递增地决定
    - 设计动机：解决 reverse KL 在高熵区域梯度方差大、多样性坍缩的问题，同时在低熵区域保持精确模仿的优势

2. **在策略训练的高效整合**:

    - 功能：在不增加额外采样成本的前提下实现熵感知蒸馏
    - 核心思路：forward KL 通常需要从教师分布采样，但本文在学生轨迹上利用重要性加权近似 forward KL 梯度，避免了额外的教师采样开销。教师的 logits 在标准在策略蒸馏中已经计算完毕，因此熵的计算只需要额外一次 softmax 归一化，开销可忽略
    - 设计动机：保持在策略蒸馏的训练效率优势，不引入额外的前向传播或采样步骤

3. **多样性保持的梯度稳定化**:

    - 功能：稳定高熵区域的梯度信号，防止训练震荡
    - 核心思路：在高熵 token 上，reverse KL 的梯度 $\nabla_\theta D_{\mathrm{KL}}(p_\theta \| p_T)$ 方差大且方向不稳定，因为教师分布扁平时对数比值 $\log(p_\theta / p_T)$ 对参数变化敏感。切换到 forward KL 后，梯度变为 $\nabla_\theta [-\sum p_T \log p_\theta]$，等价于以教师分布为目标的交叉熵，梯度方向由教师分布锚定，显著更稳定
    - 设计动机：通过梯度稳定化间接提升学生模型在高熵区域的 token 级熵（生成多样性），改善 student-teacher 对齐

### 训练策略
训练流程沿用标准在策略蒸馏：学生采样生成序列，教师计算每个 token 的条件分布和熵，根据熵值计算混合 KL 损失，用标准优化器（如 AdamW）更新学生参数。教师模型为 Qwen3-32B，学生模型分别为 Qwen3-0.6B-Base、Qwen3-1.7B-Base 和 Qwen3-4B-Base。

## 实验关键数据

### 主实验

| 学生模型 | 方法 | 6个数学基准 Pass@8 (avg) | vs. 基线 |
|----------|------|-------------------------|----------|
| Qwen3-0.6B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-0.6B-Base | Entropy-Aware (本文) | baseline + 1.37 | **+1.37** |
| Qwen3-1.7B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-1.7B-Base | Entropy-Aware (本文) | baseline + 2.39 | **+2.39** |
| Qwen3-4B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-4B-Base | Entropy-Aware (本文) | baseline + 5.05 | **+5.05** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 纯 Reverse KL | 基线水平 | 标准在策略蒸馏，高熵区域多样性差 |
| 纯 Forward KL | 略低于基线 | 全局 mode-covering 导致低熵区域拟合不精确 |
| 固定混合权重 | 小幅提升 | 不随 token 熵变化的静态混合无法最优适配 |
| 熵感知自适应混合（本文） | 最优 | 动态切换兼得精确性与多样性 |

### 关键发现
- 增益随学生模型规模增大而增大（0.6B: +1.37, 1.7B: +2.39, 4B: +5.05），说明更大的学生模型更能受益于保持高熵区域的多样性
- 在 token 级分析中，本文方法显著维持了学生模型的 token 级熵，避免了生成多样性坍缩
- 在高熵 token 上，学生与教师之间的 forward KL 显著降低，表明 student-teacher 对齐更好
- Pass@8 指标的提升比 Pass@1 更显著，进一步验证了多样性保持的重要性——多条推理路径中至少有一条正确的概率更高

## 亮点与洞察
- 将教师 token 级熵作为 mode-seeking / mode-covering 切换信号，简洁而有效——这个设计几乎不增加计算开销（熵从已有 logits 直接算），却带来显著收益
- 揭示了 reverse KL 在语言模型蒸馏中被忽视的"高熵盲区"问题，为蒸馏目标的选择提供了新视角
- 方法的通用性好：可以作为插件应用到任何在策略蒸馏框架中，不需要修改采样策略或网络结构

## 局限与展望
- 当前仅在数学推理任务上验证，尚未在代码生成、开放域对话等其他多样性要求高的任务上验证泛化性
- 熵阈值的选择依赖于经验调参，自适应阈值选择机制有待探索
- 教师模型为 Qwen3-32B，其他教师-学生架构组合（如跨家族蒸馏）的效果未知
- 未探讨与其他蒸馏增强技巧（如数据增强、课程学习）的组合效果

## 相关工作与启发
本文延续了语言模型知识蒸馏的研究线，与 GKD（Generalized Knowledge Distillation）、MiniLLM 等在策略蒸馏方法形成对比。关键启发在于：蒸馏目标不应该是全局固定的，而应该根据教师的局部不确定性动态调整。这个 insight 可以迁移到强化学习中的 reward shaping（不确定区域降低 reward 权重）和对比学习中的 hard negative mining（根据 anchor 的熵选择负样本策略）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)

</div>

<!-- RELATED:END -->
