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
方法要解决的核心问题是：在策略蒸馏一律用 reverse KL，会在教师不确定的高熵 token 上逼着学生坍缩到单一续写、还把梯度搅得不稳。本文不改变在策略蒸馏的标准 pipeline（学生采样 → 教师打分 → 计算损失 → 梯度更新），只在「算损失」这一步动手：对学生采样轨迹上的每个 token 位置 $t$，先用教师条件分布的熵 $H_t = H(p_T(\cdot|x_{<t}))$ 判断教师此刻有多确定，再据此在 reverse KL 和 forward KL 之间动态调配权重。教师确信时就精确模仿、教师含糊时就保留多样性。

### 关键设计

**1. 教师熵感知的 KL 目标切换：让蒸馏目标随教师的逐 token 不确定性浮动。**

标准在策略蒸馏一刀切地用 reverse KL $D_{\mathrm{KL}}(p_\theta \| p_T)$，它是 mode-seeking 的——把学生的概率质量往教师峰值上挤。这在教师确信（低熵）时没问题，但教师面对多条合理推理路径（高熵）时，就会强行只拟合其中一种，多样性骤降。本文的做法是把"什么时候该 mode-seek、什么时候该 mode-cover"交给教师熵来裁决：在每个位置算出 $H_t$，低熵时仍用 reverse KL 精确贴合教师峰值，高熵时切到 forward KL $D_{\mathrm{KL}}(p_T \| p_\theta)$ 去覆盖教师的多模态分布。两者不是硬开关，而是按熵单调递增的权重 $\lambda_t$ 平滑混合：

$$\mathcal{L} = (1-\lambda_t)\, D_{\mathrm{KL}}^{\mathrm{rev}} + \lambda_t\, D_{\mathrm{KL}}^{\mathrm{fwd}}$$

其中 $\lambda_t$ 随 $H_t$ 增大而增大——教师越不确定，forward KL 的话语权越高。这样既保住了低熵区的精确模仿，又在高熵区把多样性让了出来，正好对症 reverse KL 一刀切忽略「教师不确定性随 token 位置动态变化」这一事实。

**2. 在策略训练的高效整合：熵感知不额外掏采样成本。**

切到 forward KL 的天然麻烦是它通常要从教师分布采样，会凭空多出一轮教师开销。本文绕开这一点：在学生已有的采样轨迹上用重要性加权近似 forward KL 的梯度，不再单独向教师采样。而熵的计算更便宜——教师 logits 在标准在策略蒸馏里本就已经前向算好，要拿到 $H_t$ 只需对这些 logits 多做一次 softmax 归一化，开销可忽略。整套熵感知机制因此完全嵌进原有 pipeline，不引入额外的前向传播或采样步骤，在策略蒸馏的效率优势原封不动。

**3. 多样性保持的梯度稳定化：换目标的同时也换来更稳的梯度。**

高熵 token 上 reverse KL 难训不只是多样性问题，梯度本身就不稳。教师分布扁平时，reverse KL 的梯度 $\nabla_\theta D_{\mathrm{KL}}(p_\theta \| p_T)$ 依赖对数比值 $\log(p_\theta / p_T)$，分母处处接近、比值对参数极敏感，导致梯度方差大、方向漂移。切到 forward KL 后梯度变成 $\nabla_\theta [-\sum p_T \log p_\theta]$，这正是以教师分布为目标的交叉熵——方向被教师分布牢牢锚定，不再随学生自身的扁平输出乱晃，因而显著更稳。这条稳定化是设计 1 的副产物但同样关键：它间接抬高了学生在高熵区的 token 级熵（即生成多样性），并改善 student-teacher 对齐。

### 训练策略
训练流程沿用标准在策略蒸馏：学生采样生成序列，教师计算每个 token 的条件分布和熵，按熵值算出混合 KL 损失，用标准优化器（如 AdamW）更新学生参数。教师模型为 Qwen3-32B，学生模型分别为 Qwen3-0.6B-Base、Qwen3-1.7B-Base 和 Qwen3-4B-Base。

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
- [\[ICLR 2026\] Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems](../../ICLR2026/model_compression/rejuvenating_cross-entropy_loss_in_knowledge_distillation_for_recommender_system.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)

</div>

<!-- RELATED:END -->
