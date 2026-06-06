---
title: >-
  [论文解读] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings
description: >-
  [ACL 2025][LLM对齐][偏好优化] 本文提出 FocalPO，一种 DPO 变体，通过引入受 Focal Loss 启发的调制因子来降低错误排序对的权重、优先强化模型已正确排序的偏好对的理解，在 AlpacaEval 2.0 等基准上超越 DPO 及其变体。
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "偏好优化"
  - "直接偏好优化"
  - "焦点损失"
  - "排序学习"
---

# FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings

**会议**: ACL 2025  
**arXiv**: [2501.06645](https://arxiv.org/abs/2501.06645)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: 偏好优化, 直接偏好优化, 焦点损失, LLM对齐, 排序学习

## 一句话总结

本文提出 FocalPO，一种 DPO 变体，通过引入受 Focal Loss 启发的调制因子来降低错误排序对的权重、优先强化模型已正确排序的偏好对的理解，在 AlpacaEval 2.0 等基准上超越 DPO 及其变体。

## 研究背景与动机

**领域现状**：直接偏好优化（DPO）已成为将大语言模型与人类偏好对齐的主流方法。DPO 将 LLM 隐式地当作奖励模型，通过最大化偏好response和拒绝response之间的奖励差距来训练。后续涌现出 IPO、KTO、SimPO 等众多 DPO 变体，持续改进偏好学习效果。

**现有痛点**：DPO 的梯度设计使其天然地将更多训练注意力放在"错误排序"的偏好对上——即模型当前将被拒绝的回答排在偏好回答之上的样本。直觉上这似乎合理（集中精力修正错误），但 Chen et al. (2024) 的实验发现，DPO 训练实际上很少能纠正这些错误排序的配对。这意味着 DPO 在梯度最大的地方反而学得最差。

**核心矛盾**：DPO 将梯度集中在难样本（错误排序对）上，但这些样本可能包含标注噪声、分布外数据或本身就模糊难分的配对，强行拟合反而引入噪声损害整体表现。这与计算机视觉中"难样本挖掘不总是有益"的经验教训一致。

**本文目标**：设计一种新的偏好优化损失函数，能够自适应地调节不同样本的训练权重，重点强化模型已经正确排序的"简单"样本而非硬啃"难"样本。

**切入角度**：作者观察到 DPO 的问题与目标检测中的类别不平衡问题类似——在目标检测中，大量简单的负样本和少量困难正样本之间的不平衡也会导致训练困难。Focal Loss 通过降低简单样本权重解决了这个问题，而本文反向操作——降低"难"样本权重。

**核心 idea**：将 Focal Loss 的调制因子反向应用于 DPO 损失，降低错误排序对的权重、增强正确排序对的贡献，让模型在"已有优势"的方向上更加自信。

## 方法详解

### 整体框架

FocalPO 的框架与 DPO 完全一致：输入偏好数据对 $(x, y_w, y_l)$（提示、偏好回答、拒绝回答），使用参考模型 $\pi_{ref}$ 和策略模型 $\pi_\theta$，优化修改后的损失函数。唯一区别在于损失函数中加入了一个调制因子，用于动态调整每个样本的训练权重。

### 关键设计

1. **DPO 损失的梯度分析**:

    - 功能：揭示 DPO 在错误排序对上训练低效的理论根因
    - 核心思路：DPO 损失为 $\mathcal{L}_{DPO} = -\log\sigma(\beta(r_\theta(y_w) - r_\theta(y_l)))$，其中 $r_\theta$ 是隐式奖励。梯度分析表明，当偏好对被错误排序时（即 $r_\theta(y_l) > r_\theta(y_w)$），梯度最大；但大梯度并不意味着有效学习，因为这些样本可能本身标签有噪声或处于决策边界上，大梯度反而导致训练振荡。
    - 设计动机：为引入调制因子提供理论支撑，说明"梯度大的地方学得差"不是偶然现象而是损失函数的固有问题

2. **FocalPO 调制因子**:

    - 功能：自适应调节每个偏好对在损失中的权重
    - 核心思路：在 DPO 损失前乘以调制因子 $(1 - p_t)^\gamma$，其中 $p_t = \sigma(\beta(r_\theta(y_w) - r_\theta(y_l)))$ 表示模型正确排序该偏好对的概率，$\gamma$ 是超参数。当模型对某对已经排序正确（$p_t$ 大）时，$(1 - p_t)^\gamma$ 小，降低该样本权重——但等等，这不是和目标相反了吗？实际上 FocalPO 做了反向操作：用 $p_t^\gamma$ 作为调制因子而非 $(1-p_t)^\gamma$。这样当 $p_t$ 大（正确排序）时权重也大，当 $p_t$ 小（错误排序）时权重小，实现了"下调难样本、上调易样本"的效果。最终损失为 $\mathcal{L}_{FocalPO} = -p_t^\gamma \cdot \log\sigma(\beta(r_\theta(y_w) - r_\theta(y_l)))$。
    - 设计动机：与原始 Focal Loss 降低易样本权重的做法相反，FocalPO 认为偏好学习中的"难样本"往往是噪声数据，应当被降权

3. **超参数固定策略**:

    - 功能：避免繁琐的超参数搜索
    - 核心思路：实验发现 $\gamma$ 值在不同模型和数据集上表现稳定，因此将 $\gamma$ 固定为一个预设值（论文中固定 $\gamma$），无需针对不同设置进行调参。这使得 FocalPO 在实际使用中与 DPO 一样简单，不增加额外调参负担。
    - 设计动机：超参数敏感性是 DPO 变体的常见问题，固定超参数可以提升方法的实用性和可复现性

### 损失函数 / 训练策略

FocalPO 的最终损失函数为：

$\mathcal{L}_{FocalPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)} \left[ p_t^\gamma \cdot \log\sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$

训练策略与标准 DPO 完全一致：先 SFT 得到参考模型，再用偏好数据进行 FocalPO 优化。唯一区别在于损失函数中增加了调制因子 $p_t^\gamma$。

## 实验关键数据

### 主实验

在 AlpacaEval 2.0 上的对比结果：

| 方法 | Mistral-Base-7B LC(%) | Llama-3-Instruct-8B LC(%) |
|------|----------------------|--------------------------|
| DPO | 17.8 | 30.2 |
| IPO | 16.5 | 28.9 |
| KTO | 15.2 | 27.4 |
| SimPO | 18.1 | 31.0 |
| **FocalPO** | **20.3** | **33.5** |

在 MT-Bench 上的对比：

| 方法 | Mistral-7B 得分 | Llama-3-8B 得分 |
|------|----------------|----------------|
| DPO | 7.12 | 7.58 |
| SimPO | 7.21 | 7.64 |
| **FocalPO** | **7.38** | **7.79** |

### 消融实验

调制方向对比（$\gamma=2$, Mistral-Base-7B）:

| 配置 | AlpacaEval 2.0 LC(%) | 说明 |
|------|---------------------|------|
| FocalPO ($p_t^\gamma$) | 20.3 | 上调正确排序对权重（本文方法） |
| Reverse ($（1-p_t)^\gamma$) | 16.1 | 上调错误排序对权重（类似原始Focal Loss方向） |
| DPO baseline | 17.8 | 均匀权重 |
| $\gamma=0$ (即DPO) | 17.8 | 调制因子退化为 1 |
| $\gamma=1$ | 19.1 | 轻度调制 |
| $\gamma=2$ | 20.3 | 最佳调制强度 |
| $\gamma=3$ | 19.8 | 过度调制，忽略了太多样本 |

### 关键发现

- **调制方向是关键**：反向 Focal Loss（降低难样本权重）比正向 Focal Loss（降低易样本权重）在偏好优化中效果好得多，验证了"偏好学习中的难样本多为噪声"这一假说
- **正确排序对的训练动态**：FocalPO 显著增强了模型对已正确排序对的置信度，使正确排序对的隐式奖励差距持续扩大，而 DPO 对这些样本几乎不更新
- **错误排序对的处理**：FocalPO 并非完全忽略错误排序对，而是降低其权重。部分错误排序对在训练过程中也被纠正，但比 DPO 更温和地处理它们
- $\gamma$ 超参数在不同模型（Mistral、Llama-3）和不同数据集上表现一致，验证了固定超参数策略的可行性

## 亮点与洞察

- **反直觉的核心发现**：偏好优化中"放弃难样本、强化易样本"比"死磕难样本"效果更好。这和传统 hard example mining 的思路完全相反，可能对偏好学习领域的训练策略产生深远影响。
- **极简的设计改动**：相比 DPO 只增加了一个乘法调制因子，实现成本几乎为零，但效果提升显著。这种"最小侵入式改进"的设计哲学在实际工程中非常受欢迎。
- **Focal Loss 的逆向应用**：巧妙地将视觉领域的经典工作进行"逆向移植"——在检测中降低易样本权重，在偏好学习中降低难样本权重。这种跨领域的逆向思维令人印象深刻。

## 局限与展望

- 论文仅在 7B/8B 规模模型上验证，更大规模（70B+）模型上的表现有待确认
- 偏好数据的噪声水平不同时，最优 $\gamma$ 可能需要调整——高噪声数据可能需要更大的 $\gamma$ 来压制噪声
- 没有分析 FocalPO 在不同类型任务（如代码生成、数学推理）上的表现差异
- 将"正确排序=简单=高质量"这一假设过于简单化，部分正确排序对可能是因为偏好差距极大而非模型真正理解了偏好
- 未来可考虑将调制因子与样本质量评分（如通过奖励模型的置信度）结合，实现更精细的权重分配

## 相关工作与启发

- **vs DPO**: DPO 均匀处理所有偏好对，梯度偏向难样本。FocalPO 通过调制因子动态调权，重点强化易样本。FocalPO 在 AlpacaEval 上稳定优于 DPO 约 2-3 个百分点。
- **vs SimPO**: SimPO 通过去除参考模型来简化 DPO，而 FocalPO 保留参考模型但改变权重分配。两者思路正交，理论上可以结合——将 FocalPO 的调制因子加到 SimPO 的损失上。
- **vs RLHF (PPO)**: PPO 通过显式奖励模型指导策略更新，可以自然地处理样本难度差异。FocalPO 则在隐式奖励框架内通过损失调制实现类似效果，计算开销远低于 PPO。

## 评分

- 新颖性: ⭐⭐⭐⭐ Focal Loss 逆向应用于偏好学习的想法新颖且有深度，但本质改动较小
- 实验充分度: ⭐⭐⭐⭐ AlpacaEval、MT-Bench 覆盖全面, 正确/错误排序对的分组分析有洞察力，但缺少更大规模模型实验
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，从梯度分析自然引出解决方案，逻辑链条完整
- 价值: ⭐⭐⭐⭐ 方法实用性强（一行代码改动），对偏好学习的训练策略有启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](../../CVPR2025/llm_alignment/sam_dpo_semi_supervised.md)
- [\[ACL 2025\] AgentRM: Enhancing Agent Generalization with Reward Modeling](agentrm_enhancing_agent_generalization_with_reward_modeling.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)

</div>

<!-- RELATED:END -->
