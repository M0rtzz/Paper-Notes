---
title: >-
  [论文解读] Preference Optimization for Combinatorial Optimization Problems
description: >-
  [ICML 2025][LLM对齐][偏好优化] 将RLHF中的偏好优化思想引入组合优化（COP），把定量奖励信号转化为定性偏好信号，结合熵正则化目标和局部搜索微调，在TSP/CVRP/FFSP等标准基准上实现了1.5x-2.5x的收敛加速和更优解质量。
tags:
  - ICML 2025
  - LLM对齐
  - 偏好优化
  - 组合优化
  - REINFORCE
  - 熵正则化
  - 局部搜索
---

# Preference Optimization for Combinatorial Optimization Problems

**会议**: ICML 2025  
**arXiv**: [2505.08735](https://arxiv.org/abs/2505.08735)  
**代码**: 无  
**领域**: 强化学习 / 组合优化  
**关键词**: 偏好优化, 组合优化, REINFORCE, 熵正则化, 局部搜索

## 一句话总结

将RLHF中的偏好优化思想引入组合优化（COP），把定量奖励信号转化为定性偏好信号，结合熵正则化目标和局部搜索微调，在TSP/CVRP/FFSP等标准基准上实现了1.5x-2.5x的收敛加速和更优解质量。

## 研究背景与动机

**领域现状**: 强化学习（RL）已成为神经组合优化（RL4CO）的主流范式，REINFORCE及其变体被广泛用于训练端到端神经求解器（AM、POMO、Sym-NCO等），无需专家标注即可从环境交互中学习启发式策略。

**现有痛点**: 
   - **奖励信号衰减**: 随着策略改进，采样解之间的奖励差 |r(x,τ)-b(x)| 趋近于零，导致梯度消失、收敛缓慢
   - **探索效率低**: 组合动作空间呈指数增长，传统轨迹级熵正则化计算不可行（需要枚举所有轨迹）
   - **额外推理开销**: 许多方法将局部搜索（Local Search）作为后处理步骤提高解质量，但引入额外推理时间

**核心矛盾**: REINFORCE依赖数值奖励差驱动学习，但COP中后期该差值极小；同时轨迹空间太大，无法有效正则化探索

**本文目标**: 设计一种不依赖奖励绝对数值、自带探索能力、能无缝集成局部搜索的通用RL优化框架

**切入角度**: 借鉴RLHF中DPO的思想——将定量奖励转化为定性偏好比较，通过重参数化避免不可行计算

**核心 idea**: 用偏好比较代替绝对奖励值，通过熵正则化目标的解析解重参数化奖励函数，使策略优化转化为分类问题

## 方法详解

### 整体框架

PO（Preference Optimization）框架包含三个核心模块：(1) 偏好比较模块——对同一问题实例的多个采样解进行成对比较，生成无冲突偏好标签；(2) 策略优化模块——基于重参数化的熵正则化目标，用偏好分类损失直接更新策略；(3) 局部搜索微调模块——在策略收敛后，用局部搜索生成高质量偏好对，进一步突破局部最优。

### 关键设计

1. **偏好信号生成（Preference Comparison）**: 

    - **功能**: 把定量奖励r(x,τ)转化为定性偏好标签y=𝟙(r(x,τ₁)>r(x,τ₂))
    - **核心思路**: 对每个问题实例x采样N个解{τ₁,...,τ_N}，用ground truth奖励函数（如TSP路径长度）做成对比较，生成偏好对
    - **设计动机**: 偏好标签对奖励的仿射变换具有不变性——𝟙(k·r+b > k·r'+b) = 𝟙(r > r')，这意味着无论奖励如何缩放或平移，偏好关系始终保持，从根本上避免了奖励信号衰减问题。同时由于COP的奖励函数是客观物理量度（如路径长度），生成的偏好标签天然无冲突，具有传递性

2. **重参数化熵正则化目标（Reparameterized Entropy-Regularized Objective）**:

    - **功能**: 将最大熵RL目标的解析最优策略代入偏好模型，消除不可行的分区函数Z(x)
    - **核心思路**: 
        - 出发点是最大熵RL目标: max E[r(x,τ)] + αℋ(π)
        - 其解析最优策略为: π*(τ|x) = (1/Z(x))·exp(α⁻¹·r(x,τ))
        - 反解得: r̂(x,τ) = α·log π(τ|x) + α·log Z(x)
        - 将该奖励代入Bradley-Terry偏好模型: p(τ₁≻τ₂|x) = σ(α·[log π(τ₁|x) - log π(τ₂|x)])
        - 关键发现: Z(x)在奖励差中自然抵消（因为它只依赖x不依赖τ），Proposition 3.1证明了这种抵消不影响最优策略
    - **设计动机**: 直接计算轨迹级熵在COP中不可行（指数量级的轨迹数），但通过重参数化巧妙地将熵正则化隐式融入偏好分类目标中，无需显式枚举任何轨迹

3. **偏好模型选择与梯度分析**:

    - **功能**: 支持多种偏好模型（Bradley-Terry、Thurstone、Plackett-Luce），分析其隐式优势信号特性
    - **核心思路**: 最终优化目标为 max E[𝟙(r(x,τ₁)>r(x,τ₂))·log p_θ(τ₁≻τ₂|x)]，其梯度中的 g(τ,τ',x) - g(τ',τ,x) 项作为"量不变优势信号"——当r(x,τ₁)>r(x,τ₂)时，梯度始终增大π_θ(τ₁)、减小π_θ(τ₂)
    - **设计动机**: 与REINFORCE中优势值趋于零不同，PO的优势信号由sigmoid函数控制，始终保持有意义的区分能力

4. **局部搜索微调（Fine-Tuning with Local Search）**:

    - **功能**: 在策略收敛后，将局部搜索（如2-Opt、swap*）集成到训练中而非推理时
    - **核心思路**: 对每个采样解τ执行少量LS迭代得到LS(τ)，构造偏好对(τ, LS(τ), y)，其中y=𝟙(r(x,LS(τ))>r(x,τ))。优化目标变为max E[log p_θ(LS(τ)≻τ|x)]，即让策略学会模仿LS改进后的解
    - **设计动机**: 
        - REINFORCE难以利用LS：LS产生的off-policy样本会导致分布偏移，需要重要性采样纠正
        - PO的偏好目标天然兼容off-policy数据，可以将LS输出视为专家示范（类似模仿学习），无需重要性采样
        - 推理时不需要LS，不增加额外推理开销

### 损失函数 / 训练策略

- **主训练损失**: 基于BT模型的偏好分类损失
    - L_PO = -E[𝟙(r(x,τ₁)>r(x,τ₂))·log σ(α·[log π_θ(τ₁|x) - log π_θ(τ₂|x)])]
    - 对batch中每个实例采样N个解，做O(N²)的成对比较
- **微调损失**: L_finetune = -E[log σ(α·[log π_θ(LS(τ)|x) - log π_θ(τ|x)])]
- **训练流程**: 先用PO训练T步至收敛 → 再用PO+LS微调T_FT步
- **超参数α**: 对内建探索机制的模型（POMO、Sym-NCO）使用较低α值，避免过度探索影响收敛

## 实验关键数据

### 主实验

**TSP & CVRP（10k实例评估）**:

| 模型+算法 | TSP-50 Gap | TSP-100 Gap | CVRP-50 Gap | CVRP-100 Gap |
|-----------|-----------|------------|------------|-------------|
| POMO (RF) | 0.04% | 0.15% | 0.94% | 1.76% |
| POMO (PO) | 0.02% | 0.07% | 0.68% | 1.37% |
| POMO (PO+FT) | 0.00% | 0.03% | 0.53% | 1.19% |
| Sym-NCO (RF) | 0.16% | 0.39% | 1.31% | 2.07% |
| Sym-NCO (PO) | 0.08% | 0.28% | 1.20% | 1.88% |

**FFSP（1k实例评估）**:

| 模型+算法 | FFSP20 MS↓ | FFSP50 MS↓ | FFSP100 MS↓ |
|-----------|-----------|-----------|------------|
| MatNet (RF) | 27.3 | 51.5 | 91.5 |
| MatNet (PO) | 27.0 | 51.3 | 91.1 |
| MatNet (RF+Aug) | 25.4 | 49.6 | 89.7 |
| MatNet (PO+Aug) | 25.2 (最优) | 49.5 (最优) | 89.2 (最优) |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 偏好模型选择 | BT > PL > Thurstone > Exp (TSP-100) | Bradley-Terry模型表现最优 |
| 收敛速度 | PO用40%-60%训练量达到RF同等水平 | 1.5x-2.5x加速 |
| 优势值分布 | PO: 宽分布 vs RF: 窄集中于0 | PO显著改善区分能力 |
| 策略一致性 | PO >> RF，PO+FT 进一步提升 | p(π(τ₁)>π(τ₂) | r(τ₁)>r(τ₂)) |
| 轨迹熵 | PO显著高于RF（早期训练） | 验证PO的内在探索能力 |
| 泛化（TSPLib） | ELG(PO) 3.00% vs ELG(RF) 3.08% | zero-shot持续改进 |

### 关键发现

1. **收敛加速显著**: PO在POMO和Sym-NCO上训练80 epoch即可匹配RF训练200 epoch的效果
2. **解质量一致提升**: 在所有测试的神经求解器（AM、POMO、Sym-NCO、Pointerformer、MatNet、ELG）上，PO均优于或持平RF
3. **局部搜索微调非常有效**: POMO+PO+FT在TSP-100上将gap从0.07%降至0.03%，接近最优
4. **探索能力增强**: PO训练早期的轨迹熵显著高于RF，表明其从熵正则化目标继承了良好的探索特性

## 亮点与洞察

1. **跨领域创新**: 将RLHF/DPO的偏好优化思想首次系统地引入组合优化领域，是一个非常优雅的理论桥接
2. **理论推导简洁完整**: 从最大熵RL → 最优策略解析解 → 重参数化 → 偏好模型的推导链条清晰，且Proposition 3.1保证了Z(x)消除的合法性
3. **通用即插即用**: PO不绑定特定模型架构，可以替换任何现有REINFORCE-based方法的训练算法
4. **LS集成的理论解释**: 通过偏好框架自然规避了LS引入的off-policy问题，这是REINFORCE难以做到的
5. **偏好标签无冲突**: 利用COP奖励函数的客观性生成天然一致的偏好标签，避免了RLHF中人类标注不一致的问题

## 局限与展望

1. **O(N²)比较开销**: 成对比较带来二次复杂度，当采样数N较大时计算开销值得关注
2. **超参数α的选择**: 论文中α需要根据模型特性手动调整（内建探索的用低α），缺乏自适应策略
3. **局部搜索依赖**: 微调阶段的LS方法（2-Opt、swap*）是问题特定的，对新COP问题需要设计对应的LS算子
4. **重参数化假设**: 将π_θ近似为最优策略π*是一个较强假设，在训练早期可能不成立
5. **仅验证了经典COP**: 未在更复杂的实际场景（如带时间窗的VRP、多目标优化）中验证
6. **偏好模型理论分析不足**: 不同偏好模型（BT vs Thurstone vs PL）在不同问题上的优劣缺乏系统理论解释

## 相关工作与启发

- **DPO (Rafailov et al., 2024)**: PO的核心灵感来源——将KL正则化目标中的奖励重参数化为策略，但PO使用熵正则化而非KL正则化，适配COP无参考策略的场景
- **POMO (Kwon et al., 2020)**: 最强基线之一，PO在其上实现了最显著的提升
- **SAC (Haarnoja et al., 2017)**: PO的熵正则化目标与SAC同源，但PO进一步将其与偏好学习结合
- **启发**: 偏好优化的思想可能适用于其他奖励信号不稳定的RL场景（如机器人控制中的稀疏奖励问题），以及多目标优化中难以定义单一标量奖励的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ (跨领域迁移思路新颖，但核心技术来自DPO)
- 实验充分度: ⭐⭐⭐⭐⭐ (覆盖3类COP、6种求解器、泛化实验、消融全面)
- 写作质量: ⭐⭐⭐⭐⭐ (理论推导清晰，实验展示充分)
- 价值: ⭐⭐⭐⭐ (通用框架且效果一致提升，但主要是训练效率改进而非突破性结果)

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [BOPO: Neural Combinatorial Optimization via Best-anchored and Objective-guided Preference Optimization](bopo_neural_combinatorial_optimization_via_best-anchored_and_objective-guided_pr.md)
- [Self-Consistency Preference Optimization](self-consistency_preference_optimization.md)
- [ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)
- [MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)

<!-- RELATED:END -->
