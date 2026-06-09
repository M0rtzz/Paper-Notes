---
title: >-
  [论文解读] Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control
description: >-
  [NeurIPS 2025][强化学习][移动端控制] 提出SoLS算法，通过不对称策略更新机制（成功时激进学习、失败时保守正则化）和成功转换回放（STR），实现基础模型在移动应用控制任务上的高效强化学习微调，在AndroidWorld上达到51.3%成功率。
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "移动端控制"
  - "离策略学习"
  - "基础模型微调"
  - "样本效率"
---

# Succeed or Learn Slowly: Sample Efficient Off-Policy Reinforcement Learning for Mobile App Control

**会议**: NeurIPS 2025  
**arXiv**: [2509.01720](https://arxiv.org/abs/2509.01720)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 强化学习, 移动端控制, 离策略学习, 基础模型微调, 样本效率

## 一句话总结

提出SoLS算法，通过不对称策略更新机制（成功时激进学习、失败时保守正则化）和成功转换回放（STR），实现基础模型在移动应用控制任务上的高效强化学习微调，在AndroidWorld上达到51.3%成功率。

## 研究背景与动机

移动应用控制是一个极具挑战的多步交互任务：动作空间大且上下文相关，每步模拟耗时数秒，且大多数环境只提供稀疏的二元奖励（成功/失败）。现有方法存在明显瓶颈：

**GPT-4o提示工程方法**：虽然能利用强大的先验知识，但每步需要多次API调用，推理延迟高达40-60秒，且运营成本昂贵

**SFT微调方法**：在人类演示数据上训练的小模型泛化能力有限，尤其是当目标任务分布与训练数据存在显著偏移时

**标准RL方法**：在基础模型微调场景下面临两个根本问题——负优势样本的策略梯度更新会扰乱模型已学习的表征，导致性能退化

论文的核心洞察在于：**正优势样本的更新不需要策略正则化，而负优势样本的更新则可能损害模型性能**。这是因为当优势函数为负时，策略梯度会降低当前动作token的概率，从而不可控地提升词汇表中其他token的概率，破坏模型的语义结构。

## 方法详解

### 整体框架

系统采用两阶段流水线：首先在AndroidControl数据集上进行SFT预训练（获得基本的动作预测能力和输出格式），然后使用SoLS+STR在AndroidWorld环境中进行在线RL微调。基础模型为Llama-3-8B-Instruct，仅使用文本UI树作为输入。

### 关键设计

1. **不对称策略更新（SoLS核心机制）**

   SoLS基于离策略actor-critic框架，但对正负优势样本采用完全不同的更新策略。优势函数定义为 $A(s,a) = R - V^{\pi_\theta}(s)$，其中 $R$ 是蒙特卡洛回报。策略梯度为：

    $\nabla\mathcal{L}_{ac} = \begin{cases} -\mathbb{E}_{s,a\sim\hat{D}}\left[A \cdot \frac{\nabla\pi_\theta(a|s)}{\pi_b(a|s)}\right] & \text{if } A > 0 \text{ or } 1-\epsilon \leq \frac{\pi_\theta(a|s)}{\pi_b(a|s)} \leq 1+\epsilon \\ 0 & \text{otherwise} \end{cases}$

   当优势为正时，直接使用重要性采样比率进行无约束更新，最大化从成功经验中的学习效率。当优势为负时，仅在重要性采样比率处于 $[1-\epsilon, 1+\epsilon]$ 范围内才允许更新，否则完全跳过。这种设计在标准PPO的基础上增加了双侧约束——PPO仅约束比率下界，而SoLS同时约束上界，防止负样本导致策略过度偏移。

2. **成功转换回放（STR）**

   针对开放世界环境中轨迹生成成本高、成功率极低的问题，STR使用哈希表将每个任务映射到其成功动作的列表中。每个任务保留最近50个成功时间步，训练时从每个任务中采样n个成功转换与在线数据混合：

    $\mathcal{D} = \bigcup_{t \in \text{tasks}} \text{sample}(\mathcal{D}_{\text{STR}}(t), n) \cup \mathcal{D}_{on}$

   STR构建了SFT分布和目标分布之间的桥梁，通过从偶发成功中引导学习，避免了稀缺成功经验的浪费。

3. **值函数设计**

   值网络通过在Transformer最终隐藏层上添加仿射层+sigmoid激活实现，使用蒙特卡洛目标训练以避免额外的目标网络：

    $\mathcal{L}_{cr} = \mathbb{E}_{R,s\sim\mathcal{D}}\left[(R - V_\phi^{\pi_\theta}(s))^2\right]$

### 损失函数 / 训练策略

联合优化actor和critic损失：$\mathcal{L} = \mathcal{L}_{ac} + \lambda \cdot \mathcal{L}_{cr}$。训练采用数据并行架构，每个并行进程维护独立的STR实例。RL环境为AndroidWorld模拟器，每步执行约4-5秒。

## 实验关键数据

### 主实验

| 方法 | 输入类型 | Easy | Medium | Hard | Overall |
|------|---------|------|--------|------|---------|
| SeeAct (GPT-4o) | screen+UI tree | 36.1 | 17.9 | 0.0 | 22.5 |
| T3A (GPT-4o) | UI tree | 66.7 | 21.4 | 12.5 | 40.0 |
| GPT-4o+UGround-7B | screen | 69.4 | 28.6 | 12.5 | 43.8 |
| GPT-4o+AriaUI | screen+UI tree | 66.7 | 28.6 | 6.3 | 41.3 |
| SFT | UI tree | 38.9 | 9.5 | 4.2 | 22.1±2.7 |
| PPO | UI tree | 53.7 | 8.3 | 6.3 | 28.3±0.7 |
| DigiRL-STR | UI tree | 55.6 | 32.1 | 12.5 | 38.8±0.0 |
| **SoLS-STR** | UI tree | **68.5** | **40.5** | **16.6** | **51.3±1.2** |

### 消融实验

| 配置 | 成功率 | 说明 |
|------|--------|------|
| SoLS (无STR) | ~38.8% | 与DigiRL-STR持平 |
| SoLS-STR | 51.3% | STR带来显著提升 |
| A2C-STR | 32.1% | 无不对称更新，性能差60% |
| PPO-STR | ~35% | 离策略PPO变体 |
| DigiRL (无STR) | ~30% | STR对DigiRL也有帮助 |

### 关键发现

1. SoLS-STR相比DigiRL-STR实现32.5%的相对提升，超越所有GPT-4o方法
2. 不对称更新贡献最大：SoLS-STR vs A2C-STR的60%相对提升直接验证了核心假设
3. 推理速度约0.9秒/步，比UGround-7B快60倍
4. Medium难度任务提升最为显著（40.5% vs 32.1%），说明RL训练帮助学习域内知识
5. Files、Maps、Markor等类别在训练过程中成功率提升最大

## 亮点与洞察

1. **问题分析精准**：从负优势梯度导致token概率重分布的视角解释了为什么标准RL微调基础模型会导致性能退化，这一洞察具有普适价值
2. **设计简洁有效**：仅需修改PPO的裁剪方式（双侧约束+正样本不裁剪），实现了显著的性能提升
3. **实用性强**：8B模型单次前向推理即可完成任务，0.9秒延迟远优于多步提示工程

## 局限与展望

- 不对称约束可能导致提前收敛到局部最优
- 强依赖SFT初始策略的质量
- 无法处理需要视觉输入、长期记忆或完全未见过交互模式的任务
- 高度随机环境中可能出现策略震荡

## 相关工作与启发

- **DigiRL**: 移动应用控制的在线RL，使用拒绝采样避免负更新
- **GRPO (DeepSeek-R1)**: 无critic的RL算法，但需要每个提示64个响应，开销远大于SoLS
- **WoLF**: 名称灵感来源，根据胜负调整学习率

## 评分

- 新颖性: ⭐⭐⭐⭐☆ — 不对称更新思路有深刻洞察但不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ — AndroidWorld真实环境评估，多基线对比充分
- 写作质量: ⭐⭐⭐⭐☆ — 结构清晰，动机阐述到位
- 价值: ⭐⭐⭐⭐⭐ — 小模型+RL击败GPT-4o，实践意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Bootstrap Off-policy with World Model (BOOM)](bootstrap_off-policy_with_world_model.md)
- [\[NeurIPS 2025\] Finite-Sample Analysis of Policy Evaluation for Robust Average Reward Reinforcement Learning](finite-sample_analysis_of_policy_evaluation_for_robust_average_reward_reinforcem.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[NeurIPS 2025\] A Differential and Pointwise Control Approach to Reinforcement Learning](a_differential_and_pointwise_control_approach_to_reinforceme.md)
- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](../../ICML2026/reinforcement_learning/debiased_model-based_representations_for_sample-efficient_continuous_control.md)

</div>

<!-- RELATED:END -->
