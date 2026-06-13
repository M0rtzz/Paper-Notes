---
title: >-
  [论文解读] Self-Improving Embodied Foundation Models
description: >-
  [NeurIPS 2025][机器人][具身基础模型] 本文提出一种面向具身基础模型的两阶段后训练方法：第一阶段通过行为克隆和 steps-to-go 预测进行监督微调，第二阶段利用 steps-to-go 预测生成的自奖励函数和成功检测器实现在线 RL 自我改进，仅需 1-3% 额外数据即可实现 1.5x 以上的成功率提升，并首次展示了机器人自主学习超出模仿数据分布之外的新技能。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "具身基础模型"
  - "自我改进"
  - "强化学习后训练"
  - "Steps-to-go预测"
  - "机器人操作"
---

# Self-Improving Embodied Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2509.15155](https://arxiv.org/abs/2509.15155)  
**代码**: 暂无  
**领域**: 强化学习  
**关键词**: 具身基础模型, 自我改进, 强化学习后训练, Steps-to-go预测, 机器人操作

## 一句话总结
本文提出一种面向具身基础模型的两阶段后训练方法：第一阶段通过行为克隆和 steps-to-go 预测进行监督微调，第二阶段利用 steps-to-go 预测生成的自奖励函数和成功检测器实现在线 RL 自我改进，仅需 1-3% 额外数据即可实现 1.5x 以上的成功率提升，并首次展示了机器人自主学习超出模仿数据分布之外的新技能。

## 研究背景与动机
基础模型（Foundation Models）在 web 规模数据预训练后已经能被微调为机器人低级控制策略（RT-2, Octo, π0 等），且继承了预训练带来的显著泛化能力。然而，具身基础模型（EFM）的训练至今局限于行为克隆（监督学习）。反观 LLM 领域，预训练后的标准流程是：SFT → RL。RL 后训练已被证明能显著且快速地提升 LLM 在下游任务的性能，成为基础模型训练的关键环节。

将 RL 后训练应用于真实世界机器人面临独特挑战：**奖励工程**（reward engineering）问题——为每个操作任务手动设计奖励函数需要反复试错、修补，且在真实环境中测量奖励需要大量工程投入。随着任务种类增加，手动奖励设计变得不可持续。

核心 idea：利用 steps-to-go 预测（即"距离完成目标还有多少步"的时间距离估计）作为桥梁——它自然地生成了一个形状良好的**数据驱动奖励函数**和一个鲁棒的**成功检测器**，且这两个工具继承了底层基础模型在 web 规模预训练中获得的泛化能力。这消除了任务特定的奖励工程需求，使一个人类操作员就能监控多台机器人自主训练。

## 方法详解

### 整体框架
两阶段后训练框架：
- **Stage 1 (SFT)**：从预训练 PaLI（30亿参数视觉-语言模型）出发，用机器人模仿学习数据集同时训练行为克隆和 steps-to-go 预测
- **Stage 2 (Self-Improvement)**：冻结一个 Stage 1 模型用于计算奖励和检测成功，初始化另一个 Stage 1 模型作为策略，通过在线 RL（REINFORCE）自主练习并改进

### 关键设计
1. **Steps-to-go 预测与数据驱动奖励函数**:

    - Stage 1 训练两个目标：行为克隆损失 $\mathcal{L}_\text{BC} = -\mathbb{E}[\log p_\text{action}^\text{EFM}(a_t | o_t, g_{t'})]$ 和 steps-to-go 损失 $\mathcal{L}_\text{steps-to-go} = -\mathbb{E}[\log p_\text{steps-to-go}^\text{EFM}(t'-t | o_t, g_{t'})]$
    - 定义时间距离 $d(o, g) := \mathbb{E}_{p_\text{steps-to-go}}[\text{steps-to-go}]$
    - 奖励函数为 **时间距离差**：$r(o_t, a_t, o_{t+1}, g) := d(o_t, g) - d(o_{t+1}, g)$
    - 数学推导表明这个奖励隐含了势能形状奖励（potential-based shaping）：
    $r = \underbrace{(1-\gamma) \cdot V^\mu(o_{t+1}, g)}_{\text{core reward}} + \underbrace{[\gamma \cdot V^\mu(o_{t+1}, g) - V^\mu(o_t, g)]}_{\text{reward shaping}}$
    - 其中 $V^\mu$ 是数据集策略 $\mu$ 的值函数。Core reward 部分让策略在 $\mu$ 擅长的区域获得更高奖励，shaping 部分提供基线降低方差
    - 设计动机：完全数据驱动，无需手工设计，自动继承预训练模型的泛化能力

2. **Steps-to-go 成功检测器**:

    - 成功判定：$\text{success}(o, g) := \mathbb{1}[d(o, g) \leq s]$，其中 $s$ 是一个很小的步数阈值
    - 比显式训练二分类成功检测器更鲁棒，即使在低数据量下也稳定可靠
    - 设计动机：终止成功的 episode 避免收集冗余的"静止在成功状态"数据

3. **On-policy Self-Improvement 循环**:

    - 使用 REINFORCE 策略梯度：$-c \cdot R_t \cdot \log p_\text{action}^\text{EFM}(a_t | o_t, g)$
    - Monte Carlo 回报 $R_t = \sum_{i=t}^T \gamma^{i-t} \cdot r(o_i, a_i, o_{i+1}, g)$，折扣因子 $\gamma = 0.9$
    - 不使用经验回放、不训练值函数，消除了 deadly triad 的两个顶点（off-policy + bootstrapping）
    - 每轮收集足够数据后执行 $N$ 次策略更新，然后清空缓冲区重新开始
    - 一个人类操作员可同时监控多台机器人站点，仅在异常时手动干预
    - 设计动机：最大化训练稳定性和可靠性，为真实世界部署奠定基础

### 损失函数 / 训练策略
- Stage 1：联合优化 $\mathcal{L}_\text{BC} + \mathcal{L}_\text{steps-to-go}$（+ 可选辅助任务如指令预测）
- Stage 2：REINFORCE 损失，权重系数 $c = 5\times 10^{-2}$
- Stage 2 冻结一个独立的 Stage 1 模型用于奖励计算（避免奖励信号随训练漂移）
- PaLI 模型用 RT-2 参数化将连续动作 tokenize 为语言 token

## 实验关键数据

### 主实验

| 领域 | 数据量 | BC (Stage 1) | Self-Improvement (Stage 2) | 额外数据比例 |
|------|--------|-------------|---------------------------|------------|
| sim LanguageTable 10% | 10% imitation | 25% | 60% | +1% episodes |
| sim LanguageTable 20% | 20% imitation | 35% | 70% | +1.5% episodes |
| sim LanguageTable 80% | 80% imitation | 45% | **75%** | +2% episodes |
| real LanguageTable 20% | 20% imitation | ~62% | **~88%** | +3% episodes |
| real LanguageTable 80% | 80% imitation | ~63% | **~87%** | +3% episodes |
| sim Aloha 5K | 5K imitation | ~40% | ~65% | +2.5K episodes |
| sim Aloha 10K | 10K imitation | ~55% | **~75%** | +2K episodes |

### 消融实验

| 奖励模型 | 10% 数据 | 20% 数据 | 80% 数据 | 说明 |
|---------|---------|---------|---------|------|
| PaLI (多模态预训练) | **60%** | **70%** | **75%** | 最佳 |
| Uni-PaLI (单模态预训练) | ~40% | ~50% | ~65% | 显著低于 PaLI |
| Scratch (随机初始化) | 高方差 | 高方差 | ~55% | 低数据量完全失败 |

| BananaTable 泛化任务 | 成功率 | 说明 |
|---------------------|--------|------|
| Self-Improvement 前 | ~63% | 策略从未见过香蕉 |
| Self-Improvement 后 | **~85%** | 8小时自主练习 |

### 关键发现
- Self-Improvement 的样本效率远超扩大模仿数据：10% 数据 + 1% 自主练习 > 20% 纯模仿数据 > 80% 纯模仿数据中的多个情况
- 多模态预训练对 Self-Improvement 至关重要：PaLI 在 20% 数据量下的 Self-Improvement 效果优于 Uni-PaLI 在 80% 数据量下
- BananaTable 实验展示了**行为泛化**（而非仅语义泛化）：策略学会了推香蕉的特殊技巧（从中间或尖端推），超越了模仿数据中的行为模式
- Real2Sim 迁移中，仅 3% 额外数据就将目标域性能从 22% 提升到 59%

## 亮点与洞察
- **首次将 RL 后训练引入机器人基础模型训练流程**：借鉴 LLM 的 SFT→RL 范式，但通过 steps-to-go 创造性地解决了奖励工程难题。这一方法的优雅之处在于奖励函数自然地继承了基础模型的泛化能力——同一个模型既是策略又是奖励的来源。
- **BananaTable 实验的启示性**：与之前的语义泛化（如 RT-2 在新语境下执行相同动作）不同，BananaTable 展示了**行为泛化**——策略学会了全新的操作技巧。这说明 web 规模预训练 + 在线 Self-Improvement 的组合能解锁模仿学习永远无法覆盖的行为空间。

## 局限与展望
- Self-Improvement 超过性能峰值后会出现退化，缺少有效的 early stopping / 自适应正则化机制
- 仅使用 on-policy REINFORCE，无数据复用；off-policy 方法可能进一步减少所需的机器人小时数
- Steps-to-go 估计对分布外失败状态的建模不充分——模仿数据中没有失败恢复轨迹
- 成功检测器虽然鲁棒，但基于固定阈值，无法处理部分成功的细粒度评估
- 目前仅在双臂操作和桌面推块两个场景验证，尚未扩展到腿式运动或更复杂的长时间多步任务

## 相关工作与启发
- **vs RT-2**: RT-2 是本文 Stage 1 的直接等价物（BC 微调 VLM），本文在此基础上增加了 stages-to-go + RL 后训练，性能提升显著
- **vs RoboCat**: RoboCat 用 hindsight relabeling + BC 迭代来改进策略，但 hindsight relabeled SL 存在已知失败模式。本文使用显式 RL 优化
- **vs Code-as-Rewards**: LLM 写奖励代码的方法需要反复迭代、真实环境难以测量、需要独立的成功检测器，不适合通用机器人学习

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在真实机器人上验证基础模型的 RL 后训练，steps-to-go 作为桥梁的设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 仿真+真实、两种平台、多种数据量、预训练消融、域迁移、行为泛化全面覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，数学直觉与视觉直觉兼备，实验叙述层层递进
- 价值: ⭐⭐⭐⭐⭐ 为机器人基础模型指出了 SFT→RL 后训练的系统性路径，BananaTable 的行为泛化具有范式意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](../../CVPR2025/robotics/universal_actions_for_enhanced_embodied_foundation_models.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[ICML 2025\] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](../../ICML2025/robotics/sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)

</div>

<!-- RELATED:END -->
