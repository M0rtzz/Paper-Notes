---
title: >-
  [论文解读] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow
description: >-
  [AAAI 2026][强化学习][离线强化学习] 本文将MeanFlow从视觉生成任务重新改造为离线RL的生成式策略，提出一种残差形式的直接噪声到动作映射，实现单步采样的表达性策略，可在单阶段训练中与Q函数稳定联合优化，在OGBench和D4RL的73个任务上取得了强劲性能。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "离线强化学习"
  - "生成式策略"
  - "MeanFlow"
  - "单步采样"
  - "Q学习"
---

# One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow

**会议**: AAAI 2026  
**arXiv**: [2511.13035](https://arxiv.org/abs/2511.13035)  
**代码**: [https://github.com/HiccupRL/MeanFlowQL](https://github.com/HiccupRL/MeanFlowQL)  
**领域**: 强化学习 / 离线RL  
**关键词**: 离线强化学习, 生成式策略, MeanFlow, 单步采样, Q学习

## 一句话总结
本文将MeanFlow从视觉生成任务重新改造为离线RL的生成式策略，提出一种残差形式的直接噪声到动作映射，实现单步采样的表达性策略，可在单阶段训练中与Q函数稳定联合优化，在OGBench和D4RL的73个任务上取得了强劲性能。

## 研究背景与动机

**领域现状**：离线RL从固定数据集学习策略，面临表达性与效率的权衡。高斯策略单步快速但无法建模多模态动作分布；Flow/Diffusion策略表达性强但需多步迭代采样，与Q学习结合时需通过时间反向传播（BPTT），训练不稳定。

**现有痛点**：现有解决方案采用两阶段蒸馏——先用行为克隆训练多步生成策略，再蒸馏为单步策略并与Q值联合优化。但蒸馏引入表达性瓶颈，且增加训练复杂度。直接将MeanFlow用于RL会遇到早期训练阶段动作超出边界需裁剪的问题，导致策略输出与Bellman目标不一致，训练不稳定。

**核心矛盾**：需要一个策略既像Flow模型一样具有强多模态建模能力，又像高斯策略一样支持单步采样和稳定Q学习——这在之前的框架中是矛盾的。

**本文目标**：设计一个支持单步噪声→动作生成的生成式策略，能直接与Q函数联合训练（单阶段），无需蒸馏。

**切入角度**：MeanFlow通过建模平均速度场实现单步采样，但其"速度估计→速度积分"的两步推理在RL中导致动作越界。将其改写为残差形式 $g(a_t,b,t) = a_t - u(a_t,b,t)$，将速度估计和动作生成合并为单个网络前向。

**核心 idea**：将MeanFlow的两步过程（估计速度→积分得动作）合并为单步残差映射 $g_\theta$，配合适当的初始化策略（零初始化/小方差Kaiming初始化）确保早期训练输出在有效范围内，同时通过UAT保证表达能力不损失。

## 方法详解

### 整体框架
输入状态 $s$ 和高斯噪声 $e \sim \mathcal{N}(0,I)$，单步生成动作 $\hat{a} = g_\theta(e, b=0, t=1) = e - u_\theta(e, b=0, t=1)$。训练目标结合MeanFlow Identity损失（行为克隆）和Q值最大化（策略改进）。

### 关键设计

1. **残差MeanFlow策略重构**:

    - 功能：实现可微的单步噪声→动作映射
    - 核心思路：定义 $g(a_t,b,t) = a_t - u(a_t,b,t)$，其中 $u$ 是MeanFlow的平均速度场。当 $b=0, t=1$ 时退化为 $g(e,0,1) = e - u(e,0,1)$——即单步生成。关键区别在于用 $a_t$（数据-噪声插值）而非纯噪声 $\epsilon$ 作为输入，通过UAT保证在 MLP 足够大时 $g_\theta$ 可近似任意连续映射
    - 设计动机：朴素的 $a = \epsilon - u(\epsilon, b, t)$ 在玩具实验中无法拟合多模态分布。使用 $a_t$ 插值作为input保留了flow matching的条件概率路径结构

2. **MeanFlow Identity训练损失**:

    - 功能：无需显式速度积分即可训练平均速度场
    - 核心思路：$\mathcal{L}_{MFI}(\theta) = \mathbb{E}||g_\theta(a_t,b,t) - \text{sg}(g_{tgt})||_2^2$，其中目标 $g_{tgt}$ 由MeanFlow Identity推导得到。使用stop-gradient防止模式坍塌。训练时从数据中采样 $(s,a)$，从高斯采样 $e$，构造 $a_t = (1-t)a + te$，优化 $g_\theta$ 满足MeanFlow恒等式
    - 设计动机：直接利用MeanFlow的理论框架，避免了ODE求解器的不稳定性

3. **Q学习联合优化与实用增强**:

    - 功能：在单阶段训练中同时进行行为克隆和策略改进
    - 核心思路：总目标 = MFI损失（行为克隆正则）+ Q值最大化 + 自适应BC正则权重。额外引入value-guided rejection sampling提升推理质量——采样多个噪声，选Q值最高的动作
    - 设计动机：单步映射使Q值反向传播直达策略参数（无BPTT），训练稳定高效

### 损失函数 / 训练策略
$\mathcal{L}_\pi = -Q_\phi(s, g_\theta(e,0,1)) + \alpha \cdot \mathcal{L}_{MFI}$。Critic用标准Bellman误差训练。$\alpha$ 自适应调整。

## 实验关键数据

### 主实验

| 方法 | OGBench (73 tasks avg) | D4RL avg | 推理步数 | 训练阶段 |
|---|---|---|---|---|
| Gaussian (SAC-style) | 一般 | 一般 | 1步 | 单阶段 |
| Diffusion Policy | 竞争力 | 竞争力 | 多步 | 两阶段 |
| Flow Policy + Distillation | 竞争力 | 竞争力 | 1步 | 两阶段 |
| **MeanFlowQL** | **强劲** | **强劲** | **1步** | **单阶段** |

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| 原始MeanFlow（两步推理） | 训练不稳定 | 动作越界+裁剪问题 |
| 朴素残差形式 | 欠拟合 | 无法建模多模态 |
| 修正残差形式（本文） | 最优 | 保持表达性+训练稳定 |
| 无rejection sampling | 略降 | 采样质量影响性能 |
| 无自适应BC正则 | 略降 | BC-Q平衡重要 |

### 关键发现
- 残差形式的选择至关重要——朴素形式在玩具实验中完全无法拟合多模态分布
- 单阶段训练比两阶段蒸馏更简单且最终策略表达性更好
- Value-guided rejection sampling是低成本高收益的推理增强
- 在73个任务上表现稳定，在offline-to-online设定下也有竞争力

## 亮点与洞察
- **MeanFlow从生成到RL的巧妙迁移**：原本用于图像生成的单步方法被重新构造为RL策略，解决了flow policy与Q学习的兼容性问题
- **残差形式的深入分析**：不只提出一种方案，而是系统分析了多种重构变体并解释为何只有特定形式有效，分析透彻
- **消除两阶段训练的复杂性**：单阶段端到端训练比蒸馏更简洁，也避免了蒸馏带来的表达性损失

## 局限与展望
- 基于MeanFlow的理论假设（如速度场平滑性），在极高维动作空间的适用性有待验证
- Value-guided rejection sampling增加了推理成本（虽然只是线性倍数）
- 仅验证了离线RL，纯在线RL场景的适用性未探索
- 可结合世界模型进一步提升仅从离线数据学习的效果

## 相关工作与启发
- **vs IDQL/SfBC（两阶段蒸馏）**: 需先BC训练再蒸馏，本文单阶段完成且表达性更好
- **vs Diffusion Policy（如DDPO）**: 多步推理+BPTT不稳定，本文单步+可微
- **vs Gaussian Policy**: 单步但无法建模多模态，本文保持单步采样的同时具备flow级表达性
- MeanFlow重构为RL策略的思路可推广到其他需要单步采样的场景（如实时控制）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将MeanFlow引入RL并解决了兼容性问题，残差重构分析深入
- 实验充分度: ⭐⭐⭐⭐⭐ 73个任务覆盖OGBench和D4RL，含offline和offline-to-online
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论推导完整
- 价值: ⭐⭐⭐⭐⭐ 解决了生成式策略与Q学习结合的核心瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/reinforcement_learning/offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] MFPO: 用 Few-step MeanFlow Policy 把 MaxEnt RL 跑到接近 Gaussian policy 的速度](../../ICML2026/reinforcement_learning/mean_flow_policy_optimization.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)
- [\[NeurIPS 2025\] CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design](../../NeurIPS2025/reinforcement_learning/core_constraint-aware_one-step_reinforcement_learning_for_simulation-guided_neur.md)

</div>

<!-- RELATED:END -->
