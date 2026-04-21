---
title: >-
  [论文解读] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning
description: >-
  [ICLR 2026][图像生成][Flow Matching] 通过在流匹配训练中注入可控噪声扩大策略覆盖范围，并结合熵引导的采样机制在在线微调时动态平衡探索与利用，在有限交互预算下显著提升离线到在线RL的样本效率。
tags:
  - ICLR 2026
  - 图像生成
  - Flow Matching
  - 离线-在线RL
  - 噪声注入
  - 探索-利用平衡
  - 熵引导采样
---

# Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning

**会议**: ICLR 2026  
**arXiv**: [2602.18117](https://arxiv.org/abs/2602.18117)  
**代码**: [GitHub](https://github.com/CTID282/FINO)  
**领域**: 流匹配/强化学习  
**关键词**: Flow Matching, 离线-在线RL, 噪声注入, 探索-利用平衡, 熵引导采样

## 一句话总结
通过在流匹配训练中注入可控噪声扩大策略覆盖范围，并结合熵引导的采样机制在在线微调时动态平衡探索与利用，在有限交互预算下显著提升离线到在线RL的样本效率。

## 研究背景与动机

**领域现状**：生成模型（扩散/流匹配）因能建模多模态分布，在离线RL中作为策略表示表现优异。Flow Q-Learning (FQL) 已在离线设置中证明了流匹配策略的有效性。

**现有痛点**：离线预训练的策略过度约束于数据集分布，在线微调阶段探索能力不足。现有方法（如FQL）将在线微调简单视为离线预训练的延续，未专门设计探索机制。在antmaze-giant任务中，FQL agent几乎只沿数据集中存在的上方路径到达目标，完全忽略了其他可行路线。

**核心矛盾**：离线RL需要保守约束（避免分布外动作），但在线阶段需要广泛探索（超出数据覆盖）。这两个阶段对策略分布的要求本质对立。

**本文目标**：如何在不增加数据集的前提下，让策略在离线预训练阶段就学到比数据集更广的动作覆盖，并在在线微调时有效利用这种多样性？

**切入角度**：观察到flow matching的条件概率路径在 $\sigma_{\min}=0$ 时会将分布坍缩到单个数据点上，限制了覆盖范围。通过向flow matching注入受控噪声，可以扩展条件概率路径的方差。

**核心 idea**：在flow matching目标中注入噪声以扩大策略支持集，配合熵引导采样在在线阶段自适应地平衡探索与利用。

## 方法详解

### 整体框架
FINO (Flow matching with Injected Noise for Offline-to-online RL) 基于FQL框架，包含两个核心组件：(1) 离线预训练阶段注入噪声的流匹配训练；(2) 在线微调阶段的熵引导采样机制。输入为状态-动作对数据集，输出为能在在线交互中高效探索的策略。

### 关键设计

1. **噪声注入流匹配 (Noise Injection for Flow Matching)**:

    - 功能：在流匹配训练过程中沿插值路径注入时间依赖的高斯噪声 $\epsilon_t \sim \mathcal{N}(0, \alpha_t^2 I)$
    - 核心思路：修改训练目标为 $\mathcal{L}_{\text{FINO}}(\theta) = \mathbb{E}[\|v_\theta(t, s, x_t + \epsilon_t) - (x_1 - (1-\eta)x_0)\|^2_2]$，其中 $\alpha_t^2 = (\eta^2 - 2\eta)t^2 + 2\eta t$，$\eta \in [0,1]$ 控制噪声幅度。当 $\eta=0$ 时退化为标准flow matching。
    - 设计动机：Theorem 2 证明FINO诱导的边际概率路径方差大于等于标准FM，即 $\text{Var}(X_t^{\text{FINO}}) \geq \text{Var}(X_t^{\text{FM}})$，使策略覆盖更广的动作空间。Theorem 1 保证噪声注入后仍构成有效的连续归一化流。

2. **熵引导采样 (Entropy-Guided Sampling)**:

    - 功能：在线微调时从策略中采样多个候选动作，基于Q值构建softmax采样分布，并自适应调节温度参数
    - 核心思路：采样概率 $p(i) = \frac{\exp(\xi \cdot Q_\phi(s, a_i))}{\sum_j \exp(\xi \cdot Q_\phi(s, a_j))}$，温度 $\xi$ 按策略熵自适应更新：$\xi_{\text{new}} = \xi - \alpha_\xi[\mathcal{H} - \bar{\mathcal{H}}]$
    - 设计动机：固定温度无法适应学习过程的动态变化。策略熵高时增大 $\xi$ 偏向利用，熵低时减小 $\xi$ 偏向探索，实现自动平衡。

3. **实现细节**:

    - 功能：解决one-step policy分布不可求导的熵估计问题
    - 核心思路：从同一状态采样多个动作，用高斯混合模型(GMM)拟合后估计熵。$\eta=0.1$（基于动作范围 $[-1,1]$），候选动作数为动作维度的一半。
    - 设计动机：one-step policy通过蒸馏和Q值优化得到，其分布不可直接计算熵。

### 损失函数 / 训练策略
- 离线阶段：同时训练flow policy（Eq.7）、one-step policy（Eq.5）和Q网络（TD loss）
- 在线阶段：继续优化三个网络，每 $N_\xi$ 步更新一次温度 $\xi$
- 推理时：确定性选择Q值最高的动作

## 实验关键数据

### 主实验
在OGBench和D4RL共45个任务上评估，10个随机种子取平均：

| 任务类别 | 指标 | FINO | FQL | Cal-QL | 提升 |
|---------|------|------|-----|--------|------|
| OGBench humanoidmaze-medium | 在线微调后得分 | 最优 | 3±3 | 0±0 | 显著 |
| D4RL antmaze (6任务平均) | 在线微调后得分 | 最优 | 次优 | - | 稳定提升 |
| D4RL adroit (4任务平均) | 在线微调后得分 | 最优 | 次优 | - | 稳定提升 |
| OGBench (5任务平均) | 在线微调后得分 | 最优 | 次优 | - | 显著 |

### 消融实验

| 配置 | 关键表现 | 说明 |
|------|---------|------|
| Full FINO | 最优 | 噪声注入 + 熵引导采样 |
| w/o 噪声注入 (η=0) | 明显下降 | 退化为标准FQL |
| w/o 熵引导 | 下降 | 固定温度无法自适应 |
| 仅噪声注入 | 中等 | 缺乏在线阶段的探索-利用平衡 |

### 关键发现
- 在antmaze-giant任务中，FINO能发现多条到达目标的路线（包括下方路径），而FQL只走上方路径
- 噪声注入使策略在toy实验中明显覆盖了数据点周围更广的区域，但仍以数据为中心
- 在高维动作空间任务（如humanoidmaze）中优势更显著，因为噪声注入提供的探索在高维中更关键

## 亮点与洞察
- **噪声注入的理论保证**：三个定理完整论证了噪声注入的合理性——保持有效流（Theorem 1）、扩大覆盖（Theorem 2）、构成合法分布（Proposition 1）。理论严谨且实际有效。
- **离线为在线服务的设计理念**：不把离线训练独立看待，而是从一开始就为后续在线微调做准备（注入噪声保留多样性），这种前瞻性设计值得借鉴。
- **自适应温度**方案简洁优雅，用策略熵作为信号闭环调节探索-利用权衡，无需手动调参。

## 局限与展望
- $\eta$ 统一设为0.1，未考虑不同任务/不同状态的最优噪声幅度可能不同
- 熵估计依赖GMM拟合，在极高维动作空间中GMM的准确性可能下降
- 候选动作数设为动作维度的一半，这一启发式规则缺乏理论依据
- 实验主要在locomotion和navigation任务上，未验证在manipulation等精细操作任务上的效果

## 相关工作与启发
- **vs FQL**: FQL直接用标准flow matching做策略，FINO在训练目标中注入噪声扩大覆盖。FINO在在线微调后性能显著优于FQL，尤其在需要探索的任务中。
- **vs Cal-QL/RLPD**: 这些方法不使用生成模型作为策略，FINO利用flow matching的表达力优势在复杂任务中更有竞争力。
- **可迁移思路**：噪声注入扩大分布覆盖的思路可以迁移到其他生成模型场景，如扩散模型的fine-tuning、条件生成的多样性增强等。

## 评分
- 新颖性: ⭐⭐⭐⭐ 噪声注入flow matching的想法新颖，理论分析扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 45个任务、10个种子、多个baseline、详细消融
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论推导完整，大量附录支撑
- 价值: ⭐⭐⭐⭐ 为离线-在线RL提供了实用且有理论保障的解决方案

<!-- RELATED:START -->

## 相关论文

- [Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [DiffusionNFT: Online Diffusion Reinforcement with Forward Process](diffusionnft_online_diffusion_reinforcement_with_forward_process.md)
- [Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)

<!-- RELATED:END -->
