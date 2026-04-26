---
title: >-
  [论文解读] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data
description: >-
  [NeurIPS 2025][图像生成][强化学习] 提出 CompFlow，通过复合流匹配架构（在离线流输出分布上构建在线流）估计离线-在线环境间的动态差异（Wasserstein 距离），并结合高动态差异区域的主动探索策略，在 27 个动态偏移 RL 任务中平均回报超越最强基线 14.2%。
tags:
  - NeurIPS 2025
  - 图像生成
  - 强化学习
  - 流匹配
  - 最优传输
  - 动态偏移
  - Wasserstein距离
  - 离线数据
---

# Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data

**会议**: NeurIPS 2025  
**arXiv**: [2505.23062](https://arxiv.org/abs/2505.23062)  
**代码**: [GitHub](https://github.com/Haichuan23/CompositeFlow)  
**领域**: image_generation  
**关键词**: 强化学习, 流匹配, 最优传输, 动态偏移, Wasserstein距离, 离线数据

## 一句话总结

提出 CompFlow，通过复合流匹配架构（在离线流输出分布上构建在线流）估计离线-在线环境间的动态差异（Wasserstein 距离），并结合高动态差异区域的主动探索策略，在 27 个动态偏移 RL 任务中平均回报超越最强基线 14.2%。

## 研究背景与动机

在线 RL 通常需要大量环境交互，在机器人、医疗、野生动物保护等现实场景中成本极高。利用预采集的离线数据可以提升样本效率，但当离线数据的**转移动力学**与在线环境不同时（称为 shifted dynamics），直接使用会导致分布失配、策略退化。

现有方法的核心缺陷：
- **H2O**、**BC-PAR** 等用 KL 散度或互信息估计动态差异，但当两个动力学的支持集不匹配时，这些度量**可能未定义或不稳定**
- **BC-VGDF** 基于值估计过滤，但 bootstrapping 偏差和目标非平稳性导致估计不可靠
- 所有方法仅做被动过滤（剔除高差异数据），未主动探索高差异区域

CompFlow 的核心洞察：**流匹配与最优传输之间存在理论联系**——训练好的流模型的传输代价近似 Wasserstein 距离。利用此联系可以得到**定义良好、支撑集无关**的动态差异估计。

## 方法详解

### 整体框架

CompFlow 包含三个核心模块：

1. **离线流模型**：从高斯先验学习离线动力学 $p_{\text{off}}(s'|s,a)$
2. **在线复合流模型**：在离线流的输出分布上构建，学习从 $p_{\text{off}}$ 到 $p_{\text{on}}$ 的残差传输
3. **动态差异引导的策略训练**：基于 Wasserstein 距离估计的动态差异，选择性合并离线数据 + 主动探索高差异区域

### 关键设计

**复合流架构**：

传统方法从高斯先验 $x_0 \sim \mathcal{N}(0, \mathbf{I})$ 直接学习在线动力学。CompFlow 分两段：

- **离线流**（$t: 0 \to 1$）：$x_1 = \psi_\theta^{\text{off}}(x_0, 1 \mid s, a)$，将高斯映射到离线转移分布
- **在线流**（$t: 1 \to 2$）：$x_2 = \psi_\phi^{\text{on}}(x_1, 2 \mid s, a)$，从离线分布映射到在线转移分布

理论保证（Theorem 3.1）：当 $W_2(p_G, p_{\text{on}}) > W_2(\hat{p}_{\text{off}}, p_{\text{on}})$ 时（离线分布比高斯更接近在线分布），复合流享有更紧的泛化误差界。

**Wasserstein 距离估计**：利用 OT-FM（最优传输流匹配）训练在线流时，传输代价自然近似 2-Wasserstein 距离。Monte Carlo 估计器：

$$\hat{\Delta}(s,a) = \left(\frac{1}{M} \sum_{j=1}^{M} \left\|\psi_\theta^{\text{off}}(x_0^{(j)}, 1 \mid s,a) - \psi_\phi^{\text{on}}(\psi_\theta^{\text{off}}(x_0^{(j)}, 1 \mid s,a), 2 \mid s,a)\right\|_2^2\right)^{1/2}$$

**主动探索策略**：在标准 actor-critic 的动作选择中加入探索奖励：

$$a = \arg\max_{a \in \mathcal{A}} \left[Q(s,a) + \beta\, \hat{\Delta}(s,a)\right]$$

$\beta$ 控制探索强度。Theorem 3.5 证明：在高动态差异区域收集更多样本可以进一步缩小与最优策略的性能差距。

**数据选择**：每次迭代中，仅保留动态差异低于 $\xi$-分位数阈值的离线转移加入回放缓冲区：

$$\mathcal{B} = \{(s,a) \in \mathcal{D}_{\text{off}}: \hat{\Delta}(s,a) \leq \hat{\Delta}_{\xi\%}\} \cup \mathcal{D}_{\text{on}}$$

### 损失函数

**Critic 损失**：

$$\mathcal{L}_Q = \mathbb{E}_{\mathcal{D}_{\text{on}}}[(Q_\varsigma(s,a) - y)^2] + \mathbb{E}_{\mathcal{D}_{\text{off}}}[\mathbf{1}(\hat{\Delta}(s,a) \leq \hat{\Delta}_{\xi\%})(Q_\varsigma(s,a) - y)^2]$$

目标值 $y = r + \gamma Q_\varsigma(s', a') + \beta \hat{\Delta}(s,a)$

**Policy 损失**（策略改进 + 行为克隆正则项）：

$$\mathcal{L}_\pi = \mathbb{E}_{s, a \sim \pi_\varphi}[Q_\varsigma(s,a)] - \omega\, \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{off}}, \tilde{a} \sim \pi_\varphi}[\|a - \tilde{a}\|_2^2]$$

## 实验关键数据

### 主实验

Gym-MuJoCo 三种动态偏移（形态/运动学/摩擦），三种数据质量（MR/M/ME），27 个任务：

| 方法 | 平均回报↑ | 最优任务数 | 相对 SAC 提升 |
|:--|:--:|:--:|:--:|
| SAC (纯在线) | 878 | 0 | — |
| BC-SAC | 1920 | — | +118.7% |
| H2O | 1783 | — | +103.1% |
| BC-VGDF | 1868 | — | +112.8% |
| BC-PAR | 1803 | — | +105.4% |
| **CompFlow** | **2193** | **19/27 最优，5 并列** | **+149.8%** |

CompFlow 在 27 个任务中 24 个达到最优或并列，平均回报比最强基线（BC-SAC）高 14.2%。

### 消融实验

**复合流 vs 直接流**：在验证集上，复合流的转移动力学 MSE 在训练过程中始终显著低于从高斯先验直接学习的流模型，验证了 Theorem 3.1 的理论预测。

**数据选择比例 $\xi$**：$\xi \in \{20, 30, 50, 70\}$，中等值（30 或 50）通常最优。$\xi$ 过大会引入高差异的低质量转移，过小则浪费有用数据。

**探索强度 $\beta$**：$\beta = 0$（无探索）在所有 5 个测试任务中均表现最差或倒数第二，验证了主动探索的必要性。不同任务的最优 $\beta$ 不同（摩擦任务偏好大 $\beta$，形态任务偏好中等 $\beta$）。

### 关键发现

1. 大多数现有基线（H2O、BC-VGDF、BC-PAR）性能与 BC-SAC 相当，表明它们未能有效利用离线数据中的结构信息
2. 基于 KL/互信息的动态差异估计在大偏移或支撑集不匹配时失效，而 Wasserstein 距离始终稳定
3. 野生动物保护实验中，CompFlow 超越最强基线 8.8%，在真实场景中验证了方法的实用价值
4. 离线数据利用 + 主动探索两个机制缺一不可

## 亮点与洞察

- ⭐ 复合流的设计简洁而深刻——在离线流输出上构建在线流，既共享结构知识又可精确测量差异
- ⭐ 将流匹配-最优传输-Wasserstein 距离三者联系起来作为动态差异估计工具，理论和实践均严密
- 主动探索高差异区域的策略有扎实的理论支撑（Theorem 3.5 给出了具体的性能差距缩减量）
- 方法通用性强，可与任意 actor-critic 算法（如 SAC）即插即用

## 局限性

- 复合流需要训练两个流模型（离线 + 在线），计算开销较高
- Theorem 3.1 假设离线分布比高斯更接近在线分布——对于差异极大的离线数据可能不成立
- 实验主要在 MuJoCo 连续控制任务上验证，未涉及高维观测（图像）或离散动作空间
- 探索强度 $\beta$ 和数据选择比例 $\xi$ 需要针对每个任务调参

## 相关工作与启发

将流匹配引入跨域 RL 的动态差异估计是一个新视角。与 DARC（基于分类器的差异估计）、H2O（基于 KL 的 Q 值惩罚）相比，Wasserstein 距离具有更好的数学性质（度量性、连续性、不要求支撑集重合）。野生动物保护实验展示了方法在社会公益领域的应用潜力。

## 评分

⭐⭐⭐⭐ (4/5)

| 维度 | 评分 |
|:--|:--:|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |

理论贡献突出，复合流架构设计巧妙。实验覆盖 27 个任务 + 真实场景，说服力强。主要缺憾是计算开销分析不足，且未在更复杂环境（如视觉 RL）中验证。

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Curly Flow Matching for Learning Non-gradient Field Dynamics](curly_flow_matching_for_learning_non-gradient_field_dynamics.md)
- [\[ICML 2025\] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers](../../ICML2025/image_generation/elucidating_flow_matching_ode_dynamics_with_respect_to_data_geometries_and_denoi.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](value_gradient_guidance_for_flow_matching_alignment.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)

<!-- RELATED:END -->
