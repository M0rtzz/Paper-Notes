---
title: >-
  [论文解读] Tree-Guided Diffusion Planner
description: >-
  [NeurIPS 2025][图像生成][扩散规划] 提出Tree-guided Diffusion Planner (TDP)，将测试时扩散规划形式化为树搜索问题，通过双层采样（粒子引导生成多样父轨迹 + 快速条件去噪生成子轨迹）在探索与利用之间取得平衡，在非凸目标和不可微约束下显著超越现有方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - 扩散规划
  - 树搜索
  - 测试时引导
  - 零样本规划
  - 轨迹生成
---

# Tree-Guided Diffusion Planner

**会议**: NeurIPS 2025  
**arXiv**: [2508.21800](https://arxiv.org/abs/2508.21800)  
**代码**: [项目主页](https://tree-diffusion-planner.github.io)  
**领域**: 扩散模型 / 规划与生成  
**关键词**: 扩散规划, 树搜索, 测试时引导, 零样本规划, 轨迹生成

## 一句话总结

提出Tree-guided Diffusion Planner (TDP)，将测试时扩散规划形式化为树搜索问题，通过双层采样（粒子引导生成多样父轨迹 + 快速条件去噪生成子轨迹）在探索与利用之间取得平衡，在非凸目标和不可微约束下显著超越现有方法。

## 研究背景与动机

扩散模型已成为离线规划的有力框架，能从离线演示数据中生成连贯的轨迹。然而现有测试时引导规划方法面临多个根本性限制：

**梯度引导的局限性**：标准梯度引导（如classifier guidance）假设引导函数是凸的、可微的，但现实世界的规划任务常涉及非凸目标（如多目标导航）和不可微约束（如必须经过中间点的路径规划）。

**分布内偏好（In-distribution Preference）**：预训练扩散模型倾向于生成与训练数据分布一致的轨迹，难以发现需要组合性创新的解决方案。梯度引导容易陷入局部最优，因为它优先考虑学习分布内的局部最优轨迹。

**探索-利用困境**：引导强度 $\alpha$ 高度依赖任务，穷举调参在测试时引导场景下代价高昂。现有方法未能充分解决在保持轨迹可行性的同时最大化引导得分的平衡问题。

**监督方法的局限**：MCTD等序列方法需要训练任务特定的值估计器，Hierarchical Diffuser依赖训练时标注的子目标分布，D-MPC需要少样本微调——这些都限制了零样本泛化能力。

## 方法详解

### 整体框架

TDP采用双层（bi-level）轨迹采样框架：第一层通过粒子引导（Particle Guidance）生成多样的父轨迹实现探索；第二层通过快速条件去噪从父轨迹分支生成子轨迹实现利用。所有轨迹构成树结构，选择叶节点得分最高的路径作为最终方案。

### 关键设计

1. **状态分解（State Decomposition）**：给定测试时引导函数，自动将状态向量分为观察状态和控制状态。通过检查引导函数对每个状态特征的梯度——梯度非零为观察状态，梯度为零为控制状态。例如在KUKA机器人臂任务中，方块位置是观察状态（影响目标函数），机器人关节角度是控制状态（不直接影响目标但控制动力学）。这种基于梯度的自动分类无需任务先验知识。

2. **父轨迹分支（Parent Branching）**：在控制状态上应用固定势能粒子引导（PG），利用径向基函数（RBF）的梯度 $\nabla\Phi$ 在轨迹间施加排斥力，促进多样性探索。在观察状态上施加任务梯度引导 $\nabla\mathcal{J}$。整体去噪步骤为：

$$\boldsymbol{\mu}^{i-1}_{\text{control}} \leftarrow \boldsymbol{\mu}^{i}_{\text{control}} + \alpha_p \Sigma^i \nabla\Phi(\boldsymbol{\mu}^{i}_{\text{control}})$$
$$\boldsymbol{\mu}^{i-1}_{\text{obs}} \leftarrow \boldsymbol{\mu}^{i}_{\text{obs}} + \alpha_g \Sigma^i \nabla\mathcal{J}(\boldsymbol{\mu}^{i}_{\text{obs}})$$

PG与梯度引导共同构成统一的条件分布，组合引导项为 $g_{\text{TDP}} = g_{\text{gg}} + g_{\text{pg}}$。

3. **子树扩展（Sub-Tree Expansion）**：对每个父轨迹，随机选择分支点 $b \sim \text{Unif}(0, T_{\text{pred}})$，将父轨迹在分支点之后部分加噪（使用 $N_f \ll N$ 步的快速去噪），并以父轨迹前缀 $\boldsymbol{C} = \{s_k\}_{k=0}^b$ 为条件生成子轨迹：

$$\boldsymbol{\tau}_{\text{child}}^{N_f} \sim q_{N_f}(\boldsymbol{\tau}_{\text{parent}}, \boldsymbol{C})$$

子轨迹实现两个作用：(1) 修复父轨迹因粒子引导引入的动力学不可行性；(2) 在父轨迹附近进行高效局部搜索以找到更优解。

### 理论保证

Proposition 1证明了双层采样的必要性：从标准高斯初始化的引导采样可能收敛到偏离数据流形的局部最优（off-subspace），而从无条件样本附近初始化的引导采样能收敛到全局最优。这解释了为什么TDP先生成on-subspace的父轨迹再做引导子树扩展的策略优于直接梯度引导。

## 实验关键数据

### 主实验——Maze2D Gold-picking

| 方法 | Medium | Large | 单任务平均 | Multi-Medium | Multi-Large | 多任务平均 |
|------|--------|-------|-----------|-------------|-------------|-----------|
| Diffuser | 10.1 | 4.3 | 6.8 | 7.7 | 9.9 | 8.8 |
| Diffuserγ (TAT) | 12.3 | 9.3 | 10.8 | 8.6 | 23.1 | 15.9 |
| MCSS | 17.2 | 25.0 | 21.1 | 32.3 | 57.5 | 44.9 |
| MCSS+SS | 17.4 | 21.2 | 19.3 | 29.2 | 58.0 | 43.6 |
| TDP (w/o child) | 19.0 | 30.4 | 24.7 | 35.3 | 59.1 | 47.2 |
| TDP (w/o PG) | 39.1 | 41.1 | 40.1 | 75.9 | 64.9 | 70.4 |
| **TDP** | **39.8** | **47.6** | **43.7** | **74.7** | **70.0** | **72.4** |

### 消融实验——Robot Arm Manipulation

| 方法 | PnWP | PnP (stack) | PnP (place) | PnP平均 |
|------|------|------------|------------|---------|
| Diffuser | 31.13 | 51.50 | 21.31 | 36.41 |
| AdaptDiffuser | 39.72 | 60.54 | 36.17 | 48.36 |
| MCSS | 35.69 | 59.91 | 31.37 | 45.64 |
| MCSS+SS | 36.24 | 56.80 | 35.50 | 46.15 |
| TDP (w/o child) | 35.53 | 60.00 | 32.19 | 46.10 |
| TDP (w/o PG) | 66.63 | 59.42 | 36.94 | 48.18 |
| **TDP** | **66.81** | **61.17** | **36.94** | **49.06** |

### 关键发现

- **非凸任务提升最为显著**：在PnWP任务中TDP（66.81）比MCSS（35.69）提高近一倍，说明双层采样在非凸引导函数下优势巨大
- **粒子引导是多模态探索的关键**：TDP (w/o PG)在PnWP上从66.81降至66.63差距不大，但在Maze2D多任务中TDP vs TDP (w/o PG)差距明显，PG在需要广泛探索的场景更重要
- **子树扩展提供精细优化**：TDP (w/o child)在PnWP仅35.53，远低于完整TDP的66.81，证实局部搜索对发现全局最优至关重要
- **AntMaze中找到的目标数比MCSS高11%**，同时减少了每个目标的到达步数，说明探索效率更高
- TDP超越了需要任务特定训练的AdaptDiffuser，证明零样本测试时规划的能力

## 亮点与洞察

- **将规划问题重构为树搜索**的视角新颖且自然——父节点提供大范围探索，子节点提供局部精化，完美匹配探索-利用平衡
- **状态分解**的自动化设计使方法无需任务先验知识，具有良好的通用性
- **理论分析**（Proposition 1）清晰解释了为什么从on-subspace初始化优于标准高斯初始化，为方法提供了理论支撑
- 完全零样本、无需额外训练的特性使其适用于各种新增测试时目标

## 局限与展望

- 双层轨迹生成引入额外计算开销（粒子间成对距离计算 + 子轨迹生成的额外前向传播）
- 粒子引导的超参数（$\alpha_p$, $\alpha_g$）仍需指定，虽然比单一引导强度更鲁棒但仍有调参需求
- 实验场景相对简单（2D迷宫、机器人臂），在更高维度和更复杂动力学下的表现待验证
- 闭环规划中每步都需重新搜索，实时性可能受限
- 未与基于学习的测试时适应方法做深入对比

## 相关工作与启发

- TDP将扩散模型的采样多样性与树搜索的结构化探索相结合，为扩散模型在复杂决策问题中的应用提供了新范式
- 粒子引导（PG）促进样本多样性的思路来自图像生成领域，本文将其创新性地用于轨迹空间的多样性探索
- 状态分解的思路可以推广到其他需要区分可控/不可控状态的生成式规划场景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 双层采样框架和状态分解设计新颖，将树搜索与扩散规划的结合有理论支撑
- **实验充分度**: ⭐⭐⭐⭐ 三个不同任务域的全面评估，消融深入，但缺少更复杂真实场景的验证
- **写作质量**: ⭐⭐⭐⭐ 问题动机阐述清晰，理论与实验衔接自然，但部分符号较密集
- **价值**: ⭐⭐⭐⭐ 解决了扩散规划中测试时引导的核心痛点，零样本能力有实用前景

<!-- RELATED:START -->

## 相关论文

- [Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner](../../ICML2025/image_generation/task-agnostic_pre-training_and_task-guided_fine-tuning_for_versatile_diffusion_p.md)
- [Schrödinger Bridge Matching for Tree-Structured Costs and Entropic Wasserstein Barycentres](schrödinger_bridge_matching_for_tree-structured_costs_and_entropic_wasserstein_b.md)
- [Safe and Stable Control via Lyapunov-Guided Diffusion Models](safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)
- [Guided Diffusion Sampling on Function Spaces with Applications to PDEs](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [KLASS: KL-Guided Fast Inference in Masked Diffusion Models](klass_kl-guided_fast_inference_in_masked_diffusion_models.md)

<!-- RELATED:END -->
