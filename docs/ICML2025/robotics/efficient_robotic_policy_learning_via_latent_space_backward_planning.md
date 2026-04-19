---
title: >-
  [论文解读] Efficient Robotic Policy Learning via Latent Space Backward Planning
description: >-
  [ICML 2025][机器人][机器人规划] 提出潜在空间反向规划（LBP），从最终目标出发递归预测越来越接近当前状态的中间子目标，在保持任务对齐的同时大幅提升规划效率，在 LIBERO-LONG 仿真和真实机器人长时域任务上达到 SOTA。
tags:
  - ICML 2025
  - 机器人
  - 机器人规划
  - 潜在空间
  - 反向规划
  - 子目标预测
  - 长时域操作
---

# Efficient Robotic Policy Learning via Latent Space Backward Planning

**会议**: ICML 2025  
**arXiv**: [2505.06861](https://arxiv.org/abs/2505.06861)  
**代码**: [Project Page](https://lbp-authors.github.io)  
**领域**: 机器人  
**关键词**: 机器人规划, 潜在空间, 反向规划, 子目标预测, 长时域操作

## 一句话总结

提出潜在空间反向规划（LBP），从最终目标出发递归预测越来越接近当前状态的中间子目标，在保持任务对齐的同时大幅提升规划效率，在 LIBERO-LONG 仿真和真实机器人长时域任务上达到 SOTA。

## 研究背景与动机

机器人规划面临一个根本性的"三难困境"（trilemma）：**效率、精度、充分的未来引导**难以兼得。

现有方法分为两大类，各有致命缺陷：

**视频规划方法**（UniPi, HiP, Seer, GR-1 等）：逐帧预测未来图像作为策略引导。虽然能提供丰富的未来信息，但计算成本极高，且误差沿时序持续累积，生成物理不一致的帧会误导下游策略。

**粗粒度子目标规划方法**（SuSIE, MimicPlay 等）：仅预测稀疏的中间子目标，提高了效率。但它们采用**前向规划**范式——从当前状态逐步向前预测子目标，容易因累积误差导致偏离最终任务目标（off-task）。已有方法引入可达性/最优性检查来纠偏，但这只是事后补救，增加了复杂度而无法从根本上解决问题。

核心问题：能否在长时域多阶段任务中同时做到**高效**和**准确**的规划？

LBP 的回答是：模拟人类规划的思维方式——先想象最终结果，再逆向分解为逐渐可操作的小目标。

## 方法详解

### 整体框架

LBP（Latent Space Backward Planning）包含三个核心模块：

1. **潜在目标预测器**（Latent Goal Predictor $f_g$）：将任务语言描述 + 当前观测映射到最终目标的潜在空间表征 $z_g$
2. **反向子目标预测器**（Backward Subgoal Predictor $f_w$）：从 $z_g$ 出发，递归地预测时间上越来越接近当前状态的中间子目标 $w_1, w_2, \ldots, w_n$
3. **子目标融合策略**（Goal-Fusion Policy $\pi$）：通过 Perceiver 风格的交叉注意力自适应地融合子目标信息，指导动作生成

整个规划在**潜在空间**中进行（使用 DecisionNCE 或 SigLIP 编码器），而非像素空间，大幅降低计算量。

### 关键设计

#### 1. 潜在目标预测（Grounding Task Objective as Latent Goals）

语言指令在长时域任务中通常退化为任务标识符，缺乏细粒度的引导能力。LBP 学习一个目标预测模型 $f_g$，将当前状态 $z_t$ 和语言特征 $\phi_l$ 映射到最终潜在目标 $z_g$：

$$\max_{f_g} \sum_{\tau \in \mathcal{D}_z} \sum_{1 \leq t \leq H} \mathbb{E}_{p(z_g, \phi_l | \tau)} \log f_g(z_g | z_t, \phi_l)$$

这样的目标不是固定的，而是**依据当前场景动态生成**的——例如"把棕色杯子放到白色杯子前面"这一指令，最终目标状态取决于白色杯子的实际位置。

#### 2. 反向递归子目标预测（Backward Subgoal Prediction）

这是 LBP 最核心的创新。与传统前向规划不同，LBP 采用**从目标到当前状态**的反向规划：

- **第一步**：以 $z_g$（最终目标）为锚点预测第一个子目标 $w_1$，它在时间上接近最终目标
- **递归**：每个后续子目标 $w_i$ 从前一级子目标 $w_{i-1}$ 预测得到，离当前状态越来越近

所有层级的子目标预测器可以**共享同一个统一模型** $f_w$，因为它们结构相同。定义递归规划系数 $\lambda = \frac{\Gamma(w_i) - t}{\Gamma(w_{i-1}) - t}$，统一目标为：

$$\max_{f_w} \sum_{\tau \in \mathcal{D}_z} \sum_{1 \leq t < H} \mathbb{E} \left[ \sum_{i=1}^{n} \log f_w(z_{\lambda_i} | z_t, z_{\lambda_{i-1}}, \phi_l) \right]$$

训练目标包含两个项：

- **真值监督项**：用轨迹中的真实子目标作为监督
- **自洽项**：以 $f_w$ 自身的预测作为输入进行监督，确保测试时递归推理的一致性

**反向规划的三大优势**：

- 子目标序列覆盖整个任务时域，提供**从粗到细**的非均匀时间采样
- 以最终目标为锚点保证任务对齐，**减少累积误差**
- 递归式预测减少规划步数，提升**计算效率**

#### 3. 子目标融合（Goal-Fusion Module）

子目标序列 $c = \{w_n, \ldots, w_1, z_g, \phi_l\} \in \mathbb{R}^{(n+2) \times N_z}$ 维度过高，直接拼接会加重策略学习负担。LBP 引入 Perceiver 风格的交叉注意力：

- 用一个可学习的查询向量 $z \in \mathbb{R}^{1 \times N_z}$ 对子目标序列做交叉注意力
- 输出压缩后的上下文嵌入 $z_c$
- 自适应地从不同距离的子目标中提取最相关的信息

这使策略能够**动态平衡短期与长期引导**：大幅移动时更关注远距离子目标以避免阻碍未来进展，精细操作时更关注近距离子目标。

### 损失函数 / 训练策略

LBP 采用**三阶段分别训练**：

1. **目标预测器 $f_g$**：两层 MLP，最大似然估计（Eq. 2）
2. **子目标预测器 $f_w$**：两层 MLP，带自洽正则化的最大似然（Eq. 5）
3. **低层策略 $\pi$**：ResNet-34 主干 + FiLM 语言注入 + 残差 MLP，使用**扩散损失**（denoising step=25），动作分块长度为 6

训练超参：高层规划器 batch=64, 100k steps；低层策略 batch=64/128, 200k/400k steps（仿真/真实）。

默认设置：3 步规划（最终目标 + 2 个中间子目标），$\lambda = 0.5$。

## 实验关键数据

### 主实验

在 LIBERO-LONG（10 个长时域机器人操作任务，每任务 50 条专家示范）上的结果：

| 方法 | 类型 | Avg. Success (%) | 相对 LBP |
|------|------|:---:|:---:|
| MTACT | 多任务策略 | 41.0 | -47.6 |
| OpenVLA | 大模型策略 | 54.0 | -34.6 |
| MVP | 预训练表征 | 68.2 | -20.4 |
| MPI | 交互表征 | 77.3 | -11.3 |
| Seer | 视频规划 | 78.6 | -10.0 |
| SuSIE | 图像编辑子目标 | 76.3 | -12.3 |
| **LBP (SigLIP)** | 潜在反向规划 | 85.0 | -3.6 |
| **LBP (DecisionNCE)** | 潜在反向规划 | **88.6** | — |

真实机器人实验（AIRBOT 6DoF，4 个长时域任务，200 条示范）：

| 任务 | LCBC | GLCBC | SuSIE | **LBP** |
|------|:---:|:---:|:---:|:---:|
| Stack 3 cups (Avg. Score) | 78.7 | 84.6 | 60.4 | **84.6** |
| Move cups (Avg. Score) | 60.4 | 62.9 | 46.2 | **77.9** |
| Stack 4 cups (Avg. Score) | 55.0 | 45.5 | 42.5 | **72.5** |
| Shift cups (Avg. Score) | 41.8 | 36.1 | 17.7 | **67.1** |

关键发现：**任务越长（阶段越多），LBP 的优势越明显**。在最难的 Shift cups（5 阶段）中，LBP 在最终阶段得分 26.6，而所有 baseline 为 0。

### 消融实验

| 配置 | Avg. Success (%) | 说明 |
|------|:---:|------|
| 无目标无子目标（LCBC） | 77.3 | 仅语言条件 |
| 仅最终目标 $z_g$ | 83.3 | +6.0%，视觉目标有效 |
| $z_g$ + 1 个子目标 ($\lambda=0.5$) | 85.6 | +2.3%，子目标进一步提升 |
| $z_g$ + 2 个子目标 ($\lambda=0.5$) | **88.6** | 最优配置 |
| $z_g$ + 3 个子目标 ($\lambda=0.5$) | 83.0 | 过多子目标反而下降 |
| Goal-fusion → average pooling | 79.0 | -9.6%，朴素池化大幅损害性能 |
| 前向规划 vs 反向规划 | — | 前向规划 MSE 在远距离子目标处急剧增大，反向规划全程保持低误差 |

### 关键发现

1. **反向规划 vs 前向规划**：在 3000 个采样点上对比子目标预测 MSE，前向规划的误差随子目标距离增大而急剧膨胀（尤其在最难的 Shift Cups 任务），而反向规划全程保持低误差
2. **并行规划 vs 反向规划**：并行预测虽然没有误差累积，但由于需要同时监督所有子目标，导致全程预测精度偏低
3. **子目标数量**：2 个中间子目标最优，过多反而降低性能——体现了 LBP 的高效性
4. **$\lambda$ 鲁棒性**：$\lambda = 0.5$ 和 $\lambda = 0.75$ 结果相近，框架对该超参数不敏感
5. **泛化性**：在 Shift Cups 任务上面对干扰物品和不同背景，LBP 仍显著优于 LCBC 基线

## 亮点与洞察

1. **反向规划范式的启发式价值**：从终点规划到起点，这一思路简洁而深刻，类似人类"以终为始"的思维方式。与前向规划相比，反向规划天然避免了累积误差导致的 off-task 问题
2. **轻量级实现**：目标预测器和子目标预测器都只是两层 MLP，无需像 SuSIE 那样训练重型图像编辑扩散模型，也无需像 Seer 那样做逐帧视频预测
3. **统一子目标预测器**：不同层级的子目标共享同一模型 $f_w$，参数高效且训练简洁
4. **非均匀时间采样**：子目标序列天然形成近密远疏的分布——近期提供精确操作引导，远期保持任务对齐，这比均匀采样更符合实际需求
5. **Goal-fusion 的必要性**：消融显示，简单平均池化导致 9.6% 的性能下降，说明自适应融合不同距离的子目标至关重要

## 局限与展望

1. **子目标选择机制**：当前使用固定的 $\lambda$ 做均匀递归，未来可结合**关键帧检测**方法自适应选择信息量最大的子目标时间点
2. **潜在空间质量**：LBP 依赖预训练编码器（DecisionNCE / SigLIP）的表征质量，更好的机器人专用编码器可能进一步提升性能
3. **真实世界复杂度**：真实实验仅涉及杯子操作（pick-and-place），对更复杂的操作（工具使用、柔性物体）未验证
4. **训练效率**：三阶段分别训练，未探索端到端联合训练的可能性
5. **场景泛化**：只测了桌面操作场景，跨任务/跨环境的泛化能力有待验证

## 相关工作与启发

- **SuSIE**（Black et al., ICLR 2024）：图像编辑扩散模型生成子目标图像，在像素空间操作导致计算成本高且容易产生幻觉
- **Seer**（Tian et al., ICLR 2025）：端到端预测性逆动力学模型，联合预测动作和未来帧
- **DecisionNCE**（Li et al., ICML 2024）：通过隐式偏好学习构建体现多模态表征的潜在空间
- **Diffusion Policy**（Chi et al., RSS 2023）：扩散模型建模动作分布，LBP 的低层策略采用此方案
- **Perceiver**（Jaegle et al., ICML 2021）：跨模态注意力架构，启发了 LBP 的 goal-fusion 模块

## 评分

| 维度 | 分数 (1-5) | 评语 |
|------|:---:|------|
| 新颖性 | 4 | 反向规划思路新颖，统一子目标预测器设计精巧 |
| 技术深度 | 4 | 理论分析清晰，递归自洽训练设计合理 |
| 实验充分度 | 4 | 仿真+真实机器人，消融全面，前向/并行对比有说服力 |
| 写作质量 | 4 | 动机清晰，图示直观，逻辑连贯 |
| 实用价值 | 4 | 轻量 MLP 实现，适合实时部署 |
| **总分** | **4.0** | 一篇解决实际问题的扎实工作，反向规划范式值得在更多场景推广 |

<!-- RELATED:START -->

## 相关论文

- [Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [Domain Expansion: A Latent Space Construction Framework for Multi-Task Learning](../../ICLR2026/robotics/domain_expansion_a_latent_space_construction_framework_for_multi-task_learning.md)
- [Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](../../ICCV2025/robotics/resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)
- [ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](../../NeurIPS2025/robotics/thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)

<!-- RELATED:END -->
