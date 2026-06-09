---
title: >-
  [论文解读] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning
description: >-
  [ICLR 2026][图像生成][offline RL] 提出 Single-Step Completion Policy (SSCP)，通过在流匹配框架中预测"完成向量"（从任意中间状态到目标动作的归一化方向），将多步生成策略压缩为单步推理…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "offline RL"
  - "flow matching"
  - "single-step generation"
  - "completion vector"
  - "policy learning"
  - "D4RL"
---

# SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning

**会议**: ICLR 2026  
**arXiv**: [2506.21427](https://arxiv.org/abs/2506.21427)  
**代码**: [GitHub](https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy)  
**领域**: 图像生成  
**关键词**: offline RL, flow matching, single-step generation, completion vector, policy learning, D4RL

## 一句话总结
提出 Single-Step Completion Policy (SSCP)，通过在流匹配框架中预测"完成向量"（从任意中间状态到目标动作的归一化方向），将多步生成策略压缩为单步推理，在 D4RL 上与多步扩散/流策略持平但训练快 64×、推理快 4.7×，并扩展到 GCRL 中将层级策略扁平化。

## 研究背景与动机
**领域现状**：扩散/流匹配生成策略在离线 RL 中因能捕捉多模态动作分布而表现优异（DQL、CAC 等）。但它们需要数十步迭代采样，推理延迟高。

**现有痛点**：
   - **推理效率**：扩散策略每步需 5-50 次去噪，不适合实时控制（DQL ~1.27ms vs 确定性策略 ~0.1ms）
   - **训练不稳定**：将策略梯度通过多步采样链反传（BPTT）导致梯度不稳定、训练耗时（DQL ~8 小时 vs TD3+BC ~30 分钟）
   - **Shortcut 方法的 bootstrap 问题**：Frans et al. 2024 提出的 shortcut 模型用自身预测作为训练目标（self-consistency loss），在 RL 等动态目标场景中不稳定

**核心矛盾**：生成策略的表达力 vs 推理/训练效率不可兼得？

**核心 idea**：在流匹配的中间时间步 $\tau$ 处，预测直接到达目标 $x_1$ 的完成向量（而非速度场），用真实数据监督（非 bootstrap），实现单步生成

## 方法详解

### 整体框架
SSCP 想解决的是「生成策略表达力强、但推理要迭代几十步」这对矛盾。它的出发点是标准流匹配：在噪声 $z$ 和目标动作 $x_1$ 之间走一条线性插值路径 $x_\tau = (1-\tau)z + \tau x_1$，时间 $\tau$ 从 0 到 1。常规流策略学的是每一点的瞬时速度场，推理时要沿这条路径一小步一小步积分。

SSCP 的关键转变是：在路径上的任意中间点 $x_\tau$，不只学瞬时速度，还额外学一个「完成向量」——从当前点直接指向终点 $x_1$ 的方向。训练时模型同时拟合两个量：瞬时速度 $h_\theta(x_\tau,\tau,d{=}0)$（用标准流损失约束），以及完成向量 $h_\theta(x_\tau,\tau,d{=}1{-}\tau)$（用真实 $x_1$ 直接监督）。推理时直接从纯噪声 $z$ 出发，令剩余跨度 $d{=}1$，一步到位输出动作 $\pi_\theta(s)=z+h_\theta(z,s,0,1)$。这样既保留了流匹配捕捉多模态分布的表达力，又把采样从几十步压到一步。

### 关键设计

**1. 完成向量（Completion Vector）：让中间点一步跳到终点，且用真值而非自预测监督。**

流策略推理慢的根源是只知道局部速度、必须积分。完成向量直接回答「从 $x_\tau$ 怎样一步到 $x_1$」：它是从当前点到目标的归一化方向，乘以剩余跨度 $1{-}\tau$ 就能还原终点 $\hat{x}_1 = x_\tau + h_\theta(x_\tau, \tau, 1{-}\tau) \cdot (1{-}\tau)$，训练目标就是让这个还原值贴近真实动作：

$$\mathcal{L}_{completion} = \mathbb{E}\big[\|x_\tau + h_\theta(x_\tau, \tau, 1{-}\tau)(1{-}\tau) - x_1\|^2\big]$$

这里最要紧的区别在监督信号：Frans et al. 2024 的 shortcut 模型把模型自身的预测当作训练目标（self-consistency loss / bootstrap），在 RL 这种目标动态变化的场景里会自我累积误差、训练不稳；而完成损失用的是数据集里真实的 $x_1$，是固定的真值监督，从根上避开了自一致性带来的不稳定。之所以敢这么直接回归，是因为 RL 的动作空间维度通常很低（<20 维），完成向量的回归难度远小于高维图像生成——这也是该思路在控制任务里好用、却未必能照搬到图像生成的关键前提。

**2. 三目标联合训练（SSCQL）：流损失保表达力、完成损失保单步质量、Q 损失管价值。**

光有完成向量还不够，离线 RL 还要兼顾分布匹配和动作价值，所以 actor 的目标是三项相加：

$$\mathcal{L}_\pi = \alpha_1 \mathcal{L}_{flow} + \alpha_2 \mathcal{L}_{completion} + \mathcal{L}_{\pi_Q}$$

其中流损失 $\mathcal{L}_{flow}$ 约束瞬时速度场，维持流匹配对多模态分布的拟合能力；完成损失 $\mathcal{L}_{completion}$ 约束单步生成的动作质量，同时起到行为约束（BC 正则化）的作用，把策略拉回数据分布；$\mathcal{L}_{\pi_Q}$ 是 Q-learning 策略梯度，负责把动作朝高价值方向优化。三者缺一不可——少了流损失表达力垮，少了完成损失单步质量垮。Critic 侧则是标准的 twin Q-learning 加 target network。

**3. 单步推理：一次前向就出动作，确定性中保留多模态。**

推理时把时间设到起点、剩余跨度设满，即 $\tau{=}0,\,d{=}1$，模型从纯噪声一步给出动作 $\pi_\theta(s) = z + h_\theta(z, s, 0, 1)$。这只是单次前向传播，速度和 TD3+BC 这类确定性策略相当。值得注意的是，它对固定的输入噪声 $z$ 是确定性输出，但对不同的 $z$ 采样会落到不同的模态，因此整体上仍然能表达多模态动作分布——既快又没丢掉生成策略的核心优势。

**4. Goal-Conditioned 扩展（GC-SSCP）：把完成的思想从「压缩生成步骤」推广到「压缩决策层级」。**

完成模型的用途不止压缩采样步数。在目标条件 RL（GCRL）里，HIQL 这类方法用高层 + 低层两级策略协作决策。GC-SSCP 把同样的完成思路套上去：训练一个扁平的单层策略，让它的输出去匹配层级策略组合后的结果，推理时一步直接决策。可以这样类比——SSCP 把多步流生成压缩成单步，GC-SSCP 则把多层级决策压缩成单层，二者都是「用完成模型把一串过程折叠成一次预测」。

### 损失函数 / 训练策略
Actor 用 $\alpha_1 \mathcal{L}_{flow} + \alpha_2 \mathcal{L}_{completion} + \mathcal{L}_{\pi_Q}$ 三项联合优化，Critic 用 twin Q-learning + target network 软更新。优化器为 Adam，batch size 256，整体训练约 16 分钟（DQL 约 8 小时）。

## 实验关键数据

### D4RL 离线 RL 主实验

| 方法 | 类型 | D4RL 平均(9任务) | 训练时间 | 推理延迟 | 去噪步数 |
|------|------|----------------|---------|---------|---------|
| DQL | 扩散策略 | 87.9 | ~8h | 1.27ms | 5 |
| CAC | 流策略 | 85.1 | ~5h | 0.85ms | 2 |
| TD3+BC | 确定性 | 85.2 | ~30min | 0.08ms | 1 |
| **SSCQL** | **单步完成** | **87.9** | **~16min** | **0.27ms** | **1** |

SSCQL 与最强扩散基线 DQL 持平，但**训练快 64×、推理快 4.7×**。

### 离线到在线微调

| 方法 | 稳定性 | 说明 |
|------|--------|------|
| DQL | 经常退化（>10%） | 多步采样链导致微调不稳定 |
| CAC | 经常退化 | 同上 |
| Cal-QL | 稳定 | 专为 O2O 设计的 SOTA |
| **SSCQL** | **稳定提升** | 单步避免了 BPTT 不稳定性 |

### 在线 RL

| 方法 | HalfCheetah | Hopper | Walker2d |
|------|-------------|--------|----------|
| DQL | 较差 | 较差 | 较差 |
| CAC | 较差 | 较差 | 较差 |
| **SSCQL** | **最优** | **最优** | **最优** |

### Goal-Conditioned RL (OGBench)

GC-SSCP（扁平策略）平均超越 HIQL（层级策略），说明完成模型成功将层级结构压缩为扁平决策。

### 关键发现
- 动作空间低维（<20 维）使完成向量的直接回归可行——这是 SSCP 在 RL 中有效但可能在图像生成中不适用的关键原因
- 流损失和完成损失缺一不可：流损失保证表达力，完成损失保证单步质量
- 多步扩散/流策略在 O2O 微调和在线 RL 中不稳定——BPTT 是罪魁祸首
- GC-SSCP 展示了完成模型在策略压缩（不仅是生成步骤压缩）中的更广泛应用

## 亮点与洞察
- **真值监督替代 bootstrap**是最核心的创新：bootstrap 的自一致性损失在 RL 动态目标场景中不可靠，而完成向量可以用数据直接监督。简单但关键
- **64× 训练加速 + 4.7× 推理加速**同时保持等价性能——这使流策略在实时控制中变得可行
- **从生成压缩到决策压缩**：SSCP → GC-SSCP 的扩展展示了完成模型的通用性——不仅压缩采样步骤，还能压缩决策层级

## 局限与展望
- $\alpha_1, \alpha_2$ 平衡系数需要调优，不同任务可能需要不同设置
- 早期 $\tau$ 处的完成预测可能不准确（噪声大、信息少），理论分析缺失
- 仅在 MuJoCo 连续控制任务上验证，未在高维动作空间（如机器人操作、自动驾驶）上测试
- 与蒸馏方法（如 consistency model）的对比缺失

## 相关工作与启发
- **vs DQL (Wang et al. 2022)**：DQL 用扩散策略 + DDPG+BC，需 5 步去噪；SSCQL 用完成策略，1 步，性能等价但快 64×
- **vs Shortcut Models (Frans et al. 2024)**：shortcut 用 bootstrap 自一致性目标，不稳定；SSCP 用真值完成向量，稳定
- **vs CAC**：CAC 用流匹配 + 2 步去噪 + 一致性蒸馏；SSCP 更简单直接，无需蒸馏

## 评分
- 新颖性: ⭐⭐⭐⭐ 完成向量替代 bootstrap 的思路简单但有效，真值监督的洞察有价值
- 实验充分度: ⭐⭐⭐⭐⭐ D4RL + O2O + Online + BC + GCRL，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 清晰的渐进式展开
- 价值: ⭐⭐⭐⭐⭐ 让生成策略在实时控制中变得可行，64× 训练加速有巨大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](../../AAAI2026/image_generation/efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)
- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[NeurIPS 2025\] Scaling Offline RL via Efficient and Expressive Shortcut Models](../../NeurIPS2025/image_generation/scaling_offline_rl_via_efficient_and_expressive_shortcut_models.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
