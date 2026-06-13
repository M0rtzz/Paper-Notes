---
title: >-
  [论文解读] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation
description: >-
  [AAAI2026 Oral][机器人][robotic manipulation] 提出层次化目标驱动框架 H-GAR，通过先预测目标观测再合成中间观测、并利用历史动作记忆库细化粗粒度动作，实现了观测与动作的显式双向交互，在仿真和真实机器人操控任务上取得 SOTA。 统一的视频与动作预测模型在机器人操控领域展现了巨大潜力：…
tags:
  - "AAAI2026 Oral"
  - "机器人"
  - "robotic manipulation"
  - "goal-conditioned planning"
  - "observation-action interaction"
  - "扩散模型"
  - "coarse-to-fine refinement"
---

# H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation

**会议**: AAAI2026 Oral  
**arXiv**: [2511.17079](https://arxiv.org/abs/2511.17079)  
**代码**: 待确认  
**领域**: 机器人  
**关键词**: robotic manipulation, goal-conditioned planning, observation-action interaction, diffusion policy, coarse-to-fine refinement

## 一句话总结

提出层次化目标驱动框架 H-GAR，通过先预测目标观测再合成中间观测、并利用历史动作记忆库细化粗粒度动作，实现了观测与动作的显式双向交互，在仿真和真实机器人操控任务上取得 SOTA。

## 背景与动机

统一的视频与动作预测模型在机器人操控领域展现了巨大潜力：未来观测为规划提供上下文线索，动作序列揭示交互如何影响环境。然而现有方法（如 UVA、PAD、UniPi 等）存在两个根本缺陷：

1. **目标无关的观测生成**：模型在预测未来观测序列时缺乏显式目标语义引导，生成的序列虽然视觉上合理，但与任务目标语义不一致，导致下游规划精度下降。
2. **隐式的观测-动作建模**：观测和动作通常并行生成或仅松耦合，未显式建模二者的因果关系，削弱了时序一致性和适应性。

这两个问题在长时域多阶段操控任务（如抽屉开关+物体放置）中尤为突出，导致关键步骤失败。

## 核心问题

如何在统一视频-动作预测框架中引入**显式的目标锚定**和**结构化的观测-动作双向交互**，使生成的动作既与任务目标语义对齐，又保持时间上的连贯性？

## 方法详解

### 整体架构

H-GAR 采用粗到细的层次化设计，包含四个阶段：

1. **编码阶段**：将历史观测 $\{O_{t-h+1},\dots,O_t\}$ 通过预训练 VAE 编码为 latent tokens，结合 CLIP 编码的文本指令 $T_I$ 和 masked 未来观测，输入 Transformer 编码器得到联合表征 $\mathbf{Z}_{t+1:t+h'}$。
2. **目标预测阶段**：利用视频 diffusion decoder 从最终步的 latent $\mathbf{Z}_{t+h'}$ 生成目标观测 $O_{t+h'}$（即任务完成后的最终视觉状态），同时生成粗粒度动作序列。
3. **Goal-Conditioned Observation Synthesizer (GOS)**：以目标观测 latent 和粗粒度动作 latent 为条件，合成中间观测特征。
4. **Interaction-Aware Action Refiner (IAAR)**：利用中间观测反馈和历史动作记忆库，将粗粒度动作细化为精细、时序一致的动作。

### GOS 模块

GOS 引入可学习查询 $\mathbf{Q}_{\text{Inter}} \in \mathbb{R}^{(h'-1)\times N\times D}$ 来表示中间帧的 latent：

- **自注意力聚合目标信息**：将查询与目标观测 latent $\mathbf{Z}_{t+h'}$ 拼接后做 Self-Attention，使目标语义注入查询。
- **交叉注意力注入动作上下文**：更新后的查询以粗动作 latent $\mathbf{Z}_{t+1:t+h'}$ 为 Key/Value 做 Cross-Attention，再经 FFN 输出中间观测特征 $\mathbf{Z}_{\text{Inter}}$。

这一设计使中间观测同时反映「要到哪里去」（目标）和「怎么过去」（动作）。

### IAAR 模块

IAAR 分两步细化粗动作：

1. **历史动作交互**：以粗动作 latent 作为 Query，Historical Action Memory Bank $\mathcal{H}_t$ 作为 Key/Value 做注意力，注入时序行为先验。
2. **观测反馈细化**：以上一步输出作为 Query，中间观测特征 $\mathbf{Z}_{\text{Inter}}$ 作为 Key/Value 做 Cross-Attention，得到最终精细动作。

### Historical Action Memory Bank

记忆库存储历史精细动作 latent，当超过阈值时采用**相似度驱逐策略**：计算相邻动作 latent 的余弦相似度，合并最相似的一对（取平均），保持记忆多样性。这优于 FIFO 和随机删除策略。

### 训练目标

总损失由四部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{goal}} + \mathcal{L}_{\text{coarse}} + \mathcal{L}_{\text{inter}} + \mathcal{L}_{\text{fine}}$$

各项均为 diffusion denoising loss，分别监督目标观测、粗动作、中间观测和精细动作。训练时对未来观测施加位置一致的随机 mask 以避免信息泄漏，推理时从空白图像开始。

## 实验关键数据

### 仿真实验（成功率）

| 方法 | PushT | PushT-M | Libero-10 |
|------|-------|---------|-----------|
| Diffusion Policy-C | 0.91 | 0.68 | 0.53 |
| UVA | 0.96 | 0.85 | 0.89 |
| PD-VLA | 0.82 | 0.71 | 0.92 |
| **H-GAR** | **0.99** | **0.90** | **0.94** |

三个基准全部 Rank 1，PushT 达到 0.99。

### 真实机器人实验（ALOHA 平台）

| 任务 | H-GAR | UVA | PD-VLA |
|------|-------|-----|--------|
| Object Placement (两阶段) | 9/10 → 8/10 | 7/10 → 6/10 | 8/10 → 7/10 |
| Drawer Manipulation (三阶段) | 7/10 → 6/10 → 6/10 | 6/10 → 5/10 → 3/10 | 6/10 → 6/10 → 4/10 |
| Towel Folding | 8/10 | 5/10 | 6/10 |
| Mouse Arrangement | 6/10 | 3/10 | 4/10 |

长时域多阶段任务优势显著，如 Drawer Manipulation 最终阶段 6/10 vs UVA 3/10。

### 观测生成质量（FVD ↓）

H-GAR 在 8-step 生成下 Libero-10 FVD 49.01（UVA 51.10），Mouse Arrangement FVD 28.43（UVA 32.78），且 FVD 与成功率呈强负相关。

### 关键消融

- GOS + IAAR（含记忆库）完整模型最优；去掉任一模块性能均显著下降。
- 目标帧条件优于随机帧和均匀采样帧策略。
- 记忆库大小 32 为最佳平衡点；相似度驱逐策略优于 FIFO 和随机策略。

## 亮点

1. **层次化粗到细范式**设计优雅：先锚定目标→生成粗动作→合成中间观测→精细化动作，逻辑链清晰。
2. **GOS 和 IAAR 的双向交互**使观测与动作显式耦合，打破了现有方法中二者松耦合的瓶颈。
3. **Historical Action Memory Bank** 配合相似度驱逐策略，简洁有效地编码时序行为先验。
4. **真实机器人验证充分**，4 类任务覆盖短时域/长时域/精细操控，长时域任务优势突出。
5. FVD 与成功率的负相关分析提供了观测生成质量→操控性能的直接证据。

## 局限与展望

1. **领域分类不当**：本文核心是机器人操控/策略学习，而非 object detection，当前分类可能需要重新归类到 robotics。
2. **计算开销**：层次化多阶段设计增加了推理复杂度，论文未讨论推理延迟，对实时控制场景可能存在瓶颈。
3. **目标观测预测的误差传播**：粗到细的级联设计中，若目标观测预测偏差较大，后续 GOS 和 IAAR 可能将错误放大。
4. **记忆库容量固定阈值**：简单的阈值+合并策略可能不适用于极长时域任务，自适应记忆管理值得探索。
5. **泛化性验证有限**：仅在 ALOHA 平台测试，未涉及灵巧手或移动操控等更复杂场景。

## 与相关工作的对比

| 方法 | 目标锚定 | 观测-动作交互 | 粗到细 | 记忆机制 |
|------|---------|-------------|--------|---------|
| Diffusion Policy | ✗ | ✗ | ✗ | ✗ |
| UniPi | ✗ | 隐式 | ✗ | ✗ |
| UVA | ✗ | 联合优化 | ✗ | ✗ |
| PAD | ✗ | 联合去噪 | ✗ | ✗ |
| LBP | 隐式 latent goal | ✗ | ✗ | ✗ |
| **H-GAR** | **显式目标观测** | **GOS+IAAR 双向显式** | **✓** | **Historical Memory Bank** |

H-GAR 是首个将显式目标观测锚定、双向观测-动作交互和历史动作记忆统一在一个层次化框架中的方法。

## 启发与关联

1. **粗到细的级联思路**可迁移到其他序列决策问题（如自动驾驶轨迹规划）：先预测终态作为锚点，再逐步细化。
2. **相似度驱逐的记忆管理策略**简单但有效，可用于需要维护有限大小历史缓冲区的任何在线学习场景。
3. 观测生成质量与下游任务性能的相关性分析思路，可用于评估视频生成在其他具身任务（如导航）中的作用。
4. 与 idea 方向关联：若将 GOS 替换为 3D 场景表示生成器，可能在 3D 操控规划中获得更强的空间推理能力。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 层次化目标驱动+双向交互设计新颖，但个别模块（如 diffusion decoder、cross-attention）为标准组件
- 实验充分度: ⭐⭐⭐⭐⭐ — 仿真+真实机器人+消融+可视化+相关性分析，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、图表丰富，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 对机器人操控中的视频-动作联合建模提供了有效的层次化解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](../../NeurIPS2025/robotics/human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)

</div>

<!-- RELATED:END -->
