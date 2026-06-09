---
title: >-
  [论文解读] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
description: >-
  [ICLR 2026][机器人][MLLM] 提出OmniEVA——通过任务自适应门控路由器动态注入3D位置编码(仅在需要时启用几何推理)和具身感知推理框架(将物理约束融入规划循环),解决了空间MLLM的两大gap：几何适应性差(2D-only或硬编码3D)和具身约束缺失(理论可行但实际不可执行的计划)…
tags:
  - "ICLR 2026"
  - "机器人"
  - "MLLM"
  - "任务自适应3D接地"
  - "门控路由"
  - "具身感知推理"
  - "GRPO"
---

# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning

**会议**: ICLR 2026  
**arXiv**: [2509.09332](https://arxiv.org/abs/2509.09332)  
**代码**: [项目页面](https://github.com/OmniEVA-Project)  
**领域**: 具身智能/3D推理  
**关键词**: MLLM, 任务自适应3D接地, 门控路由, 具身感知推理, GRPO

## 一句话总结
提出OmniEVA——通过任务自适应门控路由器动态注入3D位置编码(仅在需要时启用几何推理)和具身感知推理框架(将物理约束融入规划循环),解决了空间MLLM的两大gap：几何适应性差(2D-only或硬编码3D)和具身约束缺失(理论可行但实际不可执行的计划),在8个基准中7个达到SOTA。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：MLLM用于具身智能→空间理解+推理+行动。两条路线：(1) 2D RGB直接输入→缺3D信息; (2) 3D-LLM硬编码3D注入→不灵活。

**现有痛点**：

### 现有痛点

**现有痛点**：(1) **几何适应性gap**：2D-only模型在3D推理任务(堆叠/遮挡处理/导航)失败; 3D-LLM硬编码注入→3D输入嘈杂或不必要时反而引入噪声

### 核心矛盾

**核心矛盾**：(2) **具身约束gap**：网络图像/视频训练的模型忽略机器人物理约束→计划理论可行但物理无法执行(抓取位/工作空间/运动学)

**切入角度**：(1) 门控路由器动态决定是否需要3D → 按需注入; (2) TE-GRPO训练让模型学习尊重物理约束。

## 方法详解

### 整体框架

OmniEVA 在一个 MLLM 主干之上挂两个互补部件：前端的任务自适应门控路由器(TAGR)决定要不要把 3D 几何信息喂给模型，后端的具身感知推理把机器人物理约束写进训练奖励里。整套系统经过 SFT 加 TE-GRPO 两阶段训练，使模型既能在需要时做几何推理，又只产出物理上能执行的计划。

### 关键设计

**1. 任务自适应门控路由器(TAGR)：让模型自己决定何时需要 3D。** 已有做法要么纯 2D 输入丢掉几何信息，要么硬编码地把 3D 注入每一次推理——后者在堆叠、遮挡这类任务上有用，但在不需要几何的 2D 任务上反而引入噪声。TAGR 把"是否注入 3D"变成一个可学习的二值开关。它先从深度图重建世界坐标，按 patch 做平均后过正弦编码，得到 3D 位置编码 $V^p \in \mathbb{R}^{N \times H_p \times W_p \times d_v}$；同时用句子 Transformer 编码指令得到任务条件 $V^T$，用视觉编码器输出均值池化得到场景条件 $V_{avg}^I$。两个条件拼接后过 MLP 得到二维 gate logits，再用 Gumbel-Softmax 转成可微的二值决策。门控开($=1$)时注入几何 $V^{final} = V^I + V^p$，门控关($=0$)时保持纯 2D $V^{final} = V^I$。因为开关由当前任务和场景共同决定，模型能在导航、抓取这类几何敏感任务上启用 3D，而在纯 2D 推理上自动关闭，避免无用几何带来的退化。

**2. 具身感知推理：把物理可行性拆成四个原始技能并写进奖励。** 在网络图像/视频上训练的模型常输出理论可行但机器人做不到的计划，因为它不感知抓取位、工作空间边界和运动学约束。OmniEVA 先把具身规划分解成四个可独立评估的原始技能——Where2Go(导航目标选择)、Where2Grasp(抓取位估计)、Where2Approach(接近位姿)、Where2Fit(放置适配性)，让"可执行性"变得可度量。在此基础上提出 TE-GRPO(Task- and Embodiment-aware GRPO)：在后训练阶段用 GRPO(Group Relative Policy Optimization)做强化学习，但奖励同时考虑任务目标、物体可供性、工作空间边界与运动学可行性。这样模型不只追求"答对",还被持续约束去尊重机器人本体的物理限制，把可执行计划比例从纯 SFT 的约 65% 提到约 90%。

### 损失函数 / 训练策略

整体走两阶段。第一阶段做监督微调(SFT)，混合 2D 与 3D 的 VQA 以及具身推理数据，让模型同时具备几何理解和指令跟随的基础能力；第二阶段切到 TE-GRPO 后训练，用上面带物理约束的奖励做强化学习，专门优化计划的可执行性。两阶段配合让"会推理"和"能落地"这两件事分别由 SFT 和 RL 负责。

## 实验关键数据

### 8个基准(2D + 3D + 视频)


### 主实验

| 基准类型 | 模型 | 性能 |
|---------|------|------|
| 2D空间推理 | OmniEVA | **SOTA** |
| 3D空间推理 | OmniEVA | **SOTA (7/8)** |
| 目标导航(HM3D) | OmniEVA | **排行榜第一** |
| 目标导航(MP3D) | OmniEVA | **排行榜第一** |

### 4个原始技能基准


### 消融实验

| 技能 | OmniEVA vs SOTA | 说明 |
|------|----------------|------|
| Where2Go | +5% | 导航目标选择 |
| Where2Grasp | +8% | 抓取位估计 |
| Where2Approach | +6% | 接近策略 |
| Where2Fit | +7% | 放置适配 |

### 关键发现
- 门控路由器在~40%任务上选择关闭3D→这些任务确实不需要3D→验证了自适应策略
- 硬编码3D in baseline模型→在不需要3D的2D任务上反而降低性能→证明了TAGR的价值
- TE-GRPO后训练比纯SFT→可执行计划比例从~65%→~90%

## 亮点与洞察
- **"按需3D"的设计哲学**：不是"给所有任务都加3D"→而是让模型自己学习何时需要→这比人工规则更灵活更准确。
- **原始技能基准的贡献**：4个新基准(Where2Go/Grasp/Approach/Fit)→首次系统评估具身计划的可执行性。
- **TE-GRPO连接了LLM训练和机器人学**：将GRPO(LLM后训练主流方法)与物理约束奖励结合→是LLM-for-robotics的自然且有效的融合方式。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 任务自适应3D+具身感知推理的双重创新
- 实验充分度: ⭐⭐⭐⭐⭐ 8+4基准+消融+排行榜
- 写作质量: ⭐⭐⭐⭐ 架构描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对具身MLLM有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](../../NeurIPS2025/robotics/mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ACL 2025\] Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](../../ACL2025/robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](../../CVPR2026/robotics/recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)

</div>

<!-- RELATED:END -->
