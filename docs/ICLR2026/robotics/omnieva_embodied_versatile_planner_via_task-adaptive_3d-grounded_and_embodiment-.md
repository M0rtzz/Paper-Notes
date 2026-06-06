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

### 任务自适应门控路由器(TAGR)

1. **3D位置编码**：深度图→世界坐标→patch级平均→正弦编码→$V^p \in \mathbb{R}^{N \times H_p \times W_p \times d_v}$

2. **门控决策**：
    - 任务条件：句子Transformer编码指令→$V^T$
    - 场景条件：视觉编码器输出均值池化→$V_{avg}^I$
    - 拼接→MLP→2维gate logits→Gumbel-Softmax→二值决策

3. **动态注入**：
    - Gate=1: $V^{final} = V^I + V^p$ (加3D位置编码)
    - Gate=0: $V^{final} = V^I$ (纯2D)
    - 不同任务/场景自动选择→避免无用3D的噪声

### 具身感知推理

1. **原始技能分解**：
    - Where2Go: 导航目标选择
    - Where2Grasp: 抓取位估计
    - Where2Approach: 接近位姿
    - Where2Fit: 放置适配性

2. **TE-GRPO (Task- and Embodiment-aware GRPO)**：
    - 后训练阶段用GRPO(Group Relative Policy Optimization)
    - 奖励考虑：任务目标 + 物体可供性 + 工作空间边界 + 运动学可行性
    - 确保生成的计划可执行

### 两阶段训练
- Stage 1: 监督微调(SFT)→2D+3D VQA+具身推理数据
- Stage 2: TE-GRPO后训练→强化学习优化可执行性

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
- [\[ACL 2025\] Task-aware MoILE: Hierarchical-Task-Aware Multi-modal Mixture of Incremental LoRA Experts for Embodied Continual Learning](../../ACL2025/robotics/hierarchical-task-aware_multi-modal_mixture_of_incremental_lora_experts_for_embo.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[ICML 2026\] EMBGuard: Constructing Hazard-Aware Guardrails for Safe Planning in Embodied Agents](../../ICML2026/robotics/embguard_constructing_hazard-aware_guardrails_for_safe_planning_in_embodied_agen.md)
- [\[CVPR 2026\] Recurrent Reasoning with Vision-Language Models for Estimating Long-Horizon Embodied Task Progress](../../CVPR2026/robotics/recurrent_reasoning_with_vision-language_models_for_estimating_long-horizon_embo.md)

</div>

<!-- RELATED:END -->
