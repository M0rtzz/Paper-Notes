---
title: >-
  [论文解读] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning
description: >-
  [NeurIPS 2025][机器人][VLA safety] 首次系统性地将安全强化学习（SafeRL）的 CMDP 框架应用于视觉-语言-动作模型（VLA）的安全对齐，通过建模-激发-约束-保证四阶段集成安全方法（ISA）…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "VLA safety"
  - "constrained MDP"
  - "强化学习"
  - "embodied AI"
  - "robot safety"
---

# SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning

**会议**: NeurIPS 2025  
**arXiv**: [2503.03480](https://arxiv.org/abs/2503.03480)  
**代码**: [项目主页](https://pku-safevla.github.io)  
**领域**: LLM对齐  
**关键词**: VLA safety, constrained MDP, safe reinforcement learning, embodied AI, robot safety

## 一句话总结
首次系统性地将安全强化学习（SafeRL）的 CMDP 框架应用于视觉-语言-动作模型（VLA）的安全对齐，通过建模-激发-约束-保证四阶段集成安全方法（ISA），在移动操作任务上实现 83.58% 的安全违规成本下降同时保持任务性能（+3.85%）。

## 研究背景与动机

**VLA 模型的安全挑战**：VLA 模型（如 RT-2、OpenVLA）正在成为通用机器人策略，但在真实世界部署中面临对环境、机器人自身和人类的物理安全风险

**现有方法的不足**：LLM/VLM 的安全对齐方法（RLHF 等）聚焦于抽象意图级安全，无法直接应用于物理世界安全约束；现有 VLA 训练（IL 或标准 RL）缺乏显式安全约束机制

**核心问题**：如何在不损失性能的前提下将安全约束显式集成到 VLA 中？

## 方法详解

### 整体框架：集成安全方法（ISA）

ISA 包含四个相互关联的阶段：Modeling（建模安全需求）→ Eliciting（激发不安全行为）→ Constraining（SafeRL 约束策略）→ Assuring（针对性安全评估）

### 问题建模：CMDP 框架

将 VLA 安全对齐形式化为约束马尔可夫决策过程（CMDP），VLA 策略将观测历史映射到动作，同时约束安全成本不超过阈值：

$$\pi^* = \arg\max_{\pi_\theta \in \Pi_\mathcal{C}} \mathcal{J}(\pi_\theta)$$

定义了5类安全关键组件：
- **Corners**：狭窄角落导致卡死或反复碰撞
- **Blind Spots**：短期空间感知失败与已见但不可见障碍物碰撞
- **Fragile Collections**：操作过程中对附近易碎物品造成附带损害
- **Critical Points**：间接动作使不稳定物品掉落
- **Dangerous Equipment**：与危险设备的禁止交互

### 风险激发：Safety-CHORES 基准

利用 ProcTHOR 生成 15 万个多样化室内场景，结合 Objaverse 提供 80 万个 3D 资产，在 AI2THOR 模拟器中系统化激发安全违规。

### 约束策略学习：Lagrangian 方法

通过 Lagrangian 松弛将约束优化转化为无约束的 min-max 问题：

$$\min_\theta \max_{\lambda \geq 0} [-\mathcal{J}_r(\theta) + \sum_{i=0}^n \lambda_i \mathcal{J}_{c_i}(\theta)]$$

动态 Lagrange 乘子自适应平衡奖励与成本目标，先优化安全、再最大化任务性能。

### 安全保证

从三个维度评估：测试时安全（标准测试集+OOD）、长尾安全（低频危险事件）、极端失败安全（任务不可完成时行为）。

## 实验关键数据

### 主实验：ISA vs 基线方法

| 方法 | ObjNav SR↑/CC↓ | PickUp SR↑/CC↓ | Fetch SR↑/CC↓ |
|------|----------------|----------------|---------------|
| SPOC-DINOv2 | 0.43/13.50 | 0.86/10.29 | 0.14/13.97 |
| FLaRe | 0.822/12.36 | 0.912/7.08 | 0.605/43.36 |
| **ISA** | **0.865/1.85** | **0.928/0.37** | **0.637/8.08** |

CC 平均下降 **83.58%**，SR 平均提升 **+3.85%**。

### OOD 鲁棒性

| 扰动 | ObjNav CC | PickUp CC | Fetch CC |
|------|-----------|-----------|----------|
| 无扰动 | 1.85 | 0.37 | 8.98 |
| +Color | 3.10 | 1.82 | 15.34 |
| +All | 3.21 | 0.41 | 12.50 |

安全行为在 OOD 下表现稳定。

### 极端失败场景

| 方法 | 平均 CC |
|------|---------|
| FLaRe | 71.68 |
| SPOC | 14.63 |
| **ISA** | **2.20** |

即使任务完全失败（SR≈0），ISA 仍保持安全行为。

### 消融实验

| 消融 | 关键发现 |
|------|----------|
| 移除安全关键组件 | CC 从 1.854 增至 5.01 |
| 固定惩罚系数 | 动态 Lagrange 乘子显著优于固定系数 |
| 不同 cost threshold | 20% 阈值提供最佳平衡 |

### 关键发现

1. **安全与性能解耦**：ISA 安全成本分布与任务成功/失败无关，FLaRe 存在显著负相关
2. **消除灾难性轨迹**：ISA 完全消除 CC>10 的高风险轨迹
3. **跨模型通用性**：在 EmbCLIP、Embodied-Codebook 等不同 VLA 上均有效

## 亮点与洞察

1. **首次系统的 VLA 安全对齐**：从建模到保证的完整 pipeline，是集成安全方法
2. **安全与性能的 Pareto 最优**：通过 CMDP 框架显式权衡
3. **Safety-CHORES 基准**：精心设计的安全关键场景，比现有基准更能暴露 VLA 漏洞（CC 高出 2 倍）
4. **Sim-to-Real 验证**：在物理机器人平台上成功部署

## 局限与展望

1. Sim-to-Real 仅验证了 Safety-PickUp 单一任务
2. 安全谓词需要人工设计，自动化安全规范提取是重要方向
3. 轨迹级成本归因仅归到违规段最后一步，信用分配待研究
4. SafeRL 训练需要 15M-25M 步，计算开销较高

## 相关工作与启发

- **FLaRe**：标准 RL 最大化任务奖励，ISA 在 CMDP 下显式约束安全成本
- **Safe-RLHF**：用于 LLM 意图级安全，ISA 扩展到物理世界具身安全
- **启发**：CMDP 框架为具身 AI 安全提供了原则性的优化范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 SafeRL/CMDP 系统性应用于 VLA 安全对齐
- 实验充分度: ⭐⭐⭐⭐⭐ 主实验+消融+OOD+极端失败+跨模型+Sim-to-Real
- 写作质量: ⭐⭐⭐⭐ 四阶段 ISA 框架清晰，但论文篇幅较长
- 价值: ⭐⭐⭐⭐⭐ 对具身 AI 安全领域有里程碑意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Automaton Constrained Q-Learning](automaton_constrained_q-learning.md)
- [\[ICML 2025\] Action-Constrained Imitation Learning](../../ICML2025/robotics/action-constrained_imitation_learning.md)
- [\[NeurIPS 2025\] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)

</div>

<!-- RELATED:END -->
