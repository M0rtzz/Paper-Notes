---
title: >-
  [论文解读] Automaton Constrained Q-Learning
description: >-
  [NeurIPS 2025][机器人][LTL] 提出 ACQL（Automaton Constrained Q-Learning），将线性时序逻辑（LTL）任务规范转化为自动机，结合目标条件学习和最小安全约束，首次在连续控制环境中可扩展地同时支持时序目标序列和非平稳安全约束。
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "LTL"
  - "自动机"
  - "安全约束"
  - "目标条件RL"
  - "CMDP"
---

# Automaton Constrained Q-Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.05061](https://arxiv.org/abs/2510.05061)  
**代码**: 无  
**领域**: 强化学习 / 安全RL  
**关键词**: LTL, 自动机, 安全约束, 目标条件RL, CMDP

## 一句话总结

提出 ACQL（Automaton Constrained Q-Learning），将线性时序逻辑（LTL）任务规范转化为自动机，结合目标条件学习和最小安全约束，首次在连续控制环境中可扩展地同时支持时序目标序列和非平稳安全约束。

## 研究背景与动机

### 解决思路

**解决思路**：现实机器人任务需要达成**序列子目标**同时遵守**动态安全约束**（如仓库机器人需按顺序补货并避障、保持电池可达范围）

### 核心矛盾

**核心矛盾**：目标条件 RL（GCRL）能达到单个目标但无法推理目标序列或确保安全；安全 RL 只能处理**静态**安全约束

### 领域现状

**领域现状**：LTL 可表达复杂时序任务，但现有 LTL+RL 方法依赖稀疏二值奖励，在复杂连续环境中表现差

### 现有痛点

**现有痛点**：核心差距**：没有可扩展的方法能同时支持时序目标和安全约束

## 方法详解

### 整体框架

ACQL 构建了一个增强乘积 CMDP：

1. 将 STL（Signal Temporal Logic）规范翻译为确定性 Büchi 自动机（DBA）
2. 从自动机提取**安全约束映射** $S: \mathcal{Q} \to \Phi$ 和**活性条件/子目标映射** $G: \mathcal{Q} \to \mathcal{G}^+$
3. 将 MDP 与自动机组合为增强 CMDP，状态空间 $\mathcal{S}^A = \mathcal{S} \times \mathcal{G}^+ \times \mathcal{Q}$
4. 学习奖励 Q 函数 $Q^r$ 和安全 Q 函数 $Q^c$，策略为约束最大化：$\pi^*(s^A) = \arg\max_{a: Q^c(s^A,a) > \mathcal{L}} Q^r(s^A, a)$

### 关键设计

1. **目标条件 + HER 解决稀疏奖励**:
    - 功能：在乘积 CMDP 状态中包含子目标列表 $g^+$，使用 HER 从失败尝试中学习
    - 核心思路：自动机状态对应的子目标被显式编码到观测中，HER 可追溯性地为到达过的子目标分配奖励
    - 设计动机：此前 LTL+RL 方法的核心瓶颈是稀疏奖励；GCRL 技术可大幅缓解此问题

2. **最小安全约束替代累积代价**:
    - 功能：用最小安全约束 $\mathbb{E}[\min_t c^A(s_t, a_t)] > \mathcal{L}$ 替代标准 CMDP 的累积代价约束
    - 核心思路：将安全判定转为分类问题（安全/不安全），而非回归未来累积代价；使用 $Q^c(s,a) = \mathbb{E}[\min_t c^A(s_t, a_t)]$ 和渐进调度的折扣因子 $\gamma_c \to 1$
    - 设计动机：累积代价预测需建模长期依赖，方差大且收敛慢；最小安全简化为二分类，且 $\mathcal{L}=0$ 对所有任务通用

### 损失函数 / 训练策略

- 奖励 Q 函数：标准 TD 目标 $y^r_t = r^A_t + \gamma Q^r_{\bar\psi}(s^A_{t+1}, \pi_j(s^A_{t+1}))$

- 安全 Q 函数：基于 HJ 可达性的目标 $y^c_t = \gamma_c \min\{c^A(s_t, a_t), Q^c_{\bar\theta}(s^A_{t+1}, \pi_j(s^A_{t+1}))\} + (1-\gamma_c) c^A(s_t, a_t)$

- $\gamma_c$ 从小逐渐调度到 1.0（三时间尺度随机逼近，保证收敛）

## 实验关键数据

### 主实验（表格）

| 任务 | 智能体 | CRM-RS | LOF | **ACQL** |
|------|--------|--------|-----|---------|
| 双目标序列导航 | Ant | 0% 成功 | ~20% | **~80%** |
| 分支导航 | Ant | 0% 成功 | ~10% | **~70%** |
| 单目标+安全约束 | Quadcopter | ~30% | ~40% | **~90%** |
| 双目标+消失安全 | PointMass | ~50% | ~20% | **~95%** |

### 消融实验

- 去除 HER：性能大幅下降，验证了子目标+HER 对克服稀疏奖励的关键作用
- 用累积代价替代最小安全：安全约束违反率上升，且 $\mathcal{L}$ 需要手动调优
- 固定 $\gamma_c=1$（不调度）：安全函数学习不稳定

### 关键发现

- ACQL 在所有测试任务上显著优于 CRM-RS 和 LOF
- 非平稳安全约束（如"到达 $g_1$ 之前避开区域 $o_1$，之后可以经过"）对基线方法是致命挑战
- 成功部署在 6-DOF 机械臂上，在有安全约束的储物柜环境中完成目标到达任务

## 亮点与洞察

- **首次统一**：将 Safe RL 和 GCRL 提升到 LTL 可表达的任务类别
- 最小安全约束比累积代价更简洁、通用（$\mathcal{L}=0$ 对所有任务适用），且学习更稳定
- 单一目标条件策略处理所有子目标（不像 LOF 需要为每个 edge 训练单独策略）
- 收敛性有理论保证（三时间尺度随机逼近）

## 局限与展望

- 仅在 recurrence class LTL 公式上测试，更一般的 LTL 未覆盖
- 实验环境中的障碍物仅通过任务规范引入（无物理障碍），可能不够真实
- 仅限在线 RL 设定，未涉及零样本泛化到新 LTL 任务
- 扩展到高维连续观测空间（如视觉输入）的能力未验证

## 相关工作与启发

- Reward Machines (CRM-RS) 使用终止rollout来隐式处理安全，不够鲁棒
- LOF 需为每个子目标训练独立策略，扩展性差
- 来自 HJ 可达性分析的最小安全思想优雅地桥接了形式方法和深度 RL

## 评分

- 理论创新：⭐⭐⭐⭐⭐
- 实验验证：⭐⭐⭐⭐
- 实用价值：⭐⭐⭐⭐
- 写作质量：⭐⭐⭐⭐
- 综合评分：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [\[ICML 2025\] Action-Constrained Imitation Learning](../../ICML2025/robotics/action-constrained_imitation_learning.md)
- [\[CVPR 2025\] Coordinated Manipulation of Hybrid Deformable-Rigid Objects in Constrained Environments](../../CVPR2025/robotics/coordinated_manipulation_hybrid_deformable_rigid_objects.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)

</div>

<!-- RELATED:END -->
