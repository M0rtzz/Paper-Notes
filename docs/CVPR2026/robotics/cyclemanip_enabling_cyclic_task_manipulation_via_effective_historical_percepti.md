---
title: >-
  [论文解读] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding
description: >-
  [CVPR 2026][机器人][循环操作] CycleManip 首次系统性地解决机器人循环操作任务（如摇瓶子N次），通过成本感知的历史采样策略增强历史感知，配合多任务学习辅助目标提升历史理解，以端到端模仿学习方式实现循环次数可控的操作。
tags:
  - "CVPR 2026"
  - "机器人"
  - "循环操作"
  - "机器人操作"
  - "模仿学习"
  - "历史感知"
  - "多任务学习"
---

# CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding

**会议**: CVPR 2026  
**arXiv**: [2512.01022](https://arxiv.org/abs/2512.01022)  
**代码**: [https://isee-laboratory.github.io/CycleManip/](https://isee-laboratory.github.io/CycleManip/)  
**领域**: 机器人  
**关键词**: 循环操作, 机器人操作, 模仿学习, 历史感知, 多任务学习

## 一句话总结
CycleManip 首次系统性地解决机器人循环操作任务（如摇瓶子N次），通过成本感知的历史采样策略增强历史感知，配合多任务学习辅助目标提升历史理解，以端到端模仿学习方式实现循环次数可控的操作。

## 研究背景与动机
1. **领域现状**：机器人操作领域的模仿学习和VLA模型在顺序任务上表现出色，但对循环任务（重复动作+准确终止）的研究几乎空白。
2. **现有痛点**：（i）短观察窗口的策略无法区分循环的不同阶段（每次摇瓶后视觉观测几乎相同）；（ii）缺乏包含充足数据和自动评估工具的循环任务基准。
3. **核心矛盾**：循环任务是非马尔可夫过程，正确决策不仅取决于当前观测，还取决于已累积的进度。但扩展观察范围会大幅增加计算开销。
4. **本文目标**：设计端到端的模仿学习框架，使机器人能执行循环动作并在正确时刻停止。
5. **切入角度**：将观测分为高开销（视觉）和低开销（本体感知），差异化采样；用多任务学习促进循环阶段理解。
6. **核心idea**：成本感知采样（视觉稀疏+本体感知密集）+ 进度预测辅助任务 = 循环感知策略。

## 方法详解

### 整体框架
给定用户指令和机器人观测，成本感知采样策略对高低开销观测采用不同采样，所有观测和语言指令编码为扩散条件预测动作。同时观测特征用于预测任务进度（辅助任务），增强模型对循环阶段的理解。

### 关键设计

1. **成本感知历史采样策略**:
    - 功能：在低计算开销下扩展观察范围
    - 核心思路：将观测分为低开销（末端执行器位姿差分）和高开销（点云/RGB）两类。低开销观测全量密集采样（计算几乎免费），高开销观测用启发式帧采样——一半用二分采样覆盖全历史，一半用指数采样保留近期细节（$t-2^k$）。
    - 设计动机：末端执行器的循环特性比关节位置更明显且更容易建模；使用位姿差分而非绝对位置避免了位置偏差。

2. **多任务学习进度预测**:
    - 功能：让模型隐式学习循环阶段特征
    - 核心思路：引入辅助任务预测当前进度 $b_t$（当前帧号/最大帧号，离散化为10类分类问题）。通过多层MLP融合后单层MLP预测进度。总损失 = MSE动作损失 + CE进度损失。
    - 设计动机：纯模仿学习的监督信号在每个循环中都相同（继续执行），无法让模型区分不同阶段。进度预测迫使模型学习区分性特征。

3. **CycleManip基准**:
    - 功能：提供循环任务评估平台
    - 核心思路：基于RoboTwin 2.0构建8个循环操作任务（锤钉子、摇瓶子、切胡萝卜等），每任务200条演示轨迹，循环次数1-8次。自动评估仅当操作成功且循环次数正确时才判定为成功。
    - 设计动机：无标准化基准阻碍了循环任务研究的发展。

### 损失函数 / 训练策略
$\mathcal{L} = \alpha \cdot \text{MSE}(a_t, a_t^*) + \beta \cdot \text{CE}(b_t, b_t^*)$，使用扩散策略框架。

## 实验关键数据

### 主实验

| 任务 | CycleManip成功率 | Baseline成功率 | 循环准确率 |
|------|----------------|--------------|----------|
| 锤钉子 | 高 | 低 | 高 |
| 摇瓶子 | 高 | 极低 | 高 |
| 切胡萝卜 | 中高 | 低 | 中高 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full CycleManip | 最优 | 完整框架 |
| w/o 进度预测 | 显著下降 | 辅助任务关键 |
| w/o 密集本体采样 | 下降 | 历史感知重要 |
| 仅视觉扩展 | 计算开销大但效果有限 | 证明成本感知采样的必要性 |

### 关键发现
- 方法在通用操作任务上也有很好的适应性，不限于循环任务。
- 可作为即插即用模块应用于VLA模型（如Pi0等）。
- 跨平台验证（双臂夹持器、灵巧手、人形机器人）证明了通用性。
- 密集本体感知采样的计算开销几乎可忽略，是最具性价比的历史建模方式。

## 亮点与洞察
- **首次系统性定义循环操作任务**，填补了机器人操作研究的空白。
- **成本感知采样**的设计直觉优秀：用免费的本体感知替代昂贵的视觉历史来捕捉循环模式。
- 进度预测辅助任务是一个简单但有效的trick。

## 局限与展望
- 进度预测离散化为10类可能不够精细。
- 当前仅支持固定循环次数，对"直到混合均匀"等动态终止条件未探索。
- 复杂物理交互（如不同材质的摩擦）可能需要更精细的力反馈。

## 相关工作与启发
- **vs Diffusion Policy**: 标准扩散策略使用短观察窗口，无法处理循环任务。CycleManip通过历史感知和理解扩展了其能力。
- **vs VLA models**: VLA模型也依赖短期观测，CycleManip的即插即用设计可直接增强它们。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义循环操作问题，方法实用但不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 8任务+3平台+仿真+真实+VLA集成
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 填补重要空白，对机器人实际部署有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](../../NeurIPS2025/robotics/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[CVPR 2026\] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition](pixel-level_scene_understanding_in_one_token_visual_states_need_what-is-where_co.md)

</div>

<!-- RELATED:END -->
