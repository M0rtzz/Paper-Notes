---
title: >-
  [论文解读] Multi-person Pose Forecasting with Individual Interaction Perceptron and Prior Learning
description: >-
  [ECCV 2024][时间序列][多人姿态预测] 本文提出 IAFormer（Interaction-Aware Pose Forecasting Transformer），通过设计交互感知模块（IPM）来评估每个人与事件的交互程度，并引入交互先验学习模块（IPLM）来积累高频交互模式的先验知识，从而实现语义层面的多人姿态预测，在多个多人场景数据集上显著超越现有方法。
tags:
  - ECCV 2024
  - 时间序列
  - 多人姿态预测
  - 交互感知
  - Transformer
  - 先验学习
  - 事件角色建模
---

# Multi-person Pose Forecasting with Individual Interaction Perceptron and Prior Learning

**会议**: ECCV 2024  
**作者**: Peng Xiao, Yi Xie, Xuemiao Xu, Weihong Chen, Huaidong Zhang
**代码**: https://github.com/ArcticPole/IAFormer  
**领域**: 人体姿态预测 / 时序预测  
**关键词**: 多人姿态预测, 交互感知, Transformer, 先验学习, 事件角色建模

## 一句话总结

本文提出 IAFormer（Interaction-Aware Pose Forecasting Transformer），通过设计交互感知模块（IPM）来评估每个人与事件的交互程度，并引入交互先验学习模块（IPLM）来积累高频交互模式的先验知识，从而实现语义层面的多人姿态预测，在多个多人场景数据集上显著超越现有方法。

## 研究背景与动机

**领域现状**：人体姿态预测（Human Pose Forecasting）旨在根据历史姿态序列预测未来姿态，是理解人类意图的重要问题。当前方法主要通过深度学习模型学习单人或多人的时序运动规律来进行预测。

**现有痛点**：现有多人姿态预测方法往往忽略了每个人在事件中的角色差异。在多人场景中，不同的人参与事件的程度不同——有些人是核心参与者，有些人只是旁观者。现有方法对所有人一视同仁地建模交互关系，导致在复杂、多种交互同时发生的场景中性能有限。此外，很多方法倾向于学习轨迹层面的运动规律（trajectory pose forecasting），而非语义层面的动作预测（semantic pose forecasting）。

**核心矛盾**：多人场景中的交互具有选择性——事件通常只涉及场景中部分人。如果不区分人与事件的关系强弱，模型会被无关人物的运动模式干扰，无法聚焦于真正有意义的交互关系。

**本文目标** (1) 如何量化每个人与正在发生的事件之间的交互程度？(2) 如何利用交互程度信息来指导特征提取和姿态预测？(3) 如何让模型具备高频交互模式的先验知识以做出更语义化的预测？

**切入角度**：作者提出"事件通常只涉及场景中部分人"这一关键观察，由此设计了交互感知模块来为每个人分配"参与度评分"。同时，通过先验学习模块积累常见的交互模式（如握手、对话、传球等），使预测具有语义理解能力。

**核心 idea**：通过交互感知模块评估每个人的事件参与度，并结合先验学习积累的高频交互模式知识，实现对多人姿态的语义感知预测。

## 方法详解

### 整体框架

IAFormer 接收多人的历史姿态序列作为输入，输出未来每个人的姿态序列。整体流程为：(1) 首先通过 Interaction Perceptron Module (IPM) 评估每个人与事件的交互水平；(2) 基于交互评估结果，通过注意力机制提取交互感知的人体特征（区分高交互人物和低交互人物）；(3) 利用 Interaction Prior Learning Module (IPLM) 注入高频交互的先验知识；(4) 基于融合后的特征预测每个人的未来姿态。

### 关键设计

1. **交互感知模块（Interaction Perceptron Module, IPM）**:

    - 功能：为场景中的每个人计算一个交互程度评分（interaction level），量化其与当前事件的关联程度
    - 核心思路：IPM 接收所有人的姿态特征，通过学习人与人之间的相对运动模式和空间关系来推断谁是事件的核心参与者。具体地，它比较每个人的运动特征与整体事件特征之间的相似度，产出一个标量评分。高评分的人被认为是事件核心参与者，其特征在后续注意力计算中获得更大权重
    - 设计动机：在真实多人场景中，有些人只是经过或站在背景中，对预测核心事件中人物的姿态变化贡献很小。IPM 让模型能自动过滤无关人物的干扰信息，聚焦于有意义的交互

2. **交互感知注意力（Interaction-Aware Attention）**:

    - 功能：基于 IPM 的评分结果，自适应地提取每个人的交互感知特征
    - 核心思路：在标准 Transformer 的注意力机制基础上，将 IPM 产出的交互评分作为注意力权重的调制因子。对于高交互的人物对，注意力权重被放大；对于低交互的人物对，权重被抑制。这使得特征提取过程能够区分"交互相关"和"交互无关"的信息
    - 设计动机：标准注意力不区分交互强度，会平等地对待所有人之间的关系。在复杂多人场景中这会引入大量噪声。交互感知注意力让模型"知道该看谁"

3. **交互先验学习模块（Interaction Prior Learning Module, IPLM）**:

    - 功能：学习和积累高频交互模式的先验知识库，使预测具有语义理解能力
    - 核心思路：IPLM 维护一个可学习的先验知识库（prior memory），存储常见交互模式的原型特征（如"两人面对面对话"、"一人向另一人传递物品"等动作模式）。在预测时，当前场景的交互特征与先验库中的原型做匹配，匹配到的先验被融合进预测特征中。这使得模型不只是做轨迹级别的外推，而能理解"他们正在做什么"并据此做出语义合理的预测
    - 设计动机：纯数据驱动的姿态预测容易陷入简单的运动外推，无法处理运动模式的突变（如突然转向、开始新动作）。先验学习让模型具备"常识"，能在轨迹线索不明确时给出语义合理的预测

### 损失函数 / 训练策略

训练使用 L2 姿态重建损失为主损失，衡量预测姿态与真实未来姿态之间的距离。同时对 IPM 的交互评分施加辅助监督（如基于是否存在物理接触或共同参与同一动作的标注），确保交互评估的语义正确性。IPLM 的先验库通过端到端学习自动从数据中提取高频交互模式。

## 实验关键数据

### 主实验

| 数据集 | 指标 (MPJPE mm) | IAFormer | 之前SOTA | 提升 |
|--------|-----------------|----------|---------|------|
| CMU-Mocap (2人) | MPJPE@1000ms | 显著领先 | - | - |
| UMPM (多人) | MPJPE@1000ms | 显著领先 | - | - |
| CHI3D (2人交互) | MPJPE@1000ms | 显著领先 | - | - |
| Human3.6M (单人) | MPJPE@1000ms | 有竞争力 | - | - |
| 合成人群数据 | MPJPE@1000ms | 显著领先 | - | 多人优势明显 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full IAFormer | 最优 | 完整模型 |
| w/o IPM | 性能下降明显 | 无法区分交互程度，受无关人物干扰 |
| w/o IPLM | 中等下降 | 缺乏先验知识，语义预测能力减弱 |
| w/o 交互感知注意力 | 较大下降 | 退化为标准注意力，多人优势消失 |
| IPM + 标准注意力 | 有提升但不充分 | 只有评分没有融合机制效果有限 |

### 关键发现
- IPM 模块贡献最大，尤其在人数较多的场景中（5人以上），说明交互程度评估是核心
- IPLM 在交互模式丰富的数据集（CHI3D）上效果最为突出，说明先验知识在复杂交互中价值大
- 在单人场景（Human3.6M）中，IAFormer 仍有竞争力但优势不再明显，这符合预期——交互建模模块在单人场景中无用武之地
- 随着场景中人数增加，IAFormer 相比 baseline 的优势逐渐放大

## 亮点与洞察
- **"事件通常只涉及部分人"的观察**非常有洞見力。这个简单但被忽视的观察直接指导了 IPM 的设计，思路可以迁移到多人活动识别、群体行为分析等任务中
- **先验学习模块**的设计思路——维护一个可学习的交互模式原型库——是一种很好的引入领域知识的方式，可以迁移到其他需要"常识推理"的运动预测任务（如车辆交互预测、社交机器人导航）
- 将交互程度作为连续值而非二分类来建模，使得模型能处理渐进式的交互参与，比硬分组更灵活

## 局限与展望
- 先验学习模块的容量固定，对于罕见的交互模式可能覆盖不足；可以考虑动态扩展先验库
- 论文主要在实验室数据集上验证，真实城市街景等大规模场景的效果有待验证
- IPM 的交互评估主要基于运动特征，未融入场景上下文（如对象检测、场景语义分割），限制了对场景因果关系的理解
- 未考虑长时预测（>2秒）的场景，长时间预测中交互角色可能动态变化

## 相关工作与启发
- **vs MRT (Motion Representation Transformer)**: MRT 建模所有人之间的全局关系，不区分交互强度。IAFormer 通过 IPM 引入选择性注意力，在多人场景中效果更好
- **vs SoMoFormer**: SoMoFormer 使用社交力模型来建模人际交互，但缺乏语义层面的交互理解。IAFormer 的 IPLM 补充了语义先验
- **vs TBIFormer**: TBIFormer 考虑了身体部位级别的交互，与 IAFormer 的人物级别交互评估是互补的，两者结合可能进一步提升效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 交互程度量化和先验学习的组合在姿态预测领域是新颖的
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个数据集和不同人数场景，消融实验系统
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法表述流畅，key insight 表达到位
- 价值: ⭐⭐⭐⭐ 对多人姿态预测提出了有效的交互建模框架，开源代码增加了实用价值

<!-- RELATED:START -->

## 相关论文

- [Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [CPiRi: Channel Permutation-Invariant Relational Interaction for Multivariate Time Series Forecasting](../../ICLR2026/time_series/cpiri_channel_permutation-invariant_relational_interaction_for_multivariate_time_se.md)
- [M2FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting](../../AAAI2026/time_series/m2fmoe_multi-resolution_multi-view_frequency_mixture-of-experts_for_extreme-adap.md)
- [MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/time_series/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [Coherent Multi-Agent Trajectory Forecasting in Team Sports with CausalTraj](../../AAAI2026/time_series/coherent_multi-agent_trajectory_forecasting_in_team_sports_with_causaltraj.md)

<!-- RELATED:END -->
