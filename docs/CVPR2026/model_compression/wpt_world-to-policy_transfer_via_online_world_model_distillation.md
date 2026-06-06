---
title: >-
  [论文解读] WPT: World-to-Policy Transfer via Online World Model Distillation
description: >-
  [CVPR 2026][模型压缩][世界模型] WPT 提出世界-策略转移训练范式，通过可训练的奖励模型将世界模型的未来预测知识注入教师策略，再通过策略蒸馏和世界奖励蒸馏转移到轻量学生策略，实现79.23驾驶得分（闭环）且推理速度提升4.9倍。
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "世界模型"
  - "策略蒸馏"
  - "奖励模型"
  - "自动驾驶"
  - "在线蒸馏"
---

# WPT: World-to-Policy Transfer via Online World Model Distillation

**会议**: CVPR 2026  
**arXiv**: [2511.20095](https://arxiv.org/abs/2511.20095)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 世界模型, 策略蒸馏, 奖励模型, 自动驾驶, 在线蒸馏

## 一句话总结
WPT 提出世界-策略转移训练范式，通过可训练的奖励模型将世界模型的未来预测知识注入教师策略，再通过策略蒸馏和世界奖励蒸馏转移到轻量学生策略，实现79.23驾驶得分（闭环）且推理速度提升4.9倍。

## 研究背景与动机
1. **领域现状**：世界模型在自动驾驶中用于捕捉时空动态、预测未来场景，但现有方法存在紧密运行时耦合或依赖离线奖励信号的问题。
2. **现有痛点**：直接集成世界模型的方法导致严重的推理延迟；将世界模型作为模拟器的方法依赖模拟器保真度。
3. **核心矛盾**：世界模型提供宝贵的未来预测知识，但部署时不能承受其计算开销。
4. **本文目标**：在训练时利用世界模型知识，部署时仅用轻量策略网络，实现"训练时用世界模型，部署时丢弃"。
5. **切入角度**：通过奖励模型作为桥梁，将世界模型的预测与策略的轨迹选择关联，然后蒸馏到学生。
6. **核心idea**：可训练的交互式奖励模型评估候选轨迹与世界模型预测的一致性→教师策略学会未来感知规划→策略蒸馏+世界奖励蒸馏转移到学生。

## 方法详解

### 整体框架
训练阶段：世界模型预测未来状态→奖励模型评估多模态候选轨迹→教师策略选择最优轨迹。蒸馏阶段：学生通过策略蒸馏（对齐规划表示）和世界奖励蒸馏（匹配教师最优奖励轨迹）从教师学习。部署时只用学生策略。

### 关键设计

1. **可训练交互式奖励模型**:
    - 功能：评估候选轨迹与未来世界状态的一致性
    - 核心思路：每条候选轨迹 $\tau_i$ 与世界模型预测的未来状态 $F_{t+1}^w$ 结合，通过轨迹编码器和两个奖励头评估——（1）模仿奖励：评估轨迹与人类驾驶偏好的一致性；（2）模拟奖励：基于PDM评分等驾驶质量指标打分。最终奖励为两者加权。
    - 设计动机：将世界模型的预测能力转化为可优化的奖励信号，使策略能端到端地从未来预测中学习。

2. **策略蒸馏**:
    - 功能：将教师的规划表示能力转移到轻量学生
    - 核心思路：对齐教师和学生的规划表示（planning queries经过decoder后的特征），使学生在单次前向传播中就能产生接近教师的规划。
    - 设计动机：学生网络简单，直接学习端到端映射避免了多模态轨迹生成和世界模型交互的开销。

3. **世界奖励蒸馏**:
    - 功能：让学生学会匹配教师在预测未来世界中的最优轨迹
    - 核心思路：鼓励学生输出的轨迹在世界模型预测的未来中获得与教师最优轨迹相近的奖励，即匹配教师选择的奖励最高轨迹。
    - 设计动机：仅对齐表示不够，还需要对齐"什么轨迹在未来世界中是最好的"这一决策逻辑。

### 损失函数 / 训练策略
教师训练：模仿损失（imitation reward）+ 仿真奖励（simulation reward）。蒸馏：策略蒸馏损失 + 世界奖励蒸馏损失。

## 实验关键数据

### 主实验

| 基准 | 指标 | WPT | 之前SOTA | 提升 |
|------|------|-----|---------|------|
| 开环 | L2误差 | 0.61m | - | 竞争力 |
| 开环 | 碰撞率 | 0.11% | - | SOTA |
| 闭环 | 驾驶得分 | 79.23 | - | SOTA |
| 推理速度 | 加速比 | 4.9× | 1× | 显著提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full WPT (教师) | 最优 | 世界模型增强的教师 |
| Full WPT (学生) | 接近教师 | 蒸馏保留大部分增益 |
| w/o 世界奖励蒸馏 | 下降 | 奖励蒸馏很重要 |
| w/o 奖励模型 | 显著下降 | 奖励模型是核心 |

### 关键发现
- 学生策略在推理速度提升4.9倍的同时保留了教师大部分性能增益。
- 世界奖励蒸馏相比纯策略蒸馏提供了额外的提升，说明决策逻辑的转移很重要。
- WPT在不同轻量策略架构上都有效，说明框架通用性好。

## 亮点与洞察
- **"训练时用世界模型，部署时丢弃"**的范式很有吸引力：获得世界模型的好处又不付部署代价。
- **可训练奖励模型**作为知识转移的桥梁是巧妙设计，将不可微的世界知识转化为可微的学习信号。
- 双重蒸馏（表示+奖励）比单一蒸馏更完整。

## 局限与展望
- 依赖预训练世界模型的质量，低保真度预测会导致误导性奖励。
- 闭环评估的场景多样性有限。
- 未来可探索将此范式应用到通用机器人决策中。

## 相关工作与启发
- **vs WoTE/DriveDPO**: 将世界模型直接集成到策略中，推理时需要自回归rollout。WPT将此开销完全转移到训练阶段。
- **vs DriveWorld类模拟器方法**: 依赖模拟器保真度且主要在合成环境评估。WPT直接在真实数据上训练。

## 评分
- 新颖性: ⭐⭐⭐⭐ 训练-部署解耦的世界模型使用范式新颖
- 实验充分度: ⭐⭐⭐⭐ 开环+闭环+多策略架构验证
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，流程描述完整
- 价值: ⭐⭐⭐⭐⭐ 在效率和性能之间取得了很好的平衡

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[CVPR 2026\] Memory-Efficient Transfer Learning with Fading Side Networks via Masked Dual Path Distillation](memory_efficient_transfer_learning_with_fading_side_networks.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](../../ICML2026/model_compression/entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](../../ICLR2026/model_compression/pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)

</div>

<!-- RELATED:END -->
