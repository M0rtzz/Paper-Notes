---
title: >-
  [论文解读] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning
description: >-
  [NeurIPS 2025][自动驾驶][运动预测] UniMotion 提出了一个基于 decoder-only Transformer 的统一运动框架，通过任务感知的交互模式和训练策略同时支持运动仿真、轨迹预测和自车规划三大任务，联合训练促进任务间知识共享，微调后在 Waymo 数据集上同时达到多个任务的 SOTA 表现。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 运动预测
  - 轨迹仿真
  - 自动驾驶规划
  - 统一框架
  - GPT式模型
---

# UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning

**会议**: NeurIPS 2025  
**arXiv**: [2602.00566](https://arxiv.org/abs/2602.00566)  
**代码**: https://github.com/LogosRoboticsGroup/UniMotion (有)  
**领域**: 自动驾驶  
**关键词**: 运动预测, 轨迹仿真, 自动驾驶规划, 统一框架, GPT式模型

## 一句话总结

UniMotion 提出了一个基于 decoder-only Transformer 的统一运动框架，通过任务感知的交互模式和训练策略同时支持运动仿真、轨迹预测和自车规划三大任务，联合训练促进任务间知识共享，微调后在 Waymo 数据集上同时达到多个任务的 SOTA 表现。

## 研究背景与动机

自动驾驶中的运动理解包含三个核心任务：运动仿真（生成多样化的智能体行为）、轨迹预测（预测交通参与者的未来轨迹）和自车规划（为自车生成可行驾驶轨迹）。尽管这些任务本质上共享相同的基础能力——理解多智体交互、建模运动行为、推理时空动态——现有方法通常为每个任务设计专用模型，阻碍了跨任务泛化和系统可扩展性。

更关键的是，这种任务隔离忽视了任务间的互利潜力。例如，仿真任务对生成多样性的追求可以为预测任务提供更广泛的参考方向，而预测任务对长程精度的要求可以提升仿真的运动合理性。

本文将运动任务抽象为两个基本类别：**多样运动生成**（对应仿真）和**长程轨迹预测**（对应预测和规划），在此统一的概念框架上构建模型。选择 decoder-only Transformer 是因为其简洁性、可扩展性和跨任务的通用生成能力。

## 方法详解

### 整体框架

UniMotion 以 decoder-only Transformer 为骨干，包含：
1. **输入表示**：将智能体轨迹和高精地图通过 tokenization 离散化为 token 序列（2,048 个智能体 token + 1,024 个地图 token），使用固定时间间隔切分轨迹并聚类构建 token 词汇表
2. **场景上下文编码**：堆叠自注意力模块编码地图嵌入
3. **运动解码**：使用因子化注意力机制（时间自注意力 → 智体-地图交叉注意力 → 智体-智体自注意力），配合相对位置编码确保坐标变换不变性
4. **两阶段训练**：联合训练 + 任务特定微调

### 关键设计

1. **联合训练的双目标策略**：

    - **下一 Token 预测（NTP）**用于仿真：采用因果注意力掩码，从条件概率 P(A_t | A_{<t}) 建模逐步运动生成
    - **长程未来回归（LFR）**用于预测：将 token 分类替换为归一化轨迹回归，每个 token 预测完整长程未来轨迹而非短片段。训练时所有 token 密集监督，推理时仅需单次前向传播
    - 规划任务自然受益于仿真和预测的训练目标，采用两阶段推理：先联合预测所有智能体未来轨迹→tokenize预测结果→渐进生成自车轨迹

2. **任务特定微调策略**：

    - **仿真微调**：引入类似 RLHF 的强化学习微调。使用 GRPO 生成 n 个场景 rollout，计算运动学奖励（GT 在生成轨迹分布下的对数似然）和碰撞奖励（指示函数），仅对一个 rollout 梯度更新以降低计算开销
    - **预测微调**：引入轻量 Transformer 解码器，基于意图锚点（intention points）生成多模态轨迹。使用 Gaussian NLL 损失监督多模态输出
    - **规划微调（Pred2Gen）**：用预测结果替代真值 token 进行自车生成，过滤不正确预测后替换为真值，端到端微调以消除训练-推理的分布不匹配

3. **注意力掩码设计**：不同任务使用不同的注意力掩码——仿真用因果掩码，预测用序列到序列掩码（历史 token 互相可见，未来 token 可见历史和在前的未来 token），实现同一模型下灵活的多任务处理。

### 损失函数 / 训练策略

联合训练损失：$\mathcal{L} = \mathcal{L}_{ntp}(\mathcal{A}_s, \hat{\mathcal{A}}_s) + \mathcal{L}_{lfr}(\mathcal{A}_f, \hat{\mathcal{A}}_f)$

其中 NTP 使用交叉熵损失，LFR 使用 smooth-L1 损失。微调阶段各任务有专属损失：仿真微调在交叉熵上叠加 GRPO 策略损失（权重 0.1）；预测微调用 NLL + 交叉熵 + 辅助 LFR 回归；规划微调监督自车生成（NTP）和周围智体预测（LFR，权重 0.5）。

训练使用 AdamW 优化器，batch size 48，8 张 A6000 GPU 训练 30 epoch，学习率 5e-4 余弦衰减。微调使用相同epochs但仿真和规划用更低学习率 5e-5。

## 实验关键数据

### 主实验 - Sim Agents Challenge

| 方法 | Realism Meta | Kinematic | Interactive | Map-based | minADE |
|------|-------------|-----------|-------------|-----------|--------|
| LLM2AD | 0.7779 | 0.4846 | 0.8048 | 0.9109 | 1.2827 |
| UniMM | 0.7829 | 0.4914 | 0.8089 | 0.9161 | 1.2949 |
| CATK | 0.7846 | 0.4931 | 0.8106 | 0.9177 | 1.3065 |
| **UniMotion** | **0.7851** | **0.4943** | 0.8105 | **0.9187** | 1.3036 |

### 主实验 - WOMD Prediction Leaderboard

| 方法 | minADE↓ | minFDE↓ | MR↓ | mAP↑ | Soft mAP↑ |
|------|---------|---------|-----|------|-----------|
| MTR++ | 0.5906 | 1.1939 | 0.1298 | 0.4329 | 0.4414 |
| EDA | 0.5718 | 1.1702 | 0.1169 | 0.4487 | 0.4596 |
| RMP-YOLO | 0.5737 | 1.1697 | 0.1160 | 0.4523 | 0.4673 |
| **UniMotion** | **0.5718** | **1.1643** | 0.1162 | **0.4534** | 0.4642 |

### 主实验 - Waymo Planning

| 方法 | Collision↓ | Red Light↓ | Off Route↓ | Err@1s↓ | Err@3s↓ | Err@5s↓ |
|------|-----------|------------|-----------|---------|---------|---------|
| DIPP | 1.802 | 1.235 | 0.506 | 0.227 | 1.187 | 3.335 |
| **UniMotion** | **1.565** | 1.309 | **0.477** | **0.083** | **0.591** | **2.246** |

### 消融实验

| NTP | LFR | 仿真 Kin. | 仿真 Inter. | 预测 minADE | 预测 mAP |
|-----|-----|-----------|------------|------------|----------|
| ✓ | - | 0.4884 | 0.7961 | 0.7697 | 0.2629 |
| - | ✓ | 0.4401 | 0.7742 | 0.6668 | 0.2935 |
| ✓ | ✓ | **0.4892** | **0.7968** | **0.6508** | **0.3147** |

### 关键发现

- 联合训练不仅不损害各任务性能，还能提升预测精度（对比仅 LFR，mAP 从 0.2935 提升到 0.3147）
- NTP 和 LFR 的联合训练具有协同效应：多样生成方向可作为预测目标的指导
- 任务特定微调在所有任务上带来一致且显著的提升
- 仅用 RL 微调（无闭环监督）效果不佳，需结合一致性约束
- 预测微调中保留意图锚点至关重要，去除后 mAP 反而下降

## 亮点与洞察

- 将运动任务抽象为"多样生成"和"长程预测"两类基本能力的概念框架清晰且有洞察力
- 在单个 decoder-only 模型中通过注意力掩码和训练目标实现多任务的设计极为优雅
- RL 微调借鉴了 LLM 中 RLHF/GRPO 的成功经验，运动学奖励 + 碰撞奖励的设计自然且有效
- Pred2Gen 微调策略直接解决了规划推理中预测token与训练token的分布不匹配问题
- 在仿真、预测、规划三个排行榜上同时达到或接近 SOTA，验证了统一框架的可行性

## 局限与展望

- 因子化注意力和相对位置编码依赖detailed positional信息，限制了利用更多 LLM 相关技术的可能
- 未探索跨数据集学习（如同时在 Waymo 和 nuPlan 上训练）
- 受限于计算资源，未充分探索更大规模模型的 scaling 效果
- 红灯违规率不佳（1.309 vs DIPP 的 1.235），仅用日志轨迹监督无法约束交通规则遵守
- tokenization 设计更强调生成多样性，可能牺牲仿真任务中的预测精度（minADE 相对较差）

## 相关工作与启发

- 继承了 BehaviorGPT、SMART 等 GPT 式运动模型的自回归生成思路，但创新性地将其扩展到预测和规划
- 与 LLM 多任务训练（如用不同 prompt 区分任务）的思路一脉相承
- RL 微调策略直接借鉴了 GRPO 和 DAPO 的简化设计
- 启发：未来可将此统一框架与感知模块端到端连接

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](../../ECCV2024/autonomous_driving/unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling](flow_matching-based_autonomous_driving_planning_with_advanced_interactive_behavi.md)
- [A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning](../../CVPR2025/autonomous_driving/a_neuro-symbolic_framework_combining_inductive_and_deductive_reasoning_for_auton.md)
- [Availability-aware Sensor Fusion via Unified Canonical Space](availability-aware_sensor_fusion_via_unified_canonical_space.md)
- [Long-term Traffic Simulation with Interleaved Autoregressive Motion and Scenario Generation](../../ICCV2025/autonomous_driving/long-term_traffic_simulation_with_interleaved_autoregressive_motion_and_scenario.md)

<!-- RELATED:END -->
