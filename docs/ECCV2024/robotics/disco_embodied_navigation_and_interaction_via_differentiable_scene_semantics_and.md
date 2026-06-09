---
title: >-
  [论文解读] DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control
description: >-
  [ECCV2024][机器人][embodied navigation] 提出 DISCO 框架，通过可微分场景语义表示和双层粗-细动作控制，在 ALFRED 基准上实现具身导航与交互的显著性能提升（未见场景成功率超越 SOTA +8.6%，且无需逐步指令）。
tags:
  - "ECCV2024"
  - "机器人"
  - "embodied navigation"
  - "mobile manipulation"
  - "differentiable scene representation"
  - "affordance"
  - "dual-level control"
  - "ALFRED"
---

# DISCO: Embodied Navigation and Interaction via Differentiable Scene Semantics and Dual-Level Control

**会议**: ECCV2024  
**arXiv**: [2407.14758](https://arxiv.org/abs/2407.14758)  
**代码**: [AllenXuuu/DISCO](https://github.com/AllenXuuu/DISCO)  
**领域**: 机器人  
**关键词**: embodied navigation, mobile manipulation, differentiable scene representation, affordance, dual-level control, ALFRED

## 一句话总结

提出 DISCO 框架，通过可微分场景语义表示和双层粗-细动作控制，在 ALFRED 基准上实现具身导航与交互的显著性能提升（未见场景成功率超越 SOTA +8.6%，且无需逐步指令）。

## 背景与动机

构建通用家庭助手智能体是具身 AI 的长期目标，要求代理具备任务规划、环境建模和物体交互能力。现有方法主要分为两类：

1. **神经策略方法**（如 Seq2Seq、E.T.、M-Track）：端到端学习动作，但需要大量训练轨迹和标注，且存在长视野任务与无记忆感知之间的矛盾
2. **基于地图的规划方法**（如 HLSM、FILM、Prompter）：构建场景模型辅助规划，但执行缺乏灵活性，难以在运行时自适应调整

两类方法各有局限：神经策略数据饥饿、泛化差；地图规划方法的离散 cell 表示对感知不完美敏感，需要手工规则修补。

## 核心问题

如何构建一个具身代理，使其能基于 verb-noun 指令对（如 "Pickup Lettuce"）高效完成移动操控任务？具体需解决三个子问题：

1. 如何建立兼具丰富语义、动态更新、可查询、可泛化的场景表示？
2. 如何在有限模仿数据下实现高效的移动操控？
3. 如何将原语任务整合到长视野具身指令跟随应用中？

## 方法详解

### 感知系统

从自我中心 RGB 帧出发，使用三个神经网络预测像素级信息：

- **深度估计**：U-Net 架构，深度离散为 50 个 bin（每 bin 10cm），交叉熵损失训练
- **实例分割**：Mask R-CNN，85 类物体，COCO 预训练后微调
- **Affordance 估计**：U-Net 架构，预测 1 类导航 + 7 类交互（如 pickable、openable），二值交叉熵损失

所有训练数据通过 AI2THOR 模拟器收集，严格禁止使用未见场景数据。

### 可微分场景表示

将场景建模为 20m × 20m 空间，离散为 80 × 80 网格（每格 25cm × 25cm）。每个网格分配 256 维嵌入 $s_i$，同时初始化 $N^o + N^a$ 个语义查询向量 $q_j$（物体类 + affordance 类）。

**查询机制**：通过内积 + sigmoid 计算网格 $i$ 属于类别 $j$ 的概率：

$$p_{i,j} = \sigma(s_i^T q_j)$$

**在线优化**：每步将自我中心帧转为带语义的点云并投影到俯视图，生成软标签 $y_i^j$（归一化语义点比例）。对可见网格使用交叉熵损失，通过梯度下降同时更新 $s_i$ 和 $q_j$（学习率 0.01，每步 10 次迭代）。

关键优势：相比离散 cell 表示，连续可微表示在历史与当前观测间做软权衡，缓解感知不完美问题，无需手工修补规则。

### 双层粗-细动作控制

对于一个 verb-noun 原语任务（如 "Pickup Lettuce"），执行三阶段：

1. **随机游走**：目标物体尚未被检测器发现时，利用导航 affordance 查询可达性地图，随机选取可达点并用 BFS 规划路径
2. **粗控制（全局线索）**：目标物体被发现后，查询场景表示获取物体分布与 affordance 分布的联合概率图，选择最大概率网格为目标，BFS 规划到目标 1m 范围内
3. **细控制（局部线索）**：用 ResNet50 神经策略，输入为 RGB + 深度 + 目标物体掩码的拼接，通过物体类别特定的分类器预测动作。仅处理短视野调整（4 步以内），通过模仿专家动作训练

细控制只需 316,935 帧训练数据（ALFRED 默认轨迹含 1,051,308 帧），更加数据高效。

### 应用：具身指令跟随

集成到 ALFRED 长视野任务中：用微调 BERT 解析自然语言指令为 ALFRED 内部参数，通过模板转为 verb-noun 子目标序列。例如 pick_clean_then_put 任务转为 (Pick, Lettuce) → (Clean, Lettuce) → (Put, DinningTable)。

## 实验关键数据

在 ALFRED 测试集上的主要结果：

| 设置 | 已见场景 SR | 未见场景 SR | 未见场景 GC |
|------|-----------|-----------|-----------|
| DISCO（含逐步指令） | **59.5%** | **56.5%** | **66.8%** |
| DISCO（仅高层目标） | **58.0%** | **54.7%** | **65.5%** |
| Prompter（含逐步指令） | 53.2% | 45.7% | 58.8% |
| CAPEAM（含逐步指令） | 51.8% | 46.1% | 57.3% |

关键发现：

- 未见场景成功率超 SOTA +10.4%（含指令）/ +11.0%（无指令）
- **无逐步指令的 DISCO 仍优于使用逐步指令的 SOTA**（54.7% vs 46.1%）
- PLWSR 指标为 Prompter 的 1.57x-1.75x，证明执行效率更高

消融实验（验证集）：

| 消融项 | 已见 SR 变化 | 未见 SR 变化 |
|--------|------------|------------|
| 去除可微表示 → 离散 cell | -9.9% | -12.3% |
| 去除导航 affordance | -9.3% | -9.0% |
| 去除交互 affordance | -5.1% | -3.7% |
| 去除细控制 | -4.3% | -3.3% |
| 去除粗控制 | -43.8% | -45.9% |

## 亮点

1. **可微场景表示设计精巧**：用网格嵌入 + 语义查询的内积-sigmoid 机制，实现可微优化替代离散 cell 的手工更新规则，泛化能力显著更强（未见场景提升 12.3%）
2. **双层控制范式实用**：粗控制高效处理长距离导航，细控制仅聚焦短视野调整，数据需求降低 3x 以上
3. **无需逐步指令即超越 SOTA**：证明了方法的鲁棒性和规划能力，减少对细粒度人类标注的依赖
4. **Affordance 融合自然**：将 openable 等属性自动融入决策，避免手写规则

## 局限与展望

1. **依赖模拟器真值训练感知模块**：深度、分割、affordance 均用 AI2THOR 真值训练，向真实世界迁移时感知质量可能大幅下降
2. **Affordance 类别固定**：7 类交互 affordance 由模拟器定义，缺乏开放词汇泛化能力
3. **场景表示为 2D 俯视图**：忽略了高度维度信息，对多层环境或堆叠物体场景可能不足
4. **仅在 AI2THOR/ALFRED 验证**：离散动作空间、有限场景多样性，实际机器人部署差距较大
5. **语言理解依赖模板匹配**：BERT + 模板的指令解析方式较脆弱，不支持灵活的自然语言输入

## 与相关工作的对比

| 方法 | 场景表示 | 控制方式 | 未见 SR |
|------|---------|---------|--------|
| FILM | 离散 2D cell | 手工规则 | 26.5% |
| HLSM | 3D voxel | 神经策略 | 16.3% |
| Prompter | 离散 2D cell + 搜索 | 手工规则 | 45.7% |
| CAPEAM | 上下文记忆 | 上下文规划 | 46.1% |
| **DISCO** | **可微连续嵌入** | **粗细双层** | **56.5%** |

DISCO 的优势在于：(1) 可微表示比离散 cell 更鲁棒；(2) 双层控制兼具地图规划的全局效率和神经策略的局部灵活性；(3) affordance 融入消除手工规则。

## 启发与关联

- **可微场景表示的思路可迁移**：内积查询 + 在线梯度优化的范式可应用于其他需要动态空间语义建模的任务（如语义 SLAM、场景图构建）
- **粗细控制范式通用性强**：全局规划 + 局部调整的分层策略在更广泛的机器人操控场景中值得借鉴
- **与 foundation model 结合的潜力**：当前用 BERT + 模板做指令解析，替换为 VLM/LLM 可能进一步提升规划灵活性和开放场景泛化能力

## 评分
- 新颖性: 8/10 — 可微场景表示和双层控制的结合具有新意，但各模块本身较为经典
- 实验充分度: 9/10 — 充分的消融实验和定性分析，结果有说服力
- 写作质量: 8/10 — 结构清晰，动机和方法阐述较好
- 价值: 8/10 — 在 ALFRED 上大幅推进 SOTA，但受限于模拟器环境

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[CVPR 2026\] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition](../../CVPR2026/robotics/pixel-level_scene_understanding_in_one_token_visual_states_need_what-is-where_co.md)
- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/robotics/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[ECCV 2024\] AFF-ttention! Affordances and Attention models for Short-Term Object Interaction Anticipation](aff-ttention_affordances_and_attention_models_for_short-term_object_interaction_.md)
- [\[ICML 2026\] PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making](../../ICML2026/robotics/psg-nav_probabilistic_scene_graph_navigation_via_multiverse_decision_making.md)

</div>

<!-- RELATED:END -->
