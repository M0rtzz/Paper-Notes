---
title: >-
  [论文解读] NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation
description: >-
  [ICCV 2025][视觉语言导航] 提出 NavQ，一种前瞻性 VLN 智能体，通过 Q-model 在单次前向传播中预测每个候选动作的长期未来语义聚合特征（Q-feature），结合 A* 式搜索策略在目标导向导航中取得显著提升。
tags:
  - ICCV 2025
  - 视觉语言导航
  - Q学习
  - 前瞻性决策
  - "A*搜索"
  - 自监督预训练
---

# NavQ: Learning a Q-Model for Foresighted Vision-and-Language Navigation

**会议**: ICCV 2025  
**arXiv**: [2510.16457](https://arxiv.org/abs/2510.16457)  
**代码**: [https://github.com/woyut/NavQ_ICCV25](https://github.com/woyut/NavQ_ICCV25)  
**领域**: Reinforcement Learning / Vision-and-Language Navigation  
**关键词**: 视觉语言导航, Q学习, 前瞻性决策, A*搜索, 自监督预训练

## 一句话总结

提出 NavQ，一种前瞻性 VLN 智能体，通过 Q-model 在单次前向传播中预测每个候选动作的长期未来语义聚合特征（Q-feature），结合 A* 式搜索策略在目标导向导航中取得显著提升。

## 研究背景与动机

**领域现状**：目标导向的视觉语言导航（VLN）要求智能体根据目标物体描述在真实 3D 环境中导航。现有方法（如 DUET、HAMT 等）主要基于历史信息做单步决策，缺乏对动作未来后果的预判。

**现有痛点**：
- **局部方法**（如 NeRF/扩散模型合成邻域视图）仅预测单步未来，无法捕捉长程语义信息。
- **世界模型方法**（如 DreamWalker）虽能多步展开，但需逐步 rollout，计算开销大，且在 RGB 空间预测容易失真和过拟合。
- 目标导向 VLN 本质上是一个搜索问题，A* 算法已证明启发式函数（评估未来成本）可大幅提升搜索效率，但 VLN 领域缺乏有效的未来启发函数。

**核心矛盾**：长视野 vs 高效率——如何在不进行耗时逐步展开的情况下，一次性获取长程未来信息？

**本文切入角度**：借鉴 Q-learning 中 Q 值累积未来奖励的思想，将"Q-值"替换为"Q-特征"——累积未来观测的聚合表征。不依赖奖励函数，Q-model 可在大量无标注轨迹上自监督预训练。

## 方法详解

### 整体框架

NavQ 在 DUET 基线上增加一个**未来分支**，与全局编码器（GE，基于历史）并行运行：
1. **Q-model**：为每个候选动作生成 Q-feature，聚合该方向的潜在未来观测
2. **Future Encoder (FE)**：将任务无关的 Q-feature 与文本指令交互，生成目标导向的未来评分
3. 未来评分与历史评分融合，实现 A* 式平衡：过去进展 + 未来前景

### 关键设计

1. **Q-function 定义**：
   $$Q(T, a) = R(\mathcal{A}) + \gamma \mathbb{E}_{a' \sim \pi}[Q(T \cup \{\mathcal{A}\}, a')]$$
   
   不同于传统 Q-learning 预测标量奖励，此处 $R(\cdot)$ 是节点的文本描述特征（由 BLIP 等生成），$Q$ 输出是特征向量。解耦了奖励与导航指令，从而可在无标注场景上预训练。

2. **Rollout 策略设计**：若用随机策略，多路径共享节点会导致不同候选的 Q-feature 区分度不足。作者添加**最短路径偏好约束**：每条 rollout 路径必须是从当前节点出发的最短路径段。这确保每个未来节点的特征仅累积到唯一一个候选动作的 Q-feature 中。
   
   展开为显式公式：$Q(T, a) = \sum_{N, t} P_\pi(N, t | T, a) \cdot \gamma^t \cdot R(N)$
   
   无需 RL 技术（TD误差等），直接枚举有限图上所有可达节点计算监督信号。

3. **文本空间预测（提升泛化）**：Q-model 不在 RGB 视觉空间预测（风格/纹理信息会导致虚假关联），而在文本描述的特征空间预测，聚焦高层语义关系。每个节点的 36 个视图用 BLIP 生成文本描述，取平均文本特征作为 $R(N)$。

4. **MAE 热启动**：Q-model 先做 MAE 预训练（掩码重建轨迹 token），为 Q-learning 提供良好初始化。

5. **Future Encoder + 进度预测**：FE 为 4 层 Graph Transformer，结构同 GE。为保证 GE 和 FE 的功能分解：GE 输出送 MLP 预测"已走距离"（历史进度），FE 输出送 MLP 预测"剩余距离"（未来目标）。这恰好对应 A* 中 g(n) + h(n) 的分解。

### 损失函数 / 训练策略

三阶段训练：
- **Stage 1**：Q-model 自监督预训练（MSE loss，30k 步，lr=1e-5），仅需随机轨迹无需标注
- **Stage 2**：Agent 预训练（MLM + SAP + OG + MRC + 进度预测，100k 步，lr=5e-5），Q-model 冻结
- **Stage 3**：Agent 在线微调（DAgger + 伪专家策略，20k 步，lr=1e-5）

## 实验关键数据

### 主实验：REVERIE 数据集

| 方法 | SR↑ | SPL↑ | RGS↑ | RGSPL↑ |
|------|-----|------|------|--------|
| DUET (baseline) | 46.98 | 33.73 | 32.15 | 23.03 |
| GOAT (CVPR24) | 53.37 | 36.70 | 38.43 | 26.09 |
| VER (CVPR24) | 55.98 | 39.66 | 33.71 | 23.70 |
| **NavQ (Ours)** | **53.22** | **38.89** | **36.84** | **27.12** |
| NavQ (w. extra scenes) | 54.10 | 39.22 | 37.57 | 27.29 |

NavQ 相比 DUET baseline 全面提升：SR +6.24, SPL +5.16, RGSPL +4.09。RGSPL 指标超越所有对比方法。利用额外无标注场景可进一步提升。

### 消融实验

| 配置 | OSR | SR | SPL | RGS | RGSPL |
|------|-----|-----|-----|-----|-------|
| (1) 无未来分支 (baseline) | 54.42 | 48.14 | 33.38 | 30.19 | 21.05 |
| (2) 只有 FE, 无 QM | 54.84 | 48.20 | 33.92 | 32.52 | 23.14 |
| (3) 只有 QM, 无 FE | 53.25 | 48.48 | 32.22 | 33.03 | 21.86 |
| (4) QM+FE, 无进度损失 | 55.98 | 51.55 | 35.79 | 34.51 | 23.81 |
| (5) **完整 NavQ** | **60.47** | **53.22** | **38.89** | **36.84** | **27.12** |
| (6) GT Q-feature, 无 FE | 60.18 | 54.36 | 41.71 | 37.03 | 28.59 |
| (7) GT Q-feature + FE | 65.38 | 59.27 | 47.04 | 39.68 | 31.62 |

关键观察：(a) QM 和 FE 需配合使用，单独一个提升有限；(b) 进度损失对分工贡献显著；(c) GT Q-feature 的上界证实了 Q-feature 设计（含最短路径策略 + 衰减因子）的有效性，SPL 可达 +14%。

| 衰减因子 γ | OSR | SR | RGSPL |
|-----------|-----|-----|-------|
| 0 (仅邻域) | 56.66 | 51.12 | 25.16 |
| 0.3 | 59.73 | 51.95 | 26.61 |
| **0.5** | **60.47** | **53.22** | **27.12** |
| 0.7 | 57.06 | 50.89 | 23.85 |

γ=0 退化为单步世界模型效果，γ=0.5 在特征质量和训练难度间取得最佳平衡。

### 关键发现

- Q-model 在无标注数据上训练优于仅用有标注轨迹训练（泛化更好），样本量差异是关键因素。
- Q-model 是任务无关的：同一预训练模型在 REVERIE 和 SOON 两个不同数据集上均有效。
- SOON 结果：OSR +7.88, SR +2.81, SPL +4.07（val unseen），进一步验证泛化能力。

## 亮点与洞察

- **Q-value → Q-feature 的创新跳跃**：将强化学习中的标量值函数推广为向量特征函数，免除奖励设计，天然兼容自监督。
- **最短路径 rollout 策略**保证了每个未来节点仅分配给一个候选动作，使 Q-feature 具有区分性——这是关键的设计洞察。
- **A* 思想在 VLN 中的落地**：GE 对应 g(n)（已走代价），FE 对应 h(n)（预估剩余代价），通过进度监督实现功能分离。

## 局限与展望

- 当前基于离散导航图，尚未扩展到连续环境。
- Q-model 的预测仍与 GT Q-feature（上界）有显著差距（SPL 38.89 vs 47.04），提升 Q-model 表达能力是主要改进方向。
- 文本描述依赖于图像字幕模型（BLIP）的质量，可能引入噪声。

## 相关工作与启发

- 与 Q* [Wang 2024] 在 LLM 推理中结合 Q-learning 和 A* 搜索的思路类似，但 NavQ 面向具身导航，且使用特征向量而非标量 Q 值。
- VLV [Chang 2020] 也在导航中使用 Q-learning，但受限于封闭集类别标签；NavQ 通过 Q-feature 实现任务无关的通用建模。
- 与世界模型方法（DreamWalker, PathDreamer）互补：它们在 RGB 空间逐步 rollout，NavQ 在潜空间一次性聚合。

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — Q-value 到 Q-feature 的概念创新、A* 落地设计
- 技术深度：⭐⭐⭐⭐ — rollout 策略的理论分析、三阶段训练
- 实验充分度：⭐⭐⭐⭐⭐ — REVERIE+SOON、多维消融、GT上界分析、γ分析
- 实用性：⭐⭐⭐⭐ — 适用于家庭助手等意图级导航场景

<!-- RELATED:START -->

## 相关论文

- [Embodied Navigation with Auxiliary Task of Action Description Prediction](embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](../../AAAI2026/reinforcement_learning/vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)
- [Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](../../NeurIPS2025/reinforcement_learning/provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)
- [EvoLM: In Search of Lost Language Model Training Dynamics](../../NeurIPS2025/reinforcement_learning/evolm_in_search_of_lost_language_model_training_dynamics.md)
- [Octopus: Embodied Vision-Language Programmer from Environmental Feedback](../../ECCV2024/reinforcement_learning/octopus_embodied_vision-language_programmer_from_environmental_feedback.md)

<!-- RELATED:END -->
