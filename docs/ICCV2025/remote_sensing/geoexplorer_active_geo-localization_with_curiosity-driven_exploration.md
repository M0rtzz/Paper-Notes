---
title: >-
  [论文解读] GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration
description: >-
  [ICCV 2025][遥感][主动地理定位] 提出 GeoExplorer，一个结合目标导向和好奇心驱动内在奖励的主动地理定位（AGL）智能体，通过联合动作-状态动力学建模和好奇心探索实现更鲁棒的 UAV 搜索策略，在未知目标和环境中展现出优越的泛化能力。 主动地理定位（AGL）是指在预定义搜索区域内…
tags:
  - "ICCV 2025"
  - "遥感"
  - "主动地理定位"
  - "好奇心驱动探索"
  - "强化学习"
  - "UAV 导航"
  - "多模态目标"
---

# GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration

**会议**: ICCV 2025  
**arXiv**: [2508.00152](https://arxiv.org/abs/2508.00152)  
**代码**: [项目主页](https://limirs.github.io/GeoExplorer/)  
**领域**: 遥感  
**关键词**: 主动地理定位, 好奇心驱动探索, 强化学习, UAV 导航, 多模态目标

## 一句话总结

提出 GeoExplorer，一个结合目标导向和好奇心驱动内在奖励的主动地理定位（AGL）智能体，通过联合动作-状态动力学建模和好奇心探索实现更鲁棒的 UAV 搜索策略，在未知目标和环境中展现出优越的泛化能力。

## 研究背景与动机

主动地理定位（AGL）是指在预定义搜索区域内，通过导航 UAV 智能体到达目标位置的任务。目标可以多种模态（航拍图像、地面图像、文本）给出，但其位置在推理时未知。这在搜救行动中至关重要。

现有方法（如 GOMAA-Geo）依赖**外在奖励**（基于距离的奖励），存在三个问题：

**距离估计不可靠**：推理时目标位置未知，基于距离的奖励无法计算，学到的策略可靠性降低

**环境建模不充分**：仅预测动作序列，不建模状态转移，无法理解动作如何改变环境

**泛化能力弱**：量身定制于训练环境的距离估计难以适应未见目标和新环境

**核心洞察**：引入好奇心驱动的内在奖励（目标无关）与目标导向的外在奖励互补——好奇心奖励基于状态预测与实际观测的差异，不依赖目标位置，提供密集、目标无关、内容相关的探索引导。

## 方法详解

### 整体框架

GeoExplorer 的训练分为三个阶段：
1. **特征表示**：使用对齐的多模态编码器处理不同模态的目标
2. **动作-状态动力学建模（DM）**：因果 Transformer 有监督预训练，联合建模动作和状态转移
3. **好奇心驱动探索（CE）**：基于 PPO 的 Actor-Critic RL，结合外在和内在奖励

### 关键设计

1. **多模态特征表示**：使用三个对齐的编码器：

    - 航拍图像编码器 Sat2Cap（ViT，与 CLIP 对齐）
    - 地面图像编码器 CLIP_img
    - 文本编码器 CLIP_text
    - 所有编码器预训练后冻结，确保不同模态的目标在同一嵌入空间

2. **动作-状态动力学联合建模**：使用因果 Transformer（Falcon-7B）同时预测：

    - 最优动作 $\hat{a}_t = \text{CausalTrans}(s_t | x_{t-1}, s_{goal})$（哪个动作让智能体更接近目标）
    - 状态表示 $\hat{s}_t = \text{CausalTrans}(a_{t-1} | x_{t-1}, s_{goal})$（动作如何影响环境）
    - 无需修改模型架构，仅增加状态建模损失 $\mathcal{L}_{State} = \sum_{t=1}^{N-1} \|\hat{s}_t - s_t\|^2_2$
    - 总 DM 损失：$\mathcal{L}_{DM} = \mathcal{L}_{Action} + \alpha \mathcal{L}_{State}$

3. **好奇心驱动内在奖励**：利用 DM 阶段学到的状态预测 $\hat{s}_{t+1}$ 与实际状态 $s_{t+1}$ 的差异度量"出乎意料"程度：

    - MSE 方式：$r^{in}_t = \|\hat{s}_{t+1} - s_{t+1}\|^2_2$
    - 余弦相似度方式：$r^{in}_t = -\cos(\hat{s}_{t+1}, s_{t+1})$
    - 最终奖励：$r^{CE}_t = r^{ex}_t + \beta r^{in}_t$，其中 $\beta = 0.25$
    - 关键：内在奖励归一化到 [-1, 1] 后再加权

### 损失函数 / 训练策略

- DM 阶段：有监督预训练，随机生成 {start, goal} 轨迹对
- CE 阶段：冻结因果 Transformer，仅训练 Actor-Critic 动作预测头（MLP 结构）
- 搜索网格 5×5，搜索预算 B=10，评估距离 C∈{4,5,6,7,8}
- 仅在 Masa 数据集上训练，在其他数据集上零样本迁移评估

## 实验关键数据

### 主实验（表格）

**Masa 数据集验证集（成功率 SR）**：

| 方法 | C=4 | C=5 | C=6 | C=7 | C=8 |
|------|-----|-----|-----|-----|-----|
| Random | 0.141 | 0.058 | 0.064 | 0.025 | 0.024 |
| DiT | 0.201 | 0.296 | 0.357 | 0.422 | 0.456 |
| GOMAA-Geo | 0.409 | 0.506 | 0.717 | 0.803 | 0.785 |
| **GeoExplorer** | **0.432** | **0.532** | **0.816** | **0.923** | **0.950** |

**未见目标泛化（SwissViewMonuments）**：

| 方法 | C=4 | C=5 | C=6 | C=7 | C=8 |
|------|-----|-----|-----|-----|-----|
| GOMAA-Geo | 0.403 | 0.383 | 0.627 | 0.728 | 0.783 |
| **GeoExplorer (I)** | **0.413** | **0.533** | **0.770** | **0.889** | **0.883** |
| **GeoExplorer (G)** | **0.416** | **0.517** | **0.773** | **0.878** | 0.783 |

### 消融实验（表格）

| Action Loss | State Loss | $r^{ex}$ | $r^{in}$ | C=4 | C=5 | C=6 | C=7 | C=8 |
|:-:|:-:|:-:|:-:|-----|-----|-----|-----|-----|
| ✓ | | ✓ | | 0.409 | 0.506 | 0.717 | 0.803 | 0.785 |
| ✓ | ✓ | ✓ | | 0.398 | 0.494 | 0.761 | 0.841 | 0.865 |
| ✓ | | ✓ | MSE | 0.389 | 0.494 | 0.741 | 0.862 | 0.914 |
| ✓ | ✓ | ✓ | COS | 0.396 | 0.539 | 0.813 | 0.905 | 0.935 |
| ✓ | ✓ | ✓ | **MSE** | **0.432** | **0.532** | **0.816** | **0.923** | **0.950** |

状态建模 + 好奇心奖励的完整组合效果最佳；即使不显式监督状态预测，内在奖励本身也能带来提升。

### 关键发现

- 长路径提升最显著：C=8 时 GeoExplorer 比 GOMAA-Geo 提升 0.1643
- 跨域迁移：在 xBD-disaster（灾后环境 + 灾前目标）上平均提升 0.0556
- 好奇心奖励具有内容感知性：从森林到城市的过渡 patch 获得最高内在奖励
- 探索更全面：C=4 时 GeoExplorer 有 30.79% 访问在搜索区域内部（GOMAA-Geo 仅 20.08%）
- 增大搜索预算时优势扩大（更多探索空间）

## 亮点与洞察

- 将好奇心驱动 RL 引入 AGL 任务，无需额外组件即可集成到序列建模框架
- 联合动作-状态建模无需修改 Transformer 架构——仅加一个状态损失即可
- 提出新的 SwissView 基准，特别是 SwissViewMonuments 子集评估未见目标泛化
- 好奇心奖励的可视化分析非常直观：语义跳变（如森林→城市）获得高奖励

## 局限与展望

- 搜索空间为离散网格（5×5），未来需扩展到连续状态和动作空间
- 未考虑 UAV 的自我位姿噪声和观测变形等真实部署问题
- 内在奖励与外在奖励的最优平衡方式需更深入研究
- 训练集（Masa）场景较单一，可能限制在更多样环境中的表现

## 相关工作与启发

- 相比 GOMAA-Geo（仅动作建模 + 外在奖励），GeoExplorer 增加状态建模和内在奖励两个维度
- 好奇心驱动 RL 在经典控制/游戏任务中已有广泛研究，本文首次将其引入地理定位
- 状态预测的"副产品"用于构建内在奖励，设计巧妙

## 评分

- 新颖性: ⭐⭐⭐⭐ （好奇心 RL + AGL 的首次结合）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 个基准 + 新数据集 + 全面消融 + 丰富可视化）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，补充材料详尽）
- 价值: ⭐⭐⭐⭐ （对搜救 UAV 部署有实际意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Scaling Image Geo-Localization to Continent Level](../../NeurIPS2025/remote_sensing/scaling_image_geo-localization_to_continent_level.md)
- [\[ICCV 2025\] Information-Bottleneck Driven Binary Neural Network for Change Detection](information-bottleneck_driven_binary_neural_network_for_change_detection.md)
- [\[CVPR 2026\] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](../../CVPR2026/remote_sensing/rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)
- [\[ECCV 2024\] ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](../../ECCV2024/remote_sensing/congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)
- [\[ICCV 2025\] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)

</div>

<!-- RELATED:END -->
