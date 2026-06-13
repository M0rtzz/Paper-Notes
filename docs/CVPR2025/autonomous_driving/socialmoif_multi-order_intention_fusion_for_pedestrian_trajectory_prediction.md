---
title: >-
  [论文解读] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction
description: >-
  [CVPR 2025][自动驾驶][行人轨迹预测] SocialMOIF 提出了一种多阶意图融合模型，通过一阶直接交互层和高阶邻居间接交互层全面捕获社交意图，结合基于挤压定理思想的轨迹分布近似器和首次引入 KAN 网络的全局轨迹优化器，在 ETH/UCY、SDD、NBA、NuScenes 多个数据集上实现了 SOTA 性能。
tags:
  - "CVPR 2025"
  - "自动驾驶"
  - "行人轨迹预测"
  - "多阶意图交互"
  - "全局轨迹优化"
  - "KAN网络"
  - "分布近似"
---

# SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction

**会议**: CVPR 2025  
**arXiv**: [2504.15616](https://arxiv.org/abs/2504.15616)  
**代码**: [https://github.com/XiaodZhao/SocialMOIF](https://github.com/XiaodZhao/SocialMOIF)  
**领域**: 自动驾驶  
**关键词**: 行人轨迹预测, 多阶意图交互, 全局轨迹优化, KAN网络, 分布近似

## 一句话总结

SocialMOIF 提出了一种多阶意图融合模型，通过一阶直接交互层和高阶邻居间接交互层全面捕获社交意图，结合基于挤压定理思想的轨迹分布近似器和首次引入 KAN 网络的全局轨迹优化器，在 ETH/UCY、SDD、NBA、NuScenes 多个数据集上实现了 SOTA 性能。

## 研究背景与动机

**领域现状**：行人轨迹预测是智能交通和自动驾驶的关键任务。现有方法分为知识驱动（如社会力模型、游戏论方法）和数据驱动（基于 LSTM 的社会池化、基于 Transformer 的时空建模、基于图神经网络的交互建模）。

**现有痛点**：(1) 现有方法主要关注目标行人与邻居间的**直接交互**（一阶），忽略了邻居群体内部交互**间接传导**到目标行人的高阶影响。Kim 等人虽然建模了 N 阶交互但平等对待各阶，当邻居数量多时反而削弱了低阶意图的主导作用。(2) 生成模型中的潜变量分布缺乏显式引导，可解释性差。(3) 下游轨迹生成通常是逐时间步串行的，累积误差且效率低。

**核心矛盾**：如何在强化一阶直接交互的主导地位的同时，有效纳入高阶间接影响？如何让生成模型的潜变量分布更可解释且可控？

**本文目标**：(1) 设计多阶意图融合网络，同时捕获直接和间接交互信息；(2) 设计轨迹分布近似器显式引导潜变量；(3) 设计全局轨迹优化器实现并行预测。

**切入角度**：高阶意图交互可以分解为邻居间的一阶交互。通过多子空间并行捕获邻居内部交互矩阵，用可学习的影响因子 $\eta_m$ 加权后与一阶交互矩阵融合，保证低阶主导、高阶补充。

**核心 idea**：用分层注意力机制分别建模一阶（目标-邻居）和高阶（邻居-邻居传导到目标）意图交互，融合后通过分布近似器和 KAN 优化器生成高质量的并行轨迹预测。

## 方法详解

### 整体框架

输入历史轨迹 $g_i^{1:T_H}$（8 帧），输出预测轨迹 $\hat{g}_i^{T_H+1:T_F}$（8 帧）。Pipeline 包含四个核心组件：(1) 多阶意图融合模型 (MOIF) 从位置、速度、距离、角度等提取并融合多阶社交意图；(2) 轨迹分布近似器用多阶意图作为下界、真实未来轨迹作为上界显式引导潜变量分布；(3) 全局轨迹优化器通过 KAN 网络在时间维度并行优化；(4) 距离-方向融合损失函数综合监督动态变化。

### 关键设计

1. **多阶意图融合模型 (MOIF)**:

    - 功能：全面捕获目标行人与邻居间的直接和间接意图交互
    - 核心思路：分两层设计。**高阶意图交互层**：提取 $N_n$ 个邻居的绝对位置和速度，通过多头注意力在 M 个子空间并行计算邻居间意图系数矩阵 $W_U^m$（总意图数 $\Omega = N_n^2$）。**一阶意图交互层**：构建目标行人与邻居间的丰富特征（位置、距离 $d$、速度夹角 $\theta$、预测终点距离 $e$），通过注意力计算直接交互矩阵 $W_S$。融合时高阶通过可学习因子 $\eta_m$ 加权后与一阶叠加：$A_i = (\sum_m \eta_m W_U^m + W_S) V_S$
    - 设计动机：一阶输入包含更丰富的相对特征（距离、角度、预测终点距离）来强化其主导地位，而高阶通过可学习权重 $\eta_m$ 自适应控制影响程度，避免邻居多时高阶压制低阶

2. **轨迹分布近似器**:

    - 功能：显式引导潜变量分布，提高生成轨迹的质量和模型可解释性
    - 核心思路：受挤压定理启发，以多阶融合意图 $I_i$ 作为下界、真实未来轨迹特征 $B_i$ 作为上界，通过重参数化近似潜变量分布 $q_\varphi(\cdot|I_i^{t-1}, B_i^t) \sim N(u_{i\varphi}^t, \sigma_{i\varphi}^t)$。训练时同时有 $\varphi$（含 GT 信息）和 $\vartheta$（仅推理用）两套参数，通过 KL 散度拉近二者。用 RNN 在每个时间步更新意图 $I_i^t$，将潜变量和生成轨迹拼接后更新
    - 设计动机：常规生成模型（如 CVAE）中潜变量分布隐式学习，不够可控。通过设定上下界显式约束分布范围，使模型在安全敏感任务中更有解释性

3. **基于 KAN 的全局轨迹优化器**:

    - 功能：首次引入 KAN 网络实现全时域并行轨迹优化
    - 核心思路：将解码器输出的 $g_i^{T_H+1:T_F}$ 在时间维度展平为 $\Gamma^0 \in \mathbb{R}^{2T_F}$，通过 L 层 KAN（每层的激活函数矩阵 $\Phi_\ell$ 是可学习的）进行全局优化：$\hat{g}_i \leftarrow \Gamma^L = (\Phi_L \circ \cdots \circ \Phi_0) \Gamma^0$。最终映射回原始维度得到整个预测周期的最终轨迹
    - 设计动机：传统方法逐时间步串行生成轨迹，累积误差且效率低。KAN 的可学习激活函数比固定激活的 MLP 更灵活，适合捕捉轨迹的非线性全局模式

### 损失函数 / 训练策略

总损失结合距离、方向和 KL 散度三部分：

$L = \sum_{t} \{ \mathbb{E}[L_{dis} + L_{angle}] - KL[q_\varphi \| q_\vartheta] \}$

- **距离损失** $L_{dis} = \|\hat{g}_i^t - g_{igt}^t\|$：标准位移误差
- **方向损失** $L_{angle} = -\arccos(\text{cos\_sim}(\hat{g}_i^t - \hat{g}_i^{t+1}, g_{igt}^t - g_{igt}^{t+1}))$：监督预测方向与真实方向一致
- **KL 散度**：训练模式（含 GT）和测试模式（纯推理）的潜变量分布对齐

训练在两张 RTX 4090 上进行，M=6 个子空间，L=3 层 KAN，8 帧历史预测 8 帧未来，使用 best-of-20 评估协议。

## 实验关键数据

### 主实验

ETH/UCY 平均 ADE/FDE（米）：

| 方法 | ETH | Hotel | Univ | Zara1 | Zara2 | Avg |
|------|-----|-------|------|-------|-------|-----|
| EqMotion | 0.36/0.54 | 0.12/0.16 | 0.21/0.39 | 0.16/0.27 | 0.11/0.19 | 0.19/0.31 |
| E-V2-Net-SC | 0.23/0.30 | 0.10/0.13 | 0.18/0.24 | 0.13/0.16 | 0.11/0.16 | 0.15/0.20 |
| **SocialMOIF** | 0.26/0.31 | **0.10/0.12** | **0.11/0.17** | **0.10/0.17** | **0.09/0.14** | **0.13/0.18** |

其他数据集 ADE/FDE：

| 数据集 | 之前 SOTA | SocialMOIF | 提升 |
|--------|-----------|------------|------|
| NBA-Rebound | 0.54/0.79 | **0.34/0.66** | -37%/-16% |
| NBA-Scores | 0.46/0.76 | **0.30/0.56** | -35%/-26% |
| SDD | 0.21/0.34 | **0.17/0.24** | -19%/-29% |
| NuScenes | 1.04/1.47 | **0.92/1.56** | -12%/+6% |

### 消融实验

各组件贡献（SDD/NuScenes ADE/FDE）：

| 组 | 关键组件 | SDD | NuScenes |
|----|----------|-----|----------|
| 1 | 仅位置+距离+方向损失 | 0.46/0.71 | 1.26/2.15 |
| 2 | +速度+角度 | 0.45/0.66 | 1.24/2.09 |
| 逐步添加 | +高阶意图+分布近似+KAN优化+方向损失 | → **0.17/0.24** | → **0.92/1.56** |

### 关键发现

- ETH/UCY 平均 ADE/FDE 比此前 SOTA (E-V2-Net-SC) 降低 13.3%/10.0%
- NBA 数据集上优势巨大（ADE 降低 35-37%），说明在强交互场景中多阶融合特别有效
- 方向损失的引入对各基线模型都有普适性提升（论文补充材料中验证）
- 定性分析显示模型正确处理了转向、减速避让、拥挤场景等复杂情况
- KAN 网络首次在轨迹预测中使用，实现了全时域并行优化

## 亮点与洞察

- 多阶意图融合的设计理念合理：一阶主导 + 高阶补充，通过可学习因子自适应平衡
- 首次将 KAN 引入轨迹预测的全局优化，是一个有趣的跨领域创新
- 距离-方向融合损失简单有效，关注了行人运动的动态状态而非仅终点位置
- 轨迹分布近似器通过上下界约束提升了 CVAE 框架的可解释性

## 局限与展望

- NuScenes 上 FDE 略有上升（1.56 vs 1.47），说明全局优化在某些场景可能引入偏差
- ETH 场景中平行人流交互弱，性能不如 E-V2-Net-SC，说明方法更适合强交互场景
- 高阶意图的子空间数 M=6 和 KAN 深度 L=3 是固定超参，自适应设定可能更好
- 未考虑静态环境约束（如障碍物、道路边界），纯社交力建模在复杂城市场景中可能不足
- 方向损失中的 arccos 在梯度计算时可能存在数值稳定性问题

## 相关工作与启发

- Social Force Model [12] 开创了行人交互建模，SocialMOIF 可视为其深度学习时代的延续
- V2-Net + SocialCircle [41] 系列是直接对标的 SOTA，SocialMOIF 通过更精细的交互建模超越
- KAN [24] 的可学习激活函数理论为轨迹优化提供了参数效率和表达力的良好折中
- 启发：轨迹预测中"谁影响谁"的交互建模远未饱和，高阶传导效应值得更多探索

## 评分

- **新颖性**: 7/10 — 多阶意图融合和 KAN 引入有新意，但基本框架（注意力+CVAE）较常规
- **实验充分度**: 9/10 — 四个主流数据集全面评估，消融实验详尽，定性分析丰富
- **写作质量**: 7/10 — 方法描述清晰但公式密集，部分符号不够直观
- **价值**: 8/10 — NBA 等强交互场景的大幅提升令人印象深刻，开源代码增加实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Certified Human Trajectory Prediction](certified_human_trajectory_prediction.md)
- [\[AAAI 2026\] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction](../../AAAI2026/autonomous_driving/catformer_causal_temporal_transformer_with_dynamic_contextual_fusion_for_driving.md)
- [\[CVPR 2025\] Physical Plausibility-aware Trajectory Prediction via Locomotion Embodiment](physical_plausibility-aware_trajectory_prediction_via_locomotion_embodiment.md)
- [\[CVPR 2025\] GDFusion: Rethinking Temporal Fusion with a Unified Gradient Descent View for 3D Semantic Occupancy Prediction](gdfusion_temporal_fusion_occupancy.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)

</div>

<!-- RELATED:END -->
