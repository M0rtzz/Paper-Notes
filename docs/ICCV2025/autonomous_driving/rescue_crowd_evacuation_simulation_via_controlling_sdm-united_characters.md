---
title: >-
  [论文解读] RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters
description: >-
  [自动驾驶] 提出首个在线 SDM（感知-决策-运动）统一 3D 疏散仿真框架 RESCUE，集成 3D 自适应社会力模型和个性化步态控制器，实现数百智能体的实时个性化疏散模拟。
tags:
  - 自动驾驶
---

# RESCUE: Crowd Evacuation Simulation via Controlling SDM-United Characters

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2507.20117](https://arxiv.org/abs/2507.20117)
- **代码**: [项目页面](http://cic.tju.edu.cn/faculty/likun/projects/RESCUE)
- **领域**: 自动驾驶
- **关键词**: 人群疏散模拟, 社会力模型, 物理引擎, 个性化步态, 3D角色控制

## 一句话总结

提出首个在线 SDM（感知-决策-运动）统一 3D 疏散仿真框架 RESCUE，集成 3D 自适应社会力模型和个性化步态控制器，实现数百智能体的实时个性化疏散模拟。

## 研究背景与动机

人群疏散模拟对公共安全至关重要，但现有方法无法同时满足以下需求：

**传统仿真的简化表示**：社会力模型（SFM）使用简化 2D 表示，力反馈只能用不真实的替代计算，无法反映 3D 碰撞、跌倒等真实物理行为。

**角色控制的扩展不足**：基于物理引擎的角色控制方法在密集场景中缺乏个性化运动，可能导致跌倒和碰撞。扩散模型的运动生成面临可控性和物理真实性挑战。

**两个关键需求**：3D 近身感知（拥挤中的在线动态调整）和个性化步态（不同属性个体表现不同行为）。

## 方法详解

### SDM 统一框架

模拟人脑的感知-决策-运动循环：

1. **感知模块**：每个智能体感知自身状态（位置、速度、人形状态）、他人状态（相对根位置）和环境状态
2. **决策模块**：3D 自适应 SFM 决策机制，A* 搜索找路径+SFM 驱动+避碰
3. **运动模块**：Pacer 路径跟随控制器 + 个性化步态转换器

### 3D 自适应 SFM 决策机制

**基础力**：驱动力 + 排斥力

**闪避力**（核心创新）：计算垂直于期望方向的闪避方向：

$$F_{\text{evasive}} = A \, \text{sgn}(o_i \cdot p_i) \, p_i$$

其中 $A$ 为二值掩码（前方有障碍且侧方有空间时为1），$p_i$ 为垂直向量。

**个性化 SFM 系数优化**：五组智能体（青年、中年、老年、患者、残疾人），校准速度使模拟速度匹配文献中的真实速度。

**最终决策**：$\tilde{v}_{i,t+1} = v_i + \Delta t (F_{\text{drive}} + F_{\text{repulsive}} + F_{\text{evasive}})$

### 个性化步态控制器

基于 CAMDM 扩散模型将非个性化动作帧转换为个性化动作帧：$a^0 = G(a^t, t; c)$。通过步态帧匹配（4个关键事件）对齐个性化和非个性化帧对。模拟中仅替换上半身动作。

### 部件级力可视化

为 24 个身体部件集成力传感器，通过颜色梯度可视化接触力大小。

## 实验

### 定量比较

| 方法 | 平均成功率 | 平均跌倒次数 |
|------|-----------|------------|
| OmniControl | 0.48 | — |
| MaskedMimic | 0.60 | 18.55 |
| **RESCUE** | **0.84** | **12.26** |

### 速度多样性分析

| 群体 | 速度特征 |
|------|---------|
| 青年 | 最高中值速度，变异性大 |
| 中年 | 中等速度分布 |
| 老年 | 较慢速度，范围窄 |
| 患者 | 速度有限 |
| 残疾人 | 最低速度 |

### 关键发现

1. **成功率最高**：0.84 vs MaskedMimic 0.60，得益于 3D 自适应 SFM 的在线避障
2. **跌倒次数最少**：12.26 vs 18.55，闪避力有效避免拥堵导致的摔倒
3. **密度-宽度踩踏分析**：相同宽度下人数越多踩踏越严重；相同密度下通道越窄踩踏越严重
4. **地形影响验证**：不平地面和滑地面更易导致跌倒

## 亮点与洞察

1. **脑科学启发**：SDM 循环赋予框架生物学合理性
2. **闪避力的实用创新**：3D 环境中闪避比等待更合理
3. **个性化系数校准**：补偿物理引擎摩擦因素，确保模拟速度匹配文献值
4. **部件级力可视化**：为疏散分析提供前所未有的洞察

## 局限性

1. 100STYLE 数据集的运动风格有限
2. 五组分类较为粗糙
3. 其他多智能体环境的适用性需验证

## 相关工作

- **人群模拟**：社会力模型、Agent-Based 模型
- **角色控制**：Pacer、MaskedMimic、DRL 方法
- **运动生成**：扩散模型、OmniControl、EMAGE

## 评分

- **新颖性**: ⭐⭐⭐⭐ — SDM 统一框架+3D 自适应 SFM 首创
- **技术深度**: ⭐⭐⭐⭐ — 多模块深度整合
- **实验充分性**: ⭐⭐⭐⭐⭐ — 多场景分析，用户研究
- **写作质量**: ⭐⭐⭐⭐ — 可视化丰富

<!-- RELATED:START -->

## 相关论文

- [SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [Resonance: Learning to Predict Social-Aware Pedestrian Trajectories as Co-Vibrations](resonance_learning_to_predict_social-aware_pedestrian_trajectories_as_co-vibrati.md)
- [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [Point-to-Region Loss for Semi-Supervised Point-Based Crowd Counting](../../CVPR2025/autonomous_driving/point-to-region_loss_for_semi-supervised_point-based_crowd_counting.md)
- [LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)

<!-- RELATED:END -->
