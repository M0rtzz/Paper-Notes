---
title: >-
  [论文解读] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement
description: >-
  [ICCV 2025][自动驾驶][轨迹预测] 提出 Soft-Braid Attention，通过"软交叉点"显式建模轨迹间和轨迹与车道间的时空拓扑关系来指导多智能体轨迹精炼，在 Argoverse v2 和 INTERACTION 两个数据集上对四种基线方法均实现显著提升，建立了轨迹精炼任务的新 SOTA。
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "轨迹预测"
  - "轨迹精炼"
  - "拓扑结构"
  - "辫子理论"
  - "多智能体交互"
---

# SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement

**会议**: ICCV 2025  
**arXiv**: [2507.04263](https://arxiv.org/abs/2507.04263)  
**代码**: [github.com/Liwen-Xiao/SRefiner](https://github.com/Liwen-Xiao/SRefiner)  
**领域**: 自动驾驶  
**关键词**: 轨迹预测, 轨迹精炼, 拓扑结构, 辫子理论, 多智能体交互

## 一句话总结

提出 Soft-Braid Attention，通过"软交叉点"显式建模轨迹间和轨迹与车道间的时空拓扑关系来指导多智能体轨迹精炼，在 Argoverse v2 和 INTERACTION 两个数据集上对四种基线方法均实现显著提升，建立了轨迹精炼任务的新 SOTA。

## 研究背景与动机

### 问题定义

多智能体未来轨迹预测是自动驾驶安全决策的核心任务。轨迹精炼（Trajectory Refinement）作为一种后处理策略，将基线模型预测的粗糙轨迹作为输入，通过建模轨迹与场景的精细交互来输出更准确、更合理的轨迹。

### 已有方法的不足

现有轨迹精炼方法的核心问题在于**隐式交互建模**，未能利用轨迹的拓扑结构信息：

**QCNet**：将粗预测轨迹编码为锚点查询，再与场景上下文融合来预测轨迹偏移，但交互是隐式的

**R-Pred**：使用局部注意力机制在邻近车辆间精炼，但缺乏显式的拓扑指导

**MTR++**：利用意图点引导车辆间信息交换，但仍属于隐式关系学习

**SmartRefine**：场景自适应精炼策略，动态调整迭代次数和注意力范围，但未考虑拓扑关系

辫子理论（Braid Theory）已被证明在机器人领域有效（如系绳无人机路径规划、抓取等），直接应用于轨迹预测（BeTop）虽有效但存在三个关键局限：

- **忽视非交叉交互**：例如红车减速让蓝车，两条轨迹不交叉但行为逻辑相关
- **忽略时间动态**：辫子拓扑只关注空间交叉关系，不编码时间维度的动态交互
- **表达能力有限**：只能回答轨迹是否交互，无法捕捉交互的具体方式

### 核心动机

需要一种**显式且富表达力**的拓扑表示来指导轨迹精炼。关键洞察是：即使两条轨迹不在空间上交叉，它们在同一时刻最接近的点（"软交叉点"）处的运动状态和空间关系仍然编码了丰富的交互信息。

## 方法详解

### 整体框架

SRefiner 是一个多迭代、多智能体轨迹精炼框架，包含三个核心模块：
1. **Trajectory-Trajectory Soft-Braid Attention**：建模轨迹间的软辫子拓扑
2. **Trajectory-Lane Soft-Braid Attention**：建模轨迹与车道间的软辫子拓扑
3. **迭代精炼与拓扑更新**：每次迭代使用上一轮精炼后的轨迹更新拓扑信息

### 关键设计

#### 1. **软辫子拓扑（Soft-Braid Topology）**

- **功能**：为任意两条轨迹 $y_i$ 和 $y_j$ 定义"软交叉点"，编码在该点处的运动状态和空间关系，构成一个富表达力的拓扑描述符。
- **核心思路**：

  定义软交叉点为两条轨迹在同一时刻距离最近的点：
  $$t_{ij} = \arg\min_t \|y_i(t) - y_j(t)\|$$
  $$P_{i_j} = y_i(t_{ij}), \quad P_{j_i} = y_j(t_{ij})$$

  软辫子拓扑编码包含六个维度的信息：
  $$\tilde{\sigma}_{i \leftarrow j} = [\dot{y}_i^{(i)}(t_{ij}), \dot{y}_j^{(i)}(t_{ij}), \ddot{y}_i^{(i)}(t_{ij}), \ddot{y}_j^{(i)}(t_{ij}), d_{ij}, \theta_{ij}^{(i)}]$$

  其中 $\dot{y}$ 和 $\ddot{y}$ 分别为速度和加速度（在智能体 $i$ 的局部坐标系中），$d_{ij}$ 为软交叉点间的距离，$\theta_{ij}$ 为连线方向角。

- **设计动机**：
    - 与辫子拓扑（只有交叉/非交叉的二值关系）不同，软辫子拓扑为**所有**轨迹对建立拓扑连接
    - 在局部坐标系中表示保证了旋转不变性
    - 编码速度和加速度捕捉了时间动态（如加速/减速/等待等行为）
    - 距离和角度刻画了空间关系的细节

#### 2. **轨迹-轨迹软辫子注意力（Trajectory-Trajectory Soft-Braid Attention）**

- **功能**：利用软辫子拓扑信息指导轨迹间的信息融合。
- **核心思路**：

  通过多头交叉注意力实现信息交互，将拓扑信息融入 Key 和 Value：
  $$F_i = \text{MHCA}(Q: F_i, \; K: \{F_j + \varphi(\tilde{\sigma}_{i \leftarrow j})\}_{j \in \Omega(i)}, \; V: \{F_j + \varphi(\tilde{\sigma}_{i \leftarrow j})\}_{j \in \Omega(i)})$$

  其中 $\varphi(\cdot)$ 是 3 层 MLP，$\Omega(i) = \{j | d_{ij} \leq \tau_a\}$ 为邻域集合（$\tau_a = 50$m）。

- **设计动机**：通过将拓扑特征添加到 Key/Value 中，注意力权重的计算自然会考虑拓扑关系——距离近且运动状态相关的轨迹对会获得更高的注意力权重。

#### 3. **轨迹-车道软辫子注意力（Trajectory-Lane Soft-Braid Attention）**

- **功能**：将软辫子拓扑扩展到轨迹与高精地图车道线之间的交互建模。
- **核心思路**：

  定义轨迹 $y_i$ 上距离车道 $L_k$ 最近的点为软交叉点 $C_{i_k}$：
  $$t_{ik} = \arg\min_t \|y_i(t) - L_k\|$$

  轨迹-车道拓扑特征：
  $$\tilde{\lambda}_{i \leftarrow k} = [\dot{y}_i^{(i)}(t_{ik}), \ddot{y}_i^{(i)}(t_{ik}), d_{ik}, \theta_{ik}^{(i)}]$$

  同样通过 MHCA 融合，邻域阈值 $\tau_l = 10$m。

- **设计动机**：轨迹与车道的关系对于保证轨迹合理性至关重要（如不偏离可行驶区域、不侵入对向车道），通过编码智能体在最近点处的运动状态，可以捕捉到"是否正在靠近/远离车道"等动态信息。

#### 4. **迭代精炼与渐进拓扑更新**

- **功能**：多次迭代精炼轨迹，每次使用上一轮精炼后的轨迹重新计算拓扑。
- **核心思路**：
  $$\tilde{B}_{l-1}, \tilde{B}'_{l-1} = \mathcal{S}(Y_{l-1}, L)$$
  $$F_l = \text{SoftBraidAttn}(F_{l-1}, L, \tilde{B}_{l-1}, \tilde{B}'_{l-1})$$
  $$Y_l = \varphi(F_l) + Y_{l-1}$$

  默认迭代 $I=3$ 次。

- **设计动机**：初始粗轨迹的拓扑可能不准确，使用精炼后的轨迹更新拓扑可以逐步提高拓扑信息的质量，形成正反馈循环。

### 损失函数 / 训练策略

- 使用 **Joint Winner-Takes-All (WTA) Loss**：在 $K$ 个模式中选择与真实轨迹联合位移误差最小的模式进行优化
- 最优模式选择：$k_l = \arg\min_{k \in [1,K]} \frac{1}{N}\sum_{i=1}^N \|Y_{l,i,k} - Y_{\text{gt},i}\|$
- 使用 **Huber Loss** 监督每个迭代的输出
- 总损失为所有迭代损失的平均：$\mathcal{L} = \frac{1}{I}\sum_{l=1}^I \mathcal{L}_l$

## 实验关键数据

### 主实验

在 Argoverse v2 和 INTERACTION 两个数据集上对四种基线方法的提升效果：

| 数据集 | 基线 | 指标 | 原始值 | +SRefiner | 提升 |
|--------|------|------|--------|-----------|------|
| AV2 Val | Forecast-MAE | avgMinFDE↓ | 1.642 | 1.477 | -10.1% |
| AV2 Val | FJMP | avgMinFDE↓ | 1.920 | 1.736 | -9.6% |
| AV2 Test | Forecast-MAE | avgMinFDE↓ | 1.679 | 1.521 | -9.4% |
| INTER Test | AutoBots | minJointFDE↓ | 1.015 | 0.906 | -10.7% |
| INTER Test | FJMP | minJointFDE↓ | 0.945 | 0.867 | -8.3% |
| INTER Test | HPNet | minJointFDE↓ | 0.823 | 0.797 | -3.2% |

与其他精炼方法的比较（Argoverse v2，基线 Forecast-MAE）：

| 精炼方法 | avgMinFDE↓ | avgMinADE↓ | actorMR↓ | 延迟 |
|----------|-----------|-----------|----------|------|
| Baseline | 1.642 | 0.717 | 0.194 | – |
| DCMS | 1.601 | 0.702 | 0.190 | 5ms |
| R-Pred | 1.554 | 0.683 | 0.187 | 12ms |
| QCNet | 1.520 | 0.674 | 0.185 | 58ms |
| MTR++ | 1.495 | 0.670 | 0.183 | 54ms |
| **SRefiner** | **1.477** | **0.658** | **0.183** | 28ms |

### 消融实验

| 配置 | avgMinFDE↓ | avgMinADE↓ | actorMR↓ |
|------|-----------|-----------|----------|
| 基线（无精炼） | 1.642 | 0.717 | 0.194 |
| +Traj-Traj + Traj-Lane（无拓扑更新） | 1.522 | 0.673 | 0.186 |
| +Traj-Traj + 拓扑更新（无 Traj-Lane） | 1.497 | 0.670 | 0.183 |
| +Traj-Lane + 拓扑更新（无 Traj-Traj） | 1.514 | 0.678 | 0.184 |
| **完整模型** | **1.477** | **0.658** | **0.183** |

拓扑类型对比：

| 拓扑类型 | avgMinFDE↓ |
|----------|-----------|
| 无拓扑 | 1.530 |
| Braid Topology (BeTop) | 1.512 |
| Soft-Braid（仅 Traj-Traj） | 1.497 |
| **Soft-Braid（完整）** | **1.477** |

### 关键发现

1. SRefiner 在所有四种基线和两个数据集上都实现了显著提升，证明其强泛化能力
2. 软辫子拓扑优于辫子拓扑（1.477 vs 1.512），验证了"软交叉点"设计的有效性
3. SRefiner 达到 SOTA 精度的同时延迟仅 28ms，优于 QCNet (58ms) 和 MTR++ (54ms)
4. 渐进拓扑更新策略贡献显著（移除后 avgMinFDE 从 1.477 退化至 1.522）
5. 从可视化看，SRefiner 有效减少了碰撞轨迹和偏离可行驶区域的轨迹

## 亮点与洞察

1. **从辫子理论到深度学习**：将代数拓扑中的辫子理论引入轨迹预测，是跨学科融合的典范
2. **"软交叉点"的设计妙处**：将离散的交叉/非交叉关系放松为连续的距离-运动状态表示，既保留了拓扑的结构化先验，又增强了表达能力
3. **即插即用**：SRefiner 可以无缝集成到现有的多智能体轨迹预测方法中
4. **效率优势**：相比 SmartRefine 需要为每个智能体单独推理（$24 \times N$ ms），SRefiner 同时精炼所有智能体（28ms）

## 局限与展望

1. **仅考虑车辆**：未涵盖行人和自行车等异构交通参与者
2. **拓扑计算开销**：每次迭代都需要重新计算所有轨迹对的软交叉点，计算复杂度为 $O(N^2 T)$
3. **固定邻域阈值**：$\tau_a = 50$m 和 $\tau_l = 10$m 为固定值，未考虑场景自适应
4. **仅适用于有 HD Map 的场景**：Trajectory-Lane 模块依赖高精地图信息
5. **未考虑多模态不确定性**：软辫子拓扑在不同预测模式下的表示是固定的

## 相关工作与启发

- 与 **BeTop** 的区别：BeTop 直接应用辫子拓扑的二值交叉关系，SRefiner 提出了更富表达力的"软辫子"连续拓扑
- 与 **MTR++** 的区别：MTR++ 通过意图点隐式建模交互，SRefiner 通过显式拓扑特征指导交互
- 从机器人领域（tethered drone planning）借鉴辫子理论到自动驾驶轨迹预测，展示了跨领域技术迁移的潜力

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 将辫子理论扩展为"软辫子"用于轨迹精炼，概念新颖且有理论基础
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集、四种基线、五种精炼方法对比、详细消融，但缺少真实场景下游任务评估
- **写作质量**: ⭐⭐⭐⭐ — 从辫子理论到软辫子的动机链条清晰，图示直观
- **价值**: ⭐⭐⭐⭐ — 为轨迹精炼提供了新的拓扑视角，即插即用的设计实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](../../CVPR2025/autonomous_driving/trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](../../ICML2025/autonomous_driving/hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[ICCV 2025\] DONUT: A Decoder-Only Model for Trajectory Prediction](donut_a_decoder-only_model_for_trajectory_prediction.md)
- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](where_what_why_towards_explainable_driver_attention_prediction.md)

</div>

<!-- RELATED:END -->
