---
title: >-
  [论文解读] GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction
description: >-
  [ICML 2025][自动驾驶][逆强化学习] 首次将最大熵逆强化学习框架与向量化场景表示相融合，提出 GoIRL 轨迹预测框架：通过可学习的 Feature Adaptor 将图特征聚合到网格空间以适配 IRL，再用层级参数化轨迹生成器（Bézier曲线+精细化模块）和 MCMC 概率融合机制实现多模态轨迹预测，在 Argoverse 和 nuScenes 上达到 SOTA 并展现出相比监督模型显著更强的泛化能力。
tags:
  - ICML 2025
  - 自动驾驶
  - 逆强化学习
  - MaxEnt IRL
  - 向量化表示
  - 多模态轨迹预测
  - Bézier曲线
  - MCMC采样
---

# GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction

**会议**: ICML 2025  
**arXiv**: [2506.21121](https://arxiv.org/abs/2506.21121)  
**代码**: 无  
**领域**: 自动驾驶 / 轨迹预测  
**关键词**: 逆强化学习, MaxEnt IRL, 向量化表示, 多模态轨迹预测, Bézier曲线, MCMC采样

## 一句话总结

首次将最大熵逆强化学习框架与向量化场景表示相融合，提出 GoIRL 轨迹预测框架：通过可学习的 Feature Adaptor 将图特征聚合到网格空间以适配 IRL，再用层级参数化轨迹生成器（Bézier曲线+精细化模块）和 MCMC 概率融合机制实现多模态轨迹预测，在 Argoverse 和 nuScenes 上达到 SOTA 并展现出相比监督模型显著更强的泛化能力。

## 研究背景与动机

**领域现状**：轨迹预测的主流方法属于行为克隆（BC）范式——直接在大量驾驶数据上做监督学习拟合轨迹分布。近年来向量化表示（如 LaneGCN 的车道图卷积）替代了栅格化 BEV 图像，显著提升了精度。

**现有痛点**：BC 范式存在两个核心问题：(1) 协变量偏移（covariate shift）——当测试场景显著偏离训练分布时（如临时路障改变了可行驶区域），监督模型无法适应，产生严重的 compounding error；(2) 模态坍塌——训练时只有一条 GT 轨迹，模型难以学习一对多的映射关系。

**核心矛盾**：IRL 范式理论上能解决上述两个问题——奖励驱动+交互式学习天然处理分布偏移，最大熵原则天然捕获多模态性——但 IRL 方法依赖栅格化表示，在特征质量上远逊于当前的向量化监督模型。

**本文目标** (1) 将向量化场景表示引入 IRL 来弥补信息损失；(2) 缓解 IRL 在大状态空间下的计算开销；(3) 将 IRL 推断的粗粒度策略转化为精细的连续轨迹。

**切入角度**：观察到 IRL 的网格操作与向量化特征之间的鸿沟可以用一个可学习的适配器弥合——将图特征按物理位置映射到网格单元，再用 CNN 降维。

**核心 idea**：用 Feature Adaptor 将向量化特征注入 MaxEnt IRL 网格空间，让 IRL 基预测器首次获得与向量化监督模型同等的场景感知能力。

## 方法详解

### 整体框架

GoIRL 采用两阶段架构：**策略推断 → 轨迹生成**。

输入为驾驶上下文 $\mathcal{C} = \{\mathcal{X}, \mathcal{O}, \mathcal{M}\}$（目标智能体历史轨迹、周围智能体轨迹、HD地图信息）。第一阶段用图编码器提取向量化特征，通过 Feature Adaptor 转换到网格空间，运行 MaxEnt IRL 获得奖励分布和策略；第二阶段在采样的粗粒度路径上条件生成连续轨迹，经精细化模块得到最终预测。整个过程将多模态轨迹分布分解为：

$$P(\hat{\mathcal{Y}}|\mathcal{C}) = \sum_{\hat{\tau} \in \mathbb{S}(\mathcal{C})} P(\hat{\mathcal{Y}}|\hat{\tau}, \mathcal{C}) P(\hat{\tau}|\mathcal{C})$$

### 关键设计

1. **图导向上下文编码器（Graph-Oriented Context Encoder）**

    - 功能：以向量化方式编码驾驶场景，提取丰富的拓扑和语义特征
    - 核心思路：采用双层图结构——车道图（用 LaneGCN 的 dilated LaneConv 提取车道节点特征）+可行驶区域图（用 PointNet-like 网络提取占据节点特征）；智能体运动用 1D CNN + FPN 编码；最终用多层图注意力融合所有特征
    - 设计动机：向量化表示相比栅格化能更好地捕获复杂拓扑连接和长距离交互，是 GoIRL 性能提升的基础

2. **Feature Adaptor（特征适配器）**

    - 功能：将图特征无缝映射到 IRL 需要的网格空间
    - 核心思路：构建以目标智能体为中心的均匀网格，将可行驶区域上的融合特征按物理位置分配到对应网格单元，不可行驶区域填零；用带步长的 CNN 将高分辨率 $\mathcal{C}_{fine}$ 降采样为低分辨率 $\mathcal{C}_{coarse}$；再通过 $1 \times 1$ 卷积得到奖励分布 $\mathcal{R}$
    - 设计动机：IRL 依赖网格形状输入，而向量化特征是无序的图节点——Feature Adaptor 的桥接使两者首次兼容，CNN 降采样有效缓解大状态空间带来的 IRL 计算开销

3. **MaxEnt IRL 策略推断 + 层级轨迹生成**

    - 功能：从奖励分布推导多模态策略，并将离散路径转化为连续轨迹
    - 核心思路：将问题建模为有限 MDP $\{\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}\}$，9个离散动作（8方向+停止）；通过近似值迭代求解 soft-optimal 策略 $\pi(a|s) = \exp(Q(s,a) - V(s))$，$V(s) = \log \sum_a \exp(Q(s,a))$；用 MCMC 采样多条不同路径。条件于采样路径，用 Bézier 曲线递推预测控制点表示连续轨迹（保证平滑性），再用精细化模块利用完整轨迹检索局部上下文特征做位置修正。概率融合将 MCMC 分布与分类概率加权组合
    - 设计动机：MaxEnt 原则使策略倾向高熵分布，天然捕获多模态性；Bézier 参数化保证连续性和光滑性；精细化模块融合历史和预测实现时序一致性；MCMC 融合弥补分类概率对不确定性刻画的不足

### 损失函数 / 训练策略

IRL 阶段用 MaxEnt IRL 负对数似然监督奖励分布学习；轨迹生成阶段用 Winner-Takes-All 策略只对最佳预测施加 smooth L1 回归损失。两阶段联合端到端训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | GoIRL | QCNet | MTR | 提升 |
|--------|------|-------|-------|-----|------|
| Argoverse 1 | minADE₆↓ | **最优** | 次优 | 第三 | 显著 |
| Argoverse 1 | minFDE₆↓ | **最优** | 次优 | 第三 | 显著 |
| Argoverse 1 | MR₆↓ | **最优** | 次优 | 第三 | 显著 |
| nuScenes | minADE₅↓ | **最优** | - | 次优 | 显著 |
| nuScenes | minFDE₅↓ | **最优** | - | 次优 | 显著 |

### 消融实验

| 配置 | 影响 | 说明 |
|------|------|------|
| 去除 Feature Adaptor | minFDE退步~14% | 最关键组件，证明向量化注入的必要性 |
| 去除 MCMC 概率融合 | MR退步~9% | 概率融合增强多模态覆盖 |
| 去除精细化模块 | ADE/FDE退步 | 局部上下文检索提升精度 |
| 去除 Bézier 参数化 | 轨迹平滑性下降 | 参数化保证连续性 |

### 关键发现

- Feature Adaptor 是最关键组件，证实向量化特征注入是 GoIRL 性能的支柱
- 在可行驶区域变化的 OOD 场景中，GoIRL 明显优于监督模型——验证了 IRL 奖励驱动范式的泛化优势
- IRL 推断出的奖励分布具有良好可解释性——高奖励区域与合理行驶路径高度吻合

## 亮点与洞察

- **首次 MaxEnt IRL + 向量化表示**：Feature Adaptor 打破了 IRL 长期局限于栅格化输入的壁垒，使IRL预测器首次在大规模benchmark上超越监督方法，是连接两种技术路线的关键桥梁
- **OOD泛化优势的量化验证**：通过可行驶区域变化实验证明了IRL范式相比BC的根本优势——奖励函数编码环境约束而非行为分布，因此对环境变化有天然鲁棒性

## 局限与展望

- IRL 内循环的近似值迭代仍带来额外计算开销，推理速度低于纯监督方法
- 离散 9 方向动作空间粒度有限，大曲率转弯可能表达不精确
- Feature Adaptor 的最近邻分配对稀疏可行驶区域可能不够精确，可考虑注意力机制
- 仅验证了单智能体预测场景，多智能体交互下的表现未知

## 相关工作与启发

- **vs PGP (Guo et al., 2022)**：同为 IRL 基预测器，但 PGP 使用栅格化表示导致性能受限，GoIRL 通过 Feature Adaptor 解决了这一根本局限
- **vs QCNet/MTR++**：主流监督方法精度高但泛化弱，GoIRL 在标准场景超越的同时泛化优势更突出
- **vs GAIL/AIRL**：对抗式 IRL 方法计算成本更高、训练不稳定，GoIRL 的 MaxEnt IRL 更简洁高效

## 评分

- 新颖性: ⭐⭐⭐⭐ Feature Adaptor 巧妙弥合两种范式，首次在大规模benchmark上证明IRL可超越监督方法
- 实验充分度: ⭐⭐⭐⭐ 两个大规模benchmark+泛化实验+详细消融，但缺少推理速度对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法描述详细、图示直观
- 价值: ⭐⭐⭐⭐⭐ IRL在预测领域的复兴之作，OOD泛化结论对自驾安全有实际价值

<!-- RELATED:START -->

## 相关论文

- [BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning](r3dm_enabling_role_discovery_and_diversity_through_dynamics_models_in_multi-agen.md)
- [SeqGrowGraph: Learning Lane Topology as a Chain of Graph Expansions](../../ICCV2025/autonomous_driving/seqgrowgraph_learning_lane_topology_as_a_chain_of_graph_expansions.md)
- [Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](../../ICCV2025/autonomous_driving/foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](../../ICCV2025/autonomous_driving/generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

<!-- RELATED:END -->
