---
title: >-
  [论文解读] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning
description: >-
  [ICML2025][离线强化学习] 提出 Graph-Assisted Stitching (GAS) 框架，用基于图搜索的子目标选择替代显式高层策略学习，通过时间距离表示 (TDR) 空间中的聚类构图与最短路径规划，在离线 HRL 中实现高效的跨轨迹拼接，在最具挑战的 antmaze-giant-stitch 任务上从前 SOTA 的 1.0 飙升至 88.3。
tags:
  - ICML2025
  - 离线强化学习
  - 层级RL
  - 轨迹拼接
  - 图搜索
  - 时间距离表示
  - 子目标选择
---

# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning

**会议**: ICML2025  
**arXiv**: [2506.07744](https://arxiv.org/abs/2506.07744)  
**代码**: [GitHub](https://github.com/qortmdgh4141/GAS)  
**领域**: 离线层级强化学习 (Offline Hierarchical RL)  
**关键词**: 离线强化学习, 层级RL, 轨迹拼接, 图搜索, 时间距离表示, 子目标选择

## 一句话总结

提出 Graph-Assisted Stitching (GAS) 框架，用基于图搜索的子目标选择替代显式高层策略学习，通过时间距离表示 (TDR) 空间中的聚类构图与最短路径规划，在离线 HRL 中实现高效的跨轨迹拼接，在最具挑战的 antmaze-giant-stitch 任务上从前 SOTA 的 1.0 飙升至 88.3。

## 研究背景与动机

离线层级强化学习 (offline HRL) 通过高层策略生成子目标、低层策略执行动作来解决长视野稀疏奖励任务。然而现有方法存在三大瓶颈：

**子目标采样缺乏时间感知**：多数方法在轨迹中按固定步数间隔采样子目标候选 (如每 $c$ 步取一个)，忽略了状态之间的真实时间距离关系，导致冗余或低效的子目标

**长视野推理能力不足**：HIQL 等方法在 antmaze-medium 上表现良好，但在 antmaze-giant 等超长视野环境中急剧下降，因为稀疏奖励使高层策略的学习信号过弱

**跨轨迹拼接能力缺失**：离线数据集包含多条目标不同的轨迹，理想情况下应能拼接不同轨迹中的有用片段形成新路径，但现有方法缺乏有效的跨目标拼接机制

核心思路：**既然高层策略难以学好，不如用图搜索替代它**——在 TDR 空间中构建状态图，通过 Dijkstra 最短路径算法直接选择子目标序列。

## 方法详解

GAS 包含四个核心组件：

### 1. 时间距离表示 (TDR) 预训练

将状态嵌入到潜空间 $\mathcal{H}$，使得两个状态的欧氏距离等于最优策略在它们之间转移所需的最小时间步数：

$$d^*(s, g) = \|\psi(s) - \psi(g)\|_2$$

通过 IQL 风格的 expectile 回归损失在离线数据上训练：

$$\mathcal{L} = \mathbb{E}_{(s,s',g)\sim\mathcal{D}}\left[\ell_\tau^2\left(-\mathbb{1}\{s \neq g\} + \gamma \bar{V}(s',g) - V(s,g)\right)\right]$$

其中 $V(s,g) = -\|\psi(s) - \psi(g)\|_2$，$\ell_\tau^2(x) = |\tau - \mathbb{1}(x<0)| \cdot x^2$。

### 2. TD-Aware 图构建

在 TDR 空间中以固定时间距离间隔 $H_{\text{TD}}$ 对所有状态进行聚类：

- 第一个状态作为第一个簇的中心
- 后续状态若与现有簇中心距离 $\leq H_{\text{TD}}$ 则分入最近簇，否则新建簇
- 簇中心 = 簇内状态均值 → 作为图节点
- 在距离 $\leq H_{\text{TD}}$ 的节点间建边

不同轨迹中语义相似的状态被聚合到同一节点，**自然实现跨轨迹拼接**。

### 3. 时间效率 (TE) 指标过滤

为避免低质量状态 (来自次优轨迹) 污染图结构，定义 TE 度量当前状态处转移的效率：

$$\theta_{\text{TE}} = \cos\left(\psi(s_{\text{opt}}) - \psi(s_{\text{cur}}),\ \psi(s_{\text{reached}}) - \psi(s_{\text{cur}})\right)$$

其中 $s_{\text{opt}}$ 是沿轨迹到达时间距离 $H_{\text{TD}}$ 的最优未来状态，$s_{\text{reached}}$ 是实际 $H_{\text{TD}}$ 步后到达的状态。仅保留 $\theta_{\text{TE}} \geq 0.9$ 的高效状态用于建图，实际使用状态仅占 2%–8%，节点数不到数据集总体的 1%。

### 4. 任务规划与执行

推理时不需要高层策略：

1. Dijkstra 算法预计算所有图节点到最终目标的最短距离
2. 找到当前状态在 $H_{\text{TD}}$ 范围内的可达节点
3. 选择 $v_{\text{subgoal}} = \arg\min_{v \in \mathcal{V}_{\text{near}}} (\text{Dists}[v] + \|h_{\text{cur}} - v\|_2)$
4. 低层策略沿方向向量 $\vec{h}_{\text{dir}} = \text{dir}(\psi(s_t), \psi(s_{\text{sub}}))$ 执行动作

低层策略训练采用 DDPG+BC 损失 + IQL 值函数，子目标采样也改为按 TDR 距离而非固定步数。

## 实验关键数据

### State-based 基准 (OGBench)

| 任务 | GCBC | HIQL | HHILP | **GAS** |
|------|------|------|-------|---------|
| antmaze-medium-navigate | 33.1 | 95.3 | 96.3 | **96.3** |
| antmaze-large-navigate | 23.4 | 89.9 | 86.8 | **93.2** |
| antmaze-giant-navigate | 0.0 | 67.3 | 53.1 | **77.6** |
| antmaze-medium-stitch | 43.2 | 92.0 | 96.0 | **98.1** |
| antmaze-large-stitch | 2.3 | 71.7 | 34.1 | **96.3** |
| antmaze-giant-stitch | 0.0 | 1.0 | 0.0 | **88.3** |
| antmaze-large-explore | 0.0 | 2.9 | 2.4 | **94.2** |
| scene-play | 5.4 | 40.0 | 43.4 | **73.6** |
| kitchen-partial | 69.5 | 73.1 | 66.7 | **87.3** |

### 关键发现

- **拼接任务突破性提升**：antmaze-giant-stitch 从 HIQL 的 1.0 → **88.3** (+8730%)
- **探索型数据集优势巨大**：antmaze-large-explore 从 HIQL 2.9 → **94.2**
- **视觉环境同样有效**：visual-antmaze-giant-stitch 55.8 (前 SOTA 仅 3.6)

### TE 过滤消融

| 任务 | 全量状态建图 | TE 过滤后 | 提升 |
|------|-------------|----------|------|
| antmaze-giant-navigate | 63.4 | **77.6** | +14.2 |
| antmaze-giant-stitch | 75.3 | **88.3** | +13.0 |
| antmaze-large-explore | 75.4 | **94.2** | +18.8 |

## 亮点与洞察

1. **范式转换**：用图搜索替代高层策略学习，避开了稀疏奖励下高层策略难训练的根本性困难
2. **TE 指标设计精妙**：仅用余弦相似度就能有效过滤次优状态，使用 2%–8% 数据反而比全量数据效果更好
3. **统一的 $H_{\text{TD}}$ 参数**：一个超参数同时控制聚类间距、建边阈值、子目标采样距离和执行时的可达范围，保证了训练与测试的一致性
4. **拼接能力极端场景验证**：在 antmaze-giant-stitch 这一需要拼接能力最强的任务上实现从 1.0 到 88.3 的突破

## 局限与展望

1. **视觉环境性能仍有差距**：相比 state-based 环境，视觉环境整体性能偏低，作者承认缺乏针对高维视觉输入的表示学习
2. **$H_{\text{TD}}$ 需要逐环境调优**：不同任务的最优 $H_{\text{TD}}$ 不同，缺乏自适应选择机制
3. **图结构为静态**：推理时构建一次图后不再更新，无法根据执行反馈在线修正
4. **Dijkstra 每 episode 重算**：虽然节点数 <1%，但大规模环境中可能仍有延迟

## 相关工作与启发

- **HIQL** (Park et al., 2023)：用值函数学习的潜空间表示做高层策略，GAS 的 TDR 和低层训练借鉴了 HIQL 的框架
- **HHILP** (Park et al., 2024c)：提出了 TDR 表示学习，GAS 在此基础上用图搜索替代高层策略
- **OGBench** (Park et al., 2025a)：提供了包含 navigate/stitch/explore 三类数据集的标准化评估基准

## 评分

- 新颖性: ⭐⭐⭐⭐ — 图搜索替代高层策略 + TE 过滤指标是核心创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 涵盖 locomotion/navigation/manipulation，state/visual，5 项消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图示直观，公式完整
- 价值: ⭐⭐⭐⭐⭐ — 在离线 HRL 拼接任务上实现数量级提升，实用意义显著

<!-- RELATED:START -->

## 相关论文

- [Structural Information-based Hierarchical Diffusion for Offline Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/structural_information-based_hierarchical_diffusion_for_offline_reinforcement_le.md)
- [Hierarchical Reinforcement Learning with Targeted Causal Interventions](hierarchical_reinforcement_learning_with_targeted_causal_interventions.md)
- [Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [Meta-Black-Box-Optimization through Offline Q-function Learning (Q-Mamba)](meta-black-box-optimization_through_offline_q-function_learning.md)

<!-- RELATED:END -->
