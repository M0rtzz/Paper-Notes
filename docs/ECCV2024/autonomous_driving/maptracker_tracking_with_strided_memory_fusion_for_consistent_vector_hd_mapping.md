---
title: >-
  [论文解读] MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping
description: >-
  [ECCV 2024][自动驾驶][向量高精地图] 将在线向量高精地图构建重新定义为追踪任务，通过双表示（BEV栅格+道路元素向量）的步进式记忆缓冲区融合机制实现时间一致的高精地图重建，在nuScenes和Argoverse2上分别以76.1和76.9 mAP大幅超越现有方法。
tags:
  - ECCV 2024
  - 自动驾驶
  - 向量高精地图
  - 追踪范式
  - 记忆机制
  - BEV感知
  - 时间一致性
---

# MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping

**会议**: ECCV 2024  
**arXiv**: [2403.15951](https://arxiv.org/abs/2403.15951)  
**代码**: 有 (https://map-tracker.github.io)  
**领域**: 自动驾驶 / 高精地图  
**关键词**: 向量高精地图, 追踪范式, 记忆机制, BEV感知, 时间一致性

## 一句话总结

将在线向量高精地图构建重新定义为追踪任务，通过双表示（BEV栅格+道路元素向量）的步进式记忆缓冲区融合机制实现时间一致的高精地图重建，在nuScenes和Argoverse2上分别以76.1和76.9 mAP大幅超越现有方法。

## 研究背景与动机

在线向量高精地图（Vector HD Mapping）从车载传感器数据实时重建道路几何结构（人行横道、车道分隔线、道路边界），对自动驾驶至关重要。如能从单次行驶中重建一致的城市高精地图，将极大降低全球数万城市的地图制作成本。

现有方法存在以下关键问题：

**检测而非追踪**：MapTRv2、StreamMapNet等主流方法采用**逐帧检测**范式，每帧从头检测道路元素而不强制执行时间一致性。虽然可能利用前一帧的重建结果作引导（如StreamMapNet的条件检测），但无法保证同一道路元素在不同帧被识别为同一实例。

**记忆机制的局限**：StreamMapNet等方法采用标准RNN风格的循环潜在嵌入作为记忆，将全部历史信息压缩到单个潜在记忆中。在车辆遮挡严重的复杂环境中，单一记忆容易发生"记忆丢失"——被遮挡的道路结构难以从单一记忆中恢复。

**Ground Truth和评估指标的缺陷**：现有基准数据集（MapTR、StreamMapNet版本）的GT存在帧间不一致问题（如人行横道合并策略不当、分隔线被错误切分），标准mAP指标也不惩罚时间不一致的重建结果，无法衡量方法的时间一致性。

**核心洞察**：高精地图重建本质上是一个**追踪**问题——道路元素持续存在于世界中，应该跨帧关联同一元素而非每帧重新检测。结合冗余但鲁棒的多帧记忆缓冲区（而非单一记忆），可以显著提升时间一致性。

## 方法详解

### 整体框架

MapTracker的核心是**双表示记忆机制**，包含两类记忆缓冲区：
1. **BEV记忆**：鸟瞰视图坐标系下的2D潜在图像，积累空间上下文
2. **VEC记忆**：每个道路元素的潜在向量，维护元素级追踪信息

传感器流被依次处理：图像骨干网络提取特征 → BEV模块构建和融合BEV记忆 → VEC模块利用BEV信息和向量记忆进行追踪和几何重建。

### 关键设计

1. **BEV记忆缓冲区与步进融合**：解决单一记忆的信息丢失问题。

   BEV记忆 $\mathbf{M}_{\text{BEV}}(t) \in \mathbb{R}^{50 \times 100 \times 256}$ 是以车辆为中心的2D潜在图像，覆盖左右15m、前后30m区域。缓冲区保存最近20帧的记忆，使记忆机制冗余但鲁棒。

   **BEV查询传播**：通过车辆运动的仿射变换 $P_{t-1}^t$ 和双线性插值将前一帧BEV记忆对齐到当前帧。新进入视野的区域用可学习的embedding $\mathbf{M}_{\text{BEV}}^{\text{init}}$ 初始化（MaskBlend操作）。

   **步进式记忆选择**：不使用所有20帧（计算量大且冗余），而是根据车辆行驶距离选择4个距离步进的记忆（距当前位置1m/5m/10m/15m最近的帧），通过坐标对齐后用轻量残差卷积块融合。

   **设计动机**：距离步进而非时间步进确保了空间覆盖的均匀性和效率。同时包含近距离（精细细节）和远距离（全局上下文）的记忆信息。

2. **VEC记忆与追踪式查询传播**：将高精地图构建从检测范式转变为追踪范式。

   VEC记忆 $\mathbf{M}_{\text{VEC}}(t) \in \{\mathbb{R}^{512}\}$ 是一组向量潜在表示，每个对应一个活跃的道路元素。初始化时分为两部分：

   $$\mathbf{M}_{\text{VEC}}(t) = [\mathbf{M}_{\text{VEC}}^{\text{prop}}(t), \mathbf{M}_{\text{VEC}}^{\text{new}}(t)]$$

   - $\mathbf{M}_{\text{VEC}}^{\text{prop}}(t)$：**传播查询**，从 $\mathbf{M}_{\text{VEC}}(t-1)$ 中继承的已追踪道路元素，通过PropMLP对齐坐标系。PropMLP接收运动变换 $P_{t-1}^t$ 的旋转四元数和平移向量的位置编码，与向量潜在拼接后映射。
   - $\mathbf{M}_{\text{VEC}}^{\text{new}}(t)$：**新检测查询**，100个可学习embedding $\mathbf{M}_{\text{VEC}}^{\text{init}}$，用于发现新进入视野的道路元素。

   **向量记忆融合**：对每个道路元素，从缓冲区中选择同一元素的历史潜在向量（相同距离步进策略），通过cross-attention融合。查询是当前帧的传播潜在，键/值是历史帧的对齐潜在。

   **设计动机**：借鉴了TrackFormer/MOTR等视觉目标追踪的查询传播范式。传播查询提供跨帧关联，新检测查询处理新元素发现，两者组合实现了完整的追踪+检测闭环。

3. **一致性感知基准与评估指标**：不只改进方法，还修复了评估体系。

   **GT改进**：修复现有基准的两类问题——nuScenes中大型人行横道偶发分裂为小环路，Argoverse2中分隔线被错误切分为短段。通过帧间最优二部匹配建立道路元素的时间对应关系（ground truth tracks）。

   **一致性mAP（C-mAP）**：在标准mAP流程的基础上增加时间一致性检查。如果一个重建元素的"祖先"（同一track中前一帧的元素）没有被匹配为正确检测，则当前帧的匹配也被视为不一致并移除。这有效惩罚了时间不一致的重建。

   **设计动机**：标准mAP独立评估每帧，一个方法即使每帧输出都"正确"但帧间不一致，也无法被发现。C-mAP填补了这一评估空白。

### 损失函数 / 训练策略

**总损失**：

$$\mathcal{L} = \mathcal{L}_{\text{BEV}} + \mathcal{L}_{\text{track}} + \lambda_5 \mathcal{L}_{\text{trans}}$$

- **BEV损失**：$\mathcal{L}_{\text{BEV}} = \lambda_1 \mathcal{L}_{\text{focal}} + \lambda_2 \mathcal{L}_{\text{dice}}$，分割监督
- **追踪损失**：$\mathcal{L}_{\text{track}} = \lambda_3 \mathcal{L}_{\text{focal}} + \lambda_4 \mathcal{L}_{\text{line}}$，扩展MOTR的匹配损失，传播查询使用前帧匹配结果继承标签，新查询进行匈牙利匹配
- **变换损失**：$\mathcal{L}_{\text{trans}}$ 训练PropMLP保持向量几何和类别不变
- 权重：$\lambda_1=10.0, \lambda_2=1.0, \lambda_3=5.0, \lambda_4=50.0, \lambda_5=0.1$

**三阶段训练**：
1. 预训练图像骨干和BEV编码器（仅 $\mathcal{L}_{\text{BEV}}$）
2. 冻结其他参数热启动向量解码器（大batch size加速收敛，500 warmup后开启向量记忆）
3. 联合训练所有参数

- 优化器：AdamW，初始lr=5e-4，余弦衰减至1.5e-6
- 8×RTX A5000，nuScenes训练72 epochs约3天，推理约10 FPS

## 实验关键数据

### 主实验

**nuScenes数据集**（使用一致性GT）：

| 方法 | Backbone | APp | APd | APb | mAP | C-mAP |
|------|----------|-----|-----|-----|-----|-------|
| MapTRv2 | R50 | 69.6 | 68.5 | 70.3 | 69.5 | 50.5 |
| StreamMapNet | R50 | 70.0 | 72.9 | 68.3 | 70.4 | 56.4 |
| **MapTracker** | **R50** | **80.0** | **74.1** | **74.1** | **76.1** | **69.1** |

**Argoverse2数据集**（使用一致性GT）：

| 方法 | Backbone | APp | APd | APb | mAP | C-mAP |
|------|----------|-----|-----|-----|-----|-------|
| MapTRv2 | R50 | 68.3 | 75.6 | 68.9 | 70.9 | 56.1 |
| StreamMapNet | R50 | 70.5 | 74.2 | 66.1 | 70.3 | 57.5 |
| **MapTracker** | **R50** | **77.0** | **80.0** | **73.7** | **76.9** | **68.3** |

MapTracker在两个数据集上均大幅领先：mAP提升**8%+**，C-mAP提升**19%+**。

### 消融实验

**核心组件消融**（nuScenes一致性GT）：

| 方法 | 任务 | 记忆 | mAP | C-mAP | 说明 |
|------|------|------|-----|-------|------|
| Baseline (无时序) | 检测 | 无 | 69.9 | 56.1 | StreamMapNet去掉时序模块 |
| StreamMapNet | 条件检测 | 单一记忆 | 70.4 | 56.4 | RNN风格记忆 |
| MapTracker (仅追踪) | 追踪 | 单一记忆 | 70.8 | 62.4 | 追踪范式显著提升C-mAP |
| +记忆融合 | 追踪 | 缓冲区(最近4帧) | 74.9 | 68.1 | 记忆缓冲区大幅提升性能 |
| **+步进选择** | **追踪** | **步进缓冲区** | **76.1** | **69.1** | 距离步进进一步优化 |

**步进距离选择消融**：

| 缓冲区大小 | 步进距离(m) | mAP | C-mAP |
|-----------|------------|-----|-------|
| 4 | 无(最近4帧) | 74.9 | 68.1 |
| 20 | {0,0,0,0} | 75.0 | 68.2 |
| 20 | {1,5,10,15} | **76.1** | **69.1** |

### 关键发现

- **追踪 vs 检测**：仅切换到追踪范式（不加记忆融合），C-mAP就从56.4跳到62.4（+6%），证明追踪对时间一致性的核心贡献
- **记忆融合的巨大增益**：添加记忆缓冲区+融合后mAP从70.8到74.9（+4.1%），C-mAP从62.4到68.1（+5.7%），是最大的性能提升来源
- **距离步进优于时间步进**：{0,0,0,0}（重复最近帧）mAP=75.0 vs {1,5,10,15}（距离步进）mAP=76.1，距离步进提供更好的空间覆盖
- **地理非重叠评估**：在更严格的非重叠测试集上，MapTracker优势更大（100×50m范围：36.2 vs 23.0 mAP），展现更强泛化能力
- **GT质量影响**：所有方法在修复后的一致性GT上性能均有提升，说明GT质量对训练至关重要

## 亮点与洞察

1. **范式转变**：将高精地图从"逐帧检测"重新定义为"持续追踪"，与道路元素的物理特性（持续存在于世界中）完美契合
2. **双表示记忆设计**：BEV栅格记忆捕获空间上下文，VEC向量记忆维护元素级追踪，两者互补且各自有缓冲区和融合机制
3. **基准贡献**：不仅改进方法，还修复了GT数据和评估指标，推动了更公平和有意义的评估
4. **简单而有效的记忆选择**：距离步进策略无需学习，仅通过空间覆盖的均匀性即获得显著增益

## 局限与展望

- **推理速度**：约10 FPS，相比MapTRv2等单帧方法慢，因为需要维护和融合记忆缓冲区
- **依赖车辆运动估计**：记忆对齐需要准确的帧间运动变换，对定位误差敏感（训练时加入随机扰动缓解）
- **nuScenes非重叠集性能下降**：坐标回归误差较大，说明几何泛化仍有提升空间
- 可探索：将追踪+记忆融合的设计扩展到3D目标检测或占据网格等其他自动驾驶感知任务
- 可探索：结合LiDAR等多模态传感器进一步提升遮挡场景的鲁棒性

## 相关工作与启发

- **TrackFormer/MOTR**：提供了查询传播的追踪范式，MapTracker将其引入高精地图领域
- **StreamMapNet**：最直接的基线，采用条件检测+RNN式记忆，MapTracker在其基础上升级为追踪+缓冲区记忆
- **BEVFormer**：提供了透视图到BEV的cross-attention机制，MapTracker直接继承
- **Sparse4Dv2/v3**：在3D目标检测中使用RNN式记忆和时序去噪，类似思想在高精地图中的对应
- 启发：**追踪范式+冗余记忆**的组合可能是所有需要时间一致性的在线感知任务的通用解法

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 追踪范式和步进记忆融合的组合设计虽非单点突破但组合效果显著
- **实验充分度**: ⭐⭐⭐⭐⭐ — 两个数据集、多个GT版本、核心组件消融、步进距离消融、地理非重叠评估，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，Fig. 1的全局地图对比非常有说服力
- **价值**: ⭐⭐⭐⭐⭐ — 大幅提升了高精地图的时间一致性，修复了基准数据和指标，对自动驾驶社区贡献突出

<!-- RELATED:START -->

## 相关论文

- [MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving](risk-aware_self-consistent_imitation_learning_for_trajectory_planning_in_autonom.md)
- [Accelerating Online Mapping and Behavior Prediction via Direct BEV Feature Attention](accelerating_online_mapping_and_behavior_prediction_via_dire.md)
- [Stream Query Denoising for Vectorized HD-Map Construction](stream_query_denoising_for_vectorized_hd-map_construction.md)
- [LiDAR-Event Stereo Fusion with Hallucinations](lidarevent_stereo_fusion_with_hallucinations.md)

<!-- RELATED:END -->
