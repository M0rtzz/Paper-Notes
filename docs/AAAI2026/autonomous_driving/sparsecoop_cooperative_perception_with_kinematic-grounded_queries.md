---
title: >-
  [论文解读] SparseCoop: Cooperative Perception with Kinematic-Grounded Queries
description: >-
  [AAAI 2026][自动驾驶][协同感知] 提出 SparseCoop——首个完全稀疏的协同感知框架，通过运动学锚定查询（KGQ）、粗到精聚合模块和协同实例去噪策略，完全抛弃密集 BEV 表示，在 V2X-Seq 和 Griffin 数据集上以最低通信开销和最高计算效率达到 SOTA 性能（AP 0.530，传输仅 3.17×10⁴ BPS）。
tags:
  - AAAI 2026
  - 自动驾驶
  - 协同感知
  - 稀疏查询
  - 3D目标检测与跟踪
  - V2X通信
  - 车路协同
---

# SparseCoop: Cooperative Perception with Kinematic-Grounded Queries

**会议**: AAAI 2026  
**arXiv**: [2512.06838](https://arxiv.org/abs/2512.06838)  
**代码**: [github.com/wang-jh18-SVM/SparseCoop](https://github.com/wang-jh18-SVM/SparseCoop)  
**领域**: 自动驾驶  
**关键词**: 协同感知, 稀疏查询, 3D目标检测与跟踪, V2X通信, 车路协同

## 一句话总结

提出 SparseCoop——首个完全稀疏的协同感知框架，通过运动学锚定查询（KGQ）、粗到精聚合模块和协同实例去噪策略，完全抛弃密集 BEV 表示，在 V2X-Seq 和 Griffin 数据集上以最低通信开销和最高计算效率达到 SOTA 性能（AP 0.530，传输仅 3.17×10⁴ BPS）。

## 研究背景与动机

### 协同感知的必要性

单车系统固有地受限于传感器视场约束、远距离感知衰减和严重遮挡，这些挑战构成了自动驾驶安全部署的关键瓶颈。协同感知通过多智能体（V2V车-车、V2I车-路、V2D车-无人机）的信息交换创建集体感知系统。

### 现有方法的三大困境

#### 1. 密集 BEV 特征的根本缺陷
主流方法共享密集 BEV 特征图提供统一空间网格，但存在：
- **通信和计算成本随感知范围二次增长**
- **抽象的场景级特征难以在不同智能体间精确对齐**（尤其在时间异步和视角差异下）

#### 2. 稀疏查询方法的新问题
新兴的稀疏查询方法虽然更高效，但面临：
- **几何表示不足**：通常仅用单个参考点锚定查询，无法处理大视角旋转和时间偏移
- **融合策略欠优**：简单线性网络（表达力有限）或全局注意力（忽略自车传感器数据的细粒度上下文）
- **训练不稳定**：不同智能体视角和遮挡导致共同观测对象稀少，正样本不足

#### 3. 伪稀疏的局限
很多"稀疏"方法仍依赖密集 BEV 组件，继承了计算规模问题。

## 方法详解

### 整体框架

SparseCoop 的完整流程：

1. **稀疏实例提取**（每个智能体独立执行）：从多视图图像特征中生成运动学锚定查询（KGQ）
2. **运动学锚定关联**：利用 KGQ 的丰富状态向量进行跨智能体的精确时空对齐和匹配
3. **粗到精聚合**：先粗融合匹配对，再通过多上下文精炼整合所有实例
4. **协同实例去噪**（仅训练时）：从扰动的 GT 框初始化去噪查询，提供稳定的监督信号

### 关键设计

#### 1. 运动学锚定查询（Kinematic-Grounded Query, KGQ）

**功能**：为每个检测实例定义一个丰富的显式状态向量，替代简单的参考点表示。

**核心定义**：每个 KGQ 定义为 $\{\mathcal{F}, \mathcal{S}\}$，其中 $\mathcal{F}$ 是语义特征向量，$\mathcal{S}$ 是 11 维显式状态向量：

$$\mathcal{S} = (x, y, z, l, w, h, \sin(\theta), \cos(\theta), v_x, v_y, v_z)$$

包含 3D 位置、尺寸、朝向角和速度，远比单参考点包含更多几何和运动学信息。

**设计动机**：简单的参考点锚定无法应对协同感知中的大视角旋转和时间偏移。显式的状态向量不仅支持精确的时空对齐，还为匹配和融合提供了多维度的线索。

#### 2. 运动学锚定关联（Kinematic-Grounded Association, KGA）

**功能**：在不同智能体之间实现鲁棒的实例匹配。

**时空对齐**：
- **延迟补偿**：利用状态向量中的速度 $(v_x, v_y, v_z)$，通过恒速运动模型将协作实例的状态预测到自车当前时间戳
- **坐标投影**：通过变换矩阵 $\mathbf{T}_{co \to ego}$ 将实例投影到自车坐标系
- **特征更新**：使用旋转感知 MLP 更新特征向量

**几何-外观匹配（GAM）**：构建成对代价矩阵 $C$，包含两个互补组件：
- **几何相似性**：状态向量之间的加权 L1 距离
- **外观相似性**：特征向量之间的余弦距离

关联结果输出三组实例：（1）匹配对、（2）未匹配自车实例、（3）未匹配协作实例。

**交互范围设计**：定义一个较小的交互范围 $R_{int}$（V2X-Seq 最优 30m，Griffin 最优 15m），仅在此范围内进行融合。范围外的协作实例直接输出，避免自车低质量数据污染协作端的高质量检测。

#### 3. 粗到精聚合（Coarse-to-Fine Aggregation, CFA）

**功能**：有效融合匹配和未匹配实例的信息。

**粗融合**：对匹配对使用轻量级线性网络融合特征向量：

$$\mathcal{F}_{\text{fused}} = \text{MLP}([\mathcal{F}_{\text{ego}}; \widetilde{\mathcal{F}_{\text{co}}}])$$

**多上下文精炼**：融合后的 KGQ 和所有未匹配 KGQ 经过迭代精炼，每个精炼阶段包含：
- **时序交叉注意力**：链接当前帧与前一帧实例，理解运动并保持跟踪一致性
- **协作交叉注意力**：与完整的对齐协作 KGQ 集合交互，获取被遮挡区域的信息
- **自注意力**：捕捉当前帧所有实例之间的关系，推理场景布局并避免重复检测
- **可变形聚合**：从自车多尺度图像特征中采样，将抽象实例表示与原始视觉数据关联

**设计动机**：与仅用时序或协作上下文的方法不同，本文主张**自车图像特征对协同感知同样关键**——通过可变形聚合在原始视觉数据中精炼定位。

#### 4. 协同实例去噪（Cooperative Instance Denoising, CID）

**功能**：解决稀疏协同感知中正样本监督信号稀缺的训练不稳定问题。

**问题分析**：
- V2X-Seq 中大量 GT 对象仅对一个智能体可见（如约 58% 仅路侧可见）
- 即使同一对象对两个智能体可见，早期训练中的预测可能偏差太大而无法匹配

**噪声注入**：
- **观测噪声**：模拟传感器误差，在本地坐标系对 GT 属性添加均匀分布扰动（位置 ±2.0m，其他 ±0.5）
- **变换噪声（创新）**：模拟标定误差和时间异步，对变换矩阵添加随机旋转（σ=2°）和平移（σ=1m）

**去噪管线**：
- 去噪实例通过 tracking ID 直接匹配（提供大量稳定的匹配对）
- 使用定制注意力掩码严格隔离正常和去噪管线（防止信息泄漏）

### 损失函数 / 训练策略

- 基于 Sparse4D 框架的标准检测和跟踪损失
- 去噪管线与正常管线共享网络权重，但保持注意力隔离
- 训练时去噪实例提供额外的匹配对监督
- 高置信度 KGQ 被分配 tracking ID 并传播到后续帧（循环机制）

## 实验关键数据

### 主实验

#### V2X-Seq 和 Griffin-25m 数据集性能对比

| 方法 | V2X-Seq AP↑ | V2X-Seq AMOTA↑ | TC (BPS)↓ | Griffin AP↑ | Griffin AMOTA↑ | FPS↑ |
|------|------------|----------------|-----------|------------|----------------|------|
| No Fusion | 0.166 | 0.130 | 0 | 0.375 | 0.365 | 8.10 |
| Early Fusion | 0.243 | 0.209 | 8.19×10⁷ | 0.607 | 0.670 | 5.17 |
| V2X-ViT (ECCV22) | 0.268 | 0.287 | 2.56×10⁶ | 0.465 | 0.508 | 7.56 |
| CoopTrack (ICCV25) | 0.390 | 0.328 | 5.64×10⁴ | 0.479 | 0.488 | 6.23 |
| **SparseCoop** | **0.530** | **0.421** | **3.17×10⁴** | **0.559** | **0.509** | **11.64** |
| 提升（vs CoopTrack） | **+35.9%** | **+28.4%** | **-43.8%** | **+16.7%** | **+4.3%** | **+86.8%** |

SparseCoop 在所有方面全面领先：检测 AP 提升 35.9%，跟踪 AMOTA 提升 28.4%，同时传输成本降低 43.8%，推理速度提升 86.8%（11.64 vs 6.23 FPS）。

### 消融实验

#### 各模块贡献（V2X-Seq）

| 配置 | AP↑ | AMOTA↑ | 说明 |
|------|-----|--------|------|
| **完整模型** | **0.530** | **0.421** | - |
| 去除延迟补偿 (LC) | 0.505 | 0.414 | AP -4.7% |
| 去除几何外观匹配 (GAM) | 0.502 | 0.414 | AP -5.3% |
| 去除粗融合 (CFF) | 0.489 | 0.375 | **AMOTA -10.9%** |
| 去除多上下文精炼 (MCR) | 0.512 | 0.379 | AMOTA -10.0% |
| 去除观测噪声 (ON) | 0.521 | 0.416 | AP -1.7% |
| 去除变换噪声 (TN) | 0.531 | 0.394 | AMOTA -6.4% |
| **去除全部去噪** | **0.521** | **0.352** | **AMOTA -16.4%** |

粗融合和去噪对跟踪性能影响最大（AMOTA 分别下降 10.9% 和 16.4%），证明了融合质量和训练稳定性的关键作用。

### 关键发现

1. **完全稀疏可行且高效**：无需密集 BEV 即可达到 SOTA，传输成本仅 3.17×10⁴ BPS
2. **交互范围是关键超参数**：过大导致低质量数据污染，过小导致重复检测（V2X-Seq 最优 30m，Griffin 15m）
3. **通信延迟鲁棒性优异**：200ms 延迟下 SparseCoop 的 AP 超过所有方法（包括 0 延迟的 early fusion），归功于运动学补偿
4. **变换噪声比观测噪声更重要**：去除变换噪声导致 AMOTA 下降 6.4%，因为它直接模拟了跨智能体的标定和异步误差

## 亮点与洞察

- **彻底摆脱 BEV**：首个真正意义上的完全稀疏协同感知框架
- **KGQ 的优雅设计**：11 维状态向量同时服务于延迟补偿、坐标变换、匹配和融合定位
- **交互范围的深刻理解**：不是所有协作数据都应融合——远场直接输出、近场精心融合
- **去噪策略巧妙**：利用 GT 框的先验知识为训练提供稳定的匹配对，同时严格隔离防止信息泄漏

## 局限与展望

- 目前仅支持两个智能体（一对一），需要扩展到多智能体协同
- 恒速运动模型对非匀速运动的补偿精度有限
- 交互范围需要针对不同数据集手动调整
- 仅使用相机输入，可扩展至 LiDAR 和雷达模态的协同
- 去噪策略中的噪声参数需要与实际系统误差匹配

## 相关工作与启发

- **Sparse4D 系列**：稀疏实例提取的基础框架，SparseCoop 在其上构建协同模块
- **DN-DETR/MaskDINO**：去噪训练技术从检测领域迁移到协同感知
- **V2X-Seq/Griffin**：协同感知的标准评估基准
- **QUEST/CoopTrack**：稀疏查询协同方法的先驱，SparseCoop 解决了它们的关键缺陷

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （完全稀疏 + KGQ + 协同去噪，系统性创新）
- 实验充分度: ⭐⭐⭐⭐⭐ （两个数据集，全面消融，延迟鲁棒性分析）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，问题-方案对应明确）
- 价值: ⭐⭐⭐⭐⭐ （对协同感知范式有重要推动，兼顾性能、效率和鲁棒性）

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](../../ICCV2025/autonomous_driving/cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](../../NeurIPS2025/autonomous_driving/urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[AAAI 2026\] RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving](radarmp_motion_perception_for_4d_mmwave_radar_in_autonomous_driving.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)

<!-- RELATED:END -->
