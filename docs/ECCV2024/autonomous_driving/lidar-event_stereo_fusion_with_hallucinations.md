---
title: >-
  [论文解读] LiDAR-Event Stereo Fusion with Hallucinations
description: >-
  [ECCV 2024][自动驾驶][事件相机] 提出将LiDAR稀疏深度点与事件立体相机融合的首个框架，通过在事件堆叠表示（VSH）或原始事件流（BTH）中"幻觉"（插入虚构事件）来弥补事件相机在无运动/无纹理区域的信息缺失，大幅提升事件立体匹配精度。
tags:
  - ECCV 2024
  - 自动驾驶
  - 事件相机
  - LiDAR融合
  - 立体匹配
  - 深度估计
  - 事件幻觉
---

# LiDAR-Event Stereo Fusion with Hallucinations

**会议**: ECCV 2024  
**arXiv**: [2408.04633](https://arxiv.org/abs/2408.04633)  
**代码**: [有](https://eventvppstereo.github.io/)  
**领域**: 自动驾驶 / 深度估计  
**关键词**: 事件相机, LiDAR融合, 立体匹配, 深度估计, 事件幻觉

## 一句话总结

提出将LiDAR稀疏深度点与事件立体相机融合的首个框架，通过在事件堆叠表示（VSH）或原始事件流（BTH）中"幻觉"（插入虚构事件）来弥补事件相机在无运动/无纹理区域的信息缺失，大幅提升事件立体匹配精度。

## 研究背景与动机

### 领域现状
事件相机（neuromorphic cameras）通过异步报告像素亮度变化，具有微秒级时间分辨率和极高动态范围，非常适合快速运动和极端光照下的深度估计。事件立体匹配将事件流编码为结构化表示（如直方图、体素网格、MDES等），再用深度网络估计视差图。

### 核心痛点
事件相机仅在亮度变化时触发，因此在以下场景会产生**灾难性失败**：

**无运动场景**：相机或物体静止时完全没有事件

**大面积无纹理区域**：如天空、墙面、路面等亮度均匀区域不触发事件
3. 半稠密的事件数据使得立体匹配中的对应点搜索极其困难

### 现有方案与局限
- RGB立体匹配中，LiDAR融合方法（拼接输入、调制代价体积、Virtual Pattern Projection）已被广泛研究
- 但 **事件立体 + LiDAR融合** 领域完全空白
- 直接套用RGB融合方法存在问题：LiDAR固定帧率（通常10Hz）与事件相机的异步采集天然矛盾——要么仅在LiDAR可用时使用深度（大多时间浪费），要么降低处理速率到LiDAR频率（丧失事件相机微秒级分辨率优势）

### 核心互补性洞察
事件相机与LiDAR天然互补：
- **事件相机**：在物体边界（亮度变化剧烈处）提供丰富信息，但LiDAR在此处稀疏
- **LiDAR**：在无纹理、无运动区域可靠测距，但事件相机在此处无信息

### 核心 idea
受RGB领域Virtual Pattern Projection (VPP)启发，设计一种 **"幻觉"机制**：利用LiDAR深度点在事件数据中插入虚构的匹配线索。在已知某像素深度（即视差）的前提下，在左右视图的对应位置注入相同的distinctive patterns，让立体网络更容易找到正确匹配。

## 方法详解

### 整体框架

根据对事件立体网络的访问级别，定义三种框架：
- **白盒（White box）**：可访问网络和堆叠表示的实现
- **灰盒（Gray box）**：可访问堆叠表示但不可访问网络内部
- **黑盒（Black box）**：堆叠表示和网络均不可访问

提出两种幻觉策略：VSH（适用于灰盒）和BTH（适用于黑盒），均不需修改立体网络本身。

### 关键设计

#### 1. 虚拟堆叠幻觉（Virtual Stack Hallucination, VSH）

**功能**：在已构建的事件堆叠表示上直接注入虚拟模式，增强匹配区分度。

**核心思路**：给定左右事件堆叠 $\mathcal{S}_L, \mathcal{S}_R$（尺寸 $W \times H \times C$）和LiDAR深度测量集合 $Z$，对每个深度点 $z(x,y)$：

1. 将深度转换为视差：$d(x,y) = \frac{bf}{z(x,y)}$
2. 计算右图对应位置：$x' = x - d(x,y)$
3. 在左右堆叠的对应位置注入相同的虚拟模式：

$$\mathcal{S}_L(x,y,c) \leftarrow \mathcal{A}(x,y,x',c), \quad \mathcal{S}_R(x',y,c) \leftarrow \mathcal{A}(x,y,x',c)$$

虚拟模式 $\mathcal{A}$ 从均匀分布中随机采样：

$$\mathcal{A}(x,y,x',c) \sim \mathcal{U}(\mathcal{S}^-, \mathcal{S}^+)$$

其中 $\mathcal{S}^-, \mathcal{S}^+$ 为堆叠中的最小/最大值。可选择单像素或局部窗口（3×3效果最佳），并支持alpha blending。

**设计动机**：事件堆叠在无事件区域完全空白（semi-dense），注入matching-consistent的随机模式能极大增强局部区分度。在左右对应位置注入相同模式直接提供了正确视差的匹配线索。对事件堆叠的效果比RGB图像更显著，因为作用在更稀疏的数据上。

#### 2. 回溯时间幻觉（Back-in-Time Hallucination, BTH）

**功能**：直接在原始事件流中插入虚构事件，无需访问堆叠表示。

**核心思路**：在从 $t_d$ 时刻向前采样的事件历史 $\mathcal{E}_L, \mathcal{E}_R$ 中，对每个深度点 $d(\hat{x},\hat{y})$，注入一对虚构事件：

$$\hat{e}^L = (\hat{x}, \hat{y}, \hat{p}, \hat{t}), \quad \hat{e}^R = (\hat{x}', \hat{y}, \hat{p}, \hat{t})$$

满足三个约束：
- **时间有序性**：$\hat{t}$ 在事件历史时间范围内
- **几何约束**：$\hat{x}' = \hat{x} - d(\hat{x},\hat{y})$
- **一致性约束**：左右虚构事件的极性 $\hat{p}$ 和时间戳 $\hat{t}$ 相同

**单时间戳注入**：在固定时间戳 $t_z$ 注入 $K_{\hat{x},\hat{y}}$ 对随机极性的事件。关键优势：即使 $t_z < t_d$（LiDAR数据过时），只要在事件历史时间范围内，仍可有效利用。

**重复注入（Repeated Injection）**：更高级策略——将事件历史分为 $B$ 个时间bin，在每个bin中独立进行注入。每个深度点仅在随机分配的一个bin中注入，增强时间维度的区分度。使用 $B=12$ 个注入点、每点注入2个虚构事件。

**设计动机**：BTH无需访问堆叠表示（黑盒兼容），且能利用事件数据的时间维度优势。重复注入特别增强了对LiDAR数据时间偏移（misaligned，$t_z < t_d$）的鲁棒性。

#### 3. 框架级别的适配

- 8种堆叠表示全面支持：Histogram, Voxel Grid, MDES, Concentration, TORE, Time Surface, ERGO-12, Tencode
- 支持预训练模型直接应用（无需重训练）和从头训练两种模式
- 遮挡处理、均匀/非均匀patch等细节继承自VPP

### 训练策略

- 骨干网络：基于SE-CFF的AANet变体
- 训练 25 epochs，batch size 4，最大视差192
- Adam优化器，lr=$5 \times 10^{-4}$，cosine衰减
- 随机裁剪和垂直翻转数据增强
- VSH额外引入2-15ms CPU开销，BTH引入10ms

## 实验关键数据

### 主实验

**DSEC数据集 - 预训练模型（8种表示平均排名）**：

| 融合方法 | 1PE↓ Avg Rank | 2PE↓ Avg Rank | MAE↓ Avg Rank | 说明 |
|---------|--------------|--------------|--------------|------|
| Baseline (无融合) | 3.00 | 3.00 | 3.00 | 纯事件立体 |
| Guided [Poggi] | - | - | - | 代价体积调制，改善有限 |
| **VSH (Ours)** | **1.75** | **1.38** | **1.50** | 灰盒策略 |
| **BTH (Ours)** | **1.25** | **1.63** | **1.13** | 黑盒策略，最优 |

**DSEC数据集 - 重训练模型（8种表示平均排名）**：

| 融合方法 | 1PE↓ Avg Rank | 2PE↓ Avg Rank | MAE↓ Avg Rank |
|---------|--------------|--------------|--------------|
| Concat [LidarStereoNet] | 3.38 | 3.00 | 3.13 |
| Guided+Concat [CCVNorm] | 3.63 | 3.50 | 3.38 |
| Guided [Poggi] | 5.00 | 5.00 | 5.00 |
| **VSH (Ours)** | **1.38** | **1.88** | **1.13** |
| **BTH (Ours)** | 1.63 | 1.38 | 1.88 |

重训练时VSH表现最优，1PE常降至10%以下（如ERGO-12: 9.25%）。

**M3ED数据集 - 跨域泛化（预训练）**：

| 表示 | Baseline 1PE | VSH 1PE | BTH 1PE | 相对改善 |
|------|-------------|---------|---------|---------|
| Histogram | 37.70 | 20.19 | 22.32 | ~46% |
| ERGO-12 | 36.33 | 22.53 | 20.41 | ~44% |
| Tencode | 43.56 | 28.24 | 22.61 | ~48% |

在M3ED上改善更为惊人，1PE从30-40%+降至20%左右。

### 消融实验

**DSEC搜索集上的超参数消融（1PE，8种表示平均）**：

| 配置 | 效果 | 说明 |
|------|------|------|
| VSH: 单像素 vs 3×3 patch vs 5×5 | 3×3最优 | 适当patch增强区分度 |
| VSH: 随机模式 vs 均匀模式 | 均匀更好 | 统一模式更有效 |
| VSH: alpha=0 vs 0.5 vs 1.0 | 0.5最优 | 原始内容与模式的平衡 |
| BTH: 单次注入 vs 重复注入 | 重复注入更优 | 利用时间维度 |
| BTH: 1 vs 2 vs 4 虚构事件 | 2个即饱和 | 少量事件即足够 |
| BTH: 随机极性 vs 均匀极性 | 均匀更好 | 增强一致性 |

### 关键发现

1. **Guided方法在事件立体上效果有限**：16线LiDAR过于稀疏，且代价体积调制在无事件区域帮助不大
2. **VSH和BTH显著优于所有现有RGB融合方法的迁移**：1PE降低2-3%（预训练）或更多（重训练）
3. **BTH在预训练场景最优，VSH在重训练场景最优**：BTH更灵活（黑盒），VSH更直接（可训练优化）
4. **过时LiDAR数据仍可有效利用**：BTH的重复注入策略使得$t_z < t_d$时仅有微小精度下降，保持了事件相机微秒级分辨率的优势
5. **方法对所有8种事件表示通用**：非特定某种表示设计

## 亮点与洞察

1. **问题定义的开创性**：首次探索事件立体+LiDAR融合，识别到两种传感器的天然互补性
2. **"幻觉"的优雅设计**：不修改网络、不改变表示格式，仅通过数据层面的注入实现大幅改善
3. **黑盒兼容性**：BTH甚至不需要访问堆叠表示，具有极强通用性
4. **异步传感器的优雅处理**：利用事件历史的时间范围，将过时LiDAR数据也能无缝整合

## 局限与展望

1. VSH需要访问堆叠表示（灰盒限制），BTH的遮挡处理不如VSH完善
2. 实验中LiDAR稀疏点的对齐依赖额外的里程计和ICP配准
3. 虚构事件的模式为简单随机/均匀，可探索学习更优的注入模式
4. 未考虑事件相机和LiDAR的在线标定误差
5. 可扩展到单目事件深度估计 + LiDAR融合

## 相关工作与启发

- **VPP (Virtual Pattern Projection)** [Bartolomei, CVPR 2024]：RGB立体中投射虚拟模式的先驱，本文将思想迁移到事件域
- **SE-CFF** [Nam et al., CVPR 2024]：事件立体匹配SOTA框架，本文基于其实现
- **DSEC** [Gehrig et al., RA-L 2021]：大规模室外事件立体数据集
- 启发：传感器融合的关键不是简单拼接数据，而是找到每种传感器的失败模式并用另一种传感器弥补

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 开辟事件立体+LiDAR融合新方向，幻觉机制设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ — 2个数据集、8种表示、多种融合对比、预训练/重训练双模式、超参消融全面
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰、方法描述细致、实验组织条理分明
- **价值**: ⭐⭐⭐⭐ — 高实用性（自动驾驶主流传感器组合），通用性强（8种表示均适用）

<!-- RELATED:START -->

## 相关论文

- [Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)
- [MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [DVLO: Deep Visual-LiDAR Odometry with Local-to-Global Feature Fusion and Bi-Directional Structure Alignment](dvlo_deep_visuallidar_odometry_with_localtoglobal_featu.md)
- [Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)

<!-- RELATED:END -->
