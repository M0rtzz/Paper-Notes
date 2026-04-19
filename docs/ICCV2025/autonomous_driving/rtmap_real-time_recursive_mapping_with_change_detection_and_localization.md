---
title: >-
  [论文解读] RTMap: Real-Time Recursive Mapping with Change Detection and Localization
description: >-
  [ICCV 2025][自动驾驶][HD Map] 提出RTMap——首个端到端框架，同时解决多次遍历在线HD地图构建中的三大核心挑战：基于先验地图的定位、道路结构变化检测和概率感知众包地图融合，在TbV和nuScenes上同时提升地图质量和定位精度。
tags:
  - ICCV 2025
  - 自动驾驶
  - HD Map
  - 众包地图
  - 变化检测
  - 地图定位
  - 不确定性建模
  - 多次遍历融合
---

# RTMap: Real-Time Recursive Mapping with Change Detection and Localization

**会议**: ICCV 2025  
**arXiv**: [2507.00980](https://arxiv.org/abs/2507.00980)  
**代码**: [github.com/CN-ADLab/RTMap](https://github.com/CN-ADLab/RTMap)  
**领域**: Autonomous Driving / 在线高精地图构建  
**关键词**: HD Map, 众包地图, 变化检测, 地图定位, 不确定性建模, 多次遍历融合

## 一句话总结

提出RTMap——首个端到端框架，同时解决多次遍历在线HD地图构建中的三大核心挑战：基于先验地图的定位、道路结构变化检测和概率感知众包地图融合，在TbV和nuScenes上同时提升地图质量和定位精度。

## 研究背景与动机

在线HD地图构建已成为自动驾驶的主流范式，使车辆能在行驶中实时生成HD地图。但现有方法存在关键局限：

**单次遍历的局限性**：MapTR/MapTRv2等方法将建图视为单次过程，无法利用多次经过同一地点的丰富上下文信息。遮挡和感知不准确导致地图质量受限

**众包增强缺乏探索**：多Agent多次遍历的众包增强可以扩展感知范围、解决遮挡、提升地图精度，但需要两个关键能力：
   - **精确定位**：在先验地图中确定车辆准确位置以对齐当前观测
   - **变化检测**：检测道路结构变化（如车道修改、施工区域），保持地图新鲜度

**现有解决方案的碎片化**：定位和变化检测被独立研究，但它们本质上都在解决多次遍历地图元素间的检索、对应和差异问题，统一框架更合理

RTMap首次将先验辅助在线HD建图、地图定位和变化检测统一在一个端到端框架中，并引入不确定性建模来同时提升定位精度和众包融合质量。

## 方法详解

### 整体框架

RTMap包含在线（车载）和离线（云端）两个模块：

- **车载模型** $\mathrm{RTMapModel}(\mathbf{I}_t, \mathcal{M}_{t-1})$：输入当前帧多传感器图像和先验地图，输出地图元素 $\mathbf{M}_t$、不确定性 $\mathbf{U}_t$、对应关系 $\mathbf{D}_t$ 和端到端位姿 $\mathbf{T}_t^{\mathbb{E}}$
- **定位器** $\mathrm{Localize}(\cdot)$：利用推导出的对应关系进行显式优化求解位姿 $\mathbf{T}_t^{\mathbb{R}}$
- **云端众包** $\mathrm{CSrc}(\cdot)$：异步融合多次遍历观测更新全局先验地图

通过GPS提供米级初始位姿 $\mathbf{T}_t^0$，从全局先验地图中截取局部地图 $\mathcal{M}_{t-1}$ 作为车载操作的输入。

### 混合查询（Hybrid Queries）

为同时处理定位、建图和变化检测，设计三类查询：

$$\mathbf{Q}_{\mathrm{hybrid}} = \{\mathbf{Q}_{\mathrm{map}}, \mathbf{Q}_{\mathrm{fake}}, \mathbf{Q}_{\mathrm{new}}\} + \mathbf{Q}_{\mathrm{hie}}$$

- $\mathbf{Q}_{\mathrm{prior}}$：从先验地图编码而来，每个prior query用固定点数的统一表示编码（前2维为XOY坐标，其余N维为类别one-hot编码）
- $\mathbf{Q}_{\mathrm{new}}$：剩余query用零填充，负责检测当前遍历的新元素
- 训练后 $\mathbf{Q}_{\mathrm{prior}}$ 进一步分化为 $\mathbf{Q}_{\mathrm{map}}$（匹配的可靠元素，用于定位）和 $\mathbf{Q}_{\mathrm{fake}}$（过时的元素，标记为变化）
- $\mathbf{Q}_{\mathrm{hie}}$ 为MapTR中的层次查询嵌入

### 存在性感知匹配（Existence-aware Matching）

**训练阶段**：
- $\mathbf{Q}_{\mathrm{map}}$ 使用**预分配**到对应的现有地图元素
- $\mathbf{Q}_{\mathrm{new}}$ 使用标准**匈牙利匹配**与剩余地图元素配对
- $\mathbf{Q}_{\mathrm{fake}}$ 由于对应的地图元素不存在，不进行预分配
- 不同类型query在decoder层中展现出不同行为：map queries逐层向正确位置移动，fake queries参考点不稳定，new queries检测新元素

**推理阶段**：
- 无GT标注，$\mathbf{Q}_{\mathrm{map}}$ 和 $\mathbf{Q}_{\mathrm{fake}}$ 混在 $\mathbf{Q}_{\mathrm{prior}}$ 中
- 由于训练时fake queries未被预分配，其**分类置信度显著低于**map queries
- 因此使用置信度阈值即可有效区分两者

### 损失函数

**复合几何损失**（用于地图元素顶点）：

$$\mathbf{L}_{\mathrm{pts}} = \lambda_1 \cdot \mathbf{L}_{\mathrm{nll}} + \lambda_2 \cdot \mathbf{L}_{\mathrm{mht}}$$

其中NLL损失建模每个顶点的Laplace分布不确定性：

$$\mathbf{L}_{\mathrm{nll}} = \sum_{v=1}^{V}\sum_{k=1}^{2}\left(\log(2\sigma_v^k) + \frac{|\mathbf{m}_v^k - \mu_v^k|}{\sigma_v^k}\right)$$

- $\mu_v^k, \sigma_v^k$：第 $v$ 个顶点第 $k$ 维的位置和尺度参数
- $\mathbf{L}_{\mathrm{mht}}$：保留MapTRv2的曼哈顿距离损失
- 超参数：$\lambda_1=0.03$，$\lambda_2=5.0$

**位姿辅助损失**：利用decoder输出中map queries的特征，通过共享MLP+最大池化预测delta pose，用smooth L1 Loss监督。

### MAP定位与众包

**优化定位**：利用匹配对应 $\mathbf{D}_t$ 和不确定性 $\mathbf{U}_t$，对点到点残差进行最大后验（MAP）估计：

$$\min_{\mathbf{T}^{\mathbb{R}}} \sum_{\mathbf{D}_t} -\log\left(\exp\left(-\frac{1}{2}\|\mathbf{T}^{\mathbb{R}}\cdot\mathbf{m}_t^i - m_{t-1}^i\|_{\mathbf{g}_t^i}^2\right)\right)$$

其中协方差使用高斯混合 $\mathbf{g}_t^i = \mathbf{u}_t^i \oplus u_{t-1}^i$，使用Levenberg-Marquardt算法求解。

**概率感知众包**：云端使用union-find算法构建位置求解器：

$$\min_{m_t} \sum \frac{1}{2}\|m_t - \mathbf{T}_t^j \cdot \hat{\mathbf{m}}_t^j\|^2_{\mathbf{u}_t^j} + \frac{1}{2}\|m_t - m_{t-1}\|^2_{u_{t-1}}$$

通过匈牙利投票确定拓扑结构，高斯混合模型持续精化概率密度。

## 实验

### 数据集与设置

- **TbV**：200+场景含真实道路变化（车道拓扑、道路边界、人行横道），专为变化检测设计
- **nuScenes**：1000场驾驶场景，含标注HD地图，用于定位评估
- 引入位姿扰动的更真实评估：横向 $\mathcal{N}(0, 0.75^2)$ m，纵向 $\mathcal{N}(0, 1.5^2)$ m，航向 $\mathcal{N}(0, 0.85^2)$ °

### 主实验：众包地图质量（TbV）

| 方法 | 场景 | 遍历次数 | Cycle(%) | Ped.(%) | Div.(%) | Avg mAP(%) |
|------|------|----------|----------|---------|---------|-------------|
| MapTRv2 | Straight | Ave. | 31.7 | 42.0 | 37.3 | 37.1 |
| HRMapNet | Straight | Ave. | 34.2 | 43.7 | 39.8 | 39.2 |
| MapTracker | Straight | Ave. | 35.7 | 44.6 | 39.6 | 39.9 |
| RTMap(w/o U) | Straight | 2 | 28.6 | 60.5 | 31.7 | 40.2 |
| **RTMap** | **Straight** | **2** | **32.7** | **68.6** | **35.5** | **45.6** |
| RTMap(w/o U) | Straight | 3 | 35.7 | 74.4 | 42.0 | 50.7 |
| **RTMap** | **Straight** | **3** | **40.9** | **84.3** | **47.6** | **57.6** |
| MapTRv2 | Turning | Ave. | 28.2 | 31.6 | 18.3 | 26.0 |
| **RTMap** | **Turning** | **3** | **42.3** | **85.2** | **38.8** | **55.4** |

**关键发现**：
- 引入位姿噪声后现有方法精度大幅下降，RTMap通过众包逐步提升
- 3次遍历后RTMap在直行场景mAP达57.6%，远超MapTracker的39.9%
- 概率密度 $\mathbf{U}$ 对众包至关重要，带来约7% mAP提升

### 消融实验：定位精度

| 方法 | 横向Mean(m) | 横向90th(m) | 纵向Mean(m) | 航向Mean(°) |
|------|-------------|-------------|-------------|-------------|
| RTMap ($\mathbf{Q}_{\mathrm{prior}}$) | 0.163 | 0.318 | 0.686 | 0.332 |
| **RTMap ($\mathbf{Q}_{\mathrm{map}}$)** | **0.125** | **0.256** | **0.633** | **0.317** |

| 方法(nuScenes) | 横向Mean(m) | 纵向Mean(m) | 航向Mean(°) |
|------|-------------|-------------|-------------|
| RTMap ($\mathbf{T}^{\mathbb{E}}$) | 0.142 | 0.589 | 0.521 |
| **RTMap ($\mathbf{T}^{\mathbb{R}}$)** | **0.121** | **0.586** | **0.368** |
| $\mathbf{T}^{\mathbb{R}}$+$\mathbf{L}_{\mathrm{pose}}$ | 0.122 | 0.590 | 0.371 |
| $\mathbf{T}^{\mathbb{R}}$+$\mathbf{L}_{\mathrm{pts}}$ | **0.118** | 0.609 | 0.376 |

**关键发现**：
- 混合查询区分 $\mathbf{Q}_{\mathrm{map}}$ 和 $\mathbf{Q}_{\mathrm{fake}}$ 有效提升定位（排除过时元素的干扰）
- 显式优化 $\mathbf{T}^{\mathbb{R}}$ 优于端到端回归 $\mathbf{T}^{\mathbb{E}}$，尤其在航向角上
- NLL损失 $\mathbf{L}_{\mathrm{pts}}$ 和位姿辅助损失 $\mathbf{L}_{\mathrm{pose}}$ 各自贡献互补

### 变化检测（TbV）

| 方法 | Acc_c(%) | Acc_r(%) | mAcc(%) |
|------|----------|----------|---------|
| TbV baseline | 40.0 | 68.2 | 54.1 |
| **RTMap** | **48.9** | 66.0 | **57.4** |

RTMap对变化类别的召回率更高，实用中更高召回能确保安全。

## 亮点与洞察

1. **首个统一框架**：将定位、变化检测、建图三大任务融合在端到端模型中，互相促进而非独立优化
2. **混合查询设计精妙**：利用query分化自然区分已有/过时/新增地图元素，推理时仅靠置信度即可区分
3. **不确定性的多重价值**：概率密度同时服务于定位（Mahalanobis距离加权）和众包融合（噪声感知合并）
4. **自进化记忆**：先验地图作为"记忆"随遍历次数增加持续自我改善，3次遍历质量即大幅超越单次方法

## 局限性

1. 目前仅使用相机，未融合LiDAR等多模态传感器
2. 验证仅限结构化道路，尚未扩展到非结构化路面和更复杂城市场景
3. 众包机制通过MAP投票逐步过滤临时结果，尚未区分永久结构变化和暂时遮挡
4. 训练使用人工扰动的先验地图（公开数据集缺乏多次遍历数据），与真实众包场景有差距

## 相关工作

- **在线HD建图**：HDMapNet、MapTR/v2、MapTracker等单次遍历方法；PrevPredMap、HRMapNet等利用先验的方法
- **先验辅助方法**：P-MapNet（SD先验+HD先验）、PriorDrive（统一向量编码器）、SMERF、U-BEV
- **变化检测**：ExelMap（元素级插入/删除检测）
- **地图定位**：BEV-Locator、EgoVM（厘米级定位），但将定位作为独立任务

## 评分

- **新颖性**：⭐⭐⭐⭐⭐（首个统一定位+变化检测+众包建图的端到端框架）
- **技术深度**：⭐⭐⭐⭐⭐（混合查询、存在性匹配、概率建模、MAP优化多个精巧设计）
- **实验完整性**：⭐⭐⭐⭐（多数据集多任务评估完整，但缺乏真实多次遍历大规模验证）
- **实用价值**：⭐⭐⭐⭐⭐（直接面向产业级自动驾驶众包地图场景）
- **总体推荐**：⭐⭐⭐⭐⭐（系统性极强的工作，有望推动在线HD建图进入众包时代）

<!-- RELATED:START -->

## 相关论文

- [GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network](../../ICML2025/autonomous_driving/when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy.md)
- [ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)
- [Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [LookOut: Real-World Humanoid Egocentric Navigation](lookout_real-world_humanoid_egocentric_navigation.md)

<!-- RELATED:END -->
