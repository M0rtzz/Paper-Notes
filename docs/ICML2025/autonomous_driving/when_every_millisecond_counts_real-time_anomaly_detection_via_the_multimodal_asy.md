---
title: >-
  [论文解读] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network
description: >-
  [ICML 2025][自动驾驶][实时异常检测] 提出多模态异步混合网络，结合事件相机的高时间分辨率（异步 GNN 处理）和 RGB 相机的丰富空间特征（CNN 处理），在交通异常检测中实现 579 FPS 的推理速度和 1.17s 的平均响应时间，首次将事件流引入自动驾驶异常检测领域。
tags:
  - ICML 2025
  - 自动驾驶
  - 实时异常检测
  - 事件相机
  - 多模态融合
  - 异步图神经网络
  - 自动驾驶安全
---

# When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network

**会议**: ICML 2025  
**arXiv**: [2506.17457](https://arxiv.org/abs/2506.17457)  
**代码**: [有](https://github.com/PKU-XD/EventAD)  
**领域**: 自动驾驶  
**关键词**: 实时异常检测, 事件相机, 多模态融合, 异步图神经网络, 自动驾驶安全

## 一句话总结

提出多模态异步混合网络，结合事件相机的高时间分辨率（异步 GNN 处理）和 RGB 相机的丰富空间特征（CNN 处理），在交通异常检测中实现 579 FPS 的推理速度和 1.17s 的平均响应时间，首次将事件流引入自动驾驶异常检测领域。

## 研究背景与动机

自动驾驶异常检测（如突然出现的行人、车辆异常行为）对于驾驶安全至关重要。现有方法主要面临以下矛盾：

**精度 vs 速度的矛盾**：SOTA 方法（如 TTHF）使用复杂深度网络追求更高检测精度，但推理延迟高（仅 3 FPS），在实际驾驶中数百毫秒的延迟可能决定安全制动与碰撞的差异

**帧级检测的时间盲区**：传统 RGB 相机以固定帧率采集图像（如 30 FPS），帧间约 33ms 的"时间盲区"中高速目标可能已移动显著距离

**已有方法忽视响应时间指标**：多数方法仅关注 AUC、AP 等检测精度指标，未将响应时间纳入评估体系

**核心矛盾**：如何在保持高检测精度的同时实现毫秒级响应时间？

**本文切入角度**：引入事件相机（Event Camera）这一新型传感器。事件相机以微秒级时间分辨率异步检测亮度变化，产生稀疏事件流，天然适合捕捉快速动态变化。将事件流与 RGB 图像互补融合，用异步 GNN 处理事件流以保持低延迟，同时用 CNN 提取 RGB 的空间特征。

**核心 idea**：设计多模态异步混合网络，异步处理事件流（不等待帧同步），实现帧间异常检测能力，将推理速度提升到 ~600 FPS。

## 方法详解

### 整体框架

网络包含两个并行分支和一个异常检测头：
- **RGB 分支**：ResNet50 提取空间外观特征 + YOLOX 目标检测
- **事件分支**：异步 GNN（DAGr）处理事件流提取时序动态特征
- **融合 + 检测**：单向特征融合（CNN→GNN）+ GRU 时序建模 + 注意力机制 + 风险评分预测

### 关键设计

1. **异步事件图构建与处理**: 事件相机输出事件流 $E = \{e_i = (x_i, t_i, p_i)\}$，其中 $x_i$ 为像素坐标、$t_i$ 为时间戳、$p_i \in \{-1, 1\}$ 为极性。亮度变化超过阈值 $C$ 时触发：$|\Delta L| > C$。将事件建模为图节点，归一化时空坐标 $\hat{x}_i = (u_i/W, y_i/H)$，$\hat{t}_i = \beta t_i$，基于时空邻近性建边（半径 $R$ 内，每节点最多 16 个邻居）。使用 DAGr（Deep Asynchronous GNN）的残差图卷积层和样条卷积处理：

   $$f_i' = W_c f_i + \sum_{j \in \mathcal{N}(i)} W(e_{ij}) f_j$$

   样条卷积通过查找表加速推理，时间复杂度远低于标准注意力机制。设计动机：保留事件流的异步稀疏特性，避免将事件聚合为帧表示带来的信息损失和延迟。

2. **单向多模态融合（CNN→GNN）**: 将 CNN 中间层特征图 $G_I = \{g_I^l\}_{l=1}^L$ 通过空间采样增强 GNN 节点特征：

   $$f_i' = [f_i, g_I(\hat{x}_i)]$$

   即在每个事件节点的空间位置采样对应 CNN 特征进行拼接。关键设计是**单向共享**——仅 CNN 特征流向 GNN，GNN 不反馈到 CNN。设计动机：在事件稀疏场景（如静止或慢速运动时几乎无事件产生）中，RGB 特征可弥补事件信息不足；同时避免双向通信增加计算延迟。

3. **时空关系学习与注意力异常检测**: 对每个检测目标 $i$，提取事件特征 $o_{t,i} = \text{AsyncGNN}(E_{t,i}; \theta_{\text{GNN}})$ 和 CNN 特征 $g_{t,i}$，拼接后降维得到融合特征 $f_{t,i}$。分别用两个 GRU 建模时序依赖：

   $$h_{b,t,i} = \text{GRU}(b_{t,i}, h_{b,t-1,i}; \theta_1)$$
   $$h_{f,t,i} = \text{GRU}(f_{t,i}, h_{f,t-1,i}; \theta_2)$$

   其中 $b_{t,i}$ 为边界框特征，$f_{t,i}$ 为融合特征。注意力机制动态分配目标权重：$\alpha_{b,t} = \text{softmax}(\tanh(H_{b,t}^\top w_b))$，使模型聚焦于潜在异常目标。最终风险评分：$s_{t,i} = \text{softmax}(\phi(\hat{h}_{t,i}; \theta_3))$。

### 损失函数 / 训练策略

- **目标检测损失**：IoU loss + class loss + regression loss（YOLOX 框架）
- **异常检测损失**：加权交叉熵，ROL 数据集中负类权重 0.27、正类权重 1.0（处理类别不平衡）
- **优化器**：GRU-注意力模块使用 Adam（lr=0.001），GNN-ResNet 使用 AdamW（lr=2×10⁻⁴）
- **学习率调度**：ReduceLROnPlateau
- **训练规模**：RGB 组件 30 epochs（batch 64），事件组件 150,000 iterations（batch 32，约 2500 遍数据）
- **事件数据生成**：使用 v2e 工具将 RGB 视频转换为模拟事件流（现有真实数据集缺少事件模态）

## 实验关键数据

### 主实验

在 ROL 和 DoTA 两个交通异常检测基准上评测。

| 方法 | AUC (ROL) | AUC (DoTA) | mTTA (ROL) | FPS | mResponse (ROL) |
|--------|------|------|------|------|------|
| FOL-Ensemble | 0.849 | 0.866 | 2.05s | 33 | 2.16s |
| MAMTCF | 0.841 | 0.862 | 2.01s | 98 | 1.88s |
| AM-Net | 0.855 | 0.874 | 2.18s | 61 | 1.96s |
| STFE | 0.862 | 0.881 | 2.23s | 77 | 2.04s |
| TTHF | 0.871 | 0.891 | 2.35s | **3** | 2.46s |
| **Ours** | **0.879** | **0.896** | **2.80s** | **579** | **1.17s** |

核心优势：
- AUC 最优（ROL 0.879、DoTA 0.896），同时 FPS 是 TTHF 的 **193 倍**
- mResponse 仅 1.17s（ROL），比 TTHF 的 2.46s 快 **一倍以上**
- mTTA 达到 2.80s，比所有方法提前更多时间预警

### 消融实验

| 配置 | AUC | AP | mTTA | mAP | 说明 |
|------|-----|-----|------|-----|------|
| RGB+Event 仅 | 0.805 | 0.479 | 1.44 | 41.66 | 基础多模态 |
| +GRU | 0.817 | 0.508 | 1.98 | 43.59 | 时序建模关键 |
| +Attention | 0.823 | 0.518 | 2.06 | 35.76 | 聚焦异常目标 |
| +BBox | 0.839 | 0.531 | 2.11 | 43.29 | 空间定位提升 |
| +Object | 0.845 | 0.539 | 1.96 | 42.94 | 目标级特征 |
| +Two-stage | 0.868 | 0.561 | 2.51 | 43.82 | 两阶段优化 |
| Full Model | **0.879** | **0.570** | **2.80** | **45.15** | 全部组件 |

网络深度与骨干消融：

| 配置 | AUC | FPS | mResponse | 说明 |
|------|-----|-----|-----------|------|
| 4 层（默认） | 0.879 | 579 | 1.17s | 速度最优 |
| 5 层 | 0.885 | 312 | 1.31s | 精度微升、速度减半 |
| 6 层 | 0.892 | 166 | 1.56s | 精度-速度权衡 |
| CNN→Swin | 0.881 | 278 | 1.44s | Transformer 精度微升 |
| CNN→ViT-B | 0.886 | 213 | 1.51s | 全局建模更强 |

### 关键发现

1. **GRU 是最关键的组件**：引入 GRU 后 AUC 从 0.805 提升到 0.817，mTTA 从 1.44s 提升到 1.98s，说明时序积累对异常预判至关重要
2. **事件模态的独特价值**：事件流提供了帧间异常检测能力，在高速目标突然出现的场景中可以在 RGB 帧到达前就检测到异常
3. **速度优势压倒性**：579 FPS 意味着每次推理仅约 1.7ms，远低于 RGB 帧间隔（33ms），使得实时响应成为可能
4. **网络深度可扩展**：增加层数可微幅提升精度（+1.3% AUC），但延迟线性增长，提供了精度-速度权衡的灵活性

## 亮点与洞察

- **首次将事件相机引入交通异常检测**：开创性地利用事件流的高时间分辨率和异步特性解决响应时间问题
- **帧间检测能力**：传统方法只能在帧到达时检测，本文利用事件流在两帧之间就能发现高速异常目标（图 5 展示了行人突然出现的帧间检测）
- **单向融合设计精妙**：CNN→GNN 的单向特征共享既解决了事件稀疏时的信息不足，又避免了反向通信带来的延迟
- **mResponse 指标贡献**：提出跨多阈值的平均检测延迟指标 mResponse，比单阈值评估更全面地衡量实时性能
- **实用导向**：明确将响应时间纳入评价体系，强调"检测到但来不及反应"等于没检测到

## 局限与展望

1. **事件数据为模拟生成**：使用 v2e 将 RGB 视频转为事件流，与真实事件相机存在域差距。未来可用 DSEC 等真实事件驾驶数据集验证
2. **目标检测依赖 RGB 帧**：YOLOX 检测框仍需等待 RGB 帧，事件流仅用于特征增强，未完全发挥事件相机的低延迟优势。可探索纯事件检测降低整体延迟
3. **场景覆盖有限**：ROL 和 DoTA 都是前视 dashcam 场景，未测试多视角、恶劣天气等条件
4. **注意力机制简单**：使用基础 tanh+softmax 注意力，可考虑交叉注意力或 Transformer 进一步提升异常聚焦能力（但需权衡速度）
5. **训练策略复杂**：两种模态分别训练，参数调整复杂，端到端联合训练可能更优

## 相关工作与启发

- **vs TTHF**：TTHF 通过文本信息融合获得更高 AUC-Frame（0.847 vs 0.736），但仅 3 FPS，实际不可部署。本文在 AUC 上超越 TTHF 且速度快 193 倍
- **vs MOVAD**：MOVAD 首次提出在线交通异常检测概念，本文进一步推进到毫秒级实时检测
- **vs DAGr**：借鉴 DAGr 的异步 GNN 架构处理事件流，将其从目标检测扩展到异常检测
- **启发**：事件相机 + 异步处理的范式可推广到其他安全关键感知任务（如行人意图预测、碰撞预警），尤其适合时间敏感场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将事件相机引入交通异常检测，帧间检测概念新颖
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、全面消融、骨干对比，但事件数据为模拟生成
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰（响应时间公式化），架构图直观，评估指标完善
- 价值: ⭐⭐⭐⭐⭐ 将响应时间提升到全新量级，高度实用导向，对自动驾驶安全有直接意义

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] RTMap: Real-Time Recursive Mapping with Change Detection and Localization](../../ICCV2025/autonomous_driving/rtmap_real-time_recursive_mapping_with_change_detection_and_localization.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](../../CVPR2025/autonomous_driving/reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](../../NeurIPS2025/autonomous_driving/spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)

<!-- RELATED:END -->
