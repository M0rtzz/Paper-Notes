---
title: >-
  [论文解读] Stream Query Denoising for Vectorized HD-Map Construction
description: >-
  [ECCV 2024][自动驾驶][HD地图构建] 提出 Stream Query Denoising (SQD) 策略，通过对前一帧 GT 添加噪声并训练网络恢复当前帧 GT 来增强流式 HD 地图构建中的时序一致性建模，在 nuScenes 和 Argoverse2 上全面超越 StreamMapNet。
tags:
  - ECCV 2024
  - 自动驾驶
  - HD地图构建
  - 时序建模
  - 查询去噪
  - 流式推理
  - 矢量化地图
---

# Stream Query Denoising for Vectorized HD-Map Construction

**会议**: ECCV 2024  
**arXiv**: [2401.09112](https://arxiv.org/abs/2401.09112)  
**代码**: 论文中提及 code will be available soon  
**领域**: 自动驾驶  
**关键词**: HD地图构建, 时序建模, 查询去噪, 流式推理, 矢量化地图

## 一句话总结

提出 Stream Query Denoising (SQD) 策略，通过对前一帧 GT 添加噪声并训练网络恢复当前帧 GT 来增强流式 HD 地图构建中的时序一致性建模，在 nuScenes 和 Argoverse2 上全面超越 StreamMapNet。

## 研究背景与动机

矢量化 HD 地图构建正在从单帧检测向时序流式推理演进，StreamMapNet 是其代表性工作。然而，直接将流式 query 传播范式从目标检测迁移到地图构建面临独特挑战：

**曲线不同于框**：目标检测中物体在帧间主要做刚性运动（平移+旋转），box 的时序预测相对简单。但道路线在帧间会部分增长、部分截断，线上每个点需要学习不同的偏移量，这种复杂变化难以显式建模。

**流式训练困难**：实验表明直接加入 streaming 策略反而使性能下降 0.7 mAP——网络难以学会应对曲线在不同帧间的持续变化。

**Query Denoising 不能直接用于曲线**：DN-DETR 针对 bounding box 设计，曲线的噪声策略尚未被探索。

核心观察：如果在前一帧 GT 上加噪声来模拟 stream query 的预测行为，让网络学会从噪声中恢复当前帧 GT，就能有效学习时序一致性。

## 方法详解

### 整体框架

SQD-MapNet = StreamMapNet + SQD 策略。整体流程：

1. 多视角图像 → backbone + FPN → BEVFormer 得到 BEV 特征
2. Polyline Decoder 使用可学习 query 提取地图元素
3. 时序 query 传播：top-k 高分 query 经坐标变换后传递到下一帧
4. **SQD（仅训练时）**：对前一帧 GT 加噪生成去噪 query，与正常 query 一同送入 decoder，用当前帧 GT 监督去噪输出

SQD 在推理时移除，不增加推理开销。

### 关键设计

1. **Normal Query Denoising — 曲线噪声策略**：首次将曲线统一为最小外接矩形表示 $(x, y, w, h)$，在此基础上设计三种噪声：

    - **Box Shifting（线平移）**：对矩形中心加偏移 $(\Delta x, \Delta y)$，约束 $|\Delta x| < \frac{\lambda_1 w}{2}$，曲线内各点相对矩形位置不变
    - **Box Scaling（角度旋转+尺度变换）**：对矩形宽高进行缩放，$h' \in [(1-\lambda_3)h, (1+\lambda_3)h]$，同时实现旋转和长度变化
    - 噪声 query 生成：类别 → 可学习 embedding $C_q$；点坐标 → 位置编码 → MLP 融合为位置 embedding $Pos_q$；最终 $Q_{denoise} = \text{MLP}^{(fuse)}(\text{Concat}(C_q, Pos_q))$

2. **Adaptive Temporal Matching (ATM)**：建立前一帧 GT 与当前帧 GT 的一对一对应关系。方法是用 ego-motion 矩阵将前帧 GT 变换到当前帧坐标系，然后计算双向 Chamfer Distance。根据每条曲线自身的尺度设定自适应阈值：
    $\delta = \alpha \frac{w + h}{2}$
   只有 CD 小于阈值的才建立匹配。这样避免了固定阈值忽略曲线尺度差异的问题。

3. **Dynamic Query Noising**：匹配后的曲线已有因 ego-motion 产生的自然偏差，再添加等量随机噪声不合理。因此根据 Chamfer Distance 动态衰减噪声：
    $R_{decay} = 1 - \frac{D}{\gamma \cdot \frac{\delta}{\alpha}}$
   自然偏差越大（D 越大），额外添加的随机噪声越小。最终噪声实例：$B_{ins} = \{x,y,w,h\} + \eta \cdot R_{decay}$

### 损失函数 / 训练策略

总训练损失 = 地图损失 + 去噪损失：

- 地图损失：$\mathcal{L}_{map} = \lambda_1 \mathcal{L}_{Focal} + \lambda_2 \mathcal{L}_{line} + \lambda_3 \mathcal{L}_{trans}$
- 去噪损失：$\mathcal{L}_{denoise} = \lambda_4 \mathcal{L}_{Focal}^{DN} + \lambda_5 \mathcal{L}_{line}^{DN}$

训练策略：
- 单帧训练阶段使用 Normal Query Denoising，流式训练阶段使用 Stream Query Denoising
- nuScenes 训练 24 epochs，Argoverse2 训练 30 epochs
- backbone 默认 ResNet-50，BEV 提取用 BEVFormer 单层 encoder
- 8× V100 GPU，batch size 32

## 实验关键数据

### 主实验（nuScenes val）

| 方法 | Backbone | 感知范围 | AP_ped | AP_div | AP_bound | mAP |
|------|----------|---------|--------|--------|----------|-----|
| MapTR | R50 | 60×30m | 46.3 | 51.5 | 53.1 | 50.3 |
| StreamMapNet | R50 | 60×30m | 60.4 | 61.9 | 58.9 | 60.4 |
| **SQD-MapNet** | **R50** | **60×30m** | **63.0** | **62.5** | **63.3** | **63.9** |
| StreamMapNet | R50 | 100×50m | 62.9 | 63.1 | 55.8 | 60.6 |
| **SQD-MapNet** | **R50** | **100×50m** | **67.0** | **65.5** | **59.5** | **64.0** |
| **SQD-MapNet** | **V2-99** | **60×30m** | **74.2** | **72.3** | **75.6** | **74.0** |

### 消融实验

| 配置 | mAP | 说明 |
|------|-----|------|
| 单帧 baseline | 59.9 | 无时序 |
| + Temporal Stream | 59.2 (-0.7) | 直接加流式反而掉点 |
| + Dynamic Query Noising | 62.9 (+3.7) | SQD 核心贡献 |
| + Adaptive Temporal Matching | 63.9 (+1.0) | ATM 进一步提升 |

| 去噪策略 | mAP |
|----------|-----|
| 无去噪 | 59.2 |
| Normal Query Denoising | 62.7 |
| Stream Query Denoising | 63.9 |

### 关键发现

- 直接使用流式时序传播反而降性能（-0.7），说明曲线的时序建模确实困难
- SQD 策略贡献 +4.7 mAP（从 59.2 到 63.9），是性能提升的核心
- 自适应匹配阈值 $\alpha=0.1$ 时最佳（63.5），固定阈值最高只到 62.8
- 噪声衰减率 $\gamma=0.2$ 时达到最优平衡（63.9）
- Normal DN 和 Stream DN 有互补作用：Stream DN 包含部分 Normal DN 的效果

## 亮点与洞察

1. **首次将 Query Denoising 引入 HD 地图构建**：发现曲线和框的本质差异（部分增长/截断），提出了统一的最小外接矩形 + box shifting/scaling 噪声方案
2. **Stream Query Denoising 一石二鸟**：通过对前帧 GT 加噪来模拟 stream query 的行为，同时学习时序 query 建模和普通 query 去噪
3. **自适应设计贯穿始终**：每条曲线根据自身尺度设定匹配阈值（ATM）和噪声衰减率（Dynamic QN），而非一刀切
4. **训练时增强、推理时无开销**：SQD 仅在训练时使用，不影响推理效率

## 局限与展望

1. 仅在 StreamMapNet 一个流式基线上验证，缺少更多流式方法的验证
2. 不同地图元素（人行横道、车道线、路边界）的噪声策略是否应有差异，未深入探讨
3. 前帧 GT 到当前帧的匹配是离线计算的，若实时计算开销如何尚未分析
4. 可探索将 SQD 扩展到 3D 目标检测的 streaming 方法（如 StreamPETR）

## 相关工作与启发

- DN-DETR/DINO 的 query denoising 思路启发了本工作，但曲线 vs 框的差异需要全新设计
- StreamMapNet 的时序 query 传播机制是本文的基础
- ATM 的自适应匹配思想可推广到其他需要跨帧关联的任务（如视频分割）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将 query denoising 从框扩展到曲线，stream query denoising 的设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — nuScenes + Argoverse2 两个数据集，消融覆盖各关键组件和超参敏感性
- **写作质量**: ⭐⭐⭐⭐ — 动机分析清晰，图2（曲线帧间变化）很好地说明了问题
- **价值**: ⭐⭐⭐⭐ — SQD 策略即插即用，对流式 HD 地图构建有实际推动作用

<!-- RELATED:START -->

## 相关论文

- [MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2026/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](../../CVPR2025/autonomous_driving/uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [Enhancing Vectorized Map Perception with Historical Rasterized Maps](enhancing_vectorized_map_perception_with_historical_rasterized_maps.md)
- [SafeMap: Robust HD Map Construction from Incomplete Observations](../../ICML2025/autonomous_driving/safemap_robust_hd_map_construction_from_incomplete_observations.md)

<!-- RELATED:END -->
