---
title: >-
  [论文解读] Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene
description: >-
  [ECCV2024][自动驾驶][目标检测] 提出 LiSe 方法，将 2D 图像信息引入无监督 3D 目标检测，通过自步学习（self-paced learning）中的自适应采样和弱模型聚合策略，大幅提升远距离和小目标的检测能力。
tags:
  - ECCV2024
  - 自动驾驶
  - 目标检测
  - LiDAR-Camera Fusion
  - Self-paced Learning
  - Pseudo Labels
---

# Approaching Outside: Scaling Unsupervised 3D Object Detection from 2D Scene

**会议**: ECCV2024  
**arXiv**: [2407.08569](https://arxiv.org/abs/2407.08569)  
**代码**: [GitHub](https://github.com/Ruiyang-061X/LiSe)  
**领域**: autonomous_driving  
**关键词**: Unsupervised 3D Object Detection, LiDAR-Camera Fusion, Self-paced Learning, Pseudo Labels

## 一句话总结

提出 LiSe 方法，将 2D 图像信息引入无监督 3D 目标检测，通过自步学习（self-paced learning）中的自适应采样和弱模型聚合策略，大幅提升远距离和小目标的检测能力。

## 背景与动机

无监督 3D 目标检测旨在不使用任何标注数据的情况下发现和定位场景中的 3D 物体，对自动驾驶安全至关重要。现有方法（如 MODEST、OYSTER）主要依赖 LiDAR 点云：

- **LiDAR 的稀疏性问题**：远距离或小尺寸目标的点云极度稀疏，返回点数极少，导致前景与背景难以区分
- **自训练中的偏差**：迭代训练过程中，模型倾向于过拟合近距离、大尺寸等容易检测的样本，逐渐丧失对困难样本的检测能力
- **2D 图像的互补价值**：RGB 图像纹理丰富，开放词汇 2D 检测器对远处和小物体有很好的识别能力，可以弥补 LiDAR 的不足

作者观察到 LiDAR 和 2D 图像在不同距离和不同分辨率的物体上具有互补性，因此首次尝试将两者融合用于无监督 3D 检测任务。

## 核心问题

1. 如何在无监督设定下有效融合 LiDAR 和 2D 图像信息来生成高质量伪标签？
2. 自训练过程中模型对常见/容易样本的过拟合导致长尾样本（远距离、小体积物体）检测能力下降。
3. 不同轮次训练得到的模型各有偏好，如何将多个"弱模型"整合为一个全面的"强模型"？

## 方法详解

### 整体框架

LiSe 由三个核心组件构成：LiDAR 与 2D 场景融合生成伪标签、自适应采样策略、弱模型聚合，统一在自步学习流程中。

### 1. LiDAR 与 2D 场景融合

**LiDAR 伪框生成**：

- 采用多次遍历（multi-traversal）方法：同一位置多次经过时，位置不变的点为静态背景，发生位移的点为前景物体
- 计算每个点的持续性分数（ppScore），低分表示动态点
- 构建图结构，使用改进的 DBSCAN 聚类，滤除静态簇后对前景簇拟合 3D 包围框

**图像伪框生成**：

- 使用 GroundingDINO 开放词汇检测器获取 2D 检测框
- 将 2D 框作为 prompt 输入 SAM（Segment Anything Model）获取精细 2D 掩码
- 利用相机内外参矩阵将 LiDAR 点投影到 2D 平面，保留落在掩码内的 3D 点
- 对保留的 3D 点使用区域生长算法聚类并拟合 3D 包围框

**距离感知融合策略**：

$$\mathcal{B}_{final} = \mathcal{B}_{LiDAR} \cup \{b_i \mid d(b_i) \geq d_{min},\, b_i \in \mathcal{B}_{img}\}$$

仅融合距离超过 $d_{min}$（实验中为 10m）的图像伪框，因为近距离 LiDAR 已足够准确，图像框反而会引入冲突。

### 2. 自适应采样策略（Adaptive Sampling）

针对自训练中模型对简单样本过拟合的问题：

- **距离-体积度量**：按距离（近 0-30m / 远 >30m）和体积（小 <5m³ / 大 >5m³）将物体分为四组
- 计算训练前的初始分布 $Q_{init}$ 和推理后的分布 $Q$
- 推理后占比增大的组（模型已擅长）→ 下一轮降采样
- 推理后占比减小的组（模型变差）→ 下一轮上采样

采样分数公式：

$$R(g_i) = \begin{cases} 1 - (Q(g_i) - Q_{init}(g_i)) & \text{if } Q(g_i) > Q_{init}(g_i) \\ 1 + (Q_{init}(g_i) - Q(g_i)) & \text{if } Q(g_i) \leq Q_{init}(g_i) \end{cases}$$

### 3. 弱模型聚合（Weak Model Aggregation）

不同轮次的模型因采样率不同而擅长不同类型目标（如第 $t$ 轮擅长大物体，第 $t+1$ 轮擅长小物体），称为"弱模型"。聚合方式：

$$\Theta_t = \lambda \cdot \Theta_{t-1} + (1 - \lambda) \cdot \theta_t \quad (T_s \leq t \leq T)$$

从第 $T_s$ 轮开始，用聚合系数 $\lambda$ 将当前弱模型权重与历史聚合模型加权平均，逐步构建综合能力更强的模型。实验中 $T_s=8$，$\lambda=0.999$。

### 自步学习整体流程

1. **种子训练**：用融合伪标签训练初始检测器
2. **自训练迭代**（共 $T=10$ 轮）：上一轮模型推理 → 自适应采样调整伪标签分布 → 训练新模型 → 弱模型聚合

## 实验关键数据

骨干网络为 PointRCNN，评价指标为 AP_BEV 和 AP_3D（IoU=0.25）。

### nuScenes 主要结果

| 方法 | 0-30m | 30-50m | 50-80m | 0-80m |
|------|-------|--------|--------|-------|
| 有监督上界 | 39.8/34.5 | 12.9/10.0 | 4.4/2.9 | 22.2/18.2 |
| MODEST (T=10) | 24.8/17.1 | 5.5/1.4 | 1.5/0.3 | 11.8/6.6 |
| OYSTER (T=2) | 26.6/19.3 | 4.4/1.8 | 1.7/0.4 | 12.7/8.0 |
| **LiSe (T=10)** | **35.0/24.0** | **11.4/4.4** | **4.8/1.3** | **19.8/11.4** |

- 全范围：相比 OYSTER 提升 **+7.1% AP_BEV** 和 **+3.4% AP_3D**
- 远距离（50-80m）：AP_BEV 4.8% **超越有监督模型**的 4.4%

### Lyft 主要结果

| 方法 | 0-30m | 50-80m | 0-80m |
|------|-------|--------|-------|
| MODEST (T=10) | 73.8/71.3 | 27.0/24.8 | 57.3/55.1 |
| **LiSe (T=10)** | **76.7/74.0** | **46.6/43.7** | **65.6/62.5** |

- 远距离（50-80m）提升 **+19.4% AP_BEV** 和 **+18.9% AP_3D**，效果极其显著

### 消融实验关键发现

- 单独使用 2D 伪框在近距离不如 LiDAR，但融合后全面提升
- 距离感知融合中 >10m 阈值最优，避免近距离模态冲突
- 自适应采样中距离和体积两个因子互补，联合使用效果最佳
- 弱模型聚合从较晚轮次（$T_s=8$）开始、使用大 $\lambda=0.999$（慢更新）效果最好

## 亮点

1. **首创 LiDAR+2D 融合的无监督 3D 检测**：利用 GroundingDINO + SAM 生成图像端伪框，有效补充远距离和小目标
2. **距离感知融合**的设计简洁有效，仅引入一个阈值就解决了模态冲突问题
3. **自适应采样**从数据分布动态调整的角度缓解长尾过拟合，思路通用
4. **弱模型聚合**利用不同轮次模型的互补性，无需额外训练开销就能获得更强模型
5. 远距离检测超越有监督方法，验证了 2D 信息在远距离感知中的独特价值

## 局限与展望

1. **依赖预训练的 2D 模型**：GroundingDINO 和 SAM 的质量直接影响图像端伪标签质量，在新领域可能需要适配
2. **计算开销较大**：需要运行 GroundingDINO + SAM + LiDAR 管线 + 10 轮自训练，训练成本高
3. **融合方式较为简单**：仅在伪标签层面做后融合（late fusion），未探索特征级融合的可能性
4. **类别无关检测**：当前仅检测物体的位置和大小，不区分具体类别
5. 距离阈值 $d_{min}$ 和体积阈值 5m³ 均为人工设定，可能不适用于所有场景

## 与相关工作的对比

| 对比维度 | MODEST/OYSTER | 运动流方法 | LiSe |
|---------|-------------|---------|------|
| 输入模态 | 仅 LiDAR | 仅 LiDAR | LiDAR + 图像 |
| 伪标签来源 | 多次遍历/聚类 | 场景流 | 多次遍历 + 开放词汇检测 + SAM |
| 远距离能力 | 弱 | 弱 | 强 |
| 训练策略 | 朴素自训练 | 单次伪标签 | 自步学习（自适应采样+模型聚合） |
| 长尾处理 | 无 | 无 | 距离-体积自适应采样 |

## 启发与关联

- **2D 基础模型赋能 3D 感知**：开放词汇检测器 + SAM 的组合为无标注 3D 检测提供了可靠的图像端先验，这种范式可以扩展到其他 3D 任务
- **自步学习的 3D 适配**：将距离和体积等 3D 特有属性引入采样策略，是通用自步学习在 3D 领域的有效定制
- **弱模型聚合 vs. 模型汤**：与 Model Soup 等方法类似但在自训练中在线聚合，无需额外选模过程
- 可以考虑将此方法扩展到开放词汇 3D 检测，利用 GroundingDINO 的文本对齐能力为伪标签赋予类别语义

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在无监督 3D 检测中融合 2D 场景，自适应采样和弱模型聚合设计合理
- 实验充分度: ⭐⭐⭐⭐ — 在 nuScenes 和 Lyft 上验证，消融实验覆盖各组件和超参数
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表直观，动机阐述充分
- 价值: ⭐⭐⭐⭐ — 远距离检测超越有监督方法，证明了无监督 LiDAR-Camera 融合的潜力

<!-- RELATED:START -->

## 相关论文

- [SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras](simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)
- [Cubify Anything: Scaling Indoor 3D Object Detection](../../CVPR2025/autonomous_driving/cubify_anything_scaling_indoor_3d_object_detection.md)
- [OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)
- [Weakly Supervised 3D Object Detection via Multi-Level Visual Guidance](weakly_supervised_3d_object_detection_via_multi-level_visual_guidance.md)
- [Detecting As Labeling: Rethinking LiDAR-camera Fusion in 3D Object Detection](detecting_as_labeling_rethinking_lidar-camera_fusion_in_3d_object_detection.md)

<!-- RELATED:END -->
