---
title: >-
  [论文解读] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction
description: >-
  [ECCV 2024][自动驾驶][3D Occupancy] 提出以物体为中心的 3D 语义高斯表示替代传统密集体素，用一组稀疏的 3D 语义高斯描述场景并通过高斯到体素的 splatting 生成占用预测，在性能可比的情况下将内存消耗降低 75%-82%。
tags:
  - ECCV 2024
  - 自动驾驶
  - 3D Occupancy
  - 3D Gaussian Splatting
  - 稀疏表示
  - 以物体为中心
  - 语义占用
---

# GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction

**会议**: ECCV 2024  
**arXiv**: [2405.17429](https://arxiv.org/abs/2405.17429)  
**代码**: [GitHub](https://github.com/huang-yh/GaussianFormer)  
**领域**: 自动驾驶  
**关键词**: 3D Occupancy, 3D Gaussian Splatting, 稀疏表示, 以物体为中心, 语义占用

## 一句话总结

提出以物体为中心的 3D 语义高斯表示替代传统密集体素，用一组稀疏的 3D 语义高斯描述场景并通过高斯到体素的 splatting 生成占用预测，在性能可比的情况下将内存消耗降低 75%-82%。

## 研究背景与动机

3D 语义占用预测是视觉自动驾驶系统鲁棒性的关键任务，但现有场景表示方式存在明显缺陷：

**体素表示冗余**：密集体素方法对每个 3D 位置分配等量资源，但驾驶场景超过 90% 是空区域，导致计算和存储浪费

**平面表示信息损失**：BEV 和 TPV 通过压缩高度维度提升效率，但不可避免地丢失细粒度 3D 结构信息

**网格方法缺乏灵活性**：网格表示无法适应不同物体尺度和区域复杂度（如公交车 vs 行人，道路 vs 植被），资源分配不合理

**动态场景建模困难**：网格表示难以捕捉场景动态，因为移动的是物体而非网格

本文的核心思想受 3D Gaussian Splatting 启发：用高斯混合模型的通用近似能力来描述场景，每个高斯自适应地覆盖一个感兴趣区域。

## 方法详解

### 整体框架

GaussianFormer 包含三个阶段：
1. **图像编码**：Backbone + FPN 提取多尺度 2D 特征
2. **高斯生成（Image to Gaussians）**：通过交替的自编码、图像交叉注意力和属性细化迭代更新 3D 高斯
3. **高斯到体素 Splatting**：通过局部聚合将稀疏高斯转化为密集的 3D 占用预测

### 关键设计

#### 以物体为中心的 3D 场景表示

每个 3D 语义高斯由 d 维向量表示：G = (m, s, r, c)
- **m** (3维): 均值/位置
- **s** (3维): 尺度
- **r** (4维): 旋转（四元数）
- **c** (|C|维): 语义 logits

位置 p 处的占用预测为所有高斯在该点贡献的加和：o_hat(p) = sum_i exp(-0.5 * (p-m_i)^T * Sigma_i^{-1} * (p-m_i)) * c_i。其中协方差矩阵 Sigma = R * S * S^T * R^T。与体素表示相比，高斯的均值和协方差属性允许自适应分配资源——大物体用大高斯覆盖，小物体用小高斯精细描述。

#### GaussianFormer 模型（Image to Gaussians）

**高斯属性和查询**：分为两组特征——
- 高斯属性 G（物理属性，是学习目标）
- 高斯查询 Q（高维特征向量，隐式编码 3D 信息）

两者均初始化为可学习向量，通过 B 个 block 迭代细化。

**自编码模块**：将高斯视为位于其均值处的点云，使用 3D 稀疏卷积处理（而非 deformable attention），计算复杂度为线性。高斯数量 P 远小于体素总数，稀疏卷积能充分利用高斯的稀疏性。

**图像交叉注意力模块（ICA）**：对每个高斯，根据其均值和协方差生成 3D 参考点，投影到多视图图像特征上采样。使用 deformable attention 聚合图像信息。参考点偏移根据协方差计算以反映高斯分布形状。

**属性细化模块**：从高斯查询 Q 通过 MLP 解码中间属性，均值采用残差更新（保持迭代过程中的连贯性），其他属性（尺度、旋转、语义）直接替换（避免 sigmoid/softmax 的梯度消失问题）。

#### 高斯到体素 Splatting

直接遍历所有高斯代价过高（复杂度 O(XYZ * P)），利用高斯分布的局部性，仅考虑每个体素的邻域高斯：

1. 将高斯嵌入目标体素网格（按均值定位）
2. 根据尺度属性计算每个高斯的邻域半径
3. 建立（高斯索引, 体素索引）对的列表
4. 按体素索引排序，得到每个体素应聚合的高斯集合
5. 局部聚合生成占用预测

整个过程通过 CUDA 实现以充分利用 GPU 并行计算能力。

### 损失函数 / 训练策略

- **损失函数**：Cross Entropy Loss + Lovasz-Softmax Loss，对每个细化模块的输出施加监督
- **总损失**：L = sum_{i=1}^{B} (L_ce^i + L_lov^i)
- **优化器**：AdamW，weight decay 0.01，学习率预热 500 iter 后最大 2e-4，余弦衰减
- **训练**：20 epochs，batch size 8，nuScenes 使用 ResNet101-DCN（FCOS3D 预训练），KITTI-360 使用 R50
- **高斯数量**：nuScenes 144000 个，KITTI-360 38400 个
- **GaussianFormer blocks**：4 个
- **数据增强**：随机翻转 + 光度畸变

## 实验关键数据

### 主实验（nuScenes val，SurroundOcc 标注）

| 方法 | SC IoU | SSC mIoU |
|------|--------|----------|
| MonoScene | 23.96 | 7.31 |
| BEVFormer | 30.50 | 16.75 |
| TPVFormer* | 30.86 | 17.10 |
| OccFormer | 31.39 | 19.03 |
| SurroundOcc | 31.49 | 20.30 |
| **GaussianFormer** | 29.83 | 19.10 |

GaussianFormer 以可比的性能（mIoU 19.10 vs SurroundOcc 20.30）实现了仅用 SurroundOcc **17.8%** 的内存消耗，仅用 OccFormer **24.8%** 的内存消耗。

### 消融实验

**高斯数量对性能和内存的影响**（nuScenes）：

| 高斯数量 | mIoU | 内存 (MB) |
|---------|------|-----------|
| 36000 | 17.06 | 5755 |
| 72000 | 18.40 | 8115 |
| 144000 | 19.10 | 12741 |
| 288000 | 19.23 | 21993 |

144000 个高斯是性能-效率的最佳平衡点。

**模块消融**：
- 稀疏卷积自编码 vs 无自编码：+0.75 mIoU
- ICA 中协方差感知偏移 vs 固定偏移：+0.39 mIoU
- 迭代细化（4 blocks）vs 单次预测：显著提升

### 关键发现

1. 144000 个高斯即可达到与密集方法（640000 体素）可比的性能
2. 内存节省 75%-82% 是实质性的工程贡献，对边缘设备部署极为重要
3. 高斯表示能自适应物体尺度——大物体被大高斯覆盖，小物体被紧凑高斯覆盖
4. 稀疏卷积在非结构化高斯上的应用是有效的自编码方案

## 亮点与洞察

- **首个以物体为中心的占用表示**：跳出了体素/平面等网格表示的思维定式，用高斯分布的灵活性描述场景
- **3D-GS 的新应用**：将 3D Gaussian Splatting 从离线渲染迁移到在线感知，且是 3D-to-3D（非渲染到 2D），具有原创性
- **CUDA 实现的 splatting**：局部聚合 + 排序的高效实现使得稀疏到密集的转换可行
- **与下游任务的天然结合**：高斯以物体为中心，天然适合运动预测和端到端驾驶

## 局限与展望

1. 高斯数量仍是需要手动设定的超参数，未实现自适应数量调整
2. 在尺度极端的场景（如远处小行人）中可能表示不足
3. SC IoU（29.83）低于密集方法（31.49），说明几何完整性有待提升
4. 缺乏时序建模的设计，未利用历史帧信息
5. 高斯到体素的 splatting 仍涉及密集体素网格，最终输出不是完全稀疏的

## 相关工作与启发

- **3D Gaussian Splatting**：提供了高斯表示的物理形式，但 GaussianFormer 是在线学习而非离线优化
- **TPVFormer**：三平面表示的效率和 GaussianFormer 的高斯表示形成对比
- **SurroundOcc**：密集体素方法的代表，GaussianFormer 以极少内存达到可比性能
- **SparseOcc**：另一种稀疏占用思路（mask-based），与高斯表示互补
- 高斯表示可进一步扩展到 4D 场景预测和开放词汇占用

## 评分

- **创新性**: 5/5 - 首创以物体为中心的占用表示，3D-GS 的全新应用
- **实验质量**: 4/5 - 双数据集验证，消融详细，但性能未超越 SOTA
- **写作质量**: 4/5 - 动机和方法阐述清晰，图示优秀
- **实用价值**: 4/5 - 内存节省显著，但需要 CUDA 支持的 splatting 模块

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](../../ICCV2025/autonomous_driving/semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ECCV 2024\] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](occworld_learning_a_3d_occupancy_world_model_for_autonomous_driving.md)
- [\[CVPR 2025\] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](../../CVPR2025/autonomous_driving/gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)

<!-- RELATED:END -->
