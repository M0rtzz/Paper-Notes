---
title: >-
  [论文解读] Dense-SfM: Structure from Motion with Dense Consistent Matching
description: >-
  [CVPR 2025][3D视觉][运动恢复结构] 提出 Dense-SfM 框架，通过高斯泼溅进行轨迹扩展解决稠密匹配产生的碎片化轨迹问题，结合基于 Transformer 和高斯过程的多视图核化匹配精炼模块，实现高精度稠密 SfM 重建。
tags:
  - CVPR 2025
  - 3D视觉
  - 运动恢复结构
  - 稠密匹配
  - 高斯泼溅
  - 特征轨迹
  - 多视图一致性
---

# Dense-SfM: Structure from Motion with Dense Consistent Matching

**会议**: CVPR 2025  
**arXiv**: [2501.14277](https://arxiv.org/abs/2501.14277)  
**代码**: [https://icetea-cv.github.io/densesfm/](https://icetea-cv.github.io/densesfm/)  
**领域**: 3D视觉  
**关键词**: 运动恢复结构, 稠密匹配, 高斯泼溅, 特征轨迹, 多视图一致性

## 一句话总结

提出 Dense-SfM 框架，通过高斯泼溅进行轨迹扩展解决稠密匹配产生的碎片化轨迹问题，结合基于 Transformer 和高斯过程的多视图核化匹配精炼模块，实现高精度稠密 SfM 重建。

## 研究背景与动机

传统 SfM 依赖稀疏关键点匹配，在无纹理区域精度和密度有限。近年来稠密匹配方法（如 DKM、RoMa）虽能在低纹理区域产生可靠匹配，但由于其逐对匹配的本质，会产生**碎片化的特征轨迹**——大部分轨迹仅覆盖两个视图，难以直接用于 SfM。

现有解决方案 DFSfM 采用量化匹配，将子像素匹配合并到网格节点，虽能增加轨迹长度和一致性，但：
- 量化操作会显著降低匹配精度
- 减少匹配数量，降低点云密度
- 严重依赖后续精炼模块

作者的核心洞察是：可以利用 Gaussian Splatting 来评估 3D 点在不同视图中的可见性，从而扩展轨迹长度，而无需牺牲原始匹配精度。

## 方法详解

### 整体框架

三阶段流程：
1. **初始 SfM**: 用稠密匹配器（DKM/RoMa）进行双视图匹配，通过双向验证筛选可靠匹配，三角化构建初始 SfM 模型
2. **轨迹扩展**: 用 Gaussian Splatting 评估每个 3D 点的可见性，将点投影到更多视图以延长轨迹
3. **迭代精炼**: 通过多视图核化匹配模块和几何 Bundle Adjustment 迭代优化

### 关键设计

1. **双向验证的稠密匹配**:
    - 功能：从稠密匹配结果中筛选出高可靠性的对应点
    - 核心思路：对图像 A→B 和 B→A 进行双向稠密匹配，计算往返误差 $\|p_a - p_{a'}\|_2 \leq \epsilon_p$（$\epsilon_p=3$px），类似于最近邻互匹配思想。先用非极大值抑制采样，再用双向验证过滤
    - 设计动机：稠密匹配会产生大量低质量匹配，双向验证是高效的质量控制手段

2. **基于 Gaussian Splatting 的轨迹扩展**:
    - 功能：解决逐对匹配产生的碎片化短轨迹问题，将轨迹从2视图扩展到多视图
    - 核心思路：用初始 SfM 的 3D 点初始化小高斯（位置设为点坐标，旋转为单位矩阵，不透明度为1），然后优化和稠密化以覆盖场景，同时固定初始高斯参数。渲染时通过可见性公式 $M = [\max_{r \in R}\{\alpha_{SfM}\prod_{j=1}^{N_{SfM}-1}(1-\alpha_j)\} > \epsilon_v]$ 判断点是否可见（$\epsilon_v=0.5$），将可见点投影到新视图并经几何验证后加入轨迹
    - 设计动机：Gaussian Splatting 训练和渲染速度快，可高效判断遮挡关系；投影方式保持匹配精度不损失

3. **多视图核化匹配模块**:
    - 功能：精炼扩展后的特征轨迹，提升多视图一致性
    - 核心思路：分为两条路径——(1) **特征路径**：用 Transformer 的自注意力和交叉注意力处理参考视图和查询视图的特征图；(2) **坐标嵌入路径**：用高斯过程计算后验均值 $\mu(\mathbf{F}_R|\mathbf{F}_{Q_i})$，利用指数余弦相似度核函数，结合位置编码提供空间几何信息。两路径结果拼接后送入 CNN 解码器，输出每像素坐标概率分布和置信度分数
    - 设计动机：DFSfM 的精炼模块仅用统计方差作为不确定性度量；本文直接通过网络学习置信度，用联合损失 $\mathcal{L} = \frac{1}{N}\sum s_{Q_i} \cdot \|p_{Q_i}-p_{gt}\|_2 - \alpha \log s_{Q_i}$ 同时优化精度和可靠性

### 损失函数 / 训练策略

- 精炼模块在 MegaDepth 数据集上训练，使用 SfM 模型提供的 GT 轨迹
- 训练时对 GT 2D 位置加随机噪声作为输入
- 损失函数结合精度项和置信度正则项（$\alpha=20$），避免模型过于不自信
- Bundle Adjustment 迭代两次，重投影误差超过 $\epsilon_f=3$px 的点被过滤

## 实验关键数据

### 主实验（ETH3D 3D三角化）

| 方法 | Accuracy@2cm (%) | Completeness@2cm (%) | 说明 |
|------|------|----------|------|
| SP+SG+PixSfM | 87.04 | 2.77 | 检测器方法 |
| LoFTR+DFSfM | 89.01 | 11.07 | 半稠密+量化 |
| RoMa+DFSfM | 88.42 | 9.79 | 稠密+量化 |
| **RoMa+Ours** | **92.62** | **17.06** | 稠密+GS轨迹扩展 |

### 消融实验（来自论文 Tab. 3）

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无轨迹扩展 | 较低精度和完整度 | 碎片化短轨迹限制精炼 |
| +轨迹扩展 (GS) | 显著提升 | 更长轨迹提供更多几何约束 |
| DFSfM精炼模块 | 次优 | 基于统计量的不确定性度量 |
| 本文精炼模块 | 最优 | 学习的置信度+高斯过程 |

### 关键发现

- Dense-SfM (RoMa) 在 ETH3D 上 accuracy@2cm 达到 92.62%，completeness@2cm 达到 17.06%，全面超越所有基线
- 相比 RoMa+DFSfM（使用量化匹配），accuracy 提升 4.2%，completeness 提升 74%——充分证明了避免量化的优势
- 在 Texture-Poor SfM 数据集上同样表现优异，验证了对低纹理场景的有效性
- 轨迹扩展使后续精炼产生更好的轨迹，二者相辅相成
- 多视图核化匹配中的高斯过程路径提供了重要的位置信息补充

## 亮点与洞察

- **GS 作为可见性工具**: 将 Gaussian Splatting 从渲染工具转变为 SfM 流程中的可见性判断工具，思路新颖
- **精度零损失的轨迹扩展**: 通过投影+几何验证而非重新匹配来扩展轨迹，完全保持原始匹配精度
- **特征+坐标双路径**: Transformer 处理外观特征，高斯过程处理空间位置，两种信息源互补
- **与 MASt3R-SfM 的对比**: MASt3R-SfM 在 ETH3D 三角化上表现很差（accuracy@2cm 仅43.9%），说明端到端方法在精确三角化上仍有差距

## 局限与展望

- 需要训练 Gaussian Splatting 模型增加计算开销
- 双向稠密匹配本身计算量大（每对图像需要两次前向传播）
- 轨迹扩展依赖初始 SfM 的位姿质量
- 未探索与学习型 SfM 方法（如 VGGSfM）的结合
- 非极大值抑制的半径设置因数据集不同而异

## 相关工作与启发

- DFSfM 的量化策略虽然增加了轨迹一致性，但牺牲了精度和密度——Dense-SfM 优雅地避开了这个 trade-off
- PixSfM 通过特征度量调整精炼关键点，本文则通过 Transformer+GP 双路径架构获得更好效果
- 启发：稠密匹配+SfM 的融合仍有设计空间，轨迹一致性是核心挑战

## 评分

- 新颖性: ⭐⭐⭐⭐ GS 轨迹扩展思路新颖，GP+Transformer 双路径精炼设计巧妙
- 实验充分度: ⭐⭐⭐⭐ ETH3D、Texture-Poor、IMC 三个数据集覆盖面广，消融研究充分
- 写作质量: ⭐⭐⭐⭐ 流程图清晰，各模块动机交代充分
- 价值: ⭐⭐⭐⭐ 解决了稠密匹配融入SfM的实际痛点，实用性强

<!-- RELATED:START -->

## 相关论文

- [Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)
- [A Unified Image-Dense Annotation Generation Model for Underwater Scenes](a_unified_image-dense_annotation_generation_model_for_underwater_scenes.md)
- [MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)

<!-- RELATED:END -->
