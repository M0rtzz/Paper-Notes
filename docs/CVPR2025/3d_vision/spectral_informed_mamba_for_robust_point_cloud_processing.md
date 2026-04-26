---
title: >-
  [论文解读] Spectral Informed Mamba for Robust Point Cloud Processing
description: >-
  [CVPR 2025][3D视觉][点云分析] 提出基于图拉普拉斯谱的点云 Mamba 遍历策略 SST，通过表面感知谱遍历（SAST）实现等距变换不变的分类、分层局部遍历（HLT）实现精确分割、遍历感知重定位（TAR）解决 Mamba 中 MAE 的 token 放置问题。
tags:
  - CVPR 2025
  - 3D视觉
  - 点云分析
  - Mamba
  - 图拉普拉斯谱
  - 自监督学习
  - 遍历策略
---

# Spectral Informed Mamba for Robust Point Cloud Processing

**会议**: CVPR 2025  
**arXiv**: [2503.04953](https://arxiv.org/abs/2503.04953)  
**代码**: [GitHub](https://github.com/AliBahri94/SI-Mamba.git)  
**领域**: 3D视觉/点云处理  
**关键词**: 点云分析, Mamba, 图拉普拉斯谱, 自监督学习, 遍历策略

## 一句话总结

提出基于图拉普拉斯谱的点云 Mamba 遍历策略 SST，通过表面感知谱遍历（SAST）实现等距变换不变的分类、分层局部遍历（HLT）实现精确分割、遍历感知重定位（TAR）解决 Mamba 中 MAE 的 token 放置问题。

## 研究背景与动机

状态空间模型（SSM）如 Mamba 因线性复杂度成为 Transformer 的高效替代。然而将 Mamba 应用于点云面临三个关键问题：

- **3D 网格遍历不适合点云**：现有方法（Point-Mamba, PCM）简单将 2D 图像的网格遍历扩展到 3D，但 3D 网格中相邻的 patch 不一定在物体表面上相邻，且遍历顺序依赖视角
- **遍历策略与任务不匹配**：适合分类的全局遍历不适合分割等需要局部精确性的任务
- **MAE 中的 token 位置问题**：Mamba 对 token 顺序敏感，不能像 Transformer 那样随意放置可学习 token

核心洞察：图拉普拉斯算子的特征向量提供了与等距变换无关的表面流形参数化，可以定义更鲁棒的遍历顺序。

## 方法详解

### 整体框架

SST（Spectral Spatial Traversing）方法包含三个组件：SAST 用于分类任务的谱遍历，HLT 用于分割任务的分层遍历，TAR 用于 MAE 自监督预训练中的 token 重定位。

### 关键设计一：表面感知谱遍历（SAST）

- **功能**：为分类任务定义等距不变的 patch 遍历顺序
- **核心思路**：构建 patch 连接图，计算随机游走拉普拉斯 $L_{rw} = I - D^{-1}W$ 的前 $s$ 个非常数最小特征向量。每个特征向量 $v^{(k)}$ 定义一个遍历顺序（按特征值升序和降序各一次），在每个 Mamba block 中执行 $s \times 2$ 次遍历后拼接。特征向量经规范化处理解决符号歧义和重特征值排序歧义
- **设计动机**：拉普拉斯谱具有等距不变性（旋转/平移不改变谱），且低频特征向量编码物体表面流形的平滑参数化，比 3D 网格遍历更好地捕捉表面邻接关系

### 关键设计二：分层局部遍历（HLT）

- **功能**：为分割任务定义同时考虑全部 $s$ 个特征向量的精确遍历
- **核心思路**：受归一化割启发，基于特征向量值的均值对 token 进行递归二分——先按第一个特征向量分为两组，每组再按第二个特征向量细分，依此类推。最终每个 token 获得一个 $s$ 位二进制编码 $b_i = [b_i^{(1)}, ..., b_i^{(s)}]$，按字典序遍历
- **设计动机**：SAST 中各特征向量独立使用，无法区分物体的特定部位（如左臂 vs 右臂）。HLT 通过组合所有特征向量的信息实现更精确的空间定位

### 关键设计三：遍历感知重定位（TAR）

- **功能**：解决 Mamba 中 MAE 预训练的 token 放置问题
- **核心思路**：在 MAE 解码器之前，将可学习 token 恢复到其被遮蔽的原始位置（而非像 Transformer 那样追加到序列末尾），保持谱遍历定义的空间邻接关系
- **设计动机**：Mamba 是方向敏感的序列模型，token 顺序直接影响状态传播。将可学习 token 放在末尾会破坏空间连续性

### 损失函数

自监督预训练使用 Chamfer 距离重建损失：$\mathcal{L}_{rec} = \frac{1}{N_m} \sum_{i=1}^{N_m} \text{Chamfer}(\mathcal{S}_i, \hat{\mathcal{S}}_i)$。

## 实验关键数据

### 主实验：ModelNet40 分类

| 方法 | 参数量 | 整体准确率 (%) |
|------|-------|-------------|
| Point-MAE | 22.1M | 93.2 |
| Point-M2AE | 12.4M | 93.4 |
| Point-Mamba | - | 93.6 |
| **SI-Mamba (Ours)** | ~12M | **94.0** |

### ScanObjectNN 分类（最难设置 PB_T50_RS）

| 方法 | 准确率 (%) |
|------|----------|
| Point-MAE | 85.2 |
| Point-M2AE | 86.4 |
| **SI-Mamba** | **88.1** |

### ShapeNetPart 分割

| 方法 | 类别 mIoU (%) | 实例 mIoU (%) |
|------|-------------|-------------|
| Point-MAE | 84.2 | 86.1 |
| **SI-Mamba (HLT)** | **84.8** | **86.5** |

### 消融实验

| 遍历策略 | ModelNet40 Acc. |
|---------|---------------|
| 3D Grid (基线) | 93.6 |
| SAST (2个特征向量) | 93.8 |
| SAST (4个特征向量) | **94.0** |
| HLT (分割) | 最优分割性能 |

### 关键发现

- SAST 在 ScanObjectNN 上优势最明显（+1.7%），因为该数据集包含旋转和平移变换
- HLT 在分割任务上显著优于 SAST，验证了任务特异性遍历策略的必要性
- TAR 解决了 Mamba MAE 预训练的 token 位置问题，使自监督性能接近 Transformer

## 亮点与洞察

1. **谱图理论与 SSM 的优雅结合**：用拉普拉斯特征向量定义遍历顺序，理论基础扎实
2. **等距不变性有实际价值**：视角无关的遍历策略对真实世界点云（传感器位置不固定）特别重要
3. **分类 vs 分割的遍历策略区分**：SAST 和 HLT 分别针对全局和局部任务设计

## 局限与展望

- 拉普拉斯特征分解有额外计算开销，虽然稀疏矩阵可用高效算法
- 特征向量数量 $s$ 需要手动选择
- 未探索大规模室外点云（如自动驾驶场景）
- 特征向量的重特征值歧义处理可能不够鲁棒

## 相关工作与启发

- **Point-Mamba, PCM**：Mamba 应用于点云的先驱工作
- **Point-MAE, Point-M2AE**：Transformer-based 点云 MAE
- 谱遍历的思想可推广到其他需要序列化不规则数据的 SSM 任务

## 评分

⭐⭐⭐⭐ — 谱图理论与 Mamba 的结合理论优雅，SAST/HLT/TAR 各解决一个明确问题。在多个基准上的一致提升验证了方法有效性。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)
- [\[CVPR 2025\] MICAS: Multi-grained In-Context Adaptive Sampling for 3D Point Cloud Processing](micas_multi-grained_in-context_adaptive_sampling_for_3d_point_cloud_processing.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](../../ICCV2025/3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](../../AAAI2026/3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)

<!-- RELATED:END -->
