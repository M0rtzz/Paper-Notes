---
title: >-
  [论文解读] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising
description: >-
  [ICCV 2025][3D视觉][点云去噪] 提出Noise2Score3D,基于Tweedie公式的全无监督点云去噪框架,从噪声数据直接学习得分函数,实现单步去噪;引入点云全变分度量估计未知噪声参数。
tags:
  - ICCV 2025
  - 3D视觉
  - 点云去噪
  - 无监督学习
  - Tweedie公式
  - 得分函数
  - 全变分
---

# Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising

**会议**: ICCV 2025  
**arXiv**: [2503.09283](https://arxiv.org/abs/2503.09283)  
**代码**: [GitHub](https://github.com/)  
**领域**: 3D视觉  
**关键词**: 点云去噪, 无监督学习, Tweedie公式, 得分函数, 全变分

## 一句话总结

提出Noise2Score3D,基于Tweedie公式的全无监督点云去噪框架,从噪声数据直接学习得分函数,实现单步去噪;引入点云全变分度量估计未知噪声参数。

## 研究背景与动机

3D点云常因传感器误差和环境因素而受噪声干扰。现有深度学习去噪方法面临两个核心问题:

**监督方法依赖配对数据** — 实际场景中获取干净数据困难或不可能

**现有无监督方法局限性大**:
   - TotalDn依赖空间先验,迭代过程缓慢
   - ScoreDenoise中的"得分"基于位移而非似然,无法利用贝叶斯统计工具
   - 泛化性差(跨数据集/噪声级别需重训练)
   - 推理效率低(需迭代去噪)

Noise2Score在2D图像去噪中展示了Tweedie公式的威力:通过将去噪重构为得分函数估计问题,提供了理论优雅且灵活的框架。本文将此思想扩展到3D点云。

## 方法详解

### 理论基础 — Tweedie公式

对于高斯噪声: $\mathbf{y} = \mathbf{x} + \sigma\epsilon$

Tweedie公式给出后验期望的显式表达:
$$E_{p(\mathbf{x}|\mathbf{y})}(\mathbf{x}) = \mathbf{y} + \sigma^2 \nabla_\mathbf{y} \log p(\mathbf{y})$$

其中 $\nabla_\mathbf{y} \log p(\mathbf{y})$ 是噪声数据分布的得分函数。**关键优势**: 给定得分函数和噪声参数,去噪只需一步。

### 得分估计网络

基于KPConv架构:
- 5个编码阶段(14个块): 3→256→512→1024→2048通道
- 4个解码块: 3072→128
- 最终全连接层输出3维得分向量
- 参数量24.3M

使用**Amortized Residual DAE (AR-DAE)**损失:
$$\mathcal{L}_{AR-DAE} = \frac{1}{N}\sum_{i=1}^N \|\sigma_t \cdot S(y'_i) + u\|^2$$

其中 $y'_i = y_i + u \cdot \sigma_t$ 是扰动版本,$u \sim \mathcal{N}(0, I)$。

### 未知噪声参数估计 — 点云全变分

$$TV_{PC} = \sum_{i=1}^N \sum_{j \in \text{neighbors}(i)} w_{i,j} \cdot \sqrt{\|\mathbf{p}_i - \mathbf{p}_j\|^2 + \epsilon^2}$$

通过最小化去噪结果的全变分来估计最优噪声参数:
$$\sigma^* = \arg\min_\sigma TV_{PC}(\hat{x}(\sigma))$$

### 去噪流程

1. 训练: 从噪声点云学习得分函数 $S(y)$
2. 去噪: $\hat{x} = y + \sigma^2 S(y)$（单步完成）
3. 若噪声未知: 在候选 $\sigma$ 范围内搜索最优值

## 实验

### ModelNet-40去噪性能对比 (CD×$10^4$)

| 方法 | 10K点1%噪声 CD | 10K点3%噪声 CD | 50K点1%噪声 CD |
|------|---------------|---------------|---------------|
| Bilateral | 5.865 | 31.034 | 3.711 |
| GLR | 6.592 | 12.890 | 1.860 |
| TotalDn | 8.079 | 29.617 | 5.044 |
| Score-U | 5.514 | 18.239 | 2.696 |
| **Noise2Score3D** | **4.891** | **12.456** | **1.654** |

### 泛化能力验证

| 训练集 | 测试集 | CD↓ | P2M↓ |
|--------|--------|-----|------|
| ModelNet-40 | ModelNet-40 | 最佳 | 最佳 |
| ModelNet-40 | PU-Net | 仍优 | 仍优 |

### 关键发现

1. 在无监督方法中达到SOTA性能,某些设置下接近监督方法
2. **泛化能力突出** — 同一预训练权重可跨数据集和噪声级别使用,无需重训练
3. **高效推理** — 已知噪声参数时单步去噪,优于迭代方法
4. 点云全变分是有效的去噪质量指标,可自动选择最优噪声参数

## 亮点与洞察

1. **贝叶斯统计工具的引入** — Tweedie公式将去噪与得分估计解耦,理论优雅
2. **单步去噪** — 区别于所有现有无监督方法的迭代过程
3. **噪声模型无关性** — 同一损失函数/预训练权重适用于不同噪声模型
4. **点云全变分** — 首次为点云去噪引入无参考质量度量

## 局限性

- 主要验证了高斯噪声,对其他噪声分布的实际效果需验证
- KPConv网络参数较大(24.3M)
- 噪声参数未知时需在候选范围内搜索

## 相关工作

- **传统方法**: 双边滤波, 低秩近似, 图拉普拉斯正则
- **监督方法**: PointCleanNet, ScoreDenoise, IterativePFN
- **无监督方法**: TotalDn, DMR-U, Score-U
- **得分匹配**: Denoising Score Matching, AR-DAE

## 评分

- 新颖性: ⭐⭐⭐⭐ (Tweedie公式到3D点云的扩展)
- 技术深度: ⭐⭐⭐⭐⭐ (理论基础扎实,贝叶斯框架完整)
- 实验充分度: ⭐⭐⭐⭐ (多噪声级别,跨数据集泛化)
- 实用价值: ⭐⭐⭐⭐⭐ (无需干净数据,单步去噪,自动噪声估计)

<!-- RELATED:START -->

## 相关论文

- [U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](../../NeurIPS2025/3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)
- [P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](../../ECCV2024/3d_vision/p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)
- [Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [Revisiting Point Cloud Completion: Are We Ready For The Real-World?](revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

<!-- RELATED:END -->
