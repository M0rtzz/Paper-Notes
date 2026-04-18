---
title: >-
  [论文解读] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds
description: >-
  [ECCV 2024][3D视觉][隐式场] 提出一种非线性隐式滤波器，在不需要法线的情况下对神经SDF的隐式场进行平滑同时保留尖锐几何细节，并通过扩展到非零等值面实现全场一致性正则化。
tags:
  - ECCV 2024
  - 3D视觉
  - 隐式场
  - 符号距离函数
  - 点云重建
  - 双边滤波
  - 特征保持
---

# Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds

**会议**: ECCV 2024  
**arXiv**: [2407.13342](https://arxiv.org/abs/2407.13342)  
**代码**: https://list17.github.io/ImplicitFilter (有项目页)  
**领域**: 3D视觉  
**关键词**: 隐式场, 符号距离函数, 点云重建, 双边滤波, 特征保持

## 一句话总结

提出一种非线性隐式滤波器，在不需要法线的情况下对神经SDF的隐式场进行平滑同时保留尖锐几何细节，并通过扩展到非零等值面实现全场一致性正则化。

## 研究背景与动机

**领域现状**: 神经符号距离函数（Neural SDF）已成为从3D点云重建表面的主流方法，通过过拟合单个点云上的MLP来预测空间中每个点的符号距离值，然后用Marching Cubes提取零等值面。

**现有痛点**: 现有方法（NeuralPull、GridPull、DIGS等）仅对单个查询点施加约束（如eikonal约束、梯度方向约束），忽略了邻域几何信息。导致重建表面包含噪声且遗漏尖锐边角等几何细节。

**核心矛盾**: 离散点云缺乏显式的符号距离监督，神经网络的连续性并不能保证所有位置的预测正确，尤其在点密度不足的区域（如尖锐边缘）无法获得可靠的几何引导。

**本文目标**: 如何利用邻域几何信息来平滑SDF隐式场中的噪声，同时保留高频几何特征（尖锐边角）。

**切入角度**: 借鉴图像处理中的双边滤波思想，设计一种作用于隐式场的非线性滤波器，同时考虑点的空间位置和SDF梯度（作为法线代理），并巧妙地扩展到非零等值面以正则化整个距离场。

**核心 idea**: 通过最小化查询点到邻域点梯度方向的加权投影距离来滤波隐式场，同时将输入点沿梯度推到非零等值面以实现全场一致性。

## 方法详解

### 整体框架

方法基于无监督神经SDF学习框架。给定无方向的3D点云 $\boldsymbol{P}=\{\boldsymbol{p}_i\}_{i=1}^N$，训练一个MLP $f_\theta: \mathbb{R}^3 \to \mathbb{R}$ 预测符号距离。训练损失由四项组成：零等值面滤波损失、全场滤波损失、距离损失和Chamfer距离损失。核心创新在于设计了一种基于SDF梯度的双边隐式滤波器。

### 关键设计

1. **零等值面双边滤波 (Level Set Bilateral Filtering)**: 假设所有输入点位于零等值面 $\mathcal{S}_0$ 上，对于零等值面上的点 $\bar{\boldsymbol{p}}$，不是简单地取邻域点的加权平均（会过度平滑尖锐特征），而是最小化到邻域点梯度方向的**加权投影距离**。双向投影滤波算子为：

$$d_{bi}(\bar{\boldsymbol{p}}) = \frac{\sum_{\boldsymbol{p}_j \in \mathcal{N}}(|\boldsymbol{n}_{p_j}^T(\bar{\boldsymbol{p}}-\boldsymbol{p}_j)| + |\boldsymbol{n}_{\bar{p}}^T(\bar{\boldsymbol{p}}-\boldsymbol{p}_j)|)\phi(\|\bar{\boldsymbol{p}}-\boldsymbol{p}_j\|)\psi(\boldsymbol{n}_{\bar{p}}, \boldsymbol{n}_{p_j})}{\sum_{\boldsymbol{p}_j \in \mathcal{N}}\phi(\|\bar{\boldsymbol{p}}-\boldsymbol{p}_j\|)\psi(\boldsymbol{n}_{\bar{p}}, \boldsymbol{n}_{p_j})}$$

其中 $\phi$ 是基于空间距离的高斯权重，$\psi$ 是基于法线相似度的高斯权重：$\psi(\boldsymbol{n}_{\bar{p}}, \boldsymbol{n}_{p_j}) = \exp\left(-\frac{1-\boldsymbol{n}_{\bar{p}}^T\boldsymbol{n}_{p_j}}{1-\cos(\sigma_n)}\right)$，法线由SDF梯度归一化得到 $\boldsymbol{n} = \nabla f_\theta / \|\nabla f_\theta\|$。

设计动机：与简单均值滤波不同，投影到梯度方向可以保留尖锐特征——在尖锐边缘处，法线差异大使得跨边的邻域点权重 $\psi$ 趋近于零。

2. **采样与拉取机制 (Sampling via NeuralPull)**: 直接在零等值面上采样困难，因此借鉴NeuralPull思想，在表面附近随机采样查询点 $\boldsymbol{q}$，然后沿梯度方向拉取到零等值面：

$$\hat{\boldsymbol{q}} = \boldsymbol{q} - f_\theta(\boldsymbol{q})\frac{\nabla f_\theta(\boldsymbol{q})}{\|\nabla f_\theta(\boldsymbol{q})\|}$$

零等值面滤波损失：$L_{zero} = \sum_{\hat{\boldsymbol{q}} \in \hat{\boldsymbol{Q}}} d_{bi}(\hat{\boldsymbol{q}})$。效率优化：由于 $\hat{\boldsymbol{q}}$ 随训练动态变化需反复搜索邻域，改用原始查询点 $\boldsymbol{q}$ 的最近邻 $NN(\boldsymbol{q})$ 的邻域来近似。

3. **非零等值面扩展 (Extension to Non-Zero Level Sets)**: 将滤波从零等值面扩展到整个SDF场。对于位于等值面 $\mathcal{S}_{f_\theta(\boldsymbol{q})}$ 上的查询点 $\boldsymbol{q}$，通过将输入点反向沿梯度投影到该等值面来构造邻域：

$$\mathcal{N}(\boldsymbol{q}, \mathcal{S}_{f_\theta(q)}) = \left\{\hat{\boldsymbol{p}} \mid \hat{\boldsymbol{p}} = \boldsymbol{p} + f_\theta(\boldsymbol{q})\frac{\nabla f_\theta(\boldsymbol{p})}{\|\nabla f_\theta(\boldsymbol{p})\|}, \boldsymbol{p} \in \mathcal{N}(\hat{\boldsymbol{q}}, \mathcal{S}_0)\right\}$$

全场滤波损失：$L_{field} = \sum_{\boldsymbol{q} \in \boldsymbol{Q}} d_{bi}(\boldsymbol{q})$。

设计动机：仅在零等值面滤波不够，不同等值面之间可能存在不一致，通过跨等值面滤波提升整个SDF场的一致性与正则性。

4. **梯度约束 (Gradient Constraint)**: 隐式滤波可能退化为零梯度的平凡解。使用Chamfer距离损失 $L_{CD}$ 作为梯度约束（比eikonal项更宽松且更有效），通过将拉取点与原始点云的双向最近距离来约束SDF的值和梯度。

### 损失函数 / 训练策略

总损失：

$$L = L_{zero} + \alpha_1 L_{field} + \alpha_2 L_{dist} + \alpha_3 L_{CD}$$

其中 $L_{dist} = \frac{1}{N}\sum|f_\theta(\boldsymbol{p}_i)|$ 约束输入点位于零等值面，$\alpha_1=\alpha_2=1, \alpha_3=10$。滤波参数 $\sigma_n=15°$，$\sigma_p$ 设为邻域最大距离。网络使用OccNet架构 + SAL几何初始化。

## 实验关键数据

### 主实验

**ABC & FAMOUS数据集** (F-Score阈值=0.01):

| 方法 | ABC $CD_{L2}$ | ABC $CD_{L1}$ | ABC F-S. | FAMOUS $CD_{L2}$ | FAMOUS $CD_{L1}$ | FAMOUS F-S. |
|------|----------|----------|------|------------|------------|-------|
| NeuralPull | 0.095 | 0.011 | 0.673 | 0.100 | 0.012 | 0.746 |
| SIREN | 0.022 | 0.012 | 0.493 | 0.025 | 0.012 | 0.561 |
| DIGS | 0.021 | 0.010 | 0.667 | 0.015 | 0.008 | 0.772 |
| **Ours** | **0.011** | **0.009** | **0.691** | **0.008** | **0.007** | **0.778** |

**ShapeNet数据集** (3000+物体):

| 方法 | $CD_{L2}\times100$ | NC | F-Score(0.002) | F-Score(0.004) |
|------|------------|------|------------|------------|
| GridPull | 0.0086 | 0.9723 | 0.9896 | 0.9923 |
| **Ours** | **0.0032** | **0.9779** | **0.9976** | **0.9985** |

**边缘Chamfer距离** (ABC数据集，$ECD_{L2}\times100$):

| 方法 | P2S | NeuralPull | DIGS | **Ours** |
|------|-----|-----------|------|----------|
| $ECD_{L1}$ | 0.0496 | 0.0501 | 0.0786 | **0.0256** |
| $ECD_{L2}$ | 1.055 | 1.255 | 2.493 | **0.399** |

### 消融实验

**损失函数组合** (FAMOUS数据集):

| 损失组合 | $CD_{L1}$ | $CD_{L2}$ | F-S. | NC |
|---------|----------|----------|------|------|
| $L_{pull}$ 仅 | 0.012 | 0.083 | 0.742 | 0.884 |
| $L_{CD}$ 仅 | 0.010 | 0.031 | 0.757 | 0.891 |
| $L_{CD}+L_{zero}$ | 0.008 | 0.018 | 0.772 | 0.905 |
| $L_{CD}+L_{zero}+L_{field}$ | 0.008 | 0.011 | 0.769 | 0.908 |
| **Full (Ours)** | **0.007** | **0.008** | **0.778** | **0.911** |

### 关键发现

- 零等值面滤波 $L_{zero}$ 带来最大提升（$CD_{L2}$从0.031降到0.018），有效去噪并保留几何特征
- 非零等值面扩展 $L_{field}$ 进一步提升全场一致性
- Chamfer距离约束比eikonal约束更适合本方法（去掉eikonal后性能反而略好）
- 双向投影（$d_{bi}$）比单向投影（$d$）提升显著：F-S.从0.726→0.778
- 边缘Chamfer距离指标上，Ours的优势最明显（仅为DIGS的16%），验证了特征保持的有效性

## 亮点与洞察

- **隐式场上的双边滤波**：将经典图像处理中的双边滤波推广到SDF隐式场，并利用梯度作为法线代理，避免依赖GT法线
- **非零等值面正则化**：通过沿梯度将点推到任意等值面来实现全场平滑，思路巧妙且对SDF一致性有实际帮助
- **特征保持机制**：法线相似度权重 $\psi$ 使得跨尖锐边缘的邻域点自动获得低权重，天然保留sharp features
- 方法与现有SDF学习框架（NeuralPull、GridPull等）兼容，是一个即插即用的改进

## 局限与展望

- 需要在表面附近密集采样，计算邻域和梯度开销较大
- 假设所有输入点位于表面上，对于含outlier的噪声点云可能不鲁棒
- 滤波参数 $\sigma_n, \sigma_p$ 虽然在一定范围内鲁棒，但对极端几何（极薄结构）可能需要自适应调整
- 场景级重建中提取的是0.001等值面而非零等值面（开放场景不封闭），这一限制值得进一步探索

## 相关工作与启发

- 与DIGS（使用散度引导平滑表面）和EPI（平滑隐式表面粗糙性）不同，本文首次通过**局部几何滤波**来优化隐式场
- 启发：滤波思想可推广到其他隐式表示（如UDF、NeRF的密度场）
- LOP（局部投影算子）的点云滤波传统在神经隐式表达中的现代化延续

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将双边滤波推广到SDF隐式场并扩展到非零等值面，理论贡献扎实
- **实验充分度**: ⭐⭐⭐⭐⭐ — ShapeNet/ABC/FAMOUS/SRB/3D Scene五个数据集，含边缘Chamfer距离等针对性指标，消融全面
- **写作质量**: ⭐⭐⭐⭐ — 公式推导清晰，图示直观解释了滤波优于均值的原因
- **实用价值**: ⭐⭐⭐⭐ — 作为即插即用模块可广泛集成到现有SDF学习框架中
