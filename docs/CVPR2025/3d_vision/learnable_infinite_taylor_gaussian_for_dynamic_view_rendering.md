---
title: >-
  [论文解读] Learnable Infinite Taylor Gaussian for Dynamic View Rendering
description: >-
  [CVPR 2025][3D视觉][dynamic scene] 提出可学习无穷 Taylor 级数（Learnable Infinite Taylor Formula）建模动态场景中高斯基元的位置/旋转/缩放随时间的演化，用三阶 Taylor 展开捕捉大运动、MLP+LBS 构造 Peano 余项补偿高阶项，实现无近似误差的运动建模，N3DV 和 Technicolor 数据集上超越 SOTA。
tags:
  - CVPR 2025
  - 3D视觉
  - dynamic scene
  - 3D Gaussian Splatting
  - Taylor series
  - Peano remainder
  - novel view synthesis
---

# Learnable Infinite Taylor Gaussian for Dynamic View Rendering

**会议**: CVPR 2025  
**arXiv**: [2412.04282](https://arxiv.org/abs/2412.04282)  
**代码**: [项目页](https://ellisonking.github.io/TaylorGaussian)  
**领域**: 3d_vision  
**关键词**: dynamic scene, 3D Gaussian Splatting, Taylor series, Peano remainder, novel view synthesis

## 一句话总结

提出可学习无穷 Taylor 级数（Learnable Infinite Taylor Formula）建模动态场景中高斯基元的位置/旋转/缩放随时间的演化，用三阶 Taylor 展开捕捉大运动、MLP+LBS 构造 Peano 余项补偿高阶项，实现无近似误差的运动建模，N3DV 和 Technicolor 数据集上超越 SOTA。

## 研究背景与动机

**领域现状**: 动态 3D Gaussian Splatting (3DGS) 在动态场景重建中取得显著进展，但如何准确建模高斯基元属性（位置、旋转、缩放）随时间的连续变化仍是核心挑战。海量时变参数在有限光度数据约束下极易导致收敛到次优解。

**现有方案的局限**:
1. **端到端隐式方法** (如 D3DGS): 用 MLP 直接预测所有高斯点的变形，缺乏显式监督，难以生成高质量变换场，时空一致性弱
2. **时间条件多项式方法**: 显式可解释，但需要大量手工设计，难以在不同场景间泛化
3. **4DGS**: 用六平面分解建模时空特征，但处理多视角困难
4. **StreamRF**: 在线训练策略不适应大视角变化

**核心动机**: 能否结合隐式方法的灵活性和显式多项式的可解释性？Taylor 公式天然适合以多项式逼近复杂函数，且其 Peano 余项可以用神经网络学习，构成完整级数。

## 方法详解

### 整体框架

框架包含四个核心阶段：高斯初始化 → 稀疏点采样 → 高斯点插值 → 高斯变换场建模。变换场建模分为两部分：
- **Taylor 展开部分** $f_k(t)$: 三阶显式多项式建模大运动
- **Peano 余项部分** $\mathcal{H}_k(t)$: MLP 编码 + LBS 插值建模高阶残差

完整变换 $\mathcal{T}_i(t) = f_k(t) + \mathcal{H}_k(t)$，无近似误差。

### 关键设计

#### 1. 三阶 Taylor 展开变换场

对位置运动、缩放一致性、旋转运动分别建立时间关于 Taylor 展开：

- **位置**: $\bm{p}_i(t) = \sum_{k=0}^{n} \frac{1}{k!} f_p^{(k)}(t_\tau)(t - t_\tau)^k$，其中 $f_p^{(k)}(t_\tau)$ 为可学习参数（Taylor 系数），$t_\tau$ 为时间中心
- **缩放**: $\bm{s}_i(t) = \sum_{k=0}^{m} \frac{1}{k!} f_s^{(k)}(t_\tau)(t - t_\tau)^k$
- **旋转**: $\bm{q}_i(t) = \sum_{k=0}^{l} \frac{1}{k!} f_q^{(k)}(t_\tau)(t - t_\tau)^k$（四元数表示）

三阶展开足以捕获加速、减速等大规模运动模式，同时保持计算效率。

#### 2. Peano 余项的变形场建模（GP-LP 架构）

将高斯点分为两类：
- **全局高斯基元 (GP)**: 通过最远点采样选取少量代表性骨架点，用 MLP 直接预测其时变偏移 $\Delta_{GP} = MLP(GP)$
- **局部高斯基元 (LP)**: 大量细节点，其余项通过 **Linear Blend Skinning (LBS)** 从邻近 GP 插值获得

LP 的位置偏移：$\Delta\mu_i^t = \sum_{j \in \mathcal{N}} w_{ij}(R_j^t(\mu_i - p_j) + p_j + \Delta d_j^t)$

权重 $w_{ij}$ 基于高斯核 RBF 计算（可学习半径参数 $r_j$），确保空间一致性（LP 与 GP 位置关系不变）和时间一致性（相邻点运动刚性约束）。

#### 3. 时间相关的不透明度建模

不透明度建模为时间中心 $\mu_i^\tau$ 周围的 RBF 衰减：$\sigma_i(t) = \sigma_i^s \cdot e^{-s_i^\tau |t - \mu_i^\tau|^2}$，其中 $\sigma_i^s$ 是静态不透明度，$s_i^\tau$ 是时间尺度因子。这使得高斯基元可以在特定时间窗口内"出现"或"消失"。

### 损失函数

沿用标准 3DGS 渲染损失（光度一致性损失），无额外特殊损失。关键在于模型本身的数学表达力使优化更容易收敛。

## 实验关键数据

### 主实验表

**N3DV 数据集 (4 scene avg)**:

| 方法 | Cook Spinach PSNR | Sear Steak PSNR | Flame Steak PSNR | Cut Roast Beef PSNR |
|------|-----------|-----------|------------|-------------|
| 4DGS | 28.12 | 29.07 | 25.04 | 29.71 |
| SWinGS | 31.96 | 32.21 | 32.18 | 31.84 |
| D3DGS | 20.53 | 25.02 | 23.02 | 22.35 |
| **Ours** | **32.59** | **33.12** | **33.34** | **33.06** |

**Technicolor 数据集**:

| 方法 | Birthday PSNR | Painter PSNR | Train PSNR | Fatma PSNR |
|------|----------|---------|---------|---------|
| D3DGS | 33.81 | 37.38 | - | 38.40 |
| STG | 33.87 | 37.30 | 33.36 | 37.28 |
| **Ours** | **34.72** | **38.37** | **35.30** | **38.91** |

Technicolor Birthday 上相比 4DGS 提升 **58.25%**。

### 消融实验表

| 配置 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| w/o Time-opacity | 31.17 | 0.952 | 0.096 |
| w/o Time-motion | 29.24 | 0.920 | 0.154 |
| w/o Time-rotation | 31.21 | 0.953 | 0.103 |
| w/o Time-scale | 31.40 | 0.953 | 0.097 |
| w/o Peano remainder | 31.51 | 0.935 | 0.103 |
| **Ours Full** | **33.03** | **0.970** | **0.052** |

### 关键发现

- 去掉 Time-motion 影响最大（PSNR 下降近 4 dB），说明位置运动建模是核心
- 去掉 Peano 余项导致 1.5 dB 下降，验证了构建完整 Taylor 级数（而非截断近似）的必要性
- 时间不透明度、旋转、缩放各贡献约 1.6-1.9 dB

## 亮点与洞察

1. **数学优雅性**: 将 Taylor 公式完整搬到动态高斯建模，三阶显式展开 + Peano 余项神经网络 = 无近似误差的完整级数，这个思路非常巧妙
2. **GP-LP 分层设计**: 仅对少量骨架点（GP）使用 MLP 预测，大量细节点（LP）通过 LBS 插值获得，既保证了效率又维持了时空一致性
3. **可解释性与灵活性兼具**: Taylor 系数直接对应物理含义（速度、加速度等），同时 Peano 余项提供任意复杂度的补偿能力
4. **显著的性能提升**: 在两个公开数据集上全面超越，尤其在处理复杂运动场景时优势明显

## 局限与展望

1. 当前仅在多视角设置下验证，**单目动态场景**重建能力未探索
2. GP 点数量的选择需要手动调节（最远点采样的数量），**自适应 GP 选取**机制值得探索
3. Taylor 展开阶数固定为 3，更高阶可能进一步提升复杂运动准确度，但计算开销会增加
4. 缺乏对**拓扑变化**（如物体出现/消失、形变大的情况）的专门处理

## 相关工作与启发

- **D3DGS** (Yang et al.): 隐式变形场方案的代表，本文的 Peano 余项部分与之互补
- **4DGS** (Wu et al.): 六平面分解 + 轻量 MLP 预测变形，在多视角处理上不如本文
- **SC-GS** (Huang et al.): 稀疏控制点引导编辑，与本文 GP-LP 设计思路类似
- **启发**: Taylor 展开 + 网络学习余项的范式可推广到任何需要建模连续时变函数的场景（如 4D 人体重建、动态 SLAM）

## 评分

⭐⭐⭐⭐ — 数学建模优雅、实验全面，SOTA 提升明显；原创性在于将完整 Taylor 级数引入 4D Gaussian 建模。

<!-- RELATED:START -->

## 相关论文

- [ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [IRIS: Inverse Rendering of Indoor Scenes from Low Dynamic Range Images](iris_inverse_rendering_of_indoor_scenes_from_low_dynamic_range_images.md)
- [DropoutGS: Dropping Out Gaussians for Better Sparse-view Rendering](dropoutgs_dropping_out_gaussians_for_better_sparse-view_rendering.md)
- [WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments](wildgs-slam_monocular_gaussian_splatting_slam_in_dynamic_environments.md)
- [BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel View Synthesis](../../ICCV2025/3d_vision/billboard_splatting_bbsplat_learnable_textured_primitives_fo.md)

<!-- RELATED:END -->
