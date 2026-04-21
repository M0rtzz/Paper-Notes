---
title: >-
  [论文解读] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes
description: >-
  [CVPR 2026][3D视觉][4D高斯溅射] 提出 GP-4DGS，将变分高斯过程（GP）整合到 4D 高斯溅射中，通过时空组合核和变分推断实现概率化运动建模，同时赋予 4DGS 不确定性量化、运动外推和自适应运动先验三大新能力。
tags:
  - CVPR 2026
  - 3D视觉
  - 4D高斯溅射
  - 高斯过程
  - 不确定性量化
  - 运动外推
  - 动态场景重建
---

# GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes

**会议**: CVPR 2026  
**arXiv**: [2604.02915](https://arxiv.org/abs/2604.02915)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 4D高斯溅射、高斯过程、不确定性量化、运动外推、动态场景重建

## 一句话总结
提出 GP-4DGS，将变分高斯过程（GP）整合到 4D 高斯溅射中，通过时空组合核和变分推断实现概率化运动建模，同时赋予 4DGS 不确定性量化、运动外推和自适应运动先验三大新能力。

## 研究背景与动机

**领域现状**：4D 高斯溅射（4DGS）是动态新视角合成的主流方法，通过将 3D 高斯原始体随时间变形来建模动态场景。现有方法如 D-3DGS（MLP变形）、4DGS（HexPlane）、STG（多项式变形）已达到很好的视觉质量。

**现有痛点**：(1) 现有方法将运动视为确定性优化问题，施加手工设计的运动先验（多项式变形、刚性约束等），这些固定先验在所有原始体上统一应用，对观测不充分或被遮挡的区域不合适；(2) 缺乏对运动预测的不确定性估计机制；(3) 无法在训练帧之外进行时间外推。

**核心矛盾**：固定的确定性运动先验无法区分观测良好和观测稀疏的区域——对前者过度约束，对后者约束不足。需要一种能根据观测置信度自动调节正则化强度的机制。

**本文目标** (1) 如何为 4DGS 的运动预测提供原理性的不确定性量化？(2) 如何从观测良好的区域学习运动先验并传播到稀疏/未观测区域？(3) 如何实现训练帧之外的时间外推？

**切入角度**：高斯过程天然是概率函数族上的分布，其核函数定义了数据之间的相关结构。将 GP 用于建模变形场，可同时实现自适应先验、不确定性量化和外推——这些能力直接来自GP的概率公式化，无需额外建模。

**核心 idea**：用变分高斯过程替代确定性变形函数，通过时空组合核捕获几何和运动相关性，在GP后验均值引导4DGS优化的同时自然获得不确定性和外推能力。

## 方法详解

### 整体框架
输入是单目视频。每个高斯原始体的 4D 坐标 $\mathbf{x}=(\bm{p}, t)$（规范3D位置+时间）作为 GP 输入，输出 9 维变形向量（3维平移+6维连续旋转表示）。GP 通过变分推断（inducing points）实现计算可扩展性。采用 GP-GS 交替优化策略：(1) GP 训练阶段在高置信度数据上学习运动先验；(2) GS 优化阶段用 GP 预测作为正则化指导。

### 关键设计

1. **时空组合核设计**:

    - 功能：捕获空间几何平滑性和时间运动周期性两种截然不同的相关结构
    - 核心思路：将核分解为空间和时间两部分 $k_i(\mathbf{x},\mathbf{x}') = k_i^{\text{spatial}}(\bm{p},\bm{p}') + k_i^{\text{temporal}}(\mathbf{x},\mathbf{x}')$。空间部分使用各向异性 Matérn 核（而非 RBF），因为 Matérn 能处理不连续性，适合建模空间不连通的物体。时间部分将逐轴 Matérn 核与周期核 $k^{\text{periodic}}(t,t') = \sigma^2 \exp(-2\sin^2(\pi|t-t'|/\tau)/\ell^2)$ 相乘，捕获运动周期性同时保持空间局部性
    - 设计动机：标准 GP 核假设各向同性相关，与时空数据中空间和时间维度截然不同的相关结构根本不匹配。周期核为时间外推提供了强归纳偏置

2. **变分高斯过程与 Inducing Points**:

    - 功能：将 GP 推断从 $\mathcal{O}(N^3)$ 降到 $\mathcal{O}(NM^2+M^3)$，使得数万个高斯原始体的推断可行
    - 核心思路：引入 $M$ 个诱导点 $\mathbf{Z}=\{\mathbf{z}_m\}_{m=1}^M$（$M \ll N$），参数化变分后验 $q(\mathbf{u}_i)=\mathcal{N}(\mathbf{m}_i, \mathbf{S}_i)$。通过最大化 ELBO 联合优化核超参、诱导点位置和变分参数。推断时每个查询点复杂度仅 $\mathcal{O}(M)$。诱导点初始化使用 Chronos 提取时间序列特征再 k-means 聚类选择代表性规范位置，时间轴均匀采样
    - 设计动机：精确 GP 推断对于典型的数万个高斯原始体完全不可行。基于时间序列特征的初始化比随机或速度基线取得更高的 ELBO

3. **GP-GS 交替优化策略**:

    - 功能：形成 GP 和 4DGS 之间的协同反馈循环
    - 核心思路：Stage 1（GP训练）：通过累积渲染贡献 $C_k = \sum_{\mathbf{I}}\sum_{\mathbf{r}} \omega_{k,t}^{\pi}(\mathbf{r})$ 衡量每个原始体的置信度，选 $C_k > \tau_C$ 的高置信子集训练 GP，并在空间坐标注入噪声作为正则化。Stage 2（GS优化）：用 GP 后验均值 $\bar{\bm{\mu}}$ 作为伪指导信号，通过 $\mathcal{L}_{\text{GP}} = \frac{1}{NT}\sum_{k,t} \delta_{(k,t)} \|\mathbf{y}_{(k,t)} - \bar{\bm{\mu}}_{(k,t)}\|^2$ 正则化偏离 GP 预测的原始体，阈值 $\tau_\delta$ 在训练中逐步收紧。每 2000 步缓存一次 GP 预测
    - 设计动机：可靠观测逐步精炼运动先验，运动先验反过来稳定观测不足区域的重建——形成自强化循环

### 损失函数 / 训练策略
总损失 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{GP}}\mathcal{L}_{\text{GP}}$，$\lambda_{\text{GP}}=0.1$。重建损失沿用 SoM 的光度损失、D-SSIM、流损失和平滑损失。不确定性通过蒙特卡洛采样从 GP 后验获取（对旋转的非线性变换无法直接传播方差）。

## 实验关键数据

### 主实验（DyCheck）

| 方法 | mPSNR ↑ | mSSIM ↑ | mLPIPS ↓ |
|------|---------|---------|----------|
| D-3DGS | 11.92 | 0.49 | 0.66 |
| 4DGS | 13.42 | 0.49 | 0.56 |
| HyperNeRF | 15.99 | 0.59 | 0.51 |
| Gaussian Marbles | 15.84 | 0.54 | 0.57 |
| SoM | 17.09 | 0.65 | 0.39 |
| **GP-4DGS** | **17.38** | **0.65** | **0.37** |

### 运动外推实验（PSNR ↑）

| 方法 | 周期运动(5帧) | 周期运动(15帧) | 非周期运动(5帧) | 非周期运动(15帧) |
|------|-------------|-------------|--------------|--------------|
| Linear extrapolation | 11.55 | 8.11 | 15.02 | 11.92 |
| **GP-4DGS** | **17.62** | **16.65** | **15.27** | **13.22** |

### 不确定性量化（AUSE-MSE ↓ ×10⁻²）

| 方法 | Top 20 帧 | Top 40 帧 | 所有帧 |
|------|----------|----------|-------|
| Random | 9.76 | 9.30 | 10.98 |
| UA-4DGS | 7.60 | 8.11 | 8.62 |
| **GP-4DGS** | **7.22** | **8.00** | **8.49** |

### 关键发现
- 在 DyCheck 挑战子集上性能差距更大（mPSNR 14.56→15.02，mLPIPS 0.53→0.51），证明 GP 先验对稀疏观测区域最有价值
- 周期运动外推优势巨大（17.62 vs 11.55），因为周期核精确捕获了循环动力学
- GP 指导有效正则化了运动轨迹，消除噪声和波动，产生物理合理的运动模式
- Inducing point 时间序列初始化比随机和速度KNN初始化一致取得更高 ELBO（平均 1.53 vs 1.10 vs 1.37）
- 在 DAVIS 极端视角偏移下保持了更好的几何结构完整性

## 亮点与洞察
- 将概率建模引入 4DGS 的思路非常优雅：不确定性量化和运动外推是 GP 概率公式化的自然副产物，无需额外设计。这种"免费获得"额外能力的方式比专门设计模块更干净
- GP-GS 交替优化的设计精巧：置信度加权的数据选择确保 GP 从可靠数据学习，GP 预测反过来指导不可靠区域——像是一种"自举学习"
- 时空组合核的设计有物理直觉：空间 Matérn 允许不连续（不同物体）、时间周期核编码运动规律性，两者正交组合

## 局限与展望
- GP 推断仍引入额外计算开销（每2000步更新一次GP缓存），对实时应用可能是瓶颈
- 周期核假设运动具有周期性，对于非周期复杂运动外推仍然有限（非周期场景提升较小）
- 仅在 DyCheck（7个场景）和 DAVIS 上评估，场景多样性有限
- 变分 GP 的近似精度受 inducing points 数量限制，极高分辨率场景可能需要更多inducing points

## 相关工作与启发
- **vs SoM（STG）**: SoM 使用多项式变形作为固定先验，GP-4DGS 学习数据自适应先验，在挑战子集上优势明显（14.56→15.02）
- **vs Stochastic GS**: Stochastic GS 对高斯属性建模随机变量但仅限静态场景，本文首次将概率建模扩展到4D动态场景
- **vs UA-4DGS**: UA-4DGS 尝试动态场景不确定性估计，但 GP-4DGS 通过核相关结构提供更可靠的校准（AUSE 更低）
- **vs D-3DGS/4DGS**: 这些确定性方法在稀疏观测区域缺乏约束，GP 的空间相关先验有效填补了这一空白

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 GP 与 4DGS 结合是全新的方向，时空核设计和交替优化策略有深度
- 实验充分度: ⭐⭐⭐⭐ 重建质量+外推+不确定性+消融全面覆盖，但评估场景数量较少
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，方法动机清晰，图示精美
- 价值: ⭐⭐⭐⭐ 为4DGS引入概率建模开辟新方向，不确定性量化对自动驾驶等安全关键应用有实际价值

<!-- RELATED:START -->

## 相关论文

- [4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4.md)
- [RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous_time_4d_gaussian.md)
- [Sparse4DGS: 4D Gaussian Splatting for Sparse-Frame Dynamic Scene Reconstruction](../../AAAI2026/3d_vision/sparse4dgs_4d_gaussian_splatting_for_sparse-frame_dynamic_scene_reconstruction.md)
- [Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](../../ICLR2026/3d_vision/uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)

<!-- RELATED:END -->
