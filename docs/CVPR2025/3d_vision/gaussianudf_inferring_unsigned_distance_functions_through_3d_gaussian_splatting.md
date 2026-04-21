---
title: >-
  [论文解读] GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][无符号距离函数] 本文提出 GaussianUDF，通过将 2D 高斯平面贴合到曲面上，利用自监督和梯度推断为近场和远场分别提供无符号距离监督，首次在 3DGS 框架内高效推断连续 UDF，实现高质量开放曲面重建。
tags:
  - CVPR 2025
  - 3D视觉
  - 无符号距离函数
  - 3D高斯溅射
  - 开放曲面重建
  - 2D高斯
  - 自监督
---

# GaussianUDF: Inferring Unsigned Distance Functions through 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.19458](https://arxiv.org/abs/2503.19458)  
**代码**: https://lisj575.github.io/GaussianUDF/ (有)  
**领域**: 3D视觉  
**关键词**: 无符号距离函数, 3D高斯溅射, 开放曲面重建, 2D高斯, 自监督

## 一句话总结
本文提出 GaussianUDF，通过将 2D 高斯平面贴合到曲面上，利用自监督和梯度推断为近场和远场分别提供无符号距离监督，首次在 3DGS 框架内高效推断连续 UDF，实现高质量开放曲面重建。

## 研究背景与动机

**领域现状**：从多视角图像重建开放曲面（如衣服、树叶等非封闭物体）是数字化的重要任务。主流策略是通过体渲染学习无符号距离函数（UDF），然后从其零水平集提取曲面。UDF 相比 SDF 的优势在于能处理任意拓扑，不要求曲面封闭。

**现有痛点**：现有 UDF 方法（NeuralUDF、2S-UDF、VRPrior 等）都基于 NeRF 的体渲染，需要光线追踪寻找交点，训练效率低下（通常需要 7-9 小时）。3DGS 以其显式表示和高效光栅化成为加速重建的有力候选，但核心障碍是：3DGS 的离散显式表示（一堆高斯原语）与 UDF 的连续隐式表示之间存在本质鸿沟。

**核心矛盾**：UDF 在零水平集附近的梯度场极其复杂——梯度在曲面上是未定义的（不连续），这使得直接用梯度投影来学习 UDF 会导致优化不稳定。同时，高斯中心点稀疏且不均匀，仅靠点的 Chamfer Distance 无法提供足够的曲面信息。

**本文目标** 如何在 3DGS 框架内桥接离散高斯与连续 UDF，实现高效且准确的开放曲面重建。

**切入角度**：用 2D 高斯（薄平面）来近似曲面，然后利用整个高斯平面（而非仅中心点）来提供曲面附近的距离监督。远场用梯度投影拉向中心点，近场用高斯法线方向的采样点提供自监督。

**核心 idea**：用 2D 高斯贴合曲面 + 自监督近场 + 梯度推断远场，在 3DGS 内建立离散高斯到连续 UDF 的可微桥梁。

## 方法详解

### 整体框架
GaussianUDF 联合优化两个组件：一组 2D 高斯原语 $\{g_i\}_{i=1}^I$ 和一个 MLP 参数化的 UDF 网络 $f$。2D 高斯通过可微溅射渲染图像并最小化渲染误差来学习场景外观，同时通过约束将高斯贴合到 UDF 的零水平集上。基于贴合的高斯，近场和远场分别获得 UDF 的监督信号。输入为多视角 RGB 图像，输出为连续的 UDF 场，最终用 MeshUDF 算法提取开放曲面。

### 关键设计

1. **梯度推断的远场监督（Gradient-based Far-field Inference）**:

    - 功能：为距离曲面较远的空间区域提供粗粒度 UDF 监督
    - 核心思路：在高斯中心点附近随机采样查询点 $\{q_j\}$，用 Neural-Pull 的方式沿 UDF 梯度方向投影到零水平集得到投影点 $q_j' = q_j - f(q_j) \cdot \nabla f(q_j) / |\nabla f(q_j)|$，然后用投影点与高斯中心点之间的 Chamfer Distance $L_{far}$ 作为损失。这鼓励 UDF 的零水平集与高斯中心点一致
    - 设计动机：高斯中心点提供了曲面的稀疏采样，梯度投影可以将远处的查询点"拉"向曲面，但由于中心点稀疏不均匀，仅靠此损失会产生孔洞和噪声

2. **自监督的近场监督（Self-supervised Near-field Supervision）**:

    - 功能：利用 2D 高斯平面的完整结构，为曲面附近区域密集地提供 UDF 真值
    - 核心思路：利用 2D 高斯本身就是薄平面的特性。在每个高斯平面上采样根点 $\{r_i^h\}$，沿法线方向以随机偏移 $t_b$ 生成采样点 $e_{i,h}^b = r_i^h + t_b \cdot n_i / \|n_i\|$。关键洞察：如果高斯平面已贴合曲面，则偏移量 $|t_b|$ 就是采样点到曲面的无符号距离。因此用 $L_{near} = \|f(e_i^b) - |t_b|\|_1$ 作为自监督信号
    - 设计动机：远场监督只利用了高斯中心点（0 维），而自监督利用了整个高斯平面（2 维），能覆盖足够的曲面区域填补中心点之间的空隙。这是弥补高斯稀疏性的核心机制

3. **高斯到零水平集的投影约束（Gaussian Projection to Zero Level Set）**:

    - 功能：确保 2D 高斯中心精确位于 UDF 的零水平集上，使高斯真正贴合曲面
    - 核心思路：用 UDF 梯度将高斯中心投影到零水平集得到 $\mu_i' = \mu_i - f(\mu_i) \cdot \nabla f / |\nabla f|$，但在反向传播时**阻断通过 $f$ 的梯度**。然后用 $L_{proj} = \|\mu_i' - \mu_i\|_2$ 直接更新高斯位置。这与 GSPull 等方法的区别在于不通过 UDF 网络传梯度，避免了零水平集附近复杂梯度场导致的优化不稳定
    - 设计动机：UDF（不同于 SDF）在零水平集上梯度不连续且复杂，直接反传梯度会导致振荡。通过 stop-gradient 将投影目标视为固定锚点，稳定了优化过程

### 损失函数 / 训练策略
总损失：$L = (1-\lambda_1)L_{rgb} + \lambda_1 L_{ssim} + \lambda_2 L_{far} + \lambda_3 L_{near} + \lambda_4 L_{proj} + \lambda_5 L_{depth} + \lambda_6 L_{norm}$，其中 $L_{depth}$ 是深度蒸馏损失（约束同一射线上高斯交点的间距），$L_{norm}$ 是法线一致性损失（对齐高斯法线与深度图导出的法线）。训练 30k 迭代，$L_{far}$ 在第 9k-12k 迭代启用，之后加入 $L_{near}$ 和 $L_{proj}$。UDF 网络为 8 层 256 隐单元 MLP，带位置编码和 ReLU 激活，最后一层用绝对值确保非负。

## 实验关键数据

### 主实验
DF3D 数据集（开放曲面，12 件衣物）上的 Chamfer Distance（×10⁻³）：

| 方法 | 类型 | 平均 CD↓ | 训练时间 |
|------|------|---------|---------|
| NeuS | SDF | 4.36 | 5.7h |
| 2DGS | SDF | 3.81 | 6min |
| GOF | SDF | 2.49 | 47min |
| NeuralUDF | UDF | 2.15 | 8.6h |
| 2S-UDF | UDF | 1.98 | 7.8h |
| VRPrior | UDF | 1.71 | 9.2h |
| **GaussianUDF** | **UDF** | **1.60** | **1.6h** |

DTU 数据集（封闭曲面，15 个场景）上的平均 CD：

| 方法 | 平均 CD↓ |
|------|---------|
| 2DGS | 0.80 |
| GOF | 0.74 |
| GSPull | 0.75 |
| VRPrior | 0.71 |
| **GaussianUDF** | **0.68** |

### 消融实验

| 配置 | DTU CD↓ |
|------|---------|
| Only Far | 0.99 |
| Far + Near | 0.78 |
| Far + Proj | 0.88 |
| w/o Warp | 0.74 |
| w/o Near | 0.77 |
| w/o Proj | 0.76 |
| **Full Model** | **0.68** |

### 关键发现
- $L_{near}$（自监督近场损失）贡献最大：去掉后 CD 从 0.68 涨到 0.77，验证了用高斯平面提供密集监督的核心价值
- 仅用远场损失（Only Far）效果最差（0.99），因为高斯点云本身稀疏且有噪声，无法提供可靠的几何信息
- $L_{proj}$（投影约束）对精度的贡献（0.76→0.68）反映了将高斯精确贴合到零水平集的重要性
- 在开放曲面任务上，SDF 方法（NeuS、2DGS、GOF）因为试图构建封闭网格，误差是 UDF 方法的 1.5-2.7 倍
- 在封闭曲面（DTU）上，GaussianUDF 即使假设开放曲面仍优于专门设计的 SDF 方法

## 亮点与洞察
- **自监督机制的巧妙性**：利用 2D 高斯"本身就是薄平面"这个几何特性，沿法线采样即可得到距离真值，完全不需要额外标注。这种"表示即监督"的思路可推广到其他隐式场学习任务
- **Stop-gradient 的稳定化**：UDF 零水平集附近梯度场复杂是学界公认的难题，作者用 stop-gradient 将投影目标固定、转而直接移动高斯位置，用显式约束替代隐式梯度传播，简单但有效
- **点云变形验证**：用学到的 UDF 将任意形状的点云（苹果、甜甜圈）变形为目标衣物形状，直观证明了距离场的全局准确性，而非仅在零水平集附近正确

## 局限与展望
- 对无纹理结构的重建能力弱于 SDF 方法，UDF 的高灵活性反而增加了优化难度
- 从 UDF 提取网格仍然是开放问题（MeshUDF 等算法的质量限制了最终重建）
- 训练时间 1.6 小时虽然比 NeRF 方法快 5 倍，但比 2DGS（6 分钟）慢很多，主要瓶颈在 UDF MLP 和采样
- 缺少 mask 监督的实验，对于有前景/背景分离的场景可能有进一步提升空间

## 相关工作与启发
- **vs VRPrior**: VRPrior 需要额外的深度先验数据来学习 UDF，GaussianUDF 是纯自监督的，且精度更高（1.60 vs 1.71）
- **vs 2DGS/GOF**: 这些 SDF 方法学的是封闭曲面，对开放曲面会产生双层面问题。GaussianUDF 用 UDF 天然支持开放拓扑
- **vs GSPull**: GSPull 用类似的梯度投影学 SDF，但 SDF 的零水平集梯度更稳定。GaussianUDF 的 stop-gradient 策略是针对 UDF 梯度不稳定的专门设计
- 该工作表明 3DGS 与隐式场可以互补：高斯提供高效渲染和稀疏曲面采样，MLP 提供连续距离场

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在 3DGS 框架内实现 UDF 推断，自监督和 stop-gradient 机制设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 覆盖开放/封闭曲面、合成/真实数据，消融彻底，但缺少大规模场景测试
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，图示丰富，但公式略多
- 价值: ⭐⭐⭐⭐ 为开放曲面重建提供了高效新方案，桥接了显式和隐式表示

<!-- RELATED:START -->

## 相关论文

- [HybridGS: Decoupling Transients and Statics with 2D and 3D Gaussian Splatting](hybridgs_decoupling_transients_and_statics_with_2d_and_3d_gaussian_splatting.md)
- [ActiveGAMER: Active GAussian Mapping through Efficient Rendering](activegamer_active_gaussian_mapping_through_efficient_rendering.md)
- [SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)
- [PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)

<!-- RELATED:END -->
