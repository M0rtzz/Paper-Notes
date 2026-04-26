---
title: >-
  [论文解读] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling
description: >-
  [ICCV 2025][人体理解][3D形变模型] imHead 提出首个大规模隐式 3D 头部形变模型，通过全局-局部解耦架构在 4,000 个身份的数据集上训练，实现了紧凑的隐式表示与局部面部编辑的兼顾，在重建精度和编辑灵活性上超越现有方法。
tags:
  - ICCV 2025
  - 人体理解
  - 3D形变模型
  - 隐式函数
  - 头部建模
  - 局部编辑
  - 大规模数据集
---

# ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling

**会议**: ICCV 2025  
**arXiv**: [2510.10793](https://arxiv.org/abs/2510.10793)  
**代码**: [项目主页](https://rolpotamias.github.io/imHead/)  
**领域**: 人体理解  
**关键词**: 3D形变模型, 隐式函数, 头部建模, 局部编辑, 大规模数据集

## 一句话总结

imHead 提出首个大规模隐式 3D 头部形变模型，通过全局-局部解耦架构在 4,000 个身份的数据集上训练，实现了紧凑的隐式表示与局部面部编辑的兼顾，在重建精度和编辑灵活性上超越现有方法。

## 研究背景与动机

3D 形变模型（3DMM）是人脸建模的核心技术，广泛应用于游戏、图形学和虚拟现实。但传统基于 PCA 的 3DMM 存在两个根本性限制：

**线性模型的表达力不足**：PCA 模型无法捕捉人脸的复杂局部变化，生成的表面过于光滑，缺少高频细节（如发丝、皱纹）。虽然非线性方法（如图神经网络）有所改进，但仍无法达到真实感要求。

**拓扑一致性约束**：3DMM 要求数据集中所有扫描具备一致的拓扑结构和精确的稠密对应关系，建立这种对应关系极为耗时且容易出错，限制了 3DMM 仅能建模面部区域，难以扩展到完整头部。

深度隐式函数（DIF）通过神经网络估计空间点到表面的符号距离，提供了连续的、无需拓扑约束的 3D 表示。现有的隐式 3DMM（如 NPHM）虽然能建模完整头部，但面临两个问题：

- **全局纠缠的隐式空间**：NPHM 将隐式空间分割为局部组件并附加全局身份编码，但全局信息"烘焙"在局部网络中，导致无法进行局部编辑。
- **小规模数据集**：现有方法仅在 <300 个身份上训练，身份多样性严重不足，无法捕捉真实世界的分布。

imHead 的核心洞察是：**保留单一紧凑的全局身份空间，通过中间层的区域特定表示来实现局部编辑**，而非直接分割隐式空间。

## 方法详解

### 整体框架

imHead 是一个 auto-decoder 风格的隐式模型 $\mathcal{M}: (\mathbf{x}, \mathbf{z}_{id}, \mathbf{z}_{exp}) \mapsto y \in \mathbb{R}$，由三个核心模块组成：

1. **Decomposition Network（DecNet）**：将全局身份编码分解为局部区域嵌入
2. **Structure Blending Fusion Network（FusionNet）**：聚合局部特征并预测全局隐式场
3. **Expression Warping Module**：学习观测空间到规范空间的映射以建模表情变形

### 关键设计

1. **数据集构建（4,000 身份）**

   利用 MimicMe 数据集的原始扫描（5,000 人、20 种表情），通过以下流程构建大规模完整头部数据集：

   - 多视角渲染 + RetinaFace 检测 2D 关键点 → 三角化提取 3D landmarks
   - ICP 刚性配准到 FLAME 规范空间 → 拟合优化得到软对应
   - NPHM 模型拟合填充完整头部
   - 非刚性 ICP（NICP）注册恢复身份细节

   最终过滤后保留 4,000 个身份（~50,000 次扫描），是之前隐式头部数据集的 **10 倍**。人口统计覆盖 57% 男性 / 43% 女性，年龄 1-81 岁，多种族。

2. **全局-局部解耦的身份网络**

   **DecNet $\mathcal{T}_\theta$**：通过简单的线性投影层将全局身份编码 $\mathbf{z}_{id} \in \mathbb{R}^{256}$ 映射为 $K=39$ 个局部嵌入 $\{\mathbf{z}_{id}^j \in \mathbb{R}^{32}\}$。这种设计的精妙之处在于：全局隐式空间保持紧凑（仅 256 维，比 NPHM 的 2176 维小 8.5 倍），同时通过中间表示实现局部可编辑性。

   **Local-Part Networks $\{g_j\}_{j=0}^K$**：$K$ 个独立的局部网络，每个接收查询坐标 $\mathbf{x}$ 和局部嵌入 $\mathbf{z}_{id}^j$，提取高维特征 $\mathbf{f}_x^j = g_j(\mathbf{x} - \mathbf{k}_j, \mathbf{z}_{id}^j)$。其中 $\mathbf{k}_j$ 为对应区域的关键点（由 LandmarkNet 回归），作为局部坐标系的原点。对称区域共享网络参数。使用位置编码 $\gamma(\mathbf{x} - \mathbf{k}_j)$ 捕获高频细节。

   **FusionNet $\mathcal{F}_\theta$**：基于距离的加权聚合局部特征：$\hat{\mathbf{f}}_x = \sum_j^K w(\mathbf{x}, \mathbf{k}_j) \mathbf{f}_x^j$，权重 $w(\mathbf{x}, \mathbf{k}_j) = \frac{e^{-\|\mathbf{x}-\mathbf{k}_j\|_2/\sigma}}{\sum_j^K e^{-\|\mathbf{x}-\mathbf{k}_j\|_2/\sigma}}$，最后由融合网络回归 SDF 值 $y = \mathcal{F}_\theta(\mathbf{x}, \hat{\mathbf{f}}_x)$。

   关键区别于 NPHM：imHead 不直接混合局部神经场（会导致编辑时的不连续性），而是通过融合中间特征来引导全局隐式场，确保编辑过程的平滑性。

3. **反向表情变形（Backward Expression Warping）**

   与 NPHM 的前向变形不同，imHead 采用反向变形：$\Delta \mathbf{x} = \mathcal{E}(\mathbf{x}_{obs}, \mathbf{z}_{id}, \mathbf{z}_{exp})$，$\mathbf{x}_{can} = \mathbf{x}_{obs} + \Delta \mathbf{x}$。

   优势：前向变形需要迭代式 root-finding 来建立软对应，对初始化敏感且计算开销大。反向变形直接将观测点映射到规范空间，拟合过程更平滑，速度提升 **3 倍**（40s vs 138s）。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{eik} + \lambda_{kpt}\mathcal{L}_{kpt} + \lambda_{sym}\mathcal{L}_{sym} + \lambda_{reg}\mathcal{L}_{reg}$$

- $\mathcal{L}_{rec}$：表面点 SDF 趋零 + 梯度匹配法向量
- $\mathcal{L}_{eik}$：Eikonal 正则化（梯度单位范数）
- $\mathcal{L}_{kpt}$：关键点回归损失
- $\mathcal{L}_{sym}$：对称区域的隐式编码对称约束
- $\mathcal{L}_{reg}$：身份和表情编码的正则化

## 实验关键数据

### 主实验

**身份重建（中性表情，单扫描）**：

| 方法 | NPHM CD↓ | NPHM NC↑ | NPHM F@5mm↑ | MimicMe CD↓ | MimicMe NC↑ | MimicMe F@5mm↑ |
|------|----------|----------|-------------|-------------|-------------|----------------|
| FLAME | 1.244 | 0.943 | 0.632 | 1.336 | 0.929 | 0.606 |
| NPHM† | 0.514 | 0.980 | 0.866 | 0.598 | 0.967 | 0.827 |
| monoNPHM† | 0.514 | 0.980 | 0.866 | 0.593 | 0.968 | 0.829 |
| **imHead-Full†** | **0.459** | **0.988** | **0.898** | **0.533** | **0.986** | **0.873** |

imHead 在两个数据集上全面超越所有基线，同时隐式空间仅 256 维（NPHM 2176 维压缩 8.5 倍）。

**表情重建**：imHead-Full 在 NPHM 数据集上 CD=0.485, F@5mm=0.912，在 MimicMe 上 CD=0.563, F@5mm=0.878，同样领先。拟合速度 40s vs NPHM 的 138s。

### 消融实验

| 配置 | NPHM CD↓ | NPHM NC↑ | NPHM F@5mm↑ | MimicMe CD↓ | 说明 |
|------|----------|----------|-------------|-------------|------|
| w. Local Lat. (d=312) | 0.876 | 0.915 | 0.689 | 0.874 | 纯局部隐式空间，严重不足 |
| w. Local Lat. (d=1248) | 0.775 | 0.948 | 0.743 | 0.767 | 大局部空间，仍逊于全局 |
| w. Local+Global (d=1344) | 0.494 | 0.964 | 0.841 | 0.569 | NPHM 式设计，性能接近但空间大 5× |
| w/o FusionNet | 0.595 | 0.954 | 0.808 | 0.674 | 直接回归局部 SDF，法线不光滑 |
| w/o Local Canonical | 0.723 | 0.934 | 0.723 | 0.884 | 缺失局部坐标系，性能大幅下降 |
| **imHead-Full** | **0.459** | **0.988** | **0.898** | **0.533** | 完整模型 |

### 关键发现

- 全局隐式空间比局部空间能更好地捕捉数据分布中的全局模式，即使局部空间维度高达 1248 维也不如 256 维的全局空间。
- FusionNet 的中间特征融合对于避免编辑时的不连续性至关重要。
- 大规模数据集的引入（imHead-MimicMe vs imHead-NPHM）在 MimicMe 测试集上将 CD 从 0.571 降到 0.546（20% 提升），验证了数据规模的重要性。
- imHead 对含噪输入鲁棒，1.5 倍标准差的高斯噪声下仍能保持身份特征的合理重建。

## 亮点与洞察

- **全局/局部的优雅平衡**：通过 DecNet 将紧凑全局空间分解为局部中间表示，兼顾了压缩性能和编辑灵活性，设计极为精巧。
- **局部编辑的自然涌现**：模型训练仅优化重建，但天然支持面部区域独立编辑和区域互换（如交换两个身份的鼻子、头发等），无需任何额外约束。
- **规范空间的对应关系保持**：即使在极端表情（如张嘴）下，反向变形仍能保持面部的拓扑一致性，类似传统 3DMM 的优势。

## 局限与展望

- 隐式模型推理速度慢于显式 3DMM，需要大量点采样和 marching cubes 后处理。
- 难以精确建模头发丝等高频薄面结构。
- 局部编辑受固定锚点数量限制，边界处可能受相邻局部网络影响。
- 数据集存在种族偏差（73% 白人），尤其是头发区域多样性不足。
- 没有 1-1 的稠密对应映射（不像显式模型）。

## 相关工作与启发

- SPAGHETTI 的中间 part-level 表示思想对本文有直接启发，但本文将其应用于人脸这一更具挑战性的领域。
- 反向变形的设计选择巧妙避开了前向变形的 root-finding 问题，实际上简化了整个拟合流程。
- 大规模数据集的构建流程（raw scan → FLAME 配准 → NPHM 补全 → NICP 细节恢复）具有重要的参考价值。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 全局-局部解耦架构设计新颖，反向变形简化流程
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖重建/生成/编辑/对应关系等多维度，消融详尽
- **写作质量**: ⭐⭐⭐⭐ 技术细节清晰，公式推导完整
- **价值**: ⭐⭐⭐⭐⭐ 大规模隐式头部模型 + 局部编辑能力，对 3D 人脸/头部领域有基础性贡献

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[CVPR 2025\] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars](../../CVPR2025/human_understanding/rgbavatar_reduced_gaussian_blendshapes_for_online_modeling_of_head_avatars.md)
- [\[CVPR 2026\] LCA: Large-scale Codec Avatars - The Unreasonable Effectiveness of Large-scale Avatar Pretraining](../../CVPR2026/human_understanding/lca_large-scale_codec_avatars_the_unreasonable_effectiveness_of_large-scale_avata.md)
- [\[NeurIPS 2025\] GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](../../NeurIPS2025/human_understanding/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [\[ECCV 2024\] PetFace: A Large-Scale Dataset and Benchmark for Animal Identification](../../ECCV2024/human_understanding/petface_a_large-scale_dataset_and_benchmark_for_animal_identification.md)

<!-- RELATED:END -->
