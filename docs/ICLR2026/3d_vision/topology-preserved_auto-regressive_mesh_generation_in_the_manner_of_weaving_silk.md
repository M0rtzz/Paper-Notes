---
title: >-
  [论文解读] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk
description: >-
  [3D视觉] 提出一种类似"织丝"的网格 tokenization 算法，通过顶点分层和排序提供规范的拓扑框架，保证生成网格的流形性、水密性、法线一致性和部件感知性，同时达到 SOTA 压缩效率。
tags:
  - 3D视觉
---

# Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk

- **会议**: ICLR 2026
- **arXiv**: [2507.02477](https://arxiv.org/abs/2507.02477)
- **代码**: [项目页面](https://gaochao-s.github.io/pages/MeshSilksong/)
- **领域**: 3D 视觉 / Mesh 生成
- **关键词**: Mesh Generation, Auto-regressive, Topology Preservation, Mesh Tokenization, Manifold, Watertight

## 一句话总结

提出一种类似"织丝"的网格 tokenization 算法，通过顶点分层和排序提供规范的拓扑框架，保证生成网格的流形性、水密性、法线一致性和部件感知性，同时达到 SOTA 压缩效率。

## 研究背景与动机

自回归网格生成是近期3D内容生成的重要方向，但现有方法（MeshAnything v2, EdgeRunner, BPT, DeepMesh）将网格视为简单的三角形集合，缺乏对整体拓扑结构的感知，导致：

**无法保证水密性**：生成存在孔洞，影响 3D 打印、物理模拟等应用

**无法保证部件感知**：缺乏连通分量概念，无法捕捉小但重要的部件（如眼睛）

**法线方向不一致**：面法线翻转导致渲染伪影

**非流形拓扑**：局部补丁方法（BPT, DeepMesh）无法保证流形

**核心贡献**：首个同时保证流形性、水密性检测、法线一致性和部件感知的网格 tokenization 算法。

## 方法详解

### 整体框架

三个顺序步骤：
1. **顶点分层和排序** → 2. **层间邻接矩阵压缩** → 3. **Token 打包**

### 1. 顶点分层和排序

对每个连通分量：
- 通过 y-z-x 坐标排序确定起始半边 $j$-$m$
- 用 BFS 根据到起点 $j$ 的最短图距离确定层索引 $L$
- 每层顶点通过半边遍历进行排序（类似数学归纳证明）
- 顶点标记为 $\mathcal{V}_i^L$，$L$ 为层索引，$i$ 为层内顺序

### 2. 层邻接矩阵压缩

顶点连接分为两类：

**自层矩阵 $\mathcal{S}_L$**（蓝边）：同层顶点间的连接
- 对称 0-1 矩阵，大小 $M \times M$
- 使用固定窗口大小 $W=8$ 的二进制压缩
- 覆盖 99.1% 的情况，极端情况用 COO 格式

**层间矩阵 $\mathcal{B}_L$**（红边）：相邻层间的连接
- 0-1 矩阵，大小 $M \times N$
- 使用 RLE-like 压缩：连续 "1" 编码为 $(x, y)$（起始列索引, 长度）
- 进一步升级为等价 "Stars and Bars" 问题，每行仅需一个 token

**Token 打包**：每个顶点 $\mathcal{V}_i^L$ 产生 4 个 token：
- 2 个位置 token $V_{(L,i)}$（使用 block-offset 表示压缩）
- 1 个自层拓扑 token $S_{(L,i)}$
- 1 个层间拓扑 token $B_{(L,i)}$

### 3. 几何性质保证

**(1) 流形拓扑**：生成过程中边连接仅存在于同层或相邻层，三角形逐层填充，严格保证流形拓扑。

**(2) 水密性检测/修复**：孔洞必然由自层/层间矩阵中的 0 被错误预测引起，可直接检测和纠正。

**(3) 法线一致性**：定义层 $L$ 半边方向为升序、层 $L-1$ 为降序，保证每个三角面的顶点逆时针遍历，从而法线一致。

**(4) 部件感知**：通过特殊 token $C$ 标记连通分量起始，支持小部件生成。

### 非流形处理算法

对非流形边（被 3 个以上面共享）：
- 不同于 Libigl 的度优先合并策略
- 额外检测非流形顶点周围的"边图"结构 $\mathcal{G}$
- 确保边图形成纯环以维护表面完整性

### 训练策略

**渐进式平衡采样**：

$$p_j^{PB}(t) = (1 - t/T) p_j^{IB} + (t/T) p_j^{CB}$$

早期偏向实例平衡采样（学简单），后期偏向类别平衡采样（学复杂），每 100 面为一类。

### 损失函数

标准交叉熵损失：

$$\mathcal{L}_{ce} = -\sum_{t=1}^{T-1} S_{t+1} \log \hat{S}_t$$

## 实验

### 数据集和评估指标

- **训练**：gObjaverse (~280k) + ShapeNetV2 + 3D-FUTURE + Toys4K (~100k)，无手动筛选
- **评估**：500 个 gObjaverse 保留样本
- **指标**：CD (Chamfer Distance), HD (Hausdorff Distance), NC (Normal Consistency), |NC|, Bits-per-face, Compression Ratio

### 主要结果

| 方法 | CD↓ | HD↓ | NC↑ | |NC|↑ | Bits/face↓ | Comp. Ratio↓ |
|------|-----|-----|-----|-------|------------|--------------|
| EdgeRunner* | 0.053 | 0.144 | 0.418 | 0.789 | 29.61 | 0.47 |
| TreeMeshGPT | 0.030 | 0.103 | 0.706 | 0.892 | 42.00 | 0.22 |
| BPT | 0.027 | 0.094 | 0.770 | 0.909 | 28.48 | 0.26 |
| **Ours** | **0.025** | **0.087** | **0.792** | **0.924** | **26.65** | **0.22** |

### 消融实验

| 消融 | CD↓ | HD↓ | NC↑ | |NC|↑ |
|------|-----|-----|-----|-------|
| w/ 重采样 | 0.025 | 0.087 | 0.792 | 0.924 |
| w/o 重采样 | 0.032 | 0.103 | 0.700 | 0.880 |
| 流形+非流形数据 | 0.022 | 0.080 | 0.801 | 0.932 |
| 仅流形数据 | 0.027 | 0.090 | 0.688 | 0.871 |

### 几何性质对比

| 方法 | 无损 | 流形 | 水密 | 法线一致 | 部件感知 |
|------|------|------|------|----------|----------|
| Ours | ✓ | ✓ | ✓ | ✓ | ✓ |
| MeshAnything v2 | ✓ | ✗ | ✗ | ✗ | ✗ |
| EdgeRunner | ✓ | ✓ | ✗ | ✗ | ✗ |
| BPT | ✓ | ✗ | ✗ | ✗ | ✗ |

## 亮点

1. **织丝的优雅类比**：层间编织的思想自然保证了拓扑性质，设计巧妙
2. **唯一同时保证 5 项几何性质**的 tokenization 方法
3. **SOTA 压缩效率**：26.65 Bits/face, 0.22 压缩比
4. **实用性强**：支持 3D 打印、物理模拟、动画绑定等下游应用
5. **在线非流形处理**：有效扩展可训练数据规模

## 局限性

1. 词表大小较高（最多 10,267），但在 bits-per-face 上仍最优
2. 需要预定义最大层顶点数 $m$，限制了极端复杂网格
3. 目前仅支持三角形网格，混合多边形支持是未来方向
4. 模型规模约 500M 参数，训练需 16×H800 约 15 天

## 相关工作

- **VQ-VAE 方法**：MeshGPT (Siddiqui et al., 2024) 用编码器-解码器进行 mesh tokenization
- **直接量化**：MeshXL, BPT 直接量化三角形顶点坐标
- **树遍历**：EdgeRunner, TreeMeshGPT 通过树结构保证流形但压缩效率低
- **强化学习**：DeepMesh 通过人类反馈微调提升美观度

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 层间编织的思想原创且优雅
- **实用性**: ⭐⭐⭐⭐⭐ — 几何性质保证直接服务于工业应用
- **清晰度**: ⭐⭐⭐⭐ — 算法描述清晰，图示直观
- **意义**: ⭐⭐⭐⭐⭐ — 解决了自回归网格生成中的关键拓扑问题

<!-- RELATED:START -->

## 相关论文

- [MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](../../ICCV2025/3d_vision/meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)
- [UrbanGS: A Scalable and Efficient Architecture for Geometrically Accurate Large-Scene Reconstruction](urbangs_a_scalable_and_efficient_architecture_for_geometrically_accurate_large-s.md)
- [Universal Beta Splatting](universal_beta_splatting.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)

<!-- RELATED:END -->
