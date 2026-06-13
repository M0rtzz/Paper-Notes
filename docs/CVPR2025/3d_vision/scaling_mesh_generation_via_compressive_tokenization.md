---
title: >-
  [论文解读] Scaling Mesh Generation via Compressive Tokenization
description: >-
  [CVPR 2025][3D视觉][网格生成] 本文提出 Blocked and Patchified Tokenization (BPT)，一种将三角网格序列压缩约 75% 的高效表征方法，使自回归 Transformer 首次能处理超过 8k 面的高精度网格，在点云/图像条件生成中达到产品级质量，并验证了网格面数对生成性能的正相关 scaling 规律。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "网格生成"
  - "压缩表征"
  - "Transformer"
  - "高面数网格"
  - "块索引"
---

# Scaling Mesh Generation via Compressive Tokenization

**会议**: CVPR 2025  
**arXiv**: [2411.07025](https://arxiv.org/abs/2411.07025)  
**代码**: [https://whaohan.github.io/bpt](https://whaohan.github.io/bpt)  
**领域**: 3D视觉  
**关键词**: 网格生成, 压缩表征, 自回归Transformer, 高面数网格, 块索引

## 一句话总结

本文提出 Blocked and Patchified Tokenization (BPT)，一种将三角网格序列压缩约 75% 的高效表征方法，使自回归 Transformer 首次能处理超过 8k 面的高精度网格，在点云/图像条件生成中达到产品级质量，并验证了网格面数对生成性能的正相关 scaling 规律。

## 研究背景与动机

**领域现状**：原生网格（native mesh）生成是 3D 内容创建的核心需求——相比 NeRF/3D Gaussian 等神经表征，网格具有明确的拓扑结构，可直接用于游戏、影视和仿真。近年来 MeshGPT、MeshXL 等工作通过自回归 Transformer 直接生成顶点和面的序列，保持了人工制作级的拓扑质量。

**现有痛点**：现有方法受限于网格序列过长。一个三角面由 3 个顶点、每个顶点 3 个坐标组成，即 9 个 token/面。MeshAnything 只能处理最多 800 面的网格，MeshAnythingV2 扩展到 1600 面。这些低面数网格严重缺乏细节，无法满足产品级需求。现有压缩方法（如 AMT、EdgeRunner）的压缩率不够，最优也只能达到约 47%。

**核心矛盾**：Transformer 的上下文窗口有限（通常 4k-9.6k tokens），而高面数网格（>4k 面）在现有表征下的序列长度远超上下文窗口。需要更极致的压缩来扩大可训练网格的面数范围，从而利用更丰富的训练数据。

**本文目标**：设计一种压缩率达 75% 的网格 tokenization 方法，使模型能利用 8k+ 面的高质量网格训练，显著提升生成性能和鲁棒性。

**切入角度**：作者从两个层面压缩——顶点层面用分块索引替代笛卡尔坐标将 3 个 token 压到 1-2 个，面层面用 patch 聚合消除共享顶点的重复。两者正交叠加实现约 75% 压缩。

**核心 idea**：将顶点从 (x,y,z) 三坐标表示转为 (block_id, offset_id) 二元索引表示 + 将相邻面聚合成以高度数顶点为中心的 patch，从而在顶点和面两个层级同时压缩序列长度。

## 方法详解

### 整体框架

BPT 将三角网格 $\mathcal{M}$ 转化为压缩的 1D token 序列，供标准自回归 Transformer 建模。输入网格先按 z-y-x 排序，然后：(1) 每个顶点的 3D 坐标转为 block-wise 索引，相同 block 内的连续顶点共享 block index，得到约 50% 压缩；(2) 识别具有最多未访问面的顶点作为 patch 中心，将其连接的所有面聚合为一个 patch，消除中心顶点的重复出现，再叠加约 50% 压缩。最终压缩率约 75%。生成时，Transformer 以点云或图像为条件，通过 cross-attention 注入条件信息，自回归地生成 BPT 序列。

### 关键设计

1. **Block-wise Indexing（分块索引）**:

    - 功能：将 3D 坐标 $(x,y,z)$（需 3 个 token）压缩为 $(b_i, o_i)$（最多 2 个 token）
    - 核心思路：将量化后的坐标空间沿每个轴等分为 $B$ 段，每段长度 $O$。block index $b_i = (x_i \mid O) \cdot B^2 + (y_i \mid O) \cdot B + z_i \mid O$ 标记顶点所在块，offset index $o_i = (x_i \% O) \cdot O^2 + (y_i \% O) \cdot O + z_i \% O$ 标记块内偏移。由于顶点按 z-y-x 排序，相邻顶点大概率在同一 block 内，所有连续同 block 的顶点可共享一个 block index，进一步压缩。词汇表大小为 $B^3 + O^3$（如 $B=8, O=16$ 时为 512+4096=4608），远小于朴素索引的 $128^3 \approx 200$ 万
    - 设计动机：朴素将 $(x,y,z)$ 映射为单一索引的词汇表大小 $r^3$ 不可承受；分块思路将指数级词汇表拆为两个多项式级子词汇表，同时利用排序后的空间局部性进一步消除 block index 冗余

2. **Patchified Aggregation（面片聚合）**:

    - 功能：将共享顶点的相邻面聚合为 patch，消除顶点重复出现
    - 核心思路：类比图像生成中的 patch 概念。算法流程：找到第一个未访问的面，选择其中连接最多未访问面的顶点作为 patch 中心 $v_c$，将 $v_c$ 连接的所有面聚合为 patch $P_c = (v_c, v_1, v_2, ..., v_n)$，其中 $v_c$ 只需出现一次而非重复出现于每个面中（原本 $v_c$ 平均需出现 6 次）。标记已访问的面后继续。使用双重 block 词汇表——patch 中心的 block index 和普通顶点的 block index 用不同词汇表，通过词汇表类型隐式标记 patch 的起始位置，无需额外特殊 token
    - 设计动机：原始表征中每个顶点出现次数等于其连接的面数（平均 6 次），造成大量冗余。patch 聚合将中心顶点从 ~6 次降到 1 次，其他顶点也从面数降到 patch 数。这不仅压缩序列，还增强了空间局部性——同一 patch 内的顶点空间相邻，降低 Transformer 的长程依赖需求

3. **条件网格生成架构**:

    - 功能：支持点云和图像两种条件的网格生成
    - 核心思路：使用 24 层、hidden size 1024 的标准自回归 Transformer，通过 cross-attention 注入条件。点云条件：用 Michelangelo 风格预训练编码器提取点云特征，训练时随机采样 4096 点。图像条件：先用 DINO 提取图像特征，通过额外的 DiT 扩散模型生成点云条件特征（桥接图像和点云空间），再喂给点云条件生成模型。训练分两阶段：先在 1.5M 大规模数据上训练，再在 0.3M 高质量数据上微调
    - 设计动机：点云是与网格最自然的配对模态——都是几何表示。图像条件通过扩散模型桥接，避免直接从 2D 到 3D 的巨大 gap。两阶段训练平衡了泛化性和质量

### 损失函数 / 训练策略

标准的自回归交叉熵损失 $L(\theta) = \prod_i p(p_i | p_{1:i-1}, c; \theta)$。训练使用 AdamW 优化器（$\beta_1=0.9, \beta_2=0.99$），学习率 $10^{-4}$，在 4 台 8×L40 机器上训练约 7 天。采样温度 0.7，上下文窗口 9600 tokens。使用 flash attention 和 bf16 混合精度加速。

## 实验关键数据

### 主实验

**点云条件生成**：

| 方法 | Hausdorff Distance↓ | Chamfer Distance↓ |
|------|---------------------|-------------------|
| MeshAnything | 0.301 | 0.136 |
| MeshAnythingV2 | 0.265 | 0.114 |
| **BPT (Ours)** | **0.166** | **0.094** |

### 消融实验

**Block/Offset 大小选择**（$|B| \cdot |O| = 128$）：

| (|B|, |O|) | Hausdorff↓ | Chamfer↓ |
|------------|-----------|---------|
| (4, 32) | 0.209 | 0.111 |
| **(8, 16)** | **0.166** | **0.094** |
| (16, 8) | 0.256 | 0.126 |

**Scaling 面数**的效果：

| 最大面数 | Hausdorff↓ | 趋势 |
|---------|-----------|------|
| 1600 面 | ~0.30 | 基线 |
| 3200 面 | ~0.23 | 提升明显 |
| 4800 面 | ~0.19 | 继续提升 |
| 8000 面 | ~0.166 | 最优 |

**压缩率对比**：

| 方法 | 压缩率↓ |
|------|--------|
| MeshXL / MeshAnything | 1.00 |
| MeshGPT / PivotMesh | 0.67 |
| MeshAnythingV2 / EdgeRunner | 0.46-0.47 |
| **BPT** | **0.26** |

### 关键发现

- BPT 实现了 SOTA 的 26% 压缩率（即压缩 74%），比第二好的方法再压缩约 44%
- 面数从 1600 scaling 到 8000 带来了持续的性能提升（Hausdorff 从 ~0.30 降到 0.166），验证了"面数越多、数据越丰富、生成越好"的 scaling 规律
- 截断训练 + 滑动窗口推理的工程 trick 并不能替代真正的长序列训练——截断会降低生成完整性和鲁棒性
- BPT 生成的网格在 AVD (Average Vertex Distance) 指标上在所有上下文长度下都最优，证明其空间局部性最强

## 亮点与洞察

- BPT 的设计哲学是"从两个正交维度同时压缩"——顶点级（坐标→索引）和面级（面→patch），两者独立贡献约 50% 的压缩率并可叠加
- 双重 block 词汇表隐式编码 patch 边界的设计巧妙——不需要特殊 token 就能区分 patch 起始，零额外开销
- 验证了网格生成领域的"scaling law"：面数（数据复杂度）是当前性能瓶颈，而非模型大小。一旦突破表征瓶颈，现有模型就能显著提升
- 图像→点云特征→网格的两阶段桥接策略有效解决了 2D-3D 模态 gap

## 局限与展望

- 当前模型仅 500M 参数，作者认为通过增大模型规模可进一步提升性能
- 固定的 7-bit 量化分辨率（128 级）限制了极精细几何的表达能力
- 仅验证了三角网格，对四边形/混合网格的泛化性未知
- 未来方向：(1) 更大模型 + 更多数据的 scaling 实验；(2) 探索其他序列建模架构（如 Mamba）来更好利用网格的归纳偏置；(3) 支持纹理/材质的联合生成

## 相关工作与启发

- 与 MeshGPT 的关系：MeshGPT 用自编码器将面转为潜空间 token，BPT 直接在原始坐标空间压缩，更简洁且无信息损失
- 与 MeshAnythingV2 (AMT) 的对比：AMT 压缩率约 46%，BPT 进一步到 26%；更重要的是 BPT 的空间局部性更强，生成的网格更完整
- 启发：分块索引的思路可推广到其他需要在有限词汇表下表示高维离散空间的场景（如体素生成、分子结构生成）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — block-wise indexing + patch aggregation 的组合简洁有效，虽非全新概念但在网格领域首次系统性应用
- **实验充分度**: ⭐⭐⭐⭐⭐ — 压缩率对比、scaling 实验、截断对比、block size 消融、点云/图像条件生成全面验证
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富直观
- **价值**: ⭐⭐⭐⭐⭐ — 解决了网格生成的核心瓶颈问题，使产品级质量的网格生成成为可能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing](treemeshgpt_artistic_mesh_generation_with_autoregressive_tree_sequencing.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](../../ICCV2025/3d_vision/meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[ICML 2025\] FreeMesh: Boosting Mesh Generation with Coordinates Merging](../../ICML2025/3d_vision/freemesh_boosting_mesh_generation_with_coordinates_merging.md)
- [\[CVPR 2025\] DAGSM: Disentangled Avatar Generation with GS-enhanced Mesh](dagsm_disentangled_avatar_generation_with_gs-enhanced_mesh.md)
- [\[CVPR 2025\] MVGenMaster: Scaling Multi-View Generation from Any Image via 3D Priors Enhanced Diffusion Model](mvgenmaster_scaling_multi-view_generation_from_any_image_via_3d_priors_enhanced_.md)

</div>

<!-- RELATED:END -->
