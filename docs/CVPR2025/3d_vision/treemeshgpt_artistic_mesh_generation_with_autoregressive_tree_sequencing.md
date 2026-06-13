---
title: >-
  [论文解读] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing
description: >-
  [CVPR 2025][3D视觉][网格生成] 提出 TreeMeshGPT，通过基于三角形邻接关系的动态树结构遍历来序列化网格，实现每面仅需 2 个 token 的高效表示（压缩率约 22%），将艺术网格生成能力扩展到 5500 面，同时显著减少法线翻转问题。 3D 生成后通常使用 Marching Cubes 提取网格…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "网格生成"
  - "Transformer"
  - "树结构序列化"
  - "点云条件"
  - "艺术网格"
---

# TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing

**会议**: CVPR 2025  
**arXiv**: [2503.11629](https://arxiv.org/abs/2503.11629)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 网格生成, 自回归Transformer, 树结构序列化, 点云条件, 艺术网格

## 一句话总结

提出 TreeMeshGPT，通过基于三角形邻接关系的动态树结构遍历来序列化网格，实现每面仅需 2 个 token 的高效表示（压缩率约 22%），将艺术网格生成能力扩展到 5500 面，同时显著减少法线翻转问题。

## 研究背景与动机

3D 生成后通常使用 Marching Cubes 提取网格，但产出的网格过度密集、线框杂乱，不适合游戏和VR等需要实时渲染的应用。艺术家手工创建的网格紧凑、结构规整且语义对齐，但过程极其耗时。

自回归网格生成方法逐渐兴起：MeshGPT 开创了面排序+VQ-VAE+Transformer 的范式，但 MeshAnything 每面需 9 个 token，限于 800 面。MeshAnythingV2 和 EdgeRunner 利用三角邻接缩短序列，扩展到 1600 和 4000 面。但真实应用需要更高面数来准确表示复杂表面，且现有方法存在间隙、缺失和法线翻转等伪影。

**核心问题**: 如何进一步压缩 token 序列、降低训练难度、提高生成质量，同时解决法线一致性问题？从"下一个 token 预测"到"从动态树中检索下一个 token"的范式转换。

## 方法详解

### 整体框架

TreeMeshGPT 使用 24 层 Transformer 解码器（1024 维隐藏层，16 头注意力）。输入为点云条件（8192 点，通过交叉注意力编码为 2048 个 latent code $\mathbf{Z}$）拼接自回归序列。序列中每步输入为一条有向网格边 $(v_1^n, v_2^n)$，输出为对面顶点 $v_3^n$ 或 [STOP] 标签。使用半边数据结构和 DFS 遍历构建输入-输出序列对。7-bit 量化，支持最多 5500 面。

### 关键设计1: 自回归树序列化 (Autoregressive Tree Sequencing)

**功能**: 以每面仅 2 个 token 的效率将三角网格序列化为 Transformer 可处理的序列。

**核心思路**: 将三角网格转换为等价的树结构，其中每个节点表示一条有向边 $(v_i, v_j)$。使用动态栈管理 DFS 遍历：(1) 从最低位置的边及其孪生边初始化栈；(2) 每步从栈顶弹出一条边作为输入 $I_n = (v_1^n, v_2^n)$；(3) 若存在对面顶点 $v_3^n$，输出该顶点，并将两条新边 $(v_3^n, v_2^n)$ 和 $(v_1^n, v_3^n)$ 按逆时针方向压栈；(4) 若为边界或已访问，输出 [STOP]。

**设计动机**: 传统 next-token 预测将每步输出作为下步输入，序列顺序与网格拓扑关系弱相关。树遍历确保每步生成都是局部扩展——从最后生成的三角面直接扩展，大幅降低训练难度。半边数据结构的逆时针方向约束天然保证法线一致性，从根本上避免法线翻转问题。

### 关键设计2: 层次化 MLP 头顶点预测

**功能**: 在单个序列步内预测完整的 3D 坐标。

**核心思路**: 与先前方法每个坐标作为独立 token 不同，本文使用层次化 MLP 头在一步内预测量化的 $x$, $y$, $z$ 坐标。先预测 $x$，其嵌入参与预测 $y$，$y$ 的嵌入再参与预测 $z$，保持坐标间的顺序依赖。损失为三轴交叉熵之和：$\mathcal{L} = \mathcal{L}_{CE}(\mathbf{O}_x, \hat{\mathbf{O}}_x) + \mathcal{L}_{CE}(\mathbf{O}_y, \hat{\mathbf{O}}_y) + \mathcal{L}_{CE}(\mathbf{O}_z, \hat{\mathbf{O}}_z)$。

**设计动机**: 将 3D 坐标分成三个独立 token 使序列长度增加 3 倍。层次化预测在保持坐标间依赖的同时，仅需一个序列步。消融实验显示层次化比同时预测三坐标更容易采样。

### 关键设计3: 推理时调整策略

**功能**: 提高生成质量和稳定性。

**核心思路**: (1) 重复面检测——跟踪已生成面，若预测顶点会形成重复三角形则自动替换为 [STOP]；(2) 自适应 EOS 增强——在长序列中 [EOS] 难以被采样（虽在 top-5 logits 中），对 [EOS] logit 施加递增因子，仅当成为最高 logit 时才选择；(3) 动态温度——栈长度 <10 时温度 1.0，<30 时 0.4，否则 0.2。

**设计动机**: 自回归生成中累积误差导致长序列质量下降。重复面检测避免拓扑错误；自适应 EOS 解决模型不知何时停止的问题；动态温度在生成初期鼓励多样性、后期收紧分布保证质量。

### 损失函数

三轴坐标独立的交叉熵损失之和。[STOP] 和 [EOS] 标签作为 z 轴额外类别。使用 teacher-forcing 训练。

## 实验关键数据

### 定量结果 (Objaverse 验证集, 200 样本)

| 模型 | CD↓ | NC↑ | |NC|↑ |
|------|------|------|-------|
| MeshAnything | 0.0115 | 0.223 | 0.853 |
| MeshAnythingV2 | 0.0102 | 0.167 | 0.843 |
| **TreeMeshGPT** | **更优** | **更优** | **更优** |

### 面数容量对比

| 方法 | 最大面数 | 每面 token 数 |
|------|---------|-------------|
| MeshAnything | 800 | 9 |
| MeshAnythingV2 | 1,600 | ~4.5 |
| EdgeRunner | 4,000 | ~4.5 |
| **TreeMeshGPT** | **5,500** | **~2** |

### 关键发现

1. **面数容量大幅提升**: 从 800→5500 面（相比 MeshAnything 提升 6.9x），从 4000→5500（相比 EdgeRunner 提升 37.5%）。
2. **法线一致性显著改善**: NC 指标大幅优于 MeshAnything（0.223）和 MeshAnythingV2（0.167），因为半边结构天然强制法线方向一致。
3. **压缩率约 22%**: 每面约 2 token vs 朴素方法 9 token，压缩率是 MeshAnythingV2 和 EdgeRunner 的约两倍。
4. **泛化到真实扫描**: 在 GSO 数据集上也展示了良好的定性结果。
5. **局部预测降低难度**: 动态栈管理使 Transformer 每步仅需做局部预测，训练更容易收敛。

## 亮点与洞察

- **范式突破**: 从"预测下一个 token"到"从动态树中检索下一个 token"，改变了自回归网格生成的基本模式。
- **数据结构为先**: 利用半边数据结构的拓扑/方向约束天然解决法线翻转问题，比后处理更优雅。
- **DFS 的局部性**: DFS 遍历保证每步生成与最近生成的面相邻，而非网格远处的面，减少了"长距离跳跃"的训练难度。

## 局限与展望

- **5500 面仍有限**: 对于极复杂模型可能不够，但已覆盖大量实用场景。
- **流形假设**: 训练网格必须是流形且无翻转法线，限制了训练数据量。
- **DFS 顺序唯一性**: 同一网格可能有多种 DFS 遍历顺序，当前选择最低位置起始可能不是全局最优。
- 未来可探索 BFS 遍历、更大搜索空间、与 3D 生成流水线的更紧密集成。

## 相关工作与启发

- **MeshGPT/MeshAnything**: 开创面排序+自回归生成范式，TreeMeshGPT 通过树遍历将压缩率提升一倍。
- **EdgeRunner**: 利用三角邻接优化序列，但仍使用标准 next-token 预测。
- **启发**: 在自回归生成中，数据结构（如半边、树遍历）的选择可能比网络架构的改进更关键。

## 评分

⭐⭐⭐⭐ — 树序列化设计精妙，将面数容量从 800 推至 5500 的同时解决法线翻转问题。数据结构驱动的创新比单纯架构改进更有深度。22% 压缩率是领域内的新 benchmark。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](scaling_mesh_generation_via_compressive_tokenization.md)
- [\[CVPR 2025\] StageDesigner: Artistic Stage Generation for Scenography via Theater Scripts](stagedesigner_artistic_stage_generation_for_scenography_via_theater_scripts.md)
- [\[ICLR 2026\] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](../../ICLR2026/3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[ICML 2025\] FreeMesh: Boosting Mesh Generation with Coordinates Merging](../../ICML2025/3d_vision/freemesh_boosting_mesh_generation_with_coordinates_merging.md)

</div>

<!-- RELATED:END -->
