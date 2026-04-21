---
title: >-
  [论文解读] VertexRegen: Mesh Generation with Continuous Level of Detail
description: >-
  [ICCV 2025][3D视觉][网格生成] 提出VertexRegen，受渐进网格启发将网格生成重新定义为边折叠(edge collapse)的逆操作——顶点分裂(vertex split)的学习，实现连续细节层级的"随时停止"网格生成。
tags:
  - ICCV 2025
  - 3D视觉
  - 网格生成
  - 渐进网格
  - 连续细节层级
  - 自回归
  - 顶点分裂
---

# VertexRegen: Mesh Generation with Continuous Level of Detail

**会议**: ICCV 2025  
**arXiv**: [2508.09062](https://arxiv.org/abs/2508.09062)  
**代码**: [项目页](https://vertexregen.github.io)  
**领域**: 3D视觉  
**关键词**: 网格生成, 渐进网格, 连续细节层级, 自回归, 顶点分裂

## 一句话总结

提出VertexRegen，受渐进网格启发将网格生成重新定义为边折叠(edge collapse)的逆操作——顶点分裂(vertex split)的学习，实现连续细节层级的"随时停止"网格生成。

## 研究背景与动机

现有自回归网格生成方法（MeshGPT, MeshXL等）的根本局限：
- **部分到完整**生成范式：中间步骤产生不完整网格（缺面）
- **缺乏细节控制**：必须完成整个序列才能得到有效网格
- 提前停止 = 缺失面片，而非低分辨率网格

VertexRegen的创新：**粗到细**生成范式，每个中间步骤都是有效网格，只是细节层级不同。

## 方法详解

### 渐进网格回顾

两个可逆操作：
- **边折叠** $\text{ecol}(v_s, v_t)$：合并两个相邻顶点，移除两个面和一个顶点
- **顶点分裂** $\text{vsplit}(v_s, v_l, v_r, v_t)$：逆操作，恢复顶点和两个面

渐进网格表示：$\text{PM}(\mathcal{M}) = (\mathcal{M}_0, \{\text{vsplit}_0, \cdots, \text{vsplit}_{n-1}\})$

边折叠顺序由QEM（二次误差度量）决定。

### VertexRegen序列化

完整序列：
```
M: [<bos>, [M_0 sequence], <sep>, [vsplit_0], ..., [vsplit_n-1], <eos>]
```

**粗网格 $\mathcal{M}_0$ 标记化**：沿用MeshXL方案（每面9个token），但仅编码极其粗糙的初始网格（平均仅18面 vs 原始457面）。

**顶点分裂标记化**：利用半边数据结构消除歧义，每次操作仅需记录 $v_s, v_l, v_r, v_t$ 四个顶点（12 tokens或边界情况10 tokens）。

### 半边数据结构的关键作用

顶点分裂需确定 $v_s$ 的邻居环中哪一半属于 $v_t$。通过半边遍历从 $\mathcal{H}_{ls}^1$ 到 $\mathcal{H}_{\cdot s}^r$，精确确定分属关系：

$$\{v_k | \mathcal{H}_{\cdot s}^k, k \neq r\} = N_{k+1}(v_s) - \{v_l, v_r, v_t\}$$

### 引导解码

维护状态机，实时执行顶点分裂。强制约束：
- $v_s$ 必须是当前网格中的顶点
- $(v_s, v_l)$ 和 $(v_s, v_r)$ 必须是有效边
- 仅允许 $v_l$ 或 $v_r$ 之一为 `<nil>`

## 实验

### 无条件网格生成

| 方法 | 标记化 | COV↑ | MMD↓ | 1-NNA(%) | JSD↓ |
|------|--------|------|------|----------|------|
| MeshXL | 展平坐标 | 51.76 | 8.30 | 50.84 | 3.81 |
| MeshAnything V2 | AMT | 50.33 | 8.50 | 52.25 | 4.84 |
| EdgeRunner | EdgeBreaker | 51.39 | 7.81 | 49.44 | 3.22 |
| **VertexRegen** | **渐进** | 51.03 | 8.29 | 50.22 | **2.89** |

VertexRegen生成质量与SOTA可比，同时独有连续细节层级能力。JSD最优，表明生成分布与参考分布更接近。

### 面数约束下的生成 (@400面)

| 方法 | COV↑ | MMD↓ | 1-NNA |
|------|------|------|-------|
| MeshXL (w/ FCC) | 41.20 | 10.03 | 59.06 |
| **VertexRegen** | **50.92** | **8.31** | **51.03** |

在低面数约束下VertexRegen大幅领先，因为其从粗到细的生成保持了完整结构。

### 压缩效率

| 方法 | 压缩比 |
|------|--------|
| MeshXL | 1.0 |
| MeshAnything V2 | 0.46 |
| EdgeRunner | 0.47 |
| VertexRegen (w/ HE) | 0.73 |
| VertexRegen (w/o HE) | 0.89 |

半边数据结构将序列长度减少22%。

## 亮点与洞察

1. **范式转变**：从"部分到完整"到"粗到细"，根本性地改变了网格生成的特性
2. **半边数据结构妙用**：消除顶点分裂的邻居归属歧义
3. **随时生成(Anytime Generation)**：任何时刻停止都得到有效网格
4. **继承渐进网格的理论保证**：每步顶点分裂都是确定性逆操作

## 局限性

- 粗网格 $\mathcal{M}_0$ 仍需MeshXL方式生成，占总长5.68%
- 压缩效率(0.73)不如AMT(0.46)，因为顶点分裂需4个顶点
- 预测无效顶点分裂会中断链式生成
- 需要引导解码维护状态机，增加推理复杂度

## 相关工作

- MeshGPT, MeshXL: 自回归网格生成
- Progressive Meshes (Hoppe): 渐进网格理论
- EdgeRunner: EdgeBreaker压缩标记化

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (渐进网格+生成模型的完美结合)
- 技术深度: ⭐⭐⭐⭐⭐ (半边数据结构+引导解码的精巧设计)
- 实验充分度: ⭐⭐⭐⭐ (无条件+条件+消融)
- 实用价值: ⭐⭐⭐⭐⭐ (连续LOD是实际刚需)

<!-- RELATED:START -->

## 相关论文

- [MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing](meshpad_interactive_sketch-conditioned_artist-reminiscent_mesh_generation_and_ed.md)
- [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)

<!-- RELATED:END -->
