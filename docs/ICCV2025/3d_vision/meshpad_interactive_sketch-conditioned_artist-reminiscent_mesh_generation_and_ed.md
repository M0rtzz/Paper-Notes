---
title: >-
  [论文解读] MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing
description: >-
  [3D视觉] MeshPad 将草图驱动的 3D 网格创建与编辑解耦为"添加"和"删除"两个子任务，基于三角序列表示和 Transformer 自回归生成，并提出顶点对齐推测解码器实现 2.2× 加速，让交互式网格编辑在几秒内完成。
tags:
  - 3D视觉
---

# MeshPad: Interactive Sketch-Conditioned Artist-Reminiscent Mesh Generation and Editing

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.01425](https://arxiv.org/abs/2503.01425)
- **代码**: [项目页面](https://derkleineli.github.io/meshpad/)
- **领域**: 3D视觉
- **关键词**: 三角网格生成, 草图条件编辑, 自回归Transformer, 推测解码加速, 交互式3D建模

## 一句话总结
MeshPad 将草图驱动的 3D 网格创建与编辑解耦为"添加"和"删除"两个子任务，基于三角序列表示和 Transformer 自回归生成，并提出顶点对齐推测解码器实现 2.2× 加速，让交互式网格编辑在几秒内完成。

## 研究背景与动机

三角网格是 3D 制作中最主流的表示形式，但现有的网格生成方法存在几个关键不足：

**不可编辑**：MeshGPT、MeshAnything 等方法只能生成完整形状，不支持局部编辑

**迭代创作需求**：艺术内容创作是反复修改的过程，需要快速交互式编辑能力

**编辑粒度粗**：基于 SDS 或隐式表示（SDF/NeRF）的 3D 编辑方法修改非目标区域，且无法输出 artist-reminiscent 的网格

**推理速度慢**：自回归网格生成逐 token 预测，难以达到交互速度

MeshPad 的核心洞察：通过将编辑分解为添加和删除，可以（a）自动从已有 3D 数据生成训练数据而无需真实编辑序列，（b）仅修改目标区域同时保留未编辑部分。

## 方法详解

### 整体框架

MeshPad 的输入是草图图像 $\mathcal{I}$（含黑色保留区域 $\mathcal{I}_k$ 和红色编辑区域 $\mathcal{I}_r$），输出是对应的 3D 三角网格。框架由三个核心组件构成：

1. **草图条件网格删除网络**：分类网络，标记每个顶点是否应被删除
2. **草图条件网格添加网络**：自回归生成网络，生成新的网格三角形序列
3. **顶点对齐推测解码器**：加速自回归生成

### 网格表示与编辑分解

网格 $\mathcal{M}$ 由三角形集合组成，每个三角形 $\mathcal{F} = \{v_1, v_2, v_3\}$。编辑操作将网格分为互斥的两部分：
- $\mathcal{M}_k$：保留部分（对应黑色草图线条）
- $\mathcal{M}_r$：编辑部分（对应红色草图线条）

**删除操作**：对网格序列中每个顶点预测二值标签，删除标记为红色区域的顶点对应的三角形：

$$\mathcal{M}_r' = \{\mathcal{F} \in \mathcal{M} | \exists v \in \mathcal{F}: v \in \mathcal{V}_r'\}; \quad \mathcal{M}_k' = \mathcal{M} \setminus \mathcal{M}_r'$$

**添加操作**：以现有网格 $\mathcal{M}_k$ 和新草图为条件，自回归生成新三角形序列 $\mathcal{M}_r'$，并与 $\mathcal{M}_k$ 合并。

### 网络架构

- **骨干网络**：基于 OPT（Open Pre-trained Transformer），使用 MeshAnythingV2 的预训练权重初始化
- **图像编码器**：冻结的 RADIO 2.5-h 编码草图输入
- **Token化**：采用 MeshAnythingV2 的 tokenizer，将网格表示为控制 token（`<split>`, `<start>`, `<end>`）和顶点坐标 token 的有序序列
- **添加网络**：因果注意力 + 自回归生成，仅生成 $\mathcal{M}_r'$ 的 token 序列
- **删除网络**：双向注意力 + 分类头，对每个顶点输出删除/保留标签

自回归生成的核心公式：

$$P(S_r'^{(i+1)} | S_k, \mathcal{I}, S_r'^{(1...i)}) = \text{OPT}(S_k, \mathcal{I}, S_r'^{(1...i)})$$

### 顶点对齐推测解码器

关键创新：利用网格序列的结构化特性——每个顶点恰好由 3 个 token ($V_x, V_y, V_z$) 组成。推测器接收 $V_x$ 对应的 Transformer 隐藏状态，一次预测 $V_y$ 和 $V_z$：

$$P(V'_{\{y,z\}}) = \text{Speculator}(E_x, V'_x)$$

关键设计选择：
- **联合训练**：推测器与 OPT 联合训练（而非冻结 OPT 后独立训练），使隐藏状态包含足够上下文信息
- **顶点对齐**：仅预测 $V_y, V_z$，降低预测头的复杂度
- **OPT 损失仅监督** $V_x$ 和控制 token

### 训练数据生成

利用自监督方式从 ShapeNet 3D 数据自动生成训练对，无需真实编辑序列：

1. 从完整网格 $\mathcal{M}_c$ 中随机采样两个体积 $\mathcal{L}_a$ 和 $\mathcal{L}_b$
2. 切割得到 $\mathcal{M}_k$ 和 $\mathcal{M}_r$
3. 通过 Canny 边缘检测在法线和深度渲染图上生成合成草图
4. 随机数据增强弥合合成草图与手绘草图的域差

### 损失函数

- **添加网络**：交叉熵损失，监督 token 序列预测 + 推测器的独立交叉熵损失
- **删除网络**：二值交叉熵损失，监督每个顶点的删除/保留标签

## 实验关键数据

### 主实验：草图条件网格生成

| 数据集 | 方法 | CD↓ | LPIPS↓ | CLIP↑ | FID↓ |
|--------|------|-----|--------|-------|------|
| ShapeNet | LAS | 15.02 | 0.2742 | 94.61 | 46.90 |
| ShapeNet | LAS-MA | 22.06 | 0.2963 | 93.63 | 18.52 |
| ShapeNet | SENS | 8.95 | 0.2753 | 93.36 | 81.88 |
| ShapeNet | SENS-MA | 29.43 | 0.3348 | 91.88 | 42.93 |
| ShapeNet | **MeshPad** | **6.20** | **0.1790** | **95.85** | **9.38** |
| IKEA | LAS | 16.27 | 0.2970 | 94.83 | 69.54 |
| IKEA | **MeshPad** | **6.78** | **0.1837** | **96.67** | **29.67** |

MeshPad 在 CD 上比最优基线 SENS 提升 **22%+**，在 FID 上提升极为显著。

### 消融实验：推测解码器设计

| 方法 | CD↓ | LPIPS↓ | CLIP↑ | FID↓ | T/s↑ |
|------|-----|--------|-------|------|------|
| w/o speculator | 7.66 | 0.1765 | 95.59 | 32.59 | 60.7 |
| w/o vert-alignment | 9.00 | 0.1992 | 94.43 | 35.65 | 138.9 |
| w/o joint training | 57.13 | 0.5134 | 84.52 | 211.46 | 130.8 |
| **MeshPad (full)** | **6.78** | **0.1837** | **96.67** | **29.67** | **131.1** |

### 关键发现

1. **联合训练至关重要**：冻结 OPT 训练推测器导致性能崩溃（CD 从 6.78 暴涨到 57.13）
2. **顶点对齐保持质量**：与普通 MLP 推测器相比，顶点对齐在相当速度下保持更好的生成质量
3. **加速 2.16×**：推测解码使生成速度从 60.7 T/s 提升到 131.1 T/s
4. **用户偏好碾压**：90%+ 参与者在二元对比中偏好 MeshPad 的结果
5. **交互速度**：编辑操作（约 1/5 完整形状）仅需 1-5 秒

## 亮点与洞察

1. **编辑分解范式**：将复杂的网格编辑拆解为添加+删除，既简化了问题又实现了自动训练数据生成
2. **部分生成**：添加网络只生成编辑部分的 token，天然保留未编辑区域——这是相比 SDS-based 方法的核心优势
3. **推测解码创新**：利用网格序列的 xyz 三元组结构信息，比通用 NLP 推测解码更高效
4. **Sketch UI**：基于 Gradio + Three.js 的浏览器端交互界面，降低 3D 建模门槛
5. **新形状生成能力**：通过迭代编辑可以创建训练集中未见的新形状类别

## 局限性

1. **上下文长度限制**：受 Transformer 模型限制，生成的网格面数有限（<768 面），远低于现代 CG 应用的百万级需求
2. **仅训练于 ShapeNet**：55 个类别的覆盖有限，对开放世界物体的泛化能力尚需验证
3. **草图-网格对齐非精确**：不保证生成的网格与草图像素级对齐
4. **从头生成 vs 编辑**：从头生成复杂形状不如迭代编辑效果好

## 相关工作与启发

- **MeshGPT/MeshAnything 系列**：建立了自回归三角网格生成的基础，但不支持编辑
- **SENS/LAS**：草图到 SDF/Occupancy Grid 然后 Marching Cubes，产出非 artist-reminiscent 网格
- **推测解码**：从 NLP 领域引入，利用域特定结构（顶点三元组）进行了巧妙适配
- **启发**：将复杂任务分解为更小的可控子任务是一个通用且有效的策略

## 评分

⭐⭐⭐⭐ (4/5)

优势在于编辑分解思路优雅、推测解码器设计巧妙、用户研究全面；不足在于面数限制较大、数据集覆盖有限。整体是直接网格生成领域迈向交互式编辑的重要一步。

<!-- RELATED:START -->

## 相关论文

- [Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)

<!-- RELATED:END -->
