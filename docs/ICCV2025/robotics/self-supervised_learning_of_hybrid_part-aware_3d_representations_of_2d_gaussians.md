---
title: >-
  [论文解读] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics
description: >-
  [ICCV 2025][机器人][部件感知重建] 提出 PartGS，一个自监督的部件感知3D重建框架，将2D Gaussian Splatting与超二次曲面混合耦合，通过参数共享和多种正则化实现同时高质量几何分解和纹理重建，在DTU、ShapeNet和真实场景上在重建精度上比SOTA提升75.9%…
tags:
  - "ICCV 2025"
  - "机器人"
  - "部件感知重建"
  - "2D高斯溅射"
  - "超二次曲面"
  - "自监督"
  - "形状分解"
---

# Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics

**会议**: ICCV 2025  
**arXiv**: [2408.10789](https://arxiv.org/abs/2408.10789)  
**代码**: [zhirui-gao/PartGS](https://zhirui-gao.github.io/PartGS)  
**领域**: 机器人  
**关键词**: 部件感知重建, 2D高斯溅射, 超二次曲面, 自监督, 形状分解

## 一句话总结

提出 PartGS，一个自监督的部件感知3D重建框架，将2D Gaussian Splatting与超二次曲面混合耦合，通过参数共享和多种正则化实现同时高质量几何分解和纹理重建，在DTU、ShapeNet和真实场景上在重建精度上比SOTA提升75.9%，PSNR提升16.13dB。

## 研究背景与动机

**部件感知3D重建** 旨在将物体/场景分解为有意义的结构化部件，而非低层表示（点云、网格等）。这与认知科学中人类将3D环境理解为有意义部件组合的观点一致。结构化几何分解增强了场景可解释性，并有助于物理仿真、编辑、内容生成等下游任务。

现有方法存在三个核心问题：

**依赖3D监督**：EMS、MonteBoxFinder等方法需要3D点云或体素输入，无法直接从多视图图像工作，限制了实际应用。

**几何与外观的矛盾**：PartNeRF用多个NeRF建模部件，但隐式场的复杂组合导致次优渲染质量和低效分解。DBW使用超二次曲面+UV纹理图进行分解，虽然分解合理，但几何和外观重建精度不足（无法捕获细节）。

**速度瓶颈**：PartNeRF约8小时/场景，GaussianBlock（并行工作）也需数小时。

**核心洞察**：超二次曲面擅长表示广泛的基本形状原语（球、立方体、圆柱等的连续参数化族），适合部件级分解；而2D Gaussian Splatting擅长高保真纹理和几何细节重建。将两者耦合——让Gaussian分布在超二次曲面表面、共享姿态参数——可以同时获得合理的部件分解和高质量渲染。

## 方法详解

### 整体框架

PartGS 采用 **两阶段优化策略**：

- **Block-level 阶段**：使用混合超二次曲面+Gaussian表示，将场景分解为基本形状块
- **Point-level 阶段**：解耦Gaussian与超二次曲面的约束，允许Gaussian自由偏移以精化几何

### 混合表示的参数化

场景 $\mathcal{S}$ 被分解为 $M$ 个混合块：$\mathcal{S} = \mathcal{B}_1 \cup \ldots \cup \mathcal{B}_M$

每个混合块 $\mathcal{B}_i$ 由**超二次曲面**和其表面上的**2D Gaussian**组成，参数包括：

**形状与尺度参数**：超二次曲面由两个形状参数 $\epsilon_1, \epsilon_2$ 和三个尺度参数 $s_1, s_2, s_3$ 控制，顶点坐标为：
$$\mathbf{v} = [s_1 \cos^{\epsilon_1}(\theta) \cos^{\epsilon_2}(\varphi); \; s_2 \sin^{\epsilon_1}(\theta); \; s_3 \cos^{\epsilon_1}(\theta) \sin^{\epsilon_2}(\varphi)]$$

**姿态参数**：旋转 $\mathbf{R}_i$ 和平移 $\mathbf{t}_i$，变换为：$\hat{\mathbf{v}}_i^j = \mathbf{R}_i \mathbf{v}_i^j + \mathbf{t}_i$

**关键耦合设计**：2D Gaussian的中心均匀采样在超二次曲面的三角面上。其旋转矩阵 $\mathrm{R}_v = [r_1, r_2, r_3]$ 和缩放 $\mathrm{S}_v$ 由面顶点位置确定（跟随 GaMeS），无需独立学习几何参数。$r_1$ 对齐面法线，$r_2$ 从质心指向 $v_1$，$r_3$ 通过Gram-Schmidt正交化得到。

**不透明度参数**：每个块有可学习的不透明度 $\tau_i$，训练中低于阈值的块被移除，实现自适应部件数量。

**纹理参数**：2D Gaussian的球谐系数控制视点相关的纹理。

### Block-level 分解：优化与正则化

仅用渲染损失会导致块定位不稳定，因此引入四个正则化项：

**渲染损失**（标准3DGS损失）：
$$\mathcal{L}_{\text{ren}} = (1 - \lambda) L_1 + \lambda L_{\text{D-SSIM}}$$

**覆盖损失**：确保混合块覆盖物体区域，且不延伸到边界外。基于超二次曲面的内外函数 $D_i(x) = \Psi_i(x) - 1$ 定义光线与块的交互关系：
$$\mathcal{L}_{\text{cov}} = \sum_{r \in \mathcal{R}} l_r L_{\text{cross}}(r) + (1 - l_r) L_{\text{non-cross}}(r)$$

**重叠损失**：通过蒙特卡洛方法惩罚同时处于多个块内部的采样点：
$$\mathcal{L}_{\text{over}} = \frac{1}{N} \sum_{x \in \Omega} \text{ReLU}(\sum_{i \in \mathcal{M}} \mathcal{O}_i^x - k)$$
其中软占据函数 $\mathcal{O}_i^x = \tau_i \cdot \text{sigmoid}(-D_i(x) / \gamma)$。

**简约损失**：惩罚块不透明度以促进使用最少数量的块：$\mathcal{L}_{\text{par}} = \frac{1}{M} \sum_{i} \sqrt{\tau_i}$

**不透明熵损失**：将块不透明度推向二值（0或1）：
$$\mathcal{L}_{\text{opa}} = \frac{1}{|\mathcal{R}|} \sum_{r} L_{ce}(\max_{i} \tau_i(x^r), l_r)$$

总损失为加权求和：$\mathcal{L} = \mathcal{L}_{\text{ren}} + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}} + \lambda_{\text{over}} \mathcal{L}_{\text{over}} + \lambda_{\text{par}} \mathcal{L}_{\text{par}} + \lambda_{\text{opa}} \mathcal{L}_{\text{opa}}$

**自适应块数量**：不透明度低于阈值 $t$ 的块被移除；使用DBSCAN聚类未被覆盖的初始点云，为每个聚类引入新块。

### Point-level 精化

解耦Gaussian与超二次曲面的约束，允许独立优化。添加进入约束防止一个块的Gaussian穿越到相邻块内部：

$$\mathcal{L}_{\text{enter}} = \frac{1}{N} \sum_{x \in \Omega} \sum_{m \in \mathcal{M} \setminus \{\delta\}} \text{ReLU}(-D_m(x))$$

## 实验

### 主实验一：DTU数据集定量对比

| 方法 | 输入 | 可渲染 | 部件感知 | 平均CD↓ | PSNR↑ | 时间↓ |
|------|------|--------|---------|---------|-------|-------|
| EMS | 3D GT | ✗ | ✓ | 4.65 | - | - |
| MBF | 3D GT | ✗ | ✓ | 2.50 | - | - |
| PartNeRF | Image | ✓ | ✓ | 8.54 | 17.97 | ~8h |
| DBW | Image | ✓ | ✓ | 4.76 | 16.44 | ~2h |
| **PartGS (Block)** | Image | ✓ | ✓ | **4.19** | **19.84** | **~30m** |
| **PartGS (Point)** | Image | ✓ | ✓ | **0.98** | **35.04** | **~40m** |
| 2DGS (非部件) | Image | ✓ | ✗ | 0.81 | 34.07 | ~10m |

PartGS Point-level 的CD=0.98逼近非部件方法2DGS（0.81），同时保持部件分解能力。相比DBW（SOTA部件方法），CD改进79%、PSNR提升18.6dB、速度快3倍。

### 主实验二：ShapeNet数据集对比

| 方法 | 输入 | Airplane CD | Table CD | Chair CD | Gun CD | 平均CD |
|------|------|-------------|----------|----------|--------|--------|
| EMS | 3D GT | 3.40 | 6.92 | 19.0 | 2.02 | - |
| PartGS (Block) | Image | - | - | - | - | 4.19 |
| PartGS (Point) | Image | - | - | - | - | 0.98 |

在ShapeNet上同样展现了显著的重建精度优势，且能处理不同类别的多样化形状。

### 消融实验

| 策略 | 影响 |
|------|------|
| 无覆盖损失 | 块不能完整覆盖物体，出现未覆盖区域 |
| 无重叠损失 | 块之间严重重叠，分解质量下降 |
| 无简约损失 | 使用过多冗余的块 |
| 无DBSCAN添加 | 复杂物体的新出现区域无法被覆盖 |
| 无进入约束（点级） | Gaussian穿越块边界，破坏分解连续性 |

## 亮点与洞察

1. **混合耦合表示的优雅设计**：Gaussian共享超二次曲面的姿态，使表示更紧凑高效（无需独立学习Gaussian几何）
2. **自监督实现部件分解**：无需3D标注或分割监督，仅通过多视图渲染损失+正则化约束自动发现物体部件
3. **两阶段的递进策略**：Block级保证分解质量、Point级保证重建精度，两者解耦又互补
4. 在保持部件感知能力的同时，重建质量逼近甚至超越非部件方法（PSNR 35.04 vs 2DGS 34.07）

## 局限性

1. 超二次曲面的表达能力有限：对高度不规则的形状（如树木、动物毛发），基本形状原语的假设可能不成立
2. 初始块数量 $M$ 是超参数，虽然有自适应机制但仍需经验设定
3. 主要在物体级别验证，更大规模的室内/室外场景分解尚未探索
4. 仅与2024年及之前的部件感知方法对比，缺少与最新并行工作的全面比较

## 相关工作

- **形状分解/抽象**：Blocks World、EMS（超二次曲面概率恢复）、MonteBoxFinder（立方体+MCTS）
- **图像驱动的结构化3D**：PartNeRF（椭球+NeRF）、DBW（超二次曲面+UV纹理）、ISCO
- **网格+Gaussian混合**：SuGaR、GaMeS（Gaussian绑定到三角面）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 5 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| 总评 | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Weakly-Supervised Learning of Dense Functional Correspondences](weakly-supervised_learning_of_dense_functional_correspondences.md)
- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[CVPR 2025\] Lift3D Foundation Policy: Lifting 2D Large-Scale Pretrained Models for Robust 3D Robotic Manipulation](../../CVPR2025/robotics/lift3d_policy_lifting_2d_foundation_models_for_robust_3d_robotic_manipulation.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)
- [\[ICCV 2025\] Bridging Domain Generalization to Multimodal Domain Generalization via Unified Representations](bridging_domain_generalization_to_multimodal_domain_generalization_via_unified_r.md)

</div>

<!-- RELATED:END -->
