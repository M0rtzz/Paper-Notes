---
title: >-
  [论文解读] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models
description: >-
  [ICML2025][医学图像][3D医学体积] 提出 Raptor（Random Planar Tensor Reduction），一种完全免训练的方法，利用冻结的 2D 基础模型（DINOv2-L）对 3D 医学体积沿三轴提取视觉 token，再通过随机投影大幅压缩维度，在 10 个医学任务上超越所有需要大规模预训练的 SOTA 方法。
tags:
  - ICML2025
  - 医学图像
  - 3D医学体积
  - 免训练嵌入
  - 随机投影
  - 基础模型
  - DINOv2
  - 降维
---

# Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models

**会议**: ICML2025  
**arXiv**: [2507.08254](https://arxiv.org/abs/2507.08254)  
**代码**: [github.com/sriramlab/raptor](https://github.com/sriramlab/raptor)  
**领域**: 3D医学影像  
**关键词**: 3D医学体积, 免训练嵌入, 随机投影, 基础模型, DINOv2, 降维

## 一句话总结

提出 Raptor（Random Planar Tensor Reduction），一种完全免训练的方法，利用冻结的 2D 基础模型（DINOv2-L）对 3D 医学体积沿三轴提取视觉 token，再通过随机投影大幅压缩维度，在 10 个医学任务上超越所有需要大规模预训练的 SOTA 方法。

## 研究背景与动机

3D 医学影像（MRI/CT）的基础模型面临两大瓶颈：

**计算复杂度**：将卷积或 Transformer 架构从 2D 扩展到 3D 时，计算开销呈立方或更高阶增长，训练成本极高（如 VoCo 使用 8×H100，SuPreM 使用 8×A100 训练 7 天以上）

**数据稀缺**：最大的 3D 医学数据集仅约 160K 体积，比 2D 图像数据集（1.2B 图像）小几个数量级

与此同时，2D 图像基础模型（如 DINOv2，1.2B 图像训练）已非常成熟。核心问题是：**能否不训练任何 3D 模型，直接复用 2D 基础模型的能力来处理 3D 体积数据？**

## 方法详解

### 整体流程

Raptor 的核心思想是三步：**三轴切片 → 2D 基础模型编码 → 随机投影压缩**，全程无需训练。

### Step 1: 三轴体积采样

对输入体积 $\mathbf{x} \in \mathbb{R}^{D \times D \times D}$，沿轴向（axial）、冠状（coronal）、矢状（sagittal）三个方向各取 $D$ 个切片，得到 $\mathbf{S} \in \mathbb{R}^{3 \times D \times (D \times D)}$，共 $3 \times D$ 个 2D 图像。

### Step 2: 2D 基础模型编码

使用冻结的 DINOv2-L（304M 参数，ViT 架构，patch size $T=16$）对每个切片提取 token：

$$\mathbf{z} = \text{concat}_{1 \leq i \leq 3}[\phi(\mathbf{S}_i)] \in \mathbb{R}^{3 \times D \times d \times p^2}$$

其中 $d=1024$ 为 token 维度，$p = D/T = 16$ 为每边 patch 数。对于 $D=256$，原始表示约 201M 个值（383MB），是原始体积的 127 倍。

### Step 3: 随机投影压缩

- **均值池化**：沿切片方向求均值，将 $3 \times D \times d \times p^2$ 压缩为 $3 \times d \times p^2$
- **随机投影**：采样 $\mathbf{R} \in \mathbb{R}^{K \times d}$，其中 $R_{kl} \sim \mathcal{N}(0,1)$，将 token 维度从 $d$ 压缩到 $K$
- **展平**：最终 Raptor 嵌入为

$$\mathbf{v} = \text{flatten}\left(\text{concat}_{1 \leq i \leq 3}\left[\mathbf{R} \frac{1}{D}\sum_{j=1}^{D}\mathbf{z}_{ij}\right]\right) \in \mathbb{R}^{3Kp^2}$$

典型设置下（$K=100, p=16$），嵌入大小为 $768 \times K = 76800$ 维，比原始体积小约 99%。精简版 Raptor-B 使用 $K=10$，再小 10 倍。

### 理论依据

随机投影的有效性由 **Johnson–Lindenstrauss 引理** 保证：高维点映射到 $\mathbb{R}^K$ 后，点对距离以 $(1 \pm \varepsilon)$ 的精度高概率保持。时间复杂度为 $\mathcal{O}(p^2 d N(D+K))$，优于 PCA 的 $\mathcal{O}(p^2 d^2 N)$。

## 实验关键数据

### 分类任务（3D MedMNIST, 6 数据集，AUROC/ACC）

| 方法 | Organ | Nodule | Fracture | Adrenal | Vessel | Synapse |
|------|-------|--------|----------|---------|--------|---------|
| SuPreM | 0.999/0.968 | 0.891/0.848 | 0.645/0.492 | 0.906/0.869 | 0.964/0.929 | 0.907/0.879 |
| VoCo | 0.992/0.870 | 0.797/0.836 | 0.699/0.535 | 0.913/0.872 | 0.799/0.880 | 0.844/0.830 |
| Merlin | 0.976/0.766 | 0.809/0.861 | 0.691/0.549 | 0.836/0.801 | 0.870/0.879 | 0.833/0.825 |
| **Raptor** | **0.999/0.961** | **0.929/0.870** | 0.677/0.502 | 0.926/0.845 | **0.966/0.922** | **0.943/0.911** |

### 额外分类数据集

| 方法 | CC-CCII (AUC) | CTRG-C (AUC) | CTRG-B (AUC) |
|------|---------------|--------------|--------------|
| SuPreM | 0.988 | 0.613 | 0.717 |
| **Raptor** | **0.997** | **0.620** | 0.711 |

### 回归任务（UK Biobank 脑 MRI，$r^2$，10 个脑区平均）

| 方法 | 平均 $r^2$ |
|------|-----------|
| Merlin | 0.313 |
| SuPreM | 0.299 |
| Raptor-B | 0.356 |
| **Raptor** | **0.389** |

Raptor 回归任务平均比 Merlin 提升 +24%，比 SuPreM 提升 +30%，比 SLIViT 提升 +47%。

### 数据效率

在 Synapse 数据集上，仅用 10 个样本即可达到全量（1230 样本）性能的 77%，100 个样本达到 88%。

### 嵌入效率

| 方法 | 参数量 | 嵌入大小 |
|------|--------|---------|
| VoCo | 294.9M | $3072 \times 3^3$ |
| Merlin | 124.7M | $2048 \times 14 \times 7^2$ |
| SuPreM | 5.1M | $128 \times 12^3$ |
| **Raptor** | 304.4M (DINOv2-L) | $3 \times 100 \times 16^2$ |
| **Raptor-B** | 同上 | $3 \times 10 \times 16^2$ (比 SuPreM 小 28.8×) |

## 亮点与洞察

1. **完全免训练**：无需在 3D 数据上训练任何模型，仅用冻结的 2D DINOv2 + 随机投影，大幅降低计算门槛（单张 RTX 2080 Ti 约 6.5s/体积）
2. **反直觉的强效果**：未经过医学领域训练的通用 2D 模型，通过三轴切片+压缩竟然超越了所有在医学数据上专门预训练的 3D 模型
3. **模型无关性**：Raptor 可无缝替换底层 2D 基础模型，随着 2D 模型进步自动获益
4. **极致压缩**：Raptor-B（$K=10$）嵌入仅 7680 维，比 SuPreM 小 28.8 倍，但性能相当甚至更优
5. **三轴互补**：消融实验证实三轴采样互补性强，单轴特征丢失可被其他轴恢复（类似三角定位）

## 局限与展望

1. **部分数据集表现一般**：Fracture3D 等数据集上效果不佳（AUC 0.677 vs VoCo 0.699），说明某些任务仍需领域特定先验
2. **空间分辨率受限**：模拟实验显示当特征小于 16px 时检测能力急剧下降（AUC 降至 ~0.5），受限于 ViT patch size
3. **均值池化损失信息**：沿切片方向取均值会丢失细粒度空间位置信息
4. **仅适用于体素数据**：不直接适用于点云、网格等其他 3D 表示
5. **下游仍需训练分类器**：虽然嵌入免训练，但下游任务仍需拟合逻辑回归或 MLP

## 相关工作与启发

- **SuPreM / VoCo / Merlin / MISFM**：代表了当前 3D 医学预训练的主流范式，均需大量计算和数据
- **DINOv2**：本文成功证明了通用 2D 基础模型在医学 3D 领域的迁移能力
- **Johnson–Lindenstrauss 引理**：为随机投影降维提供了严格的理论保证
- 启发：对于数据稀缺的高维模态（如 4D fMRI、视频），类似的"切片+2D编码+压缩"范式值得探索

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 免训练+随机投影处理 3D 体积是全新范式
- 实验充分度: ⭐⭐⭐⭐⭐ — 10 个数据集 + 6 个基线 + 完整消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论与实验兼顾
- 价值: ⭐⭐⭐⭐⭐ — 极大降低 3D 医学影像分析门槛，单 GPU 即可运行

<!-- RELATED:START -->

## 相关论文

- [Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [Foundation Models for Clinical Records at Health System Scale](foundation_models_for_clinical_records_at_health_system_scale.md)
- [Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](../../CVPR2025/medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](../../ICLR2026/medical_imaging/improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)
- [PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations](../../NeurIPS2025/medical_imaging/polypose_deformable_2d3d_registration_via_polyrigid_transformations.md)

<!-- RELATED:END -->
