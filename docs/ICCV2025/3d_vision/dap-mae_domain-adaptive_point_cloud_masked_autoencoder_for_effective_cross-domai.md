---
title: >-
  [论文解读] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning
description: >-
  [ICCV 2025][3D视觉][masked autoencoder] 提出 DAP-MAE，通过异构域适配器（HDA）和域特征生成器（DFG）协同学习多域点云数据，仅需一次预训练即可适配物体分类、表情识别、部件分割和3D检测等多种下游任务。
tags:
  - ICCV 2025
  - 3D视觉
  - masked autoencoder
  - cross-domain learning
  - 点云
  - 域适应
  - 自监督学习
---

# DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning

**会议**: ICCV 2025  
**arXiv**: [2510.21635](https://arxiv.org/abs/2510.21635)  
**代码**: [https://github.com/CVI-SZU/DAP-MAE](https://github.com/CVI-SZU/DAP-MAE)  
**领域**: 3D视觉 / 点云分析 / 自监督学习  
**关键词**: masked autoencoder, cross-domain learning, 点云, 域适应, 自监督学习

## 一句话总结

提出 DAP-MAE，通过异构域适配器（HDA）和域特征生成器（DFG）协同学习多域点云数据，仅需一次预训练即可适配物体分类、表情识别、部件分割和3D检测等多种下游任务。

## 研究背景与动机

**点云数据稀缺**：相比2D数据，不同领域的点云数据规模极为有限（物体域、人脸域、场景域各有独立数据集），这限制了监督学习方法的性能。

**MAE 预训练的域限制**：
   - 现有点云 MAE（如 Point-MAE、3DFaceMAE、PiMAE）需要在与下游任务相同的域上分别预训练
   - 这导致预训练的冗余——每个域都要独立训练一次模型
   - 未能充分利用不同域的点云数据

**简单混合域数据的失败**：
   - 将来自不同域的点云直接混合进行 MAE 预训练，下游任务可能将跨域信息视为噪声干扰
   - 实验验证：ReCon-SMC 在 𝕆+𝔽+𝕊 混合预训练后，表情识别从 87.69% 降至 87.23%，物体检测从 42.7% 降至 42.5%

**核心问题**：如何有效利用多域点云数据进行联合预训练，使模型在各个域的下游任务上都能获益？

## 方法详解

### 整体框架

DAP-MAE 基于 Transformer 的 MAE 架构，引入两个关键组件：

- **异构域适配器（HDA）**：预训练时用适应模式分域处理，微调时用融合模式综合利用多域知识
- **域特征生成器（DFG）**：通过对比学习提取域特征，引导下游任务的特征适配

**预训练数据**：ShapeNet（物体域 𝕆）、FRGCv2（人脸域 𝔽）、S3DIS（场景域 𝕊），每个点云统一采样 4096 个点。

**预训练流程**：
1. FPS + KNN 将点云分割为 patch，随机划分可见和遮蔽部分
2. PointNet tokenize + HDA 适应模式处理
3. Transformer encoder 编码可见 token
4. Decoder 重建遮蔽 patch（Chamfer Distance 损失）
5. DFG 提取域特征（对比损失）

### 关键设计

#### 1. 异构域适配器（HDA）

HDA 包含三个并行 MLP（对应 𝕆、𝔽、𝕊 三个域），每个 MLP 由两层 FC + BN + ReLU 组成。

**适应模式（预训练）**：
- 根据输入点云所属的域 d，选择对应的 MLP_d 处理 token
- 不同域的数据通过不同 MLP 路径，使 encoder 能在共享参数下学习各域特有的几何信息
- 公式：$\mathcal{T}_{vis} = \text{MLP}_d(\mathcal{T}'_{vis})$

**融合模式（微调）**：
- 冻结三个 MLP 的参数，新增两个 MLP 生成融合系数
- 三个 MLP 同时处理输入 token，以下游任务域对应的 MLP 输出为主，其他两个为辅
- 通过学习的系数线性加权融合：$\text{To}^{(1)} = \text{To}_d^{(1)} + \sum_{d' \neq d} \alpha_{d'} \text{To}_{d'}^{(1)}$
- 两次融合分别作用于两层 FC 的输出

#### 2. 域特征生成器（DFG）

DFG 使用交叉注意力机制从 encoder 特征中提取域特征和类别特征：

- 设置 class token $\hat{c}$ 和三个 domain token $\{\hat{o}, \hat{f}, \hat{s}\}$
- 预训练时：根据输入域选择对应 domain token，与 class token 拼接作为 Q，encoder 特征作为 K/V
- 通过对比损失训练：同域特征相似度高，异域特征相似度低
- $l(d_i, d_j) = 1 - \cos(d_i, d_j)$（同域），$\max(0, \cos(d_i, d_j) - a)$（异域）

微调时，DFG 输出的类特征 c、域特征 d 和点云特征 ℱ 共同输入下游任务头。

### 损失函数 / 训练策略

**预训练损失**：
$$\mathcal{L} = w_1 \mathcal{L}_{rec} + w_2 \mathcal{L}_{con}$$

- 重建损失 $\mathcal{L}_{rec}$：Chamfer Distance
- 对比损失 $\mathcal{L}_{con}$：域特征对比学习
- 最优权重：$w_1 = 100, w_2 = 0.001$（增大重建损失权重，避免对比损失过拟合）

**训练配置**：batch size 512，AdamW 优化器，lr=0.0005，cosine schedule，300 epochs，NVIDIA V100 GPU。

**微调策略**：
- HDA 的三个 MLP 参数冻结，仅训练融合系数 MLP
- 不同下游任务使用不同配置（lr、epoch、点数等），详见实验部分

## 实验关键数据

### 主实验

DAP-MAE 在5个下游任务上评测：物体分类、Few-shot、部件分割、表情识别、3D物体检测。

#### 物体分类（ScanObjectNN）

| 方法 | PM | OBJ_BG | OBJ_ONLY | PB_T50_RS |
|------|-----|--------|----------|-----------|
| Point-MAE | PC | 90.02 | 88.29 | 85.18 |
| Point-FEMAE | PC | 95.18 | 93.29 | 90.22 |
| ReCon-SMC (baseline) | PC | 94.15 | 93.12 | 89.73 |
| 𝕆+𝔽+𝕊 ReCon-SMC | PC | 94.32 | 93.12 | 89.90 |
| **DAP-MAE (Ours)** | **PC** | **95.18** | **93.45** | **90.25** |
| I2P-MAE | PC+I | 94.14 | 91.57 | 90.11 |
| ReCon-full | PC+I+T | 95.18 | 93.63 | 90.63 |

DAP-MAE 仅用单模态（PC）就超越大多数跨模态方法，比 baseline ReCon-SMC 提升 1.03%（OBJ_BG）。

#### 跨域预训练对比

| 方法 | PM | ScanObjectNN | BU3DFE | BOS | AP50 | AP25 |
|------|-----|------|--------|------|------|------|
| ReCon-SMC (same-domain) | PC | 94.15 | 89.13 | 87.69 | 42.7 | 63.8 |
| 𝕆+𝔽+𝕊 ReCon-SMC | PC | 94.32 | 88.52 | 87.23 | 42.5 | 63.5 |
| **DAP-MAE** | **PC** | **95.18** | **89.83** | **88.45** | **43.2** | **64.0** |

简单混合域数据的 ReCon-SMC 在表情识别和检测上反而下降，而 DAP-MAE 在所有任务上都有提升。

#### 3D 物体检测（ScanNetV2）

| 方法 | PM | AP50 | AP25 |
|------|-----|------|------|
| VoteNet | PC | 33.5 | 58.6 |
| 3DETR | PC | 37.9 | 62.1 |
| MaskPoint | PC | 40.6 | 63.4 |
| PiMAE | PC+I | 39.4 | 62.6 |
| ACT | PC+I | 42.1 | 63.8 |
| **DAP-MAE** | **PC** | **43.2** | **64.0** |

DAP-MAE 单模态超越所有跨模态方法，比 ACT 高 1.1% AP50。

### 消融实验

#### 组件消融（ScanObjectNN OBJ_BG）

| CD | HDA | DFG | Accuracy |
|----|-----|-----|----------|
| ✗ | ✗ | ✗ | 94.15 |
| ✓ | ✗ | ✗ | 94.32 |
| ✓ | ✓ | ✗ | 94.66 |
| ✓ | ✗ | ✓ | 94.66 |
| ✓ | ✓ | ✓ | **95.18** |

HDA 和 DFG 各自独立贡献 +0.34%，两者协同再额外提升 0.52%。

#### HDA 融合模式消融

| Fusion Mode | 𝕆+𝔽 | 𝕆+𝔽+𝕊 |
|------------|------|--------|
| Adding（直接相加） | 92.59 | 92.94 |
| FC（FC预测系数） | 94.84 | 93.80 |
| MLP（MLP预测系数） | 93.80 | **95.18** |

不当的融合方式（如直接相加）导致准确率从 94.66% 暴跌至 92.59%，说明跨域数据若处理不当会成为噪声。

#### 特征组合消融

| c | d | ℱ | OBJ_BG |
|---|---|---|--------|
| ✓ | ✗ | ✗ | 93.12 |
| ✗ | ✓ | ✗ | 93.80 |
| ✗ | ✗ | ✓ | 94.49 |
| ✓ | ✓ | ✓ | **95.18** |

三种特征的组合最优，点云特征 ℱ 单独使用效果最好（94.49%），因重建损失保留了完整点云信息。

### 关键发现

1. **简单混合域数据无益甚至有害**：直接在混合域上预训练 ReCon-SMC，表情识别和检测任务性能下降
2. **HDA 冻结参数至关重要**：微调时若不冻结 HDA 的 MLP 参数，后期容易过拟合
3. **融合系数动态变化**：微调过程中，MLP 学习的融合系数呈先增后减趋势——早期多借助其他域知识，后期逐渐退出
4. **模型效率**：相比 baseline ReCon 仅增加 0.2M 参数和 0.1G FLOPs
5. **对比损失权重敏感**：$w_1=100, w_2=0.001$ 最优，对比损失过大容易过拟合

## 亮点与洞察

1. **问题定义精准**：首次系统性地研究了点云 MAE 的跨域预训练问题，并通过实验清晰地展示了简单混合的局限性
2. **双模式 HDA 设计精巧**：预训练时分域处理避免干扰，微调时融合利用多域知识，一套模块完成两阶段的不同需求
3. **DFG 的引导作用**：对比学习训练的域特征不仅区分域，还能在微调时引导特征适配
4. **单次预训练多任务**：一次预训练即可适配4种不同任务和3个不同域，大幅减少预训练成本
5. **单模态超越跨模态**：不需要图像或文本的跨模态蒸馏，纯点云模态即可匹敌甚至超越 ACT、ReCon-full 等跨模态方法

## 局限与展望

1. **域扩展性**：当前模型无法在不重新训练的情况下加入新域，需要探索增量学习/持续学习策略
2. **三域固定**：HDA 的三个并行 MLP 硬编码为三个域，扩展到 N 个域需要重新设计
3. **域定义粒度**：当前将点云粗分为物体/人脸/场景三个域，更细粒度的域定义可能带来更好的效果
4. **预训练数据不均衡**：ShapeNet 50k vs FRGCv2 120k vs S3DIS 的数据量差异可能影响效果
5. **缺少户外场景**：当前仅涵盖室内场景域，自动驾驶等户外场景的适用性未验证

## 相关工作与启发

- **Point-MAE / ReCon**：DAP-MAE 直接建立在这些工作基础之上，使用 ReCon 作为 baseline
- **ACT / I2P-MAE**：跨模态蒸馏方法证明了外部知识的价值，DAP-MAE 用跨域替代跨模态达到类似效果
- **PointNet / PointNet++**：经典点云处理骨干被用于 tokenization
- 启发：相比引入图像/文本等额外模态，挖掘和利用好已有的不同域的同模态数据可能是更高效的路径

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统研究点云 MAE 跨域预训练，HDA 双模式设计创新
- **技术质量**: ⭐⭐⭐⭐ — 方法设计有充分动机，消融实验详细
- **实验充分度**: ⭐⭐⭐⭐⭐ — 5个下游任务、多个数据集、详尽消融和可视化分析
- **实用价值**: ⭐⭐⭐⭐ — 一次预训练多任务适配，显著降低训练成本
- **综合评分**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-qualit.md)

</div>

<!-- RELATED:END -->
