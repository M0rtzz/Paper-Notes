---
title: >-
  [论文解读] Efficient Spiking Point Mamba for Point Cloud Analysis
description: >-
  [3D视觉] SPM（Spiking Point Mamba）提出首个基于 Mamba 的 3D 脉冲神经网络框架，通过层次化动态编码（HDE）和脉冲 Mamba 模块（SMB），在大幅降低能耗（3.5× 以上）的同时，在 ScanObjectNN 上比前 SOTA SNN 方法提升 6-7% 的准确率。
tags:
  - 3D视觉
---

# Efficient Spiking Point Mamba for Point Cloud Analysis

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2504.14371](https://arxiv.org/abs/2504.14371)
- **代码**: 未公开
- **领域**: 3D 视觉 / 点云分析
- **关键词**: SNN, Mamba, 点云分析, 脉冲神经网络, 能量效率

## 一句话总结

SPM（Spiking Point Mamba）提出首个基于 Mamba 的 3D 脉冲神经网络框架，通过层次化动态编码（HDE）和脉冲 Mamba 模块（SMB），在大幅降低能耗（3.5× 以上）的同时，在 ScanObjectNN 上比前 SOTA SNN 方法提升 6-7% 的准确率。

## 研究背景与动机

脉冲神经网络（SNN）因其事件驱动特性在能效方面具有显著优势，但在 3D 点云分析中面临三个关键挑战：

**长距离依赖建模困难**：现有 MLP 和 Transformer 基础的 SNN 架构难以捕获不规则点序列中的远程依赖

**静态时间编码**：传统 direct encoding 沿时间维度简单重复输入，无法充分利用 SNN 时间特征提取能力

**脉冲信息退化**：脉冲驱动计算过程中信息密度显著低于 ANN（每个时间步激活减少约 30%），整合 Mamba 等序列模型时问题加剧

Mamba 提供了线性复杂度的序列建模能力，但直接迁移到 SNN 存在**时间复杂度不匹配**（连续状态转移 vs 离散脉冲事件）和**信息密度差异**的问题。

## 方法详解

### 整体框架

SPM 由三部分组成：
1. **HDE（层次化动态编码）**：将静态点云转换为层次化的动态事件表示
2. **SEL（脉冲嵌入层）**：将 token 映射到高维语义特征
3. **SMB（脉冲 Mamba 模块）**：核心组件，堆叠 N=12 次进行特征交互

### 关键设计一：层次化动态编码（HDE）

传统 SNN 使用 direct encoding，直接在时间维度重复输入，导致时间特征提取失效。HDE 根据最远点采样（FPS）的三个阶段引入动态变化：

- **早期阶段**（不稳定）：因随机初始选择导致不稳定
- **中期阶段**（稳定）：有效捕获点云骨架结构
- **晚期阶段**（冗余）：可能引入噪声

HDE 的具体策略：
- **Finite Forward Sliding**：在早期和中期动态滑动选取 F 个点，步长 l 随时间递减
- **Infinite Backward Extension**：在晚期从剩余点中动态选取 r 个点作为记忆池

### 关键设计二：脉冲 Mamba 模块（SMB）

SMB 是 SPM 的核心，包含两个分支：

**SSM 分支**：
- 输入 spike 经 FC 层和 SN 提取中间特征
- 仅翻转**时间维度**（而非 token 维度），因为稀疏脉冲矩阵的 token 翻转无意义
- 双向 SSM 学习时间步间的动态关系

**Gate 分支**：
- 将 spike 映射到高维生成门控矩阵
- 不直接做逐元素乘法（会导致严重信息丢失），而是先在 token 维度做 Element-wise Average Pooling（EAP）
- 保留重要特征维度，维持 token 间关系

SMB 的数学表达：

$$\mathbf{S}_n'' = \mathcal{SN}(\mathbf{U}_n' + \mathbf{U}_t') \circ \mathcal{SN}(\text{EAP}(\mathbf{Z}_n))$$
$$\mathbf{U}_{n+1} = \text{MLP}(\mathbf{S}_n'') + \mathbf{U}_n$$

### 关键设计三：脉冲预训练

采用非对称 SNN-ANN 异构编码器-解码器架构：
- **编码器**：N 个 SMB 堆叠（SNN）
- **解码器**：$N_d$ 个单向 SSM（ANN，$N_d < N$）
- 使用 Chamfer Distance 作为重建损失
- SNN-ANN 解耦设计：利用 ANN 解码器的建模能力增强 SNN 编码器

### 能耗分析

SPM 将 ANN 中的矩阵乘法（MAC，4.6 pJ）转换为稀疏累加（AC，0.9 pJ）。能耗主要来自 SEL 和 SMB 模块。

## 实验关键数据

### 主实验：3D 分类（ScanObjectNN + ModelNet40）

| 架构 | 方法 | 参数 | T | OBJ-BG | OBJ-ONLY | PB-T50-RS | ModelNet40 |
|------|------|------|---|--------|----------|-----------|------------|
| ANN | PointMamba | 12.3 | - | 90.2 | 89.8 | 85.4 | 92.4 |
| SNN | Spiking PointNet | 3.5 | 4 | 72.2 | 76.4 | 64.1 | 88.2 |
| SNN | P2SResLNet-B | 14.3 | 1 | 78.6 | 80.2 | 74.5 | 88.7 |
| SNN | SPT | 10.2 | 4 | 82.8 | 83.4 | 78.0 | 91.4 |
| **SNN** | **SPM (Ours)** | **12.8** | **4** | **90.2 (+7.4)** | **89.5 (+6.1)** | **84.2 (+6.2)** | **92.3 (+0.9)** |

### 消融实验：SMB 设计

| 模型 | Gate 分支 | SSM 分支 | OBJ-BG | PB-T50-RS | ModelNet40 |
|------|-------|-------|--------|-----------|------------|
| Model I（vanilla Gate + 单向） | Gate I | uni. | 88.3 | 83.1 | 91.4 |
| Model II（vanilla Gate + 双向） | Gate I | bi. | 89.3 | 83.8 | 91.8 |
| Model III（SNN Gate + 单向） | Gate II | uni. | 88.9 | 83.5 | 91.6 |
| **SMB（完整设计）** | **Gate II** | **bi.** | **90.2** | **84.2** | **92.3** |

### 时间步与能耗

| 时间步 | 能耗(mJ) | OBJ-BG | PB-T50-RS | ModelNet40 |
|--------|----------|--------|-----------|------------|
| ANN | 18.9 | 90.2 | 85.4 | 92.4 |
| 1 | 1.5 | 88.9 | 83.3 | 91.6 |
| 4 | 5.4 | 90.2 | 84.2 | 92.3 |
| 6 | 7.6 | 90.0 | 84.3 | 92.3 |

### 关键发现

1. **SNN SOTA 大幅提升**：在 ScanObjectNN 三个变体上比之前最优 SNN 方法（SPT）提升 6-7%
2. **接近 ANN 性能**：SPM 在分类上达到了 ANN 对应方法 PointMamba 的水平
3. **能耗仅为 ANN 的 1/3.5**：时间步为 4 时性能与 ANN 持平，能耗 5.4mJ vs 18.9mJ
4. **预训练有效**：脉冲预训练在 PB-T50-RS 上额外提升 2.3%（84.2→86.5）
5. **时间翻转优于 Token 翻转**：双向策略中，翻转时间维度比翻转 token 维度效果好

## 亮点与洞察

1. **首个 3D Mamba SNN**：成功将 Mamba 引入 3D SNN 领域，解决了脉冲与连续状态的兼容性问题
2. **HDE 的巧妙设计**：利用 FPS 三阶段特性自然引入时间动态，不引入额外参数
3. **EAP 门控**：通过 token 维度的平均池化避免脉冲乘法的信息丢失，简单有效
4. **异构预训练**：SNN 编码 + ANN 解码的非对称设计，推理时仅需 SNN，保持低能耗

## 局限性

1. 最大时间步仅测试到 6，受计算成本限制
2. ShapeNetPart 分割任务上与 ANN 仍有差距（84.8 vs 85.8 ins. mIoU）
3. 仅在较小规模数据集上验证，未在大规模真实场景数据集上测试
4. 依赖特定的脉冲神经元模型（LIF），对其他神经元模型的适用性未探索

## 相关工作与启发

- **PointMamba**：ANN 版本的 Point Mamba，SPM 以此为基础构建 SNN 版本
- **Spiking Point Transformer (SPT)**：前 SNN SOTA，基于 Transformer 架构
- **TA-TiTok / S4D / Mamba**：SPM 集成了 SSM 的高效序列建模与 SNN 的低能耗优势

## 评分

⭐⭐⭐⭐ — 首创性强且实验充分，在 SNN 领域实现了跨越式进步，但实际应用场景和更大规模验证有待加强。

<!-- RELATED:START -->

## 相关论文

- [Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)
- [DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](../../CVPR2025/3d_vision/dualpm_dual_posed-canonical_point_maps_for_3d_shape_and_pose_reconstruction.md)
- [RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)

<!-- RELATED:END -->
