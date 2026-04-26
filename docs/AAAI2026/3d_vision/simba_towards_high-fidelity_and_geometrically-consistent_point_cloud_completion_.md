---
title: >-
  [论文解读] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion
description: >-
  [AAAI 2026][3D视觉][点云补全] 提出 Simba 框架，首次将点云补全重构为"对几何变换场做扩散"而非"对点坐标做扩散"，通过 Sym-Diffuser 学习逐点仿射变换的条件分布来生成粗糙补全，再用级联 Mamba 架构（MBA-Refiner）逐步精修到高保真输出，在 PCN、ShapeNet、KITTI 多个基准上达到 SOTA。
tags:
  - AAAI 2026
  - 3D视觉
  - 点云补全
  - 扩散模型
  - 对称先验
  - Mamba
  - 仿射变换
---

# Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion

**会议**: AAAI 2026  
**arXiv**: [2511.16161](https://arxiv.org/abs/2511.16161)  
**代码**: [https://github.com/I2-Multimedia-Lab/Simba](https://github.com/I2-Multimedia-Lab/Simba)  
**领域**: 3D视觉  
**关键词**: 点云补全, 扩散模型, 对称先验, Mamba, 仿射变换

## 一句话总结

提出 Simba 框架，首次将点云补全重构为"对几何变换场做扩散"而非"对点坐标做扩散"，通过 Sym-Diffuser 学习逐点仿射变换的条件分布来生成粗糙补全，再用级联 Mamba 架构（MBA-Refiner）逐步精修到高保真输出，在 PCN、ShapeNet、KITTI 多个基准上达到 SOTA。

## 研究背景与动机

### 领域现状

点云补全是 3D 视觉的基础任务，目标是从残缺的部分观测恢复完整的 3D 形状。现有方法大致经历了几代发展：

**粗到细范式**（PCN、FoldingNet）：全局形状先验 → 细化

**Transformer 方法**（PoinTr、SeedFormer、CRA-PCN）：捕获长程依赖，当前主流

**对称先验方法**（SymmCompletion）：利用对称性学习逐点局部仿射变换

**扩散方法**（PDR、PCDreamer）：在点坐标空间做扩散

### 现有痛点

作者聚焦于**利用对称先验的方法**（如 SymmCompletion），指出其两个关键缺陷：

**过拟合**：回归方式倾向于记忆训练集中实例特定的变换模式，而非学到可泛化的几何对齐规则。在跨域场景（如 KITTI 真实数据）上泛化能力差。

**噪声敏感**：逐点独立回归变换，对遮挡和噪声高度敏感，导致全局结构碎片化或扭曲。

同时，直接在点坐标空间做扩散的方法也有问题：会"洗掉"部分输入中的精细细节，且计算成本高、推理慢。

### 核心矛盾与切入角度

**如何既利用对称先验中的强几何信息，又避免网络仅仅记忆特定变换模式？**

作者的关键观察：扩散模型具有强生成能力，能做多样性采样。将扩散与变换矩阵结合，可在利用几何先验的同时避免过拟合到固定解。

**核心创新**：不扩散点坐标，而是**扩散几何变换场**。学习逐点仿射变换的条件分布 $p(\mathcal{T}|\mathcal{F}_k)$，通过迭代去噪生成变换场并应用到关键点上构建完整形状，天然保留输入的精细细节。

## 方法详解

### 整体框架

两阶段设计：

- **Stage 1**：预训练 SymmGT 网络，生成目标变换矩阵（作为 Stage 2 扩散的监督目标）
- **Stage 2**：
    - **Sym-Diffuser**（对称扩散模块）：在变换场空间做条件扩散，生成粗糙补全
    - **MBA-Refiner**（级联 Mamba 精修器）：三层级联精修 + 上采样

### 关键设计

#### 1. SymmGT 预训练（Stage 1）

**功能**：生成扩散模型训练所需的"干净"目标变换场 $\mathcal{T}_{gt}$。

**核心流程**：
- 输入：部分点云 $\mathcal{P}_{in}$ 和完整 GT $\mathcal{P}_{gt}$
- 从 $\mathcal{P}_{in}$ 采样关键点 $\mathcal{P}_k$
- 共享权重的特征提取器（SA 层 + Point Transformer）分别提取关键点特征 $\mathcal{F}_k$ 和 GT 全局特征 $\mathcal{F}_{gt}$
- Cross-attention 融合后回归变换场 $\mathcal{T}_{gt} \in \mathbb{R}^{K \times 12}$，由逐点仿射矩阵 $\mathbf{A}_i \in \mathbb{R}^{3 \times 3}$ 和平移向量 $\mathbf{T}_i \in \mathbb{R}^3$ 组成
- 变换应用到关键点：$\mathcal{P}_{init} = \mathcal{P}_k \cup \{\mathbf{A}_i \mathbf{p}_i + \mathbf{T}_i\}$
- 用 Chamfer Distance 训练

**Stage 2 中冻结 SymmGT**，仅用来产生 $\mathcal{T}_{gt}$ 作为扩散的 $\mathcal{Z}_0$。

#### 2. Sym-Diffuser（对称扩散模块）

**功能**：学习变换场的条件分布，生成结构完整的粗糙补全。

**核心思路**：
- **前向过程**：标准 DDPM，$T=100$ 步，逐步给 $\mathcal{Z}_0$（目标变换场）加噪
- **反向过程**：噪声预测器 $\epsilon_\theta$ 估计噪声，恢复预测的干净变换场 $\hat{\mathcal{T}}_\theta$
- **训练目标**：受 Consistency Models 启发，使用加权 MSE 损失：

$$\mathcal{L}_{\text{proxy}} = \mathbb{E}_{t, \mathcal{Z}_0, \epsilon}\left[\lambda(t) \|\mathcal{T}_{gt} - \hat{\mathcal{T}}_\theta(\mathcal{Z}_t, t, \mathcal{F}_k)\|^2\right]$$

- **推理**：从随机高斯向量 $\mathbf{Z} \in \mathbb{R}^{N_k \times 12}$ 开始，条件于 $\mathcal{F}_k$，迭代去噪得到变换场 → 应用到关键点 → 得到粗糙补全 $\mathcal{P}_{init} = \mathcal{P}_k \cup \mathcal{P}_s$

**相比直接回归的优势**：
- 扩散模型学的是分布而非确定映射，天然避免过拟合
- 生成过程有多样性，对噪声和遮挡更鲁棒
- 在低维空间（12维变换向量）做扩散，比高维点坐标空间更高效

#### 3. MBA-Refiner（级联 Mamba 精修器）

**功能**：将粗糙补全逐步精修上采样到高保真输出。

**核心架构**：三层级联，上采样比例 $[2\times, 2\times, 4\times]$，总计 $16\times$。每层包含特征融合 + MambaForward 精修。

**异构融合策略**——在不同密度层使用不同融合方式：

- **Block 1-2**（低密度层）：**Cross-Attention 融合**，性能优先
    - 基础特征 $\mathcal{F}_l$ 分别 attend 到关键点特征 $\mathcal{F}_k$ 和对称点特征 $\mathcal{F}_s$
    - 拼接后通过 MLP 融合

$$\mathbf{f}_{in}^l = \boldsymbol{\psi}\left([\text{MCA}(\mathcal{F}_l, \mathcal{F}_g)]_{g \in \{k,s\}}\right)$$

- **Block 3**（高密度层）：**Mamba Fusion**，效率优先
    - $\mathcal{O}(N^2)$ 的注意力在高密度点云上不可承受
    - Mamba 的线性复杂度 $\mathcal{O}(N)$ 大幅降低内存和计算开销

**MambaForward 模块**：所有层共享的精修+上采样模块，包含 MLP → Mamba block (带残差连接) → 上采样层。

**设计动机**：异构设计（前两层用 attention，最后一层用 Mamba）在性能和效率之间取得最优平衡。纯 attention 内存太高（16.4GB），纯 Mamba 性能不足（CD 6.43 vs 6.34）。

### 损失函数 / 训练策略

**Stage 1 损失**：

$$\mathcal{L}_{\text{stage1}} = L_{CD}(\mathcal{P}_k \cup \{\mathbf{A}_i \mathbf{p}_i + \mathbf{T}_i\}, \mathcal{P}_{gt})$$

**Stage 2 损失**（多层级监督）：

$$\mathcal{L}_{\text{stage2}} = \mathcal{L}_{\text{proxy}} + \sum_{l=1}^{3} L_{CD}(\mathcal{P}_{out}^l, \mathcal{P}_{gt})$$

同时监督 Sym-Diffuser 和 MBA-Refiner 的每层输出。

训练设置：PyTorch，4 × NVIDIA RTX 4090。

## 实验关键数据

### 主实验

**PCN 数据集**（8 类，L1-CD ×10³ ↓ / F-Score@1% ↑）：

| 方法 | 会议 | 平均 CD ↓ | F-Score ↑ |
|------|------|----------|-----------|
| PCN | 3DV 2018 | 9.64 | 0.695 |
| PoinTr | ICCV 2021 | 8.38 | - |
| SnowflakeNet | ICCV 2021 | 7.21 | 0.801 |
| SeedFormer | ECCV 2022 | 6.74 | 0.818 |
| AdaPoinTr | TPAMI 2023 | 6.53 | 0.845 |
| CRA-PCN | AAAI 2024 | 6.39 | - |
| SymmCompletion | AAAI 2025 | 6.47 | 0.840 |
| PointCFormer | AAAI 2025 | 6.41 | 0.855 |
| PCDreamer | CVPR 2025 | 6.52 | **0.856** |
| **Simba (Ours)** | **AAAI 2026** | **6.34** | 0.853 |

整体 CD 最优，比 SymmCompletion 低 2%（6.34 vs 6.47），在 Sofa、Table、Watercraft 上尤其突出。

**ShapeNet-55/34/21**（L2-CD ×10³ ↓）：

| 方法 | 55类 Avg | 34 Seen Avg | 21 Unseen Avg |
|------|---------|-----------|-------------|
| AdaPoinTr | 0.81 | 0.73 | 1.23 |
| SVDFormer | 0.83 | 0.75 | 1.28 |
| CRA-PCN | 0.85 | 0.76 | 1.24 |
| **Simba** | **0.79** | **0.70** | **1.23** |

在 55 全类和 34 已见类上均最优，21 未见类上与 AdaPoinTr 持平，说明有良好泛化能力。

**KITTI 真实数据**（MMD ×10³ ↓）：

| 方法 | MMD ↓ |
|------|-------|
| CRA-PCN | 1.737 |
| SeedFormer | 0.516 |
| EINet | 0.967 |
| SymmCompletion | 0.970 |
| **Simba** | **0.423** |

在真实 LiDAR 数据上大幅领先，验证了变换扩散范式的跨域泛化优势——仅在合成数据训练，直接用于真实数据。

### 消融实验

**预测模块消融**（PCN，CD-L1 ×10³）：

| 配置 | CD ↓ | 说明 |
|------|------|------|
| Diffusion Model (Ours) | **6.34** | 扩散方式生成变换场 |
| Transformer Regression | 6.48 | 直接回归变换场 |

扩散优于回归 2.2%，且可视化显示回归方式产生明显结构伪影。

**渐进上采样策略消融**（总倍率 16×）：

| 配置 | CD ↓ | 说明 |
|------|------|------|
| 3层 [2×, 2×, 4×] (Ours) | **6.34** | 渐进式，最优 |
| 1层 [16×] | 6.70 | 一步到位，最差 |
| 2层 [2×, 8×] | 6.56 | 不均匀 |
| 2层 [4×, 4×] | 6.52 | 两层均匀 |

渐进多层级精修显著优于激进单步/两步上采样。

**MBA-Refiner 架构消融**：

| 配置 | 融合策略 | 内存 | CD ↓ |
|------|---------|------|------|
| [CA, CA, MFusion] (Ours) | 异构 | 14.7GB | **6.34** |
| [MLP, MLP, MFusion] | 简单融合 | 12.1GB | 6.49 |
| [CA, CA, MLP] | 无 Mamba | 12.0GB | 6.41 |
| [CA, CA, CA] | 全 attention | 16.4GB | 6.35 |
| [MFusion×3] | 全 Mamba | 13.8GB | 6.43 |

异构设计（CA+CA+Mamba）在性能（6.34）和内存（14.7GB）间取得最佳平衡。全 attention 性能接近（6.35）但多耗 11.6% 内存。

### 关键发现

1. 扩散模型学习变换分布比确定性回归更鲁棒——本质原因是避免了过拟合
2. 渐进上采样至关重要——一步 16× 上采样 CD 增加 5.7%
3. 在 KITTI 上的出色表现证明了变换扩散范式在合成→真实迁移中的优越性（MMD 降低 18%，相比 SeedFormer）
4. Mamba 在高密度点云上是 attention 的有效替代，以微小性能代价换取显著内存节省

## 亮点与洞察

1. **范式创新**："扩散变换场"而非"扩散点坐标"——在低维空间（12维）做扩散更高效，且天然保留输入细节（因为变换应用到原始关键点上）
2. **两阶段解耦**：Stage 1 生成监督目标 + Stage 2 做扩散，避免了端到端训练扩散模型的不稳定性
3. **异构级联设计**：根据点密度自适应选择融合策略（低密度用 attention，高密度用 Mamba），是工程和理论的良好结合
4. **强跨域泛化**：仅合成数据训练即在 KITTI 真实数据上 SOTA，对实际部署意义重大

## 局限与展望

1. **推理速度**：扩散模型需要多步迭代去噪（$T=100$），可能比纯前馈方法慢。论文未报告推理时间
2. **两阶段训练**：Stage 1 需要单独预训练 SymmGT，增加了总训练复杂度
3. **对称性假设**：框架建立在对称先验之上，对高度不对称物体（如不规则自然物体）可能效果受限
4. **PCN 数据集局限**：仅 8 类物体，虽然也在 ShapeNet-55 上验证了，但缺少更大规模/更多样化场景的验证
5. F-Score 略低于 PCDreamer（0.853 vs 0.856），说明表面重建精度仍有提升空间

## 相关工作与启发

- **SymmCompletion** (AAAI 2025)：本文的直接前驱，提出逐点仿射变换回归。Simba 将其从回归升级为扩散
- **Consistency Models** (2023)：启发了扩散训练目标的设计
- **Mamba** (2023)：线性复杂度序列建模，在点云领域开始崭露头角（PointMamba、3DMambaComplete）
- **PCDreamer** (CVPR 2025)：2D先验+扩散做点云补全，但直接在坐标空间扩散
- 变换场扩散的思路可推广到其他形变/变换学习任务（如配准、形变预测）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — "扩散变换场"范式新颖且原理清晰，是扩散模型在3D任务中的创新应用
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个基准（PCN/ShapeNet/KITTI）+ 详细消融（预测模块/上采样/架构）
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，图表丰富，但部分推导（如扩散训练目标）可更详细
- 价值: ⭐⭐⭐⭐⭐ — 新范式+强泛化+开源代码，对点云补全领域有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[CVPR 2025\] PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors](../../CVPR2025/3d_vision/pcdreamer_point_cloud_completion_through_multi-view_diffusion_priors.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)
- [\[AAAI 2026\] 4DSTR: Advancing Generative 4D Gaussians with Spatial-Temporal Rectification for High-Quality and Consistent 4D Generation](4dstr_advancing_generative_4d_gaussians_with_spatial-tempora.md)

<!-- RELATED:END -->
