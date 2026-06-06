---
title: >-
  [论文解读] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation
description: >-
  [CVPR 2025][医学图像][点云生成] 提出 SeaLion，通过语义部件感知的潜点扩散技术，在去噪过程中联合预测噪声和逐点分割标签，并以分割标签为条件解码点云，生成具有高质量部件间一致性和精确分割标签的 3D 点云…
tags:
  - "CVPR 2025"
  - "医学图像"
  - "点云生成"
  - "扩散模型"
  - "语义分割"
  - "部件感知"
  - "潜空间"
---

# SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation

**会议**: CVPR 2025  
**arXiv**: [2505.17721](https://arxiv.org/abs/2505.17721)  
**代码**: [https://github.com/Dekai21/SeaLion](https://github.com/Dekai21/SeaLion)  
**领域**: 医学图像  
**关键词**: 点云生成, 扩散模型, 语义分割, 部件感知, 潜空间

## 一句话总结

提出 SeaLion，通过语义部件感知的潜点扩散技术，在去噪过程中联合预测噪声和逐点分割标签，并以分割标签为条件解码点云，生成具有高质量部件间一致性和精确分割标签的 3D 点云；同时提出 part-aware Chamfer distance (p-CD) 评价指标，在 ShapeNet 和 IntrA 数据集上大幅超越 DiffFacto。

## 研究背景与动机

**领域现状**：3D 点云生成已取得显著进展（Lion、PVD、DPM 等），但现有方法主要关注无标签点云的生成。带语义分割标签的点云生成几乎是空白——已有的方法（TreeGAN、EditVAE 等）虽然能生成可分离的子部件，但这些子部件**缺乏明确的语义含义**。

**现有痛点**：DiffFacto 是唯一能生成语义标注点云的方法，但它为每个部件使用独立的 DDPM 分别生成，再通过姿态预测进行组装。这种"先独立生成各部件，再组装"的策略导致**部件间一致性差**（inter-part coherence）——组装出的形状可能不符合真实分布。现有评估指标（1-NNA-P、SNAP）也无法有效度量部件间一致性。

**核心矛盾**：分布式生成（每部件独立生成）天然无法保证部件间的一致性，而整体生成（一次性生成完整点云）又难以获得逐点语义标签。如何在保证整体一致性的同时获得精确的语义分割标签是核心挑战。

**本文目标**：设计一个同时生成高质量点云和精确分割标签的模型，且保证部件间一致性。同时提出能有效评估此任务的指标。

**切入角度**：受生成模型中间特征可用于语义分割的启发（DDPM 的中间表示包含高层语义信息），在扩散去噪过程中利用中间特征同时预测噪声和分割标签，实现语义感知的生成。

**核心 idea**：不单独生成每个部件，而是在统一的潜空间中同时扩散所有部件的潜点，利用 U-Net 的共享下采样路径学习噪声预测和分割预测的公共表征，通过两个并行上采样路径分别输出噪声和标签，再以预测标签为条件解码点云坐标。

## 方法详解

### 整体框架

SeaLion 基于 Lion 的层次化潜空间扩散框架，包含两阶段训练：第一阶段训练条件 VAE（全局编码器 $\phi_z$、点级编码器 $\phi_h$、条件解码器 $\xi_h$），以分割标签 $y$ 为条件进行编码和解码；第二阶段训练两个扩散模块（全局扩散 $\epsilon_z$、点级扩散 $\epsilon_h$），其中 $\epsilon_h$ 联合预测噪声和分割标签。推理时，先生成全局潜变量 $z_0$，再生成潜点 $h_0$ 和分割标签 $\hat{y}$，最后以 $\hat{y}$ 和 $z_0$ 为条件解码出点云。

### 关键设计

1. **语义部件感知潜点扩散技术**:

    - 功能：使生成模型在扩散过程中获得语义部件感知能力
    - 核心思路：两个关键创新——(1) 在 VAE 的编码器和解码器中引入分割编码 $y$ 作为条件，训练模型在 $y$ 引导下重建点云，使模型学会语义-几何对应；(2) 点级扩散模型 $\epsilon_h$ 同时输出噪声预测 $\hat{\epsilon}_t$ 和分割预测 $\hat{y}_t$：$\hat{\epsilon}_t, \hat{y}_t \leftarrow \epsilon_h(h_t, t, z_0)$。推理时分割预测通过 EMA（smoothing factor=0.1）逐步平滑为最终标签 $\hat{y}$，条件解码器 $\xi_h$ 以 $\hat{y}$ 为条件生成与标签严格对齐的点云。
    - 设计动机：相比传统的"先生成无标签点云，再用预训练分割模型分配伪标签"，联合生成法更简洁鲁棒——不依赖外部模型，且条件解码保证了坐标与标签的对齐。

2. **双路上采样 U-Net 架构**:

    - 功能：从共享表征中分别提取噪声预测和分割预测的任务特定特征
    - 核心思路：点级扩散 $\epsilon_h$ 使用修改的 PVCNN（Point-Voxel CNN）U-Net，包含一个共享的下采样路径学习公共表征 $r_c$，和两个并行的上采样路径分别输出噪声预测特征 $r_\epsilon$ 和分割预测特征 $r_y$。每个上采样层将上一层的任务特定特征与对应层的公共表征拼接后处理。
    - 设计动机：噪声预测和分割预测虽然共享底层几何信息，但本质是两个不同任务。共享下采样保证信息效率，分离上采样保证任务特异性。

3. **Part-aware Chamfer Distance (p-CD)**:

    - 功能：评估带分割标签点云的生成质量，特别是部件间一致性
    - 核心思路：对两个点云 $x^1$ 和 $x^2$（各含 $|P|$ 个部件），计算每个部件内的 Chamfer Distance 并求和：$\text{p-CD}(x^1, x^2) = \sum_{p \in P} \text{CD}(x^1_p, x^2_p)$。如果两个点云的部件组成不同，p-CD 定义为无穷大。基于 p-CD，可以计算 1-NNA (p-CD)、COV (p-CD)、MMD (p-CD) 等指标。
    - 设计动机：现有的 1-NNA-P（按部件平均）无法捕捉部件间不一致性——极端情况下，从不同形状随机组合部件仍能获得高 1-NNA-P 分数。p-CD 通过整体计算避免了这个漏洞，能有效检测"组装异常"的形状。

### 损失函数 / 训练策略

- **VAE 阶段**：最大化 ELBO，包含重建项 + KL 正则项，$\lambda_z$ 和 $\lambda_h$ 为平衡超参
- **扩散阶段**：全局扩散 $\mathcal{L}(\epsilon_z) = E[\|\epsilon_z(z_t,t) - \epsilon\|^2_2]$；点级扩散 $\mathcal{L}(\epsilon_h) = E[\|\hat{\epsilon}_t - \epsilon\|^2_2 + \lambda_{seg} H(y, \hat{y}_t)]$，其中 $H$ 为交叉熵，$\lambda_{seg}$ 平衡两个任务
- VAE 训练 8k epochs，扩散训练 24k epochs，Adam 优化器 lr=1e-3
- VAE 参数 22.3M，扩散参数 98.1M，RTX 3090 单卡训练

## 实验关键数据

### 主实验

| 方法 | Airplane 1-NNA↓ | Car 1-NNA↓ | Chair 1-NNA↓ | Lamp 1-NNA↓ |
|------|-----------------|------------|-------------|------------|
| Lion + SPoTr | 67.13 | 77.36 | 65.27 | - |
| DiffFacto | 81.67 | 90.51 | 77.34 | 67.13 |
| **SeaLion** | **65.40** | **73.10** | **63.14** | **61.71** |

SeaLion 在四个类别上平均超越 DiffFacto 13.33%（1-NNA p-CD）。

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| Lion (无标签) + PointNet++ | 两步法：先生成无标签点云，再分割 | 1-NNA 68.48 (airplane) |
| Lion + SPoTr (SOTA分割) | 两步法换用更好的分割模型 | 1-NNA 67.13 |
| DiffFacto | 按部件分别生成再组装 | 1-NNA 81.67 (部件间不一致) |
| SeaLion (full) | 联合生成+条件解码 | 1-NNA **65.40** |
| SeaLion (半监督, 10% 标签) | 仅 10% 数据有标签 | 性能仍然可接受 |

### 关键发现

- DiffFacto 的 1-NNA-P 到 1-NNA (p-CD) 下降剧烈（如 airplane: 约 50→81.67），验证了其部件间一致性差的问题
- SeaLion 可以半监督训练（仅少量标注数据），利用额外无标签数据提升性能，降低标注成本
- 分割预测的 mIoU 随扩散 timestep 从 T 递减到 0 逐步提升，与噪声去除的过程一致
- 生成数据用于增广训练分割模型时，能有效提升下游分割性能（尤其在医学数据 IntrA 上标注稀缺时）
- SeaLion 可用作部件编辑工具——冻结保留部件的潜点，只对其余部件做扩散-去噪，实现局部形状变化

## 亮点与洞察

- **联合生成标签与点云的范式**：不是事后分割，而是在生成过程中同步产生标签，并以标签为条件保证对齐——这个思路可以迁移到任何需要"生成带标注数据"的场景（如 2D 图像生成+语义分割）
- **p-CD 指标设计精巧**：通过一个简单的修改（按部件计算 CD 再求和）就解决了现有指标无法捕捉部件间一致性的系统性缺陷，且论文用反例（Figure 4 的随机组合实验）直观展示了旧指标的漏洞
- **半监督能力**：在医学图像等标注昂贵的领域，能利用无标签数据是重要的实际价值

## 局限与展望

- 模型参数量较大（VAE 22.3M + 扩散 98.1M），训练和推理效率有待优化
- 每个类别需要单独训练一个模型，无法跨类别泛化
- 目前仅支持固定数量部件的生成，对于部件数量变化（如不同结构的椅子）的处理有待探索
- p-CD 假设两个点云的部件组成相同，否则距离为无穷大——这个定义对于处理新颖部件组合不够灵活
- 条件解码依赖于分割标签预测的准确性，当标签预测有误时可能导致几何形状失真

## 相关工作与启发

- **vs DiffFacto**: DiffFacto 按部件分别生成再组装，部件间一致性差。SeaLion 同时扩散所有部件的潜点，天然保证整体一致性，1-NNA (p-CD) 平均优 13.33%
- **vs Lion**: Lion 是 SOTA 无标签点云生成模型。SeaLion 在其框架基础上加入分割条件和联合预测，在生成质量上也有提升（即使不考虑标签，65.40 vs 67.13 on airplane）
- **vs 两步法 (Lion + SPoTr)**: 先生成再分割的方法受限于分割模型的泛化能力，且无法保证坐标-标签对齐。SeaLion 的端到端方法更简洁鲁棒

## 评分

- 新颖性: ⭐⭐⭐⭐ 联合标签生成思路有创新，p-CD 指标填补了评测空白
- 实验充分度: ⭐⭐⭐⭐⭐ ShapeNet (6类) + IntrA (医学), 半监督实验、数据增广实验、编辑应用全面展示
- 写作质量: ⭐⭐⭐⭐ 动机清晰，Figure 4 的反例说明很有说服力
- 价值: ⭐⭐⭐⭐ 带标签点云生成+评测指标=完整的问题定义和解决方案，对下游应用有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[CVPR 2025\] Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis](latent_drifting_in_diffusion_models_for_counterfactual_medical_image_synthesis.md)
- [\[CVPR 2025\] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)
- [\[NeurIPS 2025\] Semantic and Visual Crop-Guided Diffusion Models for Heterogeneous Tissue Synthesis in Histopathology](../../NeurIPS2025/medical_imaging/semantic_and_visual_crop-guided_diffusion_models_for_heterogeneous_tissue_synthe.md)
- [\[CVPR 2025\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)

</div>

<!-- RELATED:END -->
