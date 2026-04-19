---
title: >-
  [论文解读] Orochi: Versatile Biomedical Image Processor
description: >-
  [NeurIPS 2025][医学图像][生物医学图像处理] 提出 Orochi——首个面向底层生物医学图像处理的通用基础模型，通过任务相关联合嵌入预训练（TJP）和多头层级 Mamba 架构，在配准、融合、复原和超分辨率四大任务上以轻量微调（<5% 参数）即可达到或超越专用 SOTA 模型。
tags:
  - NeurIPS 2025
  - 医学图像
  - 生物医学图像处理
  - 通用基础模型
  - 自监督预训练
  - Mamba
  - 图像配准/融合/复原/超分
---

# Orochi: Versatile Biomedical Image Processor

**会议**: NeurIPS 2025  
**arXiv**: [2509.22583](https://arxiv.org/abs/2509.22583)  
**代码**: 暂无（论文承诺开源）  
**领域**: 医学图像  
**关键词**: 生物医学图像处理, 通用基础模型, 自监督预训练, Mamba, 图像配准/融合/复原/超分

## 一句话总结

提出 Orochi——首个面向底层生物医学图像处理的通用基础模型，通过任务相关联合嵌入预训练（TJP）和多头层级 Mamba 架构，在配准、融合、复原和超分辨率四大任务上以轻量微调（<5% 参数）即可达到或超越专用 SOTA 模型。

## 研究背景与动机

深度学习在生命科学中的应用日益重要，其中底层生物医学图像处理（配准、融合、复原、超分辨率）是最关键的应用之一。当前领域面临一个核心矛盾：

**专用模型范式的三重困境**：

**任务视角**：实际生物医学成像流程通常需要多个顺序步骤（如先配准再融合），但现有每个步骤都需选择不同的专用模型

**退化视角**：不同任务的底层退化原因具有相关性——低信噪比和低分辨率都是信息丢失，遮挡和变形都是空间变换

**数据视角**：生物医学图像具有多通道、大尺度、高通量的特点，训练和推理多个专用模型效率极低

而现有平台（如 ImageJ/Fiji、napari）虽然提供了各种模型插件，但这些插件都局限于特定任务和数据集。生物学家面对海量插件往往无所适从。

本文的切入角度是：构建一个通用的底层图像处理基础模型，通过统一框架处理所有底层任务，同时利用跨任务学习获得更泛化的特征表示。核心 idea 是利用任务相关的退化作为自监督信号（而非通用的 Masked Image Modelling），因为不同退化之间的内在关联恰好对应了不同底层任务之间的关联。

## 方法详解

### 整体框架

Orochi 的设计围绕四个层面展开：

- **数据层面**：从 100+ 公开研究中收集未标注原始数据（总计超 100 TB），通过随机多尺度采样转换为训练 patch/volume
- **预训练层面**：Task-related Joint-embedding Pre-Training (TJP)
- **模型层面**：Multi-head Hierarchy Mamba
- **后训练层面**：三级微调框架（Full / Normal / Light）

### 关键设计

#### 1. 随机多尺度采样（Random Multi-scale Sampling）

从原始图像中提取不同尺度的 patch/volume：

- **多尺度缩放**：将原始图像 $I$ 缩放到 $1, 1/2, 1/4$ 三个尺度：$I_s = \downarrow_s(I)$
- **随机窗口采样**：对每个尺度的图像，用固定大小窗口 $K$ 随机采样子块：$x_s = I_s(i:i+W-1, j:j+H-1)$

**设计动机**：不同底层任务的感兴趣区域（ROI）尺度不同，多尺度采样扩展了数据多样性，使模型在预训练阶段就能学到跨尺度的特征。

#### 2. 任务相关联合嵌入预训练（TJP）

TJP 的核心思想是：利用与底层任务直接对应的四种退化作为自监督信号，让模型学习不同退化之间的内在关联。

**双遮罩重建融合（Dual-Masking Reconstructive Fusion）**：为融合任务设计。对训练数据施加两组独立遮罩 $M_A, M_B$，模型需要从两个部分遮挡的输入中联合重建原图：

$$x_A = x \odot M_A, \quad x_B = x \odot M_B$$
$$\hat{x} = f_\theta(x_A, x_B)$$

这迫使模型学习从两个不完整视角中发现互补信息并融合。

**空间变化高斯降采样**：为超分辨率任务设计。融合了噪声降采样和空间变化的高斯滤波：

$$D_{\text{LR}}(x) = \mathbf{G}_{\sigma_{\text{var}}}(\uparrow_{1/s}(\downarrow_s(x + \eta)))$$

其中高斯核的标准差在空间坐标上变化，模拟真实光学系统的非均匀模糊。

**多尺度平滑 Perlin 噪声变形**：为配准任务设计。生成逼真的多尺度变形场：

$$D_{\text{def}}(x) = \mathbf{T}(x, \Phi), \quad \Phi = \mathbf{G}_\sigma(\mathbf{Per}(\mathbf{f}, \mathbf{p}))$$

使用多octave Perlin 噪声生成层次化变形，$\tanh$ 函数限制最大位移。

**多阶段噪声模拟**：为复原任务设计。依次叠加高斯噪声、泊松噪声和椒盐噪声：

$$D_{\text{noise}}(x) = \mathbf{Bi}_p(\mathbf{Poi}(\max(0, x + \eta)))$$

**设计动机**：相比 MAE 等通用 MIM 方法只学习重建被遮挡区域，TJP 让模型直接学习与具体任务相关的退化-恢复映射。实验证明 MAE 在配准任务上表现极差（Dice 仅 71.22 vs TJP 的 83.62），因为遮罩与空间变形是完全不同的退化类型。

#### 3. 三级微调框架

- **Full**：全量微调所有参数
- **Normal**：仅微调替换的密集卷积头
- **Light**：使用深度可分离卷积（depth-wise separable conv），仅约 1-2% 参数，实现参数高效微调

### 模型架构

Multi-head Hierarchy Mamba：利用 Mamba 的线性计算复杂度，结合 Swin-Transformer 的层级设计和 patch merging 机制。详细架构见附录。

## 实验关键数据

### 主实验（四大任务综合对比）

| 任务 | 数据集 | 指标 | Orochi (Full) | Orochi (Light) | 之前 SOTA | SOTA 方法 |
|------|--------|------|--------------|---------------|----------|-----------|
| 复原 | CARE | PSNR (XY)↑ | 28.31 | **29.77** | 27.12 | UniFMIR |
| 超分辨 | HBA (4mm) | PSNR↑ | **35.33** | 34.83 | 32.41 | LIIF |
| 配准 | OASIS | Dice↑ | **83.62** | 79.61 | 82.22 | Transmorph-L |
| 融合 | VIFB | Qabf↑ | **0.41** | 0.34 | 0.39 | BSAFusion |

在所有四大任务上，Orochi 全量微调或轻量微调均达到或超越各任务的 SOTA 专用模型。

### 消融实验（预训练策略对比）

| 预训练策略 | 配准 Dice↑ | 融合 Qabf↑ | 复原 PSNR↑ | 超分 PSNR↑ |
|-----------|-----------|-----------|-----------|-----------|
| MAE (单遮罩) | 71.22 | 0.36 | 26.67 | 29.17 |
| I-JEPA (双遮罩) | 69.97 | 0.39 | 25.02 | 28.81 |
| **Orochi (TJP)** | **83.62** | **0.41** | **29.88** | **33.63** |

TJP 在配准任务上比 MAE 提升了 12.4 个 Dice 点，充分验证了任务相关退化设计的必要性。

### 关键发现

1. **轻量微调反超全量微调**：在数据有限的复原任务中（<100 训练 patch），Light 模式（1-2% 参数）反而优于 Full 模式，因为全量微调容易过拟合
2. **TJP vs MIM**：通用 MIM 在部分底层任务上完全失效，证明了任务相关退化设计的必要性
3. **域内零样本泛化**：预训练后的 Orochi 在未见过的测试图像上展现出良好的零样本处理能力
4. **融合任务的互补验证**：在着丝粒计数案例中，模型成功融合了两个部分遮挡视图的互补信息，而非简单重建

## 亮点与洞察

- **首个底层生物医学图像通用模型**：开创性地将配准、融合、复原、超分四大底层任务统一到一个框架
- **退化即任务**：将自监督退化设计与下游任务直接对应的思路，比通用 MIM 更适合底层图像处理
- **实践导向**：三级微调框架给了生物学家灵活的选择——数据少用 Light，数据多用 Full
- **数据工程**：100+ 研究、100 TB 数据量的预训练数据收集本身就是重要贡献

## 局限与展望

- 模型架构细节（Multi-head Hierarchy Mamba）放在附录中，主文信息不充分
- 代码尚未开源，复现困难
- 仅覆盖底层图像处理任务，未涉及高级语义任务（检测、分割等）
- 融合和配准任务的评价指标种类有限

## 相关工作与启发

- UniFMIR 已展示预训练基础模型在生物图像复原上的泛化能力，Orochi 将其扩展到四个任务
- I-JEPA 的联合嵌入预测思想启发了 TJP，但 TJP 替换为任务相关退化
- Mamba 的线性复杂度使得处理大尺度生物医学图像（2-5D）变得可行

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个覆盖四大底层任务的通用生物图像处理模型，TJP 设计精巧
- 实验充分度: ⭐⭐⭐⭐ 四个任务 30+ 基线对比很全面，但缺少计算效率的定量分析
- 写作质量: ⭐⭐⭐⭐ 框架清晰，但方法细节过于依赖附录
- 价值: ⭐⭐⭐⭐⭐ 填补了生物医学底层图像处理通用模型的空白，实践意义重大

<!-- RELATED:START -->

## 相关论文

- [Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)
- [RadEdit: Stress-Testing Biomedical Vision Models via Diffusion Image Editing](../../ECCV2024/medical_imaging/radedit_stress-testing_biomedical_vision_models_via_diffusion_image_editing.md)
- [Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing](unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)
- [Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [ANGEL: Learning from Negative Samples in Biomedical Generative Entity Linking](../../ACL2025/medical_imaging/learning_from_negative_samples_in_biomedical_generative_entity_linking.md)

<!-- RELATED:END -->
