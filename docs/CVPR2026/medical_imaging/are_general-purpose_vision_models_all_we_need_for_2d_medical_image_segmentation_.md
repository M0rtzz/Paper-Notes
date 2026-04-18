---
title: >-
  [论文解读] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
description: >-
  [CVPR 2026][医学图像][图像分割] 通过在三个异构医学数据集上对 11 种架构进行标准化对比实验，证明了通用视觉模型 (GP-VMs) 在 2D 医学图像分割中可以超越大多数专用医学分割架构 (SMAs)，且 XAI 分析表明 GP-VMs 无需特定领域设计也能捕获临床相关结构。
tags:
  - CVPR 2026
  - 医学图像
  - 图像分割
  - general-purpose vision models
  - empirical study
  - benchmarking
  - Grad-CAM
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**会议**: CVPR 2026  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**代码**: 有 (GitHub)  
**领域**: 医学图像分割  
**关键词**: 通用视觉模型, 医学图像分割, 实证研究, 跨数据集评估, Grad-CAM

## 一句话总结

通过统一训练协议在三个异质医学数据集上对比 11 种模型，发现通用视觉模型（GP-VMs）在标准化条件下系统性超越大多数专用医学分割架构（SMAs），挑战了"医学分割必须使用专用架构"的传统认知。

## 研究背景与动机

医学图像分割（MIS）是计算机辅助诊断的核心组件。过去十年，大量针对医学影像的专用架构（如 U-Net 及其变体）被提出，以应对低对比度、小目标结构、有限标注等领域特定挑战。与此同时，通用视觉模型（如 SegFormer、InternImage 等）在自然图像上展现了强大的泛化能力。

然而，一个根本性问题尚未得到充分研究：**医学分割任务是否真的需要专用架构，还是通用视觉模型已经足够？** 现有研究在比较时缺乏标准化评估协议，不同论文在数据集、预处理、增强策略、训练设置上差异显著，导致性能差异可能源于实验设计而非模型架构本身。

本文通过严格控制变量的跨数据集实证研究，首次在统一条件下系统性回答这一问题。

## 方法详解

### 整体框架

本文并非提出新架构，而是构建了一套**标准化基准测试框架**，在统一条件下公平比较两大类模型：

- **专用医学分割架构（SMA）**：5 种，包括 U-Net、HiFormer-B、MISSFormer、U-KAN-L、Swin-UMamba
- **通用视觉模型（GP-VM）**：6 种，分为语义分割架构（SegFormer-B3、SegNeXt-L、VWFormer×2）和视觉骨干+UPerHead 解码器（InternImage-T、TransNeXt-Tiny）

### 关键设计

1. **标准化训练协议**：所有模型使用统一设置——ImageNet 预训练编码器、$512 \times 512$ 输入分辨率、AdamW 优化器 + REX 学习率调度器、batch size 8，每个数据集使用相同的增强管道。学习率从 $\{10^{-4}, 5 \times 10^{-5}, 10^{-5}\}$ 中搜索最优值，训练 150 epochs 并采用统一的早停策略。

2. **异质数据集覆盖**：选择三个在成像模态、分类结构和数据特征上截然不同的医学数据集：
    - **ISIC'18**：皮肤镜 RGB 图像，二分类病变分割（3,565 张），特点是不规则边界
    - **NeoPolyp**：内窥镜 RGB 图像，三分类息肉分割（945 张），特点是亚型变异性大
    - **CAMUS**：超声心动图灰度图像，四分类心脏区域分割（1,996 张），特点是噪声多

3. **严格的数据泄漏防护**：通过原始字节和感知哈希相似度过滤重复/近似图像；CAMUS 数据集使用患者级别划分避免信息泄漏；所有数据集采用五折交叉验证。

### 损失函数 / 训练策略

- ISIC'18 使用 Binary Cross-Entropy 损失
- NeoPolyp 和 CAMUS 使用 Cross-Entropy 损失
- 统一使用混合精度训练（NVIDIA A100 GPU），SegNeXt 因稳定性问题除外
- 特殊处理：MISSFormer 和 HiFormer 内部使用 $224 \times 224$ 分辨率；U-KAN 缺乏 ImageNet 预训练，从头训练

## 实验关键数据

### 主实验

| 模型 | 类别 | NeoPolyp mDSC | CAMUS mDSC | ISIC'18 mDSC | 平均 mDSC |
|------|------|--------------|------------|-------------|----------|
| U-Net | SMA | 83.3±1.1 | 89.1±0.3 | - | - |
| HiFormer-B | SMA | 84.6±0.9 | 90.8±0.2 | - | 88.8 |
| MISSFormer | SMA | 82.9±1.6 | 90.4±0.1 | - | ≤87.9 |
| Swin-UMamba | SMA | **88.9±0.6** | **91.3±0.3** | - | **90.5** |
| U-KAN-L | SMA | 82.5±1.7 | 90.5±0.2 | - | - |
| SegFormer-B3 | GP-VM | 89.1±1.3 | - | - | 90.7 |
| SegNeXt-L | GP-VM | 89.2±0.7 | - | - | 90.7 |
| VW-MiT | GP-VM | **89.7±0.8** | - | - | **91.0** |
| VW-Conv | GP-VM | 89.6±1.3 | - | - | 90.9 |
| InternImage-T | GP-VM | 89.6±1.1 | - | - | 90.8 |
| TransNeXt-Tiny | GP-VM | 89.4±0.7 | - | - | 90.9 |

### 消融实验 / 类别分析

| 模型 | NeoPolyp C1 (非肿瘤性) | NeoPolyp C2 |
|------|----------------------|-------------|
| VW-MiT (GP-VM) | **66.1±4.3** | 92.7±0.9 |
| InternImage (GP-VM) | 66.0±5.7 | 92.9±0.7 |
| Swin-UMamba (SMA) | 59.2±3.8 | **92.5±0.6** |
| HiFormer (SMA) | 52.7±4.9 | 88.9±0.7 |
| U-KAN (SMA) | 36.9±12.2 | 87.1±0.9 |

在最困难的 NeoPolyp C1 类别上，GP-VM 相比最佳 SMA（Swin-UMamba）领先高达 **7 个百分点**。

### 关键发现

- **GP-VM 系统性优于大多数 SMA**：按三数据集平均 mDSC 排名，前六名全部为 GP-VM（91.0%–90.7%），最佳 SMA (Swin-UMamba) 排名第七（90.5%）
- **性能差距因数据集而异**：NeoPolyp 上差距最大（GP-VM 领先 SMA 4-7 个百分点），ISIC'18 和 CAMUS 上差距缩小至 1-2%
- **Swin-UMamba 是唯一接近 GP-VM 的 SMA**：其余 SMA 在 NeoPolyp 上落后超过 4 个百分点
- **XAI 分析**：Grad-CAM 可视化表明 GP-VM 能够捕获临床相关结构，无需显式领域特定设计

## 亮点与洞察

- **核心洞察**：在标准化条件下，"专用架构的性能优势"在很大程度上消失了——此前文献报告的优势可能主要来自实验设置差异而非架构本身
- **实用价值**：对于新的医学分割任务，应优先评估 GP-VM（尤其是 VWFormer、InternImage 等）而非盲目设计新架构
- **研究范式启示**：MIS 社区应更关注数据策划、训练协议优化和 OOD 泛化评估，而非片面追求新架构

## 局限与展望

- 仅覆盖三个 2D 数据集，缺乏 3D 体积数据（如 CT/MRI）和更多成像模态
- 11 种模型并非穷举，参数量差异（25M–60M）可能引入偏差
- 未评估 OOD 泛化（如在 NeoPolyp 上训练、Kvasir-SEG 上测试）
- 未纳入最新的基础模型（如 SAM2、BiomedParse 等）的微调版本
- 缺少计算效率（FLOPs、推理时间）的系统对比

## 相关工作与启发

- **U-Net [MICCAI 2015]**：医学图像分割的奠基性工作，本文基线之一
- **SAM [ICCV 2023]**：通用分割基础模型，已有多项医学适配工作
- **VWFormer [ICLR 2024]**：多尺度窗口注意力语义分割模型，本文最佳 GP-VM
- **Moglia et al. [2026]**：通用模型在 MIS 中的综述，但依赖原始论文的非标准化指标
- 启发：未来工作可将本框架扩展至更多模态和更严格的 OOD 评估设计

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 综合评价 | ⭐⭐⭐⭐ |
---
description: "大规模受控实验研究通用视觉模型(GP-VMs)与专用医学分割架构(SMAs)在2D医学图像分割任务上的系统性对比"
tags:
- medical_image_segmentation
- empirical_study
- vision_models
- benchmarking
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**会议**: CVPR 2026  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**代码**: [GitHub](https://github.com/)  
**领域**: 医学图像分割  
**关键词**: medical image segmentation, general-purpose vision models, empirical study, benchmarking, Grad-CAM

## 一句话总结

通过在三个异构医学数据集上对 11 种架构进行标准化对比实验，证明了通用视觉模型 (GP-VMs) 在 2D 医学图像分割中可以超越大多数专用医学分割架构 (SMAs)，且 XAI 分析表明 GP-VMs 无需特定领域设计也能捕获临床相关结构。

## 研究背景与动机

医学图像分割 (MIS) 是计算机辅助诊断和临床决策支持系统的基础组件。过去十年中，大量针对医学影像特定挑战（低对比度、小解剖结构、有限标注数据）的专用架构不断涌现，如 HiFormer、MISSFormer、U-KAN、Swin-UMamba 等。与此同时，通用视觉模型 (GP-VMs) 在自然图像上取得了显著进展。

然而，现有研究缺乏 **标准化的受控实验** 来公平比较两类模型：不同论文使用不同的数据集、预处理流水线、增强策略和评估流程，导致性能差异可能来自实验设计而非架构本身的优势。

本文的核心问题是：**医学分割任务是否真的需要专用架构，还是通用视觉模型就能达到甚至超越其性能？**

## 方法详解

### 整体框架

本文并非提出新架构，而是建立了一个 **标准化基准测试框架**，在统一的训练和评估协议下，横跨三个异构医学数据集，对 11 种架构进行全面对比。

### 关键设计

1. **模型选择策略**：选取两大类共 11 种模型
    - **专用医学分割架构 (SMA)**：U-Net (31M, CNN)、HiFormer-B (26M, CNN-ViT 混合)、MISSFormer (42M, Transformer)、Swin-UMamba (60M, State-space 混合)、U-KAN-L (25M, CNN+KAN 混合)
    - **通用视觉模型 (GP-VM)**：SegFormer-B3 (47M)、SegNeXt-L (49M)、VWFormer (两个 backbone：MiT-B3 和 ConvNeXt-S)、InternImage-T + UPerHead (58M)、TransNeXt-Tiny + UPerHead (58M)
    - 选择标准：架构多样性、计算规模可比、学术可见度、代码可用性

2. **异构数据集覆盖**：三个数据集涵盖不同成像模态和任务特征
    - ISIC'18：RGB 皮肤镜检查，二分类病变分割，3565 张
    - BKAI-IGH NeoPolyp Small：RGB 内窥镜，三分类息肉分割，945 张
    - CAMUS：灰度超声心动图，四分类心脏区域分割，1996 张

3. **标准化训练协议**：所有模型使用统一训练设置
    - ImageNet 预训练编码器，$512 \times 512$ 输入分辨率
    - AdamW + REX 学习率调度器，batch size 8
    - 学习率搜索范围 $\{10^{-4}, 5 \times 10^{-5}, 10^{-5}\}$
    - 5 折交叉验证，150 epoch 训练，统一早停条件
    - 数据增强按数据集定制但所有模型一致

### 损失函数 / 训练策略

- ISIC'18 使用二元交叉熵损失
- 多分类任务 (NeoPolyp, CAMUS) 使用交叉熵损失
- 混合精度训练 (除 SegNeXt 因不稳定外)
- 评估指标：mIoU、mDSC、mRec、mPrec（不含背景类，全局微平均）
- 使用 Grad-CAM 可视化进行可解释性分析

## 实验关键数据

### 主实验

| 模型 | 类型 | NeoPolyp mDSC | CAMUS mDSC | ISIC'18 mDSC | 平均 mDSC |
|------|------|:---:|:---:|:---:|:---:|
| VW-MiT | GP-VM | 89.7 | 91.4 | 91.8 | **91.0** |
| VW-Conv | GP-VM | 89.6 | 91.3 | 91.8 | **90.9** |
| TransNeXt | GP-VM | 89.4 | 91.7 | 91.7 | **90.9** |
| InternImage | GP-VM | 89.6 | 91.2 | 91.5 | **90.8** |
| SegFormer | GP-VM | 89.1 | 91.4 | 91.7 | **90.7** |
| SegNeXt | GP-VM | 89.2 | 91.3 | 91.5 | **90.7** |
| SU-Mamba | SMA | 88.9 | 91.3 | 91.3 | 90.5 |
| HiFormer | SMA | 84.6 | 90.8 | 91.0 | 88.8 |
| U-KAN | SMA | 82.5 | 90.5 | 90.6 | 87.9 |
| MISSFormer | SMA | 82.9 | 90.4 | 90.3 | 87.9 |
| U-Net | SMA | 83.3 | 89.1 | 89.3 | 87.2 |

### 消融实验 / 关键对比

| 指标维度 | GP-VMs 表现 | SMAs 表现 | 差距分析 |
|----------|:---:|:---:|------|
| NeoPolyp 非肿瘤息肉(C1) | 62.4-66.1% | 34.9-59.2% | GP-VMs 领先最大（7+ pp） |
| CAMUS 左心室壁(C2) | 87.7-88.4% | 85.6-88.3% | 差距较小（≈1-2 pp） |
| ISIC'18 整体 | 91.5-91.8% | 89.3-91.3% | GP-VMs 稳定优势 |
| Grad-CAM 可解释性 | 捕获临床相关区域 | 类似表现 | GP-VMs 无需特定设计也有良好关注 |

### 关键发现

- **GP-VMs 全面领先**：按平均 mDSC 排名，前 6 名全部是 GP-VMs
- **性能差异与数据集相关**：NeoPolyp 上差距最大（GP-VM vs SMA 可达 7+ pp），CAMUS 上差距较小（≈1-2 pp）
- **SU-Mamba 是最强 SMA**：在 SMA 中稳定占优，但仍略低于最佳 GP-VMs
- **GP-VMs 的 XAI 表现良好**：Grad-CAM 分析显示 GP-VMs 能在没有领域特定设计的情况下捕获临床相关结构

## 亮点与洞察

- **实验设计的严谨性**：在相同预处理、增强、优化设置下对比，消除了实验设计混淆因素，这在 MIS 领域相当稀缺
- **实用启示**：在引入新的专用架构之前，应先系统评估现有 GP-VMs 的性能
- **资源节约视角**：当 GP-VMs 已经具备竞争力时，研究精力可转向数据整理、训练协议优化和 OOD 泛化评估

## 局限与展望

- 仅覆盖三个 2D 数据集，无法完全代表临床影像的多样性（如 3D、CT、MRI 等）
- 部分模型参数量不完全可比（如 U-KAN 无 ImageNet 预训练）
- 未评估 OOD 泛化能力（如在 Kvasir-SEG 上测试 NeoPolyp 训练模型）
- 未涉及 3D 医学图像分割和半监督/少样本场景

## 相关工作与启发

- **SAM 在医学影像中的应用**：SAM 已被适配到医学场景，但本文指出即使不用 SAM 级别的基础模型，标准 GP-VMs + fine-tuning 就很有效
- **与 Moglia et al. (2026) 综述对比**：该综述也发现通用模型表现优异，但依赖了不同论文的原始结果，本文通过标准化协议提供了更可靠的证据
- 对医学影像社区的启发：资源有限时优先考虑成熟的 GP-VMs（如 InternImage、VWFormer），而非从零开始设计专用架构

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体评价 | ⭐⭐⭐⭐ |
