---
title: >-
  [论文解读] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
description: >-
  [CVPR 2026][医学图像][医学图像分割] 在统一训练评估协议下对比11个模型（5个专用医学分割架构SMA + 6个通用视觉模型GP-VM）在3个异构医学数据集上的表现，发现GP-VMs在所有数据集上系统性优于大多数SMAs（平均mDSC: VW-MiT 91.0% vs 最佳SMA SU-Mamba 90.5%），且Grad-CAM分析表明GP-VMs能捕获临床相关结构。
tags:
  - CVPR 2026
  - 医学图像
  - 医学图像分割
  - 通用视觉模型
  - 实证对比
  - 可解释性
  - Grad-CAM
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**会议**: CVPR 2026  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**代码**: GitHub（论文提及，具体链接见论文）  
**领域**: 医学图像  
**关键词**: 医学图像分割, 通用视觉模型, 实证对比, 可解释性, Grad-CAM

## 一句话总结

在统一训练评估协议下对比11个模型（5个专用医学分割架构SMA + 6个通用视觉模型GP-VM）在3个异构医学数据集上的表现，发现GP-VMs在所有数据集上系统性优于大多数SMAs（平均mDSC: VW-MiT 91.0% vs 最佳SMA SU-Mamba 90.5%），且Grad-CAM分析表明GP-VMs能捕获临床相关结构。

## 研究背景与动机

**领域现状**：医学图像分割（MIS）自U-Net以来涌现了大量专用架构，包括HiFormer（CNN-ViT混合）、MISSFormer（纯Transformer）、Swin-UMamba（状态空间模型）、U-KAN（KAN集成）等，各自针对医学图像的低对比度、小结构、标注稀缺等挑战设计了专用机制。与此同时，SegFormer、SegNeXt、ConvNeXt、InternImage、TransNeXt等通用视觉模型在自然图像语义分割基准上已取得强大性能。

**现有痛点**：文献中的性能对比通常存在严重的混淆因素——不同论文使用不同的数据集、预处理流程、数据增强策略、优化器设置和评估协议，导致性能差异可能源于实验设计选择而非架构本身的优劣。因此，"专用架构是否真的优于通用模型"这个基本问题缺乏可靠答案。

**核心矛盾**：医学影像社区的默认假设是"医学图像的特殊性要求专用架构设计"，但通用模型受益于大规模ImageNet预训练和百万级图像验证的优化经验，其学到的通用视觉特征可能已经足够强大。这两种理念之间的矛盾缺乏受控实证检验。

**本文目标** (1) 在消除所有混淆因素的统一协议下，专用医学分割架构是否比通用视觉模型有系统性优势？(2) 通用模型不仅精度高，其决策逻辑是否也与临床知识一致（通过XAI验证）？(3) 结论是否在不同成像模态和任务设置下具有普遍性？

**切入角度**：采用严格控制的实证研究方法论——统一ImageNet预训练、512×512输入分辨率、AdamW+REX调度器、相同数据增强和早停策略，对每个模型-数据集对进行学习率搜索，并使用5折交叉验证减少随机性。选择3个在模态（皮肤镜RGB/内窥镜RGB/超声灰度）、类别结构（二分类/多分类）和数据特性上存在异构性的数据集。

**核心 idea**：在严格受控的benchmarking框架下，用实证数据证明通用视觉模型已能替代大多数专用医学分割架构，领域特定设计的边际收益可能被高估。

## 方法详解

### 整体框架

这是一项纯实证对比研究，不提出新方法。研究框架分三层：(1) 模型选择——从SMA和GP-VM两大家族中各选代表性架构，确保涵盖CNN/ViT/混合/状态空间/KAN等多种范式；(2) 统一训练评估——在ISIC'18（皮肤病变二分类）、NeoPolyp（息肉多分类）、CAMUS（心脏超声多分类）3个数据集上用同一协议训练所有模型；(3) 多维分析——除分割精度外，用Grad-CAM可解释性分析揭示模型的注意力模式。

### 关键设计

1. **统一Benchmarking协议（消除混淆因素）**:

    - 功能：确保不同架构之间的对比公平、可信
    - 核心思路：所有模型统一使用ImageNet预训练编码器、512×512输入分辨率、AdamW优化器+REX学习率调度器、batch size 8。对每个模型-数据集对，在 $\{10^{-4}, 5\times10^{-5}, 10^{-5}\}$ 中搜索最优学习率，训练100 epoch后选最优，再用150 epoch的5折交叉验证评估。数据集特定的损失函数（ISIC'18用BCE，多分类用CE）和数据增强在同一数据集所有模型间完全一致
    - 设计动机：文献中大量对比因训练设置不统一而不可信——augmentation策略、学习率、epoch数等差异可能导致2-5%的性能波动，数量级与架构差异相当

2. **异构数据集覆盖（验证结论普遍性）**:

    - 功能：选择3个在模态、类别结构和数据特性上高度异构的数据集
    - ISIC'18：皮肤镜RGB图像，3565张，二分类（病变/背景），特点是边界不规则
    - NeoPolyp (BKAI-IGH)：内窥镜RGB图像，945张，3类（非肿瘤性/肿瘤性息肉+背景），特点是亚型间变异大
    - CAMUS：心脏超声灰度图像，1996张，4类（左心室/左室壁/左心房+背景），特点是超声噪声大
    - 设计动机：覆盖RGB/灰度、二分类/多分类、皮肤/消化道/心脏等异构场景，避免单一数据集的偏倚。数据集还进行了重复图像和高相似图像的过滤，CAMUS使用患者级别分组防止数据泄漏

3. **Grad-CAM可解释性分析（超越精度的评估）**:

    - 功能：分析模型注意力区域是否与临床相关结构对应
    - 核心思路：使用M3d-CAM实现的Grad-CAM，自动选择最后一层合适的层生成注意力热图。在每折50个最差样本中可视化各模型的注意力分布，评判模型是否"看到了正确的东西"
    - 设计动机：在安全关键的医学应用中，高精度但关注错误区域的模型是危险的。XAI分析为GP-VM的可靠性提供了额外证据

### 损失函数 / 训练策略

ISIC'18使用二元交叉熵损失（BCE），NeoPolyp和CAMUS使用多分类交叉熵损失（CE）。所有模型使用AdamW优化+REX学习率调度。评估指标包括mDSC（主指标）、mIoU、mRecall、mPrecision，均为global micro-averaging去除背景类后计算。训练环境为2×NVIDIA A100 GPU，使用混合精度训练。

## 实验关键数据

### 主实验

3个数据集上的5折交叉验证结果（mDSC%，均值±标准差，**加粗**表示该家族最优）：

| 模型 | 类型 | 参数量 | ISIC'18 mDSC | NeoPolyp mDSC | CAMUS mDSC | 三数据集均值 |
|------|------|--------|-------------|--------------|-----------|------------|
| VW-MiT | GP-VM | 51M | 91.7±0.5 | **89.7±0.8** | 91.6±0.1 | **91.0** |
| VW-Conv | GP-VM | 57M | 91.5±0.4 | 89.6±1.3 | 91.4±0.1 | 90.9 |
| TransNeXt | GP-VM | 58M | **91.9±0.7** | 89.4±0.7 | 91.5±0.1 | 90.9 |
| InternImage | GP-VM | 58M | 91.3±0.4 | 89.6±1.1 | 91.4±0.2 | 90.8 |
| SegNeXt | GP-VM | 49M | 91.4±0.6 | 89.2±0.7 | **91.6±0.1** | 90.7 |
| SegFormer | GP-VM | 47M | 91.3±0.8 | 89.1±1.3 | 91.5±0.1 | 90.7 |
| **SU-Mamba** | **SMA** | **60M** | **91.3±0.5** | **88.9±0.6** | **91.3±0.3** | **90.5** |
| HiFormer | SMA | 26M | 91.0±0.6 | 84.6±0.9 | 90.8±0.2 | 88.8 |
| U-KAN | SMA | 25M | 89.2±1.1 | 82.5±1.7 | 90.5±0.2 | 87.4 |
| MISSFormer | SMA | 42M | 90.3±0.8 | 82.9±1.6 | 90.4±0.1 | 87.9 |
| U-Net | SMA | 31M | 89.0±0.9 | 83.3±1.1 | 89.1±0.3 | 87.1 |

### 消融实验

NeoPolyp数据集按类别的mDSC（最困难的数据集，差异最大）：

| 模型 | C1 非肿瘤性 mDSC | C2 肿瘤性 mDSC | 整体 mDSC |
|------|------------------|---------------|----------|
| VW-MiT (GP-VM) | **66.1±4.3** | 92.7±0.9 | 89.7 |
| InternImage (GP-VM) | 66.0±5.7 | 92.9±0.7 | 89.6 |
| SU-Mamba (SMA) | 59.2±3.8 | **92.5±0.6** | 88.9 |
| HiFormer (SMA) | 52.7±4.9 | 88.9±0.7 | 84.6 |
| U-KAN (SMA) | 36.9±12.2 | 87.1±0.9 | 82.5 |
| U-Net (SMA) | 34.9±18.2 | 88.1±0.6 | 83.3 |

### 关键发现

- **GP-VMs系统性优于SMAs**：按三数据集平均mDSC排名，前6名全部是GP-VMs（91.0-90.7%），最佳SMA SU-Mamba为90.5%。差距虽小（~0.5%），但在统一协议下稳定可复现
- **NeoPolyp数据集差异最大**：GP-VMs在多类息肉分割上优势明显——VW-MiT (89.7%) vs SU-Mamba (88.9%) vs HiFormer (84.6%)。核心原因是非肿瘤性息肉（C1）极难分割，GP-VMs的C1 mDSC达66%而多数SMAs仅35-53%，差距高达7个百分点
- **ISIC'18和CAMUS差距较小**：在皮肤病变和心脏超声上，GP-VMs和最佳SMA的差距缩小到约1-2%，说明简单二分类和大数据量场景下架构差异被弱化
- **Grad-CAM验证GP-VMs的临床合理性**：GP-VMs不仅精度更高，其注意力热图也更精确地聚焦在临床相关结构上。在ISIC'18案例中，GP-VMs比某些SMA更准确地关注病变区域；在CAMUS中，GP-VMs在左心房（C3）这一困难类别上检测到更多真正阳性
- **SU-Mamba是唯一接近GP-VMs的SMA**：其Mamba状态空间模型的长程依赖建模能力使其在SMA家族中遥遥领先，但仍未超越GP-VMs

## 亮点与洞察

- **逆直觉的实证发现具有纠偏价值**：挑战了"医学图像必须用专用架构"的社区共识。在受控对比下，通用模型的大规模预训练优势压过了专用设计的领域知识嵌入，启示研究者在提出新架构前应先验证通用模型的性能
- **Benchmarking方法论本身是贡献**：统一协议+异构数据集+XAI分析的三层评估框架，比仅看分割精度的传统对比更全面可信。消除混淆因素的方法论可推广到其他医学AI子领域
- **资源配置的实践启示**：如果GP-VMs已经足够好，研究者应将精力从架构创新转向数据策展、训练协议优化和OOD泛化评估——这些在临床部署中更具实际影响

## 局限与展望

- **仅3个2D数据集**：未覆盖CT/MRI等3D成像模态，专用架构（如nnU-Net）在3D分割中的优势可能更明显
- **预训练数据量不公平**：GP-VMs受益于ImageNet大规模预训练，而U-KAN无预训练权重只能从零训练，造成一定偏倚
- **计算公平性有限**：模型参数量从25M到60M不等，未严格控制FLOPs/推理速度
- **缺少Foundation Model对比**：SAM-Med2D、MedSAM等基于SAM的医学分割方法未纳入比较
- **未评估OOD泛化**：所有评估均为域内（in-domain），未检验GP-VMs在分布外数据上的鲁棒性，而这在临床场景中至关重要

## 相关工作与启发

- **vs nnU-Net**：nnU-Net作为医学分割黄金标准通过自适应框架（而非固定架构设计）取得强大性能，与本文"架构设计不如训练策略重要"的发现一致，但本文未直接对比nnU-Net
- **vs SAM/MedSAM**：SAM系列代表了另一条路线——用基础模型+提示的方式做医学分割。本文聚焦于标准fine-tuning范式的对比，SAM系列的zero-shot/few-shot能力是互补的研究方向
- **vs TransUNet/SwinUNet**：这些早期Transformer+U-Net混合架构试图引入全局注意力，但在本文实验中MISSFormer（纯Transformer SMA）表现平平（87.9%），说明Transformer架构本身不是关键，预训练质量和规模才是

## 评分

- 新颖性: ⭐⭐⭐ 实证对比研究不提出新方法，但揭示了社区长期忽视的重要发现
- 实验充分度: ⭐⭐⭐⭐ 11模型×3数据集×5折CV设计严谨，但缺少3D和OOD评估
- 写作质量: ⭐⭐⭐⭐ 研究问题明确，实验设计严谨，讨论客观承认局限性
- 价值: ⭐⭐⭐⭐ 对医学分割领域的模型选择策略和资源分配具有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mambabased_decoder_with.md)
- [Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)

<!-- RELATED:END -->
