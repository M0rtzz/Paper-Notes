---
title: >-
  [论文解读] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis
description: >-
  [AAAI 2026][医学图像][X射线血管造影] 提出 VasoMIM，一个针对X射线血管造影的领域特定自监督预训练框架：通过解剖引导的掩码策略优先遮挡血管区域 + 解剖一致性损失保持重建图像的血管拓扑结构，结合构建的最大规模XA-170K预训练数据集，在4个下游任务6个数据集上全面超越通用SSL方法和医学SSL方法（包括在16.9亿图像上预训练的DINOv3）。
tags:
  - AAAI 2026
  - 医学图像
  - X射线血管造影
  - 自监督学习
  - 掩码图像建模
  - 血管解剖感知
  - 基础模型
  - 血管分割
  - 狭窄检测
---

# Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis

**会议**: AAAI 2026  
**arXiv**: [2602.11536](https://arxiv.org/abs/2602.11536)  
**代码**: [GitHub](https://github.com/Dxhuang-CASIA/XA-SSL)  
**领域**: 医学图像 / 自监督预训练  
**关键词**: X射线血管造影, 自监督学习, 掩码图像建模, 血管解剖感知, 基础模型, 血管分割, 狭窄检测

## 一句话总结
提出 VasoMIM，一个针对X射线血管造影的领域特定自监督预训练框架：通过解剖引导的掩码策略优先遮挡血管区域 + 解剖一致性损失保持重建图像的血管拓扑结构，结合构建的最大规模XA-170K预训练数据集，在4个下游任务6个数据集上全面超越通用SSL方法和医学SSL方法（包括在16.9亿图像上预训练的DINOv3）。

## 研究背景与动机

**领域现状**：心血管疾病是全球第一大死因，X射线血管造影是诊断的金标准。深度学习方法（UNet、Faster R-CNN等）在血管分割和狭窄检测上取得进展，但严重受限于标注数据稀缺。自监督学习（SSL）是解决方案，但该领域缺乏专用的SSL框架和大规模数据集。

**现有痛点**：
   - **通用MIM的掩码策略不适用**：血管造影中血管结构极其稀疏（仅占图像很小比例），随机/注意力引导/损失引导的掩码策略会大量遮挡背景区域，导致模型学习重建背景而非血管
   - **像素级重建目标缺乏语义判别性**：MSE损失鼓励预测低频背景纹理，而非高频血管细节
   - **缺乏大规模数据集**：不同于胸片（CheXpert 22万+）和CT，血管造影领域没有大规模预训练数据集
   - **通用视觉基础模型（如DINOv3）在跨域时性能不佳**：在自然图像上预训练的模型缺乏血管造影特定的解剖语义

**核心 idea**：在MIM中注入强解剖归纳偏置——让模型知道"哪里是血管"，强制其学习重建血管区域。

## 方法详解

### 整体框架（VasoMIM）

输入X射线血管造影 → Frangi滤波器提取血管解剖 + UNeXt-S分割器生成概率图 → 联合引导（co-guidance）→ 解剖引导的掩码策略 → ViT编码器+解码器重建 → 像素重建损失 $\mathcal{L}_{rec}$ + 解剖一致性损失 $\mathcal{L}_{cons}$

### 关键设计

1. **Frangi滤波器提取血管解剖**：

    - 多尺度Hessian分析（σ=1,2,3,4）检测管状结构
    - 自适应百分位阈值化（α=92）生成粗二值掩码
    - 区域生长去除孤立伪影，得到最终二值血管掩码 $B \in \{0,1\}^{1 \times H \times W}$

2. **解剖引导的掩码策略**：

    - **共同引导(co-guidance)**：结合Frangi滤波器掩码 $B$ 和UNeXt-S分割概率图 $M$：
    $G = \eta \cdot B + (1-\eta) \cdot M, \quad \eta=0.5$
    - UNeXt-S 可弥补 Frangi 滤波器对低对比度细小血管的遗漏
    - **Patch级采样概率**：$f(g_i) = \frac{\sum_j g_{ij}}{\sum_k \sum_j g_{kj}}$，血管密度大的patch被遮挡概率更高
    - **由弱到强引导**：早期阶段混入更多随机掩码（避免过早优化困难），后期逐渐增加解剖引导比例
    $\beta_e = \beta_0 + \frac{e}{E}(\beta_E - \beta_0)$
    - 每个epoch：$\beta_e \gamma N$ 个patch按解剖引导采样，$(1-\beta_e)\gamma N$ 个随机采样

3. **解剖一致性损失**：

    - 核心思想：原始图像和重建图像通过同一个分割器，分割结果应一致
    $\mathcal{L}_{cons} = \mathcal{L}_{CE}(\mathcal{S}(I), \mathcal{S}(I'))$
    - 使用轻量级UNeXt-S（仅0.26M参数）作为可微分代理（Frangi滤波器不可微）
    - 保证模型学到的是拓扑准确的血管表示，而非仅仅像素强度

4. **总训练目标**：
    $\mathcal{L}_{MIM} = \mathcal{L}_{rec} + \mathcal{L}_{cons}$

### XA-170K 数据集

从4个公开数据源收集177,478张X射线血管造影图像：
- CADICA：42个患者，6,594帧
- SYNTAX：231个患者，2,943张
- XCAD：1,621张
- CoronaryDominance：1,574个患者，160,320张（主要来源）

## 实验

### 下游任务与数据集
- **血管分割**：ARCADE-V, CAXF, XCAV（DSC + clDice）
- **血管段分割**：ARCADE-VS（DSC）
- **狭窄分割**：ARCADE-S（DSC）
- **狭窄检测**：Stenosis（mAP50, mAP75, mAP）

### 主实验：分割任务

| 方法 | 预训练数据 | ARCADE-V DSC | ARCADE-V clDice | XCAV DSC | ARCADE-S DSC | ARCADE-VS DSC | 平均排名 |
|------|-----------|-------------|-----------------|----------|-------------|--------------|---------|
| UNet (scratch) | - | 71.44 | 70.67 | 78.18 | 27.04 | 38.77 | 22.00 |
| MAE | XA-170K | 79.39 | 80.74 | 84.84 | 51.72 | 56.69 | 4.88 |
| DINOv3 | LVD-1698M | 79.36 | 80.90 | 82.76 | 53.57 | 54.36 | 7.25 |
| DeblurringMIM | XA-170K | 79.25 | 80.77 | 85.38 | 51.70 | 56.66 | 4.38 |
| RAD-DINO | LVD-142M+CXR-838K | 78.96 | 80.26 | 84.88 | 51.55 | 54.81 | 6.62 |
| VasoMIM-v1 | XA-170K | 79.90 | 81.57 | 85.80 | 54.52 | 58.03 | 2.12 |
| **VasoMIM** | **XA-170K** | **80.25** | **82.06** | **86.09** | **55.62** | **58.87** | **1.00** |

**关键发现**：
- VasoMIM 在所有指标上取得最佳，平均排名1.00（满分）
- 对UNet提升巨大：ARCADE-S DSC +28.58，ARCADE-VS DSC +20.10
- **领域特定预训练 > 通用大规模预训练**：DINOv3在16.9亿自然图像上预训练，但不及VasoMIM在17万血管造影上的预训练
- VasoMIM比VasoMIM-v1（会议版）进一步提升（p=1.18×10⁻⁴，配对t检验）

### 狭窄检测任务

| 方法 | mAP50 | mAP75 | mAP |
|------|-------|-------|-----|
| Faster R-CNN (scratch) | 88.37 | 19.01 | 36.63 |
| MAE | 92.30 | 24.28 | 39.69 |
| DINOv3 | 93.89 | 23.60 | 40.90 |
| VasoMIM-v1 | 94.25 | 25.01 | 40.91 |
| **VasoMIM** | **94.91** | **25.72** | **41.07** |

### 消融实验

**解剖引导掩码 + 解剖一致性损失的独立贡献**：

| 引导 | $\mathcal{L}_{cons}$ | ARCADE-V DSC | XCAV DSC |
|------|---------------------|-------------|----------|
| ✗ | ✗ | 79.31 | 84.52 |
| ✗ | ✓ | 79.85 (+0.54) | 85.79 (+1.27) |
| ✓ | ✗ | 79.87 (+0.56) | 85.92 (+1.40) |
| **✓** | **✓** | **80.25 (+0.94)** | **86.09 (+1.57)** |

- 两个组件各自独立有效，组合后进一步提升（超加性增益）
- 解剖引导掩码对XCAV提升更大（+1.40 vs +1.27），因该数据集血管更稀疏

**掩码引导可视化分析**：
- 基线（MAE随机掩码）：仅5-10%掩码patch含血管
- VasoMIM：随训练进行，血管区域掩码比例逐渐从~20%增加到~70%
- 共同引导比单一Frangi引导更准确，可捕获Frangi遗漏的低对比度血管分支

**与替代重建目标对比**：解剖一致性损失使用UNeXt-S（0.26M参数）远比使用DINOv2（86M参数）蒸馏更轻量，性能相当或更好

## 亮点与洞察

1. **领域知识驱动设计**：Frangi滤波器是血管分析的经典方法，将其作为解剖先验注入MIM框架，是"传统方法+深度学习"结合的典范
2. **"小数据胜大数据"的反直觉发现**：17万张领域数据 > 16.9亿张自然图像（DINOv3），强调了领域适配的重要性
3. **共同引导策略的互补性**：Frangi滤波器提供硬边缘但会遗漏细管，UNeXt-S提供软概率图弥补遗漏
4. **由弱到强的课程学习**：避免早期遮挡太多血管导致优化困难，体现了良好的工程直觉
5. **解剖一致性损失的巧妙设计**：用仅0.26M的轻量分割器替代不可微的Frangi滤波器，计算开销极小
6. **数据集贡献**：XA-170K是该领域最大的预训练数据集，将公开供使用

## 局限性

1. **Frangi滤波器的局限**：对噪声敏感、可能将骨骼结构误判为血管（虽然共同引导部分缓解）
2. **仅使用ViT-B/16作为backbone**：未验证更大模型（ViT-L/H）是否能进一步提升
3. **下游任务使用全量微调**：未探索参数高效微调（如LoRA），在更极端小样本场景下表现未知
4. **数据集规模仍有限**：17万张虽为该领域最大，但与胸片/CT的百万级预训练数据相比仍小
5. **仅针对冠状动脉造影**：未验证是否适用于其他血管造影场景（如脑血管、外周血管）
6. **UNeXt-S分割器用Frangi伪标签训练**：分割质量受限于Frangi滤波器质量，可能引入系统性偏差

## 相关工作

- **通用SSL**：MAE, SimMIM, DINO, iBOT, I-JEPA, DINOv3
- **医学SSL**：Model Genesis, LVM-Med, DeblurringMIM, RAD-DINO, CheXWorld, MedDINOv3
- **MIM掩码策略**：AMT（注意力引导）, HPM（损失引导）, AnatoMask, HAP（人体先验）
- **血管造影分析**：ARCADE数据集, XCAD, 基于UNet/Faster R-CNN的传统有监督方法

## 评分 ⭐⭐⭐⭐⭐

方法设计精巧、动机清晰、领域知识融合到位。实验极其全面——4个下游任务、6个数据集、20+个基线对比、详细消融。数据集贡献和scaling law验证进一步增加价值。是医学图像自监督预训练领域的高质量工作。

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[AAAI 2026\] Self-supervised Multiplex Consensus Mamba for General Image Fusion](self-supervised_multiplex_consensus_mamba_for_general_image_fusion.md)
- [\[AAAI 2026\] MIRNet: Integrating Constrained Graph-Based Reasoning with Pre-training for Diagnostic Medical Imaging](mirnet_integrating_constrained_graph-based_reasoning_with_pre-training_for_diagn.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](../../CVPR2026/medical_imaging/ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)

<!-- RELATED:END -->
