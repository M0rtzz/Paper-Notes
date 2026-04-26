---
title: >-
  [论文解读] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection
description: >-
  [CVPR 2025][医学图像][分布外检测] 本文提出了 OpenMIBOOD，一个专为医学影像设计的 OOD 检测综合基准框架，包含来自组织病理、内窥镜和脑部 MRI 三个医学领域的 14 个数据集，评估了 24 种 post-hoc 方法，揭示了自然图像 OOD 基准的结论无法直接迁移到医学场景。
tags:
  - CVPR 2025
  - 医学图像
  - 分布外检测
  - 医学影像
  - OOD基准
  - 后验方法
  - 可信AI
---

# OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection

**会议**: CVPR 2025  
**arXiv**: [2503.16247](https://arxiv.org/abs/2503.16247)  
**代码**: https://github.com/remic-othr/OpenMIBOOD (有)  
**领域**: 医学图像  
**关键词**: 分布外检测, 医学影像, OOD基准, 后验方法, 可信AI

## 一句话总结
本文提出了 OpenMIBOOD，一个专为医学影像设计的 OOD 检测综合基准框架，包含来自组织病理、内窥镜和脑部 MRI 三个医学领域的 14 个数据集，评估了 24 种 post-hoc 方法，揭示了自然图像 OOD 基准的结论无法直接迁移到医学场景。

## 研究背景与动机

**领域现状**：OOD (Out-of-Distribution) 检测是保障 AI 系统可信度的关键环节。自 2016 年以来，大量 OOD 检测方法涌现，OpenOOD 框架提供了统一的评估标准，但这些基准主要面向自然图像（如 ImageNet）。

**现有痛点**：医学影像领域缺乏系统性的 OOD 评估基准。现有少量医学 OOD 研究存在明显局限：评估方法数量有限、数据集选择不够全面、缺乏对 covariate-shifted ID（cs-ID）的系统评估。Cao et al. 的工作仅覆盖 8 种 post-hoc 方法，且部分评估场景使用自然图像，与医学场景脱节。

**核心矛盾**：在自然图像基准上表现优异的 OOD 方法，未必能在医学影像中同样有效。医学图像具有低方差、特定语义偏移（如不同扫描仪、不同染色方案）等独特特征，导致基于分类概率的方法（logits/softmax）在医学场景下远不如基于特征空间的方法有效。

**本文目标**：构建一个覆盖多个医学领域、包含精细 OOD 分级（cs-ID、near-OOD、far-OOD）的标准化基准，评估大量 post-hoc OOD 检测方法，为医学场景 OOD 检测研究提供可靠参考。

**切入角度**：沿用 OpenOOD 的分类法但做关键修改——将 cs-ID 独立出来而非与 ID 合并，因为在医学场景中区分 ID 和 cs-ID 同样重要（如不同扫描仪采集的图像）。

**核心 idea**：用三个医学领域（组织病理 MIDOG、内窥镜 PhaKIR、脑部 MRI OASIS3）的 14 个数据集，对 24 种 post-hoc 方法进行标准化的全面评估。

## 方法详解

### 整体框架
OpenMIBOOD 不是提出新的 OOD 检测算法，而是一个 benchmark 框架。其核心流程为：(1) 构建三个医学 benchmark，每个包含 ID、cs-ID、near-OOD、far-OOD 四层数据；(2) 为每个 benchmark 训练一个分类器；(3) 在分类器上运行 24 种 post-hoc OOD 检测方法并对比性能。

### 关键设计

1. **三层 OOD 分级体系**:

    - 功能：将域偏移按严重程度分为 cs-ID、near-OOD 和 far-OOD
    - 核心思路：cs-ID 指标签不变但输入特征分布变化（如不同扫描仪）；near-OOD 指语义相似但存在显著差异（如不同手术器械类型）；far-OOD 指完全不同的医学应用（如用内窥镜训练的模型遇到眼科图像）
    - 设计动机：医学场景中各层偏移的检测难度和临床意义不同，需要分开评估才能精确指导模型部署

2. **三个互补的医学基准**:

    - 功能：覆盖组织病理（MIDOG）、内窥镜（PhaKIR）和脑 MRI（OASIS3）三个领域
    - 核心思路：MIDOG 包含 10 个域的有丝分裂细胞分类，域偏移来自不同扫描仪、染色方案和物种（人/犬）；PhaKIR 包含胆囊切除术器械分类，域偏移来自烟雾遮挡、不同手术和不同手术类型；OASIS3 包含认知正常 vs 阿尔茨海默症分类，域偏移来自不同模态（T1w→T2w）、不同扫描仪和不同解剖区域
    - 设计动机：不同医学领域的图像特征差异巨大（2D 病理 vs 2D 内窥镜 vs 3D MRI），需要多领域评估才能得出可靠结论

3. **标准化评估协议**:

    - 功能：统一方法实现、超参数调优和评估指标
    - 核心思路：所有 24 种方法基于 OpenOOD 代码库实现；用 near-OOD 验证集做超参数选择；报告 AUROC、FPR@95 和 AUPRIN/AUPROUT 的调和平均值；按信息来源将方法分为分类型（蓝）、特征型（橙）、混合型（绿）三类
    - 设计动机：统一实验条件是公平比较的前提，调和平均值避免了在数据不平衡时某一指标被高估

### 损失函数 / 训练策略
分类器训练使用加权交叉熵损失函数（针对类别不平衡），配合 OneCycle 学习率调度器。MIDOG 和 PhaKIR 使用 ImageNet-1k 预训练权重，OASIS3 使用 Kinetics400 预训练的 R(2+1)D 模型。

## 实验关键数据

### 主实验

| 方法 | MIDOG nOOD | PhaKIR nOOD | OASIS3 nOOD | 平均 nOOD AUROC | 类型 |
|------|-----------|-------------|-------------|----------------|------|
| MDSEns | 91.84 | 97.11 | 99.46 | 96.14 | 特征 |
| ViM | 62.67 | 81.14 | 98.40 | 80.74 | 混合 |
| Residual | 65.78 | 76.99 | 96.70 | 79.82 | 特征 |
| MDS | 63.21 | 76.48 | 96.15 | 78.61 | 特征 |
| KNN | 61.63 | 55.44 | 97.66 | 71.58 | 特征 |
| MSP | 55.90 | 50.16 | 53.50 | 53.19 | 分类 |
| EBO | 56.85 | 40.18 | 49.39 | 48.81 | 分类 |

### 方法类型性能对比

| 方法类型 | MIDOG 平均 | PhaKIR 平均 | OASIS3 平均 |
|---------|-----------|-------------|-------------|
| 特征型 | 66.08 | 70.68 | 92.08 |
| 混合型 | 57.18 (-13%) | 50.71 (-28%) | 69.86 (-24%) |
| 分类型 | 55.81 (-16%) | 49.45 (-30%) | 52.13 (-43%) |

### 关键发现
- **特征型方法全面碾压分类型方法**：在所有三个医学基准上，基于特征空间的方法平均 AUROC 显著高于基于 logits/softmax 的方法。这可能是因为医学图像方差低（MIDOG/PhaKIR 的平均像素强度标准差仅 0.148/0.149，远低于 ImageNet 的 0.226），导致特征空间更紧凑，更适合基于距离的 OOD 检测。
- **MDSEns 的高性能来自 covariate shift 检测**：MDSEns 利用了网络所有中间层的 Mahalanobis 距离，而浅层更容易捕捉低级视觉特征的变化（如边缘、颜色）。在 MIDOG domain 5（仅有语义偏移无 covariate 偏移）上 MDSEns 性能严重下降至 71.50%，但在 domain 6a（同时有语义和 covariate 偏移）上达到 98.95%。
- **自然图像基准的最优方法≠医学场景最优**：将方法在 ImageNet-1k 和 OpenMIBOOD 上的排名对比发现，两者之间没有明显的正相关。在 ImageNet 上排名靠前的分类型方法在医学场景下表现最差。

## 亮点与洞察
- **cs-ID 独立评估的设计很有价值**：在医学场景中，不同扫描仪采集的同类图像（cs-ID）可能导致模型失效，将其与 ID 合并会掩盖这一风险。这个设计思路可以迁移到其他安全敏感的 AI 部署场景。
- **分类器过度自信现象的发现**：PhaKIR 基准中，EndoSeg18 的 OOD 器械被高置信度地分类为 Grasper，因为特征空间中 OOD 样本聚集在 Grasper 类附近。这说明分类型 OOD 方法在医学场景的局限性本质上是分类器的过度自信。
- **评估指标选择的讲究**：用 AUPRIN 和 AUPROUT 的调和平均值取代单一指标，避免了数据不平衡导致的偏差。这个做法适用于所有类别不平衡的 OOD 评估场景。

## 局限与展望
- **仅关注分类任务**：benchmark 仅覆盖分类场景，未涉及分割任务——后者在医学影像中更为常见和重要
- **仅评估 post-hoc 方法**：未涵盖需要额外训练步骤的方法（如 outlier exposure），尽管作者引用 OpenOOD 的结论认为 post-hoc 方法不弱于训练型方法
- **分类器架构固定**：每个基准仅使用一种分类器架构（ResNet50/ResNet18/R(2+1)D），未探究架构选择对 OOD 检测方法排名的影响
- **未探究 foundation model 的潜力**：随着医学影像 foundation model（如 BiomedCLIP）的发展，基于这些模型的 OOD 检测方法可能表现更优

## 相关工作与启发
- **vs OpenOOD**: OpenOOD 面向自然图像，将 cs-ID 与 ID 合并评估。本文为医学场景定制了独立的 cs-ID 评估，更适合安全敏感部署
- **vs MOOD Challenge**: MOOD 使用合成图像损坏作为 OOD，而本文使用真实的医学域偏移，更贴近临床实际
- **vs Cao et al.**: Cao et al. 仅覆盖 8 种方法和 3 个医学场景，且部分使用自然图像。本文规模大得多（24 种方法、14 个数据集），结论更可靠

## 评分
- 新颖性: ⭐⭐⭐ 核心贡献是 benchmark 构建而非新算法，但 cs-ID 独立评估的设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 24 种方法、14 个数据集、多种指标，是目前医学 OOD 检测最全面的评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，附录中数据集描述非常详细，但主文部分 LaTeX 宏渲染问题较多
- 价值: ⭐⭐⭐⭐ 揭示了自然图像基准结论不可直接迁移的关键发现，对医学 AI 部署有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] DIsoN: Decentralized Isolation Networks for Out-of-Distribution Detection in Medical Imaging](../../NeurIPS2025/medical_imaging/dison_decentralized_isolation_networks_for_out-of-distribution_detection_in_medi.md)
- [\[ICML 2025\] Out-of-Distribution Detection Methods Answer the Wrong Questions](../../ICML2025/medical_imaging/out-of-distribution_detection_methods_answer_the_wrong_questions.md)
- [\[CVPR 2026\] The Invisible Gorilla Effect in Out-of-distribution Detection](../../CVPR2026/medical_imaging/the_invisible_gorilla_effect_in_out-of-distribution_detection.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)

<!-- RELATED:END -->
