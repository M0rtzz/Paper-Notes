---
title: >-
  [论文解读] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation
description: >-
  [CVPR 2025][医学图像][SAM] 本文提出一种增强的SAM框架，通过BiomedCLIP、VQA和GPT-4生成无监督语义/位置/形状提示，并引入DPO启发的偏好对齐损失模拟人类反馈，在仅10%标注数据的半监督设置下实现了肺部、乳腺肿瘤和腹部器官分割的优异性能。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "SAM"
  - "半监督分割"
  - "偏好优化"
  - "无监督提示"
  - "医学影像"
---

# Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation

**会议**: CVPR 2025  
**arXiv**: [2503.04639](https://arxiv.org/abs/2503.04639)  
**代码**: 无  
**领域**: LLM对齐  
**关键词**: SAM, 半监督分割, 偏好优化, 无监督提示, 医学影像

## 一句话总结
本文提出一种增强的SAM框架，通过BiomedCLIP、VQA和GPT-4生成无监督语义/位置/形状提示，并引入DPO启发的偏好对齐损失模拟人类反馈，在仅10%标注数据的半监督设置下实现了肺部、乳腺肿瘤和腹部器官分割的优异性能。

## 研究背景与动机
1. **领域现状**：SAM等基础模型已扩展到医学图像分割（SAM-Med2D、MedSAM），但仍依赖专家提供的几何提示（点/框），且需要大量标注数据。
2. **现有痛点**：(1)提示生成依赖人工，效率低下；(2)许多医学数据集缺乏全面标注，限制了数据密集型基础模型的利用；(3)现有人在环方法需要复杂的领域知识和独立的奖励函数训练。
3. **核心矛盾**：基础模型需要大量标注但医学标注成本极高；人在环反馈方法虽减少标注需求，但需要复杂的奖励建模，无法端到端训练。
4. **本文目标**：(1)设计无需人工的提示生成方案；(2)跳过奖励函数训练，用简单的偏好评分实现端到端对齐。
5. **切入角度**：利用BiomedCLIP/VQA/GPT-4提供语义+位置+通用信息的无监督提示；用DPO损失替代传统RLHF中的奖励函数训练。
6. **核心idea**：无监督多源提示 + DPO驱动的半监督偏好对齐。

## 方法详解

### 整体框架
Stage 1（10%标注数据微调）：输入图像 → SAM-Med2D编码器 + BiomedCLIP(显著图→框提示) + MedVInT(形状/位置文本) + GPT-4(通用疾病信息) → 提示编码器 → 掩码解码器 → 分割图。Stage 2（剩余无标注数据对齐）：多阈值生成4个分割候选 → 虚拟标注员评分 → DPO启发损失微调解码器。

### 关键设计

1. **多源无监督提示生成**:

    - 功能：无需人工干预，自动生成包含语义、位置和形状信息的综合提示。
    - 核心思路：(1)视觉提示：用BiomedCLIP+gScoreCAM生成显著图→CRF后处理→提取边界框和点坐标；(2)文本提示：MedVInT回答关于器官/肿瘤形状和位置的VQA问题（如"What is the shape of the liver?"）；(3)通用知识：GPT-4提供疾病/器官的通用描述。三种提示拼接输入提示编码器。
    - 设计动机：现有SAM方法的提示要么需要专家（点/框），要么仅用语义信息缺乏位置/形状信息。多源信息互补提供更强信号。

2. **DPO启发的偏好对齐损失**:

    - 功能：利用未标注数据，通过模拟人类偏好反馈来改善分割质量，无需训练独立奖励函数。
    - 核心思路：对每张图像在不同阈值（0.3, 0.4, 0.5, 0.6）生成4个分割候选，根据与GT的IoU分箱评分（1-4级）。损失函数扩展标准DPO为4个候选：$\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log\sigma(\beta_1\log\frac{\pi_\psi(Y_1|I)}{\pi_{\text{fine}}(Y_2|I)} + \beta_2\log\frac{\pi_\psi(Y_2|I)}{\pi_{\text{fine}}(Y_2|I)} - \beta_2\log\frac{\pi_\psi(Y_3|I)}{\pi_{\text{fine}}(Y_3|I)} - \beta_1\log\frac{\pi_\psi(Y_4|I)}{\pi_{\text{fine}}(Y_4|I)})]$，其中 $\beta_1 > \beta_2$ 对最佳和最差候选给予更大权重。
    - 设计动机：DPO无需训练独立奖励函数，模型本身就是奖励模型。4个候选的梯度化权重比简单的成对比较提供了更丰富的偏好信号。

3. **虚拟标注员评分机制**:

    - 功能：模拟人类标注员的质量评估过程，为偏好对齐提供监督信号。
    - 核心思路：用IoU分箱（<0.4, 0.4-0.55, 0.55-0.7, >0.7）对候选打分。虽然用了GT计算IoU，但GT仅用于评分而非直接监督，因此是半监督设置。也支持排名代替评分。
    - 设计动机：模拟真实场景中标注员对分割质量的简单好/差判断，不需要精确的像素级标注。

### 损失函数 / 训练策略
Stage 1：Focal Loss + Dice Loss（20:1加权），在10%标注数据上训练15个epoch。Stage 2：DPO损失，在剩余无标注数据上训练30个epoch。$\beta_1=1, \beta_2=0.5$。Adam优化器，lr=1e-4，每10个epoch减半。

## 实验关键数据

### 主实验

| 方法 | Chest X-ray (20% data) Dice | Breast USD (20%) Dice | AMOS CT (20%) mDice |
|------|---------------------------|---------------------|-------------------|
| U-Net | 58.66 | 57.35 | 59.35 |
| nnU-Net | 60.97 | 59.47 | 65.21 |
| SAM-Med2D | 67.81 | 63.72 | 66.57 |
| **Ours (10%+10%未标注)** | **78.87** | **75.88** | **77.69** |

### 消融实验

| 配置 | Chest X-ray Dice | 说明 |
|------|-----------------|------|
| Full (提示+对齐) | 78.87 | 完整模型 |
| - 对齐 (仅提示, 20%标注) | 79.13 | 全监督提示基线 |
| - 对齐 (仅提示, 10%标注) | 75.60 | 标注减半掉3.5% |
| - 对齐 - VQA | 73.35 | VQA提示贡献+2.25% |
| - 对齐 - VQA - GPT4 | 72.76 | GPT4贡献+0.59% |
| - 对齐 - VQA - CAM (10%) | 57.02 | 仅GPT4文本极差 |

### 关键发现
- 在10-50%数据范围内，本方法持续超越所有全监督SOTA，展现了半监督的优势。
- 偏好对齐机制在仅用10%标注+10%未标注数据时，就接近了20%全监督提示方法的性能。
- 排名策略略优于评分策略，两者都显著优于仅使用最佳候选的基线。
- BiomedCLIP显著图是最重要的提示组件（贡献+15.74%），VQA和GPT-4提供增量改善。
- Stage 1使用Focal Loss + Dice Loss（20:1加权），在10%标注数据上训练15个epoch；Stage 2使用DPO损失在剩余无标注数据上训练30个epoch。Adam优化器，lr=1e-4，每10个epoch减半。
- 多阈值生成4个分割候选（阈值0.3/0.4/0.5/0.6），IoU分箱评分（<0.4/0.4-0.55/0.55-0.7/>0.7对应1-4级），DPO权重$\beta_1=1, \beta_2=0.5$对最佳和最差候选给予更大权重。

## 亮点与洞察
- **将DPO从语言模型迁移到医学分割的创新**：用阈值化生成分割候选替代语言模型的多样化生成，用IoU评分替代人类偏好标注。
- **多源无监督提示的实用性**：BiomedCLIP + VQA + GPT-4 的组合提供了不依赖专家的全面提示信息，可推广到其他医学任务。
- **半监督范式的实际意义**：仅需10%标注数据就能达到较好性能，极大降低了医学图像标注成本。
- **提示组件重要性分析**：BiomedCLIP显著图(CAM)是核心组件——去除后性能从75.60%骤降至57.02%（-18.58%）。VQA提示贡献+2.25%，GPT-4通用知识贡献+0.59%。三种提示信息互补：视觉提示定位目标区域，VQA文本提示描述形状和位置，GPT-4提供疾病通用知识。

## 局限与展望
- 虚拟标注员仍使用GT计算IoU，真实部署时需要替代方案（如基于不确定性的评分）。
- 3D分割（AMOS-CT）是逐切片处理，未充分利用3D信息。
- 在100%数据设置下与全监督方法持平或略低，说明偏好对齐的上限有限。但在10-50%低标注场景下优势明显，是实际部署中最有价值的区间。
- 未来可探索真正无GT的偏好评估方法和更高效的3D扩展。
- MedVInT和GPT-4的文本提示信息质量依赖于预训练模型对医学领域的覆盖程度，罕见疾病场景可能效果有限。
- DPO损失中4个候选的权重设置（$\beta_1=1, \beta_2=0.5$）基于经验，缺乏理论最优性分析。
- 视觉提示使用BiomedCLIP+gScoreCAM生成显著图，经CRF后处理提取边界框和点坐标。MedVInT回答关于器官/肿瘤形状和位置的VQA问题（如"What is the shape of the liver?"），GPT-4提供疾病/器官的通用描述。三种提示拼接输入提示编码器。
- 在Chest X-ray、Breast USD、AMOS CT三个不同模态数据集上评估，覆盖2D X光、超声和3D CT。

## 相关工作与启发
- **vs SAM-Med2D**: SAM-Med2D需要全监督几何提示，本文无监督提示+半监督对齐。
- **vs MedCLIP-SAM**: MedCLIP-SAM仅用CLIP语义提示生成伪标签，本文多源提示+偏好对齐更全面。
- **vs Self-Prompt SAM**: Self-Prompt从输出掩码自生成提示，本文的外部知识提示信号更强。
- **vs nnU-Net**: nnU-Net在20%数据下Dice仅60.97%（Chest X-ray），本方法在10%标注+10%未标注下达78.87%，领先+17.90%。

## 评分

### 实现细节
基于SAM-Med2D编码器。Stage 1: Focal Loss+Dice Loss(20:1), 15 epochs。
Stage 2: DPO损失, 30 epochs。Adam, lr=1e-4, 每10 epoch减半。
- 新颖性: ⭐⭐⭐⭐ DPO在医学分割中的创新应用，多源提示设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三种模态三个数据集，多种数据比例，消融全面
- 写作质量: ⭐⭐⭐⭐ 框架清晰，实验详尽
- 价值: ⭐⭐⭐⭐⭐ 在低标注场景下的实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection](openmibood_open_medical_imaging_benchmarks_for_out-of-distribution_detection.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)

</div>

<!-- RELATED:END -->
