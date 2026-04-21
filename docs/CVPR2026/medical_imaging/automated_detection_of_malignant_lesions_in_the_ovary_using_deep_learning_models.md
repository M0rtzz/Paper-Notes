---
title: >-
  [论文解读] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI
description: >-
  [CVPR 2026][医学图像][卵巢癌检测] 系统地比较了 LeNet/ResNet/VGG/Inception 四大CNN架构的15个变体在卵巢癌组织病理学图像分类上的表现，最终选择 InceptionV3-ReLU 作为基础模型(平均指标~94%)，并结合 LIME、SHAP、Integrated Gradients 三种 XAI 方法对分类结果进行可解释性分析。
tags:
  - CVPR 2026
  - 医学图像
  - 卵巢癌检测
  - CNN分类
  - 可解释AI
  - 组织病理学
  - InceptionV3
---

# Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**会议**: CVPR 2026  
**arXiv**: [2603.11818](https://arxiv.org/abs/2603.11818)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 卵巢癌检测, CNN分类, 可解释AI, 组织病理学, InceptionV3

## 一句话总结

系统地比较了 LeNet/ResNet/VGG/Inception 四大CNN架构的15个变体在卵巢癌组织病理学图像分类上的表现，最终选择 InceptionV3-ReLU 作为基础模型(平均指标~94%)，并结合 LIME、SHAP、Integrated Gradients 三种 XAI 方法对分类结果进行可解释性分析。

## 研究背景与动机

卵巢癌是全球女性第7大常见癌症，也是致死率最高的妇科癌症之一。与乳腺癌（乳腺X线/CBE）和宫颈癌（Pap检测）不同，卵巢癌目前没有可靠的早期筛查方法。现有检测手段包括：

**经阴道超声**：敏感性有限，假阳性率高

**CA-125血液检测**：特异性差，非癌因素也可导致升高

**组织活检**：确诊金标准，但为有创检查，耗时长

核心矛盾：**需要一种非侵入性、快速、准确的检测方法来降低卵巢癌的延迟诊断率**。近年来深度学习在医学影像辅助诊断中展现出强大能力，但卵巢癌领域的应用仍然相对不足。

本文的切入角度：在 Mendeley 公开的卵巢癌组织病理学数据集上，系统比较多种 CNN 变体，选出最优模型，并通过可解释 AI(XAI) 揭示模型决策依据，增强临床可信度。

## 方法详解

### 整体框架

整体 pipeline 为：数据集获取 → 数据增强 → 张量转换与归一化 → 15个CNN变体训练评估 → 最优模型选择 → XAI 可解释性分析（LIME + SHAP + Integrated Gradients）。

数据集为 Mendeley 的 OvarianCancer&SubtypesDatasetHistopathology，包含5个类别：Clear Cell、Endometrioid、Mucinous、Non Cancerous、Serous，共498张原始图像。

### 关键设计

1. **数据增强策略**:

    - 功能：从498张扩增到2490张，保持类间平衡
    - 核心思路：使用 Albumentations 库进行复合增强——旋转(最多180°)、水平/垂直翻转、亮度/对比度/饱和度/色相随机变化，每张原图生成4张增强图
    - 设计动机：原始数据集极小(每类仅~100张)，不足以训练深度CNN；Albumentations的随机概率参数比固定变换提供更大的多样性

2. **张量转换与归一化**:

    - 功能：将增强后的图像转换为TensorFlow张量，像素值从[0,255]归一化到[0,1]
    - 核心思路：使用 `image_dataset_from_directory()` 方法，标签模式设为 `int`（非one-hot），80/20随机划分训练/测试集
    - 设计动机：float32 + [0,1]归一化可加速卷积运算收敛；int标签便于未来扩展类别

3. **15个CNN变体的系统评估**:

    - 功能：覆盖 LeNet(3变体)、ResNet(4变体)、VGG(4变体)、Inception(4变体) 共15个模型
    - LeNet变体：基准(lr=0.001) → +Dropout → +Step Decay
    - ResNet变体：ResNet-34(32×32)、ResNet-34(224×224)、ResNet-50、ResNet-101，通过随机搜索优化学习率和Dropout率
    - VGG变体：VGG16-A/B/C(分别用ReLU/tanh/+lr+dropout)和VGG19，全部使用迁移学习（冻结卷积层，仅训练全连接层）
    - Inception变体：V1-A/B(ReLU/tanh)和V3-A/B(+BatchNorm/ReLU和tanh)
    - 设计动机：通过系统对比确定在该特定数据集上的最优架构

4. **XAI可解释性分析**:

    - 功能：对选定的 InceptionV3-A 模型应用三种 XAI 方法
    - **LIME**：生成局部可解释的超像素级特征重要性图，限制显示10个最重要特征
    - **Integrated Gradients**：从基线输入到实际输入的梯度积分，生成像素级归因图
    - **SHAP**：基于 Shapley 值的局部解释，展示各像素对每个类别预测的正/负贡献
    - 设计动机：VGG的迁移学习结构难以应用XAI，而从零训练的InceptionV3更适合梯度类XAI方法

### 损失函数 / 训练策略

- 所有模型输出层使用 **Softmax** 激活：$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}$
- 损失函数：分类交叉熵
- ResNet 的超参数搜索：在 lr∈[0.0001, 0.1]、dropout∈[0.0, 0.9] 范围内随机采样10组，各跑3个epoch选最优
- 评估：统一使用 Accuracy、Precision、Recall、F1-Score、ROC曲线和AUC

## 实验关键数据

### 主实验

| 模型 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| LeNet-A | 61.85% | 62.20% | 61.85% | 61.96% |
| ResNet-34(224) | 57.03% | 59.39% | 57.03% | 57.70% |
| ResNet-50 | 34.14% | 47.75% | 34.14% | 33.47% |
| VGG16-A (迁移) | **96.99%** | 96.98% | 96.99% | 96.97% |
| VGG19 (迁移) | **97.19%** | **97.31%** | **97.19%** | **97.20%** |
| InceptionV3-A | 94.58% | 94.75% | 94.58% | 94.62% |
| InceptionV1-A | 78.92% | 81.58% | 78.92% | 79.33% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| VGG16-O (Kasture原文) | 50% (原始) / 84.64% (增强20k张) | 基线对比论文 |
| VGG16-A (本文) | 77.78% (原始) / 96.99% (增强) | 张量转换+归一化大幅提升 |
| InceptionV3-A (原始数据) | 20.20% | 从零训练在小数据上极差 |
| InceptionV3-A (增强数据) | 94.58% | 增强后显著提升 |
| ReLU vs tanh (InceptionV3) | 94.58% vs 82.13% | ReLU明显优于tanh |
| ReLU vs tanh (InceptionV1) | 78.92% vs 85.74% | V1中tanh反而更好 |

### 关键发现

- VGG系列在迁移学习下表现最好(~97%)，但迁移学习的封装特性使XAI难以深入分析
- InceptionV3-A 作为从零训练的模型在增强数据集上达到94.58%，兼顾了准确性和可解释性
- 数据增强是小数据集上CNN训练的决定性因素——InceptionV3从20.20%跃升至94.58%
- 三种XAI方法在相同样本上识别出了重叠的关键特征区域，验证了分类依据的一致性

## 亮点与洞察

- **系统化对比**：15个模型变体的穷举式比较在卵巢癌组织病理学领域较为全面
- **XAI三方交叉验证**：同时使用LIME/SHAP/IG并做对比分析，增强了可解释性结论的可信度
- **模型选择的权衡思考**：不单纯追求最高准确率(VGG19=97.19%)，而是综合考虑XAI兼容性选择InceptionV3，体现了实用导向

## 局限与展望

- **数据集极小**：仅2490张增强图像(原始498张)，与临床规模差距巨大，泛化能力存疑
- **无外部验证**：所有实验在单一数据集上完成，未使用独立测试集或跨中心数据评估
- **模型架构偏旧**：未涉及 Vision Transformer、EfficientNet、ConvNeXt 等现代架构
- **无临床对比**：未与专业病理医生的诊断准确率进行对比
- **XAI分析偏定性**：三种XAI方法仅做了视觉对比，缺乏定量一致性指标(如IoU、相关系数)
- **数据来源单一**：Mendeley数据集的图像质量、染色方案可能不代表真实临床多样性

## 相关工作与启发

- Kasture et al. 在相同数据集上用VGG16达到84.64%，本文VGG16-A达96.99%，差异主要来自预处理(张量转换+归一化)
- Wang et al. 的MRI卵巢肿瘤鉴别(87% accuracy)使用不同模态，但提出了AI辅助初级医生的思路
- Hsu et al. 的超声集成CNN(ResNet-18/50 + Xception)提出80-100%置信度阈值作为临床应用标准

## 评分
- 新颖性: ⭐⭐ 方法论层面无创新，是标准CNN分类+XAI的组合应用
- 实验充分度: ⭐⭐⭐ 15个变体的系统对比较全面，但数据集太小、无外部验证
- 写作质量: ⭐⭐⭐ 结构清晰，但部分细节冗余，公式描述偏初级教科书风格
- 价值: ⭐⭐⭐ 对卵巢癌CAD有一定参考价值，但距离临床可用仍有较大差距

<!-- RELATED:START -->

## 相关论文

- [Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)
- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)

<!-- RELATED:END -->
