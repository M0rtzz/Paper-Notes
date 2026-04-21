---
title: >-
  [论文解读] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI
description: >-
  [CVPR 2026][医学图像][卵巢癌检测] 系统对比 15 种 CNN 变体（LeNet/ResNet/VGG/Inception）在卵巢癌组织病理图像五分类上的表现，最终选出 InceptionV3-A（ReLU）达 94% 综合指标，并用 LIME/SHAP/Integrated Gradients 三种 XAI 方法做对比解释分析。
tags:
  - CVPR 2026
  - 医学图像
  - 卵巢癌检测
  - CNN对比
  - 可解释AI
  - 组织病理学
  - InceptionV3
---

# Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI

**会议**: CVPR 2026  
**arXiv**: [2603.11818](https://arxiv.org/abs/2603.11818)  
**代码**: 无  
**领域**: 医学图像分类 / 可解释AI  
**关键词**: 卵巢癌检测, CNN对比, 可解释AI, 组织病理学, InceptionV3  

## 一句话总结

系统对比 15 种 CNN 变体（LeNet/ResNet/VGG/Inception）在卵巢癌组织病理图像五分类上的表现，最终选出 InceptionV3-A（ReLU）达 94% 综合指标，并用 LIME/SHAP/Integrated Gradients 三种 XAI 方法做对比解释分析。

## 研究背景与动机

**领域现状**：卵巢癌是全球女性第 7 常见癌症，致死率极高，核心难题在于缺乏有效的早期筛查手段——不像乳腺癌有乳腺 X 光、宫颈癌有 Pap 试验，卵巢癌仅能通过侵入性活检确诊。深度学习已在多种癌症检测中取得进展，但卵巢癌的 DL 方案仍有限。

**现有痛点**：(1) 现有非侵入检测方法（经阴道超声、CA-125 血检、盆腔检查）准确率不足以作为可靠筛查手段；(2) 确诊依赖活检，过程侵入性强且耗时；(3) 已有 DL 方案多使用单一模型，缺乏系统性的多架构对比和 XAI 可解释性支持，临床采纳信任度低。

**核心矛盾**：需要一个准确率高且可解释的自动化检测系统，但可用的组织病理数据集极小（仅 498 张），且高分模型（VGG 迁移学习）由于特征层预训练冻结难以用 XAI 有效解释。

**本文目标** 在小规模卵巢癌组织病理数据上构建高准确率分类模型，并通过 XAI 为临床决策提供透明依据。

**切入角度**：广撒网式对比 15 种 CNN 变体，综合考虑准确率和 XAI 可行性来选模型，而非只追求最高精度。

**核心 idea**：模型选型不仅看精度，还要看可解释性——用 InceptionV3（从头训练）替代精度更高的 VGG（迁移学习），因为前者更易被 XAI 有效解释。

## 方法详解

### 整体框架

从 Mendeley 数据集获取 5 类组织病理图像（Clear Cell、Endometri、Mucinous、Non Cancerous、Serous）共 498 张 → 使用 Albumentations 数据增强扩充至 2490 张 → 系统训练 15 种 CNN 变体 → 综合评估选出最佳模型（InceptionV3-A）→ 叠加 LIME/SHAP/IG 三种 XAI 方法做对比解释分析。

### 关键设计

1. **数据增强管线**
    - 使用 Albumentations 库做旋转（最高 180°）、水平/垂直翻转、亮度/对比度/饱和度/色调随机变换
    - 每张原图产生 4 张增强图像，从 498 扩充至 2490 张（5 类各约 498 张，保持平衡）
    - 图像转为 Tensor 后将 RGB 值从 0-255 归一化到 0-1 范围，显著提升训练稳定性
    - 训练集-测试集 80:20 随机划分（1992 训练 / 498 测试）
2. **15 种 CNN 变体系统对比**
    - **LeNet 系列**（3 种）：基础版（lr=0.001）/ +Dropout / +Step Decay，100 epoch
    - **ResNet 系列**（4 种）：ResNet-34×2 分辨率（32×32 和 224×224）、ResNet-50、ResNet-101，通过随机搜索（10 次迭代 × 3 epoch）确定最优 lr 和 dropout rate
    - **VGG 系列**（4 种）：VGG16-A/B/C 和 VGG19，均为 ImageNet 迁移学习冻结特征层只训练全连接层
    - **Inception 系列**（4 种）：InceptionV1-A (ReLU) / V1-B (Tanh) / V3-A (ReLU + BatchNorm) / V3-B (Tanh + BatchNorm)，从头训练 80 epoch
3. **XAI 三方法对比解释**
    - **LIME**：局部可解释，生成超像素级解释图（限制显示 10 个最重要特征），揭示预测的局部依据
    - **Integrated Gradients**：梯度归因方法，沿从基线到输入的路径积分梯度，生成像素级重要性图
    - **SHAP**：基于 Shapley 值的归因方法，量化每个像素对预测的边际贡献
    - 三种方法生成的高亮区域存在显著共性，验证了黑盒解释的一致性和可靠性

### 损失函数 / 训练策略

Softmax 输出层 + 交叉熵损失。LeNet 系列训练 100 epoch；ResNet 通过随机搜索确定最优超参（lr 范围 0.0001-0.1，dropout 范围 0.0-0.9）；VGG 使用 ImageNet 预训练权重冻结卷积特征层仅训练全连接层；Inception 从头训练 80 epoch。选模型时排除 VGG 系列——虽然精度最高但迁移学习的冻结特征层使 XAI 难以有效工作。

## 实验关键数据

### 主实验

| 模型 | Accuracy | Precision | Recall | F1-Score |
|------|----------|-----------|--------|----------|
| VGG19 (迁移学习) | 97.19% | 97.31% | 97.19% | 97.20% |
| VGG16-A (迁移学习) | 96.99% | 96.98% | 96.99% | 96.97% |
| **InceptionV3-A (选用)** | **94.58%** | **94.75%** | **94.58%** | **94.62%** |
| InceptionV1-B | 85.74% | 86.26% | 85.74% | 85.42% |
| LeNet-A | 61.85% | 62.20% | 61.85% | 61.96% |
| ResNet-34 (224) | 57.03% | 59.39% | 57.03% | 57.70% |
| ResNet-50 | 34.14% | 47.75% | 34.14% | 33.47% |

VGG 系列虽分数最高，但因迁移学习的黑盒特征层使 XAI 难以有效解释而被排除。

### 消融实验

| 对比维度 | 结论 |
|---------|------|
| VGG16-A vs VGG16-O (Kasture et al.) | 同数据集，本文 96.99% vs 对方 84.64%（增强数据），原始数据 77.78% vs 50% (+27.78pp)，归功于 Tensor 转换和归一化 |
| InceptionV3-A (ReLU) vs InceptionV3-B (Tanh) | 94.58% vs 82.13%，ReLU 显著优于 Tanh，差距 12.45pp |
| ResNet-34 32×32 vs 224×224 | 43.78% vs 57.03%，输入分辨率对 ResNet 影响大 |
| 三种 XAI 方法 | LIME/SHAP/IG 高亮区域存在共性，验证解释一致性 |

### 关键发现

- 模型复杂度不等于性能：ResNet-50/101 表现极差（34-43%），可能与超参搜索和训练 epoch 不充分有关
- 数据预处理（Tensor 转换 + 归一化）对小数据集有显著提升，比对方工作高 27pp
- XAI 可行性是模型选型的重要维度——精度差 2.4pp 但换来可解释性的代价值得

## 亮点与洞察

- 15 种模型的系统对比为医学影像领域的 CNN 选型提供了有价值的参考
- 选模型时综合考虑 XAI 可行性而非只追求最高精度，体现了部署导向和临床信任的考量
- XAI 对比分析展示了 LIME/SHAP/IG 三种方法的互补性和一致性
- 数据增强和归一化策略在极小数据集上的效果值得借鉴

## 局限与展望

- 数据集极小（498 张原始 → 2490 张增强），泛化能力存疑，未做交叉验证
- 全部使用经典 CNN，未尝试 ViT、医学预训练模型（如 BiomedCLIP）或更现代的架构
- 缺少多中心/多机构数据验证和真实临床场景测试
- ResNet 系列表现极差（34-57%），超参搜索策略（仅 10 次 × 3 epoch）可能严重不足
- 未报告 ROC-AUC 的类别级细节（Clear Cell vs Serous 等的混淆情况）
- 增强策略仅做简单几何/色彩变换，未用更先进的增强（如 MixUp、CutMix、Mosaic）

## 相关工作与启发

- **vs Kasture et al. (VGG16-O)**：同数据集，本文 VGG16-A 在增强数据上 96.99% vs 84.64%，证明数据预处理的价值
- **vs Hsu et al.**：对方用 ResNet-18/50/Xception 集成学习做超声卵巢癌检测，达到更高精度但依赖更大数据集
- **vs Wang et al.**：对方用 DL 做 MRI 卵巢良恶性鉴别达 87% 准确率，但使用不同成像模态
- 医学小数据场景下的增强策略和"可解释性导向模型选型"思路可为类似工作提供参考

## 评分

- 新颖性: ⭐⭐ 方法组合均为标准套路，无新架构或新技术提出
- 实验充分度: ⭐⭐ 数据集过小，缺泛化验证、交叉验证和统计检验
- 写作质量: ⭐⭐⭐ 结构清晰但部分描述冗余，公式编号与正文对应不够紧密
- 价值: ⭐⭐ 作为医学 AI 入门工作有一定参考，但创新和实验深度不足以支撑顶会水平

<!-- RELATED:START -->

## 相关论文

- [InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversionbased_reconstructionfree_anomaly_de.md)
- [Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)
- [Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiationinduced_cont.md)
- [Deep Learning–Based Estimation of Blood Glucose Levels from Multidirectional Scleral Blood Vessel Imaging](deep_learning_based_estimation_of_blood_glucose_levels_from_multidirectional_scl.md)
- [Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)

<!-- RELATED:END -->
