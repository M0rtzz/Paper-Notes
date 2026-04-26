---
title: >-
  [论文解读] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?
description: >-
  [CVPR 2026][特征归因稳定性] 提出 FASS 基准，通过强制预测不变性过滤、三轴稳定性分解（空间/排序/显著区域）和多类型扰动（几何/光度/压缩），系统评估后验特征归因方法的稳定性，揭示了现有评估体系的根本性缺陷。
tags:
  - CVPR 2026
  - 特征归因稳定性
  - 后验解释方法
  - 预测不变性
  - 扰动鲁棒性
  - XAI基准
---

# Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?

**会议**: CVPR 2026  
**arXiv**: [2604.02532](https://arxiv.org/abs/2604.02532)  
**代码**: [GitHub](https://github.com/KamalasankariSubramaniakuppusamy/xai-stability-benchmark)  
**领域**: 可解释AI / 模型压缩  
**关键词**: 特征归因稳定性, 后验解释方法, 预测不变性, 扰动鲁棒性, XAI基准

## 一句话总结

提出 FASS 基准，通过强制预测不变性过滤、三轴稳定性分解（空间/排序/显著区域）和多类型扰动（几何/光度/压缩），系统评估后验特征归因方法的稳定性，揭示了现有评估体系的根本性缺陷。

## 研究背景与动机

后验特征归因方法（如 Grad-CAM、LIME、SHAP、Integrated Gradients）广泛用于安全关键的视觉系统中，帮助从业者理解模型决策。然而，当输入发生微小但不改变模型预测的扰动时，归因结果可能发生显著变化，这对其可靠性构成严重威胁。

现有稳定性评估存在三个结构性缺陷：

**不强制预测不变性**：在计算稳定性时未检查扰动是否改变了模型预测类别。Lipschitz 连续性、max-sensitivity 等指标在扰动后预测类别改变时仍然比较归因，混淆了"模型敏感性"与"解释脆弱性"

**单标量度量**：将稳定性压缩为单一数值，无法区分空间位移、排序变化或显著区域不一致等不同失败模式

**仅评估加性噪声**：现有框架主要在 ε-ball 加性噪声下测试，忽略了实际系统中常见的几何变换、光度变化和压缩伪影

这些缺陷使得现有评估**系统性高估**了归因方法的稳定性。

## 方法详解

### 整体框架

FASS（Feature Attribution Stability Suite）是一个模块化评估流水线，包含三个阶段：
- 扰动施加 → 预测不变性过滤 → 三轴稳定性度量

每张输入图像与其扰动版本配对，仅当模型 top-1 预测类别保持不变时，才计算稳定性指标。

### 关键设计

1. **预测不变性过滤（Prediction-Invariant Filtering）**：对每个输入-扰动对，检查模型的 argmax 预测是否保持一致。不一致的对被排除并单独报告为保留率诊断。核心洞察是：比较不同预测类别下的归因没有意义，因为归因是相对于特定预测定义的。保留率（retention rate）本身也成为一级实验量，揭示"稳定性评估何时变得不可靠"。

2. **三轴稳定性分解**：

    - **SSIM（结构相似性）**：衡量归因图的空间一致性，使用 11×11 均值池化窗口，检测像素级空间位移
    - **Spearman 秩相关**：衡量特征重要性排序是否在扰动后保持，独立于幅度变化。将归因图展平后比较排名
    - **Top-k Jaccard 重叠度**：k=100，衡量最显著的前 100 个特征位置的一致性（占 224×224×3 = 150,528 维归因图的 0.07%）

3. **复合 FASS 分数**：三个指标的等权平均：FASS = (S + R + J) / 3。等权设计将三种失败模式视为同等重要。

4. **扰动分类**：

    - 几何：15° 旋转、20 像素水平平移（零填充边界）
    - 光度：亮度缩放 ×1.5、高斯噪声 σ=0.15
    - 压缩：JPEG 质量因子 40

### 损失函数 / 训练策略

本文为评估基准，不涉及模型训练。评估使用预训练模型（ResNet-50、DenseNet-121、ConvNeXt-Tiny、ViT-B/16），通过 Captum 库实现四种归因方法（IG、GradientSHAP、Grad-CAM、LIME）。

## 实验关键数据

### 主实验

评估规模：约 70,000 张图像 × 5 种扰动 × 4 模型 × 4 归因方法 = 约 640 万次归因计算。

| 数据集 | 方法 | SSIM | Spearman | Jaccard | FASS |
|--------|------|------|----------|---------|------|
| ImageNet | Grad-CAM | .885 | .966 | .314 | **.722** |
| ImageNet | IG | .706 | .603 | .060 | .457 |
| ImageNet | GradientSHAP | .681 | .570 | .037 | .429 |
| ImageNet | LIME | .342 | .582 | .072 | .332 |
| CIFAR-10 | Grad-CAM | .830 | .899 | .423 | **.717** |
| COCO | Grad-CAM | .810 | .881 | .321 | **.671** |

### 消融实验 - 预测不变性保留率

| 扰动类型 | 平均保留率 | 范围 |
|----------|-----------|------|
| 旋转 | 30.9% | 0.0–88.1% |
| 平移 | 0.1% | 0.0–0.6% |
| 亮度 | 0.8% | 0.0–9.0% |
| 噪声 | 34.5% | 0.0–94.4% |
| JPEG | 1.0% | 0.0–11.7% |

不进行预测不变性过滤时，高达 **99%** 的评估对涉及预测变化——说明不过滤就直接评估稳定性是严重有问题的。

### 消融实验 - 扰动类型影响

| 扰动类别 | SSIM | Spearman | Jaccard | FASS |
|----------|------|----------|---------|------|
| 几何 | .725 | .666 | .099 | .497 |
| 光度 | .770 | .724 | .178 | .557 |
| 压缩 | .791 | .739 | .196 | .576 |

### 关键发现

1. **Grad-CAM 是最稳定的归因方法**：在所有 12 个数据集-架构组合中 FASS 最高。其 7×7 激活图的低通滤波特性天然吸收了局部扰动效应
2. **IG 和 GradientSHAP 高度一致**：FASS 差距不超过 0.05，说明 SHAP 的 Shapley 值采样没有显著降低梯度信号的稳定性
3. **归因方法选择 > 模型架构选择**：Grad-CAM 与 IG 之间的 FASS 差距（0.21）约为同一方法在不同架构间最大差距（0.09）的两倍
4. **几何扰动暴露的不稳定性远大于光度扰动**：仅基于加性噪声的基准系统性高估归因鲁棒性
5. **LIME 展现独特的 SSIM-Spearman 分离**：空间不稳定但排序相对保持，是单一分数无法捕捉的结构性失败模式

## 亮点与洞察

- **预测不变性作为前置条件**是一个简单但极为重要的设计原则。此前的工作完全忽视了这一点，导致稳定性评估结果不可靠
- 三轴分解的设计很精妙：同一个 FASS 分数背后，不同方法的失败模式完全不同。例如 Grad-CAM 的 Spearman 接近完美但 Jaccard 较低，而 LIME 的 SSIM 极低但 Spearman 还行
- 保留率本身作为"一级实验量"的思路很值得借鉴——它不仅是诊断指标，还定义了稳定性评估本身是否可靠的边界条件

## 局限与展望

1. FASS 度量稳定性而非忠实性：一个始终给出错误但一致的归因会得到高分。联合评估稳定性和忠实性是自然的下一步
2. 每种扰动仅使用单一强度，强度扫描可能揭示非线性退化
3. 等权复合分数缺乏领域适配性，特定应用场景可能需要不同权重
4. 仅覆盖四种归因方法，未涉及 SmoothGrad、LRP 等变体和概念级方法
5. 使用预训练模型未做数据集特定微调，影响保留率（尤其 CIFAR-10 的 32→224 上采样）

## 相关工作与启发

- 本文与 Quantus、LATEC、OpenXAI 等评估框架形成鲜明对比——这些框架要么不强制预测不变性，要么仅用单一标量，要么仅测试加性噪声
- 对于实际部署场景（如医学影像、自动驾驶），本文的发现意味着：(1) 优先选择 Grad-CAM 而非像素级方法；(2) 必须在目标部署条件下（包括几何变换）测试归因稳定性
- 可启发其他"评估评估方法本身"的研究——任何稳定性度量都需要检查其前置假设是否满足

## 评分

- 新颖性: ⭐⭐⭐⭐ （评估框架设计新颖，但基本思路不复杂）
- 实验充分度: ⭐⭐⭐⭐⭐ （70K 图像、640 万次归因计算、三数据集四架构四方法）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，问题定义精准）
- 价值: ⭐⭐⭐⭐ （对 XAI 社区实践有直接指导意义）

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](../../CVPR2025/interpretability/open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)

<!-- RELATED:END -->
