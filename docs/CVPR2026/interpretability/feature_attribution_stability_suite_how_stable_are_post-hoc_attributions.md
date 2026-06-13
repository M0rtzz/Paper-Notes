---
title: >-
  [论文解读] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?
description: >-
  [CVPR 2026][可解释性][特征归因稳定性] 提出 FASS 基准，通过强制预测不变性过滤、三轴稳定性分解（空间/排序/显著区域）和多类型扰动（几何/光度/压缩），系统评估后验特征归因方法的稳定性，揭示了现有评估体系的根本性缺陷。 后验特征归因方法（如 Grad-CAM、LIME、SHAP、Integrated Gr…
tags:
  - "CVPR 2026"
  - "可解释性"
  - "特征归因稳定性"
  - "后验解释方法"
  - "预测不变性"
  - "扰动鲁棒性"
  - "XAI基准"
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

FASS（Feature Attribution Stability Suite）想回答一个被以往评估忽略的问题：当输入被扰动、但模型预测没变时，归因图还稳不稳？它把评估拆成一条流水线——先对每张图施加一种扰动，再用预测不变性过滤掉那些"扰动后模型已经改判"的样本对，最后只在预测保持一致的样本对上，从空间、排序、显著区域三个角度分别度量归因图的变化。这样得到的稳定性才真正反映"解释的脆弱性"，而不是"模型本身的敏感性"。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["原图 + 扰动图对"] --> B["扰动分类<br/>几何 / 光度 / 压缩 三类结构化扰动"]
    B --> C{"预测不变性过滤<br/>argmax 预测是否一致？"}
    C -->|"不一致：剔除并单独统计保留率"| X["排除该样本对"]
    C -->|"一致"| D2
    subgraph D2["三轴稳定性分解"]
        direction TB
        E["SSIM：空间一致性"]
        F["Spearman 秩相关：重要性排序"]
        G["Top-k Jaccard：显著区域重叠"]
    end
    D2 --> H["复合 FASS 分数<br/>(S+R+J)/3 等权合一"]
```

### 关键设计

**1. 扰动分类：跳出 ε-ball 加性噪声**

现有框架几乎只在加性噪声下测稳定性，而真实系统里更常见的是几何、光度、压缩这类结构化扰动。FASS 把扰动分成三类系统施加：几何（15° 旋转、20 像素水平平移，零填充边界）、光度（亮度缩放 ×1.5、高斯噪声 $\sigma=0.15$）、压缩（JPEG 质量因子 40）。正是这组扰动暴露出"仅靠加性噪声会系统性高估归因鲁棒性"，也构成了后续过滤与度量的输入。

**2. 预测不变性过滤：先排除"模型已经改判"的样本对**

以往的 Lipschitz、max-sensitivity 等指标有个隐患：扰动后模型 top-1 类别可能已经变了，却还拿变更后的归因和原图比，等于把"模型敏感"误算成"解释不稳"。FASS 对每个输入-扰动对先检查 argmax 预测是否一致，不一致的对直接剔除并单独统计。被保留下来的比例（retention rate，保留率）本身被当成一级实验量——它揭示了"在什么扰动下，稳定性评估本身已经不可信"。

**3. 三轴稳定性分解：一个标量盖不住三种不同的失败**

把稳定性压成单一数值，就分不清归因到底是空间移位了、还是重要性排序乱了、还是显著区域漂了。FASS 沿三条正交的轴各算一个指标：用 **SSIM**（11×11 均值池化窗口）衡量归因图的空间一致性，捕捉像素级位移；用 **Spearman 秩相关**衡量特征重要性排序是否保持，它独立于幅度变化，把归因图展平后比排名；用 **Top-k Jaccard 重叠度**（$k=100$）衡量最显著的前 100 个位置是否一致（仅占 $224\times224\times3=150{,}528$ 维归因图的 0.07%）。三条轴拆开后，同一个分数背后不同方法的失败模式才看得见。

**4. 复合 FASS 分数：三种失败模式等权合一**

为了给出一个可比较的总分，三个指标取等权平均 $\text{FASS} = (S + R + J)/3$。等权意味着把空间位移、排序变化、显著区域漂移视为同等重要，不预设哪种失败更严重——具体应用想偏重某一轴时再自行调权。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[CVPR 2025\] Open Ad-Hoc Categorization with Contextualized Feature Learning](../../CVPR2025/interpretability/open_ad-hoc_categorization_with_contextualized_feature_learning.md)
- [\[CVPR 2026\] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization](ermoe_eigen-reparameterized_mixture-of-experts_for_stable_routing.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)

</div>

<!-- RELATED:END -->
