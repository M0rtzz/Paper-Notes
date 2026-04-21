---
title: >-
  [论文解读] Enhancing Dataset Distillation via Non-Critical Region Refinement
description: >-
  [CVPR 2025][模型压缩][数据集蒸馏] 提出NRR-DD三阶段框架：用CAM选低置信度patch初始化合成图像、固定关键区域仅优化非关键区域提升信息密度、用2个距离值替代1000维软标签实现500倍存储压缩。在ImageNet-1K上IPC=10时达到46.1%（超RDED 25.7%），软标签存储从120GB降至0.2GB。
tags:
  - CVPR 2025
  - 模型压缩
  - 数据集蒸馏
  - 非关键区域优化
  - 类激活映射
  - 软标签压缩
  - 距离表示
---

# Enhancing Dataset Distillation via Non-Critical Region Refinement

**会议**: CVPR 2025  
**arXiv**: [2503.18267](https://arxiv.org/abs/2503.18267)  
**代码**: https://github.com/tmtuan1307/NRR-DD  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏、非关键区域优化、类激活映射、软标签压缩、距离表示

## 一句话总结
提出NRR-DD三阶段框架：用CAM选低置信度patch初始化合成图像、固定关键区域仅优化非关键区域提升信息密度、用2个距离值替代1000维软标签实现500倍存储压缩。在ImageNet-1K上IPC=10时达到46.1%（超RDED 25.7%），软标签存储从120GB降至0.2GB。

## 研究背景与动机

**领域现状**：大规模数据集蒸馏方法分为两类——关注类通用特征的方法（如SRe2L，学共性但丢细节）和关注实例特异特征的方法（如RDED，选真实patch但缺共性）。

**现有痛点**：（1）两类方法各有偏向，合成图像难以同时捕获类通用和实例特异特征；（2）大规模软标签存储成本巨大——ImageNet1K IPC=200需120GB存储1000维软标签，不切实际。

**核心矛盾**：RDED选取的真实patch已包含丰富的实例特征，但非关键区域（背景等）未被充分利用；同时1000维软标签存储是大规模DD部署的瓶颈。

**本文目标** 在保留真实patch实例特异性的同时增强类通用特征学习，并解决软标签的存储瓶颈。

**切入角度**：用CAM识别关键/非关键区域，固定关键区域（保留实例特征）、优化非关键区域（注入类通用特征）；用2个距离值替代全维度软标签。

**核心 idea**：固定高CAM区域保留实例特征、优化低CAM区域注入类通用知识、用距离表示压缩软标签500倍。

## 方法详解

### 整体框架
三阶段：（1）CIDD阶段用CAM选出最低置信度（非最高）的patch组合成合成图像；（2）NRR阶段固定CAM高激活区域，仅对非关键区域做梯度下降优化（交叉熵+BN正则化），让低置信度patch的非关键区域学到类通用特征；（3）DBR阶段用教师模型计算每个合成图像软标签与原始/增强one-hot标签的交叉熵距离，仅存2个距离值替代1000维向量。

### 关键设计

1. **基于CAM的低置信度初始化（CIDD）**:

    - 功能：选择最有优化空间的patch初始化合成图像
    - 核心思路：用CAM识别原图的高激活区域并提取top-t个patch，但与RDED不同，选择置信度最低的patch而非最高的。将$\beta$个低置信度patch拼接成一张合成图像。低置信度patch包含判别性特征但教师模型不太确信，因此在后续NRR阶段有最大的优化空间
    - 设计动机：高置信度patch已被模型很好识别，优化空间小；低置信度patch有更多可改进的非关键区域

2. **非关键区域优化（NRR）**:

    - 功能：在保留实例特征的同时注入类通用特征
    - 核心思路：用CAM生成非关键区域掩码$M = \max\{0, \epsilon - C\}$，仅对掩码覆盖的像素施加梯度更新$\tilde{x} = \tilde{x} - M \times \eta \nabla_{\tilde{x}} \mathcal{L}_C$。损失包含交叉熵（让低置信度样本向高置信区域移动）和BN统计正则化（对齐原始模型的running statistics）
    - 设计动机：关键区域已有实例特异特征不应被破坏；非关键区域（背景等）是"空白画布"，优化它们可以注入有助于类别整体识别的通用特征

3. **距离表示软标签压缩（DBR）**:

    - 功能：将1000维软标签压缩为2个标量
    - 核心思路：计算教师软标签与原始one-hot标签的交叉熵距离$d_{org}$，以及与增强one-hot标签的交叉熵距离$d_{aug}$。训练学生模型时用距离匹配损失代替直接KD：让学生模型的预测与one-hot的距离匹配教师的距离。每个样本仅需存2个float值
    - 设计动机：存储ImageNet1K的全维度软标签需120GB（IPC=200），不可行。2个距离值可恢复软标签60-71%的性能，存储仅0.2GB

### 损失函数 / 训练策略
NRR阶段：$\mathcal{L}_C = \mathcal{L}_{ce} + \alpha_{bn}\mathcal{L}_{bn}$。学生训练阶段：$\mathcal{L}_S = \mathcal{L}_{sce} + \alpha_{dbr}\mathcal{L}_{dbr}$。

## 实验关键数据

### 主实验

| 数据集 | IPC | NRR-DD | RDED | SRe2L | 提升(vs RDED) |
|--------|-----|--------|------|-------|-------------|
| CIFAR-10 | 10 | 72.2% | 50.2% | 29.3% | +22.0% |
| CIFAR-100 | 10 | 62.7% | 48.1% | 27.0% | +14.6% |
| Tiny-ImageNet | 50 | 61.2% | 47.6% | 41.1% | +13.6% |
| ImageNet-1K | 10 | 46.1% | 20.4% | 21.3% | +25.7% |
| ImageNet-1K | 50 | 60.2% | 38.4% | 46.8% | +21.8% |

### 软标签压缩

| 方法 | ImageNet1K IPC=50 | 存储 | 说明 |
|------|-----------------|------|------|
| 完整软标签 | 60.2% | 120GB | 基线 |
| DBR (2个距离) | 45.1% | 0.2GB | 500倍压缩，恢复60%性能 |
| One-hot | 32.4% | ~0 | 丢失太多信息 |

### 关键发现
- NRR在所有数据集和IPC设置上均大幅超越RDED和SRe2L，优势在大规模数据集上尤其明显（ImageNet-1K +25%）
- 选低置信度patch比高置信度更好，因为优化空间大——"越差越有潜力"的反直觉策略
- 非关键区域优化是核心创新，固定关键区域只改背景就能带来如此大的提升，说明背景/非关键区域的信息对分类至关重要
- DBR的2个距离值方案在巨大的压缩比（500×）下仍保留了可观的软标签信息

## 亮点与洞察
- **"固定主体、优化背景"的反直觉设计**：传统思路关注如何更好地提取主体特征,但NRR-DD发现背景/非关键区域也承载大量类别判别信息
- **低置信度选择的深层原因**：低置信度patch不是噪声——它们包含判别特征但方式不典型，NRR优化后变成了"有独特视角的好样本"
- **DBR软标签压缩的优雅性**：将1000维信息压缩为2个距离值，本质上保留了"软标签离one-hot有多远"这个最关键的知识蒸馏信号

## 局限与展望
- DBR虽压缩500倍但性能损失仍较大（60%恢复率），改进距离表示可能进一步提升
- CAM质量依赖预训练模型，模型不好时关键/非关键区域划分不准确
- NRR阶段的优化迭代次数需要调节

## 相关工作与启发
- **vs RDED**: RDED直接选高置信度patch不做优化；NRR-DD选低置信度+优化非关键区域，CIFAR-10 IPC=10上从50.2%→72.2%
- **vs SRe2L系列**: SRe2L关注类通用特征丢失实例信息；NRR-DD通过保留关键区域+优化背景实现两者兼顾
- **vs EDF**: EDF从梯度角度增强判别区域；NRR-DD从像素角度固定/优化不同区域，思路互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 非关键区域优化的想法独特，低置信度选择+背景优化的组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ CIFAR到ImageNet-1K全覆盖，软标签压缩实验丰富
- 写作质量: ⭐⭐⭐⭐ 三阶段结构清晰，动机分析充分
- 价值: ⭐⭐⭐⭐⭐ 性能提升巨大+存储压缩实用，对DD领域有重要推动

<!-- RELATED:START -->

## 相关论文

- [Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [Emphasizing Discriminative Features for Dataset Distillation in Complex Scenarios](emphasizing_discriminative_features_for_dataset_distillation_in_complex_scenario.md)
- [DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation](delt_a_simple_diversity-driven_earlylate_training_for_dataset_distillation.md)
- [Curriculum Coarse-to-Fine Selection for High-IPC Dataset Distillation](curriculum_coarse-to-fine_selection_for_high-ipc_dataset_distillation.md)
- [Grounding and Enhancing Informativeness and Utility in Dataset Distillation](../../ICLR2026/model_compression/grounding_and_enhancing_informativeness_and_utility_in_dataset_distillation.md)

<!-- RELATED:END -->
