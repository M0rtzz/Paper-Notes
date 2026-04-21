---
title: >-
  [论文解读] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning
description: >-
  [CVPR2026][医学图像][Multiple Instance Learning] 提出 PAMS（Priority-Aware Mistake Severity）方法，通过非对称严重性感知的交叉熵损失（MSCE）、语义特征混合（SFR）和非对称 Mikel's Wheel 指标，在多分类 MIL WSI 诊断中显著降低严重误诊风险。
tags:
  - CVPR2026
  - 医学图像
  - Multiple Instance Learning
  - Mistake Severity
  - Whole Slide Image
  - 非对称误分类
  - 层次化分类
  - 病理诊断
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning

**会议**: CVPR2026  
**arXiv**: [2603.13682](https://arxiv.org/abs/2603.13682)  
**代码**: 待确认  
**领域**: medical_imaging  
**关键词**: Multiple Instance Learning, Mistake Severity, Whole Slide Image, 非对称误分类, 层次化分类, 病理诊断

## 一句话总结

提出 PAMS（Priority-Aware Mistake Severity）方法，通过非对称严重性感知的交叉熵损失（MSCE）、语义特征混合（SFR）和非对称 Mikel's Wheel 指标，在多分类 MIL WSI 诊断中显著降低严重误诊风险。

## 背景与动机

1. **MIL 在病理诊断中广泛应用**：Multiple Instance Learning（MIL）将 WSI 建模为 patch bag，已成为计算病理学的主流范式，但现有方法主要关注准确率最大化，忽略了误分类的严重性差异
2. **临床场景中误分类代价不对称**：将恶性肿瘤误诊为正常（漏诊）的后果远比将正常误诊为恶性（过诊）严重，但传统交叉熵对所有错误施加相同惩罚
3. **WSI 多分类的优先级特性**：病理医生在 WSI 中观察到多种共存症状时，标注最紧急的诊断结果；类别之间存在隐含的优先级层次，这与自然图像中每个物体独立标注的方式截然不同
4. **现有 Mistake Severity 方法缺陷**：以往方法仅基于类别间距离定义严重性权重（如 CDW-CE），忽略了方向性——同样距离的误分类在不同方向上临床风险完全不同
5. **缺乏面向临床 WSI 的 MS 解决方案**：现有 MS 研究主要在自然图像上开展，未能处理 WSI 的标注约束（弱标签、复杂共存症状、类别优先级）
6. **评价指标的局限**：现有 MS 指标（ECC/EMC）基于对称距离，无法区分方向性不同的误分类，导致无法正确评估模型在安全性角度的表现

## 方法详解

### 整体框架

PAMS 将多分类问题组织为层次化结构（从最细粒度 $\mathcal{H}$ 到根节点 $\mathcal{R}$），在每个层次训练一个分类器 $f_{\theta_h}$。训练目标为 $\mathcal{L} = \lambda_1 \mathcal{L}_{MSCE} + \lambda_2 \mathcal{L}_{HA}$，同时使用 SFR 进行数据增强。

### Mistake Severity Cross-Entropy（MSCE）

- 定义非对称权重矩阵 $M^h$：当真实类 $c_i^h$ 比预测类 $c_j^h$ 更紧急时，惩罚为 $\alpha^{|i-j|}$（$\alpha > 1$）；反向误分类权重为 1
- 最终损失为 $\mathcal{L}_{MSCE} = -\sum_h \hat{p}^h M^h (\tilde{Y}^h)^\top \sum_c \tilde{Y}^h[c] \log \hat{p}^h[c]$
- 核心思想：在交叉熵前乘一个方向性正则化权重 $\hat{p}^h M^h (\tilde{Y}^h)^\top$，同时考虑预测概率分布和真实标签之间的严重性关系
- 与 Weighted CE 的区别：Weighted CE 仅基于类频率或固定权重，不考虑预测与真实标签之间的方向性差异

### 层次概率对齐（Hierarchy Alignment）

- 使用 Jensen-Shannon 散度对齐相邻层次的预测概率
- 将更细粒度层的预测 $\hat{p}^{h+1}$ 聚合为粗粒度表示 $\dot{p}^{h+1}$，与当前层 $\hat{p}^h$ 对齐
- 确保不同层次分类器对同一样本做出一致性预测

### Semantic Feature Remix（SFR）

- 给定两个不同优先级的 WSI（$Y_a \succ Y_b$），将其全部 instance 聚类为 $L$ 个簇
- 按簇中来自高优先级样本 $Z_a$ 的 patch 比例排序，取 top-$k$ 簇中的 $Z_a$ patches
- 将这些语义上代表高严重性症状的 patches 混入低优先级 bag $Z_b$，形成合成样本 $Z_{a+b}$，标签为 $Y_a$
- 使用 FAISS 库实现高效 GPU 并行聚类

### 非对称 Mikel's Wheel 指标

- 提出 AsCC（Asymmetric Classification Confidence）和 AsMC（Asymmetric Misclassification Confidence）
- 混淆权重 $W_{i,j}^h = 1 + |i-j| + \mathbb{1}(c_i^h \succ c_j^h) \times P$，其中 $P=2$
- 高优先级类被误分到低优先级时施加额外惩罚，反映真实临床风险

## 实验关键数据

### 数据集

- **BRACS**：乳腺癌 H&E 染色 WSI，547 张，7 类（正常→浸润癌），按良性/非典型/恶性三级层次
- **In-house**：结肠活检 WSI 4734 张，7 类，按良性/锯齿状/腺瘤三级层次；含 182 例复杂混合症状测试集

### 主实验结果（Table 1，BRACS + TransMIL）

| 方法 | ACC | AUC | AsCC | AsMC |
|------|-----|-----|------|------|
| Cross Entropy | 40.23 | 74.90 | 58.48 | 50.18 |
| Chang et al. | 47.51 | 79.48 | 63.98 | 51.02 |
| Hong et al. (τ=10) | 47.13 | 79.80 | 62.44 | 45.54 |
| CDW-CE | 44.83 | 79.06 | 61.05 | 47.32 |
| **PAMS (Ours)** | **47.59** | **80.61** | **64.92** | **55.65** |

PAMS 在所有指标上取得最优，AsCC 和 AsMC 提升最为显著。In-house 数据集上同样全面领先。

### 消融实验（Table 2，BRACS + TransMIL）

| 消融项 | ACC 下降 | AsMC 下降 |
|--------|----------|-----------|
| w/o MSCE | -2.46 | -4.84 |
| w/o HA | -2.84 | -0.53 |
| w/o SFR | -0.54 | -4.02 |
| 全部移除 | -7.82 | -1.76 |

- MSCE 对严重性指标贡献最大（AsMC 下降 4.84）
- SFR 对 AsMC 也有显著贡献（下降 4.02）
- 三个组件协同配合效果最佳

### CIFAR-10 自然图像实验（Table 4）

| 方法 | ACC | AsCC | AsMC |
|------|-----|------|------|
| CE | 83.24 | 87.23 | 34.84 |
| CDW-CE | 84.11 | 87.87 | 34.63 |
| **MSCE (Ours)** | **85.64** | **89.12** | **35.70** |

验证了 MSCE 在自然图像领域的泛化能力。

## 亮点

- **非对称严重性建模**：首次在 MIL WSI 诊断中引入方向性误分类惩罚，准确反映临床漏诊比过诊更危险的实际需求
- **语义数据增强 SFR**：利用弱标签信息在特征空间中智能混合样本，无需像素级标注即可模拟复杂共存症状
- **指标创新**：AsCC/AsMC 弥补了现有对称指标无法区分误分类方向的缺陷，适用于所有安全关键分类任务
- **广泛通用性**：方法在 BRACS、In-house 医学数据及 CIFAR-10 自然图像上均有效，与多种 MIL 架构兼容

## 局限与展望

- 层次结构需人工预定义，依赖领域专家知识，不同疾病可能需要不同的层次设计
- MSCE 中的 $\alpha$ 和 $P$ 超参数选择需要调优，论文将敏感性分析放在补充材料中
- SFR 依赖聚类质量，聚类数 $L$ 和 top-$k$ 的选择可能影响效果
- 仅在病理场景验证，尚未在放射影像、皮肤镜等其他医学模态上验证
- In-house 数据集未公开，可重复性受限

## 与相关工作的对比

- **vs. Weighted CE**：仅用固定权重，无法捕捉预测与真实标签间的方向性差异；MSCE 动态计算惩罚
- **vs. HXE / Soft Labels（Bertinetto et al.）**：利用LCA层次信息但对严重性指标改善有限
- **vs. HAF（Garg et al.）**：特征空间正则化方法，在 DTFD-MIL 上泛化较差
- **vs. Hong et al.**：随机 remix 策略在 In-house 数据上有效但 BRACS 上不稳定；SFR 通过语义引导更鲁棒
- **vs. CDW-CE**：基于类距离加权但仍对称，PAMS 的非对称设计更贴合临床需求

## 评分

- 新颖性: ⭐⭐⭐⭐ — 非对称严重性损失 + 语义 remix + 非对称评价指标三位一体，问题定义清晰
- 实验充分度: ⭐⭐⭐⭐ — 公开+私有数据集、多种 MIL 架构、消融实验、remix 策略对比、自然图像泛化实验
- 写作质量: ⭐⭐⭐⭐ — 图表表达清晰，问题动机铺垫有说服力，公式推导完整
- 价值: ⭐⭐⭐⭐ — 切中临床 MIL 部署中的核心安全痛点，非对称指标有通用价值

<!-- RELATED:START -->

## 相关论文

- [MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)
- [Do Multiple Instance Learning Models Transfer?](../../ICML2025/medical_imaging/do_multiple_instance_learning_models_transfer.md)
- [Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gender-adversarial_attention_multi.md)
- [Pathology-knowledge Enhanced Multi-instance Prompt Learning for Few-shot Whole Slide Image Classification](../../ECCV2024/medical_imaging/pathology-knowledge_enhanced_multi-instance_prompt_learning_for_few-shot_whole_s.md)
- [Meta-learning In-Context Enables Training-Free Cross Subject Brain Decoding](meta-learning_in-context_enables_training-free_cross_subject_brain_decoding.md)

<!-- RELATED:END -->
