---
title: >-
  [论文解读] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions
description: >-
  [CVPR 2026][医学图像][weakly supervised localization] 提出SFDA-DeP，受机器遗忘启发将源自由域适应（SFDA）建模为迭代识别并纠正预测偏差的过程——选择性降低优势类中不确定样本的置信度、保留可靠预测、联合训练像素级分类器恢复定位判别力——在跨器官/跨中心病理基准上一致优于SFDA baselines的分类和定位性能。
tags:
  - CVPR 2026
  - 医学图像
  - weakly supervised localization
  - 域适应
  - prediction debiasing
  - machine unlearning
  - histopathology
---

# Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions

**会议**: CVPR 2026  
**arXiv**: [2603.12468](https://arxiv.org/abs/2603.12468)  
**代码**: [anonymous.4open.science/r/SFDA-DeP-1797](https://anonymous.4open.science/r/SFDA-DeP-1797/)  
**领域**: 医学图像 / 计算病理 / 域适应  
**关键词**: weakly supervised localization, source-free domain adaptation, prediction debiasing, machine unlearning, histopathology

## 一句话总结

提出SFDA-DeP，受机器遗忘启发将源自由域适应（SFDA）建模为迭代识别并纠正预测偏差的过程——选择性降低优势类中不确定样本的置信度、保留可靠预测、联合训练像素级分类器恢复定位判别力——在跨器官/跨中心病理基准上一致优于SFDA baselines的分类和定位性能。

## 研究背景与动机

**领域现状**：深度WSOL模型通过图像级标签同时实现分类和ROI定位，在病理图像中已有成功应用（NEGEV、PixelCAM、SAT等）。但WSOL模型在跨域部署时（不同器官、不同中心、不同染色/扫描协议），分布偏移导致性能严重退化。

**现有痛点**：

1. 在较大域偏移（特别是跨器官）下，WSOL预测严重偏向优势类——从GlaS（结肠）迁移到CAMELYON16/17（乳腺淋巴结）时，模型可能90%+预测为癌症类
2. 传统SFDA方法（SFDA-DE、ERL、CDCL）依赖自训练，隐含假设源分类器在目标域仍有足够判别力——在严重域偏移下假设不成立
3. 偏置的伪标签在自训练迭代中被不断强化而非纠正——偏差放大效应导致分类和定位双重退化

**核心矛盾**：SFDA的自训练机制恰恰是放大WSOL预测偏差的元凶——越训练越偏。

**本文目标** 在不访问源数据的前提下，纠正域偏移导致的类别预测失衡，同时恢复定位判别力。

**切入角度**：借鉴机器遗忘思路——不是"遗忘"某个类别，而是让模型"遗忘"旧的偏置决策边界，建立新的均衡边界。

**核心 idea**：对优势类中高熵样本施加"遗忘"损失以推移决策边界，同时保留低熵样本维持稳定预测。

## 方法详解

### 整体框架

在目标域上每隔 $m$ 个epoch动态重划分"遗忘集"（优势类中高熵样本）和"保留集"（其余所有样本），通过三个损失联合优化：保留损失维持可靠预测、遗忘损失纠正偏差、像素级定位损失锚定空间判别特征。

### 关键设计

1. **遗忘-保留集动态划分与对应损失**

    - 将预测为优势类 $\mathcal{B}$ 的样本集合记为 $\mathbb{B}$，按归一化熵排序取 $\text{top}_\rho$ 最不确定样本为遗忘集 $\mathbb{B}_f$（$\rho \in \{5\%, 15\%, 25\%\}$），其余为保留集 $\mathbb{B}_r = \mathbb{T} - \mathbb{B}_f$
    - 保留损失：标准交叉熵 $\mathcal{L}_{\text{retain}} = -\log(p_i(\hat{y}))$，让模型继续预测保留集样本的伪标签
    - 遗忘损失：$\mathcal{L}_{\text{forget}} = -\log(1 - p_i(\hat{y}))$，最小化它让模型停止将遗忘集样本预测为优势类，迫使决策边界移动
    - **关键**：每 $m$ epoch用当前模型重新划分，防止错误遗忘决策不可逆积累——这是动态纠偏而非一次性操作

2. **像素级定位监督**

    - 联合训练轻量像素级分类器 $h$ 对特征图做前景/背景二分类
    - 仅对每类中熵最低的 $\text{top}_{\rho_{\text{loc}}}$ 样本提取源模型CAM作为像素级伪标签
    - 用BCE损失训练：$\mathcal{L}_{\text{loc}} = -(1-Y_p)\log(h(z_p)_0) - Y_p\log(h(z_p)_1)$
    - 目的：在分类去偏的同时锚定空间判别特征，防止定位能力在适应过程中漂移

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \lambda_{\text{retain}} \mathcal{L}_{\text{retain}} + \lambda_{\text{forget}} \mathcal{L}_{\text{forget}} + \lambda_{\text{loc}} \mathcal{L}_{\text{loc}}$

- $\lambda_{\text{retain}}, \lambda_{\text{forget}} \in \{0.2, 0.5, 1.0, 2.0\}$，$\lambda_{\text{loc}} \in \{0.5, 1.0, 5.0\}$
- 学习率从 $\{10^{-5}, 10^{-4}, 10^{-3}\}$ 选取
- CNN骨干ResNet-50，Transformer骨干DeiT-Tiny，在三种WSOL模型（PixelCAM、SAT、DeepMIL）上验证
- 数据集：GlaS（结肠）、CAMELYON16（乳腺）、CAMELYON17（5个中心）

## 实验关键数据

### 主实验

GlaS→CAMELYON系列跨器官/跨中心适应的6个目标域平均指标：

| WSOL模型 | 方法 | 平均PxAP | 平均CL | vs SFDA-DE |
|----------|------|----------|--------|-----------|
| PixelCAM | Source-only | 36.9 | 49.3 | — |
| PixelCAM | SFDA-DE | 28.0 | 54.6 | baseline |
| PixelCAM | ERL | 25.4 | 59.9 | -2.6 PxAP |
| PixelCAM | RGV | 34.7 | 52.1 | +6.7 PxAP |
| PixelCAM | **SFDA-DeP** | **44.1** | **67.1** | **+16.1 PxAP, +12.5 CL** |
| DeepMIL | Source-only | 20.9 | 49.8 | — |
| DeepMIL | SFDA-DE | 20.5 | 53.9 | baseline |
| DeepMIL | **SFDA-DeP** | **40.7** | **73.4** | **+20.2 PxAP, +19.5 CL** |
| SAT | Source-only | 21.3 | 52.1 | — |
| SAT | SFDA-DE | 21.6 | 68.7 | baseline |
| SAT | **SFDA-DeP** | **30.3** | **69.2** | **+8.7 PxAP, +0.5 CL** |

### 消融实验

| 消融项 | 关键结果 |
|--------|---------|
| 动态重采样 vs 静态划分 | 动态显著更优，防止错误遗忘决策累积 |
| 有/无 $\mathcal{L}_{\text{loc}}$ | 加入像素级损失后PxAP提升明显 |
| 遗忘比例 $\rho$ (5%-25%) | 方法对此超参不敏感 |

### 关键发现

- SFDA-DE在多个中心CL坍塌到50%（随机猜测），SFDA-DeP恢复到80%+：PixelCAM在C17-0上CL从50.0%提至86.2%，DeepMIL在C17-0上CL从50.0%提至82.8%
- 现有SFDA方法在PxAP上甚至不如Source-only（SFDA-DE: 28.0 vs Source-only: 36.9），证实偏差放大效应
- DeepMIL上SFDA-DeP的改进最大（+20.2 PxAP），说明对基础架构越弱、偏差越严重的模型效果越显著
- 定性可视化显示SFDA-DeP的CAM激活集中在肿瘤组织上，SFDA baselines常高亮背景区域

## 亮点与洞察

- 精准诊断了SFDA在WSOL上失败的根因（预测偏差放大），而非笼统归因于域偏移
- "遗忘旧决策边界、建立新均衡边界"的类比清晰且方法简洁（仅三个损失项）
- 跨三种不同WSOL架构（CNN/Transformer/MIL）和6个目标域一致有效，泛化性好
- 在SFDA-DE完全失效的场景（CL坍塌到50%即随机猜测）中仍能恢复到80%+，鲁棒性强

## 局限与展望

- 仅在二分类（肿瘤/正常）上验证，多类病理场景下优势类识别和遗忘策略需要扩展
- 遗忘/保留集划分完全基于预测熵，未利用特征空间的结构信息（如聚类密度）
- 像素级CAM伪标签质量受源模型质量限制，跨域场景下CAM本身可能不可靠
- 数据集规模偏小（GlaS仅67张训练图），更大规模数据集上的效果待验证
- 未探索目标域中存在源域没有的新类别的open-set场景

## 相关工作与启发

- **vs SFDA-DE (CVPR'22)**：经典分布估计SFDA，在预测偏差严重时PxAP反而恶化（28.0 vs source-only 36.9），多中心CL坍塌到50%
- **vs RGV (CVPR'25)**：不确定性控制SFDA，策略保守近似source-only水平（PxAP 34.7 vs 36.9），几乎无适应增益
- **vs ERL (ICLR'23)**：噪声标签学习处理域偏移，CL有改善但PxAP频繁下降（25.4），未解决定位退化
- 启发：预测偏差放大是自训练域适应的通用问题，去偏思路可推广到检测/分割等任务的SFDA

## 评分

- 新颖性: ⭐⭐⭐⭐ 将机器遗忘引入SFDA进行预测去偏，动机清晰、方法简洁有效
- 实验充分度: ⭐⭐⭐⭐ 三种WSOL模型×多个跨域设定，消融和可视化充分，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，Fig.1的偏差可视化非常直观
- 价值: ⭐⭐⭐⭐ 揭示并解决了SFDA+WSOL的核心瓶颈，对计算病理领域有实际部署意义

<!-- RELATED:START -->

## 相关论文

- [Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation](weakly_supervised_teacher-student_framework_with_progressive_pseudo-mask_refinem.md)
- [Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection](chips_efficient_clip_adaptation_via_curvature-aware_hybrid_influence-based_data_.md)

<!-- RELATED:END -->
