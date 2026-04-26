---
title: >-
  [论文解读] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation
description: >-
  [CVPR 2025][医学图像][腺体分割] 本文提出一种弱监督教师-学生框架，利用稀疏病理学家标注和 EMA 稳定的教师网络生成渐进式精炼的伪掩码，在腺体分割任务上以远少于全监督的标注量达到 mIoU 80.10 和 mDice 89.10 的优异性能。
tags:
  - CVPR 2025
  - 医学图像
  - 腺体分割
  - 弱监督语义分割
  - 教师-学生框架
  - 伪掩码精炼
  - 结直肠癌
---

# Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation

**会议**: CVPR 2025  
**arXiv**: [2603.08605](https://arxiv.org/abs/2603.08605)  
**代码**: 无  
**领域**: 医学图像 / 弱监督分割  
**关键词**: 腺体分割, 弱监督语义分割, 教师-学生框架, 伪掩码精炼, 结直肠癌

## 一句话总结
本文提出一种弱监督教师-学生框架，利用稀疏病理学家标注和 EMA 稳定的教师网络生成渐进式精炼的伪掩码，在腺体分割任务上以远少于全监督的标注量达到 mIoU 80.10 和 mDice 89.10 的优异性能。

## 研究背景与动机

**领域现状**：结直肠癌的组织病理学分级依赖于腺体结构的精确分割。深度学习方法在全监督条件下已取得很好效果，但需要大量像素级标注——制作一张全标注的全切片图像（WSI）可能需要病理学家数小时甚至数天的工作。

**现有痛点**：弱监督语义分割（WSSS）是低标注成本的替代方案，但现有基于类激活图（CAM）的方法存在严重局限：CAM 倾向于只关注最具判别性的区域（通常是腺体的局部），生成的伪掩码不完整、边界模糊。更关键的是，对于未标注的腺体区域，CAM 方法完全无法提供监督信号，导致这些区域被忽略。

**核心矛盾**：病理图像中腺体形态多样且密集分布，仅凭少量标注点很难推广到所有腺体结构。简单地用不完整的伪掩码训练会导致学生网络继承并放大教师的错误，形成恶性循环。

**本文目标**：设计一个标注高效的框架，能从稀疏标注出发，渐进式地发现和分割所有腺体区域，同时确保伪标签质量。

**切入角度**：利用教师-学生（Teacher-Student）架构的互补特性——教师网络通过 EMA 平滑保持稳定的预测，学生网络快速学习最新特征。两者的融合可以逐步扩展到未标注区域。

**核心 idea**：用 EMA 教师网络生成稳定预测，通过置信度过滤和自适应融合将教师预测与有限真实标注混合形成伪掩码，再用课程学习策略渐进提升伪掩码的覆盖范围和质量。

## 方法详解

### 整体框架
输入为 H&E 染色的病理切片图像及其稀疏标注（部分腺体的像素级标注）。框架包含一个学生网络和一个 EMA 教师网络。学生网络在每个训练步正常更新梯度，教师网络的参数通过 $\theta_T \leftarrow \alpha \theta_T + (1-\alpha) \theta_S$ 指数移动平均更新。训练循环为：教师网络预测 → 置信度过滤 → 与真实标注自适应融合 → 生成精炼伪掩码 → 学生网络训练。

### 关键设计

1. **EMA 稳定教师网络**:

    - 功能：产生时间平滑的稳定预测，避免学生网络训练初期的噪声预测反馈
    - 核心思路：教师参数为学生参数的指数移动平均，动量系数随训练递增（从 0.99 到 0.999）。由于 EMA 平均了多步的学生参数，教师的预测比任何单一时刻的学生网络更稳定，尤其在训练早期能提供更可靠的伪标签
    - 设计动机：Mean Teacher 范式已在半监督学习中证明有效，本文将其扩展到弱监督场景，应对伪标签质量不稳定的挑战

2. **置信度过滤与自适应融合**:

    - 功能：从教师预测中筛选高质量区域，并与有限真实标注融合
    - 核心思路：对教师网络的预测概率图设置动态阈值（如 >0.7 视为高置信度），只保留高置信区域作为伪标签。在有真实标注的区域始终使用真实标注；在无标注但教师高置信的区域使用教师预测；在低置信区域不提供监督信号（忽略损失）。融合公式为 $M_{fused}(x) = \mathbb{1}_{GT}(x) \cdot M_{GT}(x) + \mathbb{1}_{conf}(x) \cdot M_{teacher}(x)$
    - 设计动机：直接用教师的全部预测作为伪标签会引入大量噪声。置信度过滤确保只传递可靠信息，自适应融合在有真实标注处保持准确性

3. **课程引导的渐进精炼**:

    - 功能：随训练推进逐步扩大伪掩码的覆盖面
    - 核心思路：训练初期使用较高的置信度阈值（只采纳最可靠的伪标签），随着模型能力增强逐步降低阈值，纳入更多教师预测区域。这形成了一个课程学习策略——先学习容易的高置信区域，再逐步挑战低置信区域。同时，伪掩码的更新频率也随训练调整，早期更频繁地更新以快速扩展覆盖，后期降低频率以保持稳定
    - 设计动机：一次性用所有伪标签训练容易在噪声标签上过拟合。课程式渐进策略让模型逐步建立对腺体形态的全面理解

### 损失函数 / 训练策略
使用标准的交叉熵损失和 Dice 损失的组合 $\mathcal{L} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{Dice}$，损失仅在有标注或高置信伪标签的像素上计算。低置信区域被标记为忽略区域，不参与梯度回传。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | 全监督上界 | 弱监督SOTA |
|--------|------|------|---------|-----------|
| GlaS | mIoU | 80.10 | 85.20 | 74.30 |
| GlaS | mDice | 89.10 | 92.10 | 83.50 |
| TCGA-COAD | mDice | 84.70 | 88.50 | 78.20 |
| TCGA-READ | mDice | 82.30 | 87.10 | 76.80 |
| SPIDER | mDice | 71.50 | 82.40 | 65.10 |

### 消融实验

| 配置 | GlaS mIoU | GlaS mDice | 说明 |
|------|----------|-----------|------|
| Full model | 80.10 | 89.10 | 完整框架 |
| w/o EMA 教师 | 73.40 | 82.60 | 无稳定教师，-6.7 mIoU |
| w/o 置信度过滤 | 75.80 | 85.20 | 噪声伪标签导致 -4.3 |
| w/o 渐进精炼 | 77.20 | 86.50 | 静态阈值，-2.9 |
| w/o 自适应融合 | 76.50 | 85.90 | 全用教师预测，-3.6 |

### 关键发现
- EMA 教师网络是最关键的组件，去除后性能严重下降 6.7 mIoU，说明稳定的伪标签是成功的基础
- 在 SPIDER 数据集上性能明显下降（域偏移），反映了框架对训练数据域分布的依赖
- 与全监督方法的差距仅 5 mIoU 左右，但标注成本可降低 80% 以上
- 跨数据集评估（TCGA-COAD/READ）无需额外标注仍有不错效果，泛化能力较强

## 亮点与洞察
- **渐进式伪标签策略的优雅设计**：置信度阈值的动态调整模拟了"由易到难"的学习过程，这种课程学习思路可推广到其他弱监督任务（如弱监督目标检测、点标注分割等）
- **稀疏标注的高效利用**：框架充分利用少量精确标注作为锚点，通过教师网络向未标注区域"扩散"知识，实现了标注效率的数量级提升
- **即插即用的框架设计**：教师-学生+EMA 的架构与具体的分割网络结构解耦，可以替换为任何先进的分割backbone

## 局限与展望
- SPIDER 数据集上的性能下降表明框架对域偏移（domain shift）敏感，未来需要结合域适应技术
- 当前只针对腺体分割验证，更复杂的多类分割（如同时分割腺体和间质）的效果未知
- 置信度阈值的调度策略目前是预设的固定 schedule，未来可考虑自适应调整
- 对于极稀疏标注（如每张图只标一个腺体）的极端场景，框架的鲁棒性有待验证

## 相关工作与启发
- **vs CAM-based WSSS**：传统 CAM 方法只能定位最显著区域，且对背景区域容易生成假阳性。本文通过教师-学生框架和置信度过滤有效避免了这两个问题
- **vs Mean Teacher（半监督）**：原始 Mean Teacher 假设有大量无标注数据和少量精确标注。本文的创新在于将伪掩码精炼机制与 Mean Teacher 结合，适应了弱监督（标注不完整而非无标注）的场景
- **vs SAM（通用分割）**：SAM 需要提示（点/框），而本文的方法可以自动发现未标注的腺体区域

## 评分
- 新颖性: ⭐⭐⭐ 各组件（EMA、伪标签、课程学习）都不算新，但组合和应用有新意
- 实验充分度: ⭐⭐⭐⭐ 多个数据集验证含跨域评估，消融完善
- 写作质量: ⭐⭐⭐⭐ 结构清晰，医学背景交代充分
- 价值: ⭐⭐⭐ 实用价值高，可显著降低病理学标注成本

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2025\] CARL: A Framework for Equivariant Image Registration](carl_a_framework_for_equivariant_image_registration.md)
- [\[CVPR 2025\] WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression](wise_a_framework_for_gigapixel_whole-slide-image_lossless_compression.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)

<!-- RELATED:END -->
