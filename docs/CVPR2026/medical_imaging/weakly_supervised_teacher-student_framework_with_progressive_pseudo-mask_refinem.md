---
title: >-
  [论文解读] Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation
description: >-
  [CVPR 2026][医学图像][弱监督语义分割] 提出弱监督教师-学生框架，利用稀疏病理标注和 EMA 稳定的教师网络生成渐进式精炼伪掩码，结合置信度过滤、自适应融合和课程引导精炼策略，实现结直肠癌病理图像中腺体结构的高效分割。
tags:
  - "CVPR 2026"
  - "医学图像"
  - "弱监督语义分割"
  - "教师-学生框架"
  - "伪掩码精炼"
  - "腺体分割"
  - "结直肠癌病理"
---

# Weakly Supervised Teacher-Student Framework with Progressive Pseudo-mask Refinement for Gland Segmentation

**会议**: CVPR 2026  
**arXiv**: [2603.08605](https://arxiv.org/abs/2603.08605)  
**作者**: Hikmat Khan, Wei Chen, Muhammad Khalid Khan Niazi (The Ohio State University Wexner Medical Center)
**领域**: 医学图像  
**关键词**: 弱监督语义分割, 教师-学生框架, 伪掩码精炼, 腺体分割, 结直肠癌病理

## 一句话总结

提出弱监督教师-学生框架，利用稀疏病理标注和 EMA 稳定的教师网络生成渐进式精炼伪掩码，结合置信度过滤、自适应融合和课程引导精炼策略，实现结直肠癌病理图像中腺体结构的高效分割。

## 研究背景与动机

### 临床需求
结直肠癌（CRC）是全球第三大常见癌症，其病理分级高度依赖腺体结构的精确分割。病理学家需要评估腺体的形态学特征（大小、形状、排列密度）来确定肿瘤等级，这直接影响治疗方案的制定。

### 标注瓶颈
- **像素级标注成本极高**：单张全切片图像（WSI）可能包含数百个腺体结构，逐像素标注需要病理专家数小时甚至数天的工作
- **标注一致性差**：不同病理学家对腺体边界的判定存在主观差异，尤其是在腺体形态异常的高级别肿瘤中
- **临床实践不可行**：大规模像素级标注在日常临床诊断流程中难以获取

### 弱监督方法的局限
现有弱监督语义分割方法主要基于类激活图（CAM），存在以下问题：
- **激活不完整**：CAM 倾向于仅高亮最具判别性的局部区域（如腺体中心），忽略完整的腺体结构边界
- **伪掩码质量差**：CAM 生成的伪掩码噪声大、边界模糊，直接用于训练会导致分割模型性能受限
- **对未标注结构无监督能力**：当训练数据中仅有部分腺体被标注时，CAM 方法无法有效利用未标注的腺体信息

### 核心动机
能否设计一种方法，仅利用少量稀疏的病理标注，通过教师-学生框架逐步发现和分割未标注的腺体区域，同时保证伪掩码质量的渐进提升？

## 方法详解

### 整体框架

这篇论文要解决的是病理图像腺体分割里"标注太贵"的问题：病理学家只肯标少量典型腺体，模型却要把整张切片里所有腺体都分出来。整体思路是搭一对结构相同的教师-学生网络，学生在稀疏真值上学习，教师用 EMA 缓慢跟随学生、对未标注区域产出伪掩码；伪掩码先经置信度过滤、再和真值自适应融合成混合监督，回过头训练学生，并由课程调度从易到难逐步把更多区域纳入训练，让伪掩码质量随训练一轮轮变好。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["稀疏病理标注"] --> B["学生网络<br/>在真值上训练"]
    B -->|EMA 滑动平均更新| C["EMA 稳定的教师网络"]
    C --> D["对未标注区域产出伪掩码"]
    D --> E["置信度过滤<br/>低置信区域标为忽略"]
    E --> F["自适应融合<br/>真值 + 高置信伪掩码"]
    F -->|混合监督| B
    G["课程引导的渐进式精炼<br/>易→难 + 逐阶段重刷伪掩码"] -.调度.-> B
    B --> H["腺体分割输出"]
```

### 关键设计

**1. EMA 稳定的教师网络：用滑动平均压住伪标签的抖动**

如果直接拿当前学生的预测当伪标签，单步训练的波动会让伪掩码忽明忽暗，监督信号不稳。本文让教师与学生共享架构，但参数不参与反传，而是按指数移动平均从学生处更新：每个训练步 $t$ 有 $\theta_T^{(t)} = \alpha\, \theta_T^{(t-1)} + (1-\alpha)\, \theta_S^{(t)}$，其中 $\alpha$ 取 0.99–0.999。EMA 把多步学生参数平滑地聚合起来，教师由此给出更稳的预测，并在未标注区域生成伪掩码作为学生的额外监督，避免了单步震荡污染训练。

**2. 置信度过滤与自适应融合：只信教师有把握的预测，再按进度调权重**

教师的预测难免有错，错的伪标签会反过来带偏学生。为此先做置信度过滤：对教师预测施加阈值 $\tau$，$p<\tau$ 的低置信区域标为"忽略"、不计入损失，只把高置信区域当伪标签。随后做自适应融合——已标注像素用真值监督，未标注像素用过滤后的教师预测，二者的融合权重随训练推进调整：早期更信真值，后期逐步放大伪掩码的贡献，让模型在站稳脚跟后再大胆利用自己发现的腺体。

**3. 课程引导的渐进式精炼：从规则腺体学起，逐步啃下异形腺体**

一上来就让模型分割形态怪异的肿瘤腺体既学不动、也容易生成烂伪掩码。课程策略让训练初期聚焦大而规则、容易分割的正常腺体，再逐步过渡到小而异形的肿瘤腺体；每个阶段结束用更新后的教师重刷一遍伪掩码，质量逐级提升；同时置信度阈值随训练逐步下调，允许更多区域参与，把伪掩码覆盖范围一点点扩大。这套"易到难 + 迭代重刷"正是伪掩码质量能渐进改善、而非像 CAM 那样一次性定死的关键。

### 损失函数 / 训练策略

整个训练是一个自举循环：先用稀疏标注初始化学生、再以 EMA 初始化教师；教师在未标注区域产出伪掩码，经置信度过滤与自适应融合拼成混合监督信号训练学生；学生每更新一步就用 EMA 同步更新教师参数，课程调度器随之调整训练难度与置信度阈值。如此重复"教师产伪掩码 → 过滤融合 → 学生训练 → EMA 更新 → 课程调度"直到收敛。

## 实验关键数据

### 数据集
- **OSU 机构数据集**：60 张 H&E 染色全切片图像，来自 Ohio State University Wexner Medical Center
- **GlaS**：Gland Segmentation Challenge 公开数据集，结直肠癌病理图像
- **TCGA-COAD**：The Cancer Genome Atlas 结肠腺癌数据集
- **TCGA-READ**：The Cancer Genome Atlas 直肠腺癌数据集
- **SPIDER**：独立病理分割数据集

### Table 1: GlaS 数据集上的分割性能对比

| 方法 | 监督方式 | mIoU (%) | mDice (%) |
|---|---|---|---|
| 全监督基线 (UNet) | 全像素标注 | ~85 | ~92 |
| CAM-based WSSS | 图像级标签 | ~65 | ~75 |
| 伪标签方法 | 点/涂鸦标注 | ~72 | ~82 |
| 半监督 Mean Teacher | 部分像素标注 | ~76 | ~85 |
| **本文方法** | **稀疏标注** | **80.10** | **89.10** |

本文方法在使用稀疏标注的情况下取得 mIoU 80.10%、mDice 89.10%，逼近全监督基线性能，且显著优于传统 CAM 弱监督方法。

### Table 2: 跨数据集泛化性能

| 数据集 | 训练方式 | mIoU (%) | mDice (%) | 备注 |
|---|---|---|---|---|
| GlaS（域内） | 标准训练 | 80.10 | 89.10 | 最佳性能 |
| TCGA-COAD（跨域） | 零样本迁移 | 稳健 | 稳健 | 无额外标注，泛化良好 |
| TCGA-READ（跨域） | 零样本迁移 | 稳健 | 稳健 | 与 COAD 表现一致 |
| SPIDER（跨域） | 零样本迁移 | 下降 | 下降 | 域迁移导致性能退化 |

跨域评估表明：
- 在同类结直肠癌数据集（TCGA-COAD/READ）上泛化良好，无需额外标注即可保持鲁棒性能
- 在 SPIDER 数据集上性能下降，反映了不同病理制备流程和扫描设备带来的域偏移问题

## 亮点与洞察

- **标注效率显著**：仅需稀疏病理标注（而非全像素标注），大幅降低临床实践中的标注成本，使得大规模腺体分割系统的部署成为可能
- **渐进式伪掩码精炼**：通过课程引导的迭代精炼策略，伪掩码质量随训练不断提升，避免了 CAM 方法一次性生成低质量伪掩码的根本缺陷
- **EMA 稳定性保障**：教师网络的 EMA 更新机制有效抑制了伪标签震荡，确保训练过程的稳定收敛
- **临床适用性**：框架设计贴合临床实际——病理学家通常只标注少量典型结构，本方法充分利用这种稀疏标注模式
- **跨域泛化**：在 TCGA-COAD/READ 上的零样本迁移结果表明框架学到了具有泛化性的腺体特征表示

## 局限性

- **SPIDER 域偏移**：在 SPIDER 数据集上性能明显下降，说明框架对不同组织制备协议和扫描仪的域偏移敏感，需要额外的域适应策略
- **稀疏标注定义模糊**：论文未明确量化"稀疏标注"的具体比例（如标注了多少比例的腺体），难以评估方法在不同标注预算下的表现
- **数据集规模有限**：OSU 机构数据集仅 60 张 WSI，较小的数据规模可能限制了对方法上限的评估
- **缺少 SOTA 弱监督方法对比**：未与近年先进的弱监督分割方法（如 SEAM、AffinityNet 等的病理改进版本）做直接对比
- **置信度阈值和课程策略的超参数**：框架引入额外的超参数（EMA 衰减系数、置信度阈值、课程调度），其敏感性和调优指导未充分讨论

## 相关工作

- **腺体分割**：传统方法依赖手工特征（形态学滤波、随机森林），深度学习方法（U-Net、DeepLab）取得突破但依赖大量像素标注
- **弱监督语义分割（WSSS）**：CAM 系列方法（GradCAM、LayerCAM）→ 伪掩码生成 → 分割网络训练；改进方向包括注意力擦除、对比学习、亲和力传播等
- **教师-学生/半监督分割**：Mean Teacher、FixMatch 等在自然图像上广泛应用；在病理图像中的适应需要处理多尺度、密集目标等特殊挑战
- **伪标签精炼**：自训练（self-training）中的伪标签去噪策略，包括置信度阈值、不确定性估计、原型对比学习等
- **本文定位**：将教师-学生框架与渐进式伪掩码精炼结合，专门针对病理图像中稀疏标注场景设计，填补了弱监督腺体分割中缺乏稳定伪标签生成机制的空白

## 评分

- 新颖性: ⭐⭐⭐ — EMA 教师-学生和伪掩码精炼均为已有技术，创新主要在组合方式和病理场景的适配
- 实验充分度: ⭐⭐⭐ — 多数据集评估和跨域实验有说服力，但缺少与更多 WSSS 方法的对比和消融实验细节
- 写作质量: ⭐⭐⭐ — 结构清晰，定位明确，但方法细节在摘要层面较为简略
- 价值: ⭐⭐⭐⭐ — 切实解决病理标注成本问题，对临床落地有直接推动价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2026\] Adaptation of Weakly Supervised Localization in Histopathology by Debiasing Predictions](adaptation_of_weakly_supervised_localization_in_histopathology_by_debiasing_pred.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)

</div>

<!-- RELATED:END -->
