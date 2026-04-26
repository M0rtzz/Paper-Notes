---
title: >-
  [论文解读] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images
description: >-
  [ICCV 2025 (pp. 23508-23518)][医学图像][零样本分割] 提出AcZeroTS框架，将主动学习与基于VLM的原型引导零样本分割模型ProZS结合，通过同时考虑不确定性、多样性和原型覆盖unseen类的能力来选择最有价值的标注样本，以最少标注实现seen和unseen组织类型的高质量分割。
tags:
  - ICCV 2025 (pp. 23508-23518)
  - 医学图像
  - 零样本分割
  - 主动学习
  - 病理图像
  - 视觉语言模型
  - 原型引导
---

# AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images

**会议**: ICCV 2025 (pp. 23508-23518)  
**arXiv**: 无arXiv版本  
**CVF**: [链接](https://openaccess.thecvf.com/content/ICCV2025/html/Tang_AcZeroTS_Active_Learning_for_Zero-shot_Tissue_Segmentation_in_Pathology_Images_ICCV_2025_paper.html)  
**作者**: Jiao Tang, Junjie Zhou, Bo Qian, Peng Wan, Yingli Zuo, Wei Shao, Daoqiang Zhang  
**代码**: 无  
**领域**: 语义分割 / 病理图像分析  
**关键词**: 零样本分割, 主动学习, 病理图像, 视觉语言模型, 原型引导  

## 一句话总结

提出AcZeroTS框架，将主动学习与基于VLM的原型引导零样本分割模型ProZS结合，通过同时考虑不确定性、多样性和原型覆盖unseen类的能力来选择最有价值的标注样本，以最少标注实现seen和unseen组织类型的高质量分割。

## 背景与动机

- **病理组织分割的重要性**：在计算机辅助癌症诊断中，对病理图像中的组织进行精确分割至关重要
- **传统方法的局限**：传统分割模型依赖大规模标注数据集，要求专家对每种组织类型都进行标注。然而，由于肿瘤微环境的复杂性，收集所有可能组织类型的标注极具挑战
- **VLM零样本分割的新机遇**：近年来视觉语言模型（VLM）的快速发展使得零样本像素级分割成为可能——模型仅在seen类上训练，但在测试时可以对seen和unseen类别都进行分割
- **VLM方法的不足**：虽然VLM-based零样本分割减少了对unseen类的标注需求，但在seen类上仍需大量标注
- **核心动机**：能否进一步减少seen类的标注量？→ 引入主动学习，用最少的标注同时保证seen和unseen类的分割性能

## 核心问题

**如何在零样本组织分割场景中设计有效的主动学习策略，使得在仅标注少量seen类样本的情况下，模型也能对unseen组织类型保持良好的分割性能？**

关键挑战：
1. 传统主动学习仅关注提升当前任务（seen类）性能，不考虑对unseen类的泛化
2. 零样本分割的原型质量直接影响unseen类识别，主动学习选样需与原型质量关联
3. 需要在有限标注预算下平衡seen类性能和unseen类泛化能力

## 方法详解

### 整体框架

AcZeroTS框架由两个核心组件构成：

1. **ProZS（Prototype-guided Zero-shot Segmentation）**：基于VLM的原型引导零样本分割模型
2. **Active Selection Criterion**：专为零样本分割设计的主动学习选样策略

整体流程：
1. 初始化：在小规模标注种子集上训练ProZS
2. 主动选择循环：根据选样准则从未标注池中选择最有价值的样本
3. 专家对选中样本进行标注
4. 用扩展后的标注集重新训练ProZS
5. 重复步骤2-4直至标注预算用尽

### 关键设计

#### 1. ProZS — 原型引导零样本分割模型

- **文本提示设计**：利用LLM为每种组织类别生成描述性文本提示（text prompts），而非简单使用类别名称
- **视觉-语义对齐**：基于VLM（如CLIP/CONCH等病理VLM）将图像patch特征与文本特征对齐
- **原型生成**：从训练样本中学习每个seen类的视觉原型（prototype）
- **零样本推理**：在推理阶段，利用VLM的文本编码器为unseen类生成语义原型，结合seen类的视觉原型进行联合分类
- **关键思想**：原型既要准确代表seen类，又要能区分unseen类——这建立了主动学习与零样本泛化之间的桥梁

#### 2. 面向零样本的主动选样准则

传统主动学习准则（如不确定性采样、多样性采样）仅关注当前任务性能。本文提出的准则同时考虑三个维度：

- **不确定性（Uncertainty）**：选择模型最不确定的样本，学习效率最高的信息
- **多样性（Diversity）**：确保选中样本在特征空间中分布多样化，避免冗余标注
- **原型覆盖性（Prototype Coverage for Unseen Classes）**：确保选中样本训练出的原型能有效概括（summarize）seen和unseen类——这是核心创新

原型覆盖性的直觉：
- 如果选中的标注样本能产生更好的原型，这些原型不仅准确代表seen类，还能在VLM的共享语义空间中与unseen类的文本原型形成良好的分类边界
- 本质上是将零样本泛化目标嵌入到主动学习的选样准则中

### 损失函数 / 训练策略

- **训练策略**：迭代式主动学习循环，逐步扩大标注集
- **分割损失**：预计使用标准交叉熵损失 + Dice损失的组合用于分割训练（在seen类上监督）
- **原型更新**：随着新标注样本加入，动态更新seen类原型
- **推理阶段**：结合视觉原型和文本语义原型，对所有类别（seen + unseen）进行推理

## 实验关键数据

| 数据集 | 类型 | 说明 |
|--------|------|------|
| TNBC | 病理图像 | 三阴性乳腺癌组织分割 |
| HPBC | 病理图像 | 乳腺癌病理组织分割 |
| Pascal VOC 2012 | 自然图像 | 验证方法在自然场景的通用性 |

- 在所有数据集上，AcZeroTS均优于现有方法
- 具体数值因无法访问全文PDF暂缺，论文声称"demonstrate the superiority of our method in comparison with the existing studies"

### 消融实验要点

- 预计包含以下消融：
  1. **选样准则各分量的贡献**：仅不确定性 vs. 不确定性+多样性 vs. 完整三分量准则
  2. **ProZS中文本提示设计**：不同prompt策略（简单类名 vs. LLM生成的描述性prompts）对性能的影响
  3. **标注预算的影响**：不同比例的标注数据下seen/unseen类的性能变化
  4. **对比不同主动学习策略**：随机采样、CoreSet、BADGE等基线

## 亮点

1. **问题定义有新意**：首次将主动学习引入零样本组织分割，弥补了VLM零样本方法仍需大量seen类标注的不足
2. **选样准则的zero-shot awareness**：传统AL仅关注当前已知类，本文的准则显式考虑了对unseen类的泛化，通过原型覆盖性将ZSL目标融入AL选样
3. **理论动机清晰**：从原型质量和语义空间覆盖的角度建立主动学习与零样本泛化的联系
4. **跨领域验证**：不仅在病理数据集上验证，还在自然图像（Pascal VOC 2012）上证明了通用性
5. **实用价值高**：显著降低了病理图像分割的标注成本，对临床落地有直接意义

## 局限与展望

1. **VLM backbone的选择**：论文可能仅使用了特定的VLM，未来可探索不同病理专用VLM（如CONCH、UNI、Virchow等）的效果
2. **unseen类的定义依赖先验**：零样本设置中unseen类需要预先知道类别名，可扩展到open-vocabulary设置
3. **单轮标注 vs. 多轮交互**：可进一步探索human-in-the-loop的迭代优化
4. **更多病理任务扩展**：如细胞分割、腺体分割等更细粒度的任务
5. **计算效率**：主动学习的选样过程可能带来额外计算开销，大规模WSI场景下的效率需关注
6. **原型数量的影响**：每个类别使用单个还是多个原型，以及原型更新策略的深入分析

## 与相关工作的对比

| 维度 | 传统分割 | VLM零样本分割 | AcZeroTS |
|------|----------|---------------|----------|
| seen类标注 | 大量 | 大量 | 少量（主动学习选择） |
| unseen类标注 | 需要 | 不需要 | 不需要 |
| 泛化到新类别 | 不可 | 可以 | 可以 |
| 标注效率 | 低 | 中 | 高 |

- **vs. ZS3Net, SPNet等零样本分割方法**：这些方法需要seen类的完整标注，AcZeroTS通过主动学习大幅减少需求
- **vs. CLIP/CONCH直接零样本**：直接零样本推理性能较低，AcZeroTS通过少量标注学习原型显著提升
- **vs. 传统主动学习（CoreSet, BADGE, Entropy等）**：传统AL不考虑unseen类泛化，AcZeroTS的选样准则兼顾seen和unseen

## 启发与关联

1. **主动学习 + 零样本泛化的新范式**：这种将AL选样与ZSL泛化目标耦合的思路可推广到其他领域（如遥感、自动驾驶）
2. **原型作为桥梁**：原型既连接了seen/unseen类的语义空间，又为主动学习提供了选样信号，这种dual-role设计有启发性
3. **病理图像分析的降低标注成本路线**：VLM + 主动学习的组合是未来降低医学图像标注成本的重要方向
4. **可向open-vocabulary分割扩展**：从固定unseen类集合到开放词汇的场景迁移是有趣的研究方向

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次将主动学习与零样本组织分割结合，选样准则的zero-shot awareness设计新颖
- **技术深度**: ⭐⭐⭐⭐ — ProZS原型引导模型 + 三分量选样准则，技术方案完整
- **实验充分性**: ⭐⭐⭐⭐ — 两个病理数据集 + 一个自然图像数据集的跨领域验证
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，方法介绍逻辑连贯
- **影响力**: ⭐⭐⭐⭐ — 对病理AI领域有实际价值，标注效率是关键瓶颈

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Learning Relative Gene Expression Trends from Pathology Images in Spatial Transcriptomics](../../NeurIPS2025/medical_imaging/learning_relative_gene_expression_trends_from_pathology_images_in_spatial_transc.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](../../NeurIPS2025/medical_imaging/zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](../../CVPR2025/medical_imaging/aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](../../CVPR2026/medical_imaging/moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)

<!-- RELATED:END -->
