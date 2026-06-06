---
title: >-
  [论文解读] Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection
description: >-
  [ICCV 2025][多模态VLM][OOD detection] 提出APLGOS框架，利用视觉语言模型的提示学习能力，通过在类条件高斯分布的低概率区域合成虚拟OOD提示和图像，以更紧凑的决策边界区分已知和未知类别，在四个主流数据集上取得SOTA。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "OOD detection"
  - "提示学习"
  - "Gaussian outlier synthesis"
  - "视觉语言"
  - "对比学习"
---

# Adaptive Prompt Learning via Gaussian Outlier Synthesis for Out-of-distribution Detection

**会议**: ICCV 2025  
**arXiv**: 无  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: OOD detection, prompt learning, Gaussian outlier synthesis, vision-language model, contrastive learning

## 一句话总结

提出APLGOS框架，利用视觉语言模型的提示学习能力，通过在类条件高斯分布的低概率区域合成虚拟OOD提示和图像，以更紧凑的决策边界区分已知和未知类别，在四个主流数据集上取得SOTA。

## 研究背景与动机

OOD检测旨在让检测器在测试时区分训练中见过的类别（ID）和未见过的类别（OOD）。现有方法主要从ID数据中提取OOD伪样本来正则化模型的决策边界，但存在两个核心问题：(1) 从ID数据中提取的OOD伪样本质量不可控，难以充分覆盖OOD数据的真实分布；(2) 需要大量ID数据才能有效训练。合成方法（如GAN生成、VOS虚拟离群值）在一定程度上缓解了问题，但仍不够有效。作者观察到，视觉语言模型（VLM）拥有强大的预训练知识和表征能力，可以辅助生成更高质量的虚拟离群值，但尚无先前工作将提示学习应用于OOD检测任务。

## 方法详解

### 整体框架

APLGOS由两大模块组成：提示学习模块（PLM）和文本-图像对齐模块（TAM）。PLM负责生成ID提示和OOD伪提示，TAM通过对比学习计算图像和提示的相似度来对齐多模态数据。训练分三个阶段：第一阶段学习ID提示和对齐；第二阶段引入合成的OOD提示；第三阶段引入合成的OOD图像。所有OOD数据（提示和图像）均为虚拟合成，仅ID图像来自真实数据集。

### 关键设计

1. **ID提示生成（ChatGPT标准化Q&A）**: 预定义包含位置坐标和类别名称的Q&A对模板（如"Q: What is in the region with coordinates <loc1>,<loc2>,<loc3>,<loc4>? A: That's a <CLS>."），通过ChatGPT多轮标准化生成多样化的语句集合，训练时随机采样初始化可学习ID提示，引入位置信息实现更细粒度的区域级提示。

2. **OOD提示合成（高斯离群值采样）**: 假设ID提示嵌入在隐空间中服从类条件多元高斯分布，计算各类的经验均值和绑定协方差矩阵，在高斯分布的低概率（ε-似然）区域采样虚拟OOD提示。通过引入可学习的高斯噪声矩阵ε扩展采样空间，防止模型过度依赖ID类别分布。OOD图像也采用同样原理在图像嵌入空间的低概率区域合成。

3. **文本-图像对齐模块（TAM）**: 通过对比学习计算图像嵌入和提示嵌入的归一化相似度分数，结合对齐损失Lalign约束ID和OOD数据的决策边界。总损失还包含位置损失Lloc（隐式引入坐标信息）、分类损失Lcls和正则化项。

### 损失函数 / 训练策略

总损失L = ξ₁[γ₁τL_align^ID + γ₂(1-τ)L_align^OOD] + γ₃ξ₂[κL_loc^ID + (1-κ)L_loc^OOD] + γ₄ξ₃L_cls + γ₅ξ₄L_reg + W，其中ξ、τ、κ控制不同训练阶段使用的损失组合。三阶段训练策略逐步引入ID对齐、OOD对齐和虚拟OOD图像。ID与OOD数据比例约1:1，OOD提示采样数K=10000。

## 实验关键数据

### 主实验

| ID数据集 | OOD数据集 | 指标 | APLGOS (RegX4.0) | VOS (RegX4.0) | 提升 |
|----------|-----------|------|-------------------|---------------|------|
| PASCAL VOC | MS-COCO | FPR95↓ | 45.96% | 50.53% | -4.57% |
| PASCAL VOC | OpenImages | FPR95↓ | 47.10% | 50.27% | -3.17% |
| BDD-100k | MS-COCO | FPR95↓ | 39.48% | 42.82% | -3.34% |
| BDD-100k | OpenImages | FPR95↓ | 19.79% | 27.55% | -7.76% |
| PASCAL VOC | - | mAP↑ | 49.4% | 49.1% | +0.3% |

在BDD-100k + OpenImages组合上，FPR95降低7.76%，为最大提升。

### 消融实验

- **提示策略**: 位置信息<LOC>和ChatGPT标准化采样(RP)的组合效果最佳，FPR95从50.53%降至45.96%
- **OOD提示采样数K**: K=10000为最优，过小无法覆盖决策边界外区域，过大则随机性过强
- **高斯噪声强度α**: α=1.0最优，过小采样空间过窄，过大采样空间过大
- **ID/OOD比例**: 1:1最优，偏离该比例性能均下降

### 关键发现

- 在隐空间中合成OOD提示比直接从ID数据提取OOD伪样本更有效
- 位置信息对区域级OOD检测至关重要，引入坐标token后性能显著提升
- 该方法在使用较少ID数据时仍能保持优越性能

## 亮点与洞察

- 首次将提示学习引入OOD检测任务，巧妙利用VLM的知识实现更高质量的OOD合成
- ID提示、OOD提示和OOD图像全部为虚拟生成，减少了对真实OOD数据的依赖
- ChatGPT标准化Q&A的方式生成多样化提示，思路新颖且实用
- 高斯分布低概率区域采样的思路直观且理论基础扎实

## 局限与展望

- 方法依赖ChatGPT生成提示集合，增加了前置成本
- 高斯分布假设可能不适用于所有特征分布（实际分布可能非高斯）
- 仅在目标检测层面评估OOD，未扩展到图像分类、语义分割等任务
- 三阶段训练流程较复杂，超参数较多（α, β, γ, ξ, K等）

## 相关工作与启发

- VOS [Du et al.] 是主要基线，在特征空间合成虚拟离群值
- CoOp/CoCoOp 等提示学习方法为本文的提示设计提供了基础
- 思路可迁移到其他安全关键任务的OOD检测中

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将提示学习和高斯离群值合成结合用于OOD检测
- 技术深度: ⭐⭐⭐⭐ — 提示合成和对齐框架设计扎实
- 实验充分性: ⭐⭐⭐⭐ — 四个数据集、详尽消融、可视化分析
- 写作质量: ⭐⭐⭐⭐ — 条理清晰，图文配合好
- 实用价值: ⭐⭐⭐ — 复杂度较高，实际部署有一定门槛

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[ICCV 2025\] PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [\[ICCV 2025\] Attention to the Burstiness in Visual Prompt Tuning!](attention_to_the_burstiness_in_visual_prompt_tuning.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)

</div>

<!-- RELATED:END -->
