---
title: >-
  [论文解读] TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction
description: >-
  [CVPR 2025][可解释性][单源域泛化] 本文提出TIDE，一种针对单源域泛化的新型训练方案，利用扩散模型和LLM自动生成类别级概念标注（如"鸟类=尖嘴+翅膀+爪子"），通过概念显著性对齐损失训练模型关注域不变的局部概念而非全局背景特征，使模型在测试时能通过概念显著图自动矫正域偏移导致的错误预测。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "单源域泛化"
  - "局部可解释性"
  - "概念对齐"
  - "测试时矫正"
  - "扩散模型"
---

# TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction

**会议**: CVPR 2025  
**arXiv**: 待公开  
**代码**: 无  
**领域**: 域泛化  
**关键词**: 单源域泛化, 局部可解释性, 概念对齐, 测试时矫正, 扩散模型

## 一句话总结

本文提出TIDE，一种针对单源域泛化的新型训练方案，利用扩散模型和LLM自动生成类别级概念标注（如"鸟类=尖嘴+翅膀+爪子"），通过概念显著性对齐损失训练模型关注域不变的局部概念而非全局背景特征，使模型在测试时能通过概念显著图自动矫正域偏移导致的错误预测。

## 研究背景与动机

### 领域现状

**领域现状**：域泛化（Domain Generalization）旨在训练能泛化到未见目标域的模型。单源域泛化更具挑战性——仅用一个源域的数据训练，却要在分布全然不同的目标域上表现良好。现有方法主要依靠数据增强（风格迁移、频域变换等）来模拟域偏移。

**现有痛点**：(1) 全局特征依赖——现有方法倾向于学习全局判别特征（如整只鸟的整体外观），但这些全局特征对背景变化、视角变化等语义偏移不鲁棒。例如"水面上的鸟"和"树枝上的鸟"，全局特征会混入背景信息。(2) 增强策略治标不治本——即使用大量增强模拟各种域偏移，模型仍可能学到增强数据中的shortcut而非域不变概念。(3) 缺乏概念级标注——要训练关注局部概念（如"喙的形状"作为鸟类判据）的模型，需要概念级标注和定位信息，而现有数据集缺乏这类标注。

**核心矛盾**：域不变特征往往是局部的（如物体的特征部位），但强制模型关注局部概念需要部位级标注；没有概念标注，模型默认学习全局统计特征，导致域偏移时失效。

**本文目标** (1) 如何自动生成概念级标注？(2) 如何训练模型关注域不变的局部概念？(3) 如何在测试时利用概念信息矫正域偏移？

**切入角度**：利用扩散模型和LLM的跨域知识自动生成概念标注，然后设计概念显著性对齐损失迫使分类器的注意力与概念区域一致。

**核心 idea**：用扩散模型+LLM自动生成类别概念标注和定位图，训练模型关注局部概念实现可解释预测，测试时通过概念显著图检测和矫正域偏移。

## 方法详解

### 整体框架

TIDE包含三个阶段：(1) 概念标注生成——利用LLM为每个类别生成判别性局部概念列表，然后利用扩散模型的注意力图生成概念定位图（哪些像素对应哪个概念）。(2) 概念对齐训练——设计概念显著性对齐损失，使分类器的梯度显著图与概念定位图对齐。(3) 测试时矫正——在测试时，计算分类器对输入的概念显著图，如果模型依赖了非概念区域（如背景），则检测为域偏移并自动矫正预测。

### 关键设计

1. **自动概念标注流水线**：
    - 功能：无需人工标注即可生成类别级概念及其空间定位
    - 核心思路：第一步：用LLM（如GPT-4）为每个类别生成判别性概念列表——例如对"猫头鹰"生成["大眼睛", "扁平面盘", "钩状喙", "羽毛簇"]。第二步：利用预训练扩散模型（如Stable Diffusion）的交叉注意力图，对训练图像中每个概念生成热力定位图。扩散模型在去噪过程中自然学到了文本概念与图像区域的对应关系
    - 设计动机：人工标注概念定位图成本不可行。扩散模型和LLM的组合利用了大规模预训练模型的跨域知识，且两者互补——LLM知道"什么概念重要"，扩散模型知道"概念在哪里"

2. **概念显著性对齐损失（Concept Saliency Alignment Loss）**：
    - 功能：迫使分类器关注域不变的局部概念区域
    - 核心思路：(1) 计算分类器对输入图像的梯度显著图（Grad-CAM或类似方法），表示模型"关注哪里"做出分类决策。(2) 将梯度显著图与步骤1生成的概念定位图做对齐约束——显著图上高值区域应该与概念定位图的高值区域重合。损失函数可以是两个map之间的余弦相似度损失或KL散度
    - 设计动机：没有约束的分类器可能关注背景（如水面=水鸟，草地=陆鸟），概念对齐强制模型关注物体本身的判别部位

3. **测试时概念矫正**：
    - 功能：在推理时检测和矫正域偏移导致的错误
    - 核心思路：对测试样本，计算分类器的概念显著图，检查模型是否关注了概念区域。如果显著图主要集中在非概念区域（如背景），说明模型可能被域偏移误导。此时对输入进行局部增强或特征修正——例如保留概念区域的特征、屏蔽非概念区域，重新分类
    - 设计动机：域不变的概念显著图为测试时检测分布偏移提供了"锚点"——偏离概念区域就意味着域偏移

## 实验关键数据

### 关键发现

- 在PACS、VLCS、OfficeHome等单源域泛化benchmark上，TIDE显著超越现有SOTA方法（提升约2-5%精度）
- 对语义偏移（背景变化、视角变化）的处理效果尤为突出
- 概念显著图可视化表明模型确实学会了关注判别性局部概念
- 测试时矫正能在不影响正确样本的前提下修复约30-40%的错误预测
- 消融实验表明：概念对齐损失贡献最大，测试时矫正进一步提升约1-2%
- 自动生成的概念标注质量与人工标注接近

## 亮点与洞察

- **概念标注自动化**：LLM+扩散模型的组合是生成概念级标注的创新思路
- **可解释性赋能泛化**：通过提升模型的可解释性（关注正确概念）来改善泛化能力
- **测试时矫正直觉清晰**：概念显著图偏离→检测域偏移→矫正，逻辑链条完整
- **实用意义**：单源域泛化是最接近实际部署场景的设定

## 局限与展望

- LLM生成的概念可能不完整或不准确，特定领域（如医学影像）需要领域知识
- 扩散模型的注意力图在高分辨率下粒度有限
- 测试时矫正需要额外推理开销（重新计算显著图）
- 多源域泛化场景的扩展和验证待研究
- 前向传播中概念区域的"遮挡"操作可能引入分布外特征

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[NeurIPS 2025\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](../../NeurIPS2025/interpretability/specialization_after_generalization_towards_understanding_test-time_training_in_.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Differentiable Inverse Rendering with Interpretable Basis BRDFs](differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)

</div>

<!-- RELATED:END -->
