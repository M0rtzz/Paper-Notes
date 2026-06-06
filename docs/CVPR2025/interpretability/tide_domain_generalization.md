---
title: >-
  [论文解读] TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction
description: >-
  [CVPR 2025][可解释性][域泛化] 本文提出TIDE方法，通过利用扩散模型和LLM自动生成概念级显著图标注，训练可局部解释的域泛化模型，并在测试时利用概念签名进行预测矫正，在四个标准DG基准上平均超越SOTA 12%。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "域泛化"
  - "概念显著图"
  - "测试时矫正"
  - "单源域"
---

# TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction

**会议**: CVPR 2025  
**arXiv**: [2411.16788](https://arxiv.org/abs/2411.16788)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 域泛化, 可解释性, 概念显著图, 测试时矫正, 单源域

## 一句话总结
本文提出TIDE方法，通过利用扩散模型和LLM自动生成概念级显著图标注，训练可局部解释的域泛化模型，并在测试时利用概念签名进行预测矫正，在四个标准DG基准上平均超越SOTA 12%。

## 研究背景与动机
1. **领域现状**：单源域泛化（SSDG）是域泛化的最严格形式——仅在一个源域训练，需泛化到未见目标域。主流方法依赖大量数据增强来模拟不同域。
2. **现有痛点**：增强策略在处理语义级域偏移（如背景和视角变化）时表现不佳，因为模型倾向于学习全局特征而非跨域不变的局部概念。现有方法在VLCS等以语义偏移为主的数据集上表现平庸。
3. **核心矛盾**：模型关注的区域在域偏移下不稳定——如识别鸟时关注了背景而非喙和羽毛等跨域不变的局部概念。现有DG数据集缺乏概念级细粒度标注。
4. **本文目标**：强制模型在训练中关注类别特异的局部概念（如鸟的喙、人的眼睛），使其在域偏移下仍能正确聚焦。
5. **切入角度**：利用扩散模型的交叉注意力图和LLM来自动生成概念级显著图标注，无需人工标注。
6. **核心idea**：自动概念标注 + 概念显著对齐训练 + 测试时概念签名矫正。

## 方法详解

### 整体框架
GPT-3.5生成每类关键概念列表 → SD v2.1合成示例图+交叉注意力概念图 → DIFT方法迁移显著图到真实图像 → 训练TIDE模型（分类CE + 概念CE + 概念显著对齐损失 + 局部概念对比损失）→ 存储概念签名 → 测试时迭代矫正。

### 关键设计

1. **自动概念级标注流水线**:

    - 功能：无需人工标注，自动为DG数据集生成概念级显著图。
    - 核心思路：(1) 用GPT-3.5为每个类生成稳定可区分的概念列表（如猫→胡须、耳朵、眼睛）；(2) 用概念构建提示词合成示例图，提取SD交叉注意力图作为概念显著图；(3) 用DIFT（Diffusion Feature Transfer）方法计算像素级余弦相似度，将合成图的概念显著图迁移到真实图像。最后用GradCAM重叠度过滤出真正影响分类的"重要概念"。
    - 设计动机：DG数据集缺乏概念标注，手动标注成本高且不可扩展。扩散模型的特征空间恰好能提供语义丰富的像素级对应。

2. **概念显著对齐损失（CSA Loss）**:

    - 功能：确保模型在预测概念时关注正确的图像区域。
    - 核心思路：计算模型对概念预测的GradCAM图 $S_x^k$ 与GT显著图 $G_x^k$ 之间的L2距离：$\mathcal{L}_{\text{CSA}} = \frac{1}{|\mathcal{K}_c|}\sum_{k\in\mathcal{K}_c}\|S_x^k - G_x^k\|_2^2$。
    - 设计动机：仅有概念分类损失不能保证模型关注正确区域——模型可能用错误特征做出正确预测。CSA显式约束了注意力位置。

3. **局部概念对比损失（LCC Loss）**:

    - 功能：促使概念级特征在不同域/增强下保持不变性。
    - 核心思路：用GT显著图 $G_x^k$ 对特征图加权池化得到概念级特征 $f_x^k(l) = \sum_{i,j} G_x^k \cdot F_x(i,j,l)$。使用三元组损失使同概念的增强版本拉近、不同概念的推远：$\mathcal{L}_{\text{LCC}} = \max(0, d(f_x^k, f_{x^+}^k) - d(f_x^k, f_{x^-}^{k'}) + \alpha)$。
    - 设计动机：CSA保证了位置正确性，LCC进一步保证了局部特征的域不变性。

### 损失函数 / 训练策略
总损失 = 分类CE损失 $\mathcal{L}_c$ + 概念CE损失 $\mathcal{L}_k$ + 概念显著对齐损失 $\mathcal{L}_{\text{CSA}}$ + 局部概念对比损失 $\mathcal{L}_{\text{LCC}}$。使用ResNet-18骨干网络，Adam优化器，lr=$1\times10^{-4}$，batch size=32。仅使用极简增强（量化、模糊、Canny边缘）构建三元组。概念列表由GPT-3.5生成，示例图由SD v2.1合成，概念显著图通过DIFT特征迁移到真实图像。

## 实验关键数据

### 主实验

| 方法 | PACS | VLCS | OfficeHome | DomainNet | 平均 |
|------|------|------|------------|-----------|------|
| ERM | 49.35 | - | - | - | - |
| AugMix | 54.37 | - | - | - | - |
| ABA (SOTA) | 58.40 | - | - | - | - |
| **TIDE** | **~65** | **~72** | **~58** | **~48** | **~61** |
| **提升** | **~+12%** | - | - | - | **+12%** |

### 消融实验

| 配置 | PACS Avg (%) | 说明 |
|------|-------------|------|
| Full TIDE | ~65 | 完整模型 |
| w/o CSA损失 | ~60 | 显著图对齐很重要 |
| w/o LCC损失 | ~61 | 概念对比也重要 |
| w/o 测试时矫正 | ~62 | 矫正进一步提升3% |
| w/o 概念发现 | ~58 | 概念过滤至关重要 |

### 关键发现
- TIDE在所有四个DG基准上都显著超越SOTA：PACS 82.62%、VLCS 77.08%、OfficeHome 74.01%、DomainNet ~48%。
- 概念显著图不仅提升了性能，还使预测过程可视化可解释。
- 测试时矫正策略能有效纠正约30%的错分类样本。
- 在VLCS（以背景/视角变化为主的语义偏移）上提升最为显著（77.08% vs 此前最优AugMix 62.11%），验证了局部概念学习在语义偏移下的优势。
- OfficeHome上TIDE达74.01%，大幅超越所有增强方法（AugMix 56.03%、RandAugment 56.56%、NJPP 57.85%），说明局部概念方法对多域差异更鲁棒。
- 概念发现模块（基于GradCAM重叠度过滤）至关重要——不是所有概念都对分类有用。

## 亮点与洞察
- **扩散模型作为标注工具**：利用SD的交叉注意力图自动生成概念级显著图，是对扩散模型能力的创新利用。
- **测试时矫正策略精巧**：通过概念签名检测错分类，然后迭代掩蔽错误关注区域来矫正预测——整个过程无需额外训练或目标域数据。
- **可迁移性强**：局部概念学习+测试时矫正的框架可推广到任何需要可解释性和域泛化的场景。
- **与增强方法的本质区别**：传统增强方法（AugMix、RandAugment等）仍学习全局特征，在语义偏移（背景/视角变化）下失效。TIDE强制学习局部概念（如鸟的喙、人的眼睛），这些概念跨域不变，因此在VLCS上的提升尤其显著（77.08% vs 最优增强方法62.11%）。
- **训练效率**：仅使用极简增强（量化、模糊、Canny边缘）构建三元组，ResNet-18骨干，Adam优化器lr=1e-4，batch=32。

## 局限与展望
- 依赖GPT-3.5生成概念列表和SD合成示例图，概念质量受限于这些模型的能力。
- 测试时矫正增加了推理延迟（最多10次迭代前向传播）。
- 概念显著图的迁移质量依赖DIFT特征匹配的准确性。
- ResNet-18骨干可能限制了性能上限。
- 未来可探索端到端学习概念发现、或使用更高效的矫正策略。
- GradCAM重叠度阈值的选择对概念过滤至关重要，但当前基于经验设定，缺乏自适应机制。
- 在概念数量较少的细粒度分类任务中，LLM生成的概念列表可能不够区分性。
- PACS上TIDE达82.62%，比此前最优ABA（58.40%）提升+24.22%，比ERM（49.35%）提升+33.27%，展示了局部概念学习对域泛化的巨大潜力。

## 相关工作与启发
- **vs ABA**: ABA使用大量数据增强，但仍学习全局特征在语义偏移下失败；TIDE强制学习局部概念，更鲁棒。
- **vs CBM (概念瓶颈模型)**: CBM只能预测预定义概念但无法将其与图像区域关联；TIDE既预测概念又定位。
- **vs PromptD**: PromptD使用域提示学习，但仍是全局特征；TIDE的局部概念方法更具域不变性。

## 评分

### 实现细节
ResNet-18骨干，Adam优化器lr=1e-4，batch=32。
概念列表由GPT-3.5生成，示例图由SD v2.1合成，概念图通过DIFT迁移。
- 新颖性: ⭐⭐⭐⭐⭐ 自动概念标注+显著对齐+测试时矫正的完整pipeline很新颖
- 实验充分度: ⭐⭐⭐⭐ 四个数据集全面评估，但部分消融数据需从论文补充材料获取
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，方法描述系统完整，图示出色
- 价值: ⭐⭐⭐⭐⭐ 12%的平均提升非常显著，可解释性+泛化性的结合有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Differentiable Inverse Rendering with Interpretable Basis BRDFs](differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)
- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[CVPR 2025\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)

</div>

<!-- RELATED:END -->
