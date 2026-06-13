---
title: >-
  [论文解读] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability
description: >-
  [CVPR 2025][图像生成][扩散模型偏见] 本文提出DiffLens框架，通过稀疏自编码器（k-SAE）将扩散模型内部神经元解缠为单语义特征空间，再用基于梯度的归因方法定位驱动偏见生成的特定特征，从而实现对性别、种族等社会偏见的精细控制和缓解，同时保持图像质量。 领域现状：扩散模型虽然生成质量卓越…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型偏见"
  - "机制可解释性"
  - "稀疏自编码器"
  - "偏见特征"
  - "偏见缓解"
---

# Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability

**会议**: CVPR 2025  
**arXiv**: [2503.20483](https://arxiv.org/abs/2503.20483)  
**代码**: [项目页](https://foundation-model-research.github.io/difflens)  
**领域**: 图像生成  
**关键词**: 扩散模型偏见, 机制可解释性, 稀疏自编码器, 偏见特征, 偏见缓解

## 一句话总结
本文提出DiffLens框架，通过稀疏自编码器（k-SAE）将扩散模型内部神经元解缠为单语义特征空间，再用基于梯度的归因方法定位驱动偏见生成的特定特征，从而实现对性别、种族等社会偏见的精细控制和缓解，同时保持图像质量。

## 研究背景与动机

**领域现状**：扩散模型虽然生成质量卓越，但经常生成带有社会偏见的内容——性别、种族、年龄等方面的刻板印象。现有偏见缓解方法主要分为两类：从头训练/微调（资源密集）和引导/编辑生成过程（忽略模型内在机制）。

**现有痛点**：引导生成的方法不理解模型内部驱动偏见的具体机制，可能过度矫正或意外影响非目标属性。直接编辑隐空间（如h-space）由于神经元多语义性（一个神经元对应多个无关概念），编辑一个属性会连带改变其他属性。

**核心矛盾**：需要精确定位并修改产生偏见的具体模型组件，但扩散模型的生成性质和神经元多语义性使得细粒度分析极为困难。

**本文目标**：发现扩散模型内部因果性地驱动偏见输出的决策机制（bias features），并通过调控这些特征精确控制偏见水平。

**切入角度**：借鉴LLM可解释性中的稀疏自编码器方法，将扩散模型的神经元激活解缠为稀疏的单语义特征空间。

**核心 idea**：k-SAE解缠 → 梯度归因定位偏见特征 → 缩放偏见特征控制输出分布。

## 方法详解

### 整体框架
三阶段流程：（1）用k-SAE将U-Net bottleneck层的隐状态映射到高维稀疏语义空间；（2）训练轻量分类器评估偏见属性概率，用积分梯度法计算每个语义特征对偏见的归因分数；（3）通过放大/抑制得分最高的偏见特征来控制生成内容的偏见水平。

### 关键设计

1. **k-稀疏自编码器（k-SAE）解缠**:

    - 功能：将多语义神经元激活解缠为单语义特征空间
    - 核心思路：编码器 $s = \text{TopK}(W_{enc}(h - b_{pre}))$ 将n维隐状态映射为m维稀疏向量（m≫n），只保留k个最大激活值（fired features），其余置零。解码器重建原始隐状态。训练目标为重建MSE最小化
    - 设计动机：原始神经元是多语义的，一个神经元参与多个不相关概念的编码。k-SAE在高维空间中分离出单语义特征，使得每个维度对应一个明确的语义概念，为精确编辑奠定基础

2. **基于梯度的偏见特征定位**:

    - 功能：识别语义空间中与偏见生成最相关的特征
    - 核心思路：定义偏见度量 $F_x(s) = \Pr(y|s)$（y为性别/种族类别），用积分梯度法计算每个语义特征 $s_i$ 的归因分数 $S(s_i; x) = (s_i - s'_i) \cdot \int_0^1 \frac{\partial F_x(s' + \alpha(s-s'))}{\partial s_i} d\alpha$。在N个生成样本上聚合归因分数，选top-τ个最高归因特征作为偏见特征
    - 设计动机：积分梯度法既考虑特征值的大小也考虑其对输出的边际效应，比简单梯度或激活值更准确。只需做一次即可永久标记偏见特征

3. **偏见特征缩放控制**:

    - 功能：通过调整偏见特征的激活值来控制偏见水平
    - 核心思路：对于identified的偏见特征集合A中的每个特征，乘以缩放因子——抑制（乘小系数）减少偏见，放大（乘大系数）增加偏见。修改后的特征经解码器映射回原始隐空间继续生成
    - 设计动机：因为特征是单语义的，缩放单个特征不会影响其他属性，实现了精细粒度控制

### 损失函数 / 训练策略
k-SAE用重建MSE损失训练。偏见分类器用生成样本的属性标签（如性别分类器）训练。

## 实验关键数据

### 主实验（CelebA-HQ无条件生成，性别偏见）

| 方法 | 偏见缓解效果 | FID变化 | 其他属性保持 |
|------|------------|---------|------------|
| Prompt Editing | 中等 | 有退化 | 差（影响其他属性） |
| Attention Editing | 中等 | 轻微退化 | 中等 |
| Fair Diffusion | 较好 | 有退化 | 中等 |
| **DiffLens** | **最好** | **最小退化** | **最好** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| 直接编辑h-space（无SAE） | 改变性别同时意外改变发型、肤色等 |
| DiffLens编辑 | 精确改变性别，其他属性保持不变 |
| 不同k-SAE特征对应的语义 | 发现不同特征分别控制发型、肤色、面部结构等细粒度属性 |

### 关键发现
- DiffLens发现的偏见特征确实控制着性别、种族等属性的生成，调控后效果显著
- 与直接h-space编辑相比，DiffLens的编辑不会意外改变非目标属性（如改性别时不变发型）
- k-SAE解缠后的特征具有清晰的语义对应——不同特征分别控制头发长度、皮肤颜色、佩戴眼镜等
- 方法同时适用于无条件和条件（Stable Diffusion）扩散模型

## 亮点与洞察
- 首次将机制可解释性（SAE）方法从LLM迁移到扩散模型进行偏见分析，开辟了新的研究方向
- k-SAE在扩散模型上的解缠效果出人意料地好——不同特征确实对应了人脸的不同细粒度属性
- 积分梯度+SAE的组合不仅可用于偏见缓解，也是理解扩散模型"如何决策"的通用工具

## 局限与展望
- k-SAE需要在目标模型的激活上训练，模型更换需要重新训练
- 当前主要在人脸和简单场景上验证，复杂场景的偏见（如职业刻板印象）未充分测试
- 偏见特征和非偏见特征之间可能存在更复杂的交互，简单缩放可能不够
- 可拓展到更多偏见维度（如年龄、体型）和更多生成模型架构（如DiT）

## 相关工作与启发
- **vs Fair Diffusion（引导生成）**: Fair Diffusion在采样时注入公平性引导，但不理解模型内部机制，可能影响质量；DiffLens精确定位偏见源头
- **vs Attention Editing**: 注意力编辑作用于宏观层面，难以只改目标属性；DiffLens在细粒度语义特征层面操作
- **vs Anthropic SAE on LLM**: 将LLM可解释性中的SAE技术迁移到视觉生成模型，是这一方向的重要拓展

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将机制可解释性用于扩散模型偏见分析
- 实验充分度: ⭐⭐⭐⭐ 无条件和条件模型，多种偏见属性，对比充分
- 写作质量: ⭐⭐⭐⭐ 框架清晰，图示优秀
- 价值: ⭐⭐⭐⭐⭐ 对AI安全和模型可解释性都有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](../../ICML2025/image_generation/towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)

</div>

<!-- RELATED:END -->
