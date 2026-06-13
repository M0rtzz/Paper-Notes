---
title: >-
  [论文解读] DDIS: When Model Knowledge Meets Diffusion Model
description: >-
  [ICML2025][图像生成][Data-Free Image Synthesis] 提出DDIS——首个利用T2I扩散模型作为图像先验的无数据图像合成方法，通过Domain Alignment Guidance (DAG)在扩散采样过程中对齐BN层域统计量、Class Alignment Token (CAT)编码类特定属性，在ImageNet-1k和多域PACS上全面超越现有DFIS方法。
tags:
  - "ICML2025"
  - "图像生成"
  - "Data-Free Image Synthesis"
  - "扩散模型"
  - "Domain Alignment"
  - "Class Alignment Token"
  - "知识蒸馏"
---

# DDIS: When Model Knowledge Meets Diffusion Model

**会议**: ICML2025  
**arXiv**: [2506.15381](https://arxiv.org/abs/2506.15381)  
**代码**: 无  
**领域**: 扩散模型 / 无数据学习  
**关键词**: Data-Free Image Synthesis, Diffusion Model, Domain Alignment, Class Alignment Token, Knowledge Distillation

## 一句话总结
提出DDIS——首个利用T2I扩散模型作为图像先验的无数据图像合成方法，通过Domain Alignment Guidance (DAG)在扩散采样过程中对齐BN层域统计量、Class Alignment Token (CAT)编码类特定属性，在ImageNet-1k和多域PACS上全面超越现有DFIS方法。

## 研究背景与动机

**领域现状**：开源预训练模型广泛可用（如Hugging Face平台），但其训练数据常因隐私/版权不可获取。Data-Free Image Synthesis (DFIS)通过从模型内部知识合成替代数据来解决这一问题，使下游任务（知识蒸馏、模型剪枝）无需原始数据。

**现有痛点**：DeepInversion等传统DFIS方法直接在高维像素空间中优化噪声输入，缺乏自然图像先验，导致搜索空间巨大。生成的图像充满不自然的人工伪影，严重偏离训练数据分布，限制了合成数据在下游任务中的实用性。

**核心矛盾**：DFIS需要在不知道训练数据任何信息的前提下生成与训练分布匹配的图像。但搜索空间太大（整个自然图像空间），没有引导就会走偏。

**本文要解决什么？**：(1) 如何利用扩散模型的强大自然图像先验缩小搜索空间？(2) 如何从预训练模型中提取域知识和类知识来引导扩散生成过程？(3) 如何处理类标签的词义歧义？

**切入角度**：BN层的running statistics编码了训练集的域分布信息；而一个可学习的pseudo-word embedding可以捕捉类标签名未显式表达的细粒度类属性。将这两个知识源注入T2I扩散模型的采样过程。

**核心idea一句话**：用预训练模型的BN统计量作为域引导、用可优化的token embedding作为类引导，驱动T2I扩散模型生成与训练分布高度对齐的合成数据。

## 方法详解

### 整体框架
DDIS基于Stable Diffusion 2.1，在扩散采样过程中引入两层引导机制。输入为类标签和预训练分类器，输出为与训练分布对齐的合成图像。流程为：(1) 构建包含CAT的文本提示；(2) 从高斯噪声开始逐步去噪，每步施加DAG域引导修正噪声潜变量；(3) 最终图像通过分类器前传计算CE loss优化CAT embedding。

### 关键设计

1. **Domain Alignment Guidance (DAG)**:

    - 功能：在扩散采样每一步引导噪声潜变量，使生成图像的内部特征统计量与预训练模型中BN层的running statistics对齐
    - 核心思路：BN层的running mean $\mu_l$ 和 running variance $\sigma_l^2$ 编码了整个训练集的域特征分布。DAG将有条件得分函数分解为无条件得分加上统计对齐梯度项。具体操作是在每个时间步$t$，先将潜变量$\mathbf{z}_t$解码为像素空间图像$\hat{\mathbf{x}}_t = \mathcal{D}(\mathbf{z}_t)$，通过分类器计算图像特征统计量与BN统计量的差异损失$\mathcal{L}_{BN} = \sum_l (\|\mu_l(\hat{\mathbf{x}}_t) - \mu_l\|^2 + \|\sigma_l^2(\hat{\mathbf{x}}_t) - \sigma_l^2\|^2)$，然后用该损失的梯度修正潜变量：$\tilde{\mathbf{z}}_t = \mathbf{z}_t - \eta \nabla_{\mathbf{z}_t} \mathcal{L}_{BN}$
    - 设计动机：直接用Classifier Guidance需要在每个时间步训练时间相关分类器（data-free条件下不可能），而BN统计量是现成的、跨时间步稳定的引导信号

2. **Class Alignment Token (CAT)**:

    - 功能：学习一个pseudo-word embedding来编码类标签名未显式表达的细粒度类属性
    - 核心思路：为每个类$c$定义一个新token $S_c$，构建提示"A/An $\{S_c\}$ $\{$class label$\}$"。$S_c$的embedding $v_c$通过最小化分类器的CE loss进行优化：$\mathcal{L}_{CE}(f(\hat{\mathbf{x}}_0; \theta^*), \mathbf{c})$，其中$\hat{\mathbf{x}}_0$是经DAG引导的最终生成图像。只优化最后一步的图像（因为$p(\mathbf{x}) \approx p(\hat{\mathbf{x}}_0)$），冻结SD所有参数，只更新单个token embedding
    - 设计动机：类标签"dog"不包含具体犬种信息，"crane"可能是鹤也可能是起重机。CAT通过与分类器交互学习到训练数据中该类的真实视觉属性，同时解决词义歧义

3. **DAG与CFG的融合采样**:

    - 功能：将域引导和文本条件生成统一到一个采样流程中
    - 核心思路：先用DAG修正潜变量$\tilde{\mathbf{z}}_t$，再基于修正后的潜变量执行Classifier-Free Guidance：$\tilde{\epsilon}_t = \epsilon_\theta(\tilde{\mathbf{z}}_t; \varnothing, t) + s(\epsilon_\theta(\tilde{\mathbf{z}}_t; \tau_\phi(\mathbf{y}), t) - \epsilon_\theta(\tilde{\mathbf{z}}_t; \varnothing, t))$。使用DDIM采样器30步完成生成
    - 设计动机：CFG只能控制文本条件一致性，不能提供域知识引导；将DAG作为CFG之前的预处理步骤实现了两者的正交互补

### 损失函数 / 训练策略
CAT embedding优化使用Adam优化器，学习率0.005，最多30个epoch，每个epoch 20次梯度累积。采用gradient skipping策略——只对最终去噪步反传梯度以节省显存。当batch中>70%样本被分类器正确预测时提前停止。整个流程中SD所有参数冻结，仅优化单个1×784维的token embedding。

## 实验关键数据

### 主实验：合成图像质量（IS↑/FID↓/Precision↑/Recall↑）

| 数据集 | 域 | DDIS IS/FID | DeepInversion IS/FID | PlugInInversion IS/FID |
|--------|------|------------|---------------------|----------------------|
| ImageNet-1k | Photo | **15.92/30.31** | 9.52/187.63 | 3.51/220.62 |
| PACS | Art Painting | **4.12/133.37** | 4.00/188.53 | 2.53/208.73 |
| PACS | Cartoon | **4.04/85.41** | 3.91/148.94 | 2.81/275.86 |
| Style-Aligned | Caricature | **3.94/139.75** | 3.58/195.25 | 2.51/293.58 |
| Style-Aligned | Manga | **3.87/145.82** | 3.32/206.57 | 2.36/295.14 |

### 消融实验（PACS Art Painting）

| 配置 | IS↑ | FID↓ | Precision↑ | Recall↑ |
|------|-----|------|-----------|---------|
| Vanilla SD | 2.88 | 193.57 | 0.6429 | 0.2572 |
| SD + CAT (w/o DAG) | 3.29 | 174.31 | 0.6995 | 0.3074 |
| SD + DAG (w/o CAT) | 3.95 | 166.22 | 0.6871 | 0.2843 |
| **DDIS (DAG + CAT)** | **4.12** | **133.37** | **0.7742** | **0.3213** |

### 关键发现
- DAG和CAT互补：DAG主要贡献域对齐（FID大幅降低），CAT主要贡献类精度（Precision提升明显），两者结合效果最佳
- CAT解决了词义歧义问题：在"tiger cat""beach wagon""mail bag"等歧义类上，vanilla SD生成错误类别，DDIS通过CAT学到了正确的视觉语义
- 在无数据知识蒸馏上，DDIS合成数据用于ResNet-34→ResNet-18蒸馏在ImageNet-1k上达到41.68% Top-1（vs DeepInversion 4.67%、PlugInInversion 2.01%）
- 生成效率：100k ImageNet图像仅需30K次迭代（vs DeepInversion的8M次），总训练耗时126小时 vs 18444小时

## 亮点与洞察
- 扩散模型作为DFIS的图像先验是范式级突破——从"在像素空间搜索"变为"引导生成模型"，将搜索空间从无穷缩小到扩散模型的自然图像流形上。这个思路可迁移到其他需要无数据/少数据生成的任务
- BN统计量作为域知识的桥梁非常巧妙——running mean/var是训练集的"指纹"，且跨扩散时间步稳定，是理想的引导信号
- CAT的词义消歧能力是意想不到的副产品——单个token embedding就能编码足够的类语义来区分同音词

## 局限性
- 依赖BN层：方法仅适用于含BN的CNN分类器（ResNet/VGG），不能直接应用于ViT等使用LayerNorm的模型，限制了通用性
- Sketch域失败：抽象的素描风格过于偏离SD的自然图像先验，DAG无法有效引导
- 每类需要单独优化CAT：1000类ImageNet需要1000次独立优化，虽然单次仅7.5分钟但总开销不小
- SD本身的偏差可能传递到合成数据中
- 对于训练数据分布与SD预训练数据高度不同的场景（如医学图像），效果可能受限
- DAG每步需要将潜变量解码回像素空间+前传分类器计算梯度，增加了采样时间

## 相关工作与启发
- **vs DeepInversion**: DeepInversion直接优化像素+BN正则，DDIS引导扩散模型生成——后者有更强的自然图像先验，FID改善4-6倍
- **vs Textual Inversion (Gal et al.)**: 两者都优化pseudo-word embedding，但Textual Inversion有参考图像，DDIS则完全无数据——通过分类器CE loss替代图像重建损失来驱动优化
- **vs NaturalInversion (Kim et al. 2022)**: NI只处理小规模数据集，DDIS首次实现了ImageNet-1k规模的高质量合成
- 启发：可将DAG推广为model-agnostic的统计对齐引导（不依赖BN），或结合LoRA代替CAT进行更强的模型适配

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型引入DFIS，开创了新范式
- 实验充分度: ⭐⭐⭐⭐ 多域(photo/art/cartoon/manga/caricature)、多任务(合成/蒸馏/剪枝)、详尽消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机-方法-实验逻辑通顺
- 价值: ⭐⭐⭐⭐⭐ 对无数据学习领域有重要推进，合成数据质量实现质的飞跃
- 总体: ⭐⭐⭐⭐⭐ DFIS领域的里程碑工作，开启了扩散模型辅助无数据学习的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)
- [\[CVPR 2025\] Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing](../../CVPR2025/image_generation/using_powerful_prior_knowledge_of_diffusion_model_in_deep_unfolding_networks_for.md)
- [\[ICML 2025\] Stealix: Model Stealing via Prompt Evolution](stealix_model_stealing_via_prompt_evolution.md)
- [\[CVPR 2025\] See Further When Clear: Curriculum Consistency Model](../../CVPR2025/image_generation/see_further_when_clear_curriculum_consistency_model.md)
- [\[ICML 2025\] Towards a Mechanistic Explanation of Diffusion Model Generalization](towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)

</div>

<!-- RELATED:END -->
