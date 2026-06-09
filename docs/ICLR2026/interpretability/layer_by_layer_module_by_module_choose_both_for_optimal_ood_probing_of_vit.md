---
title: >-
  [论文解读] Layer by layer, module by module: Choose both for optimal OOD probing of ViT
description: >-
  [ICLR 2026 (CAO Workshop)][可解释性][Transformer] 通过大规模线性探测实验系统研究预训练ViT的中间层行为，发现分布偏移是深层性能退化的主因，并在模块级别揭示了最优探测点取决于偏移程度：显著偏移时探测FFN激活最优，弱偏移时探测MHSA归一化输出最优。
tags:
  - "ICLR 2026 (CAO Workshop)"
  - "可解释性"
  - "Transformer"
  - "线性探测"
  - "分布偏移"
  - "中间层表征"
  - "OOD"
---

# Layer by layer, module by module: Choose both for optimal OOD probing of ViT

**会议**: ICLR 2026 (CAO Workshop)  
**arXiv**: [2603.05280](https://arxiv.org/abs/2603.05280)  
**代码**: [GitHub](https://github.com/ambroiseodt/vit-probing)  
**领域**: 可解释性  
**关键词**: Vision Transformer, 线性探测, 分布偏移, 中间层表征, OOD

## 一句话总结
通过大规模线性探测实验系统研究预训练ViT的中间层行为，发现分布偏移是深层性能退化的主因，并在模块级别揭示了最优探测点取决于偏移程度：显著偏移时探测FFN激活最优，弱偏移时探测MHSA归一化输出最优。

## 研究背景与动机
近年来，一个引人注目的现象在基础模型中被反复观察到：**中间层的表征往往比最终层产生更具判别力的表示**。这一现象最初在自回归预训练的语言模型中被发现，后来也在通过监督学习和判别式自监督学习（如DINO、MAE）目标训练的视觉模型中被识别到。

然而，现有研究存在几个关键的认知缺口：

**原因不明**：中间层为何优于最终层？最初归因于自回归预训练的特性，但这无法解释为什么监督训练和对比学习的模型也出现同样现象

**分析粒度不足**：现有研究通常将Transformer块的输出作为一个整体进行探测，忽略了块内不同模块（MHSA、FFN、LayerNorm等）的特性差异

**实践指导缺失**：对于实际应用者来说，"应该从哪一层、哪个模块提取特征"这一关键问题缺乏系统性回答

核心猜想是：**分布偏移**（pretraining数据与下游数据之间的差异）才是深层表征退化的根本原因，与预训练方式无关。在ViT的深层，模型越来越专门化于预训练数据的分布特征，对下游数据的泛化能力下降。

更深入地，本文认为Transformer块内不同模块对分布偏移的敏感度不同，因此最优的特征提取点不仅取决于层的深度，还取决于块内的具体模块。

## 方法详解

### 整体框架
本文不提出新模型，而是把**线性探测**当作一把"卡尺"，去测量冻结的预训练 ViT 内部每一处表征对下游分类的判别力。做法是固定 ViT 参数，从某个探测点取出特征 $z$，在下游数据集上只训练一个线性分类器 $W$，用其准确率反映该位置表征的好坏。与以往只探测每个 Transformer 块最终输出（残差相加后的混合结果）不同，本文把探测变量拆成两个正交维度——层深度（layer index）和块内模块位置（module type），后者细到 MHSA 的输入/输出/归一化后输出、FFN 的输入/GELU 后中间激活/输出，以及残差连接前后的表征，从而能问出"到底从哪一层、哪个模块取特征"这个被长期忽略的问题。

### 关键设计

**1. 分布偏移假说验证：解释中间层为何反而更强**

中间层优于最终层的现象此前被归因于自回归预训练，但监督和自监督模型同样出现，说明归因不对。本文用一组"距离梯度"数据集来检验真正的成因：从几乎与 ImageNet 同分布的 ImageNet-V2、ImageNet-R，到完全异质的 EuroSAT 遥感、Flowers102 花卉，再配合监督、DINO、MAE 等多种预训练策略。结果呈现高度一致的规律——当下游分布贴近预训练分布时，最优探测点落在最终层附近；当分布差异变大时，最优点明显前移到中间层。由于这条规律跨预训练目标都成立，本文据此把退化主因从"预训练方式"重新定位到**分布偏移**：ViT 越往深层越专门化于预训练数据的统计特征，对 OOD 下游数据的泛化随之下降。

**2. 模块级细粒度分析：把"哪一层"细化到"哪个模块"**

这是全文最核心的贡献。标准探测取块的残差输出，但那是 MHSA 与 FFN 处理结果的叠加，掩盖了二者的差异。本文转而分别探测 MHSA 输出（归一化前后）、FFN 输入、FFN 的 GELU 后中间激活（非线性变换后、输出投影前）以及 FFN 输出等多个位置，得到两条与偏移强度挂钩的结论：强分布偏移下，FFN 的中间激活判别力最高，原因可能是 FFN 承担特征变换、其中间表征恰好处在"通用特征"与"任务特定特征"之间的甜蜜点；弱分布偏移下，则是 MHSA 的归一化输出最优，因为此时预训练学到的全局空间关系仍适用于下游，MHSA 的特征聚合能力更值钱。这意味着"层"和"模块"必须一起选——也正是标题 "choose both" 的由来。

**3. 跨配置系统实验：确认规律不依赖具体设置**

为排除规律只是某个特定配置的巧合，实验在四个维度上做了网格化覆盖：ViT 规模（ViT-S/B/L）、预训练方法（监督 ImageNet、DINO v1/v2、MAE 等）、patch size（16、14 等）以及 10 余个分布偏移程度各异的下游分类数据集。上述层级与模块级结论在所有组合下都稳定复现，从而支撑了"分布偏移是唯一关键因素、预训练目标无显著影响"的核心论断。

### 损失函数 / 训练策略
线性探测以标准交叉熵损失训练线性分类器，ViT 参数全程冻结，仅优化线性层。所有探测点采用统一的超参数搜索以保证公平比较；工程上基于 OmegaConf 配置系统与 accelerate 库，便于在大量"层×模块×数据集×预训练"组合上批量跑实验。

## 实验关键数据

### 主实验
在分布偏移程度不同的数据集上，探测不同层深度的准确率变化趋势：

| 数据集类型 | 分布偏移程度 | 最优层位置 | 最优模块 | 典型数据集 |
|-----------|-----------|----------|---------|----------|
| ImageNet变体 | 弱 | 最终层或接近最终层 | MHSA归一化输出 | ImageNet-V2, ImageNet-R |
| 通用自然图像 | 中等 | 中间层 | FFN激活 / MHSA输出 | CIFAR-10, STL-10 |
| 领域特定 | 强 | 中间偏浅层 | FFN中间激活 | EuroSAT, Flowers102 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 监督 vs DINO vs MAE预训练 | 规律一致 | 分布偏移效应独立于预训练方式 |
| ViT-S vs ViT-B vs ViT-L | 规律一致 | 模型规模不改变核心结论 |
| 块输出 vs FFN激活 vs MHSA输出 | FFN激活在OOD最优 | 证明标准探测方式次优 |
| 不同patch size | 规律一致 | Patch大小对结论影响不大 |

### 关键发现
- **分布偏移是唯一的关键因素**：跨所有预训练方法和模型规模，分布偏移程度与深层退化程度高度相关，而预训练目标类型无显著影响
- **标准的块输出探测始终次优**：无论分布偏移强弱，在块的内部模块中都能找到优于块输出的探测点
- **ID场景下最终层始终最优**：当预训练和下游数据分布一致时，不存在"中间层优于最终层"的现象，最终层的完整表示总是最好的
- **FFN激活是OOD的"万金油"**：在所有强偏移场景下，FFN的中间激活一致地输出最佳表征，这暗示FFN的非线性变换层是"通用特征→专用特征"转换的关键节点
- **MHSA在弱偏移时更优**：当分布差异不大时，MHSA的全局注意力模式（在预训练数据上学到的空间关系）仍然适用于下游数据

## 亮点与洞察
- **研究问题定义精准**：将一个被广泛观察但不充分理解的现象（中间层优于最终层）解构为层级别和模块级别两个正交维度进行分析
- **反直觉结论具有强实践指导价值**：从业者通常默认使用最终层或块输出特征，本文提供了根据分布偏移程度选择最优特征提取点的明确指南
- **FFN作为特征"瓶颈"的洞察**：FFN中间激活在OOD场景下最优，暗示了FFN在ViT中扮演了"特征提炼"的角色——其输入是通用的，输出是专用的，中间层处于最佳平衡点
- **实验设计的系统性**：覆盖了预训练方法、模型规模、下游数据集、模块类型等多个维度，结论的可信度高

## 局限与展望
- 作为Workshop论文，实验规模和讨论深度相对有限，一些发现需要更多理论分析支持
- 仅考虑了线性探测，非线性探测（如MLP head）和微调设置下的结论可能不同
- 未探索分布偏移程度的定量度量与最优探测点之间的定量关系——目前仅有定性的"强偏移→FFN激活，弱偏移→MHSA输出"规则
- 未考虑密集预测任务（如检测、分割），中不同模块的特征在空间信息保留方面可能有不同的表现
- 缺乏对"为什么FFN中间激活在OOD最优"的深入机理分析，如通过表征几何或特征可分性的视角
- 对于实际部署中如何在运行时判断"分布偏移强度"以决定探测策略，未给出自动化方案

## 相关工作与启发
- **Beyond the final layer (Attentive multilayer fusion for ViTs)**: 关注多层融合策略，与本文"选择单一最优层/模块"的路线互补
- **ViT-5**: 2026年的ViT改进工作，从架构角度提升ViT性能，本文则从特征利用角度提供洞察
- **Robust Representation Learning in Masked Autoencoders**: 关注MAE的鲁棒表征学习，本文的OOD分析对其有直接参考价值
- 本文启发的方向：能否设计一种自适应的特征融合策略，根据输入样本与预训练分布的距离自动选择最优的层+模块组合？

## 评分
- 新颖性: ⭐⭐⭐⭐ （模块级分析是新视角，但线性探测方法论本身不新）
- 实验充分度: ⭐⭐⭐⭐ （系统全面，但作为Workshop论文规模适中）
- 写作质量: ⭐⭐⭐⭐ （清晰简洁，核心信息突出）
- 价值: ⭐⭐⭐⭐ （为ViT特征利用提供了重要的实践指南）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](../../ICML2026/interpretability/is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)
- [\[ICML 2025\] On the Effect of Uncertainty on Layer-wise Inference Dynamics](../../ICML2025/interpretability/on_the_effect_of_uncertainty_on_layer-wise_inference_dynamics.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](../../CVPR2025/interpretability/lswag_zero_shot_nas.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](../../NeurIPS2025/interpretability/towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)

</div>

<!-- RELATED:END -->
