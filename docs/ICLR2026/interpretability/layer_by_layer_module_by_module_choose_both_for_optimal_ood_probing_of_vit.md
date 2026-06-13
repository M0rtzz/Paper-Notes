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
本文不提新模型，而是把**线性探测（linear probing）**当成一把"卡尺"，去量一个冻结的预训练 ViT 内部每一处表征对下游分类到底有多大判别力。所有实验都基于同一个在 ImageNet-21k 上预训练的 86M 参数 ViT：保持权重全程冻结，从某个"探测点"取出 CLS token 嵌入做池化，只训练一个逻辑回归分类器（用 L-BFGS 求解），用它在下游数据集上的准确率反映该位置表征的好坏。关键创新在于把"探测点"拆成两个正交维度一起扫描——**层深度**（layer index）和**块内模块**（module type）：以往工作只探测每个 Transformer 块的最终输出（即第二个残差相加 RC2 后的混合结果），本文则把一个块内可追踪的 8 个操作（LN1、MHA、残差 RC1、LN2，以及 FFN 内部的 FC1、GELU 激活 Act、FC2、残差 RC2）逐一探测，从而回答"到底从哪一层、哪个模块取特征"这个被长期忽略的问题。

### 关键设计

**1. 双轴线性探测协议：把"取哪层特征"细化到"取哪层的哪个模块"**

判断中间层是否优于最终层，标准做法只探测块输出 RC2，但 RC2 是 MHA 与 FFN 两条支路经残差叠加后的混合结果，把模块间的差异抹平了。本文保持 ViT 冻结、对 CLS token 嵌入池化后用逻辑回归探测，并把探测点沿两个正交维度展开：纵向是层深度，横向是块内 8 个操作（LN1、MHA、RC1、LN2、FFN 的 FC1 / Act / FC2、RC2）。所有探测点用统一协议训练以保证可比，于是"该取哪层、哪个模块"第一次被放进同一张表里系统比较——这正是标题 "layer by layer, module by module, choose both" 的由来。

**2. 层级视图：分布偏移才是深层退化的真凶，而非预训练目标**

中间层优于最终层的现象此前被 Skean et al. 归因于自回归预训练，但他们的视觉实验只用了 ImageNet（恰好在预训练集内，属 ID 设定）。本文把探测搬到一批分布偏移程度递增的下游数据上：由于无法直接测量偏移强度，作者用"冻结 vs 微调"的性能差作为偏移强弱的代理，把数据集按微调准确率从高到低排序（Flowers102 最贴近预训练分布，往后是 Cifar10、Cifar10-C 的 Contrast，再到 Speckle Noise 偏移最强）。规律高度一致：偏移越强，越靠后的层判别力掉得越狠，最优探测点随之前移到中间层；而在 ID 设定（在该数据集上微调过）下，最终层始终最好，根本不存在"中间层更优"。由于这条规律与是否自回归无关，本文据此把退化主因从"预训练方式"重新定位到**分布偏移**——层越深越专门化于预训练数据的统计特征，对 OOD 下游的泛化随之下降。

**3. 模块级视图：FFN 内部激活与 LayerNorm 各擅胜场，块输出反而次优**

把分析下沉到模块后，结论更具操作性。对每个数据集、每个模块取其跨层的最佳准确率汇总（见原文 Table 1），可得三点：其一，标准的块输出 RC2 在除 Flowers102 外的所有数据集上都不是最优，说明默认探测方式系统性次优；其二，FFN 第二个全连接层 FC2 几乎最差（12 个数据集里 10 个垫底），而 FFN 内部的 **GELU 激活输出 Act** 综合最佳——它在偏移强的数据上大幅领先（此时它落在中间层、把"通用特征"刚转成"任务相关特征"的甜蜜点），但在 Cifar10、Flowers102、Pets 这类简单数据上略逊；其三，其余模块差别不大，其中 FFN 前的 LayerNorm **LN2**（即 MHA 残差后的归一化输出）略高且最稳定，弱偏移时反而是更稳妥的选择。这里"综合最佳"以**胜率（win-rate）**衡量，即一个模块在多大比例的数据集上取得跨模块最高准确率，Act 的胜率最高。两条合起来就是：层和模块必须一起选——强偏移取中间层的 FFN 激活 Act、弱偏移取 LN2。

### 损失函数 / 训练策略
没有需要训练的新模型。线性探测固定 ViT 全部参数，仅对 CLS token 池化特征拟合一个逻辑回归分类器（L-BFGS 求解）。所有"层×模块×数据集"组合采用统一的探测协议与超参，以保证横向比较公平。

## 实验关键数据

### 主实验
固定 86M ViT（ImageNet-21k 预训练），在 11 个按分布偏移递增排序的下游数据集上探测各层深度，准确率趋势如下（偏移强弱以"微调-冻结"性能差为代理）：

| 数据集 | 分布偏移程度 | 最优层位置 | 最优模块 |
|-----------|-----------|----------|---------|
| Flowers102 | 弱（贴近预训练分布） | 接近最终层 | 各模块接近，RC2 / LN2 均可 |
| Cifar10 | 弱-中 | 偏后层 | LN2 / Act |
| Cifar10-C（Contrast 等） | 中-强 | 中间层 | Act（FFN GELU 激活） |
| Cifar10-C（Speckle Noise 等） | 强 | 中间偏浅层 | Act |
| DomainNet（Clipart / Sketch） | 强 | 中间层 | Act |

### 模块级对比（原文 Table 1）
汇总每个数据集、每个模块跨层的最佳准确率：

| 对比 | 关键发现 | 说明 |
|------|---------|------|
| 块输出 RC2 vs 其余 7 个模块 | RC2 在 11 个数据集里 10 个非最优 | 默认探测方式系统性次优 |
| FFN 内部 FC1 / Act / FC2 | Act 胜率最高、FC2 最差（10/12 垫底） | FFN 激活是 OOD 的最佳探测点 |
| Act vs LN2（强 vs 弱偏移） | 强偏移 Act 大幅领先、弱偏移 LN2 更稳 | 层与模块需联合选择 |
| ID（微调）vs OOD（冻结） | ID 最终层最优、OOD 中间层更稳 | 退化主因是分布偏移 |

### 关键发现
- **分布偏移是关键因素**：跨全部 11 个偏移程度各异的数据集，分布偏移程度与深层退化高度相关，规律与是否自回归预训练无关，从而把成因从"预训练目标"重新定位到"分布偏移"
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
