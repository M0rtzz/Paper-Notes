---
title: >-
  [论文解读] AIM: Amending Inherent Interpretability via Self-Supervised Masking
description: >-
  [ICCV 2025][自监督学习] 本文提出 AIM，一种基于自监督二值掩码的 top-down 特征选择机制，无需额外标注即可引导 CNN 聚焦真实判别特征、抑制虚假相关，同时获得内在可解释性和更强的 OOD 泛化能力。
tags:
  - ICCV 2025
  - 自监督学习
  - inherent interpretability
  - spurious features
  - feature pyramid
  - Energy Pointing Game
---

# AIM: Amending Inherent Interpretability via Self-Supervised Masking

**会议**: ICCV 2025  
**arXiv**: [2508.11502](https://arxiv.org/abs/2508.11502)  
**代码**: 暂无公开代码  
**领域**: 自监督学习 / 可解释性 / 鲁棒表征学习  
**关键词**: self-supervised masking, inherent interpretability, spurious features, feature pyramid, Energy Pointing Game

## 一句话总结

本文提出 AIM，一种基于自监督二值掩码的 top-down 特征选择机制，无需额外标注即可引导 CNN 聚焦真实判别特征、抑制虚假相关，同时获得内在可解释性和更强的 OOD 泛化能力。

## 研究背景与动机

**领域现状**：深度神经网络在分类任务上表现优异，但经常利用"虚假特征"（spurious features）做决策——例如在 WaterBirds 数据集上，模型学会根据背景（水面/陆地）而非鸟本身来分类，导致分布偏移时性能暴跌。

**现有痛点**：要让模型"因正确理由做出正确判断"，现有方法大多需要额外标注（bounding box、分割掩码、注意力引导图等），获取成本高且标注本身可能不完美。少数无额外标注的方法要么需要专家人工选择模型（迭代式），要么效果有限。

**核心矛盾**：DNN 在训练中同时学到了真实特征和虚假特征（Kirichenko et al. 2022 的重要发现），但缺乏一种轻量化的机制在不依赖外部监督的情况下，让模型自主区分并优先使用真实特征。

**本文目标**
   - 如何在仅有图像类别标签的情况下，让模型自动发现并聚焦于真正有判别力的空间特征？
   - 如何让模型的决策过程变得"内在可解释"（inherently interpretable），而非依赖事后归因方法？

**切入角度**：作者的核心假设——当强制模型在做分类决策前只保留一个子集的空间特征时，模型会选择最可靠（即最真实）的特征。这与 bottom-up 逐层掩码的思路不同：bottom-up 方法会导致掩码趋于全激活（fully active），需要额外的正则化损失来约束稀疏性。

**核心 idea**：用 top-down 的可学习二值掩码机制（借鉴 FPN 架构），在多尺度特征图上进行自监督的空间特征选择，自然产生稀疏掩码以过滤虚假特征。

## 方法详解

### 整体框架

AIM 在标准 CNN backbone（如 ConvNeXt-Tiny、ResNet-50）上增加了一个 top-down 通路，形成类似 FPN 的双通路架构：
- **Bottom-up 通路**：标准 backbone 的多阶段编码（如 ConvNeXt 的 4 个阶段 $S_0$~$S_3$），生成从粗到细的层级特征
- **Top-down 通路**：镜像结构 $T_0$~$T_3$，每个阶段有两个并行分支——特征处理分支和掩码估计分支
- 掩码估计器估出的二值掩码与处理后的特征逐元素相乘，形成稀疏特征表示
- 各阶段的稀疏特征从高层级向低层级逐步上采样融合，最终用于分类

输入是图像，输出是分类预测以及可视化的掩码（展示模型关注了哪些空间区域）。

### 关键设计

1. **掩码估计器（Mask Estimator）**:

    - 功能：为每个 top-down 阶段的特征图预测一个空间二值掩码 $B_\ell \in \{0,1\}^{w_\ell \times h_\ell}$
    - 核心思路：轻量 CNN（3×3 卷积 + 3 个残差块 + 全局池化分支 + 1×1 卷积）生成软注意力图 $A_\ell = M(S_\ell(x_\ell))$，再通过 Gumbel-Softmax 技巧二值化 $B_\ell = G(A_\ell)$，最终稀疏输出 $O_\ell = T_\ell(S_\ell(x_\ell)) \odot B_\ell$
    - 设计动机：相比 bottom-up 掩码策略（Verelst & Tuytelaars 2020），top-down 方式让网络在全局语义信息指导下重新评估各层特征，自然生成稀疏且聚焦的掩码，无需额外正则化就能避免掩码全激活问题

2. **Top-Down 稀疏特征融合**:

    - 功能：将各阶段产生的稀疏特征沿 top-down 方向逐步合并
    - 核心思路：每个阶段的输出 $O_\ell$ 通过最近邻插值上采样到下一阶段的分辨率，然后逐元素求和进行融合
    - 引入超参数控制 top-down 通路的终止深度——例如只融合最后 2-3 个阶段，用符号 AIM[stage, threshold] 表示配置

3. **自监督掩码监督策略**:

    - 功能：用分类损失间接监督掩码估计器的学习
    - 核心思路：在每个 top-down 阶段合并前，都用一个独立的分类器 $f_\ell$ 对稀疏特征 $O_\ell$ 计算分类损失 $\mathcal{L}_{cls}^{(\ell)}$，确保每个阶段独立学会识别重要区域
    - 这避免了对额外标注的依赖——掩码的学习完全由分类目标驱动

### 损失函数 / 训练策略

**掩码退火（Mask Annealing）**：在 OOD 数据集（如 WaterBirds、TravelingBirds）上，额外施加一个稀疏性约束。用阈值 $\tau_i$ 控制掩码活跃比例：

$$\mathcal{L}_{masks_i} = (r_i - \tau_i)^2, \quad r_i = \frac{\sum \mathbb{1}(B^i_{j,k}=1)}{\text{total pixels}}$$

训练初始 $\tau_i = 1.0$（全激活），逐 epoch 线性降低到目标值（如 0.35 或 0.25），稳定后保持。

总损失：$\mathcal{L}_{Total} = \lambda \sum_i \mathcal{L}_{masks_i} + \sum_i \mathcal{L}_{cls}^{(i)}$，其中 $\lambda = 6$。

## 实验关键数据

### 主实验

在 Waterbirds、TravelingBirds 等 OOD 数据集上，AIM 相比 baseline 有巨大提升：

| 模型 | WB-100% WG-Acc | WB-100% EPG | WB-95% WG-Acc | WB-95% EPG | TravelBirds Acc | TravelBirds EPG |
|------|--------|------|--------|------|--------|------|
| ConvNeXt-t | 39.6% | 57.2% | 81.6% | 68.3% | 59.5% | 74.4% |
| +AIM[2,25%] | 74.0% | 58.0% | 92.7% | 75.0% | 77.4% | 85.0% |
| +AIM[2,35%] | 78.1% | 68.5% | 92.3% | 71.7% | 71.0% | 77.7% |

WaterBirds-100% 最差组准确率提升约 **40 个百分点**，TravelingBirds 准确率提升约 **18 个百分点**。

在通用分类任务上同样有效：

| 模型 | ImageNet100 Acc | ImageNet100 EPG | HardImageNet Acc | CUB-200 EPG提升 |
|------|--------|------|--------|------|
| ConvNeXt-t | 89.2% | 91.4% | 96.2% | baseline |
| +AIM[2,25%] | 90.1% | 92.8% | 96.8% | +6% |

### 消融实验

| 配置 | CUB-200 Acc | 说明 |
|------|---------|------|
| Bottom-up masking [1,25%] | 72.79% (±8.51) | 底层 bottom-up 掩码方式，性能极差且不稳定 |
| ConvNeXt-t+AIM [1,25%] | 88.82% (±0.21) | Top-down 方式，大幅提升 |
| Bottom-up masking [2,25%] | 84.00% (±1.38) | 多阶段 bottom-up 略好但仍不如 top-down |
| ConvNeXt-t+AIM [2,25%] | 88.68% (±0.25) | Top-down 稳定且显著更优 |

计算开销：AIM 仅增加约 0.1-1.0 GFLOPs 和 1.9-3.7M 参数，相对于 backbone（4.5 GFLOPs / 28M 参数）开销很小。

### 关键发现

- **Top-down vs Bottom-up 是关键区别**：bottom-up 掩码方式生成的掩码倾向全激活，性能差且不稳定；top-down 自然产生稀疏掩码
- **掩码退火有助于 OOD 场景**：在标准数据集上掩码自然稀疏无需退火，但在存在强虚假相关的数据集上退火策略提供了额外的稀疏性约束
- **无中心偏差**：通过将鸟类图片裁剪到边缘位置测试，AIM 仍能正确定位目标（+2.5% 相比 baseline），证明不依赖中心偏差
- **跨架构一致性**：在 ConvNeXt-Tiny、ResNet-50、ResNet-101 上都有效

## 亮点与洞察

- **"所见即所得"的因果可解释性**：掩码直接参与前向传播决策过程，可视化掩码直接反映了分类依据，而非事后归因方法（如 GradCAM）的近似解释。这比 post-hoc 方法更可信。
- **自监督无需额外标注的特征筛选**：巧妙地利用"限制空间特征子集→迫使模型选择最可靠特征"的归纳偏置，将可解释性和泛化能力统一起来。
- **FPN 架构的新用途**：FPN 原本用于目标检测的多尺度特征融合，这里被改造为可解释性引导工具，展示了多尺度 top-down 信息流的新价值。

## 局限与展望

- **仅验证了 CNN**：目前只在 ConvNeXt 和 ResNet 上验证，未扩展到 Vision Transformer。作者在 Future Work 中提到计划扩展到 ViT/Swin-Transformer，但尚未实现。
- **掩码退火需要手动调超参**：阈值 $\tau$ 和退火时长需要针对不同数据集调节，增加了调参负担。
- **评估数据集偏窄**：主要在鸟类相关数据集（WaterBirds、TravelingBirds、CUB-200）上验证，缺少更多领域的 OOD 验证（如医学图像、遥感等）。
- **可改进思路**：可以探索将掩码机制与对比学习结合，在自监督预训练阶段就引导特征选择，而非仅在有监督分类任务中使用。

## 相关工作与启发

- **vs MaskTune (Asgari et al. 2022)**：MaskTune 在微调阶段通过掩码判别性特征来发现更多特征，但需要先训练一个完整模型再微调。AIM 在训练过程中同步学习掩码，更加端到端。
- **vs Content-adaptive downsampling (Hesse et al. 2023)**：同样使用 Gumbel-softmax 二值掩码，但采用 bottom-up 策略逐层掩码。AIM 的 top-down 策略天然更稀疏，无需额外稀疏性损失。
- **vs Post-hoc 归因方法（GradCAM 等）**：GradCAM 是事后解释，不改变模型行为；AIM 的掩码直接影响前向推理，提供因果层面的解释。
- 对于需要在部署前验证模型是否"因正确理由做出判断"的场景非常有价值，可作为模型审计工具的基础。

## 评分

- 新颖性: ⭐⭐⭐⭐ top-down 自监督掩码的思路简洁有效，但掩码+FPN 的组合并非全新概念
- 实验充分度: ⭐⭐⭐⭐ 多数据集多架构验证充分，消融实验详实，还有用户研究
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对可解释性和 OOD 泛化的实际价值高，但局限于 CNN 降低了通用性

<!-- RELATED:START -->

## 相关论文

- [Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/interpretability/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [Dataset Distillation for Pre-Trained Self-Supervised Vision Models](../../NeurIPS2025/interpretability/dataset_distillation_for_pre-trained_self-supervised_vision_models.md)
- [ArgoTweak: Towards Self-Updating HD Maps through Structured Priors](argotweak_towards_self-updating_hd_maps_through_structured_priors.md)
- [Latent Principle Discovery for Language Model Self-Improvement](../../NeurIPS2025/interpretability/latent_principle_discovery_for_language_model_self-improvement.md)
- [Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective](../../ICML2025/interpretability/evolving_prompts_in-context_an_open-ended_self-replicating_perspective.md)

<!-- RELATED:END -->
