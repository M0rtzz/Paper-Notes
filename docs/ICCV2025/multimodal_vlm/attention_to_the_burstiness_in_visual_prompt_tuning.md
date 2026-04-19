---
title: >-
  [论文解读] Attention to the Burstiness in Visual Prompt Tuning!
description: >-
  [ICCV 2025][多模态][提示学习] 本文揭示了视觉Prompt Tuning中自注意力模块数据的"爆发性"（burstiness）和非高斯分布问题，提出通过数据白化和双线性模型来学习"爆发性prompt"，在多个基准上大幅超越VPT及其变体，如CUB数据集上从42.15%提升至77.86%。
tags:
  - ICCV 2025
  - 多模态
  - 提示学习
  - Burstiness
  - Data Whitening
  - Bilinear Model
  - parameter-efficient fine-tuning
---

# Attention to the Burstiness in Visual Prompt Tuning!

**会议**: ICCV 2025  
**arXiv**: [2506.22908](https://arxiv.org/abs/2506.22908)  
**代码**: [GitHub](https://github.com/WangYZ1608/BPT)  
**领域**: 参数高效微调 / 视觉提示学习  
**关键词**: Visual Prompt Tuning, Burstiness, Data Whitening, Bilinear Model, parameter-efficient fine-tuning

## 一句话总结
本文揭示了视觉Prompt Tuning中自注意力模块数据的"爆发性"（burstiness）和非高斯分布问题，提出通过数据白化和双线性模型来学习"爆发性prompt"，在多个基准上大幅超越VPT及其变体，如CUB数据集上从42.15%提升至77.86%。

## 研究背景与动机
视觉Prompt Tuning (VPT) 是一种参数高效的微调技术，通过在输入空间学习少量参数（称为prompts）来适配预训练的Vision Transformer。然而VPT在很多数据集上表现不佳，例如在CUB-200数据集上仅42.15%的准确率。SPT等后续工作通过精心设计prompt初始化来改善性能，但这引发了一个根本问题：为什么学习prompt本身如此困难？

作者深入分析了自注意力模块中prompt与数据交互的分布特征，发现了两个关键问题：

**爆发性现象**：$\mathbf{W}_q\mathbf{W}_k^T\mathbf{X}^T$ 中少量元素具有极大的绝对值

**非高斯分布**：$\mathbf{W}_q\mathbf{W}_k^T$ 服从超拉普拉斯分布，patch embedding $\mathbf{X}$ 服从拉普拉斯分布

这些非高斯分布直觉上给prompt的学习带来了挑战。作者从数据预处理（白化）和双线性模型两个角度出发，提出学习"爆发性prompt"来适配这种数据特性。

## 方法详解

### 整体框架
VPT将prompt $\mathbf{P} \in \mathbb{R}^{m \times d}$ 与图像patch embedding $\mathbf{X} \in \mathbb{R}^{n \times d}$ 拼接后送入Transformer的自注意力模块。在注意力计算中，prompt需要与查询投影 $\mathbf{W}_q$、键投影 $\mathbf{W}_k$ 以及图像token进行多次矩阵乘法交互。BPT的核心思想是将prompt表示为两个矩阵的乘积（双线性形式），从而促进学习"爆发性"的最终prompt。

### 关键设计

1. **BPT-fWhiten（固定白化矩阵）**:

    - 功能：对 $\tilde{\mathbf{X}} = \mathbf{W}_q\mathbf{W}_k^T\mathbf{X}^T$ 进行ZCA白化，使其去相关化并方差统一
    - 核心思路：计算协方差矩阵 $\boldsymbol{\Sigma} = \frac{1}{N}\tilde{\mathbf{X}}\tilde{\mathbf{X}}^T$，通过SVD分解得到白化矩阵 $\mathbf{W} = \boldsymbol{\Sigma}^{-1/2} = \mathbf{U}\mathbf{S}^{-1/2}\mathbf{U}^T$，在学习prompt时固定白化矩阵：$\tilde{\mathbf{P}} = \mathbf{P}\mathbf{W}^T$
    - 设计动机：白化能将非高斯分布转化为更接近高斯的分布，降低prompt学习难度。仅需100张图像即可计算出有效的白化矩阵

2. **BPT-tWhiten（可调白化矩阵）**:

    - 功能：在学习prompt的同时微调白化矩阵
    - 核心思路：由于模型对白化矩阵 $\mathbf{W}$ 可微，同时优化prompt和白化矩阵：$\hat{\mathbf{P}}, \hat{\mathbf{W}} = \min_{\mathbf{P},\mathbf{W}} \ell(\mathbf{Y}, \text{MODEL}(\mathbf{X}; \mathbf{P}\mathbf{W}^T, \mathbf{H}, \boldsymbol{\Theta}))$
    - 设计动机：微调白化矩阵可进一步适配下游任务，但引入更多参数

3. **BPT-bilinear（低秩双线性prompt）**:

    - 功能：学习两个紧凑矩阵 $\mathbf{A} \in \mathbb{R}^{m \times p}$ 和 $\mathbf{B} \in \mathbb{R}^{d \times p}$，其乘积作为最终prompt
    - 核心思路：$\tilde{\mathbf{P}} = \mathbf{A}\mathbf{B}^T$，当 $p < d$ 时得到低秩prompt。例如 $m=100, d=768, p=25$ 时，计算量从VPT的 $15.1 \times 10^6$ 降至 $4.3 \times 10^6$（$3.5\times$ 减少）
    - 设计动机：双线性操作天然产生爆发性特征，且可通过控制 $p$ 灵活调节参数量和计算开销

### 实现细节
BPT的双矩阵乘法结构通过 $1 \times 1$ 卷积实现（无bias和非线性激活）。Deep版本将prompt插入到多个Transformer block中，默认在最顶部的4个block中使用prompt tuning，以平衡性能和参数量。

## 实验关键数据

### 主实验
| 数据集/方法 | Mean Acc | CUB-200 | NABirds | Flowers | Dogs | Cars |
|------------|----------|---------|---------|---------|------|------|
| VPT-S (MAE) | 57.84 | 42.15 | 57.43 | 69.15 | 77.07 | 43.38 |
| SPT-S (MAE) | 73.95 | 71.15 | 61.87 | 89.47 | 80.01 | 67.23 |
| **BPT-S (MAE)** | **80.39** | **77.86** | **72.03** | **90.37** | **81.91** | **79.77** |
| SPT-D (MAE) | 83.26 | 80.13 | 76.28 | 93.07 | 82.23 | 84.61 |
| **BPT-D (MAE)** | **84.60** | **82.00** | **78.49** | **93.72** | **82.67** | **86.11** |
| Full fine-tuning | 82.80 | 80.55 | 77.87 | 91.71 | 80.38 | 83.51 |

### 消融实验
| 配置 | #params (×10⁻²M) | IN-1K | CUB-200 | 说明 |
|------|------------------|-------|---------|------|
| VPT | 7.68 | 63.71 | 42.15 | 基线 |
| SPT | 7.68 | 69.98 | 71.15 | 当时SOTA |
| BPT-fWhiten (固定) | 7.68 | 72.09 | 77.48 | 白化显著提升 |
| BPT-tWhiten (可调) | 66.66 | 72.37 | 78.54 | 微调白化矩阵最优但参数多 |
| BPT-bilinear (随机初始化) | 6.51 | 72.15 | 77.86 | 最少参数，接近最优 |

### 关键发现
- BPT-Shallow在CUB-200上比VPT高出**35.71个百分点**（77.86 vs 42.15）
- BPT-bilinear参数量最少（6.51 vs 7.68），对随机初始化鲁棒
- prompt长度增大时BPT性能单调递增，不像VPT/GateVPT那样敏感
- 在COCO目标检测任务上，BPT同样超越VPT和SPT
- BPT训练100个epoch即能超越SPT训练400个epoch的性能

## 亮点与洞察
- 首次揭示了Transformer自注意力模块中的爆发性现象，并将其与prompt学习困难联系起来
- "学习爆发性prompt"的反直觉发现：面对爆发性数据，学习同样具有爆发性的prompt反而效果最好
- 方法极其简洁——仅一个无bias的1×1卷积，却带来巨大性能提升
- BPT-bilinear以最少参数实现最优性能，且对初始化不敏感，实用性强

## 局限与展望
- 分析仅聚焦于 $\mathbf{P}\mathbf{W}_q\mathbf{W}_k^T\mathbf{X}^T$，未覆盖注意力矩阵中的所有交互项
- 爆发性对prompt学习的促进作用缺乏理论解释，仅有经验观察
- 未探讨在数据不平衡场景下BPT的表现，也未分析预训练模型中固有的偏差影响
- 当训练数据充足时（>30% ImageNet），full fine-tuning仍优于prompt tuning方法

## 相关工作与启发
- **vs VPT**: BPT通过双线性结构大幅超越VPT，揭示了VPT性能不佳的根本原因在于数据分布，而非架构设计
- **vs SPT**: SPT依赖精心初始化（patch token聚类），BPT则对随机初始化鲁棒，且性能更优
- **vs LoRA/SSF等ft方法**: 在监督预训练设置下，BPT-Deep以18.36×10⁻²M参数达到91.72%均值准确率，超越所有比较方法

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 发现burstiness并将其与白化和双线性模型联系起来，视角独特且令人信服
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖FGVC、ImageNet、COCO检测分割，多种预训练方式和backbone规模
- 写作质量: ⭐⭐⭐⭐ 论文组织清晰，从现象到方法的逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 方法简洁高效，即插即用，对VPT领域有重要推动作用

<!-- RELATED:START -->

## 相关论文

- [CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [PRO-VPT: Distribution-Adaptive Visual Prompt Tuning via Prompt Relocation](pro-vpt_distribution-adaptive_visual_prompt_tuning_via_prompt_relocation.md)
- [FedMVP: Federated Multimodal Visual Prompt Tuning for Vision-Language Models](fedmvp_federated_multimodal_visual_prompt_tuning_for_vision-language_models.md)
- [Revisit Visual Prompt Tuning: The Expressiveness of Prompt Experts](../../ICLR2026/multimodal_vlm/revisit_visual_prompt_tuning_the_expressiveness_of_prompt_experts.md)
- [DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](../../CVPR2025/multimodal_vlm/dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)

<!-- RELATED:END -->
