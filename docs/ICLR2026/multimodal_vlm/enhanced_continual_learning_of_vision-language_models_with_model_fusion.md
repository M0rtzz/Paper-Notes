---
title: >-
  [论文解读] Enhanced Continual Learning of Vision-Language Models with Model Fusion
description: >-
  [ICLR 2026][多模态VLM][持续学习] 提出Continual Decoupling-Unifying（ConDU）框架，首次将模型融合引入VLM持续学习，通过维护统一模型并结合任务触发器进行解耦-统一迭代操作，在MTIL基准上平均性能超SOTA 2%，同时增强了零样本能力。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "持续学习"
  - "模型融合"
  - "灾难性遗忘"
  - "CLIP"
  - "零样本能力保持"
---

# Enhanced Continual Learning of Vision-Language Models with Model Fusion

**会议**: ICLR 2026  
**arXiv**: [2503.10705](https://arxiv.org/abs/2503.10705)  
**代码**: [GitHub](https://github.com/zhangzicong518/ConDU)  
**领域**: 多模态VLM  
**关键词**: 持续学习, 模型融合, 灾难性遗忘, CLIP, 零样本能力保持

## 一句话总结
提出Continual Decoupling-Unifying（ConDU）框架，首次将模型融合引入VLM持续学习，通过维护统一模型并结合任务触发器进行解耦-统一迭代操作，在MTIL基准上平均性能超SOTA 2%，同时增强了零样本能力。

## 研究背景与动机
VLM（如CLIP）通过整合视觉和文本模态实现了出色的零样本能力。然而在多个下游任务上顺序微调时，VLM同样面临灾难性遗忘问题。现有VLM持续学习方法存在明显局限：

**蒸馏方法**（如ZSCL、Dual-RAIL）需要额外参考数据集进行知识蒸馏，性能对数据集选择敏感，且需要精心调节多个超参数来平衡遗忘缓解、零样本保持和当前任务优化

**参数高效微调方法**（如DPeCLIP、MulKI）仅适用于adapter或LoRA场景，无法处理全参数微调

**核心insight**: 如果允许为每个任务维护独立微调模型，已知任务ID时直接选用对应模型即可。关键思路是：将这些独立模型的共享部分提取并融合为一个统一VLM，任务特异性差异存储在有限内存中，从而用"一个主VLM+少量辅助内存"模拟多个专用模型的行为。模型融合（model fusion）天然适合这一场景——无需访问原始训练数据即可合并多个模型。

## 方法详解

### 整体框架
ConDU维护三个组件贯穿整个持续学习过程：统一模型、任务触发器集合、原型集合。每个新任务到来时执行三步：(1) 独立微调获得任务专家；(2) 通过任务触发器解耦统一模型获得历史任务专家；(3) 将所有任务专家统一为新的统一模型。解耦和统一操作无需训练，耗时仅为微调的约1%。

### 关键设计
1. **Delta模型统一（Unifying）**: 定义delta模型为任务专家与预训练模型的参数差 $\delta^t = \theta^t - \theta^0$。统一操作对每个参数维度选择所有delta模型中绝对值最大且与总和方向一致的值：若 $\sum_i \delta^i_j > 0$ 则取 $\max_i(\delta^i_j)$，否则取 $\min_i(\delta^i_j)$。这保留了跨模型共享的最大幅度和一致方向信息。同时为每个任务计算二值掩码 $M^i_j$（标记该任务delta模型与统一delta模型在各位置是否同号）和缩放标量 $\lambda^i$（保持平均幅度一致）作为任务触发器。

2. **Delta模型解耦（Decoupling）**: 用任务触发器从统一delta模型重建各任务delta模型：$\tilde{\delta}^i = \lambda^i \cdot M^i \odot \delta^{1:t}$，再加上预训练模型得到任务专家 $\tilde{\theta}^i = \theta^0 + \tilde{\delta}^i$。这一过程在训练阶段（用于获取历史任务专家参与统一）和推理阶段都使用。

3. **语义聚合推理机制**: 对于未知任务ID或零样本场景，解耦所有任务专家。用预训练VLM提取测试样本图像特征，计算与各任务各类别原型的余弦相似度。每个任务取最高相似度作为该任务专家的权重，选择K个最高权重的专家，聚合其输出logits作为最终预测。原型定义为类别图像特征均值加文本特征：$P^i_k = f(y, \theta^0) + \frac{1}{|\mathcal{D}^t_k|}\sum_m f(x_m, \theta^0)$。

### 损失函数 / 训练策略
- 训练阶段仅标准微调（全参数或LoRA），无需额外蒸馏损失或参考数据集
- 解耦-统一操作完全无需训练
- 唯一超参数K（推理时选择的专家数量），消融显示性能对K非常不敏感

## 实验关键数据

### 主实验

**MTIL基准（11个跨域任务）:**

| 方法 | Transfer↑ | Average↑ | Last↑ |
|------|-----------|----------|-------|
| Zero-shot | 65.3 | 65.3 | 65.3 |
| ZSCL | 68.1 | 75.4 | 83.6 |
| Dual-RAIL | 69.4 | 77.8 | 86.8 |
| DPeCLIP | 69.1 | 77.5 | 86.9 |
| MulKI | 70.1 | 77.3 | - |
| **ConDU (LoRA)** | **70.3** | **78.3** | **86.2** |
| **ConDU (FT)** | **70.8** | **78.8** | **87.1** |

**Task-Agnostic MTIL（无任务ID）:**

| 方法 | Average↑ | Last↑ |
|------|----------|-------|
| 最佳基线 | 76.1 | 84.6 |
| **ConDU (LoRA)** | **78.0** | **85.1** |
| **ConDU (FT)** | **78.1** | **86.4** |

### 消融实验
- ConDU在全参数微调和LoRA两种场景下均有效，是唯一同时支持两种范式的方法
- Few-shot MTIL（每类5样本）：Transfer 70.0%/70.3%(FT/LoRA)超最佳基线1.4%，Average 72.3%/72.7%超1.3%，Last 76.6%/77.4%超1.3%
- 推理时聚合权重的专家数K对性能非常不敏感（详见附录F）
- 解耦-统一操作的时间开销仅为微调时间的约1%
- 多个专家并行前向传播的推理时间接近单模型推理
- 统一操作中"选最大绝对值+一致方向"策略优于简单平均等基线融合策略

**Few-shot MTIL对比:**

| 方法 | Transfer↑ | Average↑ | Last↑ |
|------|-----------|----------|-------|
| 最佳基线 | 68.6 | 71.4 | 76.1 |
| ConDU (FT) | 70.0 | 72.3 | 76.6 |
| ConDU (LoRA) | 70.3 | 72.7 | 77.4 |

### 关键发现
- Transfer指标超预训练VLM 5.5%，说明持续学习过程反而增强了零样本能力
- 全参数微调版本（ConDU FT）通常优于LoRA版本，说明全参数微调在持续学习中仍有优势
- 模型融合中的"选最大绝对值+一致方向"策略在保留多任务知识方面效果显著

## 亮点与洞察
- 首次将模型融合引入VLM持续学习，开辟了一个新的研究方向
- 框架设计优雅：解耦-统一操作完全无需训练，任务触发器（掩码+缩放标量）存储开销极低
- 同时兼容全参数微调和参数高效微调，灵活性远超现有方法
- 零样本能力不仅不退化反而增强，这在持续学习中非常难得

## 局限与展望
- 任务触发器中的二值掩码与统一delta模型同维度，任务数增加时存储可能成为瓶颈
- 统一操作中"选最大绝对值"的策略是否最优有待更多理论分析
- 实验仅在CLIP架构上验证，更多VLM架构（如BLIP、LLaVA）的适用性待探索
- 语义聚合推理需要前向传播多个任务专家，任务数极多时推理成本增加

## 相关工作与启发
- **与Task Arithmetic的关系**: ConDU的统一操作受TIES Merging启发，但针对持续学习场景设计了解耦机制
- **与ZSCL/Dual-RAIL对比**: 这些方法需要参考数据集和蒸馏，ConDU完全不需要
- **启发**: 模型融合视角为持续学习提供了新的思路——不再是"如何防止遗忘"，而是"如何高效存储和重建多个专家"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将模型融合引入VLM持续学习，框架设计新颖
- 实验充分度: ⭐⭐⭐⭐ MTIL基准覆盖三种设定，但仅在CLIP上验证
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，方法描述规范，数学符号统一
- 价值: ⭐⭐⭐⭐⭐ 开辟新方向，框架通用性强，无需额外数据和蒸馏设计，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](keeplora_continual_learning_with_residual_gradient_adaptation.md)
- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](../../AAAI2026/multimodal_vlm/branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)

</div>

<!-- RELATED:END -->
