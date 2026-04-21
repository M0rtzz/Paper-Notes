---
title: >-
  [论文解读] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models
description: >-
  [NeurIPS 2025][模型压缩][视觉基础模型] VESSA提出了一种利用短视频进行视觉基础模型无监督自适应的方法，通过自蒸馏框架结合LoRA参数高效微调和不确定性加权损失，在不需要任何标注数据的情况下显著提升基础模型在目标域的分类性能。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 视觉基础模型
  - 自监督微调
  - 视频适配
  - 自蒸馏
  - LoRA
---

# VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.20994](https://arxiv.org/abs/2510.20994)  
**代码**: [GitHub](https://github.com/jesimonbarreto/VESSA)  
**领域**: 模型压缩  
**关键词**: 视觉基础模型, 自监督微调, 视频适配, 自蒸馏, LoRA

## 一句话总结

提出 VESSA，一种利用短物体中心视频进行无监督微调的方法，通过自蒸馏框架配合 LoRA 和不确定性加权损失，在不需要标注数据的情况下将视觉基础模型适配到目标域，在 33 个 VFM × 22 个数据集上持续提升下游分类性能。

## 研究背景与动机

视觉基础模型（VFM）如 DINO、DINOv2、TIPS 等通过大规模自监督预训练获得了强大的通用视觉表征能力。然而当应用到与预训练数据分布不同的专业领域（如遥感、医学影像）时，性能会显著下降。传统的解决方案是有监督微调，但这依赖于大量标注数据，在很多实际场景中代价高昂甚至不可行。

在 NLP 领域，无监督持续预训练（continued pretraining）已是大语言模型适配新领域的标准做法。但在视觉领域，这一策略的有效性尚未得到验证——直接对视觉编码器模型进行自监督持续训练往往会导致灾难性遗忘和性能退化。

本文的核心动机是回答三个问题：
1. 如何在**无标注**条件下将预训练视觉模型适配到特定域？
2. 什么形式的**无标注视觉数据**最适合这种适配？
3. 什么**学习策略**能在此约束下有效适配表征？

作者发现，朴素地将自蒸馏方法直接应用于微调阶段会导致模型快速遗忘预训练知识而退化，这需要精心的训练策略设计来规避。

## 方法详解

### 整体框架

VESSA 的训练流程如 Figure 2 所示，包含三个主要模块：

1. **帧选择（Frame Selection）**：从每个视频中采样 $n$ 对帧
2. **预处理与增强（Preprocessing & Augmentation）**：对帧对施加不同的数据增强
3. **模型微调（Model Fine-tuning）**：通过自蒸馏训练适配模型

输入为短物体中心视频（如 MVImageNet、CO3D），输出为适配后的视觉表征。

### 关键设计

1. **视频帧对采样策略**：对每个视频 $V$ 的 $T$ 帧 $\{F_i\}_{i=1}^T$，随机采样起始帧 $t \sim \mathcal{U}(1, T-\delta_{\max})$ 和时间偏移 $\delta \sim \mathcal{U}(1, \delta_{\max})$，构成帧对 $(F_t, F_{t+\delta})$。这种随机化策略引入了时间多样性，使模型从不同视角学习更鲁棒的表征。实验表明，随机偏移 $\delta \in [5,10]$ 效果最佳（DINO 85.03%，DINOv2 91.85%）。相比静态图像增强，真实视频帧对提供了远超几何/光度变换的外观变化。

2. **分阶段训练与 LoRA 适配**：这是使自监督微调成功的关键。训练分两步：

    - **Phase 1**：冻结 backbone，仅训练投影头数个 epoch，使其适配现有嵌入空间。消融实验显示，不经过此步直接训练会导致 10+ 个百分点的性能下降（从 91.87% 降至 80.87%）。
    - **Phase 2**：解冻 backbone。前 $H$ 层使用 LoRA 进行低秩适配（仅在 Q/K/V 投影中插入可训练矩阵 $\Delta W = AB$，$r \ll \min(d,k)$），保持 normalization 层可训练，保护低层的通用视觉特征（如边缘、纹理）；后 $L$ 层完全解冻，允许高层语义表征充分适配。最优配置为解冻最后 2 层。

3. **不确定性加权自蒸馏损失（UWSD）**：在标准 DINO 损失基础上引入基于教师网络输出熵的自适应权重：
    $w(q) = 1 + \gamma \cdot \mathcal{H}(q)$
   最终损失为：
    $\mathcal{L}_{\text{UWSD}} = \frac{1}{N} \sum_{(q,s,s_{lc_i}) \in \mathcal{B}} w(q) \cdot \mathcal{L}_{\text{DINO}}(q, s, s_{lc_i})$
   其中 $\gamma=1$。这使模型更关注教师预测不确定的困难样本，消融显示 UWSD 带来约 1% 的额外提升。

### 损失函数 / 训练策略

- 基础损失：DINO 交叉熵蒸馏损失 $\mathcal{L}_{\text{DINO}} = -\sum_i f_t(x_{t,i}) \log f_s(x_{s,i})$
- 加权损失：UWSD，通过教师输出熵加权
- 训练配置：ViT-Base，10 epoch 投影头 + 10 epoch 全模型，batch size 256，224×224 输入
- 教师网络参数通过 EMA 更新
- 局部裁剪成对采样：从帧对的两帧分别裁剪，保持时间一致性

## 实验关键数据

### 主实验

| 数据集 | 模型 | Pretrained | ExPLoRA+video | VESSA | 提升 |
|--------|------|-----------|---------------|-------|------|
| CO3D | DINO | 78.86 | 83.64 | **85.03** | +6.17 |
| CO3D | DINOv2 | 87.86 | 89.64 | **91.85** | +3.99 |
| CO3D | TIPS | 60.02 | — | **70.56** | +10.54 |
| MVImageNet | DINO | 90.44 | 87.74 | **92.51** | +2.07 |
| MVImageNet | DINOv2 | 95.75 | 96.15 | 96.01 | +0.26 |
| MVImageNet | TIPS | 78.65 | — | **80.54** | +1.89 |

### 消融实验

| 配置 | 准确率(%) | 说明 |
|------|----------|------|
| VESSA (完整) | 91.87 | Video + UWSD + 2层解冻 + 局部裁剪 + Head训练 |
| 去掉UWSD | 90.92 | 损失约1% |
| 去掉局部裁剪 | 90.53 | 损失约1.3% |
| 去掉Head训练 | 80.87 | 损失超10%，Head训练是关键 |
| 用图像代替视频 | 88.54 | 视频带来3.3%提升 |
| 1层解冻 | 87.14 | 解冻层数敏感 |
| DINO从头训练(图像) | 33.86 | 数据不足 |
| DINO从头训练(视频) | 39.39 | 视频一致性优于图像 |

### 关键发现

- **视频 vs. 图像**：在所有实验中，视频输入持续优于静态图像。尝试用运动模拟变换（平移、旋转、缩放等）来近似视频效果未能成功（DINOv2: 81.49% vs 91.85%），说明真实视频的增益来自超越简单几何/光度变换的视觉多样性。
- **灾难性遗忘**：适配后模型在 ImageNet 上性能急剧下降（DINOv2 从 82.10% 降至 17.15%），证实该方法是专域适配而非通用增强。
- **训练效率**：CO3D 上的完整适配仅需 1.97 小时（TPU v3-8），远低于从头预训练。

## 亮点与洞察

- 首次系统验证了视觉基础模型的**无监督微调**可行性，填补了视觉-NLP 方法论之间的空白
- 投影头预训练是防止灾难性遗忘的关键——这一发现对所有自蒸馏微调场景都有参考价值
- LoRA 在低层 + 全解冻在高层的混合策略很巧妙，平衡了通用特征保持和领域适配
- 使用物体中心短视频作为无标注数据源，采集成本低但信息密度高

## 局限与展望

- 严重的灾难性遗忘，适配后丧失通用能力，无法作为通用模型
- 依赖多视角物体中心视频，在缺乏此类数据的场景中适用性受限
- 仅在分类任务（KNN 评估）上验证，未涉及检测、分割等密集预测任务
- 帧距参数 $\delta_{\max}$ 需要针对具体数据集调优
- 可探索与 rehearsal 策略结合以缓解遗忘问题

## 相关工作与启发

- **ExPLoRA**：最相似的方法，也使用 LoRA 做持续自监督学习，但仍需后续有监督微调
- **DINO/DINOv2**：作为基础模型，VESSA 可直接应用于其适配
- **VITO/ViC-MAE**：视频到图像知识迁移方法，但更复杂且主要针对像素级任务
- 启发：对于任何自蒸馏微调，随机初始化头部与预训练参数之间的梯度冲突是核心挑战

## 评分

- 新颖性: ⭐⭐⭐⭐ 无监督 VFM 适配的方向有价值，但方法组件（自蒸馏+LoRA）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 33 个 VFM × 22 个数据集，消融全面，统计检验完善
- 写作质量: ⭐⭐⭐⭐ 条理清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 为专业领域无标注适配提供了实用方案
# VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.20994](https://arxiv.org/abs/2510.20994)  
**代码**: [GitHub](https://github.com/jesimonbarreto/VESSA)  
**领域**: 模型压缩  
**关键词**: 视觉基础模型, 自监督微调, 视频适配, 自蒸馏, LoRA

## 一句话总结

VESSA提出了一种利用短视频进行视觉基础模型无监督自适应的方法，通过自蒸馏框架结合LoRA参数高效微调和不确定性加权损失，在不需要任何标注数据的情况下显著提升基础模型在目标域的分类性能。

## 研究背景与动机

视觉基础模型（VFMs）如DINO、DINOv2等通过大规模自监督预训练获得了强大的通用视觉表征能力。然而，当将这些模型应用到与预训练数据存在分布偏移的特定领域时（如遥感、医学影像等），性能往往会下降。

现有的适配方法主要依赖有监督微调，需要大量标注数据，在很多实际场景中不可行。虽然NLP领域已经广泛使用无监督继续预训练来适配语言模型到新领域，但这种策略在视觉领域尚未被证明有效。作者发现，简单地将自蒸馏方法直接用于微调阶段会导致模型退化——模型会迅速遗忘预训练知识，进入degraded state。

这引出了三个核心问题：(1) 如何在无监督条件下适配预训练视觉模型？(2) 什么形式的无标注数据最适合适配？(3) 什么学习策略能有效适配预训练视觉表征？

## 方法详解

### 整体框架

VESSA的流程包含三个主要模块：帧选择（Frame Selection）、预处理与增强（Preprocessing & Augmentation）、模型微调（Model Fine-tuning）。输入为以目标物体为中心的短视频，输出为适应目标域的视觉模型表征。

### 关键设计

1. **帧选择模块**: 从每个视频中随机采样 $n$ 对帧。对于每一对，首先随机选取起始帧索引 $t \sim \mathcal{U}(1, T-\delta_{\max})$，然后以时间间隔 $\delta \sim \mathcal{U}(1, \delta_{\max})$ 采样第二帧。这种随机化策略引入时间多样性，使模型学习跨不同视角的鲁棒表征。实验表明 $\delta$ 在 $[5,10]$ 随机采样时效果最佳，说明适度的视角差异有助于表征学习。

2. **分阶段解冻策略**: 这是避免微调退化的核心设计。具体而言：

    - **第一阶段**：冻结整个backbone，仅训练projection head几个epoch，让head适应现有的嵌入空间
    - **第二阶段**：逐步解冻backbone，对前 $H$ 层使用LoRA进行低秩适配（仅更新Query/Key/Value投影的低秩矩阵 $\Delta W = AB$，其中 $r \ll \min(d,k)$），保留底层视觉特征；对最后 $L$ 层完全解冻进行常规更新，适配高层语义表征
   
   实验发现解冻最后2层效果最优（91.87%），解冻更多层反而性能下降。

3. **不确定性加权自蒸馏损失（UWSD）**: 在标准DINO损失基础上引入不确定性加权。计算教师网络输出分布的熵 $\mathcal{H}(q)$，用于调制每个样本对损失的贡献：

   $$w(q) = 1 + \gamma \cdot \mathcal{H}(q)$$
   
   $$\mathcal{L}_{\text{UWSD}} = \frac{1}{N} \sum_{(q,s,s_{lc_i}) \in \mathcal{B}} w(q) \cdot \mathcal{L}_{\text{DINO}}(q, s, s_{lc_i})$$
   
   高熵（不确定）的教师输出获得更大权重，优先学习困难样本。$\gamma=1$ 效果稳定。

### 损失函数 / 训练策略

基础损失为DINO的cross-entropy自蒸馏损失，教师网络通过学生网络权重的EMA更新。整体策略为先训练head 10 epoch，再训练完整模型 10 epoch。使用batch size 256，输入分辨率 $224 \times 224$，每个视频采样3对帧。局部裁剪也以成对方式从不同帧中采样，保持时间一致性。

## 实验关键数据

### 主实验

| 数据集 | 模型 | Pretrained | ExPLoRA+Video | VESSA | 提升 |
|--------|------|-----------|---------------|-------|------|
| CO3D | DINO | 78.86% | 83.64% | **85.03%** | +1.39 |
| CO3D | DINOv2 | 87.86% | 89.64% | **91.85%** | +2.21 |
| CO3D | TIPS | 60.02% | — | **70.56%** | +10.54 |
| MVImageNet | DINO | 90.44% | 87.74% | **92.51%** | +4.77 |
| MVImageNet | DINOv2 | 95.75% | 96.15% | 96.01% | ≈持平 |
| MVImageNet | TIPS | 78.65% | — | **80.54%** | +1.89 |

### 消融实验

| 配置 | 准确率 | 说明 |
|------|--------|------|
| 完整VESSA | **91.87%** | 全部组件开启 |
| 去掉UWSD损失 | 90.92% | UWSD贡献约1% |
| 去掉局部裁剪 | 90.53% | 局部裁剪贡献约1.3% |
| 去掉Head训练 | 80.87% | Head训练是最关键因素（+11%） |
| 使用Image替代Video | 88.54% | 视频输入比图像输入提升3.3% |
| 解冻1层 | 87.14% | 解冻层数敏感 |
| 解冻3层 | 90.80% | 2层最优 |
| DINO从头训练(Image) | 33.86% | 数据不足 |
| DINO从头训练(Video) | 39.39% | 视频比图像好5.53% |

### 关键发现

- **Head训练是最关键组件**：不训练head直接微调，性能从91.87%暴跌到80.87%，是随机初始化projection head导致梯度不稳定的直接证据
- **视频数据始终优于图像**：在所有配置中，使用视频输入都一致地优于对应的图像输入，说明多视角时序信息提供了超越简单数据增强的有效监督信号
- **尝试用图像变换模拟视频效果失败**：添加平移、旋转、缩放等变换后准确率仅从81.60%到81.49%（DINOv2），说明视频的优势来自真实视角变化而非简单几何变换

## 亮点与洞察

- 将NLP领域"无监督继续预训练"的思想成功迁移到视觉领域，填补了视觉基础模型无监督适配的空白
- 分阶段解冻策略简单但极其有效，避免了自监督微调中常见的表征退化问题
- 仅需简单的物体中心短视频（无需标注），降低了数据采集门槛

## 局限与展望

- 存在灾难性遗忘问题：适配后模型在ImageNet上的KNN准确率从82.1%暴跌到17.15%（DINOv2），无法作为通用模型使用
- 依赖物体中心短视频，这种结构化多视角数据在很多场景中并不容易获取
- 跨数据集泛化能力有限：在MVImageNet上训练后在CO3D上评估，性能下降5-7个百分点

## 相关工作与启发

- LoRA用于自监督继续学习的思路来自ExPLoRA，但VESSA不构建新的基础模型，而是直接适配到下游任务
- 自蒸馏框架基于DINO，但引入了关键的训练策略改进，使其在微调场景下可行
- 视频到图像知识迁移的轻量级方案，相比VITO和ViC-MAE等方法不需要复杂的帧选择管线或混合损失

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地探索视觉基础模型的无监督视频适配
- 实验充分度: ⭐⭐⭐⭐⭐ 33个基础模型×22个数据集，消融非常充分
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，层层递进
- 价值: ⭐⭐⭐⭐ 实用性强，为缺乏标注数据的场景提供了可行方案

<!-- RELATED:START -->

## 相关论文

- [Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](../../CVPR2025/model_compression/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)

<!-- RELATED:END -->
