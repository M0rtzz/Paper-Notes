---
title: >-
  [论文解读] VESSA: Video-based objEct-centric Self-Supervised Adaptation for Visual Foundation Models
description: >-
  [NeurIPS 2025][模型压缩][视觉基础模型] VESSA提出了一种利用短视频进行视觉基础模型无监督自适应的方法，通过自蒸馏框架结合LoRA参数高效微调和不确定性加权损失，在不需要任何标注数据的情况下显著提升基础模型在目标域的分类性能。 视觉基础模型（VFMs）如DINO、DINOv2等通过大规模自监督预训练获得了…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "视觉基础模型"
  - "自监督微调"
  - "视频适配"
  - "自蒸馏"
  - "LoRA"
---

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[CVPR 2025\] AutoSSVH: Exploring Automated Frame Sampling for Efficient Self-Supervised Video Hashing](../../CVPR2025/model_compression/autossvh_exploring_automated_frame_sampling_for_efficient_self-supervised_video_h.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)

</div>

<!-- RELATED:END -->
