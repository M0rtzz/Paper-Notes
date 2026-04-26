---
title: >-
  [论文解读] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning
description: >-
  [CVPR 2025][模型压缩][持续学习] LoRA-DRS 提出"LoRA 减法"操作——在学习新任务前将旧任务的 LoRA 权重从预训练权重中减去以构建漂移抵抗空间（DRS），然后在该空间中通过梯度投影训练新任务的 LoRA，结合增强三元组损失提升可塑性，在无样本持续学习中实现了 SOTA 性能，尤其在长任务序列上优势显著。
tags:
  - CVPR 2025
  - 模型压缩
  - 持续学习
  - 特征漂移
  - LoRA
  - 无样本增量学习
  - 漂移抵抗空间
---

# LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning

**会议**: CVPR 2025  
**arXiv**: [2503.18985](https://arxiv.org/abs/2503.18985)  
**代码**: https://github.com/scarlet0703/LoRA-Sub-DRS  
**领域**: 模型压缩 / 持续学习  
**关键词**: 持续学习, 特征漂移, LoRA, 无样本增量学习, 漂移抵抗空间

## 一句话总结

LoRA-DRS 提出"LoRA 减法"操作——在学习新任务前将旧任务的 LoRA 权重从预训练权重中减去以构建漂移抵抗空间（DRS），然后在该空间中通过梯度投影训练新任务的 LoRA，结合增强三元组损失提升可塑性，在无样本持续学习中实现了 SOTA 性能，尤其在长任务序列上优势显著。

## 研究背景与动机

**领域现状**：持续学习需要模型在学习新类别时不遗忘旧知识。基于样本回放的方法通过存储旧数据来缓解遗忘，但在隐私和存储受限场景下不可行，因此无样本持续学习（EFCL）受到关注。近年来基于预训练模型（PTM）+ 参数高效微调（PEFT）的方法成为主流。

**现有痛点**：单纯使用 LoRA 或 Prompt Tuning 不足以解决灾难性遗忘——特征漂移随任务增多持续加剧。现有方法（如 InfLoRA、Adam-NSCL）尝试构建"不干扰旧任务"的特征子空间，但它们依赖存储的旧任务静态特征或统计量来计算子空间，这些信息随任务增多越来越过时，无法捕捉特征空间的动态演化。

**核心矛盾**：在 EFCL 设置下无法访问旧任务数据，却需要了解旧任务特征空间的当前状态以构建有效的保护子空间。旧任务统计量的"过时问题"导致这些方法在长任务序列中性能显著退化。

**本文目标** 如何在不保存旧任务数据/特征的情况下，构建一个能动态适应特征演化的训练空间，使新任务学习对旧任务的干扰最小化？

**切入角度**：借鉴 "task vector" 的研究发现——减去任务向量可以"遗忘"特定任务知识。如果在处理新任务数据前先让模型"遗忘"旧任务，得到的特征空间就自然反映了"不包含旧任务知识"的方向，在这个空间中学习新任务就不会干扰旧知识。

**核心 idea**：将旧任务的 LoRA 权重从预训练权重中减去来构建漂移抵抗空间，在该空间中约束新任务的梯度更新方向，从参数空间层面解决特征漂移问题。

## 方法详解

### 整体框架

LoRA-DRS 基于冻结的预训练 ViT-B/16 构建。每学习一个新任务 $t$，分两个阶段：**Stage 1**（DRS 构建）——通过 LoRA 减法计算修改权重 $\tilde{W_t} = W_0 - \sum_{j=1}^{t-1} B_j A_j$，将新任务数据通过修改后的模型得到各层输入特征的协方差矩阵，再通过 SVD 分解得到 DRS 投影矩阵。**Stage 2**（DRS 中训练）——展开新 LoRA 分支 $\Delta W_t = B_t A_t$，训练时将梯度投影到 DRS 中再更新参数，配合增强三元组损失提升可塑性。旧任务的 LoRA 分支和预训练权重始终冻结。

### 关键设计

1. **LoRA 减法（LoRA Subtraction, LoRA-）**:

    - 功能：在参数空间中"移除"旧任务影响，为构建 DRS 提供基础
    - 核心思路：借鉴任务向量（task vector）理论——任务向量 $V_{t-1} = W_{t-1} - W_0 = \sum_{j=1}^{t-1} B_j A_j$ 代表从预训练到学习完旧任务的参数变化。将其取反后加到预训练权重上得到 $\tilde{W_t} = W_0 - V_{t-1} = W_0 - \sum_{j=1}^{t-1} B_j A_j$。在 LoRA 设置下，这个操作极其简洁——不需要存储旧任务特征，只需已有的 LoRA 权重。用修改后的模型处理新任务数据得到各层输入特征 $\tilde{X_t^l}$，计算非中心化协方差矩阵 $\tilde{\mathcal{X}_t^l} = \frac{1}{n_t} (\tilde{X_t^l})^\top \tilde{X_t^l}$。
    - 设计动机：现有方法（InfLoRA、Adam-NSCL）使用旧任务的静态特征来构建子空间，但这些特征在持续学习过程中已经过时。LoRA 减法通过直接操作参数避免了对旧数据/特征的依赖，且天然地反映了当前模型状态。

2. **漂移抵抗空间（DRS）中的梯度投影**:

    - 功能：约束新任务的参数更新方向，最小化对旧任务的干扰
    - 核心思路：对 LoRA 减法得到的协方差矩阵 $\tilde{\mathcal{X}_t^l}$ 进行 SVD 分解，取前 $k$ 个最大特征值对应的特征向量构成投影矩阵 $P_t^l = (U_t^l)_k$，$k$ 的选择基于累积方差比阈值 $\varepsilon$。训练时，每步将梯度投影到 DRS 中：$\Delta w_{t,s}^l = P_t^l (P_t^l)^\top g_{t,s}^l$，确保参数更新只在 DRS 方向上发生。第一个任务不进行投影，直接使用原始梯度。
    - 设计动机：DRS 由"去除旧任务影响后"的新任务特征定义，在这个空间中的梯度更新自然地避开了旧任务的特征方向，实现了在不知道旧任务特征的情况下保护旧知识。

3. **增强三元组损失（Augmented Triplet Loss, ATL）**:

    - 功能：增强模型的可塑性——在 DRS 约束下仍能有效学习新类别的区分性特征
    - 核心思路：标准三元组损失 $\mathcal{L}_{TL} = \max(0, e_{ap} - e_{an} + \epsilon)$，其中正样本距离 $e_{ap}$ 取同类样本中最远的（最难正样本），负样本距离 $e_{an}$ 同时考虑当前任务中不同类的样本和旧任务的原型（prototype）。总损失 $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{TL}$。通过将旧任务原型也纳入负样本，增强了新旧类别之间的区分度。
    - 设计动机：梯度投影虽然保证了稳定性（不干扰旧任务），但会限制模型的可塑性（学习新任务的能力）。ATL 在 DRS 的约束下通过拉大类间距离来补偿可塑性的损失。

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{TL}$。使用 ViT-B/16-IN21K 预训练模型，Adam 优化器（$\beta_1=0.9, \beta_2=0.999$），batch size 128。ImageNet-R 上每个任务训练 50 epochs，CIFAR-100 上训练 20 epochs。LoRA 集成在 ViT 注意力模块的 Key 和 Value 上。推理时使用最近类均值分类器（NCM），用所有任务的 LoRA 权重合并后提取特征。

## 实验关键数据

### 主实验

ImageNet-R（200类，5次随机种子平均）：

| 方法 | 10 tasks ACC | 25 tasks ACC | 50 tasks ACC |
|------|-------------|-------------|-------------|
| LoRA-FT | 74.54 | 56.80 | 44.89 |
| CODA-Prompt | 72.15 | 63.86 | 48.89 |
| InfLoRA | 74.95 | 69.09 | 60.49 |
| EASE | 75.94 | 72.69 | 68.54 |
| Adam-NSCL | 72.24 | 62.04 | 49.82 |
| **LoRA-DRS** | 74.74 | **74.19** | **72.12** |

CIFAR-100：

| 方法 | 10 tasks ACC | 25 tasks ACC | 50 tasks ACC |
|------|-------------|-------------|-------------|
| EASE | 75.94 | 72.69 | 68.54 |
| InfLoRA | 86.44 | 77.51 | 56.65 |
| **LoRA-DRS** | 87.06 | **84.10** | **78.32** |

### 消融实验

| 配置 | 25 tasks ACC | 说明 |
|------|-------------|------|
| Full LoRA-DRS | 74.19 | 完整模型 |
| w/o DRS (直接梯度) | ~67 | 去掉梯度投影，特征漂移严重 |
| w/o ATL | ~72 | 去掉三元组损失，可塑性下降 |
| w/o LoRA- (用原始权重构建子空间) | ~71 | 不减去旧 LoRA，DRS 质量下降 |

### 关键发现

- **长任务序列优势显著**：50 tasks 时 LoRA-DRS 比 EASE 高 3.58%，比 InfLoRA 高 11.63%，但在 10 tasks 时略低于 EASE。说明 DRS 的核心价值在于解决"旧统计量过时"问题，任务越长优势越明显
- 特征漂移曲线（Fig. 1）显示 LoRA-DRS 是唯一在整个任务序列中保持低且稳定漂移的方法
- BWT 指标一致优于其他方法，证明向后迁移（旧任务性能保持）显著更好
- LoRA 减法的简洁性令人印象深刻——仅需已有 LoRA 权重的简单算术操作

## 亮点与洞察

- **"减法即遗忘"的核心洞察**：利用 task vector 的性质，通过参数减法让模型在处理新数据前"遗忘"旧任务，这样得到的特征自然地指示了"安全学习方向"。这个思路极其简洁但非常有效。在其他涉及知识管理的场景（如模型编辑、知识解学习）中也可能有用。
- **避免静态统计量的过时问题**：与 InfLoRA 和 Adam-NSCL 依赖存储的旧特征/梯度不同，LoRA-DRS 通过参数操作动态获取 DRS，不存储任何旧数据/统计量，从根本上避免了信息过时的问题。这是长任务序列中优势显著的关键原因。
- **稳定性-可塑性的解耦设计**：DRS 保证稳定性（不干扰旧任务），ATL 增强可塑性（在约束空间内最大化类间区分度），两者分工明确、互补有效。

## 局限与展望

- 10 tasks 短序列时略低于 EASE（74.74 vs 75.94），说明 DRS 构建在任务数少时可能过度保守
- LoRA rank 的选择对 DRS 质量有影响，但文中缺少对 rank 敏感性的详细分析
- 当前仅在分类任务上验证，在检测、分割等更复杂任务上的效果有待探索
- LoRA 减法的理论分析较薄弱——为什么减去旧 LoRA 后的特征空间就是"安全"的？缺乏更严格的理论证明

## 相关工作与启发

- **vs InfLoRA**: InfLoRA 利用旧任务的梯度信息设计 LoRA 降维矩阵的子空间，但梯度信息在长任务序列中逐渐过时。LoRA-DRS 通过参数减法避免了这个问题，50 tasks 时高出 11.63%
- **vs EASE**: EASE 采用语义引导的原型补充策略，在短序列中表现好但长序列中退化。LoRA-DRS 的漂移控制在长序列中更加稳定
- **vs Adam-NSCL**: Adam-NSCL 在旧任务输入特征的近似零空间中优化，但静态特征导致零空间越来越不准确。LoRA-DRS 完全避免了旧特征依赖
- **Task Vector 思想的延伸**: 本文将 task vector 的"减法=遗忘"性质巧妙应用于持续学习中构建 DRS，是 task arithmetic 在 CL 中的重要应用

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ LoRA 减法的idea简洁优美，"减法即遗忘→遗忘即保护"的逻辑链很有启发性
- 实验充分度: ⭐⭐⭐⭐ 多数据集多任务长度的全面对比，5次随机种子平均，但缺少一些分析性实验
- 写作质量: ⭐⭐⭐⭐ 方法流程清晰，Fig.1 的特征漂移可视化直观有力
- 价值: ⭐⭐⭐⭐⭐ 解决了 EFCL 中的核心痛点，长任务优势突出，方法实用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](../../NeurIPS2025/model_compression/rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)
- [\[ICML 2025\] Function-Space Learning Rates](../../ICML2025/model_compression/function-space_learning_rates.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)

<!-- RELATED:END -->
