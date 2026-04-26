---
title: >-
  [论文解读] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning
description: >-
  [ICCV 2025][视频理解][持续学习] 提出 RainbowPrompt，通过注意力变换和任务引导对齐的提示演化机制，将多个任务特定提示整合为多样性增强的统一提示，在图像分类和视频动作识别任务上平均超越现有方法 8.23%。
tags:
  - ICCV 2025
  - 视频理解
  - 持续学习
  - 提示学习
  - 知识整合
  - 类增量学习
  - 视频动作识别
---

# RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning

**会议**: ICCV 2025  
**arXiv**: [2507.22553](https://arxiv.org/abs/2507.22553)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 持续学习, 提示学习, 知识整合, 类增量学习, 视频动作识别

## 一句话总结

提出 RainbowPrompt，通过注意力变换和任务引导对齐的提示演化机制，将多个任务特定提示整合为多样性增强的统一提示，在图像分类和视频动作识别任务上平均超越现有方法 8.23%。

## 研究背景与动机

基于提示的持续学习（PCL）通过微调少量提示参数保持预训练模型冻结，是一种无需回放的有效方案。核心挑战是如何在提示中有效整合任务特定知识。

现有方法的不足：

**固定提示方法**（如 CODA-Prompt）：已学习提示的表示在新任务学习时保持不变，无法适应新任务需求，表示多样性低

**生成式提示方法**（如 ConvPrompt）：从纠缠的任务共享空间生成提示，受任务干扰和支配效应影响，多样性受限

**实验验证**（Fig.2）：核范数分析表明现有方法的提示表示多样性不足；更高的多样性与更高准确率、更低遗忘率正相关

核心动机：设计一种提示演化机制，通过变换和对齐所有累积的基础提示（包括已学习和新引入的），持续演化知识以增强表示多样性。

## 方法详解

### 整体框架

当新任务 $t$ 到来时，维护累积提示集 $\boldsymbol{\mathcal{P}} = \{\boldsymbol{p}^1, ..., \boldsymbol{p}^t\}$。通过注意力变换和任务引导对齐，将所有基础提示演化并整合为统一的 RainbowPrompt $\boldsymbol{p}_l^{\text{rainbow}(t)}$。还引入可学习概率门控自适应决定插入层。

### 关键设计

1. **注意力变换（Attention-based Transformation）**：

    - **任务条件化**：用可学习的任务嵌入 $\boldsymbol{e}^t$ 计算注意力权重，突出任务相关的提示成分
    - **任务级变换**：以新任务提示为 Query，累积提示集为 Key/Value，计算任务间亲和矩阵 $\mathcal{G}$，根据各任务对新任务的贡献重新加权
    - **特征级变换**：使用转置 Query/Key 捕获特征维度间的交叉影响（受双线性池化启发），$\hat{V} = \mathcal{F} \cdot \tilde{V}^T$，在更细粒度上整合跨特征贡献

2. **任务引导对齐（Task-Guided Alignment）**：通过非线性变换 $\text{LT}(x) = \max(0, xW_l^1)W_l^2$ 对变换后的表示进行精炼，使其适应新任务的特性同时保留各提示的固有属性。最终通过均值聚合得到 RainbowPrompt：$\boldsymbol{p}_l^{\text{rainbow}(t)} = \frac{1}{t}\sum_{i=1}^t \tilde{\boldsymbol{\mathcal{P}}}_l[i]$

3. **自适应提示插入（Adaptive Prompting）**：引入任务特定的可学习概率门控 $\boldsymbol{g}_l^t$（Bernoulli 变量），通过 Gumbel-Softmax 松弛实现可微分优化，自适应决定在哪些 ViT 层插入 RainbowPrompt，避免手动选择层的次优性。

### 损失函数 / 训练策略

总损失：$\min_{\Theta^t} \sum_i \text{CE}(\boldsymbol{z}_i, y_i) + \lambda_s \mathcal{L}_{\text{sparse}} + \lambda_m \mathcal{L}_{\text{match}}$

- $\text{CE}$：交叉熵分类损失
- $\mathcal{L}_{\text{sparse}} = \sum_l \log \alpha_l^t$：稀疏正则化，鼓励精简的提示插入模式
- $\mathcal{L}_{\text{match}} = \gamma(q(\boldsymbol{x}), \boldsymbol{e}^t)$：任务嵌入匹配损失
- $\lambda_s = \lambda_m = 0.01$
- 可学习参数 $\Theta^t = \{\boldsymbol{p}^t, \boldsymbol{e}^t, \boldsymbol{G}^t, W^{\text{evolution}}, \phi\}$
- 提示演化仅在训练时执行，测试时直接使用存储的 RainbowPrompt

## 实验关键数据

### 主实验

图像分类（ImageNet-R / CIFAR-100）：

| 方法 | ImgNet-R A₁₀↑ | ImgNet-R F₁₀↓ | ImgNet-R A₂₀↑ | CIFAR-100 A₁₀↑ | CIFAR-100 A₂₀↑ |
|------|---------------|---------------|---------------|----------------|----------------|
| L2P | 63.49 | 6.85 | 59.38 | 82.76 | 77.95 |
| DualPrompt | 68.50 | 5.14 | 63.21 | 85.07 | 80.49 |
| CODA-Prompt | 74.24 | 4.92 | 70.86 | 87.00 | 82.15 |
| ConvPrompt | 77.86 | 4.33 | 75.10 | 88.87 | 87.37 |
| **RainbowPrompt** | **79.09** | **3.90** | **78.36** | **89.86** | **90.15** |

视频动作识别（UCF-101 / ActivityNet）：

| 方法 | UCF A₁₀↑ | UCF A₂₀↑ | ActivityNet A₁₀↑ | ActivityNet A₂₀↑ |
|------|----------|----------|------------------|------------------|
| CODA-Prompt | 84.77 | 75.35 | 66.13 | 58.62 |
| CPrompt | 87.16 | 81.78 | 66.81 | 62.17 |
| ConvPrompt | 85.58 | 78.83 | 67.32 | 60.01 |
| **RainbowPrompt** | **89.03** | **84.05** | **69.87** | **70.55** |

### 消融实验

组件消融（ImageNet-R 10-task）：

| 配置 | A₁₀↑ | F₁₀↓ | 说明 |
|------|------|------|------|
| 完整模型 | 79.09 | 3.90 | - |
| 去掉任务条件化(TC) | 78.92 | 4.19 | 提供补充信息 |
| 去掉任务级变换(TLT) | 78.70 | 4.14 | 捕获任务依赖 |
| 去掉特征级变换(FLT) | 78.57 | 4.29 | 细粒度特征交互更关键 |
| 去掉任务引导对齐(TGA) | 66.31 | 4.84 | **最关键组件**，下降 12.78% |
| 去掉自适应提示(AP) | 78.13 | 4.07 | 自适应优于手动选层 |

### 关键发现

- 任务引导对齐是最关键组件，去掉后准确率骤降 12.78%
- 在 CIFAR-100 20-task 设置中，其他方法准确率下降 1.5-4.85%，而 RainbowPrompt 反而提升 0.29%
- ActivityNet 20-task 上 ConvPrompt 从 67.32% 降至 60.01%（-7.31%），RainbowPrompt 从 69.87% 升至 70.55%（+0.68%）
- 推理时丢弃 76.5% 演化参数（6.2M），仅需 18.5B MACs

## 亮点与洞察

- 提示"演化"的概念新颖：不是简单的提示选择或融合，而是通过变换和对齐让提示表示适应新任务
- 核范数作为多样性度量的理论分析有说服力，实验证实多样性与性能正相关
- 任务数量越多优势越明显，说明演化机制能有效利用累积知识
- 自适应门控避免了手动选择提示插入层的问题，不同任务/数据集的最优层分配不同

## 局限与展望

- 随任务增多，累积提示集的规模增长，演化过程的计算成本线性增加
- 任务身份在测试时未知（类增量设置），依赖任务嵌入匹配的准确性
- 目前基于 ViT-B/16，未验证更大预训练模型上的效果
- 视频动作识别仅采样 3 帧，对复杂时序动作的建模能力有限

## 相关工作与启发

- 提示演化的思路可推广到其他参数高效微调场景（如 LoRA 的持续学习）
- 双层注意力变换（任务级 + 特征级）的设计可迁移到其他需要多源知识整合的场景
- 自适应层选择机制对 Prompt Tuning 的层选择问题有通用参考价值

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](../../NeurIPS2025/video_understanding/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ICCV 2025\] Factorized Learning for Temporally Grounded Video-Language Models](factorized_learning_for_temporally_grounded_video-language_models.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)

<!-- RELATED:END -->
