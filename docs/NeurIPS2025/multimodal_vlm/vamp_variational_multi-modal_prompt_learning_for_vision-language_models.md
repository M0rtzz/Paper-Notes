---
title: >-
  [论文解读] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models
description: >-
  [NeurIPS 2025][多模态][提示学习] 提出变分多模态提示学习框架VaMP，将文本侧提示建模为隐变量并通过变分推断进行实例级不确定性建模，结合类感知先验正则化隐空间，在少样本和域泛化设置下显著提升CLIP的下游适配能力。
tags:
  - NeurIPS 2025
  - 多模态
  - 提示学习
  - 变分推断
  - CLIP
  - 少样本学习
  - 域泛化
---

# VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.22664](https://arxiv.org/abs/2511.22664)  
**代码**: [GitHub](https://visual-ai.github.io/vamp)  
**领域**: 多模态VLM  
**关键词**: Prompt Learning, 变分推断, CLIP, 少样本学习, 域泛化

## 一句话总结

提出变分多模态提示学习框架VaMP，将文本侧提示建模为隐变量并通过变分推断进行实例级不确定性建模，结合类感知先验正则化隐空间，在少样本和域泛化设置下显著提升CLIP的下游适配能力。

## 研究背景与动机

视觉-语言模型（如CLIP）在零样本设置下表现优异，但在少样本场景中适配到下游任务仍然面临挑战。Prompt Learning作为参数高效的适配策略被广泛采用，现有方法（如CoOp、MaPLe、MMRL）通常使用**固定的、所有样本共享的确定性提示**，存在三个核心痛点：

**缺乏实例级适应性**：固定提示无法根据不同输入进行个性化调整，面对分布偏移时泛化受限

**缺乏不确定性建模**：确定性参数无法捕捉少样本场景中的模型不确定性

**先验过于简单**：已有变分方法（如Bayesian Prompt Learning、Any-Shift Prompting）仅对文本分支建模、使用标准高斯先验且局限于全局层面，无法捕捉细粒度的token级语义变化

本文的切入角度是：将提示调优形式化为**变分推断**问题，在多模态（文本+视觉）框架下引入token级别的变分建模，并使用类别感知先验提供语义结构化的正则化信号。

## 方法详解

### 整体框架

VaMP由三个核心组件构成：(1) 基于图像特征的样本特异性提示生成；(2) 将文本侧提示作为隐变量的变分推断机制；(3) 基于类原型的类感知先验构建。整体在CLIP的文本和视觉编码器的多个Transformer层上进行提示注入。

### 关键设计

1. **样本特异性多模态提示生成（Sample-specific Prompt Generation）**

   不同于MaPLe/MMRL使用固定共享提示，VaMP根据输入图像动态生成文本侧提示。给定输入图像 $x$，首先通过冻结的CLIP图像编码器提取全局表征 $f_x$，然后使用 $H$ 个层特定的MLP生成器 $\{\Phi_i\}$ 为各Transformer层产生提示token：

   $$z_i = \Phi_i(f_x) \in \mathbb{R}^{M \times d}, \quad i = J, \dots, J+H-1$$

   视觉侧则使用共享的可学习提示token $\tilde{z}_i$，跨样本不变。这种非对称设计使文本侧能实现实例级适应，而视觉侧保持稳定的共享表征。

2. **变分多模态提示适配（Variational Multi-Modal Prompt Adaptation）**

   核心创新是将文本侧提示 $z_i$ 从确定性参数升级为**隐变量**。对每层引入一个MLP $\phi_i$ 预测后验分布参数：

   $$[\mu_i, \sigma_i] = \phi_i(\bar{f}_x), \quad q_\phi(z_i|x) = \mathcal{N}(\mu_i, \text{diag}(\sigma_i^2))$$

   训练目标最大化变分证据下界（ELBO）：

   $$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z|x)}[\log p(y|x,t,z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

   通过重参数化技巧 $z_i = \mu_i + \sigma_i \odot \epsilon_i$ 实现端到端训练。推理时通过蒙特卡罗采样 $S=10$ 次并平均预测结果，实现推理时集成。

3. **类感知先验构建（Class-Aware Prior）**

   标准高斯先验 $\mathcal{N}(0, I)$ 缺乏语义结构。VaMP构建类感知先验以提供全局语义锚定。训练时计算每类样本后验均值的类原型 $o_y$，再通过层特定的先验网络 $\psi_i$ 映射为先验分布参数：

   $$[\hat{\mu}_i, \hat{\sigma}_i] = \psi_i(c_y), \quad p_\psi(z_i|o_y) = \mathcal{N}(\hat{\mu}_i, \text{diag}(\hat{\sigma}_i^2))$$

   最终ELBO在各层上求和，使同类样本的提示分布在隐空间中聚集，提升类内一致性。测试时因标签不可用，退化为标准高斯先验。

### 损失函数 / 训练策略

使用变分ELBO作为训练目标，包含重建项（分类交叉熵）和KL散度正则项。所有实验采用16-shot设置，基于ViT-B/16 CLIP，单卡V100训练。类原型在训练前用冻结编码器离线计算。

## 实验关键数据

### 主实验：Base-to-Novel泛化（11个数据集平均）

| 方法 | Base | Novel | H (调和均值) |
|------|------|-------|------|
| CLIP | 69.34 | 74.22 | 71.70 |
| MaPLe | 82.28 | 75.14 | 78.55 |
| MMRL | 85.68 | 77.16 | 81.20 |
| **VaMP** | **86.45** | **78.67** | **82.37** |

VaMP在Novel类上超过MMRL **1.51%**，调和均值提升 **1.17%**。

### 域泛化（ImageNet → 4个变体）

| 方法 | ImageNet | -V2 | -Sketch | -A | -R |
|------|----------|-----|---------|-----|-----|
| MMRL | 72.03 | 64.47 | 49.17 | 51.20 | 77.53 |
| **VaMP** | **72.83** | **64.96** | **49.69** | **51.97** | **78.01** |

### 消融实验

| 配置 | Base | Novel | H | 说明 |
|------|------|-------|---|------|
| 任务特定提示（MMRL原版） | 85.68 | 77.16 | 81.20 | 固定共享提示 |
| + 样本特异性提示 | 85.93 | 78.13 | 81.84 | Novel +0.97% |
| + 变分建模 | 86.11 | 78.45 | 82.10 | Novel +0.32% |
| + 类感知先验 | 86.45 | 78.67 | 82.37 | Novel +0.22% |

### 关键发现

- 三个组件（样本特异性、变分建模、类感知先验）各自贡献独立且互补
- 在DTD等细粒度纹理数据集上VaMP的Novel准确率领先MMRL 2.67%
- 跨数据集泛化（10个目标数据集平均）VaMP达67.74%，超MMRL 0.49%
- 隐空间可视化显示更深层的后验分布更紧凑，体现层级化的不确定性细化

## 亮点与洞察

- **将提示学习与变分推断的结合做到了极致**：不是简单的全局变分，而是token级、多层的细粒度变分建模
- 类感知先验比标准高斯先验显著更优，为隐空间提供了结构化的语义指导
- 推理时蒙特卡罗采样实现了不确定性感知的集成，提升了鲁棒性
- 框架具有通用性，可适配到MaPLe和MMRL两种不同的多模态提示基线上

## 局限与展望

- 推理时需要10次采样，带来一定计算开销；可用均值近似替代以加速
- 类感知先验在测试时退化为标准高斯，未充分利用测试时的类别信息
- 仅在ViT-B/16上验证，对更大backbone（如ViT-L/14）的效果未知
- 变分建模仅限于文本侧，视觉侧仍使用确定性提示

## 相关工作与启发

- Bayesian Prompt Learning和Any-Shift Prompting是该方向的先驱，但局限于文本单模态和全局层面变分
- MMRL的多模态共享表征思路为VaMP的多层提示注入提供了基础
- 变分推断在few-shot learning中的经典应用（如VAE-based meta-learning）为本文提供了理论基础
- 启发：可将变分思想推广到视觉prompt、adapter等其他PETL方法中

## 评分

- **新颖性**: ⭐⭐⭐⭐ token级多层变分提示建模+类感知先验是有意义的系统性创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 三种泛化设置、11个数据集、完整消融、可视化分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数学推导严谨
- **价值**: ⭐⭐⭐⭐ 为prompt学习引入了结构化的概率建模范式

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] NLPrompt: Noise-Label Prompt Learning for Vision-Language Models](../../CVPR2025/multimodal_vlm/nlprompt_noise-label_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2025\] Vision-Language Model IP Protection via Prompt-based Learning](../../CVPR2025/multimodal_vlm/vision-language_model_ip_protection_via_prompt-based_learning.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)

<!-- RELATED:END -->
