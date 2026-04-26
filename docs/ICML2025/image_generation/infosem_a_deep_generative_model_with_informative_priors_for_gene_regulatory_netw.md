---
title: >-
  [论文解读] InfoSEM: A Deep Generative Model with Informative Priors for Gene Regulatory Network Inference
description: >-
  [ICML2025][图像生成][基因调控网络] 提出InfoSEM——无监督生成框架利用文本基因嵌入作为信息先验推断基因调控网络(GRN)：无需GT标签即超越监督方法38.5%，有标签作为额外先验时再提11.1%，同时发现现有监督方法学到的是基因特定偏差而非真正的调控机制。
tags:
  - ICML2025
  - 图像生成
  - 基因调控网络
  - 无监督生成模型
  - 文本嵌入先验
  - 变分推断
  - 生物标记物发现
---

# InfoSEM: A Deep Generative Model with Informative Priors for Gene Regulatory Network Inference

**会议**: ICML2025  
**arXiv**: [2503.04483](https://arxiv.org/abs/2503.04483)  
**领域**: image_generation  
**关键词**: 基因调控网络, 无监督生成模型, 文本嵌入先验, 变分推断, 生物标记物发现

## 一句话总结
提出InfoSEM——无监督生成框架利用文本基因嵌入作为信息先验推断基因调控网络(GRN)：无需GT标签即超越监督方法38.5%，有标签作为额外先验时再提11.1%，同时发现现有监督方法学到的是基因特定偏差而非真正的调控机制。

## 研究背景与动机

### GRN推断的重要性

基因调控网络揭示转录因子如何调控靶基因——对药物设计/生物标记物发现/疾病理解至关重要。

### 现有痛点

**现有痛点**：监督模型在标准基准上表现好(AUPRC~0.85)，但可能学到了GT标签的基因特定偏差（如某些基因的类别不平衡）而非真正的调控模式。

### 核心矛盾

**核心矛盾**：不用GT标签→避免偏差，但传统无监督方法性能远落后。InfoSEM用文本嵌入作为信息先验弥补这个差距。

## 方法详解

### 变分贝叶斯框架
用变分推断训练生成模型，从scRNA-seq数据推断调控关系。

### 文本基因嵌入先验
利用预训练文本嵌入编码基因的先验生物学知识——不是GT标签但包含丰富的功能/结构信息。

### GT标签作为额外先验（可选）
当有标签时，不用于直接监督而是作为先验——避免了监督学习中的偏差陷阱。

### 生物学驱动的基准框架
提出"未见基因间交互"的评估协议——更接近生物标记物发现等真实应用。

## 实验关键数据

### GRN推断（4个数据集平均）


### 主实验

| 方法 | 模式 | AUPRC相对改善 |
|------|------|-------------|
| 监督SOTA | 有标签 | 基线 |
| **InfoSEM(文本先验)** | **无标签** | **+38.5%** |
| **InfoSEM(+标签先验)** | **有标签** | **+49.6%(+11.1%)** |

### 未见基因评估


### 消融实验

| 方法 | 标准评估 | 未见基因评估 |
|------|---------|-----------|
| 监督方法 | 高 | **大幅下降** |
| **InfoSEM** | 更高 | **稳定** |

### 偏差分析

| 现象 | 解释 |
|------|------|
| 监督方法在标准评估高 | 学到了基因特定的类不平衡 |
| 监督方法在未见基因低 | 偏差不可迁移→泛化失败 |
| InfoSEM都高 | 学到了真正的调控模式 |

### 关键发现
1. 无监督+文本先验超越监督方法38.5%——范式级突破
2. GT标签作为先验而非监督→避免了偏差
3. 未见基因评估揭示了监督方法的虚假成功
4. 文本嵌入包含了丰富的基因功能信息
5. 在4个不同数据集上一致有效

## 亮点与洞察

1. "无监督超越监督"挑战了GRN领域的共识。
2. 揭示了监督方法学到偏差而非调控——对整个领域有警示。
3. 文本嵌入作为先验是低成本但高价值的创新。
4. "未见基因评估"更贴近真实应用场景。
5. GT标签用作先验而非监督的思路有广泛启发。

## 局限与展望

1. 文本嵌入的质量受限于预训练数据。
2. 仅在scRNA-seq数据上验证。
3. 对非模式生物的适用性未测试。
4. 变分推断的近似可能影响推断精度。
5. 与最新的多组学整合方法的对比缺失。

## 相关工作与启发

- 与GENIE3/DeepSEM等的区别：InfoSEM用先验而非监督。
- 与LLM-based生物学方法的关系：利用文本嵌入但方式不同。
- 启发："先验而非监督"的思路可推广到其他生物信息学任务。

## 评分
- 新颖性: 5.0/5 — 无监督超监督+偏差揭示
- 实验充分度: 4.5/5 — 4数据集+偏差分析
- 写作质量: 4.5/5
- 价值: 5.0/5 — 对GRN领域有范式级影响

## 补充

### “先验而非监督”的范式启示
GT标签作为监督会引入偏差，作为先验则提供引导但不强制。这种思路可推广到其他标注不完美的生物信息学任务。

### 为什么文本嵌入有效
预训练文本模型(如PubMedBERT)已编码了科学文献中的基因功能知识，作为先验显著优于无先验。

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Generation of Maximal Snake Polyominoes Using a Deep Neural Network](../../CVPR2025/image_generation/generation_of_maximal_snake_polyominoes_using_a_deep_neural_network.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[ICML 2025\] Learning Single Index Models with Diffusion Priors](learning_single_index_models_with_diffusion_priors.md)
- [\[CVPR 2025\] Learning Visual Generative Priors without Text](../../CVPR2025/image_generation/learning_visual_generative_priors_without_text.md)
- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](../../ICCV2025/image_generation/inference-time_diffusion_model_distillation.md)

<!-- RELATED:END -->
