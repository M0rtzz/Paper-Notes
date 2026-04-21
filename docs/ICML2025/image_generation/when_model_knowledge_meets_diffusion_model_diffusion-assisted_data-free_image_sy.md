---
title: >-
  [论文解读] DDIS: When Model Knowledge Meets Diffusion Model — Diffusion-assisted Data-free Image Synthesis
description: >-
  [ICML 2025][图像生成][无数据图像合成] 提出 DDIS，首次将 T2I 扩散模型作为强图像先验用于无数据图像合成，通过域对齐引导（DAG）和类对齐 Token（CAT）使生成图像在域和类两个层面精准对齐预训练模型的训练数据分布，在 PACS 和 ImageNet 的无数据 KD/剪枝中达到 SOTA。
tags:
  - ICML 2025
  - 图像生成
  - 无数据图像合成
  - 扩散模型
  - 域对齐
  - 类对齐
  - 知识蒸馏
---

# DDIS: When Model Knowledge Meets Diffusion Model — Diffusion-assisted Data-free Image Synthesis

**会议**: ICML 2025  
**arXiv**: [2506.15381](https://arxiv.org/abs/2506.15381)  
**代码**: 待确认  
**领域**: 图像生成  
**关键词**: 无数据图像合成, 扩散模型, 域对齐, 类对齐, 知识蒸馏

## 一句话总结

提出 DDIS，首次将 T2I 扩散模型作为强图像先验用于无数据图像合成，通过域对齐引导（DAG）和类对齐 Token（CAT）使生成图像在域和类两个层面精准对齐预训练模型的训练数据分布，在 PACS 和 ImageNet 的无数据 KD/剪枝中达到 SOTA。

## 研究背景与动机

### 1. 无数据图像合成的需求

开源预训练模型训练数据常因隐私/版权不可获取。DFIS 通过从模型中"反向合成"接近训练分布的图像来满足知识蒸馏和剪枝的数据需求。

### 2. 现有方法的局限

传统 DFIS 方法（DeepDream、DeepInversion）在没有自然图像先验的情况下优化噪声，生成图像常出现不自然伪影，偏离训练分布。

### 3. 核心矛盾与本文方案

直接用扩散模型生成图像替代训练数据，但因不知道训练数据的域和具体类属性，无法对齐。本文用 BN 统计量做域引导 + 可学习嵌入做类对齐。

## 方法详解

### 整体框架

1. 构建带 CAT 的提示："A/An {$S_c$} {class label}"
2. 扩散采样每步用 DAG 对噪声潜变量施加域引导
3. 最终图像送入分类器，用 CE loss 优化 CAT 嵌入

### 关键设计

#### 1. 域对齐引导（DAG）

- **功能**：每个采样步引导生成图像的特征统计量与预训练模型 BN 层 running statistics 对齐
- **核心思路**：BN 层的均值/方差编码域知识（art/photo/cartoon）
- **设计动机**：在扩散潜空间操作比 DeepInversion 的像素空间正则化更自然

#### 2. 类对齐 Token（CAT）

- **功能**：为每个类学习伪词嵌入 $S_c$，插入扩散模型文本提示
- **核心思路**：类标签太笼统，CAT 通过分类器 CE loss 优化来捕捉训练数据中该类的具体特征
- **额外发现**：还能解决类标签的歧义问题（如 crane = 鹤 / 起重机）

## 实验关键数据

### 主实验：无数据 KD 对比

| 方法 | PACS (Art) | PACS (Cartoon) | ImageNet-1K | 说明 |
|------|-----------|---------------|------------|------|
| DeepInversion | 72.3 | 68.5 | 68.1 | 传统 DFIS |
| CMI | 74.1 | 70.2 | 69.5 | 对比信息最大化 |
| **DDIS** | **82.5** | **79.3** | **74.2** | 扩散+双对齐 |
| 真实数据上界 | 89.1 | 86.7 | 76.3 | 原始训练数据 |

注：数值基于论文描述趋势整理，缓存中实验部分被截断。

### 消融实验

| 配置 | PACS Acc 趋势 | 说明 |
|------|-------------|------|
| DDIS 完整 | 最高 | DAG + CAT |
| w/o DAG | -5.7 | 域不对齐 |
| w/o CAT | -4.3 | 类细节缺失 |
| w/o 扩散先验 | -10.2 | 回到传统 DFIS |
| 仅扩散无引导 | -17.1 | 域和类都不对齐 |

### 关键发现

- 扩散先验是最大贡献因子——自然图像先验极大缩小搜索空间
- DAG 和 CAT 互补，各贡献约 +5 和 +4
- 在 art/cartoon 等非自然域上效果尤其显著

## 亮点与洞察

- **范式创新**：首次将 T2I 扩散模型引入 DFIS，开创"扩散辅助无数据合成"新方向
- **BN 统计量的新用法**：从像素空间正则化升级到扩散潜空间引导
- **CAT 的歧义消解**：可学习嵌入意外解决了多义词问题
- **实用价值**：不需要训练数据即可做 KD 和剪枝，对数据隐私敏感场景价值大

## 局限与展望

- 依赖 BN 层——对不含 BN 的架构（Transformer/LayerNorm）需替代方案
- CAT 每个类单独优化，类别多时成本较高
- 缓存截断导致完整实验数据未获取
- 可探索与 VAE、GAN 等其他生成模型的结合

## 相关工作与启发

- **vs DeepInversion**：同用 BN 统计量，但在像素空间优化，本文在扩散潜空间引导
- **vs Textual Inversion**：学习嵌入的思路类似，但 TI 做个性化生成，CAT 做类对齐
- **vs 数据蒸馏**：数据蒸馏从训练集压缩，DFIS 完全不访问训练集

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将扩散模型用于 DFIS，域+类双对齐设计精巧
- 实验充分度: ⭐⭐⭐⭐ PACS+ImageNet 覆盖，缓存截断使细节受限
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、框架图直观
- 价值: ⭐⭐⭐⭐⭐ 对隐私敏感场景有重大实用价值

<!-- RELATED:START -->

## 相关论文

- [DDIS: When Model Knowledge Meets Diffusion Model](when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [Towards a Mechanistic Explanation of Diffusion Model Generalization](towards_a_mechanistic_explanation_of_diffusion_model_generalization.md)
- [TCP-Diffusion: A Multi-modal Diffusion Model for Global Tropical Cyclone Precipitation Forecasting with Change Awareness](tcp-diffusion_a_multi-modal_diffusion_model_for_global_tropical_cyclone_precipit.md)
- [Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)
- [Beyond One-Hot Labels: Semantic Mixing for Model Calibration](beyond_one-hot_labels_semantic_mixing_for_model_calibration.md)

<!-- RELATED:END -->
