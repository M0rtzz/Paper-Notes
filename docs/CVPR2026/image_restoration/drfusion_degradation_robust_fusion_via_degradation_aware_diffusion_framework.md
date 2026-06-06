---
title: >-
  [论文解读] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework
description: >-
  [CVPR 2026][图像恢复][多模态] 提出退化感知扩散框架 DRFusion，通过直接回归融合图像（而非显式预测噪声）和联合观测模型校正机制，在少量扩散步骤内实现任意退化场景下的多模态图像融合。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "多模态"
  - "扩散模型"
  - "degradation-aware"
  - "joint observation model"
---

# DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework

**会议**: CVPR 2026  
**arXiv**: [2604.08922](https://arxiv.org/abs/2604.08922)  
**代码**: [https://github.com/YShi-cool/DRFusion](https://github.com/YShi-cool/DRFusion)  
**领域**: 图像融合 / 图像恢复  
**关键词**: multimodal image fusion, diffusion model, degradation-aware, joint observation model, image restoration

## 一句话总结

提出退化感知扩散框架 DRFusion，通过直接回归融合图像（而非显式预测噪声）和联合观测模型校正机制，在少量扩散步骤内实现任意退化场景下的多模态图像融合。

## 研究背景与动机

真实世界的图像融合面临噪声、模糊、低分辨率等退化挑战。传统"先恢复再融合"管线存在误差累积和部署复杂性。端到端神经网络方法简单高效但可解释性差。扩散模型理论基础强但存在三个固有限制：(1) 需要目标分布的训练数据，而融合缺乏自然的融合图像；(2) 标准扩散模型处理单域分布，融合需要建模多源互补信息；(3) 迭代采样计算成本高。

现有扩散融合方法要么只处理特定退化，要么依赖独立预训练的恢复模型，缺乏灵活统一的框架。

## 方法详解

### 整体框架

丢弃标准扩散模型的显式噪声预测步骤，仅保留反向过程，通过有限扩散迭代直接从多源退化输入映射到融合输出。在每步扩散迭代中插入联合观测校正。

### 关键设计

1. **融合导向扩散框架**：不预测噪声而是直接回归融合图像，将去噪隐含在中间表示中。这使框架接近端到端网络的灵活性，可在自监督方式下处理融合（无需融合标签），同时仅需少量扩散步骤即可获得高质量结果。

2. **联合观测模型**：将两个源图像的退化约束和融合约束统一为矩阵形式。关键创新在于将融合图像的位置用零矩阵替代（不需要预先获得融合图像），并推导出联合退化矩阵的伪逆的解析解（通过分别求解各子方程避免直接计算高维伪逆）。

3. **联合观测校正机制**：在每步 DDIM 采样后，将退化约束和融合约束同时注入，迫使中间采样对齐退化模型同时保留跨模态互补信息。噪声情况下加入缩放因子 Σ_t 控制校正强度。

### 损失函数 / 训练策略

融合权重通过网络数据驱动学习（多任务架构同时预测噪声和权重图），约束 W1 + W2 = 1。支持多种退化场景（噪声、模糊、低分辨率及其组合）的统一处理。

## 实验关键数据

### 主实验

| 融合任务 | 退化类型 | 本文 | 对比方法 | 说明 |
|---------|---------|------|---------|------|
| 红外-可见光融合 | 噪声+模糊 | 最优 | DeFusion, DDFM 等 | 退化鲁棒性强 |
| 医学图像融合 | 低分辨率 | 最优 | 多种方法 | 恢复+融合一体化 |
| 多聚焦融合 | 散焦模糊 | 有竞争力 | 多种方法 | 灵活适配 |

### 关键发现

- 在复杂退化场景下显著优于现有方法
- 少量扩散步骤（如 5-10 步）即可达到竞争性结果
- 联合观测校正对保持恢复精度至关重要
- 数据驱动的融合权重学习优于固定权重

## 亮点与洞察

- 联合观测模型将退化恢复和多模态融合统一为一个约束优化问题
- 伪逆的解析求解方法优雅避免了高维矩阵运算
- 去除显式噪声预测使框架在少步采样时仍高效
- 统一处理噪声、模糊、低分辨率及其任意组合

## 局限与展望

- 退化模型需要已知或可估计（退化算子 A 需显式给出）
- 扩散步数减少可能在某些极端退化下影响质量
- 融合权重的学习依赖于训练数据的代表性

## 相关工作与启发

- 与 DDNM 的伪逆约束思路类似，但扩展到多输入融合场景
- 为其他多输入图像处理任务提供了退化感知扩散的通用范式

## 评分

- 新颖性：⭐⭐⭐⭐ — 联合观测模型的退化感知扩散融合
- 技术深度：⭐⭐⭐⭐⭐ — 数学推导严谨，伪逆求解优雅
- 实验充分度：⭐⭐⭐⭐ — 多任务多退化类型验证
- 实用价值：⭐⭐⭐⭐ — 统一框架处理任意退化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Degradation-Aware Metric Prompting for Hyperspectral Image Restoration](../../ICML2026/image_restoration/degradation-aware_metric_prompting_for_hyperspectral_image_restoration.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)
- [\[CVPR 2026\] NEC-Diff: Noise-Robust Event–RAW Complementary Diffusion for Seeing Motion in Extreme Darkness](nec-diff_noise-robust_event-raw_complementary_diffusion_for_seeing_motion_in_ext.md)
- [\[CVPR 2026\] EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)

</div>

<!-- RELATED:END -->
