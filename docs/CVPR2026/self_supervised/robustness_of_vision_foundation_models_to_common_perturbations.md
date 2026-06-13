---
title: >-
  [论文解读] Robustness of Vision Foundation Models to Common Perturbations
description: >-
  [CVPR 2026][自监督学习][foundation model] 首次系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）的鲁棒性，提出三种鲁棒性度量并形式化五个数学性质，发现基础模型普遍不鲁棒，并提出微调方法改善鲁棒性而不牺牲效用。 视觉基础模型输出图像的嵌入向量用于下游任务，但常见编辑操作（JPEG 压缩…
tags:
  - "CVPR 2026"
  - "自监督学习"
  - "foundation model"
  - "robustness"
  - "common perturbation"
  - "embedding"
  - "CLIP"
  - "DINOv2"
---

# Robustness of Vision Foundation Models to Common Perturbations

**会议**: CVPR 2026  
**arXiv**: [2604.14973](https://arxiv.org/abs/2604.14973)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: foundation model, robustness, common perturbation, embedding, CLIP, DINOv2

## 一句话总结

首次系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）的鲁棒性，提出三种鲁棒性度量并形式化五个数学性质，发现基础模型普遍不鲁棒，并提出微调方法改善鲁棒性而不牺牲效用。

## 研究背景与动机

视觉基础模型输出图像的嵌入向量用于下游任务，但常见编辑操作（JPEG 压缩、亮度/对比度调整等）会改变嵌入向量。与对抗扰动不同，常见扰动在非对抗的真实场景中频繁发生。三个核心问题：(1) 基础模型本身有多鲁棒？(2) 下游应用有多鲁棒？(3) 如何提升鲁棒性？设计合适的度量来量化鲁棒性是关键挑战。

## 方法详解

### 整体框架

这篇论文系统研究视觉基础模型对常见扰动（JPEG 压缩、亮度调节等）有多脆弱。整体分三步：先提出能量化嵌入鲁棒性的度量并分析其数学性质，再用这套度量系统评估六个工业级基础模型在九类扰动下的表现，最后提出一个鲁棒性感知微调把鲁棒性提上来又不牺牲效用。

### 关键设计

**1. DivergenceRadius 度量：用最小包含球半径量化嵌入鲁棒性**

要谈"基础模型鲁不鲁棒"，先得有个站得住的度量。本文把一张图在各种扰动下产生的嵌入向量取最小包含球，用这个球的半径作为鲁棒性度量。它满足全部五个期望性质（有界域、单调性、最优鲁棒性、最差鲁棒性、旋转不变性），而余弦相似度和欧氏距离度量都违反"最差鲁棒性"这一条；本文还顺手证明了余弦相似度度量和欧氏距离度量等价（$\mathcal{R}_{ed} = \sqrt{\mathcal{R}_{cs}}$），说明二者本质同源，只有最小包含球的定义在数学上完备。

**2. 鲁棒性-性能线性关系：用鲁棒性值直接预测下游表现**

测出鲁棒性值之后，本文发现它非常好用：下游分类准确率、深度估计 MSE 都与图像的鲁棒性值近似呈线性关系，简单线性回归就能由鲁棒性值准确预测扰动图像的下游性能。这意味着鲁棒性度量可以当成下游性能的代理指标，部署时不必把每个下游任务都重跑一遍。

**3. 鲁棒性感知微调：在保效用的前提下压低嵌入对扰动的敏感度**

诊断出基础模型普遍不鲁棒后，本文提出鲁棒性感知微调来补救：优化目标是鲁棒性损失与效用损失的加权和——鲁棒性项拉近原图与其扰动版本的嵌入、压低对扰动的敏感度，效用项约束微调后模型在干净图上的嵌入贴近原模型、从而保住下游表示质量。实验证实这样能在多数扰动类型上提升鲁棒性而不损害效用，说明鲁棒和好用并不必然冲突。

### 损失函数 / 训练策略

微调损失 = 鲁棒性损失 $\mathcal{L}_1$ + $\lambda \cdot$ 效用损失 $\mathcal{L}_2$。鲁棒性损失 $\mathcal{L}_1 = -\frac{1}{|\mathcal{D}|}\sum_x \cos(f'(x),\, f'(P(x,k)))$ 拉近原图与其扰动版本在新模型下的嵌入；效用损失 $\mathcal{L}_2 = -\frac{1}{|\mathcal{D}|}\sum_x \cos(f(x),\, f'(x))$ 让新模型在干净图上的嵌入贴近原模型、从而保住效用；$\lambda$ 控制两者平衡（默认 $\lambda=1$）。

## 实验关键数据

### 主实验

评估 CLIP（OpenAI，3 种架构）和 DINOv2（Meta，3 种架构）在 9 类扰动下的表现：

| 发现 | 详情 |
|------|------|
| 普遍不鲁棒 | 所有基础模型对常见扰动产生显著嵌入变化 |
| 架构影响 | ViT 架构比 ResNet 架构更鲁棒 |
| 下游影响 | 玻璃模糊使零样本 ImageNet 分类准确率下降 9.4% |
| 可预测性 | 鲁棒性值可准确预测下游性能（线性关系 R² 高） |

### 消融实验

- 扰动参数域扩大时鲁棒性单调下降（验证单调性性质）
- 不同扰动类型对嵌入的影响程度差异显著
- 微调后的模型在多数扰动类型上鲁棒性提升

### 关键发现

- 基础模型的鲁棒性问题被严重忽视——简单 JPEG 压缩即可显著改变嵌入
- ViT 比 ResNet 更鲁棒可能源于 Transformer 的全局注意力机制
- 鲁棒性度量可作为下游性能预测的代理指标

## 亮点与洞察

- 形式化五个数学性质并证明哪些度量满足/违反，理论分析严谨
- DivergenceRadius 的最小包含球定义直观且数学完备
- 鲁棒性-性能线性关系具有实际应用价值

## 局限与展望

- 仅考虑了九种常见扰动，组合扰动的效果未分析
- 微调方法需要为每种扰动类型分别训练
- 多模态基础模型（如 CLIP 的文本编码器）的鲁棒性未涉及

## 相关工作与启发

- 为基础模型的部署提供了重要的鲁棒性参考基线
- DivergenceRadius 度量可推广到其他需要量化表示稳定性的场景
- 鲁棒性预测性能的线性关系简化了实际部署中的质量评估

## 评分

7/10 — 系统性强、理论分析严谨、实际价值明确，是基础模型鲁棒性研究的重要基线工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)
- [\[CVPR 2026\] TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)
- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](../../NeurIPS2025/self_supervised/implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[CVPR 2025\] SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers](../../CVPR2025/self_supervised/sata_spatial_autocorrelation_token_analysis_for_enhancing_the_robustness_of_visi.md)

</div>

<!-- RELATED:END -->
