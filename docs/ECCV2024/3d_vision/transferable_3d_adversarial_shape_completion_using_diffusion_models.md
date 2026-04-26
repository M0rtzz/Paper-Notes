---
title: >-
  [论文解读] Transferable 3D Adversarial Shape Completion using Diffusion Models
description: >-
  [ECCV 2024][3D视觉] 提出3DAdvDiff，利用3D扩散模型通过对抗性形状补全生成高质量的迁移性3D对抗点云，结合模型不确定性、集成对抗引导和显著性评分策略，在黑盒设置下对最新3D模型实现SOTA攻击成功率。
tags:
  - ECCV 2024
  - 3D视觉
---

# Transferable 3D Adversarial Shape Completion using Diffusion Models

**会议**: ECCV 2024  
**arXiv**: [2407.10077](https://arxiv.org/abs/2407.10077)  
**代码**: 无  
**领域**: 3D视觉

## 一句话总结

提出3DAdvDiff，利用3D扩散模型通过对抗性形状补全生成高质量的迁移性3D对抗点云，结合模型不确定性、集成对抗引导和显著性评分策略，在黑盒设置下对最新3D模型实现SOTA攻击成功率。

## 研究背景与动机

### 领域现状

**领域现状**：现有3D对抗攻击方法主要对xyz坐标加扰动，导致视觉质量严重下降且易被防御检测

### 解决思路

**解决思路**：大部分攻击关注白盒场景，对近年新提出的模型（PCT、PRC、GDANet等）迁移性极差

### 现有痛点

**现有痛点**：现有黑盒攻击（AdvPC、PF-Attack）仅对早期模型有效，攻击成功率（ASR）在新模型上不足5%

### 核心矛盾

**核心矛盾**：核心问题**：如何生成高质量、高迁移性的3D对抗点云，同时不破坏原始形状的自然外观

## 方法详解

### 整体框架

以部分点云（partial shape）为先验，利用预训练的3D形状补全扩散模型（PVD），在逆向生成过程中注入对抗引导，生成"看起来自然但能欺骗分类器"的完整点云。

### 关键设计

**对抗形状补全**：在扩散模型的逆向去噪过程中，对每步中间结果施加对抗梯度引导：
- 保持部分形状z_0不变，仅对补全部分x̃施加引导
- 仅在时间步T_adv=(0, 0.2T]内施加引导（早期噪声过大无意义）

**模型不确定性增强迁移性**：
- 利用点云无序特性，对每步中间点云做M次简单随机采样（Bernoulli 0.5）
- 蒙特卡洛估计M次采样下的平均对抗梯度，类似MC-Dropout的贝叶斯推理

**集成对抗引导**：
- 集成多个替代模型（PointNet、DGCNN、PRC）的logits计算对抗损失
- 使用自适应权重（按各模型正确分类比例加权）

**生成质量保持**：
- 计算每个点的显著性评分（梯度之和），仅对top-N个关键点施加扰动
- 对每步扰动施加ℓ_inf范数限制（0.16）

### 损失函数

对抗生成过程中每步的损失为交叉熵损失的梯度，通过I-FGSM风格的梯度引导修改去噪均值。整体通过最大化分类损失来生成误分类的对抗样本。

## 实验关键数据

### 主实验

ShapeNet全类别迁移攻击ASR(%)，替代模型PointNet：

| 方法 | PointNet | PointNet++ | DGCNN | PointConv | CurveNet | PCT | PRC | GDANet | Avg |
|------|----------|------------|-------|-----------|----------|-----|-----|--------|-----|
| PGD | 99.9 | 2.1 | 0.7 | 0.8 | 0.5 | 0.4 | 0.7 | 1.6 | 0.9 |
| PF-Attack | 99.6 | 24.2 | 6.7 | 5.1 | 3.8 | 1.2 | 2.4 | 1.9 | 6.2 |
| 3DAdvDiff | 99.9 | 73.2 | 12.6 | 55.3 | 40.5 | 32.6 | 25.9 | 16.0 | 36.6 |
| **3DAdvDiff_ens** | **99.9** | **97.0** | **99.9** | **94.5** | **93.5** | **80.5** | **99.9** | **85.2** | **90.1** |

### 消融实验

对防御方法的白盒攻击ASR(%)：

| 方法 | 无防御 | SRS | SOR | DUP-Net | IF-Defense | HybridTraining |
|------|--------|-----|-----|---------|------------|----------------|
| PGD | 99.9 | 5.9 | 1.0 | 0.7 | 13.8 | 1.9 |
| GeoA3 | 99.8 | 4.9 | 1.6 | 0.8 | 13.6 | 2.2 |
| SI-Adv | 92.5 | 6.3 | 1.8 | 1.4 | 17.3 | 3.3 |
| **3DAdvDiff** | **99.9** | **78.1** | **55.5** | **52.7** | **79.1** | **15.9** |

### 关键发现

- 集成版本3DAdvDiff_ens在7个黑盒模型上平均ASR达90.1%，而此前最佳PF-Attack仅6.2%
- 对5种防御方法的攻击成功率远超传统方法（SRS: 78.1% vs 5.9%）
- ShapeNet数据集存在长尾分布问题：前5类占50%数据但仅贡献14%成功对抗样本
- 不同架构模型间的梯度余弦相似度很低，解释了传统迁移攻击失败的原因
- 通过形状补全而非坐标扰动生成对抗样本，根本性地提高了视觉质量

## 亮点与洞察

- 将对抗攻击重新定义为"生成未见数据"而非"扰动已有数据"，范式转换
- 利用扩散模型逐步引导的特点，每步只需很小的对抗扰动，累积效果显著
- 模型不确定性通过点云随机下采样实现，简洁且有效地利用了点云的无序特性
- 为3D点云分类模型的鲁棒性评估建立了新基准

## 3D黑盒攻击的特有挑战

作者深入分析了3D黑盒攻击为何比2D更难：
1. ShapeNet数据集呈长尾分布，头部5类占50%数据但仅贡献14%成功对抗样本
2. 不同架构的3D模型（PointNet vs DGCNN vs PCT）之间的梯度余弦相似度非常低
3. 坐标扰动在3D中比像素扰动更容易被感知

这解释了为何传统迁移攻击方法（PGD、GeoA3等）在新模型上ASR不足2%。3DAdvDiff通过扩散模型的渐进生成（而非直接扰动）和模型不确定性绕过了这些根本性障碍。

## 局限与展望

- 依赖预训练扩散模型的质量，目前仅在ShapeNet几个类别上验证
- 生成速度较慢（需完整1000步扩散过程）
- 对抗成功率仍受部分形状选择影响
- ModelNet40上的实验结果在附录中，主实验集中在ShapeNet

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首次将扩散模型用于3D对抗攻击
- 有效性：⭐⭐⭐⭐⭐ — 黑盒ASR从6%到90%的飞跃
- 实用性：⭐⭐⭐⭐ — 安全评估工具
- 推荐度：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

<!-- RELATED:END -->
