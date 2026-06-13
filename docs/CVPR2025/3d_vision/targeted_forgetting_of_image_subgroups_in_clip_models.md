---
title: >-
  [论文解读] Targeted Forgetting of Image Subgroups in CLIP Models
description: >-
  [CVPR 2025][3D视觉][CLIP遗忘] 提出三阶段 CLIP 子群图像遗忘框架（forgetting → reminding → restoring），通过相对 Fisher Information 选择关键层进行 LoRA 微调，利用 BatchNorm 统计量对齐 retain 数据分布，再通过 model souping 恢复零样本能力，在 ImageNet-1K 和 CIFAR-10 上实现精准子群遗忘（target↓到 0%）同时保持 85-93% 的综合得分。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "CLIP遗忘"
  - "子群图像遗忘"
  - "相对Fisher信息"
  - "模型合并"
  - "知识蒸馏"
---

# Targeted Forgetting of Image Subgroups in CLIP Models

**会议**: CVPR 2025  
**arXiv**: [2506.03117](https://arxiv.org/abs/2506.03117)  
**代码**: 无  
**领域**: 模型压缩 / 机器遗忘  
**关键词**: CLIP遗忘, 子群图像遗忘, 相对Fisher信息, 模型合并, 知识蒸馏

## 一句话总结
提出三阶段 CLIP 子群图像遗忘框架（forgetting → reminding → restoring），通过相对 Fisher Information 选择关键层进行 LoRA 微调，利用 BatchNorm 统计量对齐 retain 数据分布，再通过 model souping 恢复零样本能力，在 ImageNet-1K 和 CIFAR-10 上实现精准子群遗忘（target↓到 0%）同时保持 85-93% 的综合得分。

## 研究背景与动机

**领域现状**：CLIP 等基础模型在 LAION-5B 等大规模互联网数据上预训练，具备强大的零样本分类能力。然而训练数据中包含有害内容（歧视性图像、版权违规、个人信息等），这些有问题的知识被不可避免地编码到模型参数中。

**现有痛点**：现有遗忘方法面临三个核心挑战：(1) 预训练数据不可访问——LAION-5B 的完整数据集不可获取，无法使用传统遗忘方法；(2) 粗粒度标签导致过度遗忘——用户可能只想遗忘"波音飞机"而非所有"飞机"，但粗标签无法区分子群；(3) 分布偏移——遗忘/保留数据与预训练数据分布差异大，直接微调导致灾难性遗忘。

**核心矛盾**：需要在不访问原始预训练数据的前提下，精准遗忘同一类内的特定子群（如遗忘"狨猴"而保留其他"猴子"），同时维持 CLIP 在所有其他数据集上的零样本通用能力。

**本文目标**：在 CLIP 中实现 fine-grained subgroup image forgetting，且不依赖预训练数据。

**切入角度**：分析发现直接 GA（梯度上升）导致灾难性过度遗忘——甚至相似子群也被遗忘。根本原因是参数更新不受控制地传播到与 retain 数据相关的特征空间。

**核心 idea**：三阶段方法：先用相对 Fisher Information 选出"对 forget 样本重要但对 retain 样本不重要"的层做选择性遗忘，再用分布对齐的 retain 数据提醒模型保留知识，最后用 model merging 恢复零样本能力。

## 方法详解

### 整体框架
输入：遗忘数据集 $D^f$（目标子群图像）+ 手动构建的 retain 数据集 $D^r$（同类其他子群图像）。三阶段流程：(1) Forgetting：用相对 Fisher Information 选层 + LoRA 微调遗忘；(2) Reminding：对齐 retain 数据分布后微调 + EMA 防止过拟合；(3) Restoring：model souping 恢复零样本性能。

### 关键设计

1. **相对 Fisher Information 选层（Relative Fisher for Layer Selection）**:

    - 功能：识别对遗忘数据重要但对保留数据影响小的关键层
    - 核心思路：计算每层 $l$ 的相对 Fisher Information $\mathcal{I}^l = \frac{\mathbb{E}_{D^f}[\nabla^2_{\theta^l} \text{sim}]}{\mathbb{E}_{D^r}[\nabla^2_{\theta^l} \text{sim}]}$，比值越大说明该层对遗忘数据更敏感而对 retain 数据不敏感。选择比值最高的层用 LoRA 微调，其余冻结
    - 设计动机：传统 Fisher Information 只看遗忘数据的敏感度，忽略了对其他数据的影响。相对比值平衡了"遗忘效力"和"保留安全性"

2. **分布对齐微调（Distribution-Aligned Reminding）**:

    - 功能：在 retain 数据上恢复遗忘阶段的过度遗忘，同时缩小 retain 数据与预训练数据的分布差距
    - 核心思路：对 retain 数据添加可学习扰动 $\delta_i$，通过最小化 $\mathcal{L}_a = \sum_l \|\mu_l^{img}(x_i + \delta_i) - BN_l^\mu\| + \|\sigma_l^{img}(x_i + \delta_i) - BN_l^\sigma\|$ 将 retain 样本的中间特征统计量对齐到 BN 层记录的预训练分布。然后用对齐后的数据微调模型，并使用 EMA（$\theta^{ema} = \alpha \theta^{ema} + (1-\alpha) \theta$）防止过拟合
    - 设计动机：直接用小规模 retain 数据微调会引入分布偏差并导致过拟合；BN 层的统计量隐式编码了预训练数据的全局分布信息

3. **模型合并恢复（Model Souping Restoration）**:

    - 功能：恢复 CLIP 的泛化零样本能力
    - 核心思路：用小型校准数据集 $D_m$ 搜索最优合并系数 $\alpha$，执行 $\theta = \alpha \theta^f + (1-\alpha) \theta^{ori}$。经验上模型合并能定位损失景观中的平坦最优点，提升泛化能力
    - 设计动机：遗忘和提醒阶段都会偏移 CLIP 的原始表示空间，model merging 通过与原始权重插值回拉模型

### 损失函数 / 训练策略
遗忘阶段最小化 $\mathcal{L}_f = \sum_i \frac{g^{img}(x_i^{img}) \cdot g^{txt}(x^{txt})}{\|g^{img}(x_i^{img}) \cdot g^{txt}(x^{txt})\|}$（最大化图像-文本对齐度作为遗忘损失）。提醒阶段使用标准对比损失在对齐后的 retain 数据上恢复。整个流程中关键层使用 LoRA 适配，大幅减少参数更新量。

## 实验关键数据

### 主实验：ImageNet-1K 子群遗忘
遗忘 "marmoset"（狨猴）子群，保留其他 3 种猴子：

| Backbone | 方法 | Target ↓ | Retain ↑ | ImageNet All ↑ | CIFAR ↑ | Food ↑ | STL ↑ | ObjectNet ↑ | Score |
|---|---|---|---|---|---|---|---|---|---|
| RN50 | Original | 51.0 | 54.7 | 59.8 | 70.4 | 60.9 | 92.0 | 68.9 | – |
| | GA | 0.0 | 0.9 | 32.2 | 16.4 | 22.9 | 63.3 | 22.1 | 45.3 |
| | EMMN | 0.0 | 56.7 | 28.4 | 13.1 | 20.4 | 54.9 | 20.5 | 55.6 |
| | LIP | 0.7 | 0.2 | 1.3 | 10.7 | 0.2 | 10.3 | 1.6 | 18.6 |
| | **Ours** | **0.0** | **50.2** | **54.5** | **85.7** | **62.9** | **81.5** | **45.2** | **91.0** |
| RN101 | GA | 0.0 | 0.3 | 36.2 | 19.9 | 29.3 | 58.6 | 19.4 | 47.3 |
| | **Ours** | **0.0** | **47.7** | **58.5** | **69.7** | **56.9** | **93.5** | **45.9** | **92.9** |

### 遗忘 "box turtle" 子群（RN50）

| 方法 | Target ↓ | Retain ↑ | All ↑ | CIFAR ↑ | Food ↑ | STL ↑ | ObjNet ↑ | Score |
|---|---|---|---|---|---|---|---|---|
| GA | 0.0 | 13.3 | 42.1 | 14.6 | 40.0 | 77.8 | 21.3 | 55.8 |
| EMMN | 0.1 | 57.2 | 28.1 | 12.5 | 22.1 | 64.7 | 17.6 | 54.0 |
| **Ours** | **0.0** | **69.4** | **50.6** | **54.5** | **50.5** | **87.6** | **43.1** | **85.9** |

### 消融实验

| 配置 | Target ↓ | Score |
|---|---|---|
| Full method (Ours) | 0.0 | 91.0 |
| w/o Relative Fisher (全层微调) | 0.0 | 72.3 |
| w/o Distribution Alignment | 0.0 | 78.5 |
| w/o Model Souping | 0.0 | 82.1 |
| w/o LoRA (full FT) | 0.0 | 68.7 |

### 关键发现
- **GA 和 LIP 导致灾难性过度遗忘**：GA 在 ImageNet-All 上从 59.8% 暴跌到 32.2%，CIFAR 从 70.4% 到 16.4%。LIP 更极端，CIFAR 降到 10.7%
- **Ours 精准遗忘**：Target 降到 0.0%的同时，Retain 保持 50.2%（原始 54.7%），ImageNet-All 保持 54.5%（原始 59.8%），跨数据集零样本能力几乎不受影响
- **Score 绝对领先**：RN50 上 91.0 vs 次优 EMMN 55.6（+35.4）；RN101 上 92.9 vs 49.5（+43.4）
- **三阶段缺一不可**：去掉 Relative Fisher、Distribution Alignment、Model Souping 分别导致 Score 从 91.0 降到 72.3、78.5、82.1

## 亮点与洞察
- **子群级精准遗忘**：不是遗忘整个类别，而是类内特定子群（如 Boeing vs 公版飞机），这更贴合实际需求（版权、隐私等场景）
- **BN 统计量的巧妙利用**：利用 CLIP 的 BN 层隐式存储的预训练分布信息来弥补预训练数据不可访问的问题，这个 trick 可以推广到任何需要分布对齐的场景
- **三阶段设计干净利落**：每个阶段解决一个明确问题：选择性遗忘 → 防止过度遗忘 → 恢复泛化能力，逻辑链清晰

## 局限与展望
- retain 数据集需要手动构建，在实际场景中人工成本不可忽视
- 只在 ViT 以外的 ResNet backbone 上做了 CLIP 实验，ViT-based CLIP 的结果缺失
- model souping 需要搜索合并系数 $\alpha$，增加了超参调优开销
- 遗忘效果的持久性未验证——继续训练后是否会"重新记忆"目标子群未探讨

## 相关工作与启发
- **vs CLIP-LIP**：将 LRP + LoRA 应用于 CLIP 文本编码器做概念遗忘，但在子群场景下 Score 仅 33.2-44.4；本文三阶段方法达 85.9-92.9
- **vs EMMN**：error minimization-maximization 框架虽然不需数据，但过度遗忘严重，Score 仅 49.5-55.6
- **vs 传统 Fisher-based 遗忘**：只用遗忘数据的 Fisher 信息会选到对 retain 也敏感的层，导致过度遗忘；相对 Fisher 信息解决了这个问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 子群遗忘问题的形式化和三阶段解法都是有价值的贡献
- 实验充分度: ⭐⭐⭐⭐ ImageNet+CIFAR+多数据集跨域，但缺ViT backbone
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但部分数学符号不一致
- 价值: ⭐⭐⭐⭐ 子群遗忘场景实用性强，方法可扩展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Kiss3DGen: Repurposing Image Diffusion Models for 3D Asset Generation](kiss3dgen_repurposing_image_diffusion_models_for_3d_asset_generation.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] Gaussian Eigen Models for Human Heads](gaussian_eigen_models_for_human_heads.md)
- [\[AAAI 2026\] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening](../../AAAI2026/3d_vision/clippan_adapting_clip_as_a_supervisor_for_unsupervised_pansharpening.md)
- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)

</div>

<!-- RELATED:END -->
