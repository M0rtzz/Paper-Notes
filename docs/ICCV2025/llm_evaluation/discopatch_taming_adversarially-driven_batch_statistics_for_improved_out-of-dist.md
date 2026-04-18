---
title: >-
  [论文解读] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection
description: >-
  [ICCV 2025][分布外检测] 提出DisCoPatch框架，利用对抗性VAE中BatchNorm对批统计量的内在偏向性来区分ID和OOD样本，通过推理时将同一图像的多个patch组成batch来保证分布一致性，在协变量偏移OOD检测（ImageNet-1K(-C) 95.5% AUROC）和近分布OOD检测（95.0% AUROC）上达到SOTA，模型仅25MB且延迟低一个数量级。
tags:
  - ICCV 2025
  - 分布外检测
  - 批归一化
  - 对抗性VAE
  - 协变量偏移
  - Patch策略
---

# DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection

**会议**: ICCV 2025  
**arXiv**: [2501.08005](https://arxiv.org/abs/2501.08005)  
**代码**: https://github.com/caetas/DisCoPatch  
**领域**: Others (OOD Detection)  
**关键词**: 分布外检测, 批归一化, 对抗性VAE, 协变量偏移, Patch策略

## 一句话总结
提出DisCoPatch框架，利用对抗性VAE中BatchNorm对批统计量的内在偏向性来区分ID和OOD样本，通过推理时将同一图像的多个patch组成batch来保证分布一致性，在协变量偏移OOD检测（ImageNet-1K(-C) 95.5% AUROC）和近分布OOD检测（95.0% AUROC）上达到SOTA，模型仅25MB且延迟低一个数量级。

## 研究背景与动机
OOD检测旨在识别偏离已知数据分布的样本，对保障系统安全至关重要。OOD检测涵盖三种类型的分布偏移：

**语义偏移**（如新类别）和**域偏移**（如从真实图像到手绘）：边界清晰，研究较多

**协变量偏移**（如模糊、噪声等微妙数据变化）：容易与域偏移混淆，但挑战更大，因为其变化更细微

现有OOD方法主要分为基于生成模型（VAE/GAN/NF/DDPM）、基于重建、基于特征/logit的方法。生成方法面临似然估计反直觉的问题（OOD样本可能获得更高似然）；重建方法需精心调节信息瓶颈；特征方法虽有效但受限于大型transformer的推理速度。

核心洞察：在使用BatchNorm的GAN/对抗网络中，真实样本和对抗样本形成**两个不同的域，具有不同的批统计量**。这个"双域假说"意味着BN天然具有基于批统计量分离ID和OOD样本的能力。但BN也使模型偏向于利用非鲁棒特征。

解决方案：**使用patch策略解耦BN的优势与劣势**——训练时用跨图像的patch促进模型学习鲁棒特征，推理时用同一图像的patch组成batch以确保批统计量对应单一分布。

## 方法详解

### 整体框架
DisCoPatch是一个对抗性VAE框架，包含编码器、解码器（生成器）和判别器。VAE负责重建和生成图像（都作为负样本），判别器利用BatchNorm的批统计量来区分真实patch和生成/重建patch。推理时仅使用判别器。

### 关键设计

1. **对抗性VAE判别器训练**:

    - 功能：训练判别器区分三类样本——真实patch (正)、重建patch (负)、生成patch (负)
    - 核心思路：
        - VAE从编码重建 $z_{real}$ 和随机采样 $z_{fake}$ 分别生成两类负样本
        - 判别器最小化: $\mathcal{L}_D = \mathbb{E}_{x \sim p(x)}[\log(1-\mathcal{D}(x))] + \mathbb{E}_{x \sim p_\theta(x|z_{real})}[\log(\mathcal{D}(x))] + \mathbb{E}_{x \sim p_\theta(x|z_{fake})}[\log(\mathcal{D}(x))]$
    - 设计动机：
        - 重建图像通常缺乏高频细节（模糊），训练判别器将缺高频视为"假"
        - 生成图像常有过度高频噪点，训练判别器将过量高频视为"假"
        - 两者结合使判别器对ID频谱边界的界定更紧致

2. **Patch策略与PatchNorm**:

    - 功能：将256×256图像裁成N个64×64 patch，推理时同一图像patch组成一个batch
    - 核心思路：
        - 训练时：来自不同图像的patch混合成batch → 促进学习跨图像的一致鲁棒特征
        - 推理时：同一图像的patch归一化在一起 → 确保BN的批统计量对应同一数据分布
        - 引入PatchNorm2D：保留BN的权重和偏置，但支持每组N个patch独立归一化
        - 推理时BN设置 $m=1$（完全使用当前batch统计量，忽略训练期间的running stats）
    - 设计动机：如果推理batch中混入OOD样本，其批统计量会偏离ID分布，破坏判别效果；patch策略保证每张图的评估独立且一致

3. **综合损失函数**:

    - 功能：平衡VAE重建/正则化和对抗学习
    - 核心公式：
   $\mathcal{L}_{DCP} = \|x - \mathcal{G}(z)\|^2 - \frac{\omega_{KL}}{2}\sum(\text{KL项}) + \omega_{Rec}\mathbb{E}[\text{重建对抗}] + \omega_{Gen}\mathbb{E}[\text{生成对抗}]$
    - 设计动机：端到端训练使VAE的输出质量随训练进步而自然提升，从而逐渐收紧判别器的ID边界

### 损失函数 / 训练策略
VAE损失 = 重建MSE + KL正则化 + 对抗损失（鼓励重建/生成图骗过判别器）。判别器损失 = 标准三分类交叉熵。端到端联合训练。

## 实验关键数据

### 主实验
| 模型 | Near-OOD(SSB) | Near-OOD(NINCO) | Far-OOD(iNat) | Far-OOD(DTD) | Cov.Shift |
|---|---|---|---|---|---|
| MOODv2(BEiTv2) | 85.0/58.1 | 92.7/38.2 | **99.6/1.8** | 94.3/24.7 | 70.5/73.9 |
| SCALE(ResNet-50) | 77.4/67.7 | 85.4/51.8 | 98.0/9.5 | **97.6/11.9** | 83.3/54.1 |
| NNGuide(RegNet) | 84.7/54.7 | 93.7/28.9 | 99.9/1.8 | 95.8/17.0 | 78.5/61.6 |
| RankFeat(Rv2-101) | 89.4/47.9 | 90.0/39.3 | 96.0/13.0 | 95.0/25.4 | 91.3/38.7 |
| **DisCoPatch-64** | **95.8/19.8** | 94.3/39.0 | 99.1/3.6 | 96.4/18.9 | **97.2/10.6** |

Near-OOD平均AUROC 95.0%（SOTA），协变量偏移AUROC 95.5%（大幅领先），Far-OOD接近最佳但非SOTA。

### 消融实验
| 配置 | 说明 | OOD检测效果 |
|------|------|------------|
| 标准BN (eval模式) | 使用training running stats | 较差 |
| BN (track_running_stats=False) | 仅使用当前batch统计量 | 显著提升 |
| PatchNorm (m=1) | 每组独立归一化 | 最佳 |
| DisCoPatch-16 | 少量patch | 已超越所有基线 |
| DisCoPatch-64 | 更多patch | 最佳（>64无明显收益） |

### 关键发现
- 推理时使用当前batch统计量（而非训练期running stats）对OOD检测至关重要
- 模型大小仅25MB，延迟比MOODv2低12倍、比NNGuide低19倍
- UMAP可视化显示Near-OOD样本靠近ID簇，Far-OOD样本远离，特征空间结构良好
- patch数量从16增加到64有持续收益，但>64后饱和
- 协变量偏移检测提升最大（比第二名提升5.9%绝对值），因为重建和生成恰好覆盖了高频/低频两类退化

## 亮点与洞察
- 将BN的"缺陷"（对batch统计量的依赖）转化为OOD检测的优势，思路逆向且巧妙
- 用VAE的重建（缺高频）和生成（过高频）两类负样本收紧频谱边界，设计精妙
- Patch策略同时解决了训练（学鲁棒特征）和推理（保证分布一致性）两个问题
- 模型极其紧凑和高效，完全满足实时部署需求

## 局限与展望
- Far-OOD检测不是SOTA（虽然接近且快得多）
- 当前仅验证了ImageNet-1K作为ID数据集
- 未探索在医学影像等高要求领域的适用性
- 可尝试更强的重建/生成模型（如VQ-GAN、DDPM）替代简单VAE
- 缺乏对特征传播和抑制的信号处理层面分析

## 相关工作与启发
- "双域假说"（真实vs对抗样本在BN中形成不同分布）是本文的理论基石
- Patch-based方法在异常检测中有先例（PatchCore），但用BN统计量的角度是新的
- 可推广到其他需要高效OOD检测的场景：工业检测、自动驾驶安全门

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将BN的batch统计特性与对抗VAE结合用于OOD检测，视角非常独特
- 实验充分度: ⭐⭐⭐⭐ 涵盖Near/Far/Covariate三类OOD + 多种基线 + BN消融 + 延迟对比
- 写作质量: ⭐⭐⭐⭐ 动机清晰，核心思想在introduction中阐述到位
- 价值: ⭐⭐⭐⭐⭐ 实用性极强——SOTA性能、25MB模型、极低延迟，理想的工业部署方案
