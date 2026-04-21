---
title: >-
  [论文解读] Shedding More Light on Robust Classifiers under the lens of Energy-based Models
description: >-
  [ECCV 2024][图像生成] 通过将鲁棒判别分类器重新解释为基于能量的模型（EBM），揭示了对抗训练的能量动态规律，提出了基于能量加权的对抗训练方法WEAT，并展示了鲁棒分类器隐含的生成能力。
tags:
  - ECCV 2024
  - 图像生成
---

# Shedding More Light on Robust Classifiers under the lens of Energy-based Models

**会议**: ECCV 2024  
**arXiv**: [2407.06315](https://arxiv.org/abs/2407.06315)  
**领域**: 图像生成

## 一句话总结

通过将鲁棒判别分类器重新解释为基于能量的模型（EBM），揭示了对抗训练的能量动态规律，提出了基于能量加权的对抗训练方法WEAT，并展示了鲁棒分类器隐含的生成能力。

## 研究背景与动机

- 对抗训练（AT）是增强神经网络鲁棒性的核心方法，但近年来在算法层面进展停滞，顶级方法主要依赖更多数据或更好架构获得提升
- 现有研究对AT的理解不够深入，特别是鲁棒分类器为何具有生成能力、过拟合的根本原因等问题缺少统一的解释框架
- 先前工作（Zimmermann et al.）初步建立了AT与EBM的联系，但缺乏系统性分析
- **核心动机**：利用EBM框架统一解释AT的多种现象（鲁棒过拟合、TRADES优势、样本加权、生成能力），并据此设计更好的训练策略

## 方法详解

### 整体框架

将标准判别分类器的logits重新解释为能量函数：
- **联合能量**：$E_\theta(\mathbf{x}, y) = -\theta(\mathbf{x})[y]$（负的真实类logit）
- **边际能量**：$E_\theta(\mathbf{x}) = -\log\sum_k \exp(\theta(\mathbf{x})[k])$（负的LogSumExp）
- 交叉熵损失可分解为：$\mathcal{L}_{CE} = E_\theta(\mathbf{x}, y) - E_\theta(\mathbf{x})$

### 关键设计

**1. AT能量动态三阶段发现**

通过跟踪训练过程中自然数据与对抗数据的能量差 $\Delta E_\theta(\mathbf{x}) = E_\theta(\mathbf{x}) - E_\theta(\mathbf{x}^\star)$，发现SAT训练分为三个阶段：
- 第一阶段：能量差波动较小
- 第二阶段：能量差接近稳定
- **第三阶段**：能量差急剧下降 → 对应鲁棒过拟合的发生

**2. TRADES的EBM重新解释**

将TRADES的KL散度项改写为能量形式（Proposition 1）：

$$KL(p(y|\mathbf{x}) \| p(y|\mathbf{x}^\star)) = \underbrace{\mathbb{E}_{k \sim p(y|\mathbf{x})}[E_\theta(\mathbf{x}^\star, k) - E_\theta(\mathbf{x}, k)]}_{\text{条件项}} + \underbrace{E_\theta(\mathbf{x}) - E_\theta(\mathbf{x}^\star)}_{\text{边际项}}$$

揭示TRADES隐式地对齐自然数据和对抗数据的能量，从而缓解过拟合。

**3. 能量景观平滑性**

- 发现所有RobustBench上的SOTA鲁棒模型共享一个特性：**平滑的边际能量景观**
- 鲁棒性增加 ↔ $\Delta E_\theta(\mathbf{x})$ 趋近于零 ↔ 能量景观更平滑

**4. WEAT（加权能量对抗训练）**

基于发现低能量样本容易过拟合、高能量样本对鲁棒性贡献更大，提出能量加权方案：
- 权重函数：$w = \log(1 + \exp(|E_\theta(\mathbf{x})|))^{-1}$
- 对接近零能量的样本赋予更高权重
- $E_\theta(\mathbf{x})$ 从计算图中分离（detach），避免平凡解

### 损失函数

WEAT有两种变体：
- **WEAT_NAT**：基于TRADES，自然数据上计算CE
- **WEAT_ADV**：对抗数据上计算CE + KL散度
- 使用KL散度作为内部损失生成对抗样本，权重函数应用于整个外部损失

## 实验关键数据

### 主实验

在CIFAR-10、CIFAR-100和SVHN数据集上使用ResNet-18的对比结果：

| 方法 | CIFAR-10 PGD | CIFAR-10 AA | CIFAR-100 PGD | CIFAR-100 AA | SVHN PGD | SVHN AA |
|------|-------------|------------|--------------|-------------|---------|--------|
| SAT | 49.03 | 45.37 | 23.89 | 20.99 | 50.54 | 44.87 |
| TRADES | 52.65 | 49.46 | 28.53 | 24.29 | 55.52 | 48.13 |
| MAIL-TR | 53.09 | 49.42 | 28.79 | 24.24 | 54.94 | 47.48 |
| **WEAT_NAT** | 52.43 | 49.02 | **29.71** | **24.88** | 55.31 | 48.61 |
| **WEAT_ADV** | **53.35** | **49.75** | **30.90** | **25.63** | **56.40** | **49.60** |

在Tiny-ImageNet上（ResNet-18），WEAT_ADV达到18.45% AA准确率，超越TRADES的17.24%和MART的17.79%。

### 消融实验

不同攻击方式对能量分布的影响分析：

| 攻击方式 | 边际能量$E_\theta(\mathbf{x})$偏移 | 联合能量$E_\theta(\mathbf{x},y)$偏移 | 鲁棒准确率 |
|---------|--------------------------------|-----------------------------------|---------| 
| PGD（无目标） | 大幅左移（能量降低） | 右移 | 0% |
| TRADES（KL） | 大幅左移 | 双模态分布 | 30% |
| APGD | 微小左移 | 右移 | 接近0% |
| APGD-T（有目标） | **右移**（能量升高） | 右移至目标类 | — |
| CW | 几乎不变 | 微移 | — |

关键发现：无目标攻击使对抗样本从模型视角更"像分布内"（低能量），而有目标攻击则相反。

### 关键发现

1. **能量平滑是鲁棒性的标志**：所有SOTA鲁棒模型的 $\Delta E_\theta(\mathbf{x})$ 分布都趋近于零
2. **TRADES本质上在对齐能量**：EBM视角解释了TRADES优于SAT的原因在于隐式缓解能量发散
3. **高能量样本 ≈ 误分类样本**：移除高能量正确分类样本对鲁棒性的影响等同于移除误分类样本
4. **生成能力**：通过改进的SGLD采样（PCA初始化+动量），鲁棒分类器可达到显著的IS和FID得分，无需专门的生成训练

## 亮点与洞察

- EBM视角为AT研究提供了统一的分析框架，将之前零散的观察（过拟合、样本加权、TRADES优势、生成能力）串联起来
- 三阶段训练动态发现和能量发散-过拟合的关联具有重要理论意义
- WEAT方法简单优雅：仅基于边际能量加权，无需标签信息，不需要预热期
- 生成能力分析揭示了一个有趣现象：基于KL散度训练的模型（如TRADES）生成能力反而弱于基于CE的模型

## 局限性

- WEAT在CIFAR-10上与SOTA匹配但未超越，优势主要体现在CIFAR-100和Tiny-ImageNet
- 生成能力仍远不及专用生成模型（扩散模型），更多是一种有趣的副产品
- 理论分析主要基于经验观察，能量平滑与鲁棒性的因果关系尚未严格证明
- 仅使用ResNet-18进行实验，缺少对更大模型（如WideResNet-70-16）的验证

## 评分

- **创新性**: ⭐⭐⭐⭐ — EBM视角的AT分析和WEAT方法都有明确创新
- **实用性**: ⭐⭐⭐⭐ — WEAT即插即用，能量加权不依赖标签
- **实验充分性**: ⭐⭐⭐⭐ — 四个数据集、多种攻击、多个SOTA对比，分析深入
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论推导清晰、可视化丰富、前后呼应好

<!-- RELATED:START -->

## 相关论文

- [Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)

<!-- RELATED:END -->
