---
title: >-
  [论文解读] Boosting Adversarial Transferability with Spatial Adversarial Alignment
description: >-
  [NEURIPS2025][AI安全][adversarial transferability] 提出 Spatial Adversarial Alignment (SAA)，通过空间感知对齐和对抗感知对齐两个模块微调代理模型，使其特征与见证模型对齐，从而显著提升对抗样本的跨架构迁移性（CNN→ViT 迁移率提升 25-39%）。
tags:
  - NEURIPS2025
  - AI安全
  - adversarial transferability
  - model alignment
  - cross-architecture attack
  - spatial features
  - adversarial features
---

# Boosting Adversarial Transferability with Spatial Adversarial Alignment

**会议**: NEURIPS2025  
**arXiv**: [2501.01015](https://arxiv.org/abs/2501.01015)  
**代码**: 待确认  
**领域**: ai_safety  
**关键词**: adversarial transferability, model alignment, cross-architecture attack, spatial features, adversarial features  

## 一句话总结

提出 Spatial Adversarial Alignment (SAA)，通过空间感知对齐和对抗感知对齐两个模块微调代理模型，使其特征与见证模型对齐，从而显著提升对抗样本的跨架构迁移性（CNN→ViT 迁移率提升 25-39%）。

## 背景与动机

对抗样本的可迁移性是黑盒攻击的核心：在无法获取目标模型参数和架构的前提下，攻击者只能利用代理模型（surrogate model）生成对抗样本，再借助迁移性攻击未知的目标模型。现有提升迁移性的方法包括高级优化（MI、NI 等）、数据增强（DI、TI、SSA 等）和模型修改（SGM、LinBP 等），但在跨架构场景（如 CNN→ViT）下效果仍然有限。

已有的 Model Alignment (MA) 方法仅对齐模型的最终预测 logits，存在两个关键不足：

1. **空间特征未对齐**：CNN 和 ViT 的中间层特征在语义层级和空间结构上差异很大，仅靠最终输出约束难以让中间特征趋同
2. **对抗特征未考虑**：对抗样本具有与干净样本不同的特征分布，跨模型的对抗特征相似性同样重要，但 MA 忽略了这一点

## 核心问题

如何让代理模型学到与不同架构模型共享的空间特征和对抗特征，使生成的对抗扰动可有效迁移到 CNN 和 ViT 等不同架构上？

## 方法详解

### 整体框架

SAA 利用一个见证模型（witness model）微调代理模型，包含两个核心模块：

### 1. Spatial-aware Alignment（空间感知对齐）

**全局对齐**：最小化代理模型和见证模型最终输出（logits）之间的 KL 散度：

$$\mathcal{L}_{global}(x;\theta_s) = D_{KL}(f_{\theta_s}(x), f_{\theta_w}(x))$$

**局部对齐**：将 CNN 最后卷积层特征图和 ViT 的 patch token embeddings 都变换为 $(B, C, H, W)$ 形状，按空间位置逐一对齐。用见证模型的局部伪标签监督代理模型的局部特征：

$$\mathcal{L}_{local}(x;\theta_s) = \frac{1}{HW}\sum_{q=1}^{HW} D_{CE}(z_{\theta_s}^{[q]}(x), \hat{y}_{\theta_w}^{[q]})$$

空间对齐总损失：$\mathcal{L}_{SA} = \mathcal{L}_{global} + \gamma \cdot \mathcal{L}_{local}$，其中 $\gamma=0.2$。

### 2. Adversarial-aware Alignment（对抗感知对齐）

引入自对抗策略：利用代理模型梯度迭代生成对抗样本，使代理模型输出偏离见证模型的全局特征：

$$x_{adv}^{(t+1)} = \Pi_{x,\epsilon}(x_{adv}^{(t)} + \alpha \cdot \text{sign}(\nabla_x D_{KL}(f_{\theta_s}(x_{adv}^{(t)}), f_{\theta_w}(x))))$$

然后在生成的对抗样本上也执行全局+局部对齐：

$$\mathcal{L}_{AA}(x_{adv};\theta_s) = \mathcal{L}_{global}(x_{adv};\theta_s) + \omega \cdot \mathcal{L}_{local}(x_{adv};\theta_s)$$

### 3. 总优化目标

$$\mathcal{L}_{SAA} = \mathcal{L}_{SA}(x;\theta_s) + \kappa \cdot \mathcal{L}_{AA}(x_{adv};\theta_s)$$

超参数设置：$\gamma=0.2$，$\omega=0.02$，$\kappa=0.02$。仅用原始训练数据微调 1 个 epoch，SGD 优化器（momentum=0.9）。

## 实验关键数据

在 ImageNet-compatible 数据集上评估，目标模型包括 6 个 CNN（Res18/50/101、VGG19、DN121、Inc-v3）和 4 个 ViT（ViT-B、Swin-B、PVT-v2、MobViT）。

### 与 MA 对比（代理模型 Res50，MI 攻击）

| 见证模型 | MA 平均 ASR | SAA 平均 ASR | SAA 在 ViT 上的提升 |
|---------|-----------|------------|-----------------|
| Res50   | 45.8%     | 58.8%      | +39.1%          |
| DN121   | 63.9%     | 75.8%      | +31.3%          |
| ViT-B   | 53.5%     | 63.9%      | +25.5%          |
| Swin-B  | 44.4%     | 57.5%      | +37.7%          |

### SAA 叠加现有攻击方法（代理 Res50，见证 ViT-B）

| 攻击方法 | 原始 Avg ASR | +SAA Avg ASR | 提升 |
|---------|-------------|-------------|-----|
| MI      | 42.2%       | 63.9%       | +21.7% |
| DI-MI   | 55.4%       | 78.5%       | +23.1% |
| SSA-MI  | 78.5%       | 85.1%       | +6.6%  |

### 特征相似度验证

代理 Res50 与见证 ViT-B 之间的全局特征余弦相似度：对齐前 0.0533 → 对齐后 0.1408（干净图像），提升 164%。

## 亮点

1. **首次揭示空间特征和对抗特征对跨架构迁移性的重要性**，为模型对齐提供了新视角
2. **局部对齐设计巧妙**：将 CNN 特征图和 ViT patch embedding 统一映射到相同空间尺度再逐位置对齐，弥合了架构差异
3. **即插即用**：SAA 仅需微调代理模型 1 个 epoch，可无缝叠加到 MI、DI、TI、SSA 等主流迁移攻击上
4. **跨架构提升显著**：CNN→ViT 迁移率提升 25-39%，远超仅对齐 logits 的 MA 方法

## 局限与展望

1. **依赖见证模型选择**：不同见证模型对最终迁移性影响差异较大，论文提供了经验指导但缺乏理论保证
2. **仅考虑分类任务**：未验证在检测、分割等下游任务上的效果
3. **微调开销**：虽然仅需 1 epoch，但对 ImageNet 规模数据集的微调仍有一定计算成本
4. **防御场景缺乏**：仅在标准模型上评估，缺少对对抗训练模型或防御方法的攻击评估
5. **局部对齐的尺度选择**（$H \times W$）对不同架构组合的敏感性未深入分析

## 与相关工作的对比

| 方法类别 | 代表方法 | 与 SAA 的区别 |
|---------|---------|-------------|
| 优化类 | MI, NI, VMI | 仅优化梯度，不修改模型；SAA 微调模型本身 |
| 增强类 | DI, TI, SSA | 输入变换减少过拟合；与 SAA 正交可叠加 |
| 模型修改类 | SGM, LinBP | 调整梯度传播路径；不涉及跨模型对齐 |
| 对齐类 | MA | 仅全局 logits 对齐；SAA 增加局部空间+对抗特征对齐 |

## 启发与关联

- 空间对齐的思路可迁移到知识蒸馏场景：逐位置的特征对齐比仅对齐最终输出更有效
- 自对抗策略（在对齐过程中使用对抗样本）对对抗鲁棒性研究也有启发：训练时考虑对抗特征可能提升模型泛化能力
- 跨架构的特征对齐方法有望应用于模型融合、联邦学习等需要不同架构协同的场景

## 评分
- 新颖性: 4/5（局部空间对齐+对抗感知对齐的组合是新贡献）
- 实验充分度: 4/5（10 个模型覆盖 CNN 和 ViT，叠加多种攻击方法，消融完整）
- 写作质量: 4/5（结构清晰，可视化分析充分）
- 价值: 4/5（对对抗迁移性研究有实际推动，跨架构攻击提升显著）

<!-- RELATED:START -->

## 相关论文

- [Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](../../ICCV2025/ai_safety/enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)
- [Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment](../../ICCV2025/ai_safety/towards_adversarial_robustness_via_debiased_high-confidence_logit_alignment.md)
- [Generative Adversarial Perturbations with Cross-paradigm Transferability on Localized Crowd Counting](../../CVPR2026/ai_safety/generative_adversarial_perturbations_with_cross-paradigm_transferability_on_loca.md)
- [On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [Distributional Adversarial Attacks and Training in Deep Hedging](distributional_adversarial_attacks_and_training_in_deep_hedging.md)

<!-- RELATED:END -->
