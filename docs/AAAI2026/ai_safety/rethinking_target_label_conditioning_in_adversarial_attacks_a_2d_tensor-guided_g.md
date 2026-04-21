---
title: >-
  [论文解读] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach
description: >-
  [AI安全] 提出 TGAF 框架，利用扩散模型将目标标签编码为 2D 语义张量来引导对抗噪声生成，并设计随机遮挡策略保留完整语义信息，显著提升目标对抗攻击的可迁移性。
tags:
  - AI安全
---

# Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach

- **会议**: AAAI 2026
- **arXiv**: [2504.14137](https://arxiv.org/abs/2504.14137)
- **代码**: [GitHub - TemenosMistral/TGAF](https://github.com/TemenosMistral/TGAF)
- **领域**: ai_safety / 对抗攻击
- **关键词**: 对抗样本, 目标攻击, 可迁移性, 扩散模型, 2D 语义张量, 多目标攻击

## 一句话总结

提出 TGAF 框架，利用扩散模型将目标标签编码为 2D 语义张量来引导对抗噪声生成，并设计随机遮挡策略保留完整语义信息，显著提升目标对抗攻击的可迁移性。

## 背景与动机

- **对抗攻击分类**：无目标攻击只需引起误分类，目标攻击需迫使模型输出攻击者指定的类别，后者更具威胁性（如欺骗自动驾驶将停车标志识别为限速标志）
- **可迁移性瓶颈**：目标攻击在黑盒场景下的成功率远低于无目标攻击，根本原因是容易过拟合代理模型的决策边界
- **现有方法局限**：C-GSP 和 CGNC 等多目标生成方法将标签编码为 **1D 张量**（one-hot 或 CLIP 嵌入），丢失了空间结构信息，导致生成的对抗噪声缺乏目标类别的精细视觉特征
- **核心发现**：作者从"语义特征植入"角度系统分析了两个关键因素：
    - **特征质量（Feature Quality）**：植入的目标特征的结构完整性，缺失关键判别信息会导致部分模型无法识别
    - **特征数量（Feature Quantity）**：植入的目标特征的空间充分性，不足会限制受害模型对该特征的关注

## 方法详解

### 整体架构：TGAF 框架

TGAF 包含四个核心组件：图像编码器 $\mathcal{E}$、文本到图像编码器 $\mathcal{G}$、特征融合模块 $\mathcal{F}$ 和图像解码器 $\mathcal{D}$。

给定输入图像 $\mathbf{I} \in \mathbb{R}^{C \times H \times W}$ 和目标类别 $c_t$，生成扰动 $\delta$ 使得对抗样本 $x_{adv} = x + \delta$ 在 $\ell_\infty$ 范数约束下误导模型：

$$\arg\max f_\Phi(x_{adv}) = c_t, \quad \|\delta\|_\infty \leq \epsilon$$

### 设计一：2D 目标语义张量生成（Text-to-Image Encoder）

**核心创新**：利用 Stable-Diffusion-2 的编码器和去噪 UNet，将目标类别标签转换为 2D 空间表示，而非传统的 1D 向量。

- 输入目标标签文本，经扩散模型处理得到低维潜在向量 $B \times 4 \times 64 \times 64$
- 通过卷积层和平均池化对齐到图像特征空间：$\mathbf{z}_c \in \mathbb{R}^{4 \times \frac{H}{4} \times \frac{W}{4}}$
- **关键优势**：扩散模型仅在训练前对每个目标类别运行一次，生成的 2D 张量保存到磁盘，训练和推理时直接加载，不增加推理开销
- 保留了目标类别的低级语义信息（空间结构、纹理细节），避免了 1D 编码的信息损失

### 设计二：双路特征融合策略（Feature Integration Module）

融合图像表示 $\mathbf{x}$ 和目标条件表示 $\mathbf{z}_c$，采用两种互补策略：

**卷积融合（CbF）**：通过 $1 \times 1$ 卷积学习局部特征交互：

$$\mathbf{f}_c = \text{Conv}_{1 \times 1}(\mathbf{x} \| \mathbf{z}_c)$$

**Transformer 融合（TbF）**：捕获全局空间-通道依赖关系，包含三个阶段：
1. 通道对齐：$\mathbf{z}_t = \text{Conv}_{1 \times 1}(\mathbf{z}_c)$
2. 通道注意力重校准：$\mathbf{x}_{ca} = \mathbf{x}_c \odot \text{CHA}(\mathbf{x}_c)$
3. 自注意力 + 交叉注意力融合：

$$\mathbf{f}_t = \text{CA}(\text{SA}(\mathbf{x}_{ca}), \mathbf{z}_t)$$

其中自注意力和交叉注意力分别为：

$$\text{SA}(\mathbf{x}_{ca}) = \text{Softmax}\left(\frac{\mathbf{Q}_{ca}\mathbf{K}_{ca}^T}{\sqrt{d_k}}\right)\mathbf{V}_{ca}$$

$$\text{CA}(\mathbf{x}_{sa}, \mathbf{z}_t) = \text{Softmax}\left(\frac{\mathbf{Q}_{sa}\mathbf{K}_{z_t}^T}{\sqrt{d_k}}\right)\mathbf{V}_{z_t}$$

### 设计三：动态块级随机遮挡策略（Mask Mechanism）

- 将图像划分为 $N \times N$ 个大小不等的块，训练时随机选择 2 个块进行遮挡
- **目的**：确保部分噪声区域仍保留完整的目标类别语义信息，防止噪声集中在容易映射的区域
- 解码器输出经 $\tanh$ 投影：$\delta = \epsilon \cdot \tanh(\mathbf{o})$

### 训练目标

端到端优化交叉熵损失：

$$\theta^* \leftarrow \arg\min_\theta \mathcal{L}_{\text{CE}}\left(f_\Phi(\mathbf{x}_s + \mathcal{D}_\theta(\mathcal{F}_\theta(\mathcal{E}_\theta, \mathcal{G}_\theta)), c_t\right)$$

## 实验结果

### 实验设置

- **数据集**：ImageNet 训练集训练，ImageNet-NeurIPS 1k 评估
- **代理模型**：Inc-v3 和 Res-152
- **扰动预算**：$\epsilon = 16/255$，训练 10 epochs，学习率 2e-4
- **基线**：Logit、SU、Everywhere（实例特定）；C-GSP、CGNC（实例无关）

### 表一：正常训练模型上的攻击成功率（%）

| 代理模型 | 方法 | Inc-v4 | Inc-Res-v2 | Res-152 | DN-121 | VGG-16 | ViT-B | Swin-T |
|---------|------|--------|------------|---------|--------|--------|-------|--------|
| Inc-v3 | CGNC | 59.41 | 47.98 | 42.50 | 62.91 | 52.63 | 24.81 | 28.16 |
| Inc-v3 | **TGAF** | **72.49** | **63.20** | **61.94** | **78.30** | **70.64** | **33.03** | **42.61** |
| Res-152 | CGNC | 51.59 | 34.18 | — | 85.60 | 63.36 | 34.81 | 40.84 |
| Res-152 | **TGAF** | **62.44** | **44.02** | — | **87.90** | **65.20** | **39.64** | **42.84** |

### 表二：鲁棒训练模型上的攻击成功率（%）

| 代理模型 | 方法 | Inc-v3ADV | IR-v2ENS | Res50SIN | Res50IN | Res50FINE | Res50AUG |
|---------|------|-----------|----------|----------|---------|-----------|----------|
| Inc-v3 | CGNC | 24.30 | 22.51 | 8.88 | 40.81 | 52.13 | 22.83 |
| Inc-v3 | **TGAF** | **39.69** | **34.86** | **17.76** | **64.79** | **72.36** | **43.53** |
| Res-152 | CGNC | 22.15 | 26.70 | 29.81 | 79.82 | 84.05 | 63.66 |
| Res-152 | **TGAF** | **27.73** | **32.71** | **38.07** | **84.53** | **88.48** | **68.63** |

## 关键发现

1. **2D 编码 vs 1D 编码**：2D 语义张量保留了空间结构信息，使对抗噪声包含更完整的目标类别特征（如气压计的指针、无花果的数量）
2. **双融合策略互补**：消融实验显示移除 CbF（TGAF-Conv）或 TbF（TGAF-CA）均导致性能下降，CbF 捕获局部特征，TbF 建模全局依赖
3. **遮挡策略有效**：移除遮挡策略（TGAF-N）导致平均 ASR 下降约 6-8%，替换为 CGNC 遮挡策略（TGAF-C）也有明显下降
4. **块数敏感性**：$N=3$ 效果最佳，$N=2$ 遮挡区域过大、$N=4$ 遮挡过于细碎
5. **图像质量几乎无损**：TGAF 与 CGNC 在 SSIM/LPIPS/FID 指标上差异极小（LPIPS 仅差 0.013），PSNR 略优

## 亮点

- **视角新颖**：首次从特征质量和特征数量两个维度系统分析目标攻击可迁移性
- **扩散模型的巧妙利用**：仅在预处理阶段使用一次扩散模型生成 2D 张量并缓存，不影响训练/推理效率
- **全面实验**：覆盖正常模型、鲁棒模型、多种防御方法（预处理/去噪/扩散净化），均取得 SOTA
- **跨架构迁移**：对 CNN 和 Transformer 架构均有显著提升

## 局限性

- 依赖 Stable Diffusion 生成 2D 张量，对扩散模型生成质量有一定依赖
- 在 DiffPure 等强力防御下 ASR 依然很低（<1%），说明该类防御仍是重大挑战
- 实验仅在 ImageNet 分类任务上验证，未扩展至检测/分割等下游任务
- 扰动预算固定为 $\epsilon=16/255$，较大的预算设置下的行为未探讨

## 相关工作

- **实例特定攻击**：Logit（Zhao et al. 2021）、SU/DTMI-Logit-SU（Wei et al. 2023）、Everywhere/CFM（Zeng et al. 2025）——逐样本迭代优化，效率低
- **实例无关攻击**：C-GSP（Yang et al. 2022）用 CLIP 嵌入做 1D 条件生成；CGNC（Fang et al. 2024）引入微调遮挡但仍受限于 1D 编码
- **防御方法**：对抗训练（Goodfellow et al. 2014）、预处理防御（JPEG/BitSqueezing/Smoothing）、扩散净化（DiffPure, NRP）

## 评分

⭐⭐⭐⭐ — 方法动机清晰（特征质量+数量分析）、2D 张量引导思路新颖、实验全面且提升显著；但核心依赖预训练扩散模型，防御场景下仍有很大提升空间。

<!-- RELATED:START -->

## 相关论文

- [ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias](problog4fairness_a_neurosymbolic_approach_to_modeling_and_mitigating_bias.md)
- [Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [Ghost in the Transformer: Detecting Model Reuse with Invariant Spectral Signatures](ghost_in_the_transformer_detecting_model_reuse_with_invariant_spectral_signature.md)
- [Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer Collaborative Learning](learning_to_collaborate_an_orchestrated-decentralized_framework_for_peer-to-peer.md)

<!-- RELATED:END -->
