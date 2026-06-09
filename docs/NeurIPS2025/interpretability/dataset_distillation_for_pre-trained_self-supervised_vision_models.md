---
title: >-
  [论文解读] Dataset Distillation for Pre-Trained Self-Supervised Vision Models
description: >-
  [NeurIPS 2025][可解释性][dataset distillation] 提出 Linear Gradient Matching 方法，为预训练自监督视觉模型蒸馏合成数据集：每类仅需一张合成图就能训练出接近全数据集表现的线性分类器，且蒸馏图像可跨模型架构迁移。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "dataset distillation"
  - "自监督学习"
  - "linear probing"
  - "gradient matching"
  - "CLIP/DINO"
---

# Dataset Distillation for Pre-Trained Self-Supervised Vision Models

**会议**: NeurIPS 2025  
**arXiv**: [2511.16674](https://arxiv.org/abs/2511.16674)  
**代码**: [https://georgecazenavette.github.io/linear-gm](https://georgecazenavette.github.io/linear-gm)  
**领域**: 可解释性  
**关键词**: dataset distillation, self-supervised learning, linear probing, gradient matching, CLIP/DINO

## 一句话总结
提出 Linear Gradient Matching 方法，为预训练自监督视觉模型蒸馏合成数据集：每类仅需一张合成图就能训练出接近全数据集表现的线性分类器，且蒸馏图像可跨模型架构迁移。

## 研究背景与动机

**领域现状**：数据集蒸馏旨在合成极少量图像，使从头训的模型仍能达到全量数据的性能。现有方法（DC、MTT、DM 等）都是针对从零训练的随机初始化模型。

**现有痛点**：当前视觉范式已从"从头训练"转向"预训练大模型 + 下游微调/linear probing"。现有蒸馏方法既没有针对这种范式设计，也无法利用预训练特征的优势。

**核心矛盾**：传统蒸馏需对整个网络做反向传播（MTT 等方法），在大模型场景下存在严重的内存和稳定性问题；同时蒸馏的图像往往严重过拟合单一架构，无法跨模型使用。

**本文目标**：(1) 为预训练自监督模型设计高效蒸馏方法；(2) 使蒸馏图像能跨模型架构迁移。

**切入角度**：既然下游任务只训练线性分类器，那蒸馏时也只需在线性层的梯度空间做匹配——这大大降低了优化复杂度。受 Platonic Representation Hypothesis 的启发，不同预训练模型学到相似表征，因此蒸馏图像有望跨模型泛化。

**核心 idea**：只匹配预训练特征空间中线性分类器的梯度来蒸馏合成图像，配合金字塔重参数化和可微增强来实现跨模型迁移。

## 方法详解

### 整体框架
输入一个大规模真实数据集和一个预训练自监督特征提取器 $\phi$，输出每类一张合成图像。蒸馏时将真实和合成图通过冻结的 $\phi$ 提取特征，再过一个随机初始化的线性分类器，通过匹配两者在线性层上的梯度来更新合成图像。

### 关键设计

1. **Linear Gradient Matching**:

    - 功能：每步蒸馏时采样随机线性分类器 $W \sim \mathcal{N}(0,1)^{c \times f}$，分别计算真实和合成数据的交叉熵损失对 $W$ 的梯度，然后最小化两者余弦距离。
    - 核心公式：$\ell_{\text{real}} = \text{CE}(W\phi(X_{\text{real}}); Y_{\text{real}})$，$\ell_{\text{syn}} = \text{CE}(W\phi(X_{\text{syn}}); Y_{\text{syn}})$
    - meta loss: $\mathcal{L}_{\text{meta}} = 1 - \cos\left(\text{vec}\left(\frac{\partial \ell_{\text{real}}}{\partial W}\right), \text{vec}\left(\frac{\partial \ell_{\text{syn}}}{\partial W}\right)\right)$
    - 设计动机：只需要匹配线性层梯度，避免了对整个 backbone 反传的内存爆炸问题，同时每步采样新的随机 $W$ 避免过拟合。

2. **金字塔重参数化 (Pyramid Representation)**:

    - 功能：合成图像不直接以像素存储，而是存为多尺度金字塔 $\rho = \{1\times1, 2\times2, ..., 256\times256\}$。
    - 渲染公式：$X = \text{sigmoid}\left(\sum_{r \in \rho} \text{resize}_{256}(P_r)\right)$
    - 设计动机：直接像素优化在高分辨率下会产生大量高频噪声模式，严重过拟合蒸馏用的 backbone。金字塔由粗到细渐进合成，引入强正则化，生成的图像更自然且跨模型泛化更好。

3. **颜色去相关 (Color Decorrelation)**:

    - 功能：在去相关颜色空间中学习蒸馏图像，合成后再线性变换回标准 RGB。
    - 设计动机：消除单一 backbone 可能引入的颜色偏见（如某些模型偏向蓝色调），提升跨模型通用性。

4. **多轮可微增强 (Differentiable Augmentations)**:

    - 功能：对合成图像施加水平翻转、随机裁剪、高斯噪声等可微增强，且每步生成多组增强版本（默认 10 组）拼接为 batch。
    - 设计动机：单次增强迫使一张图承载所有信息，多轮增强让优化目标变成"所有增强版本共同组成最优训练集"，极大提升蒸馏效果和跨模型性能。

### 训练策略
- 蒸馏 5000 步，分辨率 224×224，使用 ViT-B 版本的 backbone
- ImageNet-1k 由于计算限制用 3 组增强，其余用 10 组
- 金字塔渐进优化：从最低分辨率开始，逐步加入更高分辨率层级

## 实验关键数据

### 主实验（ImageNet-1k, 每类 1 张图）

| 训练集 | CLIP | DINO-v2 | EVA-02 | MoCo-v3 | 平均 |
|--------|------|---------|--------|---------|------|
| **Distilled (Ours)** | **63.0** | **75.0** | **70.3** | **63.2** | **67.9** |
| Centroids | 53.9 | 69.5 | 58.1 | 57.4 | 59.7 |
| Neighbors | 38.8 | 67.7 | 49.9 | 56.4 | 53.2 |
| Random | 31.7 | 50.3 | 37.7 | 38.8 | 39.6 |
| Full Dataset (1.3M) | 78.7 | 83.0 | 81.7 | 76.5 | 80.0 |

每类仅 1 张合成图，DINO-v2 linear probe 达到 75%（全量 83%，差 8 点），大幅超越所有真实图像 baseline。

### 消融实验（ImageNet-100, 同模型评估 & 跨模型评估）

| 配置 | 同模型平均 Acc | 跨模型平均 Acc |
|------|-------------|-------------|
| Full (完整方法) | 87.2 | 77.8 |
| - Color Decorrelation | 86.5 | 76.4 |
| - Pyramid | 85.7 | 67.1 |
| - Augmentation | 68.6 | 33.3 |

### 关键发现
- **可微增强**贡献最大：去掉后同模型掉 18.6 点，跨模型直接崩到 33.3%
- **金字塔**对跨模型迁移至关重要：去掉后跨模型掉 10.7 点，说明像素优化产生的高频模式严重过拟合特定 backbone
- 用 DINO-v2 蒸馏的数据集跨模型泛化最好（平均 63.7%），验证了高质量模型产生更通用的蒸馏结果
- 蒸馏图像的嵌入通常分布在类别聚类的边缘或外侧，蕴含高区分度特征

## 亮点与洞察
- **极简但高效**：只匹配线性层梯度就够了。利用预训练特征空间的结构性，将蒸馏从"训练整个网络"简化为"训练线性分类器"级别的问题。
- **跨模型迁移**：用 DINO 蒸馏的图像可以直接训 CLIP 的 linear probe 并获得有竞争力的性能，说明不同大模型确实在收敛到相似表征。
- **可解释性工具**：蒸馏图像可以揭示模型关注什么——在 Spawrious 数据集上，DINO 的蒸馏图清晰展示狗的品种，而 MoCo 几乎只关注背景环境，解释了为何 MoCo 在虚假相关数据上失败。

## 局限与展望
- 目前只研究了 linear probing 场景，未探索 fine-tuning 或 adapter 调优的蒸馏
- 跨架构迁移在 CLIP-MoCo 之间表现较差，可能与两者表征空间对齐度低有关
- 只在 ViT-B 上实验，未验证不同模型规模（S/L/H）的影响
- 蒸馏成本仍需 5000 步迭代，对 ImageNet-1k 级别数据集的内存开销可观

## 相关工作与启发
- **vs DC/MTT（传统蒸馏）**: 传统方法对全网络梯度/轨迹做匹配，本文只匹配线性层梯度，大幅降低计算成本且适配预训练范式
- **vs SRe2L（大模型蒸馏）**: SRe2L 等方法扩展蒸馏到大模型但在每类一张图的极端设置下效果不佳，本文专注于此超低数据场景
- 蒸馏作为可解释性工具的视角很新颖，可启发其他方向的模型理解研究

## 评分
- 新颖性: ⭐⭐⭐⭐ 将蒸馏定位为预训练模型的 linear probe 场景，角度新但技术组件（梯度匹配、金字塔、增强）各自不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 4 种 backbone、多个数据集、跨模型/对抗/细粒度/OOD 全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 图文精美，故事线清晰，可解释性分析引人入胜
- 价值: ⭐⭐⭐⭐ 在预训练模型日益普及的今天，为 few-shot linear probing 提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](../../CVPR2025/interpretability/probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)
- [\[ICCV 2025\] AIM: Amending Inherent Interpretability via Self-Supervised Masking](../../ICCV2025/interpretability/aim_amending_inherent_interpretability_via_self-supervised_masking.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](../../CVPR2025/interpretability/scaling_vision_pre-training_to_4k_resolution.md)
- [\[ICML 2026\] Interpretable Self-Supervised Learning via Representer Landmarks and Nyström Approximation](../../ICML2026/interpretability/interpretable_self-supervised_learning_via_representer_landmarks_and_nyström_app.md)
- [\[ICML 2026\] IdEst: Assessing Self-Supervised Learning Representations via Intrinsic Dimension](../../ICML2026/interpretability/idest_assessing_self-supervised_learning_representations_via_intrinsic_dimension.md)

</div>

<!-- RELATED:END -->
