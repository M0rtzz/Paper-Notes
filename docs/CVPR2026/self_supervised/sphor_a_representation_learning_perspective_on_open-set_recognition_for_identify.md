---
title: >-
  [论文解读] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks
description: >-
  [CVPR 2026][自监督学习][开放集识别] 提出 SpHOR，一种两阶段解耦训练的开放集识别方法，通过球面表示学习（vMF 分布）、正交标签嵌入和 Mixup/Label Smoothing 集成，显式塑造特征空间以更好地分离已知/未知类别，在 Semantic Shift Benchmark 上取得最高 5.1% 的 OSCR 提升。
tags:
  - CVPR 2026
  - 自监督学习
  - 开放集识别
  - 表示学习
  - 球面表示
  - von Mises-Fisher 分布
  - 正交约束
---

# SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks

**会议**: CVPR 2026  
**arXiv**: [2503.08049](https://arxiv.org/abs/2503.08049)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: 开放集识别, 表示学习, 球面表示, von Mises-Fisher 分布, 正交约束

## 一句话总结

提出 SpHOR，一种两阶段解耦训练的开放集识别方法，通过球面表示学习（vMF 分布）、正交标签嵌入和 Mixup/Label Smoothing 集成，显式塑造特征空间以更好地分离已知/未知类别，在 Semantic Shift Benchmark 上取得最高 5.1% 的 OSCR 提升。

## 研究背景与动机

**领域现状**: 深度神经网络在安全关键应用中广泛使用，但传统闭集分类假设测试时所有类别在训练中已见过。开放集识别（OSR）要求模型能将未知类别样本正确标记为"未知"，同时保持已知类别的高精度分类。

**现有痛点**: 大多数 OSR 方法将特征提取和分类器联合端到端训练，导致特征表示仅隐式适应未知类别；Vaze 等人发现简单的闭集分类训练策略就能超越许多 OSR 方法，说明分类器层面的改进已接近瓶颈。现有利用对比学习（如 SupCon）的方法也非专门为 OSR 设计。

**核心矛盾**: 欧氏空间中特征幅值可无限增长，导致开放空间无界，未知样本被错误归类的风险极高；同时，共享的类间特征（如背景纹理）使模型陷入"熟悉性陷阱"（Familiarity Trap），把语义相近但未知的类别高置信度误认为已知类。

**本文目标**: 能否通过显式设计特征表示本身（而非依赖分类器）来提升 OSR？如何让特征空间中的类别分离更清晰、开放空间建模更有效？

**切入角度**: 采用两阶段解耦训练——先学特征表示再训分类器，在表示学习阶段引入球面约束、正交标签嵌入和数据增强策略。

**核心 idea**: 将特征投影到超球面上建模为 vMF 分布混合，配合正交标签嵌入强制类间子空间正交，再通过 Mixup 模拟模糊语义从而更好地建模开放空间。

## 方法详解

### 整体框架

SpHOR 采用两阶段解耦训练：

- **Stage 1（表示学习）**: 使用编码器 + 投影网络提取 L2 归一化的球面特征，通过 vMF Alignment Loss + Orthogonality Regularizer 学习类别特异性表示。训练数据经 RandAugment → Label Smoothing → Mixup 增强。
- **Stage 2（分类器训练）**: 冻结编码器特征，仅训练线性分类器，使用标准交叉熵损失。

### 关键设计

1. **球面表示与 vMF Alignment Loss**: 将特征 L2 归一化到超球面上，每个类建模为一个 vMF 分布。损失函数 $\mathcal{L}_{\text{vMFAL}} = -\frac{1}{N}\sum_i \sum_k S_{ik} \log P_{ik}$，其中 $S_{ik}$ 为标签相似度，$P_{ik}$ 为基于余弦相似度的分类概率。理论证明该损失同时促进 Alignment（拉近同类表示和标签嵌入）和 Uniformity（分散不同类的表示）。对于 Mixup 产生的模糊样本，均匀性损失主导使其远离类中心，有效缓解熟悉性陷阱。

2. **正交性正则化 $\mathcal{R}_{\text{Ortho}}$**: 强制标签嵌入互相正交，确保每个类的特征向量占据独立的线性子空间，防止嵌入坍缩。公式为 $\log \frac{1}{|C|^2-|C|}\sum_{j\neq i}\exp(\frac{1}{\tau}(\mu_j \cdot \mu_i)^2)$，比基于 SVD 或 Equiangular Tight Frame 的方法更稳定，且避免负相关。

3. **Mixup + Label Smoothing 集成到表示学习**: 不同于在分类器阶段使用，SpHOR 将 Mixup 和 Label Smoothing 直接引入表示学习阶段。Mixup 生成语义模糊样本模拟未知类，Label Smoothing 减少过拟合。提出 Angular Separability (AS) 和 Norm Separability (NS) 两个新指标量化这些技术对特征表示的改善。

### 损失函数 / 训练策略

- 总训练损失: $\mathcal{L} = \mathcal{L}_{\text{vMFAL}} + \mathcal{R}_{\text{Ortho}}$
- Stage 2 使用标准交叉熵
- OSR 评分规则支持 MaxLogit、PostMax、KNN、NNGuide 四种后处理方式
- 超参数: Stage 1 使用 1024 维线性投影网络，Stage 2 使用线性分类器
- 训练平台: 40GB NVIDIA A100 GPU

## 实验关键数据

### 主实验：Semantic Shift Benchmark (ImageNet 预训练 ResNet-50)

| 方法 | CUB Acc↑ | CUB AUROC(Easy/Hard)↑ | CUB OSCR(Easy/Hard)↑ | SCars Acc↑ | SCars AUROC(E/H)↑ | Aircraft Acc↑ | Aircraft AUROC(E/H)↑ |
|------|----------|----------------------|---------------------|-----------|-------------------|--------------|---------------------|
| ARPL+ | 85.4 | 81.8/73.9 | 73.1/66.9 | 89.8 | 85.0/76.4 | 83.3 | 85.8/74.6 |
| MLS+Mixup+MaxLogit | 88.3 | 86.2/78.0 | 78.6/72.1 | 91.4 | 87.3/82.4 | 81.3 | 87.3/75.3 |
| SupCON+KNN | 78.2 | 88.6/75.3 | 72.8/63.1 | 91.8 | 92.1/81.2 | 88.9 | 89.9/81.4 |
| **SpHOR+MaxLogit** | **90.8** | **91.7/83.3** | **85.7/79.0** | **96.3** | **94.1/83.1** | **90.6** | **91.5/81.1** |

### 消融实验：Legacy CNN-32 Benchmark A (AUROC)

| 方法 | SVHN | CIFAR10 | CIFAR+10 | CIFAR+50 | TIN |
|------|------|---------|----------|----------|-----|
| ARPL | 95.3 | 91.0 | 97.1 | 95.1 | 78.2 |
| MLS | 97.1 | 93.6 | 97.9 | 96.5 | 83.0 |
| ConOSR | 99.1 | 94.2 | 98.1 | 97.3 | 80.9 |
| SpHOR (w/o $\mathcal{R}_{\text{Ortho}}$) | 98.9 | 94.2 | 98.0 | 96.9 | 83.8 |
| **SpHOR** | **99.1** | **94.5** | **98.2** | **97.2** | **84.1** |

### 关键发现

- SpHOR 在 Semantic Shift Benchmark 上取得全面 SOTA，闭集准确率最高达 96.3%（SCars），同时 AUROC 和 OSCR 均领先。
- 在无 ImageNet 预训练的情况下，SpHOR 相比 MLS+Mixup 仍有显著优势（如 CUB 上 OSCR 从 45.7/42.6 提升至 76.7/70.0）。
- 正交正则化 $\mathcal{R}_{\text{Ortho}}$ 贡献约 1-2% 的提升，且在无预训练时效果更显著。
- MaxLogit 是 SpHOR 最佳匹配的评分规则。

## 亮点与洞察

- **理论优雅**: 从 vMF 分布出发统一了对齐性和均匀性分析，理论推导清晰地解释了为何 Mixup 样本能被推离类中心。
- **解耦训练的有效性**: 证明了显式设计特征空间比依赖分类器训练更有效，为 OSR 提供了新范式。
- **新评价指标**: Angular Separability 和 Norm Separability 为量化 OSR 友好特征提供了客观工具。
- **正交约束优于 margin-based 方法**: 不引入额外超参数，避免负相关问题。

## 局限与展望

- 仅在 ResNet-50 和 CNN-32 上验证，缺少 ViT 等现代架构的实验。
- Mixup 生成的模糊样本与真实未知类仍有差距，可考虑更高级的合成策略。
- 两阶段训练增加了额外开销，端到端方案可能更实用。
- 正交约束在类别数远超特征维度时可能退化。

## 相关工作与启发

- **与 SupCon 的关系**: SupCon 使用通用对比学习，而 SpHOR 专门为 OSR 定制了标签嵌入和球面约束。
- **与 ARPL 的关系**: ARPL 使用互反点学习，属于原型方法，受限于欧氏空间；SpHOR 在球面上操作更适合 OSR。
- **启发**: 解耦训练 + 特征空间显式设计是一个有潜力的通用范式，可推广到 OOD 检测、持续学习等领域。

## 评分

- 新颖性: ⭐⭐⭐⭐ — vMF 分布 + 正交嵌入 + Mixup 集成到表示学习阶段的组合是新颖的
- 实验充分度: ⭐⭐⭐⭐ — 覆盖细粒度 SSB 和传统 CNN-32 两大 benchmark，消融充分
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，动机清晰，图表直观
- 价值: ⭐⭐⭐⭐ — 为 OSR 的特征学习提供了理论基础和实践方案

<!-- RELATED:START -->

## 相关论文

- [SpHOR: A Representation Learning Perspective on Open-set Recognition](sphor_a_representation_learning_perspective_on_ope.md)
- [Representation Learning for Spatiotemporal Physical Systems](representation_learning_for_spatiotemporal_physical_systems.md)
- [BoSS: A Best-of-Strategies Selector as an Oracle for Deep Active Learning](boss_a_best-of-strategies_selector_as_an_oracle_for_deep_active_learning.md)
- [TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)

<!-- RELATED:END -->
