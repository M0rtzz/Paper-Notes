---
title: >-
  [论文解读] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders
description: >-
  [ECCV 2024][自监督学习][自监督学习] ViC-MAE 将对比学习和掩码自编码器统一到一个框架中，通过把短视频片段当作增强视角（而非把图像重复为视频），在图像和视频下游任务上同时取得优秀表现——ImageNet-1K top-1 达 87.1%（超越 OmniMAE +2.4%），SSv2 达 75.9%。
tags:
  - ECCV 2024
  - 自监督学习
  - Masked Autoencoder
  - 对比学习
  - Video Representation
  - Transformer
---

# ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders

**会议**: ECCV 2024  
**arXiv**: [2303.12001](https://arxiv.org/abs/2303.12001)  
**代码**: [https://github.com/jeffhernandez1995/ViC-MAE](https://github.com/jeffhernandez1995/ViC-MAE)  
**领域**: 自监督学习 / 视频-图像表征学习  
**关键词**: Self-supervised Learning, Masked Autoencoder, Contrastive Learning, Video Representation, Vision Transformer

## 一句话总结

ViC-MAE 将对比学习和掩码自编码器统一到一个框架中，通过把短视频片段当作增强视角（而非把图像重复为视频），在图像和视频下游任务上同时取得优秀表现——ImageNet-1K top-1 达 87.1%（超越 OmniMAE +2.4%），SSv2 达 75.9%。

## 研究背景与动机

自监督视觉表征学习有两大范式：

**联合嵌入方法**（对比学习）：学习对特定变换的不变性，擅长全局语义表征，但局部特征学习不足

**掩码图像建模**（MAE）：通过重建掩码区域学习局部特征，但全局表征质量较差（线性评估仅 68%）

当前的痛点：
- **视频→图像迁移困难**：视频预训练的模型在图像任务上表现不佳，反之亦然
- **图像-视频联合训练不平衡**：OmniMAE 等方法将图像作为视频处理（复制帧），计算效率低且语义信息利用不充分
- **对比学习与 MAE 结合非平凡**：简单组合会导致表征坍塌（如使用 [CLS] token + VicReg/SimSiam 的失败尝试）

核心洞察：**视频帧间的自然变化（姿态、视角、形变）提供了标准图像增强无法模拟的强增强信号。将短视频当作同一视角的时序增强，而非将图像伪装为视频，是更高效的多模态学习方式。**

## 方法详解

### 整体框架

ViC-MAE 使用孪生 ViT 编码器（共享权重），对两个输入（来自同一视频的两帧，或同一图像的两个增强视角）分别进行：
1. **掩码图像建模（每帧独立）**：学习局部特征
2. **对比学习（跨帧/跨视角）**：通过全局池化聚合局部特征为全局表征，再用 InfoNCE 损失对齐

### 关键设计

1. **时序增强替代图像复制**：

    - 给定视频 $\{I_1, ..., I_T\}$，采用"远距采样"（Distant Sampling）：将视频等分为 $n$ 段，从每段随机采样一帧
    - 两帧时间间距约 1.06 秒——提供足够强的"自然增强"
    - 对图像数据则使用标准的双视角增强（裁剪+翻转+颜色扰动+高斯模糊）
    - 视频和图像在同一 batch 中混合训练

2. **掩码重建分支（局部特征）**：

    - 对两帧分别生成随机掩码 $M_i, M_j$（75% 掩码率），得到可见 token $X_i^{(v)}, X_j^{(v)}$
    - ViT 编码器计算可见 token 的表征，解码器附加 mask token 后重建原始 patch
    - 重建损失：$\mathcal{L}_i^{MASK} = \|{\hat{I}_i - I_i}\|_2^2$（仅在掩码位置计算）

3. **对比学习分支（全局特征）**：

    - **全局池化层 $\Omega$**：对编码器输出的局部 token 特征做均值池化，得到全局表征
    - 预测器 $\mathcal{P}$ 和目标编码器 $\mathcal{T}$（对称设计，Linear→BN→ReLU×2）
    - InfoNCE 损失：$\mathcal{L}^{NEG}_{p_i, z_j} = -\log \frac{\exp(p_i \cdot z_j / \tau)}{\sum_{k=1}^{2N} \mathbb{1}[p_i \neq z_k] \exp(p_i \cdot z_k / \tau)}$
    - 负样本来源：其他视频的帧、同一视频中时间距离过远的帧、batch 中其他图像
    - **池化避免表征坍塌的巧妙设计**：池化后的特征同时受 MAE 重建损失约束（不能为零向量），因此不需要额外的梯度停止（stop-gradient）或熵正则化

4. **损失组合与调度**：

    - 总损失：$\mathcal{L} = \mathcal{L}^{MASK} + \lambda \mathcal{L}^{NEG}$
    - 使用渐进式引入调度：训练初期以 MAE 损失为主让模型先学好局部特征，后逐步增加对比损失权重

### 损失函数 / 训练策略

- 架构：ViT-B/16 和 ViT-L/16，小型 MAE 解码器
- 预训练数据：Kinetics-400 (~300K 视频) + ImageNet-1K (~1.2M 图像)；最大规模版本使用 K710 + MiT + IN1K
- 优化器：AdamW，batch size 512
- 视频微调时：将空间 tokenizer 沿时间维度复制并缩放以初始化时序 tokenizer（标准 16 帧，skip 4）
- 数据增强：视频仅用空间增强（裁剪+翻转，scale [0.5,1]），图像用强增强（+颜色扰动+高斯模糊）

## 实验关键数据

### 主实验

ViT-L/16 在多个基准上的迁移学习结果：

| 方法 | 预训练数据 | IN1K↑ | K400↑ | Places-365↑ | SSv2↑ |
|------|-----------|:---:|:---:|:---:|:---:|
| MAE | IN1K | 85.5 | 82.3 | 59.4 | 57.7 |
| ST-MAE | K400 | 81.7 | 84.8 | 58.1 | 73.2 |
| OmniMAE | K400+IN1K | 84.7 | 84.0 | 59.4 | 73.4 |
| **ViC-MAE** | K400+IN1K | **86.0** | **86.8** | **60.0** | **75.0** |
| **ViC-MAE** | K710+MiT+IN1K | **87.1** | **87.8** | **60.7** | **75.9** |

### 消融实验

ViT-B/16 在 ImageNet-1K 线性评估：

| 消融维度 | 配置 | Top-1 | 说明 |
|---------|------|:---:|------|
| 帧间距 | 0（同一帧） | 63.25 | 无时序增强 |
| 帧间距 | 4 | 65.25 | 连续采样 |
| 帧间距 | D（远距采样） | **67.66** | 最强增强 |
| 池化方式 | GeM | 66.92 | 广义均值 |
| 池化方式 | max | 67.01 | 最大池化 |
| 池化方式 | **mean** | **67.66** | 均值池化最优 |
| 增强策略 | 仅颜色 | 65.40 | 不够 |
| 增强策略 | 仅空间 | 66.03 | 空间更重要 |
| 增强策略 | **颜色+空间** | **67.66** | 图像需强增强 |

### 关键发现

1. **远距采样是关键**：帧间距从 0→D（约 1.06 秒），线性评估性能从 63.25% 提升到 67.66%（+4.4%），说明视频的自然时序变化提供了极强的增强信号
2. **全面超越 OmniMAE**：在相同数据（K400+IN1K）和相同架构（ViT-L）下，ViC-MAE 在所有 4 个基准上全面超越 OmniMAE（IN1K +1.3%, K400 +2.8%, Places +0.6%, SSv2 +1.6%）
3. **视频→图像迁移 SOTA**：仅用视频（MiT）训练的 ViC-MAE 在 IN1K 上达 85.3%——自监督视频预训练的最佳图像迁移结果
4. **对比学习优于纯掩码**：在图像-视频比例消融中，ViC-MAE 始终优于 OmniMAE，说明对比+掩码的组合优于纯掩码预训练
5. **COCO 目标检测**：ViC-MAE (IN1K+K710+MiT) 的 AP^Box=53.2, AP^Mask=46.9，均为同类方法最优

## 亮点与洞察

1. **池化层的双重作用**：全局均值池化既是对比学习的必要聚合步骤，又天然避免了表征坍塌（因为池化特征还受 MAE 重建约束），这是一个优雅的架构设计
2. **视频作为增强**：将视频帧间变化视为自然的数据增强（而非将图像伪装为静止视频），概念简洁且高效——减少了训练 token 数量（98 vs OmniMAE 的 157）
3. **数据效率惊人**：仅用约 425 万数据点就达到了 CAN（用 JFT-300M 的 3 亿数据）相近的精度，充分说明视频数据的信息密度远超图像
4. **负样本的重要性**：实验表明使用负样本（InfoNCE）优于无负样本方法（VicReg, SimSiam），这与其他视频-图像迁移的成功方法一致

## 局限与展望

1. **与纯视频方法的差距**：在 K400 上落后于 VideoMAEV2 0.8%（ViT-L），落后 TubeViT 2.4%（ViT-L）——可能因为 ViC-MAE 不显式建模时序 token
2. **与 DINOv2 的差距**：在 IN1K 上落后 DINOv2 1.2%（ViT-L），后者使用了更多数据和更高级的训练策略
3. **微调时需初始化时序 tokenizer**：通过复制空间 token 初始化时序维度的方式较为粗糙，可探索更好的初始化策略
4. **未利用文本、音频等模态**：未来可结合文本监督（如 UMT）或音频信号进一步提升
5. **帧采样策略相对固定**：远距采样虽然效果好，但可能遗漏中等时间尺度的运动信息

## 相关工作与启发

- 与 OmniMAE 的核心区别：ViC-MAE 将视频当作增强 + 加入对比损失，OmniMAE 将图像当作视频 + 纯掩码建模。前者更高效且性能更好
- 与 C-MAE 的区别：C-MAE 使用 [CLS] token 做对比学习，而 ViC-MAE 使用池化后的特征，避免了坍塌问题
- 启发：MAE 的局部特征和对比学习的全局语义是互补的——**通过池化层桥接两者**是一个通用的设计模式，可推广到其他模态（如点云、语音的自监督学习）

## 评分

- **新颖性**: ⭐⭐⭐⭐ 对比+MAE 的组合虽非全新，但池化层避免坍塌的设计、视频作为增强的视角、渐进式损失调度等贡献清晰
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 4 个主要基准 + 8 个迁移数据集 + COCO 检测分割 + 丰富的消融（帧间距、池化、增强、数据量、图像-视频比例）
- **写作质量**: ⭐⭐⭐⭐ 方法部分清晰，实验部分有很好的 scaling 分析（图3），但公式符号略有冗余
- **实用价值**: ⭐⭐⭐⭐⭐ 一个统一的图像-视频自监督预训练模型，直接即插即用地服务于图像和视频下游任务，代码和权重已开源

<!-- RELATED:START -->

## 相关论文

- [Self-supervised Video Copy Localization with Regional Token Representation](self-supervised_video_copy_localization_with_regional_token_representation.md)
- [Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders](efficient_image_pre-training_with_siamese_cropped_masked_autoencoders.md)
- [Adaptive Multi-head Contrastive Learning](adaptive_multihead_contrastive_learning.md)
- [FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning](flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)
- [SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)

<!-- RELATED:END -->
