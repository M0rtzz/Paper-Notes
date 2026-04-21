---
title: >-
  [论文解读] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation
description: >-
  [CVPR 2025][图像分割][弱监督语义分割] ExCEL 提出利用 patch-text 对齐范式（而非传统 image-text 对齐）挖掘 CLIP 的密集知识用于弱监督语义分割，通过文本语义扩充（TSE）和视觉校准（VC）两个模块增强密集对齐能力，在仅需 3.2GB 显存和 6% 训练时间的条件下，在 PASCAL VOC 和 MS COCO 上大幅超越 SOTA。
tags:
  - CVPR 2025
  - 图像分割
  - 弱监督语义分割
  - CLIP
  - patch-text对齐
  - 类激活图
  - 视觉-语言预训练
---

# Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2503.20826](https://arxiv.org/abs/2503.20826)  
**代码**: https://github.com/zwyang6/ExCEL  
**领域**: 语义分割  
**关键词**: 弱监督语义分割, CLIP, patch-text对齐, 类激活图, 视觉-语言预训练

## 一句话总结

ExCEL 提出利用 patch-text 对齐范式（而非传统 image-text 对齐）挖掘 CLIP 的密集知识用于弱监督语义分割，通过文本语义扩充（TSE）和视觉校准（VC）两个模块增强密集对齐能力，在仅需 3.2GB 显存和 6% 训练时间的条件下，在 PASCAL VOC 和 MS COCO 上大幅超越 SOTA。

## 研究背景与动机

**领域现状**：弱监督语义分割（WSSS）旨在仅使用图像级标签实现像素级预测，通常依赖类激活图（CAM）来提供定位线索。近年来 CLIP 被引入 WSSS，如 CLIP-ES 利用 image-text 对齐生成 GradCAM，WeCLIP 直接用 CLIP 的视觉编码器做分割。

**现有痛点**：现有方法主要利用 CLIP 的全局 image-text 对齐能力，而忽视了 CLIP 在 patch-text 对齐上的密集知识潜力。全局对齐只能告诉图像中有什么物体，但无法精确定位物体的每个像素。

**核心矛盾**：patch-text 对齐面临两个关键挑战：(1) 文本语义稀疏——"a photo of [CLASS]"这样的模板只能表示物体存在，缺乏定位所需的丰富语义；(2) 视觉特征细粒度不足——CLIP 由于 image-text 配对训练的特性，倾向于提取全局表征，q-k 注意力图过于均匀，丢失了细粒度空间信息。

**本文目标** (1) 如何让文本表征更丰富以支持精确的 patch 级匹配；(2) 如何从 CLIP 的视觉特征中挖掘细粒度空间信息。

**切入角度**：作者观察到 CLIP 的中间层的 q/k/v 各自的 intra-correlation（自相关）比跨空间的 q-k attention 保留了更多细粒度信息；同时 LLM 生成的类别描述可以聚类为隐式属性空间来增强文本表征。

**核心 idea**：用 patch-text 余弦相似度替代传统 image-text 对齐来生成 CAM，并通过 LLM 扩充文本语义 + 中间层 intra-correlation 校准视觉特征来解决密集对齐中的两大瓶颈。

## 方法详解

### 整体框架

ExCEL 的输入是一张图像和类别标签，输出是像素级分割伪标签。整体 pipeline 分为四步：(1) TSE 模块扩充文本语义，生成信息丰富的类别文本表征 $T_c$；(2) SVC 模块用 intra-correlation 替代 q-k attention，从 CLIP 冻结特征中提取细粒度视觉特征 $P_s$，与 $T_c$ 计算余弦相似度生成静态 CAM；(3) LVC 模块通过轻量 adapter 学习动态分布偏移，进一步优化视觉特征生成动态 CAM；(4) 动态 CAM 精炼为伪标签监督分割网络训练。

### 关键设计

1. **Text Semantic Enrichment (TSE)**:

    - 功能：将稀疏的类别文本模板扩充为语义丰富的文本表征
    - 核心思路：首先用 GPT-4 为每个类别生成 $n=20$ 条详细描述（包含外观、颜色、形状等属性），用 CLIP 文本编码器编码为知识库 $\mathcal{T}$。然后关键一步：不是直接融合这些描述，而是对所有描述用 K-means 聚类为 $B$ 个隐式属性（如 VOC 用 112 个）。最后，用全局文本嵌入 $t_c$ 在属性空间中检索 TOP-K 最相关属性，加权聚合得到最终表征 $T_c = t_c + \lambda \sum softmax(t_c^T A_c) a_j$
    - 设计动机：显式描述可能覆盖不全且含噪声，聚类后的隐式属性不仅更紧凑，还能跨类别捕获共享知识（如"有翅膀"同时与鸟和飞机相关），补充单一类别描述的缺失信息

2. **Static Visual Calibration (SVC)**:

    - 功能：以无参数方式从 CLIP 中间层提取细粒度视觉特征
    - 核心思路：原始 CLIP 的 q-k attention 产生过于均匀的注意力图，导致不同 token 被同质化。SVC 用 Intra-correlation 替代 q-k attention：不计算 $q^T k$，而是计算 $q^T q$、$k^T k$、$v^T v$（即每个空间内部的自相关），并在最后 $N=5$ 个中间层进行累积。这相当于让每个 patch 与自身所在空间的其他 patch 比较相似度，保留了空间结构信息
    - 设计动机：q-k attention 是为了全局 image-text 对齐而训练的，天然倾向于均匀化 token 以捕获广泛语义；intra-correlation 绕过了这种均匀化效应，直接暴露 patch 之间的空间关联。无需训练即可生成媲美训练方法的 CAM（74.6% mIoU）

3. **Learnable Visual Calibration (LVC)**:

    - 功能：通过轻量 adapter 动态校准冻结视觉特征
    - 核心思路：将 CLIP 1-12 层的冻结特征分别通过独立 MLP 后拼接，再用卷积层生成动态特征 $F_d$。计算 $F_d$ 的自相似度并去均值、缩放后得到动态关系矩阵 $R$，将负值设为 $-\inf$ 以去除无关关系。最后将 $softmax(R)$ 作为分布偏移加到 SVC 的静态注意力图上
    - 设计动机：SVC 的特征是冻结固定的，无法根据具体图像动态调整。LVC 只加了一个分布偏移而不改变 CLIP 预训练权重，既保留了迁移性又增强了密集分割性能

### 损失函数 / 训练策略

训练目标为 $\mathcal{L}_{ExCEL} = \mathcal{L}_{seg} + \gamma \mathcal{L}_{div}$。$\mathcal{L}_{seg}$ 是以动态伪标签为监督的交叉熵损失。$\mathcal{L}_{div}$ 是多样性损失，利用 SVC 生成的静态伪标签的像素亲和性来监督 adapter 特征 $F_d$ 的 token 关系学习：同类 token 对的相关性应最大化，异类 token 对应最小化。$\gamma=0.1$。使用 AdamW 优化器，学习率 1e-4，VOC 训练 30K 迭代，COCO 训练 100K 迭代。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ExCEL | WeCLIP (前SOTA) | 提升 |
|--------|------|-------|-----------------|------|
| VOC val | mIoU | 78.4% | 76.4% | +2.0% |
| VOC test | mIoU | 78.5% | 77.2% | +1.3% |
| COCO val | mIoU | 50.3% | 47.1% | +3.2% |

ExCEL 的训练仅需 3.2GB 显存和前人方法 6% 的训练时间。Training-free 模式（仅 SVC+TSE 不训练）在 CAM seed 上达到 74.6% mIoU，已超越大多数需要训练的方法。

### 消融实验

| 配置 | mIoU | 说明 |
|------|------|------|
| Baseline (CLIP) | 12.1% | 原始 CLIP 直接做分割 |
| + SVC | 72.5% | Intra-correlation 替代 q-k attention |
| + SVC + TSE | 74.7% | 加入文本语义扩充，recall 提升 3.6% |
| + SVC + LVC | 75.1% | 加入可学习视觉校准 |
| ExCEL (全部) | 77.2% | 三个模块协同 |

### 关键发现

- SVC 贡献最大（+60.4% mIoU），证明 intra-correlation 远优于原始 q-k attention 用于密集定位
- 隐式属性聚类（B=112）比直接融合 20 条显式描述好 2.1%，验证了跨类别知识共享的价值
- Intra-correlation 在最后 5 层（而非仅最后 1 层）效果最好：单层 69.7% → 多层 74.6%，说明中间层的细粒度信息需要逐层累积

## 亮点与洞察

- **Intra-correlation 替代 q-k attention 是一个非常优雅的设计**：不需要任何训练参数就能将 CLIP 的 CAM 质量从 11.2% 提升到 74.6%。其核心洞察在于 q-k attention 的均匀化是 CLIP 全局对齐训练的副产品，而不是 patch 级特征本身的问题
- **隐式属性空间的设计思路可迁移**：将类别描述聚类为跨类别共享属性的思路，可以应用到任何需要文本增强引导的视觉任务中（如 open-vocabulary detection）
- **极低训练成本值得关注**：3.2GB 显存 + 6% 训练时间即超越全部 SOTA，说明充分利用预训练模型的密集知识远比暴力训练有效

## 局限与展望

- 依赖 GPT-4 生成类别描述，引入了对外部大模型的依赖；可以探索用开源 LLM 替代
- 聚类属性数量 B 需要针对不同数据集调参（VOC 112，COCO 224），自适应确定 B 值是一个改进方向
- 目前仅在 ViT-B 上验证，未探索更大的 CLIP 模型（如 ViT-L/14）是否能进一步提升

## 相关工作与启发

- **vs CLIP-ES**: CLIP-ES 用 image-text 对齐的梯度生成 GradCAM，本质上仍是全局对齐思路；ExCEL 直接在 patch-text 级别计算相似度，定位更精确
- **vs WeCLIP**: WeCLIP 也是单阶段直接用 CLIP 分割，但没有改 attention 机制也没有增强文本；ExCEL 在同样的 single-stage 设定下高出 2.0%
- **vs MaskCLIP**: MaskCLIP 只用了最后一层的 value 特征，ExCEL 的 intra-correlation 跨多层累积更全面（65.8% vs 74.6%）

## 评分

- 新颖性: ⭐⭐⭐⭐ patch-text 对齐范式和 intra-correlation 都是有价值的新思路，但各模块独立来看并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ VOC + COCO 双数据集，丰富的消融实验，CAM seed 和分割双重评估
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观，动机阐述充分
- 价值: ⭐⭐⭐⭐⭐ 极低训练成本 + SOTA 性能，实用价值很高，对 CLIP 密集知识的挖掘有启发意义

<!-- RELATED:START -->

## 相关论文

- [SSR: Semantic and Spatial Rectification for CLIP-based Weakly Supervised Segmentation](../../AAAI2026/segmentation/ssr_semantic_and_spatial_rectification_for_clip-based_weakly_supervised_segmenta.md)
- [Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation](soft_self-labeling_and_potts_relaxations_for_weakly-supervised_segmentation.md)
- [Exploring Structural Degradation in Dense Representations for Self-supervised Learning](../../NeurIPS2025/segmentation/exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](../../ICCV2025/segmentation/know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)

<!-- RELATED:END -->
