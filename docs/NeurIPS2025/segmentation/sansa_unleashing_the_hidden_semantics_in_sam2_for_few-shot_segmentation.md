---
title: >-
  [论文解读] SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation
description: >-
  [NeurIPS 2025][图像分割][图像分割] SANSA 发现 SAM2 虽然以类别无关方式预训练，但其特征中隐含了丰富的语义结构；通过在冻结的 SAM2 Image Encoder 最后两层插入轻量 AdaptFormer 适配器，将 Memory Attention 机制从视觉相似性匹配重定向为语义相似性匹配，以统一架构实现了 few-shot 分割的 SOTA，同时比竞争方法快 3 倍以上、参数量小 4-5 倍。
tags:
  - NeurIPS 2025
  - 图像分割
  - SAM2
  - semantic alignment
  - 特征适配
  - 注意力机制
---

# SANSA: Unleashing the Hidden Semantics in SAM2 for Few-Shot Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2505.21795](https://arxiv.org/abs/2505.21795)  
**代码**: [GitHub](https://github.com/ClaudiaCuttano/SANSA)  
**领域**: 图像分割  
**关键词**: Few-Shot Segmentation, SAM2, semantic alignment, 特征适配, Memory Attention

## 一句话总结

SANSA 发现 SAM2 虽然以类别无关方式预训练，但其特征中隐含了丰富的语义结构；通过在冻结的 SAM2 Image Encoder 最后两层插入轻量 AdaptFormer 适配器，将 Memory Attention 机制从视觉相似性匹配重定向为语义相似性匹配，以统一架构实现了 few-shot 分割的 SOTA，同时比竞争方法快 3 倍以上、参数量小 4-5 倍。

## 研究背景与动机

Few-shot segmentation (FSS) 旨在仅凭少量标注样本分割未见类别。现有方法通常将 FSS 解耦为两阶段流水线：先用 DINOv2 做特征匹配找到语义对应关系，再用 SAM 生成高质量掩码。这种模块化设计虽然有效，但引入了额外的计算开销和多模型协调的复杂性。

作者观察到 SAM2 的 "prompt-and-propagate" 机制天然地统一了 FSS 所需的两大能力——**密集特征匹配**（通过 Memory Attention 跨帧建立对应关系）和**高质量掩码生成**（通过 Mask Decoder）。核心问题在于：SAM2 能否从视觉相似性追踪扩展到基于共享语义概念的"语义追踪"？

作者通过实验发现了一个关键现象：在语义差异较小的数据集（如肺部 X 光、皮肤病变）上，冻结的 SAM2 表现与 SOTA 方法相当甚至更好；但在语义差异大的数据集（如 COCO、LVIS）上，性能急剧下降。直觉上的结论是 SAM2 没有学到语义表示，但作者挑战了这一解读。他们注意到 SAM2 的预训练过程——跨帧匹配物体实例——与自监督学习框架有相似性，后者被证明能通过视图不变性诱导语义理解。因此作者假设：**SAM2 确实编码了语义信息，但这些信息与面向追踪优化的实例级特征纠缠在一起**。如果假设成立，则可以通过轻量瓶颈变换解开这种结构，并且在基类上学到的语义映射可以泛化到未见类别。

## 方法详解

### 整体框架

SANSA 将 FSS 重新解释为"在伪视频中追踪语义概念"。给定 $K$ 个参考图像-掩码对和一个目标图像，将它们拼接成伪视频序列：

$$\mathcal{M} = [x_r^k, a_r^k]_{k=1}^K \cup [x_t, \varnothing]$$

利用 SAM2 的 streaming pipeline 顺序处理参考帧及其标注，通过 Memory Attention 将掩码从参考帧传播到未标注的目标帧，实现基于语义相似性的分割。

### 关键设计

1. **从物体追踪到语义追踪的重定向**：SAM2 的功能被概念性地分解为两部分——(a) 密集特征匹配：Memory Encoder 将参考掩码与帧特征融合为记忆表示 $\mathcal{I}_r^k = \mathcal{F}_r^k + \text{conv\_down}(\hat{y}_r^k)$，存入 Memory Bank，然后目标帧特征通过 Memory Attention 做交叉注意力建立密集对应：$\mathcal{F}_{t,\text{match}} = \text{Attention}(Q(\mathcal{F}_t) K([\mathcal{I}_r^0,...,\mathcal{I}_r^k])^T) V([\mathcal{I}_r^0,...,\mathcal{I}_r^k])$；(b) 高质量掩码生成：Mask Decoder 将粗糙的特征匹配结果精炼为分割输出。设计动机：参考帧编码进 Memory Bank 时不经过 Memory Attention，避免了交叉引用，保证了目标预测对参考图像顺序的不变性。

2. **SAM2 特征适配 (AdaptFormer)**：在冻结的 SAM2 Image Encoder 的**最后两层**插入 AdaptFormer 模块。给定下投影矩阵 $\mathbf{W}_{down} \in \mathcal{R}^{d,\tilde{d}}$ 和上投影矩阵 $\mathbf{W}_{up} \in \mathcal{R}^{\tilde{d},d}$，AdaptFormer 以 token-wise 方式操作：$\mathcal{A}(x) = \sigma(x \cdot \mathbf{W}_{down}) \cdot \mathbf{W}_{up}$，其中 $\sigma$ 为 ReLU，$\tilde{d} < d$ 为瓶颈维度。适配后的特征以残差方式加入 Transformer 块：$x' = \text{MLP}(x_\text{self}) + x_\text{self} + \mathcal{A}(x_\text{self})$。设计动机：只训练投影矩阵（~10M参数），保持 SAM2 冻结以保留其预训练先验。选择最后两层是因为这些层编码了更高层的语义表示。实验证明过大容量的适配器（如更大瓶颈或 MONA）反而损害泛化性。

3. **训练目标 — 伪参考帧自训练**：采用 episodic training 范式，但创新性地反转标准 $k$-shot 设置：模型接收单个标注参考图像，负责将概念传播到多个未标注目标图像。训练片段为 $\mathcal{M}_{train} = [x_r, a_r] \cup [x_t^j, \varnothing]_{j=1}^J$。关键设计是将每个目标帧的预测表示 $\mathcal{I}_t^j$ 也编码进 Memory Bank，使中间帧变成后续帧的伪参考。这一设计迫使模型从低层特征中解开语义信息，避免对单个图像对的过拟合。

### 损失函数 / 训练策略

- 使用 Binary Cross-Entropy loss 和 Dice loss 监督预测的 masklet $\{\hat{y}_t^j\}_{j=1}^J$
- 采用 AdamW 优化器，学习率 $10^{-4}$
- 严格 FSS 设置训练 5 epochs，泛化设置训练 20 epochs
- 训练时 $k=1$（单参考），序列长度 $J=3$，同一模型评估 1-shot 和 5-shot

## 实验关键数据

### 主实验

| 数据集 | 指标 (1-shot mIoU) | SANSA | 之前SOTA | 提升 |
|--------|-------------------|-------|---------|------|
| LVIS-92i | 1-shot mIoU | **48.8** | 40.5 (SegIC) | +8.3 |
| COCO-20i | 1-shot mIoU | **60.2** | 53.9 (VRP-SAM) | +6.3 |
| FSS-1000 | 1-shot mIoU | **91.4** | 90.2 (DiffewS) | +1.2 |
| LVIS-92i | 5-shot mIoU | **53.9** | 43.7 (DiffewS) | +10.2 |
| COCO-20i | 5-shot mIoU | **64.3** | 60.7 (DiffewS) | +3.6 |

### 消融实验

| 配置 | COCO-20i mIoU | 说明 |
|------|--------------|------|
| Frozen SAM2 | 32.2 | 基线，无适配 |
| Full Fine-tuning (224M) | 51.6 | 全参数微调 |
| QKV Fine-tuning (50M) | 55.3 | 仅微调 QKV 投影 |
| LoRA | 58.0 | 适配方法 |
| AdaptFormer (0.3× bottleneck) | **60.2** | SANSA (10M参数) |
| MONA (复杂适配) | 56.9 | 过大容量损害泛化 |
| 适配 All stages (0-3) | 59.4 | 全层适配 |
| 适配 Late stages (2-3) | **60.2** | 最后两层足够 |

### 关键发现

- **SANSA 仅用 234M 参数**，比 GF-SAM (945M) 快 3 倍以上，且在 LVIS-92i 上高出 13.6%
- 在可提示 FSS 中，点提示到掩码提示的性能下降仅 6.8%（VRP-SAM 为 15.5%）
- 在泛化实验 (In-Context) 中，即使不在部件级数据上训练，SANSA 也展现出跨任务泛化能力，在 Pascal-Part 上超越 DiffewS 7.5%
- PCA 可视化清晰显示适配后特征形成了按语义类别聚类的结构，且这种语义组织能泛化到未见类别

## 亮点与洞察

- **核心洞察**：SAM2 的类别无关预训练实际上隐含了丰富的语义结构，类似于自监督学习中通过视图不变性诱导的语义理解。这一发现挑战了 SAM2 "不具备语义理解能力" 的普遍看法
- **极简设计哲学**：仅在最后两层插入最简单的 AdaptFormer，~10M 可训练参数就实现 SOTA，证明了"约束更强的适配反而更好"的原则
- **统一架构优势**：不需要 DINOv2+SAM 的双模型流水线，单一 SAM2 架构同时完成特征匹配和掩码生成

## 局限与展望

- 在 5-shot COCO-20i 上相比 GF-SAM 仍有 -2.5% 的差距
- 基于 episodic training，可能受限于训练类别的多样性
- 适配器的泛化能力依赖于基类与目标类之间的语义关联程度

## 相关工作与启发

- 与 SegIC、DiffewS 等单模型方法相比，SANSA 证明了 SAM2 的 Memory Attention 是统一特征匹配和掩码生成的理想机制
- 为基础模型的"隐藏能力挖掘"提供了新范式：通过简单适配暴露预训练特征中的潜在结构
- 伪参考帧自训练策略可借鉴到其他视频理解任务中

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次揭示并利用 SAM2 的隐含语义结构，视角独特
- **实验充分度**: ⭐⭐⭐⭐⭐ LVIS/COCO/FSS-1000 全面评测，消融详尽，PCA 可视化有力
- **写作质量**: ⭐⭐⭐⭐⭐ 问题引导式叙事（三个研究问题），逻辑清晰
- **价值**: ⭐⭐⭐⭐⭐ 极简设计+SOTA性能+高效推理，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)
- [Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](../../CVPR2025/segmentation/dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [MOVE: Motion-Guided Few-Shot Video Object Segmentation](../../ICCV2025/segmentation/move_motion-guided_few-shot_video_object_segmentation.md)
- [Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2](vanish_into_thin_air_cross-prompt_universal_adversarial_attacks_for_sam2.md)
- [Mechanistic Interpretability of RNNs Emulating Hidden Markov Models](mechanistic_interpretability_of_rnns_emulating_hidden_markov_models.md)

<!-- RELATED:END -->
