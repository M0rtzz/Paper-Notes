---
title: >-
  [论文解读] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition
description: >-
  [CVPR 2025][视频理解][跨域少样本动作识别] 本文提出 TAMT，一个解耦的"预训练-微调"范式用于跨域少样本动作识别（CDFSAR），通过时序感知适配器（TAA）高效重校准冻结模型的中间特征，并利用全局时序矩调优（GTMT）捕获长短期时序协方差来生成强表示…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "跨域少样本动作识别"
  - "时序感知适配器"
  - "模型微调"
  - "协方差特征"
  - "解耦训练"
---

# TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition

**会议**: CVPR 2025  
**arXiv**: [2411.19041](https://arxiv.org/abs/2411.19041)  
**代码**: [https://github.com/TJU-YDragonW/TAMT](https://github.com/TJU-YDragonW/TAMT)  
**领域**: 视频理解 / 少样本学习  
**关键词**: 跨域少样本动作识别, 时序感知适配器, 模型微调, 协方差特征, 解耦训练

## 一句话总结

本文提出 TAMT，一个解耦的"预训练-微调"范式用于跨域少样本动作识别（CDFSAR），通过时序感知适配器（TAA）高效重校准冻结模型的中间特征，并利用全局时序矩调优（GTMT）捕获长短期时序协方差来生成强表示，在多个跨域场景中以 5 倍低的训练成本超越现有方法 13%-31%。

## 研究背景与动机

**领域现状**：少样本动作识别（FSAR）旨在利用少量标注样本进行视频动作分类。跨域 FSAR（CDFSAR）进一步引入了源域和目标域之间的域差异问题，需要在标注丰富的源域上学习知识并迁移到标注稀缺的目标域。

**现有痛点**：现有 CDFSAR 方法（如 SEEN、CDFSL-V）采用联合训练范式，将源数据和目标数据一起训练以缓解域差异。但存在两个关键问题：(1) 当有一个源域和多个目标域时，联合训练需要对每个目标域都重新训练一次模型，计算开销随目标域数量线性增长；(2) 推理阶段使用简单的最近邻分类器或微调分类器，未能充分挖掘预训练模型的潜力。

**核心矛盾**：联合训练的计算成本与多目标域适配需求之间的矛盾，以及预训练模型表征能力未被充分利用的问题。

**本文目标**：(1) 避免多目标域重复训练；(2) 高效地将预训练模型适配到目标域；(3) 生成更强的视频表征用于少样本匹配。

**切入角度**：采用解耦范式——源域预训练只做一次，目标域微调轻量快速。并设计时序感知的适配模块，因为视频理解的核心在于时序建模，而现有适配器主要关注空间信息。

**核心 idea**：用解耦的预训练-微调范式替代联合训练，通过轻量级时序感知适配器和基于一二阶矩的时序协方差特征表示，以极低参数量实现高效的跨域少样本动作识别。

## 方法详解

### 整体框架

TAMT 分为两个阶段：(1) 源域预训练——先用自监督重建（SSL）训练 VideoMAE 编码器学习通用时空结构，再用有监督分类（SL）增强语义判别力；(2) 目标域微调——冻结预训练编码器，通过 HTTN（层级时序调优网络）进行少样本适配。HTTN 包含嵌入 Transformer 后 L 层的局部 TAA 适配器和末尾的全局 GTMT 模块。最终用度量学习（欧氏距离）比较 query 和 support 的表征进行分类。

### 关键设计

1. **时序感知适配器 (TAA)**:

    - 功能：以极少可学习参数重校准冻结模型的中间视频特征
    - 核心思路：对每层 Transformer 输出的特征 $\mathbf{F} \in \mathbb{R}^{T \times M \times C}$，TAA 生成时序感知的缩放因子 $\gamma$ 和偏移因子 $\beta$，执行 $\mathbf{F'} = \gamma \odot \mathbf{F} \oplus \beta$。$\gamma$ 和 $\beta$ 通过对特征做全局平均池化后，经两层时序卷积（核大小 $k_t=3$）和瓶颈降维（$C \to C/\rho \to C$，$\rho=4$）生成。$\gamma$ 和 $\beta$ 共享降维层权重以进一步减少参数
    - 设计动机：传统全量微调在少样本场景下易过拟合且计算量大。与 NLP/图像分类中的空间适配器不同，TAA 通过时序卷积显式捕获帧间动态信息，更适合视频任务。仅需 2.8M 参数（vs FFT 的 29.9M）和 1.9GB 显存（vs FFT 的 17.5GB）

2. **全局时序矩调优 (GTMT) + 高效长短期时序协方差 (ELSTC)**:

    - 功能：利用特征分布的一二阶矩生成更强大的视频全局表征
    - 核心思路：最终表征 $\mathbf{Z} = \mathcal{H}(\mathbf{M}_2) \oplus \mathbf{M}_1$，其中一阶矩 $\mathbf{M}_1$ 是全局平均池化，二阶矩 $\mathbf{M}_2$ 通过 ELSTC 计算。ELSTC 将时序维度分为 G 组，每组内计算帧间协方差矩阵 $\mathbf{R}_{t,t'}$，捕获从短期（同帧外观）到长期（跨帧运动）的各尺度时序相关性。最后通过两层卷积聚合所有组的协方差，经线性投影对齐维度后与一阶矩相加
    - 设计动机：传统方法仅用全局平均池化（一阶矩）作为表征，丢失了丰富的二阶统计信息。协方差矩阵能描述特征的分布形状，包含帧间运动模式的刻画。分组策略将计算量降低 G 倍，使得二阶矩计算可行

3. **两阶段预训练策略**:

    - 功能：在源域上学习兼具泛化性和判别力的特征表示
    - 核心思路：先用 VideoMAE 的掩码重建目标（SSL）训练编码器 400 个 epoch，学习通用时空结构；再用交叉熵分类损失（SL）训练 140 个 epoch，增强语义判别力
    - 设计动机：纯 SSL 捕获基础特征但缺乏高层语义；纯 SL 在少样本跨域场景泛化性不足。两阶段策略实现泛化性和表征能力的平衡

### 损失函数 / 训练策略

- 预训练阶段：SSL 用均方误差（MSE）损失，SL 用交叉熵（CE）损失
- 微调阶段：CE 损失，欧氏距离作为度量函数
- 优化器：SGD + 余弦衰减学习率，微调仅 40 个 epoch
- 推理：5-way 1/5-shot，平均 10000 个 episode

## 实验关键数据

### 主实验

**K-400 → 五个目标域 (5-way 5-shot 准确率%)**:

| 方法 | HMDB | SSV2 | Diving | UCF | RareAct | 平均 |
|------|------|------|--------|-----|---------|------|
| CDFSL-V | 53.23 | 49.92 | 17.84 | 65.42 | 49.80 | 47.24 |
| SEEN | - | - | - | - | - | - |
| **TAMT (ours)** | **74.14** | **59.18** | **45.18** | **95.92** | **67.44** | **68.37** |
| 提升 | +20.91 | +9.26 | +27.34 | +30.50 | +17.64 | **+21.13** |

### 消融实验

**预训练策略 + 微调方式 (K-400 源域, 5-way 5-shot)**:

| 预训练 | 微调方式 | SSV2 | Diving | UCF | 平均 |
|--------|---------|------|--------|-----|------|
| SSL only | Frozen | 29.27 | 22.10 | 55.30 | 35.56 |
| SL only | TAMT | 45.15 | 37.96 | 89.73 | 56.48 |
| SSL+SL | FFT | 55.99 | 42.85 | 94.95 | 64.30 |
| **SSL+SL** | **TAMT** | **59.18** | **45.18** | **95.92** | **66.76** |

**效率对比**:

| 指标 | FFT | TAMT |
|------|-----|------|
| 显存 | 17.5GB | 1.9GB |
| 参数量 | 29.9M | 2.8M |
| 训练时间 | 10.6h | 7.3h |

### 关键发现

- TAMT 以仅 9.4% 的可训练参数（2.8M vs 29.9M）超越全量微调 2.46% 的平均准确率，说明在少样本场景下参数高效微调优于全量微调
- SSL+SL 两阶段预训练远优于单独任一阶段，SSL 提供泛化性（+10.28%），SL 提供判别力
- TAMT 训练计算成本约为 CDFSL-V 的 1/5（19 vs 88 GPU days），但性能提升 21%
- 在 Diving48 数据集上提升最大（+27.3%），说明时序建模对细粒度动作识别尤其重要

## 亮点与洞察

- **解耦训练范式的效率优势**：源域只需预训练一次，面对多个目标域时只需轻量微调，实际部署成本大大降低。这个思路可以迁移到其他跨域少样本任务
- **时序感知的参数高效微调**：不同于图像领域的 adapter 只关注空间信息，TAA 通过时序卷积显式建模帧间关系，是对视频 PEFT 的有意义探索
- **二阶统计量的力量**：ELSTC 用协方差矩阵捕获帧间相关性，相比仅用均值（一阶矩）能编码更丰富的运动模式，且分组策略保证了计算效率

## 局限与展望

- 仅在 ViT-S/B 骨干上验证，未在更大模型（如 ViT-L）上测试扩展性
- ELSTC 的分组数 G 和降维比 τ 作为超参数需要手动调整
- 未探索与文本/语言先验（如 CLIP）结合的可能性
- 在 RareAct 等罕见动作数据集上的提升相对较小，可能是因为这些动作确实与源域分布差异过大
- 可以考虑在 GTMT 中引入注意力机制动态选择重要的时序协方差分量

## 相关工作与启发

- **vs SEEN**: SEEN 用联合训练 + 对比学习缓解域差异，但需要对每个目标域重训练。TAMT 解耦范式更高效，且微调时更充分利用预训练模型
- **vs CDFSL-V**: CDFSL-V 用两阶段联合训练 + 课程学习，训练成本高且性能低于 TAMT。TAMT 在 K-100 源域上平均提升 31.15%
- **vs 图像 Adapter (AdaptFormer等)**: 这些方法只建模空间信息，TAMT 的 TAA 通过时序卷积额外捕获帧间动态，更适合视频任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 解耦范式和时序感知适配器的结合是合理且有效的创新
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个源域 ×5 个目标域的全面评估，消融详尽
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，动机论述充分
- 价值: ⭐⭐⭐⭐ 为 CDFSAR 提供了简单有效的 baseline，性能和效率优势明显

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)
- [\[ICCV 2025\] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition](../../ICCV2025/video_understanding/trokens_semantic-aware_relational_trajectory_tokens_for_few-shot_action_recognit.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](../../ECCV2024/video_understanding/efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[AAAI 2026\] Task-Specific Distance Correlation Matching for Few-Shot Action Recognition](../../AAAI2026/video_understanding/task-specific_distance_correlation_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
