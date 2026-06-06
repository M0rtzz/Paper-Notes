---
title: >-
  [论文解读] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition
description: >-
  [ECCV 2024][全景活动识别] 提出 SPDP-Net，通过时空邻近性建模个体间社会关系，并利用双路径 Transformer (DPATr) 架构在个体-全局和个体-社交两条路径上协同识别多粒度活动，在 JRDB-PAR 数据集上以 46.5% overall F1 大幅刷新 SOTA。
tags:
  - "ECCV 2024"
  - "全景活动识别"
  - "社交群组检测"
  - "时空邻近性"
  - "Transformer"
  - "多粒度活动"
---

# Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition

**会议**: ECCV 2024  
**arXiv**: [2403.14113](https://arxiv.org/abs/2403.14113)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: 全景活动识别, 社交群组检测, 时空邻近性, 双路径Transformer, 多粒度活动

## 一句话总结

提出 SPDP-Net，通过时空邻近性建模个体间社会关系，并利用双路径 Transformer (DPATr) 架构在个体-全局和个体-社交两条路径上协同识别多粒度活动，在 JRDB-PAR 数据集上以 46.5% overall F1 大幅刷新 SOTA。

## 研究背景与动机

全景活动识别 (PAR) 旨在从全景视频中识别三个粒度的人类活动：(i) 个体动作、(ii) 社交群组活动及群组检测、(iii) 全局活动。PAR 面临两大核心挑战：

**空间邻近性不足以判断社交关系**：已有方法仅利用单帧中个体的空间距离来推断社交关系。但如论文 Fig.1 所示，三个人在初始帧中看似属于同一群组，但随时间推移只有两人持续同行。这说明必须引入时间维度的邻近性才能准确判断社交动态。

**层级式建模多粒度活动存在信息瓶颈**：已有方法 [JRDB-PAR, MUP] 采用"个体→社交→全局"的层级结构，但实际上全局和社交活动都需要个体信息，且二者相互影响。单向层级结构无法充分捕获这种双向依赖。

## 方法详解

### 整体框架

SPDP-Net 分为两个阶段：

1. **邻近性关系编码 (Proximity-based Relation Encoding)**：利用时空位置关系增强个体特征表示，并通过特征相似性+时空邻近性进行社交群组检测。
2. **多粒度活动识别 (Multi-Granular Activity Recognition)**：通过 DPATr 的双路径架构协同建模个体、社交群组和全局三个层次的活动。

输入全景视频经 2D CNN backbone (Inception-v3) 提取帧级特征，再通过 RoIAlign 裁剪个体区域特征，经 3D 卷积降维得到 $F^{idv} \in \mathbb{R}^{N_i \times T \times d \times h \times w}$。

### 关键设计

1. **全景位置嵌入 (Panoramic Positional Embedding, PPE)**：传统位置编码只编码裁剪区域内的位置信息，丢失了个体在全景场景中的绝对位置上下文。PPE 从全场景正弦位置嵌入中提取个体区域，保留了个体在全景中的时空位置信息。具体地，对个体特征沿时间、高度、宽度三个维度依次施加带 PPE 的多头自注意力：

$$\bar{F}^{idv} = A^w(A^h(A^t(F^{idv}, e_{pn}), e_{pn}), e_{pn}) + F^{idv}$$

2. **时序广义 IoU (Temporal Generalized IoU, TGIoU)**：将 GIoU 从空间扩展到时间轴，度量两个个体在整个视频序列上的时空邻近性：

$$R_p(i,j) = \text{TGIoU}(P^i, P^j) = \frac{1}{T}\sum_{t=1}^{T}\text{GIoU}(p_t^i, p_t^j)$$

其中 GIoU 同时考虑了 bounding box 的交并比和最小包围框。TGIoU 通过跨帧平均 GIoU 来捕获动态的社交距离变化，相比单帧空间距离或欧氏距离更能反映真实社交关系。

3. **社交关系矩阵 (Social Relation Matrix)**：融合视觉相似性矩阵 $R_s$ 和时空邻近性矩阵 $R_p$：

- $R_s = \text{Softmax}(W_\theta \bar{F}^{idv} (W_\phi \bar{F}^{idv})^\top)$（可学习的视觉特征相似度）
- $R_p$ 由 TGIoU 直接计算（无需学习的物理位置关系）
- 最终关系矩阵 $R = \frac{1}{2}(R_s + R_p)$

社交群组数量通过 MLP 从关系增强特征的均值中回归得到，群组划分通过 K-means 聚类实现。

4. **双路径活动 Transformer (DPATr)**：由 $L$ 层组成，每层包含两条基于 Transformer encoder 的路径：

- **个体-全局路径 (Individual-to-Global)**：将个体特征序列前置一个可学习的 global token，通过自注意力同时编码个体间交互和全局上下文。
- **个体-社交路径 (Individual-to-Social)**：将全局路径输出的个体特征按群组分配 $\mathcal{G}$ 分组，每组前置一个可学习的 social token，捕捉各社交群组的活动动态。

通过多层堆叠，两条路径相互增强，产生协同效应，最终输出个体 ($\tilde{F}^{idv}$)、社交群组 ($F^{sg}$) 和全局 ($F^{glb}$) 活动特征。

### 损失函数 / 训练策略

总损失由六项组成：

$$\mathcal{L} = \mathcal{L}_{idv} + \mathcal{L}_R + \mathcal{L}_{aux} + \lambda_{sg}\mathcal{L}_{sg} + \lambda_{glb}\mathcal{L}_{glb} + \lambda_n\mathcal{L}_n$$

- $\mathcal{L}_{idv}$：个体动作识别的二元交叉熵损失
- $\mathcal{L}_R$：社交关系矩阵监督损失
- $\mathcal{L}_{aux}$：基于 $\bar{F}^{idv}$ 的辅助个体动作损失
- $\mathcal{L}_{sg}, \mathcal{L}_{glb}$：社交群组和全局活动的二元交叉熵损失
- $\mathcal{L}_n$：群组数量估计的 L2 损失
- 损失权重比 $\lambda_{sg}:\lambda_{glb}:\lambda_n = 3:2:5$

训练使用 Adam 优化器，60 epochs，前 15 epochs 线性 warmup，之后固定学习率 $4\times10^{-5}$，weight decay $10^{-2}$。

## 实验关键数据

### 主实验

在 JRDB-PAR 数据集上与 SOTA 方法比较：

| 方法 | $\mathcal{F}_i$ (个体) | $\mathcal{F}_p$ (社交) | $\mathcal{F}_g$ (全局) | $\mathcal{F}_a$ (总体) |
|------|---------|---------|---------|---------|
| ARG | 33.2 | 8.2 | 50.7 | 30.7 |
| JRDB-PAR | 43.4 | 24.8 | 38.8 | 35.6 |
| MUP | 47.7 | 25.1 | 51.8 | 41.5 |
| **SPDP-Net** | **51.8** | **34.2** | **53.5** | **46.5** |

社交群组检测性能对比：

| 方法 | IoU@0.5 | IoU@AUC | Mat.IoU |
|------|---------|---------|---------|
| JRDB-PAR | 37.9 | 25.2 | 22.3 |
| MUP | 46.9 | 34.2 | 28.5 |
| **SPDP-Net** | **56.4** | **42.5** | **34.3** |

### 消融实验

| 配置 | $\mathcal{F}_a$ | IoU@0.5 | 说明 |
|------|---------|---------|------|
| 仅时间注意力 (无PPE) | 42.5 | - | 基线 |
| 空间+时间注意力 + PPE | **46.5** | - | PPE 带来 +3.1% |
| 仅 $R_s$ | - | 37.6 | 视觉相似性 |
| 仅 $R_p$ | - | 55.8 | 时空邻近性，贡献更大 |
| $R_s + R_p$ | - | **56.4** | 互补 |
| GIoU (空间) | - | 48.7 | 单帧空间度量 |
| TGIoU (时空) | - | **56.4** | 时间扩展带来 +7.7% |

### 关键发现

- 时空邻近性 $R_p$ 对社交群组检测贡献远大于视觉相似性 $R_s$（IoU@0.5: 55.8 vs 37.6）
- TGIoU 比单帧 GIoU 提升 7.7% IoU@0.5，证明时间维度对社交关系判断至关重要
- DPATr 的双路径架构在所有粒度上优于并行、层级和逆层级结构，验证了多粒度活动间相互影响的假设
- 使用 ground-truth 群组检测时，$\mathcal{F}_p$ 可进一步提升 19.3%，说明群组检测的准确性是社交活动识别的瓶颈

## 亮点与洞察

1. **TGIoU 的设计简洁而有效**：仅需对已有 GIoU 做时间维度平均，无需额外参数，却能显著提升社交群组检测性能
2. **双路径架构的设计动机清晰**：实验验证了"全局上下文对社交活动有帮助"和"个体信息对两者都重要"的直觉，DPATr 通过双路径优雅地实现了这种双向信息流
3. **全景位置嵌入的必要性**：解决了裁剪个体区域后丢失全局位置信息的问题，这是 PAR 任务特有的挑战

## 局限与展望

1. 使用冻结的 Inception-v3 backbone，如换用更强的预训练模型（如 ViT）可能进一步提升
2. 群组数量估计仍有提升空间（使用 GT 群组数后性能显著提升），可考虑更精确的群组数量预测方法
3. K-means 聚类是非可微的，无法端到端训练群组分配；可探索可微聚类方案
4. 仅在 JRDB-PAR 一个数据集上验证，泛化性有待更多数据集检验

## 相关工作与启发

- **JRDB-PAR [Han et al.]**：首次提出PAR任务和层级GCN方案，本文的主要对比基线
- **MUP [Cao et al.]**：多粒度统一感知框架，采用层级聚合
- **Groupformer [Li et al.]**：时空Transformer用于群组活动识别，启发了DPATr的设计
- **GIoU [Rezatofighi et al.]**：本文将其扩展到时间维度，简单而有效

## 评分

- 新颖性: ⭐⭐⭐⭐ — TGIoU 和双路径架构都是合理且新颖的设计，动机清晰
- 实验充分度: ⭐⭐⭐⭐ — 消融实验非常全面，覆盖了每个模块的贡献；但仅在单一数据集上验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，方法描述和实验分析都很到位
- 综合价值: ⭐⭐⭐⭐ — 在 PAR 这一新兴任务上取得显著突破，对社交场景理解有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] Operational Open-Set Recognition and PostMax Refinement](operational_open-set_recognition_and_postmax_refinement.md)
- [\[ACL 2025\] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior](../../ACL2025/others/a_spatio-temporal_point_process_for_fine-grained_modeling_of_reading_behavior.md)
- [\[ECCV 2024\] Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)
- [\[ECCV 2024\] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning](stsp_spatial-temporal_subspace_projection_for_video_class-incremental_learning.md)

</div>

<!-- RELATED:END -->
