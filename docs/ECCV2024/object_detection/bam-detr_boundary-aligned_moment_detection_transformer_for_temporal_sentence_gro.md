---
title: >-
  [论文解读] BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos
description: >-
  [ECCV2024][目标检测][Temporal Sentence Grounding] 提出边界对齐的时刻检测 Transformer（BAM-DETR），用 anchor-boundary 三元组 $(p, d_s, d_e)$ 替代传统的 center-length 二元组 $(c, l)$ 来建模时刻，配合双路径解码器和基于质量的排序机制，有效解决了中心模糊导致的定位不精确问题。
tags:
  - ECCV2024
  - 目标检测
  - Temporal Sentence Grounding
  - Transformer
  - Boundary Alignment
  - 视频理解
  - Moment Retrieval
---

# BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos

**会议**: ECCV2024  
**arXiv**: [2312.00083](https://arxiv.org/abs/2312.00083)  
**代码**: [GitHub](https://github.com/Pilhyeon/BAM-DETR)  
**领域**: object_detection  
**关键词**: Temporal Sentence Grounding, Detection Transformer, Boundary Alignment, video understanding, Moment Retrieval

## 一句话总结

提出边界对齐的时刻检测 Transformer（BAM-DETR），用 anchor-boundary 三元组 $(p, d_s, d_e)$ 替代传统的 center-length 二元组 $(c, l)$ 来建模时刻，配合双路径解码器和基于质量的排序机制，有效解决了中心模糊导致的定位不精确问题。

## 背景与动机

时序句子定位（Temporal Sentence Grounding, TSG）旨在给定自然语言描述的情况下，从未裁剪视频中定位对应的时刻片段。近年来，受 DETR 启发的基于查询的方法（如 Moment-DETR、QD-DETR）通过解码学习查询来生成时刻预测，取得了显著进展。

然而现有方法存在两个核心问题：

1. **中心模糊性问题**：传统方法用 $(c, l)$（中心、长度）表示时刻，隐含假设边界到中心等距。但实际中，时刻中心帧往往与句子语义不相关（如视频中中心位置可能是低显著性的无关场景），中心预测不准确会直接导致整个片段偏移。
2. **评分排序问题**：传统方法用分类分数（句子-片段匹配分数）对候选进行排序。但一个不完整的片段也可能与句子有较高匹配度，导致不完整预测被错误地排在前面。

作者通过诊断实验验证了这些问题：在 QVHighlights 上，Moment-DETR 和 QD-DETR 当中心预测偏移增大（误差从 [0, 0.1) 到 [0.4, 0.5)）时，IoU 从 83-87% 急剧下降到 35-36%；而 BAM-DETR 在所有误差组中均保持 ~77% 的稳定 IoU。

## 核心问题

如何在时序句子定位中摆脱对精确中心预测的依赖，实现更鲁棒的边界对齐？如何让模型在排序候选时优先选择定位质量高的预测而非仅语义匹配度高的预测？

## 方法详解

### 1. 边界导向的时刻建模

核心改变：用三元组 $(p, d_s, d_e)$ 替代 $(c, l)$，其中：

- $p$ 为锚点（anchor point），可以是目标时刻内任意显著点，不必是精确中心
- $d_s$ 为锚点到起始边界的距离
- $d_e$ 为锚点到结束边界的距离

时刻表示为 $\hat{\varphi} = (p - d_s, p + d_e)$。这种非对称设计使模型只需找到目标内的任意显著锚点即可，大大降低了定位难度。

### 2. 双路径解码器（Dual-pathway Decoder）

基于直觉：锚点更新需要全局扫描寻找潜在位置，而边界更新需要在边界附近关注细粒度特征。因此设计两条并行路径：

**锚点更新路径**：

- 自注意力层：锚点查询 $\mathbf{C}_p$ 之间交互，消除冗余，利用基于当前预测的位置编码
- 全局交叉注意力层：在多模态记忆特征 $\hat{\mathcal{V}}$ 上进行全局注意力聚合，采用 Q-K 拼接而非求和以分离特征与位置编码的角色
- FFN + sigmoid 细化锚点位置

**边界更新路径**：

- 局部性增强记忆：通过 1D 卷积层生成边界敏感特征 $\hat{\mathcal{V}}_s$、$\hat{\mathcal{V}}_e$，并施加正则化使其在起止边界附近高激活，与原始特征拼接形成局部性增强记忆
- 边界聚焦注意力（Boundary-Focused Attention）：采用可变形注意力（Deformable Attention），以当前边界位置为原点，预测偏移和权重采样 $K=3$ 个邻域点进行局部特征聚合
- FFN + sigmoid 细化边界距离

两条路径共经过 $L_D = 2$ 层迭代解码。

### 3. 基于质量的评分排序

不再使用分类分数，而是直接预测每个候选的定位质量（与真实时刻的最大 IoU）：

$$\mathbf{q} = \sigma(\text{MLP}([\mathbf{C}_p \| \mathbf{C}_s \| \mathbf{C}_e]))$$

质量损失为预测质量分数与实际 IoU 的 L1 距离。匹配代价函数也去掉了分类项，仅包含 L1 距离和 GIoU 损失，实现定位导向的匹配。

### 4. 总体训练目标

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{loc}} + \lambda_{\text{qual}}\mathcal{L}_{\text{qual}} + \lambda_{\text{sal}}\mathcal{L}_{\text{sal}} + \lambda_{\text{regul}}\mathcal{L}_{\text{regul}}$$

其中 $\mathcal{L}_{\text{sal}}$ 包含 margin 排序损失、对比损失和负关系损失，$\mathcal{L}_{\text{regul}}$ 为边界敏感特征的正则化损失。

## 实验关键数据

### QVHighlights 测试集（无预训练）

| 方法 | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg. |
|------|--------|--------|---------|----------|----------|
| Moment-DETR | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| QD-DETR | 62.40 | 44.98 | 62.52 | 39.88 | 39.86 |
| **BAM-DETR** | **62.71** | **48.64** | **64.57** | **46.33** | **45.36** |
| **BAM-DETR†** | **64.07** | **48.12** | **65.61** | **47.51** | **46.91** |

### Charades-STA 测试集

| 方法 | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|------|--------|--------|--------|------|
| UniVTG | 70.81 | 58.01 | 35.65 | 50.10 |
| **BAM-DETR** | **72.93** | **59.95** | **39.38** | **52.33** |

### TACoS 测试集

| 方法 | R1@0.3 | R1@0.5 | R1@0.7 | mIoU |
|------|--------|--------|--------|------|
| QD-DETR | - | 36.77 | 21.07 | 35.76 |
| **BAM-DETR** | **56.69** | **41.54** | **26.77** | **39.31** |

### 消融实验（QVHighlights val）

| 组件 | R1@0.5 | R1@0.7 | mAP Avg. |
|------|--------|--------|----------|
| Baseline | 62.39 | 47.87 | 41.75 |
| + 边界导向建模 | 63.42 | 49.23 | 42.42 |
| + 双路径解码器 | 63.61 | 50.26 | 44.16 |
| + 质量评分 | **65.10** | **51.61** | **47.61** |

三个组件每个都有明确贡献，质量评分提升最大（+3.45 mAP Avg.）。

## 亮点

1. **巧妙的问题发现与建模**：通过诊断实验精确定位了"中心模糊"这一被忽视的问题，提出的 anchor-boundary 三元组建模是简洁而有效的解法
2. **聚焦≠全局**：双路径设计把全局搜索（找锚点）和局部精化（调边界）分开处理，符合直觉且计算开销增加很小
3. **质量评分替代分类评分**：从根本上改变了排序逻辑，使模型在推理时优先输出高质量定位结果
4. **鲁棒性强**：在反偏置 Charades-STA 上表现优异，特别是在时刻长度偏置设置下优势显著（R1@0.7: 40.74 vs QD-DETR 32.87），说明边界导向建模天然抗偏置
5. **全面的实验**：三个数据集全面 SOTA，消融实验清晰展示每个组件贡献

## 局限与展望

1. **仅限 1D 时序**：当前设计针对时序句子定位，推广到 2D 空间（如视频中空间定位）需要额外适配
2. **锚点质量依赖**：虽然降低了对中心的依赖，但锚点仍需落在目标时刻内部，对极短时刻或内容高度相似的场景可能仍有挑战
3. **可变形注意力的采样点数 K 固定为 3**：对不同长度/复杂度的时刻可能不是最优，自适应 K 值可能进一步提升
4. **未探索与大规模预训练模型（如 InternVideo）的结合**：可能通过更强的视觉特征进一步提升

## 与相关工作的对比

- **vs Moment-DETR / QD-DETR**：同属 DETR 系列查询式方法，但改变了时刻表示方式（三元组 vs 二元组）和解码器结构（双路径 vs 单一路径），在严格 IoU 阈值下提升尤为显著
- **vs UniVTG**：UniVTG 走统一框架路线，需要 4.2M 数据预训练才达到 43.63 mAP；BAM-DETR 无预训练即达 45.36，用 236K 数据预训练达 46.67
- **vs 密集回归方法（如 SSTG）**：密集回归从每帧作为锚点预测边界，但锚点位置固定；BAM-DETR 使用动态锚点逐步调整，用少量预测即可实现精准定位
- **vs 2D 目标检测 DETR 变体（DAB-DETR、DINO）**：时序时刻与空间物体有不同挑战（中心模糊、边界不清晰），BAM-DETR 的边界导向建模是对 DETR 范式在时序领域的定制化改进

## 启发与关联

1. **边界导向思想的通用性**：从 center-length 到 anchor-boundary 的思路可推广到其他回归任务（如时序动作检测、音频事件检测），凡是"中心难以精确定义"的场景都可能受益
2. **质量评分的泛化**：用 IoU 预测替代分类分数做排序是近年检测领域的趋势（如 IoU-Net、FCOS），本文将其引入时序定位并做了充分验证
3. **双路径解码的设计哲学**：全局与局部的解耦适用于很多"先粗定位再精调"的场景，如 event detection 中粗定位事件区间再精调起止

## 评分
- 新颖性: ⭐⭐⭐⭐ — 边界导向建模和双路径解码设计新颖，但各单元技术相对成熟
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集 + 鲁棒性测试 + 详尽消融 + 诊断实验
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，图表直观，Table 1 的诊断分析尤为出色
- 价值: ⭐⭐⭐⭐ — 提供了时序定位中 DETR 范式的新思路，方法简洁且效果显著

<!-- RELATED:START -->

## 相关论文

- [Sim-DETR: Unlock DETR for Temporal Sentence Grounding](../../ICCV2025/object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization](hat_history-augmented_anchor_transformer_for_online_temporal_action_localization.md)
- [AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)
- [Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)

<!-- RELATED:END -->
