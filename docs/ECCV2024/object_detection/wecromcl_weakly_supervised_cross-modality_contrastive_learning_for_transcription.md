---
title: >-
  [论文解读] WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting
description: >-
  [ECCV 2024][目标检测][文字检测] 提出 WeCromCL 框架，通过弱监督的原子级跨模态对比学习，仅利用文本转录标注（无位置标注）实现场景文字定位，将检测到的锚点作为伪标签训练单点监督文字检测器，在无边界标注的条件下达到接近全监督的性能。
tags:
  - ECCV 2024
  - 目标检测
  - 文字检测
  - 弱监督学习
  - 跨模态对比学习
  - 转录监督
  - 场景文字识别
---

# WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting

**会议**: ECCV 2024  
**arXiv**: [2407.19507](https://arxiv.org/abs/2407.19507)  
**代码**: 有 (https://github.com/ZhengyaoFang/WeCromCL)  
**领域**: 目标检测 (文字检测与识别)  
**关键词**: 文字检测, 弱监督学习, 跨模态对比学习, 转录监督, 场景文字识别

## 一句话总结

提出 WeCromCL 框架，通过弱监督的原子级跨模态对比学习，仅利用文本转录标注（无位置标注）实现场景文字定位，将检测到的锚点作为伪标签训练单点监督文字检测器，在无边界标注的条件下达到接近全监督的性能。

## 研究背景与动机

场景文字检测（Text Spotting）通常需要精确的文字边界标注（多边形/矩形标注），标注成本极高。**仅转录监督**（Transcription-only Supervision）是一种极具吸引力的替代方案——只需文字内容标注，无需位置标注。

现有仅转录监督方法的局限：

**NPTS**：将文字检测建模为序列预测任务，拼接所有文字实例为一个序列进行自回归预测。但由于文字实例间无预定义顺序，模型需拟合所有排列组合，导致训练收敛极其困难，"需要大量计算资源"

**TOSS**：借鉴 DETR 用预学习查询定位文字，但 DETR 原设计依赖位置监督，缺少位置标注时效果受限

本文的核心洞察：将仅转录监督的文字检测分解为两阶段——先通过弱监督跨模态对比学习定位锚点，再用锚点做伪标签训练单点监督检测器。

## 方法详解

### 整体框架

WeCromCL 采用两阶段流水线：

**阶段一：弱监督锚点检测**
- 输入：场景图像 + 文本转录（无位置标注）
- 输出：每个转录在图像中的锚点位置
- 方法：原子级跨模态对比学习

**阶段二：锚点引导的文字检测**
- 输入：图像 + 锚点伪标签
- 输出：文字检测与识别结果
- 方法：基于 SPTS 或改编的 SRSTS v2 单点检测器

### 关键设计

**原子级对比学习 vs. 整体级对比学习**：

| 维度 | 整体级（如 CLIP/oCLIP） | 原子级（WeCromCL） |
|------|------------------------|-------------------|
| 目标 | 图像-文本的全局语义相关性 | 转录与图像局部区域的字符级外观一致性 |
| 粒度 | 整张图像 vs. 整段文本 | 像素级激活图 vs. 逐字符匹配 |
| 定位能力 | 无法精确定位 | 可通过激活图峰值定位锚点 |

**字符级文本编码器（Character-Wise Text Encoder）**：

- 为字母表中每个字符学习独立的向量嵌入 $\mathbf{E} \in \mathbb{R}^{|\Sigma| \times C}$
- 学习位置嵌入 $\mathbf{P} \in \mathbb{R}^{L \times C}$ 保留字符的时序信息
- 融合后通过 Transformer Encoder 建模字符间关系
- 最终对所有字符取均值得到文本表示 $\mathbf{F}_T \in \mathbb{R}^C$

**软建模激活图**：

通过跨模态交叉注意力计算激活图——文本表示作为 query，图像各像素特征作为 key/value：

$$\mathbf{M}_{(i,j)} = (\mathbf{W}_T^\top \mathbf{F}_T) \cdot (\mathbf{W}_I^\top \mathbf{F}_{I,(i,j)})$$

经 softmax 归一化后，峰值位置即为锚点。激活图进一步用于聚合图像中与转录对应的视觉特征。

**负样本挖掘**：

随机选取不配对转录构造更多负样本对，通过增加图像到文本方向的负样本数量 $N_{\text{aug}}$ 提升判别能力，开销几乎可忽略。

### 损失函数 / 训练策略

对比学习损失包含两个方向：

**文本到图像方向**：

$$\mathcal{L}_i^{T2I} = -\log\frac{\exp(\text{Cosine}(\mathbf{F}_{I_i,T_i}^c, \mathbf{F}_{T_i})/\tau)}{\sum_{j=0}^{N-1}\exp(\text{Cosine}(\mathbf{F}_{I_j,T_i}^c, \mathbf{F}_{T_i})/\tau)}$$

**图像到文本方向**（含负样本挖掘）：

$$\mathcal{L}_i^{I2T} = -\log\frac{\exp(\text{Cosine}(\mathbf{F}_{I_i,T_i}^c, \mathbf{F}_{T_i})/\tau)}{\sum_{j=0}^{N+N_{\text{aug}}-1}\exp(\text{Cosine}(\mathbf{F}_{I_i,T_j}^c, \mathbf{F}_{T_j})/\tau)}$$

最终损失为两个方向的平均。

## 实验关键数据

### 主实验（表格）

WeCromCL 锚点检测性能（F-measure，单点度量）：

| 数据集 | 训练集 | 测试集 |
|--------|--------|--------|
| ICDAR 2013 | 93.2 | 90.5 |
| ICDAR 2015 | 88.6 | 83.4 |
| Total-Text | 84.3 | 80.3 |
| CTW1500 | 66.3 | 77.7 |

WeCromCL + SPTS vs. NPTS（编辑距离度量）：

| 方法 | ICDAR2015 S | W | G | Total-Text None | Full |
|------|------------|---|---|----------------|------|
| NPTS | 70.3 | 62.7 | 57.0 | 61.6 | 70.6 |
| WeCromCL + SPTS | **71.8** | **64.7** | **59.7** | **63.2** | **70.7** |

### 消融实验（表格）

字符级 vs. 词级文本编码器（测试集 F-measure）：

| 编码器类型 | IC13 | IC15 | Total-Text | CTW1500 |
|-----------|------|------|------------|---------|
| Token-wise (CLIP) | 78.6 | 64.4 | 64.9 | 65.5 |
| **Character-wise** | **90.5** | **83.4** | **80.3** | **77.7** |

WeCromCL vs. oCLIP（测试集 F-measure）：

| 方法 | IC13 | IC15 | Total-Text | CTW1500 |
|------|------|------|------------|---------|
| oCLIP (整体对比) | 72.5 | 41.7 | 42.8 | 45.9 |
| **WeCromCL (原子对比)** | **90.5** | **83.4** | **80.3** | **77.7** |

### 关键发现

1. 字符级编码器比词级编码器在所有数据集上提升超过10个F1点，证明文字检测是外观匹配而非语义匹配
2. 原子级对比学习（WeCromCL）大幅超越整体级对比学习（oCLIP），CTW1500上差距达31.8%
3. 负样本挖掘使CTW1500测试集F-measure提升10.2%
4. WeCromCL 生成的伪标签可增强全监督检测器，在标注数据不足时效果尤为显著

## 亮点与洞察

1. **问题分解思路精妙**：将困难的仅转录监督问题拆为两个可解子问题（弱监督定位 + 单点监督检测），大幅降低优化难度
2. **原子级对比学习的提出**：不同于CLIP等模型关注语义相关性，WeCromCL学习的是字符级视觉外观一致性，这是文字检测任务的本质需求
3. **聚类中心的类比**：转录作为聚类中心关联所有包含它的图像，模型在大量图像中学到该转录的共性外观模式——非常直觉的解释
4. **低成本负样本增强**：仅增加文本侧负样本（计算几乎为零），即可获得显著性能提升

## 局限与展望

1. 两阶段流水线导致锚点定位误差会传播到检测阶段，可探索端到端的联合优化
2. 当多个相同转录出现在同一图像时，激活图可能产生歧义
3. 字符级编码对非拉丁字母语言（如中文字符）的适用性有待验证
4. 尚未与最新的大规模视觉-语言模型（如 SAM + OCR 结合方案）进行对比

## 相关工作与启发

- **oCLIP / VLPT**：整体级对比学习的代表，侧重全局语义匹配
- **SPTS**：单点监督文字检测器，WeCromCL 的理想搭档
- **NPTS**：同为仅转录监督的方法，但单阶段设计导致优化困难
- 启发：弱监督定位问题可建模为跨模态对比学习，激活图峰值即为定位结果

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

## 相关论文

- [APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension](apl_anchor-based_prompt_learning_for_one-stage_weakly_supervised_referring_expre.md)
- [Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](../../ICML2025/object_detection/discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)

<!-- RELATED:END -->
