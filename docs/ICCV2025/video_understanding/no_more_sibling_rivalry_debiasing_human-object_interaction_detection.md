---
title: >-
  [论文解读] No More Sibling Rivalry: Debiasing Human-Object Interaction Detection
description: >-
  [ICCV 2025][视频理解][人-物交互检测] 发现并系统分析了 HOI 检测中的"有毒兄弟"偏差问题——高度相似的 HOI 三元组在输入端和输出端相互干扰竞争，提出"对比后校准"（C2C）和"合并后拆分"（M2S）两种去偏学习目标，在 HICO-DET 上超越 baseline +9.18% mAP、超越前 SOTA +3.59%。
tags:
  - ICCV 2025
  - 视频理解
  - 人-物交互检测
  - 有毒兄弟偏差
  - 对比-校准学习
  - 合并-拆分策略
  - 去偏
---

# No More Sibling Rivalry: Debiasing Human-Object Interaction Detection

**会议**: ICCV 2025  
**arXiv**: [2509.00760](https://arxiv.org/abs/2509.00760)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 人-物交互检测, 有毒兄弟偏差, 对比-校准学习, 合并-拆分策略, 去偏

## 一句话总结

发现并系统分析了 HOI 检测中的"有毒兄弟"偏差问题——高度相似的 HOI 三元组在输入端和输出端相互干扰竞争，提出"对比后校准"（C2C）和"合并后拆分"（M2S）两种去偏学习目标，在 HICO-DET 上超越 baseline +9.18% mAP、超越前 SOTA +3.59%。

## 研究背景与动机

人-物交互（HOI）检测旨在识别图像中人与物体的交互关系，形式化为 ⟨人, 动作, 物体⟩ 三元组。基于 DETR 的方法已取得显著进展，但作者发现了一个关键问题——**"有毒兄弟"偏差（Toxic Siblings Bias）**。

**什么是"有毒兄弟"偏差？**

该偏差存在于两个层面：

**输入级偏差（同一图像内）**：同一图像中经常共现多个共享相同物体或动作的 HOI 三元组。例如图中标记的人与凳子的交互被错误分类为"坐"而非"站"，因为图像中其他人都是坐在凳子上的。这些相似且相互干扰的特征使交互解码器难以学习区分。

**输出级偏差（跨图像）**：即使出现在不同图像中，某些类别也存在语义相似性，如"看鸟"、"喂鸟"、"抱鸟"、"放飞鸟"——它们的分类头高度相似，导致混淆和竞争。

**偏差的影响有多深？**

统计分析揭示三个关键发现：
- 兄弟类别的学习曲线呈现**相互抑制**的竞争关系：一个提升另一个就下降。
- 输入级：当图像中存在兄弟 HOI 时，样本完全错误（mAP=0）的概率平均增加 **27.59%**。
- 输出级：分类头初始化的余弦相似度每增加 0.1，AP 下降 **5.51**——兄弟越多，学习越困难。

作者的目标是：**不依赖额外预训练数据或重型外部模型（如LLM、扩散模型）**，仅通过修改学习机制来缓解偏差。

## 方法详解

### 整体框架

在 GEN-VLKT 的一阶段两分支 HOI 检测器基础上（视觉编码器 → 实例解码器 → 交互解码器），保持主体架构不变，仅在交互解码器中引入两个辅助学习目标。

### 关键设计

1. **对比后校准（Contrastive-then-Calibration, C2C）——解决输入级偏差**

   **核心思想**：让交互解码器学会抵抗兄弟偏差，优先利用空间线索聚焦正确交互区域。

   **正负样本定义**：对 HOI 三元组 $t_i^{hoi}$，正集 $\mathcal{P}_i$ 包含同一图像中共享相同交互-物体标签的三元组，负集 $\mathcal{N}_i$ 包含共享交互或物体标签但不完全相同的三元组。

   **三元组特征表示**：将语义特征与人/物空间特征拼接：
   $$\hat{v}_i^{tri} = \text{cat}(\hat{q}_i^a, \hat{f}_i^h, \hat{f}_i^o)$$

   **对比损失**：增大负集距离，减小正集距离：
   $$\mathcal{L}_{con} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\sum_{v_j^{tri}\in\mathcal{P}_i}\exp(\hat{v}_i^{tri}\odot v_j^{tri}/\tau)}{\sum_{v_k^{tri}\in\mathcal{E}_i}\exp(\hat{v}_i^{tri}\odot v_k^{tri}/\tau)}$$

   **校准损失**：将正样本的语义特征**替换**为从负集随机采样的错误语义特征，但保留正确的空间特征（强空间先验），要求交互解码器将其**纠正**回正确的三元组：
   $$\underline{v}_i^{tri} = \text{cat}(F_{text}(s^o, a), f_i^h, f_i^o), \quad (s^o, a) = \text{sample}(\mathcal{N}_i)$$

   通过 L1 损失评估纠正效果。使用注意力掩码确保附加查询不干扰原始查询。

2. **合并后拆分（Merge-then-Split, M2S）——解决输出级偏差**

   **设计动机**：直接要求模型区分大量高度相似的 HOI 类别容易受兄弟偏差影响。不如**先学共性再学差异**。

   **Merge 阶段**：利用 CLIP 文本特征对交互类别和物体类别分别聚类，得到 $M_1$ 个超级交互类别和 $M_2$ 个超级物体类别，组合成 $M = M_1 \times M_2$ 个超类。在交互解码器中添加超类分类头，用交叉熵损失训练：
   $$\mathcal{L}_{merge} = \text{CrossEntropy}(\{m_i\}_{i=1}^N, \{\hat{m}_i\}_{i=1}^N)$$

   这使模型先学会区分不同"群组"，利用兄弟间的共享特征。

   **Split 阶段**：对每个查询，找到最相似的 $k_1$ 个类别，然后在所有查询中选出最频繁的 $k_2$ 个类别特征拼接到图像特征中，用细粒度判别损失要求模型在这些最易混淆的类别中正确分类：
   $$\mathcal{L}_{split} = -\frac{1}{N_{split}}\sum_{i=1}^{N_{split}}\log\frac{\exp(q_i^a \odot t_i / \tau)}{\sum_{j=1}^{k_2}\exp(q_i^a \odot t_j / \tau)}$$

### 损失函数 / 训练策略

总损失为五项加权和：
$$\mathcal{L}_{all} = \lambda_1\mathcal{L}_{detector} + \lambda_2\mathcal{L}_{con} + \lambda_3\mathcal{L}_{cal} + \lambda_4\mathcal{L}_{merge} + \lambda_5\mathcal{L}_{split}$$

其中 $\lambda_1=1, \lambda_2=1, \lambda_3=0.5, \lambda_4=1, \lambda_5=0.5$。使用 AdamW 优化器，90 epochs，初始学习率 $10^{-4}$，60 epoch 时衰减 10 倍。超参数 $k_1=2, k_2=10, M=25$（$M_1 \times M_2 = 5 \times 5$）。

## 实验关键数据

### 主实验

在 HICO-DET 数据集上（R50 + CLIP）：

| 方法 | VLM | Full | Rare | Non-Rare |
|------|-----|------|------|----------|
| GEN-VLKT (baseline) | CLIP | 33.75 | 29.25 | 35.10 |
| HOICLIP | CLIP | 34.69 | 31.12 | 35.74 |
| ADA-CM | CLIP | 38.40 | 37.52 | 38.66 |
| BCOM | CLIP | 39.34 | 39.90 | 39.17 |
| **Ours** | **CLIP** | **42.93** | **42.41** | **43.11** |

在 V-COCO 数据集上：本文 AP^S1=69.8, AP^S2=72.1，均为最优。

R50 + BLIP2 配置下：Full=43.98，超越使用 LLM+SAM 的 SICHOI（41.79）。

### 消融实验

| 配置 | Full | Rare | Non-Rare | 说明 |
|------|------|------|----------|------|
| Baseline (GEN-VLKT-s) | 33.75 | 29.25 | 35.10 | 基线 |
| + HOR mask | 35.37 | 30.70 | 36.51 | 注意力掩码 |
| + Contrastive Loss | 36.82 | 32.29 | 37.92 | 对比学习 |
| + Calibration Loss | 38.03 | 34.05 | 39.80 | 校准损失 +1.21% |
| + Merge Loss | 39.88 | 36.12 | 40.20 | 合并目标 +1.85% |
| + Split Loss | **42.93** | **42.41** | **43.11** | 拆分目标 +3.05% |

超参数消融：$k_1=2, k_2=10, M_1 \times M_2 = 5 \times 5$ 为最佳配置。

### 关键发现

- Rare 类别提升最大（29.25→42.41，+13.16%），验证 M2S 策略有效利用头部类数据帮助尾部类。
- R50 backbone 超越使用 R101 的 CyCleHOI，说明方法的参数效率高。
- 零样本设置下也展现竞争力（RF=35.48 vs 监督方法水平），说明去偏策略的普适性。
- 不依赖 Stable Diffusion/LLM/SAM 等重型外部模型，仍超越使用这些模型的方法。

## 亮点与洞察

- **问题发现深刻**：通过学习曲线和统计分析系统性地量化了"有毒兄弟"偏差，而非仅凭直觉。
- **输入级去偏的"先对比再校准"设计精巧**：用错误语义+正确空间作为附加查询训练解码器的纠错能力，既简洁又有效。
- **输出级"先合后分"符合人类认知**：先学大类再细分，该分层学习策略巧妙利用了兄弟间的共享特征。
- **轻量化**：仅修改学习目标，不改变推理架构，不引入额外计算开销。

## 局限与展望

- 聚类超参数 $M_1, M_2$ 的选择依赖人工调优，自动确定最优聚类数更具普适性。
- 仅在静态图像 HOI 上验证，视频 HOI 检测中的时序兄弟偏差值得探索。
- C2C 中的负样本采样策略较简单（随机采样），引入更难的负例挖掘可能进一步提升效果。

## 相关工作与启发

- "兄弟偏差"的概念可推广到其他细粒度识别任务（如细粒度物体分类、关系检测）。
- 合并-拆分的分层学习策略与课程学习思想相通，可用于长尾分布问题。
- 对比+校准的双阶段策略是处理输入端语义混淆的通用思路。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题发现深刻，两个去偏目标设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多backbone、多VLM、零样本设置、详细消融
- 写作质量: ⭐⭐⭐⭐ 分析清晰，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 在 HICO-DET 上大幅领先 SOTA，方法思路有广泛迁移性

<!-- RELATED:START -->

## 相关论文

- [Locality-Aware Zero-Shot Human-Object Interaction Detection](../../CVPR2025/video_understanding/locality-aware_zero-shot_human-object_interaction_detection.md)
- [H-MoRe: Learning Human-centric Motion Representation for Action Analysis](../../CVPR2025/video_understanding/h-more_learning_human-centric_motion_representation_for_action_analysis.md)
- [Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [General Compression Framework for Efficient Transformer Object Tracking](general_compression_framework_for_efficient_transformer_object_tracking.md)
- [XTrack: Multimodal Training Boosts RGB-X Video Object Trackers](xtrack_multimodal_training_boosts_rgb-x_video_object_trackers.md)

<!-- RELATED:END -->
