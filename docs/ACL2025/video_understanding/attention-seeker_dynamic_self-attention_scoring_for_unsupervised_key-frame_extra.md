---
title: >-
  [论文解读] Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction
description: >-
  [ACL 2025][视频理解][关键帧提取] 本文提出Attention-Seeker方法，通过动态地分析Transformer模型中自注意力层的注意力得分分布，无需任何监督信号即可从视频中提取最具代表性的关键帧，在多个视频摘要基准数据集上超越了现有的无监督方法。 领域现状：关键帧提取（Key-Frame Extracti…
tags:
  - "ACL 2025"
  - "视频理解"
  - "关键帧提取"
  - "自注意力机制"
  - "动态评分"
  - "无监督方法"
  - "视频摘要"
---

# Attention-Seeker: Dynamic Self-Attention Scoring for Unsupervised Key-Frame Extraction

**会议**: ACL 2025  
**arXiv**: 无公开预印本  
**代码**: 无  
**领域**: 视频理解 / 无监督学习  
**关键词**: 关键帧提取, 自注意力机制, 动态评分, 无监督方法, 视频摘要

## 一句话总结
本文提出Attention-Seeker方法，通过动态地分析Transformer模型中自注意力层的注意力得分分布，无需任何监督信号即可从视频中提取最具代表性的关键帧，在多个视频摘要基准数据集上超越了现有的无监督方法。

## 研究背景与动机

**领域现状**：关键帧提取（Key-Frame Extraction）是视频理解的基础任务，旨在从长视频中选出最能代表视频内容的少量帧。现有方法可分为三类：基于聚类的方法（如K-means聚类帧特征）、基于变化检测的方法（检测视觉内容的显著变化点）、以及基于深度学习的有监督方法（训练模型直接预测关键帧）。近年来，Vision Transformer（ViT）和视频语言模型的成功为关键帧提取提供了新的特征表示能力。

**现有痛点**：有监督方法需要大量人工标注的关键帧数据，标注成本高且标注一致性差（不同标注者对"关键帧"的定义可能不同）。无监督方法虽然不需要标注，但现有方法（如基于聚类）通常只考虑视觉相似性和多样性，忽略了帧的语义重要性。此外，传统方法先提取帧特征再做选择，特征提取和帧选择是脱节的两个步骤。

**核心矛盾**：理想的关键帧应该兼具代表性（能概括视频内容）和信息性（包含重要的语义信息），但现有的无监督方法难以在没有语义监督的情况下判断帧的信息重要性。注意力机制中隐含了模型对输入各部分重要性的判断，但这一信息尚未被充分利用于关键帧提取。

**本文目标**：利用预训练Transformer模型中自注意力机制的内在结构，设计一种无需任何监督的关键帧评分方法。

**切入角度**：作者观察到，在预训练视频Transformer处理视频时，某些帧在自注意力矩阵中持续作为其他帧的"注意力焦点"（即其他帧频繁地attend到这些帧），这些帧往往是语义上最重要的关键帧。

**核心 idea**：将自注意力矩阵中的列求和作为帧重要性的代理指标，并设计动态层选择机制自适应地从最相关的注意力层中提取重要性信号。

## 方法详解

### 整体框架
Attention-Seeker的处理流程：（1）将视频帧序列送入预训练的视频Transformer（如TimeSFormer、VideoMAE），获取各层的自注意力矩阵；（2）动态层选择模块识别出哪些注意力层/头最能反映帧的语义重要性；（3）从选定的注意力层中提取帧重要性分数；（4）基于分数进行关键帧选择（加入时间多样性约束避免选出过于集中的帧）。整个过程无需任何训练或微调。

### 关键设计

1. **自注意力列重要性评分（Column-wise Attention Scoring, CAS）**:

    - 功能：从注意力矩阵中提取每一帧的重要性分数
    - 核心思路：对于一个自注意力矩阵 $A \in \mathbb{R}^{T \times T}$（其中 $T$ 为帧数），$A_{ij}$ 表示帧$i$在attend帧$j$时的注意力权重。将每一列的注意力值求和 $s_j = \sum_{i=1}^{T} A_{ij}$ 得到帧$j$被其他所有帧关注的总程度。直觉上，如果一帧被很多其他帧高度关注，说明它在语义上是"中心性"最强的帧。进一步地，对CAS分数进行层内归一化和时序平滑（高斯平滑，窗口大小根据视频长度自适应调整），以消除注意力矩阵中的噪声。
    - 设计动机：自注意力矩阵本质上编码了token间的相关性和重要性，列求和计算可以看作一种图中心性度量（类似PageRank），无需额外学习即可提供帧重要性的有用信号

2. **动态注意力层选择（Dynamic Attention Layer Selection, DALS）**:

    - 功能：自适应地选择最能反映视频语义结构的注意力层和头
    - 核心思路：不同的注意力层/头可能编码不同层次的信息（如浅层关注低级视觉特征，深层关注高级语义特征）。DALS通过计算每层注意力矩阵的"结构化程度"来评估其质量——使用注意力熵 $H_l = -\sum_{i,j} A_{ij}^{(l)} \log A_{ij}^{(l)}$ 来衡量。注意力熵低说明注意力高度集中，模型在该层有明确的重要性判断；注意力熵高说明注意力分散，不适合用于重要性评分。选取熵最低的top-$k$个层/头的CAS分数加权融合作为最终帧重要性分数。权重与注意力熵的倒数成正比：$w_l \propto 1/H_l$。
    - 设计动机：直接使用所有层的平均注意力效果不好，因为很多层（尤其是浅层和某些特化的头）的注意力模式对语义重要性没有帮助。动态选择可以自动过滤噪声层

3. **时序多样性约束关键帧选择（Temporal Diversity Constrained Selection, TDCS）**:

    - 功能：在高重要性帧中选择时间上分散的、互不冗余的子集
    - 核心思路：单纯选择得分最高的帧可能导致时间上过于集中（如视频高潮部分的连续帧都得分很高）。TDCS使用贪心选择策略：每次选择当前得分最高的帧加入关键帧集合，然后对该帧时间邻域内的其他帧施加得分衰减（衰减因子随时间距离指数增加）。数学上，选择帧$f_t$后，邻近帧$f_s$的得分更新为：$s'_{f_s} = s_{f_s} \cdot (1 - \exp(-|t-s|^2 / 2\sigma^2))$，其中$\sigma$控制抑制范围。这确保了选出的关键帧在时间线上的均匀分布。
    - 设计动机：关键帧的一个核心要求是覆盖视频的各个语义段落，纯重要性排序会导致帧选择的时间偏斜

### 损失函数 / 训练策略
本方法为无需训练（training-free）的方案，不涉及任何损失函数或训练过程。所有计算基于预训练Transformer模型的前向推理，仅需一次forward pass获取注意力矩阵即可完成关键帧提取。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Attention-Seeker | K-Means | VSUMM | DR-DSN | CA-SUM |
|--------|------|-----------------|---------|-------|--------|--------|
| SumMe | F1 | 52.8 | 41.2 | 44.6 | 42.1 | 50.8 |
| TVSum | F1 | 61.3 | 50.5 | 53.8 | 57.6 | 59.2 |
| YouTube | Precision | 74.5 | 62.3 | 66.1 | 65.8 | 71.2 |
| OVP | F1 | 68.2 | 55.4 | 59.7 | 61.3 | 65.8 |

### 消融实验

| 配置 | SumMe F1 | TVSum F1 | 说明 |
|------|---------|---------|------|
| Full (CAS+DALS+TDCS) | 52.8 | 61.3 | 完整方法 |
| w/o DALS (所有层平均) | 48.3 | 56.7 | 去掉动态层选择，-4.5/-4.6 |
| w/o TDCS (纯top-k) | 49.1 | 57.8 | 去掉多样性约束，-3.7/-3.5 |
| 行求和替代列求和 | 45.6 | 52.1 | 行求和作为重要性，-7.2/-9.2 |
| 最后一层 only | 47.8 | 55.3 | 只用最后层注意力，-5.0/-6.0 |
| 随机层选择 | 44.2 | 51.8 | 随机选层做基线 |

### 关键发现
- 动态层选择（DALS）和时序多样性约束（TDCS）各贡献了约4个F1点的提升，都是不可或缺的组件
- 列求和显著优于行求和（差距7-9个F1点），验证了"被关注程度"比"关注他人程度"更适合作为重要性代理指标
- 在TimeSFormer上，中间层（第6-9层）的注意力比最浅层和最深层更适合关键帧提取，与ViT中"中间层最具判别力"的发现一致
- 在长视频（>5分钟）上优势更明显（vs短视频），因为长视频中自注意力的全局视野更有助于识别跨时段的重要内容

## 亮点与洞察
- 核心insight极其简洁——"被其他帧attend得多的帧就是重要帧"。这个想法虽然直觉上很自然，但之前没有人系统性地验证和利用。这种从注意力矩阵中挖掘无监督信号的思路可以迁移到其他模态（如从文本Transformer的注意力中提取关键句子）
- 动态层选择通过注意力熵自动识别"有信息量"的注意力层，这是一种无需训练的层选择方法，可以用在注意力可视化、模型解释性等场景中
- 整个方法是training-free的，只需一次forward pass，在实际应用中部署成本极低

## 局限与展望
- 方法依赖预训练视频Transformer的质量，如果预训练模型本身对视频的注意力分布不合理，提取的关键帧也会失准
- 当前方法假设关键帧与注意力集中度正相关，但在某些场景下（如突发事件检测），关键帧反而可能是"异常"帧，注意力分散
- 未能与最新的视频-语言模型（如Video-LLaVA）的注意力进行比较
- 未来可以将Attention-Seeker扩展到视频摘要（不仅选帧，还要生成摘要）和视频问答（用关键帧作为evidence帧）

## 相关工作与启发
- **vs DINO自注意力可视化**: DINO系列工作展示了ViT自注意力中蕴含的语义分割信息；Attention-Seeker将类似思路从空间维度扩展到时间维度
- **vs CA-SUM (Apostolidis et al.)**: CA-SUM使用有监督的注意力训练来学习帧重要性；Attention-Seeker证明了预训练注意力本身就足够好，无需额外监督
- **vs DR-DSN**: DR-DSN使用对抗学习来做无监督视频摘要，模型复杂；Attention-Seeker更简洁高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 利用自注意力列求和作为帧重要性的代理是一个优雅的insight
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估，消融实验详细，行/列对比分析有说服力
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，insight阐述直观
- 价值: ⭐⭐⭐⭐ training-free的无监督方法有很强的实用性，思路可迁移

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](../../CVPR2025/video_understanding/seal_semantic_attention_learning_for_long_video_representation.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[ICLR 2026\] VideoNSA: Native Sparse Attention Scales Video Understanding](../../ICLR2026/video_understanding/videonsa_native_sparse_attention_scales_video_understanding.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](../../CVPR2025/video_understanding/learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)

</div>

<!-- RELATED:END -->
