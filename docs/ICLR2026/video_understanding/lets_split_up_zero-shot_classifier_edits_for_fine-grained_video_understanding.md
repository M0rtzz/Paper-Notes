---
title: >-
  [论文解读] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding
description: >-
  [ICLR 2026][视频理解][类别拆分] 提出了"类别拆分"(Category Splitting)新任务，通过挖掘视频分类器权重中的潜在组合结构，在零样本条件下将粗粒度动作类别拆分为细粒度子类别，无需重训或额外数据。
tags:
  - ICLR 2026
  - 视频理解
  - 类别拆分
  - 零样本编辑
  - 细粒度视频识别
  - 分类器修改
  - 组合结构
---

# Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding

**会议**: ICLR 2026  
**arXiv**: [2602.16545](https://arxiv.org/abs/2602.16545)  
**代码**: [有](https://kaitingliu.github.io/Category-Splitting/)  
**领域**: 视频理解  
**关键词**: 类别拆分, 零样本编辑, 细粒度视频识别, 分类器修改, 组合结构

## 一句话总结

提出了"类别拆分"(Category Splitting)新任务，通过挖掘视频分类器权重中的潜在组合结构，在零样本条件下将粗粒度动作类别拆分为细粒度子类别，无需重训或额外数据。

## 研究背景与动机

视频识别模型通常在固定的分类体系上训练，但这些分类体系往往过于粗糙。例如，一个"open"标签可能涵盖"open cupboard"、"open by pushing"、"open quickly"、"open halfway"等截然不同的情况。随着应用场景的演化，需要更细粒度的区分。

现有解决方案有三类缺陷：
- **重新标注+重训**：代价昂贵，需要大量标注数据和完整训练周期
- **视觉-语言模型 (VLMs)**：依赖海量视频-文本语料，专业领域数据稀缺，难以捕捉细粒度时序线索
- **持续学习**：需要每个新类的训练数据，且关注全新类别而非已有类别的细化

核心洞察：现代视频backbone已经在其特征空间中编码了丰富的潜在结构，可以被分解以区分细粒度变化，即便没有直接的监督信号。

## 方法详解

### 整体框架

方法的核心是**仅编辑分类头**（classification head），保持backbone不变，将一个粗粒度类别 $c$ 拆分为多个细粒度子类别 $\mathcal{S}^c = \{s_1^c, s_2^c, \dots, s_k^c\}$。更新后的标签空间为 $\mathcal{Y}' = (\mathcal{Y} \setminus \{c\}) \cup \mathcal{S}^c$。

编辑方法 $E$ 需满足两个性质：
- **泛化性 (Generality)**：编辑应能正确分类未见过的新子类别样本
- **局部性 (Locality)**：编辑应保持其他类别的预测不受影响

### 关键设计

#### 1. Modifier Retrieval（修饰符检索）

将每个细粒度子类别视为"粗粒度概念 + 修饰符"的组合。例如"pushing left to right"= "pushing" + "left to right"。核心步骤：

**修饰符字典构建**：对分类器中已有的细粒度类别进行分组，找到共享基础概念的"伪粗粒度类别" $\tilde{c}$。其权重向量近似为子类别权重的均值：

$$v_{\tilde{c}} = \frac{1}{|\mathcal{S}^{\tilde{c}}|} \sum_{y \in \mathcal{S}^{\tilde{c}}} w_y$$

修饰符向量通过减去共享基础得到：$v_m = w_y - v_{\tilde{c}}$

**修饰符向量迁移**：对于目标子类别，通过文本编码器 $\phi$ 计算修饰符文本嵌入的余弦相似度，同时考虑修饰符相似性和完整标签相似性进行检索：

$$v_m^* = \arg\max_{(t_y, t_m, v_m) \in \mathcal{M}_{mod}} \text{sim}(\phi(t_y), \phi(t_s^*)) + \text{sim}(\phi(t_m), \phi(t_m^*))$$

新子类别的权重向量为：$w_{s_j^c} = w_c + v_m^*$

#### 2. Modifier Alignment（修饰符对齐）

为处理字典中不存在的新修饰符，训练一个轻量对齐模块 $g_\psi: \mathbb{R}^n \to \mathbb{R}^m$，将文本嵌入直接映射到分类器权重空间。

训练数据来自两部分：
- **修饰符级别对**：$\mathcal{D}_{mod} = \{(\phi(t_m), v_m)\}$
- **类别级别对**：$\mathcal{D}_{cat} = \{(\phi(t_y), w_y)\} \cup \{(\phi(t_{\tilde{c}}), v_{\tilde{c}})\}$

使用MSE损失训练，对齐模块是单隐层MLP（384维），仅更新 $\psi$ 参数，分类器和文本编码器保持冻结。全过程仍为零样本，无需视频数据。

#### 3. Low-Shot Category Splitting（低样本拆分）

当有少量标注样本时（低至每子类别1个视频），采用**隔离微调**策略：仅微调新添加的子类别权重 $\theta_{head}'$，冻结backbone和原始分类头。使用零样本方法的权重作为初始化，效果优于粗粒度类别权重初始化。

### 损失函数 / 训练策略

- 零样本阶段无需训练数据
- 对齐模块：MSE损失，AdamW优化器，学习率 $1 \times 10^{-3}$，余弦退火，batch size 10
- 低样本微调：交叉熵损失，AdamW，学习率 $1 \times 10^{-3}$，weight decay $1 \times 10^{-3}$，batch size 16
- EMA早停（$\beta=0.95$，patience=5，$\delta=1\times10^{-3}$）

## 实验关键数据

### 主实验

基线设置：ViT-Small + MVD预训练，CLIP ViT-L/14文本编码器。数据集：SSv2-Split（54个粗类别）和FineGym-Split（42个粗类别）。

| 方法 | SSv2-A Gen. | SSv2-A Loc. | FineGym-A Gen. | FineGym-A Loc. |
|------|:-:|:-:|:-:|:-:|
| CLIP | 27.6 | 100.0 | 12.1 | 100.0 |
| FG-CLIP | 30.9 | 100.0 | 19.4 | 100.0 |
| VideoPrism | 28.2 | 100.0 | 21.7 | 100.0 |
| **Ours** | **46.3** | **98.9** | **34.2** | **97.8** |

VLMs泛化性极低，本文方法在SSv2上泛化性提升近20个百分点。

| 微调策略 | 初始化 | Generality | Locality | Mean |
|---------|-------|:-:|:-:|:-:|
| 全模型 one-shot | 粗类别 | 33.6 | 0.0 | 16.8 |
| 仅 $\theta_{head}'$ one-shot | 粗类别 | 48.4 | 98.4 | 73.4 |
| 仅 $\theta_{head}'$ one-shot | modifier alignment | **52.8** | **98.2** | **75.5** |
| 全数据微调 | 粗类别 | 86.7 | 19.2 | 52.9 |

### 消融实验

- 修饰符检索(45.0%) vs. 修饰符对齐(46.3%)：对齐提升1.3%泛化性
- zero-shot初始化 vs. 随机初始化：提升 +7.8% 泛化性
- 预训练影响：MVD(46.3%) > SIGMA(44.1%) > VideoMAE(42.9%) > 从头训练(37.0%)
- 文本编码器：CLIP(46.3%) ≈ VideoPrism(46.5%) > RoBERTa(40.9%)

### 关键发现

- 方向型拆分效果最好，涉及物体数量、意图/成功的拆分最困难
- 全数据微调反而不如one-shot微调（75.5 vs. 52.9），因为全数据对新类别产生强偏置严重破坏局部性
- 当原始标签空间中存在相同修饰符的类似类别时效果更好，但没有也有效

## 亮点与洞察

1. **任务定义创新性强**：类别拆分是一个自然但被忽视的真实场景问题
2. **VLM不如内在结构**：VLM在细粒度视频理解方面远不如挖掘视频分类器的权重结构
3. **极简但有效**：仅编辑分类头、无需backbone更新、零样本即可运行
4. **反直觉发现**：全数据微调反而不如one-shot，隔离微调+零样本初始化是最佳策略

## 局限与展望

- 依赖文本标签来识别修饰符，难以处理纯视觉差异（如速度快慢）
- 零样本泛化性仍有提升空间（46% vs. 理想的86%+）
- 仅在分类任务上验证，未探索检测/分割等下游任务
- 需要原始分类器已有一些细粒度类别来构建修饰符字典

## 相关工作与启发

- 与NLP中的模型编辑（Model Editing）概念直接关联，借用泛化性/局部性评价框架
- 与组合动作识别(Compositional Action Recognition)有天然联系
- 修饰符的"可迁移性"假设值得在更多领域验证
- 可扩展到其他视觉任务的分类细化场景

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 任务定义和零样本编辑思路均为首创
- 技术深度：⭐⭐⭐⭐ — 方法简洁优雅但技术含量适中
- 实验充分度：⭐⭐⭐⭐⭐ — 构建了专用benchmark，消融全面
- 实用价值：⭐⭐⭐⭐ — 低成本分类器更新有实际应用前景

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](../../CVPR2026/video_understanding/frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](../../NeurIPS2025/video_understanding/part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](../../CVPR2026/video_understanding/ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)

<!-- RELATED:END -->
