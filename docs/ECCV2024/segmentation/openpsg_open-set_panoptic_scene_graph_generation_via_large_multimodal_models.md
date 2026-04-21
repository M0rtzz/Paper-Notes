---
title: >-
  [论文解读] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models
description: >-
  [ECCV 2024][图像分割][panoptic scene graph] 首次定义开放集全景场景图生成（OpenPSG）任务，利用 BLIP-2 作为多模态关系解码器，结合关系查询 Transformer（RelQ-Former）实现开放集关系预测，在 PSG 数据集 PredCls R@100 达到 79.3%，闭集场景超越先前 SOTA 26.6%。
tags:
  - ECCV 2024
  - 图像分割
  - panoptic scene graph
  - open-set
  - relation prediction
  - 多模态
  - BLIP-2
---

# OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models

**会议**: ECCV 2024  
**arXiv**: [2407.11213](https://arxiv.org/abs/2407.11213)  
**代码**: https://github.com/franciszzj/OpenPSG  
**领域**: 场景图生成 / 视觉-语言  
**关键词**: panoptic scene graph, open-set, relation prediction, large multimodal model, BLIP-2

## 一句话总结
首次定义开放集全景场景图生成（OpenPSG）任务，利用 BLIP-2 作为多模态关系解码器，结合关系查询 Transformer（RelQ-Former）实现开放集关系预测，在 PSG 数据集 PredCls R@100 达到 79.3%，闭集场景超越先前 SOTA 26.6%。

## 研究背景与动机
**领域现状**：全景场景图生成（PSG）旨在分割图像中的物体并识别它们之间的关系，构建结构化的场景理解。已有方法（PSGTR、HiLo、PairNet）在闭集设定下取得进展，但都只能预测预定义的关系类别。

**现有痛点**：大模型时代下，开放集物体检测和分割已有大量工作（OpenSeeD、Grounding DINO），但开放集关系预测尚未被探索。关系预测比物体检测更复杂——模型需要同时理解不同物体并根据交互推理关系，且物体对数量呈 $N(N-1)$ 增长。

**核心矛盾**：已有开放集 SGG 方法（如 Cacao+Epic、OvSGTR）使用 CLIP 特征匹配或知识蒸馏来处理新关系，但这些方法要么受限于固定的关系嵌入空间，要么无法真正生成新颖的关系描述。

**本文目标**：实现真正的开放集全景场景图生成，即物体类别和关系类别都可以超越预定义集合。

**切入角度**：利用 LMM（大规模多模态模型）的自回归文本生成能力来预测关系——LMM 既擅长理解名词（物体）也擅长谓词（关系），用自然语言生成关系描述天然支持开放集。

**核心 idea**：用 RelQ-Former 高效提取物体对视觉特征并过滤无关对，再用 BLIP-2 自回归解码生成/判断开放集关系。

## 方法详解

### 整体框架
OpenPSG 包含三个组件：(1) Object Segmenter：使用预训练的 OpenSeeD 进行开放集全景分割，获取物体类别、掩码和视觉特征；(2) Relation Query Transformer (RelQ-Former)：用两组可学习查询提取物体对特征并判断关系是否存在；(3) Multimodal Relation Decoder：继承 BLIP-2 解码器，用文本指令引导自回归关系预测。训练时冻结 Object Segmenter 和 Multimodal Relation Decoder，仅训练 RelQ-Former。

### 关键设计
1. **Patchify + Pairwise 模块**

    - 功能：将像素解码器输出的视觉特征 $F_I \in \mathbb{R}^{h \times w \times D}$ 序列化，并构建物体对
    - 核心思路：用单个卷积层将 $F_I$ 转换为 patch 序列 $F_{Iseq} \in \mathbb{R}^{L \times D}$；将 N 个物体全排列为 $N(N-1)$ 个主客体对 $P$；对每对中两个物体的掩码做 OR 运算得到对掩码序列 $m_{seq}^{pair} \in \{0,1\}^{N(N-1) \times L}$
    - 设计动机：为后续的关系查询 Transformer 提供标准化的视觉 token 输入和掩码引导

2. **Pair Feature Extraction Query（对特征提取查询）**

    - 功能：从全图视觉特征中提取关注物体交互区域的对特征
    - 核心思路：可学习查询 $Q^{feat} \in \mathbb{R}^{E \times D}$ 先与 pair instruction（如"Extracting subject-object (person, skateboard) features"）做自注意力 $F_{SA}^{feat} = \text{Trunc}(\text{SA}(\text{Concat}(Q^{feat}, F_{Inst}^{feat})), E)$，然后用掩码交叉注意力从 $F_{Iseq}$ 中提取对特征 $F_{CA}^{feat} = \text{MaskCA}(F_{SA}^{feat}, F_{Iseq}, m_{seq})$，经 FFN、重复两次得到最终对特征 $F_I^{pair(i,j)} \in \mathbb{R}^{E \times D}$
    - 设计动机：相比简单的 mask pooling（对所有区域一视同仁），注意力机制可以让特征更聚焦于物体交互区域——消融显示这带来 R@100 +5.2% 的提升

3. **Relation Existence Estimation Query（关系存在性判断查询）**

    - 功能：快速判断物体对之间是否可能存在关系，过滤无关对
    - 核心思路：单 token 查询 $Q^{exist} \in \mathbb{R}^{1 \times D}$ 经类似流程与指令"Is there a relation between $o_i$ and $o_j$?"交互，输出经 2 层 MLP + sigmoid 得到 [0,1] 分数。阈值 $\theta=0.35$ 过滤
    - 设计动机：$N(N-1)$ 对中大部分无关系，全部送入 LMM 解码极慢。关系存在性过滤实现 **20× 加速**

4. **Generation + Judgement 双指令设计**

    - 功能：用两种互补的指令方式实现开放集关系预测
    - 核心思路：
        - Generation 指令："What are the relations between $c_i$ and $c_j$?" → 自回归生成所有可能关系，多关系用 [SEP] 分隔：$r_{i,j} = \text{Dec}(\text{Concat}(F_I^{pair(i,j)}, F_{inst}^{gen}))$
        - Judgement 指令："Please judge between $c_i$ and $c_j$ whether there is a relation $r_k$" → 对每个候选关系判断 Yes/No。利用 KV-cache 缓存 prefix：$F_{prefix}^{(i,j)} = \text{Dec}(\text{Concat}(F_I^{pair(i,j)}, F_{inst}^{judge}))$，然后对每个关系只需处理关系名 token
    - 设计动机：Generation 擅长发现新关系但偏向高频关系；Judgement 借助 LMM 的判断能力处理低频和罕见关系，且通过 prefix caching 保持与 Generation 相同的推理速度

### 损失函数 / 训练策略
- **损失函数**：$\mathcal{L} = \lambda \mathcal{L}_{exist} + \mathcal{L}_{LM}$，其中 $\mathcal{L}_{exist}$ 为二值交叉熵（关系存在性判断），$\mathcal{L}_{LM}$ 为标准语言模型交叉熵，$\lambda=10$
- **训练设置**：冻结 Object Segmenter 和 Multimodal Relation Decoder，仅训练 RelQ-Former；AdamW，lr=1e-4，weight decay=5e-2；12 epochs，第 8 epoch lr 降至 1e-5；4×A100
- **开放集划分**：基础关系:新类关系 = 7:3，训练仅用基础关系数据

## 实验关键数据

### 主实验（PSG 数据集）

| 方法 | 设定 | PredCls R@100 | PredCls mR@100 | SGDet R@100 | SGDet mR@100 |
|------|------|--------------|----------------|-------------|--------------|
| HiLo | 闭集 | - | - | 43.0 | 33.1 |
| PairNet | 闭集 | - | - | 39.6 | 30.6 |
| PSGTR | 闭集 | - | - | 36.3 | 22.1 |
| **OpenPSG** | **闭集** | **79.3** | **63.8** | **52.0** | **50.1** |
| OpenPSG | 开放集 | 61.5 | 46.0 | 36.7 | 25.4 |

### VG 数据集 PredCls

| 方法 | 设定 | R@100 | mR@100 |
|------|------|-------|--------|
| VCTree | 闭集 | 68.1 | 19.4 |
| Cacao+Epic | 闭集 | - | 40.8 |
| **OpenPSG** | **闭集** | **71.4** | **50.3** |
| OvSGTR | 开放集 | 26.7 | - |
| **OpenPSG** | **开放集** | **30.6** | **27.2** |

### 消融实验

| 配置 | PredCls R@100 | PredCls mR@100 | 说明 |
|------|--------------|----------------|------|
| Mask Pooling 特征提取 | 74.1 | 59.1 | 简单池化 |
| **RelQ-Former 注意力提取** | **79.3** | **63.8** | +5.2/+4.7 |
| 无关系存在性过滤 | 79.3 | 63.8 | 全对推理（慢 20×）|
| 有关系存在性过滤 | 78.8 | 63.0 | 性能微降但 20× 加速 |
| Generation 指令（开放集） | 59.8 | 41.6 | 偏相高频关系 |
| **Judgement 指令（开放集）** | **61.5** | **46.0** | mR 显著更优 |

### 关键发现
- 开放集训练的 OpenPSG 在 PredCls 上甚至超越了所有闭集训练的先前方法
- Judgement 指令在 mR@K 上显著优于 Generation 指令（+4.4@mR100），说明 LMM 的判断能力比生成能力更适合低频关系
- RelQ-Former 的注意力机制比 mask pooling 效果好 5%+，证明关注交互区域的重要性
- 关系存在性过滤几乎不影响性能（-0.5 R@100）但带来 20× 加速

## 亮点与洞察
- **首次定义 OpenPSG 任务**：将开放集的概念从物体检测/分割扩展到关系预测，填补了场景图生成领域的重要空白。利用 LMM 的自由文本生成能力天然解决了开放集关系的开放词汇问题。
- **高效的 RelQ-Former 设计**：两组查询分别负责特征提取和关系过滤，前者聚焦交互区域替代简单的 mask pooling，后者以 $O(1)$ 代价过滤大量无关对，整体实现了质量和效率的平衡。
- **Generation + Judgement 互补**：前者发现罕见关系、后者精确判断，这种双指令策略为 LMM 在结构化视觉推理中的应用提供了有价值的范式。

## 局限与展望
- 当前 Object Segmenter 完全冻结，分割错误会直接传播到关系预测，端到端联合训练可能带来更大提升
- 关系存在性过滤的阈值 $\theta=0.35$ 是手动设定的，自适应阈值可能更好
- 开放集设定下 SGDet 性能（36.7 R@100）与闭集差距仍较大，新关系在完整流水线中的传播效果有限

## 相关工作与启发
- **vs HiLo**: HiLo 设计高低频分支处理不均衡关系但限于闭集；OpenPSG 通过 Judgement 指令在 mR 上大幅超越
- **vs OvSGTR**: OvSGTR 用 CLIP 匹配视觉-文本关系特征做开放集，OpenPSG 用 LMM 自回归生成/判断，VG 数据集上 R@100 提升 3.9%
- **vs Cacao+Epic**: 通过外部知识图谱迁移关系知识，受限于知识图谱覆盖范围；OpenPSG 不依赖外部知识

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次定义开放PSG任务，LMM+关系查询Transformer的组合有创意
- 实验充分度: ⭐⭐⭐⭐ PSG和VG双数据集、闭集开集双设定、细致消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 开放集关系预测是重要方向，为后续研究建立了基线

<!-- RELATED:START -->

## 相关论文

- [VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [CoReS: Orchestrating the Dance of Reasoning and Segmentation](cores_orchestrating_the_dance_of_reasoning_and_segmentation.md)
- [Diffusion Models for Open-Vocabulary Segmentation](diffusion_models_for_open-vocabulary_segmentation.md)
- [A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting](a_simple_latent_diffusion_approach_for_panoptic_segmentation_and_mask_inpainting.md)
- [OLAF: A Plug-and-Play Framework for Enhanced Multi-object Multi-part Scene Parsing](olaf_a_plug-and-play_framework_for_enhanced_multi-object_multi-part_scene_parsin.md)

<!-- RELATED:END -->
