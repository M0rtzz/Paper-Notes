---
title: >-
  [论文解读] Auto-Vocabulary Semantic Segmentation
description: >-
  [ICCV 2025][语义分割][图像分割] 本文提出 Auto-Vocabulary Semantic Segmentation (AVS) 新任务，通过 AutoSeg 框架自动从图像中发现目标类别并分割，无需人为指定词汇表，在 PASCAL VOC 上达到 87.1 mIoU，远超唯一同类方法 ZeroSeg (20.1)，甚至超越部分需要指定类别的开放词汇方法。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "图像分割"
  - "open-vocabulary"
  - "BLIP"
  - "zero-shot"
  - "LLM evaluator"
---

# Auto-Vocabulary Semantic Segmentation

**会议**: ICCV 2025  
**arXiv**: [2312.04539](https://arxiv.org/abs/2312.04539)  
**代码**: 即将公开  
**领域**: 分割 / 开放词汇 / VLM  
**关键词**: auto-vocabulary segmentation, open-vocabulary, BLIP, zero-shot, LLM evaluator

## 一句话总结

本文提出 Auto-Vocabulary Semantic Segmentation (AVS) 新任务，通过 AutoSeg 框架自动从图像中发现目标类别并分割，无需人为指定词汇表，在 PASCAL VOC 上达到 87.1 mIoU，远超唯一同类方法 ZeroSeg (20.1)，甚至超越部分需要指定类别的开放词汇方法。

## 研究背景与动机

**领域现状**：语义分割方法已从闭集（固定类别）发展到开放词汇分割（OVS），后者利用 CLIP 等 VLM 能处理任意文本指定的类别。但 OVS 仍需"人在回路"——用户要么在运行时指定词汇表，要么方法在人工标注数据集上训练时"烘焙"了固定的输出类别。

**现有痛点**：(a) 需要人工指定每张图像应分割哪些类别，限制了可扩展性；(b) 想象厨房机器人需要区分各种工具和食材——不可能每次抓取都要人提供类名；(c) 固向文本查询式方法（如 LISA）对多类别场景需要多次推理，效率极低。

**核心矛盾**：VLM（CLIP、BLIP）训练在图像级别特征上，缺乏局部区域精细推理能力，直接用于精确分割边界困难；而让模型自动发现"应该分割什么"是比"分割指定类别"更难的问题。

**本文目标** (a) 消除分割流程中的人工干预，让模型自主识别图像中所有相关类别；(b) 评估自动生成的类别与标注类别的匹配——同义词、层级关系让离散类别比较困难。

**切入角度**：BLIP 的 patch embedding 天然具有局部语义信息，可以聚类后解码为文字描述——这等价于自动化的局部图像 captioning——然后用生成的类名去驱动 OVS 分割器。

**核心 idea**：用 BLIP 特征聚类+解码生成图像特定词汇表，再作为自引导输入 OVS 模型完成无人干预的语义分割。

## 方法详解

### 整体框架

AutoSeg 流程：输入图像 → BLIP 编码器提取 patch embedding → 多分辨率多 k 值聚类 → CRF + 多数滤波去噪 → BLIP 解码器将各聚类解码为文本 → 提取名词构建词汇表 → 词汇表输入 X-Decoder（OVS 模型）→ 输出语义分割 mask。评估时通过 LAVE（LLM-based 评估器）将自动类别映射到 ground truth 类别。

### 关键设计

1. **BBoost（Bootstrapped BLIP-based Vocabulary Generation）**:

    - 功能：从图像中自动发现所有相关的目标类别名称
    - 核心思路：(a) BLIP ViT 编码器将图像分为 patch embedding $\hat{\mathbf{B}} \in \mathbb{R}^{N \times d}$；(b) 对两种分辨率（384×384, 512×512）分别运行 $k \in \{2,...,8\}$ 的 k-means，得到 14 组聚类；(c) 通过 Hungarian 匹配统一各组聚类索引，为每个 patch 计算属于各聚类的概率分布；(d) CRF 去噪 + 多数滤波进一步提纯聚类边界；(e) 将每个聚类对应的 patch embedding 集合送入 BLIP text decoder 解码出描述文本
    - 设计动机：BLIP 虽然在图像级训练，但其 patch embedding 天然保留了局部语义——聚类后可以"局部 captioning"——这是一个被发现的涌现能力（emergent capability）
    - 与 ZeroSeg 的区别：ZeroSeg 需要 DINO 聚类+CLIP embedding+自定义注意力+GPT-2 共四个模型，AutoSeg 只用 BLIP 一个模型同时做聚类和 captioning

2. **Cross-clustering Consistency（跨聚类一致性）**:

    - 功能：统一多次 k-means 的聚类结果
    - 核心思路：选最多聚类的一组为参考集 $S$，其余组通过 IoU 计算+Hungarian 匹配对齐聚类索引，每个 patch 获得一个聚类概率分布 $P(n|p)$
    - 设计动机：单次 k-means 噪声大，多次聚类做 ensemble 降低方差，同时自适应聚类数量（支持弱的聚类自然消失）

3. **LAVE（LLM-based Auto-Vocabulary Evaluator）**:

    - 功能：将自动生成的类别映射到标注数据集的固定类别以计算 mIoU
    - 核心思路：使用 Llama-2-7B，输入所有自动生成类名和目标数据集类名，LLM 根据语义关系（同义词、上下位词）进行映射
    - 设计动机：传统余弦相似度匹配（如 Sentence-BERT）会出现 "taxi→road" 而非 "taxi→car" 的错误，LLM 能处理复杂语义关系

### 损失函数 / 训练策略

AutoSeg 完全无需训练或微调——BLIP 和 X-Decoder 均使用预训练权重。超参数通过贝叶斯优化在 VOC 上调优：CRF 平滑权重 6、平滑 θ=0.8、位置编码维度 256、nucleus sampling top-P=1、重复惩罚 100。

## 实验关键数据

### 主实验

Auto-Vocabulary 设置对比（自动生成词汇表 vs ground truth 词汇表）：

| 方法 | VOC mIoU | PC mIoU | ADE mIoU | CS mIoU |
|------|----------|---------|----------|---------|
| AutoSeg (Ours) | **87.1** | **11.7** | **6.0** | **30.0** |
| ZeroSeg | 20.1 | 11.4 | - | - |
| LLaVA+X-Decoder | 56.7 | 11.4 | - | 23.4 |
| SAM+BLIP+X-Decoder | 41.1 | 11.3 | - | 27.4 |
| LLaVA+LISA | 7.7 | 0.2 | - | 1.5 |

与 OVS 方法对比（OVS 方法使用 ground truth 类名作为输入）：

| 方法 | 需要词汇表 | VOC | PC | ADE | CS |
|------|-----------|------|-----|------|-----|
| CAT-Seg | ✗ | 97.2 | 19.0 | 13.3 | - |
| X-Decoder | ✗ | 96.2 | 16.1 | 6.4 | 50.8 |
| AutoSeg | ✓ (自动) | **87.1** | **11.7** | 6.0 | 30.0 |
| OVSeg | ✗ | 94.5 | 11.0 | 9.0 | - |
| OpenSeg | ✗ | 72.2 | 9.0 | 8.8 | - |

### 消融实验

| 配置 | VOC c-mIoU | PC c-mIoU | ADE c-mIoU | CS c-mIoU |
|------|------------|-----------|------------|-----------|
| AutoSeg (完整) | **71.8** | **47.7** | **29.2** | **35.8** |
| X-Decoder+BLIP | 38.0 | 35.3 | 26.7 | 29.2 |
| BBoost Embeddings only | 16.3 | 16.3 | 11.3 | 0.85 |

### 关键发现

- AutoSeg 在 VOC 上 87.1 mIoU，达到最佳 OVS 方法（97.2）的 91%——而完全无需指定类名
- 在 VOC 和 PC 上**超过了 OpenSeg、ODISE、OVSeg** 等需要指定类名的 OVS 方法
- AutoSeg 发现的类别数远多于标注类别（VOC 938 vs 20, ADE 1578 vs 847），能识别标注中缺失的细粒度类别（如 hawk、coke、dachshund）
- LAVE 映射与人工映射结果差异极小（VOC: 87.1 vs 88.2），证明 LLM 评估器可靠

## 亮点与洞察

- **BLIP patch embedding 的涌现能力**：发现 BLIP 虽然只在图像级 captioning 上训练，但 patch embedding 聚类后送入 decoder 可以做局部 captioning——这是一个令人惊喜的涌现特性，极具启发
- **多分辨率多 k 值 ensemble**：通过 14 组聚类做 ensemble 消除单次 k-means 的噪声，比单次聚类稳定很多，思路可迁移到任何使用聚类的场景
- **自引导范式（Self-Guidance）**：用自身生成的词汇表引导分割，避免了外部输入依赖，是真正的端到端无人干预系统
- **LAVE 评估器**：用 LLM 做词汇映射解决了自动生成类别与标注类别的语义匹配难题，比余弦相似度鲁棒得多

## 局限与展望

- 在类别极多的数据集（ADE 847 类、CS）上性能较弱，仍有提升空间
- BBoost 有时会遗漏非显著物体（背景区域），captioning 训练偏向前景目标
- 依赖 X-Decoder 的分割质量——如果 OVS 模型开始支持更多类别，AutoSeg 的上限也会更高
- Captioning 轮数的最优值依赖数据集特性，缺乏自适应策略

## 相关工作与启发

- **vs ZeroSeg**: ZeroSeg 需要 DINO+CLIP+GPT-2 三个模型且 VOC 只有 20.1 mIoU，AutoSeg 用单一 BLIP 就达到 87.1
- **vs LISA**: LISA 需要逐类别多次推理，在多类别场景下效率极低且效果差（7.7 mIoU）
- **vs OVS 方法**: AutoSeg 在无需类名的前提下超越了 OpenSeg、OVSeg 等需要 ground truth 类名的方法，说明自动词汇表生成具有竞争力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个真正无需人工输入的自动词汇语义分割框架，BLIP 涌现特性的发现很启发
- 实验充分度: ⭐⭐⭐⭐ 四个数据集+多种消融+OVS对比，但缺少一些 failure case 分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，AVS vs OVS 的对比图很直观
- 价值: ⭐⭐⭐⭐⭐ 消除"人在回路"是分割领域的重要进步，LAVE 评估框架可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation](stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)

</div>

<!-- RELATED:END -->
