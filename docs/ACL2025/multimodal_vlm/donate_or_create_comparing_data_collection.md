---
title: >-
  [论文解读] Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts
description: >-
  [ACL 2025][多模态][emotion detection] 本文系统比较了三种收集作者标注情感数据的策略（创建、捐赠、近期帖子），发现研究创建的数据在文本长度、情感原型性和图文关系上与真实数据存在显著差异，但创建数据仍可有效训练泛化模型，不过真实数据对准确评估模型效果不可或缺。
tags:
  - ACL 2025
  - 多模态
  - emotion detection
  - data collection
  - social media
  - author annotation
---

# Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts

**会议**: ACL 2025  
**arXiv**: [2505.24427](https://arxiv.org/abs/2505.24427)  
**代码**: https://www.uni-bamberg.de/en/nlproc/projects/item/  
**领域**: 多模态VLM  
**关键词**: emotion detection, data collection, multimodal, social media, author annotation

## 一句话总结
本文系统比较了三种收集作者标注情感数据的策略（创建、捐赠、近期帖子），发现研究创建的数据在文本长度、情感原型性和图文关系上与真实数据存在显著差异，但创建数据仍可有效训练泛化模型，不过真实数据对准确评估模型效果不可或缺。

## 研究背景与动机
1. **领域现状**：情感分析需要作者标注的数据来准确建模主观情感表达。目前主流做法有两种：让参与者"创建"(Creation)符合特定情感标签的内容，或让参与者"捐赠"(Donation)真实社交媒体帖子并标注。
2. **现有痛点**：创建数据实施简单且隐私风险低，但生成的内容可能与真实社交媒体帖子存在系统性差异；捐赠数据虽更真实，但面临隐私问题和参与者自我过滤。目前缺乏对这两种收集策略差异的系统性分析。
3. **核心矛盾**：数据收集的便捷性与数据真实性之间存在根本性权衡——研究者不清楚创建数据与真实数据到底有多大差异，也不知道这些差异对下游模型的影响有多大。
4. **本文目标** (1) 创建数据和真实数据在内容、标注和样本特征上有哪些具体差异？(2) 这些差异如何影响情感分类模型？(3) 不同收集策略是否导致参与者群体差异？
5. **切入角度**：作者设计了三种收集策略（Creation、Donation、Recent）来收集多模态社交媒体帖子，并在多个维度（文本特征、图片风格、情感评价、参与者特征、模型性能）上进行对比。
6. **核心 idea**：通过严格对比实验揭示数据收集策略对情感数据质量和模型泛化的影响。

## 方法详解

### 整体框架
本文设计了一个大规模对比实验：招募522名参与者，通过三种不同的数据收集策略收集共2,507条多模态社交媒体帖子（含文本+图片），每条帖子都有作者标注的情感标签和详细的元数据（情感评价、图文关系等），然后从五个维度系统分析策略间的差异及其对模型的影响。

### 关键设计

1. **三种数据收集策略**:

    - 功能：以不同方式获取作者标注的情感数据
    - 核心思路：Creation 要求参与者回忆某种情感事件并创建社交媒体帖子（图片从Flickr数据库选取）；Donation 要求参与者从自己的社交媒体账号中找到符合指定情感的真实帖子并提交；Recent 要求参与者提交最近5条帖子并自由标注情感（不指定情感类别）
    - 设计动机：三种策略在数据真实性、隐私保护和情感覆盖平衡性上各有侧重，对比研究可以揭示各自的优缺点

2. **多维度元数据标注体系**:

    - 功能：收集帖子的丰富上下文信息以支持深入分析
    - 核心思路：每条帖子标注包括情感强度（1-5级）、图文关系（5项Likert量表）、事件评价（15项心理学维度如可预测性、自控感等）、事件持续时间和情感持续时间。同时收集参与者的人口统计学信息
    - 设计动机：仅对比文本特征不足以揭示差异的根本原因，需要从心理学维度理解参与者如何选择和表达情感事件

3. **跨策略模型训练-测试矩阵**:

    - 功能：量化数据差异对模型性能的实际影响
    - 核心思路：将Creation和Donation各分为训练集(800条)和测试集(300条)，使用单模态模型（RoBERTa处理文本、ViT处理图像）和多模态模型（CLIP双编码器+分类头），以及零样本VLM（llama3.2-vision、llava-llama3等），构建训练×测试的交叉验证矩阵
    - 设计动机：直接衡量跨策略泛化能力，验证创建数据是否足以训练在真实数据上表现良好的模型

### 训练策略
模型在Creation和Donation上分别微调，使用交叉熵损失进行6类情感分类。每种配置重复5次取平均。零样本模型各提示5次取平均。

## 实验关键数据

### 主实验

| 训练数据 | 测试数据 | 视觉(V) F1 | 文本(T) F1 | 多模态(T+V) F1 | 零样本 F1 |
|---------|---------|-----------|-----------|--------------|---------|
| Donation | Creation | .16 | .49 | .60 | .24/.61/.56 |
| Creation | Creation | .18 | .58 | .62 | - |
| Donation | Donation | .19 | .41 | .50 | .19/.45/.46 |
| Creation | Donation | .18 | .42 | .50 | - |

### 数据差异分析

| 维度 | Creation vs Donation/Recent | 统计显著性 |
|------|---------------------------|----------|
| 文本长度 | Creation比Recent长51%，比Donation长26% | p < 0.01 |
| 事件强度 | Creation比Donation高0.34分(5分制) | p < 0.001 |
| 情感响应强度 | Creation比Donation高0.36分 | p < 0.001 |
| 参与拒绝率 | Donation/Recent远高于Creation | p < 0.001 |
| 截图类图片 | Creation几乎没有，Donation/Recent较多 | χ² p < 0.001 |

### 关键发现
- Creation数据中情感事件更加"原型化"——参与者倾向于选择情感强度高、持续时间长的典型事件，而真实帖子中情感触发更多样
- 三种策略导致参与者样本在年龄、学生比例、种族构成上显著不同，Donation/Recent的拒绝率远高于Creation
- 在Creation数据上训练的模型可以泛化到Donation测试集（F1相当），但**在Donation上测试时模型F1普遍低于在Creation上测试时**（.50 vs .62），说明真实数据更难
- 多情感帖子非常普遍：大多数anger、fear、disgust帖子同时包含其他情感，这意味着Donation中的标注可能受到目标情感提示的偏差影响
- 零样本多模态模型在图像模态上表现极差（F1仅.19-.24），说明当前VLM在情感视觉理解上能力有限

## 亮点与洞察
- 三种策略的交叉设计非常巧妙：Creation控制隐私+平衡情感、Donation提供真实性+平衡情感、Recent提供真实性+消除提示偏差，三者互补揭示了不同维度的问题
- "在创建数据上训练可泛化，但评估必须用真实数据"这一结论具有广泛的实际指导意义——对任何涉及主观标注的NLP任务都适用
- 心理学评价维度（appraisal）的引入使得差异分析不停留在表面特征，而是深入到认知过程层面

## 局限与展望
- 研究局限于英语和英国/爱尔兰文化背景，跨文化泛化性未知
- Recent数据量小且严重偏向joy（79%），无法用于训练和测试
- 图像仅从Flickr选取（Creation），无法反映真实社交媒体图片的全部类型（如截图、meme）
- 未探索如何混合不同策略收集的数据来训练更鲁棒的模型
- Creation中参与者从Flickr选图而非上传自己的图，这一设计可能引入额外偏差（数据库图片风格单一）
- 情感分类仅覆盖Ekman基本6类，未涉及更细粒度的情感体系

## 相关工作与启发
- **vs Troiano et al. (2023)**: 前者仅收集文本创建数据，本文扩展到多模态并加入真实数据对比，揭示了图文关系的重要差异
- **vs Oprea & Magdy (2020)**: 他们探索了数据捐赠方法用于反讽检测，本文系统比较了捐赠与创建的优缺点，发现两者在参与者特征上也有显著差异
- **vs Kajiwara et al. (2021)**: 同样使用作者标注，但本文新增了近期帖子策略来消除情感提示偏差
- 本文的方法论框架（收集+多维分析+交叉训练测试）可以直接迁移到其他主观标注任务，如stance detection、hate speech annotation

## 评分
- 总体评价: 扁实但重要的实证研究，对未来情感语料库构建有直接指导意义
- 新颖性: ⭐⭐⭐⭐ 系统性对比三种策略的实验设计新颖且全面
- 实验充分度: ⭐⭐⭐⭐⭐ 五个维度的分析非常深入，统计检验严谨
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 对情感数据收集的实践指导价值高

<!-- RELATED:START -->

## 相关论文

- [Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](cosyn_code_guided_synthetic_data.md)
- [AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval](megapairs_massive_data_synthesis_for_universal_multimodal_retrieval.md)
- [Harnessing PDF Data for Improving Japanese Large Multimodal Models](harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)

<!-- RELATED:END -->
