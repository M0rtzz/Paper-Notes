---
title: >-
  [论文解读] Relational Visual Similarity
description: >-
  [CVPR 2026][多模态VLM][关系相似度] 本文首次形式化定义关系视觉相似度问题（两图像间的内在关系/功能对应，而非表面属性相似），构建114K匿名描述数据集并训练relsim模型，揭示了现有相似度指标（CLIP/DINO等）在捕捉关系相似度方面的根本性缺陷。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "关系相似度"
  - "视觉类比"
  - "匿名描述"
  - "认知科学"
  - "图像检索"
---

# Relational Visual Similarity

**会议**: CVPR 2026  
**arXiv**: [2512.07833](https://arxiv.org/abs/2512.07833)  
**代码**: [https://thaoshibe.github.io/relsim](https://thaoshibe.github.io/relsim)  
**领域**: 多模态VLM  
**关键词**: 关系相似度, 视觉类比, 匿名描述, 认知科学, 图像检索

## 一句话总结
本文首次形式化定义关系视觉相似度问题（两图像间的内在关系/功能对应，而非表面属性相似），构建114K匿名描述数据集并训练relsim模型，揭示了现有相似度指标（CLIP/DINO等）在捕捉关系相似度方面的根本性缺陷。

## 研究背景与动机
1. **领域现状**：视觉相似度是计算机视觉的基础能力。现有方法（LPIPS、CLIP、DINO等）专注于属性相似度——像素级、语义级或描述级的匹配。
2. **现有痛点**：这些方法无法识别关系相似度——例如，火柴的燃烧阶段与香蕉的成熟阶段具有相同的"时间渐变"逻辑，但它们在属性上完全不同。
3. **核心矛盾**：认知科学认为属性相似度和关系相似度是人类感知的两大核心支柱，但视觉计算完全忽略了后者。关系相似度被认为是区分人类与其他物种的关键认知能力。
4. **本文目标**：将关系视觉相似度形式化为可测量的问题，并构建能捕捉关系结构的模型。
5. **切入角度**：受认知科学启发——人类通过语言或先验知识进行概念抽象来识别关系相似度。因此引入"匿名描述"（描述内在逻辑而非具体对象）作为连接关系相似图像的纽带。
6. **核心idea**：定义匿名描述（如"时间推移下{主体}的变化"），训练模型生成匿名描述，再用这些描述将具有相同关系逻辑的图像拉近。

## 方法详解

### 整体框架
三步pipeline：（1）从LAION-2B过滤114K可能包含可迁移关系结构的图像；（2）训练匿名描述生成模型为每张图像生成匿名描述；（3）在{图像, 匿名描述}对上训练relsim模型，优化将描述编码相似关系抽象的图像拉近。

### 关键设计

1. **数据过滤与策展**:
    - 功能：从大规模图像语料中提取包含可迁移关系结构的图像
    - 核心思路：从LAION-2B中过滤低质量、错误标注和关系无信息的图像，保留可能包含时间序列、结构类比、功能对应等关系模式的图像。
    - 设计动机：LAION-2B中大量图像是关系无关的（如产品照片、自拍等），直接使用会引入噪声。

2. **匿名描述模型**:
    - 功能：为图像生成描述内在关系逻辑而非具体内容的文本
    - 核心思路：训练一个专门的描述模型，输入图像输出匿名描述——这些描述不涉及任何具体可见对象，而是捕捉图像传达的关系逻辑。例如，对一张火柴燃烧图片的匿名描述是"transformation of {subject} over time"而非"burning matchsticks"。
    - 设计动机：匿名描述作为"胶水"连接具有相似内在逻辑的图像。这是将认知科学关于关系相似度需要概念抽象的洞察操作化的关键步骤。

3. **relsim关系相似度模型**:
    - 功能：学习将具有相同关系结构的图像在表示空间中拉近
    - 核心思路：在{图像, 匿名描述}数据集上微调视觉-语言模型，优化目标使得匿名描述编码相似关系抽象的图像特征更接近。
    - 设计动机：标准的视觉-语言对比学习（如CLIP）优化的是图像与其具体描述的匹配，自然偏向属性相似度。通过替换为匿名描述，将优化目标从属性对齐转向关系对齐。

### 损失函数 / 训练策略
标准的视觉-语言对比学习损失，但使用匿名描述替代常规描述。

## 实验关键数据

### 主实验

| 模型 | 属性相似度 | 关系相似度 | 说明 |
|------|----------|----------|------|
| CLIP | 高 | 低 | 仅捕捉属性 |
| DINO | 高 | 低 | 仅捕捉属性 |
| LPIPS | 高 | 极低 | 像素级 |
| relsim | 中高 | 高 | 关系感知 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full relsim | 最优 | 匿名描述训练 |
| 常规描述训练 | 关系相似度差 | 属性描述无法编码关系 |
| w/o 数据过滤 | 下降 | 噪声数据干扰 |

### 关键发现
- 所有主流视觉相似度指标在关系相似度上都表现很差，揭示了视觉计算的关键盲区。
- 匿名描述是连接关系相似图像的有效中介。
- relsim在关系图像检索和类比图像生成等应用中展示了实用价值。

## 亮点与洞察
- **开辟了一个全新的视觉理解维度**：从属性到关系的转变是概念层面的突破。
- **匿名描述**是一个非常优雅的概念：去掉具体对象只保留抽象逻辑。
- 认知科学与计算机视觉的跨学科结合值得关注。

## 局限与展望
- 关系相似度的评估本身缺乏明确的ground truth，主观性较强。
- 匿名描述的生成质量仍有提升空间。
- 未来可探索在推理和创意生成中的更多应用。

## 相关工作与启发
- **vs CLIP/DINO**: 专注于属性级别的语义匹配，无法捕捉关系相似度。relsim通过匿名描述训练填补了这一空白。
- **vs NIGHTS**: 关注中层感知相似度，仍是属性驱动的。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全新问题定义，跨学科视角独特
- 实验充分度: ⭐⭐⭐⭐ 多模型对比+应用展示，但缺少大规模用户研究
- 写作质量: ⭐⭐⭐⭐⭐ 叙事引人入胜，认知科学背景丰富
- 价值: ⭐⭐⭐⭐⭐ 揭示了视觉AI的根本性盲区，影响深远

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs](../../ICML2026/multimodal_vlm/gated_relational_alignment_via_confidence-based_distillation_for_efficient_vlms.md)
- [\[ACL 2026\] Structured and Abstractive Reasoning on Multi-modal Relational Knowledge Images](../../ACL2026/multimodal_vlm/structured_and_abstractive_reasoning_on_multi-modal_relational_knowledge_images.md)
- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[AAAI 2026\] The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations](../../AAAI2026/multimodal_vlm/the_triangle_of_similarity_a_multi-faceted_framework_for_comparing_neural_networ.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/multimodal_vlm/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)

</div>

<!-- RELATED:END -->
