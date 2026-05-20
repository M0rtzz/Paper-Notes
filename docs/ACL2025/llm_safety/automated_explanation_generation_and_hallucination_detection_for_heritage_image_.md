---
title: >-
  [论文解读] Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval
description: >-
  [ACL 2025][LLM安全][文化遗产图像检索] 本文针对文化遗产图像检索任务，提出了一个结合自动解释生成和幻觉检测的框架，利用视觉语言模型为检索结果生成可解释的文本描述，同时通过领域知识约束的幻觉检测机制确保描述的事实准确性，在多个文化遗产数据集上验证了方法的有效性。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "文化遗产图像检索"
  - "自动解释生成"
  - "幻觉检测"
  - "视觉语言模型"
  - "跨模态检索"
---

# Automated Explanation Generation and Hallucination Detection for Heritage Image Retrieval

**会议**: ACL 2025  
**arXiv**: 无公开预印本  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 文化遗产图像检索, 自动解释生成, 幻觉检测, 视觉语言模型, 跨模态检索

## 一句话总结
本文针对文化遗产图像检索任务，提出了一个结合自动解释生成和幻觉检测的框架，利用视觉语言模型为检索结果生成可解释的文本描述，同时通过领域知识约束的幻觉检测机制确保描述的事实准确性，在多个文化遗产数据集上验证了方法的有效性。

## 研究背景与动机

**领域现状**：文化遗产数字化是计算机视觉和NLP的重要应用方向。博物馆、图书馆和档案馆拥有大量文化遗产图像（如绘画、雕塑、建筑照片、手稿），需要高效的检索系统帮助研究者和公众查找特定内容。现有的文化遗产图像检索主要依赖基于视觉特征的方法（如CNN/ViT提取的全局特征进行匹配）或基于人工标注元数据（如艺术家、年代、风格标签）的文本检索。

**现有痛点**：传统图像检索系统只返回检索结果排序，不告诉用户"为什么这张图像被检索出来"，即缺乏解释性。在文化遗产领域这一问题尤为突出：一个研究者搜索"巴洛克风格的圣母图像"，系统返回了一幅画，但研究者需要知道这幅画被匹配的原因是因为其构图风格、色彩运用还是主题内容。此外，当使用VLM生成图像描述时，模型容易产生幻觉——可能错误地将一幅文艺复兴时期的作品描述为"巴洛克风格"，或者编造不存在的细节。

**核心矛盾**：文化遗产领域需要高度准确的描述（如年代、流派、材料等元数据必须精确），但通用VLM在这一专业领域的幻觉率很高，因为其预训练数据中文化遗产相关的标注远不如通用场景充分。

**本文目标**：（1）为文化遗产图像检索结果自动生成可解释的文本描述；（2）设计领域适应的幻觉检测机制，确保描述的事实准确性。

**切入角度**：作者将问题分解为"先生成后验证"的两阶段方法——先利用VLM的生成能力获得丰富的文本描述，再利用文化遗产知识库进行事后的幻觉校正。

**核心 idea**：通过领域知识图谱约束的幻觉检测-修正循环，将通用VLM的生成能力安全地迁移到高精度要求的文化遗产领域。

## 方法详解

### 整体框架
框架分为三个核心模块：（1）跨模态检索模块——基于CLIP等视觉语言模型进行图像-文本对齐和检索；（2）解释生成模块——利用多模态LLM为检索结果生成结构化的文本解释（包括视觉描述、风格分析、匹配理由）；（3）幻觉检测与修正模块——利用文化遗产知识图谱对生成的描述进行事实核查和修正。输入为用户查询（文本或图像），输出为检索结果排序和每个结果的可解释文本描述。

### 关键设计

1. **领域适应的跨模态检索（Domain-Adapted Cross-Modal Retrieval, DACR）**:

    - 功能：提升通用视觉语言模型在文化遗产领域的检索精度
    - 核心思路：基于预训练的CLIP模型，使用文化遗产领域的图像-文本对进行轻量级的矬微调（LoRA微调，仅调整约2%的参数）。训练数据来自多个公开的博物馆数据库（如WikiArt、Met Museum Open Access、Europeana）。关键是构建高质量的微调数据——除了使用现有的元数据-图像对，还利用GPT-4V对文化遗产图像生成丰富的描述，经人工验证后加入训练集。引入了领域特定的负样本挖掘策略：在同一和相近年代/流派中选择容易混淆的图像对作为hard negatives，迫使模型学习细粒度的风格和内容区分。
    - 设计动机：通用CLIP在文化遗产领域的检索性能显著低于通用场景（约20个百分点的差距），因为预训练数据中文化遗产内容占比极低

2. **结构化解释生成器（Structured Explanation Generator, SEG）**:

    - 功能：为每个检索结果生成多维度的可解释文本
    - 核心思路：使用多模态LLM（如LLaVA或GPT-4V）为每个检索结果生成结构化的解释，包含四个维度——（a）视觉描述：描述图像中可见的视觉元素（主题、构图、色彩）；（b）风格分析：识别艺术风格、流派和技法；（c）匹配理由：解释为什么该图像与查询匹配（基于共享的视觉/语义特征）；（d）元数据推断：推断可能的年代、产地和材料。使用结构化prompt模板引导LLM按照固定格式输出，每个维度约3-5句话。生成时特意设置较低的temperature（0.3）以减少生成的随机性和创造性（在这个场景下创造性=幻觉）。
    - 设计动机：非结构化的自由描述难以系统性地验证和展示；结构化输出使得每个维度可以独立地进行幻觉检测

3. **知识图谱约束的幻觉检测与修正（KG-Constrained Hallucination Detection and Correction, KGHDC）**:

    - 功能：检测生成描述中的事实错误并进行修正
    - 核心思路：构建了一个文化遗产领域知识图谱（Heritage-KG），包含艺术风格的时间线（如"巴洛克风格"的起止年代）、材料-技法的兼容关系（如"油画布"不可能出现在宋朝中国画上）、主题-流派的关联（如"圣经题材"更常见于西方宗教画）。幻觉检测采用三步策略：（a）实体提取——从生成的描述中提取所有事实性声明（年代、风格、材料等）；（b）知识图谱验证——将提取的声明与Heritage-KG中的事实进行一致性检查；（c）修正建议——对于不一致的声明，从KG中检索最可能的正确信息并替换。修正后的描述标记修正痕迹（如"[原文：巴洛克 → 修正：洛可可]"），帮助用户判断描述的可信度。
    - 设计动机：通用VLM在文化遗产领域的幻觉率达30-40%，直接展示未经验证的描述在学术场景中是不可接受的

### 损失函数 / 训练策略
跨模态检索模块的LoRA微调使用InfoNCE对比学习损失，batch size 256，学习率1e-4，训练10个epoch。幻觉检测模块不需要训练，基于规则和知识图谱匹配。解释生成使用预训练VLM的零样本能力。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(DACR+SEG+KGHDC) | CLIP原始 | DACR only | Recall@10 |
|--------|------|---------------------|---------|----------|----------|
| WikiArt | mAP | 68.5 | 48.2 | 65.3 | 79.2 |
| Met-OA | mAP | 62.3 | 41.7 | 59.1 | 73.8 |
| Europeana | mAP | 58.7 | 38.5 | 55.2 | 71.5 |
| 解释可接受率 | Human eval | 84.6% | - | - | - |
| 幻觉检出率 | Precision | 78.3% | - | - | - |

### 消融实验

| 配置 | 解释可接受率 | 幻觉率 | 说明 |
|------|-----------|--------|------|
| SEG + KGHDC | 84.6% | 12.3% | 完整方案 |
| SEG only (无幻觉检测) | 68.2% | 35.7% | 幻觉率高，不可接受 |
| SEG + 简单规则检测 | 76.5% | 21.8% | 简单规则不够精确 |
| SEG + KGHDC (无修正) | 79.1% | 18.5% | 只检测不修正 |
| 低temperature (0.1) | 80.3% | 15.2% | 生成更保守但描述单一 |
| 高temperature (0.7) | 62.1% | 42.3% | 生成丰富但幻觉严重 |

### 关键发现
- 知识图谱约束的幻觉检测将幻觉率从35.7%降低到12.3%，减少了约23个百分点，说明领域知识对于事实校正至关重要
- 解释可接受率84.6%意味着约85%的生成描述在经过幻觉检测后被人类专家判断为"可用的"，这在专业领域是一个不错的结果
- 领域适应微调（DACR）将检索mAP提升了约20个百分点（48.2→68.5），说明通用模型与文化遗产领域的gap确实很大
- temperature是影响幻觉率的最敏感超参数（0.1→0.7幻觉率从15%飙升到42%），在高准确性要求场景中低temperature是必需的

## 亮点与洞察
- "先生成后验证"的两阶段策略在专业领域VLM应用中很有普适性——它允许系统利用VLM的强大生成能力，同时通过领域知识作为安全网防止幻觉。这一模式可以迁移到医学影像报告、法律文件分析等所有需要高准确性的领域
- 文化遗产知识图谱（Heritage-KG）的构建本身就是一个有价值的贡献，可以被其他文化遗产AI研究复用
- 修正痕迹标记的设计体现了对用户信任的考量——让用户看到哪些内容被修正了，比直接给出修正后的结果更有利于建立信任

## 局限与展望
- Heritage-KG的覆盖范围有限，主要覆盖西方艺术，对亚洲、非洲等文化遗产的知识不足
- 幻觉检测的精确率78.3%意味着仍有约22%的虚假检出（将正确描述标记为幻觉），这可能导致正确信息的损失
- 当生成的描述涉及主观判断（如"这幅画表达了忧郁的情感"）时，知识图谱无法进行事实验证
- 未来可以引入专家反馈循环来持续改进幻觉检测的精度，并扩展Heritage-KG的覆盖范围

## 相关工作与启发
- **vs CHIA (Cultural Heritage Image Analysis)**: 传统的文化遗产图像分析侧重于分类和检测，不涉及解释生成；本文首次将可解释性引入文化遗产检索
- **vs GRACE (Grounded Retrieval and Caption Enhancement)**: GRACE为图像检索生成caption，但没有领域特定的幻觉检测；本文的KGHDC模块是关键区别
- **vs Woodpecker (Yin et al., 2023)**: Woodpecker提出了VLM幻觉检测的通用框架；本文将幻觉检测专门化到文化遗产领域，利用领域KG提供更精确的验证

## 评分
- 新颖性: ⭐⭐⭐⭐ 文化遗产+可解释检索+幻觉检测的组合角度新颖
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估，包含人类专家评估，消融分析全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，应用场景描述生动
- 价值: ⭐⭐⭐⭐ 对文化遗产数字化和VLM可靠应用有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation](learning_auxiliary_tasks_improves_reference-free_hallucination_detection_in_open.md)
- [\[ACL 2025\] HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs](hd-ndes_neural_differential_equations_for_hallucination_detection_in_llms.md)
- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](../../NeurIPS2025/llm_safety/imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)
- [\[ACL 2025\] Fine-grained Hallucination Detection and Mitigation in Long-form Question Answering](localizing_and_mitigating_errors_in_long-form_question_answering.md)
- [\[ACL 2025\] Monitoring Decoding: Mitigating Hallucination via Evaluating the Factuality of Partial Response during Generation](monitoring_decoding_mitigating_hallucination_via_evaluating_the_factuality_of_pa.md)

</div>

<!-- RELATED:END -->
