---
title: >-
  [论文解读] EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis
description: >-
  [CVPR2026][视觉情感分析] 构建 EmoVerse——首个同时覆盖 CES（Mikels 8 类离散情感）和 DES（1024 维连续情感空间）的大规模可解释视觉情感数据集（219K+ 图像），提出 B-A-S（Background-Attribute-Subject）三元组知识图谱标注体系和 Annotation & Verification Pipeline（Gemini/GPT-4o + EmoViT + CoT Critic Agent），并基于 Qwen2.5-VL-3B 微调实现 1024 维 DES 投射与情感归因解释。
tags:
  - CVPR2026
  - 视觉情感分析
  - 情感表示数据集
  - 知识图谱
  - 可解释性
  - 多模态大模型
---

# EmoVerse: A MLLMs-Driven Emotion Representation Dataset for Interpretable Visual Emotion Analysis

**会议**: CVPR2026  
**arXiv**: [2511.12554](https://arxiv.org/abs/2511.12554)  
**代码**: 待确认  
**领域**: multimodal_vlm  
**关键词**: 视觉情感分析, 情感表示数据集, 知识图谱, 可解释性, 多模态大模型

## 一句话总结
构建 EmoVerse——首个同时覆盖 CES（Mikels 8 类离散情感）和 DES（1024 维连续情感空间）的大规模可解释视觉情感数据集（219K+ 图像），提出 B-A-S（Background-Attribute-Subject）三元组知识图谱标注体系和 Annotation & Verification Pipeline（Gemini/GPT-4o + EmoViT + CoT Critic Agent），并基于 Qwen2.5-VL-3B 微调实现 1024 维 DES 投射与情感归因解释。

## 背景与动机

1. **领域现状**：视觉情感分析（Visual Emotion Analysis, VEA）旨在从图像中预测观者的情感反应。现有数据集（FI、EmoSet、Instagram 等）多采用离散情感分类（Mikels 8 类或 VAD 三维），标注维度单一。
2. **现有痛点**：(1) 缺乏开源的大规模可解释情感数据集——现有数据集只提供情感类别标签，不解释"为什么引发这种情感"；(2) 离散情感标签（CES）无法捕捉细粒度情感变化，连续表示（DES）的数据集几乎不存在；(3) 缺少 subject-level 的实例定位——不知道图像中哪个主体触发了哪种情感。
3. **核心矛盾**：VEA 领域急需可解释性和细粒度标注，但人工标注成本极高（1024 维连续空间不可能人工标注），传统众包方式无法覆盖 word-level、subject-level、CES、DES 四个维度。
4. **本文目标**：如何构建一个兼具 CES 和 DES、具有可解释性标注、且规模足够大的视觉情感数据集？
5. **切入角度**：利用 MLLM（Gemini 2.5、GPT-4o）做自动标注，配合多轮验证 pipeline 保证质量，引入知识图谱结构化情感归因。
6. **核心 idea**：B-A-S 三元组将情感分解为 Background（背景场景）、Attribute（视觉属性如颜色/光线）、Subject（主体对象），配合 Grounding DINO + SAM 实现 subject 定位，用 MLLM pipeline 完成标注-验证-修正闭环。

## 方法详解

### 整体框架
EmoVerse 的构建包括四个阶段：(1) 数据收集与清洗；(2) B-A-S 三元组标注 + CES/DES 生成；(3) 多轮验证与修正（Annotation & Verification Pipeline）；(4) Subject 实例定位（Grounding DINO + SAM）。最终在此数据集上微调 Qwen2.5-VL-3B 作为可解释情感分析模型。

### 关键设计

1. **B-A-S（Background-Attribute-Subject）三元组标注**:

    - 功能：将图像情感归因分解为三个维度的知识图谱结构
    - 核心思路：受知识图谱启发，每张图像标注为 $(B, A, S)$ 三元组——$B$ 描述场景背景（如"暴风雨中的海岸"）、$A$ 描述视觉属性（如"昏暗光线、冷色调"）、$S$ 描述关键主体（如"独自站立的人"）。三者共同解释情感触发原因
    - 设计动机：传统标注只给一个情感标签（如"sadness"），不解释原因。B-A-S 三元组让情感归因可追溯，支持下游可解释分析

2. **混合数据来源**:

    - 功能：从多个来源收集 219K+ 图像，确保情感多样性
    - 核心思路：(a) 现有情感数据集：EmoSet、EmoArt；(b) 通用数据集：Flickr30k（自然场景）；(c) 网络搜索：针对特定情感关键词爬取；(d) AIGC 生成：用 Seedream 模型按情感 prompt 生成约 25K 张图像，补充稀缺情感类别
    - 设计动机：单一来源的数据集存在偏置（如 EmoSet 以自然图为主），AIGC 生成可精准补充长尾情感分布

3. **Annotation & Verification Pipeline**:

    - 功能：利用 MLLM 自动标注并通过多轮验证保证标注质量
    - 核心思路：(a) **初标**：Gemini 2.5 和 GPT-4o 分别对图像生成情感标注（CES 类别 + DES 向量 + B-A-S 三元组）；(b) **情感验证**：用预训练的 EmoViT（情感分类专家模型）检验 CES 标签一致性；(c) **CoT Critic Agent**：对初标结果进行 Chain-of-Thought 批判性审查，将每条标注判定为 valid（保留）、revisable（可修正，返回重标）、discarded（丢弃）；(d) **人工抽检**：对 Critic Agent 的输出进行抽样人工验证
    - 设计动机：纯 MLLM 标注的噪声率不可忽视（尤其 DES 1024 维空间），多轮验证 + 专家模型交叉检查 + CoT 批判性审查可有效降噪

4. **Subject-Level 实例定位**:

    - 功能：为 B-A-S 中的 Subject 提供 bounding box 和 segmentation mask
    - 核心思路：(a) 将 B-A-S 三元组中的 Subject 文本描述输入 Grounding DINO，获取 bounding box；(b) 用 SAM（Segment Anything Model）基于 bbox prompt 生成像素级分割 mask
    - 设计动机：Subject-level 实例定位使模型能学习"图像中哪个区域/对象引发了哪种情感"，支持局部情感归因

5. **可解释情感模型：Qwen2.5-VL-3B 微调**:

    - 功能：在 EmoVerse 上微调多模态模型，实现 CES 分类 + DES 投射 + 情感归因文本生成
    - 核心思路：两轮微调——第一轮学习 CES/DES 预测，第二轮学习基于 B-A-S 生成情感归因解释。DES 通过 1024 维线性投射头实现。训练用交叉熵 Loss（CE Loss）
    - 设计动机：端到端多任务训练使模型同时具备情感预测和可解释性能力

### 数据集统计
- 总量：219K+ 图像
- CES 覆盖：Mikels 8 类（amusement, awe, contentment, excitement, anger, disgust, fear, sadness）
- DES 维度：1024 维连续情感空间
- 标注维度：word-level（情感词）+ subject-level（主体定位）+ CES + DES 全覆盖
- AIGC 生成：~25K 张（Seedream）

## 实验关键数据

### 数据集对比

| 数据集 | 图像数 | CES | DES | 可解释标注 | Subject 定位 |
|--------|--------|-----|-----|-----------|-------------|
| FI | 23K | ✓ | ✗ | ✗ | ✗ |
| Instagram | 42K | ✓ | ✗ | ✗ | ✗ |
| EmoSet | 118K | ✓ | ✗ | 部分 | ✗ |
| EmoArt | 80K | ✓ | ✗ | ✗ | ✗ |
| **EmoVerse** | **219K+** | **✓** | **✓** | **✓ (B-A-S)** | **✓ (bbox+mask)** |

### 关键发现
- **EmoVerse 是首个同时覆盖 CES 和 DES 的数据集**：现有所有数据集均无 DES 标注
- **B-A-S 三元组提升可解释性**：消融显示加入 B-A-S 后情感分类准确率和归因文本质量均有提升
- **AIGC 数据有效补充长尾**：去掉 Seedream 生成数据后，稀缺情感类别（disgust、fear）的分类性能明显下降
- **Annotation Pipeline 有效降噪**：CoT Critic Agent 过滤了约 15-20% 的低质量标注，人工抽检验证 pipeline 输出的准确率 > 90%
- **Qwen2.5-VL-3B 微调效果**：在 EmoVerse 测试集上 CES 分类准确率和 DES 投射相关性均优于基线方法

## 亮点与洞察
- **B-A-S 三元组是核心贡献**：将情感归因结构化为知识图谱风格的三元组，既便于自动标注又支持下游推理，比自由文本描述更规范
- **MLLM + 专家模型 + CoT Critic 的多轮验证**：这套 pipeline 为利用 MLLM 构建大规模标注数据集提供了可复用范式——不是简单地"用 GPT 标注"，而是有质量保证闭环
- **AIGC 补充长尾分布**：用生成模型按需生成特定情感图像是巧妙的数据增强思路，比单纯过采样更有效
- **Subject-level 定位**：Grounding DINO + SAM 的组合将情感分析从图像级推进到区域级，开辟了局部情感归因的新方向

## 局限与展望
- DES 1024 维的标注质量完全依赖 MLLM，缺少人工验证金标准——MLLM 对连续情感空间的理解可能存在系统性偏差
- Mikels 8 类 CES 体系相对粗粒度，未覆盖 surprise、neutral 等常见情感
- AIGC 生成的图像可能带有生成模型的风格偏见，与真实图像的情感表达存在 domain gap
- Qwen2.5-VL-3B 较小，更大模型（7B/72B）的表现未知
- 仅在 EmoVerse 自身测试集上评估，缺少与 FI、EmoSet 等外部数据集的跨数据集泛化实验
- CoT Critic Agent 的判定阈值如何选取未充分讨论

## 相关工作与启发
- **vs EmoSet**: EmoSet 是当前最大的视觉情感数据集（118K），但只有 CES 无 DES，无 subject 定位。EmoVerse 在规模、标注维度上全面超越
- **vs EmotionCLIP**: EmotionCLIP 通过对比学习做情感零样本分类，但不提供可解释归因。EmoVerse 的 B-A-S 三元组直接支持解释生成
- **vs SentiCap / ArtEmis**: 提供图像情感描述文本，但缺少结构化标注（B-A-S）和 subject 定位
- **vs Grounding DINO + SAM 的组合使用**：EmoVerse 证明了"文本描述 → 视觉定位"pipeline 在情感分析场景的有效性，可推广到其他主观感知任务

## 评分
- 新颖性: ⭐⭐⭐⭐ B-A-S 三元组和 CES+DES 双表示体系是首创，MLLM 多轮验证 pipeline 有方法论价值
- 实验充分度: ⭐⭐⭐ 缺少跨数据集泛化实验和更大模型验证
- 写作质量: ⭐⭐⭐⭐ 数据集构建流程描述详尽，pipeline 各环节动机清晰
- 价值: ⭐⭐⭐⭐ 填补了可解释视觉情感分析数据集的空白，B-A-S 标注体系和验证 pipeline 可复用性强

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](text-guided_fine-grained_video_anomaly_understanding.md)
- [\[CVPR 2026\] DINO-QPM: Adapting Visual Foundation Models for Globally Interpretable Image Classification](dino-qpm_adapting_visual_foundation_models_for_globally_interpretable_image_clas.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)

<!-- RELATED:END -->
