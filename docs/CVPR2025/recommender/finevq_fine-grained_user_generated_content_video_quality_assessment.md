---
title: >-
  [论文解读] FineVQ: Fine-Grained User Generated Content Video Quality Assessment
description: >-
  [CVPR 2025][推荐系统][视频质量评估] 构建了首个大规模细粒度 UGC 视频质量评估数据库 FineVD（6104 视频、80 万+评分、6 个维度），并提出基于大型多模态模型的 FineVQ 方法，以一个模型同时实现质量评级、质量评分和质量归因三种能力，在 FineVD 和其他 UGC-VQA 数据集上达到 SOTA。
tags:
  - "CVPR 2025"
  - "推荐系统"
  - "视频质量评估"
  - "用户生成内容"
  - "细粒度评估"
  - "大型多模态模型"
  - "指令微调"
---

# FineVQ: Fine-Grained User Generated Content Video Quality Assessment

**会议**: CVPR 2025  
**arXiv**: [2412.19238](https://arxiv.org/abs/2412.19238)  
**代码**: [https://github.com/IntMeGroup/FineVQ](https://github.com/IntMeGroup/FineVQ)  
**领域**: 推荐系统  
**关键词**: 视频质量评估, 用户生成内容, 细粒度评估, 大型多模态模型, 指令微调

## 一句话总结
构建了首个大规模细粒度 UGC 视频质量评估数据库 FineVD（6104 视频、80 万+评分、6 个维度），并提出基于大型多模态模型的 FineVQ 方法，以一个模型同时实现质量评级、质量评分和质量归因三种能力，在 FineVD 和其他 UGC-VQA 数据集上达到 SOTA。

## 研究背景与动机

**领域现状**：UGC 视频数量爆发式增长，视频质量评估（VQA）对平台的内容监控、优化和推荐至关重要。现有 UGC-VQA 方法（VSFA、SimpleVQA、DOVER 等）通常只输出一个整体质量分数。

**现有痛点**：单一的整体质量评分无法满足多样化的下游应用需求。视频处理需要知道具体哪个维度（噪声？模糊？色彩？）有问题；推荐系统需要多维度的质量信号；内容创作者需要了解具体的质量缺陷。现有数据库也缺乏多维度的细粒度标注。

**核心矛盾**：应用场景需要细粒度、多维度的质量信息，但现有数据库只提供粗粒度的整体评分标注，导致模型也只能输出单一分数。同时，用不同模型分别评估各维度效率低且一致性差。

**本文目标**：(1) 建立首个包含多维度细粒度质量标注的大规模 UGC-VQA 数据库；(2) 设计一个"一个模型搞定所有"的细粒度质量评估方法。

**切入角度**：借助大型多模态模型（LMM）强大的视觉理解和文本生成能力，通过指令微调使单个模型同时具备质量评分（回归）、质量评级（分类）和质量描述（生成）的多任务能力。

**核心 idea**：构建 6 维度（色彩、噪声、伪影、模糊、时域、总体）的细粒度标注数据库，基于 InternVL 框架通过空间+运动双视觉编码器和 LoRA 微调，训练一个 one-for-all 的视频质量评估模型。

## 方法详解

### 整体框架
输入 UGC 视频和用户提示（prompt），经过三路特征编码：(1) 图像编码器（InternViT）从均匀采样的 8 帧中提取空间特征；(2) 运动编码器（SlowFast）从整段视频提取时域运动特征；(3) 文本分词器编码用户提示。三路 token 拼接后送入预训练大语言模型（InternLM）生成质量相关的回答——可以是质量等级（评级）、质量分数（评分）或质量描述（归因）。

### 关键设计

1. **FineVD 数据库构建**:

    - 功能：提供首个大规模多维度细粒度 UGC-VQA 标注数据集
    - 核心思路：从 B 站收集 6104 个 UGC 视频（含点播和直播），22 名专业标注者在实验室环境下从色彩、噪声、伪影、模糊、时域、总体 6 个维度进行 5 级质量评分，总计 80 万+评分。同时标注失真类型标签。再用 GPT-4 生成质量相关的 QA 对并由人工校正，构建质量评级/评分/描述三种任务的训练数据
    - 设计动机：现有数据库只有整体评分，无法支撑细粒度质量评估研究。实验室标注比众包标注质量更可控

2. **双视觉编码器 + LoRA 微调**:

    - 功能：在不过度增加参数的前提下，让预训练 LMM 具备质量感知能力
    - 核心思路：空间特征用 InternViT 提取（从 8 帧），运动特征用 SlowFast 网络从全视频提取。两路特征分别通过 2 层 MLP 投影到语言空间。在图像编码器和 LLM 上都施加 LoRA 权重进行低秩适配，保留大模型通用能力的同时注入质量评估领域知识
    - 设计动机：稀疏帧采样不足以捕获时域质量（如抖动、卡顿），需要额外的运动特征提取。LoRA 避免全量微调的高成本，同时保持对不同质量维度的灵活性

3. **指令微调的多任务统一**:

    - 功能：用单个模型同时处理质量评级（分类）、质量评分（回归）和质量描述（文本生成）
    - 核心思路：设计不同类型的指令-回答对（QA pairs）：评级任务要求模型输出"good/fair/poor"等级别；评分任务要求输出 0-100 数值；描述任务要求生成自然语言描述质量问题。通过混合训练这三类 QA 对实现多任务统一
    - 设计动机：不同应用需要不同粒度的质量反馈。将它们统一到一个 LMM 框架中，既共享底层视觉理解能力，又通过指令区分任务类型

### 损失函数 / 训练策略
评分任务使用回归损失（MSE），评级和描述任务使用语言建模的交叉熵损失。训练采用两阶段策略：第一阶段冻结视觉编码器和 LLM，只训练投影层；第二阶段解冻 LoRA 权重进行端到端微调。

## 实验关键数据

### 主实验（FineVD 评分任务）

| 方法 | Overall SRCC | Overall PLCC | Noise SRCC | Blur SRCC |
|------|-------------|-------------|-----------|----------|
| FineVQ | **0.8834** | **0.8891** | **0.8444** | **0.8711** |
| DOVER | 0.8422 | 0.8393 | 0.8018 | 0.8404 |
| SimpleVQA | 0.8311 | 0.8358 | 0.8070 | 0.8466 |
| FAST-VQA | 0.8348 | 0.8474 | 0.8093 | 0.8352 |
| VIDEVAL（传统） | 0.7310 | 0.7307 | 0.6912 | 0.7610 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| DNN 方法（per-dim） | 各维度单独训练 | 每个维度单独模型 |
| FineVQ（one-for-all） | 6 维度统一 | 单模型多维度，性能更优 |
| 通用 LMM（零样本） | 质量归因准确率较低 | InternVL2 零样本 ~50-70% |
| FineVQ 质量归因 | 各维度 87-95% | 微调后显著提升 |

### 关键发现
- FineVQ 以 one-for-all 方式在所有 6 个维度上均超越单独训练的 DNN-based 方法（DOVER、SimpleVQA），证明统一模型的优势
- 运动特征编码器的引入对时域维度质量评估贡献显著
- 通用 LMM（如 InternVL2、Qwen2-VL）在质量归因任务上零样本表现不佳（~50-70%），但经 FineVQ 微调后可达 87-95%
- 在外部数据集（LSVQ、KoNViD-1k 等）上也展示了竞争性的跨数据集泛化能力

## 亮点与洞察
- **首个多维度细粒度 VQA 数据库**：FineVD 填补了 UGC-VQA 领域缺乏细粒度标注的空白，6 维度 × 6104 视频 × 22 标注者的规模足以支撑深度学习研究
- **one-for-all 设计哲学**：用指令微调统一评分/评级/描述三个任务到一个模型中，避免了多模型维护的复杂性，且共享表示反而提升了各任务性能
- **LMM 在底层视觉任务的新应用**：展示了 LMM 在传统底层视觉质量评估任务上的潜力，通过少量 LoRA 参数即可获得显著的质量感知能力

## 局限与展望
- 数据库视频全部来源于 B 站单一平台，可能存在平台特定的内容和质量分布偏差
- 6 个质量维度是预定义的，未探索更细粒度（如具体失真类型的严重程度）或自适应的维度发现
- 模型推理需要 LMM 级别的计算资源，不适合边缘设备或实时应用场景
- 评分任务的 PLCC 在噪声维度（0.7986）相对其他维度偏低，说明噪声质量评估仍有提升空间

## 相关工作与启发
- **vs DOVER**: DOVER 也考虑了技术质量和美学质量两个维度，但只有两维且不提供评级/描述能力。FineVQ 扩展到 6 维且具备多任务能力
- **vs Q-Align**: Q-Align 也用 LMM 做质量评估，但主要面向图像且只输出整体分数。FineVQ 针对视频、多维度且集成了运动编码器
- **vs SimpleVQA**: SimpleVQA 使用预训练特征+回归头架构简单高效，但缺乏质量描述和归因能力。FineVQ 在评分性能上也更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个细粒度 VQA 数据库+基于 LMM 的多任务统一框架，填补领域空白
- 实验充分度: ⭐⭐⭐⭐ 多数据集评估、与多种 baseline 对比、质量评级/评分/描述三任务均有评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据库构建过程描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ 数据库和模型均已开源，对 UGC 视频质量评估领域有较高实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](../../AAAI2026/recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[ECCV 2024\] AID-AppEAL: Automatic Image Dataset and Algorithm for Content Appeal Enhancement and Assessment Labeling](../../ECCV2024/recommender/aid-appeal_automatic_image_dataset_and_algorithm_for_content_appeal_enhancement_.md)
- [\[ACL 2025\] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences](../../ACL2025/recommender/lotus_a_leaderboard_for_detailed_image_captioning_from_quality_to_societal_bias_.md)
- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](../../ACL2026/recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)

</div>

<!-- RELATED:END -->
