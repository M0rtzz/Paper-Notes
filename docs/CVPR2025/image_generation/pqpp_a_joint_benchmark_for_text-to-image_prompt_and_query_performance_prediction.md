---
title: >-
  [论文解读] PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction
description: >-
  [CVPR 2025][图像生成][提示学习] 提出 PQPP，首个联合文本到图像生成和检索的 Prompt/Query 性能预测基准，包含超过 10K 查询和 160 万条人工标注，发现生成与检索的查询难度几乎不相关（Pearson 仅 0.135）。
tags:
  - CVPR 2025
  - 图像生成
  - 提示学习
  - 查询性能预测
  - 文生图
  - 人工标注基准
  - 扩散模型
---

# PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction

**会议**: CVPR 2025  
**arXiv**: [2406.04746](https://arxiv.org/abs/2406.04746)  
**代码**: [GitHub](https://github.com/Eduard6421/PQPP)  
**领域**: 图像生成 / 信息检索 (Image Generation / Information Retrieval)  
**关键词**: Prompt性能预测, 查询性能预测, 文生图, 人工标注基准, 扩散模型

## 一句话总结

提出 PQPP，首个联合文本到图像生成和检索的 Prompt/Query 性能预测基准，包含超过 10K 查询和 160 万条人工标注，发现生成与检索的查询难度几乎不相关（Pearson 仅 0.135）。

## 研究背景与动机

文本到图像生成（Text-to-Image Generation）已成为传统图像检索的有力替代，生成式扩散模型能产出高质量逼真图像。在信息检索领域，查询性能预测（QPP）是一个活跃的研究方向——预先判断一个查询对检索系统的难度，以便触发自动改写、资源分配等优化策略。

然而，在文本到图像生成领域，**Prompt 性能预测**是一个几乎未被探索的新问题：给定一个文本 prompt，能否预测扩散模型生成图像的质量？如果能预测某个 prompt 是困难的，系统可以主动引导用户优化 prompt、分配更多计算资源、或提前告知生成效果可能不佳。

现有最接近的工作（Bizzozzero et al.）使用自动生成的 ground-truth 评估 prompt 难度，但自动标注可能引入显著偏差。更关键的是，生成和检索两个任务使用相同的文本查询，但它们的"难度"特征是否相同？——这一问题从未被系统研究过。

本文的核心贡献是建立首个**基于人工标注的联合基准 PQPP**，包含 10,200 条文本查询，同时标注了生成和检索的性能，从而可以跨任务比较查询难度，并评估各种预测器。

## 方法详解

### 整体框架

PQPP 的构建流程：（1）从 MS COCO 的约 59 万条 caption 中通过 k-means 聚类选取 10,000 条 + DrawBench 的 200 条 = 10,200 条查询；（2）用两个生成模型（SDXL、GLIDE）分别生成 2 张图 → 收集 24.7 万条人工标注；（3）用两个检索模型（CLIP、BLIP-2）检索图像 → 收集 139 万条人工标注；（4）设计并评估多种预测器。

### 关键设计

1. **Prompt 性能标注体系（Generation Assessment）**:
    - 功能：为每个 prompt 建立基于人工判断的生成质量 ground-truth
    - 核心思路：对每个 prompt，由 SDXL 和 GLIDE 各生成 2 张图，加上 MS COCO 原始图作为校准控制，共 5 张图。147 位标注者在自定义 Web 界面上为每张图选择高相关（score=2）、低相关（1）、不相关（0）或不真实（-1）。采用 Fleiss' $\kappa$ 进行质量控制（$\kappa > 0.4$），并通过多数投票去除异常标注。最终 Prompt 性能 HBPP = 其所有生成图的平均相关分
    - 设计动机：自动评估（如 CLIP score）可能与人类感知不一致，人工标注提供了更可靠的 ground-truth

2. **Query 性能标注体系（Retrieval Assessment）**:
    - 功能：为每个 query 建立基于人工判断的检索质量 ground-truth
    - 核心思路：使用 CLIP ViT-B/32 和 BLIP-2 ViT-Large 进行检索。通过 Sentence-BERT 余弦相似度（阈值 0.7）预筛潜在相关图像（限制每个 query 最多 2000 张），再由 100 位标注者进行相关/不相关的二元判断（每张图 2 人标注）。最终保留 53 万张相关图像（约去掉了 2/3 的候选图像），使用 P@10 和 RR 度量查询性能
    - 设计动机：需要大规模人工验证来确保检索 ground-truth 的可靠性

3. **性能预测器（Performance Predictors）**:
    - 功能：在生成/检索前或后预测 prompt/query 的性能
    - 核心思路：Pre-predictors 包括文本特征（词数、WordNet synsets 数）和 fine-tuned BERT；Post-predictors 包括 fine-tuned Long-CLIP（回归头预测生成相关度/分类头预测检索相关度）、NQC 和 WIG（基于检索分数分布统计）。训练集 6,080 / 验证集 2,040 / 测试集 2,080
    - 设计动机：对比 pre/post 预测器的表现，为未来研究建立 baseline

### 损失函数 / 训练策略

- **BERT fine-tuning**: 在预训练 BERT-base-cased 上添加回归头，针对 HBPP/P@10/RR 进行训练
- **CLIP fine-tuning**: 在 Long-CLIP ViT-B/32 上添加回归/分类头，输入 (query, image) 对
- **评估指标**: Pearson 相关系数和 Kendall $\tau$ 衡量预测值与 ground-truth 的相关性

## 实验关键数据

### 主实验

生成与检索难度相关性（全集，Table 2）：

| 比较 | Pearson | Kendall $\tau$ |
|------|---------|----------------|
| HBPP vs P@10 | 0.135 | 0.093 |
| HBPP vs RR | 0.072 | 0.048 |
| P@10 vs RR | 0.560 | 0.512 |

生成性能预测（SDXL → HBPP，部分结果）：

| 预测器 | 类别 | Pearson | Kendall $\tau$ |
|--------|------|---------|----------------|
| #synsets | Pre | 0.162 | 0.113 |
| BERT fine-tuned | Pre | 0.329 | 0.234 |
| CLIP fine-tuned | Post | 0.369 | 0.262 |

### 消融实验

| 跨任务实验 | 说明 | Pearson |
|-----------|------|---------|
| SDXL 预测器 → GLIDE | 跨模型迁移 | 有所下降但仍显著 |
| 生成预测器 → 检索任务 | 跨任务迁移 | 几乎无效 |
| CLIP预测器 → BLIP-2 | 检索跨模型 | 保持有效 |

### 关键发现

- 生成和检索的查询难度几乎正交（Pearson 仅 0.135），证明 prompt 性能预测需要专门研究
- 监督式 pre-predictor (fine-tuned BERT) 在生成任务上可接近 post-predictor 的表现
- 难以生成但好检索的 prompt 通常描述特定运动姿势（MS COCO 中体育图片丰富）；难以检索但好生成的 prompt 通常包含具体属性描述（生成模型善于细节但检索难匹配）

## 亮点与洞察

- **生成与检索的正交性发现是最有价值的贡献**：看似相似的两个任务（都是 text → image），其查询难度特征截然不同。这意味着不能简单复用信息检索中的 QPP 方法来处理生成场景
- **超过 160 万条人工标注的规模令人印象深刻**：247 位标注者参与，标注流程设计严谨（控制集、校准图、多数投票去异常），为后续研究提供了高质量基准

## 局限与展望

- 仅使用 SDXL 和 GLIDE 两个生成模型，未覆盖最新的 DiT-based 模型
- 数据来源以 MS COCO caption 为主，可能不够代表真实用户 prompt 的多样性
- 检索和生成任务使用相同数据集（MS COCO），可能引入数据偏差
- 未来可扩展到视频生成、3D 生成等更多文本引导生成任务

## 相关工作与启发

- **vs Bizzozzero et al.**: 唯一同类工作但使用自动标注 ground-truth，可能引入偏差；PQPP 是首个基于人工判断的 prompt 性能预测基准
- **vs 传统文本 QPP**: 传统 QPP 方法专注于文本检索，多模态场景需要独立研究
- **vs Human Feedback 工作 (ImageReward, HPS)**: 这些工作关注改善生成质量，PQPP 关注预测查询难度——是互补关系

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个联合基准，生成-检索正交性发现具有开创意义
- 实验充分度: ⭐⭐⭐⭐ 160 万标注，多预测器对比，跨模型/跨任务实验
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，数据收集流程详尽
- 价值: ⭐⭐⭐⭐ 填补了文生图 prompt 难度预测领域的空白

<!-- RELATED:START -->

## 相关论文

- [Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [SyncVP: Joint Diffusion for Synchronous Multi-Modal Video Prediction](syncvp_joint_diffusion_for_synchronous_multi-modal_video_prediction.md)
- [Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing](towards_scalable_human-aligned_benchmark_for_text-guided_image_editing.md)
- [Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [Mono2Stereo: A Benchmark and Empirical Study for Stereo Conversion](mono2stereo_a_benchmark_and_empirical_study_for_stereo_conversion.md)

<!-- RELATED:END -->
