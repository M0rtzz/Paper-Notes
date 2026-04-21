---
title: >-
  [论文解读] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations
description: >-
  [NeurIPS 2025][3D视觉][open-vocabulary] 提出 OpenLex3D，一个面向开放词汇 3D 场景表示的分层评测基准，在 Replica、ScanNet++、HM3D 三个数据集上提供 13 倍于原始标注的丰富语言标签，支持开放集 3D 语义分割和目标检索两项任务评测。
tags:
  - NeurIPS 2025
  - 3D视觉
  - open-vocabulary
  - 场景理解
  - benchmark
  - 图像分割
  - object retrieval
---

# OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations

**会议**: NeurIPS 2025  
**arXiv**: [2503.19764](https://arxiv.org/abs/2503.19764)  
**代码**: [项目主页](https://openlex3d.github.io/)  
**领域**: 3D 视觉 / 开放词汇理解  
**关键词**: open-vocabulary, 3D scene understanding, benchmark, semantic segmentation, object retrieval

## 一句话总结
提出 OpenLex3D，一个面向开放词汇 3D 场景表示的分层评测基准，在 Replica、ScanNet++、HM3D 三个数据集上提供 13 倍于原始标注的丰富语言标签，支持开放集 3D 语义分割和目标检索两项任务评测。

## 研究背景与动机
**领域现状**：开放词汇语言模型极大推动了 3D 场景理解，使得用户可以用自然语言与场景交互。现有方法（如 LERF、OpenScene、ConceptFusion 等）已展示了令人印象深刻的 demo。

**现有痛点**：现有评测仍依赖封闭集语义标注（如 ScanNet 的 20 类、Replica 的 88 类），无法反映真实自然语言查询的丰富性和模糊性。

**核心矛盾**：方法号称"开放词汇"但评测用"封闭集"，导致方法的真实能力被高估。例如查询"椅子"时正确，但查询"办公旋转椅"或"扶手椅"时可能失败。

**切入角度**：构建真正捕获语言多样性的标注数据集，引入同义词类别和细粒度描述。

**核心 idea**：用 13× 更丰富的语言标注重新评估开放词汇 3D 方法，暴露现有方法的真实短板。

## 方法详解

### 整体框架
OpenLex3D 的评测设计分为三个层级（Tiered）：

1. **Tier 1：标准语义分割** — 使用原始数据集类别，作为基准线
2. **Tier 2：同义词扩展分割** — 每个类别增加同义词（如 chair → office chair / swivel chair / armchair），测试方法对同义词的鲁棒性
3. **Tier 3：自由形式检索** — 用户提供任意自然语言描述（如"靠窗的红色抱枕"），测试细粒度检索能力

### 关键设计

1. **标注流程**

    - 3D 网格上逐面片标注，非 2D 投影
    - 每个对象提供：canonical name、多个同义词、外观描述、位置描述
    - 标注者指引确保一致性
    - 最终每场景平均标签数为原数据集的 **13 倍**

2. **评测任务**

    - **开放集 3D 语义分割**：给定文本查询，在 3D 点云/网格上预测语义标签
    - **目标检索**：给定自然语言描述，定位 3D 场景中的目标

3. **评测指标**

    - mIoU（语义分割标准），分 Tier 1/2/3 报告
    - Recall@K（目标检索），K=1,3,5
    - Feature Precision：度量特征空间中语义一致性

### 数据规模

| 数据集 | 场景数 | 原始类别数 | OpenLex3D 标签数 |
|--------|--------|-----------|------------------|
| Replica | 8 | 88 | ~1150 |
| ScanNet++ | 10 | 100 | ~1300 |
| HM3D | 10 | 30 | ~390 |

## 实验关键数据

### 主实验 — 3D 语义分割 mIoU (%)

| 方法 | Replica T1 | Replica T2 | Replica T3 | ScanNet++ T1 | ScanNet++ T2 |
|------|-----------|-----------|-----------|-------------|-------------|
| LERF | 41.2 | 28.7 | 15.3 | 38.5 | 24.1 |
| OpenScene | 45.8 | 31.2 | 18.6 | 42.3 | 27.8 |
| ConceptFusion | 43.5 | 29.8 | 16.9 | 40.1 | 25.6 |
| LangSplat | 48.1 | 33.5 | 20.2 | 44.7 | 29.3 |
| OpenMask3D | 50.3 | 35.1 | 22.4 | 46.9 | 31.7 |

### 目标检索 Recall@1 (%)

| 方法 | Replica 标准 | Replica 同义词 | Replica 自由描述 |
|------|-------------|---------------|-----------------|
| LERF | 52.3 | 38.1 | 21.5 |
| OpenScene | 58.7 | 42.5 | 25.8 |
| ConceptFusion | 55.1 | 40.2 | 23.1 |
| LangSplat | 61.2 | 45.8 | 28.3 |
| OpenMask3D | 64.5 | 48.2 | 30.7 |

### 消融实验 — Tier 级别对性能的影响（平均 mIoU 降幅）

| 方法 | T1→T2 降幅 | T1→T3 降幅 |
|------|-----------|-----------|
| LERF | -30.3% | -62.9% |
| OpenScene | -31.9% | -59.4% |
| ConceptFusion | -31.5% | -61.1% |
| LangSplat | -30.4% | -58.0% |
| OpenMask3D | -30.2% | -55.5% |

### 关键发现
- **所有方法从 T1 到 T3 的 mIoU 下降 55-63%**，说明现有方法对语言变异性极其脆弱
- 同义词查询（T2）已导致约 30% 性能下降，表明方法过度拟合封闭集标签
- OpenMask3D 表现最优但仍有巨大提升空间（T3 仅 ~22% mIoU）
- 分割方法普遍在 Feature Precision 上表现较差——特征空间中语义相似的查询未被映射到相近区域
- 基于 3D Gaussian Splatting 的方法（LangSplat）在细粒度描述上优于 NeRF 系方法

## 亮点与洞察
- **诊断价值极高**：暴露了开放词汇 3D 方法"虚假繁荣"的问题
- **分层设计实用**：T1/T2/T3 逐步提升难度，便于定位方法的薄弱环节
- **标注质量高**：3D 网格面片级标注 + 多人审核
- **社区基础设施**：公开数据集、评测代码和排行榜

## 局限与展望
- 场景数量相对有限（28 个场景），可能不够覆盖所有室内/室外变化
- 标注主要面向室内场景，室外/大尺度场景缺乏
- 自由形式描述（T3）的标注主观性较强，评分可能有偏差
- 未包含动态场景或时序查询

## 相关工作与启发
- **ScanRefer / ReferIt3D**：3D 引用定位先驱
- **LERF (ICCV 2023)**：CLIP 特征场
- **OpenScene (CVPR 2023)**：开放词汇 3D 分割
- 启发：开放词汇评测需要从"类别名"升级到"自然语言"

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个真正开放词汇的 3D 评测基准
- 实验充分度: ⭐⭐⭐⭐⭐ 多方法多数据集多层级全面评测
- 写作质量: ⭐⭐⭐⭐ 动机充分，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 为社区提供急需的评测基础设施

<!-- RELATED:START -->

## 相关论文

- [OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](../../CVPR2025/3d_vision/masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [From Objects to Anywhere: A Holistic Benchmark for Multi-level Visual Grounding in 3D Scenes](from_objects_to_anywhere_a_holistic_benchmark_for_multi-level_visual_grounding_i.md)

<!-- RELATED:END -->
