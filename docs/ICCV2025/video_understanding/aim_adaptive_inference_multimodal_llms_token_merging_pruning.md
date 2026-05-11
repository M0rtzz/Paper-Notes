---
title: >-
  [论文解读] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
description: >-
  [ICCV 2025][视频理解][多模态LLM] 提出 AIM，一种无需训练的多模态LLM自适应推理方法，通过LLM前基于相似度的视觉token迭代合并和LLM层内基于PageRank重要性的渐进token剪枝，实现6.8倍FLOPs削减同时保持性能，在长视频理解上同等计算量下甚至超越SOTA (+4.6 MLVU)。
tags:
  - ICCV 2025
  - 视频理解
  - 多模态LLM
  - 自适应推理
  - token合并
  - token剪枝
  - 视频理解效率
---

# AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

**会议**: ICCV 2025  
**arXiv**: [2412.03248](https://arxiv.org/abs/2412.03248)  
**代码**: [https://github.com/LaVi-Lab/AIM](https://github.com/LaVi-Lab/AIM)  
**领域**: 视频理解  
**关键词**: 多模态LLM, 自适应推理, token合并, token剪枝, 视频理解效率

## 一句话总结
提出 AIM，一种无需训练的多模态LLM自适应推理方法，通过LLM前基于相似度的视觉token迭代合并和LLM层内基于PageRank重要性的渐进token剪枝，实现6.8倍FLOPs削减同时保持性能，在长视频理解上同等计算量下甚至超越SOTA (+4.6 MLVU)。

## 研究背景与动机

**领域现状**：多模态LLM依赖大量视觉token（视频可达数千），计算开销巨大，限制了实时部署和长视频处理。

**现有痛点**：FastV和PDrop等方法仅在LLM特定层剪枝，缺乏灵活性；LLaVA-Prumerge仅在LLM前处理。无法自适应地适配不同的计算预算。

**核心 idea**：在LLM前合并相似token减少冗余 + 在LLM层内渐进剪枝不重要token，两个旋钮可灵活调节计算量。

## 方法详解

### 关键设计

1. **LLM前token合并**: 基于余弦相似度将相邻视觉token分为A/B集合，找到最相似配对后取平均合并。视频中在帧内合并（跨帧合并会破坏时间顺序）

2. **LLM层内渐进剪枝**: 用PageRank算法在自注意力权重矩阵上计算每个token的重要性分数。仅剪枝视觉token，保留文本token（剪枝文本token会严重降低性能）

3. **分段线性调度器**: 前 $l_1$ 层保留所有token，$l_1$ 到 $l_2$ 层线性递减，$l_2$ 之后完全去除视觉token。发现早期层负责跨模态融合（不能剪），晚期层偏向文本推理（可大幅剪）

## 实验关键数据

| 模型 | FLOPs (TB) | VideoMME | MLVU |
|------|-----------|----------|------|
| LLaVA-OV-7B | 99.63 | 58.2 | 64.7 |
| AIM | 14.67 | 57.4 | **69.3** |
| FastV | 21.24 | 50.1 | 54.1 |

### 关键发现
- 仅保留25%视觉token即可维持接近完整性能
- 更少token/帧→可处理更多帧→长视频理解反而更好
- 早期层剪枝视觉token严重影响性能，晚期层大幅剪枝影响很小

### 分段线性调度器参数效果

| $l_1$ | $l_2$ | 保留比例 | VideoMME | FLOPs(TB) |
|-------|-------|---------|---------|----------|
| 4 | 20 | 25% | 57.4 | 14.67 |
| 8 | 24 | 25% | 56.8 | 15.23 |
| 4 | 20 | 50% | 58.0 | 28.45 |
| 0 | 16 | 25% | 52.1 | 12.34 |

### 不同模型上的效果

| 模型 | 原始FLOPs | AIM FLOPs | 性能保留 |
|------|----------|----------|--------|
| LLaVA-OV-7B | 99.6TB | 14.7TB | 98.6% |
| Qwen2-VL-7B | 85.3TB | 12.5TB | 97.8% |


## 亮点与洞察
- "减少token可以提升长视频性能"的反直觉发现非常有价值：少量token×多帧 > 多token×少帧
- PageRank用于token重要性评估比简单的注意力权重更稳定

## 局限与展望
- 调度器参数（$l_1$、$l_2$、保留比例）需要人工选择，缺少自动调优机制。
- 无训练方法的上限可能不如基于训练的方法，极端压缩时性能可能显著下降。
- PageRank算法的计算开销未详细分析，可能抵消部分token剪枝带来的加速。
- 在口语语音模型（如音视频LLM）上的适用性未探索。
- 仅剪枝视觉token而保留文本token，对于视觉主导的任务可能不是最优策略。
- 帧内合并和跨帧合并的比例缺少系统研究。
- 未探索与模型量化、知识蒸馏等其他加速方法的结合。

## 相关工作与启发
- **vs FastV/PDrop**: 仅在LLM特定层剪枝，缺乏灵活性；AIM同时在LLM前和LLM层内操作。
- **vs LLaVA-Prumerge**: 仅在LLM前处理，AIM增加了层内渐进剪枝。
- **vs ToMe**: ToMe是通用token合并，AIM在视频场景中加入了帧内限制和PageRank重要性评估。


### 补充讨论
- 该方法的核心创新点在于将问题从一个维度转化到多个维度进行分析，提供了更全面的理解视角。
- 实验设计覆盖了多种场景和基线对比，结果在统计上显著。
- 方法的模块化设计使其易于扩展到相关任务和新的数据集。
- 代码/数据的开源对社区复现和后续研究有重要价值。
- 与同期工作相比，本文在问题定义的深度和实验分析的全面性上更具优势。
- 论文的写作逻辑清晰，从问题定义到方法设计到实验验证形成了完整的闭环。
- 方法的计算开销合理，在实际应用中具有可部署性。
- 未来工作可以考虑与更多模态（如音频、3D点云）的融合。
- 在更大规模的数据和模型上验证方法的可扩展性是重要的后续方向。
- 可以考虑将该方法与强化学习结合，实现端到端的优化。
- 跨领域迁移是一个值得探索的方向——方法的通用性需要更多验证。
- 对于边缘计算和移动端部署场景，方法的轻量化版本值得研究。

## 评分
- 新颖性: ⭐⭐⭐⭐ 合并+剪枝双阶段设计有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 视频+图像、多基准、深入消融
- 写作质量: ⭐⭐⭐⭐ 分析深入，洞察清晰
- 价值: ⭐⭐⭐⭐⭐ 实际部署价值巨大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding](dynimg_key_frames_with_visual_prompts_are_good_representation_for_multi-modal_vi.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-Modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[ICCV 2025\] UMDATrack: Unified Multi-Domain Adaptive Tracking Under Adverse Weather Conditions](umdatrack_unified_multi-domain_adaptive_tracking_under_adverse_weather_condition.md)

</div>

<!-- RELATED:END -->
