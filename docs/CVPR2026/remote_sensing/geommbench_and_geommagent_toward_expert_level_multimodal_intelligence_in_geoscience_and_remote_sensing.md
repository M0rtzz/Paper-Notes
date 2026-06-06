---
title: >-
  [论文解读] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing
description: >-
  [CVPR 2026][遥感][geoscience] 提出 GeoMMBench（1053 道专家级地球科学多选题）和 GeoMMAgent（检索-感知-推理多智能体框架），系统评估 36 个 MLLM 在遥感领域的能力，揭示领域知识、感知接地和推理方面的系统性不足。
tags:
  - "CVPR 2026"
  - "遥感"
  - "geoscience"
  - "remote sensing"
  - "benchmark"
  - "multi-agent"
  - "MLLM evaluation"
---

# GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing

**会议**: CVPR 2026  
**arXiv**: [2604.08896](https://arxiv.org/abs/2604.08896)  
**代码**: [https://geo-mm-agi.github.io](https://geo-mm-agi.github.io)  
**领域**: 遥感 / 多模态基准  
**关键词**: geoscience, remote sensing, benchmark, multi-agent, MLLM evaluation

## 一句话总结

提出 GeoMMBench（1053 道专家级地球科学多选题）和 GeoMMAgent（检索-感知-推理多智能体框架），系统评估 36 个 MLLM 在遥感领域的能力，揭示领域知识、感知接地和推理方面的系统性不足。

## 研究背景与动机

MLLMs 在通用领域快速发展，但在地球科学和遥感领域的评估仍然有限。现有 RS 基准范围狭窄，主要聚焦感知任务（分类、检测、分割），且通常局限于光学影像。而地球科学需要多传感器数据融合、时空推理和跨学科知识整合，现有基准无法充分评估这些能力。

GeoMMBench 是首个专家级、知识驱动的地球科学多模态基准，跨越多学科、多传感器和多任务层级。

## 方法详解

### 整体框架

GeoMMBench 包含 1053 道图像基多选题，覆盖三个关键维度。GeoMMAgent 将 LLMs 与领域特定工具和模型整合在自适应多智能体框架中。

### 关键设计

1. **三维覆盖**：(a) 多学科——遥感、摄影测量、GIS、GNSS 及数学、物理、地理等基础学科；(b) 多传感器——光学、SAR、多光谱/高光谱、LiDAR、DEM、热成像等；(c) 多任务层级——从理论概念和数据预处理到感知任务和高级地空间应用。

2. **GeoMMAgent 三阶段智能体**：检索阶段（从外部知识库获取领域知识）→ 感知阶段（利用领域特定 RS 模型分析复杂数据）→ 推理阶段（空间推理和综合分析）。通过任务分解和工具选择实现超越通用 MLLM 的能力。

3. **专家级问题设计**：由地球科学领域 PhD 研究者和博士生通过广泛讨论制定范围和内容，来源包括教育资源、学术文献等权威参考。val 集 37 题（评估人类专家性能），test 集 1016 题。

### 损失函数 / 训练策略

GeoMMBench 为评估基准，无训练过程。GeoMMAgent 基于现有 LLMs 的零样本能力，通过工具调用增强。

## 实验关键数据

### 主实验

| 模型类别 | 代表模型 | GeoMMBench 准确率 | 说明 |
|---------|---------|----------------|------|
| 开源 MLLM | InternVL2.5, Qwen2.5-VL | 中等水平 | 领域知识不足 |
| 商用 MLLM | GPT-4o, Gemini | 较高但仍有差距 | 推理能力占优 |
| GeoMMAgent | 基于 LLM + 工具 | 显著提升 | 工具增强有效 |

val 集 37 题用于评估人类专家性能和模型选择，test 集 1016 题作为最终评估。问题由地球科学领域 PhD 研究者和博士生通过广泛讨论制定，来源包括教育资源、学术文献等权威参考。GeoMMAgent 的三阶段智能体分工：检索阶段从外部知识库获取光谱学、测绘学等领域知识，感知阶段利用 RS 专用模型分析 SAR/高光谱等复杂数据，推理阶段进行空间推理和综合分析。

### 关键发现

- 所有 36 个测试的 MLLM 在地球科学领域都存在系统性不足
- 领域知识（如光谱学、测绘学）是最大短板
- GeoMMAgent 通过工具增强显著优于独立 LLM
- 多传感器理解（特别是 SAR、高光谱）远不如光学影像
- GeoMMBench 覆盖 4 个核心学科（遥感、摄影测量、GIS、GNSS）及数学、物理、地理等基础学科
- 任务层级从理论概念和数据预处理到感知任务和高级地空间应用，提供全面评估

## 亮点与洞察

- 首个专家级地球科学多模态基准，跨学科跨传感器跨任务层级
- 36 个模型的大规模评估建立了当前性能基线
- GeoMMAgent 证明工具增强对弥合领域差距的有效性
- 问题由领域专家精心设计，确保质量和权威性

## 局限与展望

- 1053 题的规模仍有限，部分子领域覆盖可能不够
- 多选题形式可能无法完全反映开放式推理能力
- GeoMMAgent 的工具集需要持续扩展以覆盖更多任务
- 对多时相分析和变化检测等时间序列任务的覆盖仍不足
- 开源和闭源 MLLM 间的性能差距显著，提示 RS 领域仍需专门领域微调

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 首个专家级地球科学多模态基准
- 技术深度：⭐⭐⭐⭐ — 多智能体框架设计合理
- 实验充分度：⭐⭐⭐⭐⭐ — 36个模型的全面评估
- 实用价值：⭐⭐⭐⭐⭐ — 为RS-AI发展提供标准化评估

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[ICML 2026\] Any2Any: Unified Arbitrary Modality Translation for Remote Sensing](../../ICML2026/remote_sensing/any2any_unified_arbitrary_modality_translation_for_remote_sensing.md)
- [\[ECCV 2024\] Masked Angle-Aware Autoencoder for Remote Sensing Images](../../ECCV2024/remote_sensing/masked_angle-aware_autoencoder_for_remote_sensing_images.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)

</div>

<!-- RELATED:END -->
