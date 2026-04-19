---
title: >-
  [论文解读] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime
description: >-
  [CVPR 2026][图像分割][场景图生成] 提出 DSFlash，一个低延迟全景场景图生成模型，通过统一 backbone、双向关系预测和 mask 动态剪枝等设计，在 RTX 3090 上实现 56 FPS 的实时推理，同时保持 SOTA 性能（mR@50=30.9）。
tags:
  - CVPR 2026
  - 图像分割
  - 场景图生成
  - 全景分割
  - 实时推理
  - 双向关系预测
  - 动态剪枝
---

# DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime

**会议**: CVPR 2026  
**arXiv**: [2603.10538](https://arxiv.org/abs/2603.10538)  
**代码**: 接收后公开  
**领域**: 分割  
**关键词**: 场景图生成, 全景分割, 实时推理, 双向关系预测, 动态剪枝

## 一句话总结

提出 DSFlash，一个低延迟全景场景图生成模型，通过统一 backbone、双向关系预测和 mask 动态剪枝等设计，在 RTX 3090 上实现 56 FPS 的实时推理，同时保持 SOTA 性能（mR@50=30.9）。

## 研究背景与动机

场景图生成（SGG）旨在从图像中提取结构化的节点-边图表示，节点代表实例、边代表关系（如"person sitting on chair"），在 VQA、图像描述、具身推理等下游任务中很有价值。全景场景图生成（PSGG）进一步使用分割 mask 代替 bbox 进行实例定位。

**核心矛盾**：现有 PSGG 方法追求性能而忽视效率。DSFormer 达到 SOTA 的 mR@50=30.7，但推理延迟高达 458ms；即便 REACT 实现了 19ms，其 PSGG 性能也很有限。更关键的是，现有方法通常只预测关系的子集，而非完整场景图。

**切入角度**：DSFlash 从 DSFormer 出发，系统性地替换其低效组件——统一两个 backbone 为一个、设计双向预测头减半推理次数、利用 mask 信息做动态 token 剪枝——实现了"又快又全又好"的完整场景图生成。

## 方法详解

### 整体框架

DSFlash 采用两阶段架构：第一阶段用冻结的 EoMT backbone 做全景分割并提取特征；第二阶段对每对 mask 组合添加 mask embedding，经 transformer neck 和关系预测头输出双向关系。训练时使用 GT mask，推理时使用预测 mask。

### 关键设计

1. **统一 Backbone（Merged Backbones）**: DSFormer 用两个独立 backbone（一个分割、一个关系预测），DSFlash 直接从分割模型 EoMT 中抽取中间层特征张量（blocks 2/5/8/11），拼接得到 $768 \times 40 \times 40$ 的特征图，省去第二个 backbone 的开销。EoMT 始终冻结，训练仅需更新关系预测部分，大幅降低训练成本（单卡 GTX 1080 不到 24 小时）。选择 EoMT 而非 MaskDINO 是因为其纯 encoder 设计更快且易集成。

2. **双向关系预测（Bidirectional Predictions）**: DSFormer 对每对 mask $(S_0, S_1)$ 需做两次前向传播（正向/反向），DSFlash 设计门控机制一次推理同时输出两个方向的关系：

    - 先计算 $g = \sigma(\text{gate}_{mlp}(x))$
    - 正向特征 $t^{\rightarrow} = g \odot x$，反向特征 $t^{\leftarrow} = (1-g) \odot x$
    - 共享 MLP 分别预测 $z^{\rightarrow}$ 和 $z^{\leftarrow}$

   为防止模型利用标注中正/反向标签分布不均的 shortcut，训练时交换 mask 顺序做第二次前向，加入特征一致性损失：$\text{Consistency} = \frac{1}{D}\sum_{i}(t_i^{\rightarrow} - t_i^{\prime\leftarrow})^2 + (t_i^{\leftarrow} - t_i^{\prime\rightarrow})^2$。推理时只需单次前向。

3. **Mask 动态 Patch 剪枝**: 利用 mask embedding 计算时已有的 overlap ratio 信息，识别与 subject/object 均无重叠的 patch token 并丢弃。这些远离主客体的 patch 对关系分类贡献极小，剪枝几乎无额外计算开销。由于最终预测仅依赖 CLS token，模型天然支持可变 token 数。

4. **Token Merging**: 在 backbone 的每个 attention 层前用 ToMe-SD 合并相似 token，attention 后再 unmerge，减少注意力计算量同时保持分割能力。

### 损失函数 / 训练策略

- 关系预测：BCE 损失
- 特征一致性：MSE 损失
- 数据增强：DeiT III 风格（随机翻转、色彩抖动、灰度/曝光/高斯模糊三选一）
- 训练时使用 GT mask，backbone 冻结

## 实验关键数据

### 主实验

| 方法 | mR@50 ↑ | 延迟 (ms) ↓ | 参数量 |
|------|---------|------------|--------|
| DSFormer | 30.70 | 458 | 330M |
| HiLo-L | 19.08 | 427 | 230M |
| REACT | 19.00 | 19 | 43M |
| **DSFlash-L** | **30.90** | **50** | 340M |
| DSFlash-B* | 28.50 | 23 | 116M |
| DSFlash-S* | 25.05 | **18** | **40M** |

### 消融实验（增量优化）

| 优化项 | mR@50 | 延迟 (ms) | RPS ↑ | 说明 |
|--------|-------|----------|-------|------|
| Baseline (DSFormer) | 30.7 | 445 | 435 | 起点 |
| + 统一 Backbone | 25.0 | 41 (-91%) | 5,745 | 最大加速来源 |
| + 高效 Mask 编码 | 25.0 | 37 (-10%) | 7,132 | 减少数据拷贝 |
| + 门控双向预测 | 28.8 | 29 (-22%) | 11,491 | 推理次数减半+额外监督提升性能 |
| + 跳过分割上采样 | 28.5 | 23 (-21%) | 12,928 | 无需上采样到原图分辨率 |

### 关键发现

- 分割模型质量直接决定最终场景图性能（mR@50 与 mR@inf 强相关），未来更好的分割模型可直接提升 DSFlash
- 双向预测不仅减半推理次数，还因额外监督信号提升了 mR@50（25.0→28.8）
- DSFlash-S 仅 40M 参数，18ms 延迟，仍优于除 DSFormer 外的所有方法

## 亮点与洞察

- **工程与设计并重**：每个优化都有清晰的动机和量化分析，不是简单堆技巧
- **完整场景图**：不像 REACT 只预测部分关系，DSFlash 对所有 mask 对预测关系，反而更快
- **一致性损失的洞察**：发现训练集中正/反向标签分布不均会导致 shortcut learning，用一致性约束优雅解决
- **可部署性强**：单卡 GTX 1080 训练 24h，RTX 3090 上 56 FPS 推理

## 局限与展望

- mR@50 高度依赖分割质量，若分割模型对某些类别表现差则场景图也差
- PSG 数据集仅 56 类谓词，真实场景关系更丰富
- 未探索场景图的下游任务验证（如 VQA、具身推理）
- 冻结 backbone 可能限制了对特定数据集的适配

## 相关工作与启发

- EoMT 的纯 encoder 设计是实现低延迟的关键基础设施选择
- 双向预测的门控机制借鉴 GRU 的门控思想，简洁有效
- DSFormer 的 mask embedding 模块设计合理，被直接复用
- Token Merging (ToMe) 在保持信息的同时减少计算量

## 评分

- **新颖性**: ⭐⭐⭐⭐ 双向门控预测和 mask 动态剪枝有新意，整体是系统性工程优化
- **实验充分度**: ⭐⭐⭐⭐⭐ 详尽的消融和延迟分析，每个组件的贡献清晰可见
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- **价值**: ⭐⭐⭐⭐⭐ 对实时场景图生成有重要实际意义，降低了 SGG 研究的硬件门槛

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [SAP: Segment Any 4K Panorama](sap_segment_any_4k_panorama.md)
- [Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgbd_scene_understanding_via_multitask_a.md)
- [GenMask: Adapting DiT for Segmentation via Direct Mask Generation](genmask_adapting_dit_for_segmentation_via_direct_mask_generation.md)
- [CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_re.md)

<!-- RELATED:END -->
