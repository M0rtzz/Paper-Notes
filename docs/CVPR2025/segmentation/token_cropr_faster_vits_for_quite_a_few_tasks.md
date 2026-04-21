---
title: >-
  [论文解读] Token CropR: Faster ViTs for Quite a Few Tasks
description: >-
  [CVPR 2025][图像分割][ViT加速] 提出 Token CropR (Cropr)，一种基于交叉注意力的 ViT token 剪枝方法，通过辅助预测头端到端学习按任务相关性选择 token，训练后可移除辅助头实现接近随机剪枝器的吞吐量，在分类/语义分割/目标检测/实例分割四类任务上均实现 1.5-4× 加速且性能损失极小。
tags:
  - CVPR 2025
  - 图像分割
  - ViT加速
  - token剪枝
  - 交叉注意力
  - 多任务通用
  - 语义分割
---

# Token CropR: Faster ViTs for Quite a Few Tasks

**会议**: CVPR 2025  
**arXiv**: [2412.00965](https://arxiv.org/abs/2412.00965)  
**代码**: [GitHub](https://github.com/benbergner/cropr)  
**领域**: 分割/高效视觉  
**关键词**: ViT加速, token剪枝, 交叉注意力, 多任务通用, 语义分割

## 一句话总结

提出 Token CropR (Cropr)，一种基于交叉注意力的 ViT token 剪枝方法，通过辅助预测头端到端学习按任务相关性选择 token，训练后可移除辅助头实现接近随机剪枝器的吞吐量，在分类/语义分割/目标检测/实例分割四类任务上均实现 1.5-4× 加速且性能损失极小。

## 研究背景与动机

- **ViT 的效率痛点**：自注意力的 $O(n^2)$ 复杂度使序列长度成为负担，随着模型增大、分辨率提高和 patch 细化趋势加剧。
- **现有 token 剪枝的局限**：基于注意力分数的启发式方法未显式建模 token 的任务重要性；归因方法需要完整前向传播，开销过大；几乎所有方法仅在分类任务上验证。
- **密集任务的矛盾**：语义分割需要像素级预测，与剪枝 token 的理念天然矛盾——如何在剪枝后恢复所有 token 的信息是核心挑战。
- **推理效率的间接损失**：参数化剪枝模块本身引入的额外层和损失会干扰主任务，且增加推理开销。
- **设计目标**：快速（推理时最小开销）、保持高性能、适用于多种视觉任务。

## 方法详解

### 整体框架

Cropr 模块插入 ViT 中间层。每个模块包含：scorer（交叉注意力，可学习 query×token 的 key 计算分数）→ selector（Top-K 选择 keep token）→ aggregator（复用注意力矩阵做加权平均）→ 辅助任务头（提供训练信号）。推理时移除 aggregator 和辅助头，多 query 聚合为单一向量，实现 $O(M)$ 评分。密集任务使用 Last Layer Fusion (LLF) 恢复剪除 token。

### 关键设计

**设计一：交叉注意力评分 + 辅助头训练**
- **功能**：端到端学习按任务相关性对 token 评分
- **核心思路**：可学习 query $\mathbf{Q} \in \mathbb{R}^{N \times D}$ 和 token 的 key 做交叉注意力 $\mathbf{A} = \mathbf{Q} \times \mathbf{K}(\mathbf{X})^\top$，对 query 维求和得 token 分数 $\mathbf{a}$。Aggregator 复用 $\mathbf{A}$ 计算加权平均输入辅助头做中间预测，反向传播训练 scorer。关键：对 scorer 输入施加 stop-gradient，隔离辅助头和主干的梯度。
- **设计动机**：自上而下的任务信号（来自辅助头）比自下而上的启发式（attention score）更能反映 token 的任务重要性；stop-gradient 避免辅助头梯度干扰主干表示学习。

**设计二：推理时 Query 聚合优化**
- **功能**：将推理开销从 $O(N \times M)$ 降至 $O(M)$
- **核心思路**：推理时移除 aggregator 和辅助头后，只需 token 分数。利用分配律：$\mathbf{a} = \sum_{n=1}^N \mathbf{Q}_n \mathbf{K}^\top = (\sum_{n=1}^N \mathbf{Q}_n) \mathbf{K}^\top = \bar{\mathbf{q}} \mathbf{K}^\top$，预计算聚合 query $\bar{\mathbf{q}}$，推理时每个 Cropr 模块仅需一次向量-矩阵乘法 + Top-K。
- **设计动机**：语义分割中 $N = h \times w$ 可能很大，不做聚合的推理开销接近完整注意力；交换矩阵乘法顺序将复杂度降至常数级。

**设计三：Last Layer Fusion (LLF) — 密集任务的 token 恢复**
- **功能**：在最后一层 ViT block 中恢复被剪除的 token
- **核心思路**：从所有 Cropr 模块中收集被剪除的 token，在倒数第二层 ViT block 输出后按空间位置插回，与保留的 token 一起经过最后一层 ViT block 处理。被剪 token 可与深层保留 token 做 attend 获得上下文信息。不引入额外参数。
- **设计动机**：完全丢弃 token 对密集预测任务不可接受；直接拼接被剪 token 在最终层让它们"看到"深层特征，比 token 摘要或额外投影更简单有效。

### 损失函数

主任务标准损失 + 各 Cropr 模块辅助头的任务特定损失（分类用 softmax CE，分割用下采样标签的 per-patch CE，检测用多标签二元 CE）。辅助损失通过 stop-gradient 不影响主干梯度。

## 实验关键数据

### ImageNet-1k 分类 (EVA-02 backbone)

| 方法 | Top-1 Acc | 加速比 |
|------|----------|-------|
| No pruning | 85.8% | 1.0× |
| Random pruning + LLF | 83.8% | ~2× |
| **Cropr** | **89.7%** (ViT-L) | **2.1×** |

### ADE20k 语义分割

| 方法 | mIoU | 加速比 |
|------|------|-------|
| No pruning | baseline | 1.0× |
| **Cropr** | **-0.1 mIoU** | **2.0×** |

### COCO 检测 + 实例分割

| 方法 | AP_box | 加速比 |
|------|--------|-------|
| Liu et al. | ~34% speedup | small model |
| **Cropr** | **63.0** | **1.9×** |

### 关键发现

1. ADE20k 语义分割上 2× 加速仅损失 0.1 mIoU（5 seeds 中位值），是"免费午餐"级别
2. 推理时 Cropr 的吞吐量接近随机剪枝器——辅助头完全不参与推理
3. Cropr 在高分辨率（512+）和大模型（ViT-L）上加速效果更显著
4. LLF 不引入参数但比 token 摘要和额外投影等替代方案更有效

## 亮点与洞察

- **训推分离设计**：辅助头训练时学习评分，推理时完全移除——优雅地解决了"学习 vs 推理开销"的矛盾
- **Query 聚合的数学技巧**：利用线性代数分配律将评分复杂度降至 $O(M)$，是简单但关键的工程优化
- **多任务通用性**：首个在分类/分割/检测/实例分割四类任务上均有效的 token 剪枝方法
- LLF 机制极其简洁——被剪 token 只是跳过中间层直接到最后一层，无需任何额外参数

## 局限与展望

- Top-K 选择是硬剪枝，不支持自适应每图不同剪枝率（影响批处理效率但牺牲了对简单/困难输入的灵活性）
- 辅助头的设计仍需为每个任务定制（分类 vs 分割 vs 检测），增加了适配新任务的工程成本
- LLF 在极端剪枝率下性能恢复有限——最后一层无法完全弥补被跳过的多层信息损失
- 未在视频 Transformer 或 NLP Transformer 上验证，跨领域泛化能力未知

## 相关工作与启发

- Cropr 的辅助头训练 + 推理移除的范式可推广到其他需要学习式 token 选择的场景
- LLF 的"跳过中间层并在最后层融合"思路可用于任何 token 剪枝方法

## 评分

⭐⭐⭐⭐ — 设计简洁优雅，推理开销极小，是首个真正多任务通用的 token 剪枝方法。ADE20k 上 2× 加速仅损失 0.1 mIoU 的结果非常有说服力。

<!-- RELATED:START -->

## 相关论文

- [The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [The Devil is in Low-Level Features for Cross-Domain Few-Shot Segmentation](the_devil_is_in_low-level_features_for_cross-domain_few-shot_segmentation.md)
- [Dual-Agent Optimization framework for Cross-Domain Few-Shot Segmentation](dual-agent_optimization_framework_for_cross-domain_few-shot_segmentation.md)
- [DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs](../../ECCV2024/segmentation/densenets_reloaded_paradigm_shift_beyond_resnets_and_vits.md)
- [Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](../../NeurIPS2025/segmentation/mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)

<!-- RELATED:END -->
