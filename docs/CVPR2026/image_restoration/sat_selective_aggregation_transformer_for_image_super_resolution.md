---
title: >-
  [论文解读] SAT: Selective Aggregation Transformer for Image Super-Resolution
description: >-
  [CVPR 2026][图像恢复][超分辨率] 提出选择性聚合 Transformer (SAT)，通过密度驱动 token 聚合将 Key-Value 矩阵 token 数减少 97%、保持 Query 全分辨率，实现高效全局注意力建模，超越 SOTA PFT 达 0.22dB 且 FLOPs 降低 27%。
tags:
  - CVPR 2026
  - 图像恢复
  - 超分辨率
  - Transformer
  - token aggregation
  - 注意力机制
  - global modeling
---

# SAT: Selective Aggregation Transformer for Image Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2604.07994](https://arxiv.org/abs/2604.07994)  
**代码**: [https://github.com/PhuTran1005/SAT](https://github.com/PhuTran1005/SAT)  
**领域**: 图像超分辨率  
**关键词**: super-resolution, transformer, token aggregation, efficient attention, global modeling

## 一句话总结

提出选择性聚合 Transformer (SAT)，通过密度驱动 token 聚合将 Key-Value 矩阵 token 数减少 97%、保持 Query 全分辨率，实现高效全局注意力建模，超越 SOTA PFT 达 0.22dB 且 FLOPs 降低 27%。

## 研究背景与动机

基于 Transformer 的超分方法能捕获长距离依赖但面临二次计算复杂度。窗口注意力方法限制感受野，而近期方法各有不足：IPG 的图操作对硬件不友好、ATD 的外部字典引入有限额外信息、PFT 的跨层注意力链接可能传播早期层错误。

核心观察：SR 中高频区域（边缘、纹理）需要更多计算，低频区域（平滑区域）可安全聚合。现有方法对全图均匀处理导致计算分配低效。

## 方法详解

### 整体框架

SAT 采用残差组结构，交替使用局部 Transformer 块（LTB, 窗口注意力）和选择性聚合 Transformer 块（SATB, 全局注意力），形成全局-局部互补结构。

### 关键设计

1. **选择性聚合注意力 (SAA)**：非对称压缩——保持 Query 全分辨率（逐像素重建所需），仅压缩 Key-Value 矩阵。N 个 token 聚合为 K 个代表性 token（K ≈ 3% × N），将复杂度从 O(N²d) 降为 O(NKd)。

2. **密度驱动 Token 聚合 (DTA)**：基于密度峰值聚类原则选择聚合中心。计算每个 token 的局部密度（k-近邻余弦相似度）和到更高密度点的最小距离，乘积高者为聚合中心。使用分层子采样将中心选择复杂度从 O(N²) 降至 O(K²)。相似度加权聚合 + 特征范数恢复（FNR）保持特征分布一致性。

3. **全局-局部交替结构**：SAA 专注全局建模（捕获长距离依赖），与 Rwin-SA 局部注意力交替排列，互补提取深层特征。

### 损失函数 / 训练策略

标准 L1 像素损失训练。提供了严格的复杂度保证（定理 3.1）和近似界分析（定理 3.2），证明方法在质量退化可控的前提下实现大幅加速。

## 实验关键数据

### 主实验

| 数据集 | 指标 | SAT | PFT (之前SOTA) | 提升 |
|--------|------|-----|-------------|------|
| Urban100 ×4 | PSNR | +0.22dB | baseline | 显著 |
| 多数据集 | FLOPs | -27% | baseline | 效率大幅提升 |

### 消融实验

| 配置 | PSNR | 说明 |
|------|------|------|
| 无 FNR (特征范数恢复) | 下降 | FNR 对稳定训练至关重要 |
| 均匀聚合 vs 密度驱动 | 下降 | 密度感知选择中心更优 |
| 仅局部注意力 | 下降 | 全局建模不可或缺 |

### 关键发现

- Token 数量减少 97% 的情况下仍能保持甚至提升重建质量
- 密度驱动选择自然保留高频区域的细粒度 token 而合并低频区域
- FNR 对维持加权平均后的特征范数分布至关重要

## 亮点与洞察

- 非对称 Query-KV 压缩完美匹配 SR 任务需求（Query 保持逐像素，KV 可聚合）
- 密度驱动选择自适应于图像内容，高频保留、低频聚合
- 有完整的理论分析（复杂度界和近似界），增强了方法可信度
- 全局-局部交替是经过充分消融验证的最优选择

## 局限与展望

- 聚合比例（k=3%）和子采样因子 β 需要调优
- DTA 中的 k-近邻搜索仍有一定计算开销
- 对极度不规则纹理的处理效果有待验证

## 评分

- 新颖性：⭐⭐⭐⭐ — 非对称KV压缩+密度驱动聚合组合新颖
- 技术深度：⭐⭐⭐⭐⭐ — 理论分析严格
- 实验充分度：⭐⭐⭐⭐⭐ — 全面对比+充分消融
- 实用价值：⭐⭐⭐⭐ — 显著降低FLOPs同时提升性能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)

</div>

<!-- RELATED:END -->
