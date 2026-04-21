---
title: >-
  [论文解读] Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning
description: >-
  [CVPR 2025][模型压缩][类增量学习] 提出ACMap框架，通过将每个任务独立训练的adapter增量平均合并为单一adapter（保持O(1)推理复杂度），结合centroid prototype mapping对齐旧任务原型在新子空间中的表示，在5个基准上实现与SOTA EASE相当的精度同时推理速度快39倍。
tags:
  - CVPR 2025
  - 模型压缩
  - 类增量学习
  - Adapter合并
  - 原型映射
  - 无样本存储
  - 可扩展推理
---

# Adapter Merging with Centroid Prototype Mapping for Scalable Class-Incremental Learning

**会议**: CVPR 2025  
**arXiv**: [2412.18219](https://arxiv.org/abs/2412.18219)  
**代码**: https://github.com/tf63/ACMap  
**领域**: 模型压缩 / 持续学习  
**关键词**: 类增量学习, Adapter合并, 原型映射, 无样本存储, 可扩展推理

## 一句话总结

提出ACMap框架，通过将每个任务独立训练的adapter增量平均合并为单一adapter（保持O(1)推理复杂度），结合centroid prototype mapping对齐旧任务原型在新子空间中的表示，在5个基准上实现与SOTA EASE相当的精度同时推理速度快39倍。

## 研究背景与动机

**领域现状**：预训练模型+参数高效微调（adapter/prompt）已成为类增量学习(CIL)的主流范式。EASE为每个任务训练独立adapter并用各adapter提取原型做cosine分类，精度SOTA但推理时需逐adapter前向，复杂度O(T)。

**现有痛点**：现有方法存在精度-速度权衡——SimpleCIL/APER推理O(1)但精度有限（仅第一任务adapter），EASE精度SOTA但40个任务时推理慢39倍。隐私场景下不能存储旧任务样本（exemplar-free），进一步增加难度。

**核心矛盾**：adapter合并（权重平均）后旧任务原型在新子空间中位置发生偏移，原型不再准确。但没有旧任务数据无法重新计算原型。

**核心idea**：利用当前任务数据（唯一可用数据）在新旧子空间中的原型偏移作为线索，用仿射变换近似将旧原型映射到新子空间。

## 方法详解

### 整体框架

每个任务训练独立adapter（从共享初始权重出发）→ 增量平均合并为单一merged adapter → 对当前任务在merged adapter子空间中计算原型 → 用centroid prototype mapping校正旧任务原型 → cosine分类器推理仅用merged adapter（O(1)）。

### 关键设计

1. **Adapter增量平均合并**：

    - 功能：将所有任务的adapter合并为单一adapter
    - 核心思路：$\bar{\theta}_t = (1 - 1/t) \bar{\theta}_{t-1} + (1/t) \theta_t$
    - 关键条件——**初始权重替换**：第一个任务训练完后，将随机初始权重替换为 $\theta_1$，后续所有任务都从 $\theta_1$ 出发训练。这促使所有adapter在参数空间中遵循相似的训练路径，形成共享的低损失盆地
    - 损失面分析：可视化三个连续adapter的线性插值损失面，验证确实存在低损失盆地（红色区域），平均点（星号）位于盆地中央

2. **Centroid Prototype Mapping**：

    - 功能：在无旧任务数据的情况下，将旧原型映射到当前merged adapter子空间
    - 核心思路：$P_i(\bar{A}_t) \approx P_i(\bar{A}_i) + \Delta P$，其中 $\Delta P = \mathbb{E}[P_t(\bar{A}_t) - P_t(\bar{A}_i)]$ 是当前任务原型在新旧子空间中的质心偏移
    - 核心假设：当前任务原型的偏移模式可迁移到旧任务原型——这在实验中通过cosine similarity验证
    - 有趣发现：近期任务（如task 5在task 6子空间）cosine similarity很高（偏移小），但早期任务（如task 1在task 10子空间）偏移显著——这也支撑了early stopping策略

3. **Early Stopping策略**：

    - 功能：任务数超过阈值L后停止合并adapter
    - 核心理由：合并公式中 $1/t$ 趋近于0，新增adapter的贡献可忽略；且近期任务的原型偏移更小、映射更准确
    - 效果：L=10 vs L=∞结果几乎相同，但减少了训练成本

## 实验关键数据

### 主实验：5个基准平均精度/最终精度

| 方法 | CIFAR A̅/A_T | CUB A̅/A_T | IN-R A̅/A_T | IN-A A̅/A_T | VTAB A̅/A_T |
|------|:-:|:-:|:-:|:-:|:-:|
| SimpleCIL | 87.6/81.3 | 92.2/86.7 | 62.6/54.6 | 59.8/48.9 | 86.0/84.4 |
| APER | 90.7/85.2 | 92.2/86.7 | 72.4/64.3 | 60.5/49.4 | 86.0/84.4 |
| EASE | 91.5/85.8 | 92.2/86.8 | **78.3/70.6** | 65.3/55.0 | **93.6/93.6** |
| **ACMap** | **92.0/87.8** | 91.6/86.7 | 77.3/70.5 | **65.2/56.2** | 91.2/87.6 |

### 消融实验：推理时间（IN-R Task 40）

| 方法 | 推理时间(s) | 相对ACMap倍数 |
|------|:-:|:-:|
| SimpleCIL | 22.6 | ×0.96 |
| APER | 44.1 | ×1.88 |
| EASE | **916.5** | **×39.0** |
| **ACMap** | 23.5 | 1.0 |

### 关键发现

- ACMap在4/5数据集上与EASE精度相当（±1%），但**推理快39倍**
- 比SimpleCIL最终精度高16%+（IN-R: 70.5 vs 54.6），推理时间几乎相同
- 初始权重替换带来约0.5%的一致提升（CIFAR: 91.54→92.04）
- Early stopping L=10 vs L=∞几乎无差别，验证了大任务数下合并收益递减
- VTAB上ACMap弱于EASE——因为VTAB的5个域差异极大，单adapter难拟合；EASE的5个独立adapter天然匹配

## 亮点与洞察

- **权重平均的CIL新用法**：利用model merging中"共享初始权重→低损失盆地→可安全平均"的理论，优雅地解决了adapter积累带来的推理开销问题
- **Centroid Prototype Mapping的简洁巧妙**：不需要任何学习参数，仅用当前任务数据的质心偏移估计所有旧任务的原型偏移。假设简单但实证有效
- **精度-速度帕累托最优**：在Fig.1的精度-速度图上，ACMap位于帕累托前沿——比同速度的方法精度高，比同精度的方法速度快
- **Exemplar-free设计**：不存储旧任务数据，隐私友好，适合真实场景

## 局限与展望

- VTAB等域差异极大的场景下，单merged adapter表达能力不足
- Centroid prototype mapping的仿射假设在子空间变化剧烈时可能不成立（早期任务）
- 未与LoRA以外的其他PEFT方法（如prefix tuning）结合验证
- 理论分析主要是可视化验证，缺乏形式化的收敛保证

## 相关工作与启发

- **vs EASE**：EASE用T个独立adapter的拼接特征，精度稍好但O(T)推理不可扩展。ACMap通过合并+映射实现O(1)推理，是精度-速度的更优权衡
- **vs RAPF**：RAPF也做adapter merging但用SVD（计算密集）且依赖CLIP文本特征。ACMap用简单平均+视觉only，更轻量
- **Model merging启发**：本文是model merging理论在CIL中的成功应用，"共享起点→相似训练路径→可安全平均"是核心洞察

## 评分

- 新颖性: ⭐⭐⭐⭐ adapter merging + centroid mapping组合新颖，但各技术本身较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集、完整消融（损失面可视化、原型偏移分析、时间对比），结论solid
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰、图表精美（损失面、cosine similarity曲线），每个设计选择都有充分动机
- 价值: ⭐⭐⭐⭐ 解决了adapter-based CIL的核心痛点（推理扩展性），实现精度-速度帕累托最优

<!-- RELATED:START -->

## 相关论文

- [Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)
- [Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [Incremental Object Keypoint Learning (KAMP)](incremental_object_keypoint_learning.md)
- [Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)

<!-- RELATED:END -->
