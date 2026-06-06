---
title: >-
  [论文解读] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space
description: >-
  [CVPR 2026][信号/通信][条件图像检索] CLAY 提出免训练的条件视觉相似度计算方法，通过在 VLM 嵌入空间中构建文本条件子空间来调制相似度，无需重新计算数据库特征即可适应不同检索条件，并支持多条件检索。
tags:
  - "CVPR 2026"
  - "信号/通信"
  - "条件图像检索"
  - "视觉语言模型"
  - "相似度调制"
  - "免训练"
  - "超球面几何"
---

# CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space

**会议**: CVPR 2026  
**arXiv**: [2604.11539](https://arxiv.org/abs/2604.11539)  
**代码**: 无  
**领域**: 信号通信  
**关键词**: 条件图像检索, 视觉语言模型, 相似度调制, 免训练, 超球面几何

## 一句话总结
CLAY 提出免训练的条件视觉相似度计算方法，通过在 VLM 嵌入空间中构建文本条件子空间来调制相似度，无需重新计算数据库特征即可适应不同检索条件，并支持多条件检索。

## 研究背景与动机

**领域现状**：图像检索系统通常依赖固定的单一相似度度量，但人类感知相似性是自适应的——看同一张图可能关注物种、颜色、动作等不同方面。

**现有痛点**：(1) 训练型方法需要对每种条件训练特定模型，且条件变化时需重算所有数据库特征；(2) 现有方法仅支持单条件检索，无法同时指定多个关注维度；(3) 训练数据需要每种条件的配对图像。

**核心矛盾**：条件变化时重新计算数据库嵌入计算开销大，而不同条件需要不同的相似度计算方式。

**核心 idea**：将条件化过程从视觉特征提取中分离——固定视觉嵌入不变，在相似度计算空间中根据文本条件动态调制。

## 方法详解

### 整体框架
预训练 VLM 提取固定视觉特征 → 给定文本条件生成条件投影矩阵 → 将查询和数据库的视觉特征投影到条件子空间 → 在子空间中计算余弦相似度 → 返回排序结果。

### 关键设计

1. **流形感知文本子空间构建**:

    - 功能：根据文本条件构建相似度调制空间
    - 核心思路：用 LLM 扩展条件文本为多个描述性短语，通过 VLM 文本编码器获取嵌入集合，PCA 提取主方向构建正交子空间，生成条件投影矩阵 $P_c$。考虑 VLM 嵌入空间的超球面几何，投影后重新归一化
    - 设计动机：简单的一个文本嵌入不足以定义一个有意义的子空间，需要多个相关描述来跨越条件相关的语义方向

2. **对称条件相似度**:

    - 功能：用同一投影矩阵同时变换查询和数据库特征
    - 核心思路：$\text{csim}(I_q, I_d | c) = \cos(P_c \cdot f(I_q), P_c \cdot f(I_d))$，查询和数据库特征对称处理。投影矩阵 $P_c$ 可预计算并缓存，条件变化只需切换矩阵
    - 设计动机：非对称方法（只变换查询）会保留数据库特征中的条件无关信息干扰检索结果；对称方法确保双方都只保留条件相关信息

3. **多条件检索扩展**:

    - 功能：支持同时指定多个关注维度
    - 核心思路：对多个条件的投影矩阵取并集子空间，构建联合投影矩阵
    - 设计动机：现实场景中用户可能想同时按"物种"和"颜色"检索，现有方法不支持

### 损失函数 / 训练策略
完全免训练，仅利用预训练 VLM 的特征空间。

## 实验关键数据

### 主实验

| 数据集 | 指标 | CLAY | GeneCIS (训练型) | FocalLens |
|--------|------|------|-----------------|-----------|
| GeneCIS 基准 | Recall@1 | 竞争性/优 | 基线 | 基线 |
| CLAY-EVAL | MR@K | SOTA | 不支持多条件 | 不支持 |

### 消融实验

| 配置 | 检索精度 | 说明 |
|------|---------|------|
| 对称投影 | 最优 | 完整方法 |
| 非对称投影 | 下降 | 数据库侧噪声 |
| 单文本嵌入 | 大幅下降 | 子空间表达力不足 |
| 无归一化 | 下降 | 忽略超球面几何 |

### 关键发现
- 免训练方法在标准基准上达到或超过训练型方法，且计算效率更高
- 对称 vs 非对称投影的差异证明了双向过滤条件无关信息的重要性
- 超球面几何的考虑（归一化）对性能影响显著

## 亮点与洞察
- **"改变度量而非特征"**：颠覆了传统思路——不改变特征提取，而是改变比较方式，使数据库特征可完全复用
- **免训练即 SOTA**：不需要任何训练数据就达到训练型方法的性能，实用性极强

## 局限与展望
- 依赖 LLM 生成的文本描述质量
- 子空间维度的选择需要调优
- 在极细粒度条件下可能不够精确

## 相关工作与启发
- **vs GeneCIS**: 需要训练+配对数据，条件变化需重算特征
- **vs FocalLens**: 也是条件检索但需训练，不支持多条件

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将条件化从特征提取移到相似度空间，思路新颖
- 实验充分度: ⭐⭐⭐⭐ 构建了新评估数据集
- 写作质量: ⭐⭐⭐⭐ 数学表述精炼
- 价值: ⭐⭐⭐⭐⭐ 免训练+高效+多条件支持，非常实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Neural Video Compression with Context Modulation](../../CVPR2025/signal_comm/neural_video_compression_with_context_modulation.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [\[NeurIPS 2025\] Angular Steering: Behavior Control via Rotation in Activation Space](../../NeurIPS2025/signal_comm/angular_steering_behavior_control_via_rotation_in_activation_space.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[ECCV 2024\] RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](../../ECCV2024/signal_comm/raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)

</div>

<!-- RELATED:END -->
