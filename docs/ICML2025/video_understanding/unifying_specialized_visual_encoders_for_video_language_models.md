---
title: >-
  [论文解读] Unifying Specialized Visual Encoders for Video Language Models
description: >-
  [ICML 2025][视频理解][多编码器融合] MERV 提出了多编码器视频表示方法，将四种专长不同的视觉编码器（DINOv2、ViViT、SigLIP、LanguageBind）通过时空对齐和跨注意力融合整合到单一 VideoLLM 中，在视频推理基准上比基线 Video-LLaVA 提升最高 4.62%，并验证了不同编码器的互补专长。
tags:
  - ICML 2025
  - 视频理解
  - 多编码器融合
  - VideoLLM
  - 视觉表示
  - 跨注意力
---

# Unifying Specialized Visual Encoders for Video Language Models

**会议**: ICML 2025  
**arXiv**: [2501.01426](https://arxiv.org/abs/2501.01426)  
**代码**: [有](https://github.com/princetonvisualai/merv)  
**领域**: Video Understanding / Video Language Models  
**关键词**: 多编码器融合, VideoLLM, 视觉表示, 跨注意力, 视频理解

## 一句话总结

MERV 提出了多编码器视频表示方法，将四种专长不同的视觉编码器（DINOv2、ViViT、SigLIP、LanguageBind）通过时空对齐和跨注意力融合整合到单一 VideoLLM 中，在视频推理基准上比基线 Video-LLaVA 提升最高 4.62%，并验证了不同编码器的互补专长。

## 研究背景与动机

### 领域现状
当前 VideoLLM（如 Video-LLaVA）均使用单一视觉编码器（通常是 CLIP 或 LanguageBind 等对比学习模型），编码器的选择直接限制了模型的能力上限。不同编码器在不同任务上有不同优势——CLIP 善于视觉-语言对齐但细粒度物体理解差，DINOv2 善于物体级理解但语言锚定弱，ViViT 善于时序建模但语言理解差。

### 核心痛点
单一编码器的固有弱点直接限制了 VideoLLM 的推理能力。例如某些问题只有使用 ViViT 的模型能答对（需要时序推理），而另一些只有 CLIP 模型能答对（需要语义理解）。传统观点认为多编码器会带来不必要的计算开销，但这种假设忽视了编码器间的互补价值。

### 本文方案
提出 MERV，通过 (1) 时空对齐各编码器输出；(2) 轻量预融合投影；(3) 跨注意力混合策略将四种编码器的特征融合成统一表示。利用并行视觉处理使计算开销最小化。

## 方法详解

### 整体框架

MERV 遵循 LLaVA/PrefixLM 范式，将视频输入分别送入四个视觉编码器提取特征，经时空对齐后通过跨注意力融合，最终与文本 token 拼接输入 LLaMA-2 7B。四个编码器并行处理，训练可在 8 张 L40 GPU 上 24 小时内完成。

### 关键设计

1. **多编码器特征提取 (Multi-Encoder Feature Extraction)**: 选择四种互补的编码器：

    - **空间专家 DINOv2**: 无监督学习，具备强大的物体部件理解和语义理解
    - **时序专家 ViViT**: 视频监督学习，空间-时间注意力建模长时依赖
    - **图像-语言对比专家 SigLIP**: sigmoid 对比学习，联合嵌入空间理解视觉-语言关联
    - **视频-语言对比专家 LanguageBind**: 多模态联合学习，理解视频与文本的高层语义

2. **时空对齐表示与预融合投影 (Spatio-Temporally Aligned Representations)**: 不同编码器输出形状不同（如 ViViT 输出 $8 \times 14 \times 14$, LanguageBind 输出 $16 \times 16 \times 16$），通过：

    - 时间对齐：调整输入帧数使每个编码器输出相同的时间维度 $t$
    - 空间对齐：自适应 2D 平均池化统一空间维度 $h \times w$
    - 维度投影：线性层将不同编码器维度 $d_e$ 映射到 LLM 维度 $d$
    $\mathbf{x}_e := \mathcal{P}(\mathbf{v}_e) W_e \in \mathbb{R}^{\ell \times d}, \quad \ell = t \times h \times w$
   投影器仅有 $d \times \sum_e d_e$ 个可训练参数，非常轻量。

3. **跨注意力融合 (Cross-Attention Feature Fusion)**: 使用单个可学习 query $\mathbf{Q} \in \mathbb{R}^{1 \times d}$，key 为各编码器特征序列均值 $\overline{\mathbf{X}} \in \mathbb{R}^{N \times d}$，value 为原始特征 $\mathbf{X} \in \mathbb{R}^{N \times \ell \times d}$：
    $\mathbf{O} = \text{Softmax}\left(\frac{\mathbf{Q}\overline{\mathbf{X}}^\top}{\sqrt{d}}\right) \mathbf{X} \in \mathbb{R}^{\ell \times d}$
   产生一个加权的线性混合表示，融合各编码器的信息。动态权重由视觉特征决定。

### 损失函数 / 训练策略

两种训练方案：
- **MERV (frozen)**: 仅 Stage 2 指令微调，学习率 $2 \times 10^{-5}$，batch size 128，仅训练投影器和融合模块
- **MERV (full)**: Stage 1 预训练（解冻 LLM）+ Stage 2 微调，Stage 1 学习率 $1 \times 10^{-4}$

MERV (frozen) 训练时间仅为 Video-LLaVA 的 43%，性能相当或更优。

## 实验关键数据

### 主实验
| 数据集 | 指标 | MERV (frozen) | Video-LLaVA | 提升 |
|--------|------|---------------|-------------|------|
| MSVD-QA | Acc | 70.97 | 67.74 | +3.23 |
| MSRVTT-QA | Acc | 59.03 | 56.90 | +2.13 |
| TGIF-QA | Acc | 51.10 | 47.99 | +3.11 |
| Perception Test | Acc | 46.21 | 44.22 | +1.99 |
| ActivityNet-QA | Acc | 50.87 | 47.08 | +3.79 |
| NExT-QA | Acc | 63.09 | 59.61 | +3.48 |
| TVQA | Acc | 42.28 | 37.66 | **+4.62** |

MERV (full) 在 Perception Test 上达 48.41%，超越 SeViLA 的 46.2%（+2.2%）。

### 消融实验
| 配置 | 平均准确率 | FLOPs | 说明 |
|------|----------|-------|------|
| Cross-Attention (默认) | 56.83 | 17.19T | 最优融合策略 |
| Concat (Seq.) | 54.45 | 43.09T | 序列拼接计算代价高 |
| Concat (Ch.) | 56.64 | 16.29T | 通道拼接效果接近 |
| Learnable W | 55.01 | 16.24T | 静态权重效果差 |
| 64 tokens/frame | 69.08 (MSVD) | - | 最优投影 token 数 |
| 2D Avg pooling | 55.86 | 2.1M FLOPs | 最优投影器（零参数） |

### 关键发现

1. **编码器互补性验证**: 移除任意一个编码器都会降低性能，且降低幅度与该编码器的专长强度成正比
2. **ViViT 的时序专长**: 在 SSv2-MCQ 的时序敏感子集上，ViViT 达 39.77%，比第二名高 9.19%，但在全集上落后
3. **跨注意力权重可解释**: 高运动视频激活 ViViT，含文字视频激活 SigLIP，静态场景激活 DINOv2/LanguageBind
4. **并行编码效率高**: 额外编码器带来的步骤时间开销极小，被最慢的单编码器主导

## 亮点与洞察

- **打破单编码器范式**: 首次系统地验证了在 VideoLLM 中使用多编码器的价值
- **时空对齐方案优雅**: 仅用 2D 平均池化（零参数）就实现了最优特征投影，简洁有效
- **SSv2-MCQ 分析精彩**: 通过时序敏感子集定量展示了 ViViT 的时序理解优势（推vs拉、左vs右）
- **可扩展性好**: 架构可轻松添加更多编码器，计算开销由并行处理吸收

## 局限与展望

- 4 个编码器的选择基于经验，缺乏系统的编码器搜索或自动选择机制
- 数据集固定为 Video-LLaVA 的数据，更高质量的训练数据可能带来更大提升
- 融合策略是输入无关的（基于序列均值的注意力），更好的输入自适应融合可能有益
- 未利用音频等其他模态的编码器，可能遗漏某些信息

## 相关工作与启发

- 与 SPHINX、Cambrian-1 等多编码器图像 LLM 工作相关，但聚焦于视频领域的时空对齐挑战
- 编码器的专长互补思路可启发其他多模态领域（如音频+视觉+语言的联合编码）
- 2D 平均池化优于复杂投影器的发现，提示特征选择可能比特征变换更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 多编码器融合思路新颖，但融合方法本身较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个基准、详尽消融、定性分析、SSv2 深度分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，分析深入，可视化丰富
- 价值: ⭐⭐⭐⭐ 为 VideoLLM 提供了新的扩展思路和实用方法

<!-- RELATED:START -->

## 相关论文

- [Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation](scaling_video-language_models_to_10k_frames_via_hierarchical_differential_distil.md)
- [Dynamic Updates for Language Adaptation in Visual-Language Tracking](../../CVPR2025/video_understanding/dynamic_updates_for_language_adaptation_in_visual-language_tracking.md)
- [FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](../../NeurIPS2025/video_understanding/enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [Factorized Learning for Temporally Grounded Video-Language Models](../../ICCV2025/video_understanding/factorized_learning_for_temporally_grounded_video-language_models.md)

<!-- RELATED:END -->
