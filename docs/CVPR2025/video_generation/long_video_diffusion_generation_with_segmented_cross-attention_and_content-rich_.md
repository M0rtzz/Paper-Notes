---
title: >-
  [论文解读] Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation
description: >-
  [CVPR 2025][长视频生成] Presto 提出分段交叉注意力（SCA）策略，将隐状态沿时间维度分段并与对应子描述分别交叉注意力，结合精心策展的 261K 高质量长视频数据集 LongTake-HD，实现了 15 秒内容丰富且长程连贯的视频生成，在 VBench 语义得分达到 78.5%、Dynamic Degree 达到 100%。
tags:
  - CVPR 2025
  - 长视频生成
  - 分段交叉注意力
  - 视频数据策展
  - 渐进式子描述
  - DiT
---

# Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation

**会议**: CVPR 2025  
**arXiv**: [2412.01316](https://arxiv.org/abs/2412.01316)  
**代码**: https://presto-video.github.io (项目页面)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 长视频生成, 分段交叉注意力, 视频数据策展, 渐进式子描述, DiT

## 一句话总结

Presto 提出分段交叉注意力（SCA）策略，将隐状态沿时间维度分段并与对应子描述分别交叉注意力，结合精心策展的 261K 高质量长视频数据集 LongTake-HD，实现了 15 秒内容丰富且长程连贯的视频生成，在 VBench 语义得分达到 78.5%、Dynamic Degree 达到 100%。

## 研究背景与动机

**领域现状**：当前视频扩散模型主要聚焦于生成 3-8 秒的短视频片段，内容表达和丰富度受到很大限制。延长视频时长的早期方法通过插值/外推来扩展短片段，但内容多样性受限于原始短片段的容量。

**现有痛点**：自回归方法（如额外模块逐步延长视频）面临误差传播问题；多文本拼接视频虽然可以提升内容多样性，但不同场景之间过渡生硬；现有长视频方法忽视了高质量数据的重要性，导致生成视频的一致性和内容多样性低。

**核心矛盾**：长视频需要在内容多样性和长程连贯性之间取得平衡——单一文本输入信息量不足以描述长视频中丰富的场景变化，而现有的文本编码器又存在长度截断导致信息丢失的问题。

**本文目标** (1) 如何在 DiT 架构中同时处理多个渐进式文本条件？(2) 如何构建高质量的长视频训练数据集？

**切入角度**：将长视频的文本描述拆分为多个渐进式子描述（progressive sub-captions），让模型的不同时间段分别关注对应的文本条件，从而既保证内容丰富度又维持时序一致性。

**核心 idea**：将隐状态沿时间维度分段、与渐进式子描述分段交叉注意力，配合精心策展的长视频数据集，实现内容丰富且连贯的长视频生成。

## 方法详解

### 整体框架

Presto 基于 Allegro（2.8B 参数的 DiT 模型）构建。输入端将每个视频配备一个总体描述和 5 个渐进式子描述，分别用 T5 编码器编码得到 5 组文本嵌入。在 DiT 的交叉注意力层中，将隐状态沿时间维度分为 5 段，每段与对应的子描述进行交叉注意力计算。训练分为预训练（261K 数据，1500 步）和微调（47K 精选数据，500 步）两个阶段。推理时，用户输入单一 prompt，由 GPT-4o 作为"导演"生成 5 个渐进式子描述。

### 关键设计

1. **分段交叉注意力（Segmented Cross-Attention, SCA）**:

    - 功能：让隐状态的不同时间段分别关注对应的文本条件
    - 核心思路：将 N 个子描述分别编码为 $\{c_i\}_{i=1}^N$，将隐状态 $z$ 沿时间维度等分为 N 段 $\{z_i\}_{i=1}^N$，每段 $z_i$ 只与对应 $c_i$ 做交叉注意力。作者探索了三种变体：ISCA（完全隔离）、SSCA（顺序累积）、OSCA（重叠边界）。最终采用 OSCA——在相邻段的边界引入 $\delta$ 帧重叠，重叠区域的注意力输出取平均，促进段间平滑过渡。SCA 不引入任何额外参数。
    - 设计动机：单一长文本嵌入会被截断导致信息丢失，而让所有帧关注所有文本又会模糊细节。分段策略类似窗口注意力思想，在保持局部文本精度的同时，通过自注意力实现全局信息交流。

2. **LongTake-HD 数据集策展**:

    - 功能：提供高质量的长视频-多文本配对训练数据
    - 核心思路：从 890 万公开视频出发，经过时长/帧率/分辨率筛选 → 场景分割（PySceneDetect）→ 低级指标过滤（亮度、水印）→ 美学和运动内容筛选（LAION Aesthetics + 光流），最终得到 261K 单场景视频。使用 Aria 为视频和关键帧生成描述，再用 GPT-4o 进行因果式精炼，生成 5 个渐进式子描述（含摄像机运动信息）。
    - 设计动机：现有视频数据集包含大量噪声和低质量内容，且缺乏与长视频匹配的多段文本描述。高质量数据对长视频生成至关重要。

3. **渐进式子描述生成策略**:

    - 功能：为每个长视频生成连贯的、非冗余的多段叙事描述
    - 核心思路：将视频分为 N 段，分别生成独立描述，再利用 LLM 以因果方式逐段精炼——生成第 i 段描述时参考前面所有子描述和总体描述，确保每段代表故事线中的独立片段，并显式融入摄像机运动描述。这种"narrative-style"标注消除了段间的冗余描述。
    - 设计动机：传统多文本方法将多个不相干描述硬性组合（如 TALC），导致描述之间冗余且缺乏叙事连贯性。

### 损失函数 / 训练策略

采用标准扩散模型训练损失。预训练阶段在 64 张 H100 GPU 上以 batch size 256、学习率 1e-4 训练 1500 步（处理 384K 视频）；微调阶段使用精选 47K 数据训练 500 步。后处理使用 EMA-VFI 进行帧插值以进一步延长视频长度和规范速度。

## 实验关键数据

### 主实验

| 方法 | Semantic Score | Dynamic Degree | Overall Score |
|------|---------------|----------------|---------------|
| Gen-3 (商业) | 75.2 | 60.1 | 82.3 |
| Allegro (开源SOTA) | 73.0 | 55.0 | 81.1 |
| TALC (MT2V) | 44.4 | 98.6 | 58.9 |
| **Presto (本文)** | **78.5** | **100.0** | 80.2 |

用户研究（win rate %）：

| 对比方法 | Overall Win | Diversity Win | Coherence Win | Text-Video Win |
|----------|-------------|---------------|---------------|----------------|
| vs Gen-3 | 45.0 | 59.1 | 35.1 | 40.9 |
| vs Allegro | 54.9 | 68.0 | 45.1 | 51.4 |
| vs TALC | 91.8 | 95.3 | 89.5 | — |

### 消融实验

| 配置 | Overall Score | Dynamic Degree | Consistency |
|------|--------------|----------------|-------------|
| OSCA (完整) | 74.7 | 100.0 | 25.29 |
| SSCA | 73.7 ↓ | 100.0 | 25.06 ↓ |
| ISCA | 73.1 ↓ | 100.0 | 24.88 ↓ |
| w/o 精细数据过滤 | 72.0 ↓ | 97.2 ↓ | 24.06 ↓ |
| 单一长文本条件 | 71.8 ↓ | 100.0 | 24.06 ↓ |

### 关键发现

- OSCA 是三种 SCA 策略中最优的，重叠设计促进了段间平滑过渡，兼顾内容丰富度和连贯性
- 数据质量对长视频生成影响显著，去掉精细过滤后 Overall Score 下降 2.7%
- 渐进式子描述 vs 单一长文本拼接：后者 Overall Score 下降 2.9%，证明分段建模文本比简单拼接更有效
- Dynamic Degree 达到 100% 说明 SCA 在捕捉动态方面非常强

## 亮点与洞察

- **SCA 零额外参数**：分段交叉注意力不引入任何新参数或模块，可以无缝集成到任何基于 DiT 的架构中，这使得迁移成本极低。核心思想类似窗口注意力但应用于交叉注意力的时间维度，简洁有效。
- **渐进式叙事描述**：将传统的"多文本拼接"升级为因果式渐进描述，消除冗余的同时保持叙事连贯性。这种思路可以迁移到任何需要分段控制的生成任务（如长文档生成、长音频合成等）。
- **数据策展流程系统化**：从 890 万视频筛选到 261K 高质量样本，建立了完整的多级过滤和多模态标注 pipeline，对社区有重要参考价值。

## 局限与展望

- 高动态复杂场景下 Quality Score 有所下降，说明在丰富内容和视觉质量之间仍存在 trade-off
- 推理时依赖 GPT-4o 生成子描述，增加了延迟和成本
- 仅实验了 15 秒视频（88 帧 + 插值），更长视频（如分钟级）的效果未知
- 5 段固定分割可能不适用于所有视频内容，自适应分段策略值得探索

## 相关工作与启发

- **vs TALC**: TALC 也使用多文本输入，但采用硬性多场景组合，子描述间存在大量冗余。Presto 的渐进式描述消除了冗余并强化了叙事连贯性，Semantic Score 高出 34.1%
- **vs Gen-L-Video / FreeNoise**: 这些方法通过噪声调度或滑动窗口注意力延长视频，但受限于原始短片段的内容容量。Presto 从模型架构和数据两方面同时解决问题
- **vs Allegro**: Presto 直接基于 Allegro 构建，仅通过修改交叉注意力和数据就实现了显著提升，说明方法兼容性好

## 评分

- 新颖性: ⭐⭐⭐⭐ SCA 思想简洁但不算革命性，渐进式描述有新意
- 实验充分度: ⭐⭐⭐⭐⭐ VBench 定量 + 大规模用户研究 + 详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述到位
- 价值: ⭐⭐⭐⭐ 数据集和方法对长视频生成社区有实际价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](../../NeurIPS2025/video_generation/radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2025\] VideoGigaGAN: Towards Detail-rich Video Super-Resolution](videogigagan_towards_detail-rich_video_super-resolution.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)

<!-- RELATED:END -->
