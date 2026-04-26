---
title: >-
  [论文解读] Aligning Effective Tokens with Video Anomaly in Large Language Models
description: >-
  [ICCV 2025][视频理解][视频异常理解] 提出VA-GPT，通过空间有效Token选择（SETS）和时间有效Token生成（TETG）两个模块，在MLLM中高效对齐与视频异常相关的关键Token，实现对异常事件的精准检测、描述和时间定位。
tags:
  - ICCV 2025
  - 视频理解
  - 视频异常理解
  - 多模态大语言模型
  - 有效Token选择
  - 时空对齐
  - 异常检测
---

# Aligning Effective Tokens with Video Anomaly in Large Language Models

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: 无  
**领域**: video_understanding  
**关键词**: 视频异常理解, 多模态大语言模型, 有效Token选择, 时空对齐, 异常检测

## 一句话总结

提出VA-GPT，通过空间有效Token选择（SETS）和时间有效Token生成（TETG）两个模块，在MLLM中高效对齐与视频异常相关的关键Token，实现对异常事件的精准检测、描述和时间定位。

## 研究背景与动机

### 现有方法的局限性

传统视频异常检测方法存在两个根本性问题：(1) 本质上是闭集检测与分类问题，无法全面理解和解释异常；(2) 受限于有限的词汇表，难以处理未见或新颖的场景。虽然近年来视频理解MLLM（如Video-Chat、Video-ChatGPT等）在通用视频分析上取得了显著进展，但在异常检测领域表现不佳。

### 核心问题分析

**为什么现有MLLM难以处理视频异常？** 根本原因在于异常事件具有**空间和时间上的稀疏性**：在大多数情况下，只有少数帧中的小区域包含关键的异常信息。然而，现有方法将所有潜在Token以相同优先级在空间和时间维度上进行处理，导致大量与异常无关的冗余Token稀释了关键信息，造成性能退化。

### 关键洞察

作者发现异常事件往往导致局部区域产生不同的视觉变化和变动。因此，**如何让多模态架构演化出选择性Token生成和处理机制**，动态优先处理异常显著信息，同时保持全面的场景理解能力，成为核心研究问题。

## 方法详解

### 整体框架

VA-GPT基于经典的视频理解MLLM框架构建，输入视频包含T帧，使用冻结的ViT-based视觉编码器（CLIP）从每帧提取视觉Token $X_t = \{x_t^i\}_{i=1,...,N}$。核心创新在于两个模块：

1. **空间有效Token选择（SETS）**：从 $X_t$ 中选择空间有效Token $X_t^*$，替代原始的全量Token参与微调和推理
2. **时间有效Token生成（TETG）**：生成异常感知的时间先验Token $S_t^*$，直接在语言空间中为LLM提供时间信息

### 关键设计一：空间有效Token选择（SETS）

**为什么需要空间Token选择？** 在MLLM的设定中，最关键的问题是视觉和语言模态的对齐。由于文本描述主要描述异常事件，而异常事件只占整个视频的小部分，将所有视觉模式与文本Token对齐既不合理又计算昂贵。

**帧间差异计算**：对于视频中的每帧 $V_t$，以前一帧 $V_{t-1}$ 作为参考帧，利用DINOv2作为特征提取器获取Patch嵌入：

$$F_t, F_{t-1} = FE(V_t), FE(V_{t-1})$$

然后计算逐Patch的曼哈顿距离作为帧间差异图：

$$D_t = dis(F_t, F_{t-1})$$

**Token筛选策略**：根据差异图 $D_t$，取距离最大的前K比例元素赋值为1，其余为0，形成掩码 $M_t$：

$$X_t^* = \{x_t^i | m_t^i = 1, m_t^i \in M_t\}$$

**为什么用帧间差异而非其他方法？** 作者的核心假设是：相邻帧之间变化大的区域更值得关注，因为异常事件通常伴随局部区域的显著视觉变化。使用DINOv2提取的特征具有良好的区分性和稳定性，可靠地捕捉这种变化。

### 关键设计二：时间有效Token生成（TETG）

**异常感知分类器**：设计了一个简单但有效的MLP $F_A$，用于判断每帧是否与异常事件相关。利用特征编码器提取的class embedding $z$，按训练视频标注分为正常嵌入 $z_n$ 和异常嵌入 $z_a$，使用二分类损失优化：

$$\mathcal{L} = E_{z \sim z_n}[-\log\frac{1}{1+\exp(-F_A(z))}] + E_{z \sim z_a}[-\log\frac{\exp(-F_A(z))}{1+\exp(-F_A(z))}]$$

**Token生成方式**：由于分类器提供的信息是显式的，可以通过自然语言模板直接投射到LLM的文本Token空间。基于分类结果，选择高置信度包含异常事件的起止帧时间戳 `<a-start>` 和 `<a-end>`，组装为模板：

> "Known common crime types are: 'Shooting', 'Arson', 'Arrest', ... There is one of the crime types occurring from <a-start> to <a-end>"

**为什么这样设计？** 这种方式以极低成本为LLM提供关于异常事件时间信息的先验知识，无需额外的复杂模块，直接复用LLM自身的文本理解能力。

### 训练策略

采用两阶段渐进式训练：

1. **第一阶段**：使用异常视频数据微调。基于UCF-Crime数据集构造指令跟随格式的问答对，混合多种指令对（文本对话、单/多轮视觉QA、视频QA），优化除冻结视觉编码器外的所有模块
2. **第二阶段**：使用空间有效Token对齐LLM。利用从UCF-Crime数据集中每帧提取的空间有效Token进行额外的短期微调，仅需不到150次迭代即可显著提升性能

## 实验关键数据

### 主实验

| 方法 | LLM | 域内Total Acc.(%) | 域内Temporal Acc.(%) | 跨域Total Acc.(%) | 跨域Temporal Acc.(%) |
|------|-----|-------------------|---------------------|-------------------|---------------------|
| Video-ChatGPT | Vicuna-7B | 24.13 | 28.51 | 24.00 | 29.10 |
| Otter | LLaMa-7B | 22.41 | 22.17 | 25.20 | 23.80 |
| Valley | Vicuna-7B | 20.34 | 14.48 | 21.00 | 20.20 |
| Video-LLaMA2 | Vicuna-7B | 21.38 | 26.62 | 24.20 | 23.00 |
| Hawkeye | LLaVA-7B | 28.60 | 30.00 | 25.30 | 28.50 |
| LLaMA-VID (Baseline) | Vicuna-7B | 14.83 | 26.70 | 18.80 | 23.60 |
| **VA-GPT (Ours)** | Vicuna-7B | **30.69** | **35.00** | **26.20** | **31.02** |

VA-GPT在所有四项指标上均取得最佳，域内Total Acc.比基线翻倍以上，跨域泛化能力也显著领先。

### 消融实验

| 配置 | Baseline | Stage One Fine-tuning | Stage Two Fine-tuning |
|------|----------|----------------------|----------------------|
| w/o Both | 14.83 / 26.70 | - | - |
| w.SETS | 24.83 / 27.20 | 25.86 / 29.68 | 29.31 / 31.60 |
| w.TETG | 23.79 / 27.76 | 26.10 / 30.02 | 28.96 / 33.58 |
| w.Both | 25.12 / 28.81 | 27.50 / 30.77 | **30.69 / 35.00** |

（格式：Total Acc. / Temporal Acc.）

**采样率K的消融**：

| K | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|-----|-----|-----|-----|-----|
| Total Acc.(%) | 23.61 | 24.83 | **30.69** | 28.67 | 27.27 |
| Temporal Acc.(%) | 29.03 | 29.93 | **35.00** | 31.23 | 31.03 |

K=0.5为最优，过小会丢失重要信息，过大会引入过多噪声。

### 关键发现

1. **SETS和TETG具有互补性**：两者分别从空间和时间维度压缩异常信息，联合使用效果最佳
2. **数据质量至关重要**：仅用约4000个视频（远少于基线的90k+视频）即可取得优异性能，关键在于高质量的指令跟随数据
3. **SETS同时提升数据质量**：在第二阶段微调中过滤与QA无关的视觉区域，仅需不到150次迭代即可显著提升

## 亮点与洞察

1. **Token级别的对齐创新**：首次在MLLM中探索对不同Token赋予不同的可学习知识以更好地对齐视觉内容，而非简单地等权处理所有Token
2. **时间Token的自然语言模板设计**：将分类器的时间预测通过自然语言模板注入LLM，巧妙复用了LLM的文本理解能力，设计极为简洁高效
3. **跨域评估基准**：建立了基于XD-Violence的跨域评估协议，系统评估了模型在领域迁移下的鲁棒性
4. **以少胜多的数据效率**：证明了高质量数据+有效Token选择可以大幅减少训练数据需求

## 局限与展望

1. 对于复杂场景中的异常事件检测和描述仍有挑战，如多个异常事件同时发生的场景
2. SETS依赖帧间差异，对于缓慢发生的异常（如逐渐升温的火灾）可能不够敏感
3. TETG的异常类型模板是预定义的，面对全新异常类型时可能受限
4. 仅验证了Vicuna-7B规模的LLM，更大规模模型的效果有待探索

## 相关工作与启发

- **LLaMA-VID**：本文的基线模型，VA-GPT在其基础上引入Token选择机制
- **Hawkeye**：另一个异常感知视频MLLM，但未区分Token的重要性
- **DINOv2**：本文用作SETS中的特征提取器，其自监督特征的稳定性是帧间差异计算的基础
- 启发：Token级别的选择性处理思路可推广到其他需要聚焦特定信息的MLLM任务

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](distime_distribution-based_time_representation_for_video_large_language_models.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](../../CVPR2025/video_understanding/video_summarization_with_large_language_models.md)
- [\[NeurIPS 2025\] FastVID: Dynamic Density Pruning for Fast Video Large Language Models](../../NeurIPS2025/video_understanding/fastvid_dynamic_density_pruning_for_fast_video_large_languag.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](../../CVPR2025/video_understanding/on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)

<!-- RELATED:END -->
