---
title: >-
  [论文解读] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences
description: >-
  [ICCV 2025][图像生成][循环一致性] 利用循环一致性（图→文→图或文→图→文的重建相似度）作为替代人类偏好的监督信号，构建866K偏好数据集CyclePrefDB，训练的CycleReward模型在详细描述生成评估上超越所有现有方法，并可通过DPO提升VLM和扩散模型。
tags:
  - ICCV 2025
  - 图像生成
  - 循环一致性
  - 奖励模型
  - 图文对齐
  - 偏好学习
  - DPO
---

# Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences

**会议**: ICCV 2025  
**arXiv**: [2506.02095](https://arxiv.org/abs/2506.02095)  
**代码**: [https://cyclereward.github.io/](https://cyclereward.github.io/)  
**领域**: Image Generation / Vision-Language Alignment  
**关键词**: 循环一致性, 奖励模型, 图文对齐, 偏好学习, DPO

## 一句话总结

利用循环一致性（图→文→图或文→图→文的重建相似度）作为替代人类偏好的监督信号，构建866K偏好数据集CyclePrefDB，训练的CycleReward模型在详细描述生成评估上超越所有现有方法，并可通过DPO提升VLM和扩散模型。

## 研究背景与动机

图文对齐度量 $d(x,y)$ 是多模态学习的核心问题，广泛用于评估VLM/T2I模型和通过RLHF改进对齐。然而现有方法面临关键瓶颈：

**人类偏好数据昂贵且难以规模化**：ImageReward、HPSv2、PickScore等依赖大规模人类标注

**AI反馈（如GPT-4V）成本高且受限**：闭源、限速、长期可用性不保证

**现有指标对长文本描述评估不足**：大多数偏好数据集文本较短（~20-35 tokens），无法评估详细描述

**直接计算循环一致性虽可行但低效且不可微**：需要运行完整的T2I/I2T模型

核心洞察：**将文本映回图像空间后比较重建图像和原始图像**比直接比较文本和图像容易得多。更准确的描述会产生更接近原图的重建结果。

## 方法详解

### 整体框架

给定图到文映射 $F: X \to Y$ 和文到图映射 $G: Y \to X$，定义循环一致性分数：
- 图到文：$s(x \to F(x)) := d_{\text{img}}(x, G(F(x)))$，使用DreamSim计算图像相似度
- 文到图：$s(y \to G(y)) := d_{\text{text}}(y, F(G(y)))$，使用SBERT计算文本相似度

将循环一致性分数转换为偏好对：若 $s(x \to y_i) > s(x \to y_j)$，则 $y_i \succ y_j$。

### 关键设计

1. **CyclePrefDB偏好数据集构建**:

    - 使用DCI数据集（7.6K高分辨率图像+密集描述）作为输入
    - **图到文**：11个I2T模型（BLIP2到InternVL2-40B）对每张图生成多个候选描述，固定SD3为反向映射计算 $s(x \to y)$。刻意包含旧模型产生的短/幻觉描述作为负例
    - **文到图**：4个T2I模型（SD1.5到FLUX）各用3个随机种子生成图像，固定LLaVA-1.5-13B为反向映射计算 $s(y \to x)$
    - 文本限制在77 tokens内（T2I模型提示长度限制）
    - 最终产生866K偏好对（398K I2T + 468K T2I）

2. **CycleReward奖励模型训练**:

    - 骨干：BLIP（ViT-L/16编码器 + BERTbase文本编码器 + 5层MLP），共477M参数
    - 三种变体：CycleReward-I2T / T2I / Combo（联合训练）
    - I2T损失：$\mathcal{L}_{\text{img}} = -\mathbb{E}[\log \sigma(r_\theta(x,y_i) - r_\theta(x,y_j))]$
    - T2I损失：$\mathcal{L}_{\text{text}} = -\mathbb{E}[\log \sigma(r_\theta(x_i,y) - r_\theta(x_j,y))]$
    - Combo联合损失：$\mathcal{L} = \mathcal{L}_{\text{text}} + \lambda \mathcal{L}_{\text{img}}$（$\lambda=1$）

3. **DPO应用**:

    - I2T方向：对Qwen-VL-Chat使用CyclePrefDB-I2T进行DPO微调
    - T2I方向：对Stable Diffusion 1.5使用CyclePrefDB-T2I进行Diffusion DPO
    - 无需人类标注即可提升多种下游任务

### 损失函数 / 训练策略

- 奖励模型训练使用标准Bradley-Terry偏好学习损失
- DPO训练直接在偏好数据上优化模型，无需显式奖励建模
- 关键设计决策：使用DreamSim（建模人类视觉相似度）和SBERT计算循环一致性分数，消融实验证明优于LPIPS、CLIP、BERTScore

## 实验关键数据

### 主实验

图文对齐指标评估（Pairwise Accuracy %）：

| 方法 | DetailCaps-4870 | GenAI-Bench | 监督信号 |
|------|----------------|-------------|----------|
| CLIPScore | 51.66 | 49.73 | 无（预训练）|
| VQAScore (11B) | 50.24 | **64.13** | 无（预训练）|
| HPSv2 | 54.34 | 56.13 | 人类偏好 |
| PickScore | 51.01 | 57.05 | 人类偏好 |
| ImageReward | 50.70 | 56.70 | 人类偏好 |
| Raw Cycle Consistency | 56.46 | 52.52 | 循环一致性 |
| **CycleReward-Combo** | **60.50** | 55.52 | **循环一致性** |

CycleReward在详细描述评估上超越所有方法（包括人类偏好训练的模型），比VQAScore (11B)高10.26%，而模型仅477M参数（小24倍）。

DPO结果（I2T方向，Qwen-VL-Chat）：

| 模型 | DeCapBench | LLaVA-WD | MMHalBench | MMEP |
|------|-----------|----------|------------|------|
| 基线 | 26.47 | 61.67 | 2.99 | 1460.2 |
| DPO w/ VLFeedback | 28.03 | 69.17 | 3.32 | 1551.5 |
| **DPO w/ CyclePrefDB-I2T** | **30.63** | **70.00** | 3.11 | 1485.7 |

### 消融实验

循环一致性与人类偏好的一致性（Agreement Rate %）：

| 方法 | RLHF-V | POVID | HPDv2 | PaPv2 | IRDB | 均值 |
|------|--------|-------|-------|-------|------|------|
| GPT-4o | 61.3 | 60.0 | 48.1 | 45.8 | 24.8 | 48.0 |
| Raw Cycle | 58.6 | 61.2 | 60.5 | 59.8 | 54.5 | 58.9 |
| **CycleReward-Combo** | **66.5** | 63.8 | **67.7** | **65.8** | **61.3** | **65.0** |

解码器消融：使用更强的语言模型（InternVL2-26B替代LLaVA-1.5-13B）作为I2T解码器，DetailCaps评分从51.74提升到57.21。

### 关键发现

- 循环一致性信号在I2T/T2I两个方向上都比GPT-4o标注更稳定（GPT-4o在T2I评估上一致率最低24.8%）
- 训练奖励模型比直接使用原始循环一致性分数效果更好，证明蒸馏的有效性
- CyclePrefDB-I2T的DPO不仅提升captioning，还提升感知、推理、减少幻觉——尽管数据只包含captioning指令
- Best-of-N采样中CycleReward在详细描述任务上改善最大，且对长文本提示的T2I生成也有优势
- DreamSim（建模人类视觉相似度）和SBERT是最优的相似度度量选择

## 亮点与洞察

- **核心思想简洁优雅**：循环一致性提供了一种无需人类标注的自监督对齐信号，概念上极易理解
- **跨任务泛化强**：一个信号同时用于I2T和T2I两个方向，且DPO训练后泛化到captioning之外的多种VL任务
- **实用价值高**：CycleReward快速（推理时无需运行T2I模型）、可微分、477M参数轻量
- **数据效率**：CyclePrefDB规模小于VLFeedback/Pick-a-Pic，但效果相当甚至更优
- **消融全面**：相似度度量、解码器选择、数据规模、过滤策略等都有详细分析

## 局限与展望

- 监督质量依赖于预训练解码器的重建质量，生成错误可能误导偏好
- 文本长度受限于T2I模型的77 token限制，无法评估真正的长文本描述
- 在T2I评估上，VQAScore (11B)仍然优于CycleReward——循环一致性在文到图方向的信号质量仍有提升空间
- 未探索视频-语言或音频-文本等其他模态对的循环一致性
- 可进一步研究多轮循环（image→text→image→text→...）是否能提供更强信号

## 相关工作与启发

- 循环一致性在无配对数据学习中有悠久历史（CycleGAN等），本文将其扩展到跨模态对齐的偏好学习
- Image2Text2Image直接使用循环一致性作为评估指标，本文进一步蒸馏为可学习的奖励模型，获得速度和性能的双重提升
- 与RLHF/DPO方向形成互补：提供了一种不依赖人类或强AI标注的偏好数据构建方式
- DPO数据集设计的关键洞察：包含多样化质量的模型输出比仅使用最强模型更重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ 循环一致性概念非新（已有Image2Text2Image），但蒸馏为奖励模型+构建大规模偏好数据集是显著创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 指标评估+BoN采样+DPO三个维度，I2T和T2I双方向，消融极其详尽
- **写作质量**: ⭐⭐⭐⭐⭐ 逻辑清晰，动机有力，图表直观，实验设计严谨
- **价值**: ⭐⭐⭐⭐⭐ 提供了廉价可扩展的对齐信号，对RLHF/偏好学习领域有重要启发价值
