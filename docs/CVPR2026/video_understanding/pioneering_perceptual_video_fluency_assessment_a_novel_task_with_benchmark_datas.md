---
title: >-
  [论文解读] Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline
description: >-
  [CVPR 2026][视频理解][视频流畅度评估] 本文首次将视频流畅度评估（VFA）从传统视频质量评估（VQA）中独立出来，构建了首个流畅度评估数据集 FluVid（4,606 视频），并提出 FluNet 基线模型，通过时序排列自注意力（T-PSA）实现高效帧间交互…
tags:
  - "CVPR 2026"
  - "视频理解"
  - "视频流畅度评估"
  - "时序质量"
  - "基准数据集"
  - "自注意力"
  - "自监督学习"
---

# Pioneering Perceptual Video Fluency Assessment: A Novel Task with Benchmark Dataset and Baseline

**会议**: CVPR 2026  
**arXiv**: [2603.26055](https://arxiv.org/abs/2603.26055)  
**代码**: [https://github.com/KeiChiTse/VFA](https://github.com/KeiChiTse/VFA)  
**领域**: 视频理解  
**关键词**: 视频流畅度评估, 时序质量, 基准数据集, 自注意力, 自监督学习

## 一句话总结

本文首次将视频流畅度评估（VFA）从传统视频质量评估（VQA）中独立出来，构建了首个流畅度评估数据集 FluVid（4,606 视频），并提出 FluNet 基线模型，通过时序排列自注意力（T-PSA）实现高效帧间交互，SRCC/PLCC 分别达到 0.816/0.821。

## 研究背景与动机

**领域现状**：视频质量评估（VQA）是目前量化视频主观感受的主流方法，已有大量模型（如 Fast-VQA、DOVER）被广泛使用。VQA 将空间质量（噪声、色彩等）和时序质量（运动一致性、帧连续性等）混合在一起进行整体评估。

**现有痛点**：作者通过先导实验发现，现有 VQA 模型的预测高度偏向空间质量，而对时序维度（即流畅度）的评估能力严重不足。这导致 VQA 预测分数无法有效指导自适应帧率编码、帧插值等时序相关的下游任务。

**核心矛盾**：VQA 模型的"空间-时序"纠缠使得流畅度信号被大幅稀释。人眼对时序失真比空间失真更敏感，但模型输出却恰恰相反。造成这一问题的根本原因有三：（1）缺乏独立的流畅度评分标准；（2）缺乏大规模的流畅度标注数据集；（3）缺乏针对流畅度设计的模型架构。

**本文目标** 将视频流畅度评估形式化为独立感知任务 VFA；构建首个流畅度评分标准和数据集 FluVid；设计流畅度感知baseline FluNet。

**切入角度**：从视觉心理学和认知科学出发，流畅度由三个核心视频成分决定——前景、背景和相机运动。同时，现有方法的主要障碍是输入帧数不足和帧间交互不充分。

**核心 idea**：通过通道压缩+时序维度排列的自注意力机制（T-PSA），在保持计算量可控的前提下大幅扩展时序窗口，配合自监督排序训练策略，让模型学会感知流畅度差异。

## 方法详解

### 整体框架

FluNet 包含三个部分：patch embedding 层 $F_p$（单层卷积）、编码器 $F_e$（含 T-PSA 的四阶段 Transformer）和 VFA 预测头 $F_h$（两层逐点卷积）。输入视频 $V \in \mathbb{R}^{T \times H \times W \times 3}$ 先经过 $F_p$ 映射为特征图，再逐层编码，最后通过 $F_h$ 回归流畅度分数。整个架构基于 Swin Transformer 的层次化设计，四个 stage 分别含 (2,2,6,2) 个 T-PSA block。

### 关键设计

1. **时序排列自注意力（T-PSA）**:

    - 功能：在扩大时序感受野的同时保持计算效率
    - 核心思路：标准自注意力中 $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ 的通道维度均为 $C$。T-PSA 将 $\mathbf{K}$, $\mathbf{V}$ 的通道压缩为 $C/\gamma$（$\gamma=2$），然后将时序 token 排列到通道维度，使得 $\mathbf{K}_p, \mathbf{V}_p$ 的窗口从 $(D,S,S)$ 变为 $(D/\gamma,S,S)$，但通道维度恢复为 $C$，从而可以与 $\mathbf{Q}$ 正常计算注意力。这样时序窗口可以从 8 扩大到 32，而 GFLOPs 反而从 1114 降到 308
    - 设计动机：传统方法用 32 帧稀疏采样无法捕捉微妙的流畅度变化；直接增加帧数则计算爆炸。T-PSA 只扩展时序窗口 $D$ 而固定空间窗口 $S$，实现"聚焦流畅度而非空间细节"

2. **自监督排序训练策略**:

    - 功能：在无流畅度标注的情况下让模型学会判断流畅度等级
    - 核心思路：从 HD-VILA 数据集采样 2,000 个高质量锚定视频，对每个视频通过随机丢帧+复制帧合成 $K=7$ 个不同流畅度等级的视频。丢帧率按等级递增，丢帧位置随机分布在 $M=5$ 个时间区间。排序损失为 margin ranking loss：$\mathcal{L}_{\text{rank}} = \frac{1}{K}\sum_{i=0}^{K-1}\max(0, \hat{y}_{i+1} - \hat{y}_i + \beta)$，其中 $\beta=0.4$
    - 设计动机：流畅度标注需要专业人员在实验室环境下进行，成本极高。通过合成不同流畅度等级的视频对进行排序学习，可以用无标签数据训练模型的流畅度排序能力

3. **FluVid 数据集构建**:

    - 功能：提供首个面向流畅度评估的基准数据集
    - 核心思路：基于两个设计原则——（1）按影响流畅度的三个视频成分（前景/背景/相机）收集视频；（2）确保内容和参数多样性。从 SSv2 和 5 个 UGC-VQA 数据集中筛选 4,606 个视频，20 位专家按 5 级 ACR 标准标注流畅度 MOS
    - 设计动机：现有 VQA 数据集聚焦整体质量，缺乏流畅度中心的数据和标注，无法支撑 VFA 模型的训练和评估

### 损失函数 / 训练策略

训练分三阶段：（1）可选的 LSVQ 预训练使模型具备质量感知能力（FluNet++）；（2）排序学习阶段使用 $\mathcal{L}_{\text{rank}}$ 在 16,000 个合成视频上训练 30 个 epoch；（3）微调阶段使用 606 个 FluVid 视频的 L1 损失 $\mathcal{L}_{\text{ft}} = \|\hat{y}_b - y_b\|_1$ 训练 60 个 epoch。

## 实验关键数据

### 主实验

| 方法 | 类型 | 帧数 | 窗口大小 | GFLOPs | SRCC↑ | PLCC↑ |
|------|------|------|---------|--------|-------|-------|
| Fast-VQA | VQA | 32 | (8,7,7) | 279 | 0.640 | 0.633 |
| DOVER | VQA | - | - | - | 0.638 | 0.614 |
| Qwen 2.5-VL | LMM | - | - | - | 0.598 | 0.584 |
| FineVQ | LMM | - | - | - | 0.622 | 0.609 |
| Fast-VQA+128帧+排序+微调 | VQA | 128 | (8,7,7) | 1114 | 0.725 | 0.716 |
| **FluNet (Ours)** | VFA | 128 | (32,7,7) | 308 | **0.774** | **0.770** |
| **FluNet++ (Ours)** | VFA | 128 | (32,7,7) | 308 | **0.816** | **0.821** |

### 消融实验

| 配置 | SRCC↑ | PLCC↑ | 说明 |
|------|-------|-------|------|
| 仅排序学习 | 0.722 | 0.718 | 排序学习有效 |
| 仅微调 | 0.710 | 0.693 | 微调也有效 |
| 联合训练 | 0.753 | 0.748 | 联合效果不如分阶段 |
| 排序→微调 | 0.774 | 0.770 | 分阶段最优 |
| 窗口(8,7,7) | 0.736 | 0.722 | 小窗口性能差 |
| 窗口(16,7,7) | 0.758 | 0.749 | 窗口越大越好 |
| 窗口(32,7,7) | 0.774 | 0.770 | 最优窗口大小 |
| Stage 1-3 用 T-PSA | 0.779 | 0.766 | 最佳阶段配置 |
| 全部 4 个 stage 用 T-PSA | 0.774 | 0.770 | 第4阶段加不加差别不大 |

### 关键发现

- FluNet 在 GFLOPs 仅为 308 的情况下超越了 1114 GFLOPs 的 Fast-VQA（+4.9% SRCC），证明 T-PSA 的效率优势
- 增加输入帧数（32→128）对所有方法都有增益，但 T-PSA 的独特优势在于可以同时扩大时序窗口
- 排序→微调的分阶段策略优于联合训练，说明先学会排序再校准分数是更好的学习路径
- VQA 方法整体优于 LMM，说明精细的质量感知能力比通用理解更重要；但 LMM 中 Qwen 2.5-VL 表现最好，受益于其高帧率处理能力

## 亮点与洞察

- **T-PSA 的通道-时序维度交换**是一个非常巧妙的设计。通过压缩 K/V 通道维度然后将时序 token 排列到通道维度，在不增加计算量的情况下实现了 4 倍时序窗口扩展。这个思路可以迁移到任何需要长时序建模的视频任务中
- **将 VFA 从 VQA 中独立出来的洞察**本身有很大价值。作者通过定量实验证明 VQA 模型偏向空间质量，这一发现对视频生成评估也很有启发——当前用 VQA 指标评估视频生成质量可能严重低估了时序问题
- **合成排序训练**巧妙地解决了标注稀缺问题，丢帧+复制帧的合成方式虽然简单，但有效模拟了真实世界的卡顿

## 局限与展望

- FluVid 数据集仅 4,606 个视频，规模有限且主要来自 UGC 视频，缺少 AI 生成视频
- 合成的排序训练数据仅模拟了丢帧卡顿，未覆盖其他流畅度问题（如运动模糊、帧率不稳定等）
- T-PSA 的通道压缩比 $\gamma$ 固定为 2，没有探索自适应压缩策略
- 未考虑将 VFA 与 VQA 结合做联合预测的方案，而这在实际应用中可能更有价值

## 相关工作与启发

- **vs Fast-VQA**：Fast-VQA 使用稀疏帧采样+固定窗口的 attention，FluNet 通过 T-PSA 在相同计算量下实现了更大的时序窗口和更多输入帧，SRCC 从 0.640 提升到 0.774
- **vs LMM 方法（Qwen 2.5-VL, FineVQ）**：LMM 具有强大的语义理解能力但缺乏细粒度的流畅度感知。质量感知的 LMM（如 FineVQ）表现优于通用 LMM，但仍不及专门设计的 VFA 方法
- 这项工作对视频生成评估有很大启发：当前用 FVD 等指标评估视频生成质量时，流畅度维度可能被严重忽视

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义 VFA 任务并构建完整生态（标准+数据+方法），T-PSA 设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 23 种方法的全面 benchmark + 详尽消融实验 + 多维度分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，行文结构完整，figure 设计直观
- 价值: ⭐⭐⭐⭐ 填补了流畅度评估空白，对视频生成质量评估和视频处理优化有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TacSIm: A Dataset and Benchmark for Football Tactical Style Imitation](tacsim_a_dataset_and_benchmark_for_football_tactical_style_imitation.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)

</div>

<!-- RELATED:END -->
