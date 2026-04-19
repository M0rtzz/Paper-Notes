---
title: >-
  [论文解读] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos
description: >-
  [CVPR 2026][视频理解][错误归因] 本文提出 Mistake Attribution (MATT) 任务，将第一人称视频中的操作错误归因到语义（违反了指令的哪个成分）、时间（不可逆转点 PNR 在哪一帧）和空间（PNR 帧中错误区域在哪里）三个维度，通过 MisEngine 数据引擎自动从已有动作数据集构建大规模错误样本，并设计统一的 Transformer 模型 MisFormer 同时完成三个归因子任务，在多个基准上超越各子任务的专用 SOTA 方法。
tags:
  - CVPR 2026
  - 视频理解
  - 错误归因
  - 第一人称视频
  - 语义角色标注
  - 时空定位
  - 指令对齐
---

# Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos

**会议**: CVPR 2026  
**arXiv**: [2511.20525](https://arxiv.org/abs/2511.20525)  
**代码**: https://yayuanli.github.io/MATT  
**领域**: 视频理解  
**关键词**: 错误归因、第一人称视频、语义角色标注、时空定位、指令对齐

## 一句话总结

本文提出 Mistake Attribution (MATT) 任务，将第一人称视频中的操作错误归因到语义（违反了指令的哪个成分）、时间（不可逆转点 PNR 在哪一帧）和空间（PNR 帧中错误区域在哪里）三个维度，通过 MisEngine 数据引擎自动从已有动作数据集构建大规模错误样本，并设计统一的 Transformer 模型 MisFormer 同时完成三个归因子任务，在多个基准上超越各子任务的专用 SOTA 方法。

## 研究背景与动机

1. **领域现状**：物理环境下的 AI 辅助系统（如烹饪指导、组装指导）需要理解人类在执行指令时犯的错误。现有方法主要停留在错误检测层面——判断某步骤是否出错——或者给出粗粒度的错误类别（如"步骤遗漏"、"动作偏差"）。
2. **现有痛点**：粗粒度检测无法告诉用户"指令的哪个部分没有被正确执行"（语义维度）、"错误在什么时候变得不可挽回"（时间维度）以及"PNR 帧中错误具体出现在哪个区域"（空间维度）。例如指令是"拿起锤子"但实际拿了螺栓，现有方法只能告诉你"出错了"，无法指出是"物体"角色出错、出错在第 17 帧、错误区域是红色框中的螺栓。
3. **核心矛盾**：构建细粒度错误数据集极其困难——真实错误随着收集者经验增长变得越来越稀少，而人为注入的错误又会引入视觉偏差。已有错误数据集（EgoPER 599 样本、Assembly101 707 样本）规模比通用动作数据集小两个数量级。
4. **本文目标** (a) 如何大规模自动构建含语义-时间-空间三元组标注的错误数据集；(b) 如何用一个统一模型同时完成三个归因任务。
5. **切入角度**：利用语义角色标注（SRL）对动作描述进行结构化解析，然后在现有动作识别数据集中进行跨匹配（cross-matching），将"拿起筛子"的指令文本与"拿起平底锅"的视频配对，自动产生语义归因标签，同时继承原始数据集中的 PNR 时间戳和手部/物体空间标注。
6. **核心 idea**：通过语义角色交叉匹配从大规模动作语料自动构建错误样本，并用统一 Transformer 同时做语义-时间-空间三维归因。

## 方法详解

### 整体框架

输入是一段指令文本 $T$（如"cut the apple"）和一段用户执行视频 $V$，输出三元组：(1) 每个语义角色是否出错的标签 $\{y_r\}$，(2) PNR 帧时间戳 $t_{PNR}$，(3) PNR 帧中的错误区域边界框 $B_{t_{PNR}}$。系统分为两大部分：MisEngine 数据引擎负责自动构建训练数据，MisFormer 模型负责推理归因。

### 关键设计

1. **MisEngine 数据引擎**:

    - 功能：从现有动作识别数据集全自动构建带三维归因标注的错误样本
    - 核心思路：三步流程——(1) 用 AllenNLP SRL 将每条动作描述解析为语义角色组（如谓词 "Pick up"、宾语 "the sieve"）；(2) 跨样本比较每对动作描述在每个角色上是否一致，产生 $C=|\mathcal{R}|^2$ 种错配类别（谓词错、宾语错、都错、都对）；(3) 从每类错配中采样若干动作描述及其视频作为错误尝试。语义标注由交叉匹配直接产生，时间标注继承原数据集的 PNR 帧标注，空间标注继承手部/物体边界框。最终从 Ego4D 和 EPIC-KITCHENS 分别产生 257K 和 221K 样本，比现有最大错误数据集大两个数量级。
    - 设计动机：绕过真实错误收集的稀缺性和注入错误的视觉偏差问题，将错误构建转化为已有数据的组合问题

2. **MisFormer 特征提取与投影**:

    - 功能：提取视频和文本的共享多模态特征
    - 核心思路：用 InternVideo2 的文本编码器对每个语义角色子串分别编码得到 $F_R^T \in \mathbb{R}^{|\mathcal{R}| \times d}$，用视频编码器提取 $F^V \in \mathbb{R}^{L \times K \times d}$。然后通过投影块 $\mathcal{P}$（2 层 Transformer 解码器，无因果掩码），先对各角色文本特征做自注意力以交换角色间信息，再对视频特征做交叉注意力以注入视觉上下文，得到投影后的 $F_R^{T'} \in \mathbb{R}^{|\mathcal{R}| \times d}$。
    - 设计动机：InternVideo2 预训练已使文本和视频特征在同一嵌入空间，投影块进一步适配错误理解任务

3. **三个归因头（语义/时间/空间）**:

    - 功能：分别输出语义角色错误标签、PNR 帧定位、错误区域边界框
    - 核心思路：
        - **语义头**：对每个角色的投影特征 $F_r^{T'}$ 过 FFN + sigmoid，二分类输出该角色是否出错，用 BCE 损失训练
        - **时间头**：先用 2 层自注意力将逐帧视频特征 $F^V$ 下采样为帧级特征 $F^{V'} \in \mathbb{R}^{L \times d}$，再用 2 层 Transformer 解码器（$F^{V'}$ 为 query，$F_R^{T'}$ 为 key/value）生成每帧概率分布，取 argmax 为 PNR 帧，用交叉熵损失训练
        - **空间头**：从投影块最后一层交叉注意力中提取 PNR 帧对应的注意力权重，与投影文本特征拼接后过两层自注意力生成空间显著图，上采样后与 PNR 帧 RGB 拼接为 4 通道输入，用轻量 CNN 回归边界框坐标，用 Huber 损失训练
    - 设计动机：推理时时间和空间头通过门控机制——仅当语义头检测到至少一个角色出错时才触发，减少不必要计算

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_S + \mathcal{L}_T + \mathcal{L}_{spatial}$，其中 $\mathcal{L}_S$ 为二元交叉熵（语义），$\mathcal{L}_T$ 为交叉熵（时间，仅对错误样本计算），$\mathcal{L}_{spatial}$ 为 Huber 损失（空间）。

## 实验关键数据

### 主实验

| 数据集 | 任务 | 指标 | MisFormer | 之前最佳 | 提升 |
|--------|------|------|-----------|----------|------|
| EPIC-KITCHENS-M | 语义归因 | F1@0.5 | **83.89** | 77.23 (ChatGPT-4o) | +6.66% |
| Ego4D-M | 语义归因 | F1@0.5 | **56.24** | 50.95 (ChatGPT-4o) | +5.29% |
| Ego4D-M | 时间归因 | MAE(s) | **0.638** | 0.816 (EgoT2) | -21.81% |
| Ego4D-M | 空间归因 | mIoU | **59.21** | 49.88 (MediaPipe-U) | +18.70% |
| Ego4D-M | 错误检测 | F1@0.5 | **57.55** | 15.62 (EgoPED) | +41.93% |

### 消融实验

| 配置 | 语义 F1 | 时间 MAE(s) | 空间 mIoU | 检测 F1 |
|------|---------|-------------|-----------|---------|
| MisFormer (完整) | 56.24 | 0.438 | 59.21 | 57.55 |
| 换为 LaViLa 骨干 | 49.16 | 0.561 | 51.37 | 46.05 |
| 去掉投影块 $\mathcal{P}$ | 51.34 | 0.457 | 55.43 | 52.75 |
| 去掉时间归因训练 | 51.29 | 0.623 | 57.78 | 57.46 |
| 用 GradCAM 代替注意力热图 | 55.52 | 0.482 | 55.03 | 57.51 |

### 关键发现
- InternVideo2 骨干（多模态预训练）对 MATT 至关重要，换为 LaViLa 后各子任务全面下降
- 投影块 $\mathcal{P}$ 不可或缺——原始文本嵌入不足以捕捉指令与视频间的细微偏差
- Object 角色的归因始终比 Predicate 容易，表明细粒度动作理解仍是第一人称视频的难点
- 在现有小规模错误数据集 EgoPER 上从头训练效果不佳，但在 EPIC-KITCHENS-M 上预训练后微调可达到竞争力

## 亮点与洞察
- **MisEngine 的"零成本标注"设计**非常巧妙：通过语义角色交叉匹配，把已有动作识别数据变成了错误理解数据，三维标注全部继承，无需任何人工标注。这种"不采集真实错误，而是组合正确样本以模拟错误"的思路可以迁移到其他需要稀缺负样本的任务。
- **统一模型 vs 组合专家**：MisFormer 以一个统一模型在语义/时间/空间/检测四个任务上均超越或接近各自的专用 SOTA，且仅 41M+投影头参数，运行效率高（空间头 68.9 FPS）。
- 将错误理解形式化为"指令-执行偏差"的三维归因框架，为 AI 助手提供了可解释、可操作的反馈信息。

## 局限与展望
- 当前仅支持简短指令（谓词+宾语），真实场景中的长句、多步骤指令需要更丰富的角色集合
- 空间归因比专用手-物交互检测器（SSDA）弱（mIoU 59.21 vs 64.54），可考虑引入专用物体检测先验
- 数据引擎假设错误仅来自角色交叉匹配，无法覆盖"程度错误"（如切得太厚）等连续性偏差
- 预训练表征是通用视频语言对齐目标，设计专门面向错误理解的预训练目标或许能进一步提升

## 相关工作与启发
- **vs EgoPED / AMNAR**: 这些方法将错误检测视为 OOD 检测问题，未利用错误监督信号，在大规模多活动场景下失效；MisFormer 通过 MisEngine 的大规模监督学习扩展能力
- **vs MistScene**: MistScene 生成自然语言解释但无结构化归因，且不与指令对齐；MATT 提供结构化的语义-时间-空间三元组
- **vs ChatGPT-4o**: 闭源商业模型在语义归因上被 MisFormer 超越 6.66%，说明任务特定设计优于通用大模型

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 提出全新任务定义+数据引擎+统一模型，三维贡献完整
- 实验充分度: ⭐⭐⭐⭐ 覆盖四个子任务+人类验证+消融，但空间归因对比还可更深入
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，图示优秀，逻辑递进流畅
- 价值: ⭐⭐⭐⭐ 为第一人称 AI 助手的错误反馈提供了完整方法论，数据引擎思路可广泛复用

<!-- RELATED:START -->

## 相关论文

- [Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](streamgaze_gaze-guided_temporal_reasoning_and_proactive_understanding_in_streami.md)
- [LensWalk: Agentic Video Understanding by Planning How You See in Videos](lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)

<!-- RELATED:END -->
