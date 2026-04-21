---
title: >-
  [论文解读] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding
description: >-
  [CVPR 2025][视频理解][视频时序理解] Seq2Time 提出了一个数据驱动的训练范式，通过将大规模图像序列和短视频片段转化为模拟长视频时序结构的训练数据，并引入统一相对位置 token 表示，在不需要大量时间戳标注的情况下显著提升了视频 LLM 的时序理解能力（YouCook2 F1 提升 27.6%，Charades-STA R@1 提升 14.7%）。
tags:
  - CVPR 2025
  - 视频理解
  - 视频时序理解
  - 大语言模型
  - 时序定位
  - 密集视频描述
  - 知识迁移
---

# Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding

**会议**: CVPR 2025  
**arXiv**: [2411.16932](https://arxiv.org/abs/2411.16932)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 视频时序理解, 大语言模型, 时序定位, 密集视频描述, 知识迁移

## 一句话总结

Seq2Time 提出了一个数据驱动的训练范式，通过将大规模图像序列和短视频片段转化为模拟长视频时序结构的训练数据，并引入统一相对位置 token 表示，在不需要大量时间戳标注的情况下显著提升了视频 LLM 的时序理解能力（YouCook2 F1 提升 27.6%，Charades-STA R@1 提升 14.7%）。

## 研究背景与动机

**领域现状**：视频大语言模型（Video LLMs）在通用视频理解上取得了显著进展，但时序感知能力仍然不足。时序敏感的 Video LLM（如 TimeChat、VTimeLLM）通过架构创新（双 Q-Former、多阶段训练等）来提升时序理解，但依赖大量带有精确时间戳标注的长视频数据。

**现有痛点**：(1) 时间戳标注极度稀缺——TimeIT 仅 125K 视频、VTG-IT 仅 120K，远少于通用视频数据集（VideoChat2 有 800K+）；(2) 减少 12% 的训练数据就会导致性能下降 13.4%，移除任务相关数据更是暴跌 65.5%；(3) 现有数据集的字幕质量较低，视觉-语言对齐精度不如高质量图像数据集（如 LLaVA-150K）。

**核心矛盾**：训练数据（尤其是带时间戳标注的长视频）的匮乏严重制约了 Video LLM 的时序理解能力，但标注成本高昂难以大规模获取。

**本文目标**：利用丰富的图像和短视频数据来增强 Video LLM 的时序理解能力，避免对稀缺的长视频时间戳标注的依赖。

**切入角度**：Video LLM 并不真正"感知时间"，而是识别视觉内容与其在序列中位置的对应关系。既然如此，可以用图像序列中的索引-内容对应关系来模拟视频中的时间戳-事件对应关系。大规模图像和短视频数据天然拥有丰富的的位置-内容对应信息。

**核心 idea**：将图像序列中的位置索引转化为时间标注，设计三种前置任务（图像索引定位、索引图像描述、相邻位置推理）训练 Video LLM 学习序列-内容对应，再通过统一相对位置 token 将位置知识迁移到长视频时序理解。

## 方法详解

### 整体框架

Seq2Time 包含三个数据组件：(1) **图像序列数据**——从 LLaVA-ReCap 数据集采样 96 张图像组成序列，设计三种前置任务学习索引-内容对应；(2) **短视频片段序列数据**——从 Kinetics-700 采样 2-10 个短视频片段拼成长序列，训练密集描述和时序定位；(3) **统一相对位置 token**——将图像索引、视频帧位置统一编码为 4 位小数相对位置，仅需 10 个新 token（<0>-<9>）。

### 关键设计

1. **图像序列前置任务（Image Sequence Pretext Tasks）**:

    - 功能：让 Video LLM 学习"从描述定位内容位置"和"从位置生成内容描述"的能力
    - 核心思路：从 COCO118K、BLIP558K、CC3M 等高质量图像-字幕数据集中随机采样 96 张图像组成序列，设计三种互补任务：(a) **图像索引定位（IIG）**——给定字幕找到对应图像索引，模拟时间定位；(b) **索引图像描述（IIC）**——给定索引生成对应图像描述，模拟密集描述；(c) **相邻位置推理（ALR）**——给定某图像描述，识别并描述其前/后相邻图像。每类 100K 实例，共 300K。
    - 设计动机：图像字幕质量远高于视频字幕（逐图描述 vs 视频级描述），三种任务分别强化了定位、描述和序列推理能力。消融实验证明 IIG 对整体时序理解贡献最大，IIC 主要提升文本生成质量，ALR 增强描述丰富度。

2. **短视频片段序列数据（Clip Sequence Data）**:

    - 功能：用更贴近真实长视频的数据增强时序感知
    - 核心思路：从 Kinetics-700 中采样 2-10 个不同动作类别的短片段，使用 LongVA 为每个片段生成详细字幕（以动作标签为条件），然后拼接为模拟长视频。故意使用不均匀的帧率采样来避免均匀时间间距，模拟真实视频的多事件结构。基于片段位置生成密集描述和时序定位的训练数据（100K 实例）。
    - 设计动机：图像序列虽然任务更难、字幕质量更高，但与真实视频存在模态差异。片段序列在数据特性和训练目标上更接近真实长视频，两种数据互补——图像序列强化精细定位，片段序列增强视频级理解。

3. **统一相对位置 Token（Unified Relative Position Token）**:

    - 功能：在 LLM 嵌入空间中桥接图像索引和视频时间戳
    - 核心思路：将所有位置（图像索引或视频帧位置）归一化为 4 位小数：$I_{\text{norm}} = \text{round}(i/L, 4)$，其中 $i$ 为索引、$L$ 为序列长度。例如 96 张图像序列中第 7 张编码为 0.0729 → <0><7><2><9>。仅向 LLM 词表添加 10 个新 token（<0> 到 <9>），每个数字作为可学习嵌入。推理时可通过视频帧率将相对位置还原为绝对时间戳。
    - 设计动机：(1) 绝对时间在不同帧率的视频间不可比，相对位置更通用；(2) 4 位精度对 1 分钟 30fps 视频采样 96 帧仅有 0.13% 平均误差；(3) 层次化结构——第一位表示粗略位置，后续位提供细粒度定位，适合不同尺度的时序理解；(4) 10 个 token 极度高效，不增加词表负担。

### 损失函数 / 训练策略

两阶段训练：(1) 使用完整数据集（Seq2Time 400K + TimeIT 110K + Valley 40K + ShareGPT4Video 93K）训练 1 epoch；(2) 仅用 TimeIT + Valley 数据微调 3 epochs。使用 LoRA rank=32，批大小 8，每视频采样 96 帧。标准自回归交叉熵损失。

## 实验关键数据

### 主实验

| 方法 | YouCook2 SODA_c↑ | CIDEr↑ | F1↑ | Charades R@1(0.5)↑ | R@1(0.7)↑ |
|------|-----------------|--------|------|-------------------|-----------|
| VTimeLLM | - | - | - | 27.5 | 11.4 |
| TimeChat | 1.0 | 2.9 | 12.7 | 27.2 | 11.7 |
| TimeChat+Seq2Time w/o RPT | 1.2 | 3.7 | 15.7 | 29.3 | 12.8 |
| **TimeChat+Seq2Time** | **1.3** (+30%) | **4.2** (+44.8%) | **16.2** (+27.6%) | **31.2** (+14.7%) | **13.7** (+17.1%) |

### 消融实验

| 数据配置 | YouCook2 CIDEr | F1 | Charades R@0.5 |
|---------|---------------|-----|---------------|
| Baseline (TimeChat) | 2.9 | 12.7 | 27.2 |
| +IS only | 4.3 | 13.3 | 28.8 |
| +IS+MC | 4.0 | 15.9 | 30.9 |
| +IS+MC+CS (Seq2Time) | **4.2** | **16.2** | **31.2** |

| 前置任务消融 | CIDEr | F1 |
|------------|-------|-----|
| IIC+IIG+ALR (全部) | **4.3** | 13.3 |
| 去掉 IIC | 3.4 | 12.9 |
| 去掉 IIG | 2.6 | 11.9 |
| 去掉 ALR | 2.7 | **14.4** |

### 关键发现

- **图像序列数据的效果令人惊喜**：仅添加图像序列数据就能将 CIDEr 从 2.9 提升到 4.3（+48.3%），证明静态图像序列可以有效迁移时序理解能力。
- **IIG（定位任务）贡献最大**：去掉 IIG 后两个基准都大幅下降，说明"从描述找位置"是时序理解的核心能力。
- **统一相对位置 token 是关键**：去掉 RPT 时性能提升被削弱，证明统一位置表示是实现序列到时间知识迁移的桥梁。
- Video-LLaMA 实验进一步验证了通用性：基于图像序列的训练将 F1 从 0.2 提升到 3.3，即使在没有任何时序训练的通用 Video LLM 上也能注入时序意识。
- 额外视频字幕（MC）改善了时序定位但略降了文本质量，说明图像级字幕的精度对生成质量有独特贡献。

## 亮点与洞察

- **"Video LLM 不感知时间，只感知位置"的洞察**极具启发性：将时序理解解构为位置-内容对应，打开了用廉价数据增强昂贵能力的大门。这种思路可推广到其他需要位置感知的任务（如文档理解中的布局位置）。
- **仅 10 个新 token 的位置编码**设计极其高效：4 位小数足以表达 0.13% 误差的位置精度，且层次化结构自然支持多粒度时序推理。
- 三种图像序列前置任务的互补设计（定位/描述/推理）为序列学习提供了完整的能力覆盖，比简单数据增量更有效。

## 局限与展望

- 基线模型 TimeChat 的复现性能低于原文报告（因部分训练数据不可用），影响了绝对数值的参考性。
- 片段序列字幕依赖 LongVA 生成，质量受限于该模型的能力。
- 仅在 TimeChat 和 Video-LLaMA 上验证，未在更强大的 Video LLM 上测试。
- 未来可探索在更大规模图像/片段数据上的 scaling 效果，以及将方法扩展到多模态（音频+视频）时序理解。

## 相关工作与启发

- **vs TimeChat**: TimeChat 通过双 Q-Former 架构提升时序感知，是架构驱动的方式。Seq2Time 是数据驱动的，不修改任何架构，仅通过数据增强就在 TimeChat 基础上取得显著提升，两种方式正交可互补。
- **vs VTG-LLM**: VTG-LLM 引入了绝对时间 token 减少量化误差，但绝对时间在不同帧率间不通用。Seq2Time 的相对位置 token 更灵活且仅需 10 个新 token。
- **vs Grounded-VideoLLM**: 也使用相对时间 token，但 Seq2Time 首次系统性地利用图像和短视频序列数据来增强时序能力。

## 评分

- 新颖性: ⭐⭐⭐⭐ "用图像序列模拟视频时序"的思路新颖，位置 token 设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 数据消融、任务消融、缩放实验、跨模型验证非常充分
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 提供了一条低成本提升 Video LLM 时序能力的有效途径

<!-- RELATED:START -->

## 相关论文

- [TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)
- [VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM](videorefer_suite_advancing_spatial-temporal_object_understanding_with_video_llm.md)
- [R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](../../ECCV2024/video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [Efficient Transfer Learning for Video-language Foundation Models](efficient_transfer_learning_for_video-language_foundation_models.md)
- [M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)

<!-- RELATED:END -->
