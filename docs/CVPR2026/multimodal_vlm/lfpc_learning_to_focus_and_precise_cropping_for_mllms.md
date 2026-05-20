---
title: >-
  [论文解读] LFPC: Learning to Focus and Precise Cropping for MLLMs
description: >-
  [CVPR 2026][多模态VLM][多模态大语言模型] LFPC 提出两阶段纯强化学习框架，通过"信息差"机制（降低全局图像分辨率迫使模型依赖高分辨率裁剪区域）和接地损失（提升裁剪精度），解决了现有 agent-based MLLM 中"先答后裁"的虚假工具调用问题，在高分辨率 VQA 上达到 SOTA。
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "多模态大语言模型"
  - "强化学习"
  - "裁剪工具"
  - "信息差"
  - "高分辨率VQA"
---

# LFPC: Learning to Focus and Precise Cropping for MLLMs

**会议**: CVPR 2026  
**arXiv**: [2603.27494](https://arxiv.org/abs/2603.27494)  
**代码**: [https://github.com/XuanPu-Z/LFPC](https://github.com/XuanPu-Z/LFPC)  
**领域**: 多模态VLM  
**关键词**: 多模态大语言模型, 强化学习, 裁剪工具, 信息差, 高分辨率VQA

## 一句话总结

LFPC 提出两阶段纯强化学习框架，通过"信息差"机制（降低全局图像分辨率迫使模型依赖高分辨率裁剪区域）和接地损失（提升裁剪精度），解决了现有 agent-based MLLM 中"先答后裁"的虚假工具调用问题，在高分辨率 VQA 上达到 SOTA。

## 研究背景与动机

MLLM 在复杂视觉场景中的细粒度感知仍是挑战。Agent-based 方法赋予模型"裁剪工具"来主动放大感兴趣区域，但现有训练策略存在关键问题。

**核心发现**：作者对 DeepEyes 等 RL-based 模型进行分析，发现一个令人担忧的行为模式——模型在执行裁剪前就已形成答案，裁剪只是用来"确认"预有结论。构建专用评测验证了这一假设：模型对裁剪区域内容的依赖性很弱。

**核心矛盾**：SFT+RL 方法受限于教师模型能力上限且生成轨迹成本高；纯 RL 方法虽不需要教师，但模型学到的是"走过场"的裁剪行为而非真正利用裁剪信息。

## 方法详解

### 整体框架

两阶段纯 RL 训练：Stage 1 通过信息差机制训练模型依赖裁剪区域，Stage 2 通过接地损失提升裁剪精度。不需要轨迹监督。

### 关键设计

1. **信息差机制 (Information Gap)**:

    - 功能：迫使模型真正依赖裁剪区域的信息来回答问题
    - 核心思路：与之前直接输入高分辨率图像不同，LFPC 将输入图像故意降采样到较低分辨率。降采样程度由模型自身的不确定性决定——选择使模型产生与高分辨率不一致答案的适当低分辨率。但当模型决定使用裁剪工具时，裁剪区域从原始高分辨率图像提取。这在低细节全局视图和高细节局部视图之间创造了关键的"信息差"，使裁剪区域的信息成为正确回答的必要条件
    - 设计动机：如果全局图像已包含足够信息，模型自然不会真正利用裁剪。只有让全局信息"不够用"，才能激励模型主动从裁剪中获取关键细节

2. **接地损失 (Grounding Loss)**:

    - 功能：提升裁剪坐标的精度，确保裁剪到正确的区域
    - 核心思路：在第二阶段，使用少量边界框标注引入接地奖励信号。该奖励鼓励模型不仅使用裁剪工具，还要将裁剪框放在与答案相关的精确位置。这是一种弱监督——只需少量标注即可显著提升裁剪精度
    - 设计动机：Stage 1 解决了"是否依赖裁剪"的问题，但裁剪位置可能仍不够精准。少量的定位监督可以高效地提升这一能力

3. **不确定性驱动的分辨率选择**:

    - 功能：自适应确定每张图像的降采样程度
    - 核心思路：对同一问题在不同分辨率下采样答案，找到使答案开始变得不一致的分辨率阈值。这个阈值即为该样本的"信息差"边界——足够低以创造信息需求，又不至于低到完全无法理解图像
    - 设计动机：不同图像需要的细节粒度不同，统一降采样比例可能对简单问题太激进、对复杂问题不够

### 损失函数 / 训练策略

纯 RL 训练，基于 GRPO 算法。Stage 1 使用准确率奖励 + 格式奖励，Stage 2 额外加入接地奖励。不需要任何教师模型生成的轨迹数据。

## 实验关键数据

### 主实验

| 方法 | HR-Bench 4K | HR-Bench 8K | V* | 视觉Token |
|------|------------|------------|-----|-----------|
| DeepEyes | 74.0 | 68.0 | 85.9 | 16384 |
| LFPC (16K tokens) | **SOTA** | **SOTA** | **SOTA** | 16384 |
| LFPC (1K tokens) | 优于多数16K方法 | 优于多数16K方法 | 竞争力 | **1024** |

LFPC 在 16K 和 1K 两种视觉 token 预算下均达到 SOTA。

### 消融实验

| 配置 | 裁剪依赖度 | 性能 | 说明 |
|------|-----------|------|------|
| DeepEyes 基线 | 弱（先答后裁） | 基线 | 裁剪是虚假行为 |
| Stage 1 (信息差) | 强 | 显著提升 | 模型真正利用裁剪信息 |
| Stage 1 + Stage 2 (接地) | 强+精准 | SOTA | 裁剪位置更准确 |

### 关键发现

- "信息差"机制从根本上改变了模型对裁剪区域的依赖模式——从"确认性裁剪"变为"探索性裁剪"
- 1K token 预算下 LFPC 仍超过部分 16K token 方法，说明精准裁剪比大量 token 更重要
- 少量边界框标注（Stage 2）即可显著提升裁剪精度，标注成本很低

## 亮点与洞察

- **深刻的问题诊断**：发现 RL-based agent 的"先答后裁"问题，并构建专用评测验证。这种"先质疑再解决"的研究思路值得学习
- **信息差的巧妙设计**：通过控制输入信息量来引导模型行为，比修改奖励函数更直接有效。可迁移到任何 agent 工具使用场景
- **效率优势**：1K token 超过 16K token，证明"精准看什么"比"看多少"更关键

## 局限与展望

- 信息差机制的分辨率选择需要预采样，增加了预处理成本
- 当前仅支持单次裁剪，多步迭代裁剪可能进一步提升
- 接地损失的标注需求虽少但不为零
- 未来可探索多工具（裁剪+旋转+增强）的 agent 场景

## 相关工作与启发

- **vs DeepEyes**: 纯 RL 方法，但存在虚假裁剪问题，LFPC 通过信息差机制解决
- **vs SFT+RL 方法**: 需要教师模型生成轨迹，成本高且有能力上限，LFPC 完全不依赖教师
- **vs 注意力引导方法**: 通过注意力图分析重要区域，但缺乏显式的裁剪动作和信息利用保证

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 问题诊断深刻，信息差机制设计精巧
- 实验充分度: ⭐⭐⭐⭐ 多基准对比充分，但消融可以更详细
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，实验发现有说服力
- 价值: ⭐⭐⭐⭐⭐ 对 agent-based MLLM 的工具使用训练有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2026\] Venus: Benchmarking and Empowering Multimodal Large Language Models for Aesthetic Guidance and Cropping](venus_benchmarking_and_empowering_multimodal_large_language_models_for_aesthetic.md)
- [\[CVPR 2026\] MMR-AD: A Large-Scale Multimodal Dataset for Benchmarking General Anomaly Detection with MLLMs](mmrad_multimodal_anomaly_detection.md)
- [\[CVPR 2026\] FluoCLIP: Stain-Aware Focus Quality Assessment in Fluorescence Microscopy](fluoclip_stain-aware_focus_quality_assessment_in_fluorescence_microscopy.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)

</div>

<!-- RELATED:END -->
