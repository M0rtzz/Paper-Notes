---
title: >-
  [论文解读] Vision-Language Models Can't See the Obvious
description: >-
  [ICCV 2025][多模态VLM][视觉显著性] 提出 SalBench 基准测试，发现当前大型视觉-语言模型（LVLM）在检测对人类而言显而易见的视觉显著特征（如颜色、方向、大小差异）上表现极差——最先进的 GPT-4o 在检测任务上仅达到 47.6% 准确率，揭示了 LVLM 与人类视觉注意力之间的根本差距。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉显著性"
  - "LVLM评估"
  - "注意力机制"
  - "低层特征感知"
  - "benchmark"
---

# Vision-Language Models Can't See the Obvious

**会议**: ICCV 2025  
**arXiv**: [2507.04741](https://arxiv.org/abs/2507.04741)  
**代码**: [SalBench](https://salbench.github.io)  
**领域**: 多模态VLM  
**关键词**: 视觉显著性, LVLM评估, 注意力机制, 低层特征感知, benchmark

## 一句话总结

提出 SalBench 基准测试，发现当前大型视觉-语言模型（LVLM）在检测对人类而言显而易见的视觉显著特征（如颜色、方向、大小差异）上表现极差——最先进的 GPT-4o 在检测任务上仅达到 47.6% 准确率，揭示了 LVLM 与人类视觉注意力之间的根本差距。

## 研究背景与动机

当前 LVLM 在高层语义理解任务（如 VQA、MMMU 等）上表现出色，但一个关键问题被忽视了：

**Moravec 悖论**：AI 系统在高层推理上表现出色，反而在人类轻松完成的低层感知任务上可能表现不佳。例如，在一组小圆中识别一个明显的大圆，或在一排相同颜色的物体中找出不同颜色的那个

**现有 benchmark 的盲区**：MMBench、MMMU、MathVista 等都测试高层复杂任务，但没有 benchmark 系统性地评估 LVLM 的低层视觉感知能力（颜色、方向、大小等基本视觉特征）

**与人类视觉注意力机制的对齐**：人类的视觉搜索依赖于特征整合理论（Feature Integration Theory, FIT），大脑能平行处理在某个特征维度上显著不同的区域。LVLM 是否也具有这种能力？

作者想通过一个简单却精心设计的 benchmark 来量化 LVLM 与人类在底层视觉感知上的差距，从而指明改进方向。

## 方法详解

### 整体框架

SalBench 基于 P3（合成图像）和 O3（自然图像）数据集构建，包含刺激图像中一个显著目标在众多干扰物中脱颖而出的场景。在这些图像上定义了三个任务来评估 LVLM 的感知能力。

### 关键设计

1. **Odd-One-Out Detection（异物检测）**

   给定一张包含多个相似物体和一个显著不同物体的图像，模型需从预定义的特征类别列表中预测目标在哪个（些）维度上与众不同。合成图像限定为 3 个类别（颜色、形状、方向），自然图像扩展为 7 个类别（方向、颜色、焦点、形状、大小、位置、图案）。

   设计动机：直接评估模型"看到"显著目标的能力。实验发现直接提问往往得到错误回答，说明模型确实缺乏这种基本感知能力。

2. **Referring Odd-One-Out（引导式异物识别）**

   在检测任务基础上，额外提供目标的边界框坐标作为文本提示（如 "(x_min, y_min, x_max, y_max)"）。模型需判断该区域内的物体在哪些特征上与其余物体不同。

   设计动机：消除"定位"的难度，纯粹测试模型在已知目标位置后的特征差异识别能力。即使给出位置信息，模型仍然表现不佳。

3. **Visual Referring Odd-One-Out（视觉引导式异物识别）**

   改用视觉方式（红色边界框）标出目标物体，而非文本坐标。模型需通过视觉注意力聚焦到高亮区域并判断差异。

   设计动机：测试模型整合视觉高亮信息的能力，更贴近人类在识别"被标记物体"时的自然交互方式。

### 损失函数 / 训练策略

SalBench 本身是一个评估基准而非训练方法。但作者额外进行了**训练实验**来验证是否可以通过训练改善性能：

- 生成了 100 万个合成显著性图像-文本对用于对齐阶段
- 生成了 100 万条指令微调数据
- 结合 Cambrian 数据集的 200 万条自然图像数据
- 使用 LLaVA 训练流程，测试了 4 种模型变体（LLama3.1-8B/Qwen2-7B × CLIP/SigLip）

**关键发现**：即使在域内显著性数据上训练，性能依然很低（检测 16-19%），说明当前架构和训练方式本身不适合捕捉显著性信息。

## 实验关键数据

### 主实验

**SalBench 零样本 F1 分数（检测/引导/视觉引导，自然/合成图像）**：

| 模型 | Detection NAT | Detection SYN | Referring NAT | Visual Ref. SYN |
|------|--------------|--------------|--------------|----------------|
| GPT-4o | 47.6 | 89.2 | 47.3 | 73.5 |
| Claude-sonnet | 48.2 | 86.7 | 51.1 | 87.7 |
| Qwen2-VL-72B | 41.6 | 88.8 | 44.6 | 74.7 |
| Qwen2-VL-7B | 32.5 | 55.7 | 32.5 | 57.4 |
| LLaVA 1.6-7B | 24.5 | 16.3 | 21.4 | 16.6 |
| InternVL-2-8B | 20.0 | 58.7 | 23.0 | 23.0 |

所有模型在自然图像上的性能远低于合成图像（差距 30-40%）。

### 消融实验

**合成图像按难度级别的准确率（Qwen2-VL-72B / GPT-4o）**：

| 类别 | 难度 | Qwen2-VL-72B Detection | GPT-4o Detection |
|------|------|----------------------|----------------|
| 方向 | Easy | 98.6 | 96.2 |
| 方向 | Hard | 95.7 | 98.6 |
| 大小 | Easy | 94.2 | 93.3 |
| 大小 | Hard | 46.0 | 36.8 |
| 颜色 | Easy | 100.0 | 99.8 |
| 颜色 | Hard | 60.1 | 66.1 |

**关键发现**：方向识别在各难度下都相对鲁棒，但大小和颜色在高难度（差异微小）时急剧下降。

**视觉骨干检索测试**：

| 视觉编码器 | SYN Top-1 | NAT Top-1 |
|-----------|-----------|-----------|
| SigLip-so400m | 55.3 | 87.9 |
| CLIP-ViT-Large-Patch14 | 41.2 | 78.6 |
| Random | 24.6 | 53.2 |

视觉编码器的特征表示本身就不够判别显著性信息。

### 关键发现

1. **模型规模有影响**：Qwen2-VL 从 1.5B→7B→72B，F1 分数明显提升（23.8→54.9→89.9 在合成检测上），但即使最大模型在自然图像上也仅 ~44%
2. **Few-shot 无一致提升**：增加 shot 数不一定改善性能，有时反而下降（GPT-4o 从 0-shot 47.6% 降至 3-shot 38.9%）
3. **颜色偏好**：所有模型对颜色类别的识别远好于其他特征，因为颜色是 RGB 图像直接提供的信息，而大小、形状需要更高层次编码
4. **干扰物数量影响**：随着干扰物增多（<7 到 >25），平均 F1 从 44.5% 下降到 37.4%
5. **训练也无法解决问题**：在域内数据上训练后，性能仍然很低，暗示问题可能出在视觉编码器的架构层面

## 亮点与洞察

- **Moravec 悖论的优秀实证**：第一个系统性地展示了 LVLM 在"简单"感知任务上的失败
- **任务设计巧妙**：三个递进式任务（无提示→文本位置→视觉标注）逐步降低定位难度，仍然暴露了模型的感知缺陷
- **根因分析深入**：不仅测试模型整体，还分别测试了 LLM 的 FIT 知识（GPT-4o 97.5% 了解 FIT 理论）和视觉编码器的检索能力，定位出问题根源在视觉表示端
- **实际意义**：对于需要底层视觉判断的应用场景（如工业质检、医疗影像异常检测），这一发现具有警示价值

## 局限与展望

- SalBench 主要聚焦在"odd-one-out"这一类显著性任务，未覆盖其他类型的低层感知（如纹理、深度等）
- 合成图像使用 7×7 网格排列，可能偏离真实世界场景的复杂度
- 多标签分类的评估指标（exact match 和 F1）可能低估了部分正确的识别
- 训练实验仅用 LLaVA 框架，未探索其他可能更适合低层特征学习的架构
- 未提出具体的改进方案或新的视觉编码器设计

## 相关工作与启发

本文与 MMVP（CLIP-blind pairs）、RealWorldQA、CV-Bench 等以视觉为中心的 benchmark 相关，但独特之处在于聚焦神经科学中定义良好的底层显著性概念。启发方向：（1）可能需要在视觉编码器中引入多尺度特征或显著性先验；（2）特征整合理论可以指导设计更好的视觉 backbone。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个系统性评估 LVLM 低层感知的 benchmark，发现意义重大
- **实验充分度**: ⭐⭐⭐⭐⭐ 评估了 15+ 个模型，三个任务，多种设置（zero/few-shot），骨干分析，训练实验
- **写作质量**: ⭐⭐⭐⭐ 分析清晰透彻，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ 揭示了 LVLM 的根本盲区，对未来模型设计有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] MMSafeAware: Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs](../../ACL2025/multimodal_vlm/cant_see_the_forest_for_the.md)
- [\[ECCV 2024\] BLINK: Multimodal Large Language Models Can See but Not Perceive](../../ECCV2024/multimodal_vlm/blink_multimodal_large_language_models_can_see_but_not_perceive.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](../../ACL2025/multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)

</div>

<!-- RELATED:END -->
