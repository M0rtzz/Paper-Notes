---
title: >-
  [论文解读] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning
description: >-
  [ACL 2025][语义分割][部件分割] 提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。
tags:
  - "ACL 2025"
  - "语义分割"
  - "部件分割"
  - "任务导向"
  - "指令推理"
  - "VLM"
  - "affordance"
  - "benchmark"
---

# InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning

**会议**: ACL 2025  
**arXiv**: [2505.18291](https://arxiv.org/abs/2505.18291)  
**代码**: [https://zifuwan.github.io/InstructPart/](https://zifuwan.github.io/InstructPart/)  
**领域**: 分割 / 视觉语言模型  
**关键词**: 部件分割, 任务导向, 指令推理, VLM, affordance, benchmark

## 一句话总结
提出 InstructPart，首个将任务导向指令与部件级分割结合的真实世界 benchmark——2400 张图像、48 类物体、44 类部件、9600 条人工标注的任务指令，评估发现当前 VLM 在指令驱动的部件分割上严重不足，基于 LISA+DINOv2 的 baseline 微调后性能提升约 100%。

## 研究背景与动机

**领域现状**：大型视觉语言模型（VLM）在物体级理解上表现出色——目标检测、语义分割、referring segmentation 等任务。但大多数模型将物体视为不可分割的整体，忽略了组成部件。

**现有痛点**：(1) 现有部件分割数据集（PartImageNet/Pascal-Part/PACO）只有物体-部件标注，没有任务指令——无法评估"给定任务推理哪个部件"的能力；(2) 机器人/affordance 数据集（UMD/AGD20K）的affordance 类别有限或只有点标注/来自仿真；(3) 指令中不直接提及部件名（如说"冲马桶"而非"按马桶手柄"），需要推理。

**核心矛盾**：VLM 能理解语言指令也能做分割，但将两者结合到部件粒度的任务推理上，现有模型几乎无法胜任。

**本文目标**：(1) 构建联结任务指令和部件分割的 benchmark；(2) 评估现有 VLM 的部件推理能力；(3) 提供 baseline 方法。

**切入角度**：日常家务场景——给定"倒些水"的指令和水壶图片，模型需推理出应该抓水壶的"把手"，并输出把手的分割 mask。

**核心 idea**：将部件分割从"识别"问题升级为"推理"问题——模型需要从隐含指令中推断目标部件。

## 方法详解

### 整体框架
InstructPart 是 benchmark + 简单 baseline。核心贡献在数据集构建和全面评估。

### 关键设计

1. **两个评估任务**:

    - **TRPS（Task Reasoning Part Segmentation）**：输入自然语言任务指令+图像，输出目标部件 mask。指令中**不直接提及部件名**（如"Flush the toilet"而非"Press the toilet handle"），需要推理
    - **ORPS（Oracle Referring Part Segmentation）**：输入部件名+图像，直接定位。可附加 affordance 信息（如"handle of the cup that can be held"）
    - 设计动机：TRPS 评估推理+视觉 grounding 双重能力；ORPS 仅评估 grounding，控制变量

2. **数据集构建**:

    - 2400 张真实图像，48 类物体（家居日常用品），均匀分布
    - 44 类部件 + 30 种 affordance + 37 种 action
    - 每张图配 4 条任务指令（人工 + GPT-4 改写）+ 手标分割 mask
    - Affordance 分两层：低层（pull/push/twist 等操作动作）和高层（turn on/pick up 等功能动作）
    - 6 名专家标注，GPT-4 润色+人工验证质量

3. **PISA Baseline**:

    - 功能：基于 LISA（LLaVA + SAM decoder）的改进版
    - 核心思路：用冻结 DINOv2 替代 SAM encoder 做特征提取，线性层融合多层 DINOv2 特征，保留 SAM 的 mask decoder（TransConv + 上采样交替解码）
    - 设计动机：DINOv2 在提取部件级对应关系上优于 SAM，微调后在 TRPS 上 gIoU 几乎翻倍

### 评估指标
- gIoU：所有图像 IoU 的平均
- cIoU：全局累积交集/全局累积并集
- P@50：IoU>0.5 的比例
- P@50:95：多阈值平均精度

## 实验关键数据

### 主实验（TRPS 任务，人工标注指令）

| 方法类型 | 方法 | gIoU↑ | cIoU↑ | P@50↑ |
|---------|------|-------|-------|-------|
| 开放词汇分割 | VLPart | 0.39 | 1.16 | 0.00 |
| 开放词汇分割 | OVSeg | 22.44 | 14.11 | 15.33 |
| Referring 分割 | G-SAM | 29.95 | 21.45 | 25.17 |
| 推理分割 | LISA | 32.11 | 30.25 | 30.00 |
| 推理分割 | MiniGPT-v2 | 26.29 | 19.46 | 24.00 |
| **微调 baseline** | **PISA** | **~60** | **~55** | **~55** |

（PISA 微调后约 2× 提升）

### ORPS 任务 vs TRPS 任务对比

| 模型 | ORPS gIoU | TRPS gIoU | 差距 |
|------|-----------|-----------|------|
| LISA | 34.46 | 32.11 | -2.35 |
| G-SAM | 34.33 | 29.95 | -4.38 |
| 平均 | 22.56 | 17.49 | -5.07 |

### 关键发现
- **当前 VLM 在部件级推理上严重不足**：最好的模型 LISA 在 TRPS 上只有 32.11 gIoU，远低于物体级 referring 分割的典型表现
- **推理是主要瓶颈**：ORPS（给部件名）vs TRPS（给任务指令）有约 5 个点差距，说明从指令推理到部件名是额外难点
- **VLPart 在 TRPS 上近乎失败**：开放词汇部件分割模型无法处理复杂指令，gIoU 仅 0.39
- **微调数据质量高**：用 InstructPart 训练集微调 PISA 后，TRPS 性能翻倍，说明数据集有效
- **GPT-4 改写指令不总是更好**：某些模型在人工指令上表现更好，提示指令多样性不等于更好的评估

## 亮点与洞察
- **"任务→部件"的推理范式**：填补了从物体级 grounding 到部件级任务推理的评估空白。这个范式对机器人操作、辅助技术有直接价值
- **指令中隐含部件名**：故意不在指令中提及部件名（"倒水"而非"握住把手"），迫使模型做真正的推理，更接近真实交互场景
- **Affordance 双层标注**：低层操作动作+高层功能动作，为 affordance 研究提供了更细粒度的标注

## 局限与展望
- 2400 张图规模偏小，48 类物体可能不够覆盖所有日常场景
- 每张图只有一个目标物体的标注，多物体场景未考虑
- PISA baseline 较简单（LISA+DINOv2），更强的架构（如 Grounding DINO v2 + SAM2）可能有更大提升空间
- 未评估 GPT-4V/Gemini 等闭源模型在 TRPS 上的零样本表现
- 数据集现仅含静态图像，视频场景的动态部件分割未涉及

## 相关工作与启发
- **vs LISA**: LISA 做物体级推理分割，InstructPart 将其推到部件级。LISA 在 InstructPart 上 gIoU 仅 32，说明部件推理远比物体推理困难
- **vs VLPart**: VLPart 做开放词汇部件分割但缺少指令推理，在 TRPS 上几乎失败
- **vs AGD20K**: AGD20K 有 affordance 但仅点标注+无指令。InstructPart 提供完整 mask + 任务指令

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个连接任务指令与部件分割的 benchmark，问题定义有实际意义
- 实验充分度: ⭐⭐⭐⭐ 评估了 3 类共 10 个模型，baseline 有效，但数据集偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务定义明确
- 价值: ⭐⭐⭐⭐⭐ 对机器人操作和具身 AI 有直接推动价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] General and Task-Oriented Video Segmentation](../../ECCV2024/segmentation/general_and_task-oriented_video_segmentation.md)
- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[ECCV 2024\] PartSTAD: 2D-to-3D Part Segmentation Task Adaptation](../../ECCV2024/segmentation/partstad_2d-to-3d_part_segmentation_task_adaptation.md)
- [\[ACL 2025\] DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)
- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)

</div>

<!-- RELATED:END -->
