---
title: >-
  [论文解读] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos
description: >-
  [ICCV 2025][图像分割][bimanual affordance] 本文提出从人类活动视频中自动提取精确的双手可操作区域(affordance)数据集 2HANDS，并训练基于 VLM 的 2HandedAfforder 模型，实现根据文本提示预测双手抓握的精确物体区域分割，在新提出的 ActAffordance 基准上显著优于现有方法。
tags:
  - ICCV 2025
  - 图像分割
  - bimanual affordance
  - VLM
  - hand-object interaction
  - egocentric video
---

# 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos

**会议**: ICCV 2025  
**arXiv**: [2503.09320](https://arxiv.org/abs/2503.09320)  
**代码**: https://sites.google.com/view/2handedafforder  
**领域**: 分割 / 机器人操作 / Affordance  
**关键词**: bimanual affordance, affordance segmentation, VLM, hand-object interaction, egocentric video

## 一句话总结

本文提出从人类活动视频中自动提取精确的双手可操作区域(affordance)数据集 2HANDS，并训练基于 VLM 的 2HandedAfforder 模型，实现根据文本提示预测双手抓握的精确物体区域分割，在新提出的 ActAffordance 基准上显著优于现有方法。

## 研究背景与动机

**领域现状**：Affordance grounding（可操作区域识别）是机器人操作的关键能力——机器人需要知道物体的哪些区域可以用于特定任务（如倒水时应抓住瓶身什么位置）。现有方法通常依赖人工标注的数据集，标注质量与物体部件分割相似，缺乏动作导向的精确性。

**现有痛点**：(a) 现有 affordance 数据集（IIT-AFF、AGD20K 等）标注不够精确，往往退化为粗糙的物体部件分割；(b) 多数方法不考虑任务上下文（task-agnostic），只预测通用的"热点"区域；(c) 完全忽视双手协作交互（bimanual affordance）这一重要场景。

**核心矛盾**：手与物体交互时，手本身会遮挡关键的 affordance 区域，导致直接从交互图像中提取精确接触区域非常困难。

**本文目标** (a) 如何从视频中自动提取精确的、任务导向的双手 affordance 分割 mask；(b) 如何训练一个能根据文本提示预测左右手分别交互区域的模型。

**切入角度**：利用视频级别的手部修复（hand inpainting）技术，先"去掉"遮挡手部获得完整物体视图，再通过 mask 补全得到手与物体接触的精确区域。

**核心 idea**：通过视频手部修复+mask补全自动提取精确 affordance mask，结合 VLM 实现文本驱动的双手 affordance 预测。

## 方法详解

### 整体框架

系统分两个阶段：(1) **数据提取**——从 EPIC-KITCHENS 自我中心视频中，利用手部修复和 mask 补全自动生成 278K 张带标注的 affordance 数据（2HANDS 数据集）；(2) **模型训练**——基于 VLM 的 2HandedAfforder 网络，输入图像和任务文本提示，输出左右手的 affordance 分割 mask 以及双手/单手分类。

### 关键设计

1. **Affordance 提取流水线**:

    - 功能：从人类活动视频中自动提取物体上的精确 affordance 区域
    - 核心思路：(a) 使用 VISOR 标注获取稀疏手-物体 mask，通过视频 mask 传播网络获得稠密全序列 mask；(b) 使用视频手部修复模型 VIDM 将手部区域修复为完整物体（利用 4 帧作为输入，未遮挡帧可提供线索）；(c) 使用 SAM2 将原始物体 mask 传播到修复图像上获得完整物体 mask；(d) 最终 affordance 区域 = 完整物体 mask ∩ 手部 mask
    - 设计动机：手遮挡了关键交互区域，通过修复+补全巧妙绕过遮挡问题，获得比人工标注更精确的 affordance 区域
    - 额外优势：使用视频中任务的叙述文本（narration）作为 affordance 类别标签，自然获得 73 类 affordance 和 163 类物体

2. **VLM-based 2HandedAfforder 网络**:

    - 功能：根据文本提示预测图像中的双手 affordance mask
    - 核心思路：输入文本 prompt（如"pour tea from kettle"）和图像，VLM（LLaVa-13b）生成语言 token 和 [SEG] token；SAM 图像编码器提取视觉特征；两个 SAM-style mask decoder 分别生成左手和右手 affordance mask
    - 设计动机：VLM 擅长推理但不擅长像素级任务，SAM 编码器提供强视觉特征，二者互补；双解码器设计自然处理双手场景

3. **双手分类头（Taxonomy Classifier）**:

    - 功能：预测交互是单手左/单手右/双手操作
    - 核心思路：从左手 mask decoder 的输出 token 通过 MLP 预测三分类，测试时根据分类结果决定使用哪个 mask 输出
    - 设计动机：避免在单手任务中产生冗余的另一手 mask 预测

### 损失函数 / 训练策略

- Mask 预测使用 Dice Loss + Focal Cross-Entropy Loss 组合
- 分类预测使用标准 Cross-Entropy Loss
- VLM 使用 LoRA 微调（保留预训练知识），SAM 图像编码器冻结
- 单手任务时对应手的 mask loss 权重设为 0

## 实验关键数据

### 主实验

提出 ActAffordance 基准，包含 400 个活动、由人类标注的多模态 affordance mask。

| 方法 | IoU ↑ | Precision ↑ | Dir. HD ↓ | mAP ↑ |
|------|-------|-------------|-----------|-------|
| 2HandedAfforder | **0.058** | **0.130** | **202** | **0.104** |
| LISA | 0.044 | 0.050 | 255 | 0.047 |
| 2HAff-CLIP | 0.026 | 0.064 | 292 | 0.059 |
| AffordanceLLM | 0.012 | 0.013 | 225 | 0.012 |

Cropped 版本（消除物体定位影响）：

| 方法 | IoU ↑ | Precision ↑ | Dir. HD ↓ | mAP ↑ |
|------|-------|-------------|-----------|-------|
| 2HandedAfforder | **0.086** | **0.269** | **100** | **0.240** |
| 3DOI | 0.082 | 0.224 | 109 | 0.180 |
| LISA | 0.082 | 0.122 | 130 | 0.116 |
| AffordanceLLM | 0.076 | 0.112 | 76 | 0.103 |

### 消融实验

| 配置 | 说明 |
|------|------|
| AffExtract (数据提取) | Precision=0.420, IoU=0.185，验证提取质量 |
| 2HAff vs 2HAff-CLIP | VLM 版本比 CLIP 版本精度高 2x，说明推理能力关键 |
| Ego4D 泛化测试 | 未用 Ego4D 训练，但表现与 EPIC 相当甚至更好 |

### 关键发现

- **推理能力至关重要**：2HAff (VLM) 比 2HAff-CLIP 精度高约 2 倍，说明 VLM 的语义推理远优于 CLIP 特征匹配
- 数据提取质量与人类标注对齐度合理（Precision 0.42），但 IoU 低（0.185）反映 affordance 的多模态特性——同一任务有多种合理交互区域
- 机器人演示证实预测区域可直接用于抓取规划，比通用物体分割有效

## 亮点与洞察

- **手部修复获取 affordance**：利用视频中的手部修复+mask补全绕过手遮挡问题，思路新颖且通用——任何包含手-物体交互的视频都可以作为数据源
- **Narration 作为自然类别标签**：避免了预定义固定类别体系，让 affordance 类别随任务自然涌现，覆盖面更广
- **双解码器架构**：简洁地将双手问题分解为两个并行的 mask 预测，加上分类头选择，设计优雅
- **可迁移到其他机器人任务**：affordance 区域可直接转换为 6DOF 抓取点云，已在 Tiago++ 真机上验证

## 局限与展望

- IoU 指标整体偏低（所有方法<0.1），任务本身极具挑战性
- 数据源限于厨房场景（EPIC-KITCHENS），泛化到其他环境需更多数据
- 对需要精准力控的任务（如拧瓶盖），仅靠区域分割还不够
- 未考虑 affordance 的多模态性——同一任务可能有多个合理抓握位置，模型只预测一种

## 相关工作与启发

- **vs LISA**: LISA 做全物体推理分割，不考虑 affordance 精确区域，本文通过专门的 affordance 数据训练获得更精确的区域预测
- **vs VRB/ACP**: 这些方法预测 task-agnostic 热点/热力图，本文通过文本 prompt 实现 task-aware 精确 mask
- **vs AffordanceLLM**: 虽然 AffLLM 也用 LLM，但训练数据（AGD20K）标注质量不如自动提取的 2HANDS 精确

## 评分

- 新颖性: ⭐⭐⭐⭐ 手部修复提取 affordance 思路新颖，双手 affordance 首创
- 实验充分度: ⭐⭐⭐⭐ 新基准+多基线对比+消融+真机验证，全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，流水线可视化好
- 价值: ⭐⭐⭐⭐ 对机器人操作领域有直接应用价值
