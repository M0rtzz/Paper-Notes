---
title: >-
  [论文解读] APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension
description: >-
  [ECCV 2024][目标检测][弱监督指代表达理解] 本文提出锚框提示学习方法 APL，通过设计锚框提示编码器（APE）生成位置、颜色、类别三类判别性提示，动态融入锚框特征以丰富视觉语义，再配合文本重构损失和视觉对齐损失实现精确的视觉-语言对齐，在四个 REC 基准上超越现有弱监督方法（如 RefCOCO 上比 RefCLIP 高 6.44%）。
tags:
  - ECCV 2024
  - 目标检测
  - 弱监督指代表达理解
  - 提示学习
  - 锚框特征
  - 视觉语言对齐
  - 单阶段检测
---

# APL: Anchor-based Prompt Learning for One-stage Weakly Supervised Referring Expression Comprehension

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: https://github.com/Yaxin9Luo/APL  
**领域**: 目标检测 / 指代表达理解  
**关键词**: 弱监督指代表达理解, 提示学习, 锚框特征, 视觉语言对齐, 单阶段检测

## 一句话总结

本文提出锚框提示学习方法 APL，通过设计锚框提示编码器（APE）生成位置、颜色、类别三类判别性提示，动态融入锚框特征以丰富视觉语义，再配合文本重构损失和视觉对齐损失实现精确的视觉-语言对齐，在四个 REC 基准上超越现有弱监督方法（如 RefCOCO 上比 RefCLIP 高 6.44%）。

## 研究背景与动机

**领域现状**：指代表达理解（Referring Expression Comprehension, REC）旨在根据给定的自然语言描述定位目标物体。传统方法依赖昂贵的实例级标注（即每个物体的边界框标注与文本描述的显式对应关系），训练成本很高。近年来，弱监督 REC 成为研究热点，其中 RefCLIP 是代表性的单阶段弱监督 REC 方法，它利用预训练单阶段检测网络的锚框特征表示候选物体，并通过锚框-文本排序来定位目标。

**现有痛点**：尽管 RefCLIP 表现出一定的有效性，但其视觉语义表达存在模糊和不充分的问题。具体来说，锚框特征只携带粗粒度的空间信息，缺乏位置、颜色、类别等细粒度判别信息，这使得在弱监督设定下难以准确区分相似的候选物体。此外，视觉特征与文本描述之间的语义对齐不够精准。

**核心矛盾**：弱监督 REC 的核心挑战在于——没有显式的物体-文本标注对，模型需要在仅有图像-文本对级别监督下学习细粒度的定位能力。现有方法的锚框特征过于粗粒度，无法有效编码文本描述中提及的多维度视觉属性（如"左边的红色杯子"中的位置、颜色、类别信息）。

**本文目标** (1) 如何在弱监督设定下丰富锚框的视觉语义表达能力；(2) 如何实现更精准的视觉-语言对齐。

**切入角度**：作者观察到指代表达通常包含位置（左边、上方）、颜色（红色、蓝色）、类别（杯子、人）三个维度的描述信息。如果能用提示（prompt）的方式显式编码这三类信息并注入锚框特征，就能大幅增强视觉语义的判别力。

**核心 idea**：用锚框提示编码器生成位置/颜色/类别三维提示信息，动态融入锚框特征以增强弱监督 REC 的视觉语义表达。

## 方法详解

### 整体框架

APL 采用单阶段检测框架，输入为图像和指代表达文本。图像经过 YOLOv3 骨干网络提取多尺度特征后生成大量锚框（anchor），每个锚框对应一个视觉特征。文本通过语言编码器（如 BERT 或 GloVe）得到文本特征。APL 的核心在于：(1) 用锚框提示编码器（APE）为每个锚框生成三类提示（位置、颜色、类别），并动态融合到锚框特征中；(2) 用增强后的锚框特征与文本特征进行相似度排序，选择最高分锚框作为定位结果；(3) 通过文本重构损失和视觉对齐损失两个辅助目标实现精准的视觉-语言对齐。

### 关键设计

1. **锚框提示编码器（Anchor-based Prompt Encoder, APE）**:

    - 功能：为每个锚框生成覆盖位置、颜色、类别三个方面的判别性提示
    - 核心思路：APE 包含三个子编码器。位置提示编码器根据锚框的空间坐标（中心点、宽高）生成位置嵌入；颜色提示编码器从锚框区域的图像像素中提取颜色直方图特征并编码为颜色嵌入；类别提示编码器利用预训练检测器的类别预测分布生成类别语义嵌入。三类提示通过可学习的融合机制（如注意力加权）动态注入到原始锚框特征中，形成语义增强的锚框表示
    - 设计动机：指代表达中的描述信息天然可分解为位置、颜色、类别三个维度。通过显式建模这三类信息并以提示方式注入特征，能有效弥补弱监督下锚框特征语义不足的问题

2. **文本重构损失（Text Reconstruction Loss）**:

    - 功能：作为辅助训练目标，从增强的锚框特征重构对应的文本描述
    - 核心思路：给定经过提示增强的锚框特征，通过一个文本解码器重构原始指代表达。该损失鼓励锚框特征充分编码文本中提到的各种属性信息。重构损失采用交叉熵(cross-entropy)形式计算预测token与真实token之间的差异
    - 设计动机：如果增强后的锚框特征能够准确重构出原始文本描述，说明这些特征确实捕获了文本中提及的位置、颜色、类别等关键信息。这是一种自监督式的约束，无需额外的实例级标注

3. **视觉对齐损失（Visual Alignment Loss）**:

    - 功能：确保增强后的锚框特征与原始视觉特征保持一致性
    - 核心思路：通过对比学习方式约束增强特征与原始特征之间的关系——同一锚框的增强特征和原始特征应相近（正样本对），不同锚框之间的特征应远离（负样本对）。这防止提示注入后特征发生语义漂移
    - 设计动机：直接向锚框特征注入提示可能会破坏原始的视觉表征质量。视觉对齐损失作为正则项，保证提示增强是在保持原始视觉语义的基础上添加额外判别信息，而非覆盖原有表征

### 损失函数 / 训练策略

整体训练损失由三部分组成：(1) 主损失为锚框-文本对比排序损失（contrastive ranking loss），衡量正确锚框与文本的匹配度应高于其他锚框；(2) 文本重构损失 $L_{rec}$，要求从锚框特征重构文本；(3) 视觉对齐损失 $L_{align}$，保持特征一致性。总损失为：$L = L_{rank} + \lambda_1 L_{rec} + \lambda_2 L_{align}$。训练过程中只需图像-文本对级别的监督（弱监督），无需实例级边界框标注。模型使用预训练的 YOLOv3 作为检测骨干，其权重在 COCO 训练集（排除 RefCOCO 验证/测试集中的图像）上预训练。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 APL | 之前SOTA RefCLIP | 提升 |
|--------|------|----------|-----------------|------|
| RefCOCO val | Acc@0.5 | 64.51 | ~58.07 | +6.44% |
| RefCOCO testA | Acc@0.5 | 61.91 | - | - |
| RefCOCO testB | Acc@0.5 | 63.57 | - | - |
| RefCOCO+ val | Acc@0.5 | 42.70 | - | - |
| RefCOCO+ testA | Acc@0.5 | 42.84 | - | - |
| RefCOCO+ testB | Acc@0.5 | 39.80 | - | - |
| RefCOCOg val | Acc@0.5 | 50.22 | - | - |

### 弱监督指代表达分割（Weakly RES）

| 数据集 | 指标 | 本文 APL |
|--------|------|----------|
| RefCOCO val | mIoU | 55.92 |
| RefCOCO testA | mIoU | 54.84 |
| RefCOCO testB | mIoU | 55.64 |
| RefCOCO+ val | mIoU | 34.92 |
| RefCOCOg val | mIoU | 40.13 |

### 伪标签训练其他模型

| 配置 | RefCOCO val | RefCOCO+ val | RefCOCOg val | 说明 |
|------|------------|-------------|-------------|------|
| APL_SimREC | 63.94 | 42.11 | 48.35 | 用 APL 生成伪标签训练 SimREC |
| APL_TransVG | 64.86 | 39.28 | 46.11 | 用 APL 生成伪标签训练 TransVG |

### 关键发现

- APL 在 RefCOCO 上比 RefCLIP 提升 6.44%，证明提示学习策略在弱监督 REC 中非常有效
- APL 生成的伪标签可以用于训练其他全监督 REC 模型（如 SimREC、TransVG），且性能接近直接用 APL 的结果
- 方法在弱监督指代表达分割任务上也表现出很强的泛化能力

## 亮点与洞察

- 将提示学习引入弱监督 REC 是一个很有创意的方向，通过位置/颜色/类别三维提示有效弥补了弱监督下视觉语义不足的问题
- 文本重构损失的设计很巧妙——通过要求锚框特征"说出"对应的文本描述，隐式地强制特征编码丰富的语义信息
- 伪标签训练其他模型的实验展示了方法的实用价值，可以作为弱监督到全监督的桥梁

## 局限与展望

- 依赖预训练的 YOLOv3 检测器，锚框质量直接影响最终性能；如使用更强的检测器（如 DETR 系列）可能进一步提升
- 颜色提示编码器基于颜色直方图，对复杂纹理和光照变化的鲁棒性有待验证
- 位置/颜色/类别三类提示是手工设定的，未来可探索自动发现更多维度的提示类型
- 在 RefCOCO+ 上的性能明显低于 RefCOCO，说明对纯属性描述（无位置词）的处理仍有提升空间

## 相关工作与启发

- **RefCLIP** 是本文的主要基线，开创了单阶段弱监督 REC 的范式
- **CLIP** 的视觉-语言对比学习框架为本文的锚框-文本排序提供了理论基础
- 提示学习（Prompt Learning）在 NLP 领域（如 GPT-3、P-tuning）和多模态学习中已有广泛应用，本文将其创新性地应用于弱监督目标定位
- 该方法的提示注入思路可能启发其他弱监督视觉任务（如弱监督检测、弱监督分割）

## 评分
- 新颖性: ⭐⭐⭐⭐ 提示学习 + 弱监督 REC 的结合很新颖，三维提示设计有洞察
- 实验充分度: ⭐⭐⭐⭐ 四个数据集全面评测，消融和伪标签实验丰富
- 写作质量: ⭐⭐⭐ 方法描述清晰，但部分实验细节可更详细
- 价值: ⭐⭐⭐⭐ 对弱监督视觉-语言任务有较好的参考价值

<!-- RELATED:START -->

## 相关论文

- [WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)
- [All You Need is One: Capsule Prompt Tuning with a Single Vector](../../NeurIPS2025/object_detection/all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)
- [SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection](../../ICLR2026/object_detection/spwood_sparse_partial_weakly-supervised_oriented_object_detection.md)
- [TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)

<!-- RELATED:END -->
