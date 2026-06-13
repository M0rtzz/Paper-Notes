---
title: >-
  [论文解读] Multigranular Evaluation for Brain Visual Decoding
description: >-
  [AAAI 2026 Oral][语义分割][brain decoding] 提出BASIC多粒度评估框架，从结构（四级分割mask匹配）和语义（MLLM提取对象/属性/关系图的精确率-召回率-F1）两个轴统一评估脑视觉解码质量，横跨fMRI/EEG × Image/Video/3D六种模态组合，解决现有指标饱和、缺乏神经科学基础和细粒度诊断能力的问题。
tags:
  - "AAAI 2026 Oral"
  - "语义分割"
  - "brain decoding"
  - "evaluation metric"
  - "图像分割"
  - "semantic matching"
  - "MLLM"
---

# Multigranular Evaluation for Brain Visual Decoding

**会议**: AAAI 2026 Oral  
**arXiv**: [2507.07993](https://arxiv.org/abs/2507.07993)  
**代码**: [GitHub](https://github.com/weihaox/BASIC)  
**领域**: 图像分割  
**关键词**: brain decoding, evaluation metric, segmentation, semantic matching, MLLM

## 一句话总结

提出BASIC多粒度评估框架，从结构（四级分割mask匹配）和语义（MLLM提取对象/属性/关系图的精确率-召回率-F1）两个轴统一评估脑视觉解码质量，横跨fMRI/EEG × Image/Video/3D六种模态组合，解决现有指标饱和、缺乏神经科学基础和细粒度诊断能力的问题。

## 研究背景与动机

脑视觉解码已能从fMRI/EEG神经信号重建图像、视频甚至3D形状，但评估体系严重落后于方法进展。核心痛点有三：

第一，**指标饱和**——PixCorr、SSIM、CLIP等主流指标在SOTA模型间得分趋同，无法区分解码质量差异。例如多个方法的CLIP分数几乎相同，但解码结果在语义准确性上有显著差异。

第二，**缺乏神经科学基础**——人类视觉感知是层次化的：从注意力驱动的显著物体识别，到属性感知、空间关系理解，再到场景级语义一致性。现有指标没有反映这种多层结构，无法判断解码出的细节是源于真实脑信号还是生成模型的"幻觉"。

第三，**诊断能力缺失**——黑盒式单分数指标无法告诉研究者重建在哪里失败：是物体类别错误？属性不对？还是空间关系不合理？

本文的切入点是：设计一个同时覆盖低级结构和高级语义、具有多粒度诊断能力的统一评估框架BASIC，并使其适用于所有刺激-神经影像组合。

## 方法详解

### 整体框架

BASIC（Brain-Aligned Structural, Inferential, and Contextual similarity）分为两个互补子指标：
- **BASIC-L**：低级结构相似度——基于四级分割mask的多粒度结构匹配
- **BASIC-H**：高级语义相似度——结合推理（对象/属性/关系匹配）和上下文（场景叙事一致性）

### 关键设计

1. **五维评估体系**

    - 功能：定义脑解码评估应覆盖的感知维度
    - 核心思路：Scene（布局/几何/事件/风格）、Object（类别/通用性/特异性）、Attribute（外观颜色纹理/位置/数量/文字符号）、Relation（空间/部分-整体/交互/运动）、Camera（光照/视角/运动）
    - 设计动机：源自视觉神经科学和认知心理学研究，对齐人类视觉感知的层次结构，也与多模态大模型的场景理解结构吻合

2. **BASIC-L：多粒度分割匹配**

    - 功能：量化重建图像与参考图像在空间结构上的一致性
    - 核心思路：在四个分割粒度上进行mask对应匹配：Foreground（前景显著性）→ Semantic（语义类别）→ Instance（实例级）→ Part（部件级）。对重建图和参考图分别进行多粒度分割，通过粒度感知的mask对应计算IoU和AP
    - 设计动机：单一粒度的分割匹配可能遗漏重要信息——前景分割只关注有无物体，语义分割忽略实例区分，实例分割忽略部件结构。由粗到细的层级匹配全面覆盖空间结构保真度

3. **BASIC-H：结构化语义匹配**

    - 功能：量化重建图像与参考图像在高级语义上的对应关系
    - 核心思路：三步流水线——(1) 用MLLM（如GPT-4V）为重建图和参考图生成详细的结构化描述；(2) 解析描述为语义图，提取对象集合、属性集合和关系三元组；(3) 对Object、Attribute、Relation分别计算Precision/Recall/F1，综合得到BASIC-H分数
    - 设计动机：传统特征相似度（CLIP embedding余弦距离）将多维语义压缩为单一分数，无法区分"物体正确但属性错误"和"物体数量正确但类别混淆"等情况。结构化语义匹配提供可解释的诊断信息

### 损失函数 / 训练策略

BASIC是评估指标而非训练方法，不涉及损失函数设计。框架的核心组件包括预训练分割模型（用于BASIC-L）和MLLM（用于BASIC-H），均以冻结方式使用。

## 实验关键数据

### 主实验

**NSD数据集（fMRI→Image）上BASIC-H评分**：

| 方法 | Object F1 | Attribute F1 | Relation F1 | BASIC-H |
|------|-----------|-------------|-------------|---------|
| SDRecon | 53.79 | 14.96 | 39.06 | 35.31 |
| BrainDiffuser | 58.09 | 19.43 | 43.50 | 39.71 |
| MindEye | 61.26 | 25.06 | 48.84 | 44.30 |
| DREAM | 63.56 | 25.92 | 52.91 | 46.37 |
| MindEye2 | 61.72 | 24.71 | 49.07 | 44.39 |
| NeuroVLA | **64.57** | **28.65** | **52.95** | **47.88** |
| STTM | 62.88 | 26.64 | 50.36 | 45.88 |
| MindTuner | 61.95 | 24.73 | 49.80 | 44.63 |
| BrainGuard | 62.43 | 25.84 | 50.60 | 45.43 |

**跨模态BASIC-H对比**：

| 数据集（模态） | 最佳方法 | BASIC-H |
|---------------|---------|---------|
| NSD (fMRI→Image) | NeuroVLA | 47.88 |
| CC2017 (fMRI→Video) | NeuroClips | 45.12 |
| SEED-DV (EEG→Video) | EEG2Video | 49.54 |
| EEG-Things (EEG→Image) | ATM | 30.55 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| BASIC-H 各维度 | Attribute F1 整体偏低（14-28） | 属性重建是脑解码最薄弱环节 |
| Object vs Relation | Relation F1 < Object F1 | 物体间关系比物体本身更难重建 |
| BASIC-L NeuroPictor | 25.88（最高） | 结构匹配排名与BASIC-H排名不同 |

### 关键发现

- BASIC-H在SOTA方法间保持良好区分度（35.31到47.88），而传统CLIP分数已饱和
- **Attribute是脑解码最大短板**：所有方法的Attribute F1不超过28.65，远低于Object和Relation
- **结构排名≠语义排名**：BASIC-L上NeuroPictor表现最好，但BASIC-H上NeuroVLA最优，说明两个维度捕捉了不同方面
- EEG-Image解码整体得分远低于fMRI-Image（30.55 vs 47.88），量化了两种神经影像模态的信息差距
- BASIC统一覆盖了6种刺激-神经影像组合（fMRI/EEG × Image/Video/3D），是首个如此全面的框架

## 亮点与洞察

- **首个跨模态统一评估框架**：同一指标适用于fMRI/EEG × Image/Video/3D的全部组合，横向对比首次成为可能
- **"Attribute是脑解码盲区"的发现**有重要指导意义：未来方法应专注提升颜色/纹理/材质等属性重建
- **利用MLLM做自动化语义评估**的思路巧妙：避免了传统方法需要人工标注的瓶颈，且可随MLLM能力提升而自动改善
- **结构排名≠语义排名的发现**说明单一维度评估不够——一个方法可能空间结构保真但语义混乱
- 评估维度体系有认知神经科学理论支撑，不是简单拼凑

## 局限与展望

- **MLLM幻觉风险**：MLLM生成的描述本身可能包含幻觉，引入评估噪声；特别是对模糊/低质量重建图像的描述可能不可靠
- **分割模型依赖**：BASIC-L的可靠性受底层分割模型精度限制，特别是在非自然图像（如3D渲染、视频帧）上
- **缺乏人类感知相关性验证**：未做human correlation study验证BASIC分数是否与人类主观感知判断一致
- **计算成本高**：每对图像需运行MLLM+多级分割，大规模评估的计算开销不可忽视
- **语义图构建对复杂场景可能不完整**：关系三元组提取依赖文本解析，多物体交互场景中可能遗漏
- **Contextual similarity的定义较模糊**：论文中BASIC-H主要展示了Object/Attribute/Relation的结果，全局场景一致性的量化细节不够清晰

## 相关工作与启发

与传统的8指标协议（PixCorr/SSIM/AlexNet-2/5/Inception/CLIP/EffNet/SwAV）相比，BASIC提供了可解释的多粒度评估。与n-way分类准确率等特定指标相比，BASIC具有跨数据集和跨模态的统一性。

MLLM做自动评估的思路可推广到图像生成/编辑质量评估、文本到图像生成的语义一致性评估等领域。语义图匹配方法对scene graph generation和visual question answering的评估也有参考价值。多粒度分割匹配的设计对图像分割质量本身的评估也值得借鉴。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个面向脑解码的多粒度统一评估框架，MLLM+分割的组合评估思路新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖14+方法和6种模态组合，比较全面；但缺human correlation
- 写作质量: ⭐⭐⭐⭐ 结构清晰，维度体系论述有理有据
- 价值: ⭐⭐⭐⭐ 对脑解码领域的评估标准化有重要推动作用，Attribute短板发现有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](../../CVPR2025/segmentation/comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[AAAI 2026\] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention](lwganet_addressing_spatial_and_channel_redundancy_in_remote_sensing_visual_tasks.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)

</div>

<!-- RELATED:END -->
