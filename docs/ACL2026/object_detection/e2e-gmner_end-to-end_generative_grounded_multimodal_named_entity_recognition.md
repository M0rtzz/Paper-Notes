---
title: >-
  [论文解读] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition
description: >-
  [ACL 2026][目标检测][多模态命名实体识别] 提出E2E-GMNER，首个将实体识别、语义分类、视觉定位和隐式知识推理统一在单一多模态大语言模型中的端到端GMNER框架，通过CoT推理自适应判断视觉/知识线索的可用性，并引入高斯风险感知框扰动（GRBP）提升生成式框预测的鲁棒性。
tags:
  - ACL 2026
  - 目标检测
  - 多模态命名实体识别
  - 端到端生成
  - 视觉定位
  - 高斯扰动
  - CoT推理
---

# E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition

**会议**: ACL 2026  
**arXiv**: [2604.17319](https://arxiv.org/abs/2604.17319)  
**代码**: [https://github.com/Finch-coder/E2E-GMNER](https://github.com/Finch-coder/E2E-GMNER)  
**领域**: 目标检测  
**关键词**: 多模态命名实体识别, 端到端生成, 视觉定位, 高斯扰动, CoT推理

## 一句话总结

提出E2E-GMNER，首个将实体识别、语义分类、视觉定位和隐式知识推理统一在单一多模态大语言模型中的端到端GMNER框架，通过CoT推理自适应判断视觉/知识线索的可用性，并引入高斯风险感知框扰动（GRBP）提升生成式框预测的鲁棒性。

## 研究背景与动机

**领域现状**：Grounded Multimodal Named Entity Recognition（GMNER）需要联合识别文本中的实体、预测语义类型，并将每个实体定位到图像中对应的视觉区域。现有方法如H-Index、TIGER、RiVEG等主要采用流水线架构。

**现有痛点**：（1）流水线架构将文本实体识别和视觉定位解耦为独立模块（如独立NER标注器、外部目标检测器），导致错误累积和无法联合优化；（2）现有方法通过隐式跨模态对齐解决文本-视觉歧义，但缺乏显式机制判断视觉证据或外部知识何时真正有用，导致噪声视觉线索反而降低性能；（3）生成式框预测中，单一硬目标监督对标注噪声和坐标离散化误差敏感。

**核心矛盾**：端到端统一 vs 各子任务的特异性需求——如何在单一模型中同时优化实体识别、语义分类和视觉定位三个本质不同的任务？

**本文目标**：设计首个端到端GMNER框架，消除流水线中的错误累积。

**切入角度**：将GMNER建模为指令微调的条件生成任务，利用多模态大语言模型的统一生成能力。

**核心 idea**：端到端生成+CoT自适应推理+高斯软监督，三者协同解决GMNER的三个核心问题。

## 方法详解

### 整体框架

给定图文对和任务指令，LoRA适配的多模态LLM先进行CoT推理（视觉线索分析+背景知识分析），然后自回归生成结构化实体记录（实体名|语义类型|边界框坐标），训练时用GRBP替代硬框监督。

### 关键设计

1. **端到端生成式GMNER**:

    - 功能：消除流水线架构的错误累积
    - 核心思路：将GMNER建模为条件生成：输入=[指令;(图像,文本)]，输出=[推理序列R; 实体记录集{(e_i, c_i, b_i)}]。每个实体记录序列化为"实体名|类型|[x1,y1,x2,y2]"格式，所有记录连接为最终预测。使用标准自回归MLE损失训练
    - 设计动机：单一生成过程允许实体识别和视觉定位之间的信息流动，实现真正的联合优化

2. **CoT指令微调的自适应推理**:

    - 功能：让模型自主判断何时视觉证据或背景知识有用
    - 核心思路：在生成实体记录前先输出推理序列R，包含视觉线索分析（图像中是否有与文本实体对应的视觉证据）和背景知识分析（是否需要外部知识来消歧）。训练时推理序列由更强的外部LLM通过API生成作为监督；推理时模型完全自主生成，不依赖外部模型
    - 设计动机：避免盲目使用噪声视觉线索或不相关的知识——让模型"先想后做"

3. **高斯风险感知框扰动（GRBP）**:

    - 功能：提升生成式框预测在标注噪声和离散化误差下的鲁棒性
    - 核心思路：训练时对GT框进行概率性扰动：中心位置加高斯噪声（$\delta_x, \delta_y \sim \mathcal{N}(0, \beta^2)$），宽高乘以高斯缩放因子。IoU守卫确保扰动框与原始框的IoU $\geq \tau$。这将硬目标监督替换为高斯加权的软目标——更大的扰动对应更低的概率，保持经验风险最小化的同时容忍小的几何偏差
    - 设计动机：生成式框预测将坐标离散为token序列，微小偏差就会产生不相称的大训练损失，GRBP通过软监督缓解这一问题

### 损失函数 / 训练策略

标准自回归MLE损失 $\mathcal{L} = -\sum_t \log p_\theta(y_t | y_{<t}, \text{Instruction}, I, T)$，其中框坐标在GRBP扰动后作为软目标参与训练。

## 实验关键数据

### 主实验

在Twitter-GMNER和Twitter-FMNERG基准上：

| 方法 | Twitter-GMNER (GMNER) | Twitter-GMNER (MNER) |
|------|----------------------|---------------------|
| GMDA (流水线) | 58.61 | - |
| GEM (流水线+MLLM) | 59.83 | 83.15 |
| **E2E-GMNER** | **竞争力最强** | **竞争力最强** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| w/o CoT推理 | 下降 | 自适应视觉/知识利用重要 |
| w/o GRBP | 下降 | 框预测鲁棒性受损 |
| 硬框监督 vs GRBP软监督 | GRBP优 | 容忍标注噪声 |
| 端到端 vs 流水线 | 端到端优 | 消除错误累积 |

### 关键发现

- 端到端框架在GMNER主任务上达到高度竞争性能，验证了统一优化的有效性
- CoT推理使模型在视觉线索有噪声时主动忽略它们而非被误导，这对提升实体定位精度至关重要
- GRBP的IoU守卫机制确保扰动不会过大，平衡了软监督的灵活性和准确性
- 推理时完全不依赖外部模型，保持了高效的端到端推理

## 亮点与洞察

- 首个端到端GMNER框架的意义不仅在于性能提升，更在于证明了实体识别和视觉定位可以在统一生成框架中有效协同，而非必须分步处理。
- GRBP将数据增强的思想引入监督目标设计：不是增强输入数据，而是"增强"标签——通过概率性扰动GT框来产生软监督信号。这个思路可迁移到其他生成式定位任务。
- CoT推理作为一种"注意力门控"机制：让模型在使用视觉/知识信号前先评估其可靠性，是比简单的cross-attention更智能的多模态融合策略。

## 局限与展望

- 在某些特定类别上可能仍不如专门的流水线方法（特别是使用强大外部检测器的方法）
- CoT推理的训练依赖外部LLM（如GPT-4o）生成推理序列，引入了额外的数据准备成本
- GRBP的超参数（$\beta, \gamma, \tau$）需要调优，不同数据集可能需要不同设定
- 目前仅在Twitter图文对数据集上验证，其他领域（新闻、电商）的泛化性未知

## 相关工作与启发

- **vs RiVEG (Li et al., 2024)**: 用MLLM辅助但仍为流水线架构；E2E-GMNER实现真正端到端
- **vs MAKAR (Lin et al., 2025)**: 用MLLM多智能体系统解决语义歧义，但仍有流水线组件；E2E-GMNER更简洁
- **vs MQSPN (Tang et al., 2025)**: 用集合预测缓解曝光偏差，但未解决框预测的噪声敏感问题；E2E-GMNER的GRBP直接应对此挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个端到端GMNER+GRBP软监督创新
- 实验充分度: ⭐⭐⭐⭐ 两个基准+完整消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 为多模态NER的端到端范式提供了有效示范

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Method](../../ICLR2026/object_detection/traceable_evidence_enhanced_visual_grounded_reasoning_evaluation_and_methodology.md)
- [\[CVPR 2026\] Evaluating Few-Shot Pill Recognition Under Visual Domain Shift](../../CVPR2026/object_detection/evaluating_few-shot_pill_recognition_under_visual_domain_shift.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](../../ECCV2024/object_detection/weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICML 2025\] Understanding the Emergence of Multimodal Representation Alignment](../../ICML2025/object_detection/understanding_the_emergence_of_multimodal_representation_alignment.md)

</div>

<!-- RELATED:END -->
