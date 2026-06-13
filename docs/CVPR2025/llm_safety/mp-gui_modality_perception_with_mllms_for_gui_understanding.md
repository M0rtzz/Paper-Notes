---
title: >-
  [论文解读] MP-GUI: Modality Perception with MLLMs for GUI Understanding
description: >-
  [CVPR 2025][LLM安全][GUI理解] MP-GUI设计了三个专用感知器分别提取GUI中的图形、文本和空间模态信息，通过空间结构精炼策略和自适应融合门控将三种模态组合，在有限训练数据下在多种GUI理解任务上取得了优于通用MLLM的表现。 领域现状：图形用户界面(GUI)理解是构建智能agent和自动化系统的基础能…
tags:
  - "CVPR 2025"
  - "LLM安全"
  - "GUI理解"
  - "多模态大语言模型"
  - "模态感知"
  - "空间结构建模"
  - "融合门控"
---

# MP-GUI: Modality Perception with MLLMs for GUI Understanding

**会议**: CVPR 2025  
**arXiv**: [2503.14021](https://arxiv.org/abs/2503.14021)  
**代码**: [https://github.com/BigTaige/MP-GUI](https://github.com/BigTaige/MP-GUI)  
**领域**: 人机交互 / 多模态VLM  
**关键词**: GUI理解, 多模态大语言模型, 模态感知, 空间结构建模, 融合门控

## 一句话总结
MP-GUI设计了三个专用感知器分别提取GUI中的图形、文本和空间模态信息，通过空间结构精炼策略和自适应融合门控将三种模态组合，在有限训练数据下在多种GUI理解任务上取得了优于通用MLLM的表现。

## 研究背景与动机

**领域现状**：图形用户界面(GUI)理解是构建智能agent和自动化系统的基础能力。当前多模态大语言模型(MLLM)如InternVL2、Qwen-VL等在自然图像和文档理解上表现出色，并被逐步应用于GUI任务（如元素定位、屏幕问答、GUI导航等）。

**现有痛点**：GUI与自然图像有本质区别——GUI由人工设计的图形元素（按钮、图标、文本框）按特定空间布局排列，conveying语义信息。通用MLLM虽然擅长处理图形和文本组件，但在GUI理解上面临两个关键障碍：(1) 缺乏对GUI特有空间结构（元素间的相对位置、层级包含关系、对齐约束等）的显式建模，导致元素定位和交互理解不精确；(2) 高质量GUI空间结构数据难以获取——真实应用界面涉及隐私问题，而自动标注的空间关系存在大量噪声。

**核心矛盾**：GUI理解需要同时理解"看到什么"（图形/文本内容）和"在哪里"（空间布局），但现有MLLM将这两种信息混合在统一的视觉特征中处理，缺乏对空间结构的专门建模。

**本文目标**：设计一个GUI专用的MLLM架构，显式分离和建模GUI中的图形、文本、空间三种模态，并通过自适应融合满足不同GUI任务的差异化需求。

**切入角度**：作者观察到不同GUI理解任务对三种模态的依赖程度不同——元素定位主要依赖空间信息，文本识别主要依赖文本模态，而功能理解需要图形和空间的联合推理。因此需要一个灵活的多模态融合机制。

**核心idea**：用三个独立的专用感知器分别提取GUI的图形、文本、空间模态特征，通过可学习的融合门控根据任务需要自适应地组合这三种模态信息。

## 方法详解

### 整体框架
MP-GUI基于InternVL2-8B构建，保留原有的视觉编码器和LLM主干，额外引入三个专用感知器模块。输入为GUI截图，首先通过视觉编码器提取通用视觉特征；然后三个感知器分别从中提取图形特征（按钮/图标等元素的视觉属性）、文本特征（界面中的文字内容）和空间特征（元素间的布局关系）。三种模态特征通过融合门控加权合并后，与原始视觉特征一起送入LLM进行推理。

### 关键设计

1. **图形感知器 (Graphical Perceiver)**:

    - 功能：从GUI截图中提取图形元素（按钮、图标、颜色区域等）的视觉语义特征
    - 核心思路：采用轻量级的cross-attention架构，使用一组可学习的query token从视觉编码器输出中提取与图形元素相关的特征。查询token数量固定，相当于将GUI截图的视觉信息压缩到固定维度的图形语义空间中。训练数据来自GUI元素检测和分类任务
    - 设计动机：通用视觉编码器产生的特征包含过多低层细节（纹理、渐变等），通过图形感知器过滤出与GUI元素语义相关的信息，降低后续推理的信息冗余

2. **文本感知器 (Textual Perceiver)**:

    - 功能：专门识别和理解GUI中的文字内容（标签、提示文本、输入内容等）
    - 核心思路：结构与图形感知器类似，使用另一组query token专门提取文本区域的特征。其特殊之处在于训练数据的构造——利用OCR工具自动提取GUI中的文本及其位置，然后用这些标注数据训练感知器学习文本区域的精准定位和内容识别
    - 设计动机：GUI中的文本是传达功能语义的核心元素（如"Submit"按钮、"Settings"标签），但文本在GUI中的字体小、背景复杂，通用MLLM的OCR能力在GUI场景下不够精确。专用文本感知器可以提升文本识别的精度和召回率

3. **空间感知器与空间结构精炼 (Spatial Perceiver with Refinement)**:

    - 功能：建模GUI元素间的空间关系（相对位置、包含关系、对齐方式等）
    - 核心思路：空间感知器同样基于cross-attention提取空间结构特征，但增加了空间结构精炼(Spatial Structure Refinement)策略。由于真实GUI的空间标注噪声大（自动标注的元素边界框不准确、遮挡导致的误识别等），精炼策略通过两阶段训练实现：第一阶段用Semantic UI等公开且标注质量较高的数据集预训练空间感知器，第二阶段用合成数据（通过Qwen2-VL-72B生成）进行微调，以提升对噪声标注的鲁棒性
    - 设计动机：空间关系是GUI区别于自然图像的最核心特征。元素间的"上方-下方"、"包含-被包含"、"对齐"等关系决定了GUI的功能语义，但这类结构信息在标准视觉特征中是隐式的且不可靠

### 融合门控 (Fusion Gate)
三个感知器的输出通过可学习的融合门控进行自适应加权合并。融合门控根据当前输入query（用户指令）的语义信息，动态调整三种模态的权重。例如，"点击搜索按钮"类指令会增大空间模态和图形模态的权重，而"读取输入框中的文字"类指令会增大文本模态权重。

### 训练策略
采用多步训练策略：(1) 先分别训练三个感知器（使用各自的专用数据）；(2) 再训练融合门控（固定感知器参数）；(3) 最后在下游benchmark数据上整体微调。这种分步策略降低了对大规模统一标注数据的需求，利用各模态的现有公开数据即可完成感知器训练。同时使用LoRA进行高效微调。

## 实验关键数据

### 主实验

| 基准任务 | 指标 | MP-GUI | InternVL2-8B | Qwen-VL-Chat | CogAgent | 提升(vs InternVL2) |
|---------|------|--------|-------------|-------------|---------|-------------------|
| ScreenSpot (移动端定位) | Acc | 78.2 | 66.5 | 53.1 | 71.4 | +11.7% |
| ScreenSpot (Web定位) | Acc | 72.6 | 61.8 | 48.3 | 65.2 | +10.8% |
| ScreenQA | Acc | 76.8 | 68.3 | 59.2 | 70.1 | +8.5% |
| AITW (GUI Agent) | Acc | 73.5 | 64.7 | 56.8 | 67.9 | +8.8% |
| Widget Caption | CIDEr | 142.3 | 121.6 | 98.7 | 128.4 | +20.7 |

### 消融实验

| 配置 | ScreenSpot Acc | ScreenQA Acc | 说明 |
|------|---------------|-------------|------|
| Full MP-GUI | 78.2 | 76.8 | 完整模型 |
| w/o 空间感知器 | 70.4 | 73.1 | 定位能力下降最大 |
| w/o 文本感知器 | 74.6 | 69.2 | QA任务影响更大 |
| w/o 图形感知器 | 75.1 | 74.3 | 整体均有下降 |
| w/o 融合门控 (直接拼接) | 74.8 | 73.6 | 自适应融合有2-3%增益 |
| w/o 空间结构精炼 | 73.6 | 74.5 | 精炼策略对定位贡献显著 |

### 关键发现
- 空间感知器对元素定位任务贡献最大（去掉后ScreenSpot下降7.8%），验证了GUI理解中空间结构建模的关键性
- 文本感知器对QA任务影响最大（去掉后ScreenQA下降7.6%），说明GUI问答高度依赖界面中的文本信息理解
- 融合门控比简单特征拼接好2-3%，证明了自适应模态权重调节的必要性
- 空间结构精炼策略带来4.6%的ScreenSpot提升，表明处理噪声空间标注是实用的
- MP-GUI在使用有限训练数据（主要来自公开数据集和合成数据）的情况下，仍然超越了使用更多数据训练的通用MLLM，说明架构设计的针对性比数据规模更重要

## 亮点与洞察
- **模态显式分离的设计思路**很有参考价值：不是把所有信息混在一起让模型自己学，而是根据任务领域的特点显式拆分模态、独立建模再融合。这种思路可以迁移到其他具有明确模态区分的场景（如CAD图纸理解、仪表盘解读等）
- **空间结构精炼策略**巧妙解决了数据质量问题：GUI空间标注噪声大的难题通过两阶段训练（高质量小数据预训练+合成数据微调）缓解，这是一种通用的处理噪声标注的策略
- **多步训练降低数据需求**是实用性很强的设计：各感知器独立训练避免了对统一多模态标注数据的依赖，降低了实际应用门槛

## 局限与展望
- 仅在InternVL2-8B上验证，未评估在其他MLLM（如Qwen-VL-2、LLaVA等）上的通用性
- 三个感知器的架构完全相同（都是cross-attention），针对各模态特点的差异化设计可能带来进一步提升
- 未涉及动态GUI理解（如滚动、动画效果），只处理静态截图
- 合成数据质量依赖Qwen2-VL-72B的能力上限，可能在某些复杂GUI上引入系统性偏差
- 评测局限于英文GUI界面，多语言/多文化GUI的表现未知

## 相关工作与启发
- **vs CogAgent**: CogAgent也针对GUI理解设计，但采用统一的视觉编码方案，不区分模态。MP-GUI通过显式模态分离在定位任务上超越CogAgent 6.8%
- **vs SeeClick / UGround**: 这些工作专注于GUI元素定位，MP-GUI提供了更全面的GUI理解能力，同时定位性能也更优
- **vs InternVL2 / Qwen-VL**: 通用MLLM作为底座，MP-GUI通过增加GUI专用组件获得了显著提升，验证了领域特化的价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 三模态分离+融合门控的GUI专用架构设计有创新性
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个GUI基准，消融详细，但缺少跨模型验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 为GUI Agent领域提供了有效的感知增强方案，开源代码和数据pipeline增加价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](../../ACL2026/llm_safety/risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](../../ACL2025/llm_safety/manu_modality_aware_unlearning.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](../../NeurIPS2025/llm_safety/enhancing_clip_robustness_via_crossmodality_alignment.md)
- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](../../ACL2025/llm_safety/when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [\[CVPR 2025\] IterIS: Iterative Inference-Solving Alignment for LoRA Merging](iteris_iterative_inference-solving_alignment_for_lora_merging.md)

</div>

<!-- RELATED:END -->
