---
title: >-
  [论文解读] Discovering Divergent Representations between Text-to-Image Models
description: >-
  [ICCV 2025][图像生成][文生图模型对比] 提出 CompCon（Comparing Concepts），一种进化搜索算法，自动发现两个文生图模型之间的"分歧表征"——即在哪些视觉属性上、被哪类提示词触发时，两个模型会产生截然不同的输出，并构建了 ID² 基准数据集进行系统评估。
tags:
  - ICCV 2025
  - 图像生成
  - 文生图模型对比
  - 表征差异发现
  - 进化搜索
  - 视觉属性
  - 模型偏见
---

# Discovering Divergent Representations between Text-to-Image Models

**会议**: ICCV 2025  
**arXiv**: [2509.08940](https://arxiv.org/abs/2509.08940)  
**代码**: [https://github.com/adobe-research/CompCon](https://github.com/adobe-research/CompCon)  
**领域**: 扩散模型 / 模型分析  
**关键词**: 文生图模型对比, 表征差异发现, 进化搜索, 视觉属性, 模型偏见

## 一句话总结

提出 CompCon（Comparing Concepts），一种进化搜索算法，自动发现两个文生图模型之间的"分歧表征"——即在哪些视觉属性上、被哪类提示词触发时，两个模型会产生截然不同的输出，并构建了 ID² 基准数据集进行系统评估。

## 研究背景与动机

**领域现状**：文生图（Text-to-Image, T2I）模型如 Stable Diffusion、DALL-E、PixArt、Playground 等已成为主流创作工具。每个模型由不同的训练数据、架构和优化策略塑造，因此它们对相同文本提示可能生成视觉上截然不同的图像。但目前缺乏系统化工具来理解"这些模型到底在哪些方面不同"。

**现有痛点**：对模型差异的分析主要依赖人工主观评估或 FID 等全局统计量，前者不可扩展、后者缺乏可解释性。我们知道不同模型"生成的图看起来不一样"，但很难精确说出"在什么条件下、表现在哪个视觉属性上不同"。

**核心矛盾**：模型间的差异是条件依赖的（input-dependent）——同一对模型在某些类型的提示词下可能表现一致，在另一些类型下却截然不同。简单比较总体输出分布无法捕捉这种细粒度的、依赖输入的差异。

**本文目标**：自动发现两个 T2I 模型之间的"分歧表征"（divergent representations），即找到（视觉属性, 提示词类型）的配对，使得给定某类提示词时，一个模型倾向于展示某个视觉属性，而另一个模型不会。

**切入角度**：作者借鉴了进化搜索的思想——从一组初始假设出发，通过迭代的提出→验证→筛选→进化循环，逐步发现越来越准确和有趣的分歧表征。

**核心 idea**：用 LLM/VLM 驱动的进化搜索算法自动产生"哪个视觉属性在哪类提示词下分化"的假设，然后用 VLM 分类器在生成图像上验证假设，保留通过验证的、迭代进化出更精确的描述。

## 方法详解

### 整体框架

CompCon 的流程分为三阶段：（1）给定一组提示词，分别用两个模型生成图像；（2）用 VLM 分析两组图像的视觉差异，提出"分歧假设"（如"模型 A 的输出中更常出现火焰"）；（3）通过进化搜索迭代优化假设，同时发现触发该差异的提示词特征（如"与强烈情感相关的提示词"）。最终输出是一组经过验证的（视觉属性, 触发条件）配对。

### 关键设计

1. **进化属性搜索（Evolutionary Attribute Search）**:

    - 功能：自动发现一个模型输出中存在而另一个模型中缺失的视觉属性
    - 核心思路：维护一个属性假设种群。每轮迭代中，（a）用 VLM（如 GPT-4V）观察两个模型的生成图像并提出新的视觉差异假设；（b）用 VLM 分类器对每个假设在所有图像上进行二分类（该属性是否存在）；（c）计算"分歧得分"——该属性在两个模型输出中出现频率的差值；（d）保留高分假设，淘汰低分假设，让 LLM 基于高分假设变异/交叉产生新假设
    - 设计动机：直接枚举所有可能的视觉属性是不现实的，进化搜索利用 LLM 的创造力和 VLM 的视觉判断力来高效探索巨大的假设空间

2. **提示词概念链接（Prompt Concept Linking）**:

    - 功能：找出触发某个视觉差异的提示词特征/概念
    - 核心思路：对于已发现的高分歧视觉属性，将展示该属性的图像对应的提示词和不展示的提示词分为两组，用 LLM 分析两组提示词的语义差异，提取出"触发概念"。例如，"火焰"属性主要由"涉及情感表达的提示词"触发。这一步使用迭代精化：初始描述→在更多数据上验证→根据误判案例调整描述
    - 设计动机：仅知道"模型 A 更常生成火焰"是不够的，还需要知道在什么条件下才会出现这种差异，这样用户才能理解和利用模型的特性

3. **ID² 基准数据集（Input-Dependent Differences Dataset）**:

    - 功能：提供标准化的评估框架来衡量分歧发现算法的性能
    - 核心思路：使用自动化数据生成流程创建 60 个已知的输入依赖差异。对于每个差异，提供分歧视觉属性和对应的分歧提示词描述，然后用 LLM 生成包含/不包含该属性的提示词对，再用 T2I 模型生成图像对。评估指标衡量算法能否准确召回这些已知差异
    - 设计动机：现有工作缺乏客观评价分歧发现能力的基准，ID² 填补了这一空白

### 损失函数 / 训练策略

CompCon 不涉及神经网络训练。其核心度量是分歧得分 $\Delta(a) = |P(\text{attr}=a | \text{model}_1) - P(\text{attr}=a | \text{model}_2)|$，衡量属性 $a$ 在两个模型间的出现频率差异。进化搜索使用阈值过滤（$\Delta > \tau$，$\tau = 0.2$）保留有意义的差异，并用去重机制（基于语义相似度）合并近似等价的假设。

## 实验关键数据

### 主实验

| 方法 | Recall@10 (ID²) | Precision@10 | 平均分歧得分 |
|------|-----------------|-------------|-------------|
| CompCon | **0.58** | **0.72** | **0.41** |
| LLM-only baseline | 0.32 | 0.45 | 0.28 |
| VLM-only baseline | 0.38 | 0.51 | 0.33 |
| TF-IDF baseline | 0.21 | 0.30 | 0.19 |
| Random sampling | 0.08 | 0.12 | 0.11 |

### 模型对比发现

| 模型对 | 发现的分歧表征 | 触发条件 |
|--------|---------------|---------|
| PixArt vs SD 3.5 | PixArt 生成"湿漉漉的街道" | 与孤独感相关的提示词 |
| SD 3.5 vs Playground | SD 3.5 更多展示非裔美国人 | 与媒体职业相关的提示词 |
| DALL-E 3 vs SD 3.5 | DALL-E 3 更多卡通风格 | 涉及动物的提示词 |
| PixArt vs Playground | Playground 更多高对比度光影 | 涉及建筑的提示词 |

### 关键发现

- CompCon 比 LLM-only baseline 在 Recall 上高出 81%，证明视觉信号（VLM 判断）对发现分歧至关重要
- 进化搜索相比单轮搜索提升约 30%，迭代优化能发现更多非直觉的分歧
- 发现了一些令人惊讶的偏见模式，如特定模型将某些种族与特定职业关联
- ID² 数据集包含 60 个已知差异，涵盖风格、内容、构图等多个维度

## 亮点与洞察

- **进化搜索 + LLM/VLM 的巧妙结合**：用 LLM 的语言推理能力生成假设，用 VLM 的视觉判断能力验证假设，两者通过进化框架协同工作——这种"AI 驱动的科学发现"范式可以推广到其他比较分析任务
- **Input-dependent 差异的精准定位**：不是简单比较两个模型的总体输出分布，而是精确到"什么输入下"→"什么视觉属性"不同，这种细粒度分析对模型审计和安全评估很有价值
- **发现社会偏见**：该方法不仅能发现风格差异，还能揭示模型中隐藏的种族、性别等社会偏见，对 AI 公平性研究有重要意义

## 局限与展望

- CompCon 的效果强依赖 VLM 分类器的准确性，对于微妙的视觉差异可能遗漏
- 进化搜索需要大量 API 调用（LLM + VLM），计算成本较高
- 当前只比较两个模型，扩展到多模型同时比较的效率有待提升
- ID² 数据集基于自动化流程生成，可能未覆盖所有类型的分歧
- 未来方向：将方法扩展到视频生成模型比较；结合人类反馈优化分歧发现；用发现的分歧指导模型改进

## 相关工作与启发

- **vs VisDiff**: VisDiff 比较的是数据集间的视觉差异，CompCon 比较的是模型间的差异，且考虑了输入依赖性
- **vs 传统评估（FID/CLIPScore）**: FID 和 CLIPScore 给出全局统计量，无法解释具体差异在哪里。CompCon 提供的是可解释的、具体的分歧描述
- **vs 模型审计工具**: 现有审计工具（如 DALL-E probe）主要检测已知偏见类别，CompCon 能自动发现未知的分歧模式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统化地发现 T2I 模型间的 input-dependent 分歧表征，问题定义和方法都很新颖
- 实验充分度: ⭐⭐⭐⭐ 构建了 ID² 基准数据集，对多对模型进行了系统分析
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，案例分析生动有趣
- 价值: ⭐⭐⭐⭐ 对模型理解、模型选择、偏见审计等多个场景有实际价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Models and Small Edge Models](adaptive_routing_of_text-to-image_generation_requests_between_large_cloud_model_.md)
- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model](adaptive_routing_of_text_to_image_generation_requests_between_large_cloud_model_and_light_weight_edge_model.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](pla_prompt_learning_attack_against_text-to-image_generative_models.md)

<!-- RELATED:END -->
