---
title: >-
  [论文解读] Yo'Chameleon: Personalized Vision and Language Generation
description: >-
  [CVPR 2025][图像生成][个性化生成] 提出 Yo'Chameleon，首次探索大型多模态模型（LMM）的个性化问题，通过双soft prompt + self-prompting机制 + "soft-positive"训练策略，仅用3-5张图片和32个可学习token就能实现个性化的文本理解和图像生成。
tags:
  - CVPR 2025
  - 图像生成
  - 个性化生成
  - 大型多模态模型
  - 提示学习
  - 图文统一生成
  - 少样本学习
---

# Yo'Chameleon: Personalized Vision and Language Generation

**会议**: CVPR 2025  
**arXiv**: [2504.20998](https://arxiv.org/abs/2504.20998)  
**代码**: https://thaoshibe.github.io/YoChameleon  
**领域**: 多模态VLM / 图像生成  
**关键词**: 个性化生成, 大型多模态模型, soft prompt, 图文统一生成, 少样本学习

## 一句话总结

提出 Yo'Chameleon，首次探索大型多模态模型（LMM）的个性化问题，通过双soft prompt + self-prompting机制 + "soft-positive"训练策略，仅用3-5张图片和32个可学习token就能实现个性化的文本理解和图像生成。

## 研究背景与动机

**领域现状**：大型多模态模型（如GPT-4o、Chameleon）已经成为通用AI助手，能同时理解和生成文本与图像。个性化技术在LLM和文本到图像模型中已有广泛研究——LLM通过存储个性化描述作为prompt，图像生成模型通过DreamBooth等方法进行微调。

**现有痛点**：现有LMM是通用模型，缺乏对特定用户概念的个性化知识。例如你问模型"描述<bo>并生成一张<bo>在图书馆看书的照片"，如果<bo>是你的宠物狗，模型无法给出准确回答。之前的个性化工作（如Yo'LLaVA、MyVLM）只探索了VLM的文本生成个性化，如何扩展到图像生成模态尚未被研究。

**核心矛盾**：个性化的两个关键挑战。(1) 灾难性遗忘：图像生成任务需要精细的视觉信息，通常需要微调整个模型（如DreamBooth），但这会导致LMM丢失通用知识。Soft prompt虽能保持模型冻结，但仅3-5张图训练不出好的图像生成效果。(2) 模态不兼容：为图像理解优化的soft prompt用于图像生成时会产生无关内容，反之亦然。联合训练两种任务也导致双重次优。

**本文目标**：在不破坏LMM通用能力的前提下，仅用3-5张图片就能实现同一模型的个性化文本理解和图像生成。

**切入角度**：作者发现当有~300张真实图像时，soft prompt可以达到接近全模型微调的效果。但用户只能提供3-5张。解决办法是利用视觉上相似的"soft-positive"图像来扩充训练数据，并用双重soft prompt分别处理两种任务。

**核心 idea**：用双soft prompt（分别用于图像生成和文本理解）+ self-prompting机制（模型先判断任务类型再选择prompt）+ adaptive "soft-positive"训练策略（根据相似度分配不同prompt长度），解决LMM个性化中的灾难性遗忘和模态不兼容问题。

## 方法详解

### 整体框架

基于 Chameleon 模型（通过Anole恢复图像生成能力），输入为3-5张目标概念图像，输出为个性化的文本回答或图像生成。核心思路是学习一组可训练token来编码用户概念："<sks> is <g-tokens><u-tokens>"，其中 <g-tokens> 用于图像生成，<u-tokens> 用于文本理解，共32个token。训练时保持模型权重冻结，只更新这些token和对应的分类器头矩阵。

### 关键设计

1. **"Soft-Positive"图像训练策略**:

    - 功能：解决仅3-5张正样本导致soft prompt优化效果差的问题
    - 核心思路：从LAION-5B中检索约1000张与正样本视觉相似的负样本图像。按CLIP图像相似度从高到低排序后分为$k-1$组。关键创新：相似度越高的图像分配越多的可学习token（即更长的prompt来描述更多细节），而完整的全部token只给真正的正样本。这样模型既能从相似图像中学习相关视觉特征，又能通过token数量的差异来区分正样本和soft-positive。
    - 设计动机：之前的方法要么只用3-5张正样本（数据太少）、要么用数据增强（分割+修复质量受限）、要么把所有负样本一视同仁。"soft-positive"方法引入了"相似度感知"的训练信号，比传统数据增强(CLIP-I 从低于0.7提升到0.74)和均匀负样本策略都更有效。

2. **双Soft Prompt + Self-Prompting机制**:

    - 功能：解决图像生成和文本理解两种任务在同一组prompt上不兼容的问题
    - 核心思路：学习两组独立的可训练token——<g-tokens>（k=16个token，用于图像生成）和 <u-tokens>（h=16个token，用于文本理解）。在训练时构造数据使模型必须先预测当前任务应使用哪组token（self-prompting），然后再执行任务。例如对于文本理解任务"<sks>是什么？"，target output先包含<u-tokens>再跟实际答案。这迫使模型将不同的token集与不同任务对齐。
    - 设计动机：实验发现共享token联合训练、两组token简单拼接、拼接后微调三种策略都不如self-prompting。核心原因是为一种任务优化的token表示对另一种任务缺乏语义相关性。self-prompting的巧妙之处在于token同时承担"任务模式选择"和"概念信息编码"的双重角色。

3. **概念表示为可学习Prompt**:

    - 功能：高效地将个性化概念编码到少量可训练参数中
    - 核心思路：基于Chameleon的自回归训练目标，将个性化概念表示为"<sks> is <token1><token2>...<tokenk>"。训练时只计算响应部分的loss：$p(\mathbf{X}_a) = \prod_{j=1}^{L} p_{\theta}(x_j | \mathbf{X}_{a,<j})$。可训练参数仅包含概念标识符<sks>、k个latent token和语言模型最终分类器头中与这些token对应的矩阵W。
    - 设计动机：Soft prompt方法相比全模型微调计算高效，且通过冻结模型权重完全避免灾难性遗忘。32个token (~0.001%参数) 即可达到接近全模型微调的图像生成效果。

### 损失函数 / 训练策略

使用标准自回归语言建模loss，仅在响应部分计算。训练数据由两部分组成：(1) 理解数据——包含识别数据（正样本+100easy+100hard负样本）和QA数据（10个模板问题，GPT-4o生成答案）；(2) 生成数据——正样本+soft-positive图像。优化器AdamW，学习率1e-4，每个概念训练15 epoch，batch size 4，在A100 GPU上训练。最佳checkpoint通过识别准确率和CLIP-I的平均分选择。

## 实验关键数据

### 主实验

| 方法 | Token数 | 识别准确率↑ | QA(视觉)↑ | QA(文本)↑ | CLIP-I↑ | 人脸相似度↑ |
|------|---------|------------|----------|----------|---------|-----------|
| Chameleon (原始) | 0 | 0.500 | 0.474 | 0.405 | 0.425 | 0.009 |
| Chameleon+文本prompt | ~64 | 0.727 | 0.523 | 0.716 | 0.566 | 0.012 |
| Chameleon+图像prompt (1k) | ~1k | 0.361 | 0.580 | 0.573 | 0.487 | 0.013 |
| GPT-4o+文本prompt | ~64 | 0.841 | 0.923 | 0.798 | 0.636 | 0.028 |
| GPT-4o+图像prompt (1k) | ~1k | 0.902 | 0.867 | 0.982 | 0.657 | 0.036 |
| **Yo'Chameleon (Ours)** | **32** | **0.845** | 0.604 | 0.721 | **0.783** | **0.212** |

### 消融实验

| 训练策略 | 识别准确率↑ | CLIP-I↑ | 人脸相似度↑ |
|---------|------------|---------|-----------|
| 共享prompt + 仅语言数据 | 0.784 | 0.120 | 0.032 |
| 共享prompt + 仅图像数据(正样本) | 0.104 | 0.678 | 0.188 |
| 共享prompt + 仅图像数据(soft-positive) | 0.108 | **0.742** | 0.225 |
| 共享prompt + 混合数据 | 0.564 | 0.687 | 0.193 |
| 分离prompt + 简单拼接 | 0.502 | 0.615 | 0.156 |
| 分离prompt + 拼接后微调 | 0.251 | 0.648 | 0.189 |
| **分离prompt + Self-Prompting** | **0.747** | **0.761** | **0.224** |

### 关键发现

- **Soft-positive策略显著优于数据增强**：使用soft-positive图像比通过分割+修复的数据增强方法在人脸相似度上高出约20%，因为真实图像的质量远优于合成增强数据。
- **Self-prompting是平衡多任务的关键**：共享prompt联合训练两任务会互相损害，而self-prompting使模型能在两任务上都接近单任务最优水平，说明让模型"先判断任务类型"能有效解耦不同模态的表示。
- **仅32个token即超越1k+ token的prompt方法**：Yo'Chameleon用32个可学习token在图像生成上(CLIP-I: 0.783)大幅超越GPT-4o用1k token图像prompt(0.657)，展示了学习型表示的效率优势。
- **人脸生成仍有提升空间**：当前人脸相似度0.212，合格阈值约0.4。增加token数可以提升质量但存在收益递减（16 token是性价比拐点）。

## 亮点与洞察

- **"Soft-positive"概念的提出**：创造性地将hard-negative重新定义为不同程度的"soft-positive"，并根据相似度自适应分配prompt长度。这种连续化的正/负样本处理思路可以迁移到对比学习、检索增强等场景。
- **Self-prompting的双重角色**：token既是任务选择器又是内容编码器，一组参数承担两个功能，设计优雅。这种思路可以扩展到更多模态（如音频生成），只需添加新的token组和对应的self-prompting规则。
- **"300张图像即可匹敌全模型微调"的发现**：这个实验洞察直接启发了soft-positive策略。通过分析"差距从何而来"找到了"数据量不足"这个本质原因，然后针对性地解决。

## 局限与展望

- 基于Chameleon模型，其图像生成能力本身弱于DALL-E 3等专用模型，个性化效果受限于base model能力
- 人脸相似度(0.212)仍远低于合格标准(0.4)，个性化人物肖像生成还不够实用
- 每个新概念需要独立训练约15 epoch，无法做到即时个性化（推理时zero-shot）
- QA任务上明显弱于GPT-4o，部分因为base model（Chameleon）本身的理解能力差距
- 目前仅支持单个概念的个性化，多概念组合（如"我的狗在我的花园"）尚未解决

## 相关工作与启发

- **vs DreamBooth**: DreamBooth通过全模型微调实现高质量个性化图像生成，但会导致灾难性遗忘。Yo'Chameleon通过soft prompt保持模型冻结，用efficiency换取了更好的通用性保持。
- **vs Yo'LLaVA**: 前作只做了VLM的文本生成个性化。Yo'Chameleon扩展到统一的文本+图像生成，核心新增是soft-positive策略和双prompt/self-prompting机制。
- **vs Textual Inversion**: 都是学习token来表示新概念，但Textual Inversion只用于图像生成。Yo'Chameleon在统一LMM架构下实现了理解+生成的双重个性化。
- **vs GPT-4o**: GPT-4o通过prompt工程可以做基本的个性化，但在细粒度视觉细节（特别是人脸）方面远不如学习型方法。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次探索LMM个性化，提出的soft-positive和self-prompting机制都是新颖且有效的
- 实验充分度: ⭐⭐⭐⭐ 多角度消融实验充分，但数据集仅40个概念，规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，动机推导严谨，实验分析逻辑性强
- 价值: ⭐⭐⭐⭐ 开辟了LMM个性化的新方向，但受base model限制实际应用价值待观察

<!-- RELATED:START -->

## 相关论文

- [Do Visual Imaginations Improve Vision-and-Language Navigation Agents?](do_visual_imaginations_improve_vision-and-language_navigation_agents.md)
- [Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [PersonaBooth: Personalized Text-to-Motion Generation](personabooth_personalized_text-to-motion_generation.md)
- [DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)

<!-- RELATED:END -->
