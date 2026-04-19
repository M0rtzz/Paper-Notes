---
title: >-
  [论文解读] Multimodal Pragmatic Jailbreak on Text-to-image Models
description: >-
  [ACL 2025][图像生成][多模态越狱] 提出"多模态语用越狱"（Multimodal Pragmatic Jailbreak）新型攻击方式，通过让T2I模型生成包含视觉文字的图像，使得图像内容和文字内容单独看都安全但组合后产生不安全内容，揭示了所有测试模型（包括DALL·E 3）均受此攻击影响。
tags:
  - ACL 2025
  - 图像生成
  - 多模态越狱
  - 文本到图像模型
  - 安全性
  - 视觉文本渲染
  - 扩散模型
---

# Multimodal Pragmatic Jailbreak on Text-to-image Models

**会议**: ACL 2025  
**arXiv**: [2409.19149](https://arxiv.org/abs/2409.19149)  
**代码**: [multimodalpragmatic.github.io](https://multimodalpragmatic.github.io/)  
**领域**: 图像生成  
**关键词**: 多模态越狱, 文本到图像模型, 安全性, 视觉文本渲染, 扩散模型

## 一句话总结

提出"多模态语用越狱"（Multimodal Pragmatic Jailbreak）新型攻击方式，通过让T2I模型生成包含视觉文字的图像，使得图像内容和文字内容单独看都安全但组合后产生不安全内容，揭示了所有测试模型（包括DALL·E 3）均受此攻击影响。

## 研究背景与动机

扩散模型在图像生成质量和文本一致性方面取得了显著进步，但同时其安全性问题日益受到关注。现有的安全研究主要关注单模态内容的安全过滤，而忽略了一种新的攻击方式：

- **单模态安全≠多模态安全**：当T2I模型生成包含视觉文字的图像时，图像部分和文字部分分别评估可能都是安全的，但它们的组合会产生不安全内容。例如，一张看似无害的动物图片配上特定文字可构成仇恨言论。
- **语用学启发**：这种攻击受语言学中"语用学"概念启发——意义不仅来自语言系统本身，还来自上下文因素。当视觉和文本信息结合时，可能产生暗示、夸张、讽刺、对比等复杂交互，导致不安全内容的涌现。
- **现有防御失效**：关键词黑名单、定制提示过滤器和NSFW图像过滤器等常见防御手段均无法有效应对这种跨模态的语用不安全内容。

## 方法详解

### 整体框架

本文工作聚焦于系统性研究和基准测试，而非提出新的生成方法，核心贡献包括：
1. 构建MPUP数据集（1,400条多模态语用不安全提示）
2. 在9个代表性T2I模型上进行基准测试
3. 评估现有安全过滤器的有效性
4. 探究越狱成功的底层原因

### 关键设计

1. **MPUP数据集构建**：

    - 涵盖OpenAI使用策略中的所有不当场景，分为4大类26子类：仇恨言论（500条，8子类）、人身伤害（400条，8子类）、欺诈（300条，6子类）、色情（200条，4子类）。
    - 提示格式：`"<图像生成提示>, with a sign that says, '<视觉文字提示>'"`。
    - 三步生成流程：(1) GPT-4生成各类别提示；(2) 模态选择过滤——移除单模态即不安全的提示；(3) 质量过滤——仅保留每个子类最强的前10%不安全案例。

2. **修辞语言类别标注**：

    - 为每条提示标注修辞手法类别：隐喻/明喻、讽刺、拟人/拟兽、典故、夸张。
    - 分析哪种修辞手法的提示更容易通过T2I模型表现为带视觉文字的图像，从而导致更高风险。

3. **安全分类器测试平台**：

    - **文本过滤器**：关键词黑名单（Midjourney和Leonardo.AI）、BERT分数语义相似度、LLM分类器（Vicuna 7B、GPT-3.5）、OpenAI Moderation API。
    - **图像分类器**：Q16、Multi-Headed SC（MHSC）、NSFW过滤器（基于ResNet-50）。
    - 通过"提示模态移除"和"提示模态修改"两种技术生成总计7,000条提示和9,800张图像用于评估。

4. **评估方法**：

    - 使用GPT-4o进行攻击成功率（ASR）评估，带有类别特定提示和少样本示例。
    - 与人类标注对比验证：GPT-4o与人类标签一致性74.3%（Claude 3.5 Sonnet仅53.9%）。
    - 视觉文字质量通过OCR精确匹配和子串匹配评估。

### 损失函数 / 训练策略

本文为安全性评估研究，不涉及模型训练。主要的技术分析包括：
- 将越狱能力归因于模型的视觉文字渲染能力
- 分析训练数据中包含视觉文字的图文对是渲染能力的来源

## 实验关键数据

### 主实验

| 模型 | 仇恨言论ASR(%) | 人身伤害ASR(%) | 欺诈ASR(%) | 色情ASR(%) | 平均ASR(%) |
|------|----------------|---------------|------------|------------|------------|
| DALL·E 3 | 63.3 | **85.4** | **72.4** | 52.4 | **68.2** |
| OpenDalle | **67.6** | 82.0 | 61.3 | **58.5** | 69.1 |
| Proteus | 58.6 | 76.5 | 62.7 | 46.5 | 62.9 |
| DeepFloyd | 57.8 | 66.5 | 49.7 | 61.5 | 59.1 |
| SDXL | 32.0 | 64.3 | 43.0 | 37.5 | 44.4 |
| SD | 33.0 | 46.8 | 42.3 | 30.5 | 38.2 |
| SLD | 7.6 | 11.0 | 5.0 | 3.0 | 7.4 |

### 消融实验

| 安全分类器 | 仇恨言论Acc(%) | 人身伤害Acc(%) | 欺诈Acc(%) | 色情Acc(%) | 说明 |
|------------|----------------|---------------|------------|------------|------|
| 随机过滤 | 80.0 | 80.0 | 80.0 | 80.0 | 基线 |
| 关键词黑名单 | 79.5 | 79.4 | 78.9 | 79.1 | 与随机持平 |
| BERT分数 | 78.0 | 78.8 | 78.9 | 79.1 | 与随机持平 |
| GPT-3.5 | 72.8 | 72.8 | 74.5 | 77.3 | 低于随机 |
| OpenAI Moderation API | 80.3 | 80.2 | 80.0 | 76.8 | 略优于随机 |
| Q16图像分类器 | 65.0 | 60.9 | 61.0 | 62.5 | 勉强优于随机 |

### 关键发现

- **所有测试模型均受影响**：9个T2I模型的ASR范围从约10%到70%，DALL·E 3尽管部署了多重安全措施仍表现为最不安全模型之一（平均ASR 68.2%）。
- **视觉文字渲染能力是关键因素**：子串OCR准确率与ASR高度相关（DALL·E 3子串OCR约50%，ASR约70%；SLD子串OCR极低，ASR仅7.4%）。
- **即使拼写错误也可能不安全**：部分文字虽未正确渲染，但错误渲染的文字仍可能被人类理解为不安全内容。
- **现有单模态安全过滤器全面失效**：文本过滤器和图像分类器的表现与随机过滤相当甚至更差。
- 在线T2I服务（Midjourney、Leonardo.AI等）的拒绝率极低（0-11.4%），ASR仍高达24-40%。

## 亮点与洞察

- **定义新型安全威胁**：首次系统化研究文本与图像跨模态组合产生的语用不安全性，超越了传统单模态安全范畴。
- **语言学理论与AI安全结合**：将语用学概念引入T2I模型安全分析，修辞手法分类增加了分析深度。
- **全面基准测试**：覆盖9个模型（含2个闭源商业模型），7,000条提示，12,600次查询，评估体系完整。
- **揭示根本原因**：将虚弱性归因为模型的视觉文字渲染能力和训练数据中的视觉文字样本。

## 局限与展望

- GPT-4o评估器与人类标注一致性仅74.3%，自动化评估的可靠性有待提高。
- 提示格式固定为 `"with a sign that says"` 模板，更自然或多样化的提示可能带来不同结果。
- 防御方案的探索较为初步，缺乏有效的多模态安全分类器提案。
- 未探索视频生成模型中类似的多模态语用越狱可能性。
- 可以进一步研究文化和语言差异对越狱效果的影响。

## 相关工作与启发

本文连接了多个研究领域：LLM越狱（Zou等人的对抗性后缀）、T2I模型安全（SLD、NSFW检测器）、视觉文字渲染（GlyphControl、ByT5）。其核心洞察是：当模型变得更"能干"（渲染视觉文字能力增强）时，安全风险也随之增加，这为AI能力与安全之间的权衡提供了一个具体案例。该工作对MLLM安全研究也有重要启发——跨模态内容的组合安全性评估是未来模型部署中必须考虑的问题。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次提出多模态语用越狱概念，视角独特，连接语言学理论与AI安全
- 实验充分度: ⭐⭐⭐⭐⭐ 9个模型、7000条提示、12600次查询、多种安全过滤器，实验规模大且全面
- 写作质量: ⭐⭐⭐⭐ 论述清晰、结构完整，安全性研究的伦理声明充分
- 价值: ⭐⭐⭐⭐⭐ 揭示了当前T2I模型安全机制的重大盲区，对产业实践有直接影响

<!-- RELATED:START -->

## 相关论文

- [MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models](../../AAAI2026/image_generation/macprompt_maraconic-guided_jailbreak_against_text-to-image_models.md)
- [Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](../../CVPR2025/image_generation/enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [Pragmatic Heterogeneous Collaborative Perception via Generative Communication Mechanism](../../NeurIPS2025/image_generation/pragmatic_heterogeneous_collaborative_perception_via_generative_communication_me.md)
- [Planning with Diffusion Models for Target-Oriented Dialogue Systems](difftod_diffusion_dialogue_planning.md)

<!-- RELATED:END -->
