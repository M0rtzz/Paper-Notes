---
title: >-
  [论文解读] BQA: Body Language Question Answering Dataset for Video Large Language Models
description: >-
  [ACL 2025][人体理解][肢体语言] 基于BoLD数据集通过四步半自动流水线构建了BQA——一个包含7,632个短视频的肢体语言情感识别多选QA基准，评估发现最强VideoLLM（GPT-4o/Gemini）准确率仅约60%远低于人类的85%，同时揭示了模型对面部表情的过度依赖以及针对特定种族群体的显著偏见。
tags:
  - ACL 2025
  - 人体理解
  - 肢体语言
  - VideoLLM
  - 情感识别
  - 多选QA
  - 社会偏见分析
  - 多模态
---

# BQA: Body Language Question Answering Dataset for Video Large Language Models

**会议**: ACL 2025  
**arXiv**: [2410.13206](https://arxiv.org/abs/2410.13206)  
**代码/数据**: [HuggingFace](https://huggingface.co/datasets/naist-nlp/BQA)
**机构**: 奈良先端科学技术大学院大学 (NAIST)
**领域**: 人体理解 / 视频理解 / 情感计算  
**关键词**: 肢体语言, VideoLLM, 情感识别, 多选QA, 社会偏见分析, Multimodal CoT

## 一句话总结

基于BoLD数据集通过四步半自动流水线构建了BQA——一个包含7,632个短视频的肢体语言情感识别多选QA基准，评估发现最强VideoLLM（GPT-4o/Gemini）准确率仅约60%远低于人类的85%，同时揭示了模型对面部表情的过度依赖以及针对特定种族群体的显著偏见。

## 研究背景与动机

**领域现状**: 视频大语言模型（VideoLLM）在视频摘要、视频问答等任务上取得了显著进展，能够整合多模态输入来理解视频内容。人类交流中大量信息依赖非语言线索——面部表情、眼神接触和肢体语言，而这些非语言沟通缺乏形式化规则，需要基于常识的复杂推理。

**现有痛点**: 现有工作（如SMILE）仅研究了单一情感（微笑）的检测，无法评估模型对全部人类情感的泛化理解能力。BoLD数据集虽然标注了26种情感，但其设计目标是直接预测情感值，不适合以自然语言提示的方式评测VideoLLM。

**核心矛盾**: 如果VideoLLM无法从肢体语言中准确解读人类情感，那它们将难以应用于对话系统和AI机器人等需要情感感知的未来应用场景——模型对人类无意识动作的误读可能导致严重的交互失败。

**本文目标**: 构建一个系统性的评估基准，全面测试VideoLLM对肢体语言所传达情感的理解能力，并深入分析模型在不同人口统计学维度上的偏见问题。

**切入角度**: 将已有的BoLD姿态估计数据集转化为适合VideoLLM评测的多选QA格式，利用Gemini进行半自动的数据集构建，并通过多个维度（性别/年龄/种族）的错误分析揭示模型的偏见模式。

**核心 idea**: 通过四步半自动流水线将26种情感标签的肢体语言视频转化为高质量的4选1 QA基准，证明肢体语言理解对当前VideoLLM仍是巨大挑战。

## 方法详解

### 整体框架

BQA数据集的构建基于BoLD（Body Language Dataset），后者包含从150部电影中剪辑的9,876个视频片段（每段约5秒，25fps，约125帧），由众包标注者为每个片段在10分等级上标注26种情感标签。BQA通过四步流水线将BoLD转换为多选QA格式：

1. **STEP1 候选项提取**: 将26种情感标签按心理学理论（James 1890）分为Happiness、Anger、Sadness、Pleasure四大组。正确答案取情感得分最高的标签，三个干扰项从其余三组各取一个，确保选项具有最大区分度。
2. **STEP2 问题生成**: 将四个候选选项和对应视频输入Gemini-1.5-pro，让模型生成如"视频中的人表现出什么情感？"的自然语言问题，借鉴了mCSQA的prompt设计。
3. **STEP3 质量过滤**: 用Gemini检查生成的问题是否足够客观，是否包含指向正确答案的线索信息（如"他看起来很震惊"），同时过滤含有害内容的问题。
4. **STEP4 难度标注**: 让Gemini作答生成的QA，答对标记为"Easy"，答错标记为"Hard"，用于后续的细粒度分析。

最终数据集按6:2:2划分为训练集（4,651条）、验证集（1,538条）和测试集（1,443条）。全部音频被移除，确保纯粹评估视觉理解能力。

### 关键设计1: 跨组情感分层采样策略

传统的多选QA基准往往随机选择干扰项，但在情感识别场景中，同一情感大类内的子类（如Happiness组内的Confidence和Affection）差异极小，随机选择会导致评测效度低下。BQA采用跨组分层采样：将26种情感聚类为4个心理学大类后，正确答案和3个干扰项分别来自4个不同的组。例如正确答案若为Surprise（Pleasure组），干扰项则分别从Happiness组（如Confidence）、Anger组（如Anger）、Sadness组（如Embarrassment）各取一个。这种设计保证了选项间有足够的语义距离，使QA任务评测的是模型对核心情感类别的判别能力而非细粒度子类的区分。

### 关键设计2: Gemini驱动的难度自标定与质控机制

数据集构建中Gemini扮演了多重角色——不仅生成问题，还参与质量过滤和难度标定。这带来了一个巧妙的性质：如果Gemini自己都无法正确回答它生成的问题（标记为Hard），那说明该问题具有足够的挑战性。实验证实Gemini在Hard样本上仅有8%的准确率，而在Easy样本上为91%，验证了这一标记机制的有效性。进一步分析发现Hard样本的共性特征：(1) 视频中人物表情中性、缺乏显著情感线索；(2) 面部被眼镜、帽子或太阳镜遮挡。这揭示了VideoLLM在理解肢体语言时过度依赖面部表情而非整体身体动作的倾向。

### 关键设计3: 多维偏见分析框架

BQA继承了BoLD中的人口统计元数据（性别、年龄、种族），在标准性能评测之外，系统分析了各模型在不同群体上的错误率分布。具体而言，对每个模型分别统计其在不同性别（男/女）、年龄组（儿童/青少年/成人/老年）、种族（白人/黑人/亚裔/拉美裔/原住民等）上的错误比例，并通过可视化对比发现偏见模式。这种分析框架将BQA从纯粹的性能基准提升为一个公平性评估工具。

## 实验关键数据

### 主实验: 测试集性能对比 (Table 1)

| 模型 | 帧数 | Easy | Hard | Total | Easy(CoT) | Hard(CoT) | Total(CoT) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Human (100样本) | - | 0.96 | 0.77 | **0.85** | - | - | - |
| Gemini-1.5-pro | 1fps | 0.91 | 0.08 | 0.61 | 0.94 | 0.90 | 0.92 |
| GPT-4o | 1fps | 0.78 | 0.38 | 0.60 | 0.97 | 0.95 | 0.96 |
| Phi-3.5 | 16 | 0.77 | 0.41 | 0.58 | 1.00 | 0.96 | 0.98 |
| Qwen2-VL | 16 | 0.68 | 0.27 | 0.47 | 0.98 | 0.95 | 0.97 |
| LLaVA-NeXT | 16 | 0.66 | 0.30 | 0.47 | 1.00 | 0.96 | 0.98 |
| VideoLLaMA2 | 16 | 0.15 | 0.01 | 0.08 | 0.15 | 0.01 | 0.08 |
| VideoLLaMA2 (FT) | 16 | 0.98 | 0.91 | **0.94** | 0.98 | 0.90 | 0.94 |

### 验证集性能 & 数据集统计 (Table 3 & 4)

| 模型 | Easy | Hard | Total |
|------|:---:|:---:|:---:|
| Phi-3.5 | 0.76 | 0.38 | 0.56 |
| Qwen2-VL | 0.69 | 0.32 | 0.50 |
| LLaVA-NeXT | 0.65 | 0.31 | 0.47 |
| VideoLLaMA2 | 0.40 | 0.09 | 0.24 |
| VideoLLaMA2 (FT) | 0.89 | 0.68 | 0.78 |

数据集规模：Train 4,651（Easy 2,192 / Hard 2,459）、Valid 1,538（746/792）、Test 1,443（707/736）。Self-BLEU (4-gram)在三个集上分别为0.87/0.86/0.85，表明问题多样性跨划分保持均衡。

### 关键发现

1. **肢体语言理解极具挑战**: 最强的GPT-4o和Gemini准确率仅约60%，远低于人类85%，且Hard子集上差距更大（Gemini仅8%）。
2. **CoT提升是答案泄露**: Multimodal CoT将GPT-4o从60%提升至96%，但分析发现生成的rationale中经常显式包含正确答案——这是答案泄露而非真正的推理能力提升。
3. **模型严重依赖面部表情**: Hard样本的特征是中性表情或面部遮挡，表明VideoLLM并未真正理解肢体语言，而是主要依赖面部线索进行情感判断。
4. **种族偏见显著**: LLaVA-NeXT对原住民（Native Hawaiian）相关视频的准确率仅约25%，Gemini也在该群体上表现明显偏差。性别维度上各模型无显著偏见。
5. **微调效果惊人**: VideoLLaMA2从裸模型的8%到LoRA微调后94%（LoRA r=128, alpha=256, 1 epoch），甚至超过Gemini，说明预训练数据中缺少肢体语言知识但模型具有学习能力。
6. **情感预测偏好**: 错误分析显示所有模型都不会错误预测为"Pleasure"类情感，但在正确答案为"Happiness"时经常错选"Sadness"或"Anger"等对立情感。

## 亮点与洞察

- **填补评估空白**: 首个系统评估VideoLLM肢体语言情感理解能力的基准数据集，7,632个视频覆盖26种情感和多种人口统计学维度。
- **揭示Multimodal CoT的陷阱**: CoT生成的rationale中泄露答案导致"虚假"的性能提升，这一发现对整个VLM评估社区具有警示意义——CoT得分不能等同于真实理解能力。
- **偏见分析开辟新维度**: 将性能评测与公平性分析结合，发现VideoLLM在少数族裔群体上的表现显著偏差，具有重要的社会意义。
- **难度自标定机制设计精巧**: 利用Gemini自身的答题能力来标定题目难度，无需额外人工标注成本，且实验验证了Easy/Hard标签的有效性。
- **数据污染排除论证充分**: 从问题原创性和Gemini有限准确率两个角度论证了不存在数据污染，自偏好偏见也因答案来源外部（BoLD）而被排除。

## 局限与展望

- **视频质量**: BoLD源自老电影，部分视频画质较差，可能影响模型实际能力的评估上限。
- **人类评估规模有限**: 仅3名标注者评估100个随机样本（agreement 0.79），缺乏跨文化标注者的多样性视角。
- **帧数限制**: 固定16帧输入未探索更多帧数的影响，且部分模型（如Gemini/GPT-4o）使用1fps而非固定帧数，比较不完全公平。
- **情感粒度损失**: 26种→4组的简化虽然保证了QA可行性，但丢失了细粒度情感区分的评估能力。
- **音频信息丢弃**: 移除音频虽确保纯视觉评估，但忽略了语调等重要情感线索，不反映真实多模态场景。
- **文化差异未考虑**: 肢体语言的情感表达在不同文化中可能有显著差异，数据集未对此进行控制或分析。

## 相关工作与启发

- **vs SMILE (Hyun et al., NAACL 2024)**: SMILE仅分析微笑一种情感的多模态理解，BQA则覆盖26种情感标签，评估范围大幅扩展。
- **vs Video-MME (Fu et al., 2024)**: Video-MME是综合性视频理解基准，BQA则专注于肢体语言情感理解这一特定能力维度，两者互补。
- **vs mCSQA (Sakai et al., ACL 2024)**: BQA借鉴了mCSQA的半自动QA生成方法论，将其从文本常识推理迁移到视频情感理解领域。
- **启发**: 理解"人在做什么动作"和理解"人感受到什么情感"是两个完全不同层次的任务。当前VideoLLM在动作识别上表现不错，但在情感感知上仍有巨大鸿沟。CoT等推理增强方法的表面提升可能掩盖真实能力不足，评估方法设计需要更加审慎。

## 评分

- 新颖性: ⭐⭐⭐⭐ 选题新颖且切入角度独特，首次构建系统性肢体语言理解评估基准
- 实验充分度: ⭐⭐⭐⭐ 覆盖6个VideoLLM+人类基线，偏见分析和CoT泄露分析都很完整
- 写作质量: ⭐⭐⭐ 结构清晰但部分分析（特别是偏见部分）可以更加深入定量
- 价值: ⭐⭐⭐⭐ 数据集已开源于HuggingFace，可直接用于VideoLLM的公平性和理解力评估

<!-- RELATED:START -->

## 相关论文

- [Failures to Surface Harmful Contents in Video Large Language Models](../../AAAI2026/human_understanding/failures_to_surface_harmful_contents_in_video_large_language_models.md)
- [Uncovering Strategic Egoism Behaviors in Large Language Models](../../NeurIPS2025/human_understanding/uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining](../../NeurIPS2025/human_understanding/graphchain_large_language_models_for_large-scale_graph_analysis_via_tool_chainin.md)
- [Enhancing Decision-Making of Large Language Models via Actor-Critic](../../ICML2025/human_understanding/enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models](../../NeurIPS2025/human_understanding/incentivizing_reasoning_for_advanced_instruction-following_of_large_language_mod.md)

<!-- RELATED:END -->
