---
title: >-
  [论文解读] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models
description: >-
  [NeurIPS 2025][幻觉检测][OCR] 针对多模态大模型在退化文档场景下的OCR幻觉问题，提出首个退化文档幻觉评测基准KIE-HVQA，并设计基于GRPO的多目标奖励强化学习框架，在7B参数模型上实现比GPT-4o高约28%的幻觉抑制准确率提升。
tags:
  - "NeurIPS 2025"
  - "幻觉检测"
  - "OCR"
  - "文档理解"
  - "强化学习"
  - "GRPO"
  - "视觉退化"
---

# Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.20168](https://arxiv.org/abs/2506.20168)  
**代码**: [https://huggingface.co/datasets/bytedance-research/KIE-HVQA](https://huggingface.co/datasets/bytedance-research/KIE-HVQA) (数据集公开)  
**领域**: 幻觉检测  
**关键词**: OCR幻觉, 文档理解, 强化学习, GRPO, 视觉退化

## 一句话总结
针对多模态大模型在退化文档场景下的OCR幻觉问题，提出首个退化文档幻觉评测基准KIE-HVQA，并设计基于GRPO的多目标奖励强化学习框架，在7B参数模型上实现比GPT-4o高约28%的幻觉抑制准确率提升。

## 研究背景与动机

多模态大语言模型（MLLM）在文档理解领域取得了显著进展，能够处理身份证、发票、合同等各类文档。然而，现有模型在真实场景中存在一个根本性的范式缺陷：当面对视觉退化情况（如模糊、遮挡、低对比度）时，模型无法严格遵循视觉信号，容易过度依赖语言先验或产生跨模态的幻觉内容。

这一问题的根源在于三个层面的挑战：(1) 预训练阶段缺少退化场景的关键信息提取（KIE）数据和清晰标注；(2) 指令微调阶段普遍忽视退化视觉场景的处理范式，研究者默认OCR输入是无退化的；(3) 评估阶段缺少专门量化文档理解OCR幻觉的基准。这导致模型在面对反光遮挡的身份证或低对比度报告时，会默认使用语言先验而非依靠可观察的视觉证据。

本文的切入角度是：将OCR幻觉问题建模为具有精确奖励的基本问题，利用KIE任务答案可量化的特性，通过强化学习让模型学会视觉忠实推理。

## 方法详解

### 整体框架

整个框架分为三个阶段：数据收集与构建、冷启动监督微调（SFT）、基于规则的强化学习（GRPO）。首先收集包含视觉图像描述的跨模态推理数据，然后通过SFT初始化模型的推理能力，最后通过GRPO配合精心设计的退化OCR奖励函数增强模型的泛化能力。

### 关键设计

1. **KIE-HVQA基准**: 首个专门评估退化文档OCR幻觉的基准数据集。包含2000个训练样本和400个测试实例，涵盖身份证、收据、发票三类文档。数据来源包括OCRBench（100个查询）、WildReceipt（实体答案重构）和GPT-4o生成的合成模板（200个隐私合规的虚拟证件）。每个样本模拟运动模糊、低对比度等真实退化场景，并提供像素级标注和OCR可靠性评分。评估指标包括清晰字符准确率、退化字符准确率和全局OCR性能。

2. **冷启动初始化（Cold-start Initialization）**: 解决纯语言推理模型无法直接处理多模态数据的问题。方法是先用GPT-4o将图像-问题-答案三元组转换为纯文本的伪CoT（包含图像描述和推理过程），再将这些信息与MLLM生成的详细图像描述合并，输入DeepSeek-R1生成高质量CoT数据。最终将文本CoT与对应图像配对，构建多模态CoT数据集用于冷启动。这种方法确保推理过程贴近人类认知行为。

3. **退化OCR多目标奖励函数**: 这是论文最核心的设计。根据字符退化程度将其分为三类：(a) 完全清晰字符——必须准确识别并保留；(b) 部分遮挡但人类可识别字符——标记为"异常"但仍需包含在输出中；(c) 完全不可识别字符——不应出现在OCR输出中，用空格替代以防止幻觉。例如"Beautiful"中，"B,a,u,f,u,l"清晰、"e"部分遮挡、"t,i"完全遮挡。奖励函数综合考虑清晰字符距离、不清晰字符距离和最终答案距离三个维度，使用编辑距离（Levenshtein distance）作为基础度量。

### 损失函数 / 训练策略

训练采用两阶段策略：

- **SFT阶段**: 使用Qwen-2.5-VL-7B-Instruct作为基座，在冷启动数据上微调5个epoch，学习率1e-6，批大小512，使用LLaMA-Factory框架，耗时约4小时。
- **GRPO阶段**: 在SFT模型基础上，混合TextOCR、WildReceipt和其他OCR数据集进行强化学习。GRPO对每个输入生成G个候选响应，通过组内归一化计算优势值，优化策略模型使其生成更高奖励的输出，同时通过KL散度约束防止偏离参考模型太远。使用Easy-R1框架实现。

## 实验关键数据

### 主实验

| 模型 | 清晰字符(Clr) | 不清晰字符(Nc) | 最终OCR(Final) | 平均(Avg) |
|------|-------------|--------------|--------------|----------|
| GPT-4o | 22.78 | 36.13 | 31.74 | 30.21 |
| Claude3.7-Sonnet | 19.77 | 33.73 | 26.17 | 26.56 |
| Gemini2.5-pro | 36.94 | 34.64 | 33.53 | 35.03 |
| Qwen2.5-VL-72B | 20.02 | 24.19 | 20.37 | 21.53 |
| InternVL3-78B | 6.09 | 8.59 | 6.43 | 7.04 |
| **本文(SFT+RL)** | **55.45** | **61.34** | **57.35** | **58.05** |

### 消融实验

| 配置 | 清晰字符(Clr) | 不清晰字符(Nc) | 最终OCR(Final) | 说明 |
|------|-------------|--------------|--------------|------|
| 仅清晰奖励 | 50.64 | 44.15 | 53.34 | 不清晰字符性能显著下降 |
| 仅最终奖励 | 51.06 | 54.06 | 54.24 | 不如组合奖励 |
| 全部奖励 | 55.45 | 61.34 | 57.35 | 各维度均最优 |
| 仅SFT | 49.65 | 57.25 | 49.72 | 基础能力已较强 |
| SFT+RL | 55.45 | 61.34 | 57.35 | RL带来额外提升 |

### 关键发现

- 7B参数的模型在退化文档幻觉抑制上比GPT-4o绝对提升约28%（58.05 vs 30.21）。
- 在不清晰字符识别上，本文模型（61.34%）远超GPT-4o（36.13%），证明不确定性感知机制的有效性。
- 通用OCR能力未受影响：在OCRbench的Scene（180）、Doc（179）、Info（183）三个子集上表现与GPT-4o（180/167/163）和原始Qwen2.5-VL-7B（181/181/182）相当。
- 多目标奖励组合对处理真实文档退化模式至关重要，单一奖励变体在各维度上均明显不足。

## 亮点与洞察

- 率先将KIE任务的答案可量化特性与强化学习结合，将OCR幻觉转化为精确可优化的问题。
- 三级字符退化分类（清晰/部分遮挡/完全遮挡）设计巧妙，使奖励函数能够精确引导模型行为。
- 通过拒绝回答机制增加任务难度，教会模型在无法确定时主动拒绝，而不是编造答案。
- 冷启动阶段利用GPT-4o+DeepSeek-R1的协同方案解决多模态CoT数据生成难题。

## 局限与展望

- 基准数据集规模相对有限（2000训练+400测试），退化类型和文档类型可进一步丰富。
- 目前仅在Qwen2.5-VL-7B上验证，更大规模模型的效果有待探索。
- 评估指标基于编辑距离，可能不完全反映语义层面的理解质量。
- 退化模拟主要依赖合成方式，与真实退化分布可能存在差异。

## 相关工作与启发

- DeepSeek-R1的GRPO算法为本文的强化学习框架提供了基础，但本文针对OCR任务设计了专门的多目标奖励机制。
- 现有OCR基准（OCRBench、DocLocal4K）主要关注行级别识别和文档理解，忽略了退化条件下的幻觉问题。
- 推理增强类工作（VisRL、Visual-RFT）开始将强化学习应用于MLLM视觉推理，但尚无针对OCR幻觉的专门研究。
- 启发：在需要精确答案的任务中，利用答案的量化特性构建细粒度奖励函数是一条有效路径。

## 评分

- 新颖性: ⭐⭐⭐⭐ （问题定义新颖，首个退化文档幻觉基准；但GRPO框架本身非原创）
- 实验充分度: ⭐⭐⭐⭐ （对比充分，消融完整；但数据集规模偏小）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机阐述充分）
- 价值: ⭐⭐⭐⭐ （为文档理解可靠性提供了重要方向）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](../../CVPR2025/hallucination/ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
