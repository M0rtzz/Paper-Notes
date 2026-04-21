---
title: >-
  [论文解读] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension
description: >-
  [ACL 2025][多模态][punchline comprehension] 本文提出PunchBench，一个包含6,000个图文对和54,000个问答对的多模态幽默/讽刺理解基准，通过同义/反义标题生成消除语言捷径，同时提出Simple-to-Complex Chain-of-Question (SC-CoQ)策略，在所有模型和问题格式上一致性提升punchline理解能力。
tags:
  - ACL 2025
  - 多模态
  - punchline comprehension
  - humor
  - sarcasm
  - chain-of-question
---

# PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension

**会议**: ACL 2025  
**arXiv**: [2412.11906](https://arxiv.org/abs/2412.11906)  
**代码**: https://github.com/OuyangKun10/PunchBench  
**领域**: 多模态VLM  
**关键词**: punchline comprehension, multimodal benchmark, humor, sarcasm, chain-of-question

## 一句话总结
本文提出PunchBench，一个包含6,000个图文对和54,000个问答对的多模态幽默/讽刺理解基准，通过同义/反义标题生成消除语言捷径，同时提出Simple-to-Complex Chain-of-Question (SC-CoQ)策略，在所有模型和问题格式上一致性提升punchline理解能力。

## 研究背景与动机
1. **领域现状**：多模态大语言模型（MLLM）在视觉问答、图像描述等事实性理解任务上取得了显著进展，但对幽默和讽刺等punchline（笑点/讽刺点）的理解能力尚未得到充分评估。
2. **现有痛点**：现有punchline理解基准存在三大缺陷：(1) 语言捷径——模型仅依赖文本中的偏置词或不一致性就能答对，无需真正理解图文交互；(2) 问题格式单一——仅用一种QA形式，无法全面评估模型鲁棒性；(3) 内容域狭窄——仅聚焦卡通等单一领域，覆盖不足。
3. **核心矛盾**：现有基准的设计缺陷导致无法区分"模型真正理解了punchline"和"模型利用了数据捷径"，评估结果的真实性存疑。
4. **本文目标** (1) 如何构建一个消除语言捷径、多问题格式、多领域的准确全面基准？(2) MLLM与人类在punchline理解上的差距有多大？(3) 如何提升MLLM的punchline理解能力？
5. **切入角度**：通过对标题进行同义/反义替换，生成modified标题以消除捷径偏差；同时从认知科学中"由简到难"的学习过程汲取灵感，设计SC-CoQ提示策略。
6. **核心 idea**：用同义/反义标题消除评测捷径，用由简到难的问题链提升MLLM的punchline理解。

## 方法详解

### 整体框架
PunchBench的构建分四步：(1) 从已有数据集和多媒体平台收集图文对并人工标注；(2) 生成同义和反义标题消除捷径；(3) 构建两层任务的多格式指令（感知层和推理层）；(4) 人工质量检查。在此基础上，提出SC-CoQ策略来改进模型表现。

### 关键设计

1. **同义/反义标题生成（Synonymous & Antonymous Captions）**:

    - 功能：消除模型可能利用的文本捷径
    - 核心思路：使用gpt-3.5-turbo对原始标题进行词替换（将情感词、动作词等替换为同义词/反义词）生成同义和反义标题。对于包含语义冲突成分的标题（如"I'm so glad! What a disgusting day!"），先用LLM识别冲突部分再分别处理。同义标题保持与原标题相同的punchline标签，反义标题作为对比
    - 设计动机：实验证明CogVLM2等模型能正确判断原始标题是否包含punchline，但面对同义/反义变体时性能大幅下降，说明模型依赖特定词汇而非真正理解

2. **双层多格式任务设计**:

    - 功能：从感知到推理多角度全面评估punchline理解
    - 核心思路：感知层（Punchline Perception）包含Yes/No QA（判断是否有punchline）、Matching QA（在两个标题中选出有punchline的）、Multi-option QA（四选一理解）；推理层（Punchline Reasoning）包含Yes/No QA（判断推理句是否正确解释了punchline）、Matching QA（选择正确解释）、Generation QA（自由生成解释）。每种格式都配有多种指令模板，并随机化选项顺序
    - 设计动机：单一问题格式无法全面评估——实验表明模型可能在Yes/No QA上表现好但Matching QA上失败，暴露了性能的不一致性

3. **Simple-to-Complex Chain-of-Question (SC-CoQ)**:

    - 功能：通过由简到难的问题序列逐步提升MLLM的punchline理解
    - 核心思路：在任务内部和任务之间组织从简单到复杂的问题链。具体地，先让模型回答感知层的简单问题（如Yes/No），再逐步过渡到推理层的复杂问题（如生成解释），利用前面简单问题的回答作为上下文辅助后续复杂问题的作答
    - 设计动机：复杂的punchline理解可分解为多个子技能（识别punchline存在→选择正确标题→解释原因），先掌握简单子技能再进阶，比直接面对复杂问题更有效

## 实验关键数据

### 主实验 — Punchline Perception

| 模型 | 参数量 | Yes/No (SC-CoQ) | Matching (SC-CoQ) | Multi-choice (SC-CoQ) |
|------|-------|-----------------|--------------------|-----------------------|
| GPT-4o | - | 80.7 | 67.9 | 53.1 |
| GPT-4V | - | 78.1 | 65.0 | 51.9 |
| Qwen2-VL-72B | 72B | 76.1 | 62.9 | 51.7 |
| Aria | 3.5B×8 | 74.5 | 63.6 | 50.8 |
| CogVLM2 | 19B | 71.3 | 60.8 | 46.3 |
| LLaVA | 7B | 64.8 | 57.1 | 39.1 |
| Human | - | **98.3** | **97.7** | **90.7** |

### SC-CoQ vs. 其他提示方法 (GPT-4o Perception Yes/No)

| 方法 | Accuracy |
|------|----------|
| Zero-shot | 77.5 |
| CoT | 78.6 |
| 3-shot | 79.2 |
| **SC-CoQ** | **80.7** |

### 关键发现
- MLLM与人类在punchline理解上存在巨大差距：最强模型GPT-4o在Perception Yes/No上为80.7%，人类为98.3%；Multi-choice上差距更大（53.1% vs 90.7%）
- SC-CoQ在所有模型和所有问题格式上都一致优于zero-shot、CoT和few-shot方法，且P值均<0.01，统计显著
- 模型在面对同义/反义标题时性能显著下降，证实了语言捷径确实存在
- 开源模型中Qwen2-VL-72B和Aria表现最好，接近GPT-4V水平；小模型（2B-7B）在多选题上仅略高于随机（25%）
- 推理层任务（Generation QA）难度最大，GPT-4o也仅有约53%

## 亮点与洞察
- 同义/反义标题是消除文本捷径的巧妙设计——比简单删除文本更精确地测试了"图文交互理解"能力，这个思路可以迁移到其他多模态基准中
- SC-CoQ的核心洞察是punchline理解是一个层次化能力：先感知存在性，再定位关键元素，最后推理原因。这种由简到难的范式比直接chain-of-thought更符合认知规律
- 人类在质量检查中500条指令仅1条标为"无法回答"，证明了数据集的高质量

## 局限与展望
- 数据集主要覆盖英语punchline，跨语言/跨文化的幽默理解未涉及
- SC-CoQ增加了推理步骤和token消耗，实际应用时效率需考量
- Generation QA的评估依赖参考答案相似度，可能无法完全捕捉多样化的正确解释
- 未探索微调策略——SC-CoQ仅作为推理时策略，训练时结合可能效果更好
- 6,000个图文对的规模虽不小，但每个域的分布和难度分布未详细分析
- 对punchline的定义（幽默+讽刺）可能遗漏其他需要深层理解的修辞（如反语、夸张）

## 相关工作与启发
- **vs MORE (Desai et al., 2022)**: 仅关注讽刺解释的单一任务，PunchBench覆盖幽默+讽刺、感知+推理、多问题格式
- **vs HUB (Hessel et al., 2023)**: 仅聚焦卡通域的幽默，PunchBench覆盖posts、cartoons、comments、memes多域
- **vs Chain-of-Thought**: CoT让模型自由推理，SC-CoQ则通过结构化的问题链引导推理方向，实验证明更有效
- PunchBench可作为VLM进化的长期追踪基准——随着模型能力增长，可观察punchline理解能力的提升趋势

## 评分
- 总体评价: 构建了高质量多模态幽默/讽刺理解基准，SC-CoQ策略具有实用价值
- 新颖性: ⭐⭐⭐⭐ 同义/反义标题消除捷径的设计和SC-CoQ策略都有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 12个模型、6种问题格式、4种提示方法的全面评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，示例和图表丰富
- 价值: ⭐⭐⭐⭐ 填补了MLLM punchline理解评估的空白

<!-- 数据规模: 6,000 image-caption pairs, 54,000 QA pairs, 4类多媒体域, 12个模型评测 -->

<!-- RELATED:START -->

## 相关论文

- [Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](../../NeurIPS2025/multimodal_vlm/dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)
- [MMINA: Benchmarking Multihop Multimodal Internet Agents](mmina_benchmarking_multihop_multimodal_internet_agents.md)
- [Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](../../ICCV2025/multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)

<!-- RELATED:END -->
