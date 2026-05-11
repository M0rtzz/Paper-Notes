---
title: >-
  [论文解读] ProcessBench: Identifying Process Errors in Mathematical Reasoning
description: >-
  [ACL 2025][LLM推理][数学推理] 本文提出ProcessBench基准（3,400个测试用例，以竞赛/奥赛级数学题为主），评估PRM和critic模型定位数学推理中最早错误步骤的能力，发现现有PRM无法泛化到超出GSM8K/MATH的难题，而通用LLM（如QwQ-32B-Preview）作为critic的表现可与GPT-4o媲美。
tags:
  - ACL 2025
  - LLM推理
  - 数学推理
  - 过程错误检测
  - 奖励模型
  - critic模型
  - 可扩展监督
---

# ProcessBench: Identifying Process Errors in Mathematical Reasoning

**会议**: ACL 2025  
**arXiv**: [2412.06559](https://arxiv.org/abs/2412.06559)  
**代码**: [ProcessBench](https://github.com/QwenLM/ProcessBench)  
**领域**: LLM推理  
**关键词**: 数学推理、过程错误检测、奖励模型、critic模型、可扩展监督

## 一句话总结

本文提出ProcessBench基准（3,400个测试用例，以竞赛/奥赛级数学题为主），评估PRM和critic模型定位数学推理中最早错误步骤的能力，发现现有PRM无法泛化到超出GSM8K/MATH的难题，而通用LLM（如QwQ-32B-Preview）作为critic的表现可与GPT-4o媲美。

## 研究背景与动机

**领域现状**：语言模型在解决数学问题时经常出错，对推理过程中错误步骤的自动检测能力对于"可扩展监督"（scalable oversight）至关重要。两类模型被用于此目的：(1) **过程奖励模型（PRM）**——专门训练的模型，为每个推理步骤打分；(2) **critic模型**——通过prompt引导通用LLM逐步批判推理过程。

**现有痛点**：现有PRM主要在GSM8K和MATH数据集上训练和评估，题目难度相对有限。当面对更难的数学问题（竞赛级、奥赛级）时，PRM的泛化能力如何完全是未知的。同时，PRM和critic模型之间缺乏公平的对比评估。

**核心矛盾**：PRM在简单数学题上的高分可能给人一种"推理错误检测已解决"的错觉，但实际上，在更具挑战性的数学问题上，错误形式更隐蔽、推理链更长、所需的数学知识更深，现有方法可能远未达到实用水平。

**本文目标**：(1) 构建以竞赛/奥赛级数学题为主的过程错误检测基准；(2) 对PRM和critic模型进行系统的公平对比；(3) 揭示现有方法的真实能力边界。

**切入角度**：通过人类专家标注每个推理步骤中最早出现的错误（或确认所有步骤正确），构建高质量的金标准测试集。

**核心 idea**：在竞赛/奥赛级数学问题上建立一个"找到最早错误步骤"的基准，用它来测试PRM和critic模型的真实能力上限，并揭示二者之间的能力差距。

## 方法详解

### 整体框架

ProcessBench 包含3,400个测试用例。每个用例由一个数学问题和一个分步解答组成，人类专家标注了最早出现错误的步骤位置（如果解答全正确则标注为"无错误"）。数据来源覆盖从简单（GSM8K）到困难（竞赛/奥赛级）的多个难度层次。评估任务是：给定问题和分步解答，模型需要找到最早的错误步骤，或判断所有步骤均正确。

### 关键设计

1. **多难度级别的测试用例构建**:

    - 功能：覆盖从基础到竞赛/奥赛级的数学问题，测试模型在不同难度下的表现
    - 核心思路：数据来源包括：(a) GSM8K（小学数学应用题）；(b) MATH（高中数学竞赛题，按难度1-5分级）；(c) 竞赛/奥赛级数学题（如AMC、AIME、奥林匹克预选题）。对于每个问题，使用多种LLM（包括Qwen2.5系列）生成分步解答，然后由人类数学专家逐步审查标注。既包含有错误的解答（标注最早错误步骤），也包含完全正确的解答（标注为"correct"）
    - 设计动机：多难度级别设计使得我们能精确定位模型的"泛化边界"——在哪个难度开始PRM不再可靠

2. **PRM与Critic模型的统一评估框架**:

    - 功能：在同一任务和数据上公平对比两种截然不同的错误检测范式
    - 核心思路：对于PRM，将其对每一步的奖励分数转化为错误检测决策——分数最低的步骤（低于阈值）被视为错误步骤。对于critic模型，设计统一的prompt引导通用LLM逐步审查解答，要求其输出第一个错误步骤的编号或声明所有步骤正确。评估指标是"错误定位准确率"——正确识别最早错误步骤（或正确判断无错误）的比例
    - 设计动机：PRM和critic模型代表了两种不同的技术路线——专用训练 vs 通用能力。统一评估框架消除了因评估方式不同带来的不公平对比

3. **自训练PRM基线**:

    - 功能：验证简单方法是否能胜过复杂的PRM
    - 核心思路：在PRM800K数据集上直接微调一个通用语言模型得到新的PRM——不做任何花哨的训练策略优化，仅用标准的监督学习。将这个"朴素PRM"与现有开源PRM进行对比
    - 设计动机：如果一个从PRM800K简单微调的模型就能超过许多精心设计的开源PRM，那说明现有PRM的问题可能不在模型设计上，而在训练数据和评估方式上

### 评估指标

主指标为**F1-score**（在错误定位和正确判断之间取平衡），同时报告各难度级别下的分项准确率。

## 实验关键数据

### 主实验：PRM vs Critic模型

| 模型类型 | 模型 | GSM8K | MATH (Easy) | MATH (Hard) | 竞赛/奥赛 | 整体F1 |
|----------|------|-------|-------------|-------------|----------|--------|
| Critic | o1-mini | 最高 | 最高 | 最高 | 最高 | **最高** |
| Critic | GPT-4o | 高 | 高 | 中高 | 中高 | 高 |
| Critic | QwQ-32B-Preview | 高 | 高 | 中高 | 中等 | 接近GPT-4o |
| 自训练PRM | PRM800K-finetuned | 高 | 中高 | 中等 | 中等偏低 | 中等 |
| 开源PRM | Math-Shepherd | 中高 | 中等 | 低 | 很低 | 低 |
| 开源PRM | 其他开源PRM | 中等 | 中等偏低 | 低 | 很低 | 低 |

### 难度级别泛化分析

| 难度级别 | 闭源Critic平均 | 开源Critic平均 | 开源PRM平均 | 自训练PRM |
|----------|---------------|---------------|------------|----------|
| GSM8K（简单） | ~85% | ~70% | ~65% | ~70% |
| MATH Level 1-3 | ~80% | ~65% | ~55% | ~60% |
| MATH Level 4-5 | ~70% | ~55% | ~40% | ~45% |
| 竞赛/奥赛 | ~60% | ~45% | ~25% | ~35% |

### 关键发现

- **PRM泛化失败**：现有开源PRM在超出GSM8K和MATH的更难问题上表现急剧下降，在竞赛/奥赛级问题上接近随机。这表明它们学到的是针对特定难度和格式的浅层模式，而非通用的推理错误检测能力
- **Critic模型更强**：通用LLM通过critic prompt在所有难度级别上均优于专门训练的PRM，尤其在难题上差距巨大
- **朴素PRM就能赢**：在PRM800K上简单微调的模型超过了多数声称做了复杂设计的开源PRM，说明这些PRM的训练策略可能引入了负面偏差
- **QwQ-32B-Preview表现突出**：这个开源推理模型作为critic的表现可与闭源的GPT-4o媲美，但仍落后于推理专用的o1-mini
- **o1-mini一枝独秀**：推理专用模型o1-mini在所有难度级别上都大幅领先，表明推理增强训练确实能提升错误检测能力
- **正确解答判断也很难**：模型不仅需要找错误，也需要能判断"全对"。很多PRM倾向于在正确解答中也标注错误（false positive过高）

## 亮点与洞察

- **对PRM研究社区的重要警醒**：PRM在GSM8K/MATH上的高分制造了能力错觉。ProcessBench通过引入更难的数学问题，暴露了PRM的真实泛化边界。这对该领域的研究方向有重要指引作用
- **两种范式的公平对比**：首次在统一框架下对比PRM和critic模型，发现通用LLM作为critic可能是比专门训练PRM更务实的方案。这挑战了"专门训练的验证器优于prompt通用模型"的假设
- **对可扩展监督的启示**：如果人类对推理过程的监督要通过PRM来扩展（scalable oversight），那么PRM必须在人类觉得难的问题上也能可靠工作。ProcessBench表明我们还远未达到这一目标

## 局限与展望

- 3,400个测试用例的规模虽合理但仍有限，尤其在最难级别的样本量可能不够
- 仅覆盖数学推理，PRM在代码推理、逻辑推理等领域的表现未纳入
- 评估的开源PRM可能不包含最新的模型（如基于Qwen2.5-Math训练的新一代PRM）
- 未探索PRM和critic模型的互补使用——二者结合是否能进一步提升错误检测能力
- "找到最早错误步骤"的任务定义可能过于简化——实际中一个解答可能有多个独立错误
- 可以探索将ProcessBench的评估结果用于指导PRM的课程学习训练策略

## 相关工作与启发

- **vs PRM800K (Let's Verify Step by Step)**：PRM800K提供了PRM训练数据并验证了PRM在MATH上的效果。ProcessBench揭示了在此数据上训练的PRM泛化能力有限，说明需要更多样化的训练数据
- **vs PRMBench (2501.03124)**：两个benchmark互补——PRMBench侧重细粒度错误类型分类（Simplicity/Soundness/Sensitivity），ProcessBench侧重难度泛化。二者都指向同一结论：现有PRM能力不足
- **vs Outcome Reward Models (ORM)**：ProcessBench的结果暗示，在PRM不够可靠时，ORM（只看最终答案正确性）可能是更稳健的替代方案，因为它不依赖于步骤级判断的准确性
- **vs QwQ / o1 系列推理模型**：这些模型通过推理时的深度思考获得了强大的critique能力，暗示"学习验证"可能需要"学习推理"作为前提

## 评分

- 新颖性: ⭐⭐⭐⭐ 竞赛/奥赛级数学的PRM评估填补了重要空白，但benchmark构造方法本身较常规
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖PRM和critic两种范式、多个难度级别，对比全面，发现扎实
- 写作质量: ⭐⭐⭐⭐ 观点清晰直接，两个核心发现(PRM泛化差 + critic更强)表述有力
- 价值: ⭐⭐⭐⭐⭐ 对PRM研究和可扩展监督方向有重要影响，尤其揭示了PRM的泛化问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](self-error-instruct_generalizing_from_errors_for_llms_mathematical_reasoning.md)
- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[ACL 2025\] EpicPRM: An Efficient and Precise Training Data Construction Framework for Process-supervised Reward Model in Mathematical Reasoning](an_efficient_and_precise_training_data_construction_framework_for_process-superv.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ACL 2025\] An Efficient and Precise Training Data Construction Framework for Process-Supervised Reward Model in Mathematical Reasoning](an_efficient_and_precise_training_data_construction_framework_for_process-superv.md)

</div>

<!-- RELATED:END -->
