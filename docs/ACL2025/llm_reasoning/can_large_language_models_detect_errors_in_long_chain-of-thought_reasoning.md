---
title: >-
  [论文解读] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?
description: >-
  [ACL 2025][LLM推理][长链推理] 本文提出DeltaBench——首个系统评估o1类模型长CoT推理质量和现有LLM/PRM错误检测能力的基准数据集，通过对1,236个样本的精细人工标注，揭示了o1类模型约27%推理冗余、67.8%反思无效，以及最强critic模型GPT-4-turbo-128k也仅达F1=40.8%的令人警醒的现状。
tags:
  - ACL 2025
  - LLM推理
  - 长链推理
  - 错误检测
  - 过程奖励模型
  - Critic模型
  - o1-like模型
---

# Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?

**会议**: ACL 2025  
**arXiv**: [2502.19361](https://arxiv.org/abs/2502.19361)  
**代码**: [https://github.com/OpenStellarTeam/DeltaBench](https://github.com/OpenStellarTeam/DeltaBench)  
**领域**: LLM推理  
**关键词**: 长链推理, 错误检测, 过程奖励模型, Critic模型, o1-like模型

## 一句话总结
本文提出DeltaBench——首个系统评估o1类模型长CoT推理质量和现有LLM/PRM错误检测能力的基准数据集，通过对1,236个样本的精细人工标注，揭示了o1类模型约27%推理冗余、67.8%反思无效，以及最强critic模型GPT-4-turbo-128k也仅达F1=40.8%的令人警醒的现状。

## 研究背景与动机
o1类模型（如QwQ、DeepSeek-R1、Gemini 2.0 Flash Thinking）通过生成长Chain-of-Thought推理步骤显著提升了LLM的推理能力。然而，对这些长CoT的质量和效率缺乏系统性评估：它们包含多少冗余？哪些步骤有错误？反思机制是否有效？

同时，Process Reward Model（PRM）和Critic模型作为评估推理过程质量的工具日益重要，但它们在长CoT上的表现如何，此前几乎没有研究涉及。现有PRM基准（如ProcessBench）只关注短CoT且仅评估首个错误或样本级正确性，无法满足对长CoT的细粒度分析需求。

本文的核心目标是：（1）分析o1类模型生成长CoT的效率问题；（2）度量现有PRM和Critic模型在长CoT错误检测上的能力边界。

## 方法详解

### 整体框架
DeltaBench的构建流程：
1. **查询收集**：从多个开源数据集提取数学、编程、物化生（PCB）、通用推理四个领域的题目，经嵌入聚类去重（NV-Embed-v2 + DBSCAN，得到17,510条唯一查询）、难度筛选（6个模型投票评估）、子类均匀采样
2. **长CoT生成**：使用QwQ-32B-Preview、DeepSeek-R1、Gemini 2.0 Flash Thinking等o1类模型生成长CoT解答
3. **Section切分**：将长CoT按"\n\n"分段，再用GPT-4识别每段开始/结束步骤并生成摘要，形成独立子任务粒度的section
4. **人工标注**：硕博毕业生对每个section标注策略转换、推理有用性、推理正确性、反思效率四个维度

### 关键设计

1. **Section级评估粒度**

    - 不同于传统step级别（步骤太多，标注困难）或sample级别（粒度太粗），采用section级别：每个section代表一个独立子任务
    - 这更符合人类认知模式，降低标注成本的同时保持细粒度
    - 每个section标注4个维度：策略是否转换、推理是否有用、推理是否正确（若有错误还需标注首个错误步骤+解释+修正）、是否包含反思及其是否有效

2. **错误分类体系**

    - 将推理错误分为8大类、23种具体类型：理解错误、推理错误、计算错误、格式错误、知识错误、反思错误、总结错误等
    - 不同领域错误分布差异显著：数学以推理错误（25.3%）为主；编程以推理错误+格式错误为主；物化生以理解错误+知识错误为主

3. **PRM评估方法**

    - 不使用固定阈值（因长CoT的分数分布与短CoT差异大），而采用Z-Score离群检测：$t = \mu - \sigma$
    - 分数低于阈值的section被预测为错误
    - 评估指标使用Macro-F1以缓解正负样本不平衡

### 评估指标
- 对Critic模型：Recall、Precision、Macro-F1
- 对PRM：基于Z-Score的section级预测 + HitRate@k

## 实验关键数据

### 主实验（Critic模型 F1-Score）
| 模型 | 总体F1 | 数学 | 编程 | PCB | 通用推理 |
|------|--------|------|------|-----|---------|
| GPT-4-turbo-128k | **40.76** | 37.56 | 43.06 | 45.54 | 42.17 |
| GPT-4o-mini | 37.82 | 33.26 | 37.95 | 45.98 | 46.39 |
| Doubao-1.5-Pro | 35.25 | 32.46 | 39.47 | 33.53 | 37.00 |
| DeepSeek-R1 | 28.43 | 24.17 | 29.28 | 34.78 | 35.87 |
| o1-preview | 26.97 | 22.19 | 28.09 | 33.11 | 35.94 |
| o1-mini | 19.89 | 16.71 | 21.70 | 20.37 | 26.94 |

### PRM结果
| 模型 | Recall | Precision | F1 |
|------|--------|-----------|-----|
| Qwen2.5-Math-PRM-7B | 30.30 | 34.96 | **29.22** |
| Qwen2.5-Math-PRM-72B | 28.16 | 29.37 | 26.38 |
| Llama3.1-8B-PRM-Deepseek | 11.70 | 15.59 | 12.02 |

### 消融实验 / 分析发现
| 分析维度 | 关键数字 | 说明 |
|---------|---------|------|
| 推理冗余率 | ~27% | 平均27%的section推理无用 |
| 有效反思比例 | ~32.2% | 67.8%的反思无效 |
| 计算错误占比（QwQ） | 17.9% | QwQ在细节处理上明显薄弱 |
| 基础错误占比 | 23-25% | QwQ和Gemini中约四分之一是基础性错误 |
| DeepSeek-R1自我批评降幅 | 36% | 自我评估比评估他人差36% |

### 关键发现
- **o1类模型不擅长自我批评**：DeepSeek-R1自评F1比评他人低36%，其他o1模型也有类似趋势
- **o1类模型做Critic不占优势**：o1-preview的Critic F1甚至低于Qwen2.5-32B-Instruct
- **更大的PRM不一定更好**：Qwen2.5-Math-PRM-72B不如7B版本
- **CoT长度增加时，Critic表现显著下降**：从1-3k token到4-7k token，所有Critic模型F1大幅下降
- **PRM对CoT长度更鲁棒**：因为逐section评估，但整体分数仍低于Critic模型
- **模型识别策略错误最弱**：相比计算错误（识别较好），策略错误的检测能力普遍不足

## 亮点与洞察
- **首次系统化评估长CoT质量**：填补了重要研究空白，数据和发现对理解o1类模型机制有直接价值
- **Section级粒度设计**：在标注成本和评估细粒度之间取得了好的平衡
- **"27%冗余+68%反思无效"的定量发现**：对o1类模型的效率问题提供了有力证据
- **揭示了PRM在长CoT场景的不足**：最强PRM的F1仅29.22%，说明现有PRM远未解决长CoT评估问题

## 局限与展望
- 数据集规模受限于高成本人工标注（1,236个样本），扩展性有限
- 人工标注不可避免存在主观偏差，尤其在"推理有用性"判断上
- 作为静态基准，无法实时反映o1类模型的快速进展
- 未评估最新的o1和o3模型
- **研究idea**：可以探索训练专门针对长CoT的PRM——当前PRM主要在短CoT数据上训练，迁移到长CoT效果差；用DeltaBench的标注数据微调PRM可能是改进方向

## 相关工作与启发
- ProcessBench（Zheng et al., 2024）：step级别PRM评估基准，但仅针对短CoT
- CriticBench、CriticEval：sample级别Critic评估，不支持长CoT的细粒度分析
- PRM800K（Lightman et al., 2023）：经典过程监督数据集
- 本文的发现启示：（1）PRM需要针对长CoT场景重新设计和训练；（2）o1类模型的反思机制效率低下，有很大优化空间；（3）自我批评能力是模型的关键短板

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个长CoT细粒度质量分析基准，填补空白
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖了PRM+Critic两类模型，多维度分析（错误类型/CoT长度/自评vs互评/反思效率）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据可视化丰富，发现阐述明确
- 价值: ⭐⭐⭐⭐⭐ 对理解和改进o1类模型有重要指导意义，数据集已开源

<!-- RELATED:START -->

## 相关论文

- [Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)
- [Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models](chain-of-reasoning_towards_unified_mathematical_reasoning_in_large_language_mode.md)
- [DRT: Deep Reasoning Translation via Long Chain-of-Thought](drt_deep_reasoning_translation_via_long_chain-of-thought.md)
- [Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](../../NeurIPS2025/llm_reasoning/large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [Safe: Enhancing Mathematical Reasoning in Large Language Models via Retrospective Step-aware Formal Verification](safe_math_reasoning.md)

<!-- RELATED:END -->
