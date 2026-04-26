---
title: >-
  [论文解读] Mixture of Small and Large Models for Chinese Spelling Check
description: >-
  [ACL 2025][中文拼写检查] 本文提出在Beam Search解码阶段动态混合小模型（fine-tuned BERT）和大语言模型（LLM）的概率分布来进行中文拼写纠错，无需微调LLM即可兼顾小模型的精确纠错和LLM的流畅性，在多个CSC数据集上达到SOTA。
tags:
  - ACL 2025
  - 中文拼写检查
  - 模型混合
  - Beam Search
  - BERT
  - 大语言模型
---

# Mixture of Small and Large Models for Chinese Spelling Check

**会议**: ACL 2025  
**arXiv**: [2506.06887](https://arxiv.org/abs/2506.06887)  
**代码**: [https://github.com/zhqiao-nlp/MSLLM](https://github.com/zhqiao-nlp/MSLLM)  
**领域**: NLP理解  
**关键词**: 中文拼写检查、模型混合、Beam Search、BERT、大语言模型

## 一句话总结

本文提出在Beam Search解码阶段动态混合小模型（fine-tuned BERT）和大语言模型（LLM）的概率分布来进行中文拼写纠错，无需微调LLM即可兼顾小模型的精确纠错和LLM的流畅性，在多个CSC数据集上达到SOTA。

## 研究背景与动机

**领域现状**：中文拼写检查（Chinese Spelling Check, CSC）是NLP经典任务，目标是检测和纠正文本中的拼写错误（通常是形近字、音近字替换）。当前主流方法分为两派：(1) 微调BERT类小模型，在高质量标注数据上学习纠错模式；(2) 利用LLM的语言知识直接进行纠错。

**现有痛点**：两派各有硬伤。BERT类小模型精确率高，但严重依赖训练数据的编辑模式——模型过度拟合了训练集中的错误类型和位置分布（edit pattern overfitting），遇到新模式时表现骤降。LLM虽然语言知识丰富、不受限于特定编辑模式，但在CSC上的表现反而不如精调BERT——LLM倾向于做更大范围的改写而非精确的单字纠错，导致"改多了"（过纠错）。

**核心矛盾**：小模型的精确性和LLM的泛化性难以兼得。微调LLM成本高且效果不佳；直接拼接两个模型的输出也缺乏优雅的融合方式。

**本文目标**：在不微调LLM的前提下，设计一种能动态融合小模型精确纠错能力和LLM流畅性的方法。

**切入角度**：小模型和LLM的互补性体现在概率分布层面——小模型在纠错位置给出高置信度的候选字，LLM在全局语言流畅性上提供约束。如果能在解码阶段实时混合两者的概率分布，就能取长补短。

**核心 idea**：在Beam Search解码过程中，对每个位置的token预测，动态加权混合小模型和LLM的概率分布，使最终预测既继承小模型的精确纠错信号，又受LLM的语言模型流畅性约束。

## 方法详解

### 整体框架

输入为可能含错误的中文句子。首先通过fine-tuned BERT小模型和LLM各自独立计算每个位置上各候选字的概率分布。然后在Beam Search解码阶段，对两者的概率分布进行动态加权混合，得到混合概率分布，在此基础上进行beam search选择最优纠错序列。输出为纠错后的句子。

### 关键设计

1. **动态概率分布混合**:

    - 功能：在每个解码步实时结合小模型和LLM的预测
    - 核心思路：对位置 $t$ 的混合概率为 $P_{mix}(w_t) = \alpha \cdot P_{small}(w_t) + (1-\alpha) \cdot P_{LLM}(w_t)$，其中 $\alpha$ 是混合权重。关键在于 $\alpha$ 不是全局固定值，而是根据小模型的纠错置信度动态调整——当小模型对某位置的纠错高度自信（概率分布集中）时，$\alpha$更大，更多依赖小模型；当小模型不确定时，$\alpha$减小，更多依赖LLM的语言模型判断
    - 设计动机：静态混合权重无法适应不同位置的纠错需求——有些位置是明显的拼写错误（小模型擅长），有些涉及语义理解（LLM擅长）。动态 $\alpha$ 让系统自适应地在两个模型间切换

2. **Beam Search解码整合**:

    - 功能：在序列级别（而非独立位置级别）寻找最优纠错方案
    - 核心思路：标准BERT CSC模型通常对每个位置独立预测，忽略位置间的依赖。本文将混合概率 $P_{mix}$ 接入Beam Search框架，beam中的每个候选序列的得分是各位置混合概率的乘积（对数空间为求和）。Beam size和搜索策略可调
    - 设计动机：序列级解码能捕获纠错之间的依赖关系（如改了第3个字可能影响第5个字的最优选择），比独立位置预测更合理

3. **无需微调LLM的即插即用设计**:

    - 功能：降低使用门槛，支持灵活的领域适配
    - 核心思路：LLM以零样本方式工作——只需给定input句子让LLM计算每个位置的条件概率，不需要任何CSC特定的微调。只需微调小型BERT模型。更换领域时，只需替换小模型的微调数据，LLM部分无需改动
    - 设计动机：微调LLM耗时耗力且在CSC上效果不佳（过纠错问题严重），不微调可节省大量资源。同时，不同LLM可以即插即用，方便升级

### 损失函数 / 训练策略

小模型（BERT）使用标准的CSC训练策略：交叉熵损失，在标注的纠错数据上微调。LLM不做任何微调，直接使用预训练权重计算条件概率。混合权重 $\alpha$ 的动态调节策略在验证集上调优。

## 实验关键数据

### 主实验（与现有方法对比）

| 方法 | SIGHAN15 F1 | ECSpell F1 | LEMON F1 | 类型 |
|------|------------|------------|----------|------|
| BERT-based (ReaLiSe等) | 高精确/低召回 | 高精确/低召回 | 中等 | 小模型 |
| LLM直接纠错 (GPT-4等) | 低精确/高召回 | 低 | 低 | 大模型 |
| 前SOTA | 次优 | 次优 | 次优 | - |
| MSLLM (本文) | **SOTA** | **SOTA** | **SOTA** | 混合 |

### 消融实验

| 配置 | F1变化 | 说明 |
|------|--------|------|
| Full MSLLM | 最优 | 完整混合系统 |
| 仅小模型 (α=1) | 下降 | 缺少LLM的流畅性约束 |
| 仅LLM (α=0) | 大幅下降 | LLM的过纠错问题突出 |
| 静态α | 下降 | 不如动态权重灵活 |
| w/o Beam Search (贪心) | 下降 | 序列级搜索比独立位置好 |

### 关键发现

- **混合策略的互补性显著**：小模型精确率高但召回低（保守），LLM召回高但精确率低（过激）。混合后两者的弱点互相弥补，F1显著提升
- **动态α优于静态α**：验证了"不同位置需要不同程度的模型信任"这一假设
- **无需微调LLM是重要优势**：实验显示微调LLM用于CSC反而会降低效果（因为LLM的纠错模式会变得过于激进），即插即用策略是更明智的选择
- **领域适配方便**：在不同领域数据集（通用/医学/法律）上只需替换小模型，LLM组件无需改动，降低了部署成本

## 亮点与洞察

- **问题分析透彻**：小模型的edit pattern overfitting和LLM的过纠错问题是CSC领域的公认难题，本文对两个问题的根因分析（概率分布层面的互补性）非常精准
- **解码层融合的思路很巧妙**：不在输入/特征/输出层面融合两个模型，而是在概率分布层面混合，零耦合、即插即用。这一思路可推广到任何需要融合不同规模模型的序列预测任务（如语法纠错、机器翻译后编辑等）
- **实用价值高**：无需微调LLM、SOTA性能、领域适配方便，三个特点使其在工业部署中非常有吸引力

## 局限与展望

- **需要同时运行两个模型**：推理时需要小模型+LLM同时在线计算概率，推理延迟和内存占用更高
- **动态α的自适应策略比较启发式**：基于小模型置信度的α调节缺乏理论保证，更优的混合策略值得探索（如学习型mixture）
- **实验主要在SIGHAN系列数据集**：这些经典数据集可能不完全反映真实场景的错误分布
- **仅限中文拼写检查**：是否可推广到英文拼写/语法纠错、日文/韩文纠错等尚待验证
- 未来可探索将多个小模型（针对不同错误类型）与LLM混合的"多专家"方案

## 相关工作与启发

- **vs ReaLiSe (Xu et al. 2021)**: ReaLiSe融合了字音和字形信息的BERT模型，在CSC上表现优秀但仍有过拟合问题。本文的混合方法可以直接在ReaLiSe基础上加入LLM概率，进一步提升
- **vs LLM直接CSC (如ChatGPT/GPT-4)**: 多项研究已表明LLM在CSC上的"过纠错"问题——本文通过混合策略巧妙绕过了这一问题，而非尝试修复LLM本身
- **vs Ensemble方法**: 传统模型集成在输出层投票，本文在概率分布层动态混合，粒度更细、更灵活

## 评分

- 新颖性: ⭐⭐⭐⭐ 概率分布层面的动态混合策略新颖且直觉明确
- 实验充分度: ⭐⭐⭐⭐ 多数据集+详细消融+领域适配实验
- 写作质量: ⭐⭐⭐⭐ 问题分析到位，方法阐述清晰
- 价值: ⭐⭐⭐⭐ 实用价值高，工业可部署，思路可推广

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[ACL 2025\] RMoA: Optimizing Mixture-of-Agents through Diversity Maximization and Residual Compensation](rmoa_optimizing_mixture-of-agents_through_diversity_maximization_and_residual_co.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](../../ACL2026/others/are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ACL 2025\] AnalyticKWS: Towards Exemplar-Free Analytic Class Incremental Learning for Small-footprint Keyword Spotting](analytickws_towards_exemplar-free_analytic_class_incremental_learning_for_small-.md)
- [\[ACL 2025\] The Hidden Attention of Mamba Models](the_hidden_attention_of_mamba_models.md)

<!-- RELATED:END -->
