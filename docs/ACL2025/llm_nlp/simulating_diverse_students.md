---
title: >-
  [论文解读] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents
description: >-
  [ACL 2025][LLM/NLP][student simulation] 针对 LLM 难以模拟低水平学生犯错行为的问题，提出基于知识图谱认知原型的 training-free 框架，通过认知状态建模 → 行为预测 → beam search 自精炼三阶段生成逼真的学生解答…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "student simulation"
  - "cognitive prototype"
  - "knowledge graph"
  - "beam search refinement"
  - "LLM agent"
---

# Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents

**会议**: ACL 2025  
**arXiv**: [2505.19997](https://arxiv.org/abs/2505.19997)  
**代码**: [https://mccartney01.github.io/student_sim](https://mccartney01.github.io/student_sim)  
**领域**: LLM NLP / 教育AI  
**关键词**: student simulation, cognitive prototype, knowledge graph, beam search refinement, LLM agent

## 一句话总结
针对 LLM 难以模拟低水平学生犯错行为的问题，提出基于知识图谱认知原型的 training-free 框架，通过认知状态建模 → 行为预测 → beam search 自精炼三阶段生成逼真的学生解答，在 Student_100 数据集上模拟准确率提升 100%。

## 研究背景与动机

**领域现状**：LLM 在教育场景中广泛应用于个性化辅导、课程设计和自适应评估。LLM-based agent 模拟学生行为是评估教学策略、测试智能辅导系统的关键手段。

**现有痛点**：当前 LLM 本质上被训练为"有用的助手"，倾向于生成正确答案。当要求模拟低认知水平学生时，它们仍然输出过于"高级"的响应，无法复现学生学习中的自然错误和不完美表现。实验显示，对15个不同水平学生的 naive prompt 模拟中，LLM 系统性高估了弱能力学生的认知分数。

**核心矛盾**：LLM 的训练目标（生成正确、helpful 的回答）与模拟"犯错"学生的需求根本冲突。而 fine-tuning 方法（如 MalAlgoPy）虽能引入错误模式，但会污染模型本身的知识，且忽略了错误的个性化特征——不同学生的错误模式应由其认知状态决定。

**本文目标** (1) 如何不修改模型权重就能让 LLM 模拟不同认知水平的学生？(2) 如何精确预测特定学生在新任务上会犯什么错？(3) 如何生成既包含正确解答又包含合理错误的逼真学生答案？

**切入角度**：从认知科学出发，用知识图谱显式建模学生对各知识概念的掌握程度（认知原型），而非依赖隐式神经网络表征。知识图谱的自然语言节点天然适合与 LLM 交互，无需训练即可实现精准的行为预测。

**核心 idea**：用知识图谱构建学生认知原型来预测行为，再用 beam search 自精炼生成与预测行为一致的解答，实现 training-free 的多层次学生模拟。

## 方法详解

### 整体框架
输入是学生的 $M=40$ 条历史学习记录 $P=\{(t_i, b_i, s_i)\}$（任务、行为、解答），输出是对 $N=10$ 个新任务的模拟解答。三阶段 pipeline：(1) 从历史记录构建知识图谱认知原型；(2) 将原型映射到新任务预测行为；(3) 基于预测行为通过 beam search 自精炼生成解答。

### 关键设计

1. **认知原型构建（Knowledge Graph-based Cognitive Prototype）**:

    - 功能：从学生历史记录构建显式可解释的认知状态表示
    - 核心思路：对每条学习记录迭代执行四步——(a) 概念提取：先生成任务高层描述 $d_i = \pi_{desc}(t_i)$ 识别高级概念，再结合记录提取多层次知识概念；(b) 关系提取：在概念间建立 Prerequisite_of / Used_for / Hyponym_of / Part_of 四种关系作为图的边；(c) 局部认知分析：对每个概念根据学生表现判定掌握程度为 Good / Bad；(d) 全局原型构建：处理完所有记录后，综合每个概念的历史 Good/Bad 频次生成全局认知状态总结
    - 设计动机：比起隐式神经网络，自然语言知识图谱可解释性强，且能直接作为 LLM prompt 的上下文，无需训练

2. **概念感知行为预测（Concept-aware Behavior Prediction）**:

    - 功能：预测学生在新任务上的期望行为（正确/犯错及具体错误类型）
    - 核心思路：对新任务生成概念描述，与知识图谱节点计算相似度，选取 top-$p=5$ 个最相关概念及其认知状态，找到包含最多这些概念的历史记录作为参考，综合预测行为 $\hat{b}_j = \pi_{pred}(t_j, [C_1,...,C_p], P_{\hat{j}})$
    - 设计动机：传统方法按任务文本相似度检索，容易因表面相似而检索到需要完全不同知识概念的任务（如"算阶乘"检索到"算两倍"），基于概念映射则能深入知识层面匹配

3. **Beam Search 自精炼（Self-Refinement with Beam Search）**:

    - 功能：生成与预测行为精确对齐的学生解答
    - 核心思路：初始生成弱解答 $\hat{s}_j^1$，迭代 $L=3$ 轮，每轮采样 $B=2$ 个候选，由 $\pi_{value}$ 评分（0-1 打分行为对齐度），选最高分候选进入下轮，直到分数超阈值 $\delta=0.9$ 或达到最大迭代
    - 设计动机：LLM 单次生成难以准确模拟特定错误模式，迭代精炼+自评估能逐步修正偏差；beam search 多候选采样提高找到合适解答的概率

### 训练策略
整个框架 training-free，所有组件（$\pi_{desc}$, $\pi_{node}$, $\pi_{edge}$, $\pi_{local}$, $\pi_{global}$, $\pi_{pred}$, $\pi_{refine}$, $\pi_{value}$）使用同一个 LLM，temperature=0 保证可复现性，仅 $\pi_{refine}$ 开启多样采样。

## 实验关键数据

### 主实验
在 Student_100 数据集（100名学生、5000条 Python 编程学习记录）上评测，指标为行为预测准确率 Acc、行为一致性 Con1、解答一致性 Con2（1-5分）：

| 模型 | 方法 | Acc | Con1 | Con2 |
|------|------|-----|------|------|
| GPT-4o | Similarity + IO | 0.47 | 2.62 | 2.65 |
| GPT-4o | Prototype Mapping + Refine（本文） | **0.94** | **3.77** | **3.65** |
| Claude-3.5 | Similarity + IO | 0.61 | 3.03 | 2.82 |
| Claude-3.5 | Prototype Mapping + Refine（本文） | **0.65** | **3.09** | **3.09** |
| LLaMA-3.3-70B | Similarity + Refine | 0.41 | 2.45 | 1.99 |
| LLaMA-3.3-70B | Prototype Mapping + Refine（本文） | **0.61** | **2.99** | **2.69** |

### 消融实验

| 配置 | GPT-4o Acc | GPT-4o Con2 | 说明 |
|------|-----------|-------------|------|
| Full model | 0.94 | 3.65 | 完整框架 |
| w/o 知识图谱（仅文本相似检索） | 0.47 | 2.65 | Acc 下降 50%，认知原型关键 |
| w/o 全局认知构建（仅局部状态） | 0.66 | 2.89 | 全局综合必要 |
| w/o 行为预测（直接生成解答） | - | 2.70 | 无行为描述引导，解答质量差 |
| w/o 自精炼（仅 IO） | 0.94 | 3.50 | 精炼贡献 +0.15 |
| w/o 自评估（仅精炼无评分） | 0.94 | 3.52 | 自评估引导方向重要 |

### 关键发现
- GPT-4o 的行为预测准确率从 baseline 的 0.47 跃升到 0.94（+100%），验证认知原型的核心价值
- 更强的 LLM 从自精炼中获益更多（GPT-4o: +0.15 vs LLaMA: +0.08），因其自评估能力更强
- 模拟高认知水平学生比低水平学生更容易（Con2 与认知分数正相关），因为正确解答比模拟特定错误更简单
- 历史记录从 10 增加到 40 条，Acc 从 ~0.7 稳步提升至 0.94，表明更多数据能构建更精确的认知原型

## 亮点与洞察
- 知识图谱 + LLM 的原型构建是个巧妙的"不改模型、改输入"思路，自然语言图谱节点无缝嵌入 prompt，实现 training-free 个性化模拟，可推广到任何需要个性化行为建模的场景
- Beam search 自精炼不追求"生成更好的答案"而是"生成更差但合理的答案"，这个反向优化目标的设计很新颖
- 概念级行为预测替代文本相似度检索，本质是"理解学生不会什么"而非"找相似的题"，这个范式转变可迁移到智能辅导系统的个性化出题

## 局限与展望
- 仅在 Python 编程领域验证，数学等需要不同推理链的学科可能需要调整概念提取策略
- 框架依赖多次 LLM 调用（8个组件），计算成本高，100个学生的实验需大量 API 调用
- 知识图谱的概念粒度由 LLM 决定，缺少统一的知识本体约束，可能影响跨学生的可比性
- 未考虑多模态信号（如学生操作行为、编码时间模式），仅基于文本行为建模

## 相关工作与启发
- **vs MalAlgoPy**: MalAlgoPy 定义 20 种方程变换错误并 fine-tune，但会污染模型知识且非个性化；本文 training-free 且按学生认知状态自适应引入错误
- **vs 知识追踪方法（DKT等）**: 传统方法用隐式神经网络参数化知识状态，不可解释；本文用自然语言知识图谱显式建模，可直接与 LLM 交互
- **vs Level-based 方法**: 仅用历史正确率估计学生水平过于粗糙，忽略了错误的概念特异性；认知原型能细粒度到每个知识概念

## 评分
- 新颖性: ⭐⭐⭐⭐ 将认知原型与 LLM 结合的 training-free 框架思路新颖，但 beam search 自精炼不算全新
- 实验充分度: ⭐⭐⭐⭐ 4个 LLM、18种配置组合、消融+参数分析+人工评估，且扩展到 Java/C++
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图示直观，方法描述详细
- 价值: ⭐⭐⭐⭐ 对教育 AI 中个性化模拟有直接实用价值，framework 可泛化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility](leveraging_human_production-interpretation_asymmetries_to_test_llm_cognitive_pla.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)

</div>

<!-- RELATED:END -->
