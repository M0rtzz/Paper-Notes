---
title: >-
  [论文解读] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering
description: >-
  [ACL2025][NLP理解][多跳问答] 提出 SiGIR 框架，通过端到端训练使模型具备迭代问题分解、检索、推理和自我评估能力，在推理阶段利用自我批评反馈引导 iteration-level beam search 选择最优推理路径，在三个多跳 QA 数据集上平均超越 SOTA 8.6%。
tags:
  - ACL2025
  - NLP理解
  - 多跳问答
  - 检索增强生成
  - 自我批评
  - 迭代推理
  - 束搜索
---

# Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering

**会议**: ACL2025  
**arXiv**: [2505.19112](https://arxiv.org/abs/2505.19112)  
**代码**: [zchuz/SiGIR-MHQA](https://github.com/zchuz/SiGIR-MHQA)  
**领域**: nlp_understanding  
**关键词**: 多跳问答, 检索增强生成, 自我批评, 迭代推理, 束搜索

## 一句话总结

提出 SiGIR 框架，通过端到端训练使模型具备迭代问题分解、检索、推理和自我评估能力，在推理阶段利用自我批评反馈引导 iteration-level beam search 选择最优推理路径，在三个多跳 QA 数据集上平均超越 SOTA 8.6%。

## 研究背景与动机

大语言模型在知识密集型多跳推理任务中面临三大挑战：(1) 一次性将复杂问题分解为子问题难度大，初始错误会导致后续推理偏离正确路径；(2) 迭代检索方法难以处理复杂问题规划，无法清晰表达检索意图，导致检索不准确；(3) 中间步骤缺乏指导反馈，容易产生级联错误。现有方法（如 IRCoT、Self-RAG、Auto-RAG）虽部分缓解了这些问题，但仍未能有效结合中间步骤的质量评估来引导推理过程。本文聚焦两个关键方面：迭代问题分解促进精确检索；中间步骤反馈引导推理、扩展推理空间并减少错误传播。

## 方法详解

### 整体框架

SiGIR (Self-Critique Guided Iterative Reasoning) 包含训练和推理两个阶段：

**训练阶段**分三步构建 SC-Reasoner (Self-Critique Reasoner)：

1. **Iterative Reasoner (R)**：使用 DeepSeek-V2.5 合成迭代推理轨迹（包含子问题分解、检索触发、知识推理），以交错格式组织数据训练小模型获得基础推理器
2. **Critic Model (C)**：从非重叠语料中使用 R（而非 LLM）生成推理轨迹以确保正负样本平衡，再由 LLM 评估检索质量、推理质量和整体质量，训练独立的 critic 模型
3. **SC-Reasoner (R_sc)**：用 C 对推理轨迹的子过程标注奖励信号，在特殊 token 后添加自评分，最终端到端训练得到同时具备推理和自我评估能力的模型

### SC-Reasoner 推理模式

模型在每一步中执行五个操作：(1) 判断问题是否需要分解；(2) 一次分解一个原子子问题；(3) 触发外部检索并进行知识推理；(4) 自评检索和推理质量（生成 r_retr 和 r_reas）；(5) 对原始问题进行 reduction，识别未解决部分。重复该过程直至最终答案。

### 自我批评引导的迭代推理（Iteration-level Beam Search）

推理阶段采用 iteration-level beam search：

- **分支探索**：在每个推理步，通过温度采样对子问题分解、检索和推理进行分支扩展，生成多条候选推理路径
- **候选选择**：使用累积过程奖励（r_c = r_retr + r_reas 逐步累加）对候选路径评分，保留 top-k 最优候选
- **最终选择**：推理结束后，使用累积过程奖励或结果奖励选择最终答案轨迹

### 自改进机制

SC-Reasoner 可在无标注数据上进行推理并自评质量，保留高质量轨迹用于后续训练，实现数据稀缺场景下的自我改进。

## 实验关键数据

### 主实验结果 (Table 1)

在三个多跳 QA 数据集上的 F1 结果（Mistral-7B 为默认骨干）：

| 方法 | 2WikiMQA | HotpotQA | MuSiQue |
|------|----------|----------|---------|
| IRCoT | 46.80 | 49.03 | 21.43 |
| Self-RAG | 31.60 | 54.60 | 22.00 |
| Auto-RAG | 56.06 | 50.03 | 22.08 |
| DR-Distillation | 70.51 | 58.06 | 22.74 |
| **SiGIR (Ours)** | **74.47** | **63.09** | **37.15** |

SiGIR 在 MuSiQue（3-4 跳困难问题）上提升最为显著：3-hop +47.1%、4-hop +24.37%。在 Qwen2.5 和 LLaMA2 上也取得了有竞争力的结果。

### 消融实验 (Table 2)

| 设置 | 2WikiMQA | HotpotQA | MuSiQue |
|------|----------|----------|---------|
| SiGIR 完整 | 74.47 | 63.09 | 37.15 |
| 去掉引导搜索 | 72.37 | 60.86 | 35.47 |
| 去掉奖励信号 | 67.23 | 54.94 | 29.45 |
| 去掉自我批评 | 66.96 | 56.86 | 28.62 |
| 粗粒度奖励 | 73.40 | 59.85 | 34.32 |

去掉奖励信号导致约 10% 性能下降；细粒度奖励在搜索式推理中比粗粒度更有效。

### 搜索与选择策略分析 (Table 4)

| 策略 | 平均 F1 |
|------|---------|
| 引导搜索 (完整) | 58.23 |
| 仅保留 1 条候选 | 56.81 |
| 不分支子问题 | 50.03 |
| 累积过程奖励选择 | 58.23 |
| 结果奖励选择 | 55.99 |
| 随机选择 | 55.81 |

子问题探索宽度设为 1 时性能下降 13.57%，说明多样化子问题探索的重要性。

## 亮点

- **统一框架**：将推理与自我批评统一到单一模型中，避免了分离式 generator-critic 的推理开销（吞吐量提升 4.46 倍）
- **迭代分解 + beam search**：避免一次性分解的高风险，通过逐步分解+搜索降低错误传播
- **自改进能力**：仅用 40% 标注数据 + 两轮自改进，模型性能接近 100% 数据训练的结果（仅差 2.2%）
- **检索系统灵活性**：方法天然支持混合检索（稀疏+稠密），混合检索带来平均 5.7% 的提升

## 局限性

- 子问题探索采用采样方式而非专用分解模块，可能限制探索广度
- 仅采用 SFT 训练路线，未引入强化学习优化推理轨迹
- beam search 增加了推理时的计算开销（虽然通过 GenCritic 部分缓解）
- 实验仅在英文多跳 QA 数据集上验证，跨语言和跨领域泛化性未知

## 相关工作

- **检索增强生成 (RAG)**：从单轮检索到多轮迭代检索（IRCoT, Self-RAG），SiGIR 在此基础上加入了自我批评信号引导检索方向
- **多跳问答**：问题分解方法（ProbTree, BeamAggR）vs 迭代推理方法（Auto-RAG, DR-Distillation），SiGIR 结合迭代分解与奖励引导搜索
- **推理时缩放**：采样集成（Self-Consistency）、MCTS 搜索，SiGIR 的 iteration-level beam search 平衡了性能与效率

## 评分

- 新颖性: ⭐⭐⭐⭐ 将自我批评与迭代推理统一为 beam search 的思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集+四种骨干+丰富消融分析+检索系统实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对知识密集型多跳推理有实用参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)
- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)

<!-- RELATED:END -->
