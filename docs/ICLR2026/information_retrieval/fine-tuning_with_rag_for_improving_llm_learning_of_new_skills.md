---
title: >-
  [论文解读] Fine-tuning with RAG for Improving LLM Learning of New Skills
description: >-
  [ICLR 2026][RAG distillation] 提出将 RAG 从推理时的永久依赖转化为训练时的教师信号：从 agent 失败中提取 hint、用 hint 增强的教师生成更优轨迹、然后移除 hint 蒸馏到学生模型，使学生内化检索增益而无需运行时 RAG，在 ALFWorld 达到 91% 成功率（基线 79%），WebShop 分数达 72（基线 61）。
tags:
  - ICLR 2026
  - RAG distillation
  - LLM agent
  - hint extraction
  - ALFWorld
  - WebShop
---

# Fine-tuning with RAG for Improving LLM Learning of New Skills

**会议**: ICLR 2026  
**arXiv**: [2510.01375](https://arxiv.org/abs/2510.01375)  
**代码**: [匿名仓库](https://anonymous.4open.science/r/anonymized-submission-iclr/README.md)  
**领域**: 模型压缩/知识蒸馏  
**关键词**: RAG distillation, LLM agent, hint extraction, ALFWorld, WebShop

## 一句话总结

提出将 RAG 从推理时的永久依赖转化为训练时的教师信号：从 agent 失败中提取 hint、用 hint 增强的教师生成更优轨迹、然后移除 hint 蒸馏到学生模型，使学生内化检索增益而无需运行时 RAG，在 ALFWorld 达到 91% 成功率（基线 79%），WebShop 分数达 72（基线 61）。

## 研究背景与动机

LLM agent 在多步任务中经常以可预测的方式失败：执行前置条件未满足的动作、发出冗余指令、或错误处理环境约束。现有改进方法各有局限：

**结构化 prompting**（ReAct、StateAct）：提供推理脚手架但受限于参数化知识

**自我反思**（Reflexion）：需要多次尝试和真实反馈

**检索增强**（RAG）：注入外部知识但增加运行时开销和部署复杂度

**微调**：需要大量高质量数据，且可能过拟合

核心洞察：**RAG 不需要作为永久的运行时依赖存在**。它可以作为改进训练监督信号的来源，被内化到模型参数中。具体来说，如果我们能用 RAG 生成更好的演示轨迹，然后用这些轨迹训练学生模型（但不提供 hints），学生就能学会 RAG 带来的行为改进，同时在推理时不再需要检索。

## 方法详解

### 整体框架

四阶段 pipeline：

**Stage A - 基础 Agent 采集轨迹**: 部署基础 agent（ReAct 或 StateAct）在训练集上运行，收集成功和失败的轨迹。成功轨迹构成 baseline SFT 数据集；失败轨迹用于 hint 提取。

**Stage B - 自动 Hint 提取**: 对每条失败轨迹，构造包含任务指令、初始观察、完整动作序列和结果的完整失败示例，然后用 GPT-4o 生成 1-4 条祈使句式的 hint。Hint 使用占位符（{object}、{container}）保证泛化性，并按任务类别分类存储。

**Stage C - 教师数据生成**: 给定指令和初始观察，确定任务类别，从对应类别的 hint bank 中检索 top-k（k=3）条 hint，通过量化 Qwen-2.5 7B 做 LLM re-ranking。Hint 在 episode 开始时一次性注入（t=0），教师 agent 带着 hint 运行完整 episode，仅保留成功轨迹。

**Stage D - 蒸馏训练**: 从教师轨迹中**移除 hint 字符串和 few-shot 示例**，构造蒸馏数据集。训练学生模型的 LoRA adapter，强制模型内化 hint 所传达的行为指导。

### 关键设计

1. **失败驱动的 Hint 提取**: 

    - 不需要专家监督，从 agent 自身的失败中学习
    - GPT-4o 诊断失败原因并生成修正规则
    - Hint 示例："确保在放置 {object} 之前先打开 {container}"、"使用系统化搜索模式避免遗漏 {object}"
    - 经模糊匹配去重（Levenshtein distance 阈值 0.85）
    - ALFWorld 生成 760/650 条 ReAct/StateAct hints，WebShop 生成 756/831 条

2. **一次性检索（One-shot Retrieval）**: 

    - 仅在 episode 开头 t=0 检索一次 hint，不在执行过程中动态检索
    - 限制 token 开销同时反映实际场景：指导只在任务开始时提供一次
    - 使用 LLM re-ranking 而非传统 embedding 检索，排序质量更高

3. **Hint 移除蒸馏**: 

    - 训练数据来自带 hint 的教师轨迹，但输入中移除了 hint 文本
    - 同时移除 few-shot 示例（因为它们是跨任务固定的，不提供有用训练信号）
    - 这迫使学生学习"行为"而非"文本模式"，实现真正的内化

### 损失函数 / 训练策略

- 训练目标：全序列 next-token 交叉熵损失
- 使用 QLoRA 风格：backbone 量化到 4-bit，LoRA adapter 在 bf16 精度训练
- ALFWorld：LR 2e-4，序列长度 1024，LoRA rank 64，α=128，dropout 0.10，weight decay 0.01
- WebShop：LoRA rank 16，α=32，dropout 0.20，weight decay 0.05
- WebShop 使用 token-level label smoothing（ε=0.1）缓解短轨迹上的过度自信
- 优化器：8-bit AdamW，linear schedule，batch size 2 × 4 gradient accumulation
- 单 epoch 训练，10% warmup，单张 A100 80GB

## 实验关键数据

### 主实验

Qwen-2.5 14B Instruct，ReAct 和 StateAct 的平均结果：

| 方法 | ALFWorld 成功率 | WebShop 成功率 | WebShop 分数 |
|------|----------------|---------------|-------------|
| Base | 79.85% | 38.5% | 60.87 |
| Base+RAG | 82.09% | 43.5% | 67.08 |
| SFT | 85.45% | 43.0% | 72.09 |
| **Distilled (本文)** | **91.04%** | **43.5%** | **72.40** |

Qwen-2.5 7B Instruct：

| 方法 | ALFWorld 成功率 | WebShop 成功率 | WebShop 分数 |
|------|----------------|---------------|-------------|
| Base | 26.49% | 13.0% | 28.12 |
| Base+RAG | 71.27% | 8.5% | 18.46 |
| SFT | 62.69% | 22.0% | 54.38 |
| **Distilled (本文)** | **73.88%** | **22.5%** | **61.04** |

### 效率分析（14B）

| 环境 | 方法 | Token/episode | Steps | 性能 |
|------|------|-------------|-------|------|
| ALFWorld | Base | 50.13k | 18.94 | 79.85% |
| | RAG | 53.97k | 18.69 | 82.09% |
| | Distilled | **44.82k** | **16.68** | **91.04%** |
| WebShop | Base | 7.99k | 7.16 | 60.87 |
| | RAG | 11.05k | 6.34 | 67.08 |
| | Distilled | **4.27k** | **4.98** | **72.40** |

蒸馏模型在 ALFWorld 省 10% token，WebShop 省 47% token，同时性能最优。

### 消融实验

Retrieval depth k 消融（ALFWorld 14B）：

| k | 成功率 | Steps | Tokens |
|---|--------|-------|--------|
| 1 | 83.96% | 19.11 | 52.02k |
| 3 (本文) | 82.09% | 18.69 | 53.97k |
| 6 | 84.33% | 18.13 | 50.95k |
| 9 | 76.87% | 19.27 | 57.26k |

k=3 在两个环境中均衡最优；k=9 时 hint 过多反而有害。

### 关键发现

- 蒸馏模型完全占据了精度-效率 Pareto 前沿
- 7B 蒸馏后的 WebShop 分数（61.04）接近 14B Base（60.87），实现了跨规模压缩
- 小模型（7B）在 RAG 模式下 WebShop 反而变差（hint 会误导小模型做出错误属性选择），但蒸馏可以稳定利用复杂指导
- SFT 和蒸馏的差距验证了 hint 增强轨迹确实包含了额外的行为知识

## 亮点与洞察

1. **"将运行时增强转化为训练时监督"的范式**: 这个思路对 RAG 之外的很多增强方法（如 CoT、self-critique）都适用
2. **失败驱动的自动化**: 整个 pipeline 不需要人工专家知识，从自身失败中提炼可复用的指导规则
3. **Hint 移除是关键**: 训练时有 hint 而推理时无 hint，迫使模型将显式规则内化为隐式知识
4. **效率分析很全面**: 不只看准确率，还系统分析了 token 开销、步数等效率指标
5. **跨 agent 架构验证**: 在 ReAct 和 StateAct 两种不同架构上都有效，说明方法具有通用性

## 局限与展望

1. **Hint 生成依赖 GPT-4o**: 在大规模环境中 API 调用成本不可忽视
2. **仅 t=0 一次性检索**: 无法适应 episode 中途出现的意外情况
3. **单种子评估**: 所有结果为点估计，缺乏多种子的方差分析
4. **跨域迁移未验证**: 仅在 ALFWorld 和 WebShop 上测试，新环境的泛化性未知
5. **Hint 质量上界**: 如果 GPT-4o 生成的 hint 本身质量有限（如对复杂失败的诊断不准确），性能天花板会受限

## 相关工作与启发

- **ReAct / StateAct**: 基础 agent prompting 框架，本文在此基础上增加了 hint 增强和蒸馏
- **Reflexion**: 同为从失败中学习，但需要多次尝试；本文只需单次训练
- **ExpeL / AutoGuide**: 从经验中提取知识但作为永久运行时依赖，本文将其蒸馏到参数中
- **FireAct**: 类似的微调思路但依赖 GPT-4 专家轨迹；本文用自身失败+自提取 hint 生成教师数据
- **Prompt Distillation**: 压缩复杂 prompt 到模型权重中，本文扩展到 agent 场景的动态指导蒸馏

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/information_retrieval/vladva_discriminative_fine-tuning_of_lvlms.md)
- [Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)
- [LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference](lightretriever_a_llm-based_text_retrieval_architecture_with_extremely_faster_que.md)
- [Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement](judges_verdict_a_comprehensive_analysis_of_llm_judge_capability_through_human_ag.md)
- [Revela: Dense Retriever Learning via Language Modeling](revela_dense_retriever_learning_via_language_modeling.md)

<!-- RELATED:END -->
