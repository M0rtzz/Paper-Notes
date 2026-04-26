---
title: >-
  [论文解读] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models
description: >-
  [ACL 2025][复杂常识推理] 提出 Com2，一个基于因果事件图和因果理论（干预/反事实）构建的复杂常识推理基准，包含 2500 道主题和 1254 道侦探故事题目，揭示 LLM 在推理深度与广度上的显著不足。
tags:
  - ACL 2025
  - 复杂常识推理
  - 因果图
  - 因果理论
  - 干预
  - 反事实
  - benchmark
---

# Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models

**会议**: ACL 2025  
**arXiv**: [2506.07064](https://arxiv.org/abs/2506.07064)  
**代码**: [GitHub](https://github.com/HIT-SCIR/Com2)  
**领域**: LLM Evaluation / Commonsense Reasoning  
**关键词**: 复杂常识推理, 因果图, 因果理论, 干预, 反事实, benchmark  

## 一句话总结

提出 Com2，一个基于因果事件图和因果理论（干预/反事实）构建的复杂常识推理基准，包含 2500 道主题和 1254 道侦探故事题目，揭示 LLM 在推理深度与广度上的显著不足。

## 研究背景与动机

- **问题定义**: 现有常识推理基准（如 CommonsenseQA）大多是单步推理——只要知道相关知识就能回答。但现实世界中人们更关注复杂的多步常识推理，如某事件的长期影响、突发事件的后果、反事实假设等场景。
- **现有方法局限**: (1) 数学和代码领域的复杂推理已有充分研究 (AIME、MATH)，但复杂常识推理因缺乏结构化表达和明确的 ground truth 而被忽视；(2) 现有推理 LLM (o1, R1) 的 test-time scaling 策略主要在数学/代码任务上验证，对常识推理的效果未知。
- **核心动机**: LLM 通过预训练掌握了大量简单显式知识，但面对从简单知识推导出的复杂隐式知识（如事件的长期连锁反应、反直觉场景）时表现如何？需要一个系统化的基准来回答这一问题。
- **核心挑战**: 常识知识的表达是非形式化、上下文依赖的，且通常缺乏公认的标准答案，使得构建高质量数据集困难。

## 方法详解

### 整体框架

Com2 的构建分为四步流水线：(1) **事件提议** (Event Proposal)——使用 LLM 生成具体事件和抽象事件作为种子；(2) **因果链提议** (Causal Chain Proposal)——以事件为根构建 5 节点因果链表示简单场景；(3) **因果图提议** (Causal Graph Proposal)——利用因果理论（干预、反事实等）修改因果链生成复杂场景；(4) **Com2 合成** (Com2 Synthesis)——基于因果图生成多选/多答题。

### 关键设计

1. **五种因果图场景对应五类推理任务**:
    - **Direct**: 因果链直接推理，问事件长期结果（最简单）
    - **Decision**: 双分支因果图，问如何预防不良结果（多选题）
    - **Transition**: 含因果传递问题（如场景漂移）的因果链，测试推理深度增加时可靠性
    - **Intervention**: 加入外部突发事件打断原因果链，测试模型对非常见场景的推理能力
    - **Counterfactual**: 对已发生的具体事件构建反事实假设，问"如果 X 没发生会怎样"

2. **Com2-hard 子集**: 基于 400+ 侦探故事 (BMDS)，构建多线索交织的复杂推理场景，包含 Decision、Intervention、Counterfactual 三种更高难度任务。

3. **慢思考 (Slow Thinking) 引导**: 每个样本配套系统分析、分治策略、自我修正和上下文识别等思维步骤，可作为辅助提示验证 LLM 的推理能力。

### 损失函数

无模型训练，属于 benchmark 工作。评估使用准确率 (Accuracy)；多选题 Decision 使用软 (soft) 评分策略——按正确预测选项比例计分。

## 实验

### 主实验：LLM 在 Com2 上的表现

| 模型 | Direct | Decision | Transition | Intervention | Counter. | Main 平均 | Hard 平均 | 总分 |
|------|--------|----------|------------|-------------|----------|----------|----------|------|
| Qwen2.5-32B | 83.60 | 65.16 | 48.80 | 33.80 | 72.40 | 60.73 | 54.80 | 57.77 |
| GPT-4o | 80.60 | 66.43 | 48.40 | 32.20 | 68.80 | 59.26 | 59.72 | 59.49 |
| GPT-4o-mini | 83.20 | 62.54 | 49.20 | 31.40 | 71.20 | 59.50 | 55.29 | 57.40 |
| LLaMA-3.1-8B | 83.20 | 58.04 | 47.00 | 30.40 | 71.40 | 58.01 | 53.56 | 55.79 |
| R1-distilled | 75.20 | 56.51 | 43.40 | 30.00 | 68.20 | 54.65 | 62.70 | 58.68 |
| QwQ-32B | 79.80 | 59.82 | 47.40 | 32.00 | 64.60 | 56.70 | 52.01 | 54.36 |
| o1-mini | 80.00 | 32.64 | 47.80 | 30.00 | 66.60 | 51.48 | 56.54 | 54.01 |

所有 LLM 在 Intervention 任务上表现最差（约 30%），揭示了推理广度的严重不足。

### 消融实验：Post-training 效果

| 模型 | Main 平均 (训练前) | Main 平均 (训练后) | Hard 平均 (训练前) | Hard 平均 (训练后) |
|------|-------------------|-------------------|-------------------|-------------------|
| LLaMA-3.1-8B | 58.01 | ~68 (显著提升) | 53.56 | ~58 (OOD 仍有提升) |
| Qwen2-7B | 58.13 | ~66 (显著提升) | 54.71 | ~57 |

Post-training 在 Main 上提升明显但在 Hard (OOD) 上提升有限，说明从简单任务学到的推理能力可部分迁移。

### 关键发现

- **Counterfactual 并非最难**: 按因果理论，反事实应该最难，但 LLM 在此任务上表现反而好于 Transition 和 Intervention，说明预训练赋予了 LLM 较好的假设推理能力。
- **Intervention 是最大瓶颈**: 平均仅 ~31%，说明 LLM 处理突发/非常见场景的推理广度严重不足。
- **推理 LLM 不一定更好**: 在 Com2-main 上，o1-mini 和 QwQ 反而不如通用 LLM，可能由于"过度思考" (overthinking) 导致在常识场景中陷入思维陷阱。
- **Test-time scaling 对常识推理低效**: 输出更多 token 并不一定提升性能，与数学/代码领域的 scaling law 形成对比。
- **慢思考 (Slow Thinking) 有效**: 提供引导式思维过程后，LLM 准确率显著提升，表明结构化推理引导可弥补模型缺陷。

## 亮点

- 首次系统性地利用因果事件图 + 因果理论构建复杂常识推理基准，5 种任务类型精确对应人们关心的实际场景。
- 揭示了 LLM 在复杂常识推理中的关键缺陷：推理广度不足（无法处理突发事件）和推理深度不稳定（因果传递过程中可靠性下降）。
- Com2-hard 基于侦探故事构建，场景更自然且极具挑战性，是优秀的分布外泛化测试集。
- 全面涵盖 10+ 个 LLM（通用 + 推理型），实验设计详尽。

## 局限性

- 数据合成过程依赖 ChatGPT (gpt-4o-mini)，虽有人工评估验证质量，但可能存在系统性偏差。
- 仅使用选择题/多选题格式，未涵盖开放式生成任务。
- Com2-hard 基于侦探故事构建，LLM 可能在预训练中已接触过类似故事，导致表现虚高。
- 因果图构建过程可以更精细化和步骤化，当前的 prompt-based 方法可能不够严谨。

## 相关工作

- **常识推理基准**: CommonsenseQA (Talmor et al., 2019)、OpenBookQA (Mihaylov et al., 2018) 等关注单步推理；本文扩展到多步复杂推理。
- **因果推理**: CausalNet (Luo et al., 2016)、GLUCOSE (Mostafazadeh et al., 2020) 等研究因果关系提取；本文将因果理论 (Pearl) 应用于基准构建。
- **复杂推理 LLM**: o1 (OpenAI, 2024)、DeepSeek-R1 (Liu et al., 2024) 等通过 test-time compute 提升推理能力，本文揭示其在常识域的局限性。

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 8 |
| 实验充分性 | 9 |
| 论文清晰度 | 7 |
| 实用性 | 7 |
| **总分** | **7.8** |

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](../../NeurIPS2025/llm_evaluation/can_large_language_models_master_complex_card_games.md)

<!-- RELATED:END -->
