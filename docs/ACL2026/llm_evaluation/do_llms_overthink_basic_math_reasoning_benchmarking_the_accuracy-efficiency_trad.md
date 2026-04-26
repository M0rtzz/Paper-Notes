---
title: >-
  [论文解读] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff
description: >-
  [ACL 2026][过度思考] 本文提出 LLMThinkBench，一个系统性评估 LLM 基础数学推理效率的基准，引入 Overthinking Score（准确率和 token 效率的调和平均），通过动态生成的 14 个确定性数学任务评估 53 个 LLM，发现推理模型平均生成约 18× 更多 token 但有时准确率更低，且扩展推理预算呈现收益递减。
tags:
  - ACL 2026
  - 过度思考
  - 基础数学推理
  - 准确率-效率权衡
  - 推理token
  - 基准测试
---

# Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff

**会议**: ACL 2026  
**arXiv**: [2507.04023](https://arxiv.org/abs/2507.04023)  
**代码**: [GitHub](https://github.com/ctrl-gaurav/LLMThinkBench)  
**领域**: LLM 效率 / 推理评估  
**关键词**: 过度思考, 基础数学推理, 准确率-效率权衡, 推理token, 基准测试

## 一句话总结

本文提出 LLMThinkBench，一个系统性评估 LLM 基础数学推理效率的基准，引入 Overthinking Score（准确率和 token 效率的调和平均），通过动态生成的 14 个确定性数学任务评估 53 个 LLM，发现推理模型平均生成约 18× 更多 token 但有时准确率更低，且扩展推理预算呈现收益递减。

## 研究背景与动机

**领域现状**：LLM 在复杂数学基准（GSM8K、MATH）上表现出色，推理模型通过推理时扩展（chain-of-thought）进一步提升了性能。然而，这些模型在基础数学运算上的表现和效率尚未系统评估。

**现有痛点**：(1) 在复杂基准上得分 90%+ 的模型在基础加法上可能低于 40%——复杂基准性能无法迁移到基础运算；(2) 推理模型生成过长的推理链来解决简单问题（如 234+567 生成数百 token 解释进位原理），不仅浪费计算资源，有时还降低准确率；(3) 现有评估仅关注准确率，忽略了计算浪费；(4) 静态基准存在数据污染风险；(5) 缺乏联合衡量准确率和效率的指标。

**核心矛盾**：推理模型被训练为"思考更多"以提升性能，但在基础任务上更多思考反而有害——模型将解释（explanation）与理解（understanding）混淆，产生的长文本表面上像推理但实际上不具备问题解决能力。

**本文目标**：(1) 形式化准确率-冗余度权衡；(2) 提出 Overthinking Score 指标；(3) 建立动态生成的评估协议；(4) 大规模实证研究 53 个 LLM 的推理效率。

**切入角度**：聚焦于 14 个确定性基础数学任务（排序、求和、乘法、找最大值等），这些任务有唯一正确答案且计算复杂度已知，可以精确衡量准确率和冗余度之间的关系。

**核心 idea**：更多推理 token ≠ 更好的数学推理——在基础任务上，推理模型的冗余生成不仅浪费计算，还可能因错误累积和自相矛盾而降低准确率。

## 方法详解

### 整体框架

LLMThinkBench 框架包含四个核心组件：(1) 14 个确定性基础数学任务的任务空间；(2) 准确率-冗余度二维空间的形式化；(3) Overthinking Score 指标；(4) 可安装的开源工具（PyPI: llmthinkbench），支持动态测试生成、多后端推理、层次化答案提取和报告生成。

### 关键设计

1. **Overthinking Score**:

    - 功能：将准确率和 token 效率统一为单一指标
    - 核心思路：定义 Token 效率 $E_{t,i} = 1 - \frac{\bar{T}_i - T_{min}}{T_{max} - T_{min}}$，然后用调和平均 $\mathcal{O}_i = \frac{2 \cdot A_i \cdot E_{t,i}}{A_i + E_{t,i}}$ 组合准确率和效率。调和平均严重惩罚不平衡——90% 准确率 + 10% 效率仅得 0.18 分，而 60%+60% 得 0.60 分
    - 设计动机：算术平均无法充分惩罚极端不平衡（90% 准确率+10% 效率仍得 0.55）。调和平均在所有对称齐次均值中最大程度惩罚不平衡

2. **动态测试生成协议**:

    - 功能：消除数据污染风险，确保评估公平性
    - 核心思路：基于可复现种子动态生成测试实例。列表长度从 {8,16,32,64} 采样，数值从 Uniform[-1000,1000] 采样，每折 1000 样本，3 折交叉验证（开源模型），100 样本（闭源模型，成本限制）。每个模型生成 42,000 个唯一问题
    - 设计动机：静态基准容易被训练数据污染，动态生成确保每次评估使用新数据

3. **层次化答案提取系统**:

    - 功能：从多样化的模型输出中可靠提取答案
    - 核心思路：四级提取策略——(1) 优先提取 \boxed{} 内容；(2) 解析显式答案标记（"The answer is..."）；(3) 从代码块或 Markdown 格式提取；(4) 任务特定启发式作为兜底。经 5000+ 响应验证，成功率 98.7%
    - 设计动机：不同模型输出格式差异巨大，可靠的答案提取是公平评估的前提

### 损失函数 / 训练策略

不涉及模型训练。使用现有模型的公开权重或 API 进行推理评估。评估覆盖 53 个模型，包括基座、指令微调、推理和量化变体。

## 实验关键数据

### 主实验

**部分代表性模型的 Overthinking Score 对比**

| 模型 | 参数 | 准确率 | Overthinking Score | 平均输出 Token |
|------|------|--------|-------------------|---------------|
| Phi-4 | 14B | 78.92% | **0.863** | 378.6 |
| Phi-4-reasoning-plus | 14B | 69.54% | 0.234 | 6,780.7 |
| Qwen3-14B | 14B | 86.52% | 0.727 | 3,607.6 |
| Qwen3-0.6B | 0.6B | 49.99% | 0.545 | 3,162.8 |

### 消融实验

**推理预算约束实验（Qwen3 推理模型）**

| 配置 | 准确率 |
|------|--------|
| 全预算 | 72% |
| 1024 token 限制 | 44%（-28%） |
| 推理预算 low→medium→high（GPT-5/o系列） | 准确率增益 ≈ 0 |

**量化实验（Qwen2.5 家族）**

| 配置 | 准确率变化 |
|------|-----------|
| FP16 → 8-bit | 大模型几乎不变 |
| FP16 → 4-bit | 大模型轻微下降，小模型显著下降 |

### 关键发现

- 基础数学悖论：GSM8K 上 95%+ 的模型在本文任务上低于 75%——复杂基准表现不能代表基础数学能力
- 推理模型平均生成 6,780 token vs 标准模型 378 token（18×），但准确率更低（Phi-4-reasoning-plus 69.54% vs Phi-4 78.92%）
- Overthinking Score 揭示了准确率指标掩盖的效率陷阱：Phi-4 得 0.863 远超 Phi-4-reasoning-plus 的 0.234
- token 约束下推理模型"灾难性崩溃"——从 72% 降至 44%，表明推理能力与长链推理深度绑定
- 扩展推理预算收益递减——GPT-5/o 系列从 low 到 high 推理 effort 准确率增益接近零
- 量化保留了基础推理能力，说明过度思考来自训练而非硬件限制

## 亮点与洞察

- Overthinking Score 是一个优雅且有信息量的指标——调和平均的严格惩罚使其能区分"高效正确"和"冗余正确"
- "基础数学悖论"是一个重要发现——挑战了"复杂基准得分高=数学能力强"的假设
- 动态测试生成+开源工具（PyPI 包+排行榜）使结果可复现且易扩展

## 局限与展望

- 仅覆盖 14 个确定性数学任务，未覆盖更复杂的数学推理或非数学领域
- token 效率的归一化依赖于评估集中的全局最大/最小值，可能受极端值影响
- 未分析过度思考的具体模式（如错误累积、自相矛盾的比例）
- 未探索如何训练既准确又高效的推理模型

## 相关工作与启发

- **vs ThoughtTerminator/Self-Braking**: 这些工作提出缓解过度思考的策略，本文提供了量化过度思考的指标——度量是干预的前提
- **vs GSM8K/MATH 基准**: 这些基准侧重准确率，本文补充了效率维度
- **vs Graph of Thoughts/LogicPuzzleRL**: 这些方法增强复杂推理，但未解决基础运算上的过度思考

## 评分

- 新颖性: ⭐⭐⭐⭐ Overthinking Score 是新颖且有用的指标，基础数学悖论是重要发现
- 实验充分度: ⭐⭐⭐⭐⭐ 53 个模型、量化分析、预算约束、动态生成，规模大且全面
- 写作质量: ⭐⭐⭐⭐ 形式化定义严谨，实验叙述清晰
- 价值: ⭐⭐⭐⭐⭐ 为推理模型的效率评估提供了标准化工具和深刻洞察

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](../../NeurIPS2025/llm_evaluation/evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)
- [\[ACL 2026\] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)
- [\[ICCV 2025\] On the Robustness Tradeoff in Fine-Tuning](../../ICCV2025/llm_evaluation/on_the_robustness_tradeoff_in_fine-tuning.md)

<!-- RELATED:END -->
