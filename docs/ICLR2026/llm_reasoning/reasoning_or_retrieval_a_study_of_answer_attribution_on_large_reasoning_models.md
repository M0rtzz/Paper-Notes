---
title: >-
  [论文解读] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models
description: >-
  [ICLR 2026][LLM推理][large reasoning models] 首次系统研究大型推理模型（LRM）的答案来源归因问题，揭示推理（CoT）和检索（记忆）两种机制同时竞争影响最终答案，并提出 Farl（遗忘增强强化学习）通过抑制检索捷径来提升模型的真实推理能力。
tags:
  - ICLR 2026
  - LLM推理
  - large reasoning models
  - CoT reasoning
  - memory retrieval
  - answer attribution
  - reinforcement-learning
  - unlearning
  - GRPO
---

# Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models

**会议**: ICLR 2026  
**arXiv**: [2509.24156](https://arxiv.org/abs/2509.24156)  
**代码**: [ZJUWYH/FARL](https://github.com/ZJUWYH/FARL)  
**领域**: llm_reasoning  
**关键词**: large reasoning models, CoT reasoning, memory retrieval, answer attribution, reinforcement-learning, unlearning, GRPO

## 一句话总结

首次系统研究大型推理模型（LRM）的答案来源归因问题，揭示推理（CoT）和检索（记忆）两种机制同时竞争影响最终答案，并提出 Farl（遗忘增强强化学习）通过抑制检索捷径来提升模型的真实推理能力。

## 研究背景与动机

大型推理模型（如 DeepSeek-R1、GPT o-series）通过链式思维（CoT）推理展示了强大的问题解决能力。然而，越来越多的证据表明这些模型的最终答案与其推理过程经常不一致：

**推理-答案断连**：最终答案并不总是由 CoT 推理过程直接产生，上下文偏差可以在 CoT 未承认的情况下影响输出

**双重机制假说**：模型可能同时通过"审慎推理"和"直接从内部记忆检索"两条路径生成答案

**训练方法影响不明**：蒸馏和强化学习对这两种机制的影响尚未被系统研究

核心研究问题：
- **RQ1**：LRM 是否同时使用推理和检索来得出答案？
- **RQ2**：什么因素影响两种能力的相对优势？
- **RQ3**：如何控制这两种能力的相对强度？

## 方法详解

### 整体框架

提出**推理-检索联合扰动框架**，通过分别干扰推理和检索路径，观察最终答案的变化来量化两种机制的贡献。

### 关键设计

**推理扰动**：在模型生成的 CoT 末尾注入误导性线索 $c$（如"可靠专家建议答案是 B"），将被篡改的 CoT 预填充到提示中重新生成：
$$\mathcal{M}(x \| z \| c; \theta) = y'$$
若 $y' = y_r$（误导答案），说明 CoT 变化成功影响了最终答案。

**检索扰动**：通过监督微调（SFT）"毒化"模型记忆，强制模型记住特定题目-错误答案的关联：
$$\min_\theta \ell(y_t, \mathcal{M}(x;\theta))$$
其中 $y_t$ 选择原始模型中 logit 最高的非正确答案。使用 LoRA（$r=64, \alpha=16$）、AdamW（lr=1e-4）训练 8 个 epoch。

**联合扰动**：同时施加两种扰动，制造"拔河"效应：
$$\mathcal{M}(x \| z \| c; \theta') = y'$$
设置两种条件：(i) 两种扰动指向相同错误答案 ($y_r = y_t$)；(ii) 指向不同错误答案 ($y_r \neq y_t$)。

**度量指标**：
- R-PSR（推理扰动成功率）：$\text{R-PSR} = \mathbb{E}_{(x,y)} \mathbf{1}[y' = y_r]$
- T-PSR（检索扰动成功率）：$\text{T-PSR} = \mathbb{E}_{(x,y)} \mathbf{1}[y' = y_t]$
- PER（事后解释率）：CoT 逻辑上支持被毒化答案的比例

### Farl：遗忘增强强化学习

基于关键洞察：检索机制在 RL 中可以作为"奖励黑客"的捷径——模型通过检索记忆直接获得正确答案并获取高奖励，而非通过真实推理。

Farl 在标准 GRPO 流程中加入遗忘步骤：
1. 每个 epoch：设置参考模型 → 进行 GRPO 迭代 → **执行 NPO 遗忘**
2. GRPO 优势计算：$\hat{A}_j = \frac{r(x,z_j,y_j) - \text{mean}(\{r\}_{j=1}^G)}{\text{std}(\{r\}_{j=1}^G)}$
3. 遗忘使用 Negative Preference Optimization (NPO)，抑制已记忆答案的检索路径

### 损失函数

结合 GRPO 损失 $\mathcal{J}_{\text{GRPO}}$ 和 NPO 遗忘损失 $\mathcal{L}_{\text{NPO}}$，交替优化。

## 实验关键数据

### 主实验

| 方法 | R-PSR ↓ | T-PSR ↓ | 训练域 ACC ↑ | 域外 ACC ↑ |
|------|---------|---------|-------------|-----------|
| R1-Llama-8B (Base) | 0.378 | 0.381 | 0.725 | 0.716 |
| SFT | 0.392 | 0.311 | 0.787 | 0.732 |
| RL (GRPO) | 0.259 | 0.262 | 0.869 | 0.745 |
| **Farl** | **0.197** | **0.234** | **0.891** | **0.757** |

Farl 相对基线模型：R-PSR 降低 47.8%，T-PSR 降低 38.5%，训练域准确率提升 22.8%，域外准确率提升 5.8%。

### 消融实验 / 因素分析

**问题领域**：数学/逻辑领域的 T-PSR 和 R-PSR 均最低，表明模型在此类领域更依赖推理而非记忆。

**训练方法对比**：蒸馏模型的 T-PSR 和 R-PSR 显著高于 RL 模型，说明蒸馏更倾向记忆而非推理。蒸馏模型的 PER（事后解释率）也明显更高——它们伪造 CoT 来合理化记忆中的答案。

**模型规模**：更大的模型在 PER、T-PSR、R-PSR 上均表现更低，说明大模型推理主导性更强。

**注意力机制分析**：中间层（12-16 层）的注意力头在推理/检索路径分类中获得最高 AUC。因果干预实验验证：替换高 AUC 头的激活值可以 87.2% 恢复原始答案（随机头仅 5.3%）。

### 关键发现

1. 推理和检索机制**同时存在并竞争**，两种扰动都能独立改变最终答案
2. 当两种扰动指向相同答案时，扰动效果**协同放大**
3. 蒸馏模型存在严重的**事后解释**现象：记忆毒化后不仅输出错误答案，还伪造支持该答案的 CoT
4. CoT 质量指标（cycle 提升 37.0%、diameter 提升 5.7%、small world index 提升 84.0%）表明 Farl 生成了更高质量的推理路径

## 亮点与洞察

1. **首次机制性研究**：首次系统性地探索 LRM 答案生成中推理与检索的竞争关系
2. **精巧的实验设计**：联合扰动框架可以清晰地分离和量化两种机制的贡献
3. **因果证据**：不仅有相关性分析（注意力头 AUC），还通过激活值替换提供了因果干预证据
4. **Logit 动态可视化**：逐步追踪推理过程中两条路径的 logit 竞争，直观展示了推理-检索的动态交互
5. **实用启发**：Farl 的"遗忘 + RL"范式为提升模型真实推理能力提供了新思路

## 局限性

1. Farl 虽提升推理能力，但生成更长的推理链（MTL 从 1537 增至 1914），推理效率下降
2. 受计算资源限制，仅在 R1-Llama-8B 和 R1-Qwen-7B 上验证，更大模型的结论有待验证
3. 检索扰动使用 SFT 实现，虽然验证了局部性和效率，但与真实"记忆"的关系仍需讨论
4. 训练仅在 Math&Logic 领域进行，其他领域的迁移效果有限（域外仅 +5.8%）

## 相关工作与启发

- **与推理-答案断连研究的关系**：Turpin et al.、Lanham et al. 等发现了 CoT 不忠实现象，本文更进一步揭示了其背后的双重机制
- **与记忆编辑的关系**：Meng et al. 的 ROME/MEMIT 聚焦检索机制的编辑，本文将检索作为与推理竞争的一个路径来研究
- **对 RL 后训练的启发**：揭示了 RL 训练中"奖励黑客"的新形式——模型可以通过检索记忆而非真实推理来获取奖励

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 首次系统研究 LRM 的推理-检索双重机制，洞察深刻
- **实用性**: ⭐⭐⭐⭐ — Farl 方法有效但适用范围有待扩展
- **实验完整度**: ⭐⭐⭐⭐⭐ — 从行为实验到机制分析到因果干预，层层递进
- **写作质量**: ⭐⭐⭐⭐⭐ — 研究问题驱动，结构清晰，可视化出色
- **综合评分**: ⭐⭐⭐⭐⭐ — 揭示了 LRM 的关键机制性问题，对后续研究有重要指导意义

<!-- RELATED:START -->

## 相关论文

- [Segment-Level Attribution for Selective Learning of Long Reasoning Traces](segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)
- [Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)

<!-- RELATED:END -->
