---
title: >-
  [论文解读] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering
description: >-
  [ACL 2025][模型压缩] 提出 RISE——结合 RAG 与自迭代训练的多跳问答框架，通过问题分解、检索阅读、自我批判三个动作的自我探索循环，迭代生成训练数据并多目标优化模型，在 2Wiki/HotpotQA/MuSiQue 上超越 GPT-3.5 和所有 8B 级基线。
tags:
  - ACL 2025
  - 模型压缩
---

# RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering

**会议**: ACL 2025  
**arXiv**: [2505.21940](https://arxiv.org/abs/2505.21940)  
**代码**: 无  
**领域**: LLM推理  

## 一句话总结

提出 RISE——结合 RAG 与自迭代训练的多跳问答框架，通过问题分解、检索阅读、自我批判三个动作的自我探索循环，迭代生成训练数据并多目标优化模型，在 2Wiki/HotpotQA/MuSiQue 上超越 GPT-3.5 和所有 8B 级基线。

## 背景与动机

1. **多跳问答（MHQA）仍是 LLM 难题**：需要整合多源证据并管理复杂逻辑依赖，小模型尤其容易出错。
2. **RAG 存在两类核心错误**：(a) 证据聚合错误——模型未能准确整合多个检索片段导致幻觉；(b) 推理分解错误——子问题与原问题意图不一致导致推理链偏离。
3. **全模型梯度方法代价过高**：蒸馏和人工标注微调虽有效但成本高，且人工偏差可能损害效果。
4. **自迭代与 RAG 结合的空白**：自迭代方法在代码生成和 Agent 中成功，但在 RAG 多跳问答中尚未探索。

## 方法详解

### 整体框架

RISE 是一个自迭代闭环框架，每轮包含两个阶段：**自我探索**（生成训练数据） → **迭代优化**（多目标微调模型）。

### 1. 自我探索机制

对每个问题 $q_0$，模型执行最多 20 轮探索节点：

**问题分解**：模型根据已有历史 $\mathcal{H} = \{(subq_1, suba_1), \ldots\}$ 和原问题 $q_0$，生成下一个子问题 $subq_t$；若历史信息足够则直接输出最终答案。

**检索阅读（Retrieve-then-Read）**：对子问题用检索器获取相关片段 $r_t$，模型基于检索结果生成子答案 $suba_t$。

**自我批判（Self-Critique）**：模型评估 $(subq_t, suba_t)$ 对解决原问题的相关性，输出二元判断 $\sigma_t \in \{0, 1\}$。若判为 False，回退到上一个有效节点重新生成。

三个动作分别收集数据集 $\mathcal{D}_d$（分解）、$\mathcal{D}_r$（阅读）、$\mathcal{D}_c$（批判），每类 2K~8K 样本。

### 2. 多目标联合优化

三个数据集联合训练，总损失：
$$\mathcal{L} = \alpha \mathcal{L}_d + \beta \mathcal{L}_r + \gamma \mathcal{L}_c$$

- $\mathcal{L}_d$：子问题生成的自回归损失
- $\mathcal{L}_r$：基于检索上下文的子答案生成损失
- $\mathcal{L}_c$：True/False 二分类的交叉熵损失
- 实验中采用等权 $\alpha = \beta = \gamma = 1$，避免过拟合

### 3. 问题扩展

每轮优化后，用更新的模型对种子问题做上下文学习扩展，生成更多样的训练问题，供下一轮自我探索使用。

## 实验结果

### 表2：主要结果（Accuracy %）

| 方法 | 模型 | 2Wiki | HotpotQA | MuSiQue | NQ | WebQ | TriviaQA |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive LLM | LLaMA-3.1-8B | 35.90 | 27.30 | 11.30 | 57.50 | 61.25 | 71.50 |
| GPT-3.5-turbo | GPT-3.5 | 47.10 | 41.50 | 19.10 | 57.25 | 58.30 | 80.25 |
| CoT | LLaMA-3.1-8B | 43.00 | 34.60 | 16.20 | 56.75 | 62.00 | 71.75 |
| GenGround | LLaMA-3.1-8B | 37.90 | 36.10 | 17.80 | 48.50 | 44.50 | 75.25 |
| **RISE** | **LLaMA-3.1-8B** | **49.40** | **40.50** | **21.70** | **59.50** | **62.50** | **80.25** |

RISE 在所有 MHQA 数据集上超越 GPT-3.5，比同模型 Naive RAG 提升 6-14 个百分点。

### 消融实验（Round 1 数据，Accuracy %）

| 配置 | 2Wiki | HotpotQA | MuSiQue |
|:---|:---:|:---:|:---:|
| w/o 分解 | 37.63 | 33.89 | 11.08 |
| w/o 检索阅读 | 40.59 | 33.06 | 9.46 |
| w/o 自我批判 | 38.98 | 33.89 | 10.27 |
| 分别训练 | 40.86 | 34.72 | 10.54 |
| **RISE（联合训练）** | **41.13** | **35.83** | **11.89** |

三个子任务缺一不可，联合训练优于分别训练。

### 迭代提升

- 精度随迭代轮次持续上升（4 轮），推理链长度先升后降，表明分解能力逐步优化
- 与 GPT-4o 的批判一致性从 Round 1 的 60-74% 提升至 Round 4 的 78-81%

## 亮点

- **自迭代 + RAG 的创新结合**：首次将自迭代训练范式引入 RAG 多跳问答，无需大模型蒸馏或人工标注
- **三任务协同自我探索**：分解、阅读、批判形成闭环，自动生成高质量训练数据
- **多目标联合优化**：三类数据互补学习，联合训练效果 > 分别训练
- **8B 模型超越 GPT-3.5**：在 MHQA 任务上 LLaMA-3.1-8B 经 RISE 训练后全面超越 GPT-3.5

## 局限性

- **检索器未优化**：框架依赖外部检索器但未对其进行自改进，检索质量是瓶颈
- **仅在 LLaMA-3.1-8B 上验证**：未测试更大/更小模型的效果
- **自我探索效率问题**：每个问题最多 20 轮探索节点，大规模应用时训练数据收集成本较高
- **等权策略非最优**：作者为避免过拟合选择等权，但 Table 1 显示 (α=2,β=2,γ=2) 达 44.27% vs 等权 41.13%

## 相关工作对比

| 维度 | RISE | Self-RAG | GenGround | CoT |
|:---|:---|:---|:---|:---|
| 检索 | 多轮 RAG | 自适应检索 | 交替生成+检索 | 无 |
| 自改进 | 自迭代微调 | 反思标记训练 | 无 | 无 |
| 问题分解 | 显式分解+批判 | 无 | 子问题引导 | 隐式链式 |
| 训练数据 | 自我探索生成 | 人工标注+GPT4 | 无 | 无 |
| 多跳能力 | 强（迭代增强） | 弱 | 中 | 中 |

## 评分

- ⭐⭐⭐⭐ 新颖性：RAG + 自迭代的结合是新探索方向，三任务闭环自我探索设计完整
- ⭐⭐⭐ 实用性：需 4 轮迭代训练，成本仍非微调级别的最优，但不依赖大模型标注
- ⭐⭐⭐⭐ 实验充分度：3 个 MHQA + 3 个 SHQA 数据集，消融+迭代分析+三能力单独评估覆盖全面
- ⭐⭐⭐ 写作质量：结构清晰但部分公式符号不一致，相关工作与方法的区分度可更强

<!-- RELATED:START -->

## 相关论文

- [TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)

<!-- RELATED:END -->
