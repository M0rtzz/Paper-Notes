---
title: >-
  [论文解读] Predicting LLM Reasoning Performance with Small Proxy Model
description: >-
  [ICLR 2026][LLM评测][scaling law] 提出 rBridge，通过使用 frontier 模型的推理 trace 作为 gold label 并按 token 级任务对齐加权 NLL，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能…
tags:
  - "ICLR 2026"
  - "LLM评测"
  - "scaling law"
  - "proxy model"
  - "reasoning"
  - "pre-training data selection"
  - "negative log-likelihood"
---

# Predicting LLM Reasoning Performance with Small Proxy Model

**会议**: ICLR 2026  
**arXiv**: [2509.21013](https://arxiv.org/abs/2509.21013)  
**代码**: 有（计划开源数据集和代码）  
**领域**: LLM评测  
**关键词**: scaling law, proxy model, reasoning, pre-training data selection, negative log-likelihood

## 一句话总结
提出 rBridge，通过使用 frontier 模型的推理 trace 作为 gold label 并按 token 级任务对齐加权 NLL，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，在数据集排名任务中实现 100× 以上的计算节省。

## 研究背景与动机
预训练大语言模型的成本极高，因此业界常用小型代理模型（proxy model）来评估预训练设计选择。scaling law 和数据集排名不变性等研究已证明这一策略在一般任务上的可行性。

**核心矛盾**：推理能力表现出**涌现行为**（emergent behavior），只在足够大的模型（通常 >7B）上才可靠地出现。小模型（≤1B）在推理 benchmark 上噪声极大甚至斜率反向（如 1B 模型在 MATH500 上随训练进行准确率不升反降）。这意味着：
- 用小模型的准确率直接预测大模型表现 → 失败
- 实践者被迫使用 7B-15B 的"代理模型"，开销仍然巨大（单次 7B/500B tokens 训练 >50K USD）

**作者的分析揭示两个根本问题**：

**评估目标错位**：小模型的训练目标是 NTP（next-token prediction），但评估用的是 Acc./Pass@K——与训练目标完全不同的离散指标

**任务对齐缺失**：即使使用 NLL，gold label 的选择也至关重要：如果 gold label 分布外（OOD），NLL 信号同样嘈杂。同时，标准 NLL 对所有 token 等权处理，未区分任务关键 token 和格式化 token

## 方法详解

### 整体框架
rBridge 的核心思想：设计一个 proxy 评估指标 $\text{metric}^p$，使得 $\text{corr}(\text{metric}^p(\pi^p), \text{metric}^t(\pi^t))$ 最大化。即找到一个在小模型上计算的指标，使其与大模型的目标指标（Acc./Pass@K）高度相关。

### 关键设计
1. **推理 Trace 作为 Gold Label**：

    - 使用 frontier 模型 $\pi^\phi$ 生成的**推理 trace** $R^\phi$（不含最终答案格式部分）作为 gold label
    - **为什么有效**：(a) $R^\phi$ 更接近预训练数据分布（连续长文本），NLL 降低 74.7%；(b) $R^\phi$ 是通向正确答案的推理过程，天然与目标任务对齐
    - 对比 ScalingBench 的方案：ScB 使用 $R^\phi + A^\phi$（trace + 答案），但答案中含 "\\n", "Final Answer:", "I hope it is correct." 等格式伪影，严重 OOD

2. **Token 级任务对齐加权**：
    $\text{rBridge NLL}(\text{token}_i) = -\log(p^p(\text{token}_i)) \cdot \frac{1}{|\text{token}_i|} \sum_{\text{letter} \in \text{token}_i} p^\phi(\text{letter})$
   
    - 每个 token 的 NLL 乘以 frontier 模型对该 token 的信心度（概率）
    - **直觉**：frontier 模型高置信的 token（如 "sum modulo 9"）是推理关键步骤，应获得更高权重；低置信的 token（如换行符、编号）是格式化噪声，降低权重
    - **Tokenizer 无关**：使用**字母级**概率平均法处理不同 tokenizer 的分词差异
    - 最后对权重因子做 MinMax 归一化以放大效果

3. **一次性前期成本**：frontier 模型生成推理 trace 的成本 <$10/benchmark，加权计算仅需几秒 CPU 时间

### 训练策略
本文**不训练新模型**，而是使用预训练中间检查点作为 proxy 模型，计算 rBridge NLL 来预测目标模型性能。

## 实验关键数据

### 主实验（1B→13B 代理-目标关系）

| 方法 | 平均 Train R² ↑ | 平均 Test MAE ↓ |
|------|:---:|:---:|
| Acc./Pass@1 | 0.304 | 3.709 |
| iSFT | 0.290 | 5.123 |
| TED | 0.375 | 3.377 |
| MPCA | 0.194 | 302.642 |
| NLL | 0.485 | 5.173 |
| $R^\phi$ (仅 trace NLL) | 0.867 | 1.455 |
| **rBridge** | **0.874** | **1.384** |

- rBridge 在 12 个设定（6 benchmark × R²/MAE）中的 10 个取得最好
- 1B→32B: R²=0.826, MAE=1.481，同样最优
- 1B→13B+SFT: R²=0.846, MAE=1.304，最优

### 消融实验（数据集排名 <100M→1.2B）

| 代理规模 | rBridge DAcc | 最佳基线 DAcc | 计算节省 |
|:---:|:---:|:---:|:---:|
| 3.7M (87.3M tokens) | ~70% | ~50% (随机水平) | 733.4× |
| 6M (81.6M tokens) | ~75% | ~55% | 100.2× |
| 97.9M | ~80% | ~75% | - |

- 在最小 proxy（3.7M）下，rBridge 比基线高 **27% DAcc**
- 实现相同排名精度，rBridge 节省 **100.2× 到 733.4×** FLOPs

### 关键发现
1. **rBridge 优于 7-13× 更大的模型**：1B 模型使用 rBridge 的预测能力超过 7B/13B 模型直接用准确率评估
2. **零样本跨数据集迁移**：在 OLMo-Mix-1124 上拟合的 rBridge→Acc 函数可零样本迁移到另一个数据集，5 个 benchmark 中排名全对（5/5），MAE 仅 0.043-1.417（1 个异常值 9.716）
3. **消融证实每个组件有效**：从标准 NLL → 使用 $R^\phi$ → 添加加权 → MinMax 归一化，每步都带来一致提升

## 亮点与洞察
- **深刻的洞察**：小模型在推理 benchmark 上"失败"不是因为缺乏信息，而是因为评估方式（Acc.）与小模型的能力不匹配
- **NLL 回归本源**：预训练的目标是 NTP，评估也应回归 NLL——但关键在于用什么做 gold label
- **Frontier 模型概率作为自动标注器**：token 概率天然编码了"哪些步骤是关键推理"，免去人工标注
- **实用性极强**：两阶段数据集优化框架（<100M 初筛 → 1B 精排）具有直接的工业应用价值

## 局限与展望
- Frontier 模型并非 100% 准确，少量错误的推理 trace 可能引入噪声（但实验显示过滤掉错误 trace 改善甚微）
- Frontier 模型偶尔无法按要求格式生成（当前仅重试一次后放弃）
- 零样本迁移仅在 1B→7B 上验证了一个数据集对，规模受限于计算预算
- 未验证在非推理任务上的效果（虽然非推理任务已有成熟方案）
- 两阶段优化框架仅作为讨论提出，未完整实验验证

## 相关工作与启发
- 与 scaling law 文献（Kaplan et al., Hoffmann et al.）互补：rBridge 专门解决推理涌现带来的 proxy 困难
- 与 DataDecide benchmark 直接对比：在其任务上以极低成本取得最优排名
- 对预训练数据配比优化（Xie et al., Liu et al.）的启示：NLL/perplexity 是次优指标，rBridge 更适合推理导向的数据选择
- 广泛启发：任何需要"用小模型预测大模型"的场景都可考虑 rBridge 范式

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心 insight（NLL + frontier trace + token 加权）并非全新，但组合巧妙且效果显著
- 实验充分度: ⭐⭐⭐⭐⭐ 三阶段实验设计严谨，6个 benchmark × 多模型规模 × 消融 × 跨数据集迁移
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题分析循序渐进，但符号和表格较密集
- 价值: ⭐⭐⭐⭐⭐ 直接降低预训练探索成本 100×+，工业落地价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance](../../ACL2025/llm_evaluation/language_complexity_measurement_as_a_noisy_zero-shot_proxy_for_evaluating_llm_pe.md)
- [\[ACL 2026\] Reasoning Model Is Superior LLM-Judge, Yet Suffers from Biases](../../ACL2026/llm_evaluation/reasoning_model_is_superior_llm-judge_yet_suffers_from_biases.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](../../ACL2026/llm_evaluation/retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[ACL 2025\] Where Are We? Evaluating LLM Performance on African Languages](../../ACL2025/llm_evaluation/where_are_we_evaluating_llm_performance_on_african_languages.md)

</div>

<!-- RELATED:END -->
