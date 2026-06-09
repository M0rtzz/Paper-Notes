---
title: >-
  [论文解读] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling
description: >-
  [ICLR 2026][LLM效率][Mixture-of-Experts] 提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销…
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "Mixture-of-Experts"
  - "Expert Parallelism"
  - "all-to-all通信"
  - "模型-数据协同调度"
  - "Token-Expert亲和性"
---

# Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling

**会议**: ICLR 2026  
**arXiv**: [2503.04398](https://arxiv.org/abs/2503.04398)  
**代码**: 基于SGLang实现（约5000行Python + Triton内核）  
**领域**: LLM效率  
**关键词**: Mixture-of-Experts, Expert Parallelism, all-to-all通信, 模型-数据协同调度, Token-Expert亲和性  

## 一句话总结
提出语义并行(Semantic Parallelism)范式，通过预测token-expert路由路径并协同调度模型放置与数据分发，大幅削减MoE推理中专家并行的all-to-all通信开销，在Attention-DP场景下吞吐提升最高2.78×，Attention-TP场景下延迟降低最高24.9%。

## 研究背景与动机
**MoE模型推理受all-to-all通信瓶颈制约**：EP(Expert Parallelism)将专家分布到多GPU，但需要两次all-to-all集合通信路由token到远程专家再返回，即使在400GB/s高速互联上仍占MoE层前向延迟的59.2%

**现有方案将模型放置和数据调度割裂**：专家放到哪个GPU和token发到哪个GPU被当作独立问题处理，导致大量不必要的跨设备通信

**Token具有上下文无关的专家亲和性**：实验发现token对特定专家的激活高度集中且稳定（top-k专家累计激活概率中位数达0.833-0.976），这提供了预测路由的基础

**DeepSeek-V3/R1、Qwen3等MoE模型的广泛部署**使得EP通信优化成为关键产业需求

## 方法详解

### 整体框架
Sem-MoE 把 MoE 推理的通信优化从"事后治理 all-to-all"前移到"事前预测路由"：先离线对 token-expert 亲和性做画像，据此把高度共激活的专家聚到同一 GPU；在线时再用同一份亲和性把请求或 token 直接送往其最可能命中的专家所在设备，从而让大部分激活变成本地访问、远程 all-to-all 流量被压到最低。模型放置与数据分发由同一个目标统一求解，而不是各自为政。

### 关键设计

**1. 上下文无关的 Token-Expert 亲和性画像：为预测式调度提供可靠先验。** 整套方法成立的前提，是 token 路由是否足够稳定到可以"预测"。作者在 ShareGPT 上对 DeepSeek-V2-Lite 做 profiling，发现同一个 token 在不同上下文中会一致地路由到相同的 top-k 专家子集——各层路由命中的 F1-score 中位数高达 0.833–1.000，而非 top-k 专家的最大热度仅约 0.05，激活高度集中。据此为每个 token 维护一张激活频率表 $T^{(L)} \in \mathbb{N}^{t\times N}$（$t$ 为词表大小、$N$ 为专家数），归一化后即得 token 对各专家的路由概率 $R_{ij}$，作为后续放置与调度的统一依据。这一画像离线一次完成，且具备跨数据集零样本迁移能力，避免了在线统计的额外开销。

**2. 离线模型调度——把共激活专家聚到一处：从源头减少跨设备激活。** 拿到亲和性后，作者把"专家放到哪块 GPU"建模成一个 0-1 整数规划：将专家划分为若干等大小的 cluster（对应各 GPU），优化目标是 $L = \theta\cdot L_{\text{balance}} + (1-\theta)\cdot L_{\text{remote}}$，其中前项约束各 cluster 负载均衡、后项最小化跨 cluster 的远程激活量，$\theta$ 在两者间权衡；约束要求每个专家恰属一个 cluster 且各 cluster 专家数相等。直接求解 ILP 代价高，作者用交替优化在专家归属与 cluster 中心之间迭代逼近，把经常一起被激活的专家放进同一张卡。这样一来，token 命中的多个专家更可能落在本地，远程 all-to-all 的需求被结构性地削减。

**3. 在线数据调度——让数据主动靠近专家：把"路由"变成"投递"。** 放置固定后，在线阶段反过来用亲和性决定数据往哪送，并针对两类 Attention 并行分别设计。Attention-DP 下做**请求间调度**：把整条请求投递到其 token 最可能激活的专家所在 rank，即 $S_r = \arg\max_j \sum_{i\in r} R_{ij}$，再叠加 workload-aware 的均衡策略防止某些 rank 过热。Attention-TP 下做更细粒度的**请求内 token 调度**：单看单层预测不够准，作者利用相邻层专家选择的马尔可夫依赖，用一个 2-gram 设备转移模型增强预测，并设计 Shuffled-Reduce-Scatter (SRS) 与 Shuffled-AllGather (SAG) 两个融合通信原语，把投机性的 token 重排直接嵌进 TP 既有的通信阶段，使额外开销仅约 1%。两种场景共用同一套亲和性，区别只在调度粒度。

**4. 系统实现——让算法收益真正落到延迟上：消除工程瓶颈。** 协同调度的理论收益要靠高效内核兑现。作者基于 SGLang 实现约 5000 行 Python 加自定义 Triton 内核，重写的 argsort 内核比 PyTorch 原生快 25%，并集成 DeepEP 通信库执行优化后的 all-to-all，确保减少的远程激活能切实转化为端到端的吞吐与延迟收益。

## 实验

### Attention-DP场景（吞吐 under SLO约束）

| 模型 | vs SGLang (TTFT SLO) | vs SGLang (E2E SLO) | vs MoETuner (TTFT) | vs MoETuner (E2E) |
|------|---------------------|--------------------|--------------------|-------------------|
| DeepSeek-V2-Lite | +31% | +221% | +32% | +278% |
| Qwen3-30B-A3B | +98% | +11% | +35% | +32% |

### Attention-TP场景（延迟降低）

| 模型 | 输入长度256 | 输入长度512 | 输入长度1024 |
|------|-----------|-----------|------------|
| DeepSeek-V2-Lite p99 TTFT | -12.21% | -10.60% | -18.89% |
| Qwen3-30B-A3B p99 TTFT | -17.16% | -24.90% | -3.80% |

### 关键发现
- 本地激活率(LAR)比vanilla提升37-43%，对应EP层延迟降低41.8-46.6%
- 协同调度算法的LAR比最佳baseline(MoETuner)高15.4%，负载不平衡率更低
- 跨数据集零样本迁移表现验证了调度策略的泛化性

## 亮点
- 提出"语义并行"新范式概念，将通信优化从被动治理转为主动预防
- 揭示token-expert亲和性的上下文无关特性，为预测式调度提供理论基础
- 同时优化模型放置和数据调度，是系统性而非局部的方案
- SRS/SAG融合原语设计精巧，将token重排嵌入已有通信流，额外开销仅1%

## 局限性
- 仅在8-GPU单节点验证，跨节点/低速互联场景效果待验证
- 预测模型需要离线profiling数据，冷启动时无法发挥作用
- 对路由机制高度动态的MoE变体更换gate函数后需重新profiling
- 未评估与KV缓存优化或量化技术的联合使用

## 相关工作
- **专家放置**：MoETuner(ILP优化放置)、ExFlow(层间专家亲和性)、EPLB(DeepSeek的负载均衡)
- **MoE推理系统**：DeepSpeed-MoE、Tutel、vLLM、SGLang
- **预取/卸载**：Pre-gated MoE(修改架构预测下层专家)——Sem-MoE无需修改架构
- 本文是首个同时优化模型调度和数据调度的工作

## 评分 ⭐⭐⭐⭐⭐
- **新颖性**: 5/5 — 语义并行范式+协同调度思想原创性强
- **实验充分度**: 4/5 — 两种模型两种场景，但仅单节点
- **写作质量**: 4/5 — 系统描述清晰，图示质量高
- **价值**: 5/5 — MoE推理的核心痛点，产业价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Ladder Residual: Parallelism-Aware Architecture for Accelerating Large Model Inference](../../ICML2025/llm_efficiency/ladder-residual_parallelism-aware_architecture_for_accelerating_large_model_infe.md)
- [\[ACL 2026\] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](../../ACL2026/llm_efficiency/alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)

</div>

<!-- RELATED:END -->
