---
title: >-
  [论文解读] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints
description: >-
  [ICML 2026][LLM效率][排队论] 本文建立首个显式纳入 KV 缓存显存动态的 LLM 推理排队模型，给出闭形稳定性条件 $\lambda < \mu(1-\delta)$，使运维人员可直接计算所需 GPU 数…
tags:
  - "ICML 2026"
  - "LLM效率"
  - "排队论"
  - "KV 缓存"
  - "显存约束"
  - "稳定性条件"
  - "吞吐量预测"
---

# A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints

**会议**: ICML 2026  
**arXiv**: [2605.04595](https://arxiv.org/abs/2605.04595)  
**代码**: 论文附录提供  
**领域**: LLM 推理效率 / 系统  
**关键词**: 排队论, KV 缓存, 显存约束, 稳定性条件, 吞吐量预测

## 一句话总结
本文建立首个显式纳入 KV 缓存显存动态的 LLM 推理排队模型，给出闭形稳定性条件 $\lambda < \mu(1-\delta)$，使运维人员可直接计算所需 GPU 数；在单 GPU、8 GPU 集群与 LongBench 真实数据上验证误差均 $\leq 10\%$。

## 研究背景与动机

**领域现状**：LLM 推理服务同时受计算与显存两类约束。KV 缓存加速 decoding 但在长上下文下占用大量显存。系统设计需要平衡吞吐、延迟与硬件成本。

**现有痛点**：经典排队论只建模计算约束；现有 LLM 系统论文（Wu/Yang/Li 等）关注调度算法但缺乏闭形稳定性判定；KV 缓存内存呈非线性动态——prompt 阶段按 chunk 增、decode 阶段按 token 逐增——难以套用标准 bin packing 框架。

**核心矛盾**：内存不是静态约束而是随时间演化的；不同请求处于不同阶段，共享显存高度耦合，简单的平均速率近似失效。

**本文目标**：给出严格、可计算的稳定性条件，让设计者根据 $\lambda$ 和系统参数直接估算所需 GPU 数。

**切入角度**：构造离散时间 Markov 链，把状态定义为"进行中的请求集合 + 各自进度"；定义 Lyapunov 势函数为"剩余生命周期内存×时间"，对 drift 做下界估计。

**核心 idea**：每个请求的"内存×时间"开销可写成显式函数 $g(s,o)$，于是服务速率 $\mu = M / (\bar b \mathbb E[g(s,o)])$，稳定条件为 $\lambda < \mu(1-\delta)$，$\delta = \text{ess sup}(s+o)/M$ 是松弛项。

## 方法详解

### 整体框架

两层：

- **系统层**：请求 FCFS/SJF 调度，混合连续批处理——多请求的 prompt chunks 与 decode tokens 在同一批次。显存约束 $\sum_{i\in S(t)} c_i^{(t)} \hat s + d_i^{(t)} \leq M$，其中 $c_i^{(t)}$ 是请求 $i$ 已处理 chunk 数，$d_i^{(t)}$ 是已生成 token 数。
- **稳定性层**：Markov 链状态包含每个请求的剩余进度。势函数 $V(t) = \sum_i g(s_i, o_i)$ 是所有未完成请求的总"内存×时间"。一阶 drift $\mathbb E[V(t+1)-V(t)] = -M(1-\delta) + \lambda \bar b \mathbb E[g(s,o)]$。

### 关键设计

1. **分段内存函数与生命周期视角**:

    - 功能：把 prompt 与 decode 两阶段的复杂内存动态合成一个解析量。
    - 核心思路：定义 $g(s,o) = \frac{(1+s/\hat s)s}{2} + s\cdot o + \frac{(1+o)o}{2}$，分别对应 prompt 累积、prompt 与 decode 重叠期、decode 累积三部分。无论请求处于哪个阶段，$V(t)$ 每步在满载下减少 $M(1-\delta)$，把动态非线性约束转化为线性的"面积擦除"。
    - 设计动机：用单一标量函数概括生命周期占用，简化 drift 分析。

2. **分块预填与连续批处理**:

    - 功能：控制每批计算复杂度，允许灵活混合多请求。
    - 核心思路：prompt 按固定 chunk 大小 $\hat s$ 分块，使每批 attention 计算 $O(M)$（与总 KV 缓存成正比，而非 prompt 长度平方）。连续批处理在显存允许时动态接纳新请求；溢出时退回 CPU 交换（本文假设罕见）。
    - 设计动机：消除长 prompt 的二次开销，使每步处理时间近似恒定，从而排队论分析成立。

3. **工作守恒调度 + 松弛项**:

    - 功能：保证 GPU 不空闲并合理预留突发请求空间。
    - 核心思路：任意工作守恒策略在 $\delta = \text{ess sup}(s+o)/M$ 的松弛下都稳定；松弛对应"最坏情况下一个最大请求到来时仍能容纳"。这与 vLLM 的 `gpu_memory_utilization=0.9` 等实际设置吻合。
    - 设计动机：把现实系统的保守显存利用率显式纳入理论。

## 实验关键数据

### 单 GPU 验证（合成 P:D 比例）

| Prompt:Decode | $\mu_{\text{gpu}}$ (req/s) | $\mu_{\text{theory}}$ | Gap |
|--------------|-------|-------|------|
| 1:1 | 3.387 | 3.263 | 3.66% |
| 2:1 | 3.650 | 3.956 | 8.38% |
| 1:2 | 2.969 | 2.902 | 2.25% |
| 混合（2:1→1:2 时变） | 3.137 | 3.385 | 7.90% |

### 真实数据集（LongBench v2，单 GPU）

| 指标 | 值 | 说明 |
|------|----|------|
| $\mu_{\text{gpu}}$ | 0.610 req/s | 实测 |
| $\mu_{\text{theory}}$ | 0.561 req/s | 预测 |
| Gap | 8.03% | 真实长上下文场景 |

### 8 GPU 集群

| 配置 | $\mu_{\text{gpu}}$ | $8\mu_{\text{theory}}$ | Gap |
|------|------|------|------|
| 1:1 P/D | 26.71 | 25.81 | 3.38% |

### 稳定性实验（单 GPU, 1:1）

| $\lambda$ | 关系 | 队列行为 | 理论预测 |
|---------|------|--------|--------|
| 1, 3 | $\lambda < \mu$ | 有界 (<5) | ✓ 稳定 |
| 5, 20, 50 | $\lambda > \mu$ | 线性增长 | ✓ 不稳定 |

### 关键发现
- **理论-实测 Gap 始终 $\leq 10\%$**：覆盖合成/真实、单 GPU/8 GPU 场景，预测精度令人意外。
- **线性可扩展**：8 GPU 集群 Gap 3.38%，与单 GPU 同量级，公式 $\mu_{\text{multi}} = 8\mu_{\text{single}}$ 成立。
- **P:D 比例影响显著**：1:2（长生成）比 2:1（长 prompt）吞吐低，与理论符号一致。
- **稳定/不稳定阶跃明显**：$\lambda$ 越过 $\mu$ 阈值后队列长度迅速线性发散，验证了 drift 论证。

## 亮点与洞察
- **首个 KV 显存排队模型**：填补排队论与 LLM 推理的空白，理论贡献清晰。
- **闭形稳定条件实用性强**：运维直接用 $\lceil \lambda/\mu \rceil$ 估算 GPU 数，无需模拟。
- **势函数构造优雅**：用"内存×时间面积"统一两阶段动力学，drift 分析极简。
- **跨场景验证严谨**：合成、真实、单卡、多卡均验证，Gap $\leq 10\%$ 提供高信度。

## 局限与展望
- 假设批处理时间恒定，忽略 CPU-GPU I/O、注意力不规则性等额外波动。
- 不支持 TP/PP 等并行策略；需把 $M, \bar b$ 替换为 TP-effective 值。
- 未考虑动态批大小、主动预加载、推测解码等高级调度。
- 真实流量长尾/突发情况下，平均到达率假设可能不准确。
- 改进方向：动态调节 chunk size 的在线控制；CPU 缓存交换的精细建模；多模型共置场景。

## 相关工作与启发
- **vs 经典排队论**：M/M/1, M/G/1 等不建模共享资源，本文把内存作为耦合变量引入。
- **vs LLM 调度算法文献（Wu/Yang/Li）**：他们设计最大化效率的调度；本文给出稳定性上界，为调度算法分析提供基础。
- **启发**：类似"生命周期资源占用"模型可推广到数据中心电源/散热等其他受约束系统。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次严格把 KV 显存纳入排队论框架，理论贡献明确。
- 实验充分度: ⭐⭐⭐⭐⭐ 合成 + 真实、单 GPU + 集群、稳定性验证俱全。
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，系统模型清晰。
- 价值: ⭐⭐⭐⭐⭐ 直接指导 LLM 服务的资源规划。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)
- [\[ACL 2025\] KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](../../ACL2025/llm_efficiency/kv_latent_cache_reduction.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)

</div>

<!-- RELATED:END -->
