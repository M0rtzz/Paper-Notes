---
title: >-
  [论文解读] WarmServe：一次加载多模型的 GPU 预热机制
description: >-
  [ICML 2026][LLM效率][GPU 预热] WarmServe 通过分析 LLM 服务工作负载的长期周期性规律，主动将多个模型参数预加载到 GPU，配合优化的放置算法和动态 KV 缓存预留策略，使系统能在请求突发时快速启动新实例——尾部 TTFT 相比现有系统降低 50.8 倍。
tags:
  - "ICML 2026"
  - "LLM效率"
  - "GPU 预热"
  - "多 LLM 服务"
  - "工作负载预测"
  - "冷启动"
  - "资源效率"
---

# WarmServe：一次加载多模型的 GPU 预热机制

**会议**: ICML 2026  
**arXiv**: [2512.09472](https://arxiv.org/abs/2512.09472)  
**代码**: https://github.com/LLMServe/WarmServe  
**领域**: LLM 效率 / 多模型服务  
**关键词**: GPU 预热, 多 LLM 服务, 工作负载预测, 冷启动, 资源效率

## 一句话总结
WarmServe 通过分析 LLM 服务工作负载的长期周期性规律，主动将多个模型参数预加载到 GPU，配合优化的放置算法和动态 KV 缓存预留策略，使系统能在请求突发时快速启动新实例——尾部 TTFT 相比现有系统降低 50.8 倍。

## 研究背景与动机

**领域现状**：多 LLM 服务系统需在共享 GPU 集群中并发部署多模型以提高资源利用率。主流方案有两类——（1）自动扩展：根据当前负载动态创建实例但冷启动延迟大；（2）GPU 共享：在同一 GPU 上并置多模型但严重受限 KV 缓存容量。

**现有痛点**：自动扩展在请求突发时需现场加载模型参数，导致严重 TTFT；GPU 共享虽避免初始化延迟，但每个模型分到的 KV 缓存极少。

**核心矛盾**：现有系统缺乏对未来工作负载特征的感知——自动扩展只能被动响应，GPU 共享的放置策略必须随时间保持稳定。

**关键观察**：虽然短期请求到达具有随机性，但实际生产环境中 LLM 服务的长期统计特性表现出强周期性——峰值负载在 5 分钟窗口内可以以平均 7.3% 相对误差精度预测。

**切入角度**：充分利用这种可预测性，采用主动式预热策略——在预测到未来负载突增前主动将备用模型副本加载到空闲 GPU 上。

**核心 idea**：引入"一次加载多模型"机制——将多个模型参数同时加载到单个 GPU 内存中；某模型遇请求突发时立即利用已预热参数启动活跃实例，然后快速驱逐其他模型参数。驱逐权重比按需加载快得多。

## 方法详解

### 整体框架
GPU 集群工作节点分三类——空闲（idle）、通用（universal）、专用（dedicated）。系统在空闲节点上预热多 LLM 转为通用节点；某预热模型收突发请求时该节点升级为专用节点同时驱逐其他模型；也允许在专用节点的未用 KV 缓存空间预热。

### 关键设计

1. **工作负载预测（Corrective Seasonal Predictor, CSP）**:

    - 功能：基于历史数据预测下个时间窗口内各模型的平均/峰值负载。
    - 核心思路：结合季节性模式 $P_{k,i} = \frac{1}{D}\sum_{d=1}^{D}L_{k-d,i}$ 和修正项 $\Delta_{k,i} = \frac{\sum_{w=1}^{\min(i,N)}(L_{k,i-w}-P_{k,i-w})\cdot 2^{w-1}}{2^{\min(i,N)}-1}$，最终预测 $\hat{L}_{k,i} = P_{k,i} + \Delta_{k,i}$。修正项对最近误差给予更高权重。
    - 设计动机：LLM 工作负载虽短期不可预测但长期呈现周期性；加入修正项能快速适应当前趋势，达到 92.7% 预测精度。

2. **模型放置算法**:

    - 功能：决定哪些模型副本需预热及放置在哪些 GPU，最小化跨模型预热干扰。
    - 核心思路：为每个待预热副本计算优先级分数（基于预期负载与当前实例数差距、冷启动延迟等），按分数降序排列。对每个副本贪心选择最优 GPU 组——优先选高分数副本能被保护（不被低分数副本驱逐）的 GPU 组。
    - 设计动机：LLM 跨多 GPU 分布（张量并行），单 GPU 释放会连锁驱逐多模型形成"跨模型干扰"；放置算法通过优先级隔离确保重要模型不被次要模型破坏。

3. **主动预热 + KV 缓存预留**:

    - 功能：在负载下降、模型实例即将释放前，利用该实例未用 KV 缓存空间提前加载新模型参数。
    - 核心思路：自动扩展器检测到负载下降要关闭某些实例时，这些实例通常还有充足未用 KV 缓存；系统计算所需保留的 KV 缓存为 $R = \max(C \cdot Q/B, T + C/B)$，超出部分可用于预热，空间不足时动态驱逐预热权重。
    - 设计动机：LLM 检查点超大（128GB+），传统预热在短暂窗口内常失败；通过在即将释放的 GPU 上"潜伏式"预加载，将 I/O 拉长到 GPU 仍在运行时的闲置期。

## 实验关键数据

### 主实验

| 系统 | P95 TTFT(s) | P99 TTFT(s) | 相对改进 | 最大 RPS |
|------|-----------|-----------|--------|--------|
| SLLM-GPU | 1.23 | 3.45 | - | 10 |
| MuxServe | 0.89 | 2.34 | - | 6 |
| WarmServe（无主动预热） | 0.18 | 0.31 | 6.8×-11.1× vs SLLM | 20 |
| WarmServe（完整） | 0.17 | 0.29 | 7.2×-11.9× vs SLLM / 5.2× vs MuxServe | 25 |

在 15 RPS、$\alpha$=0.5 设置下，WarmServe 相比 SLLM-GPU 实现 1.53-50.79× P99 TTFT 降低。

### 消融实验

| 配置 | 100ms 内 TTFT 比例 | 说明 |
|------|---------------|------|
| 完整模型 | 100% | baseline |
| 去掉模型预热 | 15% | 性能崩溃 |
| 去掉放置算法 | 29% | 干扰剧增 |
| 去掉主动预热 | 88% | 仍改进但比完整差 32.87× |
| 预热窗口 3 分钟 | 46% | 窗口过小预测不稳定 |
| 预热窗口 40 分钟 | 30% | 窗口过大无法捕捉短期变化 |

### 关键发现
- 模型预热提供数十倍 TTFT 改进。
- 主动预热策略带来改进最显著（高达 32 倍）。
- 放置算法在高负载下防止模型干扰雪崩。
- 5 分钟预热窗口最优。
- 工作负载预测：5 分钟窗口下平均负载预测精度 94.7%，峰值 92.7%。

## 亮点与洞察
- **发现 LLM 工作负载的长期周期性**：打破"LLM 请求完全不可预测"的认知。
- **一次加载多模型的创新**：完美结合资源效率与性能优势。
- **KV 缓存的双重用途**：将 KV 缓存从单纯激活值存储拓展为预热的临时存储。
- **贪心放置算法的优先级隔离思想**：简单高效，运行时无需求解复杂整数规划。

## 局限与展望
- 工作负载预测的适用边界——对完全新模型或特殊业务事件可能失效。
- 多数据中心/多租户场景缺失。
- 模型尺寸差异的处理不足——并置 7B + 70B 时效果有限。
- 改进：融合在线学习；多模型集成预测；详细分析预热失败率和能耗影响。

## 相关工作与启发
- **vs ServerlessLLM/MuxServe**：WarmServe 通过预热中间层在两者之间找到新的设计空间。
- **vs serverless 预热**：WarmServe 特化于 LLM 的三大挑战（跨多 GPU 依赖、极端模型尺寸、KV 缓存）。
- **vs KV 缓存优化**：利用缓存未用空间作为预热临时存储，体现"充分利用系统资源"哲学。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐  识别 LLM 工作负载长期可预测性，创新的一次加载多模型机制。
- 实验充分度: ⭐⭐⭐⭐  单机 + 大规模模拟 + 消融 + 预测精度验证。
- 写作质量: ⭐⭐⭐⭐⭐  逻辑清晰，动机递进自然。
- 价值: ⭐⭐⭐⭐⭐  50× TTFT 改进具实际部署价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ICML 2026\] SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm](siamesenorm_breaking_the_barrier_to_reconciling_prepost-norm.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ICML 2026\] Sparser Block-Sparse Attention via Token Permutation](sparser_block-sparse_attention_via_token_permutation.md)

</div>

<!-- RELATED:END -->
