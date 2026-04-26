---
title: >-
  [论文解读] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation
description: >-
  [ACL 2025 (Oral)][模型压缩][KV缓存压缩] 提出 SCOPE 框架，针对长上下文生成任务分别优化 prefill 和 decoding 阶段的 KV 缓存压缩策略——prefill 阶段保留完整缓存以维持理解能力，decoding 阶段采用滑动窗口选择 heavy hitters，并通过 adaptive 和 discontinuous 策略进一步优化内存和传输效率。
tags:
  - ACL 2025 (Oral)
  - 模型压缩
  - KV缓存压缩
  - 长上下文生成
  - 注意力机制
  - 推理优化
  - 内存效率
---

# SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation

**会议**: ACL 2025 (Oral)  
**arXiv**: [2412.13649](https://arxiv.org/abs/2412.13649)  
**代码**: [https://github.com/Linking-ai/SCOPE](https://github.com/Linking-ai/SCOPE)  
**领域**: LLM效率  
**关键词**: KV缓存压缩, 长上下文生成, 注意力机制, 推理优化, 内存效率  

## 一句话总结

提出 SCOPE 框架，针对长上下文生成任务分别优化 prefill 和 decoding 阶段的 KV 缓存压缩策略——prefill 阶段保留完整缓存以维持理解能力，decoding 阶段采用滑动窗口选择 heavy hitters，并通过 adaptive 和 discontinuous 策略进一步优化内存和传输效率。

## 研究背景与动机

**领域现状**：大语言模型在推理时需要维护 Key-Value (KV) 缓存来存储历史 token 的注意力信息。随着输入上下文和生成长度的增加，KV 缓存的内存占用成为推理效率的核心瓶颈。目前已有许多 KV 缓存压缩方法，如 SnapKV、PyramidKV、H2O、StreamingLLM 等。

**现有痛点**：现有方法几乎全部聚焦于 prefill 阶段的压缩（即处理输入阶段），而忽略了 decoding 阶段（即生成输出阶段）的优化。这在长输出生成任务中尤为突出——数学推理、代码生成等任务需要生成很长的输出序列，decoding 阶段的 KV 缓存持续增长。

**核心矛盾**：作者发现两个关键现象：(1) prefill 阶段过度压缩会损害模型对完整上下文的理解能力，尤其在复杂推理任务中，模型需要完整的上下文信息来规划推理路径；(2) 在长输出推理任务中，heavy hitters（高注意力分值 token）的分布会随着生成长度的增加而发生偏移，固定的 heavy hitter 选择在 decoding 后期失效。

**本文目标**：设计一个同时优化 prefill 和 decoding 两个阶段的 KV 缓存压缩框架，在保持推理质量的同时大幅降低内存消耗。

**切入角度**：通过可视化注意力热图，作者清晰地展示了 heavy hitter 偏移现象——早期 decoding 关注的 token 和晚期关注的 token 有显著差异。这意味着需要一个动态的 heavy hitter 选择机制来适应 decoding 过程中注意力分布的变化。

**核心 idea**：prefill 不压缩（保留完整上下文理解），decoding 用滑动窗口动态追踪 heavy hitters，两阶段分别处理。

## 方法详解

### 整体框架

SCOPE 是一个两阶段（Stage-level）KV 缓存压缩框架。在 prefill 阶段，保留完整的 KV 缓存以维持模型对输入的充分理解。在 decoding 阶段，引入基于滑动窗口的策略动态选择最重要的 heavy hitters，同时用 adaptive 和 discontinuous 策略进一步降低内存占用和传输开销。SCOPE 被设计为一个即插即用（plug-in）组件，可以与已有的 prefill-only 压缩方法（如 SnapKV、PyramidKV）兼容叠加。

### 关键设计

1. **Prefill 阶段全保留策略（Full Preservation）**:

    - 功能：在 prefill 阶段保留完整的 KV 缓存
    - 核心思路：与现有方法不同，SCOPE 主张在 prefill 阶段不做任何压缩。理由是推理任务（如数学推理）需要模型对完整输入上下文有全局理解，才能制定正确的推理策略。实验表明 prefill 阶段的压缩对推理准确率有不成比例的负面影响。当然，SCOPE 也支持与其他 prefill 压缩方法叠加使用——此时 SCOPE 仅负责 decoding 阶段的优化。
    - 设计动机：作者通过实验发现，在 LongGenBench 上保留全部 prefill KV cache 比压缩后的结果好很多，特别是在需要多步推理的 GSM8K+ 等任务上。

2. **基于滑动窗口的 Heavy Hitter 选择（Sliding Window Selection）**:

    - 功能：在 decoding 阶段动态选择最相关的 KV 缓存
    - 核心思路：维护一个固定大小的滑动窗口来追踪近期的 heavy hitters。每生成一个新 token 后，根据最新的注意力分布重新计算 heavy hitter 得分，并更新窗口内保留的 KV 对。滑动窗口机制使得 heavy hitter 的选择能够紧跟注意力分布的偏移——随着推理链的深入，模型关注的 token 自然会发生变化，滑动窗口保证了这种变化被正确追踪。
    - 设计动机：实验中观察到 heavy hitter 偏移现象：推理前期模型关注问题描述中的关键信息，后期则更关注中间推理步骤。固定的 heavy hitter 集合无法适应这种变化。

3. **Adaptive 与 Discontinuous 优化策略**:

    - 功能：进一步优化内存使用和数据传输效率
    - 核心思路：**Adaptive 策略**根据当前缓存的实际使用情况动态调整保留预算——当某些注意力头的 heavy hitters 集中度低时分配更多预算，集中度高时减少预算。**Discontinuous 策略**允许保留非连续的 KV 缓存片段，不要求保留的 token 在序列上连续——这样可以只保留真正重要的 token 而跳过不相关的部分，减少了内存碎片但需要特殊的 gather 操作。
    - 设计动机：统一大小的预算分配不够灵活（不同层和头的重要 token 分布差异大），强制连续存储浪费了预算在不重要的 token 上。

### 损失函数 / 训练策略

SCOPE 是一个免训练（training-free）的推理加速方法，不涉及模型训练或微调。所有策略都在推理时应用。

## 实验关键数据

### 主实验

在 LongGenBench（4K 和 8K 两个设置）上评估，使用 Llama3.1-8B-Instruct，任务包括 GSM8K+、MMLU+、CSQA+。

| 方法 | 阶段优化 | GSM8K+ (4K) | MMLU+ (4K) | CSQA+ (4K) | 平均 |
|------|---------|-------------|------------|------------|------|
| Full KV (无压缩) | — | 基线 | 基线 | 基线 | 基线 |
| H2O | prefill | 明显下降 | 中等下降 | 轻微下降 | 下降 |
| SnapKV | prefill | 中等下降 | 轻微下降 | 轻微下降 | 中等下降 |
| StreamingLLM | prefill | 大幅下降 | 大幅下降 | 大幅下降 | 大幅下降 |
| **SCOPE (slide)** | decoding | 接近基线 | 接近基线 | 接近基线 | **最佳** |
| **SCOPE (adaptive)** | decoding | 接近基线 | 接近基线 | 接近基线 | **最佳** |
| SnapKV + SCOPE | prefill + decoding | 优于SnapKV | 优于SnapKV | 优于SnapKV | 显著提升 |

### 消融实验

| 配置 | GSM8K+ Acc | 说明 |
|------|-----------|------|
| Full KV (无压缩) | 最高 | 上界 |
| SCOPE (slide only) | 接近Full KV | 基础滑动策略已非常有效 |
| SCOPE (slide + adaptive) | 略优于slide | adaptive 分配更精细 |
| SCOPE (slide + discontinuous) | 略优于slide | 允许非连续保留更灵活 |
| SCOPE (full) | 最优 | 三种策略组合 |
| 仅 H2O decoding | 明显不如SCOPE | 固定 heavy hitter 不如滑动 |
| 不同窗口大小 | 随窗口增大提升 | 256-512 为性价比最优区间 |

### 关键发现

- **SCOPE 作为 plug-in 效果显著**：将 SCOPE 叠加到 SnapKV 等 prefill 方法上后，性能显著优于单独使用 prefill 方法，验证了 decoding 阶段优化的必要性
- **heavy hitter 偏移确实存在**：通过注意力热图可视化，不同 decoding 阶段关注的 token 差异很大，验证了滑动窗口设计的合理性
- **GSM8K+ 等推理任务受 KV 压缩影响最大**：需要多步推理的任务对 KV 缓存的完整性最敏感，SCOPE 的优势在此类任务上最明显
- **内存效率**：SCOPE 在几乎不损失准确率的情况下显著减少了 decoding 阶段的 KV 缓存内存占用

## 亮点与洞察

- **两阶段分治的思路非常清晰**：将 prefill 和 decoding 的 KV 缓存管理解耦，各自采用最优策略。这个思路简洁有效，而且之前很少有工作关注 decoding 阶段的压缩。
- **heavy hitter 偏移的发现有启发性**：这个不仅对 KV 压缩有意义，对注意力机制的理解也有价值——说明推理过程中模型的信息需求是动态变化的。
- **即插即用设计**：SCOPE 可以与任何已有的 prefill 压缩方法叠加使用，降低了迁移成本。这种"增量式"的设计思路值得借鉴——不是要替代已有方法，而是补充已有方法忽略的部分。

## 局限与展望

- 评估主要基于 LongGenBench 单一benchmark，在其他长上下文任务上的泛化性还需验证
- Discontinuous 策略的 gather 操作可能引入额外的计算开销，在不同硬件上的加速比需要实测
- 滑动窗口大小的选择需要人工调参，尚未实现全自动的预算分配
- 未来可以结合量化（如 KV cache quantization）和 SCOPE 进行联合优化，进一步压缩内存
- 对于非常长的输出（如 >16K tokens），滑动窗口的策略是否仍然有效还需要验证

## 相关工作与启发

- **vs SnapKV / PyramidKV**：这些方法只优化 prefill 阶段的 heavy hitter 选择，忽略了 decoding 阶段。SCOPE 补充了这个缺失环节，且两者可以叠加使用。
- **vs H2O**：H2O 虽然也有 decoding 阶段的 eviction 策略，但使用固定的累积注意力分数来选择 heavy hitters。SCOPE 的滑动窗口设计更好地适应了 heavy hitter 偏移现象。
- **vs StreamingLLM**：StreamingLLM 采用极端的只保留 sink token + 最近 token 的策略，在推理任务上损失较大。SCOPE 在保留重要 token 方面更精细。

## 评分

- 新颖性: ⭐⭐⭐⭐ 两阶段分治和 heavy hitter 偏移的发现是核心贡献，但单个技术（滑动窗口、adaptive 等）并不新颖
- 实验充分度: ⭐⭐⭐⭐ LongGenBench 上的实验覆盖了多个任务和基线，plug-in 实验有说服力。但缺少更多 benchmark 和实际延迟/吞吐数据
- 写作质量: ⭐⭐⭐⭐ 动机部分的两个观察讲得很好，方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 作为 ACL 2025 Oral，填补了 decoding 阶段 KV 压缩的空白，实用价值高

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](../../NeurIPS2025/model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](../../NeurIPS2025/model_compression/homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](../../ICML2025/model_compression/rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)

<!-- RELATED:END -->
