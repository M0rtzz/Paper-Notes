---
title: >-
  [论文解读] Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling
description: >-
  [NeurIPS 2025][模型压缩][长上下文微调] 针对长上下文监督微调（Long-SFT）中长短序列混合导致的训练效率低下问题，提出动态数据调度器Skrull，通过分布感知的上下文并行（DACP）和全局数据调度（GDS）两个组件，在真实Long-SFT场景中实现平均3.76倍（最高7.54倍）的训练加速。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "长上下文微调"
  - "数据调度"
  - "上下文并行"
  - "训练效率"
  - "大语言模型"
---

# Skrull: Towards Efficient Long Context Fine-tuning through Dynamic Data Scheduling

**会议**: NeurIPS 2025  
**arXiv**: [2505.19609](https://arxiv.org/abs/2505.19609)  
**代码**: 暂无  
**领域**: 模型压缩  
**关键词**: 长上下文微调, 数据调度, 上下文并行, 训练效率, 大语言模型

## 一句话总结

针对长上下文监督微调（Long-SFT）中长短序列混合导致的训练效率低下问题，提出动态数据调度器Skrull，通过分布感知的上下文并行（DACP）和全局数据调度（GDS）两个组件，在真实Long-SFT场景中实现平均3.76倍（最高7.54倍）的训练加速。

## 研究背景与动机

长上下文能力对LLM至关重要，主流模型（Llama3、Qwen2.5、GPT-4）支持128K甚至1M tokens的上下文窗口。要获得这种能力，通常需要在混合长短序列的数据集上进行Long-SFT。例如Llama3的微调数据中99.89%是短序列（平均1K tokens），仅0.11%是长序列（平均37K tokens）。

然而，这种**异构序列长度分布**给现有训练框架带来了两难困境：

**并行策略困境**：长序列需要上下文并行（CP）和其他内存缩减策略来避免OOM，但这些策略对短序列引入了不必要的通信开销和GPU利用率低下。实验表明，CP度越高，短序列的kernel执行效率下降越严重。

**负载均衡困境**：Attention模块的$O(n^2)$计算复杂度与$O(n)$的内存消耗使得计算均衡和内存均衡无法同时满足。

**GPU利用率低**：为容纳最长序列而设置的内存缩减策略导致处理短序列时GPU内存严重空闲。

现有框架（DeepSpeed、Megatron）对所有序列采用统一的并行配置，无法同时高效处理长短序列，导致端到端性能次优。

## 方法详解

### 整体框架

Skrull从数据调度的角度切入，包含两个层次的调度机制：（1）DACP在micro-batch级别进行细粒度调度，选择性地对序列进行分片；（2）GDS在global batch级别进行粗粒度调度，生成最优的micro-batch划分方案。两者协同工作，通过离线性能建模和在线轻量级启发式算法实现近零开销的运行时调度。

### 关键设计

1. **分布感知上下文并行（DACP）**：核心思想是将micro-batch中的序列动态分为两类——**分布式序列**（需要CP分片的长序列）和**本地序列**（完全在单个设备上处理的短序列）。两类序列在同一CP组内处理，不增加GPU数量。关键优势在于：本地序列避免了不必要的通信开销，且DACP可以将分布式序列的通信与本地序列的计算重叠（overlap），进一步隐藏通信时延。调度目标形式化为优化问题：
$$\min_{D,P} \max_j \text{Time}_j, \quad \text{Time}_j = \max(T_{comm}(V), T_{comp}(\text{Local}_j)) + T_{comp}(\text{Dist})$$
其中$D$是序列分类数组（分布式/本地），$P$是本地序列的设备分配矩阵，约束条件包括内存限制$\sum_k S_k \cdot P_{kj} + D_k \cdot S_k / N \leq C$。

2. **全局数据调度（GDS）**：仅依赖DACP不够——异构长度也导致micro-batch间的负载不均衡。GDS在global batch范围内（保持优化器数学等价性）进行粗粒度的batch划分。通过将长短序列配对打包到micro-batch中，既扩展了DACP的优化空间，又改善了内存利用率。联合优化公式为：
$$\min_{B,D,P} \max_i \sum_j \text{Time}_{ij}$$
其中$B_{kij}$表示第$k$个序列分配到第$i$个DP rank的第$j$个micro-batch。

3. **轻量级启发式调度算法**：由于精确求解MILP问题耗时太长，作者设计了集成到DataLoader中的启发式算法，引入近零开销。DACP调度遵循三原则：**避免分片**（优先本地处理）、**优先计算均衡**、**回滚机制**（内存超限时回滚决策）。GDS调度采用基于FLOPs估计的bin-packing算法平衡DP worker间工作量，并通过交错排列长短序列优化micro-batch构成。

### 损失函数 / 训练策略

Skrull的核心贡献在系统层面而非算法层面。通过离线profiling建立性能模型：内存消耗$Memory(S) = \alpha S + \beta$（线性于序列长度），计算量$FLOPs(S_k) = 20bh^2 S_k + 4bhh_{kv}S_k + 4bhS_k^2$（Linear模块线性+Attention模块二次）。Skrull不改变任何训练内容和全局batch的序列顺序，仅改变累加顺序，由于浮点非结合性仅产生微小数值差异，收敛行为完全等价。

## 实验关键数据

### 主实验——训练加速效果

| 模型 | 数据集 | Skrull vs DeepSpeed | Skrull vs Sorted Batching |
|------|--------|---------------------|--------------------------|
| Qwen2.5-0.5B | Wikipedia | ~7.54× | ~6.85× |
| Qwen2.5-0.5B | LMsysChat1M | ~6.17× | ~5.40× |
| Qwen2.5-0.5B | ChatQA2-Long-SFT | ~2.79× | ~1.86× |
| Qwen2.5-7B | Wikipedia | ~2.60× | ~2.30× |
| Qwen2.5-7B | LMsysChat1M | ~2.14× | ~1.90× |
| Qwen2.5-7B | ChatQA2-Long-SFT | ~1.35× | ~1.20× |
| **平均** | — | **3.76×** | **3.45×** |

### 消融实验——组件贡献与参数影响

| 调度策略 | 加速比 | 说明 |
|---------|--------|------|
| Round-Robin w/ 回滚 | 1.17× | 简单轮询无法有效均衡 |
| Round-Robin w/o 回滚 | OOM | 缺失回滚机制导致内存溢出 |
| Skrull w/ 回滚 | 1.40× | 启发式调度显著优于轮询 |
| Skrull w/o 回滚 | OOM | 回滚机制是安全保障的关键 |

### 关键发现

- Qwen-0.5B的平均加速（5.50×）显著高于Qwen-7B（2.03×），因为小模型的BucketSize更大，调度空间更充裕
- 长尾分布数据集（Wikipedia、LMsysChat1M）的优化空间大于双峰分布（ChatQA2-Long-SFT）
- BatchSize增大带来持续但逐渐饱和的加速；BucketSize增大提升性能但也增加OOM风险
- DACP和GDS的逐步启用验证了两者的独立有效性和协同增益
- 回滚机制是**必需的**安全保障，无回滚均导致OOM

## 亮点与洞察

- **视角新颖**：从数据调度（而非模型/算法优化）角度解决Long-SFT效率问题，提供了一个正交于现有方法的优化维度
- **理论与实践平衡**：严谨地将调度过程形式化为联合优化问题，同时提出实用的启发式近似
- **回滚机制设计精妙**：在贪心调度中引入可逆操作防止OOM，体现了系统设计的工程智慧
- 不改变训练语义，收敛行为完全等价，可与PEFT等其他技术正交组合

## 局限与展望

- BucketSize需手动配置，当前依赖离线profiling
- 对ChatQA2-Long-SFT + Qwen-7B场景的加速有限（1.35×），因大多数序列本身就超过BucketSize
- 启发式算法虽然实用但非最优，可能存在进一步优化空间
- 目前仅在DeepSpeed上实现，尚未适配Megatron-LM等其他框架
- 可与LoRA等PEFT方法结合使用以扩大BucketSize，但未在实验中充分验证

## 相关工作与启发

与LongAlign（sorted batching）、Chunkflow（固定大小chunk训练）、HotSPA（动态并行配置）等工作相关。Skrull采用固定并行配置+动态数据调度的方案，与动态并行方法正交。其数据调度思路可广泛应用于任何长短数据混合的训练场景，如RLHF。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 数据调度视角新颖，联合优化建模严谨
- **实验充分度**: ⭐⭐⭐⭐ 多模型多数据集验证，逐步消融充分，但缺少更多模型规模的测试
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，数学建模严谨，但符号较多需仔细阅读
- **价值**: ⭐⭐⭐⭐⭐ 解决了Long-SFT中一个重要且普遍的系统效率问题，平均3.76×加速实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)

</div>

<!-- RELATED:END -->
