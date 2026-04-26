---
title: >-
  [论文解读] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference
description: >-
  [ACL 2026][模型压缩][KV缓存压缩] 提出ASL（Adaptive Selection Layer），通过监控token注意力分数排名的方差来自适应确定KV缓存剪枝的层位置，在困难任务上显著优于固定层选择方法，同时保持无需训练。
tags:
  - ACL 2026
  - 模型压缩
  - KV缓存压缩
  - 自适应层选择
  - 注意力剪枝
  - 长上下文推理
  - 无训练方法
---

# Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference

**会议**: ACL 2026  
**arXiv**: [2601.07667](https://arxiv.org/abs/2601.07667)  
**代码**: [GitHub](https://github.com/TANIGUCHIREI/ASL)  
**领域**: Model Compression / KV Cache Optimization  
**关键词**: KV缓存压缩, 自适应层选择, 注意力剪枝, 长上下文推理, 无训练方法

## 一句话总结

提出ASL（Adaptive Selection Layer），通过监控token注意力分数排名的方差来自适应确定KV缓存剪枝的层位置，在困难任务上显著优于固定层选择方法，同时保持无需训练。

## 研究背景与动机

**领域现状**：KV缓存是LLM推理的主要内存瓶颈，层级token剪枝（在特定层选择重要token子集并剪除其余）是主流的压缩方案。

**现有痛点**：现有层级剪枝方法（如FastKV、GemFilter）使用预定义的固定选择层——这种设计对简单任务（如QA）有效，但在困难任务（如KV检索）上严重退化。原因是困难任务中问题与上下文的语义相似度高，早期层难以区分相关token。

**核心矛盾**：固定选择层面临根本性的trade-off——早选节省计算但损失精度，晚选保持精度但减少内存节省。不同任务的最优选择层差异巨大。

**本文目标**：设计一种自适应方法，根据任务难度自动确定最佳token选择层。

**切入角度**：观察到注意力分数排名在不同任务中收敛到稳定子集的速度不同——简单任务在中间层即稳定，困难任务需要更深层才稳定。

**核心 idea**：监控token排名的方差作为"注意力聚焦度"的指标，当方差降到阈值以下时触发token选择。

## 方法详解

### 整体框架

ASL在prefilling阶段运行：从第 $L_{min}$ 层开始，在每 $L_{obs}$ 个连续层上计算pooled注意力分数的排名方差。当相对方差低于用户指定阈值时，在该层执行one-shot token选择。后续可与SnapKV等方法联合优化decoding阶段。

### 关键设计

1. **基于排名方差的自适应选择**:

    - 功能：根据任务难度自动确定token剪枝的最佳层
    - 核心思路：计算pooled注意力分数 $PA = \text{pool}(\text{softmax}(\frac{\mathbf{q}_w \mathbf{k}_c + \mathbf{m}_w}{\sqrt{d}}))$，对连续 $L_{obs}$ 层的token排名计算方差，方差小说明注意力已稳定聚焦于固定子集
    - 设计动机：排名方差比原始注意力分数更稳健——不关心具体分数值，只关心"哪些token被关注"是否稳定

2. **阈值控制的自适应trade-off**:

    - 功能：让用户通过单一参数控制精度-效率平衡
    - 核心思路：用户指定阈值 $\theta$，方差降到 $\theta$ 以下即触发选择。$\theta$ 越高越早选择（更快但可能损失精度），$\theta$ 越低越晚选择（更精确但更慢）
    - 设计动机：不同应用场景对精度和速度的需求不同，单一参数控制比手动调整选择层更实用

3. **与现有方法的无缝集成**:

    - 功能：与SnapKV等方法联合优化整个推理管道
    - 核心思路：ASL优化prefilling阶段（确定选择层），SnapKV优化decoding阶段（压缩选择层之前的KV缓存）。也可与GemFilter结合使用两遍策略
    - 设计动机：ASL是正交改进，可以直接替换现有方法中的固定层选择组件

### 损失函数 / 训练策略

ASL完全无需训练，仅在推理时运行。两个超参数 $L_{min}$ 和 $L_{obs}$ 分别控制起始监控层和观察窗口大小。

## 实验关键数据

### 主实验

| 方法 | KV检索(困难) | QA(简单) | NIAH | 内存占用 |
|------|------------|---------|------|--------|
| FastKV(固定层) | 严重退化 | 强 | 中等 | 低 |
| GemFilter(固定层) | 退化 | 强 | 中等 | 低 |
| ASL(自适应) | 显著提升 | 保持 | 提升 | 可比 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 阈值敏感性 | 平滑过渡 | 不同阈值产生连续的精度-速度trade-off |
| 跨任务适应性 | InfiniteBench 10任务 | 不同任务自动选择不同深度的层 |
| 256K上下文 | 有效工作 | 长上下文场景同样适用 |

### 关键发现
- 简单任务（QA）注意力在中间层（~15层）即稳定，困难任务（KV检索）需要到更深层（~25层以上）
- ASL在困难任务上大幅超越固定层方法，同时在简单任务上保持相当性能
- 相对方差是有效的"任务难度探针"——无需预先知道任务类型即可自适应

## 亮点与洞察
- 将"什么时候选"的问题从超参数调优转化为自动检测，显著提升了实用性
- 观察驱动的方法设计——从注意力模式的跨层演化规律出发，逻辑链条清晰
- 完全无需训练，开箱即用，且与现有方法正交可组合

## 局限与展望
- 当前仅在Llama 3.1 8B上验证，需要在更多模型架构上测试
- 监控排名方差有一定计算开销（尽管很小），在极端低延迟场景可能需要优化
- 阈值的最优值仍需用户根据场景选择
- 未来可探索progressive版本——在多个自适应选择的层逐步剪枝

## 相关工作与启发
- **vs FastKV/GemFilter**: 用自适应选择替代固定层，从根本上解决任务敏感性问题
- **vs PyramidKV/DynamicKV**: 这些方法自适应分配预算但不自适应选择层，两者互补
- **vs SnapKV**: ASL优化prefilling阶段的层选择，SnapKV优化decoding阶段的token保留，可组合使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 排名方差作为任务难度探针的想法简洁有效
- 实验充分度: ⭐⭐⭐⭐ 多benchmark、多上下文长度的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 观察→动机→方法→验证的逻辑链条非常清晰
- 价值: ⭐⭐⭐⭐ 对LLM长上下文推理优化有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](a_layer-wise_analysis_of_supervised_fine-tuning.md)
- [\[CVPR 2026\] FAIR-Pruner: Leveraging Tolerance of Difference for Flexible Automatic Layer-Wise Neural Network Pruning](../../CVPR2026/model_compression/fair-pruner_leveraging_tolerance_of_difference_for_flexible_automatic_layer-wise.md)
- [\[ICML 2025\] OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM Inference](../../ICML2025/model_compression/orthorank_token_selection_via_sink_token_orthogonality_for_efficient_llm_inferen.md)
- [\[NeurIPS 2025\] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](../../NeurIPS2025/model_compression/dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)
- [\[ACL 2026\] CBRS: Cognitive Blood Request System with Bilingual Dataset and Dual-Layer Filtering](cbrs_cognitive_blood_request_system_with_bilingual_dataset_and_dual-layer_filter.md)

<!-- RELATED:END -->
