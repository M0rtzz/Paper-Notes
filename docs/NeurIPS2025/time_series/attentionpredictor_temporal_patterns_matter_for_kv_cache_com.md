---
title: >-
  [论文解读] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression
description: >-
  [NeurIPS 2025][时间序列][KV缓存压缩] AttentionPredictor是首个学习型方法直接预测注意力模式以实现KV缓存压缩和关键token识别，通过轻量CNN捕捉注意力分数的时空模式，实现13倍KV缓存压缩和5.6倍推理加速，统一预测模型仅21KB可跨所有Transformer层共享。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "KV缓存压缩"
  - "注意力预测"
  - "时序模式"
  - "LLM推理加速"
  - "缓存预取"
---

# AttentionPredictor: Temporal Patterns Matter for KV Cache Compression

**会议**: NeurIPS 2025  
**arXiv**: [2502.04077](https://arxiv.org/abs/2502.04077)  
**代码**: [GitHub](https://github.com/MIRALab-USTC/LLM-AttentionPredictor)  
**领域**: 时间序列 / 高效推理  
**关键词**: KV缓存压缩, 注意力预测, 时序模式, LLM推理加速, 缓存预取

## 一句话总结
AttentionPredictor是首个学习型方法直接预测注意力模式以实现KV缓存压缩和关键token识别，通过轻量CNN捕捉注意力分数的时空模式，实现13倍KV缓存压缩和5.6倍推理加速，统一预测模型仅21KB可跨所有Transformer层共享。

## 研究背景与动机

**领域现状** LLM长上下文推理中KV缓存是主要内存瓶颈（7B模型在128K上下文下需72GB）。稀疏注意力方法通过仅保留"关键token"来压缩缓存。

**现有痛点** (1) 启发式方法（H2O、SnapKV）使用静态规则评估token重要性，无法捕捉动态注意力变化；(2) 学习型方法（SeerAttention）需要每层训练独立模型（总计101MB），不直接建模注意力分数分布。

**核心矛盾** 注意力模式具有复杂的动态时序特性（re-access、sequential、seasonal），但现有方法要么用简单启发式（不够准），要么用仅编码key/hidden的学习方法（不够直接）。

**本文目标** 直接预测下一步的注意力分数分布，以此精确识别关键token实现高压缩比。

**切入角度** 注意力分数在时间轴上具有可预测的时空规律（源于query自相似性和位置编码的固有属性），可建模为2D时间序列预测问题。

**核心 idea** 将KV缓存压缩形式化为注意力分数的2D时序预测问题，用轻量CNN预测器替代启发式规则来识别关键token。

## 方法详解

### 整体框架
AttentionPredictor的核心流程：(1) 预填充阶段准备注意力历史序列 $\mathcal{A}_H$；(2) 解码阶段每步：对当前注意力分数做block-wise max-pooling压缩→更新历史序列→用预训练CNN预测下一步注意力→选出Top-K关键token位置→扩展为最终索引集 $S$。

### 关键设计

1. **注意力时序模式发现与理论分析**:
    - 功能：揭示注意力分数的三种可预测时序模式
    - 核心思路：(1) **Re-access** $(\delta_t=1,\delta_i=0)$: 持续关注固定token；(2) **Sequential** $(\delta_t=1,\delta_i=1)$: 注意力逐token递进（源于RoPE的相对位置依赖）；(3) **Seasonal** $(\delta_t>1,\delta_i=0)$: 周期性重访固定位置。统一描述为 $a_{t,i} \approx a_{t+\delta_t, i+\delta_i}$
    - 设计动机：理论证明query高自相似性（余弦自相关0.87）和RoPE位置编码是这些模式的内在来源，使得时序预测可行

2. **轻量时空CNN预测器**:
    - 功能：从注意力历史预测下一步注意力分布
    - 核心思路：两层2D卷积捕捉多尺度时空特征 + 一层1D卷积聚焦时间维度。用1D卷积替代全连接层使模型适应可变空间维度。整个模型仅**21KB**（LLaMA-3.1-8B的百万分之一），**所有层和头共享同一个预测器**
    - 设计动机：注意力时序模式是LLM的固有属性（query相似性+位置编码），不因层/头/数据集变化，因此可以用单一轻量模型统一捕捉

3. **跨token KV缓存预取框架**:
    - 功能：隐藏预测和缓存传输延迟以加速解码
    - 核心思路：与现有跨层预取不同，AttentionPredictor利用当前token推理时间预测下一token的关键缓存索引并异步从CPU预取对应KV——跨token预取可利用更长的传输时间窗口
    - 设计动机：跨层方法的预取窗口仅为单层推理时间，不够隐藏大规模传输；跨token方法的窗口是完整的单token推理时间

### 附加技术
- **Block-wise压缩**：max-pooling将注意力向量压缩 $1/b$ 倍（$b=16$），减少预测计算
- **分布误差校准**：每 $M=5$ 步计算一次完整注意力来校正稀疏计算的分布偏移

## 实验关键数据

### 主实验——LongBench（LLaMA-3.1-8B，1K/2K/4K预算）

| 方法 | 1K平均 | 2K平均 | 4K平均 |
|------|--------|--------|--------|
| Full cache | 48.17 | 48.17 | 48.17 |
| StreamingLLM | 42.35 | 41.64 | - |
| H2O+ | 44.25 | 46.34 | - |
| SnapKV | 46.37 | 47.44 | - |
| Quest | 46.92 | - | - |
| **AttentionPredictor** | **48.58** | **48.82** | - |

### 性能数据

| 指标 | 数值 |
|------|------|
| KV缓存压缩比 | 13× |
| 缓存卸载场景加速比 | 5.6× |
| 预测模型大小 | **21KB**（LLM的百万分之一） |
| SeerAttention模型大小 | 101MB（AttentionPredictor的0.02%） |

### 消融实验

| 配置 | 注意力恢复率 | 说明 |
|------|-------------|------|
| H2O (启发式) | 较低 | 累积分数有偏 |
| Quest (压缩检索) | 中等 | 大page size下降 |
| AttentionPredictor | **最高** | 直接预测最准确 |
| +校准 (M=5) | 进一步提升 | 修正稀疏偏移 |

### 关键发现
- 在1K预算的极端压缩下，AttentionPredictor几乎无损（48.58 vs Full 48.17）
- 预测精度比启发式方法高5%+ ,且比固定模板方法（MInference）也高5%
- 仅用约3%的注意力数据训练即可泛化到全数据集（LongBench训练→GSM8K泛化）

## 亮点与洞察
- "注意力预测是时间序列预测"的视角新颖且理论支撑充分（query自相似性+RoPE）
- 21KB模型跨层共享是极致轻量的设计，对比SeerAttention的101MB体现了范式优势
- 跨token预取框架相比跨层预取有更大的延迟隐藏窗口，是实用的系统优化

## 局限与展望
- block-wise压缩可能在需要精细token级区分的任务上牺牲精度
- 校准步每M步需计算完整注意力，在极长序列上仍有固定成本
- 目前仅在7B-8B模型上验证，更大模型上的性能待确认

## 相关工作与启发
- **vs H2O/SnapKV**: 启发式累积分数，AttentionPredictor用学习预测器捕捉动态变化
- **vs SeerAttention**: 每层独立模型+间接编码key，AttentionPredictor单一21KB模型直接预测注意力
- **vs InfiniGen**: 跨层预取窗口短，AttentionPredictor的跨token预取有更大时间预算

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个直接预测注意力模式的学习型压缩方法，理论分析深入
- 实验充分度: ⭐⭐⭐⭐⭐ LongBench/InfiniteBench/AIME/GSM8K/MMLU多基准全面评估
- 写作质量: ⭐⭐⭐⭐ 理论分析和系统设计相辅相成
- 价值: ⭐⭐⭐⭐⭐ 对长上下文LLM推理有直接工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] How Patterns Dictate Learnability in Sequential Data](how_patterns_dictate_learnability_in_sequential_data.md)
- [\[CVPR 2025\] FLAVC: Learned Video Compression with Feature Level Attention](../../CVPR2025/time_series/flavc_learned_video_compression_with_feature_level_attention.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series](syntsbench_rethinking_temporal_pattern_learning_in_deep_learning_models_for_time.md)

</div>

<!-- RELATED:END -->
