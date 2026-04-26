---
title: >-
  [论文解读] RefreshKV: Updating Small KV Cache During Long-form Generation
description: >-
  [ACL 2025][LLM效率][KV缓存压缩] 提出RefreshKV推理方法，通过在生成过程中周期性地在全KV缓存注意力和小KV缓存注意力之间交替，并基于全注意力步的注意力模式动态更新小KV缓存，在不永久丢弃任何token的前提下，实现与驱逐式方法相当的加速且大幅提升长文本生成任务性能。
tags:
  - ACL 2025
  - LLM效率
  - KV缓存压缩
  - 长文本生成
  - 稀疏注意力
  - 动态缓存刷新
  - 推理加速
---

# RefreshKV: Updating Small KV Cache During Long-form Generation

**会议**: ACL 2025  
**arXiv**: [2411.05787](https://arxiv.org/abs/2411.05787)  
**代码**: [https://github.com/carriex/refreshkv](https://github.com/carriex/refreshkv)  
**领域**: LLM效率  
**关键词**: KV缓存压缩, 长文本生成, 稀疏注意力, 动态缓存刷新, 推理加速

## 一句话总结
提出RefreshKV推理方法，通过在生成过程中周期性地在全KV缓存注意力和小KV缓存注意力之间交替，并基于全注意力步的注意力模式动态更新小KV缓存，在不永久丢弃任何token的前提下，实现与驱逐式方法相当的加速且大幅提升长文本生成任务性能。

## 研究背景与动机
- **领域现状**：长上下文LLM推理的主要瓶颈在于KV缓存的线性内存增长和注意力的二次计算开销。现有方法（如StreamingLLM、H2O、SnapKV）通过永久驱逐"不重要"的token来维持小KV缓存
- **现有痛点**：
    - 驱逐式方法一旦移除token就无法恢复，这在**短文本生成**任务中影响较小
    - 但在**长文本生成**任务中，性能急剧退化——因为不同生成步骤需要的重要token会动态变化
    - 例如：SnapKV和H2O在HTML转TSV任务中F1直接为0，完全失败
- **核心矛盾**：过早且永久地驱逐token vs. 维持全KV缓存的高计算开销
- **切入角度**：保留全KV缓存（不牺牲内存），但大部分解码步只在小KV缓存上计算注意力，周期性地用全注意力步刷新小缓存
- **核心idea**：连续token的注意力模式相似，因此可以用最近一次全注意力步的注意力分数来构建后续若干步的小KV缓存

## 方法详解

### 整体框架
RefreshKV维护两个KV缓存：
- $C_f$（全缓存）：包含所有token的KV，容量为L
- $C_p$（部分缓存）：仅包含topK重要token的KV，容量远小于L

交替执行两种模式：
1. **部分注意力步**：用 $C_p$ 计算注意力生成token（低延迟）
2. **全注意力步**：用 $C_f$ 计算注意力生成token，并根据注意力分数重新构建 $C_p$（刷新）

### 关键设计
1. **部分缓存构建与刷新**:

    - 初始化：预填充阶段用最后一个token的注意力分数选topK token
    - 刷新：每次全注意力步后，用当前步的注意力分数重新选topK token，替换整个 $C_p$
    - 使用max pooling聚合周围token的注意力分数（而非原始分数），保持信息完整性
    - 对GQA模型，在同一group内的query head间取max聚合

2. **自适应调度策略（Dynamic Stride）**:

    - 问题：固定步长（每N步全注意力）对不同层和输入不够灵活
    - 方案：基于**query向量相似度**决定是否触发全注意力
    - 具体：每隔S步（QC stride），计算当前query向量与最近全注意力步的query向量的余弦相似度
    - 若相似度 > 阈值 $s$ → 沿用部分缓存（注意力模式未变）
    - 若相似度 ≤ $s$ → 触发全注意力并刷新缓存
    - **逐层独立决策**：每层可有不同的全注意力触发频率

3. **部分缓存的维护**:

    - 在部分注意力步中，新生成token的KV添加到 $C_p$
    - 为保持固定大小，移除 $C_p$ 中全注意力步时注意力分数最低的token的KV
    - 关键区别：若不刷新，这就退化为SnapKV

4. **Continued Pre-training**:

    - 动机：LLM训练时用全注意力，推理时用稀疏注意力产生训练-推理不一致
    - 方案：在训练时模拟RefreshKV的注意力模式（前L个token全注意力，后S个token部分注意力）
    - 简化训练：固定stride=50，不刷新部分缓存
    - 在Arxiv子集上预训练120k样本，短上下文(8K)训练的收益可迁移到长上下文(16K)

### 损失函数 / 训练策略
- 推理方法本身无需训练
- Continued Pre-training使用标准next token prediction loss
- 在所有token上计算loss，前L个用全注意力，后S个用部分注意力

## 实验关键数据

### 主实验

| 数据集 | 指标 | RefreshKV(QC=5) | SnapKV | H2O | Vanilla |
|--------|------|------|----------|------|------|
| Arxiv | PPL↓ | **2.27** | 2.54 | 2.48 | 2.22 |
| Book | PPL↓ | **7.31** | 7.78 | 7.60 | 7.07 |
| HTML→TSV | F1↑ | **17** | 0 | 0 | 33 |
| GovReport | R-L↑ | **32.56** | 28.06 | 27.41 | 34.11 |
| Chain-of-key | Acc↑ | **25** | 12 | 10 | 56 |
| RULER | Acc↑ | **86** | 79 | 21 | 90 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| w/o refresh（只做偶尔全注意力） | PPL=2.50, HTML=0 | 性能接近SnapKV，说明核心收益来自刷新而非偶尔全注意力 |
| w/o full attention（全注意力步也只用刷新后的部分缓存） | PPL=2.32, HTML=16 | 与完整RefreshKV一致，进一步证实刷新是关键 |
| Dynamic vs Fixed stride | 多任务 | 动态步长在相似或更少的全注意力步数下，一致性地优于固定步长 |
| Continued Pre-training | PPL | 各stride都有小幅提升，RefreshKV获益最大（如stride=50: 3.13→3.05） |

### 关键发现
- 驱逐式方法（SnapKV、H2O）在长文本生成任务上彻底失败（HTML→TSV为0分）
- RefreshKV恢复了Vanilla性能的52%（HTML任务），是唯一能在Chain-of-key上生成2+key链的方法
- 随时间推移，SnapKV的PPL相对Vanilla持续恶化，而RefreshKV保持稳定比率
- 动态步长在不同任务上自动调节有效stride（如GovReport stride≈14 vs Book stride≈12）
- 不同层有不同的最优刷新频率，支持逐层独立决策的设计

## 亮点与洞察
- 提出了一个清晰的"驱逐不如刷新"的insight：保留全KV缓存做存储，但计算时只用小缓存
- Chain-of-key合成任务的设计非常巧妙——清晰展示了"需要动态回溯查找"的场景，驱逐方法必然失败
- 消融实验非常精炼：w/o refresh vs w/o full attention 两组实验干净地分离了两个因素的贡献
- 自适应调度策略基于query相似度而非固定步长，允许不同层不同行为，设计上更灵活

## 局限与展望
- 不减少内存占用（仍需存储全KV缓存），对显存瓶颈的场景帮助有限
- 仍存在性能-速度权衡，无法完全恢复Vanilla性能
- 相似度阈值 $s$ 是全局设定的，未探索逐层设定
- Continued Pre-training规模较小且仅在Arxiv领域
- **可探索方向**：(1) 与KV量化方法结合减少内存；(2) 结合层级特定策略（如PyramidKV）设定不同层的K大小；(3) 扩展到视觉Transformer

## 相关工作与启发
- SnapKV在预填充阶段选一次重要token后就不再更新，是RefreshKV的特例（不刷新）
- H2O使用累积注意力分数动态维护heavy hitter，但仍是永久驱逐
- StreamingLLM的sink token + 滑动窗口策略简单但信息损失严重
- MInference关注预填充阶段的稀疏注意力加速，与RefreshKV关注解码阶段互补

## 评分
- 新颖性: ⭐⭐⭐⭐ "刷新而非驱逐"的思路简洁优雅，但在技术实现上偏简单
- 实验充分度: ⭐⭐⭐⭐⭐ 语言建模、短输出、长输出、合成任务全覆盖，消融设计精准
- 写作质量: ⭐⭐⭐⭐⭐ 动机阐述清晰，Chain-of-key任务设计直观，伪代码简洁
- 价值: ⭐⭐⭐⭐ 填补了KV缓存方法在长文本生成任务上的评测空白，方法可直接用于现有LLM

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](kv_latent_cache_reduction.md)
- [\[ACL 2025\] SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)

<!-- RELATED:END -->
