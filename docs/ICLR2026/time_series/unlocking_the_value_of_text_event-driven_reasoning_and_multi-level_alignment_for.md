---
title: >-
  [论文解读] VoT: 事件驱动推理与多层对齐解锁文本价值用于时间序列预测
description: >-
  [ICLR 2026][时间序列][多模态时间序列预测] 提出 VoT，一种通过事件驱动推理（利用 LLM 对外生文本进行结构化推理获取数值预测）和多层对齐（表征级内生文本对齐 + 预测级自适应频率融合）充分挖掘文本信息价值的多模态时间序列预测方法，在 10 个领域的真实数据集上全面超越现有方法。
tags:
  - "ICLR 2026"
  - "时间序列"
  - "多模态时间序列预测"
  - "事件驱动推理"
  - "文本对齐"
  - "自适应频率融合"
  - "LLM推理"
---

# Unlocking the Value of Text: Event-Driven Reasoning and Multi-Level Alignment for Time Series Forecasting

**会议**: ICLR 2026  
**arXiv**: [2603.15452](https://arxiv.org/abs/2603.15452)  
**领域**: 时间序列  
**关键词**: 多模态时间序列预测, 事件驱动推理, 文本对齐, 自适应频率融合, LLM推理

## 一句话总结

提出 VoT，一种通过事件驱动推理（利用 LLM 对外生文本进行结构化推理获取数值预测）和多层对齐（表征级内生文本对齐 + 预测级自适应频率融合）充分挖掘文本信息价值的多模态时间序列预测方法，在 10 个领域的真实数据集上全面超越现有方法。

## 研究背景与动机

现有时间序列预测方法大多仅依赖数值数据，但真实世界中突变事件（如 2008 金融危机、2020 COVID-19 导致的失业率飙升）难以仅从历史数值模式预测。文本信息可提供事件驱动的预测指导，但现有多模态方法面临两大挑战：

### 挑战 1：文本利用不充分

| 方法类型 | 问题 |
|---------|------|
| 内生文本方法（如 LLM-Mixer） | 使用统计摘要/数据描述，与时序信息大量重叠 |
| 外生文本方法（如 CMIN、DualTime） | 仅做表征级融合，难以挖掘深层语义 |
| **VoT（本文）** | **同时使用 LLM 推理 + 特征提取，支持内生 + 外生文本** |

### 挑战 2：模态对齐困难

文本描述事件驱动的突变，时序捕捉细微数值波动——两者存在显著模态鸠间，简单融合难以实现互补。

## 方法详解

### 整体框架：双分支架构

VoT 包含两个互补分支：
- **事件驱动预测分支**：通过 LLM 推理从外生文本生成数值预测
- **数值预测分支**：将内生文本与时序表征对齐后生成预测

最终通过自适应频率融合整合两个分支的输出。

### 关键设计 1：事件驱动推理

**三步生成流水线**：

1. **模板生成**：LLM 根据数据描述和样本生成结构化模板 $\mathcal{D}$
2. **摘要生成**：使用模板从原始外生文本中提取预测相关摘要 $\mathcal{S}_i$
3. **推理预测**：Reasoner（推理型 LLM）基于摘要和历史时序生成数值预测：

$$\hat{\mathbf{Y}}^{\text{event}}_i, \mathcal{R}_i = \text{Reasoner}(\mathcal{P}_{\text{reason}}, \mathcal{S}_i, \mathbf{X}_i)$$

**历史上下文学习（HIC）**：
- **训练阶段**：Reasoner 先生成预测，再对照真值生成纠正推理 $\mathcal{C}_i$，构建知识库 $\mathcal{K} = \{(\text{Embed}(\mathcal{S}_i), \mathcal{C}_i)\}_{i=1}^M$
- **推理阶段**：检索最相似历史样本的纠正推理作为 ICL 示例：

$$\hat{\mathbf{Y}}^{\text{event}}_j = \text{Reasoner}(\mathcal{P}_{\text{ICL}}, \mathcal{C}_{\tilde{i}}, \mathcal{S}_j, \mathbf{X}_j)$$

### 关键设计 2：多层对齐

**表征级——内生文本对齐（ETA）**：
- 使用两组可学习查询 $\mathbf{Q}^{\text{tr}}, \mathbf{Q}^{\text{se}}$ 从文本表征中提取趋势和季节性语义
- 通过交叉注意力实现时序-文本对齐
- 使用**分解对比学习**在样本级对齐趋势和季节性分量

**预测级——自适应频率融合（AFF）**：

将两个分支的预测进行 FFT 分解为低/中/高频分量：

$$\mathcal{F}^{\text{num}} = \text{FFT}(\hat{\mathbf{Y}}^{\text{num}}), \quad \mathcal{F}^{\text{event}} = \text{FFT}(\hat{\mathbf{Y}}^{\text{event}})$$

通过可学习权重 $w_*^b$ 自适应融合各频段分量：

$$\mathcal{F}_{\text{fused}} = \sum_* \sum_b w_*^b \mathcal{F}_*^b, \quad \hat{\mathbf{Y}}_{\text{final}} = \text{iFFT}(\mathcal{F}_{\text{fused}})$$

### 训练目标

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ts}} + \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{final}}$$

## 实验关键数据

### 主实验：与时序方法和多模态方法对比（10 个领域平均）

| 方法 | Agriculture MSE | Climate MSE | Economy MSE | Health MSE | Weather MSE |
|------|----------------|-------------|-------------|------------|-------------|
| iTransformer | 0.220 | 1.135 | 0.222 | 1.519 | 1.231 |
| PatchTST | 0.228 | 1.184 | 0.210 | 1.432 | 1.145 |
| GPT4TS | 0.220 | 1.184 | 0.217 | 1.341 | - |
| TaTS | 0.215 | 1.180 | 0.215 | 1.356 | - |
| CALF | 0.250 | 1.286 | 0.207 | 1.491 | - |
| **VoT** | **0.209** | **1.078** | **0.201** | **1.205** | **0.968** |

### 完胜记录

- **VoT 在所有 10 个领域的 20 项指标（MSE+MAE）上全部排名第一**
- 对比纯时序方法取得一致提升（如 Health: 1.205 vs 1.432 PatchTST，-15.9%）
- 对比现有多模态方法同样全面领先（如 Climate: 1.078 vs 1.180 TaTS，-8.6%）

### 关键发现

1. **文本推理的有效性**：事件驱动推理分支能有效捕捉外部事件对时序的影响
2. **HIC 的价值**：历史纠正推理作为 ICL 示例显著提升推理准确性
3. **频率融合的必要性**：不同数据集对文本和数值信息的依赖在不同频段不同
4. **跨领域一致性**：在金融、气候、健康、交通等 10 个差异极大的领域上均有效

## 亮点与洞察

1. **首个结合 LLM 推理与特征提取的多模态时序预测方法**：不仅使用 LLM 提取表征，更利用其推理能力生成数值预测
2. **HIC 的创新设计**：在训练时保存"错误→纠正"对，推理时检索相似纠正作为 ICL 指导，无需微调即可增强推理
3. **频域融合的精妙**：不同频段分别学习文本与数值的权重，适应不同领域特性
4. **双分支互补架构**：事件驱动分支处理突变，数值分支处理常规波动，AFF 实现最优组合

## 局限性

1. LLM 推理（Reasoner）和知识库构建增加了显著的计算和延迟开销
2. 依赖外生文本的可用性和质量，缺少文本的领域可能退化为纯时序模型
3. HIC 检索质量受知识库大小和多样性限制
4. 频率分频策略的超参数选择可能因领域而异
5. 未考虑文本时间对齐的噪声和延迟问题

## 评分 ⭐⭐⭐⭐⭐

方法设计全面精巧，在 10 个领域 20 项指标上全面第一实属罕见。将 LLM 的推理能力真正引入时序预测，HIC 和 AFF 的设计都很有新意。唯一遗憾是计算开销较大，实际部署需权衡。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] PATRA: Pattern-Aware Alignment and Balanced Reasoning for Time Series Question Answering](../../ICML2026/time_series/patra_pattern-aware_alignment_and_balanced_reasoning_for_time_series_question_an.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](../../NeurIPS2025/time_series/masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)

</div>

<!-- RELATED:END -->
