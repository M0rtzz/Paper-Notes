---
title: >-
  [论文解读] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models
description: >-
  [ACL 2025 (Short Paper)][时间序列][大语言模型] 本文系统评估了 LLM 作为零样本时间序列预测器的有效性，发现 LLM 对输入噪声极度敏感——即使少量噪声也会使性能大幅下降，甚至不如简单的领域专用模型（如 DLinear），建议未来应聚焦于对 LLM 进行微调以更好地处理数值序列。
tags:
  - ACL 2025 (Short Paper)
  - 时间序列
  - 大语言模型
  - 零样本预测
  - 噪声敏感性
  - 鲁棒性
---

# Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models

**会议**: ACL 2025 (Short Paper)  
**arXiv**: [2506.00457](https://arxiv.org/abs/2506.00457)  
**代码**: [GitHub](https://github.com/junwoopark92/revisiting-LLMs-zeroshot-forecaster)  
**领域**: 时间序列  
**关键词**: 大语言模型、零样本预测、时间序列、噪声敏感性、鲁棒性

## 一句话总结

本文系统评估了 LLM 作为零样本时间序列预测器的有效性，发现 LLM 对输入噪声极度敏感——即使少量噪声也会使性能大幅下降，甚至不如简单的领域专用模型（如 DLinear），建议未来应聚焦于对 LLM 进行微调以更好地处理数值序列。

## 研究背景与动机

**领域现状**：LLM 在自然语言处理等多种任务上展示了强大的零样本能力，引发了将其直接用于时间序列预测的研究兴趣。LLMTime 等工作通过将数值序列转换为文本 prompt，让 LLM 进行零样本预测，在某些场景下取得了令人瞩目的结果。

**现有痛点**：然而社区对 LLM 零样本时间序列预测的有效性存在分歧——一些研究声称 LLM 表现优异，另一些则质疑 LLM 是否真正"理解"数值模式。现有的正面结果可能受到评估设置的影响，例如仅在干净的合成数据或特定数据集上测试。

**核心矛盾**：真实世界的时间序列数据几乎总是含有噪声（传感器误差、测量不确定性、随机波动等），但现有评估大多在理想化条件下进行。如果 LLM 的预测能力在面对真实噪声时急剧退化，那其零样本预测的实用价值就大大存疑。

**本文目标**：在严格控制的实验条件下，系统比较 LLM 零样本预测与领域专用模型的表现，特别关注噪声条件下的鲁棒性差异。

**切入角度**：作者从鲁棒性角度切入——不仅在干净数据上比较性能，还系统地注入不同类型和程度的噪声，观察各模型的降效幅度。

**核心 idea**：通过噪声敏感性分析揭示 LLM 零样本预测的根本脆弱性，并指出微调而非 prompting 才是 LLM 应用于时间序列的正确方向。

## 方法详解

### 整体框架

本文是一项评估性研究而非方法论文。实验框架如下：(1) 选择多个 LLM（GPT-3.5、GPT-4o、LLaMA）和多个领域专用基线（DLinear、RLinear、N-BEATS、ARIMA 等）；(2) 在多种真实世界（Monash、Informer benchmark）和合成数据集上进行预测；(3) 对数据注入多种噪声（高斯噪声、常值噪声、缺失值、周期噪声）后重新评估；(4) 对比干净条件和噪声条件下各模型的性能变化。

### 关键设计

1. **多类型噪声注入框架（Multi-Type Noise Injection Framework）**:

    - 功能：系统性地测试模型在不同噪声条件下的表现
    - 核心思路：定义四种噪声类型——(a) 高斯噪声：在每个时间步加入 $\mathcal{N}(0, \sigma^2)$ 的随机扰动；(b) 常值噪声：在随机位置插入固定异常值；(c) 缺失值噪声：随机将部分时间步设为空；(d) 周期噪声：叠加一个特定频率的正弦波。每种噪声按不同强度级别（低/中/高）注入
    - 设计动机：真实时间序列中的噪声形式多样，单一类型的噪声测试不足以全面评估鲁棒性。四种噪声覆盖了最常见的数据质量问题

2. **LLM 零样本预测协议（Zero-Shot LLM Forecasting Protocol）**:

    - 功能：标准化 LLM 作为零样本预测器的评估流程
    - 核心思路：沿用 LLMTime 的范式——将历史数值序列转换为逗号分隔的文本字符串，作为 prompt 输入 LLM，让 LLM 续写预测值。对数值做适当的缩放和格式化处理。生成多次取中位数作为点预测。评估指标为 MAE 和 MSE
    - 设计动机：统一评估协议确保公平比较，遵循已有文献的标准做法以保证可比性

3. **跨维度对比分析（Cross-Dimensional Comparison）**:

    - 功能：从多个角度揭示 LLM 预测的弱点
    - 核心思路：不仅比较绝对性能，还分析 (a) 噪声导致的性能下降幅度（degradation ratio），(b) 不同数据集特征对 sensitivity 的影响，(c) 模型规模与鲁棒性的关系。通过合成数据（已知生成过程的数学函数）进一步控制变量，观察 LLM 对简单模式的识别能力在噪声下如何变化
    - 设计动机：全面的对比可以帮助定位问题根源——是 LLM 无法理解数值本身，还是对文本表示中的数值扰动特别敏感

### 训练策略

本文不涉及训练。领域专用模型使用各自标准训练流程，LLM 为直接 API 调用或本地推理。

## 实验关键数据

### 主实验

在 Informer 五个真实数据集上的平均 MAE 对比（干净 vs 高斯噪声）：

| 模型 | 干净数据 MAE | 噪声数据 MAE | 下降率 |
|------|------------|------------|--------|
| GPT-4o (零样本) | 0.412 | 0.687 | -66.7% |
| GPT-3.5 (零样本) | 0.458 | 0.731 | -59.6% |
| LLaMA-3 (零样本) | 0.523 | 0.812 | -55.3% |
| DLinear | 0.289 | 0.324 | -12.1% |
| RLinear | 0.301 | 0.338 | -12.3% |
| N-BEATS | 0.275 | 0.312 | -13.5% |
| ARIMA | 0.342 | 0.378 | -10.5% |

### 消融实验

不同噪声类型对 GPT-4o 在 Monash 数据集上的影响：

| 噪声类型 | 干净 MAE | 加噪后 MAE | 下降率 |
|---------|---------|-----------|--------|
| 无噪声 | 0.389 | - | - |
| 高斯噪声 (σ=0.1) | - | 0.542 | -39.3% |
| 常值噪声 (5%) | - | 0.611 | -57.1% |
| 缺失值 (10%) | - | 0.498 | -28.0% |
| 周期噪声 | - | 0.467 | -20.1% |

### 关键发现

- **LLM 在干净数据上的表现已经不如领域专用模型**——即使在最有利条件下，GPT-4o 的 MAE 也比 DLinear 高约 40%
- **噪声放大效应极其显著**：LLM 的性能在高斯噪声下下降 50-67%，而领域专用模型仅下降 10-14%，差距进一步拉大
- 常值噪声（异常值）对 LLM 影响最大，这可能与 LLM 的 token 化方式有关——异常数值会产生不寻常的 token 序列
- 合成数据实验表明，即使是简单的 sigmoid 函数，加入少量噪声后 LLM 的预测形状就会严重失真
- 模型规模增大（GPT-3.5 → GPT-4o）在干净数据上有帮助，但在噪声下的改善有限，说明规模不是解决鲁棒性的关键

## 亮点与洞察

- **"小噪声摧毁大模型"的标题和核心发现非常有警示意义**：直接挑战了 LLM "万能"的叙事，对社区的过度乐观情绪是有价值的纠偏
- **噪声类型分类和系统性评估框架可复用**：四种噪声类型+多级强度的评估协议可以作为未来 LLM 数值能力评估的标准组件
- **方向性建议很务实**：不是否定 LLM 在时间序列中的潜力，而是明确指出"微调 > 零样本"的路径更有前景，对后续研究有明确指导

## 局限与展望

- 作为 short paper，实验规模有限，未覆盖更多 LLM（如 Claude、Gemini）和更大规模的时序数据集
- 仅关注了零样本设置，未比较 few-shot 或 in-context learning 在噪声下的表现
- 未深入分析 LLM 对噪声敏感的根本原因——是 tokenization 问题、数值精度问题、还是模式识别能力不足
- 提出"应微调 LLM"的建议但未给出微调方案的实证验证

## 相关工作与启发

- **vs LLMTime (Gruver et al., 2023)**: LLMTime 首次展示了 LLM 零样本时序预测的可能性，本文是对其结论的直接挑战和深化——指出其在干净数据上的积极结果不能推广到真实噪声环境
- **vs Time-LLM (Jin et al., 2024)**: Time-LLM 选择了微调路线，与本文的建议方向一致
- **vs 传统统计模型 (ARIMA)**: 即使是最经典的统计方法在噪声下也显著优于 LLM 零样本预测，凸显了领域知识的重要性

## 评分

- 新颖性: ⭐⭐⭐ 噪声敏感性分析是有价值的视角，但核心实验设计较为直接
- 实验充分度: ⭐⭐⭐ 覆盖了多种模型、数据集和噪声类型，但作为 short paper 深度有限
- 写作质量: ⭐⭐⭐⭐ 论述清晰，发现表述到位，结论不过度
- 价值: ⭐⭐⭐⭐ 对社区关于 LLM 时序预测的讨论有重要贡献，方向性建议有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](../../ICLR2026/time_series/relational_transformer_toward_zero-shot_foundation_models_for_relational_data.md)
- [\[ACL 2025\] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)
- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](../../NeurIPS2025/time_series/how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](../../NeurIPS2025/time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/time_series/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)

<!-- RELATED:END -->
