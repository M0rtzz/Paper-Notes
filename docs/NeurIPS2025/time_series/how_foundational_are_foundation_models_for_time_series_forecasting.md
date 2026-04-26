---
title: >-
  [论文解读] How Foundational are Foundation Models for Time Series Forecasting?
description: >-
  [NeurIPS 2025][时间序列][时间序列基础模型] 通过合成数据与真实电力消耗数据的系统性实验，揭示时间序列基础模型(TSFM)的零样本泛化能力高度依赖于预训练数据分布，在领域偏移场景下仅49.5K参数的轻量专用模型SAMFormer从头训练即可超越500M+参数的微调TimesFM。
tags:
  - NeurIPS 2025
  - 时间序列
  - 时间序列基础模型
  - 零样本预测
  - 微调
  - 领域迁移
  - 轻量模型
---

# How Foundational are Foundation Models for Time Series Forecasting?

**会议**: NeurIPS 2025  
**arXiv**: [2510.00742](https://arxiv.org/abs/2510.00742)  
**作者**: Nouha Karaouli, Denis Coquenet, Elisa Fromont, Martial Mermillod, Marina Reyboz
**机构**: Univ. Rennes / CNRS / Inria / IRISA; Univ. Grenoble Alpes / CEA
**代码**: 数据集公开 (Zenodo)  
**领域**: 时间序列  
**关键词**: 时间序列基础模型, 零样本预测, 微调, 领域迁移, 轻量模型

## 一句话总结

通过合成数据与真实电力消耗数据的系统性实验，揭示时间序列基础模型(TSFM)的零样本泛化能力高度依赖于预训练数据分布，在领域偏移场景下仅49.5K参数的轻量专用模型SAMFormer从头训练即可超越500M+参数的微调TimesFM。

## 研究背景与动机

### 基础模型在NLP/CV的成功

基础模型(Foundation Models)在NLP（如BERT）和CV（如ViT）领域已经取得了革命性成功。通过在大规模多样化数据上预训练，这些模型展现出强大的零样本和少样本迁移能力，在各种下游任务上持续优于从头训练的任务特定模型。

### 时间序列基础模型的挑战

受NLP/CV成功的启发，研究者提出了时间序列基础模型(TSFMs)，如TimesFM、TimeGPT和TiReX，期望通过大规模预训练捕获通用时间模式表示。然而，时间序列数据具有独特挑战：

- **领域特异性结构**：季节性、趋势、不规则采样等在不同应用间差异巨大
- **分布偏移**：即使是同一大类下的数据也存在显著的统计特性差异
- **个体行为差异**：真实场景中的个体级数据（如个人用电模式）与预训练中使用的人口级聚合数据分布截然不同

### 核心研究问题

1. TSFMs能否泛化到超出其预训练分布的场景？
2. 与轻量专用替代方案相比，TSFMs在实际应用中是否具有竞争力？

## 方法详解

### 评估框架设计

论文设计了一个从合成数据到真实数据的系统性评估流程，逐步增加任务难度和分布偏移程度。

### 模型选择

| 模型 | 类型 | 参数量 | 特点 |
|------|------|--------|------|
| TimesFM | Decoder-only Transformer TSFM | 500M+ | Google发布，大规模合成+真实数据预训练 |
| TimeGPT | Transformer FM (托管API) | 未公开 | Nixtla，100B+时间序列观测值预训练 |
| TiReX | xLSTM架构 TSFM | 未公开 | 零样本长短horizon预测 |
| SAMFormer | 轻量注意力模型 | 49.5K | 通道维度注意力，从头训练 |

### 合成数据基准

构建四个合成数据集测试零样本能力，均包含2688个时间步（8周，30分钟采样间隔）：

- **D1, D2**：谐波对齐的正弦波，完全可观测的周期信号——测试对简洁周期模式的识别和外推能力
- **D3, D4**：随机采样的非谐波正弦波，复杂且部分可观测的周期——测试对不完整模式的泛化能力

### 真实世界评估

使用 **Elec_Consumption** 数据集——一个覆盖2023-2024年共两年的单户家庭日用电量私有小数据集。该数据集反映个体消费行为，与TSFM预训练中使用的通用人口级数据集形成明确的分布偏移。

### 微调实验设置

- 优化器：Adam，学习率 $10^{-4}$，权重衰减 0.01，批大小 64
- 滑窗分帧：context=128, horizon=128
- 训练：最多100 epochs，early stopping (patience=10)
- SAMFormer：5个不同随机种子取均值和标准差
- 硬件：NVIDIA Tesla V100 GPU

## 实验关键数据

### 表1：合成数据零样本MAE（D1-D4）

| 数据集 | Horizon | TimeGPT | TiReX | TimesFM |
|--------|---------|---------|-------|---------|
| D1 | 128 | 0.89 | **0.11** | 0.13 |
| D1 | 256 | 1.08 | **0.21** | 0.22 |
| D1 | 512 | 1.09 | 0.37 | **0.34** |
| D2 | 128 | 0.80 | 0.29 | **0.15** |
| D2 | 256 | 1.25 | 0.72 | **0.35** |
| D2 | 512 | 1.57 | 1.11 | **0.72** |
| D3 | 128 | 1.86 | **1.10** | 1.13 |
| D3 | 512 | **2.29** | 3.30 | 3.50 |
| D4 | 128 | 1.30 | **0.78** | 0.89 |
| D4 | 512 | **2.31** | 2.80 | 2.98 |

**关键发现**：在简单周期模式(D1/D2)上TiReX和TimesFM表现优异；在复杂非谐波模式(D3/D4)上，长horizon时所有模型性能急剧下降，TimeGPT反而因保守策略表现更稳定。

### 表2：真实数据(Elec_Consumption)零样本及微调MAE

| 模型 | 15-7 | 30-7 | 60-30 | 128-128 | 365-365 |
|------|------|------|-------|---------|---------|
| TimeGPT | 6.60 | 6.52 | 5.60 | 6.91 | 6.44 |
| TiReX | 6.94 | 5.71 | 4.61 | **3.78** | 5.90 |
| TimesFM | **5.07** | 5.83 | **4.08** | 4.63 | **5.30** |

**微调对比（context=128, horizon=128）**：

| 模型 | 参数量 | MAE |
|------|--------|-----|
| TimesFM (微调) | 500M+ | 4.49 ± 0.00 |
| SAMFormer (从头训练) | 49.5K | **4.28 ± 0.05** |

**关键发现**：仅49.5K参数的SAMFormer从头训练即超越经过微调的500M+参数TimesFM，参数效率提升超过10000倍。

## 亮点

1. **实验设计精巧**：从合成数据到真实数据的渐进式评估，系统性地揭示TSFM在不同分布偏移程度下的表现退化模式
2. **发现具有实际指导意义**：明确指出"one-size-fits-all"承诺在时间序列领域不成立，为实际部署提供决策依据——当预训练-目标相似度高时用TSFM，否则用轻量专用模型
3. **参数效率对比惊人**：49.5K vs 500M+参数（万倍差距），轻量模型仍胜出，强有力地质疑了"越大越好"的预训练范式
4. **关注实际部署场景**：使用私有个体级用电数据而非公开基准，更贴近真实应用中的领域偏移挑战
5. **数据集公开**：所有使用的数据集均通过Zenodo公开，支持可复现性

## 局限性

1. **评估范围有限**：仅测试单变量时间序列预测，未涵盖多变量预测、异常检测、分类等其他时间序列任务
2. **真实数据集过于简单**：仅使用单户两年日用电数据，样本量极小（~730天），难以代表大规模真实场景
3. **TSFM选择不全面**：仅测试TimesFM/TimeGPT/TiReX三个模型，未包含Chronos、Lag-Llama、MOIRAI等近期重要TSFM
4. **微调策略单一**：仅使用默认Adam配置微调TimesFM，未探索学习率调度、LoRA等参数高效微调策略，可能未充分发挥TSFM潜力
5. **缺乏统计显著性检验**：TimesFM微调结果方差为0.00（确定性），SAMFormer报告了标准差但未做正式统计检验
6. **合成数据过于理想化**：正弦波组合难以反映真实时间序列中的突变、缺失值、噪声等复杂特征
7. **未分析计算成本**：虽然提到参数效率，但未量化训练时间、推理延迟、内存占用等实际部署关键指标

## 相关工作

- **时间序列基础模型**：TimesFM (Das et al., 2024)、TimeGPT (Garza et al., 2024)、TiReX (Auer et al., 2025)、FEDformer (Zhou et al., 2022) 代表大规模预训练路线
- **TSFM基准评估**：GIFT-Eval (Aksu et al., 2024) 跨领域泛化评估、OpenTS (2024) 可复现测试套件、Nixtla Arena (2024) 全面评估
- **质疑TSFM有效性**：Xu et al. (2025, ICLR) 指出简单自回归基线可与TSFM竞争；Zhao et al. (2025) 发现延长微调可能导致TSFM性能退化
- **轻量专用模型**：SAMFormer (Ilbert et al., 2024) 利用锐度感知最小化和通道注意力的紧凑Transformer
- **基础模型通用讨论**：Bommasani et al. (2021) 提出基础模型的机遇与风险；Yuan et al. (2025) 质疑"一刀切"评估范式

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐ | 研究问题重要但并非首次提出，实质是经验验证性工作 |
| 技术深度 | ⭐⭐ | 无新方法/模型提出，主要是实验对比和观察总结 |
| 实验质量 | ⭐⭐⭐ | 合成到真实的渐进设计有亮点，但规模偏小、模型覆盖有限 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，论证逻辑流畅，图表规范 |
| 实用价值 | ⭐⭐⭐⭐ | 对TSFM实际部署决策有直接参考价值 |
| 总评 | 3.2/5 | 选题及时重要、结论有启发，但技术贡献有限、实验规模不足以支撑强结论 |

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Synthetic Series-Symbol Data Generation for Time Series Foundation Models](synthetic_series-symbol_data_generation_for_time_series_foundation_models.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)
- [\[NeurIPS 2025\] How Patterns Dictate Learnability in Sequential Data](how_patterns_dictate_learnability_in_sequential_data.md)
- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)

<!-- RELATED:END -->
