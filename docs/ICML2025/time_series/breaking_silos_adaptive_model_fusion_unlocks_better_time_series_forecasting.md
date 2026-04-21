---
title: >-
  [论文解读] Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting
description: >-
  [ICML 2025][时间序列][时间序列预测] 提出 TimeFuse——一个样本级自适应模型融合框架，通过元特征描述输入时间序列特征并训练可学习融合器预测最优模型组合权重，在多个预测基准上对 SOTA 模型实现近乎普遍的改进（95.1% 样本优于最佳单模型）。
tags:
  - ICML 2025
  - 时间序列
  - 时间序列预测
  - 模型融合
  - 元学习
  - 集成方法
  - 自适应权重
---

# Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting

**会议**: ICML 2025  
**arXiv**: [2505.18442](https://arxiv.org/abs/2505.18442)  
**代码**: https://github.com/ZhiningLiu1998/TimeFuse  
**领域**: 时间序列  
**关键词**: 时间序列预测, 模型融合, 元学习, 集成方法, 自适应权重

## 一句话总结
提出 TimeFuse——一个样本级自适应模型融合框架，通过元特征描述输入时间序列特征并训练可学习融合器预测最优模型组合权重，在多个预测基准上对 SOTA 模型实现近乎普遍的改进（95.1% 样本优于最佳单模型）。

## 研究背景与动机

**领域现状**：时间序列预测模型不断进步（Transformer、Mamba、MLP 等），在 benchmark 数据集上你追我赶。

**现有痛点**：细粒度样本级分析揭示了一个被忽视的事实——没有单一模型在所有样本上一致最优；即使排名最高的模型也只在约 23.2% 的测试样本上排名第一；每个模型都有独特的擅长领域。

**核心矛盾**：单模型范式浪费了不同模型的互补优势。

**本文目标**：如何自适应地利用不同模型在不同样本上的独特优势？

**切入角度**：用元特征描述每个输入时间序列的特性，训练融合器预测最优模型组合权重。

**核心 idea**：从"选最好的单模型"转向"为每个样本找最优的模型组合"。

## 方法详解

### 整体框架
1. 构建模型动物园（model zoo）：独立训练 k 个预测模型
2. 元特征提取：为每个输入计算统计/时序/频谱特征
3. 融合器训练：学习从元特征到模型组合权重的映射
4. 推理时：提取元特征→预测权重→加权组合各模型预测

### 关键设计

1. **多维元特征提取**:

    - 功能：为每个输入时间序列计算全面的特征描述
    - 核心思路：三类特征——统计特征（偏度、峰度）、时序特征（平稳性、变化率）、频谱特征（主频、谱熵）
    - 设计动机：这些特征捕捉了不同模型擅长的时间序列类型——如 TimeMixer 擅长高频谱复杂度，非平稳 Transformer 擅长低平稳性

2. **可学习融合器**:

    - 功能：从元特征预测 k 个模型的组合权重 $w_1, \dots, w_k$
    - 核心思路：MLP 网络，输入元特征，输出 softmax 归一化的权重
    - 设计动机：端到端学习使融合器能自动发现元特征与模型优势间的关联

3. **跨数据集联合训练**:

    - 功能：融合器在多个数据集的样本上联合训练
    - 核心思路：元特征是数据集无关的描述，因此融合器可泛化到未见过的数据集
    - 设计动机：增加训练多样性提升零样本泛化能力

### 损失函数 / 训练策略
- 最小化融合预测的 MSE 损失
- 融合器训练与基模型训练解耦
- 支持任意异构基模型

## 实验关键数据

### 主实验

| 数据集 | 最佳单模型 MSE | TimeFuse MSE | 改善样本比例 |
|--------|-------------|-------------|------------|
| ETTh1 | 0.376 | **0.358** | 89.2% |
| Weather | 0.151 | **0.142** | 92.4% |
| Traffic | 0.360 | **0.344** | 95.1% |

### 消融实验

| 配置 | MSE | 说明 |
|------|-----|------|
| 均匀权重集成 | 0.368 | 不自适应 |
| 仅统计特征 | 0.362 | 缺少频谱信息 |
| 全部元特征 | **0.358** | 最优 |
| 单数据集训练 | 0.364 | 泛化性差 |
| 跨数据集训练 | **0.358** | 泛化性好 |

### 关键发现
- 在高达 95.1% 的样本上优于最佳单模型——near-universal improvement
- 融合器权重可解释：高频谱复杂度→更多权重给 TimeMixer；低平稳性→更多权重给 Non-stationary Transformer
- 零样本泛化到未见数据集也有效

## 亮点与洞察
- **"没有万能模型"的细粒度发现**非常有说服力，数据分析透彻
- 元特征作为桥梁使融合器跨数据集转移，是关键设计选择
- 框架通用性强——任何新模型都可直接加入 model zoo

## 局限与展望
- 需要维护和推理多个基模型，计算开销线性增长
- 元特征设计目前是手工的，自动特征学习可能更好
- 未考虑模型间的相关性——可能有冗余

## 相关工作与启发
- **vs 传统集成（bagging/boosting）**: 静态组合，不自适应
- **vs Model Selection**: 只选一个模型，丢弃其他模型的信息
- 对 AutoML 和模型选择研究有启发

## 评分
- 新颖性: ⭐⭐⭐⭐ 样本级自适应融合视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 14模型×7数据集，分析极其充分
- 写作质量: ⭐⭐⭐⭐⭐ 可视化和分析出色
- 价值: ⭐⭐⭐⭐⭐ 实用的通用预测改进框架

<!-- RELATED:START -->

## 相关论文

- [LightGTS: A Lightweight General Time Series Forecasting Model](lightgts_a_lightweight_general_time_series_forecasting_model.md)
- [Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](../../ICLR2026/time_series/adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)
- [WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)
- [TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state](timepro_efficient_multivariate_long-term_time_series_forecasting_with_variable-_.md)
- [When Will It Fail?: Anomaly to Prompt for Forecasting Future Anomalies in Time Series](when_will_it_fail_anomaly_to_prompt_for_forecasting_future_anomalies_in_time_ser.md)

<!-- RELATED:END -->
