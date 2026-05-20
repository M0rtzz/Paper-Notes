---
title: >-
  [论文解读] SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series
description: >-
  [NeurIPS 2025][时间序列][时间序列预测] 提出SynTSBench合成数据驱动评估范式，通过可编程的特征配置和理论最优基准，系统评估时间序列预测模型在趋势、周期、依赖性、噪声鲁棒性等维度的实际建模能力。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "时间序列预测"
  - "合成数据"
  - "模型评估"
  - "理论最优基准"
  - "鲁棒性分析"
---

# SynTSBench: Rethinking Temporal Pattern Learning in Deep Learning Models for Time Series

**会议**: NeurIPS 2025  
**arXiv**: [2510.20273](https://arxiv.org/abs/2510.20273)  
**代码**: [GitHub](https://github.com/TanQitai/SynTSBench)  
**领域**: 时间序列预测 / 评估基准  
**关键词**: 时间序列预测, 合成数据, 模型评估, 理论最优基准, 鲁棒性分析

## 一句话总结

提出SynTSBench合成数据驱动评估范式，通过可编程的特征配置和理论最优基准，系统评估时间序列预测模型在趋势、周期、依赖性、噪声鲁棒性等维度的实际建模能力。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：当前时间序列预测领域的评估存在两个核心问题：

1. **缺乏特征隔离能力**：真实时间序列数据包含交织的趋势、季节性、不同长度依赖等成分，无法分离进行针对性评估。不清楚模型性能提升来自真正捕获特定模式还是利用了训练数据中的偶然相关
2. **缺乏理论性能边界**：基于观测数据的评估无法建立"最优预测应该是什么样"的基准，导致无法区分有意义的泛化改进和过拟合噪声

这反映了一个根本方法论缺口：现有方法以数据复杂度作为验证标准，忽略了对时间序列属性和理论解空间的系统刻画。

## 方法详解

### 整体框架

SynTSBench建立三个核心分析维度：

1. **时间特征分解与能力映射**：构建具有已知模式的合成数据，逐一评估模型对各类模式的捕获能力
2. **数据异常鲁棒性分析**：在干净信号上注入渐进噪声和各类异常，量化噪声容忍阈值和恢复能力
3. **理论最优基准对标**：利用合成数据已知生成过程建立各模式类型的理论最优解，直接对比模型预测与数学最优值

### 关键设计

1. **可编程时间特征合成**:
    - 功能：设计涵盖11种趋势函数和10种周期模式的合成数据集，加上短/长程依赖、多变量相关等
    - 核心思路：通过ARMA过程生成短/长程依赖；用随机游走和白噪声测试无依赖基线；构建延迟关系、线性可加、条件交互、非线性变换等多变量关系
    - 设计动机：隔离混淆因素，使模型性能可直接归因于对特定时间特征的捕获能力

2. **渐进噪声注入与异常测试**:
    - 功能：在多个SNR水平注入高斯噪声，测试均匀、Laplace、t分布、Lévy稳定等不同噪声分布，引入点异常、脉冲异常、均值漂移和趋势变化
    - 核心思路：控制变量地修改干净合成信号，量化每种模型的噪声容忍阈值和异常恢复能力
    - 设计动机：真实数据中不可避免含噪声和异常，需要系统评估模型鲁棒性而非仅在干净数据上比较

3. **理论最优解对标**:
    - 功能：利用合成数据的已知生成过程计算各模式类型的理论最优预测
    - 核心思路：例如线性趋势的最优MSE为0，AR过程的最优解可以解析计算
    - 设计动机：没有理论最优，模型优化如同"盲盒调参"；有了最优基准，可以明确性能差距和改进空间

### 损失函数 / 训练策略

所有模型统一使用滑动窗口协议，输入长度96，预测长度 $\{10, 24, 48, 96, 192\}$，训练/验证/测试比例为7:1:2。评估15个Fine-tuning模型（Autoformer、PatchTST、iTransformer、DLinear、TimesNet等）和3个零样本模型（Chronos、TimeMoE、Moirai）。

## 实验关键数据

### 主实验（表格）

趋势信号预测（MSE/MAE，4个预测长度平均）：

| 趋势类型 | 最优模型 | 最优MSE | 最差模型 | 最差MSE | 理论最优 |
|----------|---------|---------|---------|---------|---------|
| 指数趋势 | DLinear | 3.40e-08 | Autoformer | 8.45 | 0 |
| 线性趋势 | PaiFilter | 1.34e-12 | Autoformer | 2.02e-03 | 0 |
| 高斯趋势 | PaiFilter | 4.09e-06 | Autoformer | 2.00e-02 | 0 |
| Gompertz趋势 | TimeMixer | 2.85e-06 | Autoformer | 3.56e-03 | 0 |

### 消融实验

- **趋势能力差异巨大**：DLinear在线性/指数趋势上接近理论最优，但Autoformer在指数趋势上误差高出8个数量级
- **周期捕获**：Transformer类模型在复杂周期模式上不一定优于MLP类
- **噪声鲁棒性**：随SNR降低，不同模型退化速度差异显著
- **零样本模型**：Chronos/TimeMoE/Moirai在可预测模式上远不如Fine-tuning模型

### 关键发现

- 当前深度学习模型**并未在所有时间特征类型上普遍接近理论最优**
- 不同架构（Transformer、MLP、CNN、RNN、KAN）各有擅长的模式类型，没有"万能"架构
- 简单的DLinear在趋势预测上可以比复杂的Transformer好8个数量级
- 模型在合成数据上的表现差异远大于在标准真实数据集上的差异，说明现有基准掩盖了重要能力差异

## 亮点与洞察

- **方法论贡献**：填补了时间序列评估的方法论空白——从"在真实数据上比谁好"到"系统分析各项能力"
- **理论最优基准**：首次为各时间特征类型建立可计算的理论最优值
- **实用性**：帮助从业者根据任务特点选择合适的模型架构
- 揭示了"在标准基准上差距微小的模型在特定能力上可能差异巨大"这一重要洞察

## 局限与展望

- 合成数据的复杂度仍然有限，难以完全模拟真实世界的复杂交互
- 仅评估了单变量/少量多变量场景，未充分测试高维时间序列
- 理论最优基准仅适用于合成数据，不能直接推广到真实数据
- 缺少对大语言模型时间序列方法的深入分析
- 相关的复杂真实场景模拟（经济指标等）数据集设计细节不足

## 相关工作与启发

- **ProbTS**：统一了不同预测范式的评估，但依赖真实数据
- **TFB**：提供标准化管线和多领域覆盖，但同样缺少特征隔离
- 本文与这些框架互补——SynTSBench用于诊断模型能力，标准基准用于衡量实际性能
- 对时间序列社区的模型选择实践具有重要指导意义

## 评分

⭐⭐⭐⭐ — 方法论创新突出，揭示了现有评估框架的盲区，实验全面系统

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_total_electron_con.md)
- [\[NeurIPS 2025\] Connecting the Dots: A Machine Learning Ready Dataset for Ionospheric Forecasting Models](connecting_the_dots_a_machine_learning_ready_dataset_for_ionospheric_forecasting.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[NeurIPS 2025\] In-Context Learning of Stochastic Differential Equations with Foundation Inference Models](in-context_learning_of_stochastic_differential_equations_with_foundation_inferen.md)

</div>

<!-- RELATED:END -->
