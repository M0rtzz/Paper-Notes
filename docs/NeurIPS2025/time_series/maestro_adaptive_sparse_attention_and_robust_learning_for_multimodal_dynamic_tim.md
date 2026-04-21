---
title: >-
  [论文解读] MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series
description: >-
  [NeurIPS 2025][时间序列][多模态时间序列] 提出 MAESTRO 框架，通过符号化分词、自适应注意力预算、稀疏跨模态注意力和动态 MoE 路由，解决多模态时间序列中模态异质性和任意缺失的问题，在完整/缺失模态场景下均显著超越基线。
tags:
  - NeurIPS 2025
  - 时间序列
  - 多模态时间序列
  - 稀疏注意力
  - 混合专家
  - 缺失模态
  - 传感器融合
---

# MAESTRO: Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series

**会议**: NeurIPS 2025  
**arXiv**: [2509.25278](https://arxiv.org/abs/2509.25278)  
**代码**: [GitHub](https://github.com/payalmohapatra/MAESTRO)  
**领域**: Medical Imaging / Multimodal Learning  
**关键词**: 多模态时间序列, 稀疏注意力, 混合专家, 缺失模态, 传感器融合

## 一句话总结

提出 MAESTRO 框架，通过符号化分词、自适应注意力预算、稀疏跨模态注意力和动态 MoE 路由，解决多模态时间序列中模态异质性和任意缺失的问题，在完整/缺失模态场景下均显著超越基线。

## 研究背景与动机

多模态传感器监测（如心电、皮肤电导、加速度计等）在临床医疗和日常生活中广泛应用，但现有多模态学习方法存在三大局限：(1) 依赖单一主模态作为对齐锚点，容易过度依赖预设的主模态；(2) 采用成对模态交互建模，当模态数量较大时组合爆炸不可行；(3) 假设所有模态完整可用，无法处理传感器故障导致的任意缺失。作者认为，将多源传感器数据简单地视为多变量时间序列是次优的，因为各模态在语义上是不相交的，需要显式建模模态内和模态间的交互。

## 方法详解

### 整体框架

MAESTRO 包含四个核心模块：(1) 缺失感知的符号化分词将原始时间序列转换为离散符号表示；(2) 自适应注意力预算门控根据模态可用性和任务相关性分配稀疏注意力容量；(3) 稀疏跨模态注意力处理长多模态拼接序列；(4) 稀疏 MoE 路由实现输入条件自适应的动态专家分配。

### 关键设计

1. **缺失感知符号化分词 (SAX Tokenization)**: 基于符号聚合近似（SAX）方法对每个模态独立进行时间序列压缩和符号化。关键创新是保留一个额外符号 $s_0$ 表示缺失模态。利用归一化时间序列的高斯分布特性，将连续值映射到 $\alpha$ 个等概率符号。理论上证明了 SAX 变换不仅在模态内保持距离下界，还跨模态保持样本相似性的相对对比关系（Corollary 3.2），保证有意义的跨模态结构被保留。

2. **自适应稀疏注意力与注意力预算 (Adaptive Attention Budgeting)**: 每个模态拥有独立的编码器，使用稀疏自注意力机制（基于 max-mean 稀疏度度量选择 top-$\upsilon$ 查询）。核心创新在于注意力预算 $\mathbf{u}_i$ 由可学习的门控函数 $\mathcal{G}(\mathbf{m}_i; \theta_a)$ 自适应控制，其中 $\mathbf{m}_i$ 是模态可用性向量。该门控随任务目标端到端训练，自动为更相关的模态分配更多注意力资源。

3. **稀疏跨模态注意力 (Sparse Cross-Modal Attention)**: 将各模态编码器的输出沿时间维度拼接，加上模态嵌入和位置嵌入，形成统一的长多模态序列 $\mathbf{c} \in \mathbb{R}^{\hat{D} \times \hat{L}}$。对该序列施加稀疏注意力，将复杂度从 $\mathcal{O}(\hat{L}^2)$ 降至 $\mathcal{O}(\hat{L} \log \hat{L})$。这种设计天然支持不同采样率的模态、实现跨模态时变注意力，且避免了成对交互建模的组合爆炸。

4. **稀疏 MoE 动态路由**: 采用标准稀疏 MoE 层处理跨模态表示，由可训练路由器 $\mathcal{R}$ 选择 top-$k$ 专家。通过渐进式模态丢弃（curriculum learning）训练——先用完整模态样本训练，逐步增加缺失概率至 $p_{max}$——实现无需辅助负载均衡损失的隐式专家特化。

### 损失函数 / 训练策略

使用标准交叉熵损失 $\mathcal{L}_{CE}$。采用渐进式模态丢弃：模态丢弃概率 $p(\tau)$ 在热身期 $\tau_{warmup}$ 后线性增长至 $p_{max}$，作为正则化鼓励模型适应不同模态组合。MoE 采用 top-$k$（通常 $k=1$）路由，对 $k>1$ 的情况通过平均 logits 聚合。整体使用 80/10/10 的训练/验证/测试划分，每个数据集三个不同 split，每个 split 三次试验，报告均值。

## 实验关键数据

### 主实验

| 数据集 | 指标 | MAESTRO | 最佳基线 | 提升 |
|--------|------|---------|----------|------|
| WESAD (10模态) | Acc | 0.77 | 0.71 (FlexMoE) | +8% 相对 |
| DaliaHAR (5模态) | F1 | 0.84 | 0.79 (FuseMoE) | +5% 相对 |
| DSADS (5体位9轴) | Acc | 0.88 | 0.85 (FuseMoE) | +4% 相对 |
| MIMIC-III (17模态) | F1 | 0.30 | 0.27 (FlexMoE) | +11% 相对 |

### 消融实验

| 配置 | 完整模态 Acc 下降 | 40%缺失 Acc 下降 | 说明 |
|------|-------------------|-------------------|------|
| 去掉 SAX | -8% | -7% | 符号化分词贡献显著 |
| 去掉 MoE | -5% | -22% | MoE 对缺失场景至关重要 |
| 去掉模态丢弃 | -2% | -9% | 缺失场景下训练策略关键 |
| 全部换成全注意力 | +3% Acc | - | GFLOPs 增加 20% |

### 关键发现

- 在40%模态缺失条件下，MAESTRO 比最强缺失鲁棒基线平均 F1 提升 59%
- 稀疏注意力仅牺牲最多3%精度，但降低约20% GFLOPs（仅 6.13 GFLOPs，1.39M 参数）
- 多模态处理比多变量处理平均高 4%，MAESTRO 进一步高出 8%
- MAESTRO 对噪声输入（高斯噪声、电气干扰尖峰）展现出固有鲁棒性
- 符号化表示带来平均 6% 的相对性能提升
- 压缩比敏感性研究显示，过大或过小的压缩比均会降低性能

## 亮点与洞察

- 将传感器时间序列从"多变量"视角转为"多模态"视角是一个有价值的范式转变
- 符号化表示同时实现了序列压缩、缺失表示和噪声鲁棒三重功能，设计巧妙
- 无辅助损失的 MoE 隐式特化通过渐进丢弃实现，简洁有效
- 跨模态稀疏注意力避免成对建模，比 MULT 计算效率提升 200%+

## 局限与展望

- 仅在分类任务上验证，未涉及回归或时间序列预测任务
- SAX 超参（压缩比、字母表大小）需要针对每个数据集调优
- MIMIC-III 上绝对性能仍较低(Acc 0.78)，复杂任务的增益有限
- 可探索扩展到非同步采样的在线学习场景

## 相关工作与启发

- FlexMoE/FuseMoE 以 MoE 处理缺失模态，但依赖成对建模或缺失库补全
- MULT 的成对 Transformer 计算代价高（~27 GFLOPs vs MAESTRO 6 GFLOPs）
- SAX 符号化思路可迁移至其他需要处理缺失传感器的时间序列任务
- ShaSpec 通过共享模态表示处理缺失，但在高缺失率下表现不佳
- InceptionTime/ResNet1D 等多变量方法忽略模态异质性，性能天花板较低
- iTransformer 的反转注意力在某些数据集上反而表现不佳（DSADS 仅 0.62）
- 自适应注意力预算的设计思路与 Neural Architecture Search 中的资源分配有异曲同工之妙
- 符号化表示与 VQ-VAE 的离散化思想相关，但 SAX 不需要学习码本

## 评分

- 新颖性: ⭐⭐⭐⭐ 多个设计创新（SAX缺失表示、自适应预算、无损MoE）组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 4数据集10基线、消融/复杂度/敏感性/噪声实验全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰、理论与实验结合好
- 价值: ⭐⭐⭐⭐ 对多模态传感器融合领域有实际推动作用

<!-- RELATED:START -->

## 相关论文

- [Learning Soft Sparse Shapes for Efficient Time-Series Classification](../../ICML2025/time_series/learning_soft_sparse_shapes_for_efficient_time-series_classification.md)
- [Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [Causality-Aware Contrastive Learning for Robust Multivariate Time-Series Anomaly Detection](../../ICML2025/time_series/causality-aware_contrastive_learning_for_robust_multivariate_time-series_anomaly.md)
- [AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)
- [Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)

<!-- RELATED:END -->
