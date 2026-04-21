---
title: >-
  [论文解读] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data
description: >-
  [ICLR2026][时间序列][time-series foundation model] 提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征的通用时序基础模型，结合混合掩码预训练策略，首次实现无需任务特定修改即可适配所有生成式时序任务。
tags:
  - ICLR2026
  - 时间序列
  - time-series foundation model
  - 注意力机制
  - representation learning
  - hybrid masking
  - multi-task
---

# GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data

**会议**: ICLR2026  
**arXiv**: [2502.03264](https://arxiv.org/abs/2502.03264)  
**代码**: [GitHub](https://github.com/MMTS4All/GTM)  
**领域**: time_series  
**关键词**: time-series foundation model, frequency-domain attention, representation learning, hybrid masking, multi-task

## 一句话总结

提出 GTM，一个通过频域注意力机制捕获时间粒度感知特征的通用时序基础模型，结合混合掩码预训练策略，首次实现无需任务特定修改即可适配所有生成式时序任务。

## 背景与动机

时序基础模型（TSFM）近年取得进展，但仍面临两大挑战：(1) 标量时序序列的表达能力有限，现有模型大多只在时域建模，未充分挖掘频域信息；(2) 下游任务种类繁多（预测、插补、异常检测、分类等），现有模型需要针对不同任务做修改（如更换 tokenization、调整预训练策略、替换 projection head 等）。

作者通过对大规模多领域时序数据做 FFT + 2D 核密度估计分析，发现不同时间粒度（秒级、分钟级、小时级等）的时序数据在频域的幅值-频率和相位-频率联合分布存在显著差异。这一关键观察直接驱动了模型设计——需要频域模块来捕获多粒度表征。

## 核心问题

1. 如何在模型中显式建模不同时间粒度下频域分布的差异，以增强表征学习质量？
2. 如何设计统一的预训练框架，使单一模型无需任何任务特定修改即可适配所有生成式下游任务？

## 方法详解

### 整体架构

GTM 采用 Decoder-only Transformer 架构，包含三个核心模块：

- **Input Embedding**：对原始时序施加 Reversible Instance Normalization、Channel Independence (CI)、patching 和 masking，转为单变量掩码 token 序列，再加线性嵌入和位置编码
- **N-stack Decoder Backbone**：每个 decoder block 包含一个标准时域自注意力层 + 一个 Fourier Attention 模块
- **Output Projection**：统一的线性投影层 + 反归一化，自回归式生成输出

### Fourier Attention 机制

这是本文核心创新。时域自注意力输出 $\mathbf{H}_{\text{TemAttOut}}$ 经过逐列 FFT 变换到频域后，通过以下步骤处理：

1. **时间粒度编码**：将采样粒度表示为五元组 (day, hour, minute, second, millisecond)，例如 ETTm 数据集编码为 [0, 0, 15, 0, 0]
2. **粒度感知注意力**：引入 5 个可学习 key embedding 对应 5 种典型粒度，计算 query 与各 key 的点积 + softmax 得到注意力权重 $\alpha$
3. **低秩频域模块**：5 个粒度对应 5 组低秩矩阵 $\{A_i, B_i\}$，按注意力权重加权聚合
4. **全局频域模块**：一个全连接矩阵 $W_{\text{full}}$ 并行捕获与粒度无关的频域模式
5. **逆 FFT**：将频域输出变换回时域

最终输出：$\mathbf{H}_{\text{out}} = \text{iFFT}\left(\sum_{i=1}^{5} \alpha_i (A_i B_i) \mathbf{H}_{\text{FFT}} + W_{\text{full}} \mathbf{H}_{\text{FFT}}\right)$

### 预训练框架：混合掩码策略

统一重建和自回归两种目标，核心设计包括：

- **混合掩码**：引入超参数 `pred_ratio` 控制尾部连续掩码的概率。采样 $r \sim \mathcal{U}(0,1)$，若 $r \leq$ `pred_ratio` 则施加尾部连续掩码（模拟预测），否则随机掩码（模拟重建）
- **Span Shuffling**：随机采样多个连续 patch span，随机排列后拼接为目标序列
- **2D 位置编码**：使模型感知被掩码 span 的长度信息
- **注意力机制**：对掩码重建部分使用 full attention，对自回归生成部分使用 causal attention，防止信息泄露
- **预训练数据**：使用大规模公开数据集 UTSD-12G，覆盖多领域多粒度

### 下游任务适配

得益于统一架构和预训练策略，GTM 适配生成式任务仅需轻微预处理调整（去除 masking 和 2D 位置编码），无需修改模型结构。对于分类等判别式任务，仅替换输出投影层。

## 实验关键数据

### 长期预测

在 ETTh1、ETTm1、Weather、Traffic、Electricity 五个数据集上，GTM 在预测长度 $T \in \{96, 192, 336, 720\}$ 平均表现最优。代表性结果：

| 数据集 | GTM (MSE/MAE) | PatchTST | TimesNet |
|--------|--------------|----------|----------|
| ETTh1 | 0.404/0.429 | 0.413/0.434 | 0.458/0.450 |
| Weather | 0.225/0.266 | 0.225/0.263 | 0.259/0.287 |
| Electricity | 0.161/0.254 | 0.159/0.252 | 0.192/0.295 |

### 插补

在 ETTh1 上 MSE 比次优模型降低 23.1%、MAE 降低 12.1%；ETTm1 上 MSE 降低 25.0%、MAE 降低 8.6%。

### 异常检测

在 MSL、SMAP、SWaT、SMD、PSM 五个数据集上平均 F1 达 87.01%，超过所有基线（次优 GPT4TS 为 86.72%）。

### 分类

在 10 个分类数据集上获得 5 个最优和 4 个次优，超过 UniTS、GPT4TS、TimesNet 等多任务 TSFM。

### Zero-shot 预测

与 Timer-1B、MOIRAI-S、MOMENT、TimesFM、Chronos-S1 对比，GTM 在 5 个数据集平均 MSE 最低（0.380 vs 次优 Timer 的 0.392）。

### 预训练有效性

预训练 GTM vs 随机初始化 GTM：预测 MSE 降低 0.5%~7.8%，插补 MSE 降低 1.2%~11.7%，异常检测 F1 提升 1.2%。

### 可扩展性

GTM 遵循 scaling law：更深更宽的模型、更大的预训练数据均带来性能提升。但当模型深度不足时，增加宽度可能无法改善性能。

### 计算效率

GTM 参数量 35.73M，训练速度 0.290s/iter，推理内存 1.25GB，在 A100 上单变量推理延迟仅 0.043s/item，FFT/iFFT 操作引入的额外开销极小。

## 亮点

1. **频域注意力机制**：首次在 TSFM 中引入时间粒度感知的频域建模，通过低秩分解 + 全局模块的设计既高效又有效
2. **生成式任务无关性**：首个无需任何任务特定修改即可适配所有生成式时序任务的基础模型
3. **混合掩码预训练**：巧妙地通过概率控制将重建和自回归统一为单一预训练目标
4. **实验全面**：覆盖预测、插补、异常检测、分类、zero-shot、few-shot、消融、可扩展性、计算效率等多维度评估

## 局限与展望

1. 频域粒度仅编码为 5 种类型，对不规则采样或混合粒度的时序可能不够灵活
2. 分类任务仍需替换投影层，未实现完全的任务无关性
3. Channel Independence 策略可能丢失多变量间的相关性信息
4. Zero-shot 在部分数据集（如 ETTm1、Traffic）上不及 MOIRAI-S 或 Timer，泛化性仍有提升空间
5. 预训练数据 UTSD-12G 的领域覆盖和质量对性能影响未充分讨论

## 与相关工作的对比

| 模型 | 特点 | GTM 优势 |
|------|------|----------|
| Timer | 自回归预训练，需任务特定修改 | GTM 统一预训练，无需修改 |
| UniTS | 双塔 Transformer + task tokenization | GTM 更简洁，无需 task token |
| UP2ME | MAE 预训练 + Graph Transformer 微调 | GTM 单一架构即可 |
| PatchTST | 仅预测任务，CI + patch | GTM 在此基础上加频域建模 |
| MOIRAI | 跨频率学习但仅预测 | GTM 支持多任务 |
| Time-MOE | MOE 设计，多分辨率预测 | GTM 通过频域注意力实现粒度感知 |

## 启发与关联

- 频域注意力的低秩设计思路值得借鉴：不同粒度对应不同低秩子空间，非常符合信号处理直觉
- 混合掩码策略将 BERT 式重建和 GPT 式自回归统一到一个框架，这一思路可推广到其他模态的基础模型
- 时间粒度作为先验知识编码进模型，对于处理多源异构时序数据具有参考价值
- 可考虑将频域注意力与 state space model（如 Mamba）结合，进一步降低计算复杂度

## 评分
- 新颖性: 8/10 — 频域注意力 + 混合掩码预训练的组合具有显著创新性
- 实验充分度: 9/10 — 覆盖面广，消融和效率分析详尽
- 写作质量: 7/10 — 结构清晰但部分符号较密集
- 价值: 8/10 — 首个生成式任务无关的 TSFM，工业部署潜力大

<!-- RELATED:START -->

## 相关论文

- [GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)
- [PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering](../../AAAI2026/time_series/mask_the_redundancy_evolving_masking_representation_learning_for_multivariate_ti.md)
- [Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)

<!-- RELATED:END -->
