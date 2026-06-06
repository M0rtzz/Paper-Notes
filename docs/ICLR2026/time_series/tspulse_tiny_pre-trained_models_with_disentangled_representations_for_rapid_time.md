---
title: >-
  [论文解读] TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series
description: >-
  [ICLR 2026][时间序列][Time Series Pre-trained Model] 提出 TSPulse，仅 1M 参数的超轻量时间序列预训练模型，通过双空间掩码重建和双嵌入解耦策略，在分类（+5-16%）、异常检测（+20%）、插补（+50%）和相似性检索（+25%）四大任务上超越 10-10…
tags:
  - "ICLR 2026"
  - "时间序列"
  - "Time Series Pre-trained Model"
  - "Disentangled Representations"
  - "Dual-Space Reconstruction"
  - "Anomaly Detection"
  - "Tiny Model"
---

# TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time Series

**会议**: ICLR 2026  
**arXiv**: [2505.13033](https://arxiv.org/abs/2505.13033)  
**代码**: [https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)  
**领域**: Time Series  
**关键词**: Time Series Pre-trained Model, Disentangled Representations, Dual-Space Reconstruction, Anomaly Detection, Tiny Model

## 一句话总结

提出 TSPulse，仅 1M 参数的超轻量时间序列预训练模型，通过双空间掩码重建和双嵌入解耦策略，在分类（+5-16%）、异常检测（+20%）、插补（+50%）和相似性检索（+25%）四大任务上超越 10-100 倍大的模型。

## 研究背景与动机

时间序列分析涵盖预测、异常检测、插补、分类和检索等多种下游任务。近年来，借鉴 NLP 和 CV 的成功，时间序列社区开始探索大规模预训练模型：

**专用模型**：TimesFM、Chronos、Moirai 专注于预测任务

**通用模型**：Moment、UniTS 扩展到分类、异常检测和插补

**跨域模型**：Time-LLM、GPT4TS 尝试将 LLM 适配到时间序列

核心问题：**现有预训练模型参数量巨大**（数百M到数十亿），导致部署和微调成本高昂。TTM 证明 1-5M 参数的紧凑模型可在预测任务上提供竞争性性能，但仅限于预测。

研究空白：**能否构建一个 ~1M 参数的预训练模型，同时在多种非预测诊断任务上达到 SOTA？**

## 方法详解

### 整体框架

TSPulse 基于 TSMixer 轻量架构构建，核心流程为：
- 输入 $\mathbf{X} \in \mathbb{R}^{S \times C}$ → 掩码 → 双空间编码（时域+频域）→ TSMixer骨干 → 迷你解码器 → 多目标输出头

### 关键设计

1. **双空间掩码重建（Dual-Space Masked Reconstruction）**

    - 同时在时域和频域进行掩码输入的重建
    - 核心直觉：某些模式在时域更易检测（如突刺），其他在频域更显著（如周期性）
    - 时域输入 $\mathbf{X}_m$ 通过 FFT 转换获得频域表示 $\mathbf{X}^f_m$
    - 关键设计：**不显式掩码频域**，而是将时域掩码信号直接送入 FFT，自然传播掩码到频域
    - 编码后拼接：$\mathbf{Input}_E = [\mathbf{Time}_E; \mathbf{FFT}_E; \mathbf{Reg}_E] \in \mathbb{R}^{C \times K \times D}$

2. **双嵌入解耦（Dual-Embedding Disentanglement）**

    - **详细嵌入**（前 $2N$ 个 patch embedding）：用于全信号重建，捕获细粒度时域和频域模式
    - **语义嵌入**（后 $R$ 个 register embedding）：用于高层语义重建，编码全局特征
    - 语义嵌入通过两个任务监督：
        - 频率签名预测：$\mathcal{L}_{prob} = \text{CE}(\mathbf{X}^f_{prob}, \mathbf{Y}^f_{prob})$（对数幅值谱的softmax分布）
        - 短期预测：$\mathcal{L}_{future} = \text{MSE}(\mathbf{X}_{future}, \mathbf{Y}_{future})$
    - 设计动机：不同下游任务需要不同层次的信息——分类需要语义嵌入，插补需要详细嵌入

3. **TSLens（分类微调组件）**

    - 替代标准池化的学习机制，自适应地从双嵌入中提取相关特征
    - 通过 mini decoder（预训练权重初始化 + 通道混合）→ 降维投影 → flatten → 线性分类头
    - 动态聚焦局部和全局表示中最具信息量的特征

4. **多头三角测量异常检测（Multi-Head Triangulation）**

    - 利用三个预测头从不同视角检测异常：
        - $\text{Head}_{time}$：时域重建偏差 → 检测突刺异常
        - $\text{Head}_{fft}$：频域重建偏差 → 检测周期性异常
        - $\text{Head}_{future}$：短期预测偏差 → 检测趋势异常
    - 融合策略：$\text{Head}_{ensemble}$（最大值融合）或 $\text{Head}_{triang.}$（基于小验证集选最佳头）
    - 首个在单一轻量框架中统一多空间输出进行三角测量的预训练模型

5. **混合掩码策略（Hybrid Masking）**

    - 传统块掩码不适用于真实世界插补（缺失值是不规则的）
    - 混合策略：同时掩码完整 patch 和部分点级位置
    - 关键设计：掩码 token $\mathbf{M} \in \mathbb{R}^{1 \times pl}$ 定义在原始 patch 级别（非嵌入空间），支持灵活的部分掩码
    - 消融显示：去除混合预训练的模型在混合掩码评估下性能暴降 79%

6. **通道混合的恒等初始化**

    - 预训练采用单变量模式（channel-independent）
    - 微调时启用通道混合（channel-mixing），但新增的混合层用**恒等权重初始化**
    - 避免随机初始化在预训练层之间造成激活突变和梯度不稳定

### 损失函数 / 训练策略

多目标加权损失函数的联合最小化：
- $\mathcal{L}_{time1} = \text{MSE}(\mathbf{X}, \mathbf{Y})$：时域重建（仅掩码位置）
- $\mathcal{L}_{time2} = \text{MSE}(\mathbf{X}, \mathbf{Y}')$：从FFT空间反变换的时域重建
- $\mathcal{L}_{fft} = \text{MSE}(\mathbf{X}^f, \mathbf{Y}^f)$：频域重建
- $\mathcal{L}_{prob} = \text{CE}(\mathbf{X}^f_{prob}, \mathbf{Y}^f_{prob})$：频率签名
- $\mathcal{L}_{future} = \text{MSE}(\mathbf{X}_{future}, \mathbf{Y}_{future})$：短期预测

任务特化预训练：通过重新加权损失头优先级实现特化（如AD保留所有头，分类强调时域+概率头）。

预训练在 ~1B 时序样本上，8×A100 GPU 仅需一天。

## 实验关键数据

### 异常检测（TSB-AD 榜单，Figure 4）

| 方法 | 单变量 VUS-PR | 多变量 VUS-PR |
|------|-------------|-------------|
| Sub-PCA (之前SOTA) | 0.42 | - |
| CNN (之前SOTA) | - | 0.31* |
| MOMENT (ZS) | 0.38 | - |
| **TSPulse (ZS)** | **0.48** (+14%) | **0.36** (+16%) |
| **TSPulse (FT)** | **0.52** (+24%) | **0.36** (+26%*) |

*TSPulse 同时位列 TSB-AD 单变量和多变量排行榜第一

### 分类（UEA 29 数据集，Figure 5）

| 方法 | 参数量 | 平均准确率 |
|------|--------|----------|
| VQShape | ~37M | 0.701 |
| MOMENT | ~110M | 0.675 |
| UniTS | ~10M | 0.634 |
| **TSPulse** | **~1M** | **0.733** (+5-16%) |

### 插补（6个LTSF基准，Figure 6 - 混合掩码）

| 方法 | 设置 | 平均 MSE↓ |
|------|------|----------|
| MOMENT | ZS | 0.276 |
| UniTS (PMT) | PMT | 0.170 |
| **TSPulse** | **ZS** | **0.074** (+56-73%) |
| TimesNet | FT | 0.080 |
| **TSPulse** | **FT** | **0.039** (+49-51%) |

### 消融实验（Table 1）

**分类消融**:

| 变体 | 准确率 | 下降 |
|------|--------|------|
| TSPulse (完整) | 0.747 | - |
| w/o Short Embedding | 0.689 | -8% |
| w/o Long Embedding | 0.681 | -10% |
| w/o Masking | 0.691 | -8% |
| w/o CM Identity Init | 0.685 | -9% |
| w/o TSLens (Avg-Pool) | 0.675 | -11% |
| w/o TSLens (Max-Pool) | 0.645 | -16% |
| w/o Dual-space | 0.696 | -7% |

### 效率对比（Table 23）

| 模型 | 参数(M) | GPU推理(ms) | CPU推理(s) | 内存(GB) |
|------|---------|------------|-----------|---------|
| **TSPulse** | **1.06** | **7.16** | **0.06** | **0.39** |
| MOMENT(small) | 35.34 (33×) | 32.57 (5×) | 2.74 (46×) | 0.56 |
| MOMENT(large) | 341.24 (322×) | 405.42 (57×) | 21.98 (366×) | 2.30 |
| Chronos(tiny) | 8.39 (8×) | 39.81 (6×) | 66.15 (1103×) | 2.91 |

### 关键发现

1. **1M 参数击败 10-100 倍大的模型**：模型大小不是唯一决定因素，架构设计同样重要
2. **双空间学习至关重要**：去除频域分支导致分类下降 7%，插补下降 8%
3. **混合掩码预训练是插补性能的关键**：纯块掩码在混合掩码评估下暴降 79%
4. **TSLens 显著优于标准池化**：-11% (avg-pool) 和 -16% (max-pool) 的下降证明了学习注意力的价值
5. **Register token 的语义嵌入对失真鲁棒**：对噪声、幅值变化、时间偏移不敏感，对频率和形状敏感

## 亮点与洞察

- **"小而美"的哲学**：1M 参数就够了，关键在于精巧的架构设计（双空间、双嵌入、多头三角测量）
- **解耦表示的价值**：细粒度嵌入 vs 语义嵌入的分离使不同任务可以选择最适合的表示
- **多头三角测量的巧妙之处**：不同重建头天然擅长不同类型的异常，融合胜过单一视角
- **零样本即超越训练模型**：TSPulse 的零样本异常检测超越了所有在目标数据上训练的模型
- **CPU友好**：0.06秒的CPU推理时间使得GPU-free部署成为可能
- **IBM Granite 系列**：开源在 HuggingFace 上，实用性强

## 局限与展望

1. 目前未涉及预测任务（forecasting），但紧凑模型在预测上的能力已由 TTM 验证
2. 预训练数据主要覆盖特定领域（能源、交通等），其他领域的迁移性能有待验证
3. 单变量预训练 + 多变量微调的两阶段设计可能不是最优的
4. 增量学习能力缺失：无法在不遗忘旧知识的情况下持续更新
5. 少样本分类能力有待探索
6. 跨模态融合（如时间序列+文本）是有前景的未来方向

## 相关工作与启发

- **TTM (Tiny Time Mixers)**：紧凑时间序列预训练模型的先驱，但仅限预测任务
- **MOMENT**：通用时间序列基础模型，T5-encoder 架构，参数量 35-341M
- **Chronos**：T5-style 编解码器，专注预测，0.06-709M 参数
- **UniTS**：prompt-tuned 多任务模型
- **TSMixer**：TSPulse 的骨干网络，MLP-Mixer 范式替代 Transformer
- **启发**：紧凑模型 + 任务特化预训练 + 精巧的后处理组件 = 高效且强大的基础模型设计范式

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （双空间双嵌入解耦 + 多头三角测量 + 混合掩码，多项创新组合）
- 实验充分度: ⭐⭐⭐⭐⭐ （75+数据集，4大任务，全面消融，效率分析，嵌入敏感性分析）
- 写作质量: ⭐⭐⭐⭐ （内容详尽，逻辑清晰，附录极其丰富）
- 价值: ⭐⭐⭐⭐⭐ （1M参数超越100倍大模型，开源可用，对部署友好）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SwiftTS: A Swift Selection Framework for Time Series Pre-trained Models via Multi-task Meta-Learning](swiftts_a_swift_selection_framework_for_time_series_pre-trained_models_via_multi.md)
- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](../../ICML2026/time_series/learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)

</div>

<!-- RELATED:END -->
