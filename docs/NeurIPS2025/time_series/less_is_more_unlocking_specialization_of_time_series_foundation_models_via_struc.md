---
title: >-
  [论文解读] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning
description: >-
  [NeurIPS 2025][时间序列][时间序列基础模型] 揭示预训练时间序列基础模型（TSFM）存在固有的任务相关稀疏性，提出"先剪枝再微调"范式——通过结构化剪枝移除任务无关参数，使剪枝后微调的小模型显著超越直接微调的原模型，甚至胜过强专用基线。
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "时间序列基础模型"
  - "结构化剪枝"
  - "模型特化"
  - "微调"
  - "稀疏性"
---

# Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning

**会议**: NeurIPS 2025  
**arXiv**: [2505.23195](https://arxiv.org/abs/2505.23195)  
**代码**: [有](https://github.com/SJTU-DMTai/Prune-then-Finetune)  
**领域**: 时间序列预测 / 模型压缩  
**关键词**: 时间序列基础模型, 结构化剪枝, 模型特化, 微调, 稀疏性

## 一句话总结

揭示预训练时间序列基础模型（TSFM）存在固有的任务相关稀疏性，提出"先剪枝再微调"范式——通过结构化剪枝移除任务无关参数，使剪枝后微调的小模型显著超越直接微调的原模型，甚至胜过强专用基线。

## 研究背景与动机

时间序列基础模型（TSFM）如 TimesFM、Moirai、Time-MoE 等通过大规模预训练获得了出色的零样本预测能力。然而，一个令人意外的事实是：**即使经过微调，TSFM 也无法一致地超越 PatchTST 等从零训练的小型专用模型**。

作者通过实证分析揭示了 TSFM 的两个关键特性：

**注意力头输出稀疏**：多达 50% 的注意力头产生的输出相对残差范数 < 1%，几乎不影响预测。

**FFN 激活稀疏**：多达 60% 的 FFN 中间通道从未被激活（激活概率 < 5%）。

更重要的是，**稀疏模式因数据集而异**——在 Weather 和 ETTm1 上表现出不同的激活模式。这说明 TSFM 已学会为不同任务选择性激活子网络。传统微调保留完整架构并修改所有参数，可能破坏这些有价值的任务相关子结构。

## 方法详解

### 整体框架

提出 Prune-then-Finetune 两阶段部署流程：第一阶段通过结构化剪枝实现**架构特化**（移除不重要的通道和注意力头）；第二阶段通过微调实现**权重特化**（在精简参数空间内优化）。

### 关键设计

1. **剪枝单元定义**: 剪枝的最小单元是线性层的输入/输出通道。对每个线性层 $f(\cdot)$，引入二元掩码 $\mathbf{m}^{\text{in}} \in \{0,1\}^{d_{\text{in}}}$ 和 $\mathbf{m}^{\text{out}} \in \{0,1\}^{d_{\text{out}}}$：$\mathbf{h} = f(\mathbf{x} \odot \mathbf{m}^{\text{in}}) \odot \mathbf{m}^{\text{out}}$。由于残差连接的存在，输出通道掩码和下一层输入通道掩码需独立处理。

2. **基于损失的重要性评分**: 通道 $i$ 的重要性定义为移除它后损失的变化量 $s_i = |\mathcal{L}(\mathbf{m} - \boldsymbol{e}_i) - \mathcal{L}(\mathbf{m})|$。使用二阶 Taylor 展开近似：
    $s_i \approx \left| -\frac{1}{N}\sum_{n=1}^{N}\frac{\partial \mathcal{L}_n}{\partial m_i} + \frac{1}{2N}\sum_{n=1}^{N}\left(\frac{\partial \mathcal{L}_n}{\partial m_i}\right)^2 \right|$
   用 Fisher 信息近似二阶导。当通道始终输出零激活时，重要性自动为零，与 FFN 稀疏性分析一致。

3. **渐进式剪枝（Progressive Pruning）**: 避免一次剪枝过多，逐批次进行：每个 batch 计算当前重要性分数，用指数移动平均 $\tilde{s}^{(j)} = \alpha s^{(j)} + (1-\alpha)\tilde{s}^{(j-1)}$ 平滑，全局排序后移除最不重要的 K 个通道。遍历一个或多个 epoch 后完成剪枝，再微调残余参数。

### 损失函数 / 训练策略

- 剪枝阶段使用下游数据集的预测损失（MSE）计算通道重要性
- 微调阶段使用标准 MSE 优化剩余参数
- 超参数（学习率、剪枝比例）在验证集上调优
- 自回归 TSFM 每个数据集训练一个模型复用多个预测长度；非自回归 TSFM 按预测长度分别训练

## 实验关键数据

### 主实验（7 个 TSFM × 6 个数据集）

以 ETTm1 数据集为例（MSE，预测步 96/192/336/720 平均值）：

| 模型 | 微调 MSE | 剪枝+微调 MSE | 改进 | vs PatchTST |
|------|----------|-------------|------|-------------|
| Time-MoE_base | 0.391 | **0.385** | -1.5% | 胜 |
| Time-MoE_large | 0.374 | **0.369** | -1.3% | 胜 |
| Timer-XL | 0.410 | **0.396** | -3.4% | 胜 |
| Moirai_base | 0.418 | **0.407** | -2.6% | 胜 |
| Moirai_large | 0.429 | **0.423** | -1.4% | 胜 |
| Chronos-bolt | 0.422 | **0.421** | -0.2% | 胜 |
| PatchTST（基线） | 0.398 | — | — | — |

### 消融实验

| 对比配置 | 说明 |
|---------|------|
| 直接微调 vs 剪枝+微调 | 83% 的任务中剪枝+微调更优 |
| 最大改进 | 22.8%（Time-MoE_base 在 ETTm2） |
| 最大下降 | 仅 1%（剪枝+微调的下界风险可控） |
| 胜率 vs PatchTST | 从 90% 提升到 **100%**（选最佳 TSFM 时） |
| 可剪枝比例 | 最多 **90%** 参数被移除同时性能提升 |
| 推理加速 | 最高 **7× 加速** |

### 关键发现

- **剪枝提升性能而非仅压缩**：与 LLM 剪枝通常导致性能损失不同，TSFM 剪枝反而提升性能
- **更长预测步长时收益更大**：自回归模型在 720 步预测上的改进最显著（减少误差累积）
- **剪枝起到正则化作用**：相当于在任务相关参数子空间内进行微调，避免过拟合到噪声特征
- **跨域迁移性**：在一个数据集上剪枝的模型对同领域其他数据集的零样本性能也更好
- **LoRA 不足以解决问题**：参数高效微调（LoRA）不优于全微调，因为问题的根源在于架构冗余

## 亮点与洞察

- **"Less is More" 的洞察非常深刻**：TSFM 的性能瓶颈不在于参数不足，而在于无关参数干扰了特化过程
- 将剪枝从"压缩工具"重新定位为"特化工具"，是本文最大的观念转变
- 稀疏性分析（注意力头输出范数 + FFN 激活概率）提供了直觉基础
- 时间序列的非平稳性使得持久有效的特征稀少，这是剪枝特别适合 TSFM 的根本原因

## 局限与展望

- 剪枝比例需要为每个数据集调优，增加了部署成本
- 目前使用全局统一的剪枝策略，未考虑层级差异（浅层 vs 深层）
- 渐进式剪枝需要额外的训练开销（though 比完整微调通常更快）
- 未探索非结构化剪枝与结构化剪枝的组合

## 相关工作与启发

- LLM-Pruner 提出的基于损失的重要性评分是直接基础，但 LLM 剪枝通常损失性能
- 本文揭示的"剪枝反而提升性能"现象在 LLM 中罕见，表明时间序列与语言任务的本质差异
- 思路可推广：任何具有异构性的基础模型，都可能通过剪枝实现更好的下游特化

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将剪枝作为特化手段而非压缩手段，观念转变有价值
- 实验充分度: ⭐⭐⭐⭐⭐ — 7个 TSFM、6个数据集、多维度分析、完整消融
- 写作质量: ⭐⭐⭐⭐ — 实证分析→方法→实验链条清晰
- 价值: ⭐⭐⭐⭐⭐ — 为 TSFM 部署提供了简单有效的标准流程

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] How Foundational are Foundation Models for Time Series Forecasting?](how_foundational_are_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] SEMPO: Lightweight Foundation Models for Time Series Forecasting](sempo_lightweight_foundation_models_for_time_series_forecasting.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)

</div>

<!-- RELATED:END -->
