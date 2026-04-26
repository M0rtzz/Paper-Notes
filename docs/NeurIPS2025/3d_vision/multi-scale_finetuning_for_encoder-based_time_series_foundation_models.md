---
title: >-
  [论文解读] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models
description: >-
  [NeurIPS 2025][3D视觉][时间序列基础模型] 提出 MSFT（Multi-Scale FineTuning），通过因果分析揭示 naive 微调忽略尺度混淆问题，设计多尺度建模框架对 encoder-based 时间序列基础模型进行高效微调，显著超越 naive 微调和从头训练的 SOTA 方法。
tags:
  - NeurIPS 2025
  - 3D视觉
  - 时间序列基础模型
  - 多尺度建模
  - 微调
  - 因果推断
  - 参数高效微调
---

# Multi-Scale Finetuning for Encoder-based Time Series Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.14087](https://arxiv.org/abs/2506.14087)  
**代码**: https://github.com/zqiao11/MSFT (有)  
**领域**: 时间序列 / 基础模型微调  
**关键词**: 时间序列基础模型, 多尺度建模, 微调, 因果推断, 参数高效微调

## 一句话总结

提出 MSFT（Multi-Scale FineTuning），通过因果分析揭示 naive 微调忽略尺度混淆问题，设计多尺度建模框架对 encoder-based 时间序列基础模型进行高效微调，显著超越 naive 微调和从头训练的 SOTA 方法。

## 研究背景与动机

时间序列基础模型（TSFM）在零样本预测上表现出色，但如何高效微调使其适配下游任务仍是一个被忽视的问题。现有微调策略（全量微调、线性探测）存在两个核心问题：

**时间序列的多尺度本质被忽视**：时间序列数据可在不同采样尺度下展现不同的时间模式。例如小时级别的能源消耗体现局部用电模式，日级别则体现宏观趋势。Naive 微调只在原始尺度上学习，容易过拟合到单一尺度的模式。

**TSFM 的多尺度能力未被利用**：TSFM 在多尺度数据集上预训练，天然具备多尺度预测能力，但 naive 微调限制了模型只在原始尺度上学习，浪费了预训练获得的多尺度知识。

作者从因果推断角度分析：尺度 $S$ 作为混淆变量，同时影响输入 $X$（不同尺度的时间模式）和模型激活的知识 $M$（尺度特定的预训练知识），引入后门路径 $X \leftarrow S \rightarrow M \rightarrow Y$，导致虚假关联。因此需要通过 do-calculus 干预 $P(Y|do(X))$ 消除混淆效应。

## 方法详解

### 整体框架

MSFT 基于因果干预框架，通过后门调整实现 $P(Y|do(X)) = \sum_s P(Y|X,S=s,M=g(X,s))P(s)$。具体流程：
1. 将原始时间序列下采样生成多尺度序列（因子为 $2^k$）
2. 各尺度独立分词、编码
3. 拼接多尺度 token 序列，通过解耦的依赖建模捕获尺度内和跨尺度依赖
4. 多尺度预测结果加权融合

### 关键设计

1. **尺度特定知识激活**：冻结预训练参数，为每个尺度引入独立的线性适配器（input projection 层）和独立的 LoRA 模块（attention 层）。这避免了不同尺度 token 分辨率差异导致的干扰，实现了方程中 $M=g(X,s)$ 的功能——激活尺度特定的 TSFM 知识。

2. **解耦 Token 依赖建模**：分为两部分：

    - **In-scale attention**：通过掩码 $\mathbf{M}_{in}$ 确保 token 只关注同尺度的 token，避免不同尺度间因时间索引不对齐产生的虚假注意力
    - **Cross-scale aggregator**：双向（粗到细 C2F、细到粗 F2C）逐层融合相邻尺度的信息。通过线性映射 $\phi_{i,j}^l$ 将 token 映射到共享空间后，按时间对齐进行融合：C2F 用 Repeat 上采样，F2C 用 AvgPool 下采样

3. **多尺度混合输出**：各尺度独立预测 $\hat{Y}_i$，训练目标为加权损失 $\mathcal{L}_{pred} = \sum_i w_i \mathcal{L}_{pred,i}$，权重 $w_i$ 通过 softmax 学习。推理时将各尺度预测上采样到原始分辨率后加权求和，起到集成效果，缓解过拟合。

### 损失函数 / 训练策略

使用各 TSFM 原始的预测损失（MSE 或 NLL），按学习到的尺度权重加权求和。冻结预训练参数，只训练适配器、LoRA、跨尺度聚合器和尺度权重。

## 实验关键数据

### 主实验（长序列预测，MSE 指标，越低越好）

| 数据集 | 指标 | MSFT(Moirai-Base) | Full FT(Moirai-Base) | TimeMixer | SimpleTM | 提升 |
|--------|------|------|----------|------|------|
| ETTm1 | MSE | **0.332** | 0.368 | 0.381 | 0.381 | -9.8% vs FT |
| ETTm2 | MSE | **0.247** | 0.258 | 0.275 | 0.275 | -4.3% vs FT |
| Weather | MSE | **0.213** | 0.232 | 0.240 | 0.243 | -8.2% vs FT |
| Electricity | MSE | **0.169** | 0.173 | 0.182 | 0.166 | -2.3% vs FT |

MSFT 在 3 个 TSFM backbone（Moirai、Moment、UniTS）上均一致性超越 naive 微调、LoRA、AdaLoRA 等参数高效微调方法。

### 消融实验

| 配置 | ETTm1 MSE | Weather MSE | 说明 |
|------|---------|------|------|
| MSFT (完整) | **0.332** | **0.213** | 全部组件 |
| 去除 cross-scale aggregator | 0.340 | 0.220 | 跨尺度融合重要 |
| 去除 scale-specific LoRA | 0.345 | 0.222 | 尺度特定知识激活重要 |
| 去除 multi-scale mixing | 0.338 | 0.218 | 加权融合有帮助 |
| 单一尺度（K=0） | 0.361 | 0.230 | 多尺度建模是核心 |

### 关键发现
- 多尺度建模是提升的核心来源，各组件（尺度特定适配器、解耦依赖建模、加权混合）均有独立贡献
- MSFT 不仅超越各种微调方法，还超越了从头训练的 SOTA 深度学习方法（TimeMixer、SimpleTM 等）
- 在概率预测任务上同样适用（Moirai 的概率预测能力）

## 亮点与洞察
- 从因果分析的角度揭示了 naive 微调的本质问题——尺度作为混淆变量引入虚假关联，这一理论视角新颖且有说服力
- 框架设计简洁通用，可适配不同的 encoder-based TSFM 架构
- 冻结预训练参数 + 轻量适配器的设计，参数效率高且避免灾难性遗忘

## 局限与展望
- 仅针对 encoder-based TSFM，未探索 decoder-only（如 TimesFM）或 encoder-decoder（如 Chronos）架构
- 多尺度数量 K 需要手动选择，未提供自适应机制
- 下采样策略固定为平均池化，可以探索更多下采样方式（如小波变换）

## 相关工作与启发
- **vs TimeMixer**: TimeMixer 从头训练多尺度模型，而 MSFT 利用预训练 TSFM 的多尺度能力进行微调，思路不同但目标一致
- **vs LoRA**: 普通 LoRA 不区分尺度，MSFT 的尺度特定 LoRA 更能激活对应的预训练知识
- **vs Scaleformer**: Scaleformer 从粗到细迭代精化，MSFT 双向融合更灵活

## 评分
- 新颖性: ⭐⭐⭐⭐ 因果视角分析微调问题新颖，但多尺度建模本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个 backbone、多个数据集、多种微调方法对比、完善消融
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，因果分析和方法设计有机结合
- 价值: ⭐⭐⭐⭐ 为 TSFM 微调提供了实用且有理论支撑的通用框架

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)
- [\[NeurIPS 2025\] Frequency Matters: When Time Series Foundation Models Fail Under Spectral Shift](frequency_matters_when_time_series_foundation_models_fail_under_spectral_shift.md)
- [\[AAAI 2026\] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos](../../AAAI2026/3d_vision/uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)
- [\[ICLR 2026\] FeDaL: Federated Dataset Learning for General Time Series Foundation Models](../../ICLR2026/3d_vision/fedal_federated_dataset_learning_for_general_time_series_foundation_models.md)
- [\[ICLR 2026\] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models](../../ICLR2026/3d_vision/dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)

<!-- RELATED:END -->
