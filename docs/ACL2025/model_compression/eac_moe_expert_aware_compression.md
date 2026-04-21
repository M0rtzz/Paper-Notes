---
title: >-
  [论文解读] EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
description: >-
  [ACL 2025][模型压缩][MoE 量化] EAC-MoE 深入分析 MoE 模型的专家选择特性，提出两个互补模块——量化时通过逐层校准路由器缓解 expert-shift 问题（QESC），推理时基于专家选择频率动态剪枝不重要专家（PESF），在 4 个 MoE 模型上实现显著的内存压缩和推理加速且精度损失极小。
tags:
  - ACL 2025
  - 模型压缩
  - MoE 量化
  - 路由器校准
  - 动态专家剪枝
  - expert-shift
  - 推理加速
---

# EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

**会议**: ACL 2025  
**arXiv**: [2508.01625](https://arxiv.org/abs/2508.01625)  
**代码**: —  
**领域**: 模型压缩、MoE  
**关键词**: MoE 量化、路由器校准、动态专家剪枝、expert-shift、推理加速  

## 一句话总结

EAC-MoE 深入分析 MoE 模型的专家选择特性，提出两个互补模块——量化时通过逐层校准路由器缓解 expert-shift 问题（QESC），推理时基于专家选择频率动态剪枝不重要专家（PESF），在 4 个 MoE 模型上实现显著的内存压缩和推理加速且精度损失极小。

## 研究背景与动机

- **MoE 部署的双重瓶颈**：(1) 虽然 MoE 仅激活部分专家，但所有专家权重必须加载到 GPU 显存中（Mixtral-8x7B 激活参数量类似 LLaMA2-13B，但总参数量约为其 4 倍，占 94GB 显存）；(2) 低激活参数并不等价于推理加速——在长序列/批量推理中，不同 token 选择不同专家，所有专家仍需参与计算。
- **直接应用 Dense LLM 压缩方法的问题**：(1) 量化方面，低比特量化会导致路由器选错专家（expert-shift），是 MoE 性能下降的主要因素；(2) 剪枝方面，静态剪枝无法适应不同任务的专家偏好差异。
- **核心观察**：MoE 模型在不同类型任务（Math、Code、QA/CR）上展现**完全不同的专家偏好**——同类任务内的专家选择频率高度相似（余弦相似度 >0.8），但跨类任务差异显著。这意味着：量化时应确保正确选择专家而非永久评估专家重要性；剪枝时应动态评估当前任务的专家重要性。

## 方法详解

### 整体框架

EAC-MoE 从推理前（预计算）和推理中两个角度压缩 MoE 模型：(1) QESC 在离线量化阶段逐层校准路由器以缓解 expert-shift；(2) PESF 在推理时动态剪枝低频专家以减少计算。两者可组合使用。

### 关键设计

1. **量化伴随专家选择校准（QESC）**：使用 WikiText2 校准数据，逐层执行：先量化 MHSA 组件→校准 MoE 层路由器→量化所有专家。路由器校准采用 TopK-MSE 损失（仅对概率分布中 top-K 类别计算 MSE），避免大量低概率专家主导损失引入噪声。
2. **基于专家选择频率的动态剪枝（PESF）**：在推理时，对输入序列长度 $l$，若某专家被选中次数 $c < \frac{l \times K}{N} \times \alpha$（即选中次数低于平均水平的 $\alpha$ 倍），则该专家对当前序列被剪枝。特点是从专家维度而非 token 维度剪枝，可完全跳过不重要专家的计算。
3. **TopK-MSE 损失**：$\mathcal{L} = \frac{1}{K}\sum_{i \in \text{top-K}(\mathbf{W}\mathbf{x})}((\mathbf{W}\mathbf{x})_i - (\mathbf{W}\hat{\mathbf{x}})_i)^2$，仅聚焦于最可能被选中的 K 个专家，有效避免了对 64 个专家中大量低概率专家的无效优化。

### 损失函数

QESC 路由器校准使用 TopK-MSE 损失。PESF 不涉及训练损失，是纯推理时的动态决策。

## 实验

### 主实验：量化性能对比

| 比特 | 方法 | Mixtral PPL↓ | Mixtral Acc↑ | DeepSeek PPL↓ | DeepSeek Acc↑ |
|------|------|-------------|-------------|-------------|-------------|
| 16 | Baseline | 3.84 | 72.64 | 6.51 | 61.38 |
| 2.06 | GPTQ | 5.51 | 62.56 | 8.27 | 54.88 |
| 2.06 | PMQ | 5.41 | 63.25 | 8.42 | 54.79 |
| 2.06 | **QESC** | **5.09** | **66.31** | **7.99** | **57.05** |
| 3.03 | GPTQ | 4.16 | 68.92 | 6.82 | 59.33 |
| 3.03 | **QESC** | **4.14** | **72.21** | **6.71** | **61.22** |

3.03-bit QESC 在 Mixtral-8x7B 上精度损失仅 0.43%（72.21 vs 72.64），在 DeepSeek 上仅 0.16%。

### 剪枝性能对比

| 方法 | Mixtral Acc↑ | 加速比↑ | Phi3.5 Acc↑ | 加速比↑ |
|------|-------------|---------|-----------|---------|
| Baseline | 72.64 | 1.00× | 69.62 | 1.00× |
| EES | 71.40 | 1.06× | 67.96 | 1.05× |
| ODP | 71.98 | 1.05× | 68.92 | 1.04× |
| **PESF (α=0.3)** | **72.19** | **1.08×** | **69.27** | **1.12×** |
| **PESF (α=0.7)** | 58.22 | 1.13× | 67.95 | 1.30× |

### 消融实验：expert-shift 验证

| 配置 | 量化 | Expert-Shift | Mixtral PPL |
|------|-----|-------------|------------|
| 原始模型 | ✘ | ✘ | 3.84 |
| 仅 shift | ✘ | ✔ | 4.17 |
| 仅量化 | ✔ | ✘ | 4.21 |
| 量化+shift | ✔ | ✔ | 4.65 |

Expert-shift 单独造成的 PPL 增加（0.33）与量化误差本身（0.37）几乎相当，验证了校准专家选择的重要性。

### 关键发现

- Expert-shift 是 MoE 量化性能下降的核心因素之一，其影响与权重量化误差本身相当
- QESC 在所有 4 个模型的所有比特配置下均一致优于 GPTQ、PMQ 和 BSP
- PESF 的剪枝阈值 $\alpha=0.3$ 是保守甜点（精度损失 <0.5%），$\alpha=0.7$ 是激进甜点（>1.3× 加速）
- 同类任务内的专家选择频率高度相似这一发现为任务自适应的动态剪枝提供了理论依据
- 组合 QESC (3.03-bit) + PESF (α=0.3) 可将 Mixtral-8x7B 从 93GB 压缩到约 19GB，实现单卡 RTX 3090 部署

## 论文亮点

- Expert-shift 概念的提出和验证清晰有力，为 MoE 量化提供了新视角
- TopK-MSE 损失设计精巧，解决了大量低概率专家主导传统 MSE 损失的问题
- QESC 与 PESF 正交互补，可独立使用也可组合叠加
- 跨任务专家偏好分析为动态剪枝策略提供了扎实的empirical支撑

## 局限性

- 激进剪枝 (α=0.7) 在 Mixtral-8x7B 上精度下降显著（72.64→58.22），说明该模型每层仅 8 个专家时冗余度有限
- PESF 需要在推理初期收集专家选择统计信息，对极短序列（single token generation）可能不适用
- 路由器校准依赖 WikiText2 这一特定校准集，跨域泛化能力未充分验证
- 未与 QuIP、AQLM 等更先进的量化方法对比

## 相关工作

- **MoE 量化**：PMQ、BSP (Li et al., 2024a) 基于专家频率做混合精度量化，但泛化性不足
- **MoE 剪枝**：EES (Lu et al., 2024) 从 token 维度跳过低分专家，加速效果有限
- **Dense LLM 量化**：GPTQ (Frantar et al., 2022)、SmoothQuant (Xiao et al., 2023) 等未考虑 MoE 特性

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](efficientqat.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts](moqae_mixed_precision_kv_cache.md)
- [Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models](l4q_parameter_efficient_quantization_aware_finetuning.md)

<!-- RELATED:END -->
