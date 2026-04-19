---
title: >-
  [论文解读] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization
description: >-
  [ICML 2025][位置编码] 通过将 RoPE 中每个维度从单一频率扩展为傅里叶级数表示，并裁剪欠训练的低频分量，实现注意力机制的可靠周期性扩展，从而大幅提升 LLM 的长度泛化能力。
tags:
  - ICML 2025
  - 位置编码
  - 长度泛化
  - 旋转位置编码
  - 傅里叶级数
  - 频谱分析
---

# Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization

**会议**: ICML 2025  
**arXiv**: [2412.17739](https://arxiv.org/abs/2412.17739)  
**代码**: [GitHub](https://github.com/TsinghuaC3I/Fourier-Position-Embedding)  
**领域**: LLM效率  
**关键词**: 位置编码, 长度泛化, 旋转位置编码, 傅里叶级数, 频谱分析

## 一句话总结

通过将 RoPE 中每个维度从单一频率扩展为傅里叶级数表示，并裁剪欠训练的低频分量，实现注意力机制的可靠周期性扩展，从而大幅提升 LLM 的长度泛化能力。

## 研究背景与动机

**领域现状**：大语言模型通常在固定上下文长度上训练，但实际应用中经常需要处理远超训练长度的序列。RoPE（旋转位置编码）通过隐式 NUDFT 赋予注意力模式周期性，理论上允许长度外推。

**现有痛点**：RoPE 的周期性在实际网络中会被严重破坏。即便 RoPE 本身产生单一频率信号，经过线性变换和激活函数后，频谱会发生泄露和畸变，直接导致周期性失效。

**核心矛盾**：RoPE 的理论周期性与实际网络中频谱损伤之间的鸿沟——线性层引发频谱泄露（多频混叠），激活函数产生谐波畸变，而低频分量因训练长度有限而欠训练。

**本文目标** 从频域分析的视角诊断 RoPE 周期性失效的根因，并设计一种新的位置编码方案，使注意力在超出训练长度时仍能保持稳定的周期行为。

**切入角度**：将注意力中的位置依赖项视为离散信号，利用傅里叶分析工具量化频谱损伤，并基于此设计修复方案。

**核心 idea**：把 RoPE 每个维度的单频信号扩展为傅里叶级数，同时裁剪训练不充分的低频分量，从根本上修复注意力的周期性。

## 方法详解

### 整体框架

FoPE 替换标准 RoPE 中的位置编码部分。在 RoPE 中，每个维度对应单一旋转频率 $\omega_m$，位置 $n$ 处的信号为 $e^{i\omega_m n}$。FoPE 将其扩展为 $h_m(n) = H_m(n)(e^{i\omega_m n} + \sum_\omega a_\omega e^{i\omega n})$，即每个维度包含主频率及其傅里叶系数组合。同时，对频率低于 $2\pi/N$（$N$ 为训练长度）的分量执行裁剪，确保所有保留频率在训练范围内有充分周期覆盖。实现上通过一个权重矩阵 $W^F \in \mathbb{R}^{D \times (M - M_0)}$ 参数化，其中 $M_0$ 为被裁剪的低频分量数，$D$ 和初始化标准差 $\sigma$ 可配置。

### 关键设计

1. **傅里叶级数扩展 (Fourier Series Expansion)**:

    - 功能：修复线性层和激活函数导致的频谱损伤
    - 核心思路：既然线性层会将单频信号混叠为多频信号（频谱泄露），激活函数会产生谐波频率（频谱畸变），那么直接在位置编码层面就引入多频表示，让模型能够学习正确的频谱组合来恢复周期性。将 $h_m(n)$ 建模为主频 $\omega_m$ 与其他频率的线性组合
    - 设计动机：论文通过理论分析证明，线性变换 $W$ 作用于 RoPE 信号后，输出变成多个频率的叠加 $\sum_m w_m e^{i\omega_m n}$，相当于 NUDFT 矩阵的行；激活函数 $\sigma(\cdot)$ 进一步引入谐波 $k\omega_m$。因此单频表示根本无法承载网络实际需要的频谱信息

2. **低频裁剪 (Clip Floor to Zero)**:

    - 功能：消除训练不充分的低频分量对外推的负面影响
    - 核心思路：对于频率 $\omega_m < 2\pi/N$ 的分量，其在训练长度 $N$ 内连一个完整周期都无法覆盖，模型无法学到它们的正确行为。直接将这些分量的贡献置零
    - 设计动机：欠训练的低频信号在外推时行为不可预测，会破坏注意力模式的周期性。裁剪后，保留的所有频率在训练范围内至少经历一个完整周期，保证了学习的充分性

3. **零均值约束 (Zero-Frequency Mean Maintenance)**:

    - 功能：确保傅里叶级数的直流分量（零频率）不影响注意力的周期行为
    - 核心思路：在傅里叶展开中显式约束零频分量，使得注意力分数在位置维度上均值稳定，避免随序列长度增长出现偏移
    - 设计动机：零频率分量对应信号的均值偏移，如果不加约束，长序列外推时注意力分数可能系统性漂移，导致概率分布崩溃

### 损失函数 / 训练策略

FoPE 作为位置编码的替换模块，不引入额外损失函数。训练策略与标准语言模型一致，使用自回归交叉熵损失。关键超参数包括傅里叶系数矩阵的维度 $D$ 和初始化标准差 $\sigma$。FoPE 可与 YARN 等长度外推技术组合使用（FoPE + YARN），进一步提升外推效果。在微调场景中，从短上下文（如 2k）微调到目标长度（如 4k），FoPE 表现出显著优于 RoPE 的泛化能力。

## 实验关键数据

### 主实验

| 模型/方法 | 训练长度 | Passkey 2x准确率 | GovReport PPL (4-8k) | MultiNews PPL (4-8k) | TREC Acc (4-8k) |
|:--|:--|:--|:--|:--|:--|
| RoPE | 512 | ~0% | baseline | baseline | baseline |
| FoPE | 512 | ~95% | **+1.15** | **+1.87** | **+14** |
| RoPE + YARN | 2k→4k | 较好 | 一般 | 一般 | 一般 |
| FoPE + YARN | 2k→4k | **最优** | **最优** | **最优** | **最优** |

*基于 SmolLM-1.7B 微调实验。FoPE 在所有评估长度范围内均优于 RoPE，差距在超出训练长度时尤为显著。*

### 消融实验

| 组件 | Passkey 精度变化 | 困惑度变化 | 说明 |
|:--|:--|:--|:--|
| Full FoPE (FS + CF) | 最优 | 最优 | 两个组件协同增效 |
| 仅 Fourier Series (FS) | 显著提升 | 小幅提升 | 贡献最大，修复频谱损伤 |
| 仅 Clip Floor (CF) | 小幅提升 | 域内改善 | 主要帮助域内稳定性 |
| 无任何组件 (RoPE) | 基线 | 基线 | 外推时严重退化 |

### 关键发现

- 傅里叶级数扩展是性能提升的主要来源，说明频谱损伤是 RoPE 外推失败的首要原因
- 低频裁剪在域内（训练长度内）效果更明显，与傅里叶级数组合时产生协同效应
- FoPE 与 YARN 组合的效果远超 RoPE + YARN，说明 FoPE 的改进是正交且互补的
- 在 Passkey Retrieval 任务上，RoPE 在 2 倍训练长度时准确率骤降至接近 0%，而 FoPE 仍能保持约 95%

## 亮点与洞察

- 从信号处理视角深入剖析了 RoPE 失效的机制，将经验观察提升为理论理解：不是 RoPE 设计有误，而是后续网络层破坏了其周期性
- 解决方案优雅地与问题诊断对应：频谱泄露→多频表示，欠训练→频率裁剪
- 方法即插即用，不改变模型架构，不增加推理开销，实用性强
- 与 YARN 等已有方法正交互补，可叠加使用

## 局限与展望

- 论文主要在中小规模模型（SmolLM-1.7B）上验证，更大模型（如 70B+）上的表现有待确认
- 傅里叶系数矩阵引入了额外参数，虽然量不大，但在极端资源受限场景需要评估
- 频率裁剪阈值 $2\pi/N$ 是固定的，是否存在更优的自适应裁剪策略值得探索
- 论文侧重于文本 LLM，在视觉 Transformer 等其他使用 RoPE 的模型上的效果未验证

## 相关工作与启发

- **vs RoPE**: FoPE 是 RoPE 的自然扩展，保留了旋转编码的优势（相对位置感知、外推理论基础），同时通过傅里叶级数修复了实际网络中的频谱损伤
- **vs YARN/NTK-aware**: 这些方法通过调整频率基数或插值策略改善外推，但未从根本上解决频谱损伤问题。FoPE 与它们的改进方向正交
- **vs ALiBi**: ALiBi 通过线性偏置实现位置感知，不依赖旋转机制，但在长序列上的表现通常不如 RoPE 系列

## 评分

- 新颖性: ⭐⭐⭐⭐ 从频谱分析角度诊断 RoPE 失效机制并设计修复方案，角度新颖且有理论深度
- 实验充分度: ⭐⭐⭐⭐ Passkey Retrieval 和多个下游任务验证充分，消融实验清晰
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，从问题诊断到方案设计的逻辑链条完整
- 价值: ⭐⭐⭐⭐ 位置编码的即插即用改进，对 LLM 长序列应用有直接价值

<!-- RELATED:START -->

## 相关论文

- [Group Representational Position Encoding (GRAPE)](../../ICLR2026/signal_comm/group_representational_position_encoding.md)
- [FASA: Frequency-Aware Sparse Attention](../../ICLR2026/signal_comm/fasa_frequency-aware_sparse_attention.md)
- [ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](../../CVPR2025/signal_comm/abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)
- [SepLLM: Accelerate Large Language Models by Compressing One Segment into One Separator](sepllm_accelerate_large_language_models_by_compressing_one_segment_into_one_sepa.md)
- [Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)

<!-- RELATED:END -->
