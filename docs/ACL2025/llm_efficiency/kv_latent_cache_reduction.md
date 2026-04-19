---
title: >-
  [论文解读] KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding
description: >-
  [ACL 2025][LLM效率][KV Cache压缩] KV-Latent 通过直接缩减预训练模型中 Key/Value 注意力头的维度（将 KV 向量映射到低维隐空间），配合两阶段微调策略和频率感知的 RoPE 修改，仅用不到 1% 预训练量的额外训练就实现 KV Cache 50-87% 的压缩，同时基本保持模型性能。
tags:
  - ACL 2025
  - LLM效率
  - KV Cache压缩
  - 注意力头维度缩减
  - RoPE频率感知
  - 知识蒸馏
  - 推理加速
---

# KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding

**会议**: ACL 2025  
**arXiv**: [2507.11273](https://arxiv.org/abs/2507.11273)  
**代码**: https://github.com/ShiLuohe/KV-Latent  
**领域**: LLM效率  
**关键词**: KV Cache压缩, 注意力头维度缩减, RoPE频率感知, 知识蒸馏, 推理加速

## 一句话总结

KV-Latent 通过直接缩减预训练模型中 Key/Value 注意力头的维度（将 KV 向量映射到低维隐空间），配合两阶段微调策略和频率感知的 RoPE 修改，仅用不到 1% 预训练量的额外训练就实现 KV Cache 50-87% 的压缩，同时基本保持模型性能。

## 研究背景与动机

**领域现状**：Transformer 解码器架构在推理时需要为每个 token 保留 Key 和 Value 状态（KV Cache），随上下文长度线性增长，成为内存和带宽的首要瓶颈。现有 KV Cache 压缩方法主要在三个层面操作：注意力头级别（MQA/GQA 共享 KV 头）、层级别（跨层复用 KV Cache）、token 级别（驱逐/合并低重要性 token）。

**现有痛点**：头级别方法（GQA）已被广泛采用但压缩上限有限；层级别方法因跨层复用的非连续性无法优化计算强度；token 级别方法依赖 attention score，与 FlashAttention 等加速方案不兼容，且细粒度控制困难。关键是，直接缩减每个注意力头内部的 Key/Value 向量维度这一方向几乎未被探索。

**核心矛盾**：在 MHA 中，$d_h \times n_h = d$ 这个约束被默认为不可打破，但 GQA 等工作已证明 KV Cache 的低秩特性——存储完整 $d$ 维向量并非必要。问题在于：能否进一步解耦 $d_{qk}$ 和 $d_{vo}$，直接在维度层面压缩 KV Cache？

**本文目标** (1) 如何从已有预训练模型出发，以极小的额外训练代价缩减注意力头维度；(2) RoPE 在低维向量上严重不稳定（维度 <32 时振荡噪声与衰减信号同量级），如何修复；(3) Key 和 Value 的压缩对性能影响是否对称。

**切入角度**：作者观察到 $K$ 和 $V$ 本质上是将 $d$ 维隐状态下采样到 $d_h$ 维的低秩变换，对应的 $Q^\top$ 和 $O$ 做上采样。KV Cache 存储的就是这些低秩变换的结果——既然已经是隐空间表示，那就可以进一步压缩这个隐空间。

**核心 idea**：通过均匀下采样直接裁剪预训练模型的 K/V 投影矩阵维度，再用层内蒸馏+端到端微调两阶段恢复性能，同时用频率感知的 RoPE 采样消除低维下的高频噪声。

## 方法详解

### 整体框架

输入一个预训练好的 LLM（如 LLaMA-3-8B），将每层注意力的 $W_K, W_V$ 矩阵的列维度和 $W_Q$ 的列维度、$W_O$ 的行维度同步裁剪（对 QK 从 $d_h$ 缩至 $d_{qk}$，对 VO 从 $d_h$ 缩至 $d_{vo}$），然后通过两阶段训练恢复性能。推理时 KV Cache 大小直接按维度比例缩减。

### 关键设计

1. **均匀下采样初始化（Model Preparation）**:

    - 功能：从预训练模型的权重中提取缩减后的初始权重
    - 核心思路：因为 RoPE 的存在，注意力头内通道具有旋转对称性，直接对通道做均匀采样（步长为缩减比例）即可保留信息。例如将 $d_{qk}$ 缩减为 1/4 时，$\tilde{W_Q^{(i)}} = W_Q^{(i)}[:, ::4]$。FFN 层则通过 LoRA 适配（rank=256），而非全量调整
    - 设计动机：SVD 分解虽然理论上更优，但 RoPE 引入的旋转与矩阵乘法不满足交换律，使得 SVD 难以直接应用。均匀采样简单有效，且与 RoPE 的 channel-pair 旋转结构天然兼容

2. **两阶段训练策略（Two-Stage Training）**:

    - 功能：分两个阶段用少量数据（仅 1B token，来自 FineWeb-edu）恢复裁剪后的模型性能
    - 核心思路：**Stage I（层内蒸馏）**——冻结原始模型，逐层对齐：取原模型每层输入 $H_i^{(l)}$，分别通过原始层和修改后的层得到 $H_t^{(l)}$ 和 $H_p^{(l)}$，用 MSE loss 最小化两者差距 $\frac{1}{L}\sum_{l=1}^{L}\frac{||H_t^{(l)} - H_p^{(l)}||_2}{x \cdot h}$。**Stage II（端到端训练）**——用 NTP（交叉熵）或蒸馏（KL 散度）做全模型端到端训练，修复层间误差放大
    - 设计动机：Stage I 确保每层独立时输出尽量不变，但深层 LLM 中微小扰动会逐层放大，所以 Stage II 端到端训练必不可少。两阶段结合比直接端到端训练收敛更快、效果更好

3. **频率感知 RoPE（Frequency-aware RoPE）**:

    - 功能：修改 RoPE 在低维 Q/K 上的频率采样策略，消除高频噪声
    - 核心思路：原始 RoPE 的频率 $\theta_j = \theta^{-(j-1)/\delta}$ 在低维时，低编号通道（高频旋转）的振荡周期短于采样间隔，导致数值逼近失效。修改后的采样公式跳过最高频部分并加密低频采样：$\theta_j = \theta^{-2(j-1+d/8)/d}$ (前半通道) 和 $\theta_j = \theta^{-(j-1+3d/4)/d}$ (后半通道)，使得即使维度降至 16 也能保持平滑衰减
    - 设计动机：作者从 $\text{RoPE}_{\theta,d}(x) = \mathbb{1}_d \cdot \mathcal{R}_{\theta,d/2}(x) \cdot \mathbb{1}_d^\top$ 的稳定性分析出发，发现当 $d < 32$ 时自相关函数出现大量负值（意味着远距离相同向量的注意力甚至低于随机向量），根因是高频分量 $\cos(\theta^p)$ 在 $p$ 较大时剧烈振荡，而低维下采样点不足以正确逼近积分

### 损失函数 / 训练策略

- Stage I: MSE loss（层内隐状态对齐）
- Stage II: 两种选择——NTP 用交叉熵 loss（资源少）；蒸馏用 KL 散度 loss（信息量更大但需额外 forward pass）
- 实验发现 NTP 训练在有限数据量下反而优于蒸馏，因为蒸馏需要更多数据才能充分发挥优势
- FFN 使用 LoRA（Up/Down/Gate），rank 对 PPL 影响很小（16→256 仅差 0.04）

## 实验关键数据

### 主实验

| 模型 | $d_{qk}$ | $d_{vo}$ | 方法 | MMLU | OBQA | ARC | Avg | KV Cache↓ | TTFT↓ |
|------|-----------|-----------|------|------|------|-----|-----|-----------|-------|
| LLaMA3-8B | 128 | 128 | Base | 35.3 | 35.5 | 55.5 | 42.1 | - | - |
| LLaMA3-8B | 64 | 64 | Train | 35.0 | 35.1 | 53.8 | 41.3 | ↓50% | ↓8% |
| LLaMA3-8B | 64 | 64 | Distill | 31.0 | 29.1 | 39.1 | 33.1 | ↓50% | ↓8% |
| LLaMA3-8B | 16 | 16 | Train | 31.0 | 29.5 | 38.5 | 33.0 | ↓87% | ↓13% |
| LLaMA2-7B | 128 | 128 | Base | 28.9 | 29.4 | 30.7 | 29.7 | - | - |
| LLaMA2-7B | 64 | 64 | Train | 28.1 | 29.3 | 27.5 | 28.3 | ↓50% | ↓17% |

### 消融实验（不同 $d_{qk}$ 和 $d_{vo}$ 组合）

| $d_{qk}$ | $d_{vo}$ | LogPPL | KV Cache(MB) | 最大上下文(60GB) |
|-----------|----------|--------|-------------|--------------|
| 128 | 128 | baseline | 256 | 0.40M token |
| 64 | 128 | 2.47 | 172 | 0.61M token |
| 128 | 64 | 2.80 | 172 | 0.61M token |
| 64 | 64 | 2.74 | 128 | 0.81M token |
| 16 | 16 | 3.78 | 32 | 3.27M token |

### 关键发现

- **$d_{vo}$ 比 $d_{qk}$ 更重要**：同样压缩到 172MB，保留大 $d_{vo}$（PPL=2.47）远优于保留大 $d_{qk}$（PPL=2.80），说明 Value 携带的信息比 Key 更不可压缩
- **NTP 训练优于蒸馏**：在仅 1B token 训练数据下，NTP 方式(Avg=41.3) 大幅优于蒸馏(Avg=33.1)，蒸馏需要更多数据才能发挥优势
- **GQA 模型更难压缩**：已经使用 GQA 的 LLaMA3 比 MHA 的 LLaMA2 在同等压缩下性能下降更明显，因为 GQA 本身已经做了头级别压缩
- **与 token 级方法正交**：KV-Latent + PyramidInfer(50%压缩) 可叠加使用，PPL 仅从 2.509 升至 2.499，KV Cache 再降 50%
- **LoRA rank 不敏感**：rank 从 16 到 256，LogPPL 变化仅 0.04

## 亮点与洞察

- **维度级压缩是 KV Cache 压缩的新范式**：与 head 级别（GQA）、layer 级别（CLA）、token 级别（eviction）完全正交，可以叠加使用，这打开了一个新的压缩维度
- **频率感知 RoPE 的理论分析很优雅**：将 RoPE 稳定性问题转化为数值积分逼近问题——$\cos(\theta^p)$ 在大 $p$ 区域剧烈振荡，低维采样点不足，所以跳过高频、加密低频。这个 insight 可能对所有需要在低维使用 RoPE 的场景都有价值
- **Key vs Value 信息量不对称**的发现对模型架构设计有启发：未来可以给 Value 分配更多维度、给 Key 更激进地压缩

## 局限与展望

- 仅在 7B/8B 规模验证，更大模型（70B+）上的效果未知
- 仅用 1B token 训练，更多数据下蒸馏是否反超 NTP 训练值得探索
- 与 CLA（跨层复用）的对比缺失，因为 CLA 需要从头预训练
- 未涉及 SFT/RLHF 阶段的影响验证
- $d_{qk}=d_{vo}=16$ 时性能显著下降（NIH 仅 6%），存在压缩下限
- SVD 初始化因 RoPE 的非交换性无法使用，但可能存在变通方案

## 相关工作与启发

- **vs GQA/MQA**: GQA 在头数量维度共享 KV，KV-Latent 在每个头的内部维度压缩——两者完全正交，可叠加
- **vs DeepSeek-V2 (MLA)**: MLA 将 KV 压缩到联合隐空间再解码，思路类似但需要从头预训练；KV-Latent 的优势是可从已有模型出发，仅需极少额外训练
- **vs Token-level methods (H2O, PyramidInfer)**: 这些方法动态丢弃 token 的 KV 状态，KV-Latent 压缩每个 token 存储的维度——两者正交，实验证明可叠加

## 评分

- 新颖性: ⭐⭐⭐⭐ 维度级 KV Cache 压缩方向较新，但整体思路（低秩+蒸馏恢复）并不意外
- 实验充分度: ⭐⭐⭐⭐ 消融全面（QK/VO 独立分析、LoRA rank、跨方法兼容），但模型规模偏小
- 写作质量: ⭐⭐⭐⭐ RoPE 的频率分析部分理论推导清晰，整体结构工整
- 价值: ⭐⭐⭐⭐ 提供了 KV Cache 压缩的新维度，与现有方法正交，实用性强

<!-- RELATED:START -->

## 相关论文

- [SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](spindlekv_layered_kv_cache.md)
- [RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)
- [Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/llm_efficiency/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training](adaptive_grouped_pe_context_window.md)
- [Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)

<!-- RELATED:END -->
