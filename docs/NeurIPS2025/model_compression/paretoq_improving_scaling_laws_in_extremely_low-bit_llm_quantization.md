---
title: >-
  [论文解读] ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization
description: >-
  [NeurIPS 2025][模型压缩][极低比特量化] 提出 ParetoQ——首个统一 1/1.58/2/3/4 比特量化的框架，通过系统研究训练策略（全精度预训练 vs. QAT 分配）和量化函数设计（提出 SEQ 量化器），发现 2-bit 和 1.58-bit 量化在精度-模型大小折中上优于传统 4-bit，且各比特位宽均达到 SOTA。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 极低比特量化
  - 缩放定律
  - 量化感知训练
  - 2-bit量化
  - Pareto最优
---

# ParetoQ: Improving Scaling Laws in Extremely Low-bit LLM Quantization

**会议**: NeurIPS 2025  
**arXiv**: [2502.02631](https://arxiv.org/abs/2502.02631)  
**代码**: 暂无  
**领域**: 模型压缩  
**关键词**: 极低比特量化, 缩放定律, 量化感知训练, 2-bit量化, Pareto最优

## 一句话总结

提出 ParetoQ——首个统一 1/1.58/2/3/4 比特量化的框架，通过系统研究训练策略（全精度预训练 vs. QAT 分配）和量化函数设计（提出 SEQ 量化器），发现 2-bit 和 1.58-bit 量化在精度-模型大小折中上优于传统 4-bit，且各比特位宽均达到 SOTA。

## 研究背景与动机

LLM 量化领域的核心争论：**最优比特位宽是多少？**

- 一方（Dettmers & Zettlemoyer 2023）认为 4-bit 或 6-bit 是 Pareto 最优
- 另一方（Ma et al. 2024; Kaushal et al. 2024）声称 1.58-bit 足以匹配全精度性能

**为什么结论对立？** 因为缺乏统一框架——各方使用不同的训练方案、不同的量化函数、不同的基线，结论不可比。

**作者的关键观察**：之前的缩放定律研究将搜索空间简化为 $\mathcal{L}(\mathcal{N}, \mathcal{D}, \mathcal{P})$（模型大小、数据量、精度），忽略了两个关键因素：训练策略 $\mathcal{S}_{\text{train}}$ 和比特特定的量化函数 $\mathcal{F}$。正确的搜索空间是 $\mathcal{L}(\mathcal{N}, \mathcal{D}, \mathcal{P}, \mathcal{S}_{\text{train}}, \mathcal{F})$——五维空间。

**核心发现**：2-bit 和 3-bit 之间存在显著的学习行为转变——3-bit 及以上是对预训练权重的"补偿"（权重变化 10-20%），2-bit 及以下是"重建"（权重变化 ~40%）。

## 方法详解

### 整体框架

ParetoQ 的方法论分三步展开：

1. **固定量化函数，寻找最优训练策略**：$\mathcal{L}(\mathcal{N}, \mathcal{D}, \mathcal{S}_{\text{train}} | \mathcal{P}, \mathcal{F})$
2. **固定最优训练策略，寻找最优量化函数**：$\mathcal{L}(\mathcal{N}, \mathcal{F} | \mathcal{P}, \mathcal{D}^*, \mathcal{S}_{\text{train}}^*)$
3. **固定最优训练+量化，比较不同比特位宽**：$\mathcal{L}(\mathcal{N}, \mathcal{P} | \mathcal{F}^*, \mathcal{D}^*, \mathcal{S}_{\text{train}}^*)$

### 关键设计

1. **训练预算分配策略**：在固定总训练预算 $\mathcal{B}_{\text{train}} = \mathcal{B}_{\text{FP}} + \mathcal{B}_{\text{QAT}}$ 下，研究全精度预训练和QAT微调的最优分配比例。在 MobileLLM-125M 上的实验发现：

    - **~90% 用于全精度预训练 + ~10% 用于 QAT** 是最优分配，几乎对所有比特位宽成立
    - 从头 QAT（全部用于量化训练）始终不如先预训练再微调
    - 3-bit/4-bit QAT 约 10B token 饱和，1-bit/1.58-bit/2-bit 约 30B token 饱和
   
   设计动机：低比特量化（≤2-bit）需要更多训练token是因为权重"重建"比"补偿"需要更大的搜索空间。

2. **Stretched Elastic Quant (SEQ) 量化器**：针对 1.58-bit 和 2-bit 提出的关键创新。问题是：2-bit 量化有 4 个量化级别，若包含 0（如 $\{-2,-1,0,1\}$），正数只有一个级别，分布不均衡；若排除 0（如 $\{-1.5,-0.5,0.5,1.5\}$），则均衡但无法表示零值。SEQ 的解：

   $$\mathbf{W}_Q^i = \alpha \left(\lfloor \text{Clip}\left(\frac{\mathbf{W}_R^i}{\alpha}, -1, 1\right) \times \frac{k}{2} - 0.5 \rceil + 0.5 \right) / k \times 2$$

   这同时实现了均衡的量化级别和均匀覆盖全精度权重范围。3-bit/4-bit 仍使用 LSQ（含 0 更优）。

3. **统一量化公式 ParetoQ**：

   $$\mathbf{W}_Q^i = \begin{cases} \alpha \cdot \text{Sign}(\mathbf{W}_R^i), & N_{\text{bit}} = 1 \\ \alpha(\lfloor \text{Clip}(\frac{\mathbf{W}_R^i}{\alpha}, -1, 1) \times k/2 - 0.5 \rceil + 0.5)/k \times 2, & N_{\text{bit}} = 1.58, 2 \\ \alpha \lfloor \text{Clip}(\frac{\mathbf{W}_R^i}{\alpha}, n, p) \rceil, & N_{\text{bit}} = 3, 4 \end{cases}$$

   反向传播使用 STE（Straight-Through Estimator），对权重和缩放因子 $\alpha$ 分别定义梯度。$\alpha$ 初始化：1-bit 用 $\ell_1$ 均值，其余用最大绝对值。

### 训练策略细节

- AdamW 优化器，零权重衰减，16 GPU，每 GPU batch size 8
- 1/1.58/2-bit：12 万步，学习率 $2 \times 10^{-5}$，cosine 衰减
- 3/4-bit：4 万步，学习率 $1 \times 10^{-5}$，cosine 衰减
- 除 embedding 和输出层外所有权重量化

## 实验关键数据

### 主实验：LLaMA-3 8B 各比特位宽

| 方法 | Bits | ARC-e | ARC-c | PIQA | HellaS | WinoG | 平均 | Wiki2 |
|---|---|---|---|---|---|---|---|---|
| Full Precision | 16 | 81.0 | 57.7 | 81.0 | 79.5 | 73.9 | 74.6 | 6.15 |
| EfficientQAT | 2 | 69.3 | 46.8 | 76.4 | 69.0 | 66.3 | 65.5 | 9.6 |
| **ParetoQ** | **2** | **78.5** | **54.5** | **79.2** | **73.8** | **70.0** | **71.2** | **8.0** |
| 1-bit Era | 1.58 | 72.8 | 45.4 | 81.0 | 70.6 | 58.0 | 65.6 | 11.7 |
| **ParetoQ** | **1.58** | **76.3** | **51.4** | **77.7** | **71.9** | **67.7** | **69.0** | **8.6** |
| BiLLM | 1 | 33.2 | 25.6 | 54.6 | 32.7 | 50.5 | 39.3 | 38.5 |
| **ParetoQ** | **1** | **75.5** | **51.9** | **76.6** | **69.4** | **65.6** | **67.8** | **9.5** |

### 消融实验：量化函数选择的影响

| 量化器 | 1.58-bit 准确率 | 2-bit 准确率 | 3-bit 准确率 | 4-bit 准确率 |
|---|---|---|---|---|
| Min-Max (stats) | 差 | 崩溃 | 可用 | 好 |
| 范围裁剪 (stats) | 好 | 好 | 差 | 差 |
| LSQ (learnable) | 中 | 中 | **最优** | **最优** |
| **SEQ (learnable)** | **最优** | **最优** | 略差 | 略差 |

### 关键发现

- **Pareto 曲线颠覆传统认知**：1.58-bit、2-bit、3-bit 在精度-模型大小折中上均优于 4-bit
- ParetoQ 1.58-bit 8B 模型将全精度差距缩小 37.8%（相比 1-bit Era），仅用 30% 训练 token
- **ParetoQ 600M 三值模型超越了之前 SOTA 的 3B 三值模型**——用 1/5 参数
- 2-bit 和 3-bit 之间存在学习行为转变：≥3-bit 是"补偿"（权重小调整），≤2-bit 是"重建"（权重大幅改变）
- 2-bit 具有 CPU 内核加速优势，accuracy-speed 折中优于 4-bit
- 1.58-bit 和 3-bit 在硬件友好性上不如 2-bit（1.58-bit 存储复杂，3-bit 对齐困难）

## 亮点与洞察

- 方法论最大贡献：将混乱的低比特量化领域规范化为五维搜索问题，首次实现了严格的苹果对苹果比较
- SEQ 量化器的设计洞察：低比特场景下量化级别的均衡性比包含零更重要
- "补偿 vs. 重建"二分法为不同比特的 QAT 行为提供了直觉性解释
- 2-bit 作为 4-bit 的潜在替代方案具有实际部署价值——INT2 硬件支持是未来社区需要努力的方向

## 局限与展望

- 实验仅覆盖 MobileLLM 和 LLaMA-3 系列（最大 8B），更大模型（70B+）未验证
- 2-bit 内核实现仅在 CPU 上，GPU 缺乏原生 INT2 支持
- 仅量化权重，未涉及激活值量化
- 训练成本仍较高（12 万步 × 16 GPU），对资源有限的研究者不友好
- 缺少与混合精度量化方法的比较

## 相关工作与启发

- 与 1-bit Era (Ma et al. 2024) 的对比最有说服力：统一框架下 ParetoQ 用更少 token 和更简单的优化即大幅超越
- 与 Dettmers & Zettlemoyer (2023) 的 4-bit 结论相比：ParetoQ 证明了更好的量化函数设计能将 Pareto 前沿移至更低比特
- 启发：量化领域的"no free lunch"——每个比特位宽需要定制的量化函数，统一框架的价值在于提供公平基准

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一框架和 SEQ 量化器是实质贡献，但更偏工程系统性研究
- 实验充分度: ⭐⭐⭐⭐⭐ 8 个模型 × 5 个比特位宽，覆盖 PTQ/QAT/VQ 基线，极其全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，但五维搜索空间的展开略显冗长
- 价值: ⭐⭐⭐⭐⭐ 为低比特量化领域提供了权威基准，2-bit 潜力的发现有重大实际意义

<!-- RELATED:START -->

## 相关论文

- [LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [Learning Grouped Lattice Vector Quantizers for Low-Bit LLM Compression](learning_grouped_lattice_vector_quantizers_for_low-bit_llm_compression.md)
- [PTQ1.61: Push the Real Limit of Extremely Low-Bit Post-Training Quantization Methods for Large Language Models](../../ACL2025/model_compression/ptq161_low_bit_quantization.md)
- [Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment](q-palette_fractional-bit_quantizers_toward_optimal_bit_allocation_for_efficient_.md)
- [Spectra 1.1: Scaling Laws and Efficient Inference for Ternary Language Models](../../ACL2025/model_compression/scaling_laws_and_efficient_inference_for_ternary_language_models.md)

<!-- RELATED:END -->
