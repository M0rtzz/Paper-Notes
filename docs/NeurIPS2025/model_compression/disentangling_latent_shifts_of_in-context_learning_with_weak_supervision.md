---
title: >-
  [论文解读] Disentangling Latent Shifts of In-Context Learning with Weak Supervision
description: >-
  [NeurIPS 2025][模型压缩][上下文学习] WILDA 将 ICL 视为弱监督信号，用 teacher-student 框架将示例引发的潜在偏移编码进轻量 LoRA 适配器，实现无需重复 prompting 的高效推理，且 student 通过伪标签修正和覆盖扩展超越 teacher（弱到强泛化）。
tags:
  - NeurIPS 2025
  - 模型压缩
  - 上下文学习
  - 弱监督
  - 适配器算术
  - 潜在偏移解耦
  - LoRA
---

# Disentangling Latent Shifts of In-Context Learning with Weak Supervision

**会议**: NeurIPS 2025  
**arXiv**: [2410.01508](https://arxiv.org/abs/2410.01508)  
**代码**: [github.com/josipjukic/wilda](https://github.com/josipjukic/wilda)  
**领域**: 模型压缩 / LLM 效率  
**关键词**: 上下文学习, 弱监督, 适配器算术, 潜在偏移解耦, LoRA

## 一句话总结
WILDA 将 ICL 视为弱监督信号，用 teacher-student 框架将示例引发的潜在偏移编码进轻量 LoRA 适配器，实现无需重复 prompting 的高效推理，且 student 通过伪标签修正和覆盖扩展超越 teacher（弱到强泛化）。

## 研究背景与动机

**领域现状**：In-Context Learning (ICL) 让 LLM 通过 prompt 中的少量标注示例实现 few-shot 学习，无需参数更新，是低资源场景下的核心适配机制。

**现有痛点**：(a) ICL 对示例的选择和排列高度敏感，导致预测不稳定；(b) 多示例需要长上下文，推理效率低且受上下文窗口限制；(c) 示例数量超过阈值后性能反而下降——ICL 的可扩展性差。

**核心矛盾**：现有解耦方法（ICV、Batch-ICL）直接操纵注意力头或隐藏状态来分离示例的影响，但依赖线性注意力近似，忽略了 FFN 层、激活函数、残差连接等关键架构组件。

**本文目标** 如何在不修改模型内部状态的前提下，将 ICL 示例的效果"参数化"为可复用的紧凑表示？

**切入角度**：从功能性视角看 ICL——关注模型最终输出而非中间状态。ICL 输出本身就是示例效果的完整体现，可作为弱监督信号。

**核心 idea**：用 ICL 预测作为伪标签训练轻量适配器，将示例的潜在偏移编码为可复用参数。

## 方法详解

### 整体框架
WILDA 采用 teacher-student 设置：teacher 是标准 ICL 模型（处理示例+查询），student 共享同一架构但加装 LoRA 适配器（仅处理查询）。student 通过最小化与 teacher 输出的交叉熵损失来学习。

### 关键设计

1. **ICL 作为弱监督信号**:

    - 功能：用 ICL 的全概率分布（而非硬标签）作为 teacher 信号
    - 核心思路：损失函数为 $\sum_{x_q \in \mathcal{D}_{\text{unlab}}} \ell_{\text{CE}}(\mathbf{f}_{\text{teacher}}([\mathbf{X}_d^*; x_q]), \mathbf{f}_{\text{student}}(x_q))$，其中 $\mathcal{D}_{\text{unlab}}$ 是无标签数据集（仅 100 个样本即可）
    - 设计动机：不直接操纵注意力头/隐藏状态，而是从模型输出端捕获示例的完整影响（包含注意力、FFN、残差等所有组件的综合效果）

2. **适配器参数化的潜在偏移**:

    - 功能：将示例引发的效果编码为 LoRA 权重 $\mathbf{W}_{\text{ICL}}$
    - 核心思路：模型参数分解为 $\mathbf{W}_{\text{ZS}} \oplus \mathbf{W}_{\text{ICL}}$，其中 $\mathbf{W}_{\text{ZS}}$ 是零样本基础参数，$\mathbf{W}_{\text{ICL}}$ 是适配器捕获的 ICL 偏移。最终隐藏状态满足 $\mathbf{h}_{\text{LLM}}(x_q | \mathbf{W}_{\text{ZS}} \oplus \mathbf{W}_{\text{ICL}}) = \mathbf{h}_{\text{LLM}}(x_q | \mathbf{W}_{\text{ZS}}) + \Delta \mathbf{h}_{\text{ICL}}$
    - 设计动机：适配器仅占 0.1-0.3% 参数，推理时无需示例即可使用，且可与新示例组合

3. **适配器算术（Adapter Arithmetic）**:

    - 功能：对多个示例子集训练独立适配器，通过参数求和合并
    - 核心思路：将大量示例分割为 2/4/8 个子集各 16 个示例，分别训练适配器后直接相加：$\mathbf{W}_{\text{ICL}}^{\text{merged}} = \sum_k \mathbf{W}_{\text{ICL}}^{(k)}$
    - 设计动机：突破上下文窗口限制，让模型有效利用远超窗口长度的示例集

4. **三种训练变体**:

    - wilda-f（fixed）：示例集固定不变
    - wilda-s（shuffle）：每 epoch 打乱示例顺序 → 缓解顺序敏感性
    - wilda-r（resample）：每 epoch 从更大池中重采样示例

### 损失函数 / 训练策略
交叉熵损失对齐 teacher 全概率分布。训练 10 epochs，使用 LoRA 适配器（仅更新适配器参数）。同一 LLM 实例在训练中交替作为 teacher（关闭适配器）和 student（开启适配器）。

## 实验关键数据

### 主实验（16-shot，100 无标签样本，Llama 3 8B）

| 数据集 | 0-shot | n-shot ICL | Batch-ICL | wilda-s | 提升 (vs ICL) |
|--------|--------|-----------|-----------|---------|--------------|
| RTE | 62.3 | 75.1 | 77.8 | **86.0** | +10.9 |
| SST | 79.1 | 93.5 | 94.1 | **96.1** | +2.6 |
| QNLI | 64.3 | 77.0 | 78.0 | **81.4** | +4.4 |
| MNLI | 59.9 | 68.0 | 70.9 | **73.1** | +5.1 |
| CoLA | 44.6 | 58.5 | 59.8 | **64.3** | +5.8 |
| MRPC | 63.6 | 74.0 | 75.2 | **77.7** | +3.7 |
| QQP | 61.1 | 70.0 | 72.5 | **73.1** | +3.1 |
| Math (MMLU) | 31.5 | 43.5 | 36.2 | **49.5** | +6.0 |
| Misc (MMLU) | 62.5 | 84.0 | 81.0 | **88.0** | +4.0 |

wilda-s 标准差显著低于 ICL（如 RTE: 0.6 vs 6.5），stability 大幅提升。

### 适配器算术消融（Llama 3 8B，知识融合）

| 示例组合 | 方法 | RTE | SST | MMLU-Math | MMLU-Misc |
|----------|------|-----|-----|-----------|-----------|
| 2×16 | Batch-ICL | 80.2 | 95.3 | 43.5 | 83.0 |
| 2×16 | wilda-s | **87.1** | **96.4** | **51.5** | **89.5** |
| 4×16 | Batch-ICL | 84.4 | 96.4 | 45.5 | 84.5 |
| 4×16 | wilda-s | **88.4** | **97.5** | **53.5** | **91.0** |
| 8×16 | wilda-s | **92.8** | — | — | — |

随着子集数量增加，wilda-s 性能持续提升，展现出优秀的可扩展性。

### 关键发现
- **弱到强泛化**：Student 超越 teacher（ICL）是一致现象，通过伪标签修正（修正 teacher 错误）和覆盖扩展（泛化到 teacher 未见样本）两个机制实现
- wilda-s（shuffle）综合表现最佳——打乱顺序有效缓解 ICL 的位置偏差（primacy/recency effects）
- 仅需 100 个无标签样本即可有效训练适配器，数据效率极高
- OOD 泛化强：在跨数据集迁移（如 QNLI→RTE）中也显著优于 ICL

## 亮点与洞察
- **ICL 作为弱监督的视角转换**：不直接操纵注意力机制，而是从输出端捕获 ICL 全部效果。这种"黑盒"视角更完整，也更适用于复杂架构
- **适配器算术实现超长上下文 ICL**：通过拆分-训练-合并，突破上下文窗口限制。8×16=128 个示例的效果远超直接使用 128-shot ICL
- **极度参数高效**：LoRA 仅占 0.1-0.3% 参数，100 个无标签样本，10 个 epoch 训练——成本极低但效果显著

## 局限与展望
- 适配器训练仍需一些计算开销（虽然很轻量），纯推理场景下不如直接 ICL 灵活
- 伪标签质量依赖 teacher ICL 的基础能力——在 ICL 本身很差的任务上可能失效
- 仅在分类任务上验证，生成任务（如摘要、翻译）的效果未知
- 适配器算术的简单参数求和可能不是最优合并策略，可以探索加权合并或学习合并系数
- 未讨论适配器在不同模型间的迁移性

## 相关工作与启发
- **vs ICV (In-Context Vectors)**: ICV 从隐藏状态提取示例表征，依赖线性注意力近似；WILDA 从模型输出端学习，不假设特定注意力机制
- **vs Batch-ICL**: Batch-ICL 聚合多次 one-shot 的 meta-gradients，仍在操纵内部状态；WILDA 用参数化适配器更灵活
- **vs PBFT (Pattern-Based Fine-Tuning)**: PBFT 在有标签数据上微调，WILDA 仅用无标签数据 + ICL 伪标签，更符合 few-shot 场景

## 评分
- 新颖性: ⭐⭐⭐⭐ ICL-as-weak-supervision 的视角新颖，适配器算术实用
- 实验充分度: ⭐⭐⭐⭐⭐ 7 个 GLUE 任务 + 2 个 MMLU + 3 个模型，含 OOD/稳定性/融合多角度评估
- 写作质量: ⭐⭐⭐⭐ 理论动机清晰，实验全面，但部分内容重复
- 价值: ⭐⭐⭐⭐ 提供了高效稳定的 ICL 替代方案，适配器算术思路可广泛应用

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Masking Meets Supervision: A Strong Learning Alliance](../../CVPR2025/model_compression/masking_meets_supervision_a_strong_learning_alliance.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/model_compression/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](../../ICLR2026/model_compression/in-context_learning_for_pure_exploration.md)

<!-- RELATED:END -->
