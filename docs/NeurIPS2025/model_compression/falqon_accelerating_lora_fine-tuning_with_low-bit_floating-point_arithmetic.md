---
title: >-
  [论文解读] FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic
description: >-
  [NeurIPS 2025][模型压缩][LoRA加速] FALQON 通过将 LoRA 适配器直接融合 (meld) 到 FP8 量化的骨干权重中，消除了单独 LoRA 路径引入的小矩阵量化开销，结合高效梯度计算和行级代理更新机制，实现了相比现有量化 LoRA 方法约 3 倍的训练加速。 领域现状：FP8 低精度格式在现代…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "LoRA加速"
  - "FP8量化"
  - "低精度训练"
  - "大模型微调"
  - "量化开销"
---

# FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic

**会议**: NeurIPS 2025  
**arXiv**: [2510.24061](https://arxiv.org/abs/2510.24061)  
**代码**: [https://github.com/iamkanghyunchoi/falqon](https://github.com/iamkanghyunchoi/falqon)  
**领域**: 模型压缩  
**关键词**: LoRA加速, FP8量化, 低精度训练, 大模型微调, 量化开销

## 一句话总结
FALQON 通过将 LoRA 适配器直接融合 (meld) 到 FP8 量化的骨干权重中，消除了单独 LoRA 路径引入的小矩阵量化开销，结合高效梯度计算和行级代理更新机制，实现了相比现有量化 LoRA 方法约 3 倍的训练加速。

## 研究背景与动机

**领域现状**：FP8 低精度格式在现代 GPU（NVIDIA Hopper/Blackwell）上有原生硬件支持，理论上可使矩阵乘法吞吐量翻倍。LoRA 是 LLM 微调的主流 PEFT 方法，通过低秩分解减少可训练参数。

**现有痛点**：FP8 量化对大矩阵有效，但 LoRA 引入了小维度矩阵（rank 通常为 16-128）。对这些小矩阵做 FP8 量化时，量化开销（求 max、缩放）远大于计算收益，导致 FP8 LoRA 反而比 FP16 LoRA 更慢。实验显示 FP8 LoRA 吞吐量仅为 FP16 的约一半。

**核心矛盾**：独立的 LoRA 前向/反向路径需要对 $\tilde{A}$, $\tilde{B}$, $O_A$ 三个小张量单独量化，每次迭代多出 3 次量化操作。当矩阵维度 < 4K 时，$O(n^2)$ 的量化开销压倒了 $O(n^3)$ 的矩阵乘计算增益。

**本文目标**：如何真正利用 FP8 硬件加速 LoRA 微调，而非只做 weight-only 量化节省内存？

**切入角度**：既然独立 LoRA 路径引入开销，那就消除这条路径——直接将 LoRA 融入量化后的骨干权重中。量化误差本身就可以被解释为一个隐式的低秩适配器。

**核心 idea**：将 LoRA 融入 FP8 骨干权重消除小矩阵量化开销，通过拼接实现单次前向计算，再用行级代理更新机制维护融合后的权重。

## 方法详解

### 整体框架
FALQON 消除独立 LoRA 路径：初始化时用量化误差的 SVD 作为隐式 LoRA 初始化，将 $A$ 拼接进骨干实现单次前向计算，仅更新 $B$ 矩阵的梯度，通过 proxy buffer 将大幅更新选择性写回骨干。

### 关键设计

1. **Melded LoRA：将 LoRA 融入量化骨干** (Section 5.1):

    - 功能：利用 FP8 量化误差作为隐式 LoRA 初始化
    - 核心思路：$DQ_{fp8}(\tilde{W}) = W - \Delta_Q W \approx W + \hat{B}\hat{A}$，其中 $\hat{A}, \hat{B}$ 来自 $-\Delta_Q W$ 的 rank-$r$ SVD。这样量化后的骨干 $\tilde{W}$ 已隐式包含 LoRA，一次 $DQ_{fp8}(\tilde{W})x$ 同时产生骨干输出和 LoRA 适配输出
    - 设计动机：消除独立 LoRA 路径的所有小矩阵量化操作。与 IR-QLoRA 等同样利用量化误差的方法不同，FALQON 真正移除了计算开销而非仅作参数初始化

2. **高效梯度计算** (Section 5.2):

    - 功能：重新推导 melded LoRA 的梯度，进一步减少量化操作
    - 核心思路：将 $\partial\mathcal{L}/\partial B = (\partial\mathcal{L}/\partial O) \cdot x^\top A^\top$ 重写为 $(\partial\mathcal{L}/\partial O) \cdot (Ax)^\top$。由于 $Ax$ 在前向时已经通过拼接 $\tilde{W}' = [\tilde{W}; \tilde{A}]$ 计算过（$O_{merged} = \tilde{W}' \tilde{x} / (s_W s_x) = [O; O_{\hat{A}}]$），无需额外计算
    - 设计动机：只计算 $B$ 的梯度（不更新 $A$），前向时 $A$ 已嵌入骨干且为 frozen。这个"只训练 $B$"的策略借鉴了 LoRA-FA 的思路

3. **行级代理更新机制** (Section 5.3):

    - 功能：将 $B$ 的梯度更新高效传播到融合后的骨干权重
    - 核心思路：维护代理矩阵 $\Delta B$uffer 累积 $B$ 的更新。由于骨干 $\tilde{W}$ 是低精度，微小更新可能无效（量化后不变），因此选择性地只将 top-$k$ 最大行更新写回：$\mathbf{k} = \text{topk}(\sum_j |\Delta B_{i,j}|; k)$，$\tilde{W}[\mathbf{k}] = \tilde{W}[\mathbf{k}] + \Delta\text{Buffer}[\mathbf{k}] A$
    - 设计动机：$k \ll m$（输出通道数），避免无效的全量更新，同时保持 LoRA 的内存优势（只需存 $\Delta B$uffer 而非全精度 $W$）

### 损失函数 / 训练策略
- 使用 Paged AdamW，batch size 16，学习率 $2 \times 10^{-5}$，1875 训练步
- $A$ 用 FP8 量化时共用骨干的缩放因子 $s_W$，避免额外量化操作
- 单卡 RTX 4090 24GB 完成所有实验

## 实验关键数据

### 主实验
LLaMA-7B + Alpaca 数据集，MMLU 评估：

| 方法 | Time/Step (s) | 加速比 | MMLU Avg |
|------|--------------|--------|----------|
| QLoRA | 5.45 | 1.0× | 0.3272 |
| QA-LoRA | 9.44 | - | 0.3548 |
| IR-QLoRA | 8.27 | - | 0.3388 |
| FALQON | **1.80** | **3.02×** | 0.3491 |

LLaMA-13B + Alpaca 数据集：

| 方法 | Time/Step (s) | 加速比 | MMLU Avg |
|------|--------------|--------|----------|
| QLoRA | 9.37 | 1.0× | 0.4443 |
| QA-LoRA | 18.02 | - | 0.4729 |
| IR-QLoRA | 14.46 | - | 0.4349 |
| FALQON | **3.26** | **2.87×** | 0.4644 |

### 消融实验
FP8 LoRA 开销分析（LLaMA-7B, rank=64）：

| 组件 | FP16 耗时 | FP8 耗时 | 说明 |
|------|----------|---------|------|
| 计算部分 | 高 | 低（加速） | FP8 矩阵乘更快 |
| 量化开销 | 0 | 极高（~4×） | 量化 A/B/OA 拖累 |
| 总计 | 更快 | 更慢 | FP8 直接用于 LoRA 反而变慢 |

### 关键发现
- FP8 直接应用于 LoRA 的吞吐量仅为 FP16 的约 50%，即使 rank 增到 512 也无改善
- FALQON 的加速主要来自消除 3 次小矩阵量化操作（占总时间的大部分）
- Melded LoRA 拼接只增加 $r$ 行，对前向计算的额外开销可忽略
- 行级代理更新的 top-$k$ 策略有效避免了低精度下的"无效更新"
- FALQON 训练结束时权重已是 FP8 格式，无需额外的后训练量化

## 亮点与洞察
- **对 FP8 + LoRA 的深入分析**非常有价值：清楚指出问题不在精度而在量化开销，这个洞见对整个低精度训练社区有参考意义
- **Melded LoRA 初始化**巧妙利用量化误差作为隐式适配器，一举两得——补偿量化损失且无额外参数
- **拼接 $A$ 的梯度复用**是一个非常工程友好的 trick：前向一次计算同时得到输出和梯度所需的中间结果
- **端到端 FP8 工作流**消除了推理部署时的额外量化步骤，对实际部署有直接价值

## 局限与展望
- 仅冻结 $A$ 更新 $B$，可能在某些任务上不如同时更新 $A, B$
- Top-$k$ 行选择策略较简单，可能错过某些重要但分散的更新
- 目前仅在 LLaMA-7B/13B 上验证，更大模型（70B+）效果有待确认
- 依赖 FP8 硬件支持（Hopper+），老 GPU 不适用
- 理论上 melded LoRA 的初始化可能不如 Kaiming 初始化稳定

## 相关工作与启发
- **vs QLoRA**: QLoRA 用 NF4 weight-only 量化节省内存但不加速；FALQON 用 FP8 weight+activation 量化同时节省内存和加速
- **vs IR-QLoRA**: IR-QLoRA 同样利用量化误差做 SVD 初始化 LoRA，但保留独立路径不解决速度问题
- **vs FP8-LM/TorchAO**: 面向预训练大矩阵优化（验证 1.38× 加速），未解决 LoRA 小矩阵的开销问题
- **vs LoRA-FA**: FALQON 的仅训练 $B$ 借鉴了 LoRA-FA 的思路，但重新包装到 FP8 融合框架中

## 评分
- 新颖性: ⭐⭐⭐⭐ Melded LoRA + 拼接梯度复用的组合设计新颖，对 FP8 开销的分析有独到见解
- 实验充分度: ⭐⭐⭐⭐ 速度分析详尽，多数据集多模型覆盖，有详细的 breakdown 分析
- 写作质量: ⭐⭐⭐⭐ 问题分析部分非常清晰，图表有效传达核心信息
- 价值: ⭐⭐⭐⭐⭐ 直接解决了 FP8 硬件上 LoRA 微调减速的实际痛点，3× 加速对实际部署意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] EMLoC: Emulator-based Memory-efficient Fine-tuning with LoRA Correction](emloc_emulator-based_memory-efficient_fine-tuning_with_lora_correction.md)
- [\[NeurIPS 2025\] Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving](loquetier_a_virtualized_multi-lora_framework_for_unified_llm_fine-tuning_and_ser.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)

</div>

<!-- RELATED:END -->
