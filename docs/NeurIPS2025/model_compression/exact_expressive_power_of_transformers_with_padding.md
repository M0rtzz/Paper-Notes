---
title: >-
  [论文解读] Exact Expressive Power of Transformers with Padding
description: >-
  [NeurIPS 2025][模型压缩][Transformer] 本文精确刻画了带 padding 的 Transformer 的表达能力：固定深度 + 多项式 padding 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^0$，进一步结合 $O(\log^d n)$ looping 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^d$，polylog looping 收敛到 $\mathsf{NC}$，为 padding/looping 作为可并行推理时计算提供了完整理论基础。
tags:
  - NeurIPS 2025
  - 模型压缩
  - Transformer
  - padding
  - looping
  - 电路复杂度
  - TC0
---

# Exact Expressive Power of Transformers with Padding

**会议**: NeurIPS 2025  
**arXiv**: [2505.18948](https://arxiv.org/abs/2505.18948)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: Transformer表达能力, padding, looping, 电路复杂度, TC0

## 一句话总结
本文精确刻画了带 padding 的 Transformer 的表达能力：固定深度 + 多项式 padding 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^0$，进一步结合 $O(\log^d n)$ looping 恰好等于 $\mathsf{FO}$-uniform $\mathsf{TC}^d$，polylog looping 收敛到 $\mathsf{NC}$，为 padding/looping 作为可并行推理时计算提供了完整理论基础。

## 研究背景与动机

**领域现状**：Transformer 的计算能力有内在限制——固定深度 Transformer 被限制在 $\mathsf{TC}^0$ 内，即高度可并行化的问题类。Chain of Thought (CoT) 可以扩展到 $\mathsf{TC}^0$ 之外，但代价是顺序解码，牺牲并行性。

**现有痛点**：CoT 虽然表达能力强但推理慢。是否存在不增加参数、保持并行性、同时扩展 Transformer 表达能力的推理时计算方法？

**核心矛盾**：此前已知 padded Transformer 上界为 $\mathsf{TC}^0$，但是否能达到整个 $\mathsf{TC}^0$（即下界是否匹配）一直是开放问题。

**本文目标**：(a) 精确刻画 padded Transformer 的表达能力；(b) 理解 padding + looping 组合如何系统地扩展 Transformer 能力。

**切入角度**：不走标准的 $\mathsf{FO}[\mathsf{M}, \mathsf{bit}]$ 路线（`bit` 谓词难以在 Transformer 中模拟），而是使用 $\mathsf{FO}+\mathsf{M}^2$（含成对多数量词的一阶逻辑），后者等价于 $\mathsf{TC}^0$ 但不需要 `bit`。

**核心 idea**：$n^k$ 个 padding token 为 Transformer 提供了足够"存储空间"来枚举 $k$ 个变量的所有赋值组合，从而求解任意 $\mathsf{FO}+\mathsf{M}^2$ 公式。

## 方法详解

### 整体框架
本文是纯理论工作，核心结果是两个精确刻画定理：
- Theorem 1: $\mathsf{AHAT}^0_* = \mathsf{FO}\text{-uniform } \mathsf{TC}^0$（固定深度 + 多项式 padding）
- Theorem 2: $\mathsf{AHAT}^d_* = \mathsf{FO}\text{-uniform } \mathsf{TC}^d$（$O(\log^d n)$ looping + 多项式 padding）

工作模型为 Averaging-Hard-Attention Transformer (AHAT)：hard attention（只关注最大注意力分数），masked pre-norm，causal masking。

### 关键设计

1. **Padding → 变量枚举存储空间** (Lemma 2):

    - 功能：证明含 $k$ 个不同变量、嵌套深度 $\ell$ 的 $\mathsf{FO}+\mathsf{M}^2$ 公式可在 $\mathsf{uAHAT}^0_k$ 中计算
    - 核心思路：用 $n^k$ 个 padding token 枚举 $k$ 个变量的所有 $n^k$ 种赋值。每个 token $v$ 存储赋值 $v$ 下各子公式的真值。通过归纳法，每层计算一级公式：
        - 常量/变量：通过 layer-norm hash $\phi(z) = \langle z,1,-z,-1\rangle / \sqrt{2z^2+2}$ 检索位置
        - 量词 $\exists i. P$：从赋值 $v$ attention 到所有 $v'$（除变量 $i$ 外其余一致），取均值得 $c/n$，与 $1/(2n)$ 比较
        - 成对多数量词 $\mathsf{M}^2(i,j).P$：类似地，attention 到除 $i,j$ 外一致的 $v'$，得到 $c/n^2$，与 $1/2$ 比较
    - 设计动机：$\mathsf{FO}+\mathsf{M}^2$ 等价于 $\mathsf{FO}\text{-uniform } \mathsf{TC}^0$，通过证 padding 可模拟这个逻辑的所有算子，得到下界

2. **Reduction 框架引入 Transformer 分析** (Lemma 3):

    - 功能：将经典复杂度理论中的规约（reduction）和完全问题（complete problem）工具引入 Transformer 分析
    - 核心思路：若类 $\mathsf{C}$ 有一个在 $\mathsf{R}$-规约下的完全问题 $L$，且 padded Transformer 可识别 $L$ 并计算所有 $\mathsf{R}$-规约，则 $\mathsf{C} \subseteq \mathsf{AHAT}^d_*$
    - 设计动机：建立了一种模块化证明策略——只需证 Transformer 能解一个完全问题和实现规约，就可得到整个类的可识别性

3. **Looping 扩展深度** (Lemma 5):

    - 功能：证明 $O(\log^d n)$ looping + 多项式 padding 可识别整个 $\mathsf{FO}\text{-uniform } \mathsf{TC}^d$
    - 核心思路：利用图连通性问题是 $\mathsf{NL}$-complete under $\mathsf{FO}$ reductions，而 log-looped padded Transformer 可解图连通性（已知结果）。由此得到 padded Transformer 可实现 $\mathsf{L}$ 规约。再利用 wide-$\mathsf{TC}^d$ 电路求值问题在 $\mathsf{L}$ 规约下的完全性，链式推导
    - 设计动机：确立 padding 控制宽度、looping 控制深度的直觉

4. **Uniformity 坍缩定理** (Theorem 3):

    - 功能：证明对 $d \geq 1$，$\mathsf{FO}\text{-uniform } \mathsf{TC}^d = \mathsf{L}\text{-uniform } \mathsf{TC}^d$
    - 核心思路：$\mathsf{L}$ 本身在 $\mathsf{FO}\text{-uniform } \mathsf{AC}^d$ 中，因此构建电路的 $\mathsf{L}$-computation 可被 $\mathsf{FO}\text{-uniform}$ 电路模拟后与求值电路复合
    - 设计动机：作为副产品的电路复杂度新结果，加强了 Theorem 2

### 损失函数 / 训练策略
无（纯理论工作，无训练）。

## 实验关键数据

### 主实验
本文无实验，核心结果是数学定理：

| 定理 | 模型 | 等价复杂度类 |
|------|------|-------------|
| Thm 1 | 固定深度 + poly padding | $\mathsf{FO}$-uniform $\mathsf{TC}^0$ |
| Thm 2 | $O(\log^d n)$ loop + poly padding | $\mathsf{FO}$-uniform $\mathsf{TC}^d$ |
| Cor 2.1 | polylog loop + poly padding | $\mathsf{FO}$-uniform $\mathsf{NC}$ |
| Thm 3 | 电路复杂度 ($d \geq 1$) | $\mathsf{FO}$-uniform $\mathsf{TC}^d = \mathsf{L}$-uniform $\mathsf{TC}^d$ |

### 关键比较

| 推理时计算方式 | 表达能力 | 并行性 | 参数增加 |
|---------------|---------|--------|---------|
| 无 | $\mathsf{TC}^0$ | 完全并行 | 无 |
| Poly padding | $\mathsf{TC}^0$（充满整个类） | 完全并行 | 无 |
| Padding + log loop | $\mathsf{TC}^1 \supseteq \mathsf{NL}$ | 高度并行 | 无 |
| Padding + polylog loop | $\mathsf{NC}$ | 高度并行 | 无 |
| Chain of Thought | $\supseteq \mathsf{P}$ | 顺序 | 无 |

### 关键发现
- Padding 本身不扩展 Transformer 的复杂度类上界（仍是 $\mathsf{TC}^0$），但使其充满整个 $\mathsf{TC}^0$——之前只知是其子集
- 对数精度和多项式精度在 padded/looped 设置下等价
- Causal masking 和 unmasked Transformer 在 padding 下等价（Proposition 1）
- Padding + polylog looping 达到 $\mathsf{NC}$ 是保持并行性的理论上限（除非 $\mathsf{NC} = \mathsf{P}$）

## 亮点与洞察
- **首次精确刻画 padded Transformer 的表达能力**，回答了开放问题。关键技巧是绕过难以模拟的 `bit` 谓词，转而使用等价的 $\mathsf{FO}+\mathsf{M}^2$ 逻辑
- **将 reduction/completeness 工具引入 Transformer 理论**非常优雅：一个完全问题 + 规约能力 = 整个复杂度类。这套方法论可被继续用于分析新的 Transformer 变体
- **Padding 作为"计算宽度"、looping 作为"计算深度"**的对偶视角为设计推理时计算策略提供了清晰框架
- **Uniformity 坍缩定理**本身是一个有独立价值的电路复杂度新结果

## 局限与展望
- AHAT 模型（hard attention + masked pre-norm）是对标准 soft-attention Transformer 的理想化，实际模型的表达能力可能不同
- 理论指出 padding 有效，但实践中如何让模型"学会利用 padding"仍是开放问题（现有实证结果 mixed）
- Looping 的实际可训练性存疑——合适的参数化和训练策略尚未解决
- 未讨论 padding 带来的上下文长度增加导致的内存开销
- 固定 $k$ 时的 $\mathsf{AHAT}^d_k$（padding 量受限）的精细刻画仍开放

## 相关工作与启发
- **vs Merrill & Sabharwal (2023)**: 之前证明了 $\mathsf{AHAT} \subseteq \mathsf{TC}^0$ 的上界，本文补全了下界
- **vs Pfau (2024)**: 提出了 padded Transformer vs $\mathsf{TC}^0$ 的开放问题，本文给出完整回答
- **vs CoT 理论**: CoT 可达 $\mathsf{P}$ 以上但牺牲并行性；padding+looping 达到 $\mathsf{NC}$ 保持并行
- **vs Merrill (2025)**: looped Transformer 解图连通性的结果被本文作为关键引理使用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 回答了重要开放问题，引入 reduction/completeness 工具到 Transformer 理论是方法论创新
- 实验充分度: ⭐⭐⭐ 纯理论无实验，但定理体系完整自洽
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链清晰，定理陈述精炼，证明思路阐述到位
- 价值: ⭐⭐⭐⭐ 对理解 Transformer 计算能力和设计推理时计算策略有深远意义

<!-- RELATED:START -->

## 相关论文

- [Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers](deterministic_continuous_replacement_fast_and_stable_module_replacement_in_pretr.md)
- [ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](../../ICLR2026/model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [Stronger Normalization-Free Transformers](../../CVPR2026/model_compression/stronger_normalization-free_transformers.md)
- [TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](../../ICLR2026/model_compression/turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](../../ACL2025/model_compression/fedex_lora_federated_exact_aggregation.md)

<!-- RELATED:END -->
