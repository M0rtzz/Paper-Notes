---
title: >-
  [论文解读] The Counting Power of Transformers
description: >-
  [ICLR 2026][Transformer] 证明 Transformer 不仅能捕获（半）线性计数性质，还能表达所有**半代数计数性质**（即多元多项式不等式的布尔组合），从而推广了先前关于 Transformer 计数能力的所有结果，并由此推导出新的不可判定性结论。
tags:
  - "ICLR 2026"
  - "Transformer"
  - "计数性质"
  - "半代数性质"
  - "不可判定性"
  - "形式语言"
---

# The Counting Power of Transformers

**会议**: ICLR 2026  
**arXiv**: [2505.11199](https://arxiv.org/abs/2505.11199)  
**代码**: 无  
**领域**: Transformer 理论 / 形式语言  
**关键词**: Transformer 表达力, 计数性质, 半代数性质, 不可判定性, 形式语言

## 一句话总结

证明 Transformer 不仅能捕获（半）线性计数性质，还能表达所有**半代数计数性质**（即多元多项式不等式的布尔组合），从而推广了先前关于 Transformer 计数能力的所有结果，并由此推导出新的不可判定性结论。

## 研究背景与动机

- **计数性质在 Transformer 研究中的核心地位**：判断某些 token 出现次数是否多于其他 token（如 MAJORITY: $|w|_a > |w|_b$）是研究 Transformer 表达力的标准测试
- **先前结果仅覆盖线性性质**：已有工作（C-RASP、逻辑语言等）仅能处理线性表达式如 $|w|_a + |w|_b > 2 \cdot |w|_c$
- **实际需要非线性计数**：在信息检索中，共现特征如 $\#(\text{nvidia}) \cdot \#(\text{intel}) \cdot \#(\text{deal})$ 需要多项式表达
- **核心研究问题**：Transformer 能表达哪些计数性质？能否超越线性？

## 方法详解

### 计数性质的形式化

对于字母表 $\Sigma = \{a_1, \ldots, a_m\}$，Parikh 映射 $\Psi(w) = (|w|_{a_1}, \ldots, |w|_{a_m}) \in \mathbb{N}^m$ 记录每个字母的出现次数。

- **半线性计数性质**：由线性不等式的布尔组合定义，如 $|w|_a > |w|_b$
- **半代数计数性质**：由任意多元多项式不等式的布尔组合定义，如 $7|w|_a \cdot |w|_b \cdot |w|_c + 2|w|_d - 8|w|_e > 10$

### 核心定理 1：Transformer 可捕获所有半代数计数性质

**定理 1.1**：（Softmax）Transformer 可以表达所有半代数计数性质——即任意多元多项式（任意次数）不等式的布尔组合。

关键构造思路：
1. NoPE（无位置编码）的 uniform attention 层可以计算输入序列中字母计数的平均值
2. 通过多层组合和前馈网络的非线性，可以从平均值恢复计数比例
3. 利用前馈网络近似多项式函数
4. 无需位置编码或掩码

### 核心定理 2：自然子类的完整刻画

**定理 1.2**：NoPE-AHAT（无位置编码的平均硬注意力 Transformer）和 NoPE-AHAT[U]（仅 uniform 层变体）精确刻画半代数计数性质。

$$\text{NoPE-AHAT} = \text{NoPE-AHAT[U]} = \text{半代数计数性质}$$

这是惊人的，因为 AHAT 是否被 SMAT 捕获仍是开放问题。

### 核心定理 3：通用性（Turing 完备性推论）

**定理 1.3**：每个递归可枚举计数性质都是某个 NoPE-AHAT[U]（因此也是 SMAT）所识别语言的投影。仅需**两个注意力层**。

通过 Matiyasevich 对 Hilbert 第十问题的解（每个 r.e. 集都是整数多项式零点集的投影）结合定理 1.1。

### 核心定理 4：不可判定性

**定理 1.4**：给定一个仅有两个注意力层的 NoPE-AHAT[U] 或 SMAT，判定其语言是否为空是**不可判定的**。

这比先前结果惊人得多——先前需要复杂的位置编码和架构才能获得不可判定性。

### 与 C-RASP 的比较

证明 C-RASP 只能捕获**线性**计数性质。实验表明非线性性质如 $L_k: |w|_a^k \geq |w|_b$ 对 $k \geq 2$ 也能被训练，因此 C-RASP 仅是"高效可学习性质"的不完全刻画。

## 实验验证

### 非线性计数性质的可训练性

| 语言 $L_k$ | 定义 | 准确率 | 长度泛化 |
|-----------|------|--------|----------|
| $L_1$ (线性) | $\|w\|_a \geq \|w\|_b$ | 100% | ✓ |
| $L_2$ (二次) | $\|w\|_a^2 \geq \|w\|_b$ | ~100% | ✓ |
| $L_3$ (三次) | $\|w\|_a^3 \geq \|w\|_b$ | ~100% | ✓ |

### 与 PARITY 的对比

| 性质 | 可训练性 | 长度泛化 |
|------|----------|----------|
| MAJORITY | ✓ 高效 | ✓ 可泛化 |
| 半代数（$L_k$, $k \geq 2$） | ✓ 高效 | ✓ 可泛化 |
| PARITY | ✗ 困难 | ✗ 不泛化 |

- 半代数性质（包括非线性）的可训练性与线性 MAJORITY 类似
- 与 PARITY 形成鲜明对比，进一步支持 PARITY 的困难性不在于非线性

## 亮点与洞察

1. **突破线性限制**：将 Transformer 计数能力从线性扩展到任意多项式次数
2. **NoPE 的惊人表达力**：无位置编码、无掩码就能实现半代数计数
3. **完整刻画**：NoPE-AHAT 精确对应半代数计数性质
4. **深刻的数学联系**：通过 Hilbert 第十问题将代数几何与 Transformer 理论连接
5. **C-RASP 的局限性**：严格证明 C-RASP 仅"看到"线性部分

## 局限性

- 理论构造可能需要大量参数/层数，与实际训练场景有差距
- 结果聚焦于计数性质，不涉及序列顺序相关的性质
- 实验规模较小，仅验证了可训练性
- 构造性证明可能产生不自然的网络权重

## 相关工作

- **Transformer 表达力**：Hahn 2020（通信复杂度下界）、Pérez et al. 2021（图灵完备性）
- **C-RASP**：Huang et al. 2025（形式化 RASP-L 猜想）
- **形式语言与 NN**：RASP、limit transformers
- **Hilbert 第十问题**：Matiyasevich 1993

## 评分

- **创新性**: ⭐⭐⭐⭐⭐ — 开创性地将 Transformer 计数能力推广到半代数
- **技术深度**: ⭐⭐⭐⭐⭐ — 理论证明深厚，连接代数几何与计算理论
- **实验充分性**: ⭐⭐⭐ — 实验主要用于验证可训练性，规模有限
- **实用价值**: ⭐⭐⭐ — 主要为理论贡献，但对理解 Transformer 能力边界有深远意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] The Limitations and Power of NP-Oracle-Based Functional Synthesis Techniques](../../AAAI2026/others/the_limitations_and_power_of_np-oracle-based_functional_synthesis_techniques.md)
- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](../../AAAI2026/others/model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](../../ECCV2024/others/attnzero_efficient_attention_discovery_for_vision_transformers.md)
- [\[ACL 2025\] Implicit Reasoning in Transformers is Reasoning through Shortcuts](../../ACL2025/others/implicit_reasoning_in_transformers_is_reasoning_through_shortcuts.md)

</div>

<!-- RELATED:END -->
