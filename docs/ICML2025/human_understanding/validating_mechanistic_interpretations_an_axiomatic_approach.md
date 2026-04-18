---
title: >-
  [论文解读] Validating Mechanistic Interpretations: An Axiomatic Approach
description: >-
  [ICML2025][人体理解][mechanistic interpretability] 借鉴程序分析中抽象解释的思想，提出一组公理化框架来形式化定义和验证神经网络的机制解释（mechanistic interpretation），并在 2-SAT 求解器和模加法两个案例中验证了该框架的有效性。
tags:
  - ICML2025
  - 人体理解
  - mechanistic interpretability
  - axiomatic validation
  - abstract interpretation
  - 2-SAT
  - compositional analysis
---

# Validating Mechanistic Interpretations: An Axiomatic Approach

**会议**: ICML2025  
**arXiv**: [2407.13594](https://arxiv.org/abs/2407.13594)  
**代码**: [GitHub](https://github.com/nilspalumbo/axiomatic-validation)  
**领域**: 可解释性 / 机制可解释性  
**关键词**: mechanistic interpretability, axiomatic validation, abstract interpretation, 2-SAT, compositional analysis

## 一句话总结

借鉴程序分析中抽象解释的思想，提出一组公理化框架来形式化定义和验证神经网络的机制解释（mechanistic interpretation），并在 2-SAT 求解器和模加法两个案例中验证了该框架的有效性。

## 研究背景与动机

机制可解释性（Mechanistic Interpretability）旨在逆向工程神经网络的内部计算过程，将其表示为人类可理解的电路（circuit）和特征（feature）。然而当前存在两个核心问题：

**定义模糊**：什么是"有效的"机制解释缺乏统一标准，现有研究往往采用 ad-hoc 的评估方式

**缺乏组合性验证**：现有工作（如 Nanda et al. 2023 的模加法分析）仅验证单个组件的等价性，忽略了误差在多组件间的级联累积效应

作者观察到，机制可解释性与程序分析中的**抽象解释**（Abstract Interpretation, Cousot & Cousot 1977）有深层类比——两者都试图用近似的语义来描述一个复杂系统的行为。基于此，本文提出了一套公理体系，形式化地刻画何为"有效的机制解释"。

## 方法详解

### 核心形式化框架

将神经网络视为纯函数语言 $\lambda_T$ 中的程序，其基本操作包括 Embed、Unembed、Lin、ReLU、Self-Attention 等。机制解释则是另一种人类可读语言 $\lambda_H$ 中的程序。

**关键概念**：
- **分解（Decomposition）**：将模型 $t$ 分解为组件列表 $d_t = [d_t[1], d_t[2], \ldots]$，使得组件的复合等于原模型
- **抽象函数 $\alpha_i$**：将模型的实值激活映射到人类可解释的离散符号（类似 probe）
- **具体化函数 $\gamma_i$**：将抽象符号映射回模型的表示空间

### 四条核心公理

给定模型 $t$、分解 $d_t$、解释 $h$ 及其分解 $d_h$、输入分布 $\mathcal{D}$，$\epsilon$-准确的机制解释需满足：

**公理 1（$\epsilon$-前缀等价）**：每个前缀的抽象输出与解释前缀的输出一致

$$\forall i: \Pr_{x \sim \mathcal{D}}[\alpha_i \circ d_t[:i{+}1](x) = d_h[:i{+}1] \circ \alpha_0(x)] \geq 1 - \epsilon$$

**公理 2（$\epsilon$-组件等价）**：每个单独组件不引入过多误差

$$\forall i: \Pr_{x \sim \mathcal{D}}[\alpha_i \circ d_t[:i{+}1](x) = d_h[i] \circ \alpha_{i-1} \circ d_t[:i](x)] \geq 1 - \epsilon$$

**公理 3（$\epsilon$-前缀可替换性）**：用解释前缀替换模型前缀后，最终输出基本不变

$$\forall i: \Pr_{x \sim \mathcal{D}}[t(x) = d_t[i{+}1:] \circ \gamma_i \circ d_h[:i{+}1] \circ \alpha_0(x)] \geq 1 - \epsilon$$

**公理 4（$\epsilon$-组件可替换性）**：用解释的单个组件替换模型对应组件后，输出基本不变

$$\forall i: \Pr_{x \sim \mathcal{D}}[t(x) = d_t[i{+}1:] \circ \gamma_i \circ d_h[i] \circ \alpha_{i-1} \circ d_t[:i](x)] \geq 1 - \epsilon$$

> **关键区别**：公理 1/3 检验组合性（误差级联），公理 2/4 检验单组件。公理 2 不蕴含公理 1，因为误差可能在组件间累积。

### 验证方法

公理可通过统计检验高效验证：在测试集上计算违反率，使用 Clopper-Pearson 方法构建置信区间。

## 实验关键数据

### 案例一：2-SAT 求解器

训练一个 2 层 ReLU decoder-only Transformer（$d=128$, 第一层 1 头, 第二层 4 头, MLP 隐藏层 512 维）在 10 子句 5 变量的 2-SAT 问题上，测试准确率 **99.76%**。

**逆向工程发现的算法**：
- **第一层 = 解析器**：通过注意力模式将 token 序列解析为子句列表
- **第二层 = 求值器**：MLP 隐藏层中仅 34 个神经元有效，通过穷举赋值来判断可满足性

| 公理 | 组件 | 决策树解释 $\epsilon$ | 析取解释 $\epsilon$ |
|------|------|----------------------|---------------------|
| 公理 1 (前缀等价) | 第一层 | ≈0.0000374 | ≈0.0000374 |
| 公理 1 (前缀等价) | 隐藏层 | ≈0.182 | ≈0.309 |
| 公理 3 (前缀可替换) | 第一层 | ≈0.0418 | ≈0.0418 |
| 公理 3 (前缀可替换) | 隐藏层 | ≈0.0128 | ≈0.00290 |
| 公理 2 (组件等价) | 输出层 | ≈0.00433 | ≈0.00433 |

### 案例二：模加法（Nanda et al. 2023）

验证了 Nanda 等人对模加法模型的机制解释（基于三角恒等式的算法）确实满足所有公理。值得注意的是，Nanda 等人原始论文仅提供了类似公理 2 和 4 的证据，**缺少公理 1 和 3 的组合性验证**。

### 关键发现

- 决策树解释在中间表示等价性上优于析取解释（$\epsilon$: 0.182 vs 0.309），但在可替换性上两者差距很小
- $\gamma$ 函数中的**放大步骤**至关重要：去掉放大后，前缀可替换性 $\epsilon$ 从 0.0128 恶化到 0.249
- 组合性公理（1 和 3）确实不能从组件性公理（2 和 4）推出，实验证实了误差累积问题

## 亮点与洞察

1. **理论贡献清晰**：将机制解释问题从 ad-hoc 评估提升到公理化框架，类比抽象解释的思路非常优雅
2. **组合性公理的必要性**：首次明确指出仅验证单组件等价性不够，误差级联是真实存在的问题
3. **双向验证**：$\alpha$（抽象）和 $\gamma$（具体化）函数对的设计，既检查中间表示的语义一致性，又检查替换后的端到端行为
4. **实用性强**：公理验证仅需前向推理 + 统计检验，计算开销低
5. **2-SAT 案例有趣**：发现模型学会了穷举求值算法，且 MLP 神经元表现出稀疏性（512 中仅 34 个有效）

## 局限与展望

1. **仅验证了小规模模型**：2-SAT（5 变量 10 子句）和模加法（mod 113），能否扩展到实际规模的 LLM 尚不清楚
2. **$\alpha$ 和 $\gamma$ 函数需手工设计**：目前没有自动化方法来选择合适的抽象/具体化函数
3. **连续中间表示的离散化挑战**：模加法案例中，简单离散化（四舍五入到 1 位小数）导致公理 1/2 的 $\epsilon=1$，说明离散化选择对结果影响巨大
4. **公理 5/6 未实现**：论文承认当前 4 条公理可能不够充分，计划在未来工作中补充
5. **仅覆盖顺序组合**：虽然附录讨论了如何处理并行组合，但实际案例仅涉及顺序分解

## 相关工作与启发

- **因果抽象**（Geiger et al. 2021, 2024）：用因果干预验证解释，与本文互补但缺少 $\gamma$ 函数
- **Causal Scrubbing**（Chan et al. 2022）：有类似具体化操作但缺少 $\alpha$ 函数
- **电路分析**（Wang et al. 2023; Conmy et al. 2023）：发现功能子图但缺乏组合性验证标准
- **抽象解释**（Cousot & Cousot 1977）：程序分析领域的经典框架，是本文的核心理论来源

## 评分

- 新颖性: ⭐⭐⭐⭐ — 公理化框架思路新颖，抽象解释的类比有深度
- 实验充分度: ⭐⭐⭐ — 两个案例分析完整，但规模受限
- 写作质量: ⭐⭐⭐⭐ — 形式化严谨清晰，案例解析详尽
- 价值: ⭐⭐⭐⭐ — 为机制可解释性建立了评估基准，对领域有方向性意义
