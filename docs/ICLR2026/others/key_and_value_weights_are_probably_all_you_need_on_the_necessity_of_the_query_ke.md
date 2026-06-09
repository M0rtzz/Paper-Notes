---
title: >-
  [论文解读] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention
description: >-
  [ICLR 2026][自注意力] 理论证明Transformer自注意力中Query/Key/Value权重三元组存在冗余——Query权重可被替换为单位矩阵（减少25%注意力参数），GPT风格模型从头训练验证在适当超参数调整下性能不降，且训练在3倍更低权重衰减下仍然稳定。
tags:
  - "ICLR 2026"
  - "自注意力"
  - "Query权重"
  - "参数冗余"
  - "隐式正则化"
  - "架构简化"
---

# Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention

**会议**: ICLR 2026  
**arXiv**: [2510.23912](https://arxiv.org/abs/2510.23912)  
**代码**: [GitHub](https://github.com/MarkoKarbevski/Wqkv_necessity)  
**领域**: Transformer架构  
**关键词**: 自注意力, Query权重, 参数冗余, 隐式正则化, 架构简化

## 一句话总结
理论证明Transformer自注意力中Query/Key/Value权重三元组存在冗余——Query权重可被替换为单位矩阵（减少25%注意力参数），GPT风格模型从头训练验证在适当超参数调整下性能不降，且训练在3倍更低权重衰减下仍然稳定。

## 研究背景与动机

### 领域现状

**领域现状**：Transformer训练和部署资源密集→各种架构优化（量化/高效注意力/权重共享/归一化简化）。近期研究表明归一化层和注意力参数可以重排/简化。

**现有痛点**：注意力机制是否过参数化？Q/K/V三元组是否全部必要？Graef(2024)证明了无skip无归一化时Q和O都冗余，但未覆盖实际架构。

**核心矛盾**：注意力输出只通过乘积 $XW_Q$, $XW_K$, $XW_V$ 依赖输入→伸缩构造可以将一层的基变换传递到下一层→理论上$W_Q=I_d$应该无损。

**切入角度**：理论先行→渐进式证明（单层/多层/跳跃连接/权重共享）→GPT实验验证。

## 方法详解

### 整体框架

文章的论证是"理论先行、实验兜底"：先从自注意力的代数结构出发，证明 Query 权重 $W_Q$ 在一系列实际架构条件下可以被单位矩阵 $I_d$ 替换而不损失表达力，再用 GPT-2 风格模型从头训练来检验这个结论在真实优化下是否成立。核心观察是注意力对输入只通过乘积 $XW_Q$、$XW_K$、$XW_V$ 三条路径依赖，因此一层里的基变换可以被"挪"到相邻层去吸收，从而把 $W_Q$ 化简掉。

### 关键设计

**1. 单层消除定理：把基变换搬给邻层。** 注意力 logits 由 $XW_QW_K^TX^T$ 给出，$W_Q$ 在这里只起一个对输入做线性基变换的作用。Theorem 4.1 证明，在无归一化的 Transformer 中，任意单层的 $W_Q$ 都能通过对相邻权重做重参数化被消除——本质上是把 $W_Q$ 代表的伸缩/基变换吸收进上一层输出的处理里，注意力分数完全不变。这条结论甚至能用在已经预训练好的模型上：先把归一化层并入线性层（去归一化），再做这一步重参数化即可，无需重训。

**2. 多层与残差的成立条件：何时能全局令 $W_Q=I_d$。** 单层结论要推广到每一层同时成立，需要约束跨层信号如何流动。Theorem 4.2–4.3 给出两类充分条件：要么 skip 连接只环绕注意力子层（而不跨越更大的块），要么相邻层之间做跨层权重共享。这两种情形下基变换可以一层层向前传递而不互相打架，于是 $W_Q=I_d$ 能在所有层统一成立。条件之所以重要，是因为残差连接把不同层的表示叠加在一起，若 skip 的结构不对，前一层吸收的基变换会污染后续层的恒等支路。

**3. ReLU MLP 对残差的吸收边界：Theorem 8.4。** 这条结果回答了一个更基础的问题——残差连接本身能不能被 MLP 吃掉。文章精确刻画了在什么条件下一个 ReLU MLP 可以等价地吸收一条残差连接、什么时候不行，从而界定了 skip 连接为模型贡献的"额外结构表达力"到底有多大。它是一个相对独立的理论副产品，也为前两条定理里对 skip 结构的假设提供了依据。

**4. 直觉：从二次到线性的优化简化。** 去掉 $W_Q$ 不只是省参数，还改变了损失面的形状。带 $W_Q$ 时注意力 logits $XW_QW_K^TX^T$ 关于可学习权重是二次的（$W_Q$ 和 $W_K$ 相乘），令 $W_Q=I_d$ 后变成 $XW_K^TX^T$，对参数退化为线性。这解释了实验里观察到的现象：消除 $W_Q$ 后训练能在约三分之一的权重衰减下保持稳定，相当于这步化简自带了一层隐式正则化。

### 实证验证

理论给出了"可以去掉"的条件，实验则检验"去掉之后真实训练会怎样"。作者在 OpenWebText 上从头训练 GPT-2 风格模型（117M–124M）：令 $W_Q=I_d$ 的 117M 模型在验证 loss 上追平完整的 124M 基线，等于用少 8% 的参数拿到同样效果；若把省下的注意力参数重新分配给 MLP、把规模补回 124M，则反过来超越基线。这说明 $W_Q$ 携带的容量用在 MLP 上比留在注意力里更划算，且这一改动与 GQA/MQA 等键值压缩方案正交、可叠加使用。

## 实验关键数据

### GPT模型验证


### 主实验

| 配置 | 参数量 | 验证Loss | 说明 |
|------|--------|---------|------|
| 基线GPT-124M | 124M | 基线 | 完整注意力 |
| $W_Q=I$(117M) | 117M(-8%) | **匹配基线** | 少7M参数同效果 |
| $W_Q=I$+MLP扩展(124M) | 124M | **超越基线** | 参数重分配更优 |

### 训练稳定性


### 消融实验

| 配置 | 最低稳定权重衰减 |
|------|---------------|
| 标准GPT | $\lambda$ |
| $W_Q=I$ | **$\lambda/3$** |

### 关键发现
- 去掉$W_Q$后性能不降→$W_Q$确实冗余
- 省下的参数给MLP比给注意力更有价值
- 3倍更低权重衰减仍稳定→$W_Q$消除提供了隐式正则化
- 与GQA/MQA正交→可叠加使用

## 亮点与洞察
- **理论驱动的架构简化**：不是"试试去掉看行不行"，而是数学证明为什么可以去掉→提供了信心和适用条件。
- **优化简化**：$W_Q=I$后注意力logits从$XW_QW_K^TX^T$（关于学习权重二次）变为$XW_K^TX^T$（线性）→可能解释了训练稳定性提升。
- **结构表达力边界**：Theorem 8.4精确表征了何时ReLU MLP可以/不可以吸收残差连接——这是一个独立的理论贡献。

## 局限与展望
- 仅在117-124M规模验证→需要在更大模型(7B+)上确认
- 只测了$W_Q$消除，$W_K$或$W_V$消除的实验留作未来工作
- LayerNorm的存在引入额外近似→理论保证从精确变为近似
- 仅测了预训练loss，下游任务性能未评估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 理论证明注意力权重冗余性是基础贡献
- 实验充分度: ⭐⭐⭐ 规模偏小，需要更大模型验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，渐进式结构
- 价值: ⭐⭐⭐⭐ 如果大规模验证成立将对Transformer设计有重大影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](../../ACL2025/others/attention_entropy_parallel_encoding.md)
- [\[ICML 2026\] On the Coordination of Value-Maximizing Bidders](../../ICML2026/others/on_the_coordination_of_value-maximizing_bidders.md)
- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)

</div>

<!-- RELATED:END -->
