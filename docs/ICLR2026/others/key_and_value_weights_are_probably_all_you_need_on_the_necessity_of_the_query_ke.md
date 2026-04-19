---
title: >-
  [论文解读] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention
description: >-
  [ICLR 2026][自注意力] 理论证明Transformer自注意力中Query/Key/Value权重三元组存在冗余——Query权重可被替换为单位矩阵（减少25%注意力参数），GPT风格模型从头训练验证在适当超参数调整下性能不降，且训练在3倍更低权重衰减下仍然稳定。
tags:
  - ICLR 2026
  - 自注意力
  - Query权重
  - 参数冗余
  - 隐式正则化
  - 架构简化
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

**领域现状**：Transformer训练和部署资源密集→各种架构优化（量化/高效注意力/权重共享/归一化简化）。近期研究表明归一化层和注意力参数可以重排/简化。

**现有痛点**：注意力机制是否过参数化？Q/K/V三元组是否全部必要？Graef(2024)证明了无skip无归一化时Q和O都冗余，但未覆盖实际架构。

**核心矛盾**：注意力输出只通过乘积 $XW_Q$, $XW_K$, $XW_V$ 依赖输入→伸缩构造可以将一层的基变换传递到下一层→理论上$W_Q=I_d$应该无损。

**切入角度**：理论先行→渐进式证明（单层/多层/跳跃连接/权重共享）→GPT实验验证。

## 方法详解

### 理论结果

1. **Theorem 4.1 (单层免费午餐)**：任何无归一化Transformer的单层$W_Q$可被消除，仅需权重重参数化。可应用于已预训练模型（先去归一化）。

2. **Theorem 4.2-4.3 (多层消除)**：无归一化时，$W_Q=I_d$在所有层成立的条件：(i) skip连接仅在注意力子层周围，或(ii) 跨层权重共享。

3. **Theorem 8.4 (ReLU MLP与残差)**：精确求解何时残差连接可被ReLU MLP吸收，揭示了skip连接的结构表达力边界。

### 实证验证
- GPT-2风格模型(117M-124M)在OpenWebText上从头训练
- $W_Q=I_d$的117M模型匹配完整124M基线性能
- 省下的参数分配给MLP→显著超越124M基线
- 训练在3倍更低权重衰减下保持稳定→暗示隐式正则化

### 直觉解释
- 注意力依赖 $XW_QW_K^TX^T$→$W_Q$作为基变换→但这个变换可以"传送"给前一层的输出处理
- 去掉$W_Q$后注意力logits从参数的二次函数变为线性→优化更简单

## 实验关键数据

### GPT模型验证
| 配置 | 参数量 | 验证Loss | 说明 |
|------|--------|---------|------|
| 基线GPT-124M | 124M | 基线 | 完整注意力 |
| $W_Q=I$(117M) | 117M(-8%) | **匹配基线** | 少7M参数同效果 |
| $W_Q=I$+MLP扩展(124M) | 124M | **超越基线** | 参数重分配更优 |

### 训练稳定性
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

## 相关论文

- [Attention Entropy is a Key Factor for Parallel Context Encoding](../../ACL2025/others/attention_entropy_parallel_encoding.md)
- [Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)

<!-- RELATED:END -->
