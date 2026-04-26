---
title: >-
  [论文解读] Neutral Residues: Revisiting Adapters for Model Extension
description: >-
  [ICML2025][模型压缩][Adapter] 提出 **Neutral Residues**，通过在 adapter 中引入 ReLU 门控 + $\ell_1$ 稀疏局部损失 + 低方差初始化，使新增残差块在原始分布上输出近零值，在 Gemma-2B 上实现新语言学习与英语保持的最佳权衡。
tags:
  - ICML2025
  - 模型压缩
  - Adapter
  - 灾难性遗忘
  - 模型扩展
  - 门控机制
  - 多语言
  - LLM
---

# Neutral Residues: Revisiting Adapters for Model Extension

**会议**: ICML2025  
**arXiv**: [2410.02744](https://arxiv.org/abs/2410.02744)  
**代码**: 未开源  
**领域**: 模型扩展 / Model Extension  
**关键词**: Adapter, 灾难性遗忘, 模型扩展, 门控机制, 多语言, LLM

## 一句话总结

提出 **Neutral Residues**，通过在 adapter 中引入 ReLU 门控 + $\ell_1$ 稀疏局部损失 + 低方差初始化，使新增残差块在原始分布上输出近零值，在 Gemma-2B 上实现新语言学习与英语保持的最佳权衡。

## 研究背景与动机

大语言模型从头训练成本极高（Llama 3 估计数亿美元），因此**模型扩展**（model extension）成为关键需求：在不重训的情况下为已有模型添加新能力。

现有方案的局限：

- **微调（Finetuning）**：不增加容量，大量新知识导致灾难性遗忘严重
- **LoRA**：低秩更新同样不增加容量，在分布差异大时效果有限
- **Vanilla Adapter**：虽增加参数但仍存在显著遗忘

核心矛盾：**学习新分布 vs. 保持原始能力的 trade-off**。作者从数据 / 架构 / 训练三个维度联合优化这一矛盾。

## 方法详解

### 1. 混合分布训练

训练时保留 $p = 10\%$ 的近似原始分布数据（如英语 Wikipedia），在不显著拖慢新语言学习的前提下大幅减轻遗忘：

| 英语数据比例 $p$ | 英语 PPL ↓ | 法语 PPL ↓ |
|:-:|:-:|:-:|
| 0.00 | 0.720 | 0.810 |
| 0.01 | 0.707 | 0.810 |
| **0.10** | **0.687** | **0.812** |
| 0.50 | 0.683 | 0.828 |

### 2. 门控 Adapter 架构

在每个 Transformer 层的 FFN 旁**并行**添加 adapter 块，并在外部加一个 **block gate**：

$$
\text{Output} = \text{FFN}(x) + g(x) \cdot \text{Adapter}(x)
$$

其中 adapter 内部仍使用 GLU 结构：

$$
\text{Adapter}(x) = \mathbf{W}_o \left( \sigma(\mathbf{W}_g x) \odot \mathbf{W}_i x \right)
$$

门控激活函数选择 **ReLU**（而非 Sigmoid），使得在原始分布上门控输出自然趋零。

### 3. 局部稀疏损失

总训练损失为：

$$
\ell_{\text{train}} = \ell_{\text{LM}} + \alpha \, \ell_{\text{local}}
$$

- $\ell_{\text{local}}$：对原始分布数据，计算所有 adapter 输出的 $\ell_1$ 范数均值（按模型维度归一化）
- 默认 $\alpha = 0.01$
- 该损失**仅在原始分布数据上**施加，迫使 adapter 在处理英语时输出接近零 → "neutral residues"

与 Sigmoid + 交叉熵门控的对比：ReLU + $\ell_1$ 不做显式分类，让门控自适应地调节响应强度，避免门控权重矩阵奇异值过度偏斜的问题。

### 4. 低方差初始化

- 输出矩阵初始化为 **全零**（标准做法）
- 输入矩阵与门控矩阵的初始化方差从 He 初始化的 $2/d$ 降为 $1/(d \cdot L)$（$L$ 为 Transformer 层数），使模型在训练早期更长时间保持接近原始网络

### 5. FFN vs. MHA

与 LoRA 在 attention 上效果更好不同，模型扩展场景下在 **FFN** 上添加额外参数更有效，这与 MoE 的设计选择一致。

## 实验关键数据

### 主实验：Gemma-2B 扩展到四种语言（$p=0.1$，20% 额外参数）

| 目标语言 | 方法 | 遗忘（英语 avg）↑ | 学习（目标 avg）↑ |
|:-:|:-:|:-:|:-:|
| 法语 | Backbone | 53.2 | 44.6 |
| | Finetuning | 49.8 | **49.9** |
| | LoRA | 51.2 | 44.9 |
| | Adapter | 52.8 | 46.0 |
| | **Neutral Residues** | **53.3** | 48.2 |
| 丹麦语 | Finetuning | 47.2 | 42.8 |
| | **Neutral Residues** | **52.0** | **42.9** |
| 匈牙利语 | Finetuning | 45.9 | 38.5 |
| | **Neutral Residues** | **52.6** | **38.8** |
| 斯洛伐克语 | Finetuning | 46.4 | 39.2 |
| | **Neutral Residues** | **51.5** | 38.6 |

**关键发现**：Neutral Residues 在所有语言上英语保持最好（仅次于 backbone），同时目标语言学习接近或超越微调，提供最佳综合 trade-off。

### 消融实验：门控与局部损失

| 门控 | 损失 | 英语 PPL | 法语 PPL | 英语 Tasks | 法语 Tasks |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 无 | $\ell_1$ | 0.687 | 0.801 | 45.3 | 42.4 |
| Sigmoid | CE | 0.677 | 0.800 | 45.2 | 42.6 |
| **ReLU** | **$\ell_1$** | **0.674** | **0.791** | **47.1** | **43.6** |
| Adapter baseline | — | 0.686 | 0.812 | 45.1 | 41.3 |

### Perplexity 对比（EN-LM-1B，$p=0.1$）

| 方法 | 英语 PPL ↓ | 法语 PPL ↓ |
|:-:|:-:|:-:|
| 预训练模型 | 0.663 | 1.175 |
| Finetuning | 0.811 | 0.758 |
| LoRA | 0.730 | 0.818 |
| Vanilla Adapter | 0.687 | 0.812 |
| **Neutral Residues** | **0.668** | **0.793** |

## 亮点与洞察

1. **"中性残差"概念精巧**：adapter 在原始分布上输出近零 → 网络行为回退到 backbone，天然避免遗忘
2. **数据 / 架构 / 训练三层联动**：混合训练 + 门控 + 稀疏损失 + 低方差初始化，每个组件都有清晰消融验证
3. **ReLU > Sigmoid 门控**：避免门控矩阵奇异值偏斜，比显式域分类器更有效
4. **实用性强**：不要求原始训练数据，仅需近似分布（如 Wikipedia）即可工作
5. **与 MoE 理念相通但更简洁**：每层仅一个额外 FFN + gate，无需路由或负载均衡

## 局限与展望

1. **仅验证多语言场景**：虽然方法通用，但缺乏代码、数学、多模态等其他领域扩展的验证
2. **规模有限**：最大仅 Gemma-2B，未在 7B+ 模型上验证
3. **推理开销**：增加 20% 参数意味着推理时永久增加计算量（不像 LoRA 可合并）
4. **gate 的可解释性**：ReLU gate 的激活分布缺乏深入分析，何时"中性"何时"激活"的分界不够清晰
5. **无代码开源**：可复现性受限

## 评分

- 新颖性: ⭐⭐⭐⭐ — 中性残差的概念巧妙，ReLU gate + $\ell_1$ 稀疏损失组合新颖
- 实验充分度: ⭐⭐⭐⭐ — 多语言多基准、消融全面，但缺大模型和多领域验证
- 写作质量: ⭐⭐⭐⭐⭐ — 动机清晰、分析透彻、图表精美
- 价值: ⭐⭐⭐⭐ — 为模型扩展提供了实用且优雅的方案，对持续学习社区有启发

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[ICML 2025\] Random Initialization of Gated Sparse Adapters (RIGSA)](random_initialization_of_gated_sparse_adapters.md)
- [\[ACL 2025\] Limited-Resource Adapters Are Regularizers, Not Linguists](../../ACL2025/model_compression/limited-resource_adapters_are_regularizers_not_linguists.md)
- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](../../NeurIPS2025/model_compression/revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)

<!-- RELATED:END -->
