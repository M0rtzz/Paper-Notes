---
title: >-
  [论文解读] Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation
description: >-
  [ACL 2025][多语言机器翻译] 提出 Registering 方法：在源语言和目标语言 token 之间插入一组目标语言标记（registers），通过修改注意力掩码使目标生成仅依赖 registers 的激活，彻底解决多语言翻译中的 off-target 问题，使小模型 MITRE-913M 超越 NLLB-3.3B。
tags:
  - ACL 2025
  - 多语言机器翻译
  - 注册机制
  - 注意力掩码
  - off-target问题
  - decoder-only
---

# Registering Source Tokens to Target Language Spaces in Multilingual Neural Machine Translation

**会议**: ACL 2025  
**arXiv**: [2501.02979](https://arxiv.org/abs/2501.02979)  
**代码**: [GitHub](https://github.com/zhiqu22/mitre)  
**领域**: 文本生成  
**关键词**: 多语言机器翻译, 注册机制, 注意力掩码, off-target问题, decoder-only

## 一句话总结

提出 Registering 方法：在源语言和目标语言 token 之间插入一组目标语言标记（registers），通过修改注意力掩码使目标生成仅依赖 registers 的激活，彻底解决多语言翻译中的 off-target 问题，使小模型 MITRE-913M 超越 NLLB-3.3B。

## 研究背景与动机

多语言神经机器翻译（MNMT）旨在支持多种语言之间的任意翻译。传统的 MNMT 专用模型参数量小、支持零样本翻译，但性能一直落后于 LLM。

核心瓶颈是 **off-target 问题**：翻译输出没有到达目标语言。原因在于虽然输入序列开头有语言标签 $l_y$ 指示目标语言，但在注意力机制中，$l_y$ 的影响被大量源语言 token 稀释，导致生成过程无法严格遵循目标语言。

已有的解决方案各有局限：
- **LAVS**：添加语言特定词汇，代价高且阻碍知识共享
- **CL**：对齐跨语言语义表示，间接缓解
- **LCS**：用目标语言嵌入偏置编码器表示，仅限 Enc-dec 架构
- **TDO**：将 Dec-only 分成两阶段，增强翻译指令

## 方法详解

### 整体框架

Registering 的核心思想：在源序列和目标序列之间插入一组**寄存器 tokens**（registers），通过精心设计的注意力掩码，使目标 token 的生成**完全隔离于源 token**，只能通过 registers 获取源语言信息。

给定源序列 $\boldsymbol{x}' = l_y, x_1, \ldots, x_I$，插入寄存器序列 $\boldsymbol{r} = r_1, \ldots, r_{I+1}$（与源序列等长），生成过程变为：

$$y_j = \text{decoder}(\boldsymbol{x}', \boldsymbol{r}, \boldsymbol{y}_{<j})$$

### 关键设计一：寄存器初始化与注意力掩码

**寄存器初始化**：所有 register token 用目标语言标签 $l_y$ 的嵌入初始化（因为每个 register 最终都应处于目标语言空间）。

**注意力掩码规则**（基于 prefix Dec-only 方案修改）：
1. **$\boldsymbol{r}$ 关注 $\boldsymbol{x}'$**：registers 可以看到所有源 token
2. **$\boldsymbol{r}$ 内部双向注意力**：registers 之间可以相互交互
3. **$y_j$ 仅关注 $\boldsymbol{r}$ 和 $\boldsymbol{y}_{<j}$**：目标 token 看不到源 token，只能通过 registers 间接获取信息

这样设计的效果：每个 $r_i$ 会捕获其位置对齐的源 token $x_i$ 的语义，并将其"翻译"到目标语言空间。生成被严格约束在目标语言空间内。

### 关键设计二：不引入额外参数

Registering 完全通过修改注意力掩码实现，**不引入任何新参数**。这使其可以无缝应用于任何 Dec-only 模型，且参数效率优于 Enc-dec。

### 损失函数 / 训练策略

使用标准交叉熵损失，仅计算目标 token 的损失：

$$\mathcal{L}_{ce} = -\sum_{\boldsymbol{x}', \boldsymbol{y} \in \mathbb{C}} \sum_{j=1}^{J} \log p(y_j \mid \boldsymbol{x}', \boldsymbol{r}, \boldsymbol{y}_{<j})$$

预训练 MITRE 模型使用 9.3B 句对、24 种语言、194 个翻译方向。词表通过 SentencePiece 训练，大小 160K。采用 Bridge Language 策略采集数据，每个语言族选两种高资源语言作为桥接语言。

## 实验关键数据

### 主实验一：EC-40 基准（零样本翻译）

在 EC-40 上 1640 个翻译方向的 spBLEU 结果（Table 1 摘选，24 层配置）：

| 模型 | 参数量 | sup. | zero. | avg. | off-target(%) |
|------|--------|------|-------|------|---------------|
| Enc-dec vanilla | 418M | 30.28 | 8.64 | 9.69 | 26.69 |
| + CL | 418M | 30.54 | 10.79 | 11.76 | 19.99 |
| + LAVS | 430M | 30.20 | 10.03 | 11.01 | 21.73 |
| Dec-only vanilla | 368M | 29.97 | 9.84 | 10.82 | 19.01 |
| + TDO | 368M | 30.23 | 10.40 | 11.37 | 23.14 |
| **+ Ours** | **368M** | 29.88 | **12.26** | **13.12** | **3.65** |

off-target 率从 26.69% 降至 **3.65%**，几乎彻底解决 off-target 问题。零样本翻译 spBLEU 提升幅度远超所有基线。

### 主实验二：MITRE 预训练模型 vs 大模型

大规模预训练后的 spBLEU 结果对比（Table 3 摘选）：

| 模型 | 参数量 | avg. spBLEU |
|------|--------|-------------|
| M2M-1.2B | 1.2B | 24.69 |
| NLLB-3.3B | 3.3B | 30.01 |
| GPT-3.5 Turbo | - | 28.66 |
| GPT-4o mini | - | 31.09 |
| **MITRE-913M** | **913M** | **31.15** |

MITRE-913M 以 913M 参数超越 NLLB-3.3B（+1.14）和 GPT-3.5 Turbo，与 GPT-4o mini 持平。

### 消融实验

**架构选择**：Dec-only + Registering 优于 Enc-dec，因为参数效率更高且 registers 天然适配 Dec-only 的注意力结构。

**可扩展性**：当模型从 12 层扩展到 24 层时，其他方法的增益递减（从 3.88 降到 2.15），而 Registering 保持一致的增益（5.19→3.62），展现更优的扩展性。

**微调适应性**：预训练的 MITRE 在 LoRA 和全参数微调中均展现强大适应性，仅用 Flores dev（997 句对/方向）即可显著提升特定方向的翻译质量。

### 关键发现

1. off-target 问题的根源是注意力机制中源语言 token 对语言标签的稀释
2. Registers 的激活确实反映了目标语言空间中源 token 的语义（通过注意力和表示分布分析验证）
3. 简单添加语言特定参数（LAVS）不是 cost-efficient 的方案
4. Dec-only 架构在参数效率上优于 Enc-dec

## 亮点与洞察

- **概念优雅**：Registering 可以理解为 representation 级的 chain-of-thought——从目标语言的角度"重新思考"源 token
- **零额外参数**：仅通过修改注意力掩码实现，方法极其简洁
- **实际效果震撼**：913M 参数模型达到 GPT-4o mini 级别，且 off-target 率降至近零
- 方法论上融合了 gisting（mask-based 信息压缩）和 prefix-tuning 的思想

## 局限与展望

- Registers 的长度固定为源序列长度，对极长句子可能增加计算开销
- Register 初始化为统一的语言标签，未探索更智能的初始化策略
- 仅在 NLLB 数据集上预训练，数据质量和覆盖范围可能限制某些语言对的性能
- 推理时序列长度翻倍（源 + registers + 目标），对延迟有影响

## 相关工作与启发

- **Gisting** (Mu et al., 2023)：用修改后的注意力掩码将信息压缩到人工 token 中，是 Registering 的方法论灵感来源
- **NLLB** (NLLB Team, 2022)：当前 MNMT 标杆，MITRE 的主要对比对象
- **TDO** (Qu et al., 2024b)：前序工作，通过两阶段处理增强翻译指令，Registering 是其进化版

## 评分

- **新颖性**: 4/5 — 将 gisting 思想巧妙迁移到 MNMT，概念清晰且原创
- **技术深度**: 4/5 — 注意力掩码设计简洁有力，大规模预训练验证充分
- **实验充分度**: 5/5 — EC-40 + 大规模预训练 + 微调 + 多指标，极其全面
- **实用价值**: 5/5 — 开源模型，可直接使用，工业价值高

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [Unveiling the Power of Source: Source-based Minimum Bayes Risk Decoding for Neural Machine Translation](unveiling_the_power_of_source_source-based_minimum_bayes_risk_decoding_for_neura.md)
- [Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation](memorization_inheritance_seqkd.md)
- [THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)
- [Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)

<!-- RELATED:END -->
