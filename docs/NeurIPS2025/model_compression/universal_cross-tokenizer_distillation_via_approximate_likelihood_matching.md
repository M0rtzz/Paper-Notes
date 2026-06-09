---
title: >-
  [论文解读] Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching
description: >-
  [NeurIPS 2025][模型压缩][跨分词器蒸馏] 本文提出 Approximate Likelihood Matching (ALM)，一种基于二值化 f-散度的原则性跨分词器蒸馏方法，首次实现了跨根本不同分词器（如子词→字节级）的有效蒸馏和纯蒸馏。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "跨分词器蒸馏"
  - "近似似然匹配"
  - "分词器迁移"
  - "f-散度"
  - "LLM蒸馏"
---

# Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching

**会议**: NeurIPS 2025  
**arXiv**: [2503.20083](https://arxiv.org/abs/2503.20083)  
**代码**: [https://github.com/bminixhofer/tokenkit](https://github.com/bminixhofer/tokenkit)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 跨分词器蒸馏, 近似似然匹配, 分词器迁移, f-散度, LLM蒸馏

## 一句话总结

本文提出 Approximate Likelihood Matching (ALM)，一种基于二值化 f-散度的原则性跨分词器蒸馏方法，首次实现了跨根本不同分词器（如子词→字节级）的有效蒸馏和纯蒸馏。

## 研究背景与动机

知识蒸馏是创建高效语言模型的重要范式，但现有蒸馏方法要求教师模型和学生模型使用相同或相似的分词器，这极大限制了可用的教师-学生模型对。当前 LLM 领域的分词器异常多样化：不同模型（GPT、Llama、Gemma、Qwen）使用不同的词表和分词函数，且近年来出现了从子词向字符/字节级分词的趋势。

**核心矛盾**：标准蒸馏通过 KL 散度比较教师和学生的 token 级概率分布，但这要求两者共享相同的 token 空间。当分词器不同时，教师和学生将同一文本分割为不同的 token 序列，无法直接比较。已有的跨分词器方法（ULD、MinED、DSKD）使用启发式方法融入教师信息，只能作为辅助目标与主目标（如 next-token prediction）联合使用，无法实现纯蒸馏。

**本文切入角度**：将问题形式化为比较对齐的 token chunk 的似然值，通过二值化 f-散度近似来避免枚举无穷多可能的字节序列结果。

## 方法详解

### 整体框架

给定文本 x，分别用教师和学生分词器分词，计算所有 token 的 next-token 概率。找到两个序列之间的对齐 chunk（编码相同文本片段的 token 子序列），然后最小化对齐 chunk 之间似然的差异。

### 关键设计

1. **Chunk级概率对齐**：核心思想是找到教师和学生序列中编码相同文本的 token chunk，定义 chunk 级概率为 $p(\mathbf{x}, i:j) = p(T(\mathbf{x})_{i:j} | T(\mathbf{x})_{:i})$。理想情况下应计算所有可能 chunk 结果的 f-散度，但由于可能的字节序列无穷多，作者转而计算二值化 f-散度——只考虑"该 chunk 出现"和"该 chunk 不出现"两种情况。这构成了真实 f-散度的上界，并保留了关键性质（当且仅当 $p_S = p_T$ 时最小）。

2. **Outcome Chunk Debiasing（结果chunk去偏）**：子词分词器存在分词偏差（tokenization bias），例如"Hello_Wor"隐含了后面不是"ld"的信息。通过将 chunk 概率乘以一个去偏概率（下一个 token 以预分词边界字节开头的概率）来消除这种偏差。设置阈值 γ 过滤去偏概率过低的 chunk，避免概率被"擦除"到极低值。

3. **隐藏状态蒸馏**：由于 chunk 级概率仅提供 $|A_c| \times 32$ 比特的信号（相比同分词器蒸馏的 $|T(\mathbf{x})| \times |\mathcal{V}| \times 32$ 比特），额外添加隐藏状态对齐损失来丰富信号。对齐教师和学生模型中对应位置的隐藏状态，通过学习的投影函数 proj 映射维度。

4. **GradMag 损失组合**：针对多目标优化中不同损失梯度量级差异大的问题，提出简单的梯度幅值归一化方法：计算每个损失对最后一层的梯度，将权重设为 $1/\|G_W^i\|$，直接求解等梯度幅值条件，比 GradNorm 更简单且效果相当。

### 损失函数 / 训练策略

ALM 目标函数：
$$\mathcal{L}^{ALM}_{S,T}(\mathbf{x}) = \sum_{i,j,k,l \in A_c(\mathbf{x})} f(p_T^{1/\tau} \| p_S^{1/\tau}) + f(1-p_T^{1/\tau} \| 1-p_S^{1/\tau})$$

其中 f 是 f-散度的生成函数，τ 是温度超参数。可选择纯蒸馏模式（仅 ALM 损失）或混合蒸馏模式（ALM + SFT next-token prediction 损失），使用 GradMag 自动平衡。

## 实验关键数据

### 主实验：分词器迁移（Use Case 1）

| 模型 | 迁移目标 | 方法 | Avg | MMLU | BoolQ | IFEval |
|------|---------|------|-----|------|-------|--------|
| Gemma2 2B IT | 原始 | - | 58.0 | 56.9 | 83.8 | 62.5 |
| Gemma2 2B IT | →Qwen2 | SFT | 51.6 | 49.8 | 77.7 | 54.2 |
| Gemma2 2B IT | →Qwen2 | MinED | 53.0 | 51.8 | 79.6 | 57.1 |
| Gemma2 2B IT | →Qwen2 | ALM | 55.1 | 53.6 | 82.7 | 53.2 |
| Gemma2 2B IT | →Byte | SFT | 46.5 | 43.1 | 67.9 | 51.5 |
| Gemma2 2B IT | →Byte | ALM+SFT | 51.3 | 51.0 | 80.5 | 51.9 |
| Llama3.2 3B IT | →Qwen2 | ALM | 58.6 | 61.6 | 79.0 | 76.3 |

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| Outcome Chunk Debiasing | 显著提升性能 | 去除分词偏差，阈值 γ 进一步改善 |
| GradMag vs GradNorm | 持平或更优 | 更简单的损失平衡策略 |
| 纯ALM vs ALM+SFT | 子词迁移纯ALM更优 | 纯蒸馏更好保留原模型行为 |
| 字节迁移: ALM+SFT | 字节迁移ALM+SFT更优 | 极端分词变化下需保留SFT信号 |

### Use Case 2: 大→小跨分词器蒸馏

| 方法 | GSM8K | MATH | Avg |
|------|-------|------|-----|
| 教师 (OpenMath2-Llama3.1-8B) | 88.9 | 60.2 | 74.6 |
| SFT | 67.2 | 36.2 | 51.7 |
| DSKD | 65.7 | 34.9 | 50.3 |
| ALM+SFT | **70.2** | **36.4** | **53.3** |

### 关键发现

- ALM 首次实现了子词→字节级的有效蒸馏迁移（此前方法完全无效或不如 SFT）
- 将不同模型迁移到相同分词器后可进行 token 级集成，集成后性能优于单个模型
- 在效率方面，ALM 不需要 DSKD 的交叉注意力计算，也不需要 MinED 的大型 logit 矩阵对齐
- ALM 比最佳先前方法额外缩小了 34% 的教师-学生差距

## 亮点与洞察

- **原则性方法**：不同于先前启发式方法，ALM 提供了数学上有根据的蒸馏目标——二值化 f-散度是真实散度的上界且保留最优性条件
- **通用性强**：同一方法适用于子词→子词、子词→字节、大→小蒸馏、超网络训练等多种场景
- **分词器迁移即自蒸馏**：将分词器迁移重新定义为跨分词器自蒸馏的视角非常优雅
- **模型集成新可能**：通过迁移到统一分词器实现不同模型族的 token 级集成

## 局限与展望

- 二值化 f-散度是对真实散度的粗略近似，可能损失信息
- 字节级迁移仍与原始模型有较大差距，需要额外技术（如 hourglass 架构、多字节预测）
- Outcome Chunk Debiasing 只处理了结果端的偏差，条件端偏差仍未解决
- 隐藏状态对齐需要层级对应关系的先验知识

## 相关工作与启发

本文将跨分词器蒸馏从启发式方法提升为原则性框架。与 DSKD（基于交叉注意力的 token 对齐）和 MinED（基于编辑距离的最小匹配）相比，ALM 更高效且不需要预计算步骤。"分词器迁移即自蒸馏"的洞察为创建字节级模型提供了比从头训练更经济的路径。其打开了不同模型族之间知识组合的新可能。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次原则性地解决跨根本不同分词器的蒸馏问题，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 三个Use Case全面验证，消融详尽，效率比较充分
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，方法描述清晰，图表直观
- 价值: ⭐⭐⭐⭐⭐ 解决了LLM蒸馏的核心瓶颈问题，实际意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](../../AAAI2026/model_compression/ctpd_cross_tokenizer_preference_distillation.md)
- [\[NeurIPS 2025\] Explaining and Mitigating Crosslingual Tokenizer Inequities](explaining_and_mitigating_crosslingual_tokenizer_inequities.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[NeurIPS 2025\] Learning to Factorize and Adapt: A Versatile Approach Toward Universal Spatio-Temporal Foundation Models](learning_to_factorize_and_adapt_a_versatile_approach_toward_universal_spatio-tem.md)
- [\[NeurIPS 2025\] Hyperbolic Dataset Distillation](hyperbolic_dataset_distillation.md)

</div>

<!-- RELATED:END -->
