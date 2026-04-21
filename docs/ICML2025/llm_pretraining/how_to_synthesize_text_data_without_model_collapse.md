---
title: >-
  [论文解读] How to Synthesize Text Data without Model Collapse?
description: >-
  [ICML 2025][模型崩溃] 提出 Token-level Editing (ToEdit)，通过对人类数据进行 token 级别的局部重采样（而非完全生成合成数据），在理论上证明测试误差存在有限上界，从而避免 model collapse，并在预训练、持续预训练和微调三个阶段验证了有效性。
tags:
  - ICML 2025
  - 模型崩溃
  - 合成数据
  - token编辑
  - 数据分布
  - 半合成数据
---

# How to Synthesize Text Data without Model Collapse?

**会议**: ICML 2025  
**arXiv**: [2412.14689](https://arxiv.org/abs/2412.14689)  
**代码**: [GitHub](https://github.com/Xuekai-Zhu/toedit)  
**领域**: LLM/NLP  
**关键词**: 模型崩溃, 合成数据, token编辑, 数据分布, 半合成数据

## 一句话总结

提出 Token-level Editing (ToEdit)，通过对人类数据进行 token 级别的局部重采样（而非完全生成合成数据），在理论上证明测试误差存在有限上界，从而避免 model collapse，并在预训练、持续预训练和微调三个阶段验证了有效性。

## 研究背景与动机

随着生成式 AI 的普及，互联网上的合成数据将日益增多。未来 GPT-{n} 模型不可避免地要在合成数据与人类数据的混合集上进行训练。Model collapse 是这一背景下的核心风险——迭代地在自生成数据上训练会导致模型性能持续退化。

本文聚焦两个关键问题：

**合成数据对语言模型预训练的影响是什么？** 作者发现即使是非迭代地直接混合合成数据，也会损害预训练效果（非迭代 model collapse）。

**如何合成数据而不引发 model collapse？** 作者提出 token 级编辑策略，生成"半合成数据"来规避分布坍缩。

之前的理论工作（Shumailov et al., 2024; Dohmatob et al., 2024a）已证明迭代训练导致测试误差线性增长 $E_{test} = \frac{\sigma^2 d}{T-d-1} \times n$。Gerstgrasser et al. (2024) 证明数据累积可以打破 collapse，但缺乏实用的数据合成方案。本文的核心创新在于：不再"纯合成"，而是在人类原始数据上做受控的 token 级修改，同时提供理论保证。

## 方法详解

### 整体框架

ToEdit 的核心思路可分为三步：

1. **用先验语言模型推断 token 概率**：给定人类文本序列 $x = (x_1, \dots, x_t)$，使用预训练 LM（如 Llama-3-8B）计算每个 token 的条件概率 $P(x_i | x_1, \dots, x_{i-1})$。
2. **识别"过于容易"的 token**：如果某 token 的条件概率 $P(x_i | \text{context}) \geq p$（阈值 $p$），说明该 token 对模型来说太容易预测，信息量低，存在过度集中的风险。
3. **重采样替换**：对这些高置信度 token 进行重采样，用从先验分布中采样的新 token $\tilde{x}_i$ 替换，其他 token 保持原样。

这一过程只需**单次前向推理**，不涉及自回归生成，计算效率高（单张 4090 即可完成）。

### 关键设计

#### Token 编辑公式

$$
x_i' = \begin{cases} x_i, & \text{if } P(x_i | x_1, \dots, x_{i-1}) < p \\ \tilde{x}_i, & \text{if } P(x_i | x_1, \dots, x_{i-1}) \geq p \end{cases}
$$

其中 $\tilde{x}_i$ 是根据条件概率分布重新采样的 token，$p$ 是控制编辑程度的阈值。

**直觉理解**：人类文本的 token 概率呈 U 型分布——高概率（容易预测）和低概率（难以预测）的 token 最为集中。ToEdit 只替换高概率端的 token，保留了低概率端的长尾特征，从而维持分布覆盖度。

#### 为什么不用纯合成数据？——四大发现

作者通过系统实验揭示了合成数据的根本缺陷：

- **Finding I**：混入合成数据伤害预训练。在 GPT-2 (124M) 上，合成数据比例与 PPL 呈负相关——100% 合成数据的平均 PPL 从 20.99 暴增至 51.93。
- **Finding II**：合成数据分布缺失长尾且覆盖范围收窄。用 Llama-3-8B 估计 PPL 分布，人类数据覆盖 [1, 100+]，而合成数据仅集中在 [0, 14]，只覆盖人类数据的前 25%。
- **Finding III**：合成数据 N-gram 特征过度集中。将 uni-gram/bi-gram 哈希到 10000 个桶中，合成数据的响应集中在少量桶，缺乏人类数据的广泛覆盖。
- **Finding IV**：数据选择无法修正分布偏移。即使用 DSIR 重要性采样从合成数据中筛选，性能仍在原始合成数据水平波动，无法对齐人类数据分布。

#### 操作矩阵形式化

在理论分析框架中，token 编辑被形式化为对角矩阵 $M_i$ 的操作:

$$
\tilde{Y}_n = M_{n-1} \hat{Y}_n + (1 - M_{n-1}) \tilde{Y}_{n-1}
$$

$M_i$ 是对角矩阵，对角元素为 0 或 1，决定哪些数据点被编辑（1）或保留（0）。这使得每次迭代只修改部分数据，而非全部替换。

### 损失函数 / 训练策略

#### 理论保证：测试误差有限上界

**Theorem 2**：在 token 编辑设定下，经过 $n+1$ 轮迭代编辑，测试误差满足：

$$
E_{test}(\hat{w}_{n+1}) \leq \frac{2\sigma^2 d}{T - d - 1}
$$

这是一个**与迭代轮数 $n$ 无关的有限上界**。相比之下，model collapse 的误差为 $\frac{\sigma^2 d}{T-d-1} \times n$，随 $n$ 线性增长。

进一步，若编辑操作满足衰减条件 $\|M_i\| = \|M_{i-1}\| \eta$（$\eta \in (0,1)$），则上界可以更紧：

$$
E_{test}(\hat{w}_{n+1}) \leq \frac{\sigma^2 d}{T-d-1} + \sigma^2 \sqrt{\mathbb{E}[\text{tr}((X^\top X)^{-2})]} \cdot \frac{\sqrt{\mathbb{E}[\text{tr}(M_1)]}}{1-\eta}
$$

**核心逻辑**：因为每轮只修改部分 token（由 $M_i$ 控制），原始分布覆盖得以保留，噪声不会跨迭代累积。

#### 实现细节

- **先验模型**：Llama-3-8B 作为概率估计器
- **编辑阈值**：$p = 0.99$（只替换条件概率 ≥ 0.99 的 token）
- **采样策略**：top-k（$k=8$）
- **推理引擎**：vLLM 快速推理，单张 4090 即可完成数据编辑
- **无需自回归生成**：只做单次前向传播

## 实验关键数据

### 主实验

实验覆盖三个训练阶段：从头预训练、持续预训练、有监督微调。

**非迭代 model collapse（从头预训练 GPT-2 124M）**:

| 数据配比 | 22子域平均PPL↓ | 对比纯人类数据 |
|---|---|---|
| 100% Human | 20.99 / 22.59 | 基线 |
| 25% Synthetic | 22.06 / 23.91 | +1.07 / +1.32 |
| 50% Synthetic | 23.48 / 25.09 | +2.49 / +2.50 |
| 75% Synthetic | 27.60 / 28.64 | +6.61 / +6.05 |
| 100% Synthetic | 51.93 / 47.87 | +30.94 / +25.28 |

**持续预训练（Biomedicine 领域）**:

| 模型 | MQP | ChemProt | PubMedQA | RCT | USMLE | Avg |
|---|---|---|---|---|---|---|
| OLMo-1B 基线 | 52.59 | 17.2 | 51.40 | 32.70 | 28.90 | 36.63 |
| OLMo-1B CPT | 52.29 | 21.00 | 58.50 | 34.90 | 27.49 | 38.83 |
| OLMo-1B + ToEdit | **54.59** | **22.40** | **65.00** | 34.50 | 27.96 | **40.89** |
| Llama-3-8B 基线 | 66.80 | 28.59 | 60.8 | 73.85 | 40.61 | 54.13 |
| Llama-3-8B CPT | 72.29 | 29.4 | 69.1 | 72.65 | 36.76 | 56.04 |
| Llama-3-8B + ToEdit | **76.39** | **30.2** | 65.3 | 73.30 | 37.23 | **56.48** |

**SFT（指令微调 Llama-3-8B）**:

| 任务 | 原始Avg | +ToEdit Avg | 提升 |
|---|---|---|---|
| Natural Instructions | 69.34 | 69.70 | +0.36 |
| CoT | 69.01 | 69.26 | +0.25 |
| FLANv2 | 70.18 | 70.65 | +0.47 |
| Open Assistant | 69.19 | 69.44 | +0.25 |
| OSS-Instruct (代码) | 45.76 | 46.13 | +0.37 |
| Evol-Instruct (代码) | 46.62 | 46.92 | +0.30 |

### 消融实验

| 配置 | Avg (Biomedicine) | 说明 |
|---|---|---|
| $p \geq 0.99$ | **38.69** | 默认阈值，替换约27%高概率token |
| $p \geq 0.999$ | 38.48 | 更保守，替换更少token |
| $p \leq 0.1$ | 35.72 | 替换低概率token，效果差 |
| $p \leq 0.01$ | 37.46 | 替换极低概率token |
| Top-k (k=8) | 基线 | 默认，计算高效 |
| Top-p | 可比 | 动态采样范围，开销更大 |
| Rejection Sampling | 可比 | 多轮计算，开销最大 |
| k=8 vs k=64 | 差异小 | 增大k收益有限 |

### 关键发现

1. **合成数据比例与性能呈明确负相关**：100% 合成数据的 PPL 是纯人类数据的 2.5 倍。
2. **ToEdit 在三个训练阶段一致有效**：预训练 +0.36、持续预训练平均 +2 以上、SFT +0.25~0.47。
3. **高置信度 token 替换优于低置信度**：$p \geq 0.99$ 显著优于 $p \leq 0.1$，验证了方法动机——替换模型已经掌握的"容易"token 才有增益。
4. **token 概率的 U 型分布**：约 27.1% 的 token 概率在 [0.9, 1.0)，34.7% 在 [0.0, 0.1)，两端集中。
5. **理论与实践一致**：有限上界保证了多轮迭代不会 collapse，实验证实了这一点。

## 亮点与洞察

- **问题定义精准**：区分了"迭代 model collapse"和"非迭代 model collapse"，后者更贴近实际训练场景（直接混合而非迭代生成）。
- **方法极简高效**：不需要自回归生成、不需要额外训练、单次前向传播 + top-k 采样，单张 4090 即可运行，工程成本极低。
- **理论扎实**：在线性回归框架下严格证明了测试误差上界，从 $O(n)$ 降到 $O(1)$。
- **"半合成数据"概念有启发性**：不是"生成替代人类数据"的思路，而是"在人类数据上做最小修改"的思路，保留分布覆盖的同时提升数据质量。
- **四大统计发现**系统扎实：从分布、特征、数据选择三个维度彻底解释了为什么合成数据不行。

## 局限与展望

1. **理论框架局限于线性模型**：实际 LLM 是高度非线性的 Transformer，线性回归框架的上界保证能否真正迁移到实际场景存疑。
2. **改进幅度有限**：SFT 阶段提升仅 0.25~0.47 个点，预训练提升也不大（32.75→33.11），实用价值需进一步验证。
3. **依赖强先验模型**：需要 Llama-3-8B 级别的模型来估计 token 概率，对资源有限的场景仍有门槛。
4. **阈值 $p$ 的选择**：消融显示不同 $p$ 值表现有波动，缺乏自适应选择策略。
5. **未涉及多轮迭代编辑的实证验证**：理论证明了多轮迭代的上界，但实验只做了单轮编辑。

## 相关工作与启发

- **Model Collapse 理论**：Shumailov et al. (2024)、Dohmatob et al. (2024a,b,c) 奠定了理论基础；Gerstgrasser et al. (2024) 提出数据累积可以打破 collapse。本文进一步提出实用的 token 编辑方案。
- **合成数据质量**：Cosmopedia、Phi 系列等依赖高质量合成数据，但本文揭示了合成数据的分布缺陷是根本性的。
- **数据选择**：DSIR 等方法试图通过重要性采样筛选数据，本文实验表明这对合成数据的分布偏移无效。
- **启发**：这种"最小编辑"的思想可以推广到其他数据增强场景——与其生成全新数据，不如在原始数据上做受控的局部修改。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|---|---|---|
| 新颖性 | 4 | "token 级编辑"而非"纯合成"的思路新颖且实用 |
| 理论深度 | 4 | 线性回归框架下的证明严谨，但非线性场景存疑 |
| 实验充分度 | 4 | 覆盖三个训练阶段、多模型、多领域，消融完整 |
| 实用性 | 3.5 | 方法简单可落地，但改进幅度偏小 |
| 写作质量 | 4 | 逻辑清晰，四个 Finding 层层递进 |
| **综合** | **4** | 扎实的工作，问题重要、方法简洁、理论与实验俱全 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning](chameleon_a_flexible_data-mixing_framework_for_language_model_pretraining_and_fi.md)
- [Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](../../NeurIPS2025/llm_pretraining/final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization](../../ACL2025/llm_pretraining/model_performance-guided_evaluation_data_selection_for_effective_prompt_optimiza.md)
- [Metadata Conditioning Accelerates Language Model Pre-training](metadata_conditioning_accelerates_language_model_pre-training.md)

<!-- RELATED:END -->
