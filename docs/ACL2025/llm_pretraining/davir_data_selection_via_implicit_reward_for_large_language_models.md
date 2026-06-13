---
title: >-
  [论文解读] DavIR: Data Selection via Implicit Reward for Large Language Models
description: >-
  [ACL 2025][预训练][数据选择] 提出 DavIR 数据选择方法，通过对基座模型与参考模型的损失差进行**参考模型损失归一化**（而非 token 数归一化），有效消除 RHO 目标中的序列长度依赖，使仅 **6%** 的 Alpaca 数据集（3K/52K）训练出的模型优于全量数据训练模型，同时将归一化思想推广到 DPO 得到 DavIR-DPO，在 AlpacaEval 上提升 Zephyr 8% 的对齐性能。
tags:
  - "ACL 2025"
  - "预训练"
  - "数据选择"
  - "核心集选择"
  - "隐式奖励"
  - "DPO"
  - "指令微调"
  - "长度偏差"
  - "可学习性"
---

# DavIR: Data Selection via Implicit Reward for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2310.13008](https://arxiv.org/abs/2310.13008)  
**代码**: 未公开  
**作者**: Haotian Zhou, Tingkai Liu, Qianli Ma, Yufeng Zhang, Jianbo Yuan, Pengfei Liu, Yang You, Hongxia Yang  
**机构**: ByteDance, Cold Spring Harbor Laboratory, Shanghai Jiao Tong University, National University of Singapore  
**领域**: LLM预训练  
**关键词**: 数据选择, 核心集选择, 隐式奖励, DPO, 指令微调, 长度偏差, 可学习性

## 一句话总结

提出 DavIR 数据选择方法，通过对基座模型与参考模型的损失差进行**参考模型损失归一化**（而非 token 数归一化），有效消除 RHO 目标中的序列长度依赖，使仅 **6%** 的 Alpaca 数据集（3K/52K）训练出的模型优于全量数据训练模型，同时将归一化思想推广到 DPO 得到 DavIR-DPO，在 AlpacaEval 上提升 Zephyr 8% 的对齐性能。

## 研究背景与动机

**SFT 数据选择的重要性**：根据"浅层对齐假设"（Superficial Alignment Hypothesis），少量精选数据即可引导预训练 LLM 展现指令跟随能力（Zhou et al. 2023 LIMA 仅用 1K 样本）。

**现有方法的局限**：
   - **以数据为中心的方法**：AlpaGasus 用 ChatGPT 打分过滤（质量导向）、LIMA 人工标注（多样性导向），但都**忽视了基座模型自身的能力**
   - **依赖外部教师模型**：ChatGPT 标注带来安全和成本问题
   - **以模型为中心但有缺陷**：RHO（Reducible Holdout Loss）理论上好，但直接应用于语言建模时与序列长度高度相关

**核心发现**：语言建模中，token 级别的熵/损失与序列长度的 Spearman 相关性高达 **-0.97**（Albert on Alpaca）。RHO 目标继承了这种相关性，导致数据选择退化为**近似按长度排序**。

**关键创新**：一个微妙但关键的归一化改变——用参考模型损失做分母，而非 token 数——可以大幅降低长度依赖。

## 方法详解

### 核心公式：从 RHO-LM 到 DavIR

**RHO-LM（基线）**：将 Reducible Holdout Loss 推广到因果语言建模

$$S_{\text{RHO-LM}}(x,y) = \mathcal{L}_{\text{base}}(y|x) - \mathcal{L}_{\text{ref}}(y|x)$$

其中 $\pi_{\text{base}}$ 是预训练基座模型，$\pi_{\text{ref}}$ 是在全量数据上微调后的参考模型。

**问题**：$S_{\text{RHO-LM}}$ 与序列长度高度相关（Pearson 相关 0.64-0.83，见 Table 2），原因是自回归语言建模中，更长序列提供更多上下文约束后续 token 分布，导致长序列的平均损失系统性偏低。

**DavIR（本文方法）**：

$$S_{\text{DavIR}}(x_i, y_i) = \frac{\mathcal{L}_{\text{base}}(x_i, y_i) - \mathcal{L}_{\text{ref}}(x_i, y_i)}{\mathcal{L}_{\text{base}}(x_i, y_i)}$$

- 关键区别：分母用**基座模型自身损失**而非 token 数
- 数学上等价于：$1 - \mathcal{L}_{\text{ref}} / \mathcal{L}_{\text{base}}$
- 直觉：衡量模型"学会的比例"——分母归一化消除了不同长度数据的绝对损失量级差异
- 用基座或参考模型损失做分母不影响排序（简单证明见附录 C）

### 与隐式奖励的关系

DPO 的隐式奖励函数：$r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{base}}(y|x)} = \beta \cdot [\mathcal{L}_{\text{base}} - \mathcal{L}]$

- RHO-LM 的评分函数正是（常数倍的）DPO 隐式奖励
- DavIR 可视为**归一化的隐式奖励**，选择"奖励相对学习潜力最大"的数据

### DavIR 算法流程

1. 在全量数据 $D_{\text{full}}$ 上微调基座模型得到 $\pi_{\text{ref}}$
2. 对每个 $(x_i, y_i) \in D_{\text{full}}$，计算 $\mathcal{L}_{\text{base}}$ 和 $\mathcal{L}_{\text{ref}}$
3. 计算 $S_{\text{DavIR}}$ 并排序
4. 选取 top-k 高分数据组成训练集
5. 用 $\pi_{\text{base}}$ 在 top-k 数据上重新微调

### DavIR-DPO 扩展

将归一化思想推广到 DPO 训练目标：

$$\mathcal{L}_{\text{DavIR-DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \frac{\log \pi_\theta(y_w|x) / \pi_{\text{ref}}(y_w|x)}{|\log \pi_{\text{ref}}(y_w|x)|} - \beta \frac{\log \pi_\theta(y_l|x) / \pi_{\text{ref}}(y_l|x)}{|\log \pi_{\text{ref}}(y_l|x)|}\right)\right]$$

- 对 winning 和 losing response 分别用各自的参考模型损失归一化
- 目的：减少 DPO 目标对回复长度差异的依赖

## 实验关键数据

### 长度依赖性分析

| 数据集 | 模型 | RHO-LM Spearman | DavIR Spearman |
|--------|------|-----------------|----------------|
| Alpaca | gemma-2b | 0.75 | **0.30** |
| Alpaca | gemma-2-2b | 0.83 | **0.47** |
| GSM8K | gemma-2b | 0.58 | **0.06** |
| LIMA | gemma-2b | 0.20 | **0.02** |

- DavIR 显著降低了与长度的相关性（最佳情况从 0.58 降至 0.06）

### SFT 数据选择：16x 压缩率

使用 LLaMA-7B/13B 在 Alpaca 数据集上的效果（Figure 1）：
- **3K/52K = 仅 6%** 的数据即超越全量训练
- GPT-4 评估和人工评估均确认 DavIR 优势
- 随机采样的性能随数据量对数增长，远低于 DavIR

### 与其他核心集选择方法比较（Gemma-2B, AlpacaEval）

| 方法 | 3K | 5K | 7K | 10K |
|------|-----|-----|-----|------|
| Random | 10.6 | 15.9 | 17.0 | 17.6 |
| Full (52K) | - | - | - | ~18 |
| EL2N (Highest) | 10.0 | 11.3 | 12.4 | 14.3 |
| Forgetting (Highest) | 9.5 | 13.4 | 16.7 | 18.2 |
| DataInf (Highest) | 10.3 | 15.9 | 18.7 | **18.8** |
| RHO (Highest) | 9.9 | 14.5 | 15.8 | ~16 |
| **DavIR** | **~12** | **~17** | **~19** | **~19** |

- DavIR 是**唯一在所有数据量下一致超越全量基线**的方法
- 低数据regime下（<5K）差距小，但高数据量优势明显

### DavIR-DPO 结果

| DPO 变体 | 与回复长度差的 Pearson 相关 |
|----------|------------------------|
| Vanilla DPO | 0.38 |
| IPO | -0.10 |
| AOT | 0.12 |
| **DavIR-DPO** | **0.07** |

- DavIR-DPO 对长度差异的依赖最低（0.07 vs 0.38）
- 在 Zephyr-7B-SFT 上，DavIR-DPO 在 AlpacaEval 上相对提升 **8%**（length-controlled）

### 数据混合实验

- DavIR 选择的 Alpaca 子集 + GSM8K 数据混合训练，可有效平衡开放域 QA 和数学推理能力
- 全量 Alpaca 混合 GSM8K 反而出现能力冲突

## 亮点与洞察

1. **一个微小的归一化改变带来巨大收益**：分母从 token 数换为参考模型损失，看似简单但效果戏剧性，体现了对问题本质的深刻理解
2. **理论-实践闭环**：从 DPO 隐式奖励建立理论联系 → 发现长度依赖问题 → 提出归一化解决 → 再将归一化推广回 DPO
3. **"可学习性"的精确量化**：DavIR 评分直接反映模型通过训练能学到多少（相对于其现有能力），是模型中心的选择标准
4. **计算高效**：仅需计算两次前向推理（base + ref 的损失），不需要梯度/Hessian（如 DataInf），不需要 ChatGPT API
5. **统计严谨**：通过 bootstrap 抽样估计 95% 置信区间并做 t-test，提供了充分的统计显著性证据

## 局限性

1. **需要全量训练参考模型**：需要先在 $D_{\text{full}}$ 上训练一个参考模型 $\pi_{\text{ref}}$，增加了前期计算成本
2. **评估范围有限**：主要在 Alpaca/LIMA 等英文指令跟随数据集上验证，对多语言、长文本等场景未探索
3. **DavIR-DPO 实验较少**：仅在 Zephyr 一个模型上验证，优势的稳健性有待更多实验确认
4. **假设依赖**：浅层对齐假设不一定在所有场景下成立（如需要深层知识获取的领域）
5. **未探索迭代 DavIR**：理论上可以迭代执行（选择→训练→更新参考→再选择），但论文未实验

## 相关工作

- **核心集选择**：RHO (Mindermann et al. 2022)、CRAIG、DataInf、EL2N、Forgetting Score 等
- **LLM 后训练数据选择**：LIMA (Zhou et al. 2023) 人工标注、AlpaGasus (Chen et al. 2023) GPT 评分、Instruction Mining 验证集损失
- **DPO 变体**：IPO (Azar et al. 2023)、EXO (Ji et al. 2024)、AOT (Melnyk et al. 2024)、SPPO (Wu et al. 2024) 等
- **预训练数据选择**：DoReMi (Xie et al. 2023)、DSIR (Xie et al. 2023)、DRO (Oren et al. 2019)

## 评分

⭐⭐⭐⭐⭐ (5/5)

- **创新性**：⭐⭐⭐⭐⭐ 归一化改变虽简单但洞察深刻，RHO→DavIR→DavIR-DPO 的理论链条优雅
- **实验充分性**：⭐⭐⭐⭐ 多模型家族、多数据集、多基线对比，统计检验完善
- **写作质量**：⭐⭐⭐⭐ 问题提出清晰，但符号较多且行文稍显冗长
- **实用性**：⭐⭐⭐⭐⭐ 计算代价低、效果显著，直接可用于任何 LLM 后训练数据选择场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](large_vocabulary_size_improves_large_language_models.md)
- [\[ACL 2025\] Retrofitting Large Language Models with Dynamic Tokenization](retrofitting_large_language_models_with_dynamic_tokenization.md)
- [\[ACL 2025\] SCAR: Data Selection via Style Consistency-Aware Response Ranking for Efficient Instruction-Tuning](scar_style_consistency_data_selection.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->
