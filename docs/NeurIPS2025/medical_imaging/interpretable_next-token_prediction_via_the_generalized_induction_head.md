---
title: >-
  [论文解读] Interpretable Next-token Prediction via the Generalized Induction Head
description: >-
  [NeurIPS 2025][医学图像][可解释性] 提出 Induction-Gram (GIM)，一种结合精确n-gram匹配与模糊匹配的可解释语言模型，通过构建"广义归纳头"在输入上下文中检索相似序列进行下一token预测，比可解释基线提升最高25%p准确率，并在fMRI脑响应预测中提升20%。
tags:
  - NeurIPS 2025
  - 医学图像
  - 可解释性
  - 下一token预测
  - 归纳头
  - n-gram模型
  - fMRI
---

# Interpretable Next-token Prediction via the Generalized Induction Head

**会议**: NeurIPS 2025

**arXiv**: [2411.00066](https://arxiv.org/abs/2411.00066)

**代码**: [有](https://github.com/ejkim47/generalized-induction-head)

**领域**: 可解释AI / 语言建模

**关键词**: 可解释性, 下一token预测, 归纳头, n-gram模型, fMRI

## 一句话总结

提出 Induction-Gram (GIM)，一种结合精确n-gram匹配与模糊匹配的可解释语言模型，通过构建"广义归纳头"在输入上下文中检索相似序列进行下一token预测，比可解释基线提升最高25%p准确率，并在fMRI脑响应预测中提升20%。

## 研究背景与动机

大型Transformer模型在预测性能上表现出色，但其黑盒特性限制了在高风险领域（如科学、医学、政策制定）中的应用。同时，LLM的巨大规模带来了高昂的能源成本和部署困难。

传统n-gram模型可以保持完全的可解释性且计算高效，但与黑盒LLM之间存在显著的性能差距。现有的Infini-Gram模型虽然可以扩展到数万亿token的参考语料库，但在面对分布偏移时（参考语料与输入上下文存在差异）难以找到长的精确匹配，性能欠佳。

本文受到LLM中"归纳头"（induction head）机制的启发——归纳头通过识别上下文中的相似模式并复制后续token来实现上下文学习。作者尝试以可解释的方式重建这一机制，弥合n-gram模型与LLM之间的差距。

## 方法详解

### 整体框架

Induction-Gram 由三个组件组成：

1. **Infini-Gram**：基于参考语料库的精确n-gram匹配
2. **Induction-only (exact)**：在输入上下文中进行精确n-gram匹配
3. **Induction-only (fuzzy)**：在输入上下文中进行模糊匹配

最终预测按以下优先级组合（Eq.5）：
- 若参考语料库的effective n 大于输入上下文的effective n 且大于阈值τ，使用 Infini-Gram
- 若输入上下文的effective n 更大且大于阈值τ，使用精确归纳匹配
- 否则，使用模糊归纳匹配

### 关键设计

**1. 模糊匹配的相似度定义**

两个序列 $x_1$ 和 $x_2$ 的相似度基于它们是否导致相似的下一token分布，使用 Jensen-Shannon 散度衡量：

$$s(x_1, x_2) = \exp(-\text{JSD}(P_{\text{next}}(x_1), P_{\text{next}}(x_2)))$$

**2. Fuzzy Matching Model（模糊匹配模型）**

为了高效计算相似度，训练一个小型Transformer模型（3-4层），通过知识蒸馏从LLM学习。该模型输出特征嵌入，用余弦相似度近似原始相似度：

$$s_{\text{FM}}(x_1, x_2) = \exp(-(1 - \text{CosineSim}(e_1, e_2))/T)$$

其中 $T=0.1$ 为温度参数。模型使用交叉熵损失和反向KL散度损失联合训练。

**3. 模糊匹配的下一token预测**

使用滑动窗口在输入上下文中搜索与查询末尾相似的序列，将相似度分数作为浮动计数来估计下一token的概率分布：

$$P_{\text{induction(fuzzy)}}(w_i | x) = \frac{c_{\text{fuzzy}}(w_{i-k-1:i-1} w_i | x)}{\sum_{w_j \in \mathcal{V}} c_{\text{fuzzy}}(w_{i-k-1:i-1} w_j | x)}$$

### 损失函数 / 训练策略

- Fuzzy Matching Model 使用 **交叉熵损失 + 反向KL散度损失**（权重均为1.0）
- 使用 LLaMA2-7B 作为教师模型
- AdamW 优化器，学习率 0.0001，权重衰减 0.1
- 余弦学习率调度，1000步预热，训练15000-20000步
- 阈值 $\tau$ 通过BabyLM交叉验证确定：GPT-2 tokenizer为8，LLaMA-2为9

## 实验关键数据

### 主实验

**表1: 下一token预测准确率 (%) — GPT-2 tokenizer**

| 参考语料库 | 模型 | BabyLM-test | FineWeb | Pile-val |
|:---|:---|:---:|:---:|:---:|
| - | Induction-only (exact) | 36.7 | 17.2 | 37.0 |
| - | Induction-only (fuzzy) | 41.1 | 25.2 | 38.7 |
| BabyLM-dev (17.4M) | Infini-Gram | 37.6 | 14.7 | 16.0 |
| BabyLM-dev (17.4M) | **Induction-Gram** | **42.2 (+4.6)** | **25.3 (+10.6)** | **40.0 (+24.0)** |
| OpenWebText (9.04B) | Infini-Gram | 16.7 | 25.5 | 22.7 |
| OpenWebText (9.04B) | **Induction-Gram** | **41.8 (+25.1)** | **27.2 (+1.7)** | **42.7 (+20.0)** |
| Pile-train (383B) | Infini-Gram | 33.5 | 39.3 | 49.2 |
| Pile-train (383B) | **Induction-Gram** | **49.4 (+15.9)** | **38.0 (-1.3)** | **50.3 (+1.1)** |
| Unknown | LLM (GPT-2) | 46.9 | 39.0 | 52.3 |

**表2: 推测解码速度 (A40×1 GPU)**

| Draft Model | Large Model | BabyLM ms/token (↓) | 加速比 (↑) | Pile ms/token (↓) | 加速比 (↑) |
|:---|:---|:---:|:---:|:---:|:---:|
| - | LLaMA2-7B | 30.2 | 1.00× | 30.2 | 1.00× |
| TinyLLaMA-1.1B | LLaMA2-7B | 21.3 | 1.42× | 21.3 | 1.42× |
| **Induction-only (fuzzy)** | LLaMA2-7B | **17.7** | **1.71×** | **20.1** | **1.50×** |
| - | LLaMA2-13B | 52.4 | 1.00× | 52.0 | 1.00× |
| TinyLLaMA-1.1B | LLaMA2-13B | 26.7 | 1.96× | 26.3 | 1.98× |
| **Induction-only (fuzzy)** | LLaMA2-13B | **24.x** | **~2.1×** | **~25** | **~2.0×** |

### 消融实验

**fMRI 脑响应预测结果 (表3)**

| 特征模型 | 全部体素平均相关系数 | Top-10%体素平均相关系数 |
|:---|:---:|:---:|
| Eng1000 (基线) | 0.072 | 0.220 |
| Random matching + Eng1000 | ~0.069 | - |
| Naive n-gram matching + Eng1000 | ~0.068 | - |
| Infini-Gram matching + Eng1000 | - | 0.200 |
| **Induction matching + Eng1000** | **0.087 (+20%)** | **0.265 (+20%)** |
| Black-box LLaMA-2 | - | 0.268 |

### 关键发现

1. **上下文匹配 vs 语料库匹配**：仅使用 1024 token 输入上下文的 Induction-only (exact)，在BabyLM和Pile上就比使用10B token参考语料库的 Infini-Gram 高出 5.5-20%p
2. **模糊匹配的价值**：Induction-only (fuzzy) 比 exact 版本提升 1.5-8.7%p，尤其在 effective n 较低时效果更明显
3. **互补性**：Induction-Gram 在 Pile-train (383B) 参考语料库上仍能带来 15.9%p 提升
4. **推测解码加速**：作为 draft model 可实现最高 2.1× 加速，同时保持可解释性
5. **fMRI预测**：Induction matching 仅比黑盒 LLaMA-2 低 1%（0.265 vs 0.268），但完全可解释

## 亮点与洞察

1. **桥接机制可解释性与工程可解释性**：从 LLM 中发现的归纳头机制出发，用手工工程方式重建，代表了一种从 LLM 中"逆向工程"可解释组件的新范式
2. **输入上下文的分布优势**：实验表明上下文内匹配自然反映了输入查询的分布，比固定的参考语料库匹配更准确
3. **跨领域泛化**：同一框架从语言建模无缝迁移到fMRI脑响应预测，展示了方法的通用性
4. **可解释 + 高效**：模糊匹配模型仅需 3-4 层Transformer，推测解码即使比LLM草稿模型更快

## 局限与展望

1. **短上下文受限**：当输入上下文较短或信息量不足时，归纳头改进有限
2. **推理能力不足**：依赖 n-gram 级别的推理，难以处理需要深层推理的任务（类似 kNN-LM 的局限）
3. **可与 RAG 结合**：通过检索增强生成技术补充相关文档作为上下文，可能缓解短上下文问题
4. **更多 LLM 组件**：可以进一步整合 indirect object identifier、retrieval head 等机制

## 相关工作与启发

- **Infini-Gram** (Liu et al., 2024)：将 n-gram 模型扩展到万亿 token 的参考语料库
- **Induction Head** (Olsson et al., 2022)：在 transformer 中发现的上下文学习关键机制
- **kNN-LM** (Khandelwal et al., 2020)：基于最近邻的语言模型
- **推测解码** (Leviathan et al., 2023)：使用小模型加速大模型推理
- 启发：可解释模型不必牺牲全部性能，从 LLM 内部机制中可以提取可工程化的可解释组件

## 评分

| 维度 | 分数 (1-5) | 说明 |
|:---|:---:|:---|
| 创新性 | 4 | 从 LLM 机制中提取可解释组件的新范式 |
| 技术质量 | 4 | 设计简洁优雅，实验充分 |
| 实验充分度 | 5 | 多数据集、多tokenizer、两大应用场景 |
| 实用性 | 4 | 可直接用于推测解码和fMRI分析 |
| 写作质量 | 4 | 结构清晰，动机表述好 |

<!-- RELATED:START -->

## 相关论文

- [Roll the Dice & Look Before You Leap: Going Beyond the Creative Limits of Next-Token Prediction](../../ICML2025/medical_imaging/roll_the_dice_look_before_you_leap_going_beyond_the_creative_limits_of_next-toke.md)
- [STAMP: Spatial-Temporal Adapter with Multi-Head Pooling](stamp_spatial-temporal_adapter_with_multi-head_pooling.md)
- [MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [Robust or Suggestible? Exploring Non-Clinical Induction in LLM Drug-Safety Decisions](robust_or_suggestible_exploring_non-clinical_induction_in_llm_drug-safety_decisi.md)
- [FireGNN: Neuro-Symbolic Graph Neural Networks with Trainable Fuzzy Rules for Interpretable Medical Image Classification](firegnn_neuro-symbolic_graph_neural_networks_with_trainable_fuzzy_rules_for_inte.md)

<!-- RELATED:END -->
