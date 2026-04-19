---
title: >-
  [论文解读] Limited-Resource Adapters Are Regularizers, Not Linguists
description: >-
  [模型压缩][低资源机器翻译] 本文将 adapter souping（权重平均）与交叉注意力微调结合用于低资源克里奥尔语机器翻译，发现虽然方法带来了显著提升（最高 +8 BLEU），但语言关联性与 adapter 性能无有意义的协变关系——随机初始化的未训练 adapter 表现同样优秀，表明 adapter 在此设定下的作用本质是**参数正则化而非语言信息迁移**。
tags:
  - 模型压缩
  - 低资源机器翻译
  - 适配器
  - 正则化
  - 跨语言迁移
  - 克里奥尔语
---

# Limited-Resource Adapters Are Regularizers, Not Linguists

| 会议 | 领域 | arXiv | 代码 |
|------|------|-------|------|
| ACL2025 | Model Compression / Low-resource NLP | [2505.24525](https://arxiv.org/abs/2505.24525) | — |

**关键词**: 低资源机器翻译, 适配器, 正则化, 跨语言迁移, 克里奥尔语

## 一句话总结

本文将 adapter souping（权重平均）与交叉注意力微调结合用于低资源克里奥尔语机器翻译，发现虽然方法带来了显著提升（最高 +8 BLEU），但语言关联性与 adapter 性能无有意义的协变关系——随机初始化的未训练 adapter 表现同样优秀，表明 adapter 在此设定下的作用本质是**参数正则化而非语言信息迁移**。

## 研究背景与动机

### 低资源语言的翻译挑战

全球大多数语言面临数据稀缺，可靠的机器翻译（MT）仍然遥不可及。跨语言迁移学习是主流策略，但尚未弥合高低资源语言间的翻译质量差距。

### 克里奥尔语的特殊性

克里奥尔语是一类独特的低资源语言：
- 全球数亿说话者（海地语、帕皮亚门托语、桑戈语）
- 从殖民时期的语言接触中诞生，可追溯到欧洲和非洲语言
- 在不同语言维度上与不同语族有关联性
- 语言技术研究不足，但存在明确需求

### Adapter 方法的现有思路

Üstün et al. (2021) 提出的两步方法：
1. 为源语言和目标语言分别训练单语去噪 adapter
2. 冻结所有参数，仅微调解码器的交叉注意力（CA-FT）

Chronopoulou et al. (2023a) 提出 adapter souping——将多个领域 adapter 进行权重空间平均。

本文的创新在于**将源/目标语言 adapter、交叉注意力微调和 souping 三者结合**。

## 方法详解

### 实验设计

**目标语言**：海地语（hat）、帕皮亚门托语（pap）、桑戈语（sag）

**迁移语言选择的五种策略**：

1. **语系迁移**（Phylogeny）：
    - 印欧语系（IE）亲属：法语、西班牙语、葡萄牙语等
    - 尼日尔-刚果语系（NC）亲属：约鲁巴语、沃洛夫语等
2. **克里奥尔语间迁移**：克里奥尔语之间的共性
3. **类型学特征迁移**（lang2vec）：基于语法特征的距离
4. **模型表示迁移**（NLLB representations）：NLLB-200 的语言嵌入相似度
5. **分词均匀性迁移**（Subword Evenness, SuE）：分词长度均匀性

### 控制实验

- **无关语言组**：乌拉尔语系、达罗毗荼语系、CJK（中日韩）
- **随机 adapter**：未经训练的随机初始化 adapter（init）
    - 替换克里奥尔语 adapter
    - 与克里奥尔语 adapter souping

### 实现细节

- **基础模型**：NLLB-200 的 600M 蒸馏版（12层编码器/解码器，16注意力头，1024维）
- **Adapter**：在 MADLAD-400 的 10K 单语数据上训练瓶颈 adapter
- **CA-FT**：使用 NLLB-OPUS 的 10K 平行数据微调解码器交叉注意力
- **评估**：在 FLORES-200 上评估 BLEU 和 chrF

### Adapter Souping 公式

$$\theta_{soup} = \frac{1}{l} \sum_{i=1}^{l} \theta_i$$

与 init adapter souping 时，克里奥尔语 adapter 与 init 的权重比为 1:3（模拟与三个其他 adapter souping 的情况）。

## 实验

### 主实验结果（BLEU, Creole → English）

| 实验条件 | hat→eng | pap→eng | sag→eng |
|----------|---------|---------|---------|
| Base Model (CA-FT) | 33.37 | 38.97 | 10.89 |
| s 和 t Adapters | 32.33 | 40.04 | 11.40 |
| **未训练 s Adapter** | **37.07** | **45.01** | **14.91** |
| IE 迁移 | 36.44 | 46.35 | 12.46 |
| NC 迁移 | 36.06 | 46.69 | 12.29 |
| 克里奥尔语迁移 | 35.25 | 46.23 | 12.76 |
| lang2vec | 36.54 | 47.04 | 13.07 |
| NLLB Vec | 35.80 | 46.91 | 12.80 |
| SuE | 36.36 | 47.03 | 13.12 |
| **未训练 Souping** | **37.42** | **46.34** | **13.41** |
| 乌拉尔语 | 37.06 | 47.00 | 13.58 |
| CJK | 36.41 | 47.17 | 13.33 |
| 达罗毗荼语 | 36.55 | 47.27 | 13.27 |

### 核心发现

**迁移语言选择无关紧要**——两个关键证据：

1. **无关语言 ≈ 有原则选择的语言**：乌拉尔、达罗毗荼、CJK 控制组的表现与基于语系、类型学、模型表示选择的迁移语言不相上下
2. **未训练 adapter ≈ 训练过的 adapter**：随机初始化的 adapter 表现与"有意义"的语言 adapter 相当甚至更好

→ **结论**：adapter 的收益来自正则化效应，而非跨语言信息迁移。

### 加泰罗尼亚语验证实验

为排除克里奥尔语的特殊性，在加泰罗尼亚语（有明确的近亲：西、葡、奥克语）上验证：

| 实验 | 800条数据 | 10K数据 |
|------|----------|---------|
| Base Model (CA-FT) | 45.45 | 45.53 |
| s 和 t Adapters | 38.58 | 41.92 |
| spa+por+oci Souping | 41.87 | 43.74 |
| **未训练 Souping** | **43.97** | **44.75** |

即使对于有明确近亲的语言，所有 adapter 方法都无法超越基线，且随机 adapter 最接近恢复基线分数——进一步支持正则化假说。

### 正则化证据分析

1. **梯度范数和验证损失**：无正则化的 CA-FT 梯度范数和验证损失更高，与过拟合一致
2. **参数方差**：souped adapter 的参数方差显著低于单独预训练的克里奥尔语 adapter（Figure 2），支持正则化效应
3. **人工翻译评估**：海地语母语者对 33 个样本的手动评估显示，未训练 Souping 在语法错误方面优于 IE 迁移

## 亮点与洞察

1. **挑战直觉的核心发现**：跨语言迁移学习中，adapter 的语言信息可能无关紧要——这对 NLP 社区关于语言迁移的理解提出了重要挑战
2. **正则化视角**：将 adapter souping 重新解释为一种添加噪声的正则化方法，与 dropout、噪声注入等经典正则化技术形成呼应
3. **实验设计严谨**：六种有原则的迁移策略 + 三种无关语言控制 + 随机 adapter 基线 + 加泰罗尼亚语交叉验证，多层验证使结论可信
4. **对低资源 MT 的实际启示**：如果 adapter 的作用是正则化，那在实际应用中不需要费力寻找"最优迁移语言"，使用随机 adapter 即可
5. **ethics consideration**：认真讨论了克里奥尔语社区的需求和MT技术的社会影响

## 局限性

1. **只能应用于 NLLB-200 支持的少数克里奥尔语**，难以大规模验证
2. **训练数据主要来自宗教领域**（圣经翻译），不代表一般语言使用
3. **正则化假说难以严格证明**：作者承认数学上的严格证明超出本文范围
4. **样本量小**：人工评估仅 33 个样本/条件，统计力有限
5. **仅使用 600M 蒸馏模型**，更大模型上的行为可能不同

## 相关工作

- **参数高效微调**：LoRA、(IA)3、瓶颈 adapter（Houlsby et al., 2019; Pfeiffer et al., 2020）
- **跨语言迁移**：语系 adapter（Faisal and Anastasopoulos, 2022; Chronopoulou et al., 2023b）、迁移语言选择（Pires et al., 2019; Pelloni et al., 2022）
- **Adapter Souping**：Wortsman et al. (2022) 的权重空间平均用于域适应
- **克里奥尔语 NLP**：Lent et al. (2022a, 2024)、Robinson et al. (2022, 2023) 对克里奥尔语迁移学习的探索

## 评分

⭐⭐⭐⭐（4/5）

本文最大价值在于其反直觉的发现：adapter 的跨语言迁移效果可能纯粹来自正则化而非语言信息。实验设计全面、控制严谨，从多个角度支持核心论点。对低资源 MT 实践有直接指导意义。不足在于规模限制（仅 3 种克里奥尔语、600M 模型）以及正则化假说缺乏严格数学证明。

<!-- RELATED:START -->

## 相关论文

- [Random Initialization of Gated Sparse Adapters (RIGSA)](../../ICML2025/model_compression/random_initialization_of_gated_sparse_adapters.md)
- [Neutral Residues: Revisiting Adapters for Model Extension](../../ICML2025/model_compression/neutral_residues_revisiting_adapters_for_model_extension.md)
- [Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](../../NeurIPS2025/model_compression/a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)

<!-- RELATED:END -->
