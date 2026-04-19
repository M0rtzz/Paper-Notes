---
title: >-
  [论文解读] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal
description: >-
  [ACL 2025][LLM/NLP][token 粒度] 系统研究子词 token 粒度（词表大小 256~128K）对 LM surprisal 预测人类阅读时间能力的影响，发现 ~8K 词表的中等粒度在自然阅读时间预测上最优（甚至优于 GPT-2），而粗粒度 token 在花园路径句法效应上更敏感，揭示认知建模的最优分词粒度并非 NLP 通用标准。
tags:
  - ACL 2025
  - LLM/NLP
  - token 粒度
  - 子词分词
  - 惊奇度
  - 认知建模
  - 阅读时间
---

# The Impact of Token Granularity on the Predictive Power of Language Model Surprisal

**会议**: ACL 2025  
**arXiv**: [2412.11940](https://arxiv.org/abs/2412.11940)  
**代码**: [GitHub](https://github.com/byungdoh/ssm-surprisal) (有)  
**领域**: 认知建模 / 计算心理语言学  
**关键词**: token 粒度, 子词分词, 惊奇度, 认知建模, 阅读时间

## 一句话总结

系统研究子词 token 粒度（词表大小 256~128K）对 LM surprisal 预测人类阅读时间能力的影响，发现 ~8K 词表的中等粒度在自然阅读时间预测上最优（甚至优于 GPT-2），而粗粒度 token 在花园路径句法效应上更敏感，揭示认知建模的最优分词粒度并非 NLP 通用标准。

## 研究背景与动机

**语言模型的 surprisal 是认知建模的核心工具。** 在 Hale (2001) 和 Levy (2008) 的理论框架下，词级 surprisal（$-\log P(w_i | w_{<i})$）被广泛用作预测人类逐词处理难度的指标。近年来，Transformer 等神经语言模型被用于计算 surprisal 并与人类阅读时间进行拟合，为理解预测性语言处理提供了计算模型。

**影响 surprisal 质量的因素已有研究，但 token 粒度这个基础变量被忽视。** 模型架构、训练数据对 surprisal 质量的影响已被研究（Oh and Schuler, 2023; Shain et al., 2024），但子词分词的粒度——即词表大小——对认知建模能力的影响从未被系统探索过。

**Token 粒度通过两条路径影响 surprisal 质量。** 第一，**初始偏置路径**：细粒度分词（小词表）将低频长词拆成多个 token，隐式编码了词长和词频信息——例如"journey"被拆成 7 个 token 后，均匀分布下它的概率比"to"低 6 个量级。粗粒度分词（大词表）保持大多数词完整，初始概率更均匀。词长和词频恰好是影响人类阅读处理的关键变量（Barton et al., 2014; Just and Carpenter, 1980），因此某些分词方案天然更适合预测阅读时间。第二，**表示质量路径**：粗粒度 token 学到更接近词级共现统计的表示（类似 Word2Vec），而细粒度 token 将一个词分散到多个向量，增加了学习词间关联的难度。

## 方法详解

### 整体框架

(1) 用 Unigram LM (ULM) tokenizer 训练 11 种词表大小（256~128K）的分词器 → (2) 基于 Mamba-2 架构训练 3 种规模的语言模型 → (3) 在 5 个阅读时间语料库（10 种指标）上评估 surprisal 预测力 → (4) 在花园路径句法构造上评估句法敏感性。

### 关键设计

1. **11 级 token 粒度的系统化控制**:
    - 功能：训练词表大小 $|V| \in \{256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 128K\}$ 的 ULM 分词器
    - 核心思路：ULM 分词器以字符为基本单元（而非 BPE 的 bytes），通过最大化子词序列的联合概率迭代剪枝词表，在 100 万 Wiki-40B 文章上训练。$|V|=256$ 接近字符级（"journey" → 7 token），$|V|=128K$ 接近词级（"journey" → 1 token）
    - 设计动机：选用 ULM（而非 BPE）因为字符比 byte 更可解释；覆盖从 256 到 128K 的极宽范围以全面映射粒度-质量关系

2. **Mamba-2 架构解决序列长度不可比问题**:
    - 功能：用状态空间模型（SSM）替代 Transformer 训练 LM
    - 核心思路：不同粒度导致同一文本的 token 序列长度差异巨大（$|V|=256$ 下序列长度可达 $|V|=128K$ 的数倍）。Transformer 的 self-attention 有 $O(n^2)$ 空间复杂度，对长序列不友好。Mamba-2 基于 SSM 的线性复杂度天然适合处理变长序列。训练 Small/Medium/Large 三种规模：6/12/24 层，256/512/768 维嵌入，参数量 2.6M/19.8M/88M（不含嵌入层）
    - 设计动机：若用 Transformer 并设固定最大长度，不同粒度的 LM 会条件于不同量的上下文，破坏实验公平性

3. **Whitespace 概率修正**:
    - 功能：修正从 subword token 概率推导词概率时的归一化问题
    - 核心思路：ULM 分词器在 token 前添加空格前缀，朴素计算词概率时所有词概率之和可能超过 1（因为未标记词的结束位置）。应用 Oh and Schuler (2024) 的修正方法，将空格概率重新分配给前一个 token
    - 设计动机：确保词级 surprisal 在概率论上正确

### 损失函数 / 训练策略

标准 causal LM 目标（next-token prediction）。在 Wiki-40B 英文完整训练集上训练一个 epoch（5,152,219 个训练样本，10,063 个 batch × 512 样本），使用 AdamW 优化器（最大学习率 $10^{-3}$，cosine 退火至 $10^{-5}$），梯度裁剪 norm=1，半精度训练（48GB RTX 8000 GPU）。

## 实验关键数据

### 主实验——自然阅读时间预测（ΔLogLik 越高越好）

| 阶段 | 最优词表 | ΔLogLik | 最差词表 | ΔLogLik | GPT-2 Small |
|------|---------|---------|---------|---------|-------------|
| 训练前（纯分词器） | $|V|=4K$ | 2553 | $|V|=128K$ | 1899 | — |
| 训练后（Small 平均） | $|V|=8K$ | 最高 | $|V|=128K$ | 最低 | 参考线 |
| 训练后（Large 平均） | 差异缩小 | — | — | — | — |
| 训练后（三尺度平均） | **$|V|=8K$** | **最优** | $|V|=128K$ | 最低 | 被 8K 超越 |

### 花园路径实验（GPE 越大=越敏感）

| 句法构造 | 粗粒度（大词表）趋势 | 细粒度（小词表）趋势 | 说明 |
|---------|-------------------|-------------------|------|
| MV/RR（主动词/关系从句） | GPE 更大（~6ms, Small） | GPE 更小（~2ms, Small） | 粗粒度优势明显 |
| NP/S（名词短语/句子补语） | 趋势类似但弱 | — | 构造间差异 |
| NP/Z（名词短语/不及物） | 趋势类似但弱 | — | 构造间差异 |
| 人类实际效应 | — | — | 所有 LM 低估 1-2 个量级 |

### 关键发现

- **纯分词器（无训练）的 surprisal 已能预测阅读时间**：$|V|=4K$ 时 ΔLogLik≈2553，仅靠分词器编码的词长/词频信息即提供显著预测力
- **训练后 ~8K 词表综合最优**：在 5 个语料库×10 种阅读指标上平均，甚至超越 GPT-2 Small（$|V|≈50K$）
- **模型大小与粒度存在交互**：大模型（88M 参数）能部分克服初始偏置，不同粒度间差异缩小
- **最优粒度因任务而异**：自然阅读偏好中等粒度（8K），花园路径偏好粗粒度（大词表）——因为词级共现统计更有利于学习句法关系
- **困惑度与认知建模质量不完全对应**：大词表困惑度更低但 ΔLogLik 不一定更高

## 亮点与洞察

- **揭示被忽视的关键变量**：分词粒度不仅影响 NLP 性能，更深刻影响模型作为认知模型的质量——此前从未被系统研究
- **实验设计的全面性**：11 粒度 × 3 模型规模 × 5 语料库 × 10 阅读指标 + 花园路径实验，覆盖面极广
- **纯分词器即可预测阅读时间**：这一发现说明人类阅读处理对词长/词频极其敏感，分词器隐式地编码了这些信息
- **Mamba-2 架构选择**：巧妙解决了不同粒度下序列长度不可比的实验设计难题
- **不同任务需要不同最优粒度**：自然阅读 vs 花园路径的分离结果对认知建模实践有直接指导意义

## 局限与展望

- 仅在英语数据和英语母语者上验证，跨语言泛化性未知（如中文分词粒度的影响可能完全不同）
- 模型规模有限（最大 88M 参数），更大模型可能完全克服初始偏置使粒度不再重要
- 仅关注认知建模场景，不涉及 token 粒度对 NLP 应用性能的影响
- 花园路径仅覆盖 3 种句法构造（MV/RR、NP/S、NP/Z），构造覆盖面有限
- 所有模型仍大幅低估人类花园路径效应（差 1-2 个量级），分词粒度无法弥补这一根本性差距

## 相关工作与启发

- **vs Nair and Resnik (2023)**：比较形态学分词 vs BPE 对 surprisal 的影响；本文更系统——11 种粒度的连续变化
- **vs Giulianelli et al. (2024)**：从 GPT-2 token 概率推导字符级概率；本文直接从源头改变分词粒度
- **vs Oh and Schuler (2023b) / Shain et al. (2024)**：研究模型大小对 surprisal 质量的影响；本文揭示粒度是另一个被忽视的关键变量
- **启发**：认知建模领域未来应将分词粒度作为标准报告变量，而非默认使用 GPT-2/Llama 的 ~50K 词表

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统化研究被忽视的重要变量，实验设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 11粒度×3规模×5语料库×10指标+花园路径，覆盖极全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，两个实验互补，结论明确
- 价值: ⭐⭐⭐⭐ 对认知建模实践有直接指导，揭示的分词-认知关联有基础研究意义

<!-- RELATED:START -->

## 相关论文

- [Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [Enhancing Character-Level Understanding in LLMs through Token Internal Structure Learning](character_level_understanding.md)
- [Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [Representation Bending for Large Language Model Safety](repbend_representation_bending_safety.md)

<!-- RELATED:END -->
