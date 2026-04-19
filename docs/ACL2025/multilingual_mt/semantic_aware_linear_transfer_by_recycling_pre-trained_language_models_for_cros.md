---
title: >-
  [论文解读] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer
description: >-
  [ACL 2025][cross-lingual transfer] 提出 SALT（Semantic Aware Linear Transfer），通过为每个非共享词表 token 基于语义相似的共享 token 对构建独立的最小二乘变换矩阵，将目标语言 PLM 的丰富嵌入表示迁移到英语中心 LLM 的嵌入空间，在下游任务、持续预训练收敛速度和跨语言理解上均优于现有方法。
tags:
  - ACL 2025
  - cross-lingual transfer
  - embedding initialization
  - vocabulary replacement
  - PLM recycling
  - linear least squares
---

# Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer

**会议**: ACL 2025  
**arXiv**: [2505.10945](https://arxiv.org/abs/2505.10945)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: cross-lingual transfer, embedding initialization, vocabulary replacement, PLM recycling, linear least squares

## 一句话总结

提出 SALT（Semantic Aware Linear Transfer），通过为每个非共享词表 token 基于语义相似的共享 token 对构建独立的最小二乘变换矩阵，将目标语言 PLM 的丰富嵌入表示迁移到英语中心 LLM 的嵌入空间，在下游任务、持续预训练收敛速度和跨语言理解上均优于现有方法。

## 研究背景与动机

**领域现状**: 当前主流 LLM（如 Gemma、XGLM）虽包含多语言词表，但训练数据以英语为主，在目标语言上的实用性有限；其嵌入层中大量与目标语言无关的 token 浪费了参数空间。

**现有痛点**: 现有跨语言迁移方法（如 FOCUS、OFA）通过源模型嵌入的加权平均来初始化新词表嵌入，但源模型嵌入本身对目标语言的学习不充分，仅靠其浅层表示难以捕获目标语言的深层语义。

**核心矛盾**: 词表替换后需要高质量的嵌入初始化，但源模型和外部静态词向量（fastText）提供的目标语言表示都不够丰富，导致与模型上游层的知识对齐困难。

**本文解决**: 引入目标语言专用的 PLM（如韩语 GPT、阿拉伯语 BERT 等小规模预训练模型）作为嵌入源，其在目标语言上的预训练使嵌入包含更丰富的语义信息。

**切入角度**: "回收"（recycling）已有的目标语言 PLM——这些模型虽然参数规模小，但经过充分的目标语言预训练，嵌入层质量高于多语言 LLM 的目标语言嵌入。

**核心 idea**: 为每个非共享 token 独立拟合一条从 PLM 嵌入空间到 LLM 嵌入空间的线性回归线（least squares），基于语义最相似的共享 token 对确定映射关系，这种 token 级别的个性化迁移保留了 PLM 嵌入的语义丰富性。

## 方法详解

### 整体框架

SALT 分五步完成嵌入迁移：(1) 用 fastText 提取辅助子词嵌入；(2) 识别源词表 $V_s$ 和目标词表 $V_t$ 的共享词表 $V_{\text{shared}}$，并计算非共享 token 与共享 token 之间的语义相似度；(3) 用 Sparsemax 动态选取 top-k 最近邻共享 token；(4) 对每个非共享 token 用最小二乘法拟合独立变换矩阵 $X_{t_i}$ 并完成嵌入映射；(5) 在目标语言语料上进行语言自适应持续预训练以对齐嵌入层与上层权重。

### 关键设计

1. **语义感知的 token 级线性变换**: 核心公式为 $\arg\min_{X \in \mathbb{R}^{h_t \times h_s}} \|E'_{t_i} X_{t_i} - E'_{s_i}\|$，其中 $E'_{t_i}$ 和 $E'_{s_i}$ 分别是为非共享 token $v_{t_i}$ 选出的 top-k 最近邻共享 token 在 PLM 和 LLM 中的嵌入堆叠矩阵。闭式解为 $X_{t_i} = E'^{+}_{t_i} \cdot E'_{s_i}$（伪逆）。每个目标 token 有独立的变换矩阵，这是与全局线性映射方法的关键区别。
2. **Sparsemax 动态 k 值选择**: 对每个非共享 token，先用 fastText 计算其与所有共享 token 的余弦相似度集合 $C_{t_i}$，然后对 $C_{t_i}$ 施加 Sparsemax（将不相关 token 权重置零），动态决定每个 token 的近邻数量 k，避免引入噪声 token。
3. **PLM 嵌入作为迁移源**: 与 FOCUS/OFA 仅依赖源 LLM 嵌入不同，SALT 使用目标语言专用 PLM 的嵌入层，该嵌入在目标语言预训练中已学到丰富的语义表示。共享 token 直接复制源 LLM 嵌入，非共享 token 从 PLM 嵌入空间映射过来。

### 工程细节

- 源模型选用 Gemma-2b（256K 词表）和 XGLM-1.7b（256K 词表）
- 目标 PLM 涵盖 BERT、GPT、T5 三种架构，验证方法的架构通用性
- fastText 提供跨词表的辅助相似度计算，对不在 fastText 中的罕见 token 用源嵌入的正态分布初始化
- 持续预训练使用各语言 Wikipedia 语料（8M 句），CLM 目标

## 实验关键数据

### 知识基准对齐（Gemma 源模型，初始化后直接评估）

| 方法 | 德语 Avg | 阿拉伯语 Avg | 越南语 Avg |
|---|---|---|---|
| Multivariate | 31.02 | 29.03 | 33.21 |
| FOCUS | 29.82 | 28.17 | 32.76 |
| OFA | 30.70 | 29.47 | 32.52 |
| **SALT (Ours)** | **32.71** | **30.64** | **34.66** |

SALT 在所有语言和源模型上均取得最佳均分。德语 HellaSwag 达 42.58，比第二名高 3+ 分，显示对目标语言逻辑流的捕获能力更强。

### 目标语言阅读理解（Gemma 持续预训练后，EM/F1）

| 方法 | 德语 MLQA | 阿拉伯语 MLQA | 越南语 MLQA | 德语 XQuAD | 越南语 XQuAD |
|---|---|---|---|---|---|
| Multivariate | 19.70/31.68 | 10.35/25.37 | 23.44/42.16 | 22.23/34.62 | 31.01/51.74 |
| FOCUS | 25.66/37.97 | 12.63/25.63 | 27.17/44.77 | 27.21/39.00 | 35.71/54.65 |
| OFA | 23.22/37.05 | 16.49/33.52 | 25.06/45.14 | 26.52/38.99 | 35.88/55.07 |
| **SALT** | **28.07/41.47** | **18.33/36.36** | **29.70/49.63** | **31.28/44.41** | **36.30/56.90** |

### 跨语言理解（MLQA 英语-目标语言混合设置，Gemma）

| 方法 | 英语→目标 Avg EM/F1 | 目标→英语 Avg EM/F1 |
|---|---|---|
| FOCUS | 23.11/32.52 | 21.92/34.70 |
| OFA | 24.66/35.99 | 20.05/33.93 |
| **SALT** | **29.22/41.40** | **24.29/38.62** |

SALT 在跨语言 F1 上比第二名 OFA 高约 13%，说明 PLM 嵌入中英语和目标语言的共享空间有效保留。

### 关键发现

- 持续预训练收敛曲线（图 2）显示 SALT 在所有场景中收敛最快且最终 loss 最低
- FOCUS/OFA 有时甚至不如简单的多元高斯初始化（Multivariate），说明仅依赖源嵌入的加权平均可能有害
- SALT 在英语能力保留上也优于其他方法——尽管仅在目标语言上持续预训练
- 不同架构 PLM（BERT/GPT/T5）均可作为目标模型，性能差异较小，T5 在某些语言上甚至优于 GPT

## 亮点与洞察

- **PLM 回收理念**: 在 LLM 时代重新利用小规模 PLM 的嵌入——这些"过时"模型在目标语言上有不可替代的表示优势
- **token 级个性化映射**: 为每个 token 独立拟合变换矩阵，比全局映射更精细，计算成本可控（最小二乘有闭式解）
- **跨语言优势显著**: 在英语-目标语言混合场景中提升最大，说明 PLM 嵌入的双语共享空间被有效利用
- **架构无关**: BERT、GPT、T5 的嵌入均可用作迁移源，提供了极大的灵活性

## 局限性

- 依赖目标语言 PLM 的存在：对于极低资源语言（无专用 PLM），SALT 无法直接应用
- 仅在 2B 级别模型上验证，未扩展至 7B+ 的 LLM，大模型上的效果和效率待确认
- 未评估指令微调后的效果，仅关注基础模型的持续预训练阶段
- 仅测试了三种语言（德语、阿拉伯语、越南语），虽语系跨度大但覆盖有限

## 相关工作对比

### vs FOCUS (Dobler & De Melo, 2023)
FOCUS 通过 fastText 指导的加权平均从源模型嵌入构建目标嵌入，本质上仍受限于源模型对目标语言的学习程度。SALT 引入目标语言 PLM 嵌入作为更丰富的信息源，且用独立线性回归替代全局加权平均。实验中 FOCUS 在多个场景下甚至不如随机初始化，而 SALT 始终最优。

### vs OFA (Liu et al., 2024)
OFA 采用类似 FOCUS 的策略并引入 SVD 分解降维，但核心仍是源模型嵌入的加权平均。SALT 从根本上改变了嵌入来源（PLM 而非源 LLM），跨语言理解上 SALT 以约 13% F1 的优势大幅领先 OFA。

### vs 词表扩展方法 (Cui et al., 2023; Zhao et al., 2024)
词表扩展保留源词表并添加目标 token，适用但增加模型大小。SALT 采用词表替换策略，用目标语言词表完全替换源词表，模型大小不变甚至可能减小，更适合资源受限场景。

## 评分

- **新颖性**: ⭐⭐⭐⭐ PLM 嵌入回收+token 级语义感知线性映射，视角新颖且方法优雅
- **实验充分度**: ⭐⭐⭐⭐ 3 种语言×2 源模型×3 PLM 架构×多种任务，收敛曲线和跨语言分析全面
- **写作质量**: ⭐⭐⭐⭐ 五步法描述清晰，图 1 总结准确，公式推导完整
- **价值**: ⭐⭐⭐⭐ 为 LLM 时代 PLM 的再利用提供了有效路径，跨语言迁移场景实用性强

<!-- RELATED:START -->

## 相关论文

- [Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)

<!-- RELATED:END -->
