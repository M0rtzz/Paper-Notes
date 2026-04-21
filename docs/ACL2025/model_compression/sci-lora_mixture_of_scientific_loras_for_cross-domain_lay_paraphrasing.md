---
title: >-
  [论文解读] Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing
description: >-
  [ACL 2025][模型压缩] 提出 Sci-LoRA——一种混合多领域 LoRA 的框架，通过对比学习训练文本编码器+动态权重生成器+LoRA 融合模块，在无需领域标签的情况下实现跨12个学科领域的科学文本通俗化改写，在5个数据集10个指标上超越 SOTA。
tags:
  - ACL 2025
  - 模型压缩
---

# Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing

**会议**: ACL 2025  
**arXiv**: [2505.18867](https://arxiv.org/abs/2505.18867)  
**代码**: 无  
**领域**: 模型压缩  

## 一句话总结

提出 Sci-LoRA——一种混合多领域 LoRA 的框架，通过对比学习训练文本编码器+动态权重生成器+LoRA 融合模块，在无需领域标签的情况下实现跨12个学科领域的科学文本通俗化改写，在5个数据集10个指标上超越 SOTA。

## 背景与动机

1. **跨学科研究日益普遍**：现代科学研究越来越多地涉及跨学科内容（如计算机+生物、化学+AI），需要面向非专业读者的通俗化改写系统能处理多领域混合的技术文本。
2. **现有方法局限于单一领域**：已有工作（如生物医学领域的 lay summarization）只针对单一领域微调模型，忽视跨领域泛化能力，面对跨学科内容可能产生误解。
3. **单一 LoRA 存在跨域干扰**：在所有领域数据上训练一个 LoRA 会导致跨域知识互相干扰，难以兼顾领域特异性和通用性。
4. **新领域适应需要重新全量训练**：当新的交叉学科领域出现时，现有模型需要全量重训，效率低下且成本高昂。
5. **静态 LoRA 合并权重不灵活**：现有的 Mixture of LoRAs 方法多采用静态等权合并或简单路由，无法根据输入文本的领域特性动态调整各 LoRA 的贡献。
6. **领域边界模糊**：跨学科文本可能同时涉及多个领域，推理时通常没有显式的领域标签可用，需要模型自动判断输入的领域归属并加权融合。

## 方法详解

### 整体框架

Sci-LoRA 包含三个核心模块：**领域 LoRA 训练**、**适配器权重生成器**、**动态 LoRA 融合**。

### 1. 领域 LoRA 训练

- 基座模型：**Qwen2.5-7B-Instruct**（Apache-2.0 开源，长文本生成性能好）
- 为 **12 个领域**各训练一个 LoRA 适配器，每个 LoRA 只在对应领域数据上训练
- LoRA 参数：学习率 1e-4，batch size 4，rank 8，最大文档长度 2048
- 使用 LLaMA-Factory 进行高效微调，早停策略选模

### 2. 适配器权重生成器（AWG）

分两步实现无需领域标签的动态权重分配：

**文本编码器（对比学习微调）**：
- 基于 Sentence-BERT，使用跨领域子集数据做对比学习微调
- 正样本对：同领域的不同文本 $(x_i, x_j)$
- 负样本对：不同领域的文本 $(x_i, x_k)$
- InfoNCE 对比损失：$\mathcal{L} = \frac{e^{-\|x_i - x_j\|^2 / \tau}}{e^{-\|x_i - x_j\|^2 / \tau} + \sum_{k=1}^{m} e^{-\|x_i - x_k\|^2 / \tau}}$
- 目的：让编码器能更好地区分不同领域的文本表示

**权重生成**：
- 对每个领域的训练数据嵌入做 **K-Means 聚类**（K=10），选取最近质心的数据点取平均作为领域适配器表示 $r_{\triangle\theta_i}$
- 推理时根据输入文本嵌入与各领域表示的距离计算权重：$\alpha_i = \frac{1}{1 + \|E(x_i) - r_{\triangle\theta_i}\|_2}$
- 距离越近权重越大，自动实现领域相关性加权

### 3. 动态 LoRA 融合

同时维护两路生成表示并融合：

- **专业化表示**：将所有领域 LoRA 按动态权重 $\alpha$ 加权合并后注入基座模型：$r_{specialized} = \mathcal{M}(\theta + \sum_{i=1}^n \alpha_i \triangle\theta_i, x_i)$
- **通用化表示**：额外训练一个在所有领域数据上的统一 LoRA $\triangle\theta_0$：$r_{generalized} = \mathcal{M}(\theta + \triangle\theta_0, x_i)$
- **最终融合**：$\hat{r} = \beta \cdot r_{specialized} + (1-\beta) \cdot r_{generalized}$，$\beta=0.5$ 平衡领域特异性和跨域泛化

## 实验结果

### 表1：主要结果（d-BLEU % / BERTScore F1 %，部分领域展示）

| 模型 | CELLS d-BLEU | PLOS d-BLEU | ALS d-BLEU | AAD d-BLEU | CELLS BERT | PLOS BERT | ALS BERT | AAD BERT |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 5.10 | 5.76 | 11.05 | 12.40 | 81.13 | 81.81 | 83.55 | 81.80 |
| Qwen2.5 | 9.26 | 10.18 | 25.55 | 28.71 | 82.36 | 82.70 | 84.98 | 82.55 |
| DSPT5 | - | - | 24.95 | 33.53 | - | - | 85.48 | 83.70 |
| **Sci-LoRA** | **11.15** | **12.43** | **31.03** | **38.97** | **83.00** | **83.35** | **86.01** | **84.37** |

Sci-LoRA 在几乎所有领域和指标上取得最佳，d-BLEU 较 Qwen2.5 LoRA 基线平均提升 5-10 个百分点，且超越了为每个领域单独全量微调的 DSPT5。

### 表2：消融实验（AAD 领域，VTechAGP 数据集）

| 配置 | s-BLEU | d-BLEU | BERTScore | ROUGE1 | METEOR | SARI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Pre-trained | 5.63 | 12.69 | 81.97 | 42.46 | 33.98 | 34.64 |
| Single LoRA（全域） | 13.43 | 28.71 | 82.55 | 46.51 | 39.85 | 39.39 |
| Multi-LoRAs（直接多域） | 19.78 | 34.39 | 83.30 | 47.83 | 41.77 | 40.76 |
| AWG + K-Means | 23.17 | 37.08 | 83.72 | 51.13 | 43.91 | 43.56 |
| AWG + Contrastive | 18.06 | 38.62 | 84.03 | 52.07 | 44.89 | 44.00 |
| w/o Fusion | 14.86 | 31.22 | 81.59 | 50.82 | 41.42 | 43.63 |
| **Sci-LoRA（完整）** | **18.38** | **38.97** | **84.37** | **52.69** | **46.71** | **44.26** |

每个组件均有贡献：多域 LoRA > 单 LoRA；K-Means 领域表示 > 随机采样；对比学习编码器进一步提升；动态融合模块不可或缺。

### 人类评估

| 模型 | 全面性 | 通俗度 | 语义保真 | 简洁性 | 流畅度 |
|:---|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 3.25 | 2.68 | 3.08 | **3.53** | 3.15 |
| Qwen2.5 | 3.78 | 2.86 | 2.80 | 2.77 | 3.31 |
| **Sci-LoRA** | **3.82** | **2.88** | **3.45** | 3.47 | **3.40** |

Sci-LoRA 在全面性、语义保真和流畅度上最优，同时保持良好的简洁性。

## 亮点

- **无需领域标签的动态 LoRA 加权**：通过对比学习+K-Means 聚类实现自动领域识别和权重分配，推理时完全不依赖领域信息
- **双路融合策略设计巧妙**：专业化（加权多 LoRA）+ 通用化（全域 LoRA）互补，既保留领域知识又避免过拟合
- **跨 12 个领域 5 个数据集的全面评估**：10 个自动指标 + 5 维人类评估，实验规模和评估维度充分
- **可扩展架构**：新领域只需追加一个 LoRA 适配器，无需重训整个系统

## 局限性

- **领域数量扩展受限**：当前依赖 PEFT 库在推理时加载合并所有 LoRA，领域数百级时推理延迟将显著增长
- **无法处理无训练数据的新领域**：每个领域 LoRA 需要领域数据微调，不支持零样本新领域适配
- **仅在 Qwen2.5-7B 上验证**：未探索不同基座模型（如 LLaMA、Mistral）的效果差异，泛化性存疑
- **β=0.5 固定融合比例**：专业化与通用化的融合权重固定为 0.5，未根据输入自适应调整，可能非最优

## 相关工作对比

| 维度 | Sci-LoRA | DSPT5 (Cheng et al., 2025) | 单 LoRA 微调 |
|:---|:---|:---|:---|
| 训练方式 | 每域一个 LoRA + 全域一个 LoRA | 每域全量微调一个模型 | 全域数据训一个 LoRA |
| 推理需求 | 无需领域标签 | **需要**领域标签选模型 | 无需领域标签 |
| 领域扩展 | 追加 LoRA 适配器 | 需训练全新模型 | 需重训 LoRA |
| 跨域泛化 | 动态加权融合 | 无跨域能力 | 跨域干扰 |
| 性能 | 全指标最优 | 次优 | 低于 Sci-LoRA 约 5-10pp |

| 维度 | Sci-LoRA | Mixture of LoRAs (Router-based) | Mixture of LoRAs (Linear Merge) |
|:---|:---|:---|:---|
| 权重生成 | 对比学习编码器 + K-Means | 额外路由网络 | 静态等权 |
| 领域表示 | 聚类质心嵌入 | 无显式表示 | 无 |
| 融合策略 | 双路（专业+通用）动态融合 | 单路 | 单路 |
| 任务适用 | 跨域通俗化改写 | 通用多任务 | 通用多任务 |

## 评分

- ⭐⭐⭐⭐ 新颖性：对比学习驱动的动态 LoRA 权重生成 + 双路融合策略在 lay paraphrasing 任务上是新组合
- ⭐⭐⭐⭐ 实用性：跨 12 领域验证，提供可扩展的多领域适配方案，对科普写作有直接应用价值
- ⭐⭐⭐⭐ 实验充分度：5 数据集 12 领域 10 指标 + 人类评估 + 详尽消融，评估维度全面
- ⭐⭐⭐⭐ 写作质量：方法描述清晰，公式完整，消融实验和可视化（t-SNE）辅助理解

<!-- RELATED:START -->

## 相关论文

- [Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)

<!-- RELATED:END -->
