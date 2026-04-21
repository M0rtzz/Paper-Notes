---
title: >-
  [论文解读] Who Taught You That? Tracing Teachers in Model Distillation
description: >-
  [模型压缩] 本文提出"教师模型归因"新问题：给定一个蒸馏后的学生模型，能否从候选教师中识别出其训练教师？发现 n-gram 相似度和困惑度不可靠，但词性（PoS）句法模板能提供有效的教师识别信号。
tags:
  - 模型压缩
---

# Who Taught You That? Tracing Teachers in Model Distillation

| 会议 | arXiv | 代码 | 领域 | 关键词 |
|------|-------|------|------|--------|
| ACL 2025 | 2502.06659 | - | 模型压缩 / LLM 安全 | 知识蒸馏, 教师归因, 句法模板, PoS 标签, 模型溯源 |

## 一句话总结

本文提出"教师模型归因"新问题：给定一个蒸馏后的学生模型，能否从候选教师中识别出其训练教师？发现 n-gram 相似度和困惑度不可靠，但词性（PoS）句法模板能提供有效的教师识别信号。

## 研究背景与动机

**研究问题：** 在模型蒸馏（用大模型教小模型）的场景中，能否通过分析学生模型的输出来反推它的教师模型？

**现有问题：** 
- 模型蒸馏已成为用大型专有 LLM 训练高效小模型的主流方法
- 蒸馏可能违反模型提供商的服务条款（如 DeepSeek 是否蒸馏了 ChatGPT 的争议）
- 缺乏有效方法来检测未经授权的蒸馏行为
- 现有的数据溯源方法（如水印）需要在生成时嵌入，无法事后检测

**核心动机：** LLM 提供商需要工具来识别未经授权的蒸馏使用，理解教师对学生的"语言指纹"传递也有助于理解知识蒸馏的机制。

## 方法详解

### 整体框架

系统比较三种教师归因策略：
1. **困惑度方法：** 计算候选教师对学生输出的困惑度，期望真正教师给出更低困惑度
2. **相似度方法：** 测量学生与候选教师输出的文本相似度
3. **句法模板方法：** 基于 PoS（词性标注）序列模式训练分类器识别教师

### 关键设计

**实验设置：**
- **学生模型：** GPT-2（124M）和 OLMo-1B
- **候选教师集合 $\mathcal{M}$：** {Llama3-8B, Llama3-70B, Mistral-7B, Mixtral, Gemma2-9B}，均为开源模型
- **任务：** 摘要（CNN-DailyMail、Rotten Tomatoes、PubMed）、问答（OpenbookQA、CommonsenseQA）、指令遵循（Alpaca 10K）

**PoS 模板方法：**
- 使用 diversity 包提取长度为 4 的 PoS 模板
- 取所有教师输出中最常见的 50 个 PoS 模式
- 构建 PoS 模板指示特征（50维二值向量）
- 训练逻辑回归分类器（5 类），在教师数据上训练、在学生数据上测试

**核心假设：** 学生模型在蒸馏过程中会内化教师的句法偏好，这些高层次的语言结构特征比表面词汇相似度更具鉴别力。

### 评估指标

分类准确率（5 类，随机基线 0.20），BoW 余弦相似度，BERTScore，AUC-ROC。

## 实验

### 主实验结果

| 学生模型 | 特征类型 | C-D | P-M | R-T | CSQA | OBQA | QRe | Alpaca |
|----------|----------|-----|-----|-----|------|------|-----|--------|
| GPT-2 | BERT | 0.46 | 0.55 | 0.40 | 0.44 | 0.38 | 0.35 | 0.51 |
| GPT-2 | n-grams | 0.58 | 0.68 | 0.44 | 0.56 | 0.48 | 0.50 | 0.56 |
| GPT-2 | **PoS Templates** | **0.60** | **0.71** | **0.54** | **0.69** | **0.51** | **0.59** | 0.55 |
| OLMo-1B | BERT | 0.45 | 0.65 | 0.41 | 0.40 | 0.42 | 0.31 | 0.46 |
| OLMo-1B | n-grams | 0.60 | 0.62 | 0.48 | 0.55 | 0.42 | 0.58 | 0.50 |
| OLMo-1B | **PoS Templates** | **0.61** | **0.74** | 0.45 | **0.59** | 0.43 | **0.61** | 0.53 |

> 5 类分类准确率，随机基线为 0.20。PoS 模板在大多数数据集上优于 n-gram 和 BERT 特征。

### 消融实验

| 方法 | 效果 |
|------|------|
| 困惑度（Perplexity） | 教师困惑度无法可靠区分（真正教师不一定给出最低 PPL） |
| BoW + BERTScore 相似度 | AUC ≈ 0.49-0.53，接近随机，无区分能力 |
| 逻辑回归 + 相似度特征 | AUC 约 0.52，几乎无区分能力 |
| PoS Templates（核心方法） | 明显优于随机，CSQA 上达 0.69，但仍远非完美 |

### 关键发现

- **表面相似度失效：** BoW 和 BERTScore 无法区分不同教师的学生，说明蒸馏传递的不是表面词汇模式
- **困惑度也失效：** 教师模型并不总是偏好自己学生的输出（如 Gemma 对自身蒸馏学生反而给出更高 PPL）
- **句法模板有效：** PoS 模板捕捉的是更高层次的句法结构偏好，这种偏好在蒸馏中被学生保留
- **任务依赖性：** PoS 模板在推理型任务（CSQA: 0.69）上表现最好，在指令遵循（Alpaca: 0.55）上优势较小
- 准确率虽远超随机但远不完美，表明教师指纹存在但不够强，实际应用仍需改进

## 亮点

- 提出了一个新颖且有实际应用价值的问题——蒸馏教师模型的事后归因
- 系统性地排除了直觉上看似合理的方法（困惑度、文本相似度），突出了问题的挑战性
- PoS 模板的有效性揭示了蒸馏传递的是句法层面的隐含偏好，而非表面词汇特征
- 无需访问教师模型内部或使用水印策略

## 局限性

- PoS 模板分类准确率虽远超随机但离实用仍有距离（最高 0.74）
- 假设封闭集场景（教师必在候选集中），无法处理候选集外的教师
- 额外微调、数据增强、多教师蒸馏等可能模糊归因信号
- 不同教师如果训练在相同数据上，可能共享足迹，增加归因难度
- 仅使用了两种学生模型（GPT-2、OLMo-1B），泛化性待验证

## 相关工作

- **LLM 蒸馏：** Hinton (2015) 的知识蒸馏框架；Ho et al. (2023) 的推理教学；Li et al. (2023b) 的符号CoT 蒸馏；Wadhwa et al. (2024a) 的 CoT 增强蒸馏
- **来源追踪：** Li et al. (2024b) 的统计水印检测；Li et al. (2024a) 的生成时水印方法；Li et al. (2023a) 使用困惑度和对比训练检测文本来源
- **LLM 文本检测：** Shaib et al. (2024b) 发现 LLM 偏好特定句法模板（本文的直接灵感）

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 实用性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [DataDecide: How to Predict Best Pretraining Data with Small Experiments](../../ICML2025/model_compression/datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)

<!-- RELATED:END -->
