---
title: >-
  [论文解读] "My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise
description: >-
  [ACL 2025][LLM 其他][Humblebragging] 首次将 humblebragging（谦虚式自夸）检测引入计算语言学领域，提出了一个4元组形式化定义，构建了 HB-24 合成数据集，并在 ML/DL/LLM 上进行了全面基准评估，GPT-4o 在 zero-shot+定义 设定下达到 0.88 F1，超越人类标注者。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "Humblebragging"
  - "文本分类"
  - "讽刺检测"
  - "合成数据"
  - "LLM"
---

# "My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise

**会议**: ACL 2025  
**arXiv**: [2412.20057](https://arxiv.org/abs/2412.20057)  
**代码**: [有 (GitHub)](https://github.com/SharathHN/HB-24)  
**领域**: 其他  
**关键词**: Humblebragging, 文本分类, 讽刺检测, 合成数据, LLM

## 一句话总结

首次将 humblebragging（谦虚式自夸）检测引入计算语言学领域，提出了一个4元组形式化定义，构建了 HB-24 合成数据集，并在 ML/DL/LLM 上进行了全面基准评估，GPT-4o 在 zero-shot+定义 设定下达到 0.88 F1，超越人类标注者。

## 研究背景与动机

Humblebragging 是一种通过伪装成抱怨或谦虚来进行自我推销的语言现象。例如："天哪，我居然被提升为整个团队的负责人了，压力太大了！"——表面是抱怨压力，实际是炫耀升职。

这种现象在心理学、旅游研究、广告学等领域已有研究，但在计算语言学中从未被探索。自动检测 humblebragging 对多个 NLP 下游任务至关重要：

**情感分析**：一条 humblebragging 表面看似消极，但实际意图是积极的，错误判断会严重影响情感极性判断

**意图识别**：需要区分真正的抱怨和伪装的自夸

**对话理解**：在社交媒体监控和客户反馈分析中，需要区分真正的投诉和变相的炫耀

与讽刺（sarcasm）和反讽（irony）类似，humblebragging 也依赖于字面意思与真实意图之间的不一致性，但它的独特之处在于将自我推销隐藏在谦虚或抱怨的话语中。此前没有任何计算语言学的数据集或方法来处理这个问题。

## 方法详解

### 整体框架

论文从三个层面展开：(1) 提出形式化定义 (2) 构建数据集 (3) 设计检测方法并评估。

### 关键设计

#### 1. **4元组形式化定义**

将 humblebragging 定义为 $HB = \langle B, BT, HM, MT \rangle$：

- $B$（Brag）：文本中明确的自夸部分
- $BT$（Brag Theme）：自夸的主题类别（如财富、名气、工作表现等）
- $HM$（Humble Mask）：用于掩饰自夸的谦虚/抱怨部分
- $MT$（Mask Type）：掩饰类型，是谦虚还是抱怨

这个定义改编自讽刺的6元组框架，但去掉了 Speaker 和 Hearer（因为 humblebragging 通常没有特定的听众对象）。设计动机是将 humblebragging 的结构化分解，使得机器可以理解其中的「字面意思 vs 真实意图」的双层语义。

#### 2. **HB-24 数据集构建**

由于缺少现有数据集，作者采用了"合成训练、真实测试"的策略：

- **训练集**：用 GPT-4o 通过 zero-shot 和 few-shot 提示生成，共生成 11,000 条候选样本，经人工审查后筛选出 3,340 条合成 humblebrags
- **测试集**：558 条来自 Wittels (2012) 书中的真实 humblebrags
- **负样本**：包含讽刺（SARC 数据集）、反讽（SemEval-2018）、直接自夸、投诉和中性句子，所有负样本均为人工撰写

负样本的精心设计是关键：humblebragging 容易与讽刺和反讽混淆，因此在训练集中加入这些"混淆项"以提升模型的区分能力。

#### 3. **分类方法设计**

提出两种任务设定：

- **二分类**：标准的编码器分类方法（BERT、RoBERTa）
- **句子补全/问答**：将检测任务转化为 Yes/No 问答，输入格式为 `<definition><question><x><answer>`，让解码器模型在 zero-shot (Z) 和 zero-shot + 定义 (Z+D) 两种设定下完成分类

### 损失函数 / 训练策略

- 编码器模型使用 Adam 优化器 + 5折交叉验证调参
- 解码器模型在 Z 和 Z+D 设定下进行 zero-shot 推理，部分模型使用 LoRA 微调
- 特别地，Z+D 设定中将 4元组定义作为 system prompt 的一部分，引导模型理解 humblebragging 的结构

## 实验关键数据

### 主实验

| 模型 | 准确率 | 精确率 | 召回率 | F1 |
|------|--------|--------|--------|-----|
| 人类平均 | 0.80 | 0.86 | 0.71 | 0.77 |
| SVM | 0.62 | 0.72 | 0.61 | 0.56 |
| BERT-Large (F) | 0.68 | 0.76 | 0.50 | 0.61 |
| RoBERTa-Large (F) | 0.78 | 0.91 | 0.62 | 0.74 |
| GPT-4o (Z) | 0.84 | 0.78 | 0.94 | 0.85 |
| **GPT-4o (Z+D)** | **0.89** | **0.91** | **0.85** | **0.88** |
| Llama-3.1-8B (Z) | 0.49 | 0.49 | 0.99 | 0.66 |
| Llama-3.1-8B (Z+D) | 0.68 | 0.62 | 0.88 | 0.72 |
| Llama-3.1-8B (F) | 0.81 | 0.87 | 0.72 | 0.79 |

### 消融实验：4元组定义 vs 其他

| 模型设定 | 准确率 | 精确率 | 召回率 | F1 |
|----------|--------|--------|--------|-----|
| Llama (Z+D) 4元组定义 | 0.68 | 0.62 | 0.88 | 0.72 |
| Llama (Z+SOM) 情感对立模型 | 0.66 | 0.64 | 0.74 | 0.68 |

### 下游应用：情感极性分类

| 模型 | 准确率 | F1 |
|------|--------|-----|
| R-SST2（原始） | 0.53 | 0.51 |
| R-HBSC（+humblebragging 检测） | 0.82 | 0.83 |

### 关键发现

1. **GPT-4o (Z+D) 超越人类**：F1 达到 0.88，而最优人类标注者仅为 0.85，说明大模型在拥有充分的语言知识和世界知识时可以超过人类
2. **4元组定义的普适有效性**：所有解码器模型在 Z+D 设定下都优于 Z 设定，证明形式化定义能有效辅助检测
3. **合成数据微调有效**：用 HB-24 微调后，大部分模型 F1 显著提升；微调后的 RoBERTa 甚至超过了大部分 7-8B 解码器模型
4. **任务对人类也很难**：三位人类标注者中有一位 F1 仅 0.63，说明 humblebragging 检测本身具有挑战性
5. **下游任务获益显著**：加入 humblebragging 和 sarcasm 检测后，情感分类 F1 从 0.51 提升到 0.83

## 亮点与洞察

- **跨学科视角**：将心理学中的 Maslow 需求层次理论引入动机分析——humblebragging 同时满足了归属感（第3层，通过谦虚）和自尊（第4层，通过炫耀）两个层次的需求
- **从讽刺研究借鉴**：4元组定义从讽刺的6元组框架改编而来，这种"站在巨人肩膀上"的形式化策略值得学习
- **合理的评估设计**：训练用合成数据、测试用真实数据，很好地平衡了数据获取成本和评估真实性

## 局限与展望

1. 合成数据与真实数据存在分布差异（如缺少口语化表达、拉长词等），可能限制泛化能力
2. 数据集规模有限（3,340 条训练 + 约 1,100 条测试）
3. 目前仅限于英文文本，未考虑跨语言/跨文化的 humblebragging 差异
4. 仅做了二分类检测，未深入到成分识别（自动提取 B、HM 等）
5. 可以扩展到多模态场景（如图像+文本的 humblebragging 检测）

## 相关工作与启发

- 讽刺和反讽检测领域有大量研究可以借鉴，特别是基于不一致性（incongruity）的检测方法
- 合成数据生成方法（GPT-4o 提示工程）在低资源任务中的应用具有参考价值
- 将分类任务重新表述为生成/问答任务的范式，适用于 LLM 时代的各种文本分类场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次在计算语言学中引入 humblebragging 检测，4元组形式化定义新颖实用
- **实验充分度**: ⭐⭐⭐⭐ — 涵盖 ML/DL/LLM/人类对比，消融实验和下游应用验证齐全
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，示例直观
- **价值**: ⭐⭐⭐⭐ — 开辟了新的研究方向，发布了数据集和代码，对情感分析等下游任务有实际提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](../../NeurIPS2025/others/sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[ICLR 2026\] A Scalable Inter-edge Correlation Modeling in CopulaGNN for Link Sign Prediction](../../ICLR2026/others/a_scalable_inter-edge_correlation_modeling_in_copulagnn_for_link_sign_prediction.md)
- [\[AAAI 2026\] Life, Machine Learning, and the Search for Habitability: Predicting Biosignature Fluxes for the Habitable Worlds Observatory](../../AAAI2026/others/life_machine_learning_and_the_search_for_habitability_predicting_biosignature_fl.md)
- [\[ECCV 2024\] Enhancing Optimization Robustness in 1-bit Neural Networks through Stochastic Sign Descent](../../ECCV2024/others/enhancing_optimization_robustness_in_1-bit_neural_networks_through_stochastic_si.md)
- [\[ACL 2025\] DeepRTL2: A Versatile Model for RTL-Related Tasks](deeprtl2_a_versatile_model_for_rtl-related_tasks.md)

</div>

<!-- RELATED:END -->
