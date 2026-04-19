---
title: >-
  [论文解读] IPO: Your Language Model is Secretly a Preference Classifier
description: >-
  [ACL 2025][LLM/NLP][偏好优化] 提出隐式偏好优化（IPO），利用生成式LLM自身作为偏好分类器（通过"Yes/No"token的概率），替代外部奖励模型来获取偏好信号，从而实现低成本的自对齐训练。
tags:
  - ACL 2025
  - LLM/NLP
  - 偏好优化
  - 奖励模型
  - 自我改进
  - DPO
  - RLHF
---

# IPO: Your Language Model is Secretly a Preference Classifier

**会议**: ACL 2025  
**arXiv**: [2502.16182](https://arxiv.org/abs/2502.16182)  
**代码**: [github](https://github.com/shivank21/Implicit_Preference_Optimization)  
**领域**: LLM/NLP  
**关键词**: 偏好优化, 奖励模型, 自我改进, DPO, RLHF

## 一句话总结

提出隐式偏好优化（IPO），利用生成式LLM自身作为偏好分类器（通过"Yes/No"token的概率），替代外部奖励模型来获取偏好信号，从而实现低成本的自对齐训练。

## 研究背景与动机

RLHF已成为将LLM与人类偏好对齐的主流方法，但其依赖于训练外部奖励模型或人类标注偏好数据，计算和资金成本高昂。现有的Self-Rewarding方法虽然用LLM自身打分，但通常使用离散的1-5分评分，区分度不够，尤其在小模型上表现很差。此外，这些方法需要大模型作为judge，且模型的judging能力在训练中没有被更新。

本文的核心假设是：与其使用离散的prompt-based评分，不如提供一个连续的偏好幅度（preference magnitude）。作者发现，LLM本身就隐含着偏好分类能力——通过查看模型对"这个回答好不好？"这一问题输出"Yes"的概率，就能得到细粒度的偏好信号。

## 方法详解

### 整体框架

IPO框架分为两个阶段：
1. **偏好分类**：用LLM自身作为偏好分类器，通过"Yes" token的概率为回答打分
2. **自我改进训练**：基于打分构建偏好数据集，用DPO进行训练

### 关键设计

1. **基于概率的偏好分类**：给定一个指令和一个回答，使用类别特定的prompt（分为Chat/Code/Math/Safety四类）询问模型"这个回答是否合适"，提取第一个输出token中"Yes"和"No"的logit，经过softmax和归一化后得到偏好分数 $p'_{yes} = \frac{p_{yes}}{p_{yes} + p_{no}}$。分数高的回答被选为chosen，低的为rejected。

2. **类别特定的Prompt设计**：针对Chat、Code、Math、Safety四类任务设计了不同的引导prompt，并建立了自动化的prompt选择流程。对小模型尤其重要，因为它们对prompt措辞高度敏感。

3. **偏好数据集构建**：对每个指令，用SFT/Instruct模型生成4个不同回答，用IPO方法为所有回答打分，选择最高分作为chosen，最低分作为rejected，形成(Prompt, Chosen, Rejected)三元组。

### 损失函数 / 训练策略

采用标准的两阶段训练：
- **SFT阶段**：在Dolly-15k数据集上进行监督微调，优化交叉熵loss
- **DPO阶段**：使用构建好的偏好数据集，优化DPO loss进行偏好对齐

使用4k条UltraFeedback的指令作为输入prompt，用Bart-Zero Shot分类器将其分为四个类别。

## 实验关键数据

### 主实验

在RewardBench上的偏好分类准确率：

| 模型 | Chat | Code | Math | Safety | 平均 |
|------|------|------|------|--------|------|
| Qwen-2.5-7B-Inst (IPO) | 78.26 | 83.13 | 56.24 | 93.24 | 77.71 |
| Mistral-7B-Inst (IPO) | 61.25 | 70.93 | 96.20 | 83.85 | 78.05 |
| Qwen-2.5-7B-Inst (Self-Rewarding) | 58.73 | 47.93 | 40.49 | 52.20 | 49.82 |
| Mistral-7B-Inst (Self-Rewarding) | 24.55 | 1.6 | 28.18 | 15.39 | 17.43 |

DPO训练后的下游任务效果：

| 模型 | BBH | Arc-Easy | Alpaca-Eval | MMLU | IFEval | 平均 |
|------|-----|----------|-------------|------|--------|------|
| Mistral-7B-Ours | 34.60 | 82.20 | 78.20 | 37.60 | 39.19 | 54.35 |
| Mistral-7B-Reward | 30.20 | 85.20 | 77.40 | 41.00 | 31.69 | 53.10 |
| Mistral-7B-Self Rewarding | 31.20 | 77.00 | 69.60 | 33.00 | 29.31 | 48.02 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Base vs Instruct 模型 | Instruct显著更好 | 指令微调提升了偏好分类能力 |
| 大模型 vs 小模型 | 更大模型更好 | Qwen-7B >> Qwen-3B |
| 代码模型 | 与奖励模型持平 | Qwen-Coder表现优异 |
| 数学模型 | 显著较差 | 可能因CoT训练目标与二分类不兼容 |

### 关键发现

1. IPO在所有子类别上均大幅超越Self-Rewarding方法，尤其在小模型上差距更明显
2. Self-Rewarding方法的1-5分评分难以区分回答质量，经常给chosen和rejected相同分数
3. 代码专用模型天然具备偏好分类能力，但数学专用模型不具备（因训练目标差异）
4. 大多数模型在Safety类别上表现最好，说明安全对齐已广泛融入训练
5. IPO训练后的Mistral-7B在多个benchmark上匹配甚至超越基于Skywork奖励模型的训练

## 亮点与洞察

- **极简但有效的思路**：不需要训练奖励模型，不需要复杂的prompt让LLM当judge，只需要提取一个token的概率。计算成本极低。
- **揭示了LLM的隐式偏好能力**：即使是base模型也具有一定的偏好分类能力，这说明预训练过程中已经隐含地学到了质量判断。
- **连续vs离散信号**：概率值比离散评分（1-5分）提供了更细粒度的偏好信号，是偏好表示的重要改进。

## 局限与展望

1. 需要预先分类数据集（Chat/Code/Math/Safety），虽然可考虑让模型自动生成类别标签
2. 仅在1B和7B规模上评估，且使用了4k条prompt的较小子集
3. 因计算限制，使用单次DPO而非迭代DPO（Self-Rewarding的设置）
4. 所有结果均为单次运行（seed=42），鲁棒性验证不足
5. 小模型用简单prompting无法生成高质量指令，需要外部数据集

## 相关工作与启发

- 本文与VQA Score (Lin et al., 2025)的思路类似，都是将概率作为评分信号
- 相比Meta-Rewarding，IPO不需要额外的judging训练，更加轻量
- 可考虑将IPO与迭代训练结合（如SPIN），让模型的偏好分类能力随训练一起提升
- 对推理时scaling（best-of-n选择）也有直接应用价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 思路简洁巧妙，但核心idea偏incremental
- 实验充分度: ⭐⭐⭐⭐ 模型覆盖广泛，但DPO实验规模偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据展示充分
- 价值: ⭐⭐⭐⭐ 提供了一种低成本的偏好获取方案，有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [SelfElicit: Your Language Model Secretly Knows Where is the Relevant Evidence](selfelicit_evidence_highlighting.md)
- [Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)
- [Gradient-Adaptive Policy Optimization: Towards Multi-Objective Alignment of Large Language Models](gapo_multi_objective_alignment.md)
- [Are Your LLMs Capable of Stable Reasoning?](are_your_llms_capable_of_stable_reasoning.md)
- [Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)

<!-- RELATED:END -->
