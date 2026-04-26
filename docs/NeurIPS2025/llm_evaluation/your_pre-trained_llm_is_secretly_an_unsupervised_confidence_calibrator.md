---
title: >-
  [论文解读] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator
description: >-
  [NeurIPS 2025][confidence calibration] 发现 LLM 后训练（SFT/RLHF/DPO）破坏了预训练模型的置信度校准，提出 DACA 方法利用预训练模型的良好校准性，仅在预测一致样本上对齐置信度，实现无标签的后训练模型校准，ECE 最高改善 15.08%。
tags:
  - NeurIPS 2025
  - confidence calibration
  - temperature scaling
  - pre-trained LM
  - post-trained LM
  - unsupervised calibration
  - DACA
---

# Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator

**会议**: NeurIPS 2025  
**arXiv**: [2505.16690](https://arxiv.org/abs/2505.16690)  
**代码**: [GitHub](https://github.com/ml-stat-Sustech/Disagreement-Aware-Calibration)  
**领域**: llm_nlp  
**关键词**: confidence calibration, temperature scaling, pre-trained LM, post-trained LM, unsupervised calibration, DACA

## 一句话总结

发现 LLM 后训练（SFT/RLHF/DPO）破坏了预训练模型的置信度校准，提出 DACA 方法利用预训练模型的良好校准性，仅在预测一致样本上对齐置信度，实现无标签的后训练模型校准，ECE 最高改善 15.08%。

## 研究背景与动机

### 问题定义
LLM 的标准训练范式是"预训练 → 后训练"。预训练语言模型（PLM）通常具有良好的置信度校准（即模型输出的置信度能真实反映正确率），但经过 SFT、RLHF、DPO 等后训练后，模型变得**过度自信**——对正确和错误的输出都赋予很高的置信度。

### 现有方法的问题

**温度缩放（Temperature Scaling）**是最实用的后验校准方法，但**依赖有标签数据**
2. 标签获取在数学推理、医疗诊断等领域极其昂贵且费时
3. 现实部署中大量**无标签数据闲置未用**
4. 基于 prompt 的校准方法（verbalization）效果有限

### 核心洞察
预训练模型天然具有良好校准性——能否利用 PLM 的置信度来校准过度自信的 PoLM？关键发现：**直接对齐会导致欠置信**，根源在于预测不一致样本（disagreement examples）的干扰。

## 方法详解

### 整体框架

DACA（Disagreement-Aware Confidence Alignment）的核心思路：

1. 检测 PLM 和 PoLM 预测是否一致
2. 仅在一致样本上对齐两者的置信度分布
3. 通过 KL 散度最小化求解最优温度参数

### 为什么直接对齐行不通？

**朴素置信度对齐**：最小化 PLM 和 PoLM 在所有无标签数据上的 KL 散度：

$$\tau^* = \arg\min_{\tau > 0} \mathbb{E}_{\boldsymbol{x} \in \mathcal{D}} \left[ \sum_{i=1}^{k} p_i(\boldsymbol{x}) \log \frac{p_i(\boldsymbol{x})}{\sigma_i(g(\boldsymbol{x})/\tau)} \right]$$

**问题**：当两个模型预测不一致时，PLM 的置信度反映的是**自己预测的准确性**而非 PoLM 预测的准确性。由于后训练通常提升了 PoLM 的精度，PLM 置信度会**低估** PoLM 的真实正确率，导致温度参数 $\tau$ 被推向过大值。

### 理论分析

**命题 3.2**：即使 PLM 完美校准（$ECE_f=0$），完美对齐后的 PoLM 的 ECE 为：

$$ECE_g = \pi \cdot \left| \mathbb{E}_{\boldsymbol{x}} \left[ \mathbf{1}\{\arg\max_i f_i(\boldsymbol{x}) = \tilde{y}\} - \mathbf{1}\{\arg\max_i g_i(\boldsymbol{x}) = \tilde{y}\} \right] \right|$$

其中 $\pi$ 为不一致样本比例。**结论**：由于预测不一致的存在，即使理想对齐也无法达到零 ECE。

**命题 3.3**：对于不一致样本，如果 PoLM 预测类别 $c$ 但 PLM 给 $c$ 的概率 $<1/k$，则最优温度 $\tau^* = \infty$。**含义**：不一致样本上 KL 散度关于 $\tau$ 的梯度始终为正，会持续推高温度值。

### 关键设计：DACA 损失函数

$$\mathcal{L}(\tau; \boldsymbol{x}) = \mathbf{1}\{\hat{y} = \hat{y}'\} \cdot \left[ \sum_{i=1}^{k} p_i(\boldsymbol{x}) \log \frac{p_i(\boldsymbol{x})}{\sigma_i(g(\boldsymbol{x})/\tau)} \right]$$

其中 $\hat{y} = \arg\max_i f_i(\boldsymbol{x})$（PLM 预测），$\hat{y}' = \arg\max_i g_i(\boldsymbol{x})$（PoLM 预测）。

**核心操作**：用指示函数 $\mathbf{1}\{\hat{y} = \hat{y}'\}$ 过滤掉不一致样本，仅在一致样本上计算 KL 散度。

### 通用扩展

方法可推广到任意后验校准方法（向量缩放、矩阵缩放）：

$$\boldsymbol{\theta}^* = \arg\min_{\tau > 0} \mathbb{E}_{\boldsymbol{x} \in \mathcal{D}} \left[ \mathbf{1}\{\hat{y} = \hat{y}'\} \cdot \sum_{i=1}^{k} p_i(\boldsymbol{x}) \log \frac{p_i(\boldsymbol{x})}{q_i(\boldsymbol{x}; \boldsymbol{\theta})} \right]$$

### 开放域 QA 扩展

对于开放式问答，采用 P(True) 方法获取置信度：让模型判断自己生成的答案是否正确，取 $p(\text{Yes}|x, f)$ 作为置信度分数。

## 实验关键数据

### 主实验：MMLU 57 科目平均校准性能

| 模型 | 方法 | ECE(%) ↓ | MCE(%) ↓ | AECE(%) ↓ | Brier ↓ |
|------|------|----------|----------|-----------|---------|
| Qwen3-8B | Vanilla | 16.38 | 38.19 | 24.99 | 0.179 |
| Qwen3-8B | CAPE | 11.52 | 31.74 | 17.61 | 0.157 |
| Qwen3-8B | **DACA** | **8.39** | **23.70** | **12.60** | **0.144** |
| Qwen3-8B | TS (有标签) | 8.66 | 28.11 | 14.55 | 0.146 |
| Gemma-3-12B-IT | Vanilla | 23.68 | 48.51 | 35.89 | 0.235 |
| Gemma-3-12B-IT | **DACA** | **8.60** | **27.02** | **13.55** | **0.154** |
| Gemma-3-12B-IT | TS (有标签) | 9.75 | 29.80 | 15.60 | 0.159 |
| Yi-1.5-34B-Chat | Vanilla | 16.20 | 33.82 | 20.35 | 0.199 |
| Yi-1.5-34B-Chat | **DACA** | **9.47** | **19.90** | **11.70** | **0.174** |
| Llama-3-70B-IT | Vanilla | 12.87 | 36.87 | 23.84 | 0.143 |
| Llama-3-70B-IT | **DACA** | **7.84** | **24.28** | **13.16** | **0.120** |

**亮点**：DACA 在无标签设置下，性能**与有标签 TS 持平甚至超越**（如 Gemma-3-12B: 8.60 vs 9.75）。

### API 模型校准（GPT-4o + 不同 PLM）

| 校准用 PLM | PLM 原始 ECE(%) | GPT-4o ECE(%) ↓ |
|-----------|----------------|-----------------|
| 无（Vanilla） | — | 21.23 |
| Llama-3-8B | 9.45 | 7.98 |
| Qwen2.5-7B | 6.99 | 7.82 |
| **Gemma-3-12B** | **4.42** | **6.99** |

**发现**：PLM 校准越好 → DACA 对齐效果越好。可以用小模型校准大模型/闭源模型。

### 消融实验：不同后训练策略

| 后训练方式 | Vanilla ECE(%) | DACA ECE(%) |
|-----------|---------------|-------------|
| SFT | 14.85 | 4.57 |
| SFT + DPO | 25.12 | 5.42 |
| SFT + DPO + RLVR | 25.19 | 5.99 |

**发现**：后训练越"激进"（加 DPO/RLVR），过度自信越严重，但 DACA 都能有效校准。

### 开放域 QA 和选择性分类

- TruthfulQA 上 Qwen2.5-32B-Instruct: ECE 从 30.96% → 5.24%
- 选择性分类中，校准后的高置信预测准确率在所有阈值（0.5-0.95）上都显著提升
- 高阈值处优势更明显（因过度自信被有效缓解）

### 关键发现

1. Verbalization 方法（Elicitation 系列）表现远差于基于 logits 的方法
2. 模型越大，Vanilla ECE 越低（与前人研究一致）
3. DACA 对模型规模、架构、后训练策略都具有鲁棒性
4. 不一致样本上训练温度会持续增长到极大值（实验验证了理论）

## 亮点与洞察

1. **洞察极其深刻**：发现"预测不一致是朴素对齐失败的根源"，这个观察简单但解释力很强
2. **理论与实验完美配合**：Proposition 3.2/3.3 清晰解释了为什么直接对齐导致欠置信，实验完美验证
3. **实用性极强**：
    - 无需标签数据
    - 可跨架构（小 PLM 校准大 PoLM）
    - 可用于闭源 API 模型（GPT-4o, DeepSeek-V3）
    - 计算开销极低（只需一次推理 + 温度优化）
4. **方法极简但有效**：本质就是加了一个 indicator function 过滤不一致样本，但效果惊人
5. **DACA 无标签竟能打败有标签 TS**，说明 PLM 的校准信息非常丰富

## 局限与展望

1. **额外推理开销**：需要额外跑一次 PLM 推理来判断 agreement
2. **不一致样本被浪费**：过滤掉的 disagreement 样本信息完全没有利用
3. **需要可获取 logits**：对完全黑盒（连 logits 都不返回的）模型不适用
4. **校准目标为 MCQA**：虽然扩展到了开放 QA，但主实验限于选择题场景
5. **改进方向**：
    - 设计利用 disagreement 样本的方法（例如加权而非完全过滤）
    - 探索在生成任务上的适用性
    - 与 RLHF 训练过程集成，做训练时校准

## 相关工作与启发

- **与 Temperature Scaling (Guo et al. 2017) 的关系**：TS 是基础方法但需标签；DACA 用 PLM 置信度替代标签，第一个实现无标签后验校准
- **与 CAPE (Jiang et al. 2023) 对比**：CAPE 通过排列选项顺序来校准，属于 prompt 工程；DACA 是原理性方法，效果更好
- **与 Shen et al. 2024 (Thermometer) 对比**：Thermometer 训练辅助模型来预测温度，需要标签和训练；DACA 零训练
- **启发**：PLM 的校准性质是一种被低估的"资产"，可以被后训练流程更好地利用

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 第一个无标签 LLM 后验校准方法，insight 深刻且优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖多个模型族、模型规模、后训练策略、数据集、开闭源模型，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，实验呈现结构良好；motivation 到 method 的过渡自然
- **价值**: ⭐⭐⭐⭐⭐ — 解决了 LLM 部署中的核心痛点，方法简单实用，可直接落地

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Dual Consolidation for Pre-Trained Model-Based Domain-Incremental Learning](../../CVPR2025/llm_evaluation/dual_consolidation_for_pre-trained_model-based_domain-incremental_learning.md)
- [\[NeurIPS 2025\] ConfTuner: Training Large Language Models to Express Their Confidence Verbally](conftuner_training_large_language_models_to_express_their_confidence_verbally.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[NeurIPS 2025\] Leveraging Robust Optimization for LLM Alignment under Distribution Shifts](leveraging_robust_optimization_for_llm_alignment_under_distribution_shifts.md)

<!-- RELATED:END -->
