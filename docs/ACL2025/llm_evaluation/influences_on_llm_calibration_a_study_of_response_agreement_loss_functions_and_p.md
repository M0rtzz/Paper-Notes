---
title: >-
  [论文解读] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles
description: >-
  [ACL 2025][LLM评测][校准] 本文系统研究影响 LLM 校准（calibration）的三大因素——多模型响应一致性、损失函数选择和 prompt 风格，提出 Calib-n 框架通过训练辅助模型聚合多个 LLM 的响应来估计置信度，发现响应一致性和 focal loss 能显著改善校准性能。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "校准"
  - "置信度估计"
  - "辅助模型"
  - "损失函数"
  - "提示学习"
---

# Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles

**会议**: ACL 2025  
**arXiv**: [2501.03991](https://arxiv.org/abs/2501.03991)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 校准、置信度估计、辅助模型、损失函数、Prompt风格  

## 一句话总结

本文系统研究影响 LLM 校准（calibration）的三大因素——多模型响应一致性、损失函数选择和 prompt 风格，提出 Calib-n 框架通过训练辅助模型聚合多个 LLM 的响应来估计置信度，发现响应一致性和 focal loss 能显著改善校准性能。

## 研究背景与动机

**领域现状**：校准——即模型输出置信度与实际准确率的一致性——对 LLM 的可靠部署至关重要。当前获取 LLM 置信度的方法主要有三种：（1）内部概率（logit-based）；（2）口头化置信度（verbalized confidence，让模型自己说出置信度）；（3）基于辅助模型的外部置信度估计。

**现有痛点**：已有校准方法在评估时存在两个被忽视的问题：一是缺乏对不同 prompt 风格的泛化性测试——方法可能只在某种 prompt 下有效；二是缺乏对不同规模 LLM 的系统性评估。此外，现有辅助模型方法通常只基于单个 LLM 的响应，没有利用多模型间的共识信息。

**核心矛盾**：单一 LLM 的输出概率或口头置信度本身不够可靠（大模型经常过度自信），而已有辅助模型方法未充分利用多模型间的信息互补，同时对损失函数的选择缺乏系统研究。

**本文目标**：（1）定义一个控制变量的实验框架，覆盖 12 个 LLM 和 4 种 prompt 风格；（2）验证多模型响应一致性是否改善校准；（3）探索 focal loss 和 AUC 替代损失对校准的影响；（4）分析 prompt 风格对校准方法的影响。

**切入角度**：作者观察到，如果多个 LLM 对同一个问题给出相同的答案，这个答案更可能是正确的。这种跨模型的一致性信号可以作为辅助模型的额外输入特征来改善置信度估计。

**核心 idea**：构建 Calib-n 框架，用辅助模型聚合 n 个 LLM 的响应特征（包括答案文本、答案一致性、口头置信度等），结合 focal loss 来优化校准，实现比内部概率和口头置信度更鲁棒的校准估计。

## 方法详解

### 整体框架

Calib-n 的流程分为三步：（1）收集阶段——对每个问题，使用 n 个 LLM 在给定 prompt 风格下生成响应；（2）特征构建——提取每个 LLM 的响应文本、口头置信度以及跨模型的答案一致性分数；（3）训练辅助模型——以构建的特征为输入，以二元正确性标签为目标，训练一个轻量辅助模型（如逻辑回归或小型 MLP），辅助模型的输出概率即为校准后的置信度。

### 关键设计

1. **多模型响应一致性特征（Response Agreement）**:

    - 功能：捕捉多个 LLM 在同一问题上的答案共识程度
    - 核心思路：对每个问题，收集 n 个 LLM 的答案，计算目标 LLM 的答案与其他 LLM 答案的匹配比例作为一致性分数 $\text{agree}(q) = \frac{1}{n-1}\sum_{i \neq t} \mathbb{1}[a_i = a_t]$。这个分数和目标 LLM 的响应文本嵌入一起作为辅助模型的输入。
    - 设计动机：多模型共识是一种"集体智慧"信号——如果大多数模型给出相同答案，该答案的可信度更高。这比单纯依赖一个模型的 logit 更鲁棒

2. **Focal Loss 与 AUC 替代损失**:

    - 功能：改善校准优化目标，解决标准交叉熵的不足
    - 核心思路：标准二元交叉熵（BCE）平等对待所有样本，但校准任务中"容易样本"（模型非常确定且正确/错误的）贡献的梯度占主导，淹没了"困难样本"（模型不确定的边界样例）的信号。Focal loss 通过 $(1-p_t)^\gamma$ 的调制因子降低容易样本的权重：$\mathcal{L}_{\text{focal}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$。AUC 替代损失则直接优化排序质量。
    - 设计动机：校准本质上需要模型在置信度排序上准确——高置信度的样本确实应该比低置信度的更可能正确。Focal loss 更关注边界样例，这恰好是校准最关键的区域

3. **四种 Prompt 风格的系统评估**:

    - 功能：评估校准方法在不同 prompt 风格下的泛化性
    - 核心思路：设计了 zero-shot、few-shot、chain-of-thought (CoT) 和 few-shot-CoT 四种 prompt 风格，在每种风格下分别评估所有校准方法。辅助模型在每种 prompt 风格上独立训练和测试。
    - 设计动机：实际部署中 prompt 风格多变，如果校准方法只在特定 prompt 下有效，其实用性大打折扣

### 损失函数 / 训练策略

辅助模型训练使用三种损失函数：BCE（基线）、Focal Loss（$\gamma=2$）、AUC 替代损失。训练数据来自 LLM 在训练集上的响应及其正确性标签。模型结构为轻量级分类器（文本嵌入 + 一致性分数 → MLP → 置信度）。

## 实验关键数据

### 主实验

| 方法 | TriviaQA (ECE↓) | CoQA (ECE↓) | SQuAD (ECE↓) | Natural Questions (ECE↓) |
|------|-----------------|-------------|-------------|------------------------|
| Internal Prob (Llama-70B) | 0.182 | 0.201 | 0.175 | 0.193 |
| Verbalized Conf | 0.156 | 0.178 | 0.162 | 0.171 |
| Calib-1 (BCE) | 0.098 | 0.112 | 0.095 | 0.108 |
| Calib-n (BCE) | 0.082 | 0.096 | 0.081 | 0.094 |
| Calib-n (Focal) | **0.071** | **0.085** | **0.073** | **0.082** |

### 消融实验

| 配置 | 平均 ECE↓ | 说明 |
|------|----------|------|
| Calib-n + Focal (完整) | 0.078 | 最优配置 |
| Calib-1 + Focal (无一致性) | 0.091 | 去掉多模型一致性，ECE 上升 17% |
| Calib-n + BCE (无 Focal) | 0.088 | 换回标准 BCE，ECE 上升 13% |
| Calib-n + AUC Loss | 0.084 | AUC 损失有一定效果但不如 Focal |
| 仅用内部概率 | 0.188 | 基线最差 |

### 关键发现

- **多模型响应一致性带来约 17% 的校准改善**（ECE 降低），证实了跨模型共识的有效性
- **Focal loss 一致优于 BCE 和 AUC loss**，尤其在边界样例多的数据集上效果更明显
- **Few-shot prompt 是辅助模型方法的最佳搭配**——zero-shot 提供的信号太少，CoT 虽然提升准确率但对校准的帮助有限
- **辅助模型在 LLM 准确率波动时表现依然稳定**，而内部概率和口头置信度会随准确率大幅波动

## 亮点与洞察

- **多模型共识作为校准信号**是一个简单但有效的 idea——实际部署中可以用少量额外的 API 调用来获得更可靠的置信度估计，成本可控但收益明显
- **系统化的 prompt 风格影响分析**填补了校准研究中的一个重要空白——之前的工作通常只在一种 prompt 下评估，结论的泛化性存疑
- **Focal loss 在校准任务中的适配**可以推广到其他需要关注边界情况的任务，如 OOD 检测、不确定性量化等

## 局限与展望

- 辅助模型需要对每种 prompt 风格和数据集重新训练，迁移性有待验证
- 多模型一致性依赖于调用多个 LLM，推理成本线性增加，对延迟敏感的应用不友好
- 仅在 QA 任务上验证，生成式任务（摘要、翻译）的校准场景未涉及
- 一致性分数是硬匹配（exact match），对于开放式生成答案的场景需要更柔性的匹配方式

## 相关工作与启发

- **vs Temperature Scaling**: 温度缩放是最简单的后处理校准方法，但需要访问 logit，对闭源 API 不适用。Calib-n 的辅助模型方法更通用
- **vs Verbalized Confidence (Lin et al. 2022)**: 口头化置信度让 LLM 自报置信度，但模型容易过度自信。Calib-n 通过外部辅助模型避免了这个问题
- **vs Self-Consistency (Wang et al. 2023)**: 自洽性通过多次采样同一模型来估计置信度，Calib-n 则利用不同模型的差异性，信息互补更强

## 评分

- 新颖性: ⭐⭐⭐⭐ 多模型响应一致性做校准是新颖的视角，但整体框架较为直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个 LLM、4 种 prompt、4 个数据集、3 种损失函数的交叉实验非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融分析有条理
- 价值: ⭐⭐⭐⭐ 对 LLM 校准的实践指导价值高，但方法创新性一般

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Navigating Rifts in Human-LLM Grounding: Study and Benchmark](navigating_rifts_in_human-llm_grounding_study_and_benchmark.md)
- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)
- [\[ACL 2025\] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling](mis-prompt_benchmarking_large_language_models_for_proactive_error_handling.md)

</div>

<!-- RELATED:END -->
