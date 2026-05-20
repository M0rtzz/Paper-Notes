---
title: >-
  [论文解读] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models
description: >-
  [ACL 2025][LLM/NLP][置信度校准] 提出 Direct Confidence Alignment (DCA)，利用 DPO 将 LLM 的文字表达置信度（verbalized confidence）与内部 token 概率置信度（internal confidence）对齐…
tags:
  - "ACL 2025"
  - "LLM/NLP"
  - "置信度校准"
  - "DPO"
  - "内部置信度"
  - "表达置信度"
  - "LLM 可靠性"
---

# Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models

**会议**: ACL 2025  
**arXiv**: [2512.11998](https://arxiv.org/abs/2512.11998)  
**代码**: 未公开  
**领域**: LLM/NLP  
**关键词**: 置信度校准, DPO, 内部置信度, 表达置信度, LLM 可靠性  

## 一句话总结

提出 Direct Confidence Alignment (DCA)，利用 DPO 将 LLM 的文字表达置信度（verbalized confidence）与内部 token 概率置信度（internal confidence）对齐，提升模型置信度表达的一致性和透明度。

## 研究背景与动机

**核心问题：** LLM 的内部置信度（$C_i$，基于 token 概率的 softmax 值）与表达置信度（$C_v$，模型在回答中输出的置信百分比）之间存在严重不一致。例如模型内部概率仅 30% 的答案，可能在文字中声称 95% 的置信度。

**现有方案局限：**
- 温度缩放、自一致性等校准方法关注将置信度与准确率对齐，忽略了 $C_v$ 和 $C_i$ 的内在差距
- RLHF 可能破坏模型内部 logits 的校准性，使 $C_i$ 本身变得不可靠
- 黑盒模型无法获取 logits，限制了基于 $C_i$ 的校准方法

**核心动机：** 即使 $C_i$ 不完美校准，保证 $C_v$ 与 $C_i$ 的一致性仍有意义——这使模型的不确定性表达更透明、一致。DPO 的偏好对格式天然适合此类对齐任务。

## 方法详解

### 整体框架

DCA 流程包含四步：(1) 使用基础模型对问题生成带 $C_v$ 的回答 → (2) 从答案 token 的 softmax 概率中提取 $C_i$ → (3) 构建偏好数据对：原始回答（含原始 $C_v$）为 rejected，将 $C_v$ 替换为 $C_i$ 的回答为 chosen → (4) 用 DPO 训练完成对齐。

### 关键设计

1. **表达置信度提取：** 通过特定 prompt 模板要求模型在回答末尾输出 "Probability: X%"，解析数值获取 $C_v$，提取错误率 <5%

2. **内部置信度提取：** 取模型输出答案 token（如 A/B/C/D）时的 softmax 概率作为 $C_i$，直接反映模型对答案的内部确信程度

3. **偏好数据构建：** 对每个样本生成一对数据——原始回答为 rejected，将其中 $C_v$ 数值替换为 $C_i$ 数值的回答为 chosen，其余文本完全相同

### 评估指标

提出三个基于校准误差 $\epsilon = C_v - C_i$ 的新指标：
- **$\sigma_\epsilon$（校准误差标准差）**：衡量 $\epsilon$ 的变异程度
- **$\overline{|\epsilon|}$（平均绝对校准误差）**：衡量 $C_v$ 和 $C_i$ 的平均偏差
- **$\sigma_M$（校准误差标准误）**：估计平均对齐的采样不确定性

## 实验

### 主实验：DCA 对齐效果（四个数据集平均）

| 模型 | 方法 | ρ↑ | σ_ε↓ | \|ε\|↓ | σ_M↓ |
|------|------|------|------|------|------|
| Gemma-2-9B-Instruct | Vanilla | 0.34 | 16.97 | 9.91 | 0.57 |
| | **DCA** | **0.42** | **13.79** | **5.03** | **0.46** |
| Llama-3.2-3B-Instruct | Vanilla | 0.28 | 41.19 | 38.67 | 1.40 |
| | DCA | 0.23↓ | **22.88** | 44.03↑ | **0.75** |
| Mistral-7B-Instruct | Vanilla | 0.19 | 25.63 | 22.96 | 0.85 |
| | DCA | 0.13↓ | **22.93** | 48.93↑ | **0.74** |

### DCA 对准确率的影响

| 模型 | OpenBookQA | TruthfulQA | CosmosQA | MMLU |
|------|------|------|------|------|
| Gemma-2-9B Vanilla→DCA | 86.06→**86.21** | 59.68→**60.85** | 79.63→**80.01** | 72.41→72.05 |
| Llama-3.2-3B Vanilla→DCA | 47.14→**64.00** | 29.71→**37.75** | 66.43→**73.55** | 39.92→**49.77** |
| Mistral-7B Vanilla→DCA | 59.00→58.23↓ | 32.84→20.98↓ | 60.48→54.02↓ | 55.91→48.85↓ |

### 关键发现

1. **DCA 效果高度模型依赖**：Gemma-2-9B 在所有指标上一致改善（ρ +0.08，|ε| -4.88），而 Mistral-7B 在多项指标上恶化
2. Gemma 的成功可能部分因为其 $C_v$ 和 $C_i$ 初始分布已高度偏向 90-100% 区间，DCA 强化了这一集中趋势
3. $\sigma_\epsilon$ 和 $\sigma_M$ 在所有模型上普遍改善，表明 DCA 至少降低了校准误差的方差
4. DCA 对准确率影响不一致：Gemma 稳定，Llama 大幅提升（+16.86% on OpenBookQA），Mistral 显著下降（-11.86% on TruthfulQA）
5. 域内和域外数据集表现模式相似，暗示效果更依赖于模型架构而非任务类型

## 亮点

- **新颖的校准视角**：不追求与 ground-truth 准确率对齐，而是对齐模型自身的两种置信度表达，关注透明度而非正确性
- **方法简洁**：巧妙利用 DPO 的偏好对格式，仅需替换 $C_v$ 为 $C_i$ 即可构建训练数据
- **三个新指标**：$\sigma_\epsilon$、$\overline{|\epsilon|}$、$\sigma_M$ 从不同角度衡量置信度对齐质量，比单一 Spearman 相关更全面

## 局限性

- 需要访问模型 logits，不适用于 GPT-4 等闭源模型
- 方法预设 $C_i$ 比 $C_v$ 更可靠作为参考信号，但 $C_i$ 本身可能经 RLHF 后校准不佳
- 偏好数据中包含错误答案选项，导致 Mistral 准确率大幅下降
- 仅在 3 个模型上验证，且 2 个模型效果不理想，泛化性存疑
- Gemma 的"成功"可能是置信度分布坍缩到高值区间的假象

## 相关工作

- **置信度校准**：温度缩放（Guo et al.）、自一致性方法（Wang et al.）、CQO 对齐（Tao et al.）
- **表达置信度**：多次采样取均值（Tian et al.）、多温度多 prompt 策略（Xiong et al.）
- **置信度-概率对齐**：Kumar et al. 首次定义 Confidence-Probability Alignment
- **DPO**：Rafailov et al., 直接偏好优化替代 RLHF

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | 7/10 |
| 有效性 | 5/10 |
| 实验充分度 | 6/10 |
| 写作质量 | 7/10 |
| 总分 | 6/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)

</div>

<!-- RELATED:END -->
