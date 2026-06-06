---
title: >-
  [论文解读] Calibrating Verbalized Confidence with Self-Generated Distractors
description: >-
  [ICLR 2026][AIGC检测][置信度校准] 提出 DiNCo 方法，通过让 LLM **独立**评估自动生成的干扰选项（合理但错误的替代答案）来暴露其"暗示性偏差"，用干扰项上的总置信度进行归一化，并融合生成一致性与验证一致性两个互补维度，在短文本 QA 和长文本生成任务上显著改善置信度校准。
tags:
  - "ICLR 2026"
  - "AIGC检测"
  - "置信度校准"
  - "语言化概率"
  - "干扰项生成"
  - "NLI 重加权"
  - "生成-验证一致性"
---

# Calibrating Verbalized Confidence with Self-Generated Distractors

**会议**: ICLR 2026  
**arXiv**: [2509.25532](https://arxiv.org/abs/2509.25532)  
**代码**: [victorwang37/dinco](https://github.com/victorwang37/dinco)  
**领域**: AIGC检测  
**关键词**: 置信度校准, 语言化概率, 干扰项生成, NLI 重加权, 生成-验证一致性

## 一句话总结

提出 DiNCo 方法，通过让 LLM **独立**评估自动生成的干扰选项（合理但错误的替代答案）来暴露其"暗示性偏差"，用干扰项上的总置信度进行归一化，并融合生成一致性与验证一致性两个互补维度，在短文本 QA 和长文本生成任务上显著改善置信度校准。

## 研究背景与动机

**领域现状**：LLM 可以通过"语言化置信度"（verbalized confidence）直接输出其对答案的确信程度——要么让模型自报数值（如"80%"），要么通过 $P(\text{True})$ 方式计算。这种单次调用方式比多次采样高效得多，但校准质量堪忧。

**现有痛点**：
- **过度自信**：语言化置信度存在系统性过高问题，模型在错误答案上也常报出 0.8+ 的置信度
- **置信度饱和**：分数集中在少数区间（如 0.9-1.0），使得无论如何设置阈值，都无法有效区分正确与错误答案
- **跨难度不可比**：简单问题的错误答案和困难问题的正确答案可能获得相同分数

**核心观察**：作者提出"暗示性"（suggestibility）假说——当 LLM 对某个主题知之甚少时，**将 claim 放入上下文本身就会拉高模型对该 claim 的置信度**。实验验证表明，错误回答的问题其总置信度 $\beta(C)$ 显著高于正确回答的问题，证实了模型在认知不确定时更容易"来者不拒"。

## 方法详解

### 暗示性偏差建模

将语言化置信度建模为潜在真实置信度乘以暗示性偏差标量：

$$f^{\text{VC}}(c) = \beta(c) \cdot f^{\text{lat}}(c)$$

其中 $\beta(c)$ 是 claim $c$ 的暗示性偏差。关键假设是：对于逻辑相关的互斥 claim 集合 $C$，偏差近似相等 $\beta(c) \approx \beta(C)$。由于潜在置信度应满足概率归一化 $\sum_{c \in C} f^{\text{lat}}(c) = 1$，可得：

$$\beta(C) \approx \sum_{c \in C} f^{\text{VC}}(c), \quad f^{\text{NVC}}(c_0) = \frac{f^{\text{VC}}(c_0)}{\beta(C)}$$

实际使用 $\beta(C) \leftarrow \max(1, \beta(C))$ 避免 claim 集合不完备时的过度缩放。

### 干扰项生成策略

目标是找到高生成概率的互斥替代 claim 集合，最大化 $\sum_{c \in C} f^{\text{VC}}(c)$ 且 $|C| \leq K$：

| 场景 | 干扰项生成方式 | 特点 |
|------|--------------|------|
| 开源模型（logit 可用） | Beam search 生成多个高概率替代答案 | 高效覆盖概率质量，避免独立采样的重复 |
| API 模型（top-token 可用） | 伪 beam search（利用 top token 概率） | 近似 beam search 效果 |
| 纯黑盒模型 | 直接提示模型生成候选答案列表 | 无需任何概率访问 |
| 长文本生成 | 先分解为原子 claim，再为每个 claim 生成干扰项 | 适配 FactScore 评估框架 |

### NLI 重加权机制

由于生成的干扰项无法保证严格互斥，使用 NLI 模型（DeBERTa-v3-base）计算两个权重：

- **唯一性权重**：$w_{\text{unique}}(c) = \frac{1}{\sum_{c' \in C} P(\text{entail} \mid c', c)}$，对被其他 claim 蕴含的重复项降权
- **矛盾性权重**：$w_{\text{contra}}(c) = \frac{P(\text{contra} \mid c_0, c) + P(\text{contra} \mid c, c_0)}{2}$，对与原始 claim 不矛盾的项降权

归一化因子变为：

$$\beta(C) = \max\left(1, f^{\text{VC}}(c_0) + \sum_{c \in C} f^{\text{VC}}(c) \cdot w_{\text{unique}}(c) \cdot w_{\text{contra}}(c)\right)$$

### 生成-验证一致性融合

作者发现 beam search 生成的最高概率答案与验证阶段最高置信度答案仅在 59.2% 的问题上一致，表明生成器和验证器存在系统性分歧。DiNCo 将两个互补维度融合：

$$f^{\text{DiNCo}}(c) = \frac{1}{2} f^{\text{SC}}(c) + \frac{1}{2} f^{\text{NVC}}(c)$$

其中 $f^{\text{SC}}$ 是自一致性（self-consistency）估计的生成置信度，$f^{\text{NVC}}$ 是归一化后的验证置信度。推理预算 $K=10$ 时，5 个样本用于 SC，5 个干扰项用于 NVC。

## 关键设计

1. **独立评估而非联合提示**：对每个干扰项独立询问模型置信度，而非一次性呈现所有候选——如果联合呈现，模型可以通过简单算术满足概率归一化，从而掩盖不一致性
2. **NLI 重加权保证归一化质量**：通过蕴含和矛盾关系的连续权重处理部分等价/矛盾的 claim，消除简单计数的偏差
3. **双维度一致性融合**：将采样生成一致性（SC）和验证归一化一致性（NVC）作为互补信号整合，弥补单一维度的盲区

## 实验结果

### 短文本 QA 结果

| 方法 | TriviaQA ECE ↓ | TriviaQA AUC ↑ | SimpleQA ECE ↓ | SimpleQA AUC ↑ |
|------|:-:|:-:|:-:|:-:|
| VC | 0.240 | 0.817 | 0.547 | 0.644 |
| K-VC | 0.341 | 0.604 | 0.338 | 0.632 |
| MSP | 0.149 | 0.819 | 0.263 | 0.800 |
| SC | 0.236 | 0.785 | 0.220 | 0.750 |
| NVC | 0.171 | 0.853 | 0.164 | 0.729 |
| **DiNCo** | **0.097** | **0.879** | **0.089** | **0.786** |

> 以上 TriviaQA 结果为 Qwen3-8B，SimpleQA 结果为 GPT-4.1。DiNCo 在 ECE 上平均优于最佳 baseline（MSP）0.077（TriviaQA）和 0.092（SimpleQA）。

### 长文本生成结果（FactScore）

| 方法 | Qwen3-8B ECE ↓ | Qwen3-8B Pearson $r$ ↑ | Gemma-3-4B ECE ↓ | Gemma-3-4B Pearson $r$ ↑ |
|------|:-:|:-:|:-:|:-:|
| VC | 0.433 | 0.073 | 0.527 | -0.081 |
| SC | 0.162 | 0.468 | 0.197 | 0.629 |
| NVC | 0.191 | 0.444 | 0.123 | 0.695 |
| **DiNCo** | **0.076** | **0.518** | **0.172** | **0.724** |

> DiNCo 的 passage-level Pearson/Spearman 相关系数平均优于 SC 0.072/0.074。

### 饱和度与扩展性分析

- **饱和度**：DiNCo 的 $\Delta_0 = 0.998$（几乎所有样本对置信度不同），而 VC 仅 0.670，SC@100 仅 0.832
- **扩展 SC 无法弥补差距**：SC 从 10 扩到 100 个样本（FLOP 增加 7.6 倍于 DiNCo），ECE 改善微乎其微，无法追平 DiNCo
- **NLI 消融**：移除 NLI 重加权后 NVC 的 ECE 从 0.171 恶化到 0.358，证实 NLI 权重的关键作用

## 论文评价

**优点** ⭐⭐⭐⭐
- 从"暗示性偏差"角度分析过度自信，理论动机清晰且有实验验证
- 方法对开源/闭源模型均适用，且从短文本 QA 无缝迁移到长文本生成
- 仅需轻量 NLI 模型（184M 参数，<1% 总 FLOP），零资源、无训练
- 饱和度分析指标 $\Delta_\epsilon$ 的提出量化了此前仅定性讨论的问题

**不足** ⭐⭐⭐
- 干扰项质量依赖模型自身生成能力，对小模型效果可能受限
- 假设偏差 $\beta$ 在逻辑相关 claim 间近似相等，对语义距离较远的 claim 可能不成立
- 长文本场景需要额外的 claim 分解步骤，增加了流水线复杂度
- 与最近的 post-hoc calibration 方法（如温度缩放）在有标注数据场景下的对比缺失

## 相关工作与对比

| 方法类型 | 代表工作 | 与 DiNCo 的区别 |
|---------|---------|---------------|
| 语言化置信度 | P(True), Verbalized Numerical | 单次评估，受暗示性偏差影响，置信度饱和 |
| 联合多候选提示 | Top-K-VC, CaCoST | 联合呈现候选允许模型通过算术满足归一化，掩盖不一致 |
| 自一致性 | SC, SC-VC | 仅利用生成一致性，忽略验证维度 |
| 序列概率 | MSP | 依赖标准答案形式，无法扩展到长文本 |
| **DiNCo** | 本文 | 独立评估干扰项 + NLI 重加权 + 生成/验证双维度融合 |

## 总结与展望

DiNCo 从 LLM "暗示性偏差" 这一被忽视的角度出发，通过自动生成干扰项并独立评估置信度来估计和校正偏差，再融合生成与验证两个互补的一致性维度。方法在零资源设定下以极低额外开销（相比 SC 仅多 32% FLOP）实现了跨任务、跨模型的校准改善。未来方向包括：用更小模型生成干扰项以进一步降低成本、将方法扩展到多轮对话和代理决策场景、以及探索与 post-hoc 校准方法的结合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control](../../ACL2026/aigc_detection/reflex_self-refining_explainable_fact-checking_via_verdict-anchored_style_contro.md)
- [\[ICLR 2026\] Death of the Novel(ty): Beyond n-Gram Novelty as a Metric for Textual Creativity](death_of_the_novelty_beyond_n-gram_novelty_as_a_metric_for_textual_creativity.md)
- [\[ICLR 2026\] DMAP: A Distribution Map for Text](dmap_a_distribution_map_for_text.md)
- [\[ICLR 2026\] PoliCon: Evaluating LLMs on Achieving Diverse Political Consensus Objectives](policon_evaluating_llms_on_achieving_diverse_political_consensus_objectives.md)
- [\[ICLR 2026\] CLARC: C/C++ Benchmark for Robust Code Search](clarc_cc_benchmark_for_robust_code_search.md)

</div>

<!-- RELATED:END -->
