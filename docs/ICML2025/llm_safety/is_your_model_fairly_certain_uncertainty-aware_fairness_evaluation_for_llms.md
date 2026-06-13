---
title: >-
  [论文解读] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs
description: >-
  [ICML 2025][LLM安全][公平性评估] 提出不确定性感知的公平性指标 UCerF 和大规模合成数据集 SynthBias，通过联合考虑模型预测正确性与置信度来更细粒度地评估 LLM 的性别-职业偏见。 领域现状：LLM 公平性评估主要依赖 Equalized Odds (EO) 等基于准确率的离散指标…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "公平性评估"
  - "不确定性感知"
  - "性别偏见"
  - "共指消解"
  - "基准数据集"
---

# Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs

**会议**: ICML 2025  
**arXiv**: [2505.23996](https://arxiv.org/abs/2505.23996)  
**代码**: [https://github.com/apple/ml-synthbias](https://github.com/apple/ml-synthbias)  
**领域**: AI安全 / LLM公平性  
**关键词**: 公平性评估, 不确定性感知, 性别偏见, 共指消解, 基准数据集

## 一句话总结
提出不确定性感知的公平性指标 UCerF 和大规模合成数据集 SynthBias，通过联合考虑模型预测正确性与置信度来更细粒度地评估 LLM 的性别-职业偏见。

## 研究背景与动机

**领域现状**：LLM 公平性评估主要依赖 Equalized Odds (EO) 等基于准确率的离散指标，通过比较不同人口群体的正确率差异来衡量偏见程度。WinoBias 是该领域最常用的性别-职业共指消解数据集。

**现有痛点**：基于准确率的指标只关注预测是否正确，忽略了模型的置信度差异。两个模型可能准确率相同，但一个自信地做出偏见预测，另一个则不确定地猜对——在 EO 下它们被视为同样公平，但实际行为差异巨大。同时 WinoBias 规模小（仅 3168 条）、多样性不足，且依赖过时的句法线索设计，不适合评估现代 LLM。

**核心矛盾**：传统公平性指标的离散本质（每个样本只贡献二值的"对/错"）无法捕捉模型决策中的隐性偏见——一个模型可以在某个群体上"自信地正确"但在另一个群体上"勉强猜对"，这种不对称性被准确率指标完全忽略。

**本文目标**：(1) 设计能同时考虑预测正确性和不确定性的公平性指标；(2) 构建更大、更多样、更适合现代 LLM 的公平性评估数据集。

**切入角度**：作者观察到在随机采样解码下（非贪心），模型置信度直接影响输出的偏见程度。一个高置信但充满偏见的模型比一个不确定的模型在实际使用中危害更大。

**核心 idea**：将模型的正确性和不确定性统一到一个连续的"行为期望度"尺度（LSBP）上，然后将公平性定义为不同群体在该尺度上的距离。

## 方法详解

### 整体框架
UCerF 框架分两部分：(1) 一个不确定性感知的公平性指标 UCerF，将模型行为映射到 $[-1, 1]$ 的期望度尺度上再计算群体差异；(2) 一个由 GPT-4o 生成、人工验证的合成数据集 SynthBias（31,756 条样本），用于性别-职业共指消解任务的公平性评估。

### 关键设计

1. **行为期望度线性尺度 (LSBP)**:

    - 功能：将模型的预测正确性和不确定性统一到单一连续尺度上
    - 核心思路：先用 perplexity 估计模型不确定性，归一化得到确定度 $c(x_i) = (k - f_{\text{perplexity}}(x_i; G)) / (k-1) \in [0,1]$，然后根据预测是否正确赋予符号——正确时 $D(x_i) = c(x_i)$，错误时 $D(x_i) = -c(x_i)$，得到 $D \in [-1, 1]$ 的期望度分数
    - 设计动机：传统指标的二值性（0/1）丢失了"低置信命中"和"高置信偏见预测"之间的区别。LSBP 让"自信正确"得高分、"自信错误"得最低分、"不确定"居中，符合直觉

2. **UCerF 公平性指标**:

    - 功能：基于 LSBP 量化两个群体间的公平性差距
    - 核心思路：对每个最小对样本（仅修改代词）计算 $U(x_i) = 1 - \frac{1}{2}|D(x_i^A) - D(x_i^B)|$，然后对整个数据集求期望 $U(\mathbf{X}) = \mathbb{E}[U(x_i)] \in [0,1]$。1 表示完全公平，0 表示完全不公平。还提出了 group-wise 变体 $U_\text{group}$，用 TPD/FPD 替代 TPR/FPR 实现与 EO 的对标
    - 设计动机：相比 EO 只看正确率差异，UCerF 能区分"自信正确+自信错误"（高偏见）vs"自信正确+不确定正确"（中度偏见）等场景，提供更细粒度的公平性评估

3. **SynthBias 数据集**:

    - 功能：提供大规模、多样化的性别-职业共指消解评估数据
    - 核心思路：用 GPT-4o 生成候选样本，覆盖 40 个职业的所有性别刻板印象对组合。重新定义 type1（对人类也歧义）和 type2（人类可消解），而非旧的句法线索分类。经过自动规则过滤和众包标注验证（入门测试 ≥80%，动态覆盖策略到 75% 共识），最终获得 14,132 + 17,624 = 31,756 条验证样本
    - 设计动机：WinoBias 规模太小（仅 3168 条）、模板固定导致多样性不足、type1/type2 定义基于句法线索已不适合理解语义的现代 LLM

### 损失函数 / 训练策略
UCerF 是评估指标，不涉及训练。评估使用 perplexity 作为不确定性估计器，支持替换为其他估计器。

## 实验关键数据

### 主实验

| 模型 | Accuracy (WB) | EO (WB) | UCerF (WB) | Accuracy (SB) | UCerF (SB) |
|------|---------------|---------|------------|----------------|------------|
| Llama-3-70B-Inst | 高 (Top 2) | 排名 1 | 排名 1 | 有所下降 | 排名 3 |
| Mixtral-8x7B-Inst | 高 (Top 3) | 排名 2 | 排名 2 | 有所下降 | 排名 4 |
| Mistral-7B-Inst | 第 4 | 第 5 | **第 8** | 下降 | 下降显著 |
| Pythia-1B | **第 10** | 中 | **第 5** | 稳定 | 稳定 |

SynthBias 上模型准确率普遍低于 WinoBias，说明数据集更具挑战性。

### 消融实验

| 分析维度 | 关键指标 | 说明 |
|---------|---------|------|
| UCerF vs EO 排名差异 | Mistral-7B: EO 第 5 → UCerF 第 8 | 高置信偏见预测被 UCerF 惩罚 |
| 不确定模型 | Pythia-1B: Acc 第 10 但 UCerF 第 5 | 谨慎预测被 UCerF 视为更公平 |
| WB → SB 排名变化 | Llama-3-70B: 第 1 → 第 3 | SynthBias 暴露了 WinoBias 未发现的偏见 |
| Type1 任务 | Llama-3-70B: WB 第 3 → SB 第 8 | 语义歧义时暴露内在偏见 |
| MCQ 任务 | 所有模型 UCerF 提升 | 选项受限使预测更均匀 |

### 关键发现
- Mistral-7B 在准确率和 EO 上表现尚可，但 UCerF 揭示其对偏见预测过度自信，是一种"隐性偏见"
- SynthBias 比 WinoBias 更具挑战性（准确率普遍下降），且能暴露更多公平性差异——如 Llama-3-70B 在 WinoBias 上排名第一，但在 SynthBias 上降至第三
- 在 type1（无正确答案）任务中，SynthBias 的人类验证歧义性更严格，能隔离 LLM 的语义理解能力，单独评估其内在偏见

## 亮点与洞察
- **行为期望度尺度 (LSBP)** 将正确性和不确定性统一到单一连续维度上，这个思路优雅且通用——可迁移到任何需要联合评估"对不对"和"确不确定"的场景（如医疗诊断AI的公平性）
- **UCerF 揭示隐性偏见**：即使模型都答对了，置信度的不对称仍体现偏见，这种洞察是EO完全捕捉不到的
- SynthBias 的数据生成+人工验证流程（GPT-4o 生成 → 规则过滤 → 众包标注 → 严格共识阈值）是高质量合成评估数据集的良好范式

## 局限与展望
- 仅关注二元性别代词（his/her），未涉及 they/them 等性别中性代词
- 职业偏见基于美国劳动统计局数据，具有地域局限性
- 不确定性估计仅使用 perplexity，更复杂的估计器（如 MC Dropout、语义熵）可能带来不同结论
- 数据集由 GPT-4o 生成，可能继承其自身偏见
- 未探索 UCerF 在其他偏见类型（种族、年龄）和其他任务（文本生成、QA）上的适用性

## 相关工作与启发
- **vs WinoBias**: WinoBias 基于模板、规模小、依赖句法线索；SynthBias 用 LLM 生成、规模大 10 倍、基于语义歧义，更适合评估现代 LLM
- **vs Equalized Odds**: EO 是离散的（只看对/错），UCerF 是连续的（考虑置信度），能发现 EO 遗漏的"高置信偏见"和"低置信公平"
- **vs Kuzucu et al. (2023)**: 之前工作只单独比较两个群体的不确定性，UCerF 联合考虑不确定性和正确性，处理更复杂的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 将不确定性融入公平性指标是新颖的切入点，LSBP 尺度设计优雅
- 实验充分度: ⭐⭐⭐⭐ 10 个 LLM、两个数据集、intrinsic+MCQ+CoT 多任务评估，案例分析详尽
- 写作质量: ⭐⭐⭐⭐⭐ 图文并茂，用 Fig.1 的直觉示例引入问题，层层递进
- 价值: ⭐⭐⭐⭐ 提供了更细粒度的公平性评估工具，SynthBias 数据集有实际应用价值
---
title: >-
  [论文解读] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs
description: >-
  [ICML 2025][AI安全][fairness] 提出不确定性感知的公平性指标 UCerF，以及大规模性别-职业偏见评估数据集 SynthBias（31,756样本），通过联合分析预测正确性与模型不确定性来更精细地评估LLM的内在偏见。
tags:
  - ICML 2025
  - AI安全
  - fairness
  - uncertainty
  - LLM bias
  - gender-occupation bias
  - co-reference resolution
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](../../NeurIPS2025/llm_safety/evaluation_of_vision-llms_in_surveillance_video.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](../../ACL2025/llm_safety/comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)

</div>

<!-- RELATED:END -->
