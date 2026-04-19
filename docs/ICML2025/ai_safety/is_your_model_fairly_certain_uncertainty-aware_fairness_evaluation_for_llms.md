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

# Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs

**会议**: ICML 2025  
**arXiv**: [2505.23996](https://arxiv.org/abs/2505.23996)  
**代码**: [github.com/apple/ml-synthbias](https://github.com/apple/ml-synthbias)  
**领域**: AI安全 / 公平性  
**关键词**: fairness, uncertainty, LLM bias, gender-occupation bias, co-reference resolution

## 一句话总结

提出不确定性感知的公平性指标 UCerF，以及大规模性别-职业偏见评估数据集 SynthBias（31,756样本），通过联合分析预测正确性与模型不确定性来更精细地评估LLM的内在偏见。

## 研究背景与动机

- LLM在高风险决策中日益普及，但传统公平性指标（如 Equalized Odds）仅关注离散的准确率差异，忽略了模型不确定性对公平性的影响。
- **核心洞察**：两个模型可能有相同的准确率和 EO 分数，但一个对所有群体都高置信正确，另一个虽然正确但对某一群体极度不确定——后者显然更有偏见风险，尤其在高温采样场景中。
- 现有数据集 WinoBias 存在规模小（3168样本）、多样性差、语义歧义定义不适合现代LLM等问题。

## 方法详解

### 整体框架

1. **线性行为偏好量表（LSBP）**：统一正确性和不确定性到一个连续尺度 $D(x_i) \in [-1, 1]$。
2. **UCerF 指标**：基于 LSBP 衡量两个群体之间的行为差异。
3. **SynthBias 数据集**：使用 GPT-4o 生成、人工标注验证的大规模公平性评估数据集。

### 关键设计

**归一化确定性**：

$$c(x_i) = \frac{k - f_{\text{perplexity}}(x_i; G)}{k - 1} \in [0, 1]$$

其中 $k$ 为可能的预测结果数（如职业数量），$f_{\text{perplexity}}$ 为困惑度。

**行为可取性（Desirability）**：

$$D(x_i) = \begin{cases} -c(x_i), & \text{预测错误} \\ c(x_i), & \text{预测正确} \end{cases}$$

**UCerF 指标**：

$$U(x_i) = 1 - \frac{1}{2}|D(x_i^A) - D(x_i^B)|$$

$$U(\mathbf{X}) = \mathbb{E}_{x_i \in \mathbf{X}}[U(x_i)] \in [0, 1]$$

其中 $x_i^A, x_i^B$ 是最小对（如同一句子中将代词替换为 his/her）。UCerF=1 表示完美公平，0 表示完全不公平。

### SynthBias 数据集

- 使用 GPT-4o 生成，覆盖40种职业的所有组合对。
- **type1**：对人类也有歧义的句子（无法仅凭上下文解析代词）。
- **type2**：可明确解析的句子。
- 通过众包平台验证：20%的测试入门筛选（≥80%通过）、动态覆盖策略（至少4人标注且75%共识）。
- 最终：14,132条 type1 + 17,624条 type2 = **31,756条**。

## 实验关键数据

### 基准评估（10个开源LLM）

在 SynthBias type2 任务上的代表性结果：

| 模型 | 准确率 | EO排名 | UCerF排名 | 说明 |
|------|:-----:|:-----:|:-------:|------|
| Mistral-7B | 第4 | 第5 | 第8 | 高置信偏见预测，UCerF 更好地捕捉到问题 |
| Pythia-1B | 第10 | - | 第5 | 虽准确率低但预测谨慎，UCerF 给予更高公平评价 |
| Mixtral-8x7B | 高 | 高 | 高 | 高置信且公平，UCerF与EO一致 |
| Llama-3-70B | 第1(WB) | - | 第3(SB) | SynthBias 挑战样本揭示隐藏偏见 |

### 关键发现

- **UCerF vs EO 排名差异显著**：Mistral-7B 在 EO 看起来公平（第5），但 UCerF 揭示其高置信偏见预测使其实际更不公平（第8）。
- **SynthBias 比 WinoBias 更具挑战性**：模型在 SynthBias 上准确率更低，公平分平均下降6%。
- **Per-occupation 分析**：LLM 在性别比例极端的职业上偏见最严重，中间职业最公平——模型复刻了现实世界的统计偏见。
- **EO 可能夸大偏见**：当模型预测错误但不确定时，EO记录为完全不公平（TPR差异=1），而 UCerF 考虑到不确定性后给出更温和的评估（U≈0.797）。

## 亮点与洞察

1. **UCerF 填补了准确率与不确定性之间的评估空白**：传统公平性指标完全忽略模型confidence，而这在高温采样（实际部署常见场景）中至关重要。
2. **LSBP 量表的直觉设计**：将"高置信错误"视为最差行为、"不确定"为中性、"高置信正确"为最优，简洁而有效。
3. **SynthBias 的构建流程值得借鉴**：LLM生成 + 规则过滤 + 众包验证的三阶段流水线，确保数据质量。
4. **可拓展性好**：UCerF 可扩展到非二元群体（使用标准差替代绝对差），不确定性估计器也可替换。

## 局限性

- 仅使用困惑度作为不确定性估计器，更复杂的方法（如 MC Dropout、语义熵）可能提供更准确的估计。
- 聚焦于美国劳工统计局的性别-职业偏见，存在文化/地域局限性。
- 仅研究二元性别代词（he/she），未涉及性别中性代词（they/them）——虽然WinoGender涉及中性代词，但单复数歧义使其难以准确评估。
- SynthBias 由 GPT-4o 生成，可能继承其训练数据偏见。
- 公平性与性能是正交维度：Pythia-1B 虽然在 UCerF 上公平（因为不确定），但准确率极低，实际不可用。论文在附录中提供了联合评估方案。
- UCerF 依赖最小对（minimal pairs），对于没有天然配对的公平性数据集需要使用 group-wise 变体。

## 相关工作

- **公平性评估**：Equalized Odds (Hardt et al., 2016)、Demographic Parity；WinoBias (Zhao et al., 2018)、BBQ (Parrish et al., 2021)、WinoGender (Rudinger et al., 2018)。
- **LLM不确定性估计**：困惑度、MC Dropout (Gal & Ghahramani, 2016)、语义熵 (Kuhn et al., 2023)、P(True) (Kadavath et al., 2022)、Prior Networks (Malinin & Gales, 2018)。
- **不确定性+公平性交叉**：Kuzucu et al. (2023) 首次考虑不确定性群体差异但未联合分析正确性；Kaiser et al. (2022) 在视觉/表格数据上的尝试；Kuzmin et al. (2023) 将公平性和可靠性作为两个独立指标研究。
- **合成数据生成**：Guo & Chen (2024) 综述LLM生成合成数据的方法与挑战。
- **CoT与MCQ**：论文附录还评估了链式思维和多选题格式对公平性的影响，发现MCQ限制答案空间后UCerF一致提升。

## 评分

⭐⭐⭐⭐ — 指标设计简洁直观，SynthBias 数据集实用性强。通过10个LLM的系统性基准测试，令人信服地展示了考虑不确定性对公平性评估的价值。不足在于实验仅关注单一偏见维度。

<!-- RELATED:START -->

## 相关论文

- [Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [Incentivizing Time-Aware Fairness in Data Sharing](../../NeurIPS2025/ai_safety/incentivizing_time-aware_fairness_in_data_sharing.md)
- [PULSE: Practical Evaluation Scenarios for Large Multimodal Model Unlearning](../../NeurIPS2025/ai_safety/pulse_practical_evaluation_scenarios_for_large_multimodal_model_unlearning.md)
- [Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](../../NeurIPS2025/ai_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)
- [FairI Tales: Evaluation of Fairness in Indian Contexts with a Focus on Bias and Stereotypes](../../ACL2025/ai_safety/fairi_tales_evaluation_of_fairness_in_indian_contexts_with_a_focus_on_bias_and_s.md)

<!-- RELATED:END -->
