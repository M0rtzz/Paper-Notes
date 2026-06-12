---
title: >-
  [论文解读] Is Your Model Fairly Certain? Uncertainty-Aware Fairness Evaluation for LLMs
description: >-
  [ICML 2025][LLM安全][fairness] 提出不确定性感知的公平性指标 UCerF，以及大规模性别-职业偏见评估数据集 SynthBias（31,756样本），通过联合分析预测正确性与模型不确定性来更精细地评估LLM的内在偏见。
tags:
  - "ICML 2025"
  - "LLM安全"
  - "fairness"
  - "uncertainty"
  - "LLM bias"
  - "gender-occupation bias"
  - "co-reference resolution"
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

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving Your Model Ranking on Chatbot Arena by Vote Rigging](improving_your_model_ranking_on_chatbot_arena_by_vote_rigging.md)
- [\[NeurIPS 2025\] Evaluation of Vision-LLMs in Surveillance Video](../../NeurIPS2025/llm_safety/evaluation_of_vision-llms_in_surveillance_video.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](../../ACL2025/llm_safety/comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[NeurIPS 2025\] Attention! Your Vision Language Model Could Be Maliciously Manipulated](../../NeurIPS2025/llm_safety/attention_your_vision_language_model_could_be_maliciously_manipulated.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)

</div>

<!-- RELATED:END -->
