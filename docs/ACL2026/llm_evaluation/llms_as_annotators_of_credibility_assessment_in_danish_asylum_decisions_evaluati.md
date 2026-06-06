---
title: >-
  [论文解读] LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics
description: >-
  [ACL 2026][LLM评测][LLM-as-annotator] 在丹麦难民上诉委员会 (RAB) 的 273 份庇护决定文书上构建了 RAB-Cred 这个三类 ("Absent / Positive / Negative") 可信度评估专家标注数据集…
tags:
  - "ACL 2026"
  - "LLM评测"
  - "LLM-as-annotator"
  - "难民庇护"
  - "可信度评估"
  - "提示工程"
  - "错误分析"
  - "集成投票"
---

# LLMs as annotators of credibility assessment in Danish asylum decisions: evaluating classification performance and errors beyond aggregated metrics

**会议**: ACL 2026  
**arXiv**: [2605.13412](https://arxiv.org/abs/2605.13412)  
**代码**: https://github.com/glhr/RAB-Cred (有)  
**领域**: LLM 标注 / 法律 NLP  
**关键词**: LLM-as-annotator, 难民庇护, 可信度评估, 提示工程, 错误分析, 集成投票

## 一句话总结
在丹麦难民上诉委员会 (RAB) 的 273 份庇护决定文书上构建了 RAB-Cred 这个三类 ("Absent / Positive / Negative") 可信度评估专家标注数据集，系统跑了 21 个开源 LLM × 30 种 system×user prompt 组合，发现 prompt 设计比模型选择更重要、phi-4 (14B) 在零样本下能拿 94.7% F1 但单模型一致会犯"不可接受"的错，于是建议用"15 个最优模型-prompt 组合多数投票"的 ensemble，正确率能再涨 1.5 pp 到 96%。

## 研究背景与动机

**领域现状**：用 LLM 做"开箱即用文本标注员"已经在社会学、人文学科里全面铺开，法律 NLP 也有零星尝试（argument mining、legal interpretation），但绝大多数研究只测一两个 prompt × 一两个闭源大模型（如 GPT-4），导致"模型就这水平"或"prompt 就这影响"的结论高度不可靠。

**现有痛点**：作者要做的"识别庇护决定里是否做了可信度评估、做了的话是 positive 还是 negative"任务有三个一起出现的硬件：(1) 概念本身就模糊——可信度评估在难民法学界都没共识；(2) 跟"风险评估"高度同源、表述相邻、容易混淆——同一份文书可以包含两种判断；(3) 语言是丹麦语 (medium-resource) + 专业法律术语，主流开源 LLM 在丹麦语上的训练数据占比都很低 (Phi-4 自承多语数据仅 8%、Llama 3 官方只支持 8 种语言不含丹麦语)。

**核心矛盾**：领域专家既要"自动标注尽可能准"，更要"知道 LLM 在哪些案例上、以什么方式出错"——单一聚合指标 (Macro-F1) 完全没法回答后者。已有 LLM-as-annotator 文献几乎全用 dataset-level F1 评估，掩盖了 (a) 跨模型的错误一致性、(b) 跟人类标注信心的关联、(c) 错误的严重程度区分。

**本文目标**：(1) 在 under-represented 语言+领域上建一个高质量、带元数据 (标注员信心、案件结果) 的法律分类基准 RAB-Cred；(2) 系统 benchmark 21 个开源模型 × 30 个 prompt 组合，量化模型 vs prompt 谁更关键；(3) 不止看 F1，还要 zoom in 看错误的类型、跨模型一致性、与人类信心的关联、严重程度。

**切入角度**：把 LLM 标注员当成"会犯错的专业实习生"——既要看 aggregate accuracy，又要做错误根因分析；引入 instance-level PromptSensiScore (PSS) 来量化"换 prompt 时单样本预测会不会翻转"，区分 dataset-level 稳定但 instance-level 不稳的"伪鲁棒"模型 (典型如 Qwen3-30B)。

**核心 idea**：用"15 个最优 model-prompt 组合的多数投票"替代任意单一 LLM 标注员，并用专家事后给错误打三档严重度 (acceptable/understandable/unacceptable) 来回答"LLM 能在多大程度上替代人类专家"这个真问题。

## 方法详解

### 整体框架
四步流程：(1) **数据集构建**——从 RAB 官网爬 10817 份公开决定文书，分层采样 73 份 validation + 200 份 test，两位丹麦难民法专家 (H1, H2) 独立标注三类标签 + Low/Medium/High 信心，分歧 case 由 H3 决断，Cohen's $\kappa = 0.97$。(2) **Model × Prompt 网格**——21 个开源 LLM (3B-35B 参数, 9 个模型族) × (6 个 system prompt × 5 个 user prompt = 30 种组合)，每个组合在 70 个 val 样本上跑一次 greedy decoding + outlines 库做约束解码，确保输出落在合法类别集合 $\{\text{NO}, \text{POSITIVE}, \text{NEGATIVE}\} \times \text{CREDIBILITY ASSESSMENT}$。(3) **Top-15 标注员选择**——每个模型按其 top-3 prompt 的平均 Macro-F1 排序、取 top-5 model × 各自 top-3 prompt = 15 个 LLM annotator 上 test set。(4) **细粒度错误分析**——按 case-by-case 统计 ensemble agreement、PSS 量化敏感度、专家给 ensemble 错误打严重度。整个 pipeline 在单卡 H100 (80GB) 上完成。

### 关键设计

1. **三层嵌套 system prompt + 五档 user prompt 的 6×5 正交网格**:

    - 功能：把"领域知识注入"和"任务推理结构"两个维度解耦，避免 prompt 影响被混在一起。
    - 核心思路：System prompt 按嵌套层次递增——SP0 (空) → SP1 (专家 persona) → SP2 (SP1 + 原始 codebook) → SP3 (SP2 重写 + 每类典型丹麦语短语) → SP4 (SP3 + 边界 case：假设性法律构造、混合情感) → SP5 (SP1 + 专家手写"可信度 vs 风险评估 vs 中性陈述"消歧)；User prompt 按推理复杂度递增——UP1 (直接三选一) → UP1-FS (UP1 + 3 个 few-shot 样本，每类一个最难案例) → UP2 (两步：先二分类"有无"、再二分类"正/负"，对应人类标注结构) → UP3 (zero-shot CoT, Kojima et al. 2022) → UP4 (zero-shot metacognitive prompting, Wang & Zhao 2024)。所有 prompt 都是英文写、丹麦文输入。
    - 设计动机：在 LLM-as-annotator 文献里 prompt 影响往往被一句"prompt is task-specific"带过，作者要量化这种 task-specific 性——结果发现 SP3/SP4 (CS 专家+领域专家共写) > SP2/SP5 (纯领域专家写)，证明 interdisciplinary prompt design 真的有效；而 UP 影响**极度模型相关**——Phi-4 用 UP2 拿到全场最高分 (val 90.51% F1)、Gemma-3 用同一 UP2 反而崩盘 60+%。

2. **Top-3 prompt 平均 + 15 标注员集成投票**:

    - 功能：用一组多样化但都"足够好"的 LLM 标注员取众数，模拟人类多专家共识标注。
    - 核心思路：先在 val set 上对每个模型按其 top-3 prompt 的平均 Macro-F1 排序，取 top-5 模型 (phi-4 / gemma-3-27b-it / Ministral-3-14B / Mistral-Small-24B / Qwen3-30B) × 各自 top-3 prompt = 15 个 LLM annotator；test set 上跑这 15 个组合，对每个 case 取多数票作为 ensemble 预测。这套选法保证了"模型架构多样性 + prompt 多样性"两个轴都覆盖，SP5+UP2 是出现最多的组合 (3 次)。
    - 设计动机：单模型最高 test F1 是 phi-4+SP4+UP4 的 94.7%，但所有 15 个标注员都至少犯一次"不可接受错误"——单点失败不可避免。集成投票后 accuracy 跳到 96% (+1.5pp)，且对剩下 8 个错误 case，专家 H1 评估发现 4 个"可接受"、2 个"可理解"、只剩 2 个"不可接受"，证明 ensemble 真能把"严重错误"率压下去。

3. **Instance-level PromptSensiScore (PSS) 区分 dataset 稳和 instance 稳**:

    - 功能：传统 prompt 敏感度只看 aggregate F1 在不同 prompt 下的方差，会漏掉"F1 没变但具体预测翻转"的情况。
    - 核心思路：借用 Zhuo et al. (2024) 的 PSS——固定一组 prompts $\mathcal{P}$，对每个 case $x_i$ 统计在 $\mathcal{P}$ 内不同 prompt 下预测正确率的变化，记 PSS$(x_i)$ 为该 case 的预测稳定性。作者做两种切片：(a) **固定模型、变 prompt** 测 prompt sensitivity (Phi-4 PSS=0.043 最低、Qwen3-30B PSS=0.110 最高)；(b) **固定 prompt (SP5+UP2)、变模型** 测 model sensitivity (PSS=0.05，比同模型变 prompt 的 PSS 还低)。两个对比直接给出结论：**prompt 变化引入的不稳定性 > 模型变化**。
    - 设计动机：Qwen3-30B 的"aggregate 稳定但 instance 不稳"是非常隐蔽的陷阱——你看着每个 prompt 都跑出 F1 ≈ 87.9%，会以为它对 prompt 鲁棒，但其实它在不同 prompt 下错的 case 是不同的，等于"靠运气抵消"。PSS 把这种陷阱挑出来，对依赖 LLM 做大规模标注的科研社区是非常有用的诊断工具。

### 损失函数 / 训练策略
本文是**纯 zero/few-shot benchmark**，没有训练任何模型；唯一可调的"训练"是 few-shot 示例选择——选每类一个，标准是"领域专家高 confidence 但 LLM 零样本错最多"的难 case，目的是把信息密度最高的样本塞进 prompt。约束解码用 outlines 库强制 LLM 输出落在合法 schema $\text{Literal}[\text{"NO/POSITIVE/NEGATIVE CREDIBILITY ASSESSMENT"}]$ 上；5 个不支持 outlines 的模型 (Mistral 系/EuroLLM/Bielik) 改用 regex 后处理。

## 实验关键数据

### 主实验
21 个模型 × 30 prompt × 70 val 样本 = 630 组合，取 top-15 上 test set (200 样本)。Test set F1 范围 84.4% – 94.7%，outcome-as-classifier baseline 53%。代表性数字：

| 模型-Prompt | val Macro-F1 | test Macro-F1 | test Accuracy | 备注 |
|------------|--------------|---------------|---------------|------|
| phi-4 + SP4 + UP4 | 86.34 | **94.69** | **94.50** | 单模型 SOTA |
| phi-4 + SP4 + UP2 | **90.51** | 93.66 | 94.00 | val 最高 |
| Mistral-Small-24B + SP3 + UP1-FS | 85.63 | 92.52 | 92.50 | few-shot 最强 |
| Ministral-3-14B + SP4 + UP3 | 86.59 | 91.63 | 92.00 | CoT 最强 |
| Gemma-3-27b + SP4 + UP1-FS | 79.91 | 84.39 | 86.50 | 选中里最弱 |
| Qwen3-30B + SP5 + UP2 | 83.61 | 87.95 | 89.00 | 大模型不一定强 |
| **Ensemble (15 LLMs, majority vote)** | – | – | **96.00** | +1.5 pp 单模型 SOTA |
| Outcome-as-classifier baseline | – | – | 53.00 | 朴素基线 |
| Human H1 vs H2 | – | – | 98.4 ($\kappa=0.967$) | 上限 |

跨模型大小不等价于性能：Qwen2.5-32B / Qwen3-30B / aya-expanse-32B 都 plateau 在 ≤85% F1，远输于 14B 的 Phi-4；Phi-4 模型卡明确说"不为多语言设计、多语数据仅 8%"反而拿了第一，作者推测可能是高质量数据 > 多语覆盖。Inter-LLM Cohen $\kappa$ 最高仅 0.950 (两个 Gemma 同 UP 不同 SP)，远低于人类 H1-H2 的 0.967，说明 LLM 标注员之间的一致性还赶不上人类专家。

### 消融实验
错误结构与组件贡献 (test set 200 个 case)：

| 配置 / 分析 | 关键数字 | 说明 |
|------------|---------|------|
| 全 15 LLM 都对 | 144 / 200 = 72% | "简单"案例 |
| ≥ 8 LLM 都对 (多数票) | 190 / 200 = 95% | 多数投票天花板 |
| 每个 LLM 都至少 1 次"不可接受错误" | 15 / 15 | 单模型必然失误 |
| Ensemble 多数票错误数 | 8 / 200 = 4% | 比单模型 SOTA 少 1.5pp |
| Ensemble 错误的专家严重度判定 | 4 acceptable / 2 understandable / 2 unacceptable | 50% 是"专家也会犹豫"的边界 |
| Phi-4 的 instance-level PSS | 0.043 | 最稳 |
| Qwen3-30B 的 instance-level PSS | 0.110 | aggregate 稳但 instance 最不稳 |
| Prompt sensitivity 平均 | ~0.08 | > model sensitivity 0.05 |
| Gemma-3-27B F1 跨 prompt 方差 | 54% | 最不鲁棒 |
| Qwen2.5-32B F1 跨 prompt 方差 | < 5% | 最鲁棒（但小模型也行） |

### 关键发现
- **Prompt 比模型重要**：固定 prompt 换模型的 PSS (0.05) < 固定模型换 prompt 的 PSS (0.06-0.11)，说明在 LLM 标注研究里只 benchmark 一个 prompt 等于在掷骰子；多 prompt 评估应该是标配。
- **大模型 ≠ 好模型**：Phi-4 (14B) 拿了全场最高分，>30B 的 Qwen2.5-32B、Qwen3-30B、aya-expanse-32b 都≤85% F1。Phi-4 模型卡上明确说"多语数据仅 8% 不为多语言设计"，反而在丹麦语任务上吊打号称"30+ 欧洲语言"的 EuroLLM-22B。数据质量 > 数据量 + 多语覆盖。
- **CoT/metacog 收益高度依赖 SP**：UP3/UP4 只在 SP0/SP1 (无 codebook) 下比 UP1 强；当 SP 已经包含完整 codebook 时，加 CoT 几乎没用。
- **二分类拆解 (UP2) 收益模型相关**：把"三选一"拆成"先判有无、再判正负"对 Phi-4 是涨点利器，但对 Gemma-3 反而是毒药；不能假设"任务分解一定好"。
- **15 个 LLM 标注员错的 case 大多在专家信心低的边界 case 上**：所有 <75% LLM 标对的 case，都至少有一位专家信心为 Low/Medium——LLM 的失败模式跟人类的犹豫高度重合，说明数据集本身这些 case 就有内在歧义，不是模型缺陷。两个被 14/15 LLM 错标的 case 经 H1 复看后甚至"想改自己原标"，证明数据天花板低于 100%。

## 亮点与洞察
- **"专家事后给 LLM 错误打严重度"是高密度评估范式**：把"是否答对"升级成"答错了的话有多严重"，acceptable 错误 ≈ 边界 case 模糊 ≠ 真错，这种区分对实际部署比纯 F1 有用得多。可以推广到任何"专家高度参与"的 LLM 应用 (医疗诊断、内容审核、法律辅助)。
- **PSS 区分 aggregate-stable 和 instance-stable**：揭穿了 Qwen3-30B 这种"看起来很稳但其实换 prompt 就错不同 case"的伪鲁棒模型。这种 metric 对依赖 LLM 做研究的人是诊断工具——你的"稳定性"可能只是 noise 抵消的运气。
- **interdisciplinary prompt design 才赢**：纯领域专家 (SP2/SP5) 写的 prompt 不如 CS+法律共写 (SP3/SP4)。给 LLM 应用项目的启示：prompt 工程不能只丢给单一背景的人。
- **"15 个标注员投票"作为 cost-effective 人替代方案**：单跑 1 个 LLM (任意选) 风险高，但 15 个不同 model-prompt 组合的 majority vote 把 unacceptable 错误压到 1%，等于花 15× 推理成本买专家级可靠性，相对于雇 3 个领域专家做 1 万份文书标注还是便宜得多。
- **小模型 + 高质量训练 > 大模型 + 多语广撒网**：Phi-4 (14B, 多语数据 8%) 在丹麦语任务上击败 EuroLLM-22B、Qwen3-30B，给"如何为低资源语言挑 LLM"提供了反直觉证据。

## 局限与展望
- **数据集只有 273 case (val 73 + test 200 + 3 few-shot)**，作者承认对"全 1 万+ RAB 案例的多样性"统计代表性有限；模型间细微差异 (差 1-2% F1) 在 200 case 上没显著性。
- **只用开源 ≤35B 模型**——闭源大模型 (GPT-4o, Claude) 没测，因为法律敏感数据要求 offline；这是工程必要但确实留了上限的悬念。
- **单 run + greedy decoding**——没做多 seed 实验，约束解码本身也可能损害生成质量 (作者引用 Schall & de Melo 2025)。
- **English prompt + Danish input 这一跨语言选择没有充分对照**——作者承认全用丹麦语 prompt 或把文书译英可能换出不同错误谱。
- **没分析 CoT/metacog 产生的 reasoning trace 是否对齐专家推理**——只用了最终 label，留下了"LLM 是不是凭对的理由答对"这个未答的问题。
- **没做时间漂移分析**——RAB 决定文书跨 2004-2025 二十年，写作风格和匿名化策略明显变化，模型在不同时段表现是否有差异未被分析。

## 相关工作与启发
- **vs AsyLex (Barale et al. 2023)**: 唯一一个公开难民法 NLP 数据集，但是英文的；RAB-Cred 填了"低资源语言 + 难民法"的空白，且首次包含 expert confidence + 案件 outcome 元数据。
- **vs 通用 LLM-as-annotator 文献 (Pavlovic & Poesio 2024; Halterman & Keith 2025)**: 后者要么只比少数 (常为闭源) 模型，要么只看 aggregate 指标；本文用 21 模型 × 30 prompt + instance-level 错误分析 + 严重度评估三轴扩展。
- **vs Halterman & Keith (2025) "Codebook LLMs"**: 都用专家 codebook 喂 LLM 做标注，但本文进一步对比了不同 SP 的递进 (有/无 codebook, 有/无 edge case, 不同重组)，证明"光丢 codebook 不够、还得改写 + 加边界 case"。
- **vs PromptSensiScore (Zhuo et al. 2024)**: 借用了 PSS 工具但首次把它用在 LLM-as-annotator 语境，并发明"固定 prompt 变模型 vs 固定模型变 prompt"的对比切片，得出"prompt > model"结论。
- **启发**：(1) **法律/医疗等高 stakes 领域**应该把"严重度分类的错误分析"做成标配，单 F1 评估不可接受；(2) **任何依赖 LLM 大规模标注的研究**应至少跑 2-3 个 prompt + 2-3 个模型组合，并用 inter-LLM Cohen $\kappa$ 检查一致性；(3) **小模型 + 高质量数据**对低资源语言可能是更好的路径——不一定要堆参数、堆多语覆盖。

## 评分
- 新颖性: ⭐⭐⭐⭐ 数据集是新的 (Danish RAB 可信度评估)、评估范式 (15-LLM ensemble + 严重度分类) 是组合创新；PSS 在 LLM 标注语境的应用属于增量；总体是 well-designed 的应用研究。
- 实验充分度: ⭐⭐⭐⭐⭐ 21 模型 × 30 prompt × 70 val + 200 test 全跑了，外加专家 case-by-case 严重度评估、inter-LLM $\kappa$ 矩阵、instance-level PSS、时间分层采样、outcome-as-classifier 基线，几乎想到的对照都做了。
- 写作质量: ⭐⭐⭐⭐⭐ 法律背景介绍清晰、Table 1 的真实丹麦案例翻译 + 标注示例非常直观、"哪些 case 难、为什么难"讲得透；Appendix 把 prompt 全文、regex、约束解码代码都附上，可复现度极高。
- 价值: ⭐⭐⭐⭐ 为社科/法律领域用 LLM 做标注提供了可借鉴的方法论框架；RAB-Cred 数据集对 Danish NLP 社区是稀缺资源；缺点是结论 (prompt > model, ensemble 涨点) 在通用 LLM-as-annotator 语境里部分已知，迁移性可能不如方法论本身。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SCAN: Structured Capability Assessment and Navigation for LLMs](scan_structured_capability_assessment_and_navigation_for_llms.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)

</div>

<!-- RELATED:END -->
