---
title: >-
  [论文解读] StrucText-Eval: Evaluating LLM's Reasoning on Structure-Rich Text
description: >-
  [ACL 2025 (Long Paper)][LLM评测][结构化文本推理] 提出 StrucText-Eval——通过自动生成语义无关的结构化文本样本，覆盖 8 种结构化语言和 29 个任务共 5,800 个样本，以可控的嵌套深度和宽度调节难度…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM评测"
  - "结构化文本推理"
  - "benchmark自动生成"
  - "复杂度可控评测"
  - "语义无关评测"
  - "LLM能力边界"
---

# StrucText-Eval: Evaluating LLM's Reasoning on Structure-Rich Text

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2406.10621](https://arxiv.org/abs/2406.10621)  
**代码**: [MikeGu721/StrucText-Eval](https://github.com/MikeGu721/StrucText-Eval)  
**领域**: LLM评测  
**关键词**: 结构化文本推理, benchmark自动生成, 复杂度可控评测, 语义无关评测, LLM能力边界  

## 一句话总结
提出 StrucText-Eval——通过自动生成语义无关的结构化文本样本，覆盖 8 种结构化语言和 29 个任务共 5,800 个样本，以可控的嵌套深度和宽度调节难度，揭示最强开源 LLM 在困难集上仅 45.8% 而人类达 92.6%，系统性暴露了 LLM 在纯结构推理上的严重短板。

## 研究背景与动机

**领域现状**：结构化数据（JSON、YAML、XML、Markdown、LaTeX 等）在企业数据场景中无处不在。随着 LLM 在非结构化文本理解上取得巨大进步，一个自然的问题是：LLM 能否直接从纯文本形式理解和推理这些结构化数据？已有研究在图、表格、JSON 等有限类别上做了一些探索，但覆盖面窄且评测体系不完善。

**现有痛点**：现有结构化文本理解基准存在三个核心缺陷。第一，覆盖范围有限，只涉及图、表格、JSON 等少数格式，忽略了 LaTeX、Markdown、YAML、Org、自定义树等广泛使用的结构化语言。第二，多数基准依赖人工标注，制约了评测框架的可扩展性。第三，固定数据集容易被纳入训练集导致模型"作弊"，无法持续有效地评估能力。

**核心矛盾**：现有评测无法将"语义理解"和"纯结构推理"解耦。当任务中包含有意义的文本内容时，模型可以利用语义先验来"抄近路"，掩盖其在结构解析上的真实能力。同时，缺乏难度可控的生成机制使得基准无法精准定位不同复杂度下的能力边界。

**本文目标** (1) 如何设计覆盖面广、自动化、难度可控的结构化文本推理评测？(2) LLM 的纯结构推理能力究竟有多差？与人类的差距有多大？(3) 不同结构化语言和任务类型之间的难度差异由什么决定？

**切入角度**：作者观察到如果用无意义字符串填充结构化数据的语义字段，就能迫使模型只依靠结构标记本身进行推理，从而纯粹测试"结构解析能力"。同时通过参数化控制嵌套深度（depth）、宽度（width）和列数（column），可以精确生成不同难度的评测数据。

**核心 idea**：用语义无关的自动生成方法构建覆盖 8 种结构化语言 × 29 个任务的可控难度基准，纯粹评测 LLM 的结构推理能力。

## 方法详解

### 整体框架
StrucText-Eval 的核心是一套自动评测数据生成流水线。输入是任务类型、目标语言以及复杂度参数（depth/width/column），输出是一个包含四字段的样本——Reference（结构化文本）、Question（问题）、Requirement（要求）、Answer（标准答案）。整个流程分为五步：定义复杂度参数 → 构建抽象结构树 → 套用问题模板 → 通过规则算法计算 ground truth → 将抽象树和答案翻译为目标结构化语言。最终汇集为两个套件：Test（3,712 样本，depth 1-2）和 Test-Hard（2,088 样本，depth/width 最高 3×3，平均长度 16,535 字符，最长达 102,531 字符）。

### 关键设计

1. **语义无关的结构化文本生成**:

    - 功能：生成纯粹测试结构解析能力的评测样本，排除语义捷径
    - 核心思路：在构建抽象结构树时，所有叶节点的内容用无意义随机字符串填充，而非真实语义文本。模型必须通过理解结构标记（如 JSON 的花括号、YAML 的缩进、XML 的标签嵌套）来完成推理，无法依赖"常识"或"语义关联"推测答案。复杂度通过三个参数精确控制：depth 控制嵌套层数，width 控制每个非叶节点的子节点数，column 控制每个节点的字段数
    - 设计动机：现有基准使用真实语义数据，模型可能通过语义先验"猜对"答案而非真正理解结构。语义无关设计是与所有前人工作的根本区别，确保评测结果反映的是纯结构理解能力

2. **8 语言 × 29 任务的分类学覆盖**:

    - 功能：提供系统性的结构化语言和任务覆盖，揭示不同格式和任务类型的难度差异
    - 核心思路：将结构化文本分为两大类——结构化数据（Tree 自定义格式、Tabular/CSV）和半结构化数据（Object Notation: JSON/YAML/XML; Markup Language: Markdown/LaTeX/Org）。在此基础上设计 8 大类 29 个任务，包括 PathCompose（层次路径推理）、PathWalk（章节提取）、TextRetrieval（信息检索）、Syntax（语法错误检测）、Statistic（条件统计）、Join（SQL 式多表联合）、Tree.Height（树高计算）和 Node.Depth（节点深度计算），覆盖从简单信息定位到复杂多步推理的完整难度谱
    - 设计动机：前人工作仅涉及 1-2 种格式和少数任务，无法系统性评估不同结构语言间的能力差异。8×29 的网格式覆盖可支持多维度对比分析

3. **双套件难度梯度设计**:

    - 功能：区分不同层级模型的能力，并量化 LLM 与人类的能力鸿沟
    - 核心思路：Test 套件包含 3,712 个相对简单的样本（depth ≤ 2, width ≤ 2，平均 804 字符），可有效区分中低水平模型。Test-Hard 套件包含 2,088 个高复杂度样本（depth/width 达 3×3，平均 16,535 字符），确保即使是 GPT-4o 级别的模型也有足够的挑战空间。两个套件的结合形成了从能力筛选到极限测试的完整评测链
    - 设计动机：单一难度的基准无法区分中高水平模型（如 Qwen2-72B 在普通集上已达 78.4%），需要更高难度版本来暴露天花板

### 评测策略
实验采用 6 种 prompt 设计（Naive、Self-CoT、PS-CoT、w/ Hint、Few-Shot、Simple Few-Shot）和 RougeL 作为主要指标。RougeL 阈值设为 0.75——低于此分数直接判 0 分，避免长推理路径产生的虚高分数。实验验证 RougeL 与人类判断的相关性达 0.9932，优于 Exact Match（0.9501）和 BLEU。

## 实验关键数据

### 主实验

| 模型 | Test-Naive | Test-Hard (Base) | Test-Hard (3-Shot) | 备注 |
|------|-----------|-----------------|-------------------|------|
| Qwen2-72B | 78.4% | 42.5% | 61.4% | 开源最强之一 |
| Llama-3.1-70B | 75.4% | 45.8% | 58.4% | PS-CoT 提升明显 |
| Llama-3.1-405B | 74.9% | 34.4% | 48.7% | 规模大但 Hard 集较弱 |
| GPT-4o | — | 51.1% | **69.5%** | 闭源最强 |
| GLM-4-Plus | — | 47.3% | 65.8% | 接近 GPT-4o |
| Mistral-7B | ~30% | 7.0% | 21.0% | 小模型严重不足 |
| **人类** | — | **92.6%** | — | 远超所有模型 |

### 不同语言和不同 Prompt 策略下的表现差异（RougeL, Test 套件）

| 模型 | Prompt | JSON | CSV | YAML | Tree | XML | Markdown | 总体 |
|------|--------|------|-----|------|------|-----|----------|------|
| Qwen2-72B | Naive | 85.8 | 92.6 | 82.7 | 86.4 | 71.2 | 75.1 | 78.4 |
| Qwen2-72B | PS-CoT | 89.5 | 92.0 | 93.4 | 84.8 | 81.1 | 68.9 | 80.8 |
| Llama-3.1-70B | PS-CoT | 94.5 | 93.7 | 98.5 | 83.2 | 93.9 | 72.7 | 84.2 |
| Llama-3.1-405B | PS-CoT | 84.5 | 92.0 | 94.7 | 86.7 | 94.7 | 76.0 | 74.9 |
| Qwen2-7B | Naive | 70.4 | 83.5 | 68.5 | 68.9 | 57.6 | 68.0 | 30.0 |

### 关键发现
- **JSON 一骑绝尘**：所有模型在 JSON 上表现最好（Llama-3.1-70B 达 94.5%），原因是 JSON 在互联网训练数据中出现频率极高——典型的训练数据偏差效应。自定义 Tree 和 XML 表现最差
- **深度/宽度增加导致性能断崖式下降**：从 depth=1 到 depth=3，所有模型准确率大幅下滑，且模型间性能方差在高复杂度下才真正拉开——这说明普通基准无法区分强模型
- **Self-CoT / PS-CoT 对小模型有害**：Qwen2-7B 使用 Self-CoT 后从 30.0% 暴跌至 17.2%，因为小模型无法正确生成推理路径，反而引入更多错误。但对大模型（如 Llama-3.1-70B）PS-CoT 提升 +9 个点
- **Few-shot 效果随数量先升后降**：1-shot → 3-shot 持续提升，但 5-shot 出现过拟合迹象，模型开始套用示例中的特定模式而非泛化推理
- **评测指标一致性**：RougeL 与 Human Judge 相关性 0.9932，GPT-4o Judge 与 Human Judge 相关性 0.9937，验证了自动评测的可靠性
- **模型尺度效应显著**：同系列模型参数量越大表现越好（Qwen2-7B 29.6% → 72B 42.5%; Llama-3.1-8B 22.3% → 70B 45.8% → 405B 34.4%，405B 在 Hard 集反而弱于 70B，可能与 prompt 敏感性有关）

## 亮点与洞察
- **语义无关设计是最核心的创新**：用无意义字符串替代真实文本内容，强制模型从结构标记本身推理。这看似简单却彻底改变了评测的本质——从"语义辅助的结构理解"变为"纯结构解析"，暴露了模型的真实短板
- **可控复杂度 + 无限生成的双重优势**：depth/width/column 的参数化设计不仅使难度精确可调，更重要的是可以随时生成全新数据集，从根本上解决了 benchmark 被纳入训练集的"数据泄漏"问题
- **训练数据偏差的清晰证据**：JSON 与自定义 Tree 的巨大性能差异（同一模型差距可达 30+ 个点），直接证明了当前 LLM 的结构理解能力高度依赖训练数据分布，而非真正掌握了通用的结构解析规则
- **人类 vs 模型的定量对比**：92.6% vs 45.8% 的鲜明差距，为后续研究提供了清晰的改进空间和目标锚点

## 局限与展望
- **语言和任务覆盖仍有限**：仅 8 种结构化语言和 29 个任务，未涉及 SQL、TOML、Protocol Buffers、HTML DOM 查询等重要格式，实际结构化语言和推理模式远不止于此
- **语义无关设计是双刃剑**：虽然纯粹评测了结构解析能力，但现实场景中语义与结构的交互推理才是常态。未来需要设计"语义-结构联合推理"的评测
- **基线模型时效性不足**：实验中模型截至 2024 年中，未覆盖 Claude、Gemini 2.0、DeepSeek 等后续模型，结论的时效性受限
- **缺乏系统性 fine-tuning 实验**：仅在 case study 中展示了微调效果（经 5100 步训练后 Join 任务成功），但未系统探索微调数据量、多任务联合训练等关键问题
- **评测指标可进一步细化**：RougeL 在精确匹配类任务（如 Syntax 错误检测）上可能过于宽松，需要针对不同任务类型设计差异化指标

## 相关工作与启发
- **vs GraphQA / Struc-Bench**：这些工作聚焦于图结构或表格的单一格式理解，依赖语义内容且数据集固定。StrucText-Eval 在覆盖面（8 vs 1-2 种语言）、语义无关性、可扩展性上全面超越
- **vs TableLLM / TEMPTABQA**：专注于表格操作和时序推理的特定领域评测，与 StrucText-Eval 的通用结构推理定位互补
- **vs NaturalBench 等通用 LLM 评测**：StrucText-Eval 填补了通用评测体系中"结构化文本推理"的空白维度，可作为现有评测的重要补充

## 评分
- 新颖性: ⭐⭐⭐⭐ 语义无关的结构推理评测视角新颖，但 benchmark 构建方法论本身相对直接
- 实验充分度: ⭐⭐⭐⭐⭐ 12 个模型 × 6 种 prompt × 8 种语言 × 8 类任务 × 多难度级别，分析维度极其全面
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，分类学和实验分析有条理，但部分分析段落较为冗长
- 价值: ⭐⭐⭐⭐ 揭示了 LLM 在结构推理上的明显短板并提供了可靠的评测工具，对后续研究有直接指导意义
- **CoT对结构推理的有效性存疑**：在复杂结构任务中Self-CoT反而降低性能，说明结构推理可能需要不同于语义推理的prompt策略
- **可控复杂度benchmark的设计范式值得借鉴**：通过参数化生成实现难度可控和防泄露，这一方法论可推广到其他评测场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 语义无关的纯结构推理评测视角独特，自动生成方法精巧，但benchmark类工作本身范式较为成熟
- 实验充分度: ⭐⭐⭐⭐ 12个模型、6种prompt策略、8种语言×29任务的系统性评测，不过缺乏系统的fine-tuning实验及更新的模型（如Claude、Gemini Ultra）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分类学和实验设置描述详细，但部分表格数据量过大导致可读性降低
- 对我的价值: ⭐⭐⭐ 评测方法论（可控复杂度+防泄露生成）有参考价值；揭示了LLM结构推理弱点；但对非benchmark研究方向的直接启发有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoV-Eval: Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective](cov_eval_evaluating_llms_from_code_security_perspective.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2025\] Where Are We? Evaluating LLM Performance on African Languages](where_are_we_evaluating_llm_performance_on_african_languages.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](../../ACL2026/llm_evaluation/evaluating_reasoning_models_for_queries_with_presuppositions.md)

</div>

<!-- RELATED:END -->
