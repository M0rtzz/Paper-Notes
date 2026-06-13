---
title: >-
  [论文解读] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA
description: >-
  [ACL 2025 (SemEval Workshop)][LLM 其他][表格问答] 提出一种面向表格问答的多模型协同管道系统，整合 Text-to-SQL、Text-to-Code（Pandas）、端到端语义理解三条路径，通过 RAG 检索增强上下文 + Llama 3.3-70B 作为 Orchestrator 仲裁最终答案，在 SemEval-2025 Task 8 的开源赛道中以 80% 准确率排名 13/38，开发集上开源组合（88%）显著超越 GPT-4o 单模型（74%）。
tags:
  - "ACL 2025 (SemEval Workshop)"
  - "LLM 其他"
  - "表格问答"
  - "Text-to-SQL"
  - "多模型编排"
  - "RAG"
  - "开源 LLM"
---

# Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA

**会议**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2506.09657](https://arxiv.org/abs/2506.09657)  
**代码**: [GitHub](https://github.com/Nickolas-option/QA_on_Tabular_Data_SemEval2025_Task8)  
**领域**: LLM / NLP  
**关键词**: 表格问答、Text-to-SQL、多模型编排、RAG、开源 LLM

## 一句话总结

提出一种面向表格问答的多模型协同管道系统，整合 Text-to-SQL、Text-to-Code（Pandas）、端到端语义理解三条路径，通过 RAG 检索增强上下文 + Llama 3.3-70B 作为 Orchestrator 仲裁最终答案，在 SemEval-2025 Task 8 的开源赛道中以 80% 准确率排名 13/38，开发集上开源组合（88%）显著超越 GPT-4o 单模型（74%）。

## 研究背景与动机

**领域现状**：表格问答（Table QA）是将自然语言查询转化为对结构化数据的可执行操作（返回字符串、数值、布尔值或列表），广泛应用于金融、医疗、商业数据分析。当前主流方案分为两类——基于代码生成（Text-to-SQL/Python）的方法擅长数值计算和精确查询，端到端（E2E）直接回答法擅长语义理解类问题。

**现有痛点**：GPT-4o 等闭源模型在 Table QA 上表现优异，但开源模型长期落后，限制了研究的可复现性和部署灵活性。单一模型和单一方法路径难以覆盖所有类型的问题——数值聚合适合 SQL/Python，而"标题是否涉及 communication"这类语义判断适合 E2E。

**核心矛盾**：真实世界表格数据充满噪声（缺失值、emoji 列名、缩写等），LLM 生成的代码常有聚合逻辑错误、类型操作错误和参数遗漏；同时 E2E 方法面临"大海捞针"困境——完整表格注入上下文过长，关键信息易被遗漏或产生幻觉。

**本文目标** (1) 如何让多个开源模型协同工作以弥合与闭源模型的差距？(2) 如何在代码生成和语义理解之间实现互补？(3) 如何应对真实数据的噪声和格式不匹配？

**切入角度**：作者观察到不同模型和不同方法路径在不同类型问题上各有优势，单一模型的瓶颈可以通过多模型投票和编排来突破。与其追求单一更强的模型，不如设计智能的多路径管道 + 仲裁机制。

**核心 idea**：用多个开源 LLM 分别走 SQL、Python、E2E 三条路径生成候选答案，再由 Orchestrator 从中选择最可靠的，实现"群体智慧"超越单个闭源强模型。

## 方法详解

### 整体框架

系统包含两大并行路径——**Code-based**（SQL + Python）和 **End-to-End**，每条路径分别生成候选答案，最终由 Orchestrator（Llama 3.3-70B Instruct）仲裁选出最终答案。输入为自然语言问题 + 表格数据，输出为精确匹配格式的答案（Boolean / Number / Category / List）。

### 关键设计

1. **Text-to-SQL 代码生成模块**：

    - 功能：将自然语言问题转化为 SQL 查询，在内存 SQLite 数据库中执行
    - 核心思路：使用 Codestral 20.51 和 Qwen Coder 2.5-32B 两个模型分别生成 SQL 查询。prompt 中通过 RAG 注入相关行数据和推荐列名来丰富上下文，查询通过 SQLAlchemy 在 SQLite 内存数据库上执行，结果作为一个候选答案
    - 设计动机：SQL 天然适合结构化数据的精确查询和聚合操作，两个模型的独立生成提供了多样性

2. **Pandas Code 生成模块**：

    - 功能：将自然语言问题转化为 Python/Pandas 代码，在沙箱环境中执行
    - 核心思路：同样使用 Codestral 和 Qwen Coder 生成 Python 代码，在带超时保护的沙箱中执行。记录执行状态和错误信息供后续自纠错使用。Python 比 SQL 更灵活，可处理复杂的数据转换和多步骤推理
    - 设计动机：SQL 和 Python 在不同类型问题上各有优劣——SQL 擅长简单聚合，Python 更适合复杂的数据处理逻辑，两者互补可将准确率从 72% 提升至 84%

3. **RAG 检索增强模块**：

    - 功能：为 LLM 注入与问题最相关的表格行数据，解决数据格式不匹配问题
    - 核心思路：为每个相关列创建句子嵌入并建立索引，查询时检索语义最相似的 Top-3 行注入 LLM 上下文。例如回答"多少客户来自 Japan"时，RAG 可以告知模型数据集中 japan 是小写的，避免大小写不匹配
    - 设计动机：真实数据集中存在大量大小写不一致、格式变体等问题，直接让 LLM 猜测格式容易出错，RAG 提供真实数据样本作为参照

4. **自纠错机制（Self-Correction）**：

    - 功能：当生成的 SQL/Python 代码执行失败时，自动将错误信息回传 LLM 重新生成
    - 核心思路：收集 schema 信息、错误消息和原始问题，重新提交给 LLM 生成修正后的代码。这是受 ReForce 工作的启发
    - 设计动机：虽然实际效果有限（因为有 4 个代码生成 agent，通常至少有一个能成功），但在极端情况下提供了兜底保障

5. **End-to-End 语义理解模块**：

    - 功能：跳过代码生成，直接让 LLM 阅读表格文本来回答问题
    - 核心思路：使用 MiniMax-01 将表格数据转换为 Markdown 格式，与问题一同输入 LLM，直接生成答案。答案必须符合指定的数据格式（Boolean/List/Number/String）。结合 RAG 选出的相关列来限制上下文长度
    - 设计动机：某些问题需要语义理解而非精确计算，如判断标题是否涉及某话题、从姓名推断性别等。E2E 方法在这类问题上远超代码生成方法

6. **LLM Orchestrator 仲裁模块**：

    - 功能：从多个候选答案中选择最可能正确的作为最终输出
    - 核心思路：使用 Llama 3.3-70B Instruct 作为仲裁者，接收所有成功执行的候选答案及其代码，按照特定 prompt 指令评估每个候选的合理性。关键约束是 Orchestrator 不能自行生成新答案，只能在已有候选中选择
    - 设计动机：不同路径和模型在不同问题上各有正确和错误，Orchestrator 模拟"多数投票 + 逻辑评判"来提高最终准确率

### 辅助技术

- **列预测（Column Prediction）**：让 LLM 预先选择与问题相关的列，减少上下文长度，缓解 E2E 模型的"大海捞针"问题
- **列名解释（Column Renaming）**：将缩写列名重命名为可读形式（如 DMC → Duff Moisture Code），帮助 LLM 更好理解数据语义
- **Checklist 式 Prompt**：在代码生成 prompt 中加入类型匹配、实体验证、逻辑一致性等检查项，减少 LLM 跳过关键指令的情况
- **Emoji 处理**：将列名中的 emoji 替换为唯一哈希符号，避免 LLM 在处理 emoji 时插入或遗漏空格导致精确匹配失败
- **CoT 推理**：显式指示 LLM 在给出最终答案前进行详细推理，用模糊匹配从冗长回复中提取最终答案

## 实验关键数据

### 表1：开源赛道官方排名（测试集）

| 排名 | 团队 | 系统名称 | 准确率 (%) |
|:--:|:--|:--|:--:|
| 1 | xiongsishi | TeleAI | 95.02 |
| 2 | pbujno | SRPOL AIS | 89.66 |
| **13** | **anotheroption** | **anotheroption** | **80.08** |
| — | baseline | stable-code-3b-GGUF | 26.00 |

在全局排名（包含闭源模型）中，本系统以 80.08% 排名 Top-20/53，全程只使用开源模型。

### 表2：开发集消融实验结果

| 配置 | 准确率 (%) | 说明 |
|:--|:--:|:--|
| Codestral Python only | 72 | 单模型单路径基线 |
| Codestral Python + Reformulation | 68 | 问题改写反而降 4% |
| Codestral Python + SQL | 84 | SQL 互补提升 12% |
| MiniMax E2E only | 54 | 纯 E2E 较弱 |
| MiniMax E2E + Reformulation | 52 | 改写再降 2% |
| Codestral + Qwen Coder Python (w/ orch.) | 88 | 双模型 + 编排 |
| **Full pipeline (w/ orchestrator)** | **88** | 完整系统最佳 |
| GPT-4o (w/o orchestrator) | 74 | 闭源单模型基线 |
| GPT-4o (w/ pipeline) | 87 | 管道对 GPT-4o 也有效 |

### 关键发现

- **开源组合超越闭源单模型**：开源模型组合（88%）比 GPT-4o 单模型（74%）高 14 个百分点，证明编排多个弱模型可以产生"群体智慧"效应。将同样的管道应用于 GPT-4o 也能从 74% 提升到 87%，说明管道架构本身就有价值
- **SQL 互补效应显著**：Python only → Python + SQL 从 72% 跳到 84%，说明两种代码形式在不同问题类型上的互补性很强
- **问题改写适得其反**：Reformulation 在所有配置中都降低准确率（Python: 72→68%, E2E: 54→52%），原因是改写错误在管道起点传播且无恢复机制，而多模型路径已经通过 Orchestrator 提供了隐式的"错误过滤"
- **Orchestrator 决策分析**：63.4% 为一致性确认（所有候选一致），14.6% 过滤逻辑缺陷答案，12.2% 拒绝格式不匹配答案，9.8% 裁决分歧答案——Orchestrator 在 36.6% 的情况下发挥了积极作用
- **自纠错效果有限**：由于 4 个代码生成 agent 中通常至少有一个能生成可执行代码，自纠错几乎未被触发

## 亮点与洞察

- **多模型编排的"免费午餐"**：通过组合两个 coding 模型 + 一个 E2E 模型 + 一个 Orchestrator，不需要任何微调就实现了开源超闭源的效果。这证明在特定任务上，合理的系统设计比模型规模更重要
- **模块化架构的可替换性**：SQL/Python/E2E 三条路径完全独立，任何一个组件都可以单独升级替换而不影响其他部分，这为持续改进提供了便利
- **管道对强模型同样有效**：GPT-4o 在使用管道后从 74% 提升到 87%，说明这套多路径 + 仲裁的思路不仅是"弱模型的补偿策略"，而是一种通用的性能增强框架
- **负面结论同样有价值**：问题改写降低性能的发现揭示了在多模型管道中，错误传播机制与单模型系统截然不同——管道起点错误无法被下游纠正，而多路径本身已经提供了冗余

## 局限与展望

- **全候选错误无解**：当所有代码生成 agent 和 E2E 模块都给出错误答案时，Orchestrator 无法独立生成正确答案，这是多模型编排的根本局限
- **检索语义鸿沟**：当查询用语与表格数据差异较大时，embedding 检索无法完全解决术语不匹配问题
- **模型规模依赖**：系统性能对模型规模敏感——使用更小的开源模型时性能会显著下降；需要 70B 的 Orchestrator 和 32B 的 Coder，部署成本不低
- **列名解释的误判风险**：LLM 对缩写列名的解释可能不准确（如 ISI 被 GPT-4o 解释为 Fire Spread Index 而非 Initial Spread Index），需要额外的验证机制
- **缺乏动态路径选择**：当前所有问题都走全部路径，未来可以根据问题特征自动决定是否需要代码执行还是语义分析，以提高效率

## 相关工作与启发

- **vs TeleAI（第一名，95%）**：排名第一的方案可能使用了更精细的 prompt 工程或更强的模型组合，差距约 15% 暗示仍有大量提升空间，尤其在代码生成质量和复杂推理上
- **vs ReForce（Text-to-SQL 自纠错）**：本文的自纠错机制直接受 ReForce 启发，但在多 agent 场景下效果被"冗余"削弱——如果只有单个 agent，自纠错可能更关键
- **vs DataBench 原始评测**：DataBench 保留真实世界噪声（emoji、缺失值、缩写列名），比清洗后的基准更有挑战性，本文的 emoji 处理和列名解释策略值得在其他 Table QA 系统中借鉴
- **启发**：多模型编排 + Orchestrator 仲裁的范式可以推广到其他代码生成任务（如数据分析、报表生成），在无需微调的前提下提升系统鲁棒性

## 评分

- 新颖性: ⭐⭐⭐ 方法组件（SQL/RAG/E2E/Orchestrator）均为已有技术，核心贡献在于系统集成和消融分析
- 实验充分度: ⭐⭐⭐⭐ 消融实验覆盖全面，包含 Orchestrator 决策分析、GPT-4o 对比、和多种配置对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，错误分析部分尤其详细且有参考价值
- 价值: ⭐⭐⭐ 证明了开源模型组合超越闭源模型的可行性，对实际部署有参考意义，但排名和绝对性能仍有差距
---
title: >-
  [论文解读] Team Anotheroption at SemEval-2025 Task 8: Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA
description: >-
  [ACL 2025][LLM/NLP] 提出一种混合 LLM 管道系统，通过 Text-to-SQL/Code 生成、RAG 检索、自纠错机制和 LLM Orchestrator 协调多个开源模型，在 SemEval-2025 Task 8（表格问答）中达到 80% 准确率（Top-13/38），开源模型组合（88%）超越 GPT-4o 单模型（74%）。
tags:
  - ACL 2025
  - LLM/NLP
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Hierarchical Retrieval with Evidence Curation for Open-Domain Financial QA](hierarchical_retrieval_with_evidence_curation_for_open-domain_financial_question.md)
- [\[ACL 2025\] PiFi: Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models](plugin_finetuning_bridge.md)
- [\[ACL 2025\] Enhancing Open-Domain Task-Solving Capability of LLMs via Autonomous Tool Integration from GitHub](paper_2312_17294.md)
- [\[ACL 2025\] BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages](brighter_bridging_the_gap_in_human-annotated_textual_emotion_recognition_dataset.md)
- [\[ACL 2025\] Mind the (Belief) Gap: Group Identity in the World of LLMs](mind_the_belief_gap_group_identity_in_the_world_of_llms.md)

</div>

<!-- RELATED:END -->
