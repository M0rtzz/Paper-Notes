---
title: >-
  [论文解读] PERSIST: Persistent Instability in LLM's Personality Measurements
description: >-
  [AAAI 2026][LLM/NLP][LLM人格测量] PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "LLM人格测量"
  - "行为一致性"
  - "推理模式"
  - "对齐评估"
  - "心理测量"
---

# PERSIST: Persistent Instability in LLM's Personality Measurements

**会议**: AAAI 2026  
**arXiv**: [2508.04826](https://arxiv.org/abs/2508.04826)  
**代码**: [https://github.com/tosatot/PERSIST](https://github.com/tosatot/PERSIST)  
**领域**: NLP生成 / LLM评估  
**关键词**: LLM人格测量, 行为一致性, 推理模式, 对齐评估, 心理测量

## 一句话总结
PERSIST 框架系统评估 29 个 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，揭示了 CoT 推理增加变异性却降低困惑度的"推理悖论"，以及会话历史对大小模型产生相反影响的规模依赖效应，表明当前 LLM 缺乏行为一致性的架构基础。

## 研究背景与动机

### 领域现状
**领域现状**：随着 LLM 在医疗、教育、决策支持等高风险场景中的部署，行为可预测性成为可信 AI 的核心要求。欧盟 AI 法案和美国 NIST AI 风险管理框架均将"性能一致性"列为高风险 AI 应用的必要条件。现有研究通过心理学量表（如大五人格 BFI-44、暗黑三元组 SD3）测量 LLM 的行为特征，Safdari et al. (2023) 已证明 LLM 能产生与人类可比的人格测量可靠性。

### 现有痛点与挑战
**现有痛点**：(1) 已有研究依赖单次测量，完全忽略了不同部署条件下的响应变异性——实际上 Sclar et al. (2023) 发现语义等价 prompt 间的性能波动可达 76 个准确率点；(2) Prompt 敏感性问题广泛存在，Salinas and Morstatter (2024) 记录了 prompting 中的"蝴蝶效应"——单字符修改可级联导致完全不同的模型行为；(3) 缺乏全面系统地量化变异性来源和程度的评估框架。

**核心矛盾**：LLM 对齐和安全评估假设行为特征可以被可靠测量，但测量工具本身的不稳定性使这一假设根基动摇——无法可靠测量 LLM 行为特征则无法可靠评估对齐效果，这是当前 LLM 安全体系中一个尚未被充分认识的结构性问题。

### 研究目标与切入角度
**本文目标**：通过全因素实验设计，全面量化 LLM 人格测量不稳定性的来源（模型规模、推理模式、人设指令、问题措辞、会话历史）和程度，揭示不稳定性的结构性和持久性本质。

**切入角度**：构建 PERSIST 框架，覆盖 29 个模型 × 5 种人设 × 4 类问卷 × 250 种排列 × 100 种改写 × 推理/非推理模式 × 有/无会话历史，生成超过 200 万条独立测量数据。

**核心 idea**：LLM 的人格测量不稳定性是持久的、结构性的，不能仅靠模型规模增长来解决，且许多预期能提升稳定性的干预（推理、会话历史）反而加剧不稳定性。

## 方法详解

### 整体框架
PERSIST（PERsonality Stability In Synthetic Text）由三个集成模块组成：(1) **生成引擎**：基于 vLLM 进行优化推理，支持多种模型架构的高效批量推理；(2) **响应处理模块**：使用多模式解析器从 LLM 输出中提取结构化数据（支持索引响应 "1: 4"、JSON 结构、孤立数字评分等多种格式），同时提取 token 级 log 概率用于不确定性量化；(3) **分析管道**：从单条响应到问题级和特质级统计量进行层次聚合，计算跨排列的均值和标准差。单条无效响应（评分超出 1-5 范围或解析失败）即废弃整次运行以确保数据质量。

### 关键设计

1. **双版本问卷设计**：

    - 功能：区分不稳定性是人类中心化问卷的伪影还是模型内在问题
    - 核心思路：使用传统心理学问卷 BFI-44（44 个条目测量大五人格）和 SD3（27 个条目测量暗黑三元组），以及 LLM 适配版本 BFI-LLM 和 SD3-LLM（将人类特定体验翻译为行为等价的 AI 表述，如 "Is depressed, blue" → "Focuses on negative aspects"，"I'll say anything to get what I want" → "Truth is secondary to reaching a goal"），所有问卷均使用 5 点 Likert 量表
    - 设计动机：如果适配后仍表现出相同的不稳定性，则排除了问卷措辞因素，指向模型内在的结构性不稳定

2. **全因素实验设计（5 因素系统变异）**：

    - 功能：系统量化各因素对测量稳定性的独立和交互影响
    - 核心思路：(a) **问题顺序**——250 个随机排列，测试人格测量应满足的顺序不变性假设；(b) **人设指令**——5 种人设（Assistant 基线、Buddhist 正面人设、Teacher 正面人设、Antisocial 临床人设基于 DSM-5、Schizophrenia 临床人设）；(c) **推理模式**——标准 vs CoT，比较推理过程对一致性的影响；(d) **改写**——100 个语义等价改写由 Qwen3 235B-A22B 生成并经两位作者人工验证校正；(e) **会话历史**——多轮对话 vs 单轮呈现
    - 设计动机：全因素设计能分离出各因素的独立贡献，揭示变异性的多个来源及其交互效应

3. **大规模模型覆盖与稳定性度量**：

    - 功能：确保结论的跨架构和跨规模普适性
    - 核心思路：评估 29 个模型涵盖 8 个家族——Llama 3.1（8B/70B/405B Instruct）、Qwen 2.5（1.5B 至 72B 共 6 个 Instruct）、Qwen 3（1.7B 至 235B-A22B 含 MoE 共 7 个）、Gemma 2（2B/9B/27B）、Gemma 3（1B/4B/12B/27B）、DeepSeek V3/R1（均 671B）、GPT-OSS（20B/120B）、Claude Sonnet 4.5/Opus 4.1；度量使用问题级 SD（跨 250 排列）、token 级困惑度 $\text{PPL}=\exp(-\log p)$、Spearman 相关、Wilcoxon 符号秩检验和 Kruskal-Wallis 检验
    - 设计动机：覆盖 1B 到 685B 的完整规模范围和多种训练范式，避免结论受限于特定架构

### 实验配置
所有实验使用温度 τ=0 以隔离操控变量的效应（推理模式实验使用 τ=0.6），最大 token 数 16384，随机种子 42，硬件为 4× NVIDIA H100 SXM 80GB HBM3。Claude 模型在推理实验中仅运行 70 次（其余模型均 250 次）。

## 实验关键数据

### 主实验：规模效应分析

| 指标 | Spearman ρ 方向 | p 值 | 效应解读 |
|------|-----------------|------|----------|
| 模型规模 → 正面特质均分 | ↑ 正相关 | 0.001** | 越大越"友善" |
| 模型规模 → 负面特质均分 | ↓ 负相关 | <0.001*** | 越大越"温和" |
| 模型规模 → 问题级 SD | ↓ 负相关 | <0.001*** | 越大越稳定 |
| 模型规模 → 问题级困惑度 | 无显著相关 | 0.934 | 规模不降低不确定性 |
| 困惑度 ↔ 问题级 SD | ρ=0.465 | — | 仅中等相关，PPL 非完整稳定性指标 |

### 推理悖论实验

| 模型/条件 | 变异性 SD 变化 | 困惑度 PPL 变化 | 统计检验 |
|-----------|---------------|----------------|----------|
| GPT-OSS 推理强度递增 | 显著升高 | 显著下降 | Kruskal-Wallis p<0.001 |
| Qwen3 推理 vs 非推理 | 显著升高 | 显著下降 | Mann-Whitney U all p<0.001 |
| Qwen3-MoE 推理 vs 非推理 | 显著升高 | 显著下降 | Mann-Whitney U all p<0.001 |
| DeepSeek 推理 vs 非推理 | 显著升高 | 不显著 | p<0.01 (SD), n.s. (PPL) |
| Claude 推理 vs 非推理 | 显著升高 | — | Mann-Whitney U p<0.01 |

核心发现：CoT 使模型在 token 级别更确信（PPL↓），却在行为级别更不一致（SD↑）。

### 消融实验：问卷类型与改写效应

| 对比条件 | 指标 | p 值 | 结论 |
|----------|------|------|------|
| LLM 适配 vs 传统问卷 | 问题级 SD | 0.286 (n.s.) | 无显著差异 |
| LLM 适配 vs 传统问卷 | 问题级 PPL | <0.001*** | 适配版 PPL 更高 |
| 改写 vs 重排（<50B） | ΔSD | 0.244 (n.s.) | 小模型无显著影响 |
| 改写 vs 重排（≥50B） | ΔSD | <0.01** | 大模型变异显著增加 |

### 会话历史的规模依赖效应

| 模型组 | 数量 | 有历史后 SD 变化 | p 值 |
|--------|------|-----------------|------|
| <50B 小模型 | n=19 | 显著增加变异 | <0.001*** |
| ≥50B 大模型 | n=4 | 显著降低变异 | <0.001*** |

### 关键发现
- 即使 400B+ 模型在 5 分制量表上仍有 SD>0.3——规模不是解药
- CoT 推理增加变异性同时降低困惑度（"推理悖论"）
- 错位人设（反社会/精神分裂）变异性和困惑度均显著高于 Assistant 基线（p<0.05）
- 正面人设（佛教僧侣）显著降低变异性和困惑度（p<0.05）
- 会话历史帮助大模型但伤害小模型——部署策略需区分规模

## 亮点与洞察
- **"推理悖论"是核心发现**：CoT 让模型生成不同的推理链条导致不同结论，token 级确定性不等于行为一致性，挑战了"更多推理=更可靠"的直觉
- **会话历史的规模依赖效应**：大模型能从上下文中提取一致性信号而小模型被额外信息淹没，对 prompt 工程实践有直接指导价值
- **不稳定性作为错位检测信号**：欺骗性模型可能伪造正确的平均特质分数，但跨排列维持一致性要困难得多——变异模式可作为错位标志
- **方法论贡献**：200 万+独立测量和 5 因素全交叉设计建立了 LLM 行为评估的新基准，PERSIST 框架可作为未来安全认证的标准工具

## 局限与展望
- **自我报告与行为的鸿沟**：虽然已有证据表明 LLM 自我报告与行为输出相关，但自我报告可能低估实际行为不稳定性
- **缺乏正式心理测量验证**：传统和适配问卷均缺乏针对 LLM 的因子负荷和 Cronbach's α 验证
- **战略欺骗可能性**：若模型识别到评估上下文可能调整响应——随机排列和关注变异性（非均值）使欺骗更难
- **闭源模型覆盖有限**：Claude 仅 70 次运行，GPT 系列未包含最新版本
- **仅覆盖自我报告评估**：未测试博弈论场景、角色扮演等行为级评估方式

## 相关工作与启发
- **vs Safdari et al. (2023)**：他们证明 LLM 人格测量可达人类可比可靠性，但仅在特定 prompt 配置下——本文揭示了跨配置的巨大变异性
- **vs Sclar et al. (2023)**：他们发现语义等价 prompt 间性能波动达 76 点，本文将这一发现扩展到人格测量领域并系统分析了 5 个变异因素
- **vs Representation Engineering (Zou et al., 2025)**：他们用激活空间方向监控行为特征，本文指出这些特征本身不稳定
- **vs Anthropic Persona Vectors (Chen et al., 2025)**：他们提出可系统识别和控制的人格向量——本文提供了该技术需面对的基准不稳定性数据

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全因素设计揭示多个反直觉发现
- 实验充分度: ⭐⭐⭐⭐⭐ 29 模型、200 万+响应、严格统计检验
- 写作质量: ⭐⭐⭐⭐⭐ 统计分析严谨，结论有力
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 安全和对齐评估社区有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)

</div>

<!-- RELATED:END -->
---
title: >-
  [论文解读] PERSIST: Persistent Instability in LLM's Personality Measurements
description: >-
  [AAAI 2026][LLM/NLP][LLM人格测量] PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度，LLM 适配问卷与传统人类问卷表现出相似的不稳定性。
tags:
  - AAAI 2026
  - LLM/NLP
  - LLM人格测量
  - 行为一致性
  - 推理模式
  - 对齐评估
  - 心理测量
---

# PERSIST: Persistent Instability in LLM's Personality Measurements

**会议**: AAAI 2026  
**arXiv**: [2508.04826](https://arxiv.org/abs/2508.04826)  
**代码**: [https://github.com/tosatot/PERSIST](https://github.com/tosatot/PERSIST)  
**领域**: NLP生成 / LLM评估  
**关键词**: LLM人格测量, 行为一致性, 推理模式, 对齐评估, 心理测量

## 一句话总结
PERSIST 框架系统评估 25 个开源 LLM（1B-685B）在 200 万+响应上的人格测量稳定性，发现即使 400B+模型在 5 分制量表上仍有 SD>0.3 的不稳定性，且 CoT 推理悖论性地增加变异性同时降低困惑度，LLM 适配问卷与传统人类问卷表现出相似的不稳定性。

## 研究背景与动机

### 领域现状

**领域现状**：LLM 对齐和安全评估需要衡量模型的行为特征（大五人格、暗黑三元组等），通常使用心理学量表。

**现有痛点**：(1) 问题顺序、改写都可导致评分大幅变化；(2) 不清楚不稳定性是人类中心化问卷的伪影还是模型本身的问题；(3) 推理、会话历史等因素对稳定性的影响未被系统研究。

**核心矛盾**：无法可靠测量 LLM 行为特征→无法可靠评估对齐效果。

**本文目标** 全面量化 LLM 人格测量不稳定性的来源和程度。

**切入角度**：全因素设计——25模型×5人设×4问卷×250排列×100改写×推理/非推理×会话历史。

**核心 idea**：LLM 的人格测量不稳定性是持久的、结构性的，不能仅靠模型规模增长来解决。

## 方法详解

### 整体框架
PERSIST 三模块：生成引擎(vLLM)、响应处理(token 级 log 概率提取)、分析管道(层次聚合计算排列间 SD)。

### 关键设计

1. **双版本问卷**：BFI-44 + SD3（原版）+ BFI-LLM + SD3-LLM（替换人类特定概念），验证不稳定性是否为问卷设计的伪影
2. **5种人设**：Assistant(基线)、Buddhist、Teacher、Antisocial、Schizophrenia
3. **稳定性指标**：问题级 SD（跨250个排列）+ 困惑度 $\exp(-\log p)$
4. **温度自适应推理**：τ=0 确定性输出，τ=0.6 推理模式

### 损失函数 / 训练策略
纯评估研究，无训练。25 个模型通过 vLLM 和 HuggingFace Inference API 访问。

## 实验关键数据

### 主实验

| 发现 | Spearman ρ | p值 | 效应 |
|------|-----------|-----|------|
| 模型规模↑ → 正面特质↑ | 显著 | 0.001** | 更"友善" |
| 模型规模↑ → 负面特质↓ | 显著 | <0.001*** | 更"温和" |
| 模型规模↑ → 变异性↓ | 显著 | <0.001*** | 更稳定 |
| 模型规模↑ → 困惑度 | 不显著 | 0.934% | 无关联 |

### 推理悖论

| 条件 | 变异性 | 困惑度 |
|------|--------|--------|
| 无推理 | 低 | 高 |
| 有推理(CoT) | **高** (p<0.001) | **低** (p<0.01) |

### 会话历史的双刃剑效应

| 模型大小 | 有历史的效果 | p值 |
|---------|------------|------|
| <50B (n=19) | **增加**变异性 | <0.001*** |
| ≥50B (n=4) | **降低**变异性 | <0.001*** |

### 关键发现
- 即使 400B+模型仍有 SD>0.3——规模不是解药
- CoT 推理增加变异性但降低困惑度（更自信却更不一致——推理悖论）
- LLM 适配问卷与原版表现出相同的不稳定性——非问卷设计的伪影
- 错位人设(反社会、精神分裂)比基线人设变异性更高、困惑度更高
- 困惑度与变异性仅中等相关(ρ=0.465)——不是完整的稳定性指标

## 亮点与洞察
- **"推理悖论"是核心发现**：CoT 让模型更自信(困惑度↓)却更不一致(SD↑)——挑战"更多推理=更好"的直觉
- **会话历史的规模依赖效应**：帮助大模型但伤害小模型，提示工程需考虑模型规模
- **对对齐评估的警示**：测量工具本身不可靠时，对齐评估结论如何可信？

## 局限与展望
- 人格测量的 construct validity 存疑——LLM 是否真有"人格"？
- 仅使用心理学量表，未覆盖其他行为评估方式（如博弈论场景）
- 未探索微调或 RLHF 对稳定性的影响

## 相关工作与启发
- 对 LLM 安全评估方法论的直接冲击——所有基于问卷的对齐评估都需要重新审视其稳定性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 全因素设计揭示多个反直觉发现
- 实验充分度: ⭐⭐⭐⭐⭐ 25模型、200万+响应、系统统计检验
- 写作质量: ⭐⭐⭐⭐⭐ 统计分析严谨，呈现清晰
- 价值: ⭐⭐⭐⭐⭐ 对LLM安全和对齐评估社区有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)
- [\[AAAI 2026\] ICL-Router: In-Context Learned Model Representations for LLM Routing](icl-router_in-context_learned_model_representations_for_llm_routing.md)
- [\[AAAI 2026\] VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation](vspo_validating_semantic_pitfalls_in_ontology_via_llm-based_cq_generation.md)

</div>

<!-- RELATED:END -->
