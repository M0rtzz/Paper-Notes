---
title: >-
  [论文解读] BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models
description: >-
  [ACL 2026][医学图像][生物医学工具调用] BioTool 构建了一个覆盖 NCBI / Ensembl / UniProt 三大生物医学数据库 34 个常用工具、7,040 条经人工核验的「查询–API 调用」对的指令微调数据集，用它微调 4B 量级开源 LLM 后…
tags:
  - "ACL 2026"
  - "医学图像"
  - "生物医学工具调用"
  - "NCBI/Ensembl/UniProt"
  - "指令微调"
  - "小模型超越商业大模型"
---

# BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models

**会议**: ACL 2026  
**arXiv**: [2605.05758](https://arxiv.org/abs/2605.05758)  
**代码**: https://github.com/gxx27/BioTool  
**领域**: 医学 NLP / LLM 工具调用 / 指令微调数据集  
**关键词**: 生物医学工具调用、NCBI/Ensembl/UniProt、指令微调、小模型超越商业大模型

## 一句话总结
BioTool 构建了一个覆盖 NCBI / Ensembl / UniProt 三大生物医学数据库 34 个常用工具、7,040 条经人工核验的「查询–API 调用」对的指令微调数据集，用它微调 4B 量级开源 LLM 后，工具调用质量超过 GPT-5.1 / Gemini-3 Pro / Claude-4.5-Sonnet 等商业大模型 15% 以上。

## 研究背景与动机

**领域现状**：通用领域已经有 Toolformer / Gorilla / ToolBench / APIGen 等成熟的工具调用数据集与微调流水线；而在生物医学领域，主流做法仍然是 GeneGPT / ChemCrow / Biomni 这类基于 in-context learning 的 agent，把若干工具的文档塞进 prompt 里让模型现学现用。

**现有痛点**：in-context 方式存在三重瓶颈：(1) 受 context length 限制，工具数量很小（GeneGPT 只覆盖 NCBI 一小撮 API）；(2) 生物医学 API 的参数 schema 极其复杂，单靠几行 prompt 描述无法覆盖各种调用场景；(3) 把自然语言问题映射到专业 schema、identifier、参数规范的难度远高于通用工具，hallucination 严重。

**核心矛盾**：通用工具调用数据集再大，里面包含的生物医学工具也只是「沧海一粟」，无法让 LLM 在 BLAST、Variation API、UniProt sequence query 这种需要严格 schema 的场景里给出可执行调用。要让 LLM 真正成为生物医学研究者的助手，必须有一份「数据库原生」的高质量工具调用语料。

**本文目标**：(1) 系统性地从三大权威生物医学数据库挑选高频工具；(2) 自动化批量合成「查询–API 调用」对并保证语义有效；(3) 用这份数据微调中小开源 LLM，把工具调用能力打到甚至超过顶级闭源大模型。

**切入角度**：作者反向构造数据——先从工具文档枚举出多样化的 API 参数组合并实际执行，再以「真实可用的 API 响应」为种子，用 reasoning model 反推「能被这个响应回答的用户 query」，最后让 LLM judge + 人工专家把关。这种「先有正确答案再造问题」的范式天然规避了 query 与 API 错配的标注噪声。

**核心 idea**：用「response-grounded 反向 query 合成 + 多轮 LLM/人工过滤」替代「人工撰写 query→人工标 API」的传统范式，把生物医学工具调用语料的规模和质量同时拉上去，让 4B 模型在专业 schema 上超过 200×参数量的闭源模型。

## 方法详解

### 整体框架
BioTool 的构造流水线分为四个阶段：(1) **工具选择**——人工从 NCBI / Ensembl / UniProt 网站挑出 34 个研究人员高频使用的 API endpoint，覆盖变异、基因组、蛋白质组、进化、通识生物五大子领域；(2) **API 调用合成**——把每个工具的官方文档喂给 LLM，让其枚举多样化的参数组合，并真实执行调用，丢掉返回为空/无信息的样本，剩下 3,829 条 unique API call；(3) **反向 query 生成**——给 reasoning model（如 OpenAI 最新 o 系列）输入 API call + 真实响应，让其生成「能被这条响应支撑回答」的自然语言 query；(4) **质量过滤**——先用 LLM-judge 评估 API 响应是否真的能回答 query，再由生物学专家逐条人工 review，最终留下 7,040 条 (query, tool info, API arguments, observation) 四元组。

下游使用时，工具调用 LLM 仅生成 API arguments，由系统执行得到 observation，再由 base LLM（如 GPT-5.1）把 observation 整合进最终答案，实现「tool caller + answer generator」的解耦架构。

### 关键设计

1. **Response-grounded 反向 query 合成**:

    - 功能：解决「query 与 API 不对齐」的根本问题——传统人工写 query 再标 API 的方式，常常出现「API 响应根本回答不了 query」。
    - 核心思路：先随机生成多样化的 API 参数组合并执行，得到「确实能返回有用信息的 (API call, response)」；再以这条响应为锚，反向让 reasoning model 编出「最合理的用户问题」。这样保证了 query 必然被 API 响应支撑。
    - 设计动机：生物医学 query 的合理性需要 domain 知识，让 LLM 凭空写 query 容易产生 hallucination；而真实响应已经把「可回答性」内嵌在数据里，反向写比正向写更易控制质量。

2. **三层过滤的高人工核验比例**:

    - 功能：保证 7,040 条数据在生物学正确性、API schema 合规性、query-response 对齐性三个维度都达标。
    - 核心思路：第一层是执行验证（API 必须真能返回响应且不为空）；第二层是 LLM-judge（评估「响应是否充分支撑 query 的回答」）；第三层是生物学专家人工 review（重点检查生物学相关性与正确性，例如基因 ID 是否匹配物种、变异坐标是否合规）。
    - 设计动机：纯 LLM 合成数据在专业领域噪声极大，没有人工 review 就直接微调，模型会把错误的 schema 当真学进去。三层漏斗确保进入训练集的样本质量统一高。

3. **小模型微调超越大模型 in-context**:

    - 功能：证明「高质量专科数据 + 小开源模型」可以在工具调用任务上完爆「通用大数据 + 闭源大模型 + ICL」。
    - 核心思路：把 BioTool 训练集喂给 Qwen-3-4B / 8B 这类小模型做 SFT，让模型把 34 个工具的参数规范「内化」成权重，而不是临时从 prompt 中读。推理时直接生成 JSON 格式的 API arguments。
    - 设计动机：作者观察到 in-context 大模型在专业 schema 任务上的瓶颈不在「智能」而在「专精」——一旦把领域知识 hardcode 进权重，4B 小模型即可碾压 200×参数量的通用模型。这是 specialization vs. generalization 的典型权衡。

### 损失函数 / 训练策略
标准 SFT cross-entropy，target 为 (tool name, API arguments) 的 JSON 字符串；observation 不参与训练 loss，仅作为推理时由系统填充的字段。

## 实验关键数据

### 主实验：工具调用质量对比

| 模型 | 参数量 | 训练方式 | API-calling 质量 | 备注 |
|------|--------|---------|------------------|------|
| GPT-5.1 (闭源) | 未公开 | ICL | baseline | 顶级通用大模型 |
| Gemini-3 Pro | 未公开 | ICL | 接近 GPT-5.1 | |
| Claude-4.5-Sonnet | 未公开 | ICL | 三家中最强 baseline | |
| Qwen-3-4B + BioTool SFT | 4B | SFT | 比 Claude-4.5 高 **15.0%** | 本文最佳 |
| Qwen-3-8B + BioTool SFT | 8B | SFT | 进一步提升 | |

### 下游问答质量评估（人工生物学家打分）

| 配置 | 相对裸 GPT-5.1 的 normalized answer quality 提升 | 说明 |
|------|------|------|
| GPT-5.1（无工具） | 0%（基线） | 直接回答，易 hallucination |
| GPT-5.1 + Oracle BioTool API call | +**88.4%** | 上限：API call 由数据集 ground truth 提供 |
| GPT-5.1 + BioTool-fine-tuned tool caller | +**69%** | 实战：用微调好的 4B 模型作为 tool caller |
| GPT-5.1 + ICL tool calling | 远低于上述两者 | 传统做法 |

测试集规模：1,048 条 test query，由生物学专家做 head-to-head 偏好评估。

### 关键发现
- **专科数据胜于通用规模**：4B 模型在工具调用任务上击败 200×参数量的闭源大模型，说明 in-context 方式在专业 schema 任务上的边际收益已经触顶，weight-level 内化是下一步的必经之路。
- **Oracle (88.4%) 与实测 (69%) 之差**：约 20 个百分点的 gap，说明 BioTool fine-tuned caller 还有提升空间，但已经能 capture 约 78% 的工具调用收益。
- **覆盖广度的价值**：34 个工具横跨变异 / 基因组 / 蛋白质组 / 进化 / 通识 5 个子领域，使下游能处理跨学科查询（如同时需要 NCBI gene 与 UniProt protein 信息的复杂问题）。

## 亮点与洞察
- **反向构造范式**：先有「正确的 API 响应」再让 LLM 反推 query 的设计，几乎从根本上消灭了「query 与 ground-truth API call 不对齐」这一传统 tool-use 数据集的最大噪声源，值得迁移到其他垂直领域（如金融 API、地理 API、电商 API）。
- **小模型 specialization 路线的胜利**：再次印证「与其追求 200B 通用模型，不如把 4B 模型在特定 schema 上微调到极致」——对学术界 + 中小团队是非常重要的方向信号。
- **Oracle 上限分析的实验设计**：用 Oracle API call 给出 88.4% 的天花板，再用 fine-tuned caller 给出 69% 的实测值，让读者清楚知道「数据集 intrinsic quality」与「caller 实现差距」分别占多少，方法论值得借鉴。

## 局限与展望
- **工具范围仍偏窄**：34 个工具相对生物医学整体（数百 API）还是小，未来需要扩到 chemistry / proteomics imaging / clinical trial database 等。
- **人工核验成本高**：7,040 条需要专家逐条 review，难以无限 scale，下一步可以探索「专家 review + active learning 选样」的混合范式。
- **下游 base LLM 仍是 closed-source**：最终答案质量评估用的是 GPT-5.1 作为 answer generator，无法完全开源复现；理想情况下应该有一个全开源 stack。
- **没有探索多工具串行调用**：当前每条样本只对应单次 API call，复杂生物学问题往往需要多步 BLAST → annotation → cross-reference 的链式调用，未来需要扩展到 multi-step tool use。

## 相关工作与启发
- **vs Toolformer / Gorilla**：通用工具数据集，覆盖广但生物医学工具占比微乎其微，schema 复杂度也远低于 NCBI/Ensembl。本文专攻生物医学，schema 严格度更高。
- **vs GeneGPT**：同样针对 NCBI，但 GeneGPT 用 ICL 方式只能挂少量工具；BioTool 通过 SFT 一次性把 34 个工具的全套 schema 内化进权重。
- **vs ChemCrow / SciAgent**：scientific agent 路线，强调 agent 编排；BioTool 互补——提供高质量训练语料，可以用来强化 ChemCrow 那类 agent 中的 tool caller 子模块。
- **vs Biomni**：通用生物医学 agent，工具集仍偏小；BioTool 的数据可以直接补强其 caller。

## 评分
- 新颖性: ⭐⭐⭐⭐ 反向 query 合成 + 三层过滤的数据集构造范式扎实，但每个组件并非完全首创
- 实验充分度: ⭐⭐⭐⭐ 同时报告 API-quality benchmark 与人工生物学家 head-to-head 评估，Oracle 上限分析尤其加分
- 写作质量: ⭐⭐⭐⭐ 数据集 → 实验 → 人工评估三段式叙述清晰
- 价值: ⭐⭐⭐⭐⭐ 7,040 条高质量数据 + 数据集 / 代码开源，对生物医学 LLM 社区是真正可用的基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine](ritek_a_dataset_for_large_language_models_complex_reasoning_over_textual_knowled.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[ACL 2026\] RePrompT: Recurrent Prompt Tuning for Integrating Structured EHR Encoders with Large Language Models](reprompt_recurrent_prompt_tuning_for_integrating_structured_ehr_encoders_with_la.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)

</div>

<!-- RELATED:END -->
