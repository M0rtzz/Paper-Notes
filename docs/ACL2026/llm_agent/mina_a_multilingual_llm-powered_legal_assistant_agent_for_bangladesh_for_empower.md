---
title: >-
  [论文解读] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh
description: >-
  [ACL 2026][LLM Agent][法律助手] 开发 Mina——面向孟加拉国法律场景的多语言 LLM 法律助手，通过两阶段 RAG 流水线精准检索法案和条款，配合工具链和多语言嵌入，在孟加拉律师资格考试中取得 75-80% 的通过成绩，法律咨询成本仅为传统方式的 0.12-0.61%。
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "法律助手"
  - "多语言Agent"
  - "RAG"
  - "孟加拉国法律"
  - "低资源语言"
---

# Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh

**会议**: ACL 2026  
**arXiv**: [2511.08605](https://arxiv.org/abs/2511.08605)  
**代码**: [GitHub](https://github.com/)  
**领域**: LLM Agent / 法律NLP  
**关键词**: 法律助手, 多语言Agent, RAG, 孟加拉国法律, 低资源语言

## 一句话总结
开发 Mina——面向孟加拉国法律场景的多语言 LLM 法律助手，通过两阶段 RAG 流水线精准检索法案和条款，配合工具链和多语言嵌入，在孟加拉律师资格考试中取得 75-80% 的通过成绩，法律咨询成本仅为传统方式的 0.12-0.61%。

## 研究背景与动机

**领域现状**：孟加拉国司法系统积压 370-440 万案件，仅有 2,100 名法官（每 9 万人 1 名），民事纠纷拖延数十年，律师费用高昂且不受监管，公共法律援助资金有限。

**现有痛点**：(1) AI 法律助手缺乏孟加拉语支持，且未针对孟加拉法律管辖区适配；(2) 孟加拉法律体系根植于殖民时代法典，包含大量波斯语源术语，英语主导的模型无法有效处理；(3) 低收入人群面临法律语言复杂、程序不透明、费用高三重障碍。

**核心矛盾**：语言、法律体系和资源的三重低资源性——孟加拉语 NLP 工具匮乏、法律术语高度专业化且跨语言混合、目标用户缺乏法律和数字素养。

**本文目标**：构建本地化的多语言法律助手，能起草法律文件、引用法规、将复杂法律语言翻译为通俗孟加拉语解释。

**切入角度**：组合成熟组件（多语言嵌入、RAG、LangGraph Agent）并针对双语低资源法律环境深度适配，而非追求单一模块创新。

**核心 idea**：两阶段 RAG 流水线（先检索法案概要再检索具体条款）+ 自定义法律词典 + 多Agent工作流，实现管辖区特定的精准法律回答。

## 方法详解

### 整体框架
系统以 Orchestrator Agent 为中心，评估用户输入、对话历史和文档决定响应路径。内部上下文不足时触发两阶段 RAG：先用 Cohere 多语言嵌入检索相关法案，再在法案内检索具体条款。外部工具（网页搜索、文档解析等）按需调用。

### 关键设计

1. **两阶段 RAG 流水线 (Two-Stage RAG)**:

    - 功能：精准检索孟加拉法律条文，避免跨法案内容混淆
    - 核心思路：构建两个独立向量数据库——法案数据库（595 部法案的 LLM 摘要）和条款数据库（18,023 条分块索引）。查询时先用语义关键词检索 top-5 法案，再用法案 ID 过滤条款数据库找到 top-10 相关条款。检索结果经相关性检查，不足则改写查询重试
    - 设计动机：朴素 RAG 经常将不相关法案的条款混合生成回答，两阶段设计在法案级保证广度、条款级保证精度

2. **双Agent架构**:

    - 功能：分离决策和检索职责，增强可维护性
    - 核心思路：Orchestrator Agent 评估查询上下文决定是直接回答还是触发检索；RAG Agent 管理端到端检索流程。两者基于 LangGraph 状态机运行，支持跨轮持久记忆
    - 设计动机：职责分离使系统模块化，便于独立调试和扩展

3. **工具链与法律词典**:

    - 功能：处理辅助任务和殖民时代法律术语
    - 核心思路：8 个专用工具包括文档解析器（支持 pptx/docx/pdf）、关键词生成器（LLM + 正则 fallback）、网页搜索（DuckDuckGo）、问题相关性分析器、自定义法律词典（解释波斯语源和殖民时代术语）、社会经济模拟模块等
    - 设计动机：孟加拉法律文本中大量波斯语源和殖民时代术语是模型理解的主要障碍，法律词典直接弥补了这一领域知识缺口

### 损失函数 / 训练策略
本文不涉及模型训练，系统基于预训练 LLM 的提示工程和 RAG 构建。评估了 GPT-4o、Gemini 系列、Llama 系列、Qwen 等 13 个模型。

## 实验关键数据

### 主实验（律师资格考试 MCQ，2-Step RAG + Tools）

| 模型 | 2022 年 | 2023 年 | 说明 |
|------|--------|--------|------|
| Gemini-2.5-Flash | **77.0%** | **77.0%** | 最高分，匹配/超越人类平均 |
| GPT-4o | 73.6% | 72.2% | 强基线 |
| Llama3.1-70B | 42.4% | 46.2% | 开源最佳 |
| Qwen3-30B | 70.8% | 72.4% | 开源次佳 |
| w/o RAG (GPT-4o) | 18.6% | 19.2% | RAG 的重要性 |

### RAG 消融实验

| 配置 | MCQ 准确率 | 说明 |
|------|-----------|------|
| w/o RAG | 18.6% | 无检索，几乎随机猜测 |
| Naive RAG | 62.4% | 单阶段检索，常混淆法案 |
| 2-Step RAG | 69.2% | 两阶段检索，精度提升 |
| 2-Step RAG + Tools | 73.6% | 完整系统 |

### 关键发现
- RAG 是系统的核心：无 RAG 时 GPT-4o 仅 18.6%（接近随机选择的 25%），加入 2-Step RAG 后飙升至 69.2%
- 两阶段 RAG 一致优于朴素 RAG 约 7-10 个百分点，验证了分层检索设计的必要性
- 小模型（<4B）即使有 RAG 也表现很差，法律推理对模型规模有底线要求
- 成本分析显示 Mina 运营成本仅为传统法律咨询的 0.12-0.61%，成本节省 99.4-99.9%

## 亮点与洞察
- 系统级创新而非模块级创新的典范——虽然每个组件都是现有技术，但针对双语低资源法律场景的深度适配产生了实用价值
- 两阶段 RAG 的设计简单有效：先宏观定位法案再微观定位条款，避免了跨法案混淆这一关键问题
- 通过律师资格考试是一个很有说服力的评估方式，直接验证了系统在真实法律场景中的可用性

## 局限与展望
- 法律数据库当前仅覆盖 595 部法案，孟加拉法律体系还有大量未数字化的法规和判例
- 未评估在真实用户场景（非考试）中的表现，考试题目可能无法完全代表实际法律咨询需求
- 社会经济模拟模块的实际效果和评估细节不够充分
- 模型的法律建议不具备法律效力，存在误用风险

## 相关工作与启发
- **vs 通用法律AI**: 现有法律 AI 系统（如 Harvey AI）面向英美法系，无法处理孟加拉法律体系的特殊性
- **vs 通用 RAG**: 朴素 RAG 在法律场景中因法案混淆导致严重错误，两阶段设计是领域适配的关键
- **vs 多语言 LLM**: 即使支持孟加拉语的 LLM 也缺乏管辖区特定知识，RAG + 法律词典弥补了这一缺口

## 评分
- 新颖性: ⭐⭐⭐ 系统集成创新为主，单模块创新有限
- 实验充分度: ⭐⭐⭐⭐ 13个模型、三阶段考试评估、法律专家评审
- 写作质量: ⭐⭐⭐⭐ 背景详实，问题动机清晰
- 价值: ⭐⭐⭐⭐⭐ 真实解决低资源法律可及性问题，社会影响大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery](moose-copilot_a_web-based_interactive_assistant_for_unified_exploratory_and_fine.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](../../ACL2025/llm_agent/legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)

</div>

<!-- RELATED:END -->
