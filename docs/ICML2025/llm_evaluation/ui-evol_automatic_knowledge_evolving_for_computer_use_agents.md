---
title: >-
  [论文解读] UI-Evol: Automatic Knowledge Evolving for Computer Use Agents
description: >-
  [ICML 2025][计算机操作代理] 提出UI-Evol即插即用模块，通过Retrace（从截图还原实际动作序列）和Critique（对比外部知识诊断偏差并修正）两阶段自主进化GUI任务知识，在OSWorld基准上将Agent S2的成功率从19.5%提升到22%+，同时将行为标准差降低约4倍，显著增强了计算机操作代理的可靠性。
tags:
  - ICML 2025
  - 计算机操作代理
  - 知识进化
  - 自我改进
  - GUI交互
  - 轨迹回溯
---

# UI-Evol: Automatic Knowledge Evolving for Computer Use Agents

**会议**: ICML 2025  
**arXiv**: [2505.21964](https://arxiv.org/abs/2505.21964)  
**代码**: 未公开  
**领域**: LLM Agent / GUI自动化  
**关键词**: 计算机操作代理, 知识进化, 自我改进, GUI交互, 轨迹回溯

## 一句话总结

提出UI-Evol即插即用模块，通过Retrace（从截图还原实际动作序列）和Critique（对比外部知识诊断偏差并修正）两阶段自主进化GUI任务知识，在OSWorld基准上将Agent S2的成功率从19.5%提升到22%+，同时将行为标准差降低约4倍，显著增强了计算机操作代理的可靠性。

## 研究背景与动机

### 领域现状

计算机操作代理（Computer Use Agent）旨在自动与GUI交互完成任务。当前主流方法分为：(1) 单体端到端agent（如UI-TARS、SeeClick），直接用一个模型预测动作；(2) 模块化agent（如Agent S2），将任务分解为规划、感知、执行等模块。两类方法都依赖大型多模态模型（LMM）作为核心推理引擎。

为弥补LMM自身知识不足，Agent S2等先进方法引入**外部知识检索**：给定任务后先通过 Perplexica 搜索网页获取操作步骤，作为soft prior辅助规划。例如"如何在LibreOffice中全选文本"→检索到操作指南→作为提示输入agent。

### 核心问题：知识-执行鸿沟

作者对Agent S2+GPT-4o进行抽样分析发现：即使**90%的检索知识被人类判定为"正确"，agent的最佳成功率仅41%**。深入分析发现检索知识存在以下实际缺陷：

**遗漏必要中间步骤**：人类认为理所当然但agent无法推理的操作步骤

**初始条件假设不一致**：知识假设的起始状态与实际环境不符

**建议过于复杂的操作**：如建议鼠标拖拽选择文本，但agent很难精确执行拖拽；实际用 Ctrl+A 更可靠

### 核心洞察

检索到的知识本身大体可靠（90%正确率），但缺乏**与实际GUI环境和agent能力的对齐**。需要一种机制让知识在实际交互中自我进化。

## 方法详解

### 整体框架

任务指令 → 外部知识检索 → Agent执行（产生轨迹截图） → **UI-Evol**: Retrace阶段（从截图还原客观动作） → Critique阶段（对比知识、诊断偏差、生成修正） → 更新知识库 → 下轮执行使用进化后知识

UI-Evol是即插即用模块，可嵌入任何基于知识的agent系统。

### 关键设计

#### 1. Retrace阶段（轨迹回溯）

**功能**：从agent执行过程中每一步的前后截图 $(O_t, O_{t+1})$ 出发，用LMM分析UI变化，还原出**客观动作序列**（Objective Action Sequence）。

**为什么不直接用agent输出的动作日志**：agent存在幻觉和感知错误，其"主观动作序列"（Subjective Action Sequence）只代表意图而非实际效果。例如agent以为自己选中了全部文本，实际上只选中了部分。

**实现方式**：对每步$t$，将 $O_t$（操作前截图）和 $O_{t+1}$（操作后截图）输入GPT-4o，输出该步实际发生的动作描述$\hat{A}_t$和可见结果。如果前后截图无变化则标记为null。最终合并为文本格式的客观动作序列。

**设计动机**：基于视觉观察的回溯消除了无效动作的噪音，为后续Critique提供可靠的"事实基准"。

#### 2. Critique阶段（知识批判与修正）

**功能**：利用检索到的外部知识作为参考锚点，通过链式推理分析客观动作序列与知识之间的偏差，生成修正后的知识。

**四步结构化推理**（Chain-of-Thought）：
1. **完成度评估**（Completion Assessment）：任务是否完成？是否执行了多余操作？
2. **偏差检测**（Deviation Detection）：逐步对比客观动作与知识计划，识别每个偏差并推断根因（含9类根因分类：感知误解、知识缺口、语法错误、环境问题、无效假设等）
3. **替代方案探索**（Alternative Exploration）：agent是否尝试了知识计划之外的有效替代方案？
4. **改进与理据**（Mitigation with Rationales）：针对每个偏差根因生成具体修正方案，输出格式与原知识一致的精炼计划（最多15步）

**设计动机**：将检索知识作为参考锚点而非直接修正目标，利用其90%的可靠性作为对比基准。

## 实验关键数据

### 主实验（OSWorld基准，369个任务）

| 方法 | 基础模型 | 平均成功率↑ | 标准差↓ | 报告值 |
|------|---------|-----------|---------|--------|
| Agent S2 (原文) | GPT-4o | - | - | 21.1% |
| Agent S2* (复现) | GPT-4o | 19.5% | ±1.00 | - |
| **+ UI-Evol** | GPT-4o | **22.0%** | **±0.71** | - |
| Agent S2* (复现) | OpenAI-o3 | 25.6% | ±1.09 | - |
| **+ UI-Evol** | OpenAI-o3 | **28.3%** | **±0.26** | - |

### 消融实验

| 配置 | 平均成功率 | 说明 |
|------|-----------|------|
| 随机选择轨迹 + UI-Evol | 22.0% | SSR=70% |
| 基于完成度选择 + UI-Evol | 22.7% | SSR=85%，略优 |

UI-Evol对轨迹选择质量不敏感，即使输入较差的轨迹也能有效进化知识。

### 知识迁移实验

用OpenAI-o3生成的轨迹进化出的知识，迁移给GPT-4o使用 → 成功率22.4%，与GPT-4o自身轨迹进化的22.0%相当，说明进化知识可跨模型迁移。

### 关键发现

- **o3模型的标准差降低至±0.26**：约为GPT-4o的1/4，更强推理能力的模型更能利用进化知识
- **之前被忽视的不稳定性问题**：即使固定所有超参数（含temperature=0），Agent S2在3次重复实验中仍有显著波动（±1.0-1.09），这在先前研究中从未被系统分析

## 亮点与洞察

- **知识-执行鸿沟的精准诊断**：90%知识正确但只有41%成功率——这个gap的量化分析非常有洞察力，揭示了知识检索不等于知识可用
- **客观vs主观动作序列的区分**：用截图还原真实动作而非信任agent日志，这是真正做到"观察事实而非相信叙述"
- **案例说明极具说服力**：LibreOffice全选文本的例子——知识说"拖拽选择"，agent拖拽失败；UI-Evol修正为"Ctrl+A"——完美体现了从理论正确到实践可用的差距
- **并行评估框架**：30个Azure并行实例，10小时→2.5小时——这本身就是对agent评估基础设施的贡献

## 局限与展望

- **需要一次完整执行轨迹**：进化知识前agent必须先执行一遍任务（即使失败），增加了计算成本（每任务约$0.22，全基准$81.18）
- **仅一轮进化**：当前只做一次retrace+critique，理论上多轮迭代可能进一步提升
- **依赖强LMM做Retrace和Critique**：GPT-4o做Retrace、o3做Critique，如果底座模型能力不足则整个流程可能退化
- **未探索失败任务的知识利用**：当前主要利用成功或部分成功的轨迹，失败轨迹中也蕴含"不该做什么"的信息

## 相关工作与启发

- **vs Agent S2 (无UI-Evol)**：Agent S2用Perplexica检索知识作soft prior但不做修正，UI-Evol证明通过环境交互反馈进化知识可获得2.5%绝对提升
- **vs Voyager (技能库)**：Voyager在Minecraft中通过成功经验构建技能代码库，UI-Evol思路类似但面向更复杂的桌面GUI环境，且知识表示为自然语言计划而非代码
- **vs Self-Refine / CRITIC**：这些是通用的LLM自我修正方法，UI-Evol的关键区别是引入了基于截图的客观轨迹作为修正依据，而非纯粹的LLM自反思

## 评分

- 新颖性: ⭐⭐⭐⭐ Retrace+Critique的两阶段知识进化框架新颖，客观动作序列的概念有价值
- 实验充分度: ⭐⭐⭐⭐ 3次重复实验+标准差报告+消融+知识迁移+并行框架，但仅在OSWorld一个基准上验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，案例说服力强
- 价值: ⭐⭐⭐⭐ 即插即用设计+首次系统分析agent不稳定性，对GUI agent领域有实际推动

<!-- RELATED:START -->

## 相关论文

- [EvoWiki: Evaluating LLMs on Evolving Knowledge](../../ACL2025/llm_evaluation/evowiki_evaluating_llms_on_evolving_knowledge.md)
- [Communicating Activations Between Language Model Agents](communicating_activations_between_language_model_agents.md)
- [EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)
- [Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](../../NeurIPS2025/llm_evaluation/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)

<!-- RELATED:END -->
