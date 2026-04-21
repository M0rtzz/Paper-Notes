---
title: >-
  [论文解读] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?
description: >-
  [ACL 2025][LLM/NLP][LLM医学Agent] 系统综述 LLM-based Agent 在医学中的四层架构（Profile/临床规划/医学推理/外部能力增强）、四大应用场景和评估框架，覆盖 2022-2024 年 60 篇研究，提出四种 Agent 运作范式并识别幻觉管理、多模态整合和伦理等关键挑战。
tags:
  - ACL 2025
  - LLM/NLP
  - LLM医学Agent
  - 临床决策
  - 多Agent协作
  - 医学推理
  - 架构综述
---

# A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?

**会议**: ACL 2025  
**arXiv**: [2502.11211](https://arxiv.org/abs/2502.11211)  
**代码**: 无  
**领域**: LLM Agent / 医学AI  
**关键词**: LLM医学Agent, 临床决策, 多Agent协作, 医学推理, 架构综述

## 一句话总结

系统综述 LLM-based Agent 在医学中的四层架构（Profile/临床规划/医学推理/外部能力增强）、四大应用场景和评估框架，覆盖 2022-2024 年 60 篇研究，提出四种 Agent 运作范式并识别幻觉管理、多模态整合和伦理等关键挑战。

## 研究背景与动机

**领域现状**：LLM 在文本理解、生成和推理方面具有强大能力，基于 LLM 的 Agent 系统已在创意写作、复杂决策等多个领域取得显著成功。在医学领域，LLM-based Agent 已被应用于诊断支持（Kim 2024a）、患者沟通（Mukherjee 2024）和医学教育（Yu 2024）等任务。

**现有痛点**：标准 LLM 主要处理文本，而医学场景需要与外部知识库、临床指南和医疗系统深度整合。LLM-based Agent 通过外部知识检索、任务规划和工具调用实现了从"回答问题"到"解决问题"的跃迁，但面临幻觉管理、安全性和部署实施等系统性挑战。

**核心矛盾**：医学领域对 Agent 有四项独特且严格的要求——多模态整合（需处理文本、影像、实验室结果等异构数据）、临床协作（需促进多学科信息共享和人-AI 协作）、精确性和可靠性（直接影响患者安全）、透明性和可追溯性（临床决策必须可审计可解释）。当前技术与这些要求之间存在显著差距。

**本文目标** → 该领域缺乏对 Agent 架构设计、应用场景和评估框架的全面系统性综述。

**切入角度**：以"距离 Baymax 还有多远"为隐喻切入，从理想化的全能医疗 AI 助手倒推当前技术的差距与挑战。

**核心 idea**：从架构、应用和评估三个维度全面审视医学 LLM Agent 的能力边界和改进方向，用 Baymax 愿景锚定技术发展的长期目标。

## 方法详解

### 整体框架

本文提出 LLM-based 医学 Agent 的四层架构模型，包含系统配置（Profile）、临床规划（Clinical Planning）、医学推理（Medical Reasoning）和外部能力增强（External Capacity Enhancement），并定义四种 Agent 运作范式：单 Agent（Single Agent）、顺序任务链（Sequential Task Chain）、协作专家（Collaborative Experts）和迭代进化（Iterative Evolution）。

### 关键设计

1. **临床规划——从任务分解到自适应协作**:

    - 功能：将复杂医学任务分解为可执行的子任务，支持 Agent 与临床工具和数据库的交互
    - 核心思路：提出四种递进式规划架构——**任务分解**（将高层目标分解为数据摄入、假设生成、治疗规划、风险评估等步骤）；**多 Agent 跨科协作**（为放射科、病理科、检验科等分配专业 Agent，通过标准化协议聚合发现并精炼诊断，如 EHRagent Tang 2024）；**自适应规划**（MDAgents 框架 Kim 2024a 基于实时数据和任务复杂度动态调整决策，联邦学习增强适应性）；**迭代自进化**（Agent Hospital Li 2024b 维护过去病例的经验库，自主整合新数据、从历史结果学习，持续提升准确性和可靠性）
    - 设计动机：医学任务的复杂性需要层次化的规划能力——从简单任务的单 Agent 处理到复杂病例的多 Agent 协作

2. **医学推理——从链式推理到群体共识**:

    - 功能：结构化逻辑推理过程，增强诊断准确性和透明度
    - 核心思路：四种推理范式——**多步诊断推理**（Chain-of-Thought 生成逐步推理、Tree-of-Thought 并行探索多假设并丢弃低概率选项）；**反思式决策**（受 ReAct 框架启发，在推理和行动之间交替以发现不一致）；**协作群体推理**（多 Agent 框架分配基层医生和专家独立分析后通过共识聚合，减少偏差）；**记忆增强推理**（长期记忆模块积累知识和临床经验，实现持续学习和个性化洞察）
    - 设计动机：临床推理的不确定性和复杂性需要多层保障——单一推理链不够，需要反思、协作和记忆的联合增强

### 损失函数 / 训练策略

本文为综述论文。总结的主要训练/对齐策略包括：强化学习用于迭代自进化框架中的持续改进（Agent Hospital）、联邦学习用于跨机构的自适应规划（MDAgents）、RLHF 对齐用于患者交互安全（Polaris）、以及基于知识图谱和临床指南的检索增强生成。

## 实验关键数据

### 主实验（应用场景覆盖分析）

| 应用场景 | 代表系统 | 框架类型 | 关键特征 |
|---------|---------|---------|---------|
| 临床决策 | MDAgents (Kim 2024c) | 协作专家 | 多 Agent 结构化讨论提升诊断 |
| 临床决策 | Dutta & Hsiao 2024 | 自适应规划 | 模拟医患交互精炼推理，MedQA 上超基线 |
| 数据分析 | ColaCare (Wang 2024b) | 协作专家 | MIMIC-III/IV 死亡率预测和再入院分析 |
| 文档 | Sporo AI Scribe (Lee 2024) | 单 Agent | 解决临床文档变异性和复杂度 |
| 训练模拟 | Agent Hospital (Li 2024b) | 迭代进化 | 大规模重复训练生成完整交互 |
| 训练模拟 | SurgBox (Wu 2024) | 协作专家 | 手术流程训练环境，经真实手术记录验证 |
| 服务优化 | Polaris (Mukherjee 2024) | 顺序任务链 | 通用通信 + 任务特定 Agent 确保安全交互 |

### 消融实验（评估框架对比）

| 基准类别 | 代表基准 | 特点 | 局限 |
|---------|---------|------|------|
| 静态问答 | MedQA, MedMCQA, MMLU | 预设答案，测试知识 | 不反映交互式临床决策 |
| 工作流模拟 | MedChain (12163 病例), ClinicalLab (24 科 150 病) | 多阶段临床推理 | 标准化困难 |
| 自动评估 | AI-SCE, RJUA-SPs | 减少人工依赖 | 复杂场景可能不准确 |

### 关键发现

- **四种 Agent 范式适配不同复杂度**：简单任务用单 Agent，中等复杂度用顺序任务链，高复杂度用协作专家，需持续学习用迭代进化
- **幻觉是最大技术风险**：多 Agent 场景中错误可能在协作传播中放大，MedHallBench 和 HaluEval 揭示现有验证机制不足
- **静态基准不够**：MedQA 等固定问答无法捕捉临床决策的动态和交互特性，需要工作流级评估
- **偏差问题受当**：BiasMedQA 评估七种偏差，发现 SOTA 医学 LLM 的某些偏差场景准确率低至 50%
- **DeepSeek-R1 启示**：强化学习 + 长链推理可能是改进医学 Agent 自主推理的重要方向

## 亮点与洞察

- **四层架构模型**清晰地解构了医学 Agent 的能力组件——Profile 定义角色边界、Clinical Planning 管理任务分解、Medical Reasoning 保障推理质量、External Capacity Enhancement 扩展知识来源
- **四种 Agent 范式**的递进关系具有实用价值——为不同复杂度的医学场景提供了架构选型指南
- **"Baymax"隐喻**有效锚定了长期目标：距理想的全能医疗 AI 助手仍有巨大差距，主要瓶颈在幻觉管理、多模态整合和可信推理
- **Agent Hospital 的迭代进化思想**特别有启发——通过模拟患者进行大规模重复训练实现自主进化，绕过了真实临床数据获取的瓶颈

## 局限与展望

- 覆盖范围限于 2022-2024 年且以英语论文为主，可能遗漏其他语言的重要工作
- 综述以叙述性为主，缺乏不同 Agent 系统在统一基准上的定量对比
- 对隐私保护的讨论偏浅——仅提到差分隐私和匿名化，未深入探讨联邦学习场景下 Agent 协作的隐私风险
- 与物理系统（如手术机器人、护理机器人）的整合讨论停留在展望层面，缺乏具体技术路径分析
- 未详细讨论医学 Agent 的成本效益问题——开发和部署成本可能使中小型医疗机构难以受益

## 相关工作与启发

- Agent 通用综述如 Xi 2023 覆盖 LLM Agent 的总体方法论，本文聚焦医学特有需求（安全性、可追溯性、多模态整合）进行专业化分析
- 医学 AI 综述如 Topol 2019 覆盖传统 AI 医疗应用，本文聚焦 LLM 时代的 Agent 范式变革
- **启发**：(1) 医学 Agent 的核心挑战不是单一任务性能而是系统级的安全可靠性；(2) 迭代自进化框架（Agent Hospital）可能是突破临床数据瓶颈的关键路径；(3) 多 Agent 协作的科室组织模式值得借鉴到其他需要跨学科协作的复杂领域

## 评分

⭐⭐⭐ 以清晰的四层架构和四种范式为医学 Agent 领域提供了有价值的组织框架和发展地图，但叙述性综述缺乏定量对比、实操指导不足，且对核心挑战（幻觉、偏差、隐私）的解决方案讨论停留在方向层面。

<!-- RELATED:START -->

## 相关论文

- [Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](human_nlp_cooperation_survey.md)
- [MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)

<!-- RELATED:END -->
