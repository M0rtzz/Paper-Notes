---
title: >-
  [论文解读] Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems
description: >-
  [ACL2025][LLM/NLP][表征性伤害] 通过对 12 位负责评估 LLM 系统表征性伤害（representational harms）的从业者进行半结构化访谈，发现公开可用的测量工具普遍无法满足实践者需求——要么因效度/特异性不足而"不好用"（not useful）…
tags:
  - "ACL2025"
  - "LLM/NLP"
  - "表征性伤害"
  - "LLM评估"
  - "测量工具"
  - "半结构化访谈"
  - "实践者需求"
  - "测量理论"
---

# Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems

**会议**: ACL2025  
**arXiv**: [2506.04482](https://arxiv.org/abs/2506.04482)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 表征性伤害, LLM评估, 测量工具, 半结构化访谈, 实践者需求, 测量理论

## 一句话总结

通过对 12 位负责评估 LLM 系统表征性伤害（representational harms）的从业者进行半结构化访谈，发现公开可用的测量工具普遍无法满足实践者需求——要么因效度/特异性不足而"不好用"（not useful），要么因组织/制度壁垒而"用不了"（not used），并基于测量理论和实用测量框架提出系统性改进建议。

## 背景与动机

1. **表征性伤害的严重性**: LLM 系统可能以不利方式呈现某些社会群体（刻板印象、贬低、忽视），测量和缓解此类伤害对部署至关重要
2. **测量工具大量涌现但使用率低**: NLP 社区已发布大量数据集、指标、工具和基准（如 StereoSet、BBQ、ToxiGen 等），但实际被从业者采用的极少
3. **研究假设与实践需求脱节**: 已有 HCI 研究表明，研究者开发的负责任 AI 工具往往与从业者真实需求不匹配，但此前无人聚焦 LLM 表征性伤害领域
4. **表征性伤害本质上抽象且有争议**: 不同使用场景、语言、文化下的含义各异，难以精确定义和测量，使得通用工具难以适用
5. **从业者视角缺失**: 现有对测量工具缺陷的批评多来自研究者的技术评估，缺少从业者自身面临的实际挑战和制度约束的系统调研
6. **测量理论在 NLP 中应用不足**: 社会科学中成熟的测量理论（效度、信度框架）和实用测量（pragmatic measurement）理念尚未被 NLP 工具设计者充分采纳

## 方法详解

### 整体框架

"质性研究"范式：定义 7 项测量工具期望属性→设计半结构化访谈提纲→招募从业者→访谈至饱和→主题分析编码→基于测量理论提出建议。核心贡献不是算法而是对实践-研究鸿沟的系统诊断。

### 关键设计

#### 1. 测量工具期望属性体系（7 Desiderata）

- **功能**: 定义测量工具应满足的 7 项属性——效度（validity）、信度（reliability）、特异性（specificity）、可扩展性（extensibility）、可扩缩性（scalability）、可解释性（interpretability）、可操作性（actionability）
- **为什么**: 为访谈提供结构化的脚手架，确保系统覆盖从业者可能遇到的各类挑战；同时为后续建议提供分析维度
- **怎么做**: 基于作者团队自身测量表征性伤害的经验 + 对 NLP 文献中工具评估论文的系统综述归纳得出

#### 2. 半结构化访谈设计与执行

- **功能**: 对 12 位从业者（来自大科技公司、AI 创业公司、非营利组织，角色涵盖研究工程师、应用科学家、数据工程师等）进行每人 1 小时的访谈
- **为什么**: 半结构化访谈允许同时获取围绕预设维度的结构化信息和从业者自发提出的新挑战；样本量遵循质性研究的饱和原则（连续多次访谈不再发现新挑战即停止）
- **怎么做**: 先让受访者描述其角色和评估流程；再以 7 项 desiderata 为引导追问使用/放弃测量工具的原因；通过专业网络、社交媒体、冷邮件和滚雪球法招募；每人 $75 报酬；获微软研究 IRB 批准

#### 3. 主题分析（Thematic Analysis）

- **功能**: 对访谈转录文本做归纳-演绎双重编码，识别从业者面临的挑战主题
- **为什么**: 归纳法捕获预设框架之外的新发现（如数据污染问题），演绎法确保对 7 项 desiderata 的系统覆盖
- **怎么做**: 第一作者初始编码→发现新类别后扩展编码本→重新编码→全体作者讨论合成主题→至少一名其他作者对每份转录做独立编码验证→分歧通过讨论解决

#### 4. 基于测量理论的建议框架

- **功能**: 将从业者挑战映射到测量理论（systematization → operationalization → application → interrogation）和实用测量框架，提出针对性改进建议
- **为什么**: NLP 工具设计者常跳过"系统化定义概念"直接"操作化"（造工具），导致效度无法评估
- **怎么做**: 建议工具设计者：(a) 不跳过 systematization，明确文档化被测概念定义；(b) 提供效度和信度的证据及自检工具；(c) 发布测量结果分布以提升可解释性；(d) 设计模块化可扩展工具让从业者可定制

## 实验关键数据

### 表1: 受访者概况（12 人）

| 角色类型 | 所属机构类型 | 人数 |
|----------|-------------|------|
| 研究工程师 | 大科技公司 | 2 |
| 应用/研究科学家 | 大科技公司 | 3 |
| 顾问/NLP专家 | AI 创业公司/非科技大企 | 3 |
| 研究员 | AI 创业公司/非营利 | 3 |
| 数据工程师 | 大型非科技公司 | 1 |

受访者分布于北美和欧洲，覆盖大科技公司、AI 创业公司、非营利机构等多种组织类型。

### 表2: 从业者在各 desideratum 上的挑战汇总

| 期望属性 | 核心发现 | 提及人数 |
|----------|----------|----------|
| **效度 (Validity)** | 概念定义不清、数据标注错误、数据污染使信任崩塌 | 几乎所有人 |
| **特异性 (Specificity)** | 通用工具无法匹配特定系统/场景/文化语境，被迫自建 | 所有 12 人 |
| **可解释性 (Interpretability)** | 工具产出"一个数字"但无法理解其含义和阈值 | 6 人 |
| **可操作性 (Actionability)** | 无明确缓解策略则优先级下降 | 6 人 |
| **信度 (Reliability)** | 理论上重要但实践中不被优先考虑 | 0 人主动提及 |
| **可扩缩性 (Scalability)** | 仅在线评估场景下成为阻碍 | 少数 |
| **制度壁垒** | 安全合规要求、数据许可、组织文化不支持 | 多人 |

关键发现：效度和特异性是"一票否决"项——不满足则直接放弃工具；信度虽重要但因从业者先在效度关被筛掉，从未走到评估信度这一步。

### 其他核心发现

- **数据污染是特有痛点**: 50% 受访者对使用任何公开基准数据集感到不安，因 LLM 训练数据不透明
- **文化语境缺失**: 表征性伤害高度依赖文化语境，通用刻板印象数据集无法适配特定系统场景
- **低资源语言更难**: 1/3 受访者提及非英语语言评估面临工具几乎空白的困境
- **自建工具成常态**: 多位受访者因无法使用公开工具而被迫从专有数据中自建测量手段

## 亮点

- **填补关键空白**: 首个系统调研 LLM 表征性伤害测量工具与从业者需求之间鸿沟的工作
- **双重挑战模型**: "not useful"（工具本身缺陷）vs "not used"（制度/实践壁垒）的区分框架清晰有力
- **跨学科视角**: 将社会科学的测量理论和实用测量引入 NLP 工具设计，建议具体且可操作
- **方法论扎实**: 遵循质性研究规范（饱和原则、IRB 审批、双重编码、positionality 声明），在 ACL 会议中较为少见且值得推广

## 局限与展望

- **样本偏倚**: 招募困难（73 封冷邮件仅 1 人接受访谈），受访者可能偏向"遇到过挑战"的从业者，无法推断挑战的普遍性
- **仅英语访谈**: 虽个别受访者提及低资源语言，但研究整体以英语评估为中心
- **无定量验证**: 定性发现（如"效度是首要考虑"）未通过大规模问卷等定量手段验证
- **未覆盖所有伤害类型**: 聚焦表征性伤害，对分配性伤害（allocative harms）的工具适用性未做系统探讨

## 与相关工作的对比

### vs Blodgett et al. (2020) 的偏见批评综述

Blodgett 等人从**研究者视角**批评 NLP 偏见论文中概念动机不清、测量方法与声称不匹配。本文互补地从**从业者视角**出发，发现的效度和特异性问题与研究者批评高度一致，但额外揭示了制度壁垒（安全合规、组织文化）这一仅在实践中才暴露的挑战维度。

### vs Holstein et al. (2019) 的公平性工具需求调研

Holstein 等人调研了从业者对 ML 公平性工具的需求，聚焦**分配性伤害**和**传统预测模型**。本文将调研范围扩展到**表征性伤害**和**LLM 系统**，发现许多挑战（如数据污染、文化特异性）是 LLM 时代特有的新问题。

### vs Delobelle et al. (2024) 的可操作性分析

Delobelle 等人从技术角度分析测量工具缺乏可操作性的原因。本文通过从业者访谈补充了**实践层面**的可操作性障碍（如无法预判缓解策略则降低测量优先级），两者共同构成更完整的可操作性分析。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统调研 LLM 表征性伤害测量工具的从业者需求，跨学科视角新颖
- **实验充分度**: ⭐⭐⭐⭐ — 12 人访谈达饱和，主题分析方法规范，但缺少定量验证
- **写作质量**: ⭐⭐⭐⭐⭐ — 结构清晰、引用丰富、positionality 和伦理声明完善，质性研究写作范本
- **价值**: ⭐⭐⭐⭐ — 对 NLP 社区设计更实用的评估工具有直接指导意义，建议具体可操作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] Transforming Podcast Preview Generation: From Expert Models to LLM-Based Systems](transforming_podcast_preview_generation_from_expert_models_to_llm-based_systems.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Structural Reasoning Improves Molecular Understanding of LLM](structural_reasoning_improves_molecular_understanding_of_llm.md)
- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)

</div>

<!-- RELATED:END -->
