---
title: >-
  [论文解读] IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery
description: >-
  [ACL 2025][科学假设生成] 提出 IRIS，一个开源的交互式研究构思系统，通过蒙特卡洛树搜索（MCTS）扩展测试时计算、细粒度反馈机制和基于查询的文献综合，实现人机协作的科学假设生成。
tags:
  - ACL 2025
  - 科学假设生成
  - Human-in-the-Loop
  - MCTS
  - 研究构思
  - LLM辅助发现
---

# IRIS: Interactive Research Ideation System for Accelerating Scientific Discovery

**会议**: ACL 2025  
**arXiv**: [2504.16728](https://arxiv.org/abs/2504.16728)  
**代码**: 有（开源平台）  
**领域**: NLP / 科学发现与假设生成  
**关键词**: 科学假设生成, Human-in-the-Loop, MCTS, 研究构思, LLM辅助发现

## 一句话总结

提出 IRIS，一个开源的交互式研究构思系统，通过蒙特卡洛树搜索（MCTS）扩展测试时计算、细粒度反馈机制和基于查询的文献综合，实现人机协作的科学假设生成。

## 研究背景与动机

LLM 在科学发现自动化方面展现了巨大潜力，特别是在假设生成这一研究首要阶段。然而，现有方法存在以下关键问题：

**缺乏人类介入**：多数方法（如 AI-Researcher、ResearchAgent）依赖多智能体框架或扩展测试时计算，但本质上是全自动的，未有效整合人类监督

**对齐问题**：投入大量计算生成"客观新颖"的想法，却可能不符合用户的研究目标

**"奖励黑客"行为**：LLM 为了获得高评分而使用花哨术语（如 "Prompt Learning and Optimization Nexus"）或无动机地提议使用"图"结构；递归反馈循环迫使 LLM "更新颖"实际上只是在博弈评判指标

**AI 安全隐忧**：模型可能伪造科学信息，甚至出现欺骗性行为

**现有限制具体包括**：
   - 单次生成假设，忽略迭代本质
   - 粗粒度反馈（整体打分而非针对具体部分）
   - 简单的检索增强（仅附加关键词或摘要）
   - 非结构化的想法空间搜索

## 方法详解

### 整体框架

IRIS 以研究目标 $\mathcal{G}$（问题+动机）为输入，输出研究简报 $\mathcal{B}$（标题+方法+实验计划）。系统支持半自动（人指导）和全自动（MCTS 自主探索）两种模式。

### 关键设计

#### 1. 三智能体架构

**构思智能体（Ideation Agent）**：
- 生成和迭代改进研究简报
- 可在半自动模式（接收研究人员引导）和全自动模式（MCTS 驱动）之间切换

**审稿智能体（Review Agent）**：
- 负责两项任务：提供奖励分数和反馈意见
- 定义了层次化评估分类体系（基于真实科学评审标准）
- **关键创新——细粒度反馈**：不是评价整个简报，而是针对简报的*特定部分*给出*特定方面*的可操作反馈
- 研究人员验证反馈，删除不相关的部分，从而缓解"奖励黑客"

**检索智能体（Retrieval Agent）**：
- 生成针对研究目标的查询
- 采用 Ai2 Scholar QA API（Semantic Scholar，2亿+论文）
- 两阶段检索 + 三阶段生成：段落检索→重排序→引用提取→章节规划→生成引用报告
- 支持研究者上传 PDF 补充遗漏文献

#### 2. MCTS 用于假设生成

- **功能**：系统性探索广阔的研究想法空间
- **状态定义**：$s = \{$ 研究简报 $b$, 奖励 $r$, 最新反馈 $f$, 检索知识 $k\}$
- **动作空间** $\mathcal{A} = \{$ 生成, 基于检索精炼, 基于审稿精炼, 基于用户反馈精炼 $\}$
- **UCT 选择策略**：
$$\text{UCT}(n) = \frac{Q(n)}{N(n)} + c\sqrt{\frac{\ln N(n_p)}{N(n)}}$$
  其中 $c$ 为探索常数（预算紧张时降低 $c$ 以偏向利用）
- **四阶段迭代**：选择→评估→扩展→反向传播
- **设计动机**：与数学/代码（客观奖励）不同，科学构思质量是主观的，使用 Review Agent 的评分作为代理奖励
- **记忆机制**：各智能体维护轨迹级记忆，避免冗余生成

#### 3. 人机协作设计原则

- 借鉴 Amershi et al. (2019) 和 Shneiderman (2020) 的设计准则
- 最小化不透明性：MCTS 树界面提供可视化控制
- 粒度反馈替代笼统评分
- 在规划、生成、回顾各阶段保持人类监督

### 实验设置

- **LLM 后端**：Gemini-2.0-Flash (通过 LiteLLM)
- **评估指标**：
    - 绝对评分：每个假设 1-10 分
    - 相对评分：头对头比较计算 ELO rating
- **用户研究**：8 名研究者（AI/NLP、化学、物理、HCI），10 次案例研究，每次约 60 分钟

## 实验关键数据

### 自动化评估（图3）

| 指标 | Depth 0 → Depth 3 | 改善 |
|------|------------------|------|
| 绝对评分 | ~6.5 → ~7.0 | +0.5 分 |
| ELO rating | ~990 → ~1002 | +12 分 |

用户交互一致性地改善了假设质量，随交互深度增加而提升。

### 用户研究评分（表1）

| 特性/方面 | 平均评分 (1-5 Likert) |
|----------|---------------------|
| 细粒度反馈有用性 | 4.3 ± 0.7 |
| MCTS 树界面（可操控性） | 4.2 ± 0.6 |
| 文献综合质量 | 3.7 ± 0.8 |
| 可用性和控制感 | **4.5 ± 0.7** |
| 整体满意度 | 3.9 ± 0.7 |

### 定性发现

| 维度 | 比例 | 详情 |
|------|------|------|
| 可操控性 | 100%（8/8） | 所有用户重视 MCTS 树的控制力和透明性 |
| 反馈共鸣 | 87.5%（7/8） | 审稿反馈常与用户自身担忧一致 |
| 新颖洞见 | 50%（5/10） | 反馈中时有引发新思路 |
| 相关性 | 62.5%（5/8） | 假设与用户正在进行的工作有关联 |

### 关键发现

1. **交互改善质量**：用户参与的假设比纯自动生成的质量更高
2. **ELO 比绝对评分更可靠**：ELO 与人类偏好的 Pearson 相关 r=0.60，绝对评分仅 r=0.45
3. **文献检索质量因领域而异**：AI/NLP 领域较好（3.7/5），化学/物理较差，受限于 Semantic Scholar 语料覆盖
4. **可用性最受好评**（4.5/5）——说明人机协作设计确实优于全自动方案的用户体验
5. **25% 用户认为假设"显著更好"**，50% "略有改善"，100% 认为增进了对方法的理解

## 亮点与洞察

- **将 MCTS 应用于科学构思**是创新点——利用搜索树结构平衡探索与利用，比线性精炼更系统化
- **细粒度+人类验证的审稿反馈**有效解决了"奖励黑客"问题，这是全自动系统的痛点
- **开源实现**降低了学术社区的使用门槛
- **对齐问题的关注**很有前瞻性——指出 LLM 在科学构思中的"巧妙抄袭"和花哨包装问题

## 局限与展望

- 依赖研究人员作为评判者，假设其具备足够领域专业知识
- 受预算限制未使用更强的 LLM（如 Claude 3.7、o1、Gemini-2.5-Pro）
- 用户研究规模较小（N=8），结论的统计显著性有限
- 文献检索依赖 Semantic Scholar，对化学/物理等领域覆盖不足
- 未验证生成假设的实际可行性（即是否能产出有效实验）
- MCTS 计算密集，需要预算控制

## 相关工作与启发

- AI-Researcher (Si et al., 2024)：全自动但被发现有"巧妙抄袭"问题
- ResearchAgent (Baek et al., 2025)：粗粒度反馈，递归精炼导致奖励博弈
- Acceleron (Nigam et al., 2024)：早期 HITL 尝试但灵活性不足
- OpenScholar (Asai et al., 2024)：文献综合的先进系统
- 启发：未来可建立真正的"双向苏格拉底式"对话——AI 质疑研究者的选择，研究者验证 AI 的建议

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MCTS+HITL+细粒度反馈的组合在科学构思中是新思路
- **实验充分度**: ⭐⭐⭐ — 用户研究规模小（N=8），自动评估提升有限（+0.5/-12），缺少与强baseline的对比
- **写作质量**: ⭐⭐⭐⭐ — 问题动机阐述充分，系统描述详细，安全性讨论有深度
- **价值**: ⭐⭐⭐⭐ — 开源平台对学术社区有实际价值，人机协作设计理念有示范意义

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](completing_a_systematic_review_in_hours.md)
- [\[ACL 2025\] Research Borderlands: Analysing Writing Across Research Cultures](research_borderlands_analysing_writing_across_research_cultures.md)
- [\[NeurIPS 2025\] AutoSciDACT: Automated Scientific Discovery through Contrastive Embedding and Hypothesis Testing](../../NeurIPS2025/others/autoscidact_automated_scientific_discovery_through_contrastive_embedding_and_hyp.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)

<!-- RELATED:END -->
