---
title: >-
  [论文解读] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation
description: >-
  [ACL 2026][LLM Agent][Agent生成] 提出 HAG 框架，将群体 Agent 生成形式化为两阶段层次化决策过程——先用世界知识模型构建主题自适应人口分布树实现宏观分布对齐，再通过真实数据检索与 Agent 增强保证微观个体一致性，在多领域基准上将群体对齐误差平均降低 37.7%、社会学一致性提升 18.8%。
tags:
  - ACL 2026
  - LLM Agent
  - Agent生成
  - 人口模拟
  - 层次化决策
  - 主题自适应
  - Agent-Based Modeling
---

# HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation

**会议**: ACL 2026  
**arXiv**: [2601.05656](https://arxiv.org/abs/2601.05656)  
**代码**: https://github.com/Libra117/HAG  
**领域**: LLM Agent  
**关键词**: Agent生成, 人口模拟, 层次化决策, 主题自适应, Agent-Based Modeling

## 一句话总结
提出 HAG 框架，将群体 Agent 生成形式化为两阶段层次化决策过程——先用世界知识模型构建主题自适应人口分布树实现宏观分布对齐，再通过真实数据检索与 Agent 增强保证微观个体一致性，在多领域基准上将群体对齐误差平均降低 37.7%、社会学一致性提升 18.8%。

## 研究背景与动机

**领域现状**：Agent-Based Modeling（ABM）在计算社会科学、经济建模和个性化推荐等领域日益重要，这些模拟系统高度依赖用户 Agent 来模拟偏好与交互行为。Agent 的质量直接决定了模拟系统的保真度。

**现有痛点**：现有 Agent 生成方法分两类：(1) 基于数据检索的方法从真实用户日志中构建 Agent 池，但天然是静态的，无法适应未见或数据稀缺的主题；(2) 基于 LLM 生成的方法通过预定义 schema 或文本推理生成 Agent persona，但缺乏对多维属性联合分布的显式建模，每个 Agent 独立生成导致群体分布与现实不符。

**核心矛盾**：没有现有方法能同时实现"主题自适应的群体宏观分布建模"和"微观个体属性的社会学合理性"。独立生成的 Agent 可能出现属性矛盾（如年龄与职业不匹配），而静态检索无法覆盖新主题。

**本文目标**：设计一个同时满足宏观分布对齐和微观个体一致性的 Agent 群体生成框架。

**切入角度**：作者观察到人口统计结构是主题相关的（如讨论科技和讨论养老的用户群体分布差异巨大），因此将群体生成建模为层次化条件概率推断问题。

**核心 idea**：用世界知识模型（WKM）自顶向下构建主题自适应人口分布树，通过层次化条件概率捕获多维属性的联合分布，再用真实数据填充与 LLM 增强相结合生成最终群体。

## 方法详解

### 整体框架
HAG 框架分两个阶段：(1) 主题自适应分布树构建——输入目标主题，利用 WKM 推断人口属性的层次化条件概率，生成一棵从主题到完整 persona 的分布树；(2) 基于真实数据的实例化与增强——根据树的叶节点分布从 World Values Survey 数据库检索真实用户，对数据不足的节点用 LLM 进行约束增强。

### 关键设计

1. **主题自适应分布树构建**:

    - 功能：将抽象的主题转化为具体的多维人口属性联合分布
    - 核心思路：首先由 WKM 根据主题识别并排序相关人口维度（如年龄 > 性别 > 教育），形成树的层级顺序。然后逐层自顶向下扩展：每层的节点值和边权由 WKM 推断条件概率 $P(f^{(l)}=v^{(l)} | f^{(1:l-1)}=v^{(1:l-1)}, t)$ 得到。每个叶节点对应一个完整 persona，其目标比例为根到叶路径上所有边权的乘积。
    - 设计动机：通过层次化条件概率而非独立采样来建模属性间的依赖关系，确保宏观联合分布与主题匹配

2. **基于真实数据的实例化与 Agent 增强**:

    - 功能：将分布树转化为具体的 Agent 群体，保证微观个体的真实性
    - 核心思路：对每个叶节点 persona，计算所需人数 $n(\mathbf{v}) = \text{Round}(N \cdot W(\mathbf{v}|t))$。从 World Values Survey 数据库中检索匹配的真实用户（HIT 节点直接采样），对数据不足的 MISS 节点用 LLM 在该 persona 路径约束下进行增强生成。
    - 设计动机：优先使用真实数据保证微观一致性，LLM 增强受树路径约束避免产生不兼容的属性组合

3. **PACE 评估框架**:

    - 功能：从群体对齐和社会学一致性两个互补维度评估生成质量
    - 核心思路：群体对齐使用 JSD/KL 散度衡量分布保真度，用 Gini-Simpson 指数量化多样性误差；社会学一致性通过聚类提取主流原型评估典型性，并对每个个体进行内部自洽性和上下文合理性检查。
    - 设计动机：现有评估缺乏专门针对 Agent 群体生成的量化框架，需要同时考量统计对齐和语义合理性

### 训练策略
HAG 无需训练，直接利用预训练 LLM 作为 WKM 进行推理，再从现有数据库检索和增强。

## 实验关键数据

### 主实验
在 Bluesky（社交模拟）、Amazon（产品推荐）、IMDB（电影评论）三个领域上评估：

| 方法 | Bluesky JSD↓ | Bluesky KL↓ | Bluesky ArchRel↑ | Bluesky IndCon↑ |
|------|-------------|-------------|-------------------|-----------------|
| Random Select | 0.628 | 2.489 | 3.000 | 2.599 |
| Topic-Retrieval | 0.578 | 5.725 | 3.250 | 2.928 |
| LLM Generate | 0.539 | 2.487 | 3.063 | 3.197 |
| HAG-Flat | 0.401 | 2.436 | 3.750 | 3.324 |
| **HAG (Ours)** | **0.345** | **1.657** | **3.813** | **3.617** |

### 消融实验

| 配置 | JSD↓ | KL↓ | 说明 |
|------|------|-----|------|
| HAG (Full) | 0.345 | 1.657 | 完整模型 |
| HAG-Flat | 0.401 | 2.436 | 去掉层次化树，平坦化生成 |
| LLM Generate | 0.539 | 2.487 | 无树结构直接 LLM 生成 |

### 关键发现
- HAG 在所有三个领域上群体对齐误差平均降低 37.7%，社会学一致性提升 18.8%
- 层次化树结构是关键：HAG-Flat（无层次化条件概率）相比完整 HAG 在 JSD 上劣化约 16%
- 真实数据检索+增强策略有效避免了"Frankenstein Agent"问题（属性矛盾的拼凑 Agent）

## 亮点与洞察
- 将群体 Agent 生成形式化为层次化决策过程是一个优雅的建模选择，将条件概率链式分解与树结构结合，兼顾了可解释性和生成质量
- PACE 评估框架填补了 Agent 群体生成评估的空白，从统计和语义两个维度提供了系统化的评估方案，可复用于其他群体模拟任务
- 利用 WKM 的世界知识推断主题相关的人口分布，避免了依赖领域专家手动设计的瓶颈

## 局限与展望
- 树构建依赖 WKM 的质量，对于非常罕见或新兴的主题可能推断不准确
- 仅使用 World Values Survey 作为实数据源，文化和地域覆盖有限
- 维度排序对结果有影响，但自动排序的最优性缺乏理论保证
- 未来可探索动态更新树结构以适应实时变化的社会趋势

## 相关工作与启发
- **vs LLM Generate（直接生成）**: 直接用 LLM 生成 Agent 忽略了群体联合分布，HAG 通过树结构显式建模属性依赖
- **vs Topic-Retrieval（主题检索）**: 检索方法受限于已有数据覆盖，HAG 通过 WKM 推断 + LLM 增强实现无数据主题的自适应
- **vs WorldValuesBench**: HAG 继承其属性体系但扩展了动态主题自适应能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化分布树的建模思路新颖，将群体生成与条件概率推断结合
- 实验充分度: ⭐⭐⭐⭐ 三个领域覆盖面广，PACE 评估框架设计合理
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义和方法描述逻辑连贯
- 价值: ⭐⭐⭐⭐ 对 Agent 模拟领域有实用价值，评估框架可推广
- 综合: ⭐⭐⭐⭐ 问题定义清晰，方案设计合理，实验验证充分

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/llm_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[AAAI 2026\] A2Flow: Automating Agentic Workflow Generation via Self-Adaptive Abstraction Operators](../../AAAI2026/llm_agent/a2flow_automating_agentic_workflow_generation_via_self-adaptive_abstraction_oper.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[CVPR 2025\] ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Generation](../../CVPR2025/llm_agent/ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)

<!-- RELATED:END -->
