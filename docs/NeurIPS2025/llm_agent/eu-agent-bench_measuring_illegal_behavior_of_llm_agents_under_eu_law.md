---
title: >-
  [论文解读] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law
description: >-
  [NeurIPS 2025][LLM Agent][LLM agent safety] 提出 EU-Agent-Bench，首个基于欧盟法律框架的可验证智能体基准，通过 600 个良性用户请求测试 LLM 智能体的工具调用是否违反欧盟法规…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "LLM agent safety"
  - "EU law compliance"
  - "benchmark"
  - "function calling"
  - "legal AI"
---

# EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law

**会议**: NeurIPS 2025  
**arXiv**: [2510.21524](https://arxiv.org/abs/2510.21524)  
**代码**: [待确认]  
**领域**: LLM Agent  
**关键词**: LLM agent safety, EU law compliance, benchmark, function calling, legal AI

## 一句话总结

提出 EU-Agent-Bench，首个基于欧盟法律框架的可验证智能体基准，通过 600 个良性用户请求测试 LLM 智能体的工具调用是否违反欧盟法规，发现即使最佳模型（Gemini 2.5 Flash）的合法率也仅约 55%，揭示了当前对齐技术与法律可靠性之间的巨大鸿沟。

## 研究背景与动机

LLM 正从聊天助手走向**智能体（Agent）**部署，通过工具调用与环境交互。智能体系统引入了新的安全挑战：

**现有基准的不足**：
   - 大多数智能体安全基准是**领域无关**（jurisdiction-agnostic）的，不锚定具体法律体系
   - 许多基准使用**对抗性/恶意**用户输入，无法测量智能体面对良性请求时的内在违法倾向
   - 部分基准依赖 LLM 裁判（非可验证），评估不客观

**欧盟法律的特殊性**：
   - GDPR、AI Act 等法规对数据处理、消费者保护等有严格要求
   - LLM 在法律领域的知识基准表现并不完美
   - 智能体行为的合法性尚未被系统研究

**本文要填补的空白**：
   - 智能体功能调用基准 + 欧盟法律锚定 + 可验证评估标准 + 良性用户请求

## 方法详解

### 整体框架

EU-Agent-Bench 围绕 6 个场景构建，每个场景模拟一个位于欧盟的组织部署 LLM 智能体的情境。评估智能体在面对良性用户请求时，其**工具调用参数**是否违反欧盟法规。

### 关键设计

#### 1. 六大法律类别

| 类别 | 对应法规 |
|------|---------|
| 数据保护 | GDPR (Regulation 2016/679) |
| 科学不端 | EU Clinical Trials Regulation (536/2014) |
| 版权 | DSM Directive (2019/790), InfoSoc Directive (2001/29) |
| 竞争 | TFEU (Articles 101-102) |
| 偏见与歧视 | Employment Equality Directive (2000/78), Racial Equality Directive (2000/43) |
| 消费者保护 | Unfair Commercial Practices Directive (2005/29) |

#### 2. 基准构成

- **60 个人工策展的高质量用户请求**（每类 10 个）
- 通过数据增强扩展到 **600 个**（每类 100 个）
- 每个请求都是**良性的**（非恶意），但执行时可能产生合规或违规行为
- 系统提示包含基于真实行业实践的行为指导

#### 3. 可验证评估标准

核心创新：**不使用 LLM 裁判**，而是将工具调用参数与基于欧盟法规引文的评分标准（rubric）进行比对。

评估流程：
- 观察智能体第一轮回复中的工具调用
- 将函数参数值与预定义标准对比
- 安全分数为二值：0（含任何违规参数）或 1（全部合规）
- 每个请求重复 10 次，取平均

如果模型未调用必要工具，该次试验被排除。

#### 4. 与现有基准的差异化

本文系统梳理了 12 个相关基准，EU-Agent-Bench 是唯一同时满足以下四个条件的：
- ✅ 良性用户输入
- ✅ 自动可验证评估
- ✅ 锚定特定法律管辖区
- 单轮交互（多轮为未来工作）

### 损失函数 / 训练策略

本文是评估基准论文，不涉及训练。评估使用 OpenRouter API，temperature=0.7，7 个前沿模型。

## 实验关键数据

### 主实验

**模型合法率排名**（600 样本，10 次重复）：

| 模型 | 平均合法率(%) | 标准95%CI | 聚类95%CI |
|------|-------------|-----------|-----------|
| Gemini 2.5 Flash | **55.3** | [46.1, 64.5] | [46.1, 64.5] |
| Qwen3 8B | 52.7 | [49.5, 55.9] | [44.5, 60.8] |
| GPT-4.1 | 49.5 | [45.7, 53.2] | [40.2, 58.8] |
| Kimi K2 | 45.4 | [42.8, 48.1] | [37.4, 53.4] |
| Qwen3 32B | 45.1 | [42.1, 48.2] | [36.2, 54.1] |
| DeepSeek Chat v3 | 40.6 | [37.3, 44.0] | [32.3, 49.0] |
| Qwen3 14B | **38.1** | [34.6, 41.7] | [29.0, 47.3] |

三个核心观察：
1. 最佳与最差模型差距 **27.4%**，说明安全对齐技术效果差异巨大
2. 即使最佳模型也只有 55.3% 合法率，约 9/20 的请求会导致违法工具调用
3. **模型大小与合法率无关**：Qwen3 8B > Qwen3 32B > Qwen3 14B，不遵循 scaling law

### 消融实验

**注入欧盟法规文本到系统提示**（Gemini 2.5 Flash）：

在系统提示中直接提供相关欧盟法规条文的效果：
- 合法率变化极其有限，与基线接近
- 说明仅"告诉"模型法规内容不足以保证合规行为
- 需要更深层的对齐方法

### 关键发现

1. **ALL 模型都不及格**：最佳 55.3%，远不满足安全关键部署要求
2. **规模无关性**：合法率不随模型参数增长，这挑战了"更大模型更安全"的假设
3. **知识 ≠ 行动**：即使在系统提示中注入相关法规全文，模型行为改善甚微
4. **数据增强的局限**：增强后最差类别仅约 30% 的试验成功调用了必要工具，暴露了 LLM 的 prompt 敏感性
5. **不同模型在不同法律类别上表现差异大**：没有模型在所有类别上一致优秀

## 亮点与洞察

1. **首创性定位**：将智能体安全从泛化的"有害行为"锚定到具体法律管辖区（欧盟），使评估结果具有法律实践意义
2. **可验证性优先**：放弃 LLM 裁判，采用基于法规引文的确定性标准，消除评估模糊性
3. **良性输入测试内在倾向**：不测试模型能否抵御攻击，而是测量正常条件下的基础违法率，更贴近实际部署
4. **公开+保留分割策略**：公开预览集供研究，保留私有测试集防止数据污染，有利于长期基准维护
5. **"合法率不随规模增长"的发现**：对 AI 安全领域的 scaling hypothesis 提出质疑

## 局限与展望

1. **仅单轮交互**：真实智能体通常涉及多步因果依赖的工具调用链，当前设计过于简化
2. **数据增强质量下降**：增强后的请求导致工具调用成功率降低，削弱了基准鲁棒性
3. **工具参数空间受限**：为保证可验证性，工具参数限制为预定义字符串和布尔值，与真实部署的开放式工具差距大
4. **仅覆盖欧盟法律 6 个类别**：法律领域远不止这些（如税法、劳动法、金融法规等）
5. **样本量有限**：600 样本（60 原始 × 10 增强）在统计上可能不够稳健，部分置信区间较宽
6. **仅限英语交互**：欧盟是多语言环境，不同语言下的合规行为可能不同

## 相关工作与启发

- **AgentHarm/SHADE-Arena**：测试恶意输入下的智能体安全，本文互补地测试良性输入
- **Legal Agent Bench / J1-Eval**：中国法律框架下的基准，与本文的欧盟定位形成跨司法管辖区对比
- **ToolEmu / AgentDojo**：通用智能体工具使用评估，但不锚定法律
- **启发**：(1) AI 安全评估需要从"泛化有害"走向"法律特化"；(2) 合规性不能仅靠 prompt engineering，需要训练层面的介入；(3) 多司法管辖区 × 多语言的综合基准是未来方向

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个结合欧盟法律、良性输入、可验证评估的智能体基准，定位独特
- 实验充分度: ⭐⭐⭐ — 7 个模型、6 个法律类别、法规注入消融。但样本量偏小、仅单轮、仅英语
- 写作质量: ⭐⭐⭐⭐ — Workshop 论文，结构紧凑，动机论证清晰
- 价值: ⭐⭐⭐⭐ — 揭示了当前 LLM 智能体在法律合规性上的严重不足，有政策和实践参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents](agentmisalignment_measuring_the_propensity_for_misaligned_behaviour_in_llm-based.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[NeurIPS 2025\] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)
- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](../../ACL2025/llm_agent/androidgen_agent_data_scarcity.md)
- [\[ACL 2025\] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks](../../ACL2025/llm_agent/agents_under_siege_breaking_pragmatic_multi-agent_llm_systems_with_optimized_pro.md)

</div>

<!-- RELATED:END -->
