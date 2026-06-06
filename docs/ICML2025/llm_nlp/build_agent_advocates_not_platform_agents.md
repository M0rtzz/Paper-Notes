---
title: >-
  [论文解读] Build Agent Advocates, Not Platform Agents
description: >-
  [ICML 2025][LLM/NLP][LLM Agent] Position paper，指出LMA（语言模型代理）若被平台公司控制将成为加剧监控、锁定和注意力操控的"platform agents"，提出应发展用户控制的"agent advocates"来保护个人自主权…
tags:
  - "ICML 2025"
  - "LLM/NLP"
  - "LLM Agent"
  - "平台经济"
  - "用户自主权"
  - "数字权利"
  - "Agent Advocates"
---

# Build Agent Advocates, Not Platform Agents

**会议**: ICML 2025  
**arXiv**: [2505.04345](https://arxiv.org/abs/2505.04345)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: LLM Agent, 平台经济, 用户自主权, 数字权利, Agent Advocates

## 一句话总结
Position paper，指出LMA（语言模型代理）若被平台公司控制将成为加剧监控、锁定和注意力操控的"platform agents"，提出应发展用户控制的"agent advocates"来保护个人自主权，并给出三大干预措施：开放模型/算力、互操作性标准、市场监管。

## 研究背景与动机

**领域现状**：LLM Agent（LMA）正在成为AI公司的核心产品方向。Anthropic、Google DeepMind、OpenAI等已发布了可使用浏览器、执行编程任务、进行深度研究的agent产品。Meta、Google、Amazon、Microsoft等平台公司正积极布局LMA开发。

**现有痛点**：当前数字经济由平台公司主导，这些公司作为中介方存在四大问题：
   - **监控伤害**：用户被全面追踪，数据被用于广告和其他目的
   - **伪市场设计**：平台操纵双边交易以提取价值，而非促进公平竞争
   - **注意力控制**：Google占搜索市场近90%，Meta旗下三个平台占社交媒体前四中的三席
   - **治理权力**：平台单方面制定和执行规则，缺乏正当性

**核心矛盾**：LMA的默认发展路径将被平台经济的既有结构所塑造，导致"platform agents"——被平台公司控制、服务于平台利益而非用户利益的agent。这将使监控、锁定、操控等问题全面加剧。Platform agents会比现有推荐系统更深入了解用户，使个性化操控成为可能。

**本文目标**：如何打破platform agents的默认路径？如何推动开发真正服务于用户的agent？需要哪些技术和制度干预？

**切入角度**：从政治经济学和中介理论出发，将中介分为三个维度——代表人(representative) vs 中间人(go-between)、利益导向（自利 vs 他利）、是否构建性地塑造关系。现有平台是"构建性的、自利的中间人"。

**核心 idea**：应构建agent advocates——用户控制的、作为用户忠实代表的AI代理，通过开放模型、互操作标准和市场监管三大干预实现。

## 方法详解

### 整体框架

本文是position paper，没有技术方法，而是提出概念框架和政策建议。论证链路为：平台权力分析 → platform agents会如何加剧四大风险 → agent advocates如何缓解这些风险 → 实现agent advocates的三大干预措施 → 回应三类反对意见。

### 关键设计

1. **Platform Agents的四大风险**：

    - **加剧监控伤害**：Platform agents能在多种交互中积累用户的亲密知识，不再依赖统计模式，而是像一个"亲密朋友"一样了解用户，被自利商业实体部署后可造成比现在严重得多的操控
    - **市场设计操控**：用户越来越依赖agent完成日常交易，失去自主调研机会，平台可以更精准地操纵市场选择，优先推荐自营产品或付费广告商产品
    - **注意力控制升级**：Platform agents若代替用户浏览开放互联网，不仅控制信息流，还控制信息呈现方式，可进行"普遍性编辑"——在对话中微妙地引导用户
    - **治理权力强化**：通过guardrails和nudges实现细粒度行为控制，"alignment"对齐技术可被用于限制用户自由（而非保护用户安全）

2. **Agent Advocates的定义与优势**：

    - 作为用户的忠实代表（representative），而非平台的中间人
    - 可在本地硬件或加密私有云运行，用户对数据和行为有完全控制权
    - 主动防御平台监控：代替用户浏览平台获取信息，避免被追踪
    - 降低平台锁定：提供跨平台的底层互操作性（如整合不同消息服务的群聊）
    - 改变注意力-收入映射：agent可代替用户判断内容是否值得付费

3. **三大干预措施**：

    - **开放模型与算力**：需要不受平台控制的开源模型。Meta/Google的开源模型受商业许可限制（可能禁止用于削弱其平台收入的agent），中国模型受政策审查。建议自由民主国家建立公共AI基础设施
    - **互操作性与技术标准**：agent通信协议（超越自然语言的高效协议）、凭证系统（certify agent资质和规范）、交易清算机制（clearinghouses，防止agent间串通和欺诈）
    - **市场监管**：禁止平台封锁agent的GUI访问、要求提供API、保护数据可移植性，类似"网络中立"原则

### 损失函数/训练策略

不适用（position paper，无技术实验）。

## 实验关键数据

### 主实验

本文无实验。提供了假设性对比表（附录），将agent spectrum分为三个位置：

| 维度 | Platform Agent | 中间态（硬件生态） | Agent Advocate |
|------|---------------|------------------|----------------|
| 锁定 | 深度锁定，高切换成本 | 硬件生态锁定 | 完全互操作 |
| 监控 | 全面侵入式数据收集 | 有限隐私保护 | 本地数据存储 |
| 市场设计 | 操控交易、偏向自营 | 透明但有生态偏好 | 开放市场交互 |
| 注意力 | 平台利益驱动 | 关注硬件互补服务 | 用户完全自主 |

### 消融实验

通过回应三类反对意见验证论点稳健性：

| 反对意见 | 核心观点 | 作者回应 |
|---------|---------|---------|
| Platform agents不是问题 | LMA性能不够 / 市场自我纠正 / 平台agent有便利性优势 | 即使部分风险实现也值得推动advocate |
| Agent advocates不可行 | 技术障碍(loyal AI难) / 制度问题(公司可能变质) / 历史先例(去中心化常失败) | LMA比当前互联网更有利于去中心化，因为能主动削弱网络效应 |
| Agent advocates不能解决所有问题 | 恶意使用 / 多agent交互风险 / 社会影响 | Agent advocates专门针对平台风险，搭配安全治理框架是Pareto改进 |

### 关键发现

- Platform agents是默认路径的四大原因：路径依赖、用户基础、AI投资回报压力、竞争与收购能力
- Agent advocates最大的机会在于：LMA可以"跳过"平台的围墙花园，直接代表用户与对手方交互
- 仅靠法规不够（美国缺乏立法意愿），需要技术+制度双轨并行
- 关键瓶颈：开源模型受平台公司许可限制，真正独立的高性能模型仍稀缺

## 亮点与洞察

- **中介理论的三维分析框架**：将中介分为representative vs go-between、自利 vs 他利、中立 vs 构建性三个维度，精准刻画了平台的本质问题——它们是"构建性的自利中间人"。这个框架本身就具有很高的分析价值。
- **Agent微交易经济的设想**：LMA可以"看了再决定是否付费"，因为可以provably forget内容——这为信息经济提供了全新的交易模式。
- **凭证系统作为底层治理**：用户可以指定只与特定属性的agent交易（如只与同能力等级的agent交互），实现基于市场的自底向上治理。
- **"alignment用于限制用户自由"的洞察**：指出alignment技术可以被平台反向使用，不是保护用户免受AI伤害，而是强制用户遵守平台规则。

## 局限与展望

- **缺乏技术验证**：所有论点都是推理性的，没有任何原型系统或实验
- **治理悖论未充分解决**：agent advocates的开发者也有商业激励，可能最终走向platform agent方向，文中提出的凭证系统和清算机制还很初步
- **对agent能力的乐观假设**：假设agent advocates能有效代替用户浏览平台、检测追踪行为、自动化微交易决策，但这些能力尚未充分验证
- **全球视角缺失**：讨论集中在美国/欧盟，未考虑全球南方的数字鸿沟
- **可改进方向**：可以尝试构建agent advocates原型系统，验证跨平台互操作、隐私保护浏览、凭证系统等核心技术可行性

## 相关工作与启发

- **vs EU AI Act**：法规方法在欧盟可行但在美国缺政治可行性，agent advocates是技术性替代方案
- **vs 平台监管（DSA）**：监管仅限制现有行为，无法改变力量对比的根本结构
- **vs cooperative AI社区**：该工作和多agent竞争/合作的文献相关，但聚焦在政治经济学而非博弈论层面
- **vs CERN for AI提案**：本文扩展了公共AI的使命定义——不仅理解AI风险，还要积极对抗平台控制

## 评分

- 新颖性: ⭐⭐⭐⭐ 将平台经济学和中介理论引入LMA讨论，视角独特且有深度
- 实验充分度: ⭐⭐ Position paper无实验，论证主要基于推理和政策分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构完整，论证严密，三类反驳充分且诚实
- 价值: ⭐⭐⭐⭐ 对AI agent领域的发展方向和政策制定有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](../../ACL2025/llm_nlp/axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](../../ACL2025/llm_nlp/membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](../../ACL2025/llm_nlp/culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](../../ACL2025/llm_nlp/simulating_diverse_students.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](../../ACL2025/llm_nlp/agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)

</div>

<!-- RELATED:END -->
