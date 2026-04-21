---
title: >-
  [论文解读] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents
description: >-
  [NeurIPS 2025][AI治理] 提出 Policy-as-Prompt 框架，通过两阶段端到端流水线——策略树生成（POLICY-TREE-GEN）和策略即提示生成（POLICY-AS-PROMPT-GEN）——将团队已有的非结构化设计文档（PRD、TDD、代码）自动转换为可运行时执行的策略护栏，使用轻量级 LLM 作为合规"法官"，在 HR 和 SOC 应用中实现 70-73% 的输入/输出分类准确率。
tags:
  - NeurIPS 2025
  - AI治理
  - 护栏策略
  - 提示注入防御
  - 最小权限原则
  - 策略即代码
---

# Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents

**会议**: NeurIPS 2025  
**arXiv**: [2509.23994](https://arxiv.org/abs/2509.23994)  
**代码**: 无  
**领域**: AI Safety / AI Governance  
**关键词**: AI治理, 护栏策略, 提示注入防御, 最小权限原则, 策略即代码

## 一句话总结

提出 Policy-as-Prompt 框架，通过两阶段端到端流水线——策略树生成（POLICY-TREE-GEN）和策略即提示生成（POLICY-AS-PROMPT-GEN）——将团队已有的非结构化设计文档（PRD、TDD、代码）自动转换为可运行时执行的策略护栏，使用轻量级 LLM 作为合规"法官"，在 HR 和 SOC 应用中实现 70-73% 的输入/输出分类准确率。

## 研究背景与动机

**领域现状**：自主 AI Agent 正快速进入受监管和安全关键的企业场景（HR 助手、安全运维等），EU AI Act 等法规要求 AI 系统具备可审计性、可追溯性和透明度。

**现有痛点**：组织面临严重的"策略到实践"鸿沟（policy-to-practice gap）——人类可以轻松写出设计文档中的安全规则，但将这些自然语言规则转化为机器可靠执行的约束极其困难。静态安全原则要么过于严格导致大量误拒，要么过于模糊无法捕捉上下文变化。同时 AI Agent 面临多样化安全威胁：HR Agent 可能泄露薪资信息、聊天机器人可能被 prompt injection 诱导执行恶意命令。

**核心矛盾**：安全需要 just-in-time 和 context-aware，但手工编写和维护每个场景的规则成本极高且容易遗漏。现有方法要么依赖通用安全原则（TrustAgent 等），要么需要大量人工规则编写。

**本文目标** 如何自动从团队已有的设计文档中提取安全约束并编译为可实时执行的护栏，实现 secure-by-design 的 AI Agent 部署？

**切入角度**：将策略视为可执行的提示（policy-as-code for agents），借鉴软件安全中最小权限原则的思想，让 Agent 只能做设计文档明确允许的事情。

**核心 idea**：用 LLM 读设计文档提取安全规则构建策略树，再将策略树编译为少样本提示驱动的轻量级分类器，实现自动化策略执行。

## 方法详解

### 整体框架

输入为团队的 PRD、TDD 和代码 → POLICY-TREE-GEN 提取并分类安全规则 → 生成可人工审核的策略树 → POLICY-AS-PROMPT-GEN 将策略树编译为输入分类器和输出审计器 → 人在环审核后部署 → 运行时实时拦截违规请求/响应。

### 关键设计

1. **策略树生成（POLICY-TREE-GEN）**：

    - 功能：从非结构化设计文档中自动提取和分类安全规则
    - 核心思路：两步验证方法。第一步"解析与分类"——AI 系统分析文档，识别定义安全规则的语句，分为四类：ID-I（域内输入，合法请求）、OOD-I（域外输入，离题/恶意请求）、ID-O（域内输出，合法内容）、OOD-O（域外输出，数据泄露/有害内容）。每条规则由另一个 AI Agent 二次验证。第二步"示例增强"——将规则与文档中的相关示例关联，产出保留上下文溯源的结构化策略树
    - 设计动机：最小权限原则内嵌于设计——Agent 只能在设计文档明确允许的范围内操作。例如 HR Agent 由此自动限制于 HR 数据和输出，无法隐式访问财务或 IT 信息

2. **策略即提示生成（POLICY-AS-PROMPT-GEN）**：

    - 功能：将策略树编译为可部署的 prompt-based 分类器
    - 核心思路：策略树转为人类可读 Markdown → 每条规则格式化为带"正面"（合规）和"负面"（违规）标签的示例 → 合成少样本提示块 → 嵌入主模板引导 LLM 作为合规分析师。生成两个模板：输入分类器（拦截恶意/越权输入）和输出审计器（审查数据泄露/有害输出）。返回 JSON 格式结果：二分类（ID/OOD）+ 简洁理由
    - 设计动机：严格输出格式支持确定性系统动作——ID→ALLOW、OOD→BLOCK 或 ALERT，实现"默认拒绝"的安全姿态

3. **人在环审核流程**：

    - 功能：确保生成的策略在部署前经过人工验证
    - 核心思路：安全工程师可以（i）批准部署，（ii）拒绝并要求上游更新重新生成，（iii）仅修改提示文本而不改变策略树。完整的溯源链确保每条运行时决策都可追溯到原始设计文档
    - 设计动机：纯自动化策略生成可能遗漏上下文或引入误规则，人工审核是可信部署的必要环节

## 实验关键数据

### 主实验

**POLICY-TREE-GEN 模型对比**（策略提取质量）：

| 应用 | 模型 | 检测召回率 | 检测F1 | 分类Macro-F1 |
|:---|:---|:---:|:---:|:---:|
| HR | O1 | 53.3% | 60.0% | 24.5% |
| HR | GPT-OSS 120B | 17.8% | 25.0% | 4.8% |
| HR | Llama 405B | 8.9% | 14.5% | 4.5% |
| SOC | O1 | 19.4% | 22.6% | 13.0% |

**POLICY-AS-PROMPT-GEN 准确率**（策略执行质量）：

| 应用 | 模型 | 输入分类准确率 | 输出分类准确率 |
|:---|:---|:---:|:---:|
| HR | GPT-4o | 73% | 71% |
| HR | Qwen3-1.7B | 66% | 59% |
| HR | Gemma-1B | 40% | 32% |
| SOC | GPT-4o | 70% | 68% |
| SOC | Qwen3-1.7B | 66% | 61% |

### 消融分析

| 分析维度 | 发现 |
|---------|------|
| 策略提取 | O1 全面领先，Span质量高但分类差的情况常见（Llama提取准确但分类错误） |
| 策略执行 | GPT-4o 最优，SLM（Qwen3-1.7B）在低延迟场景下性价比高 |
| 领域差异 | HR 一致优于 SOC，可能因 HR 规则更结构化 |

### 关键发现

- **策略提取是瓶颈**：即使是最好的 O1 模型，检测召回率也仅 53%，意味着约一半的安全规则被遗漏，需要人工补充
- **提取 vs 分类的脱节**：Llama 405B 和 Claude 3.5 的 span 质量指标很高（提取文本准确），但分类 F1 极低（分类错误），表明"找到规则"和"正确理解规则"是两个独立挑战
- **小模型策略执行可行**：Qwen3-1.7B（1.7B参数）在策略执行上达到 66% 准确率，证明轻量级 SLM 可以胜任实时护栏场景
- **"默认拒绝"的实际价值**：73% 准确率虽不完美，但作为第一道防线能拦截大多数恶意/违规请求，模糊案例升级到人工审核

## 亮点与洞察

- **端到端自动化理念新颖**：从设计文档到运行时护栏的全流程自动化，是"DevSecOps for AI Agents"的具体实现。这种将安全左移到设计阶段的思路值得推广
- **策略溯源链的设计很巧妙**：每条运行时决策都可追溯到设计文档的具体语句，满足 EU AI Act 等法规的审计要求
- **SLM 做策略执行的实用性**：证明无需昂贵的大模型，1.7B 小模型 + 好的提示就能做有效的安全分类

## 局限与展望

- **策略提取准确率偏低**：最好模型也只有 53% 召回率，实际部署中需要大量人工补充和修正
- **评估规模有限**：仅两个领域（HR和SOC）、每个领域 100 条测试用例，无法证明跨领域泛化性
- **内部数据不可复现**：设计文档来自内部企业项目，无法公开，阻碍第三方验证
- **策略漂移未解决**：当 PRD/TDD 更新时需要手动触发重新生成，尚未实现自适应策略更新
- **对抗性鲁棒性不明确**：虽然测试了基本的 prompt injection，但未在高级对抗攻击下评估

## 相关工作与启发

- **vs TrustAgent**：TrustAgent 使用静态安全原则，Policy-as-Prompt 从具体设计文档动态生成规则，更上下文感知
- **vs CAPTURE**：同一作者团队的 prompt injection 检测工作，Policy-as-Prompt 在此基础上扩展到完整的策略执行框架
- **vs NeMo Guardrails**：NVIDIA 的护栏框架需要手工编写对话流规则，Policy-as-Prompt 自动从文档提取

## 评分

- 新颖性: ⭐⭐⭐⭐ 端到端自动化策略提取+执行的理念新颖，策略树→提示分类器的编译路径有创意
- 实验充分度: ⭐⭐⭐ 评估规模小、领域覆盖少、数据不可公开，限制了结论的可信度
- 写作质量: ⭐⭐⭐⭐ 流程描述清晰，实际案例（HR/SOC）帮助理解
- 价值: ⭐⭐⭐⭐ 对企业 AI Agent 安全部署有直接参考价值，但需提升策略提取准确率才能真正实用

<!-- RELATED:START -->

## 相关论文

- [Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All?](position_paper_if_innovation_in_ai_systematically_violates_fundamental_rights_is.md)
- [Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](../../AAAI2026/social_computing/factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)
- [OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)
- [Propaganda AI: An Analysis of Semantic Divergence in Large Language Models](../../ICLR2026/social_computing/propaganda_ai_an_analysis_of_semantic_divergence_in_large_language_models.md)
- [DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)

<!-- RELATED:END -->
