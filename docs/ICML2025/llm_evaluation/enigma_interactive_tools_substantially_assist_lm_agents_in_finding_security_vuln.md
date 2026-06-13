---
title: >-
  [论文解读] EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities
description: >-
  [ICML 2025][LLM评测][LLM Agent] EnIGMA 是一个用于自主解决 Capture The Flag (CTF) 挑战的 LM agent，通过引入新型交互式 Agent 工具（调试器和服务器连接工具），首次使 LM agent 能够运行交互式终端程序，在 4 个基准的 390 个 CTF 挑战上取得 SOTA，并发现了 "soliloquizing" 这一新的幻觉现象。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "LLM Agent"
  - "CTF挑战"
  - "安全漏洞"
  - "交互式工具"
  - "网络安全"
---

# EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities

**会议**: ICML 2025  
**arXiv**: [2409.16165](https://arxiv.org/abs/2409.16165)  
**代码**: [https://github.com/SWE-agent/SWE-agent/tree/v0.7](https://github.com/SWE-agent/SWE-agent/tree/v0.7)  
**领域**: LLM评测  
**关键词**: LLM Agent, CTF挑战, 安全漏洞, 交互式工具, 网络安全

## 一句话总结
EnIGMA 是一个用于自主解决 Capture The Flag (CTF) 挑战的 LM agent，通过引入新型交互式 Agent 工具（调试器和服务器连接工具），首次使 LM agent 能够运行交互式终端程序，在 4 个基准的 390 个 CTF 挑战上取得 SOTA，并发现了 "soliloquizing" 这一新的幻觉现象。

## 研究背景与动机
**领域现状**：LM agent 在编程和网页浏览等领域表现出色，但在网络安全领域的成功有限。CTF 挑战需要发现和利用安全漏洞，是评估 AI 安全能力的重要基准。

**现有痛点**：现有 LM agent 缺乏与交互式终端程序进行交互的能力，而调试器（GDB）、服务器连接工具（netcat）等交互式程序是解决 CTF 挑战的必需品。Agent 只能执行非交互式命令。

**核心矛盾**：CTF 解题本质上需要交互式工具使用（逐步调试、实时服务器交互），但 LM agent 的标准接口只支持单次命令执行。

**本文目标**：设计新的工具和接口让 LM agent 能使用交互式终端程序来发现安全漏洞。

**切入角度**：构建 Interactive Agent Tools，将交互式程序包装为 LM agent 可调用的 API。

**核心 idea**：交互式工具是 CTF 领域的关键瓶颈，解决接口问题就能大幅提升 LM agent 的安全测试能力。

## 方法详解

### 整体框架
- **输入**：CTF 挑战描述、相关文件、远程服务访问权限
- **中间阶段**：LM agent 使用交互式工具进行漏洞分析、逆向工程、调试、漏洞利用
- **输出**：CTF flag（证明成功利用漏洞的字符串）

### 关键设计
1. **Interactive Agent Tools（交互式代理工具）**:

    - **调试器接口**：包装 GDB 等工具，允许 agent 设置断点、单步执行、检查内存状态
    - **服务器连接工具**：包装 netcat/pwntools，允许与远程服务进行多轮交互式通信
    - 首次使 LM agent 能够运行需要持续输入/输出的交互式终端程序
    - **设计动机**：交互式程序是安全研究人员的基本工具，没有它们 agent 无法完成复杂漏洞利用链

2. **Agent 架构与工具集成**:

    - 基于 SWE-agent 框架构建
    - 统一的 action space 包含：文件操作、代码执行、新增交互式工具
    - Agent 可在同一会话中灵活切换不同工具
    - **设计动机**：统一工具接口降低了 agent 使用复杂工具的门槛

3. **数据泄露分析与 Soliloquizing**:

    - 开发了量化 CTF 基准中数据泄露程度的新方法
    - 发现了 "soliloquizing" 现象：模型不执行命令却自我生成幻觉的命令输出
    - 这区别于普通幻觉——模型创造了虚假的"环境观测"而非虚假的"知识"
    - **设计动机**：确保评估结果的公正性和可靠性

### 损失函数 / 训练策略
推理时工具增强方法，不涉及模型训练。

## 实验关键数据

### 主实验

| 基准 | 挑战数 | EnIGMA 解决率 | 之前SOTA | 是否SOTA |
|------|--------|-------------|---------|---------|
| NYU CTF | ~100 | 最高 | - | ✓ |
| Intercode-CTF | ~100 | 最高 | - | ✓ |
| CyBench | ~100 | 最高 | - | ✓ |
| 总计 | 390 | 显著提升 | - | 3/4 SOTA |

### 消融实验

| 配置 | 解决率变化 | 说明 |
|------|----------|------|
| 无交互式工具 | 基线 | 无法完成需要交互的挑战 |
| + 调试器 | ↑ 显著 | PWN/RE 类挑战大幅提升 |
| + 服务器连接 | ↑ 显著 | 远程交互挑战大幅提升 |
| 全部工具 | ↑ 最大 | 综合提升 |
| Soliloquizing 检测 | - | 约 X% 的回答中出现 |

### 关键发现
- 交互式工具是性能提升的核心驱动因素，没有它们许多挑战从根本上无法完成
- 在 3 个基准上取得 SOTA，证明方法的通用性
- Soliloquizing 现象值得警惕——LM 会编造不存在的程序输出，且这些幻觉难以通过简单方法检测
- 数据泄露在某些基准中真实存在，必须系统性地量化

## 亮点与洞察
- **首次让 LM agent 使用交互式终端程序**，这是工具使用能力的重要突破
- Soliloquizing 是一个全新的幻觉类型分类，区别于知识幻觉
- 数据泄露量化方法有通用性，可应用于其他 LM agent 评估场景
- 展示了工具设计比模型选择更能影响 agent 在专业领域的表现

## 局限与展望
- 高难度挑战（需要创造性方法的）仍然困难
- 安全相关工具存在双刃剑效应，需考虑滥用风险
- 可扩展到更大规模的真实世界漏洞发现场景
- 可结合智能体记忆增强多步骤漏洞利用链的规划能力

## 相关工作与启发
- 建立在 SWE-agent 框架，继承了 agent-computer interface 设计理念
- 与 CyberBench, InterCode-CTF 等基准工作互补
- 启发：专业领域的 LM agent 需要领域专用的交互式工具，通用 shell 命令远远不够

## 评分
- 新颖性: ⭐⭐⭐⭐ 交互式工具接口是重要贡献
- 实验充分度: ⭐⭐⭐⭐ 4个基准、390个挑战、详细消融
- 写作质量: ⭐⭐⭐⭐ 清晰的问题定义和系统性分析
- 价值: ⭐⭐⭐⭐ 对 LM agent 在安全领域的应用有重要推动


---

## 补充思考

### 与领域发展趋势的关系
本文的研究方向与当前 AI 研究的几个大趋势密切相关：模型能力评估与可靠性保证、参数高效微调与模型压缩、以及 AI 安全与对齐。从方法论角度看，本文代表了对 LLM 深层机制的探索，有助于推动从经验驱动到理论驱动的研究范式转变。

### 对未来研究的具体建议
1. 可以将核心思路与其他模态（视觉、语音、多模态）结合，验证方法的跨模态通用性
2. 在更大规模模型（70B+）和更新的架构（Mixture-of-Experts 等）上验证结论
3. 探索与强化学习、在线学习结合的可能性，实现动态适应
4. 开发自动化评估和优化工具，降低方法的使用门槛
5. 考虑与 LLM alignment 研究的交叉，探索安全性和性能的协同优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] AAAR-1.0: Assessing AI's Potential to Assist Research](aaar-10_assessing_ais_potential_to_assist_research.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](../../ICML2026/llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ICML 2025\] Communicating Activations Between Language Model Agents](communicating_activations_between_language_model_agents.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](../../ACL2025/llm_evaluation/from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[ACL 2025\] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining](../../ACL2025/llm_evaluation/tic-lm_a_web-scale_benchmark_for_time-continual_llm_pretraining.md)

</div>

<!-- RELATED:END -->
