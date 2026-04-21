---
title: >-
  [论文解读] "sudo rm -rf agentic_security" | SUDO: Screen-based Universal Detox2tox Offense
description: >-
  [ACL 2025][LLM Agent][computer-use agent] 提出SUDO攻击框架，通过Detox2tox三阶段流水线将恶意请求伪装为无害指令再恢复攻击载荷，配合基于检查清单反馈的动态迭代优化，系统性攻破Claude CUA、MANUS等计算机使用Agent的安全防护，最高达41.33%攻击成功率。
tags:
  - ACL 2025
  - LLM Agent
  - computer-use agent
  - safety attack
  - red teaming
  - jailbreak
  - agentic security
---

# "sudo rm -rf agentic_security" | SUDO: Screen-based Universal Detox2tox Offense

**会议**: ACL 2025  
**arXiv**: [2503.20279](https://arxiv.org/abs/2503.20279)  
**代码**: [github.com/AIM-Intelligence/SUDO](https://github.com/AIM-Intelligence/SUDO)  
**领域**: LLM Agent / AI安全  
**关键词**: computer-use agent, safety attack, red teaming, jailbreak, agentic security

## 一句话总结

提出SUDO攻击框架，通过Detox2tox三阶段流水线将恶意请求伪装为无害指令再恢复攻击载荷，配合基于检查清单反馈的动态迭代优化，系统性攻破Claude CUA、MANUS等计算机使用Agent的安全防护，最高达41.33%攻击成功率。

## 研究背景与动机

**领域现状**：LLM正从纯文本对话演化为computer-use agent（如Claude Computer Use、MANUS、OmniParser），能在真实桌面/网页环境中自主执行文件操作、命令行、网页浏览等任务。这些agent极大扩展了LLM的实际应用场景，但也带来了全新的安全风险面。

**现有痛点**：现有越狱（jailbreak）研究主要集中在纯文本LLM场景，针对多模态环境交互agent的安全评估严重不足。现有方法如角色扮演、程序执行框架等单轮攻击手段在computer-use agent上效果极差（ASR最高仅7.30%）。更关键的是，能操作真实系统的agent一旦被攻破，后果远比文本场景严重——可以删除系统文件、发送钓鱼邮件、窃取隐私数据。

**核心矛盾**：computer-use agent的安全防线建立在refusal training上，但这种静态防御容易被精心设计的间接指令绕过。更深层的矛盾在于——底层模型越强大，攻击反而越有效，因为更强的VLM能生成更精准的攻击执行计划，形成"能力-安全"的悖论。

**本文目标**：如何系统性地评估和攻破computer-use agent的安全防护？需要一套自动化的攻击框架和标准化的评估基准。

**切入角度**：不直接提交恶意请求，而是通过语义变换将恶意意图"洗白"后获取执行计划，再在执行时恢复攻击载荷。结合迭代反馈机制逐步突破防线。

**核心 idea**：Detox2tox——先去毒骗过安全检查获取分步计划，再回毒恢复恶意内容，配合动态反馈迭代升级攻击。

## 方法详解

### 整体框架

SUDO分为两个阶段：**静态阶段**（Static Phase）通过Detox2tox流水线一次性生成候选攻击提示，**动态阶段**（Dynamic Phase）基于agent的拒绝反馈迭代优化攻击策略。整体流程为：恶意任务 → Detoxifier去毒化 → Instruction Generator生成分步计划 → Toxifier回毒化 → 提交agent执行 → Evaluator检查清单评估 → Dynamic Updater基于反馈调整 → 重新攻击（循环直至成功或达上限）。

### 关键设计

1. **Detoxifier (去毒化器)**:
    - 功能：接收原始恶意命令/目标，移除或改写明显有害的内容，生成去毒化版本
    - 核心思路：将恶意意图隐藏在看似无害的任务描述中（如"发送钓鱼邮件" → "发送一封格式化邮件"），使agent的安全策略不会立即拒绝
    - 设计动机：直接提交恶意请求ASR为0%，必须先过安全检查这道门槛；去毒化是获取高质量执行计划的前提条件

2. **Instruction Generator (指令生成器)**:
    - 功能：将去毒化后的描述传递给SOTA VLM，生成详细的分步执行计划
    - 核心思路：利用VLM的强大能力（包括截图理解、系统信息解析）为伪装后的"无害"任务生成可操作的step-by-step计划，支持多种VLM后端（Llama 3.2 Vision、Gemini 2.0、Claude 3.5/3.7、GPT 4.5、o1）
    - 设计动机：计算机使用任务需要精确的操作序列，通用的执行计划难以奏效；利用VLM自身能力来构造绕过其防御的计划，形成"以子之矛攻子之盾"

3. **Toxifier (再毒化器)**:
    - 功能：在已生成的合法执行计划中重新注入恶意载荷，完成Detox2tox转换
    - 核心思路：将计划中的良性操作替换为攻击操作（如良性文件删除命令 → `sudo rm -rf`破坏性命令），最终输出表面无害但实际引导有害行为的攻击提示
    - 设计动机：分离"获取计划"和"注入恶意"两步，绕过端到端的安全过滤器；保留执行计划的结构完整性，仅替换关键操作内容

4. **Dynamic Updater (动态更新器)**:
    - 功能：在攻击部分失败时，基于评估反馈自动调整攻击提示并重试
    - 核心思路：使用检查清单（checklist）评估每个子目标的完成情况，根据部分失败的原因调整提示元素或强化隐藏触发器，将修改后的提示重新通过LLM生成并提交给agent
    - 设计动机：静态一次性攻击的上限有限（~20-24% ASR），迭代反馈能系统性地拆解逐层防御，每轮显著提升成功率

### 评估方法

**检查清单评估机制**：每个攻击任务被分解为多个主题要素（topical elements），每成功完成一个要素得1分，若出现越狱行为额外加1分。ASR = (matched_topics + 1) / (total_topics + 1)。这种细粒度评估不仅捕获完全成功/失败，还能记录部分成功状况，为Dynamic Updater提供可操作的反馈信号。

**SUDO Dataset基准**：人工构建50个攻击任务，覆盖4大类12子类风险场景（内容安全、社会风险、法律风险、操作风险），涵盖20种不同执行环境（网页+桌面），所有任务在真实操作系统上执行而非沙盒环境。

## 实验关键数据

### 主实验：不同Instruction Generator的攻击成功率

| 模型 | 静态ASR(%) | 动态1轮(%) | 动态2轮(%) | 动态3轮(%) |
|------|-----------|-----------|-----------|-----------|
| Claude 3.5 Haiku | 23.60 | 34.87 (+11.27) | 35.56 (+0.69) | 35.99 (+0.43) |
| Claude 3.7 Sonnet | 24.41 | 29.71 (+5.30) | 32.55 (+2.84) | 38.12 (+5.57) |
| Gemini 2.0 Flash | 24.02 | 30.09 (+6.07) | 32.19 (+2.10) | 32.95 (+0.76) |
| Llama 3.2 Vision | 19.45 | 26.45 (+7.00) | 31.19 (+4.74) | 32.69 (+1.20) |
| GPT 4.5 Preview | 21.29 | 27.99 (+6.70) | 33.82 (+5.83) | **41.33** (+7.51) |
| o1 | 24.05 | 33.79 (+9.74) | 37.29 (+3.50) | 41.09 (+3.80) |

### 与基线越狱方法对比

| 方法 | Direct | Role Play | Program Execution | Superior Model | **SUDO** |
|------|--------|-----------|-------------------|---------------|----------|
| ASR(%) | 0.00 | 3.29 | 4.67 | 7.30 | **41.33** |

SUDO相比最优基线Superior Model提升了**5.7倍**。

### 跨Agent泛化性（12个代表性任务，o1作为Instruction Generator）

| Agent | 静态ASR(%) | 动态3轮ASR(%) |
|-------|-----------|-------------|
| Claude CUA | 16.89 | 34.39 |
| MANUS | 34.86 | 63.19 |
| OmniParser V2 | 41.96 | 66.13 |

### 消融实验

| 分析维度 | 关键发现 |
|---------|---------|
| 静态 vs 动态 | 动态迭代将ASR从~20-24%提升至35-41%，提升约70-100% |
| 迭代轮次收益 | 第1轮提升最大（+5-11pp），后续轮次递减但仍有正收益 |
| 模型能力影响 | GPT 4.5和o1持续从迭代中获益更多，推理能力更强的模型攻击提升更显著 |
| 跨Agent差异 | Claude CUA防御最强（ASR最低），MANUS和OmniParser更脆弱 |

### 关键发现

- **动态迭代是核心贡献**：静态Detox2tox就能达到~20-24% ASR（已非trivial），三轮迭代后提升至41%+，证明反馈驱动的迭代攻击能系统性拆解安全防线
- **传统越狱方法在agent场景近乎失效**：Direct 0%，Role Play 3.29%，Program Execution 4.67%，说明computer-use agent需要全新的攻击范式
- **模型越强攻击越强的悖论**：GPT 4.5和o1在动态阶段的ASR增长最大，验证了"能力-安全"悖论——更强的推理能力反而帮助生成更精准的攻击计划
- **安全防护不均匀**：MANUS和OmniParser的ASR远高于Claude CUA，说明不同agent的安全机制差异巨大，行业缺乏统一的安全标准
- **攻击覆盖面广**：热力图显示Detox2tox在所有12个风险子类别上都有成功案例，不局限于特定利基场景
- **迭代收益存在收敛趋势**：后续轮次的ASR提升递减，暗示防御方可能通过增强多轮检测来建立更有效的防线

## 亮点与洞察

- **Detox2tox流水线设计精巧**：将"获取执行计划"和"注入恶意载荷"解耦，利用语义变换绕过端到端安全检查，这种思路对安全研究有重要启发，揭示了纯基于内容过滤的防御方式的根本局限。
- **实际环境而非沙盒评估**：在真实操作系统上运行攻击任务，直接观察agent是否真的执行了删除文件、发送邮件等操作，比传统文本级评估更有说服力。
- **检查清单细粒度评估**：超越简单的成功/失败二分法，捕获部分攻击成功的细节，既为动态更新提供信号，也为安全评估提供了更合理的度量标准。
- **揭示"能力-安全"悖论**：明确指出SUDO作为外部框架，底层模型的能力提升会同时增强攻击效果，这对AI安全研究的方法论有深远意义。
- **标准化基准贡献**：50个任务覆盖4大类12子类，配合20种执行环境，为computer-use agent安全评估提供了首个系统化基准。

## 局限与展望

- **目标agent覆盖有限**：主实验仅在Claude CUA上完成50个全量任务，MANUS和OmniParser仅评估了12个子采样任务，受限于服务可用性和登录限制
- **ASR绝对值仍有提升空间**：即使动态3轮最高也仅41.33%，说明现有安全防护仍有一定效果，攻击框架还有优化的余地（如更多迭代轮次、更复杂的去毒/回毒策略）
- **未考虑多agent/Agent-to-Agent场景**：随着多agent系统的普及，Detox2tox在agent间协作场景中的可扩展性尚未验证
- **防御方案缺失**：论文主要聚焦攻击能力展示，未提出相应的防御机制或缓解策略
- **评估自动化的可靠性**：使用LLM作为评估器可能存在判断偏差，特别是在复杂任务的部分成功判断上

## 相关工作与启发

| 方向 | 代表工作 | 与SUDO的区别 |
|------|---------|-------------|
| 文本越狱 | Liu et al. 2023 (Role Play / Program Exec / Superior Model) | 仅针对纯文本LLM，在computer-use agent上ASR<8% |
| Web Agent攻击 | AdvWeb (Xu 2024), EIA (Liao 2025) | 通过网页内容注入攻击，非直接prompt攻击 |
| Agent安全评估 | AgentHarm (Andriushchenko 2025), InjecAgent (Zhan 2024) | 评估文本级agent漏洞，未涉及真实环境操作 |
| Mobile Agent安全 | MobileSafetyBench (Lee 2024) | 针对Android设备控制agent的间接提示注入 |
| **SUDO** | 本文 | 首个系统性攻击computer-use agent的框架，包含Detox2tox语义变换+动态迭代+真实环境评估 |

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Detox2tox的"去毒→生成→回毒"流水线是全新攻击范式，针对computer-use agent的系统性框架此前缺失
- 实验充分度: ⭐⭐⭐⭐ 6个VLM、3个target agent、50个任务、多轮迭代的系统评估，但MANUS/OmniParser仅12个任务略显不足
- 写作质量: ⭐⭐⭐⭐ 论文标题炫酷（sudo rm -rf），框架描述清晰，实验组织有条理，命名一致性好
- 价值: ⭐⭐⭐⭐⭐ 揭示了computer-use agent这一新兴范式的重大安全盲区，SUDO Dataset为后续安全研究提供了标准化基准

<!-- RELATED:START -->

## 相关论文

- [AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](../../NeurIPS2025/llm_agent/agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)
- [REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [Agentic Knowledgeable Self-Awareness](agentic_knowledgeable_self-awareness.md)
- [Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)

<!-- RELATED:END -->
