---
title: >-
  [论文解读] AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions
description: >-
  [ICML2026][LLM Agent][GUI自动化] 提出 AutoRPA 框架，通过翻译器-构建器流水线将 ReAct 风格 GUI Agent 的交互轨迹自动蒸馏为可复用的 RPA 函数，结合混合修复策略迭代优化代码…
tags:
  - "ICML2026"
  - "LLM Agent"
  - "GUI自动化"
  - "机器人流程自动化"
  - "代码合成"
  - "轨迹蒸馏"
---

# AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions

**会议**: ICML2026  
**arXiv**: [2605.21082](https://arxiv.org/abs/2605.21082)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: GUI自动化, 机器人流程自动化, 代码合成, LLM Agent, 轨迹蒸馏  

## 一句话总结
提出 AutoRPA 框架，通过翻译器-构建器流水线将 ReAct 风格 GUI Agent 的交互轨迹自动蒸馏为可复用的 RPA 函数，结合混合修复策略迭代优化代码，在保持甚至超越原始 Agent 成功率的前提下减少 82%~96% 的 token 消耗。

## 研究背景与动机

**领域现状**：基于 LLM 的 GUI Agent（如 SeeAct、M3A）已经能够通过 ReAct 范式在多步交互中完成各种 GUI 任务。然而这类方法在每个任务实例上都需要调用 LLM 进行推理，token 消耗高、运行慢。

**现有痛点**：在实际部署场景中，大量 GUI 任务是重复性的——同一用户每天提交报告、不同用户订购机票。对这类重复任务反复调用 LLM 推理既昂贵又低效。传统 RPA 虽然运行高效，但依赖人工编写脚本，开发维护成本高且对 GUI 布局变化脆弱。

**核心矛盾**：LLM Agent 灵活但昂贵（每次都要推理），传统 RPA 高效但僵硬（人工编写、难以泛化）。直接让 LLM 生成完整代码又往往因缺乏环境知识而失败；技能学习方法虽存储成功轨迹但泛化能力有限。

**本文目标**：自动将 LLM Agent 的决策逻辑蒸馏为可泛化、低 token 消耗的 RPA 函数，使其能在不同环境状态和任务指令下稳健执行。

**切入角度**：作者观察到 ReAct Agent 虽然推理成本高，但其成功轨迹蕴含了完成任务的完整决策逻辑。如果能将这些硬编码动作转化为软编码、再合成为带条件逻辑的 RPA 代码，就能兼顾灵活性和效率。

**核心 idea**：用翻译器将 ReAct 的硬编码动作转为基于语义属性的软编码动作，再由构建器从多条翻译后轨迹合成鲁棒的 RPA 函数，并通过混合修复策略迭代优化代码质量。

## 方法详解

### 整体框架
AutoRPA 将 GUI 环境建模为 POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{G}, \mathcal{O})$，目标是为指定任务类型 $\mathcal{G}^k$ 生成 RPA 函数 $F_k$，使其在最小化 token 消耗的同时保证任务成功。整体框架分为三个阶段：（1）探索阶段：ReAct Agent 交互生成轨迹，翻译器将硬编码动作转为软编码动作；（2）RPA 生成阶段：构建器基于翻译后轨迹合成 RPA 代码，通过 RAG 检索轨迹库中的详细信息；（3）验证与修复阶段：在已见任务上验证代码，失败时由分析器定位断点、ReAct Agent 继续探索修复，构建器据此迭代改进代码。

### 关键设计

1. **翻译器-构建器流水线**:

    - 功能：将 ReAct Agent 的硬编码交互轨迹转化为可复用的 RPA 函数代码
    - 核心思路：翻译器 Agent 接收每步动作及前后观测，先进行鲁棒性分析，再将硬编码动作（如 `click(index=2)`）转为基于语义属性定位的软编码动作（如通过 `find_element` 按文本内容和元素类型查找），同时可选地插入断言语句验证执行效果。构建器 Agent 接收简化后的翻译轨迹 $\psi(\tau'_{\text{ReAct}}(g))$（去除原始观测仅保留动作和执行摘要），生成包含条件判断和循环的 RPA 函数。构建器通过 Tree-organized RAG 机制按需检索轨迹库中的详细观测信息：底层保存交互块 $(o_t, a'_t, \rho_t, o_{t+1})$，中层为简化轨迹，顶层为结论摘要
    - 设计动机：直接让 LLM 生成完整 GUI 代码往往因缺乏环境知识而失败；而硬编码动作对 GUI 布局变化脆弱。通过"先探索后翻译再合成"的流水线，既利用了 ReAct 的探索能力获取环境知识，又通过软编码提升了跨环境泛化性

2. **混合修复策略**:

    - 功能：在代码验证失败时结合 RPA 执行与 ReAct 回退进行迭代修复
    - 核心思路：对已见任务执行 RPA 代码直到首次失败，分析器 Agent 诊断断点（分析已执行轨迹和当前观测，输出失败原因、已完成子任务、可行续行方案）。ReAct Agent 从断点处（或重新开始）继续完成任务，产生修正性示范轨迹 $\tau'_{\text{hybrid}}(F_k, g_*) = F_k(g_*) \oplus (A, o_{t*}, a'_{t*}, \rho_{t*}, \ldots, C)$，构建器据此改进代码。每个任务允许修改 $M=3$ 次
    - 设计动机：与简单地让构建器调试不同，混合修复利用 ReAct Agent 在真实环境中的探索获取实际修正轨迹，为构建器提供了具体而非臆测的改进依据，大幅提升修复成功率

3. **基于树结构的轨迹检索增强生成**:

    - 功能：让构建器按需获取历史交互的详细观测信息，避免因缺少具体界面状态而生成错误代码
    - 核心思路：轨迹库 $\mathcal{D}_\tau$ 组织为三层树结构——底层为每步交互块（含截图和 DOM 信息），中层为简化轨迹（动作摘要），顶层为结论摘要。构建器可通过 `fetch_info(traj, step)` 工具函数逐层向下检索所需上下文，在需要时才拉取多模态观测，兼顾了信息完整性和 prompt 长度
    - 设计动机：直接提供完整轨迹观测信息会导致 prompt 过长；而仅提供简化轨迹又容易让构建器对界面状态产生错误假设。RAG 方式让构建器自主决定需要哪些细节

## 实验关键数据

### 主实验

实验在三个 GUI 基准上进行：AndroidWorld（116 个任务类型，20 个真实 App）、WebArena（Reddit 域 19 个任务类型）、MiniWoB++（53 个任务类型）。

| 方法 | 模型 | 时间 (min) ↓ | Tokens (k) ↓ | 成功率 (%) ↑ |
|------|------|-------------|--------------|-------------|
| SeeAct | GPT-4.1 | 5.14 | 58.8 | 25.4 |
| M3A | GPT-4.1 | 2.23 | 103.4 | 48.3 |
| ReAct† | GPT-4.1 | 3.91 | 68.7 | 50.0 |
| AutoRPA (code only) | GPT-4.1 | 1.42 | 2.7 | 47.2 |
| **AutoRPA** | **GPT-4.1** | **1.81** | **12.8** | **51.7** |
| ReAct† | GPT-5 | 8.57 | 142.5 | 74.1 |
| AutoRPA (code only) | GPT-5 | 2.72 | 6.2 | 70.7 |
| **AutoRPA** | **GPT-5** | **4.35** | **30.6** | **75.9** |

在 MiniWoB++ 上（GPT-4.1）：

| 方法 | 9 类困难任务 Tokens (k) ↓ | 成功率 (%) ↑ | 全部 53 类 Tokens (k) ↓ | 成功率 (%) ↑ |
|------|-------------------------|-------------|------------------------|-------------|
| AdaPlanner | 15.1 | 74.1 | 6.1 | 90.3 |
| AutoManual | 23.2 | 91.1 | 4.6 | 95.2 |
| ReAct† | 16.2 | 84.4 | 9.2 | 92.8 |
| AutoRPA (code only) | 1.0 | 80.0 | 0.9 | 92.5 |
| **AutoRPA** | **1.4** | **91.1** | **1.4** | **95.4** |

### 消融实验

| 配置 | 成功率 (%) |
|------|-----------|
| AutoRPA (完整) | 51.7 |
| 构建阶段去掉 ReAct | 32.5 |
| 构建阶段去掉翻译器 | 40.2 |
| 代码修复去掉 ReAct | 45.5 |
| 构建器去掉 RAG | 48.8 |

### 关键发现
- 去掉 ReAct 探索后成功率从 51.7% 暴跌至 32.5%，说明直接让 LLM 生成 GUI 代码是不可靠的，ReAct 探索提供的环境知识至关重要
- 翻译器的贡献显著（去掉后降 11.5%），软编码动作对代码泛化性至关重要
- 仅靠 RPA 代码执行（code only）就能达到与 ReAct 接近的成功率，同时 token 消耗降至原来的 4%~7%，说明大部分任务的决策逻辑确实可以被蒸馏为确定性代码
- 随着构建任务数 $N$ 增加，AutoRPA (code only) 的成功率持续逼近 ReAct，验证了更多样本能帮助生成更鲁棒的 RPA 代码
- 在 WebArena 等高度多样化的真实 Web 环境中，AutoRPA 虽然在成功率上与现有方法持平，但 token 消耗大幅降低

## 亮点与洞察
- **轨迹蒸馏范式**：将 LLM Agent 的在线推理转化为离线代码，本质是一种"推理时计算"到"编译时计算"的转换。这个思路可以迁移到任何重复性 Agent 任务场景（如数据处理流水线、测试自动化）
- **软编码翻译**：通过语义属性定位 GUI 元素而非硬编码位置/索引，巧妙解决了 GUI 布局变化导致脚本失效的经典 RPA 痛点。这个设计理念适用于所有需要跨环境泛化的自动化脚本
- **混合修复 = 代码调试 + 环境探索**：不是让 LLM 纯靠想象调试代码，而是让 Agent 在真实环境中探索获取修正轨迹。这种"在环调试"策略比纯静态代码修复更可靠

## 局限与展望
- 构建阶段仍需消耗大量 token（每个任务类型需要采样 $N$ 个任务 + 反复验证修复），作者未充分讨论构建成本与测试阶段节省的平衡点
- 对于高度多样化的任务类型（如 WebArena），单一 RPA 函数难以覆盖所有情况，仍需回退到 ReAct，AutoRPA 的优势有所减弱
- 依赖 GUI 元素的语义属性进行定位，但对于属性信息贫乏的界面（如纯图像 UI）可能不适用
- 未来可以探索自动判断何时值得为某类任务构建 RPA（投入-产出分析），以及将 RPA 函数的局部更新与增量验证结合以降低维护成本

## 相关工作与启发
- **ReAct 范式** (Yao et al., 2023)：交替推理-行动的基础范式，AutoRPA 的探索和修复阶段都基于此
- **AutoManual** (Chen et al., 2024)：从交互中归纳环境规则指导后续任务，与 AutoRPA 的技能蒸馏思路互补
- **AdaPlanner** (Sun et al., 2023)：Plan-and-Execute 范式中的技能学习方法，但依赖人工示范
- **启发**：对于任何需要重复执行的 LLM 推理任务，都可以考虑"先用高成本方法探索并收集轨迹，再蒸馏为低成本确定性流程"的策略

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)
- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](../../ACL2026/llm_agent/don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](../../AAAI2026/llm_agent/ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)

</div>

<!-- RELATED:END -->
