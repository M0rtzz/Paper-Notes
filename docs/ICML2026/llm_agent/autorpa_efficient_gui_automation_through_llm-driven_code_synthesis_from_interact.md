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
AutoRPA 把 GUI 环境建模成 POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{G}, \mathcal{O})$，要为某一类任务 $\mathcal{G}^k$ 蒸馏出一个能反复复用、token 几乎为零的 RPA 函数 $F_k$。整条流水线靠三个 Agent 接力跑：ReAct Agent 先在环境里探索拿到成功轨迹，翻译器把这些硬编码动作改写成靠语义属性定位的软编码版本，构建器再从多条软编码轨迹合成带条件和循环的 RPA 代码；代码在已见任务上验证，一旦失败就由分析器定位断点、ReAct 回真实环境补一段修正轨迹，构建器据此迭代改写，循环最多三轮。

### 关键设计

**1. 翻译器-构建器流水线：把"探索"和"写代码"拆给两个 Agent。**

直接让 LLM 凭空生成整段 GUI 代码几乎必败，因为它不知道目标界面长什么样；而 ReAct 探索出来的轨迹虽然带了完整环境知识，却是 `click(index=2)` 这种钉死在具体位置的硬编码，界面一改就废。AutoRPA 让翻译器 Agent 逐步接管：它读每一步动作及其前后观测，先做一次鲁棒性分析，再把硬编码动作翻成靠语义属性定位的软编码——比如把"点第 2 个元素"改写成"用 `find_element` 按文本内容和元素类型去找那个按钮"，必要时还顺手插一句断言来校验这步是否真的生效。构建器 Agent 拿到的是简化后的翻译轨迹 $\psi(\tau'_{\text{ReAct}}(g))$（扔掉原始观测、只留动作和执行摘要），在此基础上合成带条件判断和循环的完整 RPA 函数。这样"先探索拿环境知识、再翻译解耦位置、最后合成逻辑"的分工，既保住了 ReAct 的探索能力，又靠软编码让代码能跨不同界面状态稳定运行。

**2. 基于树结构的轨迹检索增强生成：构建器按需拉细节，而不是一次性塞满 prompt。**

构建器要写对代码就得知道界面真实状态，但把每步的截图和 DOM 全塞进 prompt 会瞬间爆长度，只给简化轨迹又会让它对界面凭空假设、写出错代码。AutoRPA 把轨迹库 $\mathcal{D}_\tau$ 组织成三层树：底层是每步完整交互块 $(o_t, a'_t, \rho_t, o_{t+1})$（含截图与 DOM），中层是只剩动作摘要的简化轨迹，顶层是一句结论摘要。构建器平时只看上层摘要，真需要某一步的具体界面时才调 `fetch_info(traj, step)` 逐层往下拉对应的多模态观测。让构建器自己决定何时取什么细节，就在"信息完整"和"prompt 不爆"之间找到了平衡。

**3. 混合修复策略：代码错了不让 LLM 闭眼调，而是回真实环境重探一段。**

合成出来的代码难免在某些任务上跑挂，常见做法是把报错丢回 LLM 让它静态调试，但 LLM 没看到真实界面，调起来基本靠猜。AutoRPA 改成"在环修复"：对一个已见任务执行 RPA 代码，跑到第一次失败就停，分析器 Agent 诊断这个断点——它读已执行的轨迹和当前观测，输出失败原因、已经完成了哪些子任务、以及可行的续行方案；接着 ReAct Agent 从断点处（或干脆从头）回到真实环境继续把任务做完，产出一段修正性示范轨迹 $\tau'_{\text{hybrid}}(F_k, g_*) = F_k(g_*) \oplus (A, o_{t*}, a'_{t*}, \rho_{t*}, \ldots, C)$。构建器拿着这段"真的能走通"的轨迹去改代码，依据是实测而非臆测，每个任务最多改 $M=3$ 次。正因为修正信号来自真实环境探索，修复成功率比纯静态调试高得多。

### 一个完整示例

以一个移动端"提交报告"任务为例，看三个 Agent 怎么接力。ReAct Agent 先在 App 里一步步点完整个流程，留下一条成功轨迹，其中有一步是 `click(index=2)`（点了那个"提交"按钮）。翻译器扫到这步，发现按索引定位很脆，于是改写成"按文本 `提交`、类型 `button` 调 `find_element` 定位后点击"，并补一句断言确认页面跳转到了成功页。构建器拿到整条软编码轨迹后，注意到不同用户的报告条目数不一样，便合成了一段带 `for` 循环的 RPA 函数来逐条填写，循环体里某一步它拿不准弹窗长什么样，就调 `fetch_info` 把那一步的 DOM 拉下来再写。代码生成后在几个已见任务上验证，某个任务因为多了一步确认弹窗而失败——分析器定位到"卡在弹窗确认"，ReAct 回到环境把弹窗点掉、走完剩余步骤，构建器据此在循环后补上处理弹窗的分支。三轮内代码稳定通过，之后这类任务就直接跑这个零推理的 $F_k$，不再调用 LLM。

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
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)

</div>

<!-- RELATED:END -->
