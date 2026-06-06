---
title: >-
  [论文解读] RExBench: Can coding agents autonomously implement AI research extensions?
description: >-
  [ACL2026][代码智能][Coding Agent] RExBench 把 coding agent 放进真实 AI 论文代码库中，让它们实现专家设计的研究扩展并用受控执行结果评分，发现当前最强 agent 也只有约三分之一成功率，距离自主完成科研扩展仍有明显差距。
tags:
  - "ACL2026"
  - "代码智能"
  - "Coding Agent"
  - "AI Research Agent"
  - "研究扩展"
  - "自动评测"
  - "代码生成基准"
---

# RExBench: Can coding agents autonomously implement AI research extensions?

**会议**: ACL2026  
**arXiv**: [2506.22598](https://arxiv.org/abs/2506.22598)  
**代码**: https://rexbench.com/  
**领域**: 代码智能  
**关键词**: Coding Agent、AI Research Agent、研究扩展、自动评测、代码生成基准

## 一句话总结
RExBench 把 coding agent 放进真实 AI 论文代码库中，让它们实现专家设计的研究扩展并用受控执行结果评分，发现当前最强 agent 也只有约三分之一成功率，距离自主完成科研扩展仍有明显差距。

## 研究背景与动机
**领域现状**：LLM agent 已经能解决一部分软件工程任务，如修复 GitHub issue、改写代码、跑实验脚本、处理数据分析流程。与此同时，AI for Science 和自动化科研系统也在尝试让 agent 执行实验设计、代码复现和论文分析。

**现有痛点**：现有基准要么偏通用软件工程，要么评估论文复现、Kaggle 风格建模或开放式研究问答。它们很少衡量一个更接近真实科研迭代的能力：在已有论文和代码库基础上，实现一个从未公开过的研究扩展，并产出与专家实现一致的实验结果。

**核心矛盾**：研究扩展任务必须真实、开放、有科学意义；但自动评测又要求任务可执行、结果可判定、环境可控。过度简化会失去科研味道，过度开放又无法稳定评分。

**本文目标**：作者希望构建一个在真实性和可自动评测之间折中的 benchmark，评估 coding agent 是否能理解论文背景、阅读原始代码、定位修改点、实现新实验，并在隔离环境中复现专家 gold solution 的数值结果。

**切入角度**：RExBench 选择 12 篇 NLP/ML 论文的代码库，由领域专家设计研究扩展任务，并把 solution 与成功标准保存在私有评测基础设施中。agent 只能拿到论文、代码库和高层扩展指令，最后提交 patch，由系统执行并比较结果。

**核心 idea**：用“实现未公开研究扩展”替代“复现已有论文结果”作为 coding agent 的科研能力测试，从而同时缓解数据污染并贴近真实研究工作流。

## 方法详解
RExBench 的任务形式是：给定一篇或多篇相关论文、原始代码库、专家写的扩展指令，agent 需要编辑代码库实现扩展实验。系统把 agent 产生的 patch 应用到原始仓库，在固定 VM 和容器环境中运行，并根据输出文件或数值结果判断是否成功。

### 整体框架
整个 benchmark 包含 12 个研究扩展任务，覆盖模型、算法、数据和评估方法四类改动。例如 WinoDict 任务要求把合成目标词替换为不同频率组的真实英文词，检查已有词义是否干扰 in-context word acquisition；Othello 任务要求改变 probe 的棋盘状态表示；Tree of Thoughts 任务要求分析算法在特定模型上的 failure mode。

每个任务由领域专家先验证原始代码可以复现，再实现 gold extension，并记录数值结果。agent 提交 patch 后，评测基础设施在同样硬件、随机种子和依赖环境中执行。最终成功取决于 agent 输出的实验结果是否落在 gold 结果的精确值或窄范围内。

### 关键设计
1. **以研究扩展为核心任务单元**:

	- 功能：评估 agent 对“已有研究之上的新假设验证”的能力。
	- 核心思路：每个任务都不是修小 bug，也不是照抄论文复现，而是让 agent 根据专家指令修改模型、数据、算法或评估流程，并得到新的实验结果。
	- 设计动机：真实科研经常从“如果我们把 X 换成 Y 会怎样”开始。RExBench 把这种扩展变成可执行任务，比单纯代码题更能测科研场景中的代码理解和实验实现能力。

2. **私有 gold solution 与受控执行评测**:

	- 功能：降低数据污染风险，并提高数值评测可信度。
	- 核心思路：gold edits 与评测脚本不公开，agent 只能提交 patch。系统在任务专用 Apptainer 容器中应用 patch、运行实验、收集结果和日志。对于有随机性的任务，gold solution 用 5 个 seed 估计均值和 $\pm 2$ 标准差范围。
	- 设计动机：如果 solution 已经公开，agent 可能靠记忆或训练数据泄漏成功。私有评测使成功更接近真实自主实现。

3. **多层指标诊断 agent 失败原因**:

	- 功能：区分代码是否能跑、是否改对文件、最终科学结果是否正确。
	- 核心思路：主指标是 final success rate；辅助指标包括 execution success rate 和 file recall。前者看实验输出是否匹配 gold，后者看代码是否正常运行，file recall 看 agent 编辑的文件与专家编辑文件的重合度。
	- 设计动机：研究代码任务的失败往往不是简单 compile error。agent 可能找到正确文件但逻辑错，也可能代码能跑但实验结果偏离。多指标能帮助分析能力瓶颈。

### 损失函数 / 训练策略
本文不训练模型，而是构建 benchmark 并评估 agent。实验测试 12 个 agent 组合，使用 aider 和 OpenHands 两类 agent framework，backbone 包括 Claude 4/3.7 Sonnet、GPT-5、o1、o4-mini、DeepSeek-R1 等。每个任务每个 agent 运行 5 次以估计随机波动。作者还额外测试两级 human-written hints：一级帮助定位信息，二级提供更具体实现步骤。

## 实验关键数据

### 主实验
主实验显示，当前 agent 对研究扩展仍然明显不足。最佳组合 OpenHands + Claude 4 Sonnet 的平均 final success rate 约为 33%，execution success rate 为 68%。

| Agent 设置 | Final Success Rate | Execution Success Rate | 主要观察 |
|--------|------|------|------|
| OpenHands + Claude 4 Sonnet | 约 33% | 68% | 最强组合，仍无法完成大多数扩展 |
| OpenHands + GPT-5 | 低于 Claude 4 | 非零且较强 | 经常语法正确但实验结果偏离 |
| Claude 3.7 / Claude 4 系列 | 明显优于弱模型 | 较高 | 能定位核心文件并产生可运行实现 |
| o1 / DeepSeek-R1 | 接近或等于 0 成功 | DeepSeek-R1 完全失败 | 推理模型在 agent loop 中可能过度思考 |
| aider + o4-mini / DeepSeek-R1 | 很低 | 常有空 patch | 非迭代式框架对复杂任务不友好 |

这些结果说明，强 backbone 能让 agent 更接近正确实现，但“可运行”与“科学结果正确”之间仍有很大差距。

### 消融实验
作者提供不同层级 hints，观察 agent 是否能利用人工提示。

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无 hints | 最佳 final success 约 33% | 真实自主研究扩展设置 |
| 信息定位 hints | 部分强 agent 提升 | 直接指出文件或关键信息后，强模型更能实施 |
| 步骤级 hints | OpenHands + Claude 4 / GPT-5 可到 43% | 具体步骤能帮助强 agent，但仍未过半 |
| 弱 agent + hints | 提升有限 | 需要基本代码理解能力才能利用提示 |
| 个别任务 | hints 可能反而降低 | 如果提示指定了模型不擅长的实现路径，agent 可能失败 |

### 关键发现
- 大多数 agent 能找到核心编辑区域，file recall 较高，但 final success 低，说明瓶颈不只是定位文件，而是正确理解实验逻辑和实现细节。
- 显式错误主要包括 Python value error、空 patch、SyntaxError、执行超时。Claude 和 GPT-5 系列较少出现 SyntaxError。
- 隐式错误更危险：代码能运行，但结果与 gold 不匹配。作者在 top-2 agent 上分析发现，隐式错误中逻辑错误与数值/参数错误约为 2:1。
- 更强模型产生更多隐式错误。这意味着未来 agent 可能越来越少“明显崩溃”，但更常给出看似合理却科学结论错误的实现。
- gold solution 的代码改动行数对 final success 有显著负面影响，回归系数为 $\beta=-0.038$ 且 $p<0.01$，说明实现工作量是主要难度来源。

## 亮点与洞察
- RExBench 把 agent 评测从“写出能过单测的代码”推进到“执行可验证研究扩展”。这对科研 agent 领域非常关键，因为真实风险往往来自实验结论错误，而不是代码是否语法正确。
- 私有 gold solution 的设计很重要。相比 PaperBench 这类论文复现任务，研究扩展天然降低了训练数据泄漏的可能性，也更能测 agent 的在线推理和代码理解。
- 论文指出强模型的失败更难调试，这是非常现实的工程提醒。可运行但结果错的 agent patch 可能误导研究者，把错误实验带入论文。
- hints 实验提供了对人机协作的启发：agent 不是简单地“有提示就会成功”。提示的粒度、路径和模型能力之间有交互，未来需要设计更稳健的 human-agent protocol。

## 局限与展望
- 为了自动评测，任务必须有明确数值目标，因此仍比真实开放科研更理想化。现实中，扩展想法往往需要多轮试错和重新定义实验。
- 当前只有 12 个任务，虽然单任务信息量大，但对相近模型能力差异的统计区分能力有限。后续社区扩展任务数量和领域覆盖很重要。
- benchmark 主要集中在 NLP/ML 研究代码。其他科学领域的实验环境、数据规模、模拟器和评测指标可能带来额外挑战。
- 过程级指标还不够。作者也建议加入 landmark evaluation 等中间检查，以缓解隐式错误的事后分析困难。
- 自动执行 machine-written code 存在安全风险，论文采用无互联网容器环境是必要防护。未来真实部署需要更严格沙箱、权限控制和审计。

## 相关工作与启发
- **vs SWE-bench**: SWE-bench 测 GitHub issue 修复，RExBench 测科研假设扩展。后者更强调实验逻辑、论文理解和数值结果。
- **vs PaperBench / Paper2Code**: 这些工作更关注论文复现或从论文到代码；RExBench 关注未公开的新扩展，因此更能缓解数据污染。
- **vs MLE-bench / MLAgentBench**: 这些 benchmark 更像 ML pipeline 或 Kaggle 式问题；RExBench 的输入包含论文语境和原始代码，任务更贴近研究迭代。
- **启发**: 未来科研 agent 评测应把“可运行”“可解释”“结果正确”“可复查”分开计量，不能只用最终 leaderboard 分数判断可靠性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用研究扩展而非复现来评测 agent，是很贴近科研场景且抗污染的设定。
- 实验充分度: ⭐⭐⭐⭐☆ 任务深、分析细，但任务数量仍偏少，领域覆盖主要集中在 NLP/ML。
- 写作质量: ⭐⭐⭐⭐☆ benchmark 构造和错误分析清楚，部分 agent 结果依赖图表，文本版需要结合附录看全数值。
- 价值: ⭐⭐⭐⭐⭐ 对科研 agent、安全执行、自动化实验评测和人机协作设计都有高参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[ICML 2026\] MARS: Modular Agent with Reflective Search for Automated AI Research](../../ICML2026/code_intelligence/mars_modular_agent_with_reflective_search_for_automated_ai_research.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ACL 2026\] SecureVibeBench: Evaluating Secure Coding Capabilities of Code Agents with Realistic Vulnerability Scenarios](securevibebench_evaluating_secure_coding_capabilities_of_code_agents_with_realis.md)

</div>

<!-- RELATED:END -->
