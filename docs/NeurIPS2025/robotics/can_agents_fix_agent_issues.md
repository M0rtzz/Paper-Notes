---
title: >-
  [论文解读] Can Agents Fix Agent Issues?
description: >-
  [NeurIPS 2025][机器人][Agent系统维护] 本文首次系统地研究了 LLM-based Agent 系统的 issue 自动修复问题——通过人工分析 201 个真实 Agent issue 构建了涵盖 6 大类 20 个子类的 Agent issue 分类体系…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "Agent系统维护"
  - "Issue自动修复"
  - "软件工程Agent"
  - "Bug分类"
  - "benchmark"
---

# Can Agents Fix Agent Issues?

**会议**: NeurIPS 2025  
**arXiv**: [2505.20749](https://arxiv.org/abs/2505.20749)  
**代码**: [https://github.com/alfin06/AgentIssue-Bench](https://github.com/alfin06/AgentIssue-Bench)  
**领域**: 机器人  
**关键词**: Agent系统维护, Issue自动修复, 软件工程Agent, Bug分类, benchmark

## 一句话总结

本文首次系统地研究了 LLM-based Agent 系统的 issue 自动修复问题——通过人工分析 201 个真实 Agent issue 构建了涵盖 6 大类 20 个子类的 Agent issue 分类体系，耗费 500 人时构建了包含 50 个可复现任务的 AgentIssue-Bench 基准，并评估发现当前最先进的软件工程 Agent（如 SWE-agent、Agentless、AutoCodeRover）在 Agent issue 上的正确修复率仅为 3.33%–12.67%，远低于它们在传统软件上的 23%–51% 修复率。

## 研究背景与动机

**领域现状**：LLM-based Agent 系统正在成为一种全新的软件范式，已被广泛应用于医疗、编程、机器人、心理咨询等多个领域。MetaGPT、AutoGen、CrewAI、GPT-Engineer 等知名 Agent 系统在 GitHub 上拥有数万甚至数十万 star，说明了这一领域的活跃度和影响力。与此同时，软件工程领域也涌现出一批专门用于自动修复代码 issue 的 SE Agent（Software Engineering Agent），如 SWE-agent、Agentless、AutoCodeRover 等，它们在传统 Python 软件的 issue 修复上已展现出不错的能力——例如 Agentless 在 SWE-bench 上可以正确修复 50.80% 的 issue。

**现有痛点**：Agent 系统作为一种新兴的软件产品，与传统软件一样不可避免地存在各种 bug 和功能需求。例如，MetaGPT 截至 2025 年 5 月已经积累了超过 800 个 GitHub issue，维护工作量巨大。然而，Agent 系统与传统软件存在根本性差异：它们需要与 LLM 提供商交互、管理 Agent 的记忆状态、调用外部工具、处理 LLM 输出的非确定性等。这些独特特性使得 Agent 系统的 issue 在性质和修复难度上都与传统软件有很大不同。目前尚不清楚现有的 SE Agent 是否能有效修复 Agent 系统中的 issue。

**核心矛盾**：现有的 SE Agent 和 issue 修复基准（如 SWE-bench 系列）都聚焦于传统软件系统，完全忽略了 Agent 系统这一新兴软件范式。Agent 系统的核心组件——LLM 控制的"大脑"（负责任务分解和规划）、感知组件（接收环境信息）、行动组件（通过工具与环境交互）、以及记忆机制——引入了许多传统软件中不存在的问题类型。例如，LLM 提供商 API 的兼容性问题、prompt 管理问题、Agent 工作流的异常循环问题等，都是 Agent 系统独有的。在没有系统性理解 Agent issue 特性的情况下，很难开发出有效的自动修复方案。

**本文目标** (1) Agent 系统中常见的 issue 类型有哪些？与传统软件的 issue 有何不同？(2) 如何构建一个可复现的 Agent issue 修复基准？(3) 当前最先进的 SE Agent 在修复 Agent issue 上的效果如何？

**切入角度**：从真实世界出发，收集 16 个主流 Agent 系统的 GitHub issue 和开发者提交的修复 patch，通过扎根理论（grounded theory）方法系统性地分析和归类 Agent issue。在此基础上，投入大量人力构建可复现的基准测试环境，每个任务都打包在 Docker 容器中，配有失败触发测试脚本，确保结果的可验证性。这种"先理解问题，再构建基准，最后评估工具"的研究范式确保了结论的可靠性。

**核心 idea**：Agent 系统的 issue 具有与传统软件截然不同的特性——涉及 LLM 提供商兼容性、工具调用、记忆管理、LLM 操作、工作流控制等 Agent 特有组件——而当前最先进的 SE Agent 在修复这些 issue 上能力极为有限，亟需开发专为 Agent 系统定制的维护工具。

## 方法详解

### 整体框架

本文的研究分为三个核心阶段：(1) **Agent Issue 分类体系构建**——从 16 个主流 Agent 系统中收集 201 个高质量 GitHub issue，通过扎根理论方法进行人工标注和分类，形成包含 6 大类 20 个子类的分类体系；(2) **AgentIssue-Bench 基准构建**——从 201 个 issue 中筛选出 50 个可稳定复现的 issue，每个 issue 配备 Docker 环境、失败触发测试、buggy 版本和 patched 版本的代码库；(3) **SE Agent 评估**——在 AgentIssue-Bench 上评估 SWE-agent、AutoCodeRover 和 Agentless 三个 SE Agent 搭配 GPT-4o 和 Claude-3.5-Sonnet 两个骨干 LLM 的修复效果。

### 关键设计

1. **Agent Issue 分类体系构建方法**:

    - 功能：从真实世界的 Agent 系统中系统性地收集和分类 issue，建立第一个 Agent issue 分类体系
    - 核心思路：首先通过 GitHub Search API 获取以"AI agents"为关键词的 50 个仓库，经过人工筛选保留真正的 LLM-based Agent 系统（过滤论文列表、教程等无关仓库），进一步筛选出拥有 1k+ star 和 30+ issue 的活跃项目，最终得到 16 个 Agent 系统（包括 MetaGPT、AutoGen、CrewAI、GPT-Engineer、BabyAGI、CAMEL、ChatDev 等）。对于每个系统的 issue，采用三个筛选标准：(i) issue 已被关闭且有开发者提交的修复 patch（作为根因理解的 ground truth）；(ii) issue 描述清晰，无误导信息；(iii) 每条 issue 只包含一个问题。最终收集到 201 条高质量 issue。随后，将其中 171 条 (85%) 用于建设分类体系、30 条 (15%) 用于评估分类体系。三位具有丰富软件开发和机器学习经验的标注员使用开放编码（open coding）方法对每个 issue 进行标注，将 issue 分解为片段并赋予描述性标签，再通过合并和关联形成结构化分类。所有标注员讨论并审查分类体系直至达成共识。评估阶段两名独立标注员的 Cohen's Kappa 达到 0.849，且未出现新的类别，验证了分类体系的泛化性和可靠性
    - 设计动机：过去缺乏对 Agent issue 的系统性理解。传统软件的 issue 分类体系无法覆盖 Agent 系统独有的问题类型（如 LLM 提供商兼容性、Agent 记忆错误、工作流异常等）。只有先建立清晰的分类体系，才能有针对性地评估和改进自动修复工具的能力

2. **六大类 Agent Issue 分类体系**:

    - 功能：将 Agent 系统的 issue 归纳为 6 大类 20 个子类，每类附有详细定义和真实案例
    - 核心思路：6 大类分别为——
      **(a) LLM 提供商不兼容性 (7.46%)**：包括依赖库不兼容（如 anthropic 库 API 变更）、不支持新模型（如缺少 GPT-4 Turbo 支持）、API 参数不兼容（如向不支持 stop 参数的 o1-preview 模型传递 stop 参数）。
      **(b) 工具相关问题 (18.41%)**：包括工具依赖缺失（如缺少 tenacity 模块）、工具配置错误（如无法单独设置 DuckDuckGo 检索器）、工具实现错误（如 SWE-agent 读取 .docx 文件时 decode 崩溃）、工具接口误用（如 Wikipedia API 的 auto_suggest 导致搜索词被纠错）。
      **(c) 记忆相关问题 (14.43%)**：包括记忆初始化错误（如 CrewAI 重置记忆时找不到 crew 实例）、记忆内容错误（如消息存储逻辑问题导致内容丢失）、记忆依赖问题（如 InnerMessage 被重命名为 AgentMessage 导致导入失败）。
      **(d) LLM 操作问题 (31.84%，占比最大)**：包括模型访问配置错误（如 API key 设置逻辑不兼容）、token 用量配置错误（如 max_tokens 默认值为 NOT_GIVEN 导致类型错误）、模型输出处理器错误（如 Gemini 模型屏蔽敏感内容后缺少异常处理）、模型依赖问题（如 transformers 版本冲突）、上下文长度问题（如输入超过最大上下文长度限制）、prompt 相关问题（如多语言设置下 Introduction/Conclusion 仍用英文生成）。
      **(e) 工作流问题 (6.97%)**：如 Agent 在多个并行前置步骤完成后后续步骤执行失败。
      **(f) 通用工具问题 (20.90%)**：包括非 LLM 相关的实现错误（如 UI/Docker/日志问题）、通用依赖问题（如 pytest 版本兼容性）、通用配置问题（如 YAML 文件编码问题）。其中只有"通用工具问题"可能在传统软件中也出现，其余五大类都与 Agent 系统的核心组件紧密相关
    - 设计动机：这一分类揭示了 Agent 系统维护的复杂性——开发者需要同时管理 LLM 提供商接口、LLM 操作逻辑、记忆机制、工具调用等多个异质组件的正确性。LLM 操作问题占比最大 (31.84%) 说明 LLM 交互是 Agent 系统最容易出错的环节

3. **AgentIssue-Bench 基准构建**:

    - 功能：从 201 个 issue 中构建一个包含 50 个可稳定复现任务的基准测试集，每个任务配备完整的可执行环境
    - 核心思路：采用了严格的三步筛选流程。**Step 1 故障复现**：拉取每个 issue 对应的 buggy commit，搭建 Agent 系统环境，手写失败触发测试脚本（failure-triggering test）来复现 issue 描述中的问题行为，过滤掉无法观察到相同 buggy 行为的 issue。**Step 2 Patch 验证**：拉取对应的 patched commit，在其上运行失败触发测试，只保留 patched 版本能通过测试的 issue。**Step 3 非脆弱性验证**：将前两步重复三次，排除因 LLM 非确定性导致的不稳定行为（flaky test）。经过这三步筛选，201 个 issue 缩减为 50 个可复现任务。每个任务实例包含：(i) 用户报告的 issue 描述文本；(ii) buggy 版本的代码库；(iii) 开发者提交的修复 patch（ground truth）；(iv) 失败触发测试脚本；(v) Docker 容器化环境（含所有依赖和配置）。所有 Docker 镜像托管在 Docker Hub 上，支持一键拉取和执行。整个复现过程耗费约 500 人时
    - 设计动机：Agent issue 的复现远比传统软件困难，主要原因有四：(i) LLM 输出的非确定性导致工作流错误难以稳定复现；(ii) 外部资源（工具、依赖库、LLM 提供商）可能在 issue 报告后发生变化；(iii) issue 描述缺乏足够的复现步骤细节；(iv) Agent 系统搭建过程中可能出现与 issue 描述不同的意外错误。正因如此，构建可复现的基准需要极大的人力投入，这也是为什么之前没有人做过 Agent issue 修复基准的原因

### 评估指标设计

本文使用三个层次的评估指标：**(1) 定位准确率 (Localization Accuracy)**——生成的 patch 是否修改了与开发者 patch 相同的位置，分为文件级和函数级两个粒度；**(2) 表面修复率 (Plausible Resolution Rate)**——生成的 patch 是否能让失败触发测试通过（但不保证语义正确）；**(3) 正确修复率 (Correct Resolution Rate)**——在表面修复的基础上，由人工标注员进一步判断生成的 patch 是否与开发者 patch 语义等价。之所以需要区分"表面修复"和"正确修复"，是因为在实践中测试的覆盖率不足，表面修复的 patch 可能只是过拟合了测试用例，并未真正解决问题（即所谓的 overfitting patch）。为消除 LLM 随机性的影响，所有实验重复三次并报告平均结果。

### SE Agent 实验设置

本文选取了三个完全开源且在传统软件 issue 修复上表现优异的 SE Agent：**(1) SWE-agent**——通过自定义 Agent-Computer Interface (ACI) 与代码仓库环境交互，能够操作文件和执行 bash 命令；**(2) AutoCodeRover**——配备了一系列代码搜索工具，通过迭代检索相关代码上下文来导航仓库和定位 issue 位置；**(3) Agentless**——用人类专家知识优化 Agent 工作流，融入层次化定位和回归测试来提高 issue 修复率。每个 SE Agent 分别搭配 GPT-4o 和 Claude-3.5-Sonnet 两个骨干 LLM 进行测试，均使用原始发布的超参数设置。所有实验在 AgentIssue-Bench 的 Docker 环境中执行，确保环境一致性。

## 实验关键数据

### 主实验

| SE Agent | 骨干 LLM | 表面修复率 | 正确修复率 | 文件级定位 | 函数级定位 | 平均成本 |
|----------|----------|-----------|-----------|-----------|-----------|---------|
| Agentless | GPT-4o | 18.67% | 6.00% | 22.97% | 14.40% | $0.65 |
| Agentless | Claude-3.5-S | 12.67% | 8.67% | 21.86% | 11.54% | $0.33 |
| AutoCodeRover | GPT-4o | 12.67% | 4.67% | 17.05% | 11.30% | $0.23 |
| AutoCodeRover | Claude-3.5-S | 17.33% | **12.67%** | **25.61%** | **18.67%** | $0.05 |
| SWE-agent | GPT-4o | 5.33% | 3.33% | 12.65% | 11.77% | $1.15 |
| SWE-agent | Claude-3.5-S | 6.67% | 6.67% | 15.58% | 11.26% | $0.57 |

对比传统软件（SWE-bench Lite）：这些 SE Agent 在传统 Python 软件 issue 上的修复率为 23.20%–50.80%，而在 Agent issue 上仅为 3.33%–12.67%，**下降幅度高达 70%–90%**。

### 按类别分解的修复情况

| Issue 类别 | 可修复占比 | 典型可修复子类 | 子类修复率 |
|-----------|-----------|-------------|-----------|
| 工具相关问题 | 3/9 (33.33%) | 工具依赖问题 | 2/3 (66.67%) |
| 记忆相关问题 | 1/8 (12.50%) | 记忆内容错误 | 1/5 (20.00%) |
| LLM 操作问题 | 2/11 (18.18%) | 模型访问配置 / Prompt 问题 | 各 1/2 (50.00%) |
| 工作流问题 | 1/6 (16.67%) | — | — |
| 通用工具问题 | 8/14 (**57.14%**) | 通用依赖问题 | 2/2 (100.00%) |
| | | 通用配置问题 | 4/6 (66.67%) |

### 关键发现

- **SE Agent 主要能修复的是"通用工具问题"**（57.14% 修复率），因为这类问题与传统软件高度相似（如日志、文件操作、UI 配置等），SE Agent 对此有积累。而 Agent 特有的问题类别（LLM 提供商兼容性、记忆、工作流等）修复率极低
- **Claude-3.5-Sonnet 总体优于 GPT-4o**：在正确修复率、表面修复率和定位准确率上，Claude 配对的 SE Agent 表现更好。AutoCodeRover + Claude-3.5-S 取得了最高的正确修复率 12.67% 和文件级定位准确率 25.61%
- **不同 SE Agent 具有互补性**：每个 SE Agent 都能独特修复 2–4 个其他 Agent 不能修复的 issue，但没有任何一个 issue 能被所有 SE Agent 同时修复。这表明组合多个 SE Agent 可能获得更好的效果
- **Agent issue 的 patch 规模远大于传统软件**：平均需要修改 66 行代码、3.58 个文件、6.79 个函数，最大的 patch 涉及 355 行、34 个文件、54 个函数。这远超 SWE-bench 中典型 issue 的修改规模
- **依赖类问题相对容易修复**：工具依赖和通用依赖类 issue 修复率最高（66.67%–100%），因为这类问题通常有明确的错误信息（如缺少库、版本冲突），即使涉及 Agent 特有组件也容易定位
- **LLM 操作问题几乎无法修复**：占比最大的 LLM 操作问题（31.84%）修复率极低。SE Agent 缺乏对 LLM API 特性的最新知识（如哪些模型支持 stop 参数、如何处理 Gemini 的内容屏蔽等），且 Agent 工作流中 LLM 交互涉及复杂的非确定性行为，难以定位根因。论文附录给出了两个典型的未修复案例：一个是 CrewAI 中 o1-preview/o1-mini 不支持 stop 参数的问题，SE Agent 不仅没有正确修复反而加剧了问题（向不支持的模型传递了 stop 参数）；另一个是 Aider 中 LLM 为单个文件生成多个 diff 导致冲突的问题，SE Agent 仅打印了错误消息而未分析根因（正确的修复是只保留第一个 diff）
- **LLM 提供商兼容性问题完全无法修复**：这类问题要求 SE Agent 了解 LLM 提供商 API 的最新变化和不同模型的参数差异，而这些知识往往不在 SE Agent 的训练数据中
- **统计显著性分析**：重复三次实验的 2-sigma 误差范围为 ±2.31% 到 ±6.67%，表明即使考虑随机波动，Agent issue 修复率仍远低于传统软件
- **成本可控但效果不佳**：应用 SE Agent 修复 Agent issue 的平均成本为 $0.05–$1.15，与修复传统软件 issue（$0.45–$2.53）相当，但修复成功率却低得多，性价比极差

## 亮点与洞察

- **首个 Agent Issue 分类体系**：通过严格的扎根理论方法构建了涵盖 6 大类 20 子类的分类体系，Cohen's Kappa 达 0.849，是后续研究 Agent 系统质量问题的重要参考框架。这一分类将 Agent 系统的复杂性拆解为可量化的维度，使得研究者可以有针对性地改进 SE Agent 在特定类别上的能力。分类体系的附录中为每个子类都提供了来自真实开源项目的详细示例，包括 issue 链接、描述和修复策略，极具实践参考价值
- **可复现性设计极为严格**：500 人时的投入、三轮非脆弱性验证、Docker 容器化封装——这是目前 Agent issue 修复领域唯一一个真正做到可复现的基准。相比之下，很多基准只提供了代码和描述，没有解决 LLM 非确定性和外部资源变化的问题。所有 Docker 镜像都托管在 Docker Hub 上，支持一键拉取和评估，大大降低了后续研究者的复现门槛
- **修复率的巨大落差揭示了根本性差异**：传统软件 50.80% vs Agent 系统 12.67%，这个落差不是量的差别而是质的差别。它说明当前 SE Agent 的核心能力（代码搜索、bug 定位、patch 生成）是为传统确定性软件设计的，面对涉及 LLM 交互、工具调用、记忆管理的 Agent 系统时根本不够用
- **"通用工具问题容易修、Agent特有问题难修"的发现提供了清晰的研究方向**：这意味着未来需要为 SE Agent 注入 Agent 系统的领域知识——包括 LLM 提供商 API 的最新文档、Agent 框架的架构模式、工作流调试方法等
- **互补性分析有价值**：通过 Venn 图分析，发现三个 SE Agent 的修复能力高度互补（每个都能独特修复 2-4 个 issue），暗示 Agent ensemble 或多策略融合可能是提升 Agent issue 修复率的一个简单有效的方向

## 局限与展望

- **基准规模较小**：50 个 issue 虽然质量高、可复现，但数量有限，可能无法覆盖所有 Agent issue 类型的边界情况。作者也承认了这一点，计划持续扩展
- **Agent 系统样本偏差**：16 个 Agent 系统主要是开源、Python-based 的项目，可能无法代表商业化或多语言 Agent 系统的 issue 特征
- **SE Agent 选择范围有限**：只评估了 SWE-agent、AutoCodeRover 和 Agentless 三个 Agent，未涵盖 Devin、OpenDevin、Moatless 等更新的工具。随着 SE Agent 快速迭代，评估结论可能需要更新
- **没有提出解决方案**：本文主要是"发现问题"的工作，指出了 Agent issue 修复的困难但没有提出改进的技术方案。后续研究可以考虑：(i) 为 SE Agent 提供 LLM 提供商 API 的实时文档检索能力；(ii) 在 SE Agent 的训练数据中加入 Agent 系统的代码和 issue 修复案例；(iii) 开发专门针对 Agent 工作流调试的交互式工具
- **非确定性问题的根本挑战**：即使构建了基准，Agent 系统中 LLM 的非确定性本质意味着很多 issue 在理论上就难以完全消除（如工作流的循环/挂起），这类问题可能需要从 Agent 架构层面而非 patch 层面来解决
- **评估指标的局限性**：语义等价性的判断依赖人工标注，存在主观性。对于复杂的 Agent 系统修复（如涉及 prompt 调整或错误处理策略变更），何为"正确"的修复本身就有多种合理方案
- **未考虑多步修复和交互式调试**：当前评估框架要求 SE Agent 一次性生成完整 patch，但实际的 Agent issue 调试往往需要多轮交互——运行 Agent、观察输出、定位问题、调整修复。引入交互式修复范式可能显著提升修复率
- **16 个 Agent 系统的详细统计**：分析涵盖了多样化的 Agent 系统，包括 MetaGPT (55.4k star, 90.7k LoC)、AutoGen (44.2k star, 198k LoC)、CrewAI (31.3k star, 171k LoC) 等，代码量从 BabyAGI 的 8.8k LoC 到 CAMEL 的 206k LoC 不等，时间跨度从 2023 年 3 月到 2024 年 7 月，使用的开源许可包括 MIT、Apache-2.0 和 CC-BY-4.0 等

## 相关工作与启发

- **vs SWE-bench 系列**：SWE-bench / SWE-bench Lite / SWE-bench Verified 都聚焦传统 Python 软件的 issue，SWE-bench Java 和 SWE-bench Multimodal 分别扩展到 Java 和前端 JavaScript 领域，SWE-Lancer Diamond 则涵盖 Expensify 商业软件，但所有这些都不涉及 Agent 系统。本文填补了这一空白，证明了 Agent 系统的维护是一个需要独立研究的方向。值得注意的是，本文构建基准的质量标准（三步筛选 + 非脆弱性验证）比很多传统软件基准更为严格
- **vs Cemri et al. (2025)**：他们也研究了 multi-agent 系统的失败模式，但聚焦于运行时失败的症状分析（通过分析失败 trajectory），而本文聚焦于 issue 修复——不仅包含 bug 修复，还包含功能请求。两者视角互补，本文还额外提供了可复现基准和 SE Agent 评估
- **vs Shao et al. (2025)**：他们研究了 LLM-integrated 系统的集成 bug，但范围更广（包括所有使用 LLM 的系统），而本文专注于 LLM-based Agent 系统这一特定类型，分析更加深入，并且提供了可复现的基准测试
- **启发**：这篇论文的核心启示是——Agent 系统已经成为一个独立的软件品类，它的维护和调试需要全新的工具和方法。传统 SE 工具直接迁移到 Agent 系统上效果很差，这与 Agent 系统的复杂组件架构、LLM 交互的非确定性、以及快速变化的外部依赖密切相关。未来可能需要"Agent for Agent"——即专门为维护 Agent 系统而设计的 Agent。此外，分类体系中"LLM 操作问题"占比高达 31.84% 但修复率极低这一发现，对于 Agent 框架的设计者也有重要参考价值：应该在架构层面提供更好的 LLM 交互抽象、更完善的异常处理机制、以及更强的可调试性

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地研究 Agent 系统的 issue 分类和自动修复，填补重要空白，但研究范式（分类+基准+评估）较为经典
- 实验充分度: ⭐⭐⭐⭐ 16个 Agent 系统、201个 issue、50个可复现任务、3个 SE Agent × 2个 LLM × 3次重复，工作量极大；但只有3个 SE Agent 且基准规模较小
- 写作质量: ⭐⭐⭐⭐ 结构清晰、动机充分、分类体系详细附有丰富示例，附录包含每个子类的完整案例
- 价值: ⭐⭐⭐⭐⭐ 揭示了 Agent 系统维护这一重要且紧迫的问题，提供了首个可复现基准和详细的诊断分析，对 SE 和 Agent 两个社区都有很高的参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](../../ICLR2026/robotics/rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[NeurIPS 2025\] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling](autotom_scaling_model-based_mental_inference_via_automated_agent_modeling.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)
- [\[NeurIPS 2025\] Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents](provable_ordering_and_continuity_in_vision-language_pretraining_for_generalizabl.md)

</div>

<!-- RELATED:END -->
