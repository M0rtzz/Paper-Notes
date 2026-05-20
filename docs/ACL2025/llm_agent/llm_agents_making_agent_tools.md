---
title: >-
  [论文解读] LLM Agents Making Agent Tools
description: >-
  [ACL 2025][LLM Agent][工具生成] 本文提出ToolMaker，一个自主将GitHub代码仓库转化为LLM兼容工具的代理框架，给定一个仓库URL和任务描述即可自动安装依赖、生成调用代码并通过闭环自修复机制调试，在涵盖多个领域15个复杂任务的新基准上正确实现了80%的任务…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "工具生成"
  - "自主代理"
  - "代码仓库转工具"
  - "闭环调试"
  - "科学工作流"
---

# LLM Agents Making Agent Tools

**会议**: ACL 2025  
**arXiv**: [2502.11705](https://arxiv.org/abs/2502.11705)  
**代码**: [GitHub](https://github.com/KatherLab/ToolMaker)  
**领域**: LLM Agent  
**关键词**: 工具生成, 自主代理, 代码仓库转工具, 闭环调试, 科学工作流

## 一句话总结

本文提出ToolMaker，一个自主将GitHub代码仓库转化为LLM兼容工具的代理框架，给定一个仓库URL和任务描述即可自动安装依赖、生成调用代码并通过闭环自修复机制调试，在涵盖多个领域15个复杂任务的新基准上正确实现了80%的任务，大幅超越了现有软件工程代理。

## 研究背景与动机

**领域现状**：工具使用（tool use）已经让大语言模型变成了强大的代理（agent），能通过动态调用外部软件组件来执行复杂的多步骤任务。但一个根本性的限制是——这些工具必须由人类开发者提前实现。当前的LLM agent生态系统依赖于手工编写的工具库（如API封装、函数定义等），每个工具都需要人类理解原始代码库并编写接口。

**现有痛点**：在生命科学和医学等领域，存在大量高度专业化的计算工具，它们通常以学术论文+代码仓库的形式发布在GitHub上。这些工具数量庞大、不断更新，人工为每个工具编写LLM兼容接口的成本极高且难以规模化。现有的软件工程代理（如SWE-Agent）虽然能处理代码相关任务，但并非专门设计用于将复杂的科研代码仓库转化为可调用工具。

**核心矛盾**：科研论文附带的开源代码仓库是一座"工具金矿"，但这些代码通常文档不全、依赖复杂、接口不统一，直接被LLM调用几乎不可能。手动封装又不可规模化。

**本文目标**：构建一个全自动的代理框架，能将任意GitHub仓库转化为LLM可直接调用的标准化工具。

**切入角度**：作者将"仓库→工具"的过程分解为三个阶段——安装（install）、创建（create）、运行（run），每个阶段都使用LLM代理在Docker隔离环境中自主操作，通过闭环自修复机制处理错误。

**核心 idea**：让LLM代理自己制造代理工具——给定一个GitHub URL和简短任务描述，ToolMaker自动完成依赖安装、代码理解、接口代码生成和调试的全流程。

## 方法详解

### 整体框架

ToolMaker的工作流程分三个阶段：**Install**阶段负责将GitHub仓库克隆到Docker容器中并安装所有依赖；**Create**阶段由LLM代理阅读仓库代码、理解功能，生成符合标准化接口的Python调用代码；**Run**阶段在新的Docker容器中执行生成的工具代码处理具体任务。每个阶段都嵌入了闭环自修复机制——如果执行出错，LLM代理会读取错误信息、分析原因并自动修改代码重试。

### 关键设计

1. **三阶段工具制造流水线（Install → Create → Run）**:

    - 功能：将复杂的"代码仓库→可用工具"过程分解为可管理的子任务
    - 核心思路：**Install**阶段在Docker中执行`git clone`、`pip install`等命令，LLM代理根据仓库的README和配置文件推断正确的安装步骤。**Create**阶段是核心——LLM代理浏览仓库结构、阅读关键代码文件，然后生成一个`implementation.py`文件，该文件实现一个标准化的函数接口（接受输入文件路径和参数，返回输出）。**Run**阶段将生成的工具代码在干净的Docker容器中执行
    - 设计动机：将"安装环境"和"理解并封装代码"分开，避免环境问题和代码理解问题相互干扰。Docker隔离确保安全性和可复现性

2. **闭环自修复调试机制（Closed-loop Self-correction）**:

    - 功能：自动诊断和修复工具代码中的错误
    - 核心思路：在Create阶段，生成代码后立即在Docker中执行测试。如果执行失败，完整的错误信息（stderr/traceback）被反馈给LLM代理，代理分析错误原因并修改代码。这个"执行→检查错误→修复→再执行"的循环最多重复N次（可配置）。关键是每次循环中代理可以访问仓库源码来辅助调试
    - 设计动机：科研代码仓库的复杂性使得一次生成正确代码几乎不可能——可能存在API变更、未文档化的参数、隐含的依赖等。闭环调试模拟了人类开发者的"试错-修正"过程

3. **标准化工具接口设计**:

    - 功能：确保生成的工具可以被任何LLM agent直接调用
    - 核心思路：定义一个统一的Python函数接口模板，所有生成的工具都必须实现该接口（如`def run(input_path: str, output_path: str, **kwargs) -> dict`）。LLM代理在Create阶段的目标就是将仓库的功能封装进这个标准接口。工具的元数据（描述、参数说明）也自动生成
    - 设计动机：标准化接口是工具可组合性和可复用性的基础。没有统一接口，即使生成了工具代码，LLM agent也无法知道如何调用

### 损失函数 / 训练策略

ToolMaker不涉及模型训练，完全基于现有LLM（如GPT-4）的零样本推理能力。评估指标为任务正确实现率（通过单元测试验证）和鲁棒性（多个测试用例下的通过率）。

## 实验关键数据

### 主实验（TM-Bench基准）

| 方法 | 任务正确率 | 单元测试通过率 | 领域覆盖 |
|------|-----------|--------------|---------|
| **ToolMaker** | **80% (12/15)** | ~85% (100+测试) | 病理、放射、基因组等 |
| SWE-Agent | ~40% | ~45% | 同上 |
| OpenHands (CodeAct) | ~33% | ~38% | 同上 |
| 人类基线 | ~93% | ~95% | 同上 |

### 消融实验

| 配置 | 任务正确率 | 说明 |
|------|-----------|------|
| ToolMaker (完整) | 80% | 三阶段+闭环调试 |
| 无闭环调试 (单次生成) | ~47% | 去掉自修复后大幅下降 |
| 减少调试轮次 (max=2) | ~60% | 调试轮次减少，部分任务无法修复 |
| 无Docker隔离 | N/A (安全风险) | 未测试 |
| 使用GPT-3.5替代GPT-4 | ~53% | 模型能力不足导致代码质量下降 |

### 关键发现

- **闭环调试是核心贡献**：去掉自修复机制后，正确率从80%降到约47%，说明一次性生成正确的工具代码极其困难，迭代调试不可或缺
- **显著超越通用软件工程代理**：ToolMaker的80%正确率远超SWE-Agent（~40%）和OpenHands（~33%），说明专门设计的工具制造流程优于通用代码生成方法
- **失败案例分析**：3个失败任务主要因为仓库代码本身有bug或需要非常规的GPU环境配置，超出了LLM代理的能力范围
- **GPT-4 vs GPT-3.5差距显著**：工具制造需要强大的代码理解和生成能力，当前只有最强的LLM才能胜任

## 亮点与洞察

- **"LLM制造工具给LLM用"的元认知想法**：这是一个优雅的递归——让AI代理自己扩展自己的工具集。随着代码仓库数量的指数增长，这种自动化方式是唯一可规模化的方案。该思路可以推广到其他需要自动化接口封装的场景。
- **Docker隔离的安全设计**：在不可信代码上执行LLM生成的安装和运行命令存在显著安全风险。ToolMaker使用Docker容器隔离每个操作阶段，确保即使生成了恶意代码也不会影响宿主环境。这是agent安全实践的良好范例。
- **面向科学工作流的定位**：论文聚焦于科学计算领域的工具自动化，这是一个有实际需求的方向——科研人员经常需要运行别人的代码但花大量时间在环境配置和接口适配上。

## 局限与展望

- 基准规模较小（仅15个任务），虽然每个任务有多个单元测试，但领域和复杂度的覆盖仍有限
- 高度依赖GPT-4的能力，使用开源模型的效果待验证
- 仅处理Python仓库，对于R、MATLAB等科研常用语言的支持未涉及
- Docker环境的GPU支持和特殊硬件需求可能制约某些科研工具的自动安装
- 生成的工具代码缺乏正式验证，可能存在细微的语义错误（通过了单元测试但在边界条件上不正确）

## 相关工作与启发

- **vs SWE-Agent (Yang et al.)**: SWE-Agent面向一般的GitHub issue修复，而ToolMaker专注于将仓库封装为可调用工具。任务定义不同导致ToolMaker的分阶段设计更有针对性
- **vs LATM (Cai et al., "Large Language Models as Tool Makers")**: LATM让LLM生成Python函数作为工具，但从简单的自然语言描述出发。ToolMaker从复杂的代码仓库出发，挑战更大，也更贴近真实科研场景
- **vs AutoGen/MetaGPT等多代理框架**: 这些框架关注代理间的协作，ToolMaker关注单代理的工具制造能力。两者是互补的——ToolMaker生成的工具可以整合到多代理框架中使用

## 评分

- 新颖性: ⭐⭐⭐⭐ "代理给代理造工具"的概念新颖且有前瞻性，分阶段+闭环调试设计合理
- 实验充分度: ⭐⭐⭐ 基准规模较小，且对手主要是通用SE代理而非专门的工具生成方法
- 写作质量: ⭐⭐⭐⭐ 动机清晰，框架图示直观，实验分析到位
- 价值: ⭐⭐⭐⭐ 对自主科学工作流和agent生态有重要推动意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[ACL 2025\] R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory](r2d2_reflective_agentic_memory.md)
- [\[NeurIPS 2025\] Distilling LLM Agent into Small Models with Retrieval and Code Tools](../../NeurIPS2025/llm_agent/distilling_llm_agent_into_small_models_with_retrieval_and_co.md)
- [\[ACL 2026\] GOAT: A Training Framework for Goal-Oriented Agent with Tools](../../ACL2026/llm_agent/goat_a_training_framework_for_goal-oriented_agent_with_tools.md)
- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](../../ACL2026/llm_agent/mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)

</div>

<!-- RELATED:END -->
