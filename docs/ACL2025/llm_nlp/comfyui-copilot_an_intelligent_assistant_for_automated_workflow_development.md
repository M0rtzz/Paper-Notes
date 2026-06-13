---
title: >-
  [论文解读] ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development
description: >-
  [ACL 2025][LLM 其他][ComfyUI] 提出 ComfyUI-Copilot，一个基于 LLM 的层次化多 agent 框架，作为 ComfyUI 插件提供智能节点/模型推荐和一键式工作流构建，覆盖 7K 节点、62K 模型和 9K 工作流的知识库，在线服务 22 国的 19K 用户并处理了 85K+ 查询。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "ComfyUI"
  - "Multi-Agent Framework"
  - "Workflow Generation"
  - "LLM Agent"
  - "AIGC"
---

# ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development

**会议**: ACL 2025  
**arXiv**: [2506.05010](https://arxiv.org/abs/2506.05010)  
**代码**: [https://github.com/AIDC-AI/ComfyUI-Copilot](https://github.com/AIDC-AI/ComfyUI-Copilot)  
**领域**: 其他  
**关键词**: ComfyUI, Multi-Agent Framework, Workflow Generation, LLM Agent, AIGC

## 一句话总结

提出 ComfyUI-Copilot，一个基于 LLM 的层次化多 agent 框架，作为 ComfyUI 插件提供智能节点/模型推荐和一键式工作流构建，覆盖 7K 节点、62K 模型和 9K 工作流的知识库，在线服务 22 国的 19K 用户并处理了 85K+ 查询。

## 研究背景与动机

ComfyUI 是 AIGC 领域最流行的开源低代码 AI 工作流平台，拥有超过 400 万活跃用户和 12K+ 社区贡献的组件。用户通过拖拽组件构建多模态任务（文生图、换脸、视频编辑等）的工作流。然而，ComfyUI 面临几个使用障碍：

**新手入门难**：文档分散在论坛和 GitHub issues 中，缺乏统一的教程系统。

**节点/模型配置复杂**：选择合适的节点和模型需要大量专业知识，不同模型间存在兼容性依赖（如特定 LoRA 仅适配特定扩散模型）。

**工作流设计成本高**：即使是经验丰富的用户，调试和构建完善的工作流也需要大量时间。

**现有自动化方案不足**：已有的自动工作流构建研究存在不稳定性（生成不可解析的工作流）和范围限制（仅支持文生图任务）。

ComfyUI-Copilot 的目标是降低新手入门门槛，同时提升专家用户的效率。

## 方法详解

### 整体框架

采用层次化多 agent 架构：**中央 assistant agent**（基于 LLM）负责任务分发和响应整合，**专业 worker agent**（工作流/节点/模型三类）负责具体任务执行，底层由三套知识库提供支撑。使用 LangChain 实现，assistant agent 根据用户指令和短期记忆（对话历史）自主选择合适的 worker agent。

### 关键设计

1. **知识库构建（7K 节点 + 62K 模型 + 9K 工作流）**: 数据来源包括热门资源平台、GitHub 仓库和 ComfyUI 官网，过滤了 NSFW 内容。对于缺乏文档的节点，设计了自动文档生成流程：搭建 ComfyUI 沙箱环境 → 克隆 GitHub 仓库并安装依赖 → 导入节点提取元数据 → 代码分块+BGE-M3 嵌入检索 → LLM 结合元数据和代码生成文档 → 质量审核。对于社区内容中缺乏功能描述的工作流和模型，利用 GPT-4o 的多模态理解能力，结合文本、效果图和 JSON 文件补充使用说明。**知识库每周持续更新**，确保覆盖最新模块。核心动机：AIGC 领域更新极快，静态知识库会迅速过时。

2. **三阶段推荐流水线（粗到细）**: 

    - **Stage 1 - 意图扩展**：用 LLM/LMM 将模糊用户指令扩展为详细任务描述。例如识别到原始图像是人像后，扩展后的意图会强调维持主体一致性。
    - **Stage 2 - 混合检索**：用 OpenAI text-embedding-3-small 计算语义相似度 $\text{sim}_S$ + 词级重叠比例 $\text{sim}_L$，综合得分 $\text{sim}_O = 0.7 \times \text{sim}_S + 0.3 \times \text{sim}_L$，取 Top-30 候选。
    - **Stage 3 - 精排**：用 GTE-Rerank 模型从 30 个候选中选出 Top-3，再结合人气指标（点赞/下载/星标）排序。这种由粗到细的设计在效率和准确性间取得了良好平衡。

3. **工作流生成（检索 + 从头生成）**: 除了通过上述流水线检索已有工作流外，还探索利用代码 LLM 从头生成工作流。工作流支持三种格式互转：ComfyUI 流图 ↔ JSON ↔ 代码（Python 风格），采用代码作为主要表示（因其富含逻辑和语义信息且与 LLM 的代码生成能力天然兼容）。用检索到的节点和代码示例提示闭源 LLM 生成工作流，同时发现微调后的 Qwen2.5-Coder-7B 在通过率和节点选择上可达到接近 Claude-3.7-Sonnet 的水平。

4. **附加功能**: 

    - **提示词优化**：帮助用户将简单描述（如"a cat"）细化为生动的详细提示。
    - **参数搜索**：支持并行实验不同参数组合（如 cfg/denoise），批量生成图像进行对比。
    - **多语言支持**：支持波兰语等多语言查询和回答。

### 交互界面

作为 ComfyUI 左侧栏插件，一键启动聊天界面，支持多轮对话和底层 LLM 切换（DeepSeek-V3/GPT-4o）。可对画布中任意节点点击提问，支持一键加载推荐工作流和节点到画布。

## 实验关键数据

### 主实验（离线评估）

| 任务 | LLM Backend | Recall@3 |
|------|-------------|----------|
| 节点推荐 (104 条指令) | DeepSeek-V3 | 88.5% |
| 节点推荐 (104 条指令) | GPT-4o | 89.4% |
| 工作流召回 (130 条指令) | DeepSeek-V3 | 90.0% |
| 工作流召回 (130 条指令) | GPT-4o | 89.2% |

两种 LLM 后端均实现 88.5%+ 的 Top-3 召回率，证明框架的鲁棒性。

### 在线用户反馈

| 指标 | 数值 |
|------|------|
| 节点推荐接受率 | 65.4% |
| 工作流推荐接受率 | 85.9% |
| 总查询量 | 85K+ |
| 用户数 | 19K |
| 覆盖国家 | 22 |
| GitHub Stars | 1.6K+ |

工作流推荐接受率（85.9%）远高于节点推荐（65.4%），说明工作流推荐的实用价值更高。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 微调 Qwen2.5-Coder-7B vs Claude-3.7 | 通过率+节点选择接近 | 开源模型可替代闭源模型 |
| 错误案例分析 | 未召回的工作流仍满足用户意图 | 说明实际效果比数字更好 |

### 关键发现

1. 工作流推荐的接受率（85.9%）显著高于节点推荐（65.4%），反映了用户对"端到端解决方案"的更大需求。
2. 即使目标工作流未被精确召回，推荐的替代工作流通常仍能满足用户需求。
3. 微调后的 7B 开源代码模型可达到接近商业闭源 LLM 的工作流生成效果，降低了部署成本。

## 亮点与洞察

- **工程完成度极高的系统论文**：从知识库构建、自动文档生成、多 agent 架构到前端插件集成，覆盖了完整的产品链路，且已有真实用户数据验证。
- **自动节点文档生成**：通过沙箱环境+代码分析+LLM 理解的流水线自动补充缺失文档，是一种可复用的技术方案。
- **知识库持续更新机制**：每周更新的设计确保了对前沿模块的覆盖，解决了 AIGC 领域快速迭代的核心挑战。
- **代码作为工作流表示**：利用 LLM 的代码生成能力来生成图形化工作流，是一种巧妙的桥接策略。

## 局限与展望

- 工作流从头生成的通过率（论文提到"仍有显著改进空间"）是当前最大瓶颈，复杂工作流的正确生成仍具挑战性。
- 节点推荐的在线接受率（65.4%）还有较大提升空间，可能需要更好的上下文理解。
- 评估主要依赖召回率和用户接受率，缺少对生成工作流质量（可执行性、图像质量等）的系统评估。
- 知识库的覆盖范围虽大（7K节点/62K模型），但 ComfyUI 生态更新极快，长尾节点的覆盖仍可能不足。
- 未讨论 LLM 调用成本和实时性，对高频使用场景的可扩展性存疑。

## 相关工作与启发

- ComfyGen 等先前工作仅支持文生图任务，ComfyUI-Copilot 扩展到了条件多模态生成任务的全覆盖。
- 多 agent 架构的设计思路（central planner + specialized workers）是 LLM Agent 领域的通用范式，本文将其成功应用于 AIGC 工作流场景。
- 启发：对于"组件化 + 可视化"类工具（不仅限于 ComfyUI），LLM agent 辅助的"意图理解 → 组件检索 → 自动组装"流水线是一种通用解决方案。

## 评分

- **新颖性**: ⭐⭐⭐ — 技术框架较为标准（检索+LLM+multi-agent），但在 ComfyUI 场景的应用是首创。
- **实验充分度**: ⭐⭐⭐ — 有离线评估和在线反馈，但评估指标较为单一，缺少工作流可执行性等深度评估。
- **写作质量**: ⭐⭐⭐⭐ — 系统描述清晰，功能展示丰富，实用性强。
- **价值**: ⭐⭐⭐⭐⭐ — 1.6K stars + 19K 用户 + 85K 查询的真实部署数据说明了极高的实用价值，对 AIGC 社区的贡献远超论文本身。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Achieving Certification-by-Design Through Model-Driven Development](achieving_certification-by-design_through_model-driven_development.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[ACL 2025\] A Measure of the System Dependence of Automated Metrics](a_measure_of_the_system_dependence_of_automated_metrics.md)
- [\[ACL 2025\] ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](consim_measuring_concept-based_explanations_effectiveness_with_automated_simulat.md)
- [\[ACL 2025\] AceCoder: Acing Coder RL via Automated Test-Case Synthesis](acecoder_acing_coder_rl_via_automated.md)

</div>

<!-- RELATED:END -->
