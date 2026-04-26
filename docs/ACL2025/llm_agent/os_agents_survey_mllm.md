---
title: >-
  [论文解读] OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use
description: >-
  [LLM Agent] 系统综述了基于多模态大语言模型（MLLM）的操作系统智能体（OS Agents），从基本概念（环境/观测/动作空间）、核心能力（理解/规划/定位）、构建方法（基础模型+智能体框架）到评估基准全面梳理，揭示了该领域从虚拟助手到通用计算设备自动化的演进路径。
tags:
  - LLM Agent
---

# OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use

- **会议**: ACL 2025
- **arXiv**: [2508.04482](https://arxiv.org/abs/2508.04482)
- **代码**: [GitHub](https://os-agent-survey.github.io/)
- **领域**: LLM Agent / 操作系统智能体 / 综述
- **关键词**: OS Agents, GUI Automation, MLLM, Foundation Model, Agent Framework

## 一句话总结

系统综述了基于多模态大语言模型（MLLM）的操作系统智能体（OS Agents），从基本概念（环境/观测/动作空间）、核心能力（理解/规划/定位）、构建方法（基础模型+智能体框架）到评估基准全面梳理，揭示了该领域从虚拟助手到通用计算设备自动化的演进路径。

## 研究背景与动机

- **现有痛点**：传统虚拟助手（Siri、Cortana、Alexa）虽提供了有限的任务自动化能力，但受限于上下文理解不足、操作范围窄、无法跨应用协作等问题，远未实现"通用计算设备自动化"的愿景。同时 OS Agent 研究快速发展（Computer Use、Apple Intelligence、AutoGLM 等相继发布），但缺乏系统性文献梳理。

- **核心矛盾**：MLLM 的理解和推理能力使 OS Agent 技术上变得可行，但实际 GUI 环境极其复杂——高分辨率界面充斥微小图标和密集文本、多步骤任务需要动态规划和错误恢复、动作空间巨大且跨平台不一致。将 MLLM 的通用能力转化为可靠 GUI 操作仍面临理解、规划和定位三重挑战。

- **本文要解决**：(1) 定义 OS Agent 的基本组件和核心能力；(2) 梳理构建 OS Agent 的两条路径：领域基础模型开发 vs 智能体框架构建；(3) 总结评估方法和基准；(4) 讨论安全隐私和个性化自进化等未来方向。

- **切入角度**：作者将 OS Agent 视为操作系统环境中的特殊 AI Agent，从"环境-感知-行动"的经典 Agent 框架出发，统一不同平台（桌面/移动/Web）的研究，提供全景式的领域概览。

## 方法详解

### 整体框架

综述结构为：基础概念（§2）→ 构建方法（§3）→ 评估（§4）→ 挑战与展望（§5）。ACL 2025 接收的 9 页精简版，覆盖核心内容。

### 关键设计

1. **OS Agent 基本组件定义（§2.1）**：

    - 功能：建立统一的概念框架
    - 核心思路：**环境**——桌面/移动/Web 三类平台，各有独特的任务复杂度和交互模式；**观测空间**——屏幕截图（视觉）、HTML/A11y 树（文本）、或多模态融合，高分辨率 GUI 中小文字和密集元素构成特殊挑战；**动作空间**——输入操作（点击/输入/滑动）、导航操作（页面切换/滚动）、扩展操作（外部工具调用）
    - 设计动机：不同平台和任务类型的研究各自为政，需要统一概念框架才能有效对比

2. **基础模型构建路线（§3.1）**：

    - 功能：梳理领域基础模型的架构和训练策略选择
    - 核心思路：架构可选 (a) 直接使用 LLM（处理 HTML）、(b) 使用现有 MLLM（LLaVA/Qwen-VL/InternVL）、(c) 拼接 LLM+视觉编码器、(d) 修改 MLLM（如 CogAgent 加入 1120×1120 高分辨率编码器、Ferret-UI 的 any-resolution 策略）。训练策略覆盖 PT（GUI 定位/屏幕理解/OCR）→ SFT（轨迹+指令合成）→ RL（AutoGLM 自进化在线课程 RL、PPO），25+ 近期模型大多采用"现有 MLLM + SFT"路线
    - 设计动机：明确技术选择空间，帮助研究者定位自己的工作

3. **智能体框架（§3.2）**：

    - 功能：描述基于非微调策略构建的 Agent 系统设计
    - 核心思路：感知模块（文本描述 TD / GUI 截图 GS / 语义定位 SG / 双重定位 DG）、规划模块（全局一次性分解 vs 迭代逐步调整）、记忆模块（自动探索 AE / 经验增强 EA / 记忆管理 MA）、动作模块（输入/导航/扩展操作组合）。20+ 近期框架（如 Agent S、OS-Copilot、Cradle），大多采用迭代规划+自动探索
    - 设计动机：非微调方法提供灵活性和效率，使 Agent 无需训练即可跨任务泛化

### 评估体系

- **协议**：离线（预记录轨迹评估）vs 在线（实时交互）；步骤级指标（动作准确率）vs 任务级指标（完成率）
- **基准**：Web（MiniWob++/WebArena/Mind2Web）、移动（AndroidWorld/AITZ）、桌面（OSWorld/WindowsAgentArena）、跨平台（CRAB/AgentStudio）

## 实验关键数据

### 基础模型训练策略统计

| 训练路线 | 代表模型数 | 代表模型 |
|---------|-----------|---------|
| 仅 SFT | ~15 | EDGE, Ferret-UI 2, UGround |
| PT + SFT | ~8 | OS-Atlas, ShowUI, SeeClick |
| SFT + RL | ~4 | AutoWebGLM, WebAI |
| PT + SFT + RL | 1 | AutoGLM（最完整路线） |

### 主流基准覆盖

| 平台 | 代表基准 | 任务规模 | 关键指标 |
|------|---------|---------|---------|
| Web | WebArena | 812 任务 | 任务成功率 |
| Web | Mind2Web | 2K+ 任务 | 步骤准确率 |
| 移动 | AndroidWorld | 116 任务 | 任务完成率 |
| 桌面 | OSWorld | 369 任务 | 任务成功率 |
| 跨平台 | AgentStudio | 200+ 任务 | 多维指标 |

### 关键发现

- **现有模型能力有限**：OSWorld 上最好模型成功率仅约 12-15%，远未达实用水平
- **视觉理解仍是瓶颈**：高分辨率 GUI 中小元素定位（grounding）是最核心技术挑战
- **SFT 是主流路线**：绝大多数基础模型采用"现有 MLLM + 领域 SFT"，RL 仍较少使用
- **跨平台泛化不足**：仅在 Web 数据预训练的模型难以泛化到移动和桌面
- **架构聚焦高分辨率**：CogAgent（1120²）、Ferret-UI（any-resolution）等表明标准 224² 视觉编码器不足

## 亮点与洞察

- **分类体系清晰**：将 OS Agent 研究统一到"环境-观测-动作"+ "理解-规划-定位"双层框架中，不同平台和方法的关系一目了然
- **及时性强**：涵盖至 2024 年 10 月最新工作（Computer Use、Apple Intelligence 等），为快速发展的领域提供了有价值的快照
- **Table 1+2 实用参考**：分别总结基础模型和框架的设计选择，为后续研究者提供清晰技术路线参考

## 局限性

- 9 页精简版，部分内容压缩较多（如评估部分）
- 对各模型的定量性能比较较少，主要是分类式描述
- 安全隐私和个性化讨论偏概念性，缺乏具体技术方案
- 截至 2024 年 10 月，部分 2025 年初更新未覆盖
- 对真实用户场景的可靠性和用户信任讨论不够深入

## 相关工作

- **vs GUI Agent 综述 (Zhang et al., 2024b)**：前者聚焦 LLM 规划能力，本文更全面覆盖基础模型训练（含 PT/SFT/RL）和框架设计
- **vs WebAgent (Gur et al., 2023)**：早期代表性工作（Flan-U-PaLM + HTML-T5），本文梳理之后大量改进
- **vs OS-Atlas (Wu et al., 2024a)**：代表性基础模型，跨平台 GUI 定位数据合成解决域泛化，本综述重点讨论

## 评分

- **新颖性**: 6/10 — 综述工作，分类体系有贡献但无新方法
- **技术深度**: 7/10 — 对构建方法四条路线分析较深入
- **实验充分度**: 5/10 — 综述无实验，定量对比较少
- **清晰度**: 8/10 — 结构清晰，图表信息密度高
- **总分**: 7/10

<!-- RELATED:START -->

## 相关论文

- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[CVPR 2025\] ATA: Adaptive Transformation Agent for Text-Guided Subject-Position Variable Background Generation](../../CVPR2025/llm_agent/ata_adaptive_transformation_agent_for_text-guided_subject-position_variable_back.md)
- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)

<!-- RELATED:END -->
