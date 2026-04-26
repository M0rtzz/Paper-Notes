---
title: >-
  [论文解读] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents
description: >-
  [ACL 2026][LLM Agent][企业隐私] 基于上下文完整性（Contextual Integrity）理论构建企业场景基准 CI-Work，揭示前沿 LLM 智能体在企业工作流中普遍存在隐私泄漏问题，且模型规模扩大反而加剧泄漏。
tags:
  - ACL 2026
  - LLM Agent
  - 企业隐私
  - 上下文完整性
  - LLM智能体
  - 信息泄漏
  - 隐私-效用权衡
---

# CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents

**会议**: ACL 2026  
**arXiv**: [2604.21308](https://arxiv.org/abs/2604.21308)  
**代码**: [https://aka.ms/ci-work](https://aka.ms/ci-work)  
**领域**: LLM Agent  
**关键词**: 企业隐私、上下文完整性、LLM智能体、信息泄漏、隐私-效用权衡

## 一句话总结

基于上下文完整性（Contextual Integrity）理论构建企业场景基准 CI-Work，揭示前沿 LLM 智能体在企业工作流中普遍存在隐私泄漏问题，且模型规模扩大反而加剧泄漏。

## 研究背景与动机

**领域现状**：LLM 智能体正被集成到企业工作流中，能够访问邮件、会议记录等内部数据来执行复杂任务，显著提升生产力。

**现有痛点**：现有隐私基准（ConfAide、PrivacyLens、CIMemories 等）主要聚焦日常生活助手场景，未能捕捉企业环境中的复杂性：（1）仅评估单一信息流，忽略企业中多个信息流并行交织的特点；（2）评估上下文简单、孤立，无法衡量在密集检索场景中区分"必要信息"与"敏感信息"的能力；（3）依赖简化上下文或短属性，无法复现企业数据的规模和密度。

**核心矛盾**：企业 LLM 智能体的核心能力（检索和使用内部数据）恰恰使其成为敏感信息泄漏的潜在载体——更高的任务效用往往伴随更多隐私违规。

**本文目标**：构建一个基于上下文完整性理论的企业级基准，系统评估 LLM 智能体在高保真企业工作流中的隐私-效用权衡。

**切入角度**：将企业信息流按组织通信分类（向上、向下、横向、对角、外部五个方向），每个实例包含"必要集"和"敏感集"的密集检索上下文。

**核心 idea**：企业隐私不是简单的信息屏蔽，而是需要在密集检索场景中精准区分必要和敏感信息，这在当前模型中尚未解决，且增大模型规模反而会加剧问题。

## 方法详解

### 整体框架

CI-Work 采用四阶段构建流程：（1）任务导向种子生成 → （2）上下文条目生成 → （3）案例剧集生成 → （4）轨迹模拟与评估。基于 ToolEmu 和 PrivacyLens 构建工具中心的模拟环境，使用 LLM 模拟企业工具（邮件、聊天、日历、会议等）。

### 关键设计

1. **五方向信息流分类**：

    - 功能：覆盖企业组织中所有典型信息流动方向
    - 核心思路：基于标准组织通信分类学，将信息流分为向下（管理→员工）、向上（汇报上级）、横向（同级协作）、对角（跨部门）、外部（利益相关者）五类
    - 设计动机：捕捉真实企业中不同角色之间的隐私规范差异

2. **自迭代精炼（Self-Iterative Refinement）**：

    - 功能：自动修正生成的上下文条目，确保必要/敏感标签一致性
    - 核心思路：生成条目后让 LLM 盲分类为 Essential/Sensitive/Ambiguous，若分类与预期不一致则触发自动修订循环
    - 设计动机：弥合 LLM 在生成与评估之间的质量差距，人工验证达到 82.5%–95.0% 的一致率

3. **双集合评估指标体系**：

    - 功能：同时量化隐私保护和任务效用
    - 核心思路：将检索条目划分为必要集 $\mathcal{E}_{\text{ess}}$ 和敏感集 $\mathcal{E}_{\text{sens}}$，设计 Leakage（泄漏率）、Violation（违规率）、Conveyance（传达率）三个指标
    - 设计动机：显式量化隐私-效用权衡，超越单一成功率评估

### 损失函数 / 训练策略

无模型训练，采用 ReAct 框架部署 LLM 智能体，使用 GPT-5.2 进行基准生成和评估，LLM-as-a-Judge 实现自动评估（与人工标注 83.0%–91.0% 一致）。

## 实验关键数据

### 主实验

| 模型 | 泄漏率(LR)↓ | 违规率(VR)↓ | 传达率(CR)↑ |
|------|------------|------------|------------|
| DeepSeek-R1 | 6.08% | 15.80% | 53.08% |
| DeepSeek-V3 | 8.37% | 21.33% | 76.92% |
| GPT-4o | 8.79% | 21.33% | 87.35% |
| GPT-5 | 11.21% | 27.83% | 93.04% |
| Grok-3 | 26.66% | 50.87% | 94.97% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 条目数量 1→12 | VR 单调上升，CR 大幅下降 | 多实体上下文引入干扰 |
| 条目长度增加 | LR/VR 均上升，CR 也上升 | 更丰富的细节扩大泄漏面 |
| 隐式用户压力 | VR 显著提升 | 强调任务完成的指令加剧泄漏 |
| 显式用户压力 | VR 接近翻倍 | 主动提供数据源使泄漏更严重 |

### 关键发现
- **隐私-效用权衡**：更高传达率与更高泄漏/违规率正相关（Pearson $r=0.40$，$p<0.05$）
- **逆向扩展现象**：更大模型（GPT-4.1 系列）反而加剧隐私泄漏，因为更强的指令遵循能力导致更倾向于响应用户需求而非遵守隐式隐私规范
- **向上交互风险更高**：向上汇报的泄漏率和违规率显著高于向下传达（VR: $p=0.006$）
- **CI-CoT 防御有效但不充分**：显著降低泄漏率但仍保持 >20% 违规率

## 亮点与洞察
- 首次系统评估企业场景下 LLM 智能体的上下文完整性，发现隐私违规率高达 15.8%–50.9%
- 揭示了反直觉的"逆向扩展"：模型越大隐私越差，因为大模型更好地理解并执行用户意图但忽略隐式社会约束
- 发现用户行为（即使无恶意的隐式压力）会导致隐私和效用的双重崩溃
- 明确指出仅靠模型规模和推理深度无法解决企业隐私问题，需要范式转变

## 局限与展望
- 基准基于合成数据和模拟环境，与真实企业部署可能存在差距
- 隐私规范因组织、司法管辖和文化而异，基准无法覆盖所有场景
- 未来方向：角色条件过滤、上下文感知的训练时对齐、从模型中心到上下文中心的架构转变

## 相关工作与启发
- **vs ConfAide/PrivacyLens**：这些基准聚焦日常助手场景和单一信息流，CI-Work 聚焦企业多实体密集检索场景
- **vs CIMemories**：关注记忆中的隐私积累，CI-Work 关注单次企业任务中的信息泄漏
- **vs WorkArena/TheAgentCompany**：这些基准关注任务执行能力，CI-Work 首次同时评估效用和隐私

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个企业级 CI 基准，揭示逆向扩展等反直觉现象
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 9 个前沿模型、5 个信息流方向、多维度分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分，图表信息丰富
- 价值: ⭐⭐⭐⭐ 对企业 AI 部署安全具有重要警示和指导意义

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)
- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](../../AAAI2026/llm_agent/coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)

<!-- RELATED:END -->
