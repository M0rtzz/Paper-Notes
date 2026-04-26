---
title: >-
  [论文解读] RISK: A Framework for GUI Agents in E-commerce Risk Management
description: >-
  [ACL 2026][LLM Agent][GUI智能体] 提出 RISK 框架，包含领域数据集（RISK-Data, 8492单步+2386多步轨迹）、基准（RISK-Bench）和基于GRPO的强化微调方法（RISK-R1），针对电商风控场景的GUI智能体，7B模型以仅7.2%的参数量超越SOTA基线，在线任务成功率达70.5%。
tags:
  - ACL 2026
  - LLM Agent
  - GUI智能体
  - 电商风控
  - 强化微调
  - 网页交互
  - 多步推理
---

# RISK: A Framework for GUI Agents in E-commerce Risk Management

**会议**: ACL 2026  
**arXiv**: [2509.21982](https://arxiv.org/abs/2509.21982)  
**代码**: [GitHub](https://github.com/RenqiChen/RISK-GUI)  
**领域**: GUI智能体  
**关键词**: GUI智能体, 电商风控, 强化微调, 网页交互, 多步推理

## 一句话总结

提出 RISK 框架，包含领域数据集（RISK-Data, 8492单步+2386多步轨迹）、基准（RISK-Bench）和基于GRPO的强化微调方法（RISK-R1），针对电商风控场景的GUI智能体，7B模型以仅7.2%的参数量超越SOTA基线，在线任务成功率达70.5%。

## 研究背景与动机

**领域现状**：电商风控需要从多个外部网站聚合异构信息（交易详情、用户档案、网站验证等），这些信息常嵌入在动态加载的子页面、交互元素或复杂DOM结构中，需要多步有状态的网页交互。

**现有痛点**：传统爬虫无法处理有状态的事件驱动交互；现有GUI Agent大多局限于单步操作，缺乏多步推理和动态内容处理能力；缺少电商风控领域的专用数据集和基准；且GUI模型训练时使用坐标定位，但部署时框架使用DOM索引+工具调用，存在训练-部署差距。

**核心矛盾**：通用GUI Agent在电商风控场景中表现不佳，因为它们缺乏领域知识、多步推理能力和处理复杂网页的经验。

**本文目标**：构建完整的电商风控GUI Agent框架——从数据收集到模型训练到实际部署。

**切入角度**：使用Browser Use框架收集高质量领域数据，结合GRPO强化微调实现从训练到部署的无缝过渡。

**核心idea**：通过四维度奖励设计（格式奖励 + 逐步准确性奖励 + 过程重加权 + 难度重加权）弥合GUI Agent训练与真实部署之间的差距。

## 方法详解

### 整体框架

RISK由三个组件构成：（1）RISK-Data——通过Qwen-VL-Max驱动Browser Use框架收集数据，经轨迹过滤、步骤清洗、信息精炼、数据增强、多步生成和难度分级6步精炼，得到8492单步+2386多步轨迹；（2）RISK-Bench——802单步+320多步轨迹，分easy/moderate/difficult三级；（3）RISK-R1——基于GRPO的强化微调框架，先SFT建立基础能力，再RFT精化。

### 关键设计

1. **框架驱动的奖励函数**：

    - 功能：弥合GUI模型训练与框架部署之间的差距
    - 核心思路：四个维度——（a）格式奖励：检查输出是否包含think/action/evaluation_previous_goal/memory/next_goal等结构，且action格式为DOM索引+工具形式而非坐标；（b）逐步准确性奖励：早期训练按工具列表中每个动作的F1>0.5分别计奖励，后期切换为整体二值奖励；（c）过程重加权：用sigmoid函数对轨迹中后期步骤赋予更高权重；（d）难度重加权：对difficult级别样本在优化目标中赋予更高权重
    - 设计动机：现有GUI-R1等方法的坐标定位奖励不适用于基于DOM索引的实际部署框架；单一二值奖励在训练早期对模型探索引导不足

2. **领域数据收集与精炼管线**：

    - 功能：构建高质量的电商风控专用数据
    - 核心思路：利用Browser Use框架+Qwen-VL-Max进行真实网页多轮交互收集原始数据，然后通过6步精炼流程（轨迹过滤→步骤清洗→信息精炼→数据增强→多步生成→难度分级）确保数据质量
    - 设计动机：通用GUI数据集缺乏电商风控领域特有的信息搜索和网站验证任务

3. **SFT→RFT两阶段训练**：

    - 功能：从基础能力建立到精细化提升
    - 核心思路：第一阶段用全部RISK-Data做SFT建立基本交互能力；第二阶段仅用单步轨迹做RFT（多步轨迹过长无法fit GPU），逐步准确性奖励从细粒度过渡到粗粒度
    - 设计动机：直接RFT不稳定，SFT先建立格式和基本能力的基础

## 实验关键数据

### 主实验

| 模型 | 单步Overall | 多步成功率 | OS-Genesis Web |
|------|-----------|----------|---------------|
| GPT-4o | 81.5 | 74.0 | 55.3 |
| Qwen2.5-VL-72B | 80.6 | 67.8 | 50.0 |
| RISK-R1-7B (ours) | **88.3** | **82.8** | **57.1** |
| GUI-R1-7B | 74.3 | 0.0 | 49.1 |
| UI-TARS-72B | 13.0 | 0.0 | 5.8 |

### 消融实验

| 配置 | 单步 | 多步 | 说明 |
|------|------|------|------|
| RISK-R1 完整 | 88.3 | 82.8 | 全部组件 |
| - 过程重加权 | 86.5 | 79.1 | 后期步骤权重均等 |
| - 逐步奖励 | 85.8 | 78.3 | 全程二值奖励 |
| - 难度重加权 | 87.1 | 80.5 | 样本权重均等 |
| 仅SFT | 83.2 | 74.7 | 无RFT |

### 关键发现
- 7B的RISK-R1超越72B的通用模型和GPT-4o，以仅7.2%的参数量达到SOTA
- 通用GUI SFT模型（UI-TARS）在领域任务上几乎完全失败，说明领域数据的必要性
- GUI-R1（坐标定位RFT）在多步任务上成功率为0，证实训练-部署差距的严重性
- 在线评估任务成功率70.5%，验证了方法的实际部署价值
- 过程重加权和逐步奖励对多步任务提升最为显著

## 亮点与洞察
- **完整的从数据到部署的框架**：RISK覆盖了数据收集、基准构建、模型训练和实际部署的完整链条
- **小模型超越大模型**：7B超越72B和GPT-4o，证明领域专注+正确训练策略的价值
- **训练-部署一致性**：奖励函数基于DOM索引+工具调用而非坐标，确保模型训练与框架部署的无缝衔接
- **有在线评估**：不仅有离线基准，还有真实环境在线评估，增加说服力

## 局限与展望
- **RFT仅用单步数据**：多步轨迹过长无法fit GPU，多步能力主要靠SFT和单步RFT迁移
- **领域局限**：仅覆盖电商风控，需验证框架在其他领域的适用性
- **依赖特定框架**：与Browser Use框架紧密耦合
- 未来方向：支持多步RFT训练、扩展到更多领域、集成更复杂的风控决策

## 相关工作与启发
- **vs GUI-R1**：使用坐标定位的通用GUI RFT，在DOM索引场景下完全失效；RISK-R1的框架驱动奖励解决了这一问题
- **vs UI-TARS**：通用GUI SFT模型，在领域任务上表现极差，凸显领域数据的重要性
- **vs Browser Use**：RISK利用其作为数据收集工具和部署框架，实现了训练-部署闭环

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架驱动的奖励设计和训练-部署一致性思路有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 离线+在线评估、多基线对比、详尽消融实验
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但部分细节需参考附录
- 价值: ⭐⭐⭐⭐ 为领域特定GUI Agent提供了可复制的完整解决方案

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] With Great Capabilities Come Great Responsibilities: Introducing the Agentic Risk & Capability Framework for Governing Agentic AI Systems](../../AAAI2026/llm_agent/with_great_capabilities_come_great_responsibilities_introducing_the_agentic_risk.md)
- [\[NeurIPS 2025\] Traj-CoA: Patient Trajectory Modeling via Chain-of-Agents for Lung Cancer Risk Prediction](../../NeurIPS2025/llm_agent/traj-coa_patient_trajectory_modeling_via_chain-of-agents_for_lung_cancer_risk_pr.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[ACL 2026\] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems](fedgui_benchmarking_federated_gui_agents_across_heterogeneous_platforms_devices_.md)
- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](../../CVPR2026/llm_agent/echotrail-gui_building_actionable_memory_for_gui_agents.md)

<!-- RELATED:END -->
