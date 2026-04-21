---
title: >-
  [论文解读] RoboAgent: Chaining Basic Capabilities for Embodied Task Planning
description: >-
  [CVPR 2026][具身任务规划] 提出 RoboAgent，一种能力驱动的具身任务规划框架，用单个 VLM 同时实现调度器和 5 种基本能力（探索引导、物体定位、场景描述、动作解码、经验总结），通过三阶段训练（SFT + DAgger + 专家引导 RL）在 EB-ALFRED 和 ALFWorld 上达到 SOTA。
tags:
  - CVPR 2026
  - 具身任务规划
  - 能力链式调用
  - 视觉语言模型
  - 强化学习
  - 多阶段训练
---

# RoboAgent: Chaining Basic Capabilities for Embodied Task Planning

**会议**: CVPR 2026  
**arXiv**: [2604.07774](https://arxiv.org/abs/2604.07774)  
**代码**: https://github.com/woyut/RoboAgent_CVPR26  
**领域**: Agent / 具身智能  
**关键词**: 具身任务规划, 能力链式调用, 视觉语言模型, 强化学习, 多阶段训练

## 一句话总结

提出 RoboAgent，一种能力驱动的具身任务规划框架，用单个 VLM 同时实现调度器和 5 种基本能力（探索引导、物体定位、场景描述、动作解码、经验总结），通过三阶段训练（SFT + DAgger + 专家引导 RL）在 EB-ALFRED 和 ALFWorld 上达到 SOTA。

## 研究背景与动机

1. **领域现状**：具身任务规划（ETP）要求智能体根据视觉观察和语言指令，在环境中执行原子动作序列完成复杂任务。VLM 虽在多模态理解上表现出色，但在涉及多轮交互、长视野推理、扩展上下文分析的具身规划中表现有限。

2. **现有痛点**：(1) 直接用 CoT 推理产生的中间思考缺乏规范化格式和直接监督，难以保证推理的正确性和实用性；(2) 依赖闭源模型或外部工具的方法无法端到端训练；(3) 标准 RL 在稀疏奖励的探索场景中难以学习有效策略。

3. **核心矛盾**：复杂规划隐含多个中间过程（意图理解、常识推理、环境分析、动作建模、进度监控），但现有方法将其混为一体，难以对中间步骤施加精细监督。

4. **本文目标** 将复杂规划分解为一系列基本视觉-语言问题，使单个 VLM 能通过显式的能力调用实现可控、透明的推理过程。

5. **切入角度**：定义一组对具身场景关键的视觉-语言能力，由调度器决定何时调用什么能力，每个能力维护自己的上下文并产生中间推理结果或环境交互。

6. **核心 idea**：用一个 VLM 同时扮演调度器和多种能力角色，将"自由格式 CoT"替换为"结构化能力调用链"，配合利用模拟器内部信息的多阶段训练。

## 方法详解

### 整体框架

RoboAgent 由一个调度器（Scheduler）和 5 种能力（Capability）组成，全部用同一个 VLM（Qwen2.5-VL-3B）实现。调度器接收任务指令和历史上下文，生成 [(能力名, 查询)] 序列；每个能力接收查询和可选的观察图像，输出动作序列或文本反馈给调度器。整个过程无需外部工具。

### 关键设计

1. **5 种基本能力的设计**:

    - 功能：将复杂规划分解为具体的视觉-语言子问题
    - 核心思路：(1) **探索引导 (EG)**：根据常识推理预测目标物体最可能的探索方向；(2) **物体定位 (OG)**：开放词汇目标检测，判断目标是否在视野中；(3) **场景描述 (SD)**：生成目标物体状态的文本描述；(4) **动作解码 (AD)**：将导航/操控指令翻译为原子动作序列；(5) **经验总结 (ES)**：总结最近动作的执行结果，分析失败原因。其中 AD 产生动作但无文本反馈，其余 4 种产生文本反馈但不生成动作。
    - 设计动机：每种能力对应 VLM 本身擅长的基本视觉-语言任务（空间推理、检测、描述、指令跟随、总结），通过结构化调用充分发挥 VLM 的内在能力，同时支持对每个能力施加精细监督。

2. **三阶段训练流水线**:

    - 功能：逐步提升模型从基础能力到复杂推理的表现
    - 核心思路：**Stage 1 (SFT-Expert)**：在专家轨迹上进行监督微调，利用模拟器内部信息（场景图谱、分割掩码、环境消息）为每种能力构建训练标签，生成 640k 样本。**Stage 2 (DAgger-SFT)**：部署阶段 1 模型采集自生成轨迹，利用语义匹配将模型的能力调用与 ground-truth 对齐，构建纠正性训练标签（690k 样本），加入物体描述和动作格式增强。**Stage 3 (RFT)**：用专家引导策略优化算法 (EIPO) 对调度器进行强化微调，以能力调用完成操纵子计划为奖励，合成多样化交互数据（25k 轨迹）。
    - 设计动机：SFT 让模型掌握基本格式和技能，DAgger 修正分布偏移，RFT 提升调度器在多样场景下的泛化能力。

3. **专家引导策略优化 (EIPO)**:

    - 功能：更稳定的调度器强化学习训练
    - 核心思路：不同于 PPO/GRPO 优化策略回报改进量，EIPO 直接最大化专家的优势函数 $A_{\pi^*}(s,a)$。由于专家策略是确定性的，$A_{\pi^*}$ 可以精确计算而无需蒙特卡洛估计。使用 GRPO 风格的组基均值作为基线，将"组内相对更好"的动作赋予正梯度，"更差"的动作赋予负梯度。目标函数为 $J(\pi) = \mathbb{E}_{s \sim D} \frac{1}{G} \sum_{i=1}^{G} [r(a^i, s) \hat{A}_{\pi^*}(s, a^i)]$。
    - 设计动机：利用专家的最优性保证 $A_{\pi^*}(s,a) \leq 0$，所有次优动作天然被抑制。相比 GRPO 使用 episode 级别回报，EIPO 使用步级别优势函数，收敛更快更稳定。

### 损失函数 / 训练策略

Stage 1-2 使用标准交叉熵损失进行 SFT。Stage 3 使用 EIPO 算法，学习率 5e-6，batch size 512，120 次策略更新迭代。整体在 4 卡 H800 上训练。基模型 Qwen2.5-VL-3B，Stage 1 学习率 1e-5，batch size 32，2 epochs。

## 实验关键数据

### 主实验

| 基准 | 方法 | 基模型 | 平均 SR |
|------|------|--------|---------|
| EB-ALFRED | Claude-3.7-Sonnet (zero-shot) | - | 67.7 |
| EB-ALFRED | WAP | Qwen2.5-VL-7B | 62.7 |
| EB-ALFRED | **RoboAgent** | **Qwen2.5-VL-3B** | **67.0** |
| ALFWorld (视觉) | SEEA-R1 | Qwen2.5-VL-7B | 36.0 |
| ALFWorld (视觉) | **RoboAgent** | **Qwen2.5-VL-3B** | **77.6** |
| ALFWorld (文本) | DynaMind | Qwen2.5-7B | 92.5/89.1 |
| ALFWorld (文本) | **RoboAgent** | **Qwen2.5-VL-3B** | **92.1/94.0** |

在 ALFWorld 视觉环境中，RoboAgent 以 77.6% SR 大幅超越所有现有 RL 方法（次优 SEEA-R1 仅 36.0%），提升幅度达 41.6 个百分点。

### 消融实验

| 训练配置 | ALFWorld SR | EB-ALFRED SR | 说明 |
|---------|-------------|--------------|------|
| SFT-expert | 44.8 | 62.0 | 仅专家轨迹 SFT |
| +DAgger (aug. gen.) | 73.1 | 64.3 | 加模型生成数据的 DAgger |
| +RFT (aug. exp.) | 74.6 | 65.7 | 加增强专家数据 RFT |
| +RFT (aug. syn.) | **77.6** | **67.0** | 完整模型，加合成数据 RFT |

### 关键发现

- **DAgger 阶段贡献最大**：ALFWorld 上从 44.8→73.1（+28.3），说明模型自采集轨迹的纠正性监督对弥补分布偏移至关重要。
- **EIPO 比 GRPO 收敛更快**：在相同迭代次数下，EIPO 相比 GRPO 达到更高的 ALFWorld SR，验证了步级优势函数的稳定性优势。
- **跨模态泛化**：同一个视觉模型直接适配文本环境，达到 92.1/94.0 (seen/unseen)，接近专为文本设计的方法，说明能力框架获得了模态无关的规划能力。
- **OOD 泛化**：在 EB-Habitat (22.3) 和 LoTa-WAH (22.1) 上优于其他开源迁移模型，但与闭源 GPT-4o (59.0) 仍有差距。

## 亮点与洞察

- **"能力即接口"的设计理念极具启发性**：不同于自由格式 CoT，结构化的能力调用让中间推理可监督、可诊断、可替换，是 VLM agent 设计的一个重要范式。
- **利用模拟器特权信息构建训练数据**：训练时利用场景图谱、分割掩码等推理时不可见的信息为能力构建高质量标签，是一个巧妙的知识蒸馏策略。
- **单模型多角色**：调度器和所有能力共享一个 3B VLM，无需外部工具或多模型协作，大幅降低了部署复杂度。这一设计可迁移到其他多工具 agent 系统中。
- **EIPO 算法可推广**：利用专家策略的确定性来获得精确优势估计，适用于任何有可靠专家策略的 RL 场景。

## 局限与展望

- 仅在 AI2-THOR/ALFRED 模拟器上训练，真实世界泛化能力未验证
- 5 种能力是预定义的，无法动态扩展新能力或根据任务自适应选择
- OOD 结果显示跨模拟器泛化仍有明显差距（vs GPT-4o 零样本差 ~37 个百分点）
- 3B 模型在复杂推理场景下可能受限，更大模型可能进一步提升
- 可以考虑引入能力的自我发现和组合机制，而非固定 5 种

## 相关工作与启发

- **vs SEEA-R1**: SEEA-R1 用 7B 模型 + RL 达到 36.0% ALFWorld SR，RoboAgent 用 3B 模型达到 77.6%，说明结构化能力调用比自由 CoT 推理更有效
- **vs 闭源零样本 (Claude/GPT-4o)**: 在 EB-ALFRED 上 RoboAgent (67.0) 接近 Claude-3.7-Sonnet (67.7)，在 ALFWorld 视觉上大幅超越 GPT-4o (24.0)，说明领域微调的小模型可以匹敌大模型
- **vs 渐进式规划 (MPO/DynaMind)**: 这些方法通过子目标分解任务，RoboAgent 通过能力分解推理过程，后者提供了更精细的监督接口

## 评分

- 新颖性: ⭐⭐⭐⭐ 能力驱动规划框架设计新颖，EIPO 算法有理论贡献，但能力定义仍较手工
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个基准（2 in-domain + 2 OOD）、3 种模态（视觉/文本/OOD）、完整阶段消融、EIPO vs GRPO 对比
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，训练流水线图示直观，但公式推导部分较密集
- 价值: ⭐⭐⭐⭐⭐ 为 VLM 具身 agent 提供了一个可复制、可扩展的范式，77.6% ALFWorld SR 是目前最强结果

<!-- RELATED:START -->

## 相关论文

- [Embodied Navigation with Auxiliary Task of Action Description Prediction](../../ICCV2025/reinforcement_learning/embodied_navigation_with_auxiliary_task_of_action_description_prediction.md)
- [Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)
- [One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)
- [AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](../../ICLR2026/reinforcement_learning/autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches](../../NeurIPS2025/reinforcement_learning/exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)

<!-- RELATED:END -->
