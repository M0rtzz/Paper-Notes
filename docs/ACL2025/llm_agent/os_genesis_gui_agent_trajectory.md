---
title: >-
  [论文解读] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis
description: >-
  [LLM Agent] 提出 OS-Genesis，一种交互驱动的 GUI Agent 轨迹合成 pipeline，通过先让 agent 在环境中探索交互再反向推导任务（Reverse Task Synthesis），结合轨迹奖励模型 (TRM) 过滤质量，生成高质量多样化的训练轨迹，在 AndroidWorld 上性能接近翻倍。
tags:
  - LLM Agent
---

# OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis

| 项目 | 内容 |
|------|------|
| 会议 | ACL 2025 |
| arXiv | [2412.19723](https://arxiv.org/abs/2412.19723) |
| 代码 | [OS-Genesis Homepage](https://os-genesis.github.io/) |
| 领域 | GUI Agent / 数据合成 |
| 关键词 | GUI Agent, Trajectory Synthesis, Reverse Task Synthesis, VLM, Reward Model |

## 一句话总结

提出 OS-Genesis，一种交互驱动的 GUI Agent 轨迹合成 pipeline，通过先让 agent 在环境中探索交互再反向推导任务（Reverse Task Synthesis），结合轨迹奖励模型 (TRM) 过滤质量，生成高质量多样化的训练轨迹，在 AndroidWorld 上性能接近翻倍。

## 研究背景与动机

**研究问题：** 基于 VLM 的 GUI Agent 需要高质量轨迹数据训练，但现有数据收集方法面临严峻瓶颈——人工标注昂贵且低效，基于预定义任务的合成方法受限于数据多样性和质量。

**现有方法的不足：** (1) 人工收集要求标注者标注完整轨迹并手动预定义高层任务，成本高且规模有限；(2) 任务驱动的模型合成方法严重依赖预定义的高层任务，限制了可扩展性和多样性；(3) 中间步骤的错误或任务目标不匹配导致合成轨迹不完整或语义不连贯。

**核心动机：** 模拟人类学习 GUI 交互的方式——先探索应用功能（交互驱动），再从已执行的操作中反向推导出有意义的任务。这种方式天然弥合了抽象指令与 GUI 动态特性之间的鸿沟。

## 方法详解

### 整体框架

OS-Genesis 包含三个核心阶段：(1) **交互驱动的功能发现** —— 在在线环境中无人工干预地遍历 UI 元素；(2) **反向任务合成 (Reverse Task Synthesis)** —— 从收集的交互三元组推导低层指令，再组合为高层任务；(3) **轨迹奖励模型 (TRM)** —— 对合成轨迹进行分级质量评估和加权采样训练。

### 关键设计

1. **交互驱动功能发现：** 在 Android 模拟器和 Chrome 浏览器中进行基于规则的 UI 元素遍历（CLICK、TYPE、SCROLL），仅在输入框交互时调用 GPT-4o 生成上下文合适的内容。收集大量 $\langle s_{pre}, a, s_{post} \rangle$ 三元组（交互前后截图 + 执行的动作）。

2. **反向任务合成：** 两级生成过程。(a) 低层：用 GPT-4o 从每个三元组推导原子操作描述 $\tau_{low}$（如"点击下拉菜单显示选项"）；(b) 高层：将低层任务关联到更广泛的用户意图 $\tau_{high}$（如"配置应用设置"）。然后用这些高层指令在环境中驱动 GPT-4o 执行，收集完整轨迹。

3. **轨迹奖励模型 (TRM)：** 不同于传统 labeler 函数二元判断（保留/丢弃），TRM 给每条轨迹打 1-5 的分级 reward，评估完成度 (Completion) 和连贯性 (Coherence)。训练时按 $P(g_i) = R_i / \sum_{k=1}^N R_k$ 概率采样轨迹，让不完整但部分有价值的轨迹也能贡献训练。

### 训练目标

两个互补的 SFT 目标：
- **规划训练 (Planning)：** $\mathcal{L}_1 = -\sum \log(p_\theta(\ell | s, h_i, c) \cdot p_\theta(a | s, h_i, c, \ell))$，同时预测低层指令和动作。
- **动作训练 (Action)：** $\mathcal{L}_2 = -\sum \log p_\theta(a | s, c, \ell)$，给定低层指令预测动作。

## 实验

### 主实验结果（AndroidWorld 成功率）

| 基础模型 | Zero-Shot | Task-Driven | Self-Instruct | **OS-Genesis** |
|---------|-----------|-------------|---------------|----------------|
| GPT-4o (M3A) | 23.70 | — | — | — |
| InternVL2-4B | 0.00 | 4.02 | 7.14 | **15.18** |
| InternVL2-8B | 2.23 | 4.46 | 5.36 | **16.96** |
| Qwen2-VL-7B | 0.89 | 6.25 | 9.82 | **17.41** |

### WebArena 成功率

| 基础模型 | Zero-Shot | Task-Driven | Self-Instruct | **OS-Genesis** |
|---------|-----------|-------------|---------------|----------------|
| InternVL2-4B | 0.00 | 4.98 | 5.81 | **7.88** |
| InternVL2-8B | 0.00 | 4.56 | 7.05 | **9.96** |
| Qwen2-VL-7B | 7.47 | 7.05 | 5.39 | **10.79** |

### 消融与分析

| 分析维度 | 发现 |
|---------|------|
| 数据多样性 | OS-Genesis 在指令和轨迹两个维度的余弦距离均最高，优于 human data 的轨迹多样性 |
| TRM vs Labeler | TRM 分级采样优于二元 labeler 过滤，尤其在高层规划任务上 |
| 数据规模 | 性能随数据量增加而提升，在约 1K 轨迹后开始饱和 |
| 与人工数据对比 | OS-Genesis 指令质量甚至优于人工编写的指令（因预定义任务可能与动态环境不匹配） |

### 关键发现

- OS-Genesis 在 AndroidWorld 上将 Qwen2-VL-7B 的成功率从 9.82% 提升至 17.41%，接近翻倍，且仅用 1K 轨迹（Self-Instruct 用了 1.5K）。
- 在 AndroidControl 的 OOD 测试中（833 个 app 中仅 20 个参与训练数据合成），OS-Genesis 仍展现出强泛化能力。
- 人工编写的指令虽然多样性高，但对应的轨迹多样性低——人类倾向于使用熟悉的操作路径。OS-Genesis 在两个维度均实现高多样性。

## 亮点

- 将 GUI 轨迹构建从"任务驱动"范式转变为"交互驱动"范式，显著提升数据的多样性和质量。
- 反向任务合成的设计直觉清晰——先探索再反推任务，天然适配 GUI 环境的动态特性。
- TRM 的分级评估避免了简单丢弃不完整轨迹导致的数据浪费。

## 局限性

- 探索阶段依赖 GPT-4o 进行输入框内容生成和反向任务合成，成本较高。
- 在 WebArena 上绝对性能仍远低于 GPT-4o Zero-Shot (16.25%)，表明合成数据在复杂 Web 任务上还有提升空间。
- 主要支持 CLICK、TYPE、SCROLL 三种动作，未覆盖拖拽、手势等更复杂的交互方式。
- 数据规模增大后出现饱和，受限于 VLM 本身能力和 GPT-4o 执行轨迹的质量。

## 相关工作

- **GUI Agent 数据：** AndroidControl (Li et al., 2024) 提供人工标注的移动端轨迹；AgentTrek (Lai et al., 2024) 等使用预定义任务驱动合成。
- **GUI Agent 系统：** M3A (Rawles et al., 2024) 基于 GPT-4o 的 Android agent；CogAgent (Hong et al., 2024) 基于 VLM 微调。
- **反向任务合成思想：** 与 Self-Instruct (Wang et al., 2023) 从 LLM 生成任务不同，OS-Genesis 从真实环境交互中反推任务，更贴近实际 GUI 功能。

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 实用性 | 8 |
| 实验充分度 | 9 |
| 写作质量 | 8 |
| 总体评分 | 8 |

<!-- RELATED:START -->

## 相关论文

- [OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use](os_agents_survey_mllm.md)
- [AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents](multiagentbench_evaluating_the_collaboration_and_competition_of_llm_agents.md)
- [HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](../../CVPR2026/llm_agent/hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)

<!-- RELATED:END -->
