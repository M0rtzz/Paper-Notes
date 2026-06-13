---
title: >-
  [论文解读] FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems
description: >-
  [ACL 2026 Findings][LLM Agent][联邦学习] FedGUI 是首个面向跨平台 GUI 代理的联邦学习综合基准，包含六个数据集覆盖移动端/网页端/桌面端，系统研究跨平台、跨设备、跨操作系统和跨数据源四种异构性对联邦 GUI 代理训练的影响。 领域现状：GUI 代理通过视觉语言模型（VLM）感知图形界…
tags:
  - "ACL 2026 Findings"
  - "LLM Agent"
  - "联邦学习"
  - "GUI代理"
  - "跨平台异构性"
  - "隐私保护"
  - "分布式训练"
---

# FedGUI: Benchmarking Federated GUI Agents across Heterogeneous Platforms, Devices, and Operating Systems

**会议**: ACL 2026 Findings  
**arXiv**: [2604.14956](https://arxiv.org/abs/2604.14956)  
**代码**: [GitHub](https://github.com/wwh0411/FedGUI)  
**领域**: Agent / GUI交互  
**关键词**: 联邦学习, GUI代理, 跨平台异构性, 隐私保护, 分布式训练

## 一句话总结
FedGUI 是首个面向跨平台 GUI 代理的联邦学习综合基准，包含六个数据集覆盖移动端/网页端/桌面端，系统研究跨平台、跨设备、跨操作系统和跨数据源四种异构性对联邦 GUI 代理训练的影响。

## 研究背景与动机

**领域现状**：GUI 代理通过视觉语言模型（VLM）感知图形界面并执行用户指令。传统方法依赖集中式数据收集和标注，成本高且不可扩展。联邦学习提供了隐私保护的分布式训练范式。

**现有痛点**：（1）现有联邦 GUI 基准（FedMABench）仅限于 Android 用户间的协作，忽略了网页和桌面用户的贡献潜力；（2）真实世界中 GUI 数据分布在不同平台（移动/网页/桌面）、设备（不同手机型号）、操作系统（Android/macOS/Windows/Ubuntu）中，这些异构性对联邦训练的影响未被研究。

**核心矛盾**：GUI 设备自然产生丰富的监督信号，但因隐私问题无法共享——联邦学习可以解决这一问题，但缺乏捕捉真实世界跨平台异构性的基准来指导算法选择。

**本文目标**：构建覆盖多平台、多设备、多OS的联邦 GUI 代理基准，回答两个关键问题：跨平台协作是否能提升性能？如何量化和应对不同维度的异构性？

**切入角度**：从九个数据源构建六个数据集，分别对应四种异构性维度，结合七种联邦学习算法和 20+ 基础模型进行系统评估。

**核心 idea**：四维异构性建模（Platform × Device × OS × Source）+ 统一动作空间 + 系统化联邦学习评估。

## 方法详解

### 整体框架
FedGUI 把"跨平台 GUI 代理能否靠联邦学习协作训练"做成一个可量化的基准：它遵循标准联邦学习协议，中央服务器协调一批异构客户端，每个客户端只在本地的 GUI 交互数据上训练、再把更新聚合成全局 VLM 模型。为了让外观迥异的移动/网页/桌面界面能在同一模型里聚合，FedGUI 提供一套统一动作空间，并据此从八个数据源切出六个数据集，分别隔离平台、设备、操作系统、数据源四种异构性来做系统评估。

### 关键设计

**1. 四维异构性数据集构建：把"哪种异构最难"拆开来量**

真实世界的 GUI 数据同时混杂着平台（移动/网页/桌面）、设备（不同手机型号）、操作系统（Android/macOS/Windows/Ubuntu）和数据源四个层面的差异，混在一起根本说不清是谁拖累了联邦训练，所以 FedGUI 从八个数据源切出六个互相隔离的数据集——FedGUI-Platform（15 个客户端）、FedGUI-Device（5 种 Android 设备）、FedGUI-OS、FedGUI-Web、FedGUI-Mobile，以及联合跨平台与跨数据源的 FedGUI-Full。

每个数据集只放大一种异构维度，这样平台级异构与设备级异构的影响就能被分别测量、互不掩盖，最终得到"平台级 > 操作系统 > 设备/数据源"的难度排序结论也才有说服力。

**2. 统一动作空间：让外观完全不同的界面在动作层对齐**

不同平台的 GUI 长得毫不相干，模型参数无法直接聚合，FedGUI 的解法是退到动作层面找公约数：识别出 CLICK、TYPE 等六种跨平台共享的基本动作，再把各平台特有的交互映射到两个独立的动作域（在系统提示里定义）。

这样即使移动端和桌面端的界面在像素层面毫无共同点，联邦聚合也能在"动作类型"这一层保持一致；否则跨平台的本地模型参数根本没有可对齐的语义，聚合出来的全局模型就只是噪声。

**3. 系统化联邦算法评估：给跨平台部署一份算法选型实证表**

光有数据集还不够，到底该用哪种联邦算法才是实践者最关心的问题，FedGUI 把 FedAvg、FedProx、FedYogi 等七种代表性算法在全部六个数据集和各异构设置上跑了一遍横评，评测指标覆盖动作类型准确率、Grounding 精度和成功率。

横评的价值在于揭示"最优算法随异构维度变化"——结果显示 FedYogi 这类自适应学习率算法在跨平台场景下最稳健，作者推测是自适应聚合更能消化不同平台之间的梯度分布差异。

### 损失函数 / 训练策略
标准联邦学习设置：本地训练用交叉熵损失，全局侧按各联邦算法自己的聚合策略合并更新；并支持 LoRA 微调以压低通信与计算成本。

## 实验关键数据

### 主实验

| 发现 | 说明 |
|------|------|
| 跨平台协作有益 | 增加参与用户（即使来自不同平台）提升模型性能 |
| 平台级异构影响最大 | 跨平台异构比平台内异构（设备/OS/数据源）更具挑战性 |
| 自适应算法最优 | FedYogi 等自适应算法在跨平台设置下表现最稳健 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅移动端 vs 全平台联邦 | 全平台更优 | 跨平台数据多样性有正面贡献 |
| IID vs Non-IID 设备分布 | Non-IID 下降 | 设备异构导致数据偏斜 |
| 不同基础模型 | 较大模型收益更高 | VLM 规模影响联邦学习效果 |

### 关键发现
- 即使来自高度异构的平台和设备，增加联邦参与者仍能提升全局模型性能——这为大规模分布式 GUI 代理训练提供了信心
- 平台级异构是最大的性能挑战，其次是操作系统，设备和数据源的影响相对较小
- FedYogi 等自适应学习率算法在跨平台场景中特别有效，可能因为自适应聚合能更好地处理不同平台的梯度分布差异

## 亮点与洞察
- **四维异构性分解**是一个有系统性的实验设计——使得每种异构性的影响可以独立分析
- 跨平台协作有益的发现有实际部署价值——意味着可以利用多种设备类型的用户数据来训练更好的统一 GUI 代理
- 统一动作空间是实现跨平台联邦学习的关键工程贡献

## 局限与展望
- 仅评估了 LoRA 微调，全参数联邦学习可能有不同的异构性动态
- 数据隐私保护仅通过联邦学习的基本框架提供，未引入差分隐私等额外保护
- 评估主要基于离线数据，缺少在真实用户交互中的在线评估
- 统一动作空间可能丢失平台特定的细粒度交互

## 相关工作与启发
- **vs FedMABench**: 仅限移动端 Android，FedGUI 扩展到移动+网页+桌面三平台
- **vs 集中式跨平台代理 (ShowUI, UI-TARS)**: 依赖集中数据收集，FedGUI 展示了分布式替代方案
- **vs 单平台 GUI 基准**: 单平台方法的泛化性差，联邦跨平台训练是更可扩展的路径

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个跨平台联邦 GUI 基准，四维异构性分析系统化
- 实验充分度: ⭐⭐⭐⭐⭐ 六个数据集、七种算法、20+ 基础模型
- 写作质量: ⭐⭐⭐⭐ 数据集构建描述清晰，实验设计系统

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ProBench: Benchmarking GUI Agents with Accurate Process Information](../../AAAI2026/llm_agent/probench_benchmarking_gui_agents_with_accurate_process_infor.md)
- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](../../ICML2026/llm_agent/scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)
- [\[ICML 2026\] Recovering Policy-Induced Errors: Benchmarking and Trajectory Synthesis for Robust GUI Agents](../../ICML2026/llm_agent/recovering_policy-induced_errors_benchmarking_and_trajectory_synthesis_for_robus.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)

</div>

<!-- RELATED:END -->
