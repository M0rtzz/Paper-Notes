---
title: >-
  [论文解读] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies
description: >-
  [ICLR 2026][多智能体系统] 提出Multi-Agent System Search（MASS）框架，通过交错优化提示词和拓扑结构的三阶段策略（局部提示优化→拓扑搜索→全局提示优化），自动发现高性能的多智能体系统设计。
tags:
  - ICLR 2026
  - 多智能体系统
  - 提示优化
  - 拓扑搜索
  - LLM Agent
  - 自动化设计
---

# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

**会议**: ICLR 2026  
**arXiv**: [2502.02533](https://arxiv.org/abs/2502.02533)  
**代码**: 无  
**领域**: signal_comm  
**关键词**: 多智能体系统, 提示优化, 拓扑搜索, LLM Agent, 自动化设计

## 一句话总结

提出Multi-Agent System Search（MASS）框架，通过交错优化提示词和拓扑结构的三阶段策略（局部提示优化→拓扑搜索→全局提示优化），自动发现高性能的多智能体系统设计。

## 研究背景与动机

1. **领域现状**: 基于LLM的多智能体系统（MAS）通过多个Agent的交互与协作，在代码生成、推理、问答等复杂任务上优于单Agent系统。

2. **现有痛点**: 设计有效的MAS需要同时考虑每个Agent的提示词设计和Agent之间的拓扑编排，两者组合成庞大的搜索空间。现有自动化方法（如ADAS、AFlow）要么只优化拓扑而忽略提示词，要么搜索空间设计粗糙。

3. **核心矛盾**: 提示词和拓扑是MAS设计中两个关键因素，但它们的交互关系复杂——提示词敏感性在级联Agent中会放大，而非所有拓扑都对性能有正面影响。联合优化的组合复杂度过高。

4. **本文目标**: 系统分析MAS设计空间中各因素的影响，并提出高效的自动化优化框架。

5. **切入角度**: 先通过实证分析揭示提示优化比简单扩展Agent数量更具token效率，且有益拓扑只是搜索空间的一小部分。基于此剪枝搜索空间，并交错优化提示和拓扑。

6. **核心 idea**: 从局部到全局、从提示到拓扑的交错优化策略，可以高效征服MAS设计的组合复杂度。

## 方法详解

### 整体框架

MASS是三阶段优化框架：**Stage 1**（1PO）对每个拓扑构建块独立进行局部提示优化；**Stage 2**（2TO）在剪枝后的搜索空间中进行拓扑优化，使用Stage 1的优化提示；**Stage 3**（3PO）对最佳拓扑进行全局联合提示优化。搜索空间包含Aggregate、Reflect、Debate、Summarize和Tool-use五类构建块。

### 关键设计

**1. 基于影响力的搜索空间剪枝**

- **功能**: 将拓扑搜索空间限制在有正面影响的子集中，降低搜索复杂度
- **核心思路**: 计算每个构建块的增量影响力 $I_{a_i} = \mathcal{E}(a_i^*) / \mathcal{E}(a_0^*)$，通过Softmax转化为选择概率 $p_a = \text{Softmax}(I_a, t)$，以拒绝采样方式剪枝搜索空间：若 $u > p_{a_i}$（$u \sim \text{Uniform}(0,1)$）则拒绝该维度
- **设计动机**: 实验表明并非所有拓扑都有正面影响（如HotpotQA上只有debate有增益），在全空间搜索会引入负面构建块，降低性能

**2. 交错三阶段优化策略**

- **功能**: 解耦MAS联合优化的组合复杂度，同时保证提示与拓扑的协同优化
- **核心思路**: Stage 1先warm-up单Agent提示 $a_0^* \leftarrow \mathcal{O}_\mathcal{D}(a_0)$，再以最小配置优化每个构建块的提示 $a_i^* \leftarrow \mathcal{O}_\mathcal{D}(a_i | a_0^*)$。Stage 2在剪枝空间中随机采样拓扑配置并评估。Stage 3对最佳拓扑全局联合优化提示
- **设计动机**: 直接在MAS上做APO（自动提示优化）因Agent间依赖和稀疏奖励而不可行；先局部后全局的策略将复杂度分解为可管理的子问题

**3. 即插即用的提示优化器集成**

- **功能**: 与任意提示优化器兼容
- **核心思路**: 使用MIPRO作为默认优化器，支持指令和少样本示例的联合优化。Bootstrap 3个示例、10个指令候选、10轮优化
- **设计动机**: 提示优化器的选择不应限制框架的适用性，即插即用设计保证了灵活性

### 损失函数 / 训练策略

优化目标为验证集上的任务指标（如MATH的准确率、DROP的F1值）。MASS框架本身是无梯度的搜索过程：Stage 1/3使用提示优化器（MIPRO），Stage 2使用拒绝采样。搜索10个拓扑候选，每个候选评估3次取平均。

## 实验关键数据

### 主实验

Gemini 1.5 Pro上8个基准任务的性能对比：

| 方法 | MATH | DROP | HotpotQA | MuSiQue | MBPP | HumanEval | LCB | 平均 |
|------|------|------|----------|---------|------|-----------|-----|------|
| CoT | 71.67 | 70.55 | 57.43 | 37.81 | 68.33 | 86.67 | 66.33 | 65.28 |
| Self-Consistency | 77.33 | 74.06 | 58.60 | 41.81 | 69.50 | 86.00 | 70.33 | 68.18 |
| Multi-Agent Debate | 78.67 | 71.78 | 64.87 | 46.00 | 68.67 | 86.67 | 73.67 | 70.26 |
| ADAS | 80.00 | 72.96 | 65.88 | 41.95 | 73.00 | 87.67 | 65.17 | 69.72 |
| **MASS** | **84.67** | **90.52** | **69.91** | **51.40** | **86.50** | **91.67** | **82.33** | **78.79** |

Gemini 1.5 Flash上MASS平均得分74.30%，对比CoT的60.87%提升13.43个百分点。

### 消融实验

| 配置 | 平均性能 | 说明 |
|------|---------|------|
| CoT (基线) | 65.28% | 单Agent零样本推理 |
| Stage 1 (1PO) | ~71% | 局部提示优化，比单Agent APO高6% |
| Stage 1+2 (1PO+2TO) | ~74% | 加入拓扑优化额外提升3% |
| Stage 1+2+3 (完整MASS) | 78.79% | 全局提示优化再提升~2% |
| 无剪枝的拓扑搜索 | 下降 | 引入负面构建块 |
| 无Stage 1的拓扑搜索 | 下降 | 未优化的Agent导致搜索在低质量空间 |

### 关键发现

- 提示优化的token效率远超简单增加Agent数量：优化后的单Agent + Self-Consistency优于默认提示的9-agent SC
- 并非所有拓扑对MAS都有正面影响——有益拓扑只是搜索空间的一小部分
- MASS可完全并行化Stage 1和Stage 2的优化，而ADAS和AFlow是迭代算法需等待前序完成
- 三个MAS设计原则：(1) 先优化个体Agent再组合；(2) 组合有影响力的拓扑；(3) 建模Agent间依赖关系（通过全局优化）

## 亮点与洞察

- **深刻的分析驱动设计**: 不急于提出方法，先通过大量实验分析MAS设计空间各因素的影响，结论令人信服
- **与NAS的类比精妙**: 借鉴神经架构搜索中"搜索空间设计比搜索算法更重要"的洞察，应用于MAS设计
- **提示优化的重要性被低估**: 揭示了大多数MAS工作忽略的关键因素
- **可并行化的优化**: 实际部署中可显著降低优化时间成本

## 局限与展望

- 搜索空间中的构建块仍需预定义，无法发现全新的Agent交互模式
- 拓扑构建规则是固定序列，可能限制了更灵活的Agent组合方式
- 优化成本仍然较高（需要大量API调用），对成本敏感的场景可能不适用
- 可探索跨任务迁移——已发现的MAS设计原则是否可直接应用于新任务

## 相关工作与启发

- DSPy和MIPRO提供了提示优化的基础设施，MASS在其上构建了MAS级别的优化
- ADAS通过meta-agent生成新拓扑但忽略提示优化，AFlow通过MCTS搜索但搜索空间未经剪枝
- 启发：在设计复杂系统时，分析各组件的影响力并剪枝搜索空间，比在完整空间上暴力搜索更高效

## 评分

- 新颖性: ⭐⭐⭐⭐ 交错优化的思路新颖，分析驱动的方法论值得借鉴
- 实验充分度: ⭐⭐⭐⭐⭐ 8个任务、4个LLM、多个基线，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 分析深入，逻辑严密，图表清晰
- 价值: ⭐⭐⭐⭐ 为MAS自动化设计提供了系统性框架和设计原则

<!-- RELATED:START -->

## 相关论文

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ECCV 2024\] Optimizing Illuminant Estimation in Dual-Exposure HDR Imaging](../../ECCV2024/signal_comm/optimizing_illuminant_estimation_in_dual-exposure_hdr_imaging.md)
- [\[ACL 2026\] Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design](../../ACL2026/signal_comm/solver-independent_automated_problem_formulation_via_llms_for_high-cost_simulati.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/signal_comm/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[CVPR 2025\] DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations](../../CVPR2025/signal_comm/ditask_multi-task_fine-tuning_with_diffeomorphic_transformations.md)

<!-- RELATED:END -->
