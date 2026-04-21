---
title: >-
  [论文解读] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
description: >-
  [CVPR 2026][机器人][VLA] 提出AtomicVLA，在π₀基础上构建统一规划-执行框架，通过自适应Think-Act切换生成原子技能抽象，并用技能引导的MoE（SG-MoE）将动作路由到专精expert执行，LIBERO-LONG成功率从85.2%提升至95.2%（+10%），真实Franka长任务+18.3%，持续学习+21%。
tags:
  - CVPR 2026
  - 机器人
  - VLA
  - 原子技能
  - Mixture-of-Experts
  - 持续学习
  - 任务规划
  - 技能路由
---

# AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots

**会议**: CVPR 2026  
**arXiv**: [2603.07648](https://arxiv.org/abs/2603.07648)  
**机构**: 中山大学、鹏城实验室、银旺智能
**领域**: 机器人操作 / 视觉-语言-动作模型  
**关键词**: VLA, 原子技能, Mixture-of-Experts, 持续学习, 任务规划, 技能路由

## 一句话总结

提出AtomicVLA，在π₀基础上构建统一规划-执行框架，通过自适应Think-Act切换生成原子技能抽象，并用技能引导的MoE（SG-MoE）将动作路由到专精expert执行，LIBERO-LONG成功率从85.2%提升至95.2%（+10%），真实Franka长任务+18.3%，持续学习+21%。

## 研究背景与动机

- **现有瓶颈**：当前VLA模型（π₀、OpenVLA等）使用单一动作解码器，将所有技能知识混合在同一组参数中。面对多步长任务时缺乏显式规划能力，面对新技能学习时存在灾难性遗忘。
- **两阶段方法的不足**：SayCan、Inner Monologue等方法用外部LLM做高层规划+独立底层控制器执行，但规划器与执行器之间缺乏相互感知，导致指令过时或不相关，且存在系统延迟。
- **核心需求**：机器人模型需要同时支持 (1) 高层推理与任务规划，(2) 精细动作生成，(3) 可扩展的持续学习——现有方法无法同时满足这三点。
- **核心思路**：将复杂任务分解为可复用的原子技能（如grasp、push、rotate），每个技能由专属expert处理，通过模块化设计实现技能库的持续扩展。

## 方法详解

### 整体框架

AtomicVLA基于π₀构建端到端框架，统一thinking和acting两种模态。输入为多相机观测 $O_t^{1:n}$ 和语言指令 $\ell$，模型首先自适应预测当前应进入thinking还是acting模式：

- **Thinking模式**：在任务开始或子技能切换时激活，生成三部分输出——任务链 $C_{0 \to k}$（将指令分解为有序子目标）、当前进度 $C_t$、原子技能抽象 $\sigma$
- **Acting模式**：在正常执行阶段激活，基于最新技能抽象 $\sigma$ 和本体感知状态 $s_t$，通过SG-MoE生成动作chunk $A_t$

### 关键设计

**1. 自适应Think-Act切换机制**

- **功能**：让模型自主决定"该想还是该做"，避免每步都做规划的计算浪费
- **核心思路**：引入两个特殊输出token [think] 和 [act]。模型在每个决策步先预测标识符，[think] 触发高层规划生成任务链和原子技能标签，[act] 触发动作chunk输出
- **设计动机**：任务初始和子技能转换时才需要规划，中间执行阶段直接生成动作即可。这种自适应切换比固定周期规划更高效，也比完全不规划的纯动作预测更鲁棒。当执行失败时（如物体掉落），模型能自动检测异常、重新触发thinking、生成新的技能抽象并恢复执行

**2. 技能引导的Mixture-of-Experts架构（SG-MoE）**

- **功能**：将不同原子技能的动作生成分配给各自专精的expert，避免技能间干扰
- **核心思路**：SG-MoE包含三个核心组件：
    - **Skill Router**：基于原子技能抽象 $\sigma$ 做路由决策。采用噪声调度式嵌入：每个原子技能映射到标量噪声级别 $\sigma \in [0,100]$，经可学习嵌入函数转换为高维向量 $Z_\sigma = E(\text{norm}(\log(\sigma)))$。Router对 $Z_\sigma$ 计算expert概率分布，激活top-1技能expert
    - **共享Expert**：保留π₀预训练权重，维持通用动作生成能力，所有token均经过此expert
    - **原子技能Expert**：每个expert专精一种原子技能，通过训练自然分化
- **设计动机**：传统MoE的token级路由使每个expert仍学习混合技能，没有明确的技能专精化。SG-MoE用语义明确的原子技能标签做路由，确保同一技能的所有动作token由同一expert处理，减少技能间干扰
- **输出融合**：最终动作为共享expert和激活技能expert的加权组合 $F_{out} = (1-w_k) \cdot F_{share}(x_t) + w_k \cdot F_k(x_t)$

**3. 模块化持续学习（Skill Expansion）**

- **功能**：在不遗忘已有技能的前提下持续获取新技能
- **核心思路**：当引入新原子技能时，只需 (a) 新增一个技能expert（随机初始化），(b) 扩展skill router的嵌入空间覆盖新技能标签，(c) 冻结所有已有expert参数——只训练新expert和更新后的router分支
- **设计动机**：传统VLA的全量微调导致灾难性遗忘（如π₀.₅学新技能后旧技能平均下降15%）。"只加不改"的模块化策略从架构层面规避遗忘，router通过复制原有权重初始化保证平滑集成

**4. 高质量Embodied Reasoning数据生成**

- **功能**：为thinking模式提供高质量训练数据（任务链+进度+技能标签）
- **核心思路**：两阶段数据构建——首先对机器人演示轨迹做主轴运动分析（比较平移/旋转分量大小 + 夹爪状态追踪），自动分割为原子技能片段（如z轴持续下降+夹爪闭合 → "pick"动作）；然后用InternVideo2.5对每个片段进行语义标注，自动校正并丰富初始的原子动作标签
- **设计动机**：传统方法依赖VLM做视频理解或光流特征分割，容易产生歧义和噪声，需要大量人工后处理。基于物理运动学的分解更精确、更少依赖人工

### 训练策略

- **Think模式**：交叉熵损失，预测任务链、进度和技能标签的token序列
- **Act模式**：Flow matching损失（继承π₀），预测连续动作chunk
- **总损失**：$\mathcal{L}_{total} = \mathcal{L}_{think} + \mathcal{L}_{act}$
- **Expert配置**：LIBERO和真实机器人实验使用5个技能expert，CALVIN使用8个技能expert
- **两种变体**：AtomicVLA基于π₀构建，AtomicVLA\*基于π₀.₅构建

## 实验关键数据

### 主实验：LIBERO Benchmark（成功率%）

| 方法 | Spatial | Object | Goal | Long | Avg. |
|------|---------|--------|------|------|------|
| Octo | 78.9 | 85.7 | 84.6 | 51.1 | 75.1 |
| OpenVLA | 84.9 | 88.4 | 79.2 | 53.7 | 76.5 |
| CoT-VLA | 87.5 | 91.6 | 87.6 | 69.0 | 81.1 |
| π₀ | 96.4 | 98.8 | 95.8 | 85.2 | 94.2 |
| π₀.₅ | 98.8 | 98.2 | 98.0 | 92.4 | 96.9 |
| **AtomicVLA** | 96.8 | 98.0 | 96.4 | **95.2** | **96.6** |
| **AtomicVLA\*** | **98.8** | **98.8** | **97.2** | **96.2** | **97.8** |

### 主实验：CALVIN ABC→D（长任务序列完成率%）

| 方法 | 1任务 | 2任务 | 3任务 | 4任务 | 5任务 | Avg.Len↑ |
|------|-------|-------|-------|-------|-------|----------|
| π₀ | 94.3 | 87.0 | 77.9 | 68.5 | 59.4 | 3.87 |
| π₀.₅ | 91.9 | 84.6 | 79.4 | 75.5 | 71.0 | 4.02 |
| **AtomicVLA** | 95.0 | 87.8 | 81.9 | 75.0 | 69.1 | **4.09** |
| **AtomicVLA\*** | 94.1 | **88.7** | **85.2** | **81.7** | **77.6** | **4.27** |

### 真实Franka机器人：长任务（成功率%）

| 方法 | 物体入盘 | 物体入抽屉 | 物体入微波炉 | Avg. | ΔAvg. |
|------|----------|------------|-------------|------|-------|
| π₀ | 45 | 55 | 10 | 36.7 | — |
| π₀.₅ | 65 | 35 | 35 | 45.0 | — |
| AtomicVLA | 65 | 60 | 45 | 56.7 | +20.0 |
| AtomicVLA\* | **75** | **60** | **55** | **63.3** | **+18.3** |

### 持续学习：技能扩展实验（成功率%）

| 方法 | Grasp | Stack | Close | Press | Open(新) | Avg. | ΔAvg. |
|------|-------|-------|-------|-------|----------|------|-------|
| π₀.₅ (基线) | 85 | 65 | 70 | 90 | — | 77.5 | — |
| π₀.₅ (持续学习) | 70 | 45 | 60 | 75 | 55 | 61.0 | **-15.0** |
| AtomicVLA\* (基线) | 95 | 80 | 70 | 100 | — | 86.3 | — |
| AtomicVLA\* (持续学习) | 90 | 80 | 80 | 100 | 70 | **82.0** | **-1.3** |

### 消融实验：SG-MoE路由机制（LIBERO-LONG%）

| 方法 | LIBERO-LONG |
|------|-------------|
| π₀ (无MoE) | 85.2 |
| + 标准token级MoE | 88.6 (+3.4) |
| + MoDE (去噪步路由) | 89.5 (+4.3) |
| + **SG-MoE (原子技能路由)** | **95.2 (+10.0)** |

### 关键发现

- **LIBERO-LONG提升最显著**：AtomicVLA在最具挑战性的长序列任务上提升10%，证明显式规划+技能分解对多步任务至关重要
- **SG-MoE远超标准MoE**：标准token级MoE和MoDE仅分别提升3.4%和4.3%，而SG-MoE提升10%——因为前两者仍是token级路由，expert学习混合技能；SG-MoE确保同一技能的所有token由同一expert处理
- **持续学习几乎无遗忘**：π₀.₅学习新技能后旧技能平均下降15%（Stack最严重降20%），AtomicVLA\*仅下降1.3%，从架构层面解决遗忘问题
- **混合训练技能干扰**：不同夹爪状态需求的任务混合训练会互相干扰（如抽屉开启不需闭合夹爪，影响抓取任务），SG-MoE通过技能隔离有效缓解
- **错误恢复能力**：AtomicVLA在执行失败时能自动检测异常并重新规划（如物体掉落后重新生成技能抽象），但CALVIN评测框架不认可恢复后的完成，报告数字可能低估真实能力

## 亮点与洞察

- **Think-Act统一范式**：不是简单的chain-of-thought叠加，而是将规划与原子技能抽象深度耦合——thinking的输出直接驱动MoE的路由决策，规划和执行在同一模型中形成闭环
- **噪声调度式技能嵌入**：借鉴扩散模型的思路将离散技能标签映射到连续嵌入空间做路由，设计巧妙且实验证明优于通用token路由
- **"只加不改"的持续学习**：不依赖正则化（EWC等）或经验回放，而是从架构设计上冻结旧expert、只训练新expert，简单有效
- **主轴分析数据生成**：用运动学主轴分析替代VLM视频理解做原子动作分割，物理先验更可靠且不依赖人工标注

## 局限性

- 原子技能标签质量依赖InternVideo2.5的生成能力，对罕见或高度专业化操作可能标注不准
- SG-MoE采用top-1路由，需要多技能协同的动作（如"边推边转"）可能需要top-k路由
- 每新增一个技能就增加一个expert，长期参数量线性增长——缺乏expert合并或剪枝机制
- CALVIN上提升（+0.22 avg len）相对LIBERO较小，可能因CALVIN任务粒度与原子技能对齐不够紧密
- 混合训练异构任务时仍存在干扰（如夹爪状态冲突），SG-MoE缓解但未完全消除

## 相关工作与启发

- **vs π₀/π₀.₅**：纯flow matching动作预测，无显式规划。AtomicVLA在其基础上增加thinking和技能路由，LIBERO-Long +10%验证规划的价值
- **vs SayCan/Inner Monologue**：外部LLM做规划+独立控制器执行，规划与执行分离导致模态gap。AtomicVLA统一在同一模型中
- **vs MoDE**：用去噪时间步做路由信号，本质仍是token级路由。SG-MoE用语义明确的技能标签路由，+5.7%证明技能级路由更优
- **MoE技能路由的通用性**：噪声调度式嵌入路由可迁移到多任务NLP——用任务描述嵌入做expert路由
- **模块化持续学习范式**：冻结旧expert+新增新expert的策略可用于视觉大模型的领域持续预训练

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Think-Act统一 + SG-MoE噪声调度路由 + 模块化持续学习，三大创新点自洽
- 实验充分度: ⭐⭐⭐⭐ LIBERO四子集 + CALVIN + 真实Franka + 消融 + 持续学习，数据充实
- 写作质量: ⭐⭐⭐⭐ 动机-方法-实验逻辑清晰，SG-MoE架构图直观，算法伪代码规范
- 价值: ⭐⭐⭐⭐⭐ 对VLA持续学习和长任务规划有重要贡献，SG-MoE思路具有通用启发价值

<!-- RELATED:START -->

## 相关论文

- [Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures.md)
- [Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)

<!-- RELATED:END -->
