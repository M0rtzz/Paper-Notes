---
title: >-
  [论文解读] ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition
description: >-
  [AAAI 2026][医学图像][机器意识] 实现ReCoN-Ipsundrum——一个可检查的智能体架构，在ReCoN感觉运动状态机上扩展了Humphrey的ipsundrum递归持续循环和可选的情感代理层，通过行为测试和因果消融实验证明：递归支撑刺激后持续性，情感耦合支撑偏好稳定性、结构化扫描和持久谨慎，并强调行为标记单独不足以归因意识。
tags:
  - AAAI 2026
  - 医学图像
  - 机器意识
  - 感知循环
  - 情感耦合
  - 因果消融
  - 意识指标
---

# ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Cognition

**会议**: AAAI 2026  
**arXiv**: [2602.23232](https://arxiv.org/abs/2602.23232)  
**代码**: [GitHub](https://github.com/xcellect/recips)  
**领域**: 人工意识 / 认知架构 / AI安全与伦理  
**关键词**: 机器意识, 感知循环, 情感耦合, 因果消融, 意识指标

## 一句话总结

实现ReCoN-Ipsundrum——一个可检查的智能体架构，在ReCoN感觉运动状态机上扩展了Humphrey的ipsundrum递归持续循环和可选的情感代理层，通过行为测试和因果消融实验证明：递归支撑刺激后持续性，情感耦合支撑偏好稳定性、结构化扫描和持久谨慎，并强调行为标记单独不足以归因意识。

## 研究背景与动机

近年来大语言模型的进步重新引发了关于"机器意识"的讨论。但"意识"是一个多层概念（现象体验、自我世界模型、鲁棒理解等），单纯的行为表现并不能揭示产生体验所需的内部过程。

**指标驱动（indicator-based）方法论**主张：意识的关联物应被视为提高或降低信度（credence）的证据，而非决定性测试；需要**跨指标和证据类型的三角验证**。这在AI领域尤为重要，因为行为可以通过异质机制实现，也可以通过"最小化"（可操纵的）实现达成。

本文的动机是采取一个**受约束的步骤**，朝向"与机制挂钩的测试"方向推进。具体灵感来自以下理论：

**Humphrey的ipsundrum假说**：感知（sentience）源于反射控制变为自我监控并自我维持，产生一个re-entrant循环的吸引子（ipsundrum），由此激发"qualiaphilia"（对感觉体验本身的偏好）等测试。

**Barrett的建构主义情感理论**：强调情感/内感受和预测-评估循环的作用，激发了可选的情感耦合设计。

**Butlin等人的AI意识评估框架**：强调AI评估应三角验证理论衍生的指标，并包含架构/因果证据，因为行为可以被操纵。

**重要声明**：作者明确表示不声称任何智能体是有意识的，而是提供一个**可检查的架构和可证伪的测试**。

## 方法详解

### 整体框架

智能体建立在**ReCoN（Request Confirmation Network）**——一个消息传递的神经符号架构之上，通过渐进式扩展来添加意识相关机制：

- **Stage A**：集中协调的反射（最小反射脚本，含感觉终端$N^s$和运动命令代理$N^m$）
- **Stage B（ReCoN基线）**：添加传出副本传感器$N^e$（低通滤波的运动命令幅度副本）
- **Stage C**：私有化并"加厚"感知——附加递归ipsundrum状态更新，强制感知脚本节点内部循环
- **Stage D**：添加门控规则——感知脚本在$N^e$超过阈值时继续循环，形成类吸引子的稳态机制

最终评估三个固定参数变体（无学习）：ReCoN（基线）、Ipsundrum（有递归无情感）、Ipsundrum+affect（有递归有情感）。

### 关键设计

1. **Ipsundrum递归动态（核心循环方程）**

   每步环境产生有符号感觉证据标量$I_t \in [-1, 1]$（正=有害，负=优美）。ipsundrum状态更新如下：

   **驱动计算**：
   $$\text{drive}_t = I_t + \pi_t E_{t-1} + b + \epsilon_t$$
   其中$E_{t-1}$是前一步再传入信号，$\pi_t$是有效精度，$b$是偏置项。

   **感觉显著度**：
   $$N_t^s = \text{clip}_{[0,1]}(F(\text{drive}_t))$$

   **"厚重时刻"积分器**（产生持续性）：
   $$X_t = d \cdot X_{t-1} + (1-d) \cdot N_t^s$$
   $$M_t = \text{clip}_{[0,1]}(h \cdot X_t)$$

   **传出副本和再传入信号**：
   $$N_t^e = d_e \cdot N_{t-1}^e + (1-d_e) \cdot M_t$$
   $$E_t = \text{clip}_{[0,1]}(g_{\text{eff}} \cdot M_t)$$

   有效递归强度诊断：$\alpha_{\text{eff}} = d + (1-d)(g_{\text{eff}} \cdot h \cdot \pi_t)$，区分被动衰减和主动维持的递归。

   设计动机：Humphrey认为感知是反射控制自我监控并自我维持的结果。该循环将瞬态感觉输入通过递归反馈转化为持续的内部状态——即"ipsundrum吸引子"，使感觉体验获得时间上的"厚度"。

2. **Barrett风格情感代理层**

   实现最小化的"身体预算"模型，灵感来自Barrett的建构主义情感理论：

   - **内感受代理传感器$N^i$**：身体预算模型状态（由预测误差和稳态控制器更新）
   - **效价读出$N^v$**：与设定点的接近程度（正=接近平衡=愉悦）
   - **唤醒度读出$N^a$**：预测误差和需求的幅度

   关键作用：情感可以调制ipsundrum参数（精度和/或反馈增益），实现**情感耦合控制**——内部状态依赖地改变$\alpha_{\text{eff}}$。正输入$I_t$消耗身体预算（成本），负输入$I_t$补充预算（收益）。非情感变体通过整流($I_t \leftarrow \max(0, I_t)$)消除负向输入的内建"愉悦"效应。

3. **策略：短时域内部展开**

   所有变体使用相同的动作选择程序：枚举动作 → 前向模型模拟感觉后果 → 评估短时域内部展开。动作内部得分包含多个成分：
   $$\text{Score} = \underbrace{w_v N^v + w_a N^a + w_s N^s + w_{bb}|bb-sp|}_{\text{情感/调节}} + \underbrace{w_{\text{epi}}|I_{\text{pred}} - I_{\text{cur}}|}_{\text{认知}} + \underbrace{\text{novelty bonus}}_{\text{好奇}} + \underbrace{w_{\text{prog}} \cdot \text{progress}}_{\text{目标}} - \underbrace{w_{\text{haz}} \cdot I_{\text{touch,pred}}}_{\text{危险}} - \text{costs}$$

   ReCoN基线将情感权重设为零，仅使用"脚本+规划"基底。

### 损失函数 / 训练策略

**本文没有训练过程——所有变体使用固定参数（无学习）**。这是刻意的设计选择，确保观察到的行为差异完全归因于架构差异而非训练过程。消融通过变体间的固定参数差异和运行时因果消融实现。

## 实验关键数据

### 主实验

**目标导向导航（能力与安全检查）**：

| 模型 | 走廊-危险接触 | 走廊-成功率 | 网格-危险接触 | 网格-成功率 | 网格-步数 |
|------|-------------|------------|-------------|------------|----------|
| ReCoN | 3.05 | 0.55 | 14.30 | 0.50 | 171.80 |
| Ipsundrum | 2.90 | 0.58 | 2.45 | 0.72 | 129.77 |
| Ipsundrum+affect | **0.00** | **0.73** | **0.26** | **0.99** | **9.54** |

Ipsundrum+affect在两个环境中都显著减少了危险接触，GridWorld中更是将成功率从50%提升到99%，步数从172减少到10。

### 消融实验

**因果消融（关键机制归因）**：

| 指标/签名 | ReCoN | Ipsundrum | Ipsundrum+affect | 机制归因 |
|-----------|-------|-----------|-----------------|---------|
| $N^s$刺激后持续性（AUC） | ≈0 | ≈0.24 | ≈0.15 | 递归→持续性 ✓ |
| 消融后AUC下降 | ≈0.00 | 19.12(20.3%) | **27.62(27.9%)** | 递归因果支撑持续性 |
| 新奇敏感度$\Delta$scenic-entry | 0.07 | 0.07 | **0.01（稳定）** | 情感→偏好稳定 ✓ |
| 扫描事件（探索游戏） | 0.9 | ≈ReCoN | **31.4** | 情感→结构化扫描 ✓ |
| 尾部持续时间（疼痛尾） | 5 | 5 | **90** | 情感→持久谨慎 ✓ |

### 关键发现

1. **递归→持续性的因果证明**：消融ipsundrum的反馈+积分器会选择性地降低刺激后$N^s$持续性（AUC下降20-28%），而对ReCoN基线无影响。这证明了持续性确实由实现的递归机制因果支撑。

2. **"持续性≠偏好稳定"的解耦**：Ipsundrum变体具有刺激后持续性，但在走廊偏好测试中仍然对新奇度敏感（$\Delta$scenic-entry=0.07），与ReCoN相同。只有加入情感耦合后才获得稳定的风景偏好（$\Delta$=0.01）。这意味着**将内部变量耦合到控制回路是关键**，仅有递归不够。

3. **情感耦合→结构化探索**：Ipsundrum+affect在无奖励探索中表现出31.4次扫描事件（≥2次原地转向），远高于ReCoN的0.9次，且动作熵低于随机基线（1.29 vs 1.99），说明是**结构化的局部调查**而非随机抖动。

4. **情感耦合→持久谨慎**：在疼痛-尾测试中（经历一次危险接触后在安全位置观察200步计划动作），Ipsundrum+affect的计划谨慎行为持续约90步，而其他两个变体仅5步。

5. **行为标记可被工程化**：核心伦理启示在于，这些"意识指标样"的行为签名可以通过简单的机制设计轻松产生。这**反过来说明不应仅凭行为标记归因意识**，必须配合架构检查和因果干预。

## 亮点与洞察

1. **从"声称意识"到"测试机制"的范式转变**：本文不试图证明智能体有意识，而是展示如何系统地设计、测试和解剖意识相关机制。这种"可检查性优先"的方法论对AI安全领域具有深远意义。

2. **精巧的Humphrey分阶段实现**：将Humphrey从反射到感知的进化叙事映射为渐进式架构扩展（Stage A→D），每步添加和消融特定机制，实现了理论到实现的清晰对应。

3. **"新奇竞争"的巧妙实验设计**：在走廊偏好测试中通过预暴露操纵熟悉度来分离新奇驱动和价值驱动的偏好，这一实验设计直接对标了Humphrey关于qualiaphilia的理论预测。

4. **因果消融的方法学价值**：运行时（在episode内部）消融特定机制组件，而非比较不同训练后的模型；这提供了更强的因果归因证据。

5. **诚实的局限性陈述**：作者坦诚承认循环性质——"消融递归减少持续性，部分因为持续性由递归实现"——并将消融定位为实现保真度和因果归因检查，而将更实质的发现放在组件解耦上。

## 局限与展望

1. **极简玩具域**：走廊和网格世界极为简单，无法代表真实世界环境中的复杂性。标量感觉输入$I_t$远不及真实视觉/触觉输入的丰富性。
2. **无学习过程**：所有变体使用固定参数，未探索学习如何影响意识指标样行为的涌现。
3. **样本量有限**：主要实验使用20个种子，部分置信区间较宽，效应估计不够精确。
4. **"内感受"高度抽象**：情感代理层是一个由外感受标量驱动的簿记抽象，与真实的生理内感受系统相差甚远。
5. **未覆盖其他意识理论**：未涉及IIT（整合信息论）、全局工作空间理论、高阶思维理论等竞争性理论框架。
6. **qualiaphilia的构建效度**：走廊偏好是"价值塑形的"——风景vs.单调直接改变$I_t$进而影响内部得分，这使得"感觉偏好"指标的理论纯净性受到质疑。

## 相关工作与启发

- **Butlin等人（2025）的AI意识指标框架**是本文的方法论基础，本文用Table 1精确映射了其实现与该框架各指标的对应关系和差距。
- **Bach & Herger（2015）的ReCoN架构**提供了感觉运动基底。本文的扩展策略——保留ReCoN的执行骨干并添加/消融额外因果结构——是一种值得借鉴的渐进式架构研究方法。
- 本文的核心贡献在方法论层面：**提供了一个"如何系统化评估AI意识指标"的实例**，而非意识本身的证据。这对AI伦理、安全和监管领域具有参考价值。
- 未来可将ipsundrum循环从标量递归扩展到结构化隐空间，或与格栅-位置细胞模型结合探索抽象表示和原型语言系统。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统实现Humphrey的ipsundrum假说并设计可操作的意识指标测试
- **技术深度**: ⭐⭐⭐⭐ — 架构设计精巧，因果消融方法学严谨，但域过于简单
- **实验充分度**: ⭐⭐⭐⭐ — 多个测试覆盖不同指标，因果消融有力，但样本量偏小
- **实用性**: ⭐⭐⭐ — 方法论贡献大于直接应用价值，为AI安全评估提供参考框架
- **写作质量**: ⭐⭐⭐⭐⭐ — 理论动机、实现细节、伦理声明三者高度统一，非常透明诚实

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Human-in-the-Loop Interactive Report Generation for Chronic Disease Adherence](human-in-the-loop_interactive_report_generation_for_chronic_disease_adherence.md)
- [\[AAAI 2026\] LungNoduleAgent: A Collaborative Multi-Agent System for Precision Diagnosis of Lung Nodules](lungnoduleagent_a_collaborative_multi-agent_system_for_precision_diagnosis_of_lu.md)
- [\[AAAI 2026\] Refine and Align: Confidence Calibration through Multi-Agent Interaction in VQA](refine_and_align_confidence_calibration_through_multi-agent_interaction_in_vqa.md)
- [\[ICLR 2026\] Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](../../ICLR2026/medical_imaging/resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)
- [\[AAAI 2026\] MAMA-Memeia! Multi-Aspect Multi-Agent Collaboration for Depressive Symptoms Identification in Memes](mama-memeia_multi-aspect_multi-agent_collaboration_for_depressive_symptoms_ident.md)

<!-- RELATED:END -->
