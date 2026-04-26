---
title: >-
  [论文解读] Position: Bridge the Gaps between Machine Unlearning and AI Regulation
description: >-
  [NeurIPS 2025][AI安全][机器遗忘] 系统分析了机器遗忘（Machine Unlearning）在欧盟人工智能法案（AIA）合规中的六大潜在应用场景，指出每个场景中 SOTA 与实际需求之间的技术差距，呼吁研究社区弥补这些差距以释放机器遗忘在 AI 监管中的潜力。
tags:
  - NeurIPS 2025
  - AI安全
  - 机器遗忘
  - AI监管
  - EU AI Act
  - 数据隐私
  - 合规
---

# Position: Bridge the Gaps between Machine Unlearning and AI Regulation

**会议**: NeurIPS 2025  
**arXiv**: [2502.12430](https://arxiv.org/abs/2502.12430)  
**代码**: 暂无  
**领域**: AI安全  
**关键词**: 机器遗忘, AI监管, EU AI Act, 数据隐私, 合规

## 一句话总结

系统分析了机器遗忘（Machine Unlearning）在欧盟人工智能法案（AIA）合规中的六大潜在应用场景，指出每个场景中 SOTA 与实际需求之间的技术差距，呼吁研究社区弥补这些差距以释放机器遗忘在 AI 监管中的潜力。

## 研究背景与动机

### 领域现状

**领域现状**：机器遗忘（MU）自诞生以来一直以 GDPR 中的"被遗忘权"为动机。如今，以欧盟人工智能法案（AIA）为代表的新一波 AI 监管法规正在推进，研究者开始探索 MU 是否能辅助这些法规的合规。

然而本文的立场是：**MU 辅助 AI 监管合规的潜力只有在研究者主动弥补技术差距的前提下才能实现**。目前 MU 的 SOTA 与其在监管合规中的预期应用之间存在相当大的鸿沟。

选择 AIA 作为案例研究的原因：

### 现有痛点

**现有痛点**：AIA 是全球首部全面的 AI 监管法案，已于 2024 年生效

### 核心矛盾

**核心矛盾**：采用基于风险的分层监管方式，对高风险 AI 系统和通用 AI 模型有详细要求

### 解决思路

**解决思路**：AIA 的原则在全球 AI 监管中具有代表性，分析结果可推广到其他法规

## 方法详解

### 整体框架

本文不提出新算法，而是**建立了 MU 与 AIA 合规之间的完整映射**，识别出六大应用场景，并逐一分析每个场景中的技术挑战。

### 关键设计（六大应用场景分析）

1. **准确性（Accuracy）**：
    - AIA 要求：高风险 AI 系统须达到与预期用途和 SOTA 匹配的适当准确度水平
    - MU 可用途径：遗忘标签错误、过时或异常的训练数据以提升准确性
    - 技术差距：识别所有导致不准确的数据点本身就很困难；部分遗忘可能适得其反；近似遗忘不应期望比精确重训更高的准确性
    - 形式化定义：遗忘算法 $U$ 是 $(\epsilon, \delta)$-遗忘器，若 $U(M; D_f, D_r)$ 的分布与 $A(D_r)$ 的分布 $(\epsilon, \delta)$-接近

2. **偏见（Bias）**：
    - AIA 要求：高风险 AI 和系统性风险 GPAI 必须缓解歧视性偏见
    - MU 可用途径：遗忘导致偏见的数据点或训练模式
    - 技术差距：如果偏见源自数据缺失（而非数据存在），MU 无法帮助；如何评估偏见仍是"开放问题"；MU 主要是后处理方法，难以解决根源性偏见

3. **机密性攻击（Confidentiality Attacks）**：
    - AIA 要求：检测和防控机密性攻击（成员推断、数据重构等）
    - MU 可用途径：遗忘易受攻击的机密信息
    - 技术差距：遗忘某些数据点可能反而暴露相邻数据的隐私（"洋葱效应"）；过度遗忘可能暴露遗忘数据本身的成员身份（"史翠珊效应"）；近似遗忘存在准确性和偏见的权衡
    - 替代方案：差分隐私（DP）在某些场景下可能更优

4. **数据投毒（Data Poisoning）**：
    - AIA 要求：预防、检测和控制数据投毒攻击
    - MU 可用途径：移除已知投毒数据的影响
    - 技术差距：识别完整的投毒样本集极具挑战性；投毒数据与干净数据可能视觉上不可区分；部分方法存在显著的准确性权衡

5. **生成式 AI 风险（GenAI Risk）**：
    - AIA 要求：缓解有害生成输出（不良医疗建议、CBRN 知识、歧视性内容等）
    - MU 可用途径：遗忘训练集中导致有害输出的数据或概念
    - 技术差距：宽泛概念（如非歧视）难以具体化为离散遗忘集；即使移除直接有害数据，模型仍可能从剩余数据的潜在信息中组装出危险输出；双重用途问题使遗忘集识别更困难

6. **版权（Copyright）**：
    - AIA 要求：GPAI 提供者须有版权合规政策，尊重数据挖掘退出请求
    - MU 可用途径：在输出阶段防止受版权保护的训练数据被再现
    - 技术差距：即使精确重训（移除版权数据）也不能保证输出不侵权，因为模型可能从剩余数据中泛化出相似表征；近似遗忘"不足以"用于版权场景

### 损失函数 / 训练策略

形式化框架：模型 $M = A(D)$，遗忘算法 $U(M; D_f, D_r)$ 产生遗忘模型 $M_u$。当 $\epsilon = \delta = 0$ 时为精确遗忘，否则为近似遗忘。三大目标需权衡：模型效用、遗忘质量和效率。

## 实验关键数据

### SOTA 能力总结（按场景）


### 主实验

| 应用场景 | MU SOTA 可行性 | 主要障碍 | 替代方案 |
|---------|---------------|---------|---------|
| 准确性 | 较高 | 识别问题数据点 | 全量重训、改进数据管线 |
| 偏见 | 中等 | 评估标准不统一、数据缺失不可遗忘 | 前/中/后处理去偏方法 |
| 机密攻击 | 中等 | 洋葱/史翠珊效应、新攻击适应性差 | 差分隐私、访问控制 |
| 数据投毒 | 较低 | 识别投毒集、准确性权衡 | 鲁棒训练、数据清洗 |
| GenAI 风险 | 较低 | 概念级遗忘困难、双重用途 | RLHF、护栏机制 |
| 版权 | 较低 | 潜在信息泛化、形式保证缺失 | 数据治理、输出过滤 |

### 核心挑战分析


### 消融实验

| 横切挑战 | 影响范围 | 当前状态 |
|---------|---------|---------|
| 遗忘集识别 | 全部6个场景 | 大多数场景下仍是未解决问题 |
| 可审计性/验证 | 全部6个场景 | 近似MU缺乏形式化保证 |
| 效用-遗忘权衡 | 准确性、偏见、机密性 | 遗忘更多数据时准确性下降 |
| 隐私-遗忘冲突 | 机密性 | 遗忘可能反而暴露信息 |
| 概念级遗忘 | GenAI风险、版权 | 分布式表征中的非离散目标 |

### 关键发现

- MU 对 AIA 合规有**潜在**价值但远非银弹——6 个场景中大多数存在相当大的技术差距
- **可审计性**是贯穿所有场景的核心挑战：监管者需要验证遗忘的有效性，但当前近似 MU 方法仅依赖经验代理指标
- 各场景之间存在相互依赖：如遗忘修复准确性可能影响公平性，去偏遗忘可能改变隐私攻击暴露面
- 在许多场景中，MU 更适合作为"反应式补救"而非"预防性保护"

## 亮点与洞察

- **首次完整映射** MU 技术与 AIA 法规条款的对应关系，为 MU 研究者提供了清晰的监管需求全景图
- 对每个应用场景同时分析了 MU 的潜力和替代方案，公平客观
- 强调"可审计性"这一被 MU 社区普遍忽视的核心需求——如果无法向监管者证明遗忘有效，技术优势无从发挥
- 指出 MU 的"洋葱效应"和"史翠珊效应"两个反直觉风险

## 局限与展望

- 作为立场论文（position paper），不包含新的算法或实验
- 主要分析 AIA，但对其他法规（如美国州级 AI 法案、加拿大 AIDA）的适用性仅简略提及
- 未深入讨论 MU 与其他技术（如 DP、RLHF）的组合使用策略
- 部分分析较为定性，缺乏量化的差距评估

## 相关工作与启发

- Cooper et al. (2024) "Machine unlearning doesn't do what you think" 对生成模型中 MU 提出质疑，本文在此基础上扩展到监管合规视角
- WMDP 基准（Li et al., 2024c）用于衡量 CBRN 知识遗忘，可作为验证工具
- 对于从事 MU 研究的学者，本文提供了 6 个与法规直接相关的研究方向，有助于提升研究的社会影响力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统性地将MU与AI法规对齐分析，视角新颖
- 实验充分度: ⭐⭐⭐ 无实验，但文献分析全面系统
- 写作质量: ⭐⭐⭐⭐⭐ 法律条款引用精确，技术分析平衡客观
- 价值: ⭐⭐⭐⭐ 为MU社区指明了法规驱动的研究方向，弥合AI安全研究与政策需求的重要桥梁

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research](machine_unlearning_doesnt_do_what_you_think_lessons_for_generative_ai_policy_and.md)
- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](../../CVPR2025/ai_safety/towards_source-free_machine_unlearning.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)

<!-- RELATED:END -->
