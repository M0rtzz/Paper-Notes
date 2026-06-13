---
title: >-
  [论文解读] ShortageSim: Simulating Drug Shortages under Information Asymmetry
description: >-
  [AAAI 2026 Oral][医疗NLP][药品短缺] 提出 ShortageSim，首个基于 LLM 多智能体：的药品短缺模拟框架，建模 FDA 监管者、制造商和购买者在信息不对称下的战略决策，在历史短缺数据上实现对解决滞后时间 84% 的预测改善，为监管策略评估提供受控测试平台。 1. 领域现状：药品短缺是全球性的医…
tags:
  - "AAAI 2026 Oral"
  - "医疗NLP"
  - "药品短缺"
  - "多智能体模拟"
  - "大语言模型"
  - "信息不对称"
  - "监管政策"
---

# ShortageSim: Simulating Drug Shortages under Information Asymmetry

**会议**: AAAI 2026 Oral  
**arXiv**: [2509.01813](https://arxiv.org/abs/2509.01813)  
**代码**: [https://github.com/Lemutisme/ShortageSim](https://github.com/Lemutisme/ShortageSim)  
**领域**: 其他  
**关键词**: 药品短缺, 多智能体模拟, 大语言模型, 信息不对称, 监管政策

## 一句话总结

提出 ShortageSim，首个基于 **LLM 多智能体**的药品短缺模拟框架，建模 FDA 监管者、制造商和购买者在信息不对称下的战略决策，在历史短缺数据上实现对解决滞后时间 84% 的预测改善，为监管策略评估提供受控测试平台。

## 研究背景与动机

1. **领域现状**：药品短缺是全球性的医疗危机——美国过去十年平均每年 130+ 起新短缺事件，平均持续时间从 2011 年的 9 个月增至 2016 年的 14 个月。美国医院每年花费至少 3.59 亿美元管理短缺相关物流。
2. **现有痛点**：(a) 信息不对称严重——制造商隐藏产能数据作为商业机密，监管者只能看到聚合短缺水平，购买者无法区分临时与永久断供；(b) 传统博弈论模型假设完全理性和完全信息，无法捕捉真实决策者如何解读模糊监管信号、动态更新信念；(c) FDA 监管干预（如发布短缺警报）可能引发囤积行为，反而加剧短缺。
3. **核心矛盾**：监管信号旨在促进信息透明以缓解短缺，但可能触发恐慌性囤积和过度产能扩张的协调失败。评估这些反事实效应需要受控实验环境。
4. **本文目标**：构建一个能模拟监管干预对竞争动态影响的框架，尤其是在信息不对称条件下。
5. **切入角度**：用 LLM 作为异质化智能体，模拟制造商、购买者和 FDA 对部分信息的主观解读和非最优决策。
6. **核心 idea**：LLM 智能体天然具备"不完全理性+主观解读"的能力，比博弈论假设更适合模拟信息不对称下的人类决策。

## 方法详解

### 整体框架

框架包含四个核心组件：(1) 环境模块——管理市场动态、状态转移和信息隔离；(2) 智能体系统——LLM 驱动的角色特定决策（制造商、购买者、FDA）；(3) 信息流——控制智能体间的通信模拟信息不对称；(4) 仿真控制器——操作执行流程并记录决策。

### 关键设计

1. **两阶段决策管线（Analyze → Decide）**

    - 功能：模拟真实决策中从信息处理到策略形成的过程。
    - 核心思路：每个智能体经过两步——(a) **收集与分析**：接收非结构化上下文（FDA 公告文本、聚合市场指标、历史趋势），生成结构化处境表征；(b) **决策生成**：基于结构化表征生成决策及详细推理。关键点：决策不要求最优——反映真实中某些利益相关者不信任竞争对手的理性并做出次优甚至不合理的选择。
    - 设计动机：两阶段设计允许通过推理输出检验 FDA 通信的效果，并捕获不同智能体推理过程的差异。

2. **信息不对称建模**

    - 功能：严格模拟制药供应链中的信息壁垒。
    - 核心思路：制造商知道自身产能和恢复时间，但不知道竞争对手状态，只能通过 FDA 公告和自身分配需求推断。购买者只看到总供给量但不知道分配和产能。FDA 只看到聚合短缺水平和强制报告的断供事件，不知道具体产能、投资和库存。时序上：断供→FDA决定是否发公告→制造商同时做投资决策→购买者同时做采购决策→市场出清。
    - 设计动机：直接反映真实供应链结构——制造商产能是商业机密、美国报告是分散自愿的、购买者库存不透明。

3. **角色特定智能体设计**

    - 功能：每种角色有不同的目标函数和信息可用性。
    - 核心思路：**制造商**——目标是利润最大化，需权衡从竞争对手断供中获利的机会与产能过剩的风险。**购买者**——面临供给不确定下的报童问题变体，优先患者安全但需平衡囤积成本。**FDA**——被动政策框架，通过公告缓解短缺而不泄露私人信息，需平衡紧迫性和稳定性。
    - 设计动机：多视角、多目标的异质化智能体比同质化假设更贴近真实市场。

### 损失函数 / 训练策略

无需训练。用 GPT-4o、Gemini 2.5 Flash、Claude Sonnet 4.5、Deepseek V3.2 作为 LLM 骨干，temperature=0.3。在 2,925 个 FDA 短缺事件和 51 条已解决历史轨迹上评估。

## 实验关键数据

### 主实验

|  模型 | 数据集 | RLP(%) | FIP(%) |
|-------|--------|--------|--------|
| ShortageSim (GPT-4o) | FDA-Disc | **4.5±3.4** | 82.6±3.0 |
| Zero-shot (GPT-4o) | FDA-Disc | -28.3±0.2 | 92.9±0.1 |
| ShortageSim (Claude) | FDA-Disc | -9.4±1.4 | 69.1±2.3 |
| Zero-shot (Claude) | FDA-Disc | -32.7±0.5 | 75.9±0.7 |
| ShortageSim (GPT-4o) | FDA-NR | -34.8±8.7 | 30.0±5.5 |
| Zero-shot (GPT-4o) | FDA-NR | -23.1±1.6 | 90.4±18.6 |

### 消融实验

| 配置 | RLP(%) | 说明 |
|------|--------|------|
| ShortageSim (完整) | 4.5 | FDA 公告启用 |
| 无 FDA 公告 | 19.1 | 解决时间显著延长 |
| Zero-shot 基线 | -28.3 | 无迭代决策，预测过早解决 |

### 关键发现

- ShortageSim 在所有 4 个 LLM 提供商上的 RLP 一致优于 zero-shot 基线（Wilcoxon 检验 p<0.05）。
- GPT-4o 的 RLP 仅 4.5%，几乎完美再现历史解决时间线。
- 禁用 FDA 公告使 RLP 从 4.5% 上升到 19.1%，**确认了监管信号在缓解短缺中的效果**。
- 主动性政策（proactive）导致更多囤积行为和更差的短缺解决效果。
- 制造商数量越多（>5），投资意愿越低（搭便车效应），短缺持续更久。
- FDA-NR 数据集效果较差，因为该子集断供原因不明确，模拟假设与实际不匹配。

## 亮点与洞察

- **LLM 作为"有限理性"智能体**：传统博弈论假设完全理性，LLM 天然具备对模糊信号的主观解读和次优决策能力，更贴近人类行为。
- **反事实政策评估平台**：可以系统性对比不同 FDA 沟通策略（被动vs主动、严重程度措辞等），这在现实中不可能做受控实验。
- **首个药品短缺领域的 LLM 多智能体框架**，开源数据集 2,925 条 FDA 短缺事件填补了依赖合成/私有数据的空缺。

## 局限与展望

- 不包含分销商和 GPO 等中间实体，可能遗漏物流层面的动态。
- 制造商对称性假设在引言中提及放松但实验中未充分探索。
- LLM 推理的可重复性有限（temperature=0.3 仍有随机性），多次实验取平均可能不够。
- FDA-NR 数据集表现较差，说明框架对断供原因的先验假设敏感。

## 相关工作与启发

- **vs 传统运筹学模型**：假设理性决策和完全信息，无法捕获信息不对称下的动态行为。ShortageSim 直接建模信息壁垒。
- **vs EconAgent**：EconAgent 模拟宏观经济现象，ShortageSim 聚焦特定供应链的微观交互。
- **vs OptiGuide**：微软的供应链优化框架用 LLM 做优化，ShortageSim 用 LLM 做行为模拟。
- 该框架可推广到其他供应链危机模拟（半导体短缺、能源供应中断等）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将 LLM 多智能体用于药品短缺模拟，信息不对称建模巧妙
- 实验充分度: ⭐⭐⭐⭐ 历史数据验证+多 LLM 提供商+反事实分析
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机充分，框架图示清晰
- 价值: ⭐⭐⭐⭐⭐ 对公共卫生政策制定有重大潜在影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Faithfulness vs. Safety: Evaluating LLM Behavior Under Counterfactual Medical Evidence](../../ACL2026/medical_nlp/faithfulness_vs_safety_evaluating_llm_behavior_under_counterfactual_medical_evid.md)
- [\[AAAI 2026\] Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)
- [\[AAAI 2026\] Real-Time Trust Verification for Safe Agentic Actions Using TrustBench](real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)
- [\[AAAI 2026\] BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives](bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)

</div>

<!-- RELATED:END -->
