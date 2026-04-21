---
title: >-
  [论文解读] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents
description: >-
  [ICLR 2026][LLM Agent][Web Agent] 受Bloom教育分类学启发，提出 Web-CogKnowledge Framework，将Web Agent能力分解为 Factual→Conceptual→Procedural 三层知识的渐进式学习，配合 Knowledge-driven CoT 推理框架训练得到 Web-CogReasoner，在Web-CogBench上以84.4%超越Claude Sonnet 4 (76.8%)和Gemini 2.5 Pro (80.4%)。
tags:
  - ICLR 2026
  - LLM Agent
  - Web Agent
  - 认知推理
  - Bloom分类学
  - Chain-of-Thought
  - 知识驱动
---

# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents

**会议**: ICLR 2026  
**arXiv**: [2508.01858](https://arxiv.org/abs/2508.01858)  
**代码**: [https://github.com/Gnonymous/Web-CogReasoner](https://github.com/Gnonymous/Web-CogReasoner) (有)  
**领域**: LLM Agent  
**关键词**: Web Agent, 认知推理, Bloom分类学, Chain-of-Thought, 知识驱动

## 一句话总结

受Bloom教育分类学启发，提出 Web-CogKnowledge Framework，将Web Agent能力分解为 Factual→Conceptual→Procedural 三层知识的渐进式学习，配合 Knowledge-driven CoT 推理框架训练得到 Web-CogReasoner，在Web-CogBench上以84.4%超越Claude Sonnet 4 (76.8%)和Gemini 2.5 Pro (80.4%)。

## 研究背景与动机

Web Agent正从早期的规则系统演变为基于LLM/LVM的智能系统。当前Web Agent面临的核心挑战是：**通用预训练知识在专门任务上存在性能瓶颈**。

具体而言：

**纯文本Agent**：仅处理HTML/Accessibility Tree，遗漏视觉线索

**纯视觉Agent**：直接从截图推理，但缺乏结构化数据

**混合Agent**：整合双模态，但仍缺乏系统化的知识基础

先前的知识增强方法往往缺乏系统性或理论支撑。论文的关键洞察来自教育学：人类学习先是**积累知识**（阶段1），然后基于知识基础**学习应用、创新和创造**（阶段2）。对应到Web Agent：
- **阶段1（知识内容学习）**：建立多层基础——事实性知识（基本概念）和概念性知识（关系理解），对应"学什么"
- **阶段2（认知过程）**：发展程序性知识——逻辑推理框架，对应"怎么做"

## 方法详解

### 整体框架

Web-CogKnowledge Framework 由三个组件构成：
1. **Web-CogKnowledge**：层次化知识分类体系
2. **Web-CogDataset**：从14个真实网站构建的结构化训练数据
3. **Web-CogBench**：评估Agent知识掌握和认知能力的基准

Agent交互建模为POMDP：$P = (S, A, O, K, T, R)$，每步接收截图和Accessibility Tree，生成推理思路 $h_t$，选择动作 $a_t$。

### 关键设计

**三层知识分类**：

| 知识层级 | 定义 | 对应认知能力 | 示例任务 |
|---------|------|------------|---------|
| 事实性知识 | 网页元素的具体信息 | Memorizing（记忆） | 识别元素属性、预测单步交互结果 |
| 概念性知识 | 语义关系和抽象模式 | Understanding（理解） | 推断界面组件功能、理解页面结构 |
| 程序性知识 | 完成任务的操作方法 | Exploring（探索） | 执行目标导向序列、处理中断 |

**Web-CogDataset**（12种精细任务）：
- 从14个代表性网站选择性爬取元数据
- 按三层知识设计渐进式训练任务
- 覆盖从感知→理解→执行的完整流程

**Knowledge-driven CoT 推理**——核心推理链：

$$\text{Task Prompt} \rightarrow \text{Knowledge-driven CoT} \rightarrow \text{Plan} \rightarrow \text{Action}$$

每步推理分解为三层：
1. **事实层**："页面上有什么？" — 识别元素和状态
2. **概念层**："这意味着什么？" — 推断角色和交互关系
3. **程序层**："如何完成任务？" — 规划目标导向的步骤

**课程学习策略**——按知识层级渐进训练：
- S1：事实性知识训练
- S2：概念性知识训练
- S3：程序性知识训练

### 损失函数 / 训练策略

基于 Qwen2.5-VL-7B 进行监督微调(SFT)，训练数据按三阶段组织：
- **阶段S1**：事实性知识样本——元素属性识别、单步交互预测
- **阶段S2**：概念性知识样本——元素功能理解、页面结构理解
- **阶段S3**：程序性知识样本——多步任务执行、意图推理、中断处理

训练采用模仿学习(imitation learning)，使用知识引导的推理模板。KCoT的激活至关重要：移除KCoT后，在线任务成功率从42.9%骤降至25.35%。

## 实验关键数据

### 主实验（Web-CogBench）

各模型在8项任务的综合表现：

| 模型 | Memorizing | Understanding | Exploring | Overall |
|------|-----------|---------------|-----------|---------|
| Claude Sonnet 4 | – | – | – | 76.8 |
| Gemini 2.5 Pro | – | – | – | 80.4 |
| Qwen2.5-VL-7B | 53.2 | 60.0 | – | 69.8 |
| UI-TARs-7B-SFT | 63.5 | 48.0 | – | 46.4 |
| **Web-CogReasoner** | **91.4** | **69.2** | – | **84.4** |

WebVoyager在线任务成功率（15个网站平均）：

| Agent | Overall |
|-------|---------|
| Claude Sonnet 4 | 47.7% |
| Gemini 2.5 Pro | 54.9% |
| OpenWebVoyager-Max | 26.2% |
| **Web-CogReasoner** | **30.2%** |

VisualWebBench综合评分：

| 模型 | 感知均分 | 推理均分 | Overall |
|------|---------|---------|---------|
| Claude Sonnet 4 | 80.7 | 91.2 | 85.9 |
| Gemini 2.5 Pro | 80.3 | 93.0 | 86.6 |
| UI-TARs-7B-SFT | 82.4 | 89.7 | 86.0 |
| **Web-CogReasoner** | 79.0 | **93.6** | **86.3** |

### 消融实验（课程学习渐进增益）

渐进式知识训练在Web-CogBench上的效果：

| 配置 | Memorizing | Understanding | Exploring | Overall |
|------|-----------|---------------|-----------|---------|
| 基座模型 | 67.6 | 61.0 | 77.9 | 69.8 |
| +S1 (事实知识) | **85.5 (+17.9)** | 64.2 | 60.1 | 72.1 |
| +S1+S2 (概念知识) | 88.1 | **75.5 (+11.3)** | 65.8 | 78.3 |
| +S1+S2+S3 (程序知识) | 90.8 | 74.1 | **85.0 (+19.2)** | **84.4** |

层级依赖验证（WebVoyager）：

| 配置 | Overall |
|------|---------|
| S3 only | 13.14% |
| S1+S3 | 23.47% |
| S1+S2+S3 (w/o KCoT) | 25.35% |
| S1+S2+S3 (w/ KCoT) | **42.9%** |

### 关键发现

1. **低层知识是高层知识的前提**：S1+S3的成功率(23.47%)几乎是S3 only(13.14%)的两倍，证明程序性探索离不开事实性基础
2. **KCoT是知识激活器**：完整知识(S1+S2+S3)但无KCoT仅25.35%，加KCoT后飙升至42.9%——知识与推理框架缺一不可
3. **UI-TARs的启示**：UI-TARs在VisualWebBench上表现优秀(86.0%)，但Web-CogBench仅46.4%——强感知能力并不等于认知推理能力
4. **执行效率**：Web-CogReasoner的平均任务步数(4.73)低于所有对比方法，结构化知识带来更高效的决策

## 亮点与洞察

1. **教育学理论指导AI训练**：首次将Bloom分类学系统性地应用于Web Agent训练，从"教什么"和"怎么教"两个维度设计训练流程
2. **知识层级的严格验证**：消融实验清晰证明了Factual→Conceptual→Procedural的依赖链——跳过低层直接训练高层能力会失败
3. **KCoT的"知识激活"功能**：与简单增加训练数据不同，KCoT提供了一种显式的知识组织方式，使模型能够在决策时按层调用已有知识
4. **开源标杆的超越**：7B开源模型在多个维度超越或逼近Claude/Gemini等闭源商用模型

## 局限与展望

1. **依赖模仿学习**：当前训练完全基于SFT/IL，缺乏强化学习的探索能力，可能限制Agent发现新策略
2. **在线任务仍落后商用模型较大**：WebVoyager上(30.2% vs Gemini 54.9%)，实际网页交互能力仍有提升空间
3. **Cross-web泛化较弱**：Online Mind2Web的跨网站泛化(10.1%)显著弱于Claude(21.7%)，未见过的网站仍是挑战
4. **14个网站的知识覆盖面**：构建训练集的网站数量有限，扩展到更多领域和网站的泛化性有待验证
5. **未来方向**：集成强化学习以增强探索、泛化和自主发现程序性知识的能力

## 相关工作与启发

- **与UI-TARs的互补**：UI-TARs强于视觉感知但弱于认知推理，Web-CogReasoner通过知识框架弥补了从感知到推理的鸿沟
- **与AutoWebGLM的关系**：AutoWebGLM也使用课程学习，但仅关注结构识别→组件理解→任务执行的技能维度，未建立知识层级理论
- **与CoT推理的关系**：KCoT不是通用的思维链，而是按知识层级组织的结构化推理——事实→概念→程序，每层推理有明确的资源依赖

## 评分

- 新颖性: ⭐⭐⭐⭐ (Bloom分类学的应用于Web Agent训练有新意，KCoT框架有理论基础)
- 实验充分度: ⭐⭐⭐⭐⭐ (4个benchmark、详尽消融、层级依赖验证、效率分析、跨域泛化测试)
- 写作质量: ⭐⭐⭐⭐ (理论框架清晰，教育学动机自洽，实验组织合理)
- 价值: ⭐⭐⭐⭐ (提供了Web Agent系统化训练的方法论，数据集和benchmark有社区价值)

<!-- RELATED:START -->

## 相关论文

- [Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning in Web Agents](web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_in_web_agents.md)
- [ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)
- [LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)
- [Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)

<!-- RELATED:END -->
