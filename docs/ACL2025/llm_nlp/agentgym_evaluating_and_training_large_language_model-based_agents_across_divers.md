---
title: >-
  [论文解读] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments
description: >-
  [ACL 2025][LLM 其他][通用Agent] 本文提出AgentGym框架，包含14种交互环境、89类任务、标准化轨迹数据集和评测基准，并提出AgentEvol自我进化算法，让LLM Agent通过跨环境探索和学习实现从模仿到自主进化，性能可达到SOTA模型水平。 领域现状：构建能够处理多样化任务并在不同环境中自我…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "通用Agent"
  - "自我进化"
  - "多环境训练"
  - "行为克隆"
  - "AgentEvol"
---

# AgentGym: Evolving Large Language Model-based Agents across Diverse Environments

**会议**: ACL 2025  
**arXiv**: [2406.04151](https://arxiv.org/abs/2406.04151)  
**代码**: [https://github.com/WooooDyy/AgentGym](https://github.com/WooooDyy/AgentGym)  
**领域**: LLM/NLP  
**关键词**: 通用Agent、自我进化、多环境训练、行为克隆、AgentEvol

## 一句话总结
本文提出AgentGym框架，包含14种交互环境、89类任务、标准化轨迹数据集和评测基准，并提出AgentEvol自我进化算法，让LLM Agent通过跨环境探索和学习实现从模仿到自主进化，性能可达到SOTA模型水平。

## 研究背景与动机

**领域现状**：构建能够处理多样化任务并在不同环境中自我进化的通用Agent是AI社区的长期目标。LLM被认为是构建此类Agent的理想基础，因其具备强大的泛化能力。当前的LLM Agent构建方法主要有两条路线。

**现有痛点**：第一条路线通过行为克隆（BC）让Agent模仿专家轨迹，但需要大量人工标注、成本高、难以扩展，且由于缺乏对环境的充分探索导致性能和泛化能力受限。第二条路线让Agent在环境中自主探索和学习，但通常局限于单一环境中的特定任务，训练出的是"专家Agent"而非"通才Agent"。

**核心矛盾**：要训练通用Agent需要"多样化环境+高质量轨迹+有效进化方法"三位一体，但现有工作缺乏统一的多环境交互平台，也没有有效的跨环境进化方法。

**本文目标**：构建一个完整的框架来支持通用LLM Agent的训练和评测，并探索Agent跨任务、跨环境的自我进化能力。

**切入角度**：类比人类学习过程——先通过模仿获取基本知识和技能，再通过与不同环境的交互探索来自主学习和适应新任务。

**核心 idea**：提出AgentGym框架（多环境平台+轨迹数据集+评测基准）和AgentEvol算法（跨环境自我进化），实现LLM Agent从行为克隆到交互式学习的进化过程。

## 方法详解

### 整体框架
AgentGym包含三个核心组件：（1）交互平台——集成14种Agent环境，通过HTTP服务提供统一API，支持实时交互、轨迹采样和在线评估；（2）数据和基准——包括扩展指令集、AgentEval评测基准和AgentTraj高质量轨迹集；（3）AgentEvol进化算法——基座Agent通过行为克隆训练后，在多环境中探索并从经验中学习。

### 关键设计

1. **统一交互平台**:

    - 功能：提供14种环境（Web浏览、具身任务、科学实验等）的标准化交互接口
    - 核心思路：每个环境部署为独立HTTP服务，客户端提供封装的统一接口。所有环境共享相同的观察/动作空间规范，Agent使用ReAct格式（先思考再行动）与环境交互。支持并发和实时反馈，使Agent可以同时在多个环境中探索
    - 设计动机：现有Agent框架要么环境数量有限（AgentBench 8个），要么不支持交互式训练。统一平台是实现跨环境进化的基础设施

2. **AgentTraj轨迹数据集**:

    - 功能：提供高质量的专家轨迹用于行为克隆基础训练
    - 核心思路：使用众包和SOTA模型（如GPT-4）在多环境中收集轨迹。通过self-instruct和指令进化方法扩展指令多样性。按统一格式整理轨迹，形成AgentTraj（基础集，约5000条轨迹）和AgentTraj-L（扩展集，约15000条轨迹）。从中选择多样且有挑战性的子集构建AgentEval评测基准
    - 设计动机：Agent在复杂环境中从零开始学习极其低效，需要先通过模仿获取基本的指令遵循能力和先验知识

3. **AgentEvol自我进化算法**:

    - 功能：让基座Agent通过环境交互自我提升，超越行为克隆的性能瓶颈
    - 核心思路：分为三个阶段：（1）探索——Agent在多个环境中尝试新任务指令，收集交互轨迹；（2）筛选——使用环境奖励信号过滤成功的轨迹；（3）学习——在筛选后的高质量轨迹上进行监督微调。关键创新是引入"动态采样"策略，根据环境难度和Agent当前能力自适应调节每个环境的探索比例。此外，使用MCTS启发的搜索在探索阶段增强轨迹多样性
    - 设计动机：行为克隆受限于专家数据的质量和数量，而自我进化通过探索可以发现数据中未覆盖的解决策略，类似于RL中的off-policy learning

### 损失函数 / 训练策略
行为克隆阶段使用标准的交叉熵损失在AgentTraj上训练。AgentEvol阶段使用迭代式的DAgger-like策略，交替执行探索-筛选-训练循环。筛选使用环境提供的二值奖励（成功/失败）。

## 实验关键数据

### 主实验

| 模型 | WebShop | ALFWorld | SciWorld | BabyAI | TextCraft | 平均 |
|------|---------|----------|----------|--------|-----------|------|
| GPT-4 | 52.3 | 78.0 | 43.2 | 90.0 | 18.0 | 56.3 |
| Lemur-70B-Chat | 38.5 | 18.0 | 19.5 | 81.1 | 6.0 | 32.6 |
| AgentGym-BC (Llama3-8B) | 45.2 | 62.0 | 34.8 | 88.9 | 12.0 | 48.6 |
| AgentGym-Evol (Llama3-8B) | **54.1** | **76.0** | **42.5** | **92.2** | **22.0** | **57.4** |

### 消融实验

| 配置 | 平均性能 | 说明 |
|------|---------|------|
| AgentEvol (Full) | 57.4 | 完整进化 |
| 仅BC（AgentTraj） | 48.6 | 行为克隆基线 |
| 仅BC（AgentTraj-L） | 53.8 | 更大数据集的BC上限 |
| AgentEvol 单环境进化 | 51.2 | 仅在一个环境中进化 |
| AgentEvol 无动态采样 | 54.6 | 均匀探索各环境 |

### 关键发现
- AgentEvol在8B参数量下超越GPT-4在Agent任务上的表现，证明了"小模型+进化"路线的可行性
- 跨环境进化（57.4）显著优于单环境进化（51.2），证明环境多样性对泛化至关重要
- 动态采样策略带来约2.8分提升，说明根据环境难度调节探索比例很有效
- 进化的AgentEvol（57.4）甚至超过了使用更多数据的BC上限（AgentTraj-L的53.8），证明自我探索确实可以发现专家轨迹未覆盖的策略

## 亮点与洞察
- AgentGym的"平台+数据+算法"三位一体设计为Agent社区提供了完整的基础设施。类似于NLP领域的GLUE/SuperGLUE，AgentEval有望成为Agent能力评测的标准基准
- AgentEvol的"模仿→探索→学习"范式很好地类比了人类的学习过程，技术上将DAgger/RFT思想迁移到Agent训练中
- 14种环境的统一化接口设计使得新环境的集成非常方便，促进了社区贡献

## 局限与展望
- 所有环境都是文本交互的，缺少视觉或多模态环境
- AgentEvol依赖环境提供的二值奖励信号，对于奖励稀疏的环境（如长期规划任务）可能效果有限
- 当前进化只迭代了2-3轮，更多轮次的进化效果和稳定性尚未充分探索
- 8B模型的Agent能力仍有限，在复杂推理和长程记忆方面与大模型有差距

## 相关工作与启发
- **vs AgentBench**: AgentBench只提供评测不提供训练，AgentGym同时支持评测和训练
- **vs AgentOhana**: AgentOhana收集了多环境轨迹但无交互平台，AgentGym提供完整的交互式训练支持
- **vs Pangu-Agent**: Pangu-Agent仅支持单环境进化，AgentGym的AgentEvol探索跨环境进化

## 评分
- 新颖性: ⭐⭐⭐⭐ 框架层面的贡献突出，AgentEvol算法相对常规但在Agent场景的验证有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 14环境89任务的全面评测，消融方分析充分
- 写作质量: ⭐⭐⭐⭐⭐ 论文组织清晰，图表精美，技术细节完整
- 价值: ⭐⭐⭐⭐⭐ 为Agent社区提供了急需的基础设施和基准，开源贡献大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents](simulating_diverse_students.md)
- [\[ACL 2025\] Geometric Signatures of Compositionality Across a Language Model's Lifetime](geometric_compositionality_lifetime.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ICLR 2026\] Evaluating Text Creativity across Diverse Domains: A Dataset and Large Language Model Evaluator](../../ICLR2026/llm_nlp/evaluating_text_creativity_across_diverse_domains_a_dataset_and_large_language_m.md)
- [\[ACL 2025\] EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents](escapebench_creative_agent.md)

</div>

<!-- RELATED:END -->
