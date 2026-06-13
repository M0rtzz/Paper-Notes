---
title: >-
  [论文解读] AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling
description: >-
  [NeurIPS 2025][机器人][Theory of Mind] AutoToM 实现完全自动化的基于模型的心智理论推理——无需人工指定 agent 模型，自动提出贝叶斯网络结构并执行贝叶斯逆规划，通过推理不确定性驱动的迭代模型调整（添加心智变量或扩展时间步）…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "Theory of Mind"
  - "Bayesian inverse planning"
  - "automated agent modeling"
  - "mental inference"
  - "LLM"
---

# AutoToM: Scaling Model-based Mental Inference via Automated Agent Modeling

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.15676](https://arxiv.org/abs/2502.15676)  
**代码**: 有  
**领域**: 心智理论 / LLM Agent  
**关键词**: Theory of Mind, Bayesian inverse planning, automated agent modeling, mental inference, LLM

## 一句话总结
AutoToM 实现完全自动化的基于模型的心智理论推理——无需人工指定 agent 模型，自动提出贝叶斯网络结构并执行贝叶斯逆规划，通过推理不确定性驱动的迭代模型调整（添加心智变量或扩展时间步），在5个ToM benchmark上以82.43%平均准确率超越GPT-4o(63.39%)、o3-mini(73.94%)等SOTA模型。

## 研究背景与动机

**领域现状**：Theory of Mind（ToM）——理解他人心智状态（目标、信念、意图）的能力——是社会智能的基石。机器ToM有两大方法流派：(a) 直接prompt LLM进行推理（SimToM、SymbolicToM等），灵活但在复杂场景下会犯系统性错误（尤其是长上下文、多agent递归推理场景）；(b) 基于模型的贝叶斯逆规划（BIP），通过构建agent的生成模型然后反向推断心智状态，鲁棒但需要人工定义agent模型（包括心智变量集合和因果结构图），泛化差。

**现有痛点**：BIP-ALM和LIMP等将BIP与LLM结合的先驱工作虽然提升了鲁棒性，但仍然要求手工指定：(a) 需要哪些心智变量（目标、信念、观察等）；(b) 变量间的因果关系（MDP/POMDP/I-POMDP结构选择）；(c) 需要考虑哪些时间步。这些手工设计限制了其适用于特定领域，无法处理开放式的ToM问题。

**核心矛盾**：LLM灵活但不鲁棒——即使是o3-mini这样的大reasoning模型在复杂ToM中也会犯系统性错误（长上下文遗忘、递归推理崩溃）；BIP鲁棒但不灵活——需要人工为每个领域设计agent模型。**核心 idea**：让LLM自动发现合适的agent模型结构，然后在该模型上做自动化的贝叶斯推理——模型发现的灵活性+贝叶斯推理的鲁棒性=可扩展的开放式机器ToM。

## 方法详解

### 整体框架
AutoToM由两个核心组件构成自我改进循环：(1) Automated Bayesian Inverse Planning——在给定agent模型上用LLM作为计算后端执行贝叶斯推理；(2) Automated Agent Model Discovery——根据推理不确定性自动提出和调整agent模型。流程：信息提取→初始模型提出→自动BIP推理→评估模型效用→效用不足则调整模型→再推理→直到置信度足够。

### 关键设计

1. **自动贝叶斯逆规划（Automated BIP）**:

    - 功能：在任意给定的agent模型（贝叶斯网络）上执行完整的推理过程
    - 核心思路：两步走——(a) 假设采样：用LLM为每个潜在心智变量生成一小组高质量假设值（类似摊销推理），结合question和可观察变量引导采样，再通过假设缩减去掉不合理的假设（评估局部条件概率）；(b) 贝叶斯推理：用LLM估计贝叶斯网络中每个局部条件概率P(子节点|父节点)，然后通过显式计算对联合分布求边缘化得到目标变量的后验P(q|X)
    - 设计动机：不同于BIP-ALM和LIMP假设固定模型结构和手工变量表示，AutoToM的BIP对任意图结构和任意变量表示都适用。支持任意阶递归推理（通过I-POMDP的嵌套信念建模），不需要领域特定实现

2. **自动Agent模型发现（Automated Model Discovery）**:

    - 功能：自动构建最适合当前ToM问题的agent模型，消除人工模型设计的瓶颈
    - 核心思路：模型M=(V^{ts:t}, X^{ts:t})由心智变量集合和可观察变量集合唯一定义。模型效用U(M,q) = R(M,q) - C(M)，其中R=-H(P(q|X))（推理结果的负熵=置信度）、C=α|M|（复杂度惩罚）。三个子模块：(a) 信息提取——用LLM从上下文中提取可观察变量（状态、动作、话语）沿时间线排列；(b) 初始模型提出——提出最小复杂度模型（仅包含回答问题最必要的变量），从最后一个时间步开始；(c) 迭代调整——变量调整（引入新心智变量如belief/observation/interactive state）和时间步调整（向前扩展更多时间步），每次选择效用增益最大的调整
    - 设计动机：不确定性驱动——只有当推理不够置信时才增加模型复杂度，避免了过度建模（浪费计算）和欠建模（精度不足）的两难。限制在MDP/POMDP/I-POMDP的变量类型空间内保证模型能解释agent行为

3. **统一形式化与可扩展设计**:

    - 功能：提供跨领域通用的ToM推理框架
    - 核心思路：将BIP统一形式化为P(V^{ts:t}|X^{ts:t})的推理问题，覆盖MDP（有目标、全观测）、POMDP（部分观测+信念维护）、I-POMDP（多agent递归推理）等模型，各自只是变量集V和X的不同配置
    - 设计动机：先前方法（BIP-ALM、LIMP）各自为特定模型类型定制实现，无法跨类型泛化。统一形式化使得一套推理引擎可处理所有模型变体

### 损失函数 / 训练策略
AutoToM无需训练参数——完全基于LLM的上下文推理（in-context inference）。关键超参数：模型效用阈值U_min（决定何时停止模型调整）、复杂度权重α。

## 实验关键数据

### 主实验（5个ToM benchmark平均准确率）

| 方法 | ToMi | BigToM | MMToM-QA | MuMA-ToM | Hi-ToM | 平均 |
|------|------|--------|----------|----------|--------|------|
| GPT-4o | 77.0 | 82.4 | 44.0 | 63.6 | 50.0 | 63.4 |
| o3-mini-high | 73.1 | 86.9 | 64.7 | 70.0 | 75.0 | 73.9 |
| Gemini 2.0 Flash Thinking | 78.0 | 82.8 | 54.0 | 82.6 | 73.5 | 74.2 |
| DeepSeek-R1 | 89.4 | 86.3 | 49.7 | 63.4 | 56.5 | 69.1 |
| BIP-ALM | 55.6 | 50.3 | 56.2 | 33.9 | 14.5 | 42.1 |
| LIMP | 44.6 | 61.7 | 55.3 | 76.6 | 6.5 | 48.9 |
| **AutoToM (GPT-4o)** | **88.3** | **86.9** | **83.0** | **81.4** | **72.5** | **82.4** |

### 消融实验

| 配置 | 平均准确率 | 相对计算量 | 说明 |
|------|----------|-----------|------|
| Full AutoToM | 82.4 | 1.0× | 最优效果 |
| w/o hypothesis reduction | ~80 | ~1.3× | 准确率微降+计算增加 |
| w/ POMDP固定 | ~78 | ~1.1× | 不灵活导致部分场景过度建模 |
| w/o variable adjustment | ~76 | ~0.8× | 无法适应需要信念/观察的场景 |
| w/ last timestep only | ~74 | ~0.6× | 丢失历史上下文 |
| w/ all timesteps | ~79 | ~1.5× | 不必要的计算开销 |

### 关键发现
- AutoToM以GPT-4o为后端（82.4%）大幅超越GPT-4o自身（63.4%）——结构化推理>纯LLM推理
- 在最具挑战性的MMToM-QA（长上下文+多模态）上提升最大：83.0% vs GPT-4o 44.0%、o3-mini 64.7%
- 随着上下文长度、agent数量和递归深度增加，AutoToM的优势越来越大（图4），而大reasoning模型的性能波动剧烈
- 换用不同LLM后端（Qwen3-235B、DeepSeek-V3、Gemini-2.5-Flash）仍一致超越对应LLM本身，验证了框架的后端无关性
- 多次运行的统计可靠性：MMToM-QA上3次运行均值82.56%±0.45%

## 亮点与洞察
- **自动模型发现是最核心的贡献**——将基于模型的ToM从"需要认知科学家手工建模"提升为"全自动化系统"，真正实现了开放式ToM推理
- **不确定性驱动的模型扩展**设计非常优雅——最小起步，按需扩展，兼顾效率和效果。本质是对model complexity做了自适应搜索
- **LLM作为概率推理后端**而非直接推理器——这一角色定位是关键洞察。LLM不擅长系统性推理，但擅长估计局部条件概率（给定具体场景评估某变量取某值的似然性）
- 产生类人的置信度估计（不只是答案，还有确信程度）——这对embodied assistance等下游任务至关重要

## 局限与展望
- 模型发现质量仍受限于LLM后端的常识推理能力——如果LLM无法正确识别需要哪些心智变量，模型可能不合适
- 假设采样和局部条件概率估计的准确性依赖LLM的上下文理解，对于非常规或反直觉的agent行为可能不准
- 多agent高阶递归推理的计算成本随递归深度指数增长
- 当前模型发现限制在MDP/POMDP/I-POMDP的变量类型空间内，可能遗漏非标准心智变量

## 相关工作与启发
- **LLM+概率推理的范式**：AutoToM展示了LLM作为概率推理后端（估计似然+生成假设）而非端到端推理器的巨大潜力，可推广到其他需要结构化推理的领域
- **自动建模与LLM**：与Li等人的统计模型自动构建、Wang等人的假设生成+程序验证思路呼应，但首次应用于BIP/ToM领域
- **human-like AI**：AutoToM产生的置信度估计与人类行为实验数据匹配，暗示其推理机制可能捕获了人类ToM的某些计算特征

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 完全自动化的模型化ToM是突破性贡献，自动模型发现+自动BIP的组合前所未有
- 实验充分度: ⭐⭐⭐⭐⭐ 5个benchmark+认知实验+embodied任务+多LLM后端+消融+统计可靠性检验
- 写作质量: ⭐⭐⭐⭐⭐ 统一形式化清晰，图表优秀，实验分析深入
- 价值: ⭐⭐⭐⭐⭐ 对社会智能AI和人机交互有深远影响，框架通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[CVPR 2025\] ShowUI: One Vision-Language-Action Model for GUI Visual Agent](../../CVPR2025/robotics/showui_one_vision-language-action_model_for_gui_visual_agent.md)
- [\[NeurIPS 2025\] Can Agents Fix Agent Issues?](can_agents_fix_agent_issues.md)
- [\[NeurIPS 2025\] Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](benchmarking_egocentric_multimodal_goal_inference_for_assist.md)
- [\[CVPR 2026\] MergeVLA: Cross-Skill Model Merging Toward a Generalist Vision-Language-Action Agent](../../CVPR2026/robotics/mergevla_cross-skill_model_merging_toward_a_generalist_vision-language-action_ag.md)

</div>

<!-- RELATED:END -->
