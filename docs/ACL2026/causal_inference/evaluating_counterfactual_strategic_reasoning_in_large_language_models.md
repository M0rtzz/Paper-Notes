---
title: >-
  [论文解读] Evaluating Counterfactual Strategic Reasoning in Large Language Models
description: >-
  [ACL2026][因果推理][反事实博弈] 本文用重复囚徒困境和石头剪刀布的标签扰动、收益扰动与联合反事实版本评测 LLM 的策略适应能力，发现很多模型在熟悉博弈中看似会玩，但在收益结构改变后仍沿用模板化策略。
tags:
  - "ACL2026"
  - "因果推理"
  - "反事实博弈"
  - "策略推理"
  - "Prisoner's Dilemma"
  - "Rock-Paper-Scissors"
  - "opponent comprehension"
---

# Evaluating Counterfactual Strategic Reasoning in Large Language Models

**会议**: ACL2026  
**arXiv**: [2603.19167](https://arxiv.org/abs/2603.19167)  
**代码**: https://github.com/dimjimitris/llm_gm_thesis  
**领域**: LLM推理 / 博弈评测 / 反事实鲁棒性  
**关键词**: 反事实博弈, 策略推理, Prisoner's Dilemma, Rock-Paper-Scissors, opponent comprehension

## 一句话总结
本文用重复囚徒困境和石头剪刀布的标签扰动、收益扰动与联合反事实版本评测 LLM 的策略适应能力，发现很多模型在熟悉博弈中看似会玩，但在收益结构改变后仍沿用模板化策略。

## 研究背景与动机
**领域现状**：LLM 已经被大量用于多智能体协作、竞争和博弈模拟，研究者常通过囚徒困境、石头剪刀布、匹配硬币等结构化博弈观察模型是否能合作、竞争、识别对手策略，并接近均衡行为。

**现有痛点**：常规博弈评测容易高估模型能力。模型可能记住“囚徒困境应该合作/背叛”“石头剪刀布应该随机化”这类模板，而不是真的根据 payoff matrix 重新计算策略。一旦动作标签被改名，或收益结构被反事实修改，流畅的解释未必能转化为正确行动。

**核心矛盾**：真正的策略推理要求模型对当前环境的标签、收益和历史交互做条件化更新；而 LLM 的行为可能更多来自预训练中见过的 canonical game pattern。二者在默认游戏中不易区分，必须通过反事实干预把表面识别和激励敏感性拆开。

**本文目标**：构造一个紧凑、可控、可复现实验框架，分别诊断 label robustness、payoff sensitivity、opponent modeling 和 token-normalized efficiency，判断模型是在理解当前博弈，还是在复现熟悉模板。

**切入角度**：作者选择两个互补游戏：囚徒困境考察合作与背叛之间的动态适应，石头剪刀布考察随机化、模式利用和三动作均衡。随后对二者施加标签扰动、收益扰动和联合扰动，让模型必须重新解释动作意义和收益结构。

**核心 idea**：用重复博弈中的反事实标签/收益干预，把 LLM 的“会说策略”与“能按新激励执行策略”区分开。

## 方法详解
本文方法本质上是一个行为评测框架。它不训练模型，而是把 LLM 放到多轮交互游戏中，让它与另一个 LLM 或算法型对手重复博弈，并记录动作、收益、对手理解速度、合作率和 token 效率。

### 整体框架
输入是一组 LLM、prompting 策略、对手类型和游戏定义。每个实验把一个 LLM 作为玩家，与同模型实例或算法型玩家对战。囚徒困境重复 16 轮，石头剪刀布重复 24 轮；每个非 self-consistency 玩家重复 5 次，self-consistency 玩家重复 2 次。

框架包含四类游戏设置：默认游戏、label-based counterfactual、payoff-based counterfactual 和 joint counterfactual。label-based 设置只改变动作名称或支配关系描述，收益不变；payoff-based 设置改变 payoff matrix，让原本的 equilibrium 不再适用；joint 设置同时改变标签和收益，形成最高压力测试。

模型侧包含 zero-shot、Chain-of-Thought、Solo Performance Prompt 以及 self-consistency 等提示形式。对手侧既有 LLM，也有 SREP、PP、MF/TFT、AP 等算法策略。这样可以同时考察模型在 LLM-LLM 协调、可预测对手利用和自适应对手对抗中的表现。

### 关键设计
1. **反事实博弈构造**:

    - 功能：把默认游戏拆成表面标签变化和深层收益变化两类压力源。
    - 核心思路：在 PD 中，用 Stag Hunt 作为 payoff-based counterfactual，把严格背叛优势改成协调问题；同时也只把 C/D 改名为 Stag/Hare，测试模型是否被动作名称影响。在 RPS 中，把某些胜负收益放大，使均匀混合策略不再是均衡。
    - 设计动机：只看默认 PD/RPS 很难知道模型是否真的读取 payoff。反事实版本能让“标签锚定”和“激励刚性”暴露出来。

2. **Opponent Comprehension 指标**:

    - 功能：衡量模型第几轮开始稳定理解并利用对手策略。
    - 核心思路：定义 $m$ 为最早轮次，使得从该轮到游戏结束，LLM 在至少 $t_p=90\%$ 的后续轮次中获得不低于对手的 payoff。$m$ 越小，说明模型越早完成对手建模；超过游戏长度则表示没有稳定理解。
    - 设计动机：总分只能说明最终赚了多少，不能说明模型是早早理解对手，还是靠后期偶然恢复。$m$ 把动态适应速度单独拉出来。

3. **性能与效率联合评估**:

    - 功能：区分“更会玩”和“更会花 token 解释”。
    - 核心思路：除了 total points、cooperation/action distribution、failure rate，作者还定义 efficiency 为 $\textit{points}/\textit{tokens}\times c$，默认 $c=1000$。
    - 设计动机：推理型模型可能输出更多 chain-of-thought，但额外 deliberation 不一定带来更快适应。效率指标能识别 reasoning-overhead mismatch。

### 损失函数 / 训练策略
本文没有训练模型，重点是评测设计。囚徒困境默认 payoff 为 $(C,C)=(4,4)$、$(C,D)=(1,6)$、$(D,C)=(6,1)$、$(D,D)=(2,2)$，16 轮累计分数范围为 16 到 96；RPS 每轮胜/负/平为 $1/-1/0$，24 轮累计分数范围为 -24 到 24。RPS payoff-based counterfactual 中，Rock-Paper 组合的胜负幅度放大为 3，理论均衡从均匀分布变为 $\pi^*(R)=0.2,\pi^*(P)=0.2,\pi^*(S)=0.6$。

## 实验关键数据

### 主实验
| 设置 | 指标 | 代表结果 | 解释 | 结论 |
|--------|------|------|----------|------|
| 默认 PD vs SREP | Total points | SREP 总是背叛时，持续 $(D,D)$ 的基线为 32 分；多数 LLM 聚集在约 30 分 | 模型通常能识别持续背叛并近似最优回应 | 简单算法对手较容易 |
| 默认 PD vs LLM | Total points | Claude 3.5/3.7 与 Llama 3.3-70B 在多种 prompting 下达到 64 分 | 64 分对应 16 轮完全互相合作 | 一些模型在 LLM-LLM 中能稳定协调 |
| 默认 PD | 不稳定案例 | Mistral Large 在 SREP 下从 18.6±10.6 到 29.8±2.2；Claude 4/DeepSeek R1 在 LLM-LLM 中为 31.4±0.0 到 49.4±15.5 | 弱模型或过度推理模型可能更不稳定 | 能力强不等于策略稳定 |
| 默认 RPS | Opponent comprehension | Claude 3.5 Sonnet v2 zero-shot 对 ZS 为 10.6±13.1，对 SPP 为 21.4±4.6，对 CoT 为 19.6±5.6 | RPS 中对手建模更慢、更接近 24 轮 horizon | 三动作无主导策略更难 |
| RPS payoff counterfactual | 理论均衡 | 从均匀 $(1/3,1/3,1/3)$ 变为 $(0.2,0.2,0.6)$ | 仍接近均匀分布说明没有按新收益重算策略 | payoff perturbation 最能暴露模板化 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Label-only counterfactual | 退化通常中等 | 强模型能保持较稳定，Mistral 等更容易因动作改名而波动 |
| Payoff-based counterfactual | 退化更强 | 需要重新计算激励，尤其在 RPS 中要从均匀随机化转向偏置混合策略 |
| Joint counterfactual | 压力最大 | 标签和收益同时变化时，near-horizon comprehension 与高方差更常见 |
| CoT / thinking variants | 效果不一致 | 对部分强模型有帮助，但 Claude 4、DeepSeek R1 等场景中会出现 overthinking 或不信任倾向 |
| Self-consistency | 降低方差但不改根本倾向 | 它常强化已有行为模式，而不是把错误策略变成正确策略 |

### 关键发现
- Payoff-based counterfactual 比 label-only 更有诊断力，因为它迫使模型重新计算收益结构，而不是只处理动作名称变化。
- 默认游戏表现不能代表反事实鲁棒性。Claude 3.7 整体最稳定，Claude 4 在 RPS 上强但反事实稳定性混合，Llama 3.3 在 PD 合作场景稳定但 RPS/payoff shift 较弱。
- 思考更多不一定更好。thinking-enabled 变体在一些设置下增加 token 消耗，却没有等比例提升 total points 或 opponent comprehension。
- RPS 比 PD 更能暴露 delayed opponent modeling，因为没有简单的合作收敛点，模型必须维持近均衡行为或识别可利用模式。

## 亮点与洞察
- 这篇论文的价值在于评测设计很“干净”：标签变化、收益变化和联合变化分别对应不同失败模式，能够把模板记忆、标签锚定、激励刚性拆开看。
- Opponent comprehension 比最终分数更有解释力。很多模型可能最后分数尚可，但如果 $m$ 很晚，说明它是在交互中慢慢撞出来，而不是一开始就理解对手。
- RPS payoff counterfactual 的 $(0.2,0.2,0.6)$ 很关键。它说明“随机化”并不是永远正确；在改变收益后继续均匀随机，反而是 canonical equilibrium persistence。
- 论文提醒我们，在 agent evaluation 中，强模型的 chain-of-thought 可能增加防御性、怀疑或探索噪声。推理过程更长并不自动意味着策略执行更稳。

## 局限与展望
- 评测只覆盖两人、同步、固定 payoff 的重复游戏，和真实多方谈判、市场、拍卖或开放式协作相比，生态有效性有限。
- 算法对手和 payoff 结构是预设的，更多自适应对手、多智能体群体博弈、非完全信息博弈可能产生不同结论。
- 所有指标都来自可观测动作和 token，用它们推断“理解”是行为层面的，不等于解释模型内部机制。
- 模型、prompt 和 counterfactual 类型仍不穷尽。未来可以加入更多开源/闭源模型、更复杂收益变换、自然语言规则歧义，以及对内部 reasoning trace 的一致性分析。

## 相关工作与启发
- **vs 常规博弈评测**: 常规 PD/RPS 只能观察模型是否会玩熟悉游戏，本文通过反事实扰动检验模型是否真的根据当前规则更新策略。
- **vs 静态反事实问答**: 许多 counterfactual benchmark 是一次性输入输出，本文把反事实放入重复交互，能观察适应速度和历史依赖。
- **vs 多智能体协作评测**: 常规 agent benchmark 更关注任务成功率，本文强调收益、对手建模、效率和失败率的多维诊断，适合作为 agentic LLM 的小型压力测试。
- **启发**：设计 LLM benchmark 时，应尽量加入规则保持但标签变化、标签保持但收益变化的对照组，否则容易把“熟悉模板执行”误判为“抽象推理”。

## 评分
- 新颖性: ⭐⭐⭐⭐ 用反事实重复博弈诊断 LLM 策略鲁棒性，问题设置紧凑且有解释力。
- 实验充分度: ⭐⭐⭐⭐ 覆盖多个 frontier LLM、prompting 策略、对手类型和指标，但游戏类型仍偏少。
- 写作质量: ⭐⭐⭐⭐ 主文逻辑清楚，附录数值充分；部分表格非常大，读者需要结合文字总结理解。
- 价值: ⭐⭐⭐⭐ 对 LLM agent 评测、策略推理和反事实鲁棒性研究都有直接借鉴价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Counterfactual Reasoning for Steerable Pluralistic Value Alignment of Large Language Models](../../NeurIPS2025/causal_inference/counterfactual_reasoning_for_steerable_pluralistic_value_alignment_of_large_lang.md)
- [\[ACL 2025\] Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models](../../ACL2025/causal_inference/counterfactual-consistency_prompting_for_relative_temporal_understanding_in_larg.md)
- [\[NeurIPS 2025\] Revealing Multimodal Causality with Large Language Models](../../NeurIPS2025/causal_inference/revealing_multimodal_causality_with_large_language_models.md)
- [\[ACL 2025\] On the Reliability of Large Language Models for Causal Discovery](../../ACL2025/causal_inference/llm_causal_discovery_reliability.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/causal_inference/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)

</div>

<!-- RELATED:END -->
