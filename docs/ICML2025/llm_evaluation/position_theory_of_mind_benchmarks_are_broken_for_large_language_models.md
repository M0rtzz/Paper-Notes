---
title: >-
  [论文解读] Position: Theory of Mind Benchmarks are Broken for Large Language Models
description: >-
  [ICML 2025][LLM评测][Theory of Mind] 这篇 Position Paper 指出当前大多数 LLM Theory of Mind（ToM）基准只测“能否预测他人行为”（Literal ToM），却没有测“能否基于该预测采取最优响应”（Functional ToM），因此会系统性高估模型在真实交互中的适应能力。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "Theory of Mind"
  - "多智能体交互"
  - "功能性适应"
  - "基准可靠性"
---

# Position: Theory of Mind Benchmarks are Broken for Large Language Models

**会议**: ICML 2025  
**arXiv**: [2412.19726](https://arxiv.org/abs/2412.19726)  
**代码**: 无公开仓库（论文未提供）  
**领域**: LLM评测  
**关键词**: Theory of Mind, LLM评测, 多智能体交互, 功能性适应, 基准可靠性

## 一句话总结

这篇 Position Paper 指出当前大多数 LLM Theory of Mind（ToM）基准只测“能否预测他人行为”（Literal ToM），却没有测“能否基于该预测采取最优响应”（Functional ToM），因此会系统性高估模型在真实交互中的适应能力。

## 研究背景与动机

过去两年，大量工作用心理学经典题（例如 Sally-Anne）或问答式任务评估 LLM 的 ToM 能力。
这些任务普遍报告了很高的准确率，进而得出“LLM 具备较强 ToM”的结论。

作者认为这里有一个关键前提被默认了：
对人类而言，预测他人行为与基于预测做决策通常具有“过程一致性”。
但对当前 LLM 而言，这种一致性并不成立。

论文给出的核心反例很直接：
在 100 轮石头剪刀布中，对手永远出 Rock。
很多 LLM 很快能预测“对手下一轮还是 Rock”（Literal ToM 很高）。
但它们自己的动作却仍接近均匀随机，而不是长期选择 Paper 去最大化收益。

因此，作者提出：
如果评测只看预测而不看行为适配，就会把“会说”误当成“会做”。
这也是本文将现有基准称为“broken”的根本理由。

## 方法详解

### 整体框架

论文在多智能体部分可观测随机博弈（POSG）框架下，区分并形式化两类 ToM 指标：

1. Literal ToM：预测其他体行为的误差。
2. Functional ToM：在给定对手策略下，自身策略相对最优响应的遗憾值。

评估流程不是训练新模型，而是固定模型权重，仅通过 prompt 与历史交互进行 in-context 适配。
这与现实部署场景一致：多数线上 LLM 服务不会为每个用户持续微调权重。

### 关键设计 1：Literal ToM（定义 2.1）

Literal ToM 被定义为对手真实动作序列与模型预测动作序列之间的距离。

直观理解：
如果模型总能猜中对方下一步出什么，Literal ToM 分数就高。

但该指标天然不涉及“模型自己下一步应当如何行动”。
所以它只能回答“是否看懂了别人”，不能回答“是否做对了自己”。

### 关键设计 2：Functional ToM（定义 2.2）

Functional ToM 用 T 步遗憾值衡量：

$\Delta_{\text{Functional}} = \sum_{t=1}^{T}(r_t^{i^*}-r_t^{i})$

其中：
$r_t^{i^*}$ 是对当前对手策略的最优响应在第 $t$ 步可获得奖励，
$r_t^{i}$ 是实际策略奖励。

直观理解：
如果模型真的“理解并利用了”对手行为规律，遗憾应接近 0。
遗憾越大，说明行动层面的 ToM 越弱。

### 关键设计 3：桥接指标 $\Delta_{\text{ToM}}$

作者还构造了一个“理性跟随预测”的上界分析：
先用 Literal ToM 预测对手动作，再假设决策器总能理性最大化收益，得到对应遗憾 $\Delta_{\text{ToM}}$。

若出现：
$\Delta_{\text{ToM}} \ll \Delta_{\text{Functional}}$

则说明问题不在“看不准对手”，而在“没有把预测转成行为”。
这正是论文反复观察到的现象。

### 任务与博弈设置

论文使用三类经典重复博弈，覆盖不同激励结构：

1. RPS（Rock-Paper-Scissors）：竞争型。
2. IBS（Iterated Battle of Sexes）：协调/合作型。
3. IPD（Iterated Prisoner’s Dilemma）：混合激励。

对手策略包括：

1. Single Action（恒定动作）。
2. Tit-for-Tat（更动态的响应式策略，见附录扩展分析）。

交互长度统一为 100 步，明显长于很多先前工作，目的是观察是否真正收敛到可适应行为。

### Prompt 家族与比较维度

文中对多类提示策略进行系统比较：

1. Generic：LM、QA、CoT。
2. In-context RL 风格：Plans+Insights、Reflexion。
3. 显式 ToM：Social QA、Social LM（先预测对方动作，再决策）。
4. 长上下文诊断：Oracle、Oracle+Max、去历史、去 payoff、S2A、CoT 3-shot。

核心问题始终一致：
即便给了更多“思考形式”或更强“提示结构”，模型能否把预测稳定转化为收益最优的行为？

## 实验关键数据

### 主结果表（RPS，Single Action，对应文中 Table 1）

| 模型 | Functional Regret $\Delta_{\text{Functional}}/T$ | Rationalized Regret $\Delta_{\text{ToM}}/T$ | ToM 准确率 |
|------|---:|---:|---:|
| Tabular (RMax) | 0.083 ± 0.004 | 0.039 ± 0.006 | 97.4% ± 0.4 |
| LLaMA-2 70B | 0.971 ± 0.067 | 0.048 ± 0.003 | 96.8% ± 0.2 |
| LLaMA-2 70B Chat | 0.857 ± 0.142 | 0.119 ± 0.006 | 92.1% ± 0.4 |
| LLaMA-2 13B | 1.015 ± 0.044 | 0.049 ± 0.003 | 96.7% ± 0.2 |
| Falcon 40B | 0.973 ± 0.040 | 0.056 ± 0.005 | 96.2% ± 0.3 |
| Mixtral 8x7B Instruct | 0.542 ± 0.070 | 0.050 ± 0.005 | 96.7% ± 0.3 |

解读：
很多模型 ToM 准确率接近 97%，但功能遗憾却接近 1.0（RPS 下几乎最差）。
这直接验证“高 Literal ToM 不等于高 Functional ToM”。

### 提示策略分析表（Mixtral 8x7B，摘自 Table 2）

| Prompt | 游戏 | $\Delta_{\text{Functional}}/T$ | $\Delta_{\text{ToM}}/T$ | ToM% |
|------|------|---:|---:|---:|
| LM | RPS | 0.542 | 0.050 | 96.7 |
| Social LM | RPS | 0.643 | 0.119 | 92.0 |
| LM | IBS | 2.055 | 0.216 | 97.3 |
| Social LM | IBS | 2.082 | 0.182 | 97.4 |
| LM | IPD | 0.949 | 0.098 | 97.8 |
| Social LM | IPD | 1.098 | 0.110 | 97.4 |

解读：
Social Prompting 虽常有帮助，但并未根治“预测-行动断裂”。
即使显式把“对手预测”写进决策流程，功能表现仍远落后于简单表格策略。

### 长上下文消融表（LLaMA-3 70B，摘自 Table 3）

| 配置 | RPS | IBS | IPD |
|------|---:|---:|---:|
| Tabular | 0.083 | 0.211 | 0.086 |
| QA | 0.444 | 1.391 | 0.996 |
| CoT | 0.213 | 2.475 | 0.892 |
| CoT + 3-shot | 0.121 | 0.526 | 2.773 |
| Social QA | 0.256 | 1.613 | 0.550 |
| Oracle | 0.238 | 1.343 | 0.839 |
| Oracle + Max | 0.153 | 0.785 | 0.767 |
| Oracle + Max - History | 1.275 | 2.060 | 0.206 |
| Oracle + Max - Payoffs | 0.103 | 0.883 | 0.241 |

解读：
1. 即使直接告知对手动作（Oracle），仍明显不如 Tabular。
2. 去掉历史在 RPS/IBS 会显著恶化，说明历史信息对适应至关重要。
3. 某些任务中去掉 payoff 反而改善，提示模型在长上下文联合推理时出现干扰。

### 补充分析：推理模型 DeepSeek-R1 Distill 32B（Table 4）

| 游戏-对手 | Tabular 功能遗憾 | DeepSeek-R1 功能遗憾 | DeepSeek-R1 ToM% |
|------|---:|---:|---:|
| RPS - Single Action | 0.083 | 0.074 | 63.7 |
| RPS - Tit-for-Tat | 0.224 | 0.906 | 38.6 |
| IBS - Single Action | 0.211 | 0.126 | 97.2 |
| IBS - Tit-for-Tat | 0.468 | 0.045 | 94.5 |
| IPD - Single Action | 0.086 | 0.121 | 83.1 |
| IPD - Tit-for-Tat | 0.248 | 4.789 | 88.9 |

解读：
推理模型出现“功能强但字面预测并不总强”的反常组合，进一步证明两类能力并非单向蕴含关系。

## 亮点与洞察

1. 概念层面最重要贡献是把 ToM 从“预测能力”升级为“可用能力”。
2. 论文用最小反例（RPS 单一动作对手）就构成了对现有评测范式的强反驳。
3. 引入 $\Delta_{\text{ToM}}$ 作为诊断桥梁非常实用，可区分“预测错”还是“决策错”。
4. 对部署实践有直接意义：若模型要获得行动权限，必须看 Functional ToM，而非只看问答分数。
5. 与 predict-then-optimize 文献形成清晰呼应：预测误差与决策误差不是等价目标。

## 局限与展望

1. 任务仍以矩阵博弈为主，生态复杂度有限，外推到真实协作系统需进一步验证。
2. 主要实验对象是开源模型；闭源前沿模型的同类评估仍不足。
3. 论文重点在“评测框架批判”，不是提出一种可大规模提升 Functional ToM 的新训练算法。
4. 虽分析了 prompt 与上下文因素，但对内部机理（为何预测无法转行为）的神经表征解释仍弱。
5. 可继续研究“结构化记忆+短规划器+动作价值头”的组合，以降低长上下文决策退化。

## 相关工作与启发

### 与人类式 ToM 基准工作

Bubeck、Kosinski、Street、Strachan 等工作强调问答任务中的 ToM 迹象。
本文指出这些任务多是静态预测，无法反映交互适应。

### 与多智能体 RL/博弈论

Lowe 等、Akata 等已关注多体行为建模。
本文的不同点是把“预测他人”与“按预测行动”显式拆开，并统一到 regret 视角。

### 与 HCI 的 mutual ToM 观点

文中呼应了“互相建模以提升协作效果”的方向。
即：ToM 的价值不在心理标签本身，而在能否改善联合任务表现。

### 对我自己的研究启发

如果未来设计 LLM agent benchmark，应默认包含三类指标：

1. 对手建模准确率（Literal）。
2. 面向收益的在线适应（Functional）。
3. 预测-决策耦合效率（可由 $\Delta_{\text{ToM}}$ 与 $\Delta_{\text{Functional}}$ 的差衡量）。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐  
  原因：把 ToM 评测从单一预测维度升级到“预测+行动”双维框架，概念创新清晰且影响面大。

- 实验充分度: ⭐⭐⭐⭐☆  
  原因：跨模型、跨博弈、跨 prompt 的对比很系统，但复杂真实任务与闭源模型覆盖仍可增强。

- 写作质量: ⭐⭐⭐⭐⭐  
  原因：从反例切入、形式化定义、再到实证诊断，叙事链条完整，论证有层次。

- 实用价值: ⭐⭐⭐⭐⭐  
  原因：对所有“让 LLM 参与决策”的场景都给出可执行评测原则，直接影响 benchmark 与上线门槛设计。

## 总结性判断

这篇论文最有价值的地方不是“又测了一个榜单”，而是纠正了评测目标本身。
如果一个系统要在多轮交互中代表用户或与用户协作，仅有高 Literal ToM 分数远远不够。
未来 ToM benchmark 若不显式加入 Functional ToM 维度，仍会继续高估 LLM 的真实社会交互能力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[AAAI 2026\] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](../../AAAI2026/llm_evaluation/lost_in_benchmarks_rethinking_large_language_model_benchmarking_with_item_respon.md)
- [\[ACL 2025\] Towards Dynamic Theory of Mind: Evaluating LLM Adaptation to Temporal Evolution of Human States](../../ACL2025/llm_evaluation/towards_dynamic_theory_of_mind_evaluating_llm_adaptation_to_temporal_evolution_o.md)
- [\[ACL 2025\] PapersPlease: A Benchmark for Evaluating Motivational Values of Large Language Models Based on ERG Theory](../../ACL2025/llm_evaluation/papersplease_a_benchmark_for_evaluating_motivational_values_of_large_language_mo.md)
- [\[ICML 2025\] G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)

</div>

<!-- RELATED:END -->
