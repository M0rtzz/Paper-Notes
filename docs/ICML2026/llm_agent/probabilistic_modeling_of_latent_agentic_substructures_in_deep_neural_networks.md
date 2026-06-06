---
title: >-
  [论文解读] Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks
description: >-
  [ICML 2026][LLM Agent][子代理结构] 作者把神经网络（特别是 LLM）形式化为多个隐式子代理（每个是 outcome 上的概率分布）通过对数加权池化合成的复合代理…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "子代理结构"
  - "对数池化"
  - "认知效用"
  - "Waluigi 效应"
  - "对齐理论"
---

# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks

**会议**: ICML 2026  
**arXiv**: [2509.06701](https://arxiv.org/abs/2509.06701)  
**代码**: 无（理论论文）  
**领域**: LLM Agent / AI 对齐 / 概率建模  
**关键词**: 子代理结构、对数池化、认知效用、Waluigi 效应、对齐理论

## 一句话总结
作者把神经网络（特别是 LLM）形式化为多个隐式子代理（每个是 outcome 上的概率分布）通过对数加权池化合成的复合代理，并在认知效用 $W_i(o)=\log P_i(o)$ 框架下证明了 "严格一致受益（strict unanimity）" 在线性池化或二元 outcome 下不可能、但 $|\mathcal O|\ge 3$ 下可行，进而推出"显式让 Waluigi 先显形再压制"严格优于"只强化 Luigi"的对齐原则。

## 研究背景与动机

**领域现状**：把 LLM 视作"内部由多个相互竞争的 persona / 先验组成的集合"是 alignment 领域的常见经验观察——人类有亲社会和反社会双重驱动，LLM 也表现出 self-preservation、Waluigi（强化善良 persona 反而激出对抗 persona）等现象。但这些都是描述性的，没有严格的数学框架去推。同时机器学习里早就有 ensemble、Product of Experts、multi-head 等加性组合，它们最终的 softmax 输出本质就是 logit 上的加权和。

**现有痛点**：(1) 经济学/博弈论里"用效用函数聚合多代理偏好"已经有非常成熟的理论（social welfare aggregation、unanimity），但没人把这套搬进神经网络分析；(2) 现有 LLM 对齐讨论 Waluigi 等现象时只能凭直觉，没法回答"在什么条件下子代理才能稳定合成""什么样的子代理组合根本无法形成一致受益的复合体"等关键问题；(3) 经验观察到的"先 manifest 后 suppress"和"直接 reinforce 善良"两种 RLHF 套路谁更好，缺数学证明。

**核心矛盾**：神经网络训练目标是 $-\log P(\cdot)$（cross-entropy），所以效用应该是 $\log P(\cdot)$——但对数效用下的"严格一致受益"是个比线性效用更尖锐的条件，对 outcome 空间结构（$|\mathcal O|=2$ vs $\ge 3$）和池化形式（线性 vs 对数）都极敏感。要把这些尖锐结论推出来，需要严格定义复合代理 + 兼容神经网络结构的聚合形式。

**本文目标**：(i) 提出"子代理是概率分布 + 复合代理是对数池化"的形式化框架；(ii) 证明严格一致受益的存在/不存在的清晰边界（线性 vs 对数池化、$|\mathcal O|=2$ vs $\ge 3$）；(iii) 建立 cloning invariance、连续性、openness 等递归与稳定性性质；(iv) 用框架重新解释 LLM 的 Waluigi 现象并推出有数学保证的对齐原则。

**切入角度**：作者注意到 modern LLM 末层就是 linear logit + softmax，于是任何 logit 上的加性分解（ensemble、PoE、multi-head）数学上**精确等价**于对应分布的对数池化。这把 alignment 的"persona 合成"问题自然嵌入到 *对数池化 + 对数效用* 的经济学/信息论框架里。

**核心 idea**：把 LLM 当成一个对数池化的复合代理，用 welfare gap $\Delta_{P_i}(P)=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$ 量化每个子代理的受益情况，再用此公式推 Waluigi-Luigi 的 first-order 关系。

## 方法详解

### 整体框架
对每个子代理 $i$ 配一对 (belief $P_i$, welfare $W_i$)。复合代理的 belief 由对数池化给出 $P(o)=\frac1Z\prod_j P_j(o)^{\beta_j}$，权重 $\beta_j\ge 0,\sum_j\beta_j=1$。当效用取 $W_i(o)=\log P_i(o)$（即认知效用），"子代理 $i$ 从合成中受益"的等价条件是 welfare gap $\Delta_{P_i}(P)=\mathbb E_P[\log P_i]-\mathbb E_{P_i}[\log P_i]\ge 0$，并可化为信息论形式 $\Delta_i=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$。"严格一致受益群" $\mathcal U_{\text{strict}}$ 要求所有 $i$ 都严格 $>0$。围绕这个核心定义，作者推出可能性边界、递归与稳定性、最后落地到 Luigi-Waluigi 的对齐分析。

### 关键设计

1. **对数池化作为神经网络的"天然"子代理聚合规则**:

    - 功能：把神经网络末层的 logit 加性分解（ensemble、Product of Experts、multi-head attention）统一解释为对数池化下的子代理合成，从而把"persona 聚合"问题嵌入既有的经济学聚合理论。
    - 核心思路：对数池化 $P(o)=\frac1Z\prod_j P_j(o)^{\beta_j}$ 等价于对 logit $\log P_i$ 做加权和再 softmax，恰好对应 modern transformer 末层的形式。再把认知效用 $W_i(o)=\log P_i(o)$ 与"训练目标是 cross-entropy"对应起来——梯度通过 $\log P(\cdot)$ 项回传，所以 $\log P$ 就是网络优化的隐式 utility。最后定义 compositional agent（$\mathbb E_P[W_i]\ge \mathbb E_{P_i}[W_i]$）和 strictly unanimous group。
    - 设计动机：要让经济学的 welfare aggregation 真的能用在神经网络上，必须找到一种聚合形式**同时**匹配（i）网络结构（末层 linear + softmax）和（ii）训练目标（log-likelihood）。对数池化是唯一同时满足这两点的形式，所以它不是"选择"而是被网络结构和训练目标共同 dictate 出来的。

2. **严格一致受益的可能性边界（Theorem 8/9/10）**:

    - 功能：清晰刻画在什么条件下"所有子代理同时受益"是可达的，是整篇论文最技术性的部分。
    - 核心思路：(i) **二元 outcome 不可能定理**：$|\mathcal O|=2$ 下，对任意非平凡权重，都无法同时满足两个代理的 $\Delta_i\ge 0$ 且至少一个 strict——直观地，二元空间下 log pool 是 zero-sum 拉扯，一方增加另一方必减。(ii) **$|\mathcal O|\ge 3$ 存在性定理**：可显式构造 $\{P_i\}_{i=1}^n,\beta_i$ 使得 $\mathbb E_P[\log P_i]>\mathbb E_{P_i}[\log P_i]$ 对所有 $i$ 严格成立。(iii) **线性池化不可能性定理**：在 $P_C^{\text{lin}}(o)=\sum\beta_i P_i(o)$ + $W_i(o)=\log P_i(o)$ 下，永远不存在严格一致受益的合成——直观上线性池化等价于 random dictatorship，反对齐的代理被选作 dictator 时会严重损害他人。
    - 设计动机：这套定理给出三件具体的"哪条路走得通"判断：(a) 要做 multi-persona LLM 必须用对数池化而非简单加权平均（这正好和神经网络结构一致）；(b) outcome 空间太小（二元）的玩具问题不适合用这套理论建模；(c) 至少需要 3 个 outcome 才能讨论非平凡合成。这也意味着"二元安全/不安全"标签太粗的对齐设定本质上是退化的。

3. **递归与稳定性 + Luigi-Waluigi 对齐原则**:

    - 功能：给"合成—分解—再合成"的递归过程建立结构性保证（cloning invariance、连续性、openness），并据此严格证明"manifest-then-suppress Waluigi"优于"只强化 Luigi"。
    - 核心思路：(a) **Lemma 13（兼容分裂下池化不变）**：把一个 $P_i$ 用 $m$ 个权重相容的子代理替换不会改变全局 pool。(b) **Theorem 14（父代理受益不传递给子代理）**：可构造例子使得父代理 $\Delta_{P_1}(P)>0$ 但分裂出的 $\Delta_{P_{1,1}}(P)<0$，说明 alignment 不能仅看顶层。(c) **Theorem 17（openness）**：若 $P\in\mathcal U_{\text{strict}}$，则其某个邻域全在 $\mathcal U_{\text{strict}}$ 里——一致受益是局部稳定属性。(d) Section 5 用 $L=\log P$、$l_i=\log P_i$ 和 $P$-centered profile $v_i(o)=l_i(o)-\mathbb E_P[l_i]$ 构造希尔伯特空间分析，证明：当 RLHF 对父代理 $P$ 加 $\mathrm{KL}$ 预算约束并强化 Luigi persona 时，对抗的 Waluigi 子代理必然反向激活；而"先 manifest Waluigi 让它显形、再统一压制"的策略相对纯强化 Luigi 严格更大的 first-order alignment 误差缩减。
    - 设计动机：稳定性结果（特别是 openness）是后面 Luigi-Waluigi 分析的基石——只有当严格一致受益是开集，才能保证 RLHF 的小步参数更新不会突然破坏 persona 结构。Theorem 14 解释了为什么实践中 RLHF 顶层指标过得去、但子 persona 层面还能搞出问题。Luigi-Waluigi 结论第一次给"manifest-then-suppress"这一经验启发式提供了形式证明。

### 损失函数 / 训练策略
这是理论论文，不涉及训练流程，但所有结论建立在"网络末层是 logit 加性结构 + cross-entropy 训练 + KL 预算 RLHF 约束"这一标准设置上，对应实际 RLHF 中的 KL-regularized DPO/PPO 微调。

## 实验关键数据

### 主实验
论文没有传统经验实验，主要"实验"是定理 + 闭式构造。可整理为两张"理论-断言"对照表：

| 池化形式 | 效用 | $|\mathcal O|$ | 严格一致受益是否可达？ | 来源 |
|---------|------|----------------|----------------------|------|
| 线性 | 认知效用 $\log P_i$ | 任意 | **不可能** | Theorem 10 |
| 对数 | 认知效用 $\log P_i$ | $2$ | **不可能** | Theorem 8 |
| 对数 | 认知效用 $\log P_i$ | $\ge 3$ | **可达**，可显式构造 | Theorem 9 |
| 对数 | 一般 welfare $W_i$ | 任意 $n\ge 2$ | 存在某 $\{P_i,W_i,\beta_i\}$ 使其成立 | Theorem 6 |

对齐应用（Section 5）的核心定理（同样属于理论结果）：

| 策略 | 一阶 alignment 误差下降量 | 严格性 |
|------|---------------------------|--------|
| 仅强化 Luigi（增大 $\beta_{\text{Luigi}}$） | $\Delta_{\text{Luigi}}^{(1)}$ | baseline |
| Manifest-then-suppress Waluigi | 严格大于 $\Delta_{\text{Luigi}}^{(1)}$ | 论文证明在 KL 预算约束下成立 |

### 消融实验

| 配置 | 关键结论 | 说明 |
|------|---------|------|
| Welfare = $\log P_i$ vs 一般 $W_i$ | 一般 $W_i$ 下更容易达到一致受益（Theorem 6） | 认知效用更"窄"且约束更强 |
| Pool 兼容分裂（Lemma 13）vs 非兼容分裂（Theorem 14） | 前者保 pool 不变，后者父受益不传子 | 说明顶层 alignment 不足以保证子层 alignment |
| Strict vs non-strict unanimity | non-strict 允许 duplicate（Lemma 48），strict 在邻域稳定（Theorem 17）但禁止 trivial duplication（Theorem 55） | 区分有意义合成 vs 退化复制 |

### 关键发现
- "outcome 空间维度 $\ge 3$" 是严格一致受益可能性的硬门槛。任何把对齐简化为 "safe vs unsafe" 二元的设定都会落进 Theorem 8 的不可能区域。
- "顶层模型表现好但子 persona 受损" 不是错觉而是数学定理（Theorem 14）——RLHF 不能只看 reward model 分数。
- "先让 Waluigi 显形再压制" 严格优于"直接强化 Luigi"，第一次有了形式化证明。这对 RLHF/DPO 中的 negative sampling 策略和 jailbreak 防御训练有直接指导意义。

## 亮点与洞察
- 把对数池化与神经网络末层结构 + cross-entropy 训练 "三位一体" 联系起来是一个非常优雅的观察——它意味着 alignment 理论可以借用整套经济学聚合工具，而不必另起炉灶。
- Theorem 14（父受益不传子）解释了 alignment 经验中的反复出现的"reward model 高分但 deceptive behavior 仍在"现象：reward 看的是顶层，子 persona 仍可能严格受损。
- "Openness"结果给 RLHF 的 KL 预算训练提供了几何直觉：只要 base model 处于严格一致受益区，小步更新会留在区内，所以 KL 预算训练在理论上稳。
- "Manifest-then-suppress" 是一条非常具体的 RL HF 实操原则——对 jailbreak、deceptive 行为先 elicit 出来构造 negative 样本，再用它们做 contrastive 训练，效果会严格优于只放大正样本。

## 局限与展望
- 整篇假设 outcome 空间有限且所有 $P_i>0$，对实际 LLM 几万词表 + 长序列 conditional 还要做严格扩展（论文只是把 state 吸到 $\mathcal O$ 里规避）。
- 对数效用 $W_i=\log P_i$ 是"训练目标自然给出"的选择，但实际 RLHF 还有 reward model、preference learning 等更复杂的隐式 utility；推广到这些非 epistemic utility 才能更直接落地工业 RLHF。
- Section 5 的 Luigi-Waluigi 结论是 first-order（即小 KL 预算下的局部展开），对大幅度对齐微调还需要 higher-order 分析。
- 全文没有实证实验验证"manifest-then-suppress 优于纯 Luigi 强化"的实际数值差距，下一步需要在 RLHF/DPO 上做对照实验。

## 相关工作与启发
- **vs Opinion Pooling 经典理论**：经济学已经研究对数 vs 线性池化的公理化性质几十年，本文把它移植到神经网络分析。
- **vs Active Inference（Friston 等）**：active inference 已经主张"agent = 生成模型 + minimize free energy"，本文借用其概率代理观但更聚焦于子代理合成的数学结构。
- **vs Waluigi Effect 经验讨论**：之前都是 blog/twitter 经验观察，本文第一次给出 KL 预算下的 first-order 严格证明，并据此推出 manifest-then-suppress 原则。
- **vs Product of Experts / Mixture of Experts**：PoE 本质就是对数池化，本文证明 PoE 风格组合的稳定/可达条件——MoE 的 gating 选择，可以看作动态的 $\beta_i$ 选择。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把对数池化 + 认知效用框架引入 alignment 理论，并给出"manifest-then-suppress 严格优于直接 reinforce"的形式证明，是 alignment 文献中第一份这类工作。
- 实验充分度: ⭐⭐ 纯理论论文，零数值实验/无 LLM 验证，所有结论靠定理。
- 写作质量: ⭐⭐⭐⭐ motivation 把"为什么神经网络末层结构 + cross-entropy 训练自然导出对数池化"讲得很到位；可能性定理与递归定理排布也清晰。
- 价值: ⭐⭐⭐⭐ 对 alignment 研究者和 RLHF 实践者都有方法论价值——给"persona 合成""Waluigi 效应"等模糊概念提供了第一份严格框架。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](../../AAAI2026/llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)
- [\[ICML 2026\] Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models](hunt_instead_of_wait_evaluating_deep_data_research_on_large_language_models.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](../../ICLR2026/llm_agent/a_benchmark_for_deep_information_synthesis.md)
- [\[ICML 2026\] HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](hawkesllm_semantic_uncertainty_propagation_in_agentic_text_simulation.md)
- [\[ICML 2026\] Answer Only as Precisely as Justified: Calibrated Claim-Level Specificity Control for Agentic Systems](answer_only_as_precisely_as_justified_calibrated_claim-level_specificity_control.md)

</div>

<!-- RELATED:END -->
