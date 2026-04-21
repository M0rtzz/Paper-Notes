---
title: >-
  [论文解读] Provable Scaling Laws for the Test-Time Compute of Large Language Models
description: >-
  [NeurIPS 2025][LLM推理][测试时计算] 提出 Knockout（淘汰赛式两两淘汰）和 League（联赛式平均胜率排序）两种两阶段测试时计算算法，在"LLM 能以非零概率生成正确解"和"LLM 两两比较优于随机"的极弱假设下，从理论上证明失败概率随测试时计算量增长呈指数或幂律衰减至零，且整个算法仅需黑盒 LLM，无需外部验证器或奖励模型。
tags:
  - NeurIPS 2025
  - LLM推理
  - 测试时计算
  - 缩放律
  - 淘汰赛
  - 联赛
  - 两两比较
  - Best-of-N
  - 可证明保证
---

# Provable Scaling Laws for the Test-Time Compute of Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2411.19477](https://arxiv.org/abs/2411.19477)  
**代码**: [GitHub](https://github.com/pan-x-c/AgentScope/tree/feature/pxc/paper_provable/examples/paper_provable_scaling_law)  
**领域**: LLM推理  
**关键词**: 测试时计算, 缩放律, 淘汰赛, 联赛, 两两比较, Best-of-N, 可证明保证

## 一句话总结

提出 Knockout（淘汰赛式两两淘汰）和 League（联赛式平均胜率排序）两种两阶段测试时计算算法，在"LLM 能以非零概率生成正确解"和"LLM 两两比较优于随机"的极弱假设下，从理论上证明失败概率随测试时计算量增长呈指数或幂律衰减至零，且整个算法仅需黑盒 LLM，无需外部验证器或奖励模型。

## 研究背景与动机

**领域现状**：测试时计算（test-time compute）已成为提升 LLM 可靠性的核心策略。典型方法包括 Best-of-N 采样（生成多个候选后用验证器选最优）、多数投票（majority voting，取出现频率最高的答案）、以及链式思维（CoT）等迭代推理方法。在高风险场景（如需要 99.9% 正确率）或 LLM Agent 多步工作流（每步都需近乎完美正确率）中，通过增加推理计算量来提升成功率已是刚需。

**现有痛点**：(1) Best-of-N 采样依赖外部验证器/奖励模型；当验证器不完美时，随 N 增大反而可能出现性能下降——因为"搜索收益最终被找到欺骗验证器的对抗性解的风险所抵消"。(2) 多数投票存在理论上的致命缺陷：即使 LLM 以 45% 概率生成正确答案，只要存在某个错误答案以 46% 概率被生成，多数投票的成功率随样本数 N 增大反而趋向零。(3) 现有经验缩放律缺乏严格理论保证，不同任务表现不一致。

**核心矛盾**：实践需要可靠的测试时缩放方法来将成功率推至任意接近 100%，但现有方法要么需要额外的外部组件（验证器），要么在理论上无法保证随计算量增加而单调改善。

**本文目标**：设计在极弱假设下具有**可证明缩放律**（provable scaling law）的测试时计算算法——即成功率可随计算量增长被提升至任意接近 100%，且仅需黑盒 LLM 本身。

**切入角度**：利用 LLM 的**两两比较能力**（pairwise comparison）而非点评估能力。核心直觉是"判断两个解哪个更好"远比"独立判断一个解是否正确"容易——这在计算复杂性理论中有深刻对应（验证 vs 搜索）。通过锦标赛/联赛结构聚合多次两两比较的结果，将比较能力的微弱优势放大为高置信度的最终选择。

**核心 idea**：生成 $N$ 个候选解 + 用 LLM 自身进行两两比较的锦标赛/联赛聚合 = 无需外部验证器的可证明测试时缩放。

## 方法详解

### 整体框架

两阶段方法，两个阶段均仅使用黑盒 LLM：

- **Stage 1 — 生成阶段**：对输入问题 $x$，独立采样 $N$ 个候选解 $y_1, \dots, y_N \sim \mathcal{M}_{\text{gen}}(x)$，可完全并行。每个候选包含完整思维链（CoT），为后续比较提供依据。
- **Stage 2 — 聚合阶段**：通过 Knockout 或 League 两种锦标赛结构，利用 LLM 的两两比较来筛选最终答案。

关键假设：$\mathcal{M}_{\text{gen}}$ 是 LLM 生成解的分布，$\mathcal{M}_{\text{comp}}$ 是 LLM 进行两两比较时输出的分布。两阶段的随机性来源包括非零温度解码、随机选择提示方法或 LLM 后端等。

### 关键设计

1. **Knockout 算法（淘汰赛式聚合）**

    - **功能**：通过多轮两两淘汰从 $N$ 个候选中选出最终答案
    - **核心思路**：将 $N$ 个候选随机配对，每对进行 $K$ 次独立比较，胜者（获得多数票者）晋级，共 $\lceil \log_2 N \rceil$ 轮后剩一个最终输出。依赖假设 (Assumption 2.1)：(A1) LLM 生成正确解的概率 $p_{\text{gen}} > 0$；(A2) 给定一对正确与错误解，LLM 以 $p_{\text{comp}} > 0.5$ 概率选出正确者。在此假设下，**定理 2.3** 证明失败概率 $P(\text{fail}) \leq (1-p_{\text{gen}})^N + \lceil \log_2 N \rceil \cdot e^{-2K(p_{\text{comp}}-0.5)^2}$，即同时放大 $N$ 和 $K$ 时呈指数衰减。**定理 2.4** 进一步证明即使固定 $K$，仅增大 $N$ 也可使成功率趋向 1（幂律衰减），消除了需要事先知道 $p_{\text{comp}}$ 的限制
    - **设计动机**：淘汰赛结构使每轮候选数减半，总计仅需 $(K+1) \times N$ 次 LLM 调用，且端到端延迟仅为 $T_{\text{gen}} + \log_2(N) \times T_{\text{comp}}$（每轮可完全并行），计算效率远优于全对全比较

2. **League 算法（联赛式聚合）**

    - **功能**：每个候选与 $K$ 个随机对手比较，用平均胜率排序选最优
    - **核心思路**：对每个候选 $y_i$，随机抽 $K$ 个对手进行独立比较，计算平均胜率 $\hat{\mu}_i$，选胜率最高者为最终输出。依赖更鲁棒的假设 (Assumption 3.1)：存在"正确且强"的解子集 $\mathcal{Y}_{\text{cs}}$，其平均胜率严格高于所有错误解的最高胜率（间隔 $\Delta > 0$），且 LLM 以非零概率 $p_{\text{cs}} > 0$ 生成这类解。**定理 3.3** 证明失败概率 $P(\text{fail}) \leq (1-p_{\text{cs}})^N + 2Ne^{-K\Delta^2/8} + 2Ne^{-(N-1)\Delta^2/8}$，呈指数衰减
    - **设计动机**：Knockout 的假设要求 LLM 对**任意一对**正确与错误解的比较都优于随机，这过于严格——一次比较错误就可能淘汰正确答案。League 转向**平均胜率**的假设，允许某些特定比较对存在系统性失败，只要正确解的整体胜率高于错误解即可，更加鲁棒

3. **混合多 LLM 策略（Mixed）**

    - **功能**：生成和比较阶段各用多个 LLM 的混合，增大假设满足的概率
    - **核心思路**：生成阶段一半候选来自 Llama3.1、一半来自 Qwen2.5；比较阶段同样混合。不同 LLM 擅长不同类型问题，混合使用使得 $p_{\text{gen}} > 0$ 和 $p_{\text{comp}} > 0.5$ 在更多问题上成立。例如 LLM$_1$ 擅长问题 $x_1$（$p_{\text{gen}}=0.2$）但不擅 $x_2$（$p_{\text{gen}}=0$），LLM$_2$ 反之，混合后两个问题都有 $p_{\text{gen}}= 0.1 > 0$
    - **设计动机**：单一 LLM 在某些问题上可能完全无法生成正确解（$p_{\text{gen}}=0$），混合多 LLM 可显著扩大算法的适用范围

### 损失函数 / 训练策略

本文是纯推理时方法，**无需任何训练或微调**。仅需零样本 CoT 提示来驱动 LLM 的生成和比较。生成阶段用标准 CoT 提示（"Let's think step by step"），比较阶段将两个候选解的完整思维链并排呈现给 LLM 并要求判断哪个更好。整个系统基于 AgentScope 多智能体框架实现，支持并行和分布式计算。

## 实验关键数据

### 主实验

| 方法 | 模型 | 数据集 | N=1 准确率 | N=64 准确率 | 提升 |
|------|------|--------|-----------|------------|------|
| Knockout | Mixed | GPQA | 45% | 55% | +10pp |
| Knockout | QwQ-32B | GPQA-diamond | 60% | 72% (N=16) | +12pp |
| Knockout (N=8) | Mixed | GPQA | — | > MV (N=64) | — |
| Knockout (N=4) | QwQ-32B | GPQA-diamond | — | > MV (N=16) | — |
| League | Mixed | GPQA | 45% | 53% (N=16) | +8pp |

- Mixed 在所有 N 下一致优于单独使用 Llama3.1 或 Qwen2.5
- Knockout 在多数模型（除 Llama3.1 外）上优于同 N 的多数投票
- 即使考虑 Knockout 需要 $(K+1)N$ 次调用的额外开销，其 N=8 的精度仍超过多数投票 N=64

### 消融实验

| 消融维度 | 设置 | 关键发现 |
|----------|------|----------|
| 筛选子集 ($\hat{p}_{\text{comp}} > \tau$) | $\tau = 0.5 / 0.6 / 0.7$ | 满足假设的子集上，准确率随 N 从 55% 提升至 80%（$\tau=0.5$），$\tau$ 越大越接近 100% |
| 任务类型 | 工程 vs 哲学 (MMLU-Pro) | 推理任务（工程）缩放效果好，知识型任务（哲学）缩放受限——因比较无法弥补知识缺失 |
| League 对手数 M | M = 1 到 N-1 | M ≥ 4-5 时精度饱和，无需全对全比较，可大幅节省计算 |
| 固定 K vs 同时放大 N,K | K=2/4 固定 | 固定 K 仍有幂律缩放（定理 2.4），同时增大 N 和 K 可获指数缩放 |

### 关键发现

- **极弱假设即可保证缩放**：只要 LLM 能偶尔生成正确解（$p_{\text{gen}} > 0$）且两两比较优于随机（$p_{\text{comp}} > 0.5$），就能从理论上保证失败率随计算量衰减至零
- **任务类型决定缩放效果**：推理密集型任务（如数学、工程）的缩放明显优于知识密集型任务（如哲学），因为前者的比较阶段可利用推理过程提供额外信息
- **League 在假设满足子集上准确率显著提升**：满足 $\hat{p}_{\text{cs}} > 0$ 且 $\hat{\Delta} > 0$ 的问题子集上，Mixed 的准确率提升高达 25%
- **多 LLM 混合扩大适用范围**：Mixed 在更多问题上满足理论假设，解释了其一致优于单模型的表现

## 亮点与洞察

- **首次为测试时计算提供形式化的可证明缩放律**：不是经验拟合的曲线，而是建立在两个极弱假设上的严格数学证明。定理给出的失败概率上界与实验观测高度一致，理论与实践完美对接
- **"比较比生成/验证容易"的深刻直觉**：判断两个解哪个更好，远比独立生成正确解或独立验证一个解更容易——当两个解并排呈现时，LLM 可以交叉检查推理步骤，错误更容易暴露。这一直觉在 LLM alignment（RLHF 用 preference 而非 reward）和社会选择理论中都有深刻对应
- **极简且无缝适配**：仅需黑盒 LLM 的生成和比较能力，无需训练验证器/奖励模型，即插即用。且淘汰赛结构天然支持并行化，端到端延迟仅对数级增长
- **两种算法形成互补**：Knockout 假设更强但计算更省（$O(N \log N)$ 次比较），League 假设更弱但计算更多（$O(NK)$ 次比较），实践中可根据任务特点选择

## 局限与展望

- **比较假设可能不总成立**：在知识密集型任务中，LLM 的比较能力可能接近甚至低于随机水平（$p_{\text{comp}} \approx 0.5$），此时算法无法有效缩放
- **计算开销较高**：Knockout 需要 $(K+1)N$ 次 LLM 调用，League 的全对全版本需要 $O(N^2)$ 次调用；虽可并行化，但总 token 消耗显著
- **假设参数未知**：$p_{\text{gen}}$、$p_{\text{comp}}$、$\Delta$ 等在实际中未知，目前无法根据目标成功率自适应确定最优超参数 $N$ 和 $K$
- **独立性假设**：理论分析假设各次比较判断相互独立，但实际中同一 LLM 的输出可能存在系统性偏差和相关性
- **对抗性错误解**：League 假设可被胜率异常高的错误解打破（类似 BoN 的验证器欺骗问题），虽然作者论证比较机制比点评估更鲁棒，但并未完全消除此风险

## 相关工作与启发

- **vs Best-of-N 采样**：BoN 需要外部 reward model 且不完美验证器会导致性能反转；本文用 LLM 自身的两两比较替代外部验证，在概念上更简洁，且比较的准确性通常高于点评估
- **vs 多数投票**：多数投票仅利用最终答案的频率信息，要求正确答案的出现概率严格最高——这在开放式任务或答案空间大时难以满足；本文利用完整推理过程的两两比较，信息利用更充分
- **vs 过程奖励模型（PRM）**：PRM 需要对每个推理步骤训练验证器，成本高且依赖标注数据；本文的方法完全免训练
- **与 LLM-as-a-Judge 的联系**：两两比较在 RLHF、Chatbot Arena 等场景中已被广泛证明有效，本文将其从评估工具升级为推理能力放大器，并给出严格理论支撑
- **启发**：该框架可扩展到 LLM Agent 工作流中——对多步流程的每个子任务应用 Knockout/League，可将整体成功率保持在高水平。此外，与 CoT/self-refinement 等方法正交且可组合使用

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 理由：首次为测试时计算提供形式化的可证明缩放律（而非经验拟合），将淘汰赛/联赛锦标赛结构与严格概率论证结合，开辟了理论驱动的推理时缩放新方向
- **实验充分度**: ⭐⭐⭐⭐ 理由：覆盖 GPQA、MMLU-Pro、MATH-500 三个数据集和 Llama3.1、Qwen2.5、GPT-4o、QwQ-32B 四个模型，且通过参数估计-子集验证方法精巧地桥接理论与实验；但每类别仅 100 题的 MMLU-Pro 子集略显不足
- **写作质量**: ⭐⭐⭐⭐⭐ 理由：定理陈述简洁优雅，假设条件清晰明了，理论与实验的桥接分析逻辑严密。全文结构层次分明：问题动机→算法设计→理论保证→实验验证→局限讨论，堪称理论型论文的范本
- **实用价值**: ⭐⭐⭐⭐ 理由：算法即插即用且实现简单（已开源），对高可靠性场景有直接应用价值；但计算开销和未知参数问题在大规模部署时仍需解决

<!-- RELATED:START -->

## 相关论文

- [Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)

<!-- RELATED:END -->
