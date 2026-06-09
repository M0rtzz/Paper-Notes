---
title: >-
  [论文解读] Reasoning Can Be Restored by Correcting a Few Decision Tokens
description: >-
  [ICML 2026][LLM推理][推理模型差距] 作者用 token 级分布散度量化 base LLM 与 reasoning LRM 的差异，发现差距高度集中在早期、规划相关、且 base 自身不确定的少量 token 上（占比 ~8%）…
tags:
  - "ICML 2026"
  - "LLM推理"
  - "推理模型差距"
  - "token级干预"
  - "稀疏控制"
  - "规划token"
  - "推理时协作解码"
---

# Reasoning Can Be Restored by Correcting a Few Decision Tokens

**会议**: ICML 2026  
**arXiv**: [2605.16874](https://arxiv.org/abs/2605.16874)  
**代码**: https://github.com/AlphaLab-USTC/RRTokenIntervention  
**领域**: LLM推理  
**关键词**: 推理模型差距、token级干预、稀疏控制、规划token、推理时协作解码

## 一句话总结
作者用 token 级分布散度量化 base LLM 与 reasoning LRM 的差异，发现差距高度集中在早期、规划相关、且 base 自身不确定的少量 token 上（占比 ~8%），据此提出"分歧门控的一令牌接管"——仅在分歧尖峰处让 LRM 出一个 token 然后立刻交还 base，用 ~4-13% 干预预算即可恢复甚至超越同尺寸 thinking 模型。

## 研究背景与动机

**领域现状**：从 OpenAI o1、DeepSeek-R1 到 Qwen3-Thinking，主流提升推理能力的范式是大规模 RLVR 后训练，让 LRM 在数学/竞赛题上显著优于其 base 版本。同时一支"潜能视角"工作（activation steering、自奖励、单样本激活）认为 base 已经具备推理机器，post-training 只是激活/放大它。

**现有痛点**：训练侧的解释偏宏观，无法回答**生成层面**的问题——base 模型究竟在生成的哪一步走错了？是均匀漂移、还是少数关键决策？没有 token 级的因果性账本，"reasoning mode 为何 work"就只能停留在假说。

**核心矛盾**：要么承认推理能力散布在长链每一步、必须全程接管（昂贵），要么相信关键决策稀疏、但缺乏定位它们的可操作信号。前者与"少量参数子空间足够引发推理"的实验观察矛盾，后者长期没有显式的 token 级度量。

**本文目标**：(1) 定义并测量 base–LRM 的 token 级行为差距；(2) 刻画这些差距 token 的数量、位置、语义性质；(3) 验证"只修少量关键 token 即可恢复推理"的假设。

**切入角度**：在同词表的 base $\mathcal{M}_b$ 和强 reasoning $\mathcal{M}_r$ 之间，沿 base 自身 rollout 计算下一 token 分布的似然散度 $s_t = \mathcal{D}_f(p_b(\cdot|x_t), p_r(\cdot|x_t))$（默认用 cross-entropy）。这个量既不要求训练，又能逐 token 暴露"两个模型在哪里分道扬镳"。

**核心 idea**：推理差距是**稀疏 + 早期 + 规划相关 + 与 base 不确定性对齐**的，因此可以用一个全局校准的分歧阈值做门控，仅在尖峰处让 LRM 接管一个 token 再立刻交回 base，用极小预算撬动整条轨迹。

## 方法详解

### 整体框架
全文围绕一个问题展开：base LLM 在生成推理链时究竟在哪几步走错了？方法因此分成前后相扣的两段。先是**诊断**——在同词表的 base $\mathcal{M}_b$ 与强 reasoning $\mathcal{M}_r$ 之间，沿 base 自己的 rollout 逐 token 算一个分歧分数 $s_t$，再统计它的稀疏性、位置、语义和对答错的预测力；接着是**干预**——把诊断里发现的"分歧尖峰"做成一个推理时门 $g_t$，每一步据此决定是让 base 继续出 token，还是临时让 LRM 出一个 token 后立刻交还。整条流程输入是 prompt $x_0$、输出是混合解码序列 $y_{1:T}$，模型参数和隐状态全程不动，唯一被改的是个别位置的 token 选择权。

### 关键设计

**1. Token 级分歧度量：把"推理差距"落到每一步上**

训练侧的解释（RLVR、CoT、backtracking）都太宏观，回答不了"base 在生成的哪一步偏离了 LRM"。作者的做法是在 base rollout 上对每个 step $t$ 算一个似然散度 $s_t = -\sum_{y} p_b(y|x_t)\log p_r(y|x_t)$，默认用 cross-entropy，也讨论了满足 $D_{\text{rKL}}=\mathcal{D}_{\text{CE}}-H_b$ 的 reverse KL。这个量不需要训练、逐 token 即可算，直接暴露两个模型在哪里分道扬镳。

实测它呈现四个互相印证的性质，共同支撑起"稀疏控制"的图景：(i) 分歧高度集中——Lorenz 曲线远离对角线、Gini 系数 $G\!\approx\!0.936$，差距压缩在约 1-8% 的 token 上；(ii) 早期偏置——top-1% 分歧 token 在归一化位置 $u=t/T$ 上强烈左偏，密度峰值落在 $u\!\approx\!0.05$；(iii) 与不确定性对齐——这些 token 和 base 自身 Shannon 熵 $H_b(t)$ 的 top-p% 高度 IoU 重合，且规划类词占比从全局 1.89% 升到 15.75%（熵）/14.13%（分歧），enrichment 约 $7.5$–$8.3\times$；(iv) 有预测力——样本内 top-100 分歧均值能预判最终答错（GSM8K AUROC 0.851，优于熵基线 0.817）。换句话说，差距既稀疏又集中在早期规划点，干预因此有了明确的目标位置，连 base 熵都能当低成本代理信号单独使用。

**2. 全局校准的双重门控：把尖峰变成可在线判定、预算可控的二值开关**

光知道"分歧集中在尖峰"还不够，得有个能在解码时实时判定、又能把接管频率攥在手里的规则。作者先在校准集上收集 $\mathcal{S}=\{s_t\}$，取 $(1-r)$ 分位数作全局阈值 $\tau = Q_{1-r}(\mathcal{S})$，同时算出尾部对全局的均值比 $\lambda = \mathbb{E}[s|s>\tau]/\mathbb{E}[s]$。运行时的门是两个条件的合取：$g_t = \mathbb{I}[s_t>\tau \land s_t>\lambda\cdot\bar{s}_{t,W}]$，其中 $\bar{s}_{t,W}=\frac{1}{W}\sum_{i=1}^W s_{t-i}$ 是滑窗局部均值。

之所以要两个条件叠加，是因为单用任一个都会出问题：固定阈值在不同 prompt 和解码阶段会让预算剧烈漂；纯局部尖峰判定又会把整体平坦、只是抖动大的段误触。合取之后，全局阈值 $\tau$ 锚定校准意义下的尾部事件、直接决定 budget，局部比率项 $\lambda$ 压住缓变高位区的连续误触发、把接管离散成一个个"尖峰"。作者还验证校准后的 $\tau$ 跨 benchmark 在同一量级，不必逐任务调参。

**3. 一令牌接管的稀疏委派解码：只改决策点，不持续接管**

有了门，解码规则就统一成一句话：$y_t \sim p_r(\cdot|x_t)$ 当 $g_t=1$，否则 $p_b(\cdot|x_t)$。每次接管只生成下一个 token，立刻把控制权交还 base，不维护任何额外状态，实际接管率 $\rho = \frac{1}{T}\sum_t g_t$ 跨问题方差很小。作者还给了更便宜的退化版——仅用 $H_b(t)$ 触发，连在线查 $p_r$ 都省了，小预算下也能恢复大部分 Pass@8。

把接管压到"一个 token"是为了把因果机制做到最小：作者认为是少数规划 token 重定向了后续轨迹，而非 LRM 一路接管。这也解释了为什么 ~4% budget 的 guided 干预能胜过 25% 的随机或纯早期注入——**关键在改哪个 token，而不是改多少 token**。

### 损失函数 / 训练策略
完全是推理时干预，**不更新任何参数**。唯一的"训练"是校准：用一小批 prompt 跑 base rollout 收集 $\{s_t\}$，定出 $\tau, \lambda$。仅有的超参是滑窗长度 $W$ 与 spike ratio $r$，且作者表明 $\tau$ 跨 benchmark 重新校准后仍落在同一量级。

## 实验关键数据

### 主实验
评测设置：base $p_b$ = Qwen3-0.6B/1.7B-Base，guide $p_r$ = Qwen3-8B (Thinking)，六个数学 benchmark（GSM8K、MATH500、AIME24/25、AMC23、OlympiadBench），报告 Accuracy / Pass@8 与 Recovery = $(P_{\text{Guided}}-P_{\text{Base}})/(P_{\text{Thinking}}-P_{\text{Base}})$。

| 模型 | 设置 | 平均 Acc / Pass@8 | Recovery |
|------|------|-------------------|----------|
| Qwen3-0.6B-Base | — | 13.0 / 36.0 | — |
| Qwen3-0.6B-Base | +Guided $\bar{\rho}\!\approx\!0.04$ | 29.1 / 61.4 | 91% |
| Qwen3-0.6B-Base | +Guided $\bar{\rho}\!\approx\!0.13$ | 52.4 / 80.0 | **157%** |
| Qwen3-0.6B (Thinking) | — | 43.4 / 64.1 | 100% baseline |
| Qwen3-1.7B-Base | +Guided $\bar{\rho}\!\approx\!0.16$ | 62.1 / 83.8 | 112% |
| Qwen3-1.7B (Thinking) | — | 64.1 / 80.3 | — |
| Qwen3-8B (Thinking) | — | 78.1 / 87.3 | 上限参考 |

仅 13% 接管预算下 0.6B-Base + Guided 就**超越**同尺寸 thinking 模型（Pass@8 80.0 vs 64.1），相对 8B teacher 也恢复 57% gap 之外的额外性能（Recovery 157%，说明并非简单回归 teacher 而是混合解码的协同效应）。AIME24 从 0.4/3.3 直接跃升到 32.5/70.0 是最显著的 case。

### 消融实验（同预算对照 + 语义对照）

| 配置 | 预算 $\rho$ | Avg Acc | Avg Pass@8 | 备注 |
|------|------------|---------|-----------|------|
| Base | 0.00 | 13.0 | 36.0 | 参照下限 |
| +Random（随机位置） | 0.25 | 26.4 | 55.2 | 6× 预算 |
| +Early-only（前缀全替） | 0.25 | 25.7 | 58.4 | 6× 预算 |
| +Guided (Ours) | **0.04** | **29.1** | **61.4** | 1× 预算 |

| 类别 | Global | 接管集占比 | Enrichment |
|------|--------|------------|------------|
| Planning（规划） | 1.9% | **33.3%** | **17.6×** |
| Execution（执行） | 98.1% | 66.7% | 0.7× |

样本级翻转统计：400 题中 152 个由错→对、仅 3 个由对→错，干预近乎单向有益。

### 关键发现
- **位置选择 > 注入本身**：6× 预算的随机/纯早期 baseline 全面输给 1× 预算的 guided，证明"用 LRM 出 token"本身不带来增益，**修哪个 token**才是关键，且仅靠"早期"也不够——必须落在分歧尖峰上。
- **Pass@8 vs $\rho$ 是强非线性"膝盖曲线"**：前几个百分点接管就拉起 ~20 个点（41%→61% at $\rho\!\approx\!3\%$），到 7-8% 已追平 0.6B-Thinking 参考线，之后边际收益快速衰减。
- **接管语义高度集中在规划**：planning token 在干预集里 17.6× 富集，执行 token 0.7×（被抑制），定性例子显示典型模式是 LRM 插入一段"stop-and-check"消歧后交还 base 做常规计算。
- **跨家族泛化**：把 base 换成 LLaMA-3.1-8B、guide 换成 DeepSeek-R1-Distill-Llama-8B，~20% 接管恢复 91% gap，机制不限于 Qwen 系列。

## 亮点与洞察
- **把"为什么 RL 有效"翻译成 token 级事实**：之前的 reasoning gap 解释（CoT、backtracking、高熵子集、low-rank 子空间）多停留在训练侧；本文用 CE 散度沿 base rollout 量化，发现 ~8% 早期规划 token 承担 ~94% 的差距（Gini 0.936），是一个直接落到生成步的微观图像。
- **诊断信号即干预信号的自闭环**：top-K 分歧均值同时是 (i) 失败预测器（AUROC 0.851）和 (ii) 干预触发器，让"哪里错了"与"在哪里改"用同一个量打通，节省了独立训练 verifier 的成本。
- **退化版用 base 熵就能跑**：当不方便在线查 $p_r$（部署成本高）时，仅用 $H_b(t)$ 触发也能在小预算下恢复大部分 Pass@8，说明"base 自身不确定性"已经携带了大部分关键位置信息——这对实际部署很友好，可启发"用单模型熵代理两模型分歧"的蒸馏型干预设计。
- **预算 < 性能的可迁移启发**：该"稀疏委派 + 一令牌接管"模板可迁到 speculative decoding、agent step-level routing、on-policy distillation 数据过滤等场景——核心思想都是**只在分布分歧尖峰处花贵的计算**。

## 局限与展望
- 主实验局限在 Qwen3 系列 + 数学题，虽然附录里 LLaMA pair 与 GPQA-Diamond 给出迁移证据，但 code、multi-hop QA、agent 任务、更大模型尚未系统验证；尤其"规划/执行"语义边界在非数学域是否仍清晰需进一步检验。
- 门控运行时**仍需同时查 $p_b$ 和 $p_r$**，部署成本未必比直接跑 thinking 模型低；论文承认是"诊断式 sparse control"而非优化过的推理系统。基于熵的退化版是更现实的部署路径，但实验只在附录。
- 校准依赖一个 hold-out 集合，跨域分布漂移时 $\tau, \lambda$ 是否仍可单次校准未深究；spike ratio $r$、窗口 $W$ 的选择缺乏自适应机制。
- Recovery 157% 让人想问：是否对某些 prompt 出现 base 与 LRM 协同解码的"集成效应"超过 LRM 单独解码？作者未拆解这部分增益来源，是值得后续分析的方向。

## 相关工作与启发
- **vs 高熵 token policy update (Wang et al., 2025a)**：那篇也强调 reasoning 集中在少数高熵 token 上，但用于 RL 训练（约束梯度）；本文是推理时干预，且明确把"自分歧"与"跨模型分歧"区分对比，证明跨模型信号更精准（AUROC 高 ~3 个点）。
- **vs Speculative decoding / RouteLLM (Ong et al., 2025)**：传统 speculative decoding 用 drafter+verifier 提速、保持等价输出；本文反过来——base 是默认 drafter、LRM 仅在分歧尖峰**改变**输出，目标是恢复 LRM 能力而非加速 LRM。
- **vs RelayLLM (Huang et al., 2026)**：RelayLLM 在小模型超过风险阈值时把困难步骤"转交"给大模型；本文用更细的 token 级分歧而非启发式风险阈值触发，且只接管一个 token 而非一段，理论上更稀疏可控。
- **vs On-policy distillation 数据过滤 (Agarwal 2024, Peng 2025)**：那条线用 teacher 分布在训练时过滤/接管 student 轨迹防止学到错误样本；本文把同一机制搬到推理时，省掉了训练但放弃了参数级修正。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把"推理差距"用 CE 散度沿 rollout 直接量化并落到 token 级干预，视角清晰；个别组件（spike gating、teacher 接管）单看不新，但闭环讲法新。
- 实验充分度: ⭐⭐⭐⭐ 六个数学 benchmark + 两个 base 尺寸 + LLaMA 跨家族 + 同预算 baseline + 语义富集 + 失败预测，证据链完整；不足在非数学域和大模型 base 上覆盖较薄。
- 写作质量: ⭐⭐⭐⭐ 四条 finding 与三个干预设计一一对应，叙事干净；图 1/5/6 的可视化很有冲击力，公式与符号一致。
- 价值: ⭐⭐⭐⭐ 给"reasoning mode 为何 work"提供了少见的 token 级因果证据，并直接给出可部署的稀疏干预方案；对 RL post-training、speculative decoding、on-policy distillation 都有启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DeFine: Decision-Making with Analogical Reasoning over Factor Profiles](../../ACL2025/llm_reasoning/define_decision-making_with_analogical_reasoning_over_factor_profiles.md)
- [\[NeurIPS 2025\] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning](../../NeurIPS2025/llm_reasoning/ktae_a_model-free_algorithm_to_key-tokens_advantage_estimation_in_mathematical_r.md)
- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ACL 2026\] Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression](../../ACL2026/llm_reasoning/can_reasoning_path_still_be_effective_as_input_bridging_post-reasoning_to_chain-.md)
- [\[ACL 2026\] Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization](../../ACL2026/llm_reasoning/is_chain-of-thought_really_not_explainability_chain-of-thought_can_be_faithful_w.md)

</div>

<!-- RELATED:END -->
