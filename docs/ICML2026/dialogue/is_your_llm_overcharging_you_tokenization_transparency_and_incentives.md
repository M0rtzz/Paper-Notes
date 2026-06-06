---
title: >-
  [论文解读] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives
description: >-
  [ICML 2026][对话系统][按 token 计费] 本文把 LLM-as-a-Service 建模成"委托-代理"问题，证明现在主流的"按 token 收费"机制天然激励服务商把同一字符串重新切成更长的 token 序列来超额收费，并且即使强制服务商公开 next-token 分布…
tags:
  - "ICML 2026"
  - "对话系统"
  - "按 token 计费"
  - "激励兼容"
  - "分词多重性"
  - "按字符计费"
  - "委托代理"
---

# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

**会议**: ICML 2026  
**arXiv**: [2505.21627](https://arxiv.org/abs/2505.21627)  
**代码**: https://github.com/Human-Centric-Machine-Learning/token-pricing (有)  
**领域**: AI 安全 / 机制设计 / LLM-as-a-Service 定价  
**关键词**: 按 token 计费、激励兼容、分词多重性、按字符计费、委托代理

## 一句话总结
本文把 LLM-as-a-Service 建模成"委托-代理"问题，证明现在主流的"按 token 收费"机制天然激励服务商把同一字符串重新切成更长的 token 序列来超额收费，并且即使强制服务商公开 next-token 分布，多收费而不被发现也只是 NP-Hard 而非不可行——作者给出一个简单启发式算法在保持合理性的前提下实测最多多收 11.2% 的 token，最后证明唯一能消除该激励的可加性定价机制是"按字符长度线性计费"。

## 研究背景与动机
**领域现状**：云端 LLM 服务（OpenAI / Gemini / Anthropic 等）几乎清一色采用 pay-per-token 计费：用户提交 prompt，服务商在自家硬件上跑模型生成输出，按返回的 token 数乘以单价收费。用户能看到的只有返回字符串和声称的 token 数，模型内部用什么 vocabulary、实际怎么切分、next-token 分布是什么样的，都在服务商一侧。

**现有痛点**：分词不唯一。同一个字符串 "Damascus" 既可以被切成 `|Dam|ascus|`（2 个 token），也可以被切成 `|Da|ma|s|cus|`（4 个 token），用户对此完全不知情。服务商完全可以把真实生成的 2-token 序列"重新报"成 4-token 来双倍收费，而字符串完全没变，用户没有任何技术手段察觉。

**核心矛盾**：信息不对称（asymmetry of information）造成的道德风险（moral hazard）——服务商完整观察生成过程，用户只观察并支付最终上报的 token 序列；只要"按 token 计费"成立且 vocabulary 中存在多字符 token，把短分词换成长分词在数学上严格能涨收入。

**本文目标**：分解为三个子问题。(1) 在 pay-per-token 下，服务商有没有结构性的撒谎激励？(2) 如果强制服务商公开 next-token 分布，让用户用"该分词在模型下可信吗"来反查，撒谎是否就被堵死了？(3) 有没有一个原则上消除这种激励的定价机制？

**切入角度**：作者用契约理论里的委托-代理（principal-agent）框架建模——把用户当 principal、服务商当 agent、计费规则当 contract，然后系统性地刻画"激励兼容"（incentive-compatibility）这一性质：在该性质下，服务商如实上报永远不比撒谎差。这是一个在拍卖、机制设计、保险等场景反复使用过的范式，第一次被搬到 LLM 定价上。

**核心 idea**：用 token 长度计费一定不激励兼容，唯一可加且激励兼容的方式是按字符数线性计费；过渡时只需令 $r_c = r_o \cdot \mathrm{tpc}$（tpc 为平均每字符 token 数），服务商平均利润率可以保持不变。

## 方法详解

### 整体框架
论文的"方法"主要是三件事拼成一个完整的论证链：(i) 用 principal-agent 模型把 LLM-as-a-Service 形式化；(ii) 在 pay-per-token 下分析撒谎激励，并构造两个具体的 reporting policy（一个无 GPU 代价但容易被识破，一个用 GPU 验证但合理性更高）来证明撒谎在现实可行；(iii) 推导激励兼容定价机制的充要刻画，证明 pay-per-character 是唯一可行解，并给出 tpc 公式让服务商无痛迁移。

输入是一个用户 prompt $q$，服务商把它喂进 LLM 得到真实 token 序列 $\mathbf{t}$（字符串记为 $s = \mathrm{str}(\mathbf{t})$），然后用某个 reporting policy $\pi$ 上报 $\tilde{\mathbf{t}} \sim \pi(\mathbf{t})$，且约束 $\mathrm{str}(\tilde{\mathbf{t}}) = s$（即用户看到的文本没变）。服务商效用 $U_\pi(\tilde{\mathbf{t}}, \mathbf{t}) = r(\tilde{\mathbf{t}}) - c_\text{gen}(\mathbf{t}) - c_\pi(\mathbf{t})$，其中 $c_\text{gen}(\mathbf{t}) \approx c_o \cdot \mathrm{len}(\mathbf{t})$ 是 GPU 生成成本，$c_\pi$ 是执行 reporting policy 本身的成本（无验证的策略 $c_\pi = 0$，需要 forward pass 验证合理性的策略 $c_\pi = c_v$ 为常数）。

### 关键设计

1. **撒谎激励的形式化与无验证启发式 (Algorithm 1)**：

    - 功能：证明 pay-per-token 下"上报越长效用越高"，并给出一个最简单的 baseline 撒谎策略 $\pi_m^R$。
    - 核心思路：在 pay-per-token 下 $r(\tilde{\mathbf{t}}) = r_o \cdot \mathrm{len}(\tilde{\mathbf{t}})$，因此对任何成本相同的两个策略 $\pi, \pi'$，只要 $\mathrm{len}(\tilde{\mathbf{t}}) > \mathrm{len}(\tilde{\mathbf{t}}')$ 必有 $U_\pi > U_{\pi'}$——上报序列越长，利润越高。Algorithm 1 给出一个零计算的实现：维护当前序列 $\tilde{\mathbf{t}}$，每轮枚举所有"可被进一步切成两个非空词"的 token，从合法切分中随机挑一个执行，重复 $m$ 次，直到所有 token 都是单字符或达到迭代上限。整个过程不需要 GPU，因为不验证合理性。
    - 设计动机：先用一个"傻"算法戳破"pay-per-token 没事"的假象——实验显示在竞争场景里，作弊服务商只要把序列拉长 $1/\alpha$ 倍，就能在每 token 单价比对手低 $\alpha$ 倍时拿到同样收入但抢更多用户，把"撒谎"变成抢市场的武器。

2. **可信赖的启发式撒谎算法 (Algorithm 2) 与 NP-Hard 屏障**：

    - 功能：在服务商被要求公开 next-token 分布（top-$p$ / top-$k$ sampling）的"透明"场景下，仍然找到看起来合理（plausible）且比真实更长的 token 序列。
    - 核心思路：先证明"找最长的合理分词"是 NP-Hard（Theorem 3，从 Hamiltonian Path 归约），所以服务商不可能在多项式时间内最优作弊。但作者观察到一个经验规律——离 BPE 最常见分词"近"的替代分词在实际模型下几乎都合理。Algorithm 2 利用 BPE 中"高 id ≈ 长 token"的性质，每轮挑当前序列里 id 最大的 token $t_i$，把它切成两个 token $(t_1', t_2')$ 使得 $\min(\mathrm{id}(t_1'), \mathrm{id}(t_2'))$ 尽可能大（max-min 启发式），切 $m$ 次后跑一次 forward pass 验证 $\hat{t}_i \in \mathcal{V}_p(\hat{\mathbf{t}}_{\leq i-1})$ 对所有 $i$ 成立——若合理则上报，否则退回真实序列。盈亏判据是 $\mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot r_o > c_v$，即"合理时多收的钱 > 一次验证成本"。
    - 设计动机：要回答"NP-Hard 是不是就保护了用户"。作者明确回答：不。验证一次 forward pass 的能耗是个与序列无关的常数 $c_v$，而每多切一个 token 多收一个 $r_o$，在主流利润率下 Algorithm 2 在 Llama-3.2-1B 上对 $p=0.99$ 能把超额收入做到 10.5%+ 且总效用始终为正。

3. **激励兼容定价机制的充要刻画 + 平滑迁移公式**：

    - 功能：刻画所有可加且激励兼容的定价机制，并给出从 pay-per-token 切换到 pay-per-character 时维持平均利润率的具体处方。
    - 核心思路：先证 Proposition 5——激励兼容意味着 $r(\tilde{\mathbf{t}})$ 只依赖于字符串 $\mathrm{str}(\tilde{\mathbf{t}})$ 而不依赖具体分词（否则服务商总能选最贵的那种分词撒谎）。再证 Theorem 6——可加 + 激励兼容当且仅当 $r(\mathbf{t}) = \sum_{\sigma \in \Sigma} \mathrm{count}_\sigma(\mathbf{t}) \cdot r(\sigma)$，即按字符（character）线性计费；若每字符同价 $r_c$，则 $r(\mathbf{t}) = |\mathrm{str}(\mathbf{t})| \cdot r_c$ 是唯一选择。推论 7 直接断言：pay-per-token 在 vocabulary 含多字符 token 时一定不激励兼容。迁移公式：$r_c = r_o \cdot \mathrm{tpc}$，其中 tpc 是数据集上"每输出的 token/字符 比"的样本平均，能让平均利润率不变。
    - 设计动机：把"该用什么计费"这种工程问题压缩成一条数学定理——不是建议而是充要条件。同时承认 pay-per-character 会让单条样本的利润率波动（因为生成成本与 token 数线性、收入与字符数线性、每 token 字符数会变），但这反过来给服务商一个良性激励：发明更好的 tokenizer / 模型去多压缩字符。

### 损失函数 / 训练策略
本文是机制设计 / 理论 + 实证论文，没有训练 loss。关键超参数：top-$p$ sampling 的 $p \in \{0.90, 0.95, 0.99\}$、温度 $T = 1.3$、Algorithm 2 的迭代数 $m$（实验显示对每个 $p$ 都存在一个单峰最优 $m$）、利润率 $\rho_o \in \{0.2, 0.4, 0.6\}$。盈亏判据 $\rho(\mathbf{t}) > 1 - \mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot c_o / c_v$ 把"撒谎是否赚"明确成利润率与验证成本之比，可以直接对照真实服务的能耗账单计算。

### 形式化定义速查
- **可加性 (additive)**：$r(\tilde{\mathbf{t}}) = \sum_i r(\tilde{t}_i)$，pay-per-token 是最简单的可加机制。
- **激励兼容 (Definition 4)**：对任意 $\mathbf{t}, \pi, \tilde{\mathbf{t}} \sim \pi(\mathbf{t})$，$U_{\pi_0}(\mathbf{t}, \mathbf{t}) \geq U_\pi(\tilde{\mathbf{t}}, \mathbf{t})$，即如实上报的策略 $\pi_0$ 是弱占优。
- **合理性 (plausibility) under top-$p$**：每一步 $\hat{t}_i \in \mathcal{V}_p(\hat{\mathbf{t}}_{\leq i-1})$，其中 $\mathcal{V}_p$ 是累积概率 $\geq p$ 的最小候选集。

## 实验关键数据

### 主实验
评测在 LMSYS Chatbot Arena 的 600 条 prompt 上，模型用 Llama / Gemma / Ministral 系列，温度 1.3，top-$p$ 采样。

| 模型 / 配置 | $p=0.99$ 超额收费 | $p=0.95$ 超额收费 | $p=0.90$ 超额收费 | 说明 |
|--------|------|------|------|------|
| Llama-3.2-1B + Alg. 2 | 11.2% | 1.8% | 0.28% | $p$ 越大、采样空间越宽，合理性约束越松，作弊空间越大 |
| Llama-3.2-1B 效用增益 (任何 $\rho_o$) | >10.5% | 1.7% | 0.4% | 即扣掉验证 forward pass 成本后仍净赚 |
| pay-per-char 后利润率 > 0 比例（$\rho_o = 0.2 / 0.4 / 0.6$） | — | 92.4% / 93.3% / 94.1% | — | 迁移后绝大多数样本仍盈利 |

### 消融实验
| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Algorithm 1（无验证） | $m$ 越大超额收费越多（线性），但合理性快速崩到 ~0 | 证明"不验证"撒谎容易被反查 |
| Algorithm 2 关于 $m$ | 超额收费 vs $m$ 呈单峰，最优 $m$ 随 $p$ 减小而减小 | $m$ 太大合理性掉得太快，反而拉低期望收益 |
| 利润率 $\rho_o$ 变化 | $\rho_o$ 越小相对效用增益越大 | 价格战中作弊收益相对放大，激励反而更强 |

### 关键发现
- 关键设计 2 贡献最大：Algorithm 2 + Theorem 3 共同回答了"透明能否堵住作弊"，结论是"理论上半堵（NP-Hard），实际上不堵（启发式照样赚 10%）"。
- $p$ 越大可作弊空间越大，但 $p$ 也是真实生成的多样性旋钮——这把"高温/高 $p$"创意写作场景指认为最脆弱场景。
- 利润率越低撒谎的相对回报越高，意味着低毛利的小服务商作弊动机最强；这与"小服务商口碑成本低"的市场现实叠加形成系统性风险。

## 亮点与洞察
- 用 Hamiltonian Path 归约把"最长合理分词"打成 NP-Hard，然后立刻用 max-min id 启发式打脸"那 NP-Hard 总安全了吧"——理论与实证组合拳非常清晰，证明算法复杂度并不等于经济安全。
- Theorem 6 把"什么计费方式才能消除撒谎激励"压成一个充要刻画，pay-per-character 不是众多候选之一而是（同价字符前提下）唯一解，这种"必然性"叙述对推动政策极有力。
- $r_c = r_o \cdot \mathrm{tpc}$ 这条迁移公式可以直接套用到任何现有 API，不改模型不改 tokenizer 不改架构，迁移成本仅是一次数据集统计，非常工程友好。

## 局限与展望
- 作者承认：pay-per-character 不能阻止服务商让模型啰嗦（artificially verbose）多输出字符；这种攻击需要质量度量类机制（如 Saig et al. 2025 的 pay-for-performance）配合。
- 假设服务商不能伪造 next-token 分布本身或换 tokenizer——对闭源模型这层假设其实很强；作者建议用 trusted execution environments / zero-knowledge proofs 解决。
- 实验只在开源模型（Llama / Gemma / Ministral）上做，prompt 来自 LMSYS Chatbot Arena 这个被批评过代表性不足的平台；闭源模型 + 真实生产流量上的效果仍开放。
- 分析只到单用户-单服务商微观层，多服务商竞争 + 用户选择反馈这类宏观市场动力学留给后续。

## 相关工作与启发
- **vs Saig et al. (2025)**：他们也用 principal-agent，但针对的是"服务商用便宜模型却按贵模型计费"这种模型替换攻击，提出 pay-for-performance；本文针对的是"同一模型内的分词重报"，两者正交互补。
- **vs Sun et al. (2025) / Cai et al. (2025)**：这两篇是审计/检测向（验证 reasoning step 是否被注水 / 验证模型是否被替换），本文是机制设计向（直接换计费规则消除激励），层次不同。
- **vs Ahia et al. (2023)**：他们指出不同语言被 BPE 切出来的 token 数差异巨大，导致非英语用户在 pay-per-token 下被天然多收费；本文的 pay-per-character 顺带也修了这个公平性问题——多语言用户每字符同价。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一篇把 LLM 定价当作机制设计问题严肃刻画，并给出充要定理的工作。
- 实验充分度: ⭐⭐⭐⭐ 三族开源模型 + 600 条多语言 prompt 覆盖了主要情形，但缺闭源模型与生产级流量验证。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链清晰：建模 → 揭露激励 → 启发式实证 → 不可避免性定理 → 平滑迁移配方，一气呵成。
- 价值: ⭐⭐⭐⭐⭐ 直接面向 LLM 商业化的核心定价规则，结论可直接进入监管与合同条款讨论，影响面很大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Your Students Don't Use LLMs Like You Wish They Did](../../ACL2026/dialogue/your_students_dont_use_llms_like_you_wish_they_did.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](../../ACL2025/dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ICML 2025\] Investigating Non-Transitivity in LLM-as-a-Judge](../../ICML2025/dialogue/investigating_non-transitivity_in_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
