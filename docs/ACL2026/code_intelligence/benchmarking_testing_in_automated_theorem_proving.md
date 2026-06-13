---
title: >-
  [论文解读] Benchmarking Testing in Automated Theorem Proving
description: >-
  [ACL 2026][代码智能][自动定理证明] 借鉴软件工程「集成测试」思想，把生成定理的语义正确性判定为「所有依赖于它的后继定理是否仍能编译通过」，构建 2206 题的 Lean 4 基准 T2，揭示出主流 LLM 编译通过率 80%+ 但语义正确率只有 ~39% 的巨大缝隙。 领域现状：LLM 自动定理证明（ATP）近…
tags:
  - "ACL 2026"
  - "代码智能"
  - "自动定理证明"
  - "Lean 4"
  - "语义正确性"
  - "集成测试"
  - "Cut Elimination"
---

# Benchmarking Testing in Automated Theorem Proving

**会议**: ACL 2026  
**arXiv**: [2604.23698](https://arxiv.org/abs/2604.23698)  
**代码**: <https://github.com/ldilab/T2>  
**领域**: 代码智能 / 定理证明 / LLM 评测  
**关键词**: 自动定理证明、Lean 4、语义正确性、集成测试、Cut Elimination

## 一句话总结
借鉴软件工程「集成测试」思想，把生成定理的语义正确性判定为「所有依赖于它的后继定理是否仍能编译通过」，构建 2206 题的 Lean 4 基准 T2，揭示出主流 LLM 编译通过率 80%+ 但语义正确率只有 ~39% 的巨大缝隙。

## 研究背景与动机

**领域现状**：LLM 自动定理证明（ATP）近两年快速进展，MiniF2F、ProofNet、PutnamBench 等基准用「编译是否通过」作为唯一正确性指标，DeepSeek-Prover、Kimina、Goedel 等专用 prover 都以此为优化目标。

**现有痛点**：编译通过 ≠ 语义正确。一个把 $a+b=b+a$ 形式化成 $a+b=a+b$ 的同义反复定理在 Lean 里也能编译通过，但完全没有捕捉到加法交换律的本意。现有补救方案要么用 BLEU/NLI 这类表层相似度（不可靠），要么靠 prover-based equivalence 检查（要 ground-truth 参考），要么靠人工检查（不可扩展）。

**核心矛盾**：理想的语义评估应该「无参考、自动、直接验语义」三者兼顾，但已有方法每条只满足其中一两条。

**本文目标**：在不依赖人工标注或参考答案的前提下，自动判定生成定理的语义正确性，并且这一判定要能区分专用 prover 与通用 LLM 的真实能力差距。

**切入角度**：Curry-Howard 同构告诉我们「证明 = 程序，定理 = 类型」。代码生成早就用单元测试代替了 BLEU——只要让所有调用该函数的下游测试通过，就可以认为它行为正确。把这一思路搬到定理证明：把待评定理 $t_{fl}$ 当作一个「函数」，把所有依赖它的后继定理 $\mathcal{T}_{succ}$ 当作「集成测试用例」，如果替换后整条依赖链都还能编译，就有强证据表明 $t_{fl}$ 语义正确。

**核心 idea**：用「Testing Accuracy (TA)」替代「Compilation Accuracy」——把生成定理代入真实 Lean 仓库，看所有调用它的后继定理是否仍能 `lake build` 通过。

## 方法详解

### 整体框架
T2 是一套「无参考、自动、直接验语义」的 ATP 评测协议，整体只做两件事：定义指标、构造数据。核心动作是把 LLM 生成的形式化定理 $t_{fl}$ 替换进真实 Lean 4 仓库中的原始定理 $t_{GT}$（保持同名），再 `lake build` 重编整条依赖链，只有所有依赖它的后继定理 $t_{succ}^{(i)}$ 都还能编译通过，才判这条形式化「语义正确」。数据侧从 5 个高质量 Lean 4 仓库自动挖出 2206 个目标定理，每个平均带 41 个后继定理充当「集成测试用例」。评测全程 zero-shot（temperature=0.6, top_p=0.95），每题生成一份完成、编译超时 600s，并把 NL proof 与 successor theorem 一起喂进 context。

### 关键设计

**1. 将集成测试形式化为 Cut Elimination：给「后继能编译=语义正确」一个证明论根据**

把「下游测试全过就认为函数行为正确」直接搬到定理证明，本来只是个工程类比，本文用证明论把它坐实。在 sequent calculus 里，证明 $X \to W$ 可以经由中间引理形成 cut 链 $X \to Y \to Z \to W$，而 cut elimination 定理保证：只要整条链都成立，中间引理就能被消去、还原出 $X \to W$ 的直接证明。把待评定理 $t_{fl}$ 看作 $X \to Y$、后继定理 $\mathcal{T}_{succ}$ 看作 $Y \to Z \to W$，于是「后继链能编译」恰好等价于「cut 可消除」，对应 $t_{fl}$ 语义正确。这一步不仅让启发式合法化，还顺带解释了为什么「后继越多、链越深，TA 越逼近真正的语义保证」。

**2. Testing Accuracy (TA) 度量：把上面的思路落成一个二值 pass/fail**

对每个问题 $P = (t_{nl}, \mathcal{T}_{pred}, \mathcal{T}_{succ})$，TA 定义为 $\texttt{TA} = \mathbb{E}_{P \sim \mathcal{D}}\left[\bigwedge_{i=1}^{k}\texttt{compiles}(t_{succ}^{(i)} \mid t_{fl})\right]$，即「$t_{fl}$ 替换 $t_{GT}$ 后，$k$ 个后继定理全部编译通过」记 1，否则记 0。为保证覆盖有意义，只收录 $|\mathcal{T}_{succ}| \geq 2$ 的目标定理，并用各仓库原生 Lean 版本 `lake build` 跑完整依赖链。相比 BLEU 给出的「0-100 连续分但与正确性弱相关」，TA 直接对齐软件工程的 unit-test pass rate，是个能横向比较模型语义能力的硬指标。

**3. 依赖感知的数据集构造管线：从真实仓库挖出天然嵌在依赖结构里的目标定理**

现有基准多是孤立的奥赛题，几乎没有后继可供 TA 评测（满足条件的只有 12 题），所以数据必须自己造。管线先用 Lean-Dojo 解析出全局依赖图 $G=(V,E)$，再按两条规则筛目标定理：非平凡性（到后继的距离 > 1，排除孤立定理）与后继覆盖度（$|\mathcal{T}_{succ}| \geq 2$）。每个目标定理用 Claude Sonnet 4.5 生成自然语言描述 $t_{nl}$（人工抽样 10 条全部正确），并进一步圈出 389 题的 Hard 子集（目标为 Prop 类型，需同时生成命题和证明）。这样挖出的 2206 个目标定理平均深度可达 7、平均后继数 1.6k，每个后继都注入一条独立的语义约束，形成远比孤立题严格的评测。

## 实验关键数据

### 主实验
在 18 个模型（Claude、GPT、Llama、DeepSeek、Kimina、Goedel 等）上对比编译通过率与 TA：

| 模型 | T2 Compile | T2 TA | T2 Hard Compile | T2 Hard TA |
|------|-----------|-------|-----------------|-----------|
| Claude-Sonnet-4.5 | 80.3 | **38.9** | 46.0 | **4.5** |
| GPT-5 | 85.7 | 37.7 | 68.3 | 3.4 |
| GPT-5-nano | 88.7 | 36.6 | 75.6 | 1.5 |
| DeepSeek-Prover-v2-7B | 62.2 | 30.0 | 35.5 | 3.2 |
| Kimina-Autoformalizer-7B | 21.9 | 20.0 | 4.3 | 1.5 |

最强模型 Claude-Sonnet-4.5 在 Full 集上编译率 80.3% 但 TA 只有 38.9%（差距 2×）；GPT-5-nano 在 Hard 上编译率 75.6% 但 TA 仅 1.5%（差距 50×）。

### 消融实验（输入 context 影响，T2 Full）

| 模型 | NL✗ ST✗ | NL✓ ST✗ | NL✗ ST✓ | NL✓ ST✓ |
|------|---------|---------|---------|---------|
| Claude-Sonnet-4.5 | 34.0 | 33.0 | 32.9 | **38.9** |
| Claude-4-Sonnet | 30.0 | 27.7 | 32.3 | **36.0** |
| Llama-3.1-70B | 28.5 | 28.9 | 29.1 | **37.0** |

NL proof 和 successor theorem **必须同时提供**才能稳定涨点；单独给 successor theorem 反而可能降分，说明模型需要 NL 提供推理上下文、需要 successor 提供语义约束，二者缺一不可。

### 关键发现
- **编译率是语义正确性的坏代理**：以编译通过预测语义正确的 precision 只有 6.89%；高 BLEU（90-100 分段）的样本中 >70% 仍语义错误。
- **专用 prover 在语义维度反而更差**：Goedel/Kimina/DeepSeek-Prover 等 specialized 模型相对通用 LLM 在 compilation 上涨、在 TA 上反降，说明 domain fine-tuning 教会的是「语法流畅」而非「语义对齐」。
- **后继数越多 TA 越严格**：仅 2 个后继时 Claude-4-Sonnet TA ~70%，达到 5 个后继时跌至 ~0%，每个后继都注入一条独立的语义约束；本基准平均深度 7、平均后继 1.6k，提供了非常严格的评测。
- **few-shot 和迭代 refinement 都无法显著缩小语义缝**：2-shot 在 Hard 上甚至有时跌点；Hilbert 借助编译反馈也只把 Hard TA 从 3.2% 推到 5.0%。

## 亮点与洞察
- 把 Curry-Howard 同构和软件集成测试串到一起，给「测试 = 证明语义评估」一个清晰的证明论解释（cut elimination），让方法不只是工程比喻而有理论支撑。
- 「编译率 80% / TA 4%」的 50× 缝隙非常震撼，揭示了过去几年 ATP 基准被大幅 over-claim 的现象——这就像代码生成时代如果只看「能不能编译」会高估模型能力一样。
- 数据集构造完全无人工：从公开 Lean 仓库的依赖图自动挖目标 + 后继，再用 LLM 生成 NL 描述，2206 题成本只有 NL 标注的 ~100 美元；这种「真实仓库 + 依赖结构」的范式可以无缝迁移到 Coq、Isabelle 等其他 proof assistant。

## 局限与展望
- TA 是二值的，无法刻画「9 个后继通过 1 个失败」与「全部失败」之间的差异；作者建议未来做分级 TA。
- 不适用于无后继的孤立定理（约 1.4%）——如仓库最前沿新加的定理，这部分只能退回 BLEU/人工。
- 仅支持 Lean 4，且数据来自 5 个仓库（主要是研究级数学）；NL 标注由 Claude Sonnet 4.5 生成，可能引入与目标 prover 同源的偏置。
- 真正语义错的反例反而很难造出来——因为名字和类型签名都给定，模型一旦写错就直接编译失败；TA 主要捕捉的是「类型对但内容错」的细粒度错误。

## 相关工作与启发
- **vs MiniF2F / ProofNet / PutnamBench**：他们只查编译是否通过，本文额外查后继链是否通过；前者全是孤立问题、几乎无后继可测，T2 直接从依赖图筛选高后继定理。
- **vs ProofNetVerif / Con-NF**：那些方法用 prover-based 等价性检查或 BEq，需要 ground-truth 参考；T2 完全无参考，只看依赖链是否仍工作。
- **vs 代码生成中 HumanEval / MBPP**：HumanEval 用单元测试做 functional correctness，本文是把同一思路首次系统化引入形式化定理；Curry-Howard 同构给了这种迁移天然的理论基础。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把代码生成的 testing-based 评估范式整体搬到 ATP，且用 cut elimination 给出严格的理论解释，是非常漂亮的范式迁移。
- 实验充分度: ⭐⭐⭐⭐⭐ 18 个模型 × 4 种 context 设置 × Full+Hard+Existing 三种集合，配以 BLEU/BEq/编译率多种对照，论证非常扎实。
- 写作质量: ⭐⭐⭐⭐ 概念框架清晰，图 1 和 cut elimination 的类比把核心 idea 解释得很直观；附录把模型清单、prompt、成本都披露完整，可复现性强。
- 价值: ⭐⭐⭐⭐⭐ 直接推翻了「专用 prover 在 ATP 上越来越强」的乐观叙事，给整个社区一记当头棒喝；TA 这一指标很可能成为今后 ATP 论文的必备评测项。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](discover_and_prove_an_open-source_agentic_framework_for_hard_mode_automated_theo.md)
- [\[ACL 2026\] SOCIA-EVO: Automated Simulator Construction via Dual-Anchored Bi-Level Optimization](socia-evo_automated_simulator_construction_via_dual-anchored_bi-level_optimizati.md)
- [\[ACL 2026\] LogicEval: A Systematic Framework for Evaluating Automated Repair Techniques for Logical Vulnerabilities in Real-World Software](logiceval_a_systematic_framework_for_evaluating_automated_repair_techniques_for_.md)
- [\[ICLR 2026\] InnoGym: Benchmarking the Innovation Potential of AI Agents](../../ICLR2026/code_intelligence/innogym_benchmarking_the_innovation_potential_of_ai_agents.md)
- [\[ICML 2026\] CentaurEval: Benchmarking Human-in-the-Loop Value in Agentic Coding](../../ICML2026/code_intelligence/centaureval_benchmarking_human-in-the-loop_value_in_agentic_coding.md)

</div>

<!-- RELATED:END -->
