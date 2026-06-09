---
title: >-
  [论文解读] ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity
description: >-
  [ICLR 2026][多语言/翻译][autoformalization] 提出 ASSESS 框架，其核心是 TransTED Similarity 指标——通过将形式化数学命题解析为算子树 (Operator Tree)，并在标准树编辑距离 (TED) 基础上融入 Lean 证明策略驱动的语义变换…
tags:
  - "ICLR 2026"
  - "多语言/翻译"
  - "autoformalization"
  - "evaluation metrics"
  - "tree edit distance"
  - "Lean"
  - "formal mathematics"
---

# ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity

**会议**: ICLR 2026  
**arXiv**: [2509.22246](https://arxiv.org/abs/2509.22246)  
**代码**: [https://github.com/XiaoyangLiu-sjtu/ASSESS](https://github.com/XiaoyangLiu-sjtu/ASSESS)  
**领域**: 形式化数学 / 评估指标  
**关键词**: autoformalization, evaluation metrics, tree edit distance, Lean, formal mathematics

## 一句话总结

提出 ASSESS 框架，其核心是 TransTED Similarity 指标——通过将形式化数学命题解析为算子树 (Operator Tree)，并在标准树编辑距离 (TED) 基础上融入 Lean 证明策略驱动的语义变换，实现了在 EPLA 基准上 70.16% 准确率和 0.35 Kappa 分数的 SOTA 性能，同时仅需 CPU 资源即可复现。

## 研究背景与动机

**领域现状**：自动形式化 (autoformalization) 是将自然语言数学命题翻译为 Lean、Isabelle 等形式化证明语言的任务，随着 LLM 的发展进步迅速。然而，如何自动评估翻译质量一直是悬而未决的关键问题——缺乏可靠的评估指标严重制约了这一领域的发展。

**现有痛点**：现有评估方法存在语义与结构不可兼得的根本困境。基于文本的指标（如 BLEU）只看 n-gram 重叠，对词汇微小变化敏感但完全忽视语义——两个语义等价但写法不同的 Lean 表达式（如 $a+b$ 和 $b+a$）可能获得很低的 BLEU 分数。基于证明的指标（如 BEq）试图通过自动证明两个命题的等价性来判断相似度，但过于严格——许多高度相似但不完全等价的命题被直接判为不相似，且受限于当前定理证明器的能力（高 false negative 率）。LLM-as-Judge 方法虽然灵活，但存在不可复现、计算成本高、需要 GPU 等问题。

**核心矛盾**：语义理解和结构匹配之间的 trade-off——要么过于宽松（BLEU 只看表面），要么过于严格（BEq 要求完全等价证明），缺乏中间地带的细粒度评估能力。

**切入角度**：形式化数学命题可以被解析为算子树 (Operator Tree, OPT)，而 Lean 中的 tactic 操作对应树上的语义保持变换。将语义变换信息融入树编辑距离计算，可以在结构匹配的基础上引入语义感知能力——既不像 BLEU 那样忽视语义，也不像证明方法那样要求完全等价。

**核心 idea**：用语义变换增强的树编辑距离（TransTED）来衡量形式化命题相似度，在纯结构和纯语义方法之间找到了最佳平衡点。

## 方法详解

### 整体框架

ASSESS 想解决的是"两条形式化命题到底有多像"这个问题——既不能像 BLEU 那样只比字面、把 $a+b$ 和 $b+a$ 判成不像，也不能像 BEq 那样非得证出完全等价、把"很接近但不全等"的翻译一刀切掉。它的做法分两步：先把每条形式化命题用 Lean Language Server 解析成一棵**算子树 (OPT)**，把符号序列变成带层次结构的树；再在标准**树编辑距离 (TED)** 之上，借 Lean 的 tactic 注入语义变换，算出最终的实数相似度 **TransTED Similarity**。整条流水线只依赖 CPU 和 Lean Language Server/REPL，不需要 GPU，因此可复现、成本低。

### 关键设计

**1. 算子树 (OPT)：把符号序列变成带结构的树。**

纯文本表示丢掉了形式化命题里最关键的层次信息——谁是算子、谁是操作数、运算优先级怎么嵌套。ASSESS 用 Lean Language Server 把命题解析成算子树：函数应用、量词、逻辑连接词这类**算子成为内部节点**，它们的操作数成为有序子节点。构建时做两项标准化：非叶节点统一加 `<SLOT>` 占位标签来区分算子位和操作数位；括号一律省略——因为优先级信息已经隐含在树的嵌套结构里。这样得到的树比序列表示能更自然地编码运算符优先级和依赖关系，为后面按结构比对打好基础。

**2. TED Similarity：先用树编辑距离量出结构差异。**

有了树就能比结构。这里把所有 OPT 看成一个**伪度量空间 (pseudometric space)**——故意允许 $d(x,y)=0$ 但 $x \neq y$，因为"语义等价但写法不同"的两棵树，距离本就应该判为 0。树编辑距离 $d_{\text{TED}}$ 定义为把一棵树经删除、插入、重标记操作变成另一棵树的最小代价，再归一化成相似度：

$$\text{sim}_{\text{TED}}(T_1, T_2) = 1 - \frac{d_{\text{TED}}(T_1, T_2)}{\max(|T_1|, |T_2|)}$$

其中 $|T|$ 是节点数。TED 在纯结构匹配上已经够用，但对语义等价表达式有系统性偏差：$a+b$ 和 $b+a$ 这种交换律等价的式子，在树上要走好几步编辑才能对齐，距离被算大了——这正是下一步要补的洞。

**3. TransTED Similarity：用 Lean tactic 把语义变换喂给距离度量。**

核心改进是让距离"懂"语义等价。论文定义了一个新的伪度量 $d^*$，要它同时满足两条约束：(a) 被 TED 上界压住，$d^*(T_1, T_2) \leq d_{\text{TED}}(T_1, T_2)$；(b) **语义变换单调性**——若命题对 $(e_x, e_y)$ 能被变换成逻辑上更强的对 $(e_u, e_v)$（即 $e_u=e_v \Rightarrow e_x=e_y$），则 $d^*(OPT(e_x), OPT(e_y)) \leq d^*(OPT(e_u), OPT(e_v))$。论文证明满足这两条的最大伪度量唯一存在 (Theorem 1)，这就是 TransTED。落到实现上：先把待比的一对命题用等号连成一个等式，再在 Lean REPL 里施加一组策划好的 tactic（如 `rw?`、`apply congrArg`、`ext`、`norm_cast` 等）做启发式搜索，并**用 TED 当启发函数**，优先尝试那些能缩小两侧 OPT 差异的变换。搜索在三种情况下停止：等价被证明、节点数超限 (NLE) 或时间超限 (TLE)。这样距离度量就"知道"交换律、量词分解等语义等价关系，在不丢结构信息的前提下拿到语义感知能力——既不像 BLEU 忽视语义，也不像 BEq 要求全等证明。

**4. EPLA 基准：给评估指标本身配一把标尺。**

要验证一个相似度指标到底好不好，得有人工标注的"真相"做对照，论文为此构建了 EPLA (Evaluating Provability and Likeness for Autoformalization) 基准。它取 miniF2F-test 和 ProofNet-test 的自然语言命题，用 4 个翻译模型（Herald Translator、Goedel-Formalizer-V2-8B、Gemini-2.5-Pro、Qwen3-Max）生成形式化翻译，经 Lean 编译器过滤后，由 7 位专家逐对标注语义等价性和结构相似性，最终得到 1,247 个标注对（831 来自 miniF2F，416 来自 ProofNet）。这套带专家标签的数据正是后面所有指标对比的裁判依据。

## 实验关键数据

### 主实验

| 指标 | Identity Match | BLEU | Majority Voting | BEq | TED Sim | **TransTED Sim** |
|------|---------------|------|----------------|-----|---------|-----------------|
| miniF2F Accuracy | 32.61% | 68.96% | 46.93% | 59.45% | 69.56% | **70.16%** |
| miniF2F Kappa | 0.05 | 0.26 | 0.14 | 0.29 | 0.31 | **0.35** |
| ProofNet Accuracy | 43.51% | 57.21% | 54.57% | 60.34% | 64.67% | **67.31%** |
| ProofNet Kappa | 0.03 | 0.18 | 0.12 | 0.28 | 0.23 | **0.30** |

TransTED Similarity 在两个数据集上的准确率和 Kappa 分数均达到 SOTA。与 BEq（基于证明的方法）相比，Kappa 从 0.29 提升到 0.35（miniF2F），说明语义变换增强的结构对比比刚性证明方法更好地平衡了精确率和召回率。

### 消融实验

| 配置 | miniF2F Acc | miniF2F Kappa | ProofNet Acc | ProofNet Kappa |
|------|------------|---------------|-------------|----------------|
| TED Similarity (无变换) | 69.56% | 0.31 | 64.67% | 0.23 |
| **TransTED Similarity (有变换)** | **70.16%** | **0.35** | **67.31%** | **0.30** |
| 提升幅度 | +0.60pp | +0.04 | +2.64pp | +0.07 |

语义变换组件是性能增益的关键因素，尤其在 ProofNet 上 Kappa 提升了 0.07，说明变换在处理更复杂的数学命题时更具优势。

### 关键发现

- **证明方法（BEq）的高精确率低召回率问题**：BEq 在 miniF2F 上精确率达 98.60% 但召回率仅 45.77%，说明自动定理证明器频繁将合法等价判为不等价。TransTED 通过渐进式变换避免了这种刚性依赖。
- **Tactic 使用模式的互补性**：`rw?` 和 `norm_cast` 是高频通用工具（探索搜索空间），而 `apply forall_congr; intro _` 和 `rw [propext and_imp]` 是低频但高采纳率的专用工具（精确解决特定逻辑结构）。通用+专用的协同构成了稳健的搜索策略。
- **TransTED 在不同阈值下表现稳定**：相比 BLEU 对阈值选择敏感，TransTED 在较宽的阈值区间内都能保持高性能。

## 亮点与洞察

- **伪度量空间的数学优雅性**：允许 $d(x,y)=0$ 但 $x \neq y$——这正好对应形式化数学中"语义等价但语法不同"的核心需求，比要求度量空间的严格指标更适合这个问题。
- **Lean tactic 作为语义桥梁**：巧妙地利用了证明助手自身的 tactic 系统作为语义变换引擎——不需要训练任何模型，不需要 GPU，只需 Lean Language Server 和一组精选 tactic，实现了高效且可复现的语义感知。
- **搜索启发函数与度量统一**：用 TED 同时作为最终度量的组成部分和搜索过程的启发函数，设计上很简洁——减小 TED 的变换就是有意义的语义等价步骤。

## 局限与展望

- **变换集有限**：当前仅使用了少量手工策划的 tactic 命令，覆盖的语义变换类型有限。更多的 Lean tactic 或自定义变换规则可能进一步提升性能。
- **搜索效率**：单个命题对的评估需要最长 10 分钟的搜索时间，大规模应用时可能成为瓶颈。
- **TransTED 计算的是上界**：由于实际变换集合有限，计算出的值是理论最优 TransTED 的上界。只有当变换序列成功证明等价时，距离才精确为0。
- **EPLA 规模有限**：1,247 个标注对虽然精心标注，但在多样性和覆盖面上可能不够——更广泛的数学领域和更多形式化语言的支持是自然的扩展方向。

## 相关工作与启发

- **vs BLEU**: BLEU 将形式化命题视为普通文本，完全忽视数学语义。TransTED 在保留结构匹配的同时引入语义感知，在 EPLA 上 Kappa 从 0.26 提升到 0.35。
- **vs BEq (基于证明)**: BEq 要求完全等价的证明成功，是一种"全或无"的评估。TransTED 提供连续值的相似度分数，更适合评估"接近正确"的翻译质量梯度。
- **vs GTED (Liu et al., 2025c)**: GTED 是首个使用算子树进行形式化评估的工作，但存在实现不稳定和变换机制仅限于变量重命名的问题。ASSESS 将其形式化为严格的伪度量空间理论，并引入了基于证明的丰富变换集合。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 Lean tactic 系统作为语义变换引擎融入树编辑距离是巧妙的创新，伪度量空间理论框架也很漂亮
- 实验充分度: ⭐⭐⭐⭐ 基线丰富，消融分析到位，tactic 使用模式分析有洞见；但 EPLA 规模偏小
- 写作质量: ⭐⭐⭐⭐ 理论部分形式化清晰，实验部分条理分明
- 价值: ⭐⭐⭐⭐ 为 autoformalization 社区提供了更可靠、高效、可复现的评估工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment](../../ICML2025/multilingual_mt/kelps_a_framework_for_verified_multi-language_autoformalization_via_semantic-syn.md)
- [\[ACL 2025\] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models](../../ACL2025/multilingual_mt/statement-tuning_enables_efficient_cross-lingual_generalization_in_encoder-only_.md)
- [\[ACL 2026\] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](../../ACL2026/multilingual_mt/reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](../../ACL2026/multilingual_mt/fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](../../ACL2026/multilingual_mt/the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)

</div>

<!-- RELATED:END -->
