---
title: >-
  [论文解读] 结构化广义线性 token mixing：用 SND + Kronecker 在复杂度与表达力之间换挡
description: >-
  [ICML 2026][LLM/NLP][token mixing] 论文提出统一的"直接输入混合 $\mathbf{A}$ + 输出递归混合 $\mathbf{B}$"框架 $Y = (I - B)^{-1} A X$ 涵盖 attention/SSM/linear recurrence/高阶递归…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "token mixing"
  - "注意力机制"
  - "SSM"
  - "高阶递归"
  - "时间复杂度"
  - "cache size"
---

# 结构化广义线性 token mixing：用 SND + Kronecker 在复杂度与表达力之间换挡

**会议**: ICML 2026  
**arXiv**: [2605.31367](https://arxiv.org/abs/2605.31367)  
**代码**: 未列  
**领域**: 注意力 / 状态空间模型 / 高效序列建模  
**关键词**: token mixing, attention, SSM, 高阶递归, 时间复杂度, cache size

## 一句话总结
论文提出统一的"直接输入混合 $\mathbf{A}$ + 输出递归混合 $\mathbf{B}$"框架 $Y = (I - B)^{-1} A X$ 涵盖 attention/SSM/linear recurrence/高阶递归，证明 sparsity pattern of $A, B$ 直接控制 $\mathcal{O}(n^{\log n})$ 到 $\mathcal{O}(n^2)$ 的复杂度梯度，提出 $f(k) = 2^k$ 和 $f(k) = k^2+1$ 两种 translation-invariant 模式给出 $\mathcal{O}(n \log n)$ 和 $\mathcal{O}(n \sqrt{n})$ 的新选择，且 cache 可缩到 $\mathcal{O}(\log n)$ 或 $\mathcal{O}(\sqrt{n})$。

## 研究背景与动机

**领域现状**：token mixing 是序列模型的核心——决定模型怎么跨 token 交换信息。Transformer 用 self-attention 全局一跳，$\mathcal{O}(n^2)$；线性 attention 用 kernel 降到 $\mathcal{O}(n)$；状态空间模型（S4/Mamba）用线性递归 $\mathcal{O}(n)$；Hybrid（Griffin/Nemotron-H）混合多种 mixer。

**现有痛点**：(1) 不同 mixer 之间的关系一直在 case-by-case 分析（Mamba-2 跟 linear attention 的等价性、Chimera 跟 SSM-on-graph 的连接），缺乏统一框架；(2) 多数 mixer 是 first-order recurrence（只看前一个状态），但 Chen et al. 2025 证明 memory 必须随 $L^\beta$ 增长才能 scale，first-order 在长序列上必然受限；(3) Higher-order recurrence（log-linear attention、ChaCAL）零散出现但没有系统理论说明"什么 sparsity pattern 给什么复杂度+表达力"。

**核心矛盾**：要表达长程依赖就要全局连接（$\mathcal{O}(n^2)$ attention）；要快就只能 local/recurrent；夹在中间的设计空间（如 $\mathcal{O}(n \log n)$ 或 $\mathcal{O}(n \sqrt{n})$）几乎没人系统探索过。

**本文目标**：(1) 给所有 causal linear token mixer 一个统一形式；(2) 系统刻画 sparsity pattern → 复杂度/cache/表达力的对应关系；(3) 设计跨越 $\mathcal{O}(n)$ 到 $\mathcal{O}(n^2)$ 全谱的新 mixer。

**切入角度**：观察到任何 causal linear token mixer 都能写成"输入直接影响 + 输出递归传播"两部分。引入矩阵 $A$ (lower triangular, 输入影响) 和 $B$ (strict lower triangular, 输出递归)，则 $Y = AX + BY$ 即 $Y = (I-B)^{-1} A X$。Attention 是 $B=0$；recurrence 是 $A$ 对角 + $B$ 次对角；hybrid 通过混合 sparsity pattern 实现。

**核心 idea**：用 sparsity pattern of $A, B$ 当 design knob——不同 pattern 对应不同复杂度（time per token、cache size）和表达力（shortest path between tokens、graph congestion）；通过 $f(k) = 2^k$（指数）或 $f(k) = k^2+1$（平方）等 translation-invariant pattern 设计新 mixer 给出 $\mathcal{O}(n \log n)$ 和 $\mathcal{O}(n \sqrt{n})$ 中间档。

## 方法详解

### 整体框架

论文把任意 causal linear token mixer 拆成"输入直接影响 + 输出递归传播"两部分，写成 Generalized Linear Recurrence Layer $y_i = \sum_{j=1}^i \alpha_{i,j} x_j + \sum_{j=1}^{i-1} \beta_{i,j} y_j$，矩阵形式即 $Y = (I-B)^{-1} A X$（$A$ lower triangular 装输入影响，$B$ strict lower triangular 装输出递归，保证 $I-B$ 可逆）。核心观察是 $A, B$ 的 sparsity pattern 同时决定复杂度（pattern 越稀，矩阵-向量乘越快）和表达力（$(I-B)^{-1}$ 展开成 dense lower-triangular，仍能让远处 token 通过多跳递归互通），于是"设计序列 mixer"被还原成"选 sparsity pattern"这一件事。

### 关键设计

**1. 统一框架：用 $(A,B)$ sparsity pattern 收编所有 causal linear mixer**

以前比较 attention 和 SSM 全靠 case-by-case：Mamba-2 跟 linear attention 的等价、Chimera 跟 SSM-on-graph 的连接，每对关系都要单独推一遍。本文发现它们其实都是同一个 $Y=(I-B)^{-1}AX$ 在不同稀疏模式下的实例——standard attention 是 $B=0,\ A=\mathrm{softmax}(QK^\top/\sqrt{d_k})$，得 $Y=AX$ 复杂度 $\mathcal{O}(n^2)$；local attention 把 $A$ 收成 banded 得 $\mathcal{O}(nk)$；gated linear recurrence 是 $A$ 对角、$B$ 次对角，展开就是 $y_t=\alpha_{t,t}x_t+\beta_{t,t}y_{t-1}$；diagonal SSM 是 recurrence 的特殊参数化加 state expansion；Mamba-2 则等价于 1-semiseparable transformation matrix，正好搭起 SSM 和 masked linear attention 的桥。统一之后，"为什么 SSM 比 attention 快"翻译成"sparsity 越稀矩阵乘越快"，"为什么 attention 表达力强"翻译成"$A$ dense 则任意 pairwise 关系可表达"——这套矩阵代数语言是后面所有系统设计的地基。

**2. Translation-invariant pattern：单个函数 $f$ 调出 $\mathcal{O}(f^{-1}(n))$ 复杂度全谱**

Longformer、BigBird 那一代 sparse attention 的连接模式是手工拼的（滑窗加全局 token），既没有 principled 的复杂度阶梯，也说不清表达力损失多少。本文改用一个严格递增的 $f:\mathbb{N}_{\ge 0}\to\mathbb{N}_{>0}$ 生成 pattern：$\alpha_{i,j}\ne 0 \iff \exists k:\ j=i-f(k)$，即 token $i$ 只能回看距离恰为 $f(k)$ 的过去 token。取 $f(k)=2^k$ 时 token $i$ 只看 $i-1,i-2,i-4,\dots,i-2^{\lfloor\log_2 i\rfloor}$，每 token 只需 $\mathcal{O}(\log n)$ 次运算。Proposition 4.2 把这件事推广成一条规律：复杂度就是 $\mathcal{O}(f^{-1}(n))$——线性 $f$ 给 $\mathcal{O}(n)$、平方 $f$ 给 $\mathcal{O}(\sqrt n)$、指数 $f$ 给 $\mathcal{O}(\log n)$。于是"选 $f$"直接等于"选复杂度档"，让此前几乎空白的 $\mathcal{O}(n\log n)$、$\mathcal{O}(n\sqrt n)$ 中间地带变成一个可滑动的工程旋钮。

**3. Shortest path 与 Congestion：把表达力量化成图论指标**

长程依赖能力此前只能定性吹，本文用 communication graph $\mathcal{G}$（token $i\to j$ 有边当且仅当 $i-j=f(k)$）给出两个可计算指标。其一是最短路径 $d(i,j)=\min\{d:\exists a\in\mathbb{N}^d,\ \sum_k f(a_k)=i-j\}$，量化"信息从 $j$ 走到 $i$ 要几跳"：$f(k)=2^k$ 时 $d(i,j)\le\log_2(i-j)$（等于差值二进制里 1 的个数），$f(k)=k^2+1$ 时借 Lagrange 四平方和定理直接锁死 $d(i,j)\le 4$。其二是 congestion $C(\mathcal{G})=\min_{\mathcal{P}}\max_i \#\{p\in\mathcal{P}:i\in p\}$，量化"copy 这类任务有多少条信息路径被迫挤过同一个节点"：标准一阶递归把一切压进单个 hidden state，congestion $=n$，而 higher-order recurrence 能压到 $\log n$ 甚至 4（Proposition 4.7-4.8 给出紧上下界）。两个指标合在一起正好解释了 $f(k)=k^2+1$ 为何能同时拿到 $\mathcal{O}(\sqrt n)$ 复杂度、4 步路径和 4 congestion 这种漂亮 trade-off。

**4. Cache-efficient pattern：把 KV cache 从 $\mathcal{O}(n)$ 收到 $\mathcal{O}(f^{-1}(n))$**

translation-invariant pattern 解决了每 token 的算力，但 cache 仍是 $\mathcal{O}(n)$——任何过去 token $j$ 都可能被任意远的未来 token 回看，所以谁都不能丢。Definition 4.10 加一条约束：解码 token $i$ 时只允许 attend $S_{i-1}\cup\{i\}$ 里的位置，cache 大小随之收成 $\mathcal{O}(f^{-1}(n))$。Proposition 4.12 给出这些保留位置的 closed-form $S_i=\{a_k\lceil(i-f(k))/a_k\rceil:k\in\mathbb{N},\ f(k)<i\}$，其中 $a_{k+1}=a_k\lceil(f(k+1)-f(k))/a_k\rceil$——保留位置都落在步长为 $a_k$ 的 lattice 上做 quantize，呈周期结构，这种规整性让它对硬件 kernel 友好、可被实现端利用。

## 实验关键数据

### 复杂度 + 表达力对照表

| Structure | Time/token | Cache | Shortest path | Congestion | Copy% | Assoc recall% | Multi-hop% |
|---|---|---|---|---|---|---|---|
| Attention | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | 1 | 1 | 100.00 | 100.00 | 39.21 |
| Local attention | $\mathcal{O}(k)$ | $\mathcal{O}(k)$ | $\infty$ | 1 | 23.75 | 26.20 | 23.59 |
| Diagonal SSM | $\mathcal{O}(1)$ | $\mathcal{O}(1)$ | $n$ | $n$ | 42.98 | 32.53 | 27.17 |
| k-th order recurrence | $\mathcal{O}(k)$ | $\mathcal{O}(k)$ | $n/k$ | $n/k$ | 74.66 | 41.12 | 39.08 |
| Dense recurrence | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ | 1 | 1 | 100.00 | 99.99 | 99.80 |
| $f(k) = 2^k$ | $\mathcal{O}(\log_2 n)$ | $\mathcal{O}(n)$ | $\le \log_2 n$ | $\le \log_2 n$ | 92.63 | 49.03 | 34.85 |
| $f(k) = 2^k$ + cache-eff | $\mathcal{O}(\log_2 n)$ | $\mathcal{O}(\log_2 n)$ | — | — | 75.47 | 52.59 | 38.63 |
| $f(k) = k^2+1$ | $\mathcal{O}(\sqrt n)$ | $\mathcal{O}(n)$ | $\le 4$ | $\le 4$ | 99.66 | 53.61 | 35.68 |
| $f(k) = k^2+1$ + cache-eff | $\mathcal{O}(\sqrt n)$ | $\mathcal{O}(\sqrt n)$ | — | — | 91.59 | 54.56 | 38.02 |

### 关键发现

- **理论复杂度 → 实证表达力的清晰阶梯**：从 Diagonal SSM（最弱）到 Attention（最强），sparse pattern $f(k) = k^2+1$ 在 $\mathcal{O}(\sqrt n)$ 复杂度下 copy 99.66% 接近 attention 的 100%——证明设计空间里有"几乎免费的中间档"。
- **Dense recurrence (infinite order) 跟 attention 一样强**：100% copy + 99.99% assoc recall + 99.80% multi-hop，说明 recurrence 不是天生比 attention 弱，关键在 order。这呼应了 Chen et al. 2025 的"memory 必须随 $L^\beta$ 增长"理论。
- **Cache-efficient 版牺牲很小**：$f(k) = 2^k$ + cache-eff 在 copy 上从 92.63 掉到 75.47，但 cache 从 $\mathcal{O}(n)$ 降到 $\mathcal{O}(\log n)$，对长序列部署很有价值。
- **Congestion 是 hard bottleneck**：Diagonal SSM congestion = $n$ → copy 只有 42.98%；$f(k) = k^2+1$ congestion = 4 → copy 99.66%，congestion 量直接预测 copy 能力。
- **Shortest path bound 是 tight 的**：Lagrange 四平方和让 $f(k) = k^2+1$ 的最短路径 $\le 4$ 是优雅数学结果，实验复现了路径短 → 长程依赖好的关系。

## 亮点与洞察

- **$Y = (I-B)^{-1} A X$ 是 elegant 的统一框架**：把 attention/SSM/linear recurrence/higher-order recurrence/Mamba-2 都嵌进单个矩阵代数，且这个统一不是"事后归纳"而是 actionable 的设计工具。
- **Translation-invariant + $f$ 的设计空间**：选 $f$ = 选复杂度，这种 single-knob design 让 architect 可以系统地在复杂度阶梯上滑动而不是 case-by-case 调。
- **Lagrange 四平方和的妙用**：$f(k) = k^2+1$ 用数论古老结果保证最短路径 $\le 4$，这种"借力外部数学"的设计很罕见且优雅。
- **Congestion 作为 expressivity metric**：从图论 routing 视角量化"信息瓶颈"，跟 Jelassi et al. 2024 关于 recurrent model copy 困境的研究形成完美闭环。
- **Cache-efficient 的 closed-form lattice**：Proposition 4.12 揭示 cache 位置在 step-$a_k$ lattice 上 periodic，对 hardware-friendly 实现（如 FlashAttention 风格的 GPU kernel）有直接启发。
- **理论指导 → 实证验证**：Table 1 整列就是理论预测的实证 ladder，没有 cherry-picking，每个 trade-off 点都有清晰数学解释。

## 局限与展望

- **只看线性 token mixer**：非线性 mixer（如 mixture-of-experts、attention with non-linear feature map）不在框架内，需要扩展。
- **Synthetic task 为主**：copy、associative recall、multi-hop recall 是 controlled task，真实语言模型 pre-training 上的 perplexity 差距（虽然论文有 LM 实验）未详尽展示在主表里。
- **$f$ 选择有限**：实验只跑了 linear、$2^k$、$k^2+1$，更精细的 $f$（如 $f(k) = k \log k$）的复杂度-表达力 trade-off 没探索。
- **Cache-efficient 的 hardware 实现还没做**：lattice quantization 的 closed-form 是理论结果，实际 GPU kernel 优化没演示。
- **Higher-order recurrence 的训练稳定性**：infinite-order Dense recurrence 表达力满分但训练是否稳定（特别在 8B+ 模型上）未给数据。
- **缺失与 Mamba-2、Linear Attention 的吞吐对比**：实证只比 task accuracy，没比 wall-clock 速度，对工程师来说少了重要 reference。

## 相关工作与启发

- **vs S4/Mamba/Mamba-2**：他们是 first-order 线性递归，本文证明 first-order 在 long sequence 上必然 sub-optimal（congestion = $n$），higher-order 是必要补强。
- **vs Log-linear attention (Guo et al. 2026)**：他们提出 logarithmic-order recurrence 实例，本文给出系统框架并把它当 $f(k) = 2^k$ 的特例。
- **vs ChaCAL (Fagnou et al. 2024) / Block-Chacal**：他们提出 infinite-order 递归实例，本文统一框架包含但同时给出更细的复杂度-表达力 trade-off。
- **vs FlashAttention**：FlashAttention 优化 standard attention 的 kernel，本文是 algorithm-level 复杂度降低，两者正交可组合（cache-efficient pattern + FlashAttention-style kernel）。
- **vs Chimera (Lahoti et al. 2025)**：他们把 SSM 推广到 graph，本文用 communication graph 当 expressivity 分析工具，两者从不同角度引入图论。
- **启发**：(1) 任何"序列模型设计"问题都可以从 $(A, B)$ sparsity pattern 出发系统探索；(2) 复杂度-表达力的 Pareto frontier 可以被 single function $f$ 参数化；(3) congestion 是被低估的 expressivity 维度，未来 architecture search 应纳入；(4) closed-form lattice for cache 给 hardware-aware design 提供 building block。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 统一框架 $(I-B)^{-1} A$ 是新视角，translation-invariant pattern + Lagrange 四平方和应用是创新组合。
- 实验充分度: ⭐⭐⭐⭐ Synthetic 三任务完整覆盖 + LM 验证，但缺 wall-clock 速度对比和大模型 scale 验证。
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨（每个 proposition 都有 proof in appendix），Figure 2 的 sparsity visualization 直观，Table 1 一表道尽 trade-off。
- 价值: ⭐⭐⭐⭐⭐ 给序列模型架构设计提供 principled framework，对 Mamba-3、新一代 hybrid model 设计有直接指导价值；对 hardware-aware kernel 开发有 closed-form 工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)
- [\[AAAI 2026\] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts](../../AAAI2026/llm_nlp/promptmoe_generalizable_zero-shot_anomaly_detection_via_visually-guided_prompt_m.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](../../ACL2025/llm_nlp/token_prepending_training_free.md)
- [\[ICML 2025\] Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence](../../ICML2025/llm_nlp/interchangeable_token_embeddings_for_extendable_vocabulary_and_alpha-equivalence.md)

</div>

<!-- RELATED:END -->
