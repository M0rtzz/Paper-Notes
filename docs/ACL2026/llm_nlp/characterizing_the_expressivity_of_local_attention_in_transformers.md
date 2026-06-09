---
title: >-
  [论文解读] Characterizing the Expressivity of Local Attention in Transformers
description: >-
  [ACL 2026][LLM/NLP][局部注意力] 作者用线性时序逻辑（LTL）作为统一刻画工具，严格证明 global-only Transformer ↔ $\mathrm{LTL}[\mathrm{P}]$、$k$-local-only ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "局部注意力"
  - "表达力"
  - "线性时序逻辑"
  - "正则语言"
  - "Transformer"
---

# Characterizing the Expressivity of Local Attention in Transformers

**会议**: ACL 2026  
**arXiv**: [2605.00768](https://arxiv.org/abs/2605.00768)  
**代码**: 无（理论 + 复现脚本基于 Delétang 等 2023 / Li & Cotterell 2025）  
**领域**: LLM 理论 / Attention  
**关键词**: 局部注意力、表达力、线性时序逻辑、正则语言、Transformer

## 一句话总结
作者用线性时序逻辑（LTL）作为统一刻画工具，严格证明 global-only Transformer ↔ $\mathrm{LTL}[\mathrm{P}]$、$k$-local-only ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$、global+local 混合 ↔ $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$，并由此证明 **local 与 global 表达力互不包含**、混合严格更强、**1-local 是 local 家族里表达力最强**，最后在合成正则语言和 WikiText-2 上经验验证理论预测。

## 研究背景与动机

**领域现状**：全局注意力是 Transformer 的灵魂——每个 token 都可看到所有前驱，序列长度 $N$ 时计算量 $O(N^2)$。Longformer / BigBird / Sparse Transformer 等模型用 local attention（每 token 只看前 $k$ 个邻居）把复杂度降到 $O(Nk)$。

**现有痛点**：local attention 通常被当作"为了省算力而牺牲表达力"的妥协，但多项实证研究反复观察到：在算力受控甚至持平条件下，加入 local attention 反而提升了机器翻译、语言建模等任务质量。这个"反直觉"现象至今没有一个严谨的解释——是凑巧的优化效应？还是 inductive bias？还是表达力真的变了？

**核心矛盾**：直觉上"看得更远"应该更强大，但 local 限制视野却经验上更好。这暗示 global 和 local 注意力捕捉的语言性质从根本上是不同维度的，而不是"包含"或"被包含"。但要证明这一点需要一种语言无关、可定量比较的形式化语义——这正是形式语言理论 + 时序逻辑提供的工具。

**本文目标**：(1) 给 local-only 和 hybrid Transformer 都找到精确对应的 LTL 片段；(2) 证明这些片段之间的包含/不包含关系，从而判定不同 attention 模式的表达力序；(3) 找出最优 window size；(4) 经验验证理论预测，并把结论延伸到 WikiText-2 自然语言场景。

**切入角度**：Li & Cotterell (2025) 已证明 fixed-precision global-attention Transformer 与 $\mathrm{LTL}[\mathrm{P}]$（含单个"past"操作符）双向等价；本文以此为支点，自然提问"加入 local mask 等价于加入什么时序操作符？"

**核心 idea**：local attention 看的是 bounded suffix → 这正对应"yesterday/$k$ 步前"算子 $\mathrm{Y}^{\leq k}$；混合注意力对应同时拥有 $\mathrm{P}$（无界 past）和 $\mathrm{Y}^{\leq k}$（bounded past）。两个算子表达力互不包含 → 自然推出 global / local 表达力互补。

## 方法详解

### 整体框架
全文是一条完整的"逻辑 → 架构 → 实验"证明链，核心思路是把 attention mask 翻译成线性时序逻辑的算子，从而让"看得多远"这种架构差异变成可比较的逻辑表达力。具体地：在逻辑侧分别刻画 $\mathrm{LTL}[\mathrm{Y}]$（=有穷后缀决定的 definite languages）、$\mathrm{LTL}[\mathrm{P}]$（=left-deterministic polynomials，Li & Cotterell 2025 已证）与 $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}]$（= locally $\mathcal{R}$-trivial monoids，可识别 locally testable languages），并证明三者两两的严格不包含关系；在 Transformer 侧把 mask 形式化为 global mask（$\mathbf{M}^*_{n,m}=1$ iff $m<n$）与 $k$-local mask（$\mathbf{M}^{\leq k}_{n,m}=1$ iff $\max(1,n-k)\leq m<n$），并用 fixed precision 限制保证可与有穷自动机对接；最后沿用 Li & Cotterell 的证明骨架建立 $k$-local Transformer ↔ $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$、hybrid Transformer ↔ $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$ 两条双向等价，再在 8 个合成正则语言和 WikiText-2 上验证理论预测。

### 关键设计

**1. 用 $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$ 与 $\mathrm{LTL}[\mathrm{P}]$ 的不可比性证明 local/global 表达力互补**

直觉总把 local 当成 global 的"省算力弱化版"，但要推翻它，需要在逻辑层面构造两个互相够不着的判定性 witness 语言。一边是 $a\Sigma^*$（"第一个字符是 a"），它可被无界 past 算子写成 $\mathrm{P}(\pi_a \land \neg \mathrm{P}\top)$，却落在 $\mathrm{Y}^{\leq k}$ 之外——因为 $\mathrm{Y}^{\leq k}$ 只看 bounded suffix，而定理 2.2 证明 $\mathrm{LTL}[\mathrm{Y}]$ 恰对应 definite languages，"开头依赖"性质并不 definite。另一边是 $L_k = \bigcup_{i=0}^{k-1} \Sigma^* a \Sigma^i$（"末尾 $k$ 步内出现 a"），它能被 $\mathrm{Y}^{\leq k} \pi_a$ 一句写出，却不是 left-deterministic polynomial，因而不在 $\mathrm{LTL}[\mathrm{P}]$ 内（命题 2.3、2.4、定理 2.9）。两个 witness 互证 $\mathrm{LTL}[\mathrm{P}]$ 与 $\mathrm{LTL}[\mathrm{Y}^{\leq k}]$ incomparable，于是 hybrid 同时拥有两个算子时严格强于任一侧（推论 2.10）——"global+local 比 global-only 严格强"这个经验观察由此落实为可证伪的形式定理。

**2. $k=1$ 是 local 家族里表达力最强的 window size**

工业实践里 sliding window 常默认开大，但本文证明扩窗反而损失表达力。对任意 $k>1$ 构造 witness $\mathbf{w}=(\mathtt{ab}^{k-1})^r \mathtt{a}$ 与 $\mathbf{w}'=(\mathtt{ab}^{k-1})^r$，前者属于 $\Sigma^*\mathtt{a}$ 而后者不属于；引理 C.1 对操作符深度 $s$ 做"$s$-close pair"的对偶归纳，证明任何深度 $\leq r$ 的 $\mathrm{LTL}[\mathrm{P}, \mathrm{Y}^{\leq k}]$ 公式都无法在 $\mathbf{w}$ 与 $\mathbf{w}'$ 的末位之间区分——关键在于 witness 的周期长度恰为 $k$，使得 $i$ 步前与 $i-k$ 步前的 token 完全相同，$\mathrm{Y}^{\leq k}$ 因此"看花了眼"。这给出 $\mathrm{LTL}[\mathrm{Y}^{\leq k}] \subsetneq \mathrm{LTL}[\mathrm{Y}]$（命题 2.11、2.12），即 1-local 等价于完整的 $\mathrm{Y}$，是 local 家族里最强的。代价是表达—深度权衡：用 1-local 实现 $\mathrm{Y}^{\leq k} \psi$ 需要 $k$ 个 $\mathrm{Y}$ 嵌套，操作符深度增加 $k-1$，对应 Transformer 层数开销最多 $\times k$，这也解释了为什么深模型才能充分享受 1-local 的好处。

**3. 位置编码 = 数值谓词，抹不平 local/global 的鸿沟**

一个常见反驳是"既然都有位置编码，global+RoPE 是不是就能模拟 local？"。本文把位置编码视为 LTL 的数值谓词 $\mathrm{MOD}_m^r$（命题 2.13），并证明只要存在 $m \geq k$ 的模运算谓词，$\mathrm{Y}^{\leq k}$ 就能被 $\mathrm{Y}$ 加 $\mathrm{MOD}$ 模拟：$\mathrm{Y}\psi = \bigvee_{i=1}^m (\mathrm{MOD}_m^i \land \mathrm{Y}^{\leq k}(\mathrm{MOD}_m^{i-1} \land \psi))$。但 SiPE 与 RoPE 对应的是模谓词的"rational 变种"，Chiang 等 2023 已证它们无法稳定提供精确的 $\mathrm{MOD}$ 语义，因此理论预言 SiPE / RoPE 都无法让 global-only 追平 hybrid——"位置编码万能"是迷思，也是"不该用 RoPE 顶替 local attention"的理论支撑。


## 实验关键数据

### 主实验：合成正则语言（最长完美泛化长度，NoPE，训长 ≤40，测长 41–500）

| 语言（所属 LTL 类） | local-1 | local-2 | local-4 | hybrid-1 | hybrid-2 | hybrid-4 | global |
|---------------------|---------|---------|---------|----------|----------|----------|--------|
| $\Sigma^*\mathtt{a}$（$\mathrm{LTL}[\mathrm{Y}]$） | **100.0** | 99.7 | **100.0** | **100.0** | 99.7 | 92.7 | 58.4 |
| $\Sigma^*\mathtt{ab}$（$\mathrm{LTL}[\mathrm{Y}]$） | **100.0** | 99.8 | **100.0** | **100.0** | 99.8 | 80.7 | 53.5 |
| $\mathtt{a}\Sigma^*$（$\mathrm{LTL}[\mathrm{P}]$） | 50.1 | 49.9 | 50.0 | **100.0** | **100.0** | 99.2 | 99.4 |
| $\Sigma^*\mathtt{a}\Sigma^*\mathtt{b}\Sigma^*$（$\mathrm{LTL}[\mathrm{P}]$） | 71.6 | 74.3 | 74.9 | **100.0** | 99.9 | **100.0** | 99.0 |
| $(\mathtt{ab})^*$（$\mathrm{LTL}[\mathrm{P,Y}]$） | 52.0 | 54.1 | 57.3 | **99.8** | 95.1 | 94.6 | 75.2 |
| $\Sigma^*\mathtt{ab}\Sigma^*$（$\mathrm{LTL}[\mathrm{P,Y}]$） | 71.2 | 85.1 | 96.2 | **99.7** | **100.0** | 71.9 | 56.9 |
| Right-det. poly.（$\mathrm{LTL}[\mathrm{S}]$ 之外） | 78.4 | 91.8 | 99.1 | 98.6 | 99.9 | 90.4 | 76.1 |
| Bounded Dyck-2（$\mathrm{LTL}[\mathrm{S}]$ 之外） | 75.2 | 94.7 | 98.4 | 89.7 | 87.4 | 83.5 | 66.7 |

数值为 5 seeds × 3 lr 的平均准确率（%），加粗为该行最优。

### 消融实验：WikiText-2 LM perplexity（GPT-2 small，12 层 / 768 d / 12 heads）

| 位置编码 | global-only | local-only $k=1$ | hybrid $k=1$ | 提升 (vs global) |
|---------|-------------|------------------|--------------|------------------|
| Learned absolute | baseline | 优于 global | **best**, $-69.7$ ppl | $-69.7$ |
| RoPE | baseline | 较 large-$k$ 更弱 | **best**, $-15.2$ ppl | $-15.2$ |
| SiPE | baseline | 较 large-$k$ 更弱 | **best**, $-11.5$ ppl | $-11.5$ |
| hybrid 随 $k$ 增大 | – | – | 退化至接近/差于 global | 验证 $k=1$ 最优 |
| local-only 在 RoPE/SiPE 下 | – | 偏好 大 $k$ | – | 与"位置编码模拟模谓词"理论吻合 |

### 关键发现
- 三组主要预测全部被合成实验证实：local-only 模型只在 $\mathrm{LTL}[\mathrm{Y}]$ 类语言上达到 100%（前两行），global-only 只在 $\mathrm{LTL}[\mathrm{P}]$ 类语言上达到 100%（第三、四行），hybrid（尤其 $k=1$）在前 6 个属于 $\mathrm{LTL}[\mathrm{P,Y}]$ 的语言上 100% 通过。
- $k=1$ 在 hybrid 内部一致最强：hybrid-2 在 $\Sigma^*\mathtt{ab}\Sigma^*$ 上还能 100%，但 hybrid-4 掉到 71.9%，hybrid-1 始终稳定在 ≥99.7%。说明大窗反而牺牲了 1-local 才有的"立即前驱"刻画能力。
- 位置编码不能让 global-only 追上 hybrid：SiPE 经常拉低性能，RoPE 在某些任务上能让 local-2/4 学到 $\Sigma^*\mathtt{a}$（验证模谓词模拟理论），但伴随其他任务退化，总体仍是 hybrid 占优。
- WikiText-2 上 hybrid-1 在所有 3 种位置编码下都拿到最低 ppl，最大幅度降 69.7（learned absolute）。这是首次在自然语言场景给"1-local + global"组合提供端到端证据。
- $\mathrm{LTL}[\mathrm{S}]$ 之外语言（bounded Dyck depth 2、right-det poly）所有模型都没有完美泛化，但 hybrid 仍给出最强的部分泛化——说明本文刻画范围之外，hybrid 的优势依然存在。

## 亮点与洞察
- **把"模型架构 vs 表达力"研究从语义直觉推进到形式化双向定理**：以前讨论 local vs global 经常停留在"sparsity 是 inductive bias"的直觉层面，本文给出严格 iff 等价。这种"架构 ↔ 逻辑片段"的对应是 mechanistic interpretability 的高级形态。
- **"1-local 最强"是直接可复用的设计原则**：很多工业 Transformer（如 GPT-NeoX、Mistral 的 SWA）默认用大 sliding window，本文从理论和 WikiText-2 经验上都建议把全局头和 1-local 头并行，而不是单纯减小 window 数。
- **解释了"位置编码不是万能"**：很多研究者以为 RoPE/SiPE 加入后 global-only 就能模拟 local 行为，本文从模谓词角度精确指出"哪种位置编码能模拟、模拟得多准"。这对未来设计新位置编码有指南意义。
- **"depth overhead"分析提供了表达力—深度权衡的量化公式**：用 $\mathrm{Y}^{\leq k}$ 替代 $\mathrm{Y}$ 最多需要 $k$ 倍层数，反过来说明深模型才能享受 1-local 的全部好处——这给"为什么大模型尤其受益于 hybrid attention"做了理论解释。
- **可迁移到其他 sparse pattern 研究**：作者的"mask → 时序算子"映射框架可以用来分析 BigBird、Longformer 的 random/global token、各种 dilated mask 等任意稀疏注意力的表达力，是一个通用工具。

## 局限与展望
- 全部理论建立在 fixed-precision + soft-max 假设上，与现代 bf16/fp16 训练有 gap；exact arithmetic 下结论可能更宽松。
- WikiText-2 实验受算力限制，只有 GPT-2 small 规模、单一数据集——作者明确承认大模型/大数据场景下定性结论可能变化。
- 只分析了"global vs local"二分；现代架构中常见的 sliding-window-with-attention-sink（如 Mistral）、global tokens（BigBird）、dilated patterns 尚未在本框架内完整刻画。
- "1-local 表达力最强但需更深"的 trade-off 没有给出闭式 depth–window pareto 边界；实践中如何选 $k$ 仍需经验。
- 位置编码分析只覆盖 SiPE/RoPE 两类，ALiBi、T5 相对偏置、NoPE 等未被严格映射到数值谓词体系。
- 改进思路：(1) 把分析推广到 cross-attention 与 encoder-decoder；(2) 将"hybrid + 1-local"在 LLaMA-3 级别上做扎实消融；(3) 形式化 attention sink / register token 的表达力。

## 相关工作与启发
- **vs Li & Cotterell (2025)**：他们建立了 global-only ↔ $\mathrm{LTL}[\mathrm{P}]$ 的基线对应，本文是直接 follow-up，新增 local/hybrid 两条对应链。证明骨架复用其 Lemma B.12 的 refinement 步骤。
- **vs Yang et al. (2024, 2026) "masked hard-attention transformer = star-free languages"**：他们处理的是 hard-attention 的另一种粒度，本文 soft-attention + fixed precision 更贴近实际部署；两者刻画的逻辑片段不同（star-free vs $\mathrm{LTL}[\mathrm{P,Y}^{\leq k}]$）。
- **vs Delétang et al. (2023) Chomsky hierarchy benchmark**：他们提供了实验范式（训短测长 + 多 seeds），本文复用了他们的 codebase，但把"经验现象"提升为"形式定理 + 验证"。
- **vs Beltagy et al. 2020 (Longformer) / Zaheer et al. 2020 (BigBird)**：经验上观察到 hybrid 优势，本文给出第一份严谨理论解释。本文进一步建议把 Longformer 默认的大 $k$ 改为 1-local + global 组合。
- **vs Chiang et al. (2023)**：他们研究 RoPE rational 变种的精确语义边界，本文借用其结论说明 SiPE/RoPE 不能稳定模拟 $\mathrm{MOD}$ 谓词。
- 启发：(1) "把架构选择映射到逻辑片段"可作为研究范式推广到 SSM、Mamba、RWKV 等新架构；(2) 任何 sparse-attention 论文都应顺带做这种"witness language" 测试集，给经验改进配上表达力验证；(3) 位置编码的设计应明确"目标是模拟哪个谓词"——这是新的目标导向。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个在 LTL 双向等价层面解释 hybrid attention 优势的工作，"1-local 最强"是显著的反直觉新结论。
- 实验充分度: ⭐⭐⭐⭐ 合成语言系统验证 8 类 LTL 片段成员，WikiText-2 覆盖 3 种位置编码 × 多个 $k$，但缺大模型实验。
- 写作质量: ⭐⭐⭐⭐⭐ 定义、命题、引理、定理结构严密，附录 60+ 页全证明 self-contained，从理论到代码全链路 traceable。
- 价值: ⭐⭐⭐⭐⭐ 直接影响 sparse attention 设计与位置编码选型，给 Transformer 表达力理论提供新基线，对工业 LLM 架构有现实指南。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](../../NeurIPS2025/llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](../../NeurIPS2025/llm_nlp/strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[AAAI 2026\] Learning Spatial Decay for Vision Transformers](../../AAAI2026/llm_nlp/learning_spatial_decay_for_vision_transformers.md)
- [\[CVPR 2025\] Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](../../CVPR2025/llm_nlp/rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)

</div>

<!-- RELATED:END -->
