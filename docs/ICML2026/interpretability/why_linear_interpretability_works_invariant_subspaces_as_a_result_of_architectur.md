---
title: >-
  [论文解读] Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints
description: >-
  [ICML 2026][可解释性][linear probing] 本文给出"为什么 transformer 的内部表征可以被简单线性方法（probe、SAE、activation steering）反复成功解码"的架构级解释：只要语义特征是通过 OV 电路或 unembedding 这类**线性接口**被读出的，它就必须落在一个跨上下文不变的线性子空间里（Invariant Subspace Necessity 定理）；并推出一个零样本应用——Self-Reference Property，即 token 本身的嵌入方向就是其概念方向，从而可以无监督地用 class token 的几何位置直接做分类。
tags:
  - "ICML 2026"
  - "可解释性"
  - "linear probing"
  - "sparse autoencoder"
  - "invariant subspace"
  - "self-reference"
  - "unembedding 几何"
---

# Why Linear Interpretability Works: Invariant Subspaces as a Result of Architectural Constraints

**会议**: ICML 2026  
**arXiv**: [2602.09783](https://arxiv.org/abs/2602.09783)  
**代码**: 暂未公开 (无)  
**领域**: 可解释性 / 表征几何 / Transformer理论  
**关键词**: linear probing、sparse autoencoder、invariant subspace、self-reference、unembedding 几何

## 一句话总结
本文给出"为什么 transformer 的内部表征可以被简单线性方法（probe、SAE、activation steering）反复成功解码"的架构级解释：只要语义特征是通过 OV 电路或 unembedding 这类**线性接口**被读出的，它就必须落在一个跨上下文不变的线性子空间里（Invariant Subspace Necessity 定理）；并推出一个零样本应用——Self-Reference Property，即 token 本身的嵌入方向就是其概念方向，从而可以无监督地用 class token 的几何位置直接做分类。

## 研究背景与动机

**领域现状**：现代 mechanistic interpretability 反复发现 transformer 的内部状态可以被极简单的线性操作"解码"：linear probe 能从 hidden state 抽取语义属性（Alain & Bengio 2016, Belinkov 2022）；sparse autoencoder (SAE) 能找出可解释的 feature direction（Bricken 等 2023, Cunningham 等 2023）；单向量 activation steering 能稳定改变模型行为（Turner 等 2023, Zou 等 2023）。

**现有痛点**：Transformer 是参数海量、深层、强非线性的系统，按理它的中间表示完全没有义务"线性可读"。但事实上线性方法广泛奏效，这背后是经验巧合还是必然？现有解释要么诉诸"经验观察"，要么从优化动力学（Jiang 等 2024 的 next-token + 梯度下降隐式偏差）切入，但没有从**架构**本身回答"必须线性"的问题。

**核心矛盾**：优化解释告诉我们"为什么这样学到了"，但无法解释"为什么所有满足该架构的模型都被迫如此"——如果换个非线性输出头，linear probe 还成立吗？作者猜想答案是不成立，根源不在优化而在 transformer 用线性矩阵（OV、unembedding）做模块间通讯这件事本身。

**本文目标**：(1) 形式化"线性接口 ⇒ 跨上下文不变线性子空间"为定理；(2) 给出可被实验验证的可证伪推论（Self-Reference Property）；(3) 在多模型多任务上验证。

**切入角度**：作者把目光放在"transformer 各模块间是怎么互相传话的"——attention 的 OV 电路 $W_O W_V$、unembedding $W_U$ 都是作用在 residual stream 上的**线性映射**。任何要走过这些接口才能影响输出的语义特征，从形式上看必须满足"线性可读"——而这等价于落在某个线性子空间里。

**核心 idea**：用"架构必要性"代替"优化偶然性"来解释线性可解释方法的成功；并据此提出"token 嵌入方向 = 概念方向"的 self-reference，零样本拿来分类。

## 方法详解

### 整体框架
本文不训练新模型，而是把"线性可解释为什么总是奏效"这个经验现象转成一个可证明的架构命题。在 4 条架构假设（加性 residual stream、OV 与 unembedding 都是线性接口、参数共享、线性输出层）下，作者先证明核心定理（Theorem 3.7，任何走线性接口被读出的语义特征必然落在跨上下文不变的线性子空间里），再用一条容量约束命题（Proposition 3.8）说明这种表征在词表远大于维度时必然稀疏因式分解为共享方向——再由此推出一个零样本应用：token 自身的嵌入方向就是它所对应概念的几何方向。最后用 8 个分类任务 × 4 个模型家族的几何对齐实验，外加一个"把 unembedding 换成 MLP head"的准实验对照来佐证因果方向。

### 关键设计

**1. Invariant Subspace Necessity 定理：把"线性可读"等价成"几何不变子空间"**

整篇论文的痛点是，transformer 是深层强非线性系统，本没有义务让中间表示"线性可读"，但 probe / SAE / steering 却屡屡奏效，缺一个非经验的解释。作者先把"可通讯特征" $f: \mathcal{C} \to \mathcal{Y}$ 形式化为两个条件：multi-context 要求存在多个不同表面 $c_1, c_2$ 都表达同一 $f$ 值（"France"和"the country of the Eiffel Tower"都指法国），linear decodability 要求存在 $\phi \in \mathbb{R}^{|V|}$ 使 $\phi^\top W_U \mathbf{h}(c) = g(f(c))$ 对所有 $c$ 成立。在此之上，由于线性接口意味着存在标量读出 $o_f(c) = \mathbf{w}_f^\top \mathbf{h}(c)$，任何上下文要给出相同的 $f$ 值，就必须在 $\mathbf{w}_f$ 方向上保持一致、只能在正交补 $\mathbf{w}_f^\perp$ 里自由变化——也就是说 $f$ 的信息只活在由 $\mathbf{w}_f$ 决定的、与上下文无关的子空间 $\mathcal{S}_f$ 内。Directional Invariance 进一步把这个子空间收紧到 $\dim(\mathcal{S}_f)=1$，即单个 direction 就足够。这条等价之所以有效，是因为它一举把 linear probe、SAE、activation steering 这些看似不同的工具统一成"在利用同一个 $\mathcal{S}_f$"，从而解释了它们为何常常给出一致结论。

**2. Capacity Constraint 命题：词表远大于维度逼出稀疏因式分解**

光证明"存在不变方向"还不够，得解释为什么 SAE 找到的是一组可复用的稀疏字典。作者从 $|\mathcal{V}| \gg d$ 这个工程现实出发：unembedding $W_U \in \mathbb{R}^{|\mathcal{V}| \times d}$ 里每个 token 的列向量 $\mathbf{w}_t$ 不可能两两正交（token 数远多于维度），只能彼此共享方向。若各上下文只激活稀疏的 feature 集合、且多个 token 共享同一语义属性，最优表示就必然因式分解为 $\mathbf{w}_t = \sum_{f \in F_t} \alpha_{t,f} \mathbf{d}_f$，其中共享方向数 $|F| \ll |\mathcal{V}|$。代回后 logit 写成 $\text{logit}_t = \sum_{f \in F_t} \alpha_{t,f}\,(\mathbf{d}_f^\top \mathbf{h}(c))$，每个因子方向 $\mathbf{d}_f$ 又重新满足"线性可解码且上下文无关"，正好落回 Theorem 3.7 的前提。这条命题说明 SAE 能成功不是巧合——容量约束、稀疏激活、多 token 共享语义这三条现实联手逼模型把表示组织成可被稀疏字典还原的形式，也就解释了为什么 SAE 字典与 linear probe 找到的方向常常重合。

**3. Self-Reference Property：token 自己就是它的概念方向**

前两条定理告诉我们概念方向 $\mathbf{d}_f$ 完全由模型参数决定，但要把它"拿出来用"通常还得训练 probe 或跑无监督 SAE。作者指出最直接的参考向量其实就是概念对应的 token 本身——token "France" 的 embedding 方向就给出 France 概念的方向，于是"I went to Paris"和"I visited Marseille"的 hidden state 都会在这个方向上有强投影，可以零样本无监督地做分类（显式 token 自指地给出方向，上下文里的隐式实例与之共享同一不变方向）。它有效是因为把"概念方向"直接落到了 token 嵌入这个零参数的几何对象上，既给出一个不依赖标签的分类 baseline，也能用来 sanity-check 训练得到的 probe 方向是否真在描述同一概念。

### 验证实验设置
主结果是上述两定理一推论的纯数学论证，验证部分不训练新模型，而是在 LLaMA3-8B、Mistral-7B、GPT2-Small、LLaMA3.2-3B 这 4 个 backbone、8 个语义分类任务（taxonomic、affective、stylistic、linguistic、descriptive 等）上测量三件事：class token 方向与对应实例 hidden state 的余弦对齐、无监督 SAE 学到的 feature 方向与 class token 方向的对齐、以及"modular division + MLP head vs. linear head"的对照实验。

## 实验关键数据

### 主实验
（论文截至 cache 范围给出定性结论；具体数表附录中）

| 验证维度 | 现象 | 解释 |
|---|---|---|
| 8 个分类任务 × 4 个模型家族 | class token 方向与同类实例 hidden state 持续高对齐 | 验证 directional invariance 在多任务、多家族上稳健 |
| 无监督 SAE feature 方向 | 与 class token 方向显著对齐 | 验证"两条路径访问同一 $\mathcal{S}_f$" |
| Modular division + MLP head（图 2 对照） | 模型找非 Fourier 解时线性 probe ~20%；找到 Fourier 解时 probe 成功 | 验证"线性 readout"才是 directional structure 的成因，MLP head 下不再必要 |

### 消融实验

| 配置 | 现象 | 说明 |
|---|---|---|
| Linear unembedding（标准 transformer） | 线性 probe 必然成功（理论保证） | Theorem 3.7 起作用 |
| MLP classification head（对照） | 线性 probe 只在偶然找到 Fourier 表示时成功 | 证明非线性 readout 解除了"必须不变子空间"的硬约束 |
| Class token zero-shot probe | 在多个任务上达到与训练 probe 相当的分类性能 | Self-Reference 的直接落地 |

### 关键发现
- **"线性接口"是关键变量而非"线性表示"**：图 2 的 modular division 对照实验最为关键——同一任务下，把 readout 换成 MLP，线性 probe 就不再普遍成功；换回线性 unembedding，directional structure 自动出现。这是这篇论文最强的实证证据，把"架构 → 表示形式"的因果方向钉死。
- **SAE 与 probe 找到的是同一组方向**：无监督 SAE feature direction 与 class token direction 对齐，意味着这两类工具不是在做不同的事，而是在用不同方法访问同一个不变子空间 $\mathcal{S}_f$，这统一了过去看似分歧的解释主义流派。
- **零样本几何 probe 可行**：不训练任何参数、只用 token 嵌入方向就能做分类，对依赖标签的 probe 是个强有力的几何 baseline，也意味着可以在没有标签的新任务上快速给出可解释方向。

## 亮点与洞察
- **架构 vs 优化二分法**：作者明确把自己的解释定位为对 Jiang 等 (2024) 优化解释的"互补"——优化决定"怎么学到"，架构决定"必须取什么形式"，这种解释分层非常清晰。
- **Theorem 3.7 的证明极短**：只用了"线性算子内核"几行就把全部主张钉牢，是 mechanistic interpretability 中少见的"少即多"的理论性贡献。
- **Self-Reference 的应用价值**：把高大上的几何定理落到"零样本无监督分类"的具体能力上，使理论部分有可验证、可工程化的落点，避免了纯数学论文常见的"看似深刻但用不上"的批评。
- **Modular division 对照实验**：这是把"必要性"主张转化为可证伪实验的关键设计——通过更换 head 让效应消失，再换回让效应出现，给"架构是因"提供了准实验级别的证据。

## 局限与展望
- 假设 1 要求"加性 residual stream"，对部分含 RMSNorm/post-norm 复杂模块的现代变体（如 Llama-3 实际使用的归一化）需要更细的论证：normalization 本身不是线性，会影响"线性接口"的精确边界。
- 假设 2 把 OV 当线性接口，但实际 attention 还有 softmax，softmax 输出虽然是凸组合但前置 query-key dot product 是非线性影响——论文未充分讨论 softmax 路径上的特征是否同样落在不变子空间。
- 实验任务限于分类（8 个），对 reasoning、in-context learning、long-context 等"非分类语义"是否还成立未验证；directional invariance 在涉及 task-specific 上下文调制时可能弱化。
- Self-Reference 假设"概念有对应 token"，对没有单 token 表达的复合概念（"我去过的国家"、"昨天写的代码"）就无法直接应用，需要扩展到 phrase embedding 或多 token pooling。
- 实验只覆盖 4 个相对小的开源模型（最大 8B），更大尺度上若 $|V|$ 持续放大、$d$ 比例缩小，因式分解的几何形态可能更复杂；超大规模上是否仍然 single-direction 还需要验证。

## 相关工作与启发
- **vs Jiang 等 (2024)** 用 next-token + 梯度下降的隐式偏差解释线性表示；本文从架构必要性给出互补解释，二者共同确认"linear representation 不是偶然"。
- **vs Park 等 (2024)** 形式化 "linear representation" 的概念几何；本文进一步指出该几何必然形成的条件。
- **vs Kantamneni 等 (2025)** 经验发现 SAE latent 在 probing 任务上未必超过 linear probe；本文给出理论解释——两者本就访问同一组不变方向。
- **vs nostalgebraist (2020) Logit Lens / Belrose 等 (2023) Tuned Lens**：这些实践方法的成立条件就是 $W_U$ 是线性、不变子空间存在，本文为它们提供了背书。
- **启发**：(1) 设计可解释友好的新架构应保留"线性最后一公里"，否则 probe/SAE 类工具会失效；(2) 在 multimodal 模型里如果想保留 linear interpretability，跨模态融合层也应尽量保持线性接口；(3) Self-Reference 的零样本分类思路可以推广为"用模型自身 token 嵌入做 contrastive probe"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"为什么 linear interpretability 有效"从经验观察上升为架构必要性定理，是 mechanistic interpretability 少见的硬理论贡献。
- 实验充分度: ⭐⭐⭐⭐ 4 个模型 × 8 个任务的对齐验证 + 一个 modular division 准实验对照足够支持主要主张；但任务全是分类、模型最大 8B，覆盖面略窄。
- 写作质量: ⭐⭐⭐⭐⭐ 假设-定义-定理-推论的论证链条干净利落，对照实验设计精巧，是这类理论性 interpretability 论文的写作范本。
- 价值: ⭐⭐⭐⭐ 给一个高速发展的领域（mechanistic interpretability）提供了亟需的统一框架，且直接孵化出零样本 probe 等可用工具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](../../ICLR2026/interpretability/decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[NeurIPS 2025\] The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?](../../NeurIPS2025/interpretability/the_non-linear_representation_dilemma_is_causal_abstraction_enough_for_mechanist.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ICML 2026\] Interpretability Can Be Actionable](interpretability_can_be_actionable.md)

</div>

<!-- RELATED:END -->
