---
title: >-
  [论文解读] How Many Different Outputs Can a Transformer Generate?
description: >-
  [ICML 2026][LLM/NLP][可达序列] 本文从"有限精度 + 有界嵌入支撑"两个最基本的架构事实出发，证明任意 transformer 只能生成有限条"可达序列"，给出可达序列长度随 prompt 长度线性增长、超过阈值后比例以 $1/|V|^n$ 指数衰减的紧上界…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "可达序列"
  - "packing number"
  - "嵌入空间几何"
  - "copying / cramming"
  - "有限精度"
---

# How Many Different Outputs Can a Transformer Generate?

**会议**: ICML 2026  
**arXiv**: [2605.22223](https://arxiv.org/abs/2605.22223)  
**代码**: https://github.com/mario-michelessa/transformers_accessibility (有)  
**领域**: LLM / NLP 理论分析  
**关键词**: 可达序列, packing number, 嵌入空间几何, copying / cramming, 有限精度  

## 一句话总结
本文从"有限精度 + 有界嵌入支撑"两个最基本的架构事实出发，证明任意 transformer 只能生成有限条"可达序列"，给出可达序列长度随 prompt 长度线性增长、超过阈值后比例以 $1/|V|^n$ 指数衰减的紧上界，并用 cramming 与 copying 实验在 Pythia/Qwen/Llama/Gemma 上验证理论斜率与实测仅差 5–10 倍。

## 研究背景与动机
**领域现状**：Transformer 在 NLP 与 CV 都是统治性架构，已有大量"approximation theory"工作证明它是通用近似器（Yun 2020a/b、Edelman 2022、Kratsios 2022），甚至能模拟图灵机、任意程序（Wei 2022、Giannou 2023），加 CoT 后表达力进一步增强。**这些理论似乎暗示 transformer 几乎"什么都能算"**。

**现有痛点**：然而实践中却反复观察到一类反直觉的失败：长序列 **复制**（Jelassi 2024）、**重复**（Barbero 2024）、**cramming**（Kuratov 2025）这些"小学生级别"的任务，一旦输入长度超过某个临界值，再大的模型再多的训练数据都救不回来，而且失败不是渐进式而是**断崖式**——短时几乎 100%，过临界点直接掉到 0。

**核心矛盾**：通用近似性是"连续空间 + 无限精度"下的渐进结果，但真实 transformer 跑在 **有限浮点精度** 上，且其内部表示被实测约束在 **有界、且高度各向异性** 的子空间内（Brody 2023、Rudman 2022）。这两个有限性意味着，可被区分的输入与可输出的序列必然是**有限集合**——和 |V|^n 的指数增长之间存在结构性鸿沟。

**本文目标**：把这个鸿沟形式化，回答三个具体问题：(i) 给定 prompt 长度 m，最多能输出多长的不同序列？(ii) 是否存在 prompt-independent 的硬上界？(iii) 这个上界能否仅由架构参数（嵌入维度 d、半径 r、精度 ε、词表 |V|）写成闭式，从而**预测**新模型在 copying / cramming 上的失败长度？

**切入角度**：把 last-layer embedding 空间按"最可能的下一 token"切成 |V| 个 argmax cell（Fig 1），那么生成一条长度 n 的序列 ⟺ 找到一段 prompt embedding，让贪心解码连续 n 步都落入正确 cell。**可生成 ⟺ 落入足够小的可行区域** —— 而有限精度的 transformer 可区分的输入数量受 **packing number** 控制。

**核心 idea**：把"可达序列数"≤"嵌入支撑的 packing number"做成定理，从而把 transformer 的生成能力上界换算成纯几何量，再用 mean-field + Wasserstein 把 prompt 长度无穷的情况也纳入同一框架。

## 方法详解

### 整体框架
这篇论文想用一条尽可能短的逻辑链回答"transformer 到底能输出多少条不同序列"：把 transformer 看成有限精度下的映射，它能区分的输入本来就是有限多个，于是它能输出的序列也必然有限。整套分析从"架构事实"出发，先把生成行为翻译成嵌入空间里的几何问题，再用 packing number 卡住可达序列的总数，导出可达长度阈值的闭式公式，最后回到 cramming / copying 两个真实任务上验证预测。

具体地，把 transformer 形式化为映射 $\tau:\bigcup_m \mathbb{R}^{d\times m}\to\Delta_{|V|}$（soft prompt 版本，hard prompt 是其特例），假设 last-layer embedding 支撑落在球 $E\subset B_d(0,r)$ 内，并按 unembedding 矩阵 $F$ 把 $E$ 切成 $|V|$ 个 argmax cell $E_i=\{x:(Fx)_i\geq (Fx)_j\,\forall j\}$（Fig 1 在 Qwen-2-0.5B 上做了 2D 切片）。剩下的工作就是数清楚"有多少条序列能在这套 cell 划分里被贪心解码连续命中"。

### 关键设计

**1. 可达序列的几何化定义：把"能否输出序列 t"变成"嵌入空间能否塞下一个精度球"**

以往的"通用近似"结果都是连续、无限精度意义下的，没法解释离散输出为什么会失败。本文的第一步就是换一套语言：先按下一个 token 的 argmax 把 last-layer embedding 划成 $|V|$ 个 cell（Section 3.1 定义，Fig 1 可视化），那么生成一条长度 $n$ 的目标序列 $t$，等价于贪心解码连续 $n$ 步都落进正确 cell，把这 $n$ 步条件叠起来得到 $E_t^m\subset B_{d\times m}(0,r)$。再引入有限精度：Assumption 4.3 要求 transformer 在每个边长 $\varepsilon$ 的立方体内取常数值，于是"序列 $t$ 可达" ⟺ $E_t^m$ 里能放下一个半径 $\varepsilon/2$ 的球（Definition 4.1）。这样一来，"transformer 能不能输出 $t$"这个离散问题就被翻译成"嵌入空间里有没有一段 ε/2 邻域整段落入 $E_t^m$"的纯几何问题——可达序列的总数自然被嵌入支撑的 packing number 卡住。顺带 Remark 3.2 把结论推广到随机解码：只要贪心不可达，任何随机采样策略的成功率都 $<50\%$。

**2. Packing-number 双轨上界：分别卡住"短 prompt"和"任意长 prompt"两种场景**

有了几何定义，可达序列数就等于嵌入支撑里能塞进的不相交精度球数，也就是 packing number，于是上界可以直接写成闭式。短 prompt 这条路（Thm 4.5 + Cor 4.6）最直接：$\tau$ 至多能区分 $P(B_{d\times m}(0,r),\|\cdot\|,\varepsilon)\leq (1+2r/\varepsilon)^{dm}$ 个输入，故可达序列数 $\leq (1+2r/\varepsilon)^{dm}$。把它和总序列数 $|V|^n$ 一比，立刻得到：一旦 $n>C\cdot m$，其中 $C=d\ln(1+2r/\varepsilon)/\ln|V|$，就必然存在不可达序列，且不可达比例以 $(1+2r/\varepsilon)^{dm}/|V|^n=O(1/|V|^n)$ 指数衰减——这正解释了"短 prompt 下可达长度阈值随 $m$ 线性增长"。但线性阈值会让人怀疑"prompt 拉得足够长是不是就能复制任意序列"，所以第二条路（Thm 4.9 + Cor 4.10）把 prompt 长度推到任意：利用 attention 的置换等变与 $L([X,X])_{:,i}=L(X)_{:,i}$ 性质（mean-field，Sander 2022），把每段 prompt $X$ 换成经验测度 $M(X)=\frac1m\sum_i\delta_{X_{:,i}}$，transformer 就成了概率测度之间的映射；配上 Wasserstein 版精度假设（Assumption 4.8），得到一个 prompt-independent 的上界 $(e+e(2r)^q/\varepsilon^q)^{(1+2r/\varepsilon)^d}$。此时虽然不再有"硬上限长度"，但可达比例依旧 $O(1/|V|^n)$ 衰减——所以再长的 prompt 也救不了 copying。两条上界合起来正好刻画了 cramming / copying 里观察到的 **sigmoid 形状**：临界长度前几乎全部可达，临界长度后比例暴跌，而且全部结论都是 prompt-、training-、compute-agnostic 的纯架构性质。

**3. 支撑域 refinement + 非均匀 cell 体积修正：把最坏情况上界收紧成可预测模型**

原始的 Cor 4.6 用了两个最坏情况假设——嵌入支撑是满球 $B_d(0,r)$、每个 cell $E_t$ 体积相同——所以预测斜率系统性偏大（满球版理论/实测比高达 14–20×）。本文用两步实测修正把它压到 5–10×。第一步是**支撑近似**（Section 5.2）：实测嵌入是高度各向异性的小子集（Rudman 2022），于是改用 axis-aligned ellipsoid（每维半径 $r_i$）和 cone（最小开口角）两种凸包络替换满球，重新算 packing number 再代回斜率公式（推导见 Appendix F）；只需 10K 随机 prompt、长度 $\ell\approx 1000$ 就能稳定估出形状参数（Fig 9）。第二步是**非均匀 cell 体积**（Section 5.3）：常见 token 占大 cell、罕见 token 占小 cell，所以先用 unembedding 矩阵 + Monte-Carlo 实测下一 token 的体积分布 $D=\{|E_t|/|E|\}$，再用 $n$ 次乘积卷积 $D^{\otimes n}$ 模拟长度 $n$ 序列的体积分布，找最小的 $n$ 使 $D^{\otimes n}$ 的中位数掉到 $1/P(E,\|\cdot\|,\varepsilon)$ 以下，即"超过一半序列不可达"的阈值（Fig 3）；当 $D$ 退化成 Dirac at $1/|V|$ 时这一步恰好回到 Cor 4.6。两步都用真实嵌入做"形状 + 密度"双修正，理论值收紧后才能直接当预测模型用——Table 1 显示 Ellipsoid + 非均匀 cell 把全部 7 个模型的比值都压到 5–11×。

### 损失函数 / 训练策略
论文是纯理论 + 黑盒探测，不训练新模型。Cramming 实验仅对一段长度 $m$ 的 soft prompt $Y\in\mathbb{R}^{d\times m}$ 用 teacher-forcing 优化 $\mathcal{L}(Y;x_{1:n})=-\sum_{i=1}^n\log p_\tau(x_i\mid[Y,x_{1:i-1}])$，模型权重全程冻结；copying 实验则在长度 ≤50 的合成串 $x_{1:n}|x_{1:n}$ 上 fine-tune 至 100% 训练精度或 10K 步，再到更长串上测 exact-match。

## 实验关键数据

### 主实验
模型横跨 Pythia (160M–2.8B)、Qwen-2.5 (0.5B / 1.5B)、Llama-3.2 (1B / 3B)、Gemma-3 (270M / 1B)。Cramming 任务每个 (n,m) 用 20 条 target 估均值，target 来自 PG19 自然文本与均匀采样的随机 token 串。

| 实验 / 模型 | 关键观察 | 数值 |
|------------|---------|------|
| Cramming (Qwen-2.5-1.5B), 固定 m | 不同 m 下"准确率 vs 长度 n"sigmoid 拟合 | 最小 $R^2=0.88$ |
| n50(m) 线性度 (PG19) | n50 ≈ Cm | $R^2_{\text{PG19}}=0.999$ |
| n50(m) 线性度 (random) | 同上，斜率更小 | $R^2_{\text{rand}}=0.995$ |
| Copying 任务，7 个模型 | 训练长度 ≤50，测更长串，断崖式下降 | sigmoid 拟合中位 $R^2=0.95$ |
| 理论斜率（满球） / 实测斜率 | 7 个模型平均 ≈ 12× | 见下消融 |

### 消融实验（Table 1：理论上界 / 实测斜率比）

| 几何假设 | Pythia-160M | Pythia-410M | Pythia-1B | Qwen-0.5B | Qwen-1.5B | Llama-1B | Gemma-270M |
|---------|------------|------------|-----------|-----------|-----------|----------|-----------|
| Ball（满球，原始 Cor 4.6） | 9.24 | 9.79 | 7.77 | 14.1 | 20.4 | 14.3 | 11.52 |
| Cone（最小开口锥） | 9.10 | 9.60 | 7.70 | 14.01 | 20.34 | 13.98 | 11.24 |
| Ellipsoid（各向异性椭球） | 7.92 | 8.15 | 6.12 | 10.96 | 15.30 | 11.86 | 11.12 |
| **Ellipsoid + 非均匀 cell** | **6.66** | **5.99** | **4.56** | **7.92** | **10.82** | **10.71** | **8.79** |
| Ellipsoid + variable ε | 8.65 | 9.83 | 7.71 | 12.32 | 18.81 | 14.63 | 13.42 |

### 关键发现
- **斜率确实线性**：n50(m) 在 PG19 与随机串两种 domain 上线性拟合 $R^2 \geq 0.995$，直接验证 Cor 4.6 的 $n^\star(m)=Cm$ 预测。
- **PG19 斜率大于随机串**：自然文本结构性强、有效信息量小，所以同样 m 下能"装下"更长的序列，与 Kuratov 2025 一致。
- **几何修正一步一步把误差压下来**：满球 → cone 几乎无收益（cone 体积仍接近球），ellipsoid 因捕捉到各向异性把比值降 1–4×，再叠加非均匀 cell 把所有模型压到 4.5–10.8×。Qwen-2.5-1.5B 始终偏大，提示其嵌入支撑可能用了更复杂的形状。
- **Copying 任务出现断崖**：fine-tune 到 100% 精度后，长串准确率近乎 step function（sigmoid R² 中位 0.95），直接验证 Cor 4.10 预测的"prompt-independent 阈值之外指数衰减"。
- **结论可直接迁移**：Remark 1.1 指出有界内部表征 + 有限精度的任何架构（包括 Mamba 等 SSM）都受同一上界约束；test-time training 的 memory 向量长度 h 也可由本文公式预测。

## 亮点与洞察
- **"几何 → 容量"翻译非常干净**：把生成能力上界写成嵌入空间的 packing number，是这类问题里最自然的形式语言；从此 transformer 的"能不能做"问题等价于"嵌入支撑里能不能塞下足够多互不相交精度球"。
- **解释性极强的两个 corollary**：Cor 4.6 解释"为什么短 prompt 下阈值随 m 线性增长"，Cor 4.10 解释"为什么再长的 prompt 也救不了 copying"——两句话覆盖了过去三年关于 transformer copying / cramming 的所有实证现象。
- **mean-field + Wasserstein 把"任意长度"纳入同一框架**：把 prompt 替换成经验测度 $M(X)$，让"prompt 无限长"的极限有了可分析的精度概念（Wasserstein-ε），是把固定 m 结果扩到 prompt-independent 的关键步。
- **可迁移 trick**：用"unembedding 矩阵 + Monte-Carlo 采样"实测 cell 体积分布 $D$，再做 n-fold 乘积卷积估临界 n —— 这套"数据驱动的几何估计"可直接用于估计任何固定 transformer 在新任务上的可达长度。
- **结论对架构形状无知**：所有上界只依赖 $d, r, \varepsilon, |V|$ 四个参数，因此既适用于 Pythia 也适用于 Mamba；理论给出闭式后，预测新模型 cramming 失败长度成本几乎为 0。

## 局限与展望
- 上界仍偏松 5–10×：Qwen-1.5B 即使叠加 ellipsoid + 非均匀 cell 仍有 10.8× 误差，说明 last-layer embedding 的真实形状比椭球更复杂（可能有低维流形结构），需要更精细的支撑模型。
- Assumption 4.8（Wasserstein 有限精度）比 ℓ∞ 精度强很多，Appendix E 给的"elementary operations"替代假设可读性较差，限制了 Thm 4.9 的直观说服力。
- 实验侧只评估了 cramming / copying 两个"机械"任务，未触及推理类任务；这类任务的目标序列可能集中在小子空间，可达性约束未必是主要瓶颈。
- 仅覆盖 ≤3B 参数模型，70B+ 的大模型是否仍服从同样比例尚未验证；也未与浮点精度（FP16/BF16）的实际 ε 值做精细对照。
- 仅给上界没给下界：理论上仍可能存在比 packing 更紧的容量限制，使得真实失败长度远低于本文阈值，特别是当 cell 形状不规则时。

## 相关工作与启发
- **vs Kuratov et al. 2025（cramming 经验研究）**：他们最早把 cramming 当作 accessibility 的可操作定义并报告 PG19/random gap，本文给出 cramming 现象的第一个严格理论解释，且把实验扩展到多模型族。
- **vs Jelassi et al. 2024（"Transformers can't copy"）**：他们实证 transformer 复制长串失败，本文证明这是嵌入空间几何的必然结果，与训练 / 模型大小无关。
- **vs Huang et al. 2025（length generalization 框架）**：Huang 等证明绝对位置编码下 copying 的不可能性，但依赖理想化假设；本文移除这些假设，并把"不可能性"量化为可预测的临界长度。
- **vs Meyer et al. 2025（memory limits）**：同样用 packing number 论证，但聚焦 memory；本文把同一工具推到 generation capability。
- **vs Chiang 2025、Strobl et al. 2024（形式语言 / 复杂度类）**：那条线分析的是 transformer 能表达哪类语言（TC⁰、电路深度），本文则定量地告诉你"能输出多少不同序列"，是表达力研究的**计数版本**——前者关心"做不做得到"，本文关心"做得到多少条"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一次把 transformer 可达序列数装进一个闭式 packing-number 上界，并能预测 cramming 阈值。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个模型族 11 个 size + 两种任务两种 target 分布，但都偏机械任务且未到 10B+ scale。
- 写作质量: ⭐⭐⭐⭐ 定义、定理、推论、实验对应清晰，几何直觉（Fig 1、Fig 3）非常友好；mean-field 那段 Wasserstein 假设跳跃稍快。
- 价值: ⭐⭐⭐⭐⭐ 给"transformer 为什么不会复制"提供了首个 architecture-agnostic 的理论，对 SSM、test-time training 等后续研究有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] One Persona, Many Cues, Different Results: How Sociodemographic Cues Impact LLM Personalization](../../ACL2026/llm_nlp/one_persona_many_cues_different_results_how_sociodemographic_cues_impact_llm_per.md)
- [\[ICML 2026\] "I've Seen How This Goes"：用渐进条件惊奇度刻画 LLM 与人类写作的多样性](ive_seen_how_this_goes_characterizing_diversity_via_progressive_conditional_surp.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](../../ACL2025/llm_nlp/can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)
- [\[ACL 2025\] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs?](../../ACL2025/llm_nlp/llm_meets_scene_graph_can_large_language_models_understand_and_generate_scene_gr.md)
- [\[ACL 2026\] Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning](../../ACL2026/llm_nlp/synthetic_eggs_in_many_baskets_the_impact_of_synthetic_data_diversity_on_llm_fin.md)

</div>

<!-- RELATED:END -->
