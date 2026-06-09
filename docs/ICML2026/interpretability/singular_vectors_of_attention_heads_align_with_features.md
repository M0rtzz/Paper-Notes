---
title: >-
  [论文解读] Singular Vectors of Attention Heads Align with Features
description: >-
  [ICML 2026][可解释性][注意力头] 本文从理论与玩具模型两侧，论证了"为什么以及何时"注意力头 QK 矩阵 $\Omega = W_Q^\top W_K$ 的奇异向量会与模型实际使用的特征方向对齐，并提出"稀疏注意力分解"作为该对齐在真实模型（GPT-2 / Pythia）中可被验证的可观测信号。
tags:
  - "ICML 2026"
  - "可解释性"
  - "注意力头"
  - "SVD"
  - "特征对齐"
  - "稀疏注意力分解"
  - "线性表征假设"
---

# Singular Vectors of Attention Heads Align with Features

**会议**: ICML 2026  
**arXiv**: [2602.13524](https://arxiv.org/abs/2602.13524)  
**代码**: https://github.com/gaabrielfranco/svf-alignment (有)  
**领域**: 机械可解释性 / Mechanistic Interpretability  
**关键词**: 注意力头, SVD, 特征对齐, 稀疏注意力分解, 线性表征假设

## 一句话总结
本文从理论与玩具模型两侧，论证了"为什么以及何时"注意力头 QK 矩阵 $\Omega = W_Q^\top W_K$ 的奇异向量会与模型实际使用的特征方向对齐，并提出"稀疏注意力分解"作为该对齐在真实模型（GPT-2 / Pythia）中可被验证的可观测信号。

## 研究背景与动机

**领域现状**：机械可解释性的核心任务是找出语言模型中"概念"的内部表征。当前主流的线性表征假设（LRH）认为，概念以一维或低维子空间方向的形式被加性叠加进激活里；近年的若干工作（Merullo 2024、Ahmad 2025、Pan 2024、Franco & Crovella 2024/2025）经验性地发现：注意力头 QK 矩阵的奇异向量往往就是这些特征方向。

**现有痛点**：上述"奇异向量 = 特征"的现象虽然被反复观察到，但一直缺乏理论解释——既不清楚为什么会出现，也不清楚在什么条件下会成立。同时，主流的特征发现手段也各有问题：linear probe 只能说明信息可解码、不能说明模型真的用了这个方向；SAE 训练昂贵，且只看激活、忽略权重；circuits 分析依赖人工选定方向。

**核心矛盾**：LRH 告诉我们激活是特征之和，但没告诉我们怎样把激活分解回去；另一方面，SVD 提供了一组天然的正交基，可它和"模型实际使用的特征"是否同一组方向，是个经验观察而非定理。

**本文目标**：把这个经验观察形式化，回答三个递进问题——(1) 奇异向量与特征对齐在玩具模型里是否稳健可复现？(2) 该现象可否从优化目标里推导出来，并给出成立条件？(3) 在无法直接观测特征的真实模型里，有没有可以验证对齐确实发生的可观测预测？

**切入角度**：作者沿用 Elhage 2022 的 toy autoencoder（学一组特征 $\{w_i\}$、把输入重构为 $W f$），再叠加一个真实的注意力头 $\Omega = W_Q^\top W_K$，让特征 $W$ 和注意力权重 $\Omega$ 在同一个 loss 下联合训练。这样特征和奇异向量都是"可观测的真值"，可以直接对照它们之间的余弦相似度。

**核心 idea**："对齐"不是巧合，而是注意力训练目标与重构损失的共同解：注意力 loss 把奇异向量拉向"感兴趣的特征对"，重构 loss 把其他无关特征推到正交方向，于是 $\Omega^\star$ 的 top 奇异向量自然就被特征"占据"，剩下空间留给噪声。

## 方法详解

整篇论文的结构是一个"实验 → 理论 → 真实模型预测"的闭环：先在玩具模型里复现现象，再用三条定理给出对齐成立的形式化条件，最后导出"稀疏注意力分解 (Sparse Attention Decomposition, SAD)"这一可在 GPT-2 / Pythia 上直接测的预测。

### 整体框架

输入是一组语义离散特征 $\{w_i \in \mathbb{R}^D\}_{i=1}^N$，每个特征以伯努利-均匀的方式被激活，组成 token $r = W f$。模型由两部分构成：(a) toy autoencoder——用 $f' = \mathrm{ReLU}(W^\top r + b)$ 重构 $f$，loss 为 $\mathcal{L}_{\text{recon}} = \|f - f'\|_2^2$；(b) 单个注意力头——对 query token $r$ 和 key 集合 $S = \{s_j\}$ 计算 logit $\ell_j = r^\top \Omega s_j$，并用 softmax 输出 $p_{\text{head}}$。注意力的训练目标由一个"特征-特征"模板 $T$ 指定：目标 logit 是 $\ell^T(r,s) = \sum_{ij} T_{ij} f_i^{(r)} f_j^{(s)}$，attention loss 是 $\mathcal{L}_{\text{attn}} = \mathrm{CE}(p_{\text{head}}, p_{\text{target}})$，总 loss 为 $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{attn}}$。

因为 $W$ 和 $\Omega$ 是模型自己学出来的，作者可以训练结束后做 SVD $\Omega = U \Sigma V^\top$，再用余弦相似度矩阵直接对齐特征列 $w_i$ 与奇异向量列 $u_k, v_k$，从而得到对齐是否成立的"真值标签"。

### 关键设计

**1. Toy 模型上的对齐-正交化耦合：把现象拆成两股优化压力**

在一个特征和奇异向量都能直接看到真值的最小设置里复现"奇异向量 = 特征"，前提是要把现象拆得足够细，作者于是设计了由简到繁的两档模板。单 feature-pair 情形下设 $T_{01} = 1$、其余为 0，训练后 $\Omega^\star$ 的谱里只剩一个显著奇异值，且 $w_0$ 与 $u_0$、$w_1$ 与 $v_0$ 的余弦相似度接近 1；多 feature-pair 情形下设 $T_{i,i+20}$ 线性递减，则多个特征会按"重要性"顺序依次占据 top 奇异向量。关键观察是同时浮现的二级现象：被注意力关心的特征会与"无关特征"正交化，而无关特征被压缩进 $D-2$ 维子空间。这两个效应在训练里是耦合的——奇异向量先动以降低 attention loss，特征随后正交化以降低 reconstruction loss。把"对齐"和"正交化"拆成相互配合的两股压力，既为后面的定理提供了锚点，也让训练动力学（图 3 的分时演化）本身成为机制证据，而不是只看终态那一张快照。

**2. 三条定理：给出对齐成立的解析条件**

要把现象上升为可证的结论，就得算出 $\Omega^\star$ 收敛后奇异向量的解析形式。令 query/key 侧特征矩阵为 $X, Y$，Gram 矩阵 $\Sigma_X = XX^\top$、$\Sigma_Y = YY^\top$。Theorem 1 给出主结论：若目标 logit 满足 $\ell^T(r,s) = 1$ 当且仅当 $x_1, y_1$ 同时出现，则收敛后 $\Omega^\star$ 是秩-1，左/右奇异向量分别为 $u_1 \propto \Sigma_X^{-1} x_1$、$v_1 \propto \Sigma_Y^{-1} y_1$，即"协方差白化后的特征方向"。Corollary 1 是其干净特例——当特征各向同性 ($XX^\top \propto I$) 时，$u_1, v_1$ 精确等于 $x_1, y_1$。Theorem 2 把结论推到现实：即便存在各向异性，只要特征间干扰 $\|E_X\|_2$ 有界，对齐仍以近似形式成立。Theorem 3 则补上正交化那一半——当 $\Omega$ 固定时，重构损失的最优解会自动把特征推成正交，从而解释了"无关特征正交化"。三条定理覆盖了各向同性 / 各向异性 / 特征自身演化三个角度，把对齐从经验观察变成对特征几何的明确刻画。作者还用 GPT-2 的 SAE 字典元素当代理特征，量化出 $\|E_X\|_2$ 落在 10–55 之间，说明 Theorem 2 的条件在真实模型里确实可被满足、定理并非空设。

**3. 稀疏注意力分解 (SAD)：在真实模型里的可测预测**

定理的结论在 GPT-2 上无法直接证伪，因为真实模型里看不到特征真值，作者于是把"奇异向量 = 特征"翻译成一个能观测的信号——注意力 logit 拆到 SVD 基上时应当稀疏。把 logit 写成 $\ell(r,s) = \sum_k r^\top u_k \sigma_k v_k^\top s$，代入 $r = W f^{(r)}$、$s = W f^{(s)}$ 后展开为 $\ell(r,s) = \sum_k \sum_{i,j} f_i^{(r)} (w_i^\top u_k) \sigma_k (v_k^\top w_j) f_j^{(s)}$；在对齐假设下只有当 $w_i, w_j$ 都与同一个 $k$ 对齐时该项才显著，所以外层关于 $k$ 的和应该是稀疏的。为了消掉 softmax 带来的偏置，作者引入"相对注意力" $\tilde{\ell}_j = \ell_j - \frac{1}{m-1} \sum_{i \neq j} \ell_i$，并用 Rolls-Tovee 稀疏度 $S(v) = (\frac{1}{n} \sum_i |v_i|)^2 / (\frac{1}{n} \sum_i v_i^2)$ 来量化，同时定义 $N_{\text{recon}}(j)$ 为"凑出相对注意力所需的最小奇异向量数"。这个指标的价值在于它把不可证伪的理论变成可证伪的预测：若 SAD 真的在真实模型里出现、且把 $U, V$ 随机旋转后立刻消失，就强支持了对齐确实来自特征-奇异向量的特定配对，而非简单地由"少数大奇异值"或低秩性质造成。

### 损失函数 / 训练策略

总 loss $\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda \mathcal{L}_{\text{attn}}$，作者在附录 A 中对 $\lambda$、特征数 $N$、上下文长度 $m$、头维度 $H$、随机种子做了完整 sweep，证明 SVF 对齐对这些超参数不敏感。Toy 模型典型配置为 $N = 20, D = 10, H = 10$（单对场景）或 $N = 100, D = H = 50$（多对场景）。真实模型侧使用 Pythia-160M 的 130 个 checkpoint 和 GPT-2 的 IOI 任务 prompt（128 个变体）。

## 实验关键数据

### 主实验

| 实验 | 模型 | 关键观测 | 数值/说明 |
|------|------|---------|---------|
| 单对齐 (图 2a) | Toy, $N=20, H=10$ | $w_0 \leftrightarrow u_0$、$w_1 \leftrightarrow v_0$ 余弦相似度 | 接近 1.0；$\Omega^\star$ 仅有 1 个显著奇异值 |
| 多对齐 (图 2b) | Toy, $N=100, H=50$ | 20 对特征同时与 top-20 奇异向量对齐 | 奇异值幅度 ≈ 线性目标 logit |
| 各向异性鲁棒性 (图 4) | Toy, anisotropy 扫到 GPT-2 范围 | 平均余弦相似度 | > 0.75，即便 $\|E_X\|_2$ 接近 GPT-2 上限 |
| SAD 在 Pythia (图 7b) | Pythia-160M, IOI 头 | $S(v)$ 训练前后变化 | 从 ~1 显著下降；随机旋转 $U,V$ 后不下降 |
| $N_{\text{recon}}$ on GPT-2 (图 9a) | GPT-2, 128 IOI prompts | 重构相对注意力所需奇异向量数 | 多数注意力头落在 1–4 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full toy model | 余弦相似度 ≈ 1 | 默认配置下对齐稳健 |
| 取消注意力头 (图 1a) | 特征各向同性排列 | 重现 Elhage 2022，没有对齐目标 |
| 加入 head + 单对 (图 1b) | $w_0, w_1$ 与其它特征正交化 | 验证 Theorem 3 的正交化压力 |
| Pythia: 随机旋转 SVD 基 (图 7b 底, 图 8 底) | $S(v)$ 不再下降 | 排除"少数大奇异值"假设 |
| RoPE (Appendix D) | 对齐仍然成立 | 位置无关和位置相关 logit 均可观察到 |

### 关键发现
- **对齐是优化耦合的结果，不是 SVD 的副产品**：通过随机旋转 $U, V$ 实验，作者证明若把奇异向量打乱再投影，相对注意力的稀疏性立即消失。换句话说，稀疏性来自特定方向上的特征-奇异向量配对，而不是矩阵谱本身。
- **最大贡献项常对应小奇异值**：图 8 中部显示相对注意力的主项往往来自谱底端，这反过来说明语义重要性 ≠ 奇异值大小，仅看 top-$k$ SVD 是不够的，必须做"按 token 的相对分解"。
- **真实模型中 1–4 个奇异向量就能解释一个头的注意力**：$N_{\text{recon}}$ 在 GPT-2 和 Pythia 上的分布说明，注意力头使用的特征子空间相当低维，使得"SVD basis 作为候选特征空间"在工程上可行。
- **对齐对各向异性具备相当容忍度**：即便 anisotropy 推到 GPT-2 实际上限，对齐余弦相似度仍 > 0.75，说明这套方法不是"只在干净玩具模型里成立"的脆弱现象。

## 亮点与洞察
- **把"经验对齐"翻译成"可证 + 可测"两件事**：定理给出对齐发生的形式化条件，SAD 给出在真实模型里独立验证的方法，两者形成完整证据链，比纯经验研究更经得起追问。
- **协方差白化的形式直接指明了实践改进路径**：定理 1 表明真正与特征对齐的不是裸 SVD，而是 $\Sigma_X^{-1} u$、$\Sigma_Y^{-1} v$ 形式的"反白化"奇异向量。这给后续工作一个明确的算子：先估出特征协方差再 unwhiten，能显著提高对齐。
- **"相对 logit + 稀疏度指标"是个可迁移的探针**：$\tilde{\ell}_j$ 和 $S(v)$ 几乎不依赖具体模型架构，可直接用来诊断任意 attention 头是否"在做单特征/少特征匹配"，对 circuit 发现、特征级 ablation 有直接价值。
- **从对齐到 SAE 的方法论替代**：如果 SVF 对齐成立，那么单次前向就能在 SVD basis 里"读出"候选特征，不再需要训练昂贵的 SAE。这是对当前 SAE-centric 的可解释性主流路线的一个清晰对照。

## 局限与展望
- 作者承认：本文未直接在真实模型上验证"SVD 派生方向就是因果意义上的特征"，而是引用 Franco & Crovella 2025 等先前工作对此提供因果证据；如果想把方法独立于这些前文使用，仍需自己的因果实验。
- 真实模型存在的"cone direction"（异常过表示方向）可能让部分奇异向量与之对齐而非与语义特征对齐，附录 C 只做了初步研究。
- 全部分析限于"特征数 ≤ 头维度 $H$"的情形；当特征数超过头容量时，附录 E 的初步实验显示最不重要的特征会共享一对奇异向量，但完整理论缺位。
- 仅分析单头，多头协同（不同头的奇异向量如何被分配 / 是否存在跨头的 superposition）未触及，是后续工作的明显方向。
- 实验只覆盖 GPT-2 / Pythia 两个相对小的模型；现代 7B+ 模型上 SVF 对齐的成立程度还需要进一步实证。

## 相关工作与启发
- **vs Merullo 2024 / Ahmad 2025 / Pan 2024 / Franco & Crovella 2024-2025**：他们经验性观察到 SVF 对齐，并基于此构建解释或 circuit 工具；本文不再止步于经验，而是给出 (a) 严格的可证条件、(b) 与 reconstruction loss 耦合的机制解释、(c) 在真实模型上可独立验证的预测（SAD）。
- **vs SAE 系工作 (Bricken 2023, Huben 2024)**：SAE 只用激活、不用权重，训练昂贵且存在特征分裂/吸收问题；本文方法直接用 $\Omega$ 的 SVD basis 做单次前向分解，廉价、可解析，并显式利用了权重信息。
- **vs Linear Probe**：probe 是相关性证据，不保证模型真的使用这个方向；SVF 对齐 + SAD 给出的是模型自己优化出来的、和注意力 logit 直接相关的方向，更接近"因果使用"的语义。
- **vs Elhage 2022 (Toy Models of Superposition)**：本文沿用其 toy autoencoder，但额外加入真实注意力头，把"特征几何"和"注意力权重谱"放进同一个 loss 里联合分析，因此能解释 SVF 对齐这一原始 toy model 触及不到的现象。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把广泛观察到的 SVF 对齐第一次给出可证条件和可测预测，机制清晰
- 实验充分度: ⭐⭐⭐ Toy 模型扫得很全，真实模型仅 GPT-2 与 Pythia-160M，缺更大模型与更多任务
- 写作质量: ⭐⭐⭐⭐ 实验-理论-真实模型预测三段闭环，定理叙述清楚，附录把鲁棒性扫得很扎实
- 价值: ⭐⭐⭐⭐ 给可解释性提供了 SAE 之外的另一条可工程化路线，思想可直接迁移到 circuit / feature ablation 工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](../../NeurIPS2025/interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[ACL 2026\] Retrieval Heads are Dynamic](../../ACL2026/interpretability/retrieval_heads_are_dynamic.md)
- [\[ICML 2026\] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning](how_few-shot_examples_add_up_a_causal_decomposition_of_function_vectors_in_in-co.md)
- [\[ICML 2026\] CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features](corrsteer_generation-time_llm_steering_via_correlated_sparse_autoencoder_feature.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](../../ACL2026/interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)

</div>

<!-- RELATED:END -->
