---
title: >-
  [论文解读] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing
description: >-
  [ICML 2026][模型压缩][幻觉] 本文把"LLM 记住随机事实"形式化为带连续置信分数的**成员测试**问题，证明在事实稀疏极限下最优记忆开销恰好等于事实/非事实输出分布之间的最小 KL 散度——即"率失真定理"——并由此推出：在 log-loss 目标下，给定有限记忆…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "幻觉"
  - "率失真定理"
  - "Bloom filter"
  - "KL 散度"
  - "成员测试"
  - "记忆容量"
---

# Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing

**会议**: ICML 2026  
**arXiv**: [2602.00906](https://arxiv.org/abs/2602.00906)  
**代码**: 暂未公开  
**领域**: LLM 安全 / 信息论  
**关键词**: 幻觉、率失真定理、Bloom filter、KL 散度、成员测试、记忆容量  

## 一句话总结
本文把"LLM 记住随机事实"形式化为带连续置信分数的**成员测试**问题，证明在事实稀疏极限下最优记忆开销恰好等于事实/非事实输出分布之间的最小 KL 散度——即"率失真定理"——并由此推出：在 log-loss 目标下，给定有限记忆，最优策略**不是弃答也不是遗忘**，而是把一定比例的非事实和事实压在同一个高置信点上，幻觉是信息论意义下的最优误差形态。

## 研究背景与动机

**领域现状**：LLM 幻觉的主流理论解释分两支：一支是 Kalai 等人 (2025) 的"分类视角"——把 LLM 看成对随机事实的二分类器，no-free-lunch 强行带来 false positive；另一支是"压缩视角"——把 LLM 看成对世界知识的有损压缩器，因此必然失真。前者解释了"为什么会错"，但没解释错的形态；后者大多是非形式化论证或假设事实数无限。

**现有痛点**：(i) "压缩→出错"无法解释为什么 LLM 更倾向**幻觉**而非**遗忘**——遗忘明显更"安全"；(ii) 即使在 closed-world 设定（所有事实有限且训练时见过）里，模型仍然在保留高置信幻觉、同时对合法 query 出现 over-refusal，单纯 no-free-lunch 无法 cover 这种 precision-recall 的怪异行为；(iii) 经典 Bloom-filter 空间下界（Carter et al. 1978; Pagh & Rodler 2001）和 LLM 幻觉理论各说各话，缺一个统一框架。

**核心矛盾**：在 LLM 的连续置信输出空间里，事实集 $\mathcal{K}$ 与非事实集 $\mathcal{U}\setminus\mathcal{K}$ 是不对称的——前者明确有限，后者远大且必须用同一段有限记忆编码。如何刻画"区分 $\mathcal{K}$ 与 $\mathcal{U}\setminus\mathcal{K}$ 所需的最小比特数"是把两类理论缝在一起的关键。

**本文目标**：(i) 给 LLM 与 Bloom filter 共用一个抽象"成员测试器"框架；(ii) 在稀疏极限 $|\mathcal{K}|/|\mathcal{U}|\to 0$ 下，给出 per-key 最小记忆开销的紧界；(iii) 在 log-loss 这一与 maximum likelihood 训练目标一致的度量下，刻画最优非事实输出分布的形状。

**切入角度**：作者把 LLM 看成 $\text{Init}+\text{Query}$ 两个算法的元组，输入 key 集合 $\mathcal{K}$ 产出 memory state $W$，对 query $i\in\mathcal{U}$ 给出置信分数 $\hat x_i \in [0,1]$。这一抽象同时涵盖 Bloom filter（$\hat x\in\{0,1\}$）与 LLM 的 likelihood 估计。然后用率失真理论刻画 $I(W;\mathcal{K})/n$ 在给定 $(\varepsilon_K,\varepsilon_N)$ 误差约束下的下界。

**核心 idea**：在稀疏极限下，per-key 最小记忆开销 = $\min_{\mu_K,\mu_N}\mathrm{KL}(\mu_K\|\mu_N)$，且 log-loss 下唯一最优解是"事实集中在一个高置信点 $x^*$ 上、非事实有一部分 $q^*$ 跟事实挤在同一个 $x^*$、剩余 $1-q^*$ 投到 0"——幻觉=空间最优。

## 方法详解

### 整体框架

作者把任何"区分 $\mathcal{K}$ 与 $\mathcal{U}\setminus\mathcal{K}$"的算法定义为 permutation-invariant 的**成员测试器** $\mathcal{M}=(\text{Init},\text{Query})$。Init 输入 $\mathcal{K}$ 输出 memory $W$，Query 输入 $(i,W)$ 输出 $\hat x_i \in [0,1]$。**记忆开销**定义为 $B(\mathcal{M})=I(W;\mathcal{K})$，即 memory state 中关于 key set 的互信息——这下界于物理 bits 数 $H(W)$。

误差由两个函数 $d^K, d^N : [0,1]\to[0,\infty]$ 刻画。Key/非 key 在置信 $\hat x$ 上的损失分别是 $d^K(\hat x), d^N(\hat x)$。约束误差水平为 $\varepsilon_K, \varepsilon_N$。对 Bloom filter，$d^K(\hat x)=1-\hat x$ 给出 FNR、$d^N(\hat x)=\hat x$ 给出 FPR；对 LLM 概率估计，$d^K(\hat x)=-\ln \hat x$、$d^N(\hat x)=-\ln(1-\hat x)$ 正是 log-loss/binary cross-entropy。统一框架的好处是两类问题之间可以平移空间下界。

整套理论分三步推：(1) 给出**非渐进 per-key 记忆下界** $B(\mathcal{M})/n \ge F_p(\mu_K,\mu_N) - \log(8n)/(2n)$，其中 $F_p(\mu_K,\mu_N)=I(X;\hat X)/p$（$X\sim\text{Bern}(p)$，$p=n/u$）；(2) 在 $p\to 0$ 稀疏极限下证 $F_p \to \mathrm{KL}(\mu_K\|\mu_N)$ 并配可达性引理 Lemma 3.7，得到主定理 Theorem 3.1；(3) 把该界套到 log-loss 上推出"幻觉通道"最优解 Theorem 4.1。

### 关键设计

1. **率失真主定理（Theorem 3.1）**:

    - 功能：把"记多少 bits 才能区分 $n$ 个 key 与 $u-n$ 个非 key 到给定误差"这个看似离散组合的问题，转化为一个连续凸优化 $\min_{\mu_K\in\mathcal{C}_K, \mu_N\in\mathcal{C}_N}\mathrm{KL}(\mu_K\|\mu_N)$。
    - 核心思路：对任意 permutation-invariant 测试器 $\mathcal{M}$，引入辅助 Bernoulli 变量 $X\sim\text{Bern}(p)$（$p=n/u$）与条件分布 $\hat X|X=1\sim \mu_K(\mathcal{M}), \hat X|X=0\sim \mu_N(\mathcal{M})$。互信息 $I(X;\hat X)$ 等于 $n$ 个 key 与剩余 $u-n$ 个非 key query 之间"区分难度"。Lemma 3.4 给出 $B(\mathcal{M})/n \ge F_p(\mu_K,\mu_N) - \log(8n)/(2n)$；Lemma 3.5 给出 $F_p$ 的下半连续性与 $\partial F_p/\partial p = -\mathrm{KL}(\mu_N\|p\mu_K+(1-p)\mu_N)/p^2$，从而在 $p\to 0$ 极限下 $F_p \to \mathrm{KL}(\mu_K\|\mu_N)$。可达性由 Lemma 3.7 用 hash-based 构造给出。Theorem 3.3 进一步给出 finite-$p$ 修正项 $-\chi^2(\mu_K^*\|\mu_N^*)/(2\ln 2)\cdot p + o(p)$。
    - 设计动机：经典 Bloom-filter 下界（Carter et al. 1978, Pagh & Rodler 2001, Hurley & Waldvogel 2007）都是 case-by-case 用计数或 mutual information 直接算。本定理把它们全部统一为 $\min\mathrm{KL}$，并精确补齐 Pagh-Rodler 留下的 $\Theta(1)$ 加性常数。

2. **log-loss 下的"幻觉通道"最优解（Theorem 4.1）**:

    - 功能：在事实/非事实分别用 log-loss $-\ln\hat x, -\ln(1-\hat x)$ 评估时，给出 $\min_{\mu_K,\mu_N}\mathrm{KL}(\mu_K\|\mu_N)$ 的闭式唯一解。
    - 核心思路：在非平凡情形 $\varepsilon_K, \varepsilon_N > 0$ 且 $e^{-\varepsilon_K} + e^{-\varepsilon_N} > 1$ 下，作者用变分法配合 KKT 验证，得到唯一最优 $\mu_K^* = \delta_{x^*},\ \mu_N^* = (1-q^*)\delta_0 + q^* \delta_{x^*}$，其中 $x^* = e^{-\varepsilon_K},\ q^* = \varepsilon_N / [-\ln(1-x^*)]$。代入即得 per-key 最小记忆量 $\mathrm{KL}(\mu_K^*\|\mu_N^*) = \log(1/q^*)$。**幻觉概率 $q^*$ 完全由记忆容量决定，与 $\varepsilon_K/\varepsilon_N$ 之间如何 trade-off 无关**。
    - 设计动机：log-loss 与 LLM 训练 maximum likelihood 严格一致；这意味着"幻觉是 LLM 在固定参数预算下数学上必然最优"的论断不依赖任何 specific 训练算法，是 information-theoretically forced。换句话说，遗忘（把 $\hat x$ 都推向 $1/2$）和 abstention（专门给一个"我不知道"标签）在 log-loss 下都次优。

3. **二侧 filter 阈值不变性 + RAG/微调的理论解释**:

    - 功能：把第二个结论从"概率估计"推广到"任意基于阈值的分类机制"，并解释为什么 RAG 和 long-tail 微调能缓解幻觉。
    - 核心思路：任何对置信 $\hat x$ 做阈值化的下游过程（包括 Kalai et al. 2025 的 generative classifier）都不能突破 $\mathrm{KL}(\mu_K\|\mu_N)$ 下界——这是"两侧 filter"（允许 FP 与 FN）的 generalization。corollary：消除 FP（幻觉）必然增大 FN（遗忘/over-refusal），后处理只能沿 frontier 滑动。而 RAG（Lewis et al. 2020）相当于把 non-parametric 外部 memory 接入，等于让 $\mathcal{M}$ 的 $B(\mathcal{M})$ 变大，自然把 frontier 整体外推；对 long-tail 事实做额外 SFT 相当于显式分配更多 parameter capacity 给随机事实，对应同一条 frontier 上的"花更多 bits 换更小 $q^*$"。
    - 设计动机：解释了三类经验现象：(a) 大模型上"abstention/IDK SFT"为何收效有限；(b) Feldman et al. 提出的"memorization 对 long-tail 必要"假说为何在信息论上成立；(c) regularization/MDL 视角下"effective memory budget" 远小于参数数为何合理——结构知识与随机事实在同一段 budget 里竞争。

### 损失函数 / 训练策略

理论论文，没有"训练"过程。实验上作者用两类设置经验验证 Theorem 4.1：(1) **synthetic random strings**：从零训小 transformer 学随机字符串集合；(2) **real-world ISBN + synthetic ID**：对预训练 LLM 做 LoRA 微调让其学一个随机 fact 集合。两种设置都直接监测 $\mu_K, \mu_N$ 的分布形态。

## 实验关键数据

### 主实验：分布形态匹配理论预测

| 设置 | 观测到的非事实输出分布 | Theorem 4.1 预测 | 一致性 |
|------|----------------------|------------------|--------|
| 小 Transformer + synthetic random strings | 双峰：$\delta_0$ + $\delta_{x^*}$ | $(1-q^*)\delta_0 + q^* \delta_{x^*}$ | ✓ |
| LoRA-tuned LLM + synthetic IDs | 显著的高置信"伪正"质量集中点 | 同上 | ✓ |
| LoRA-tuned LLM + real ISBN | 同样出现高置信幻觉聚集 | 同上 | ✓ |

关键观察：随着 memory budget（如 LoRA rank 或网络宽度）减小，$q^*$ 随 $\log(1/q^*)$ 大致线性增长——与 $\mathrm{KL}(\mu_K^*\|\mu_N^*) = \log(1/q^*)$ 严格匹配。这是理论的核心验证。

### 与经典空间下界比较

| 来源 | 下界形态 | 适用 |
|------|---------|------|
| Carter et al. (1978) | 一侧 filter, 稀疏极限 | 单侧 |
| Pagh & Rodler (2001) | 两侧 filter, 留 $\Theta(1)$ 加性常数 | 不紧 |
| Hurley & Waldvogel (2007) | 固定 $u/n$, mutual-info 风格 | 受限 |
| Li et al. (2023) | 一侧 filter, 非零 $n/u$ | 单侧 |
| **本文 Theorem 3.1** | $\min\mathrm{KL}$, 稀疏极限通用 | **全部 case 的紧界** |
| **本文 Theorem 3.3** | $\mathrm{KL} - \chi^2 p/(2\ln 2) + o(p)$ | finite-$p$ 修正 |

可达性方面：作者构造了一个 hash-based two-sided filter，$o(n)$ bits 误差内匹配下界，关闭了 Pagh-Rodler 留下的 gap。

### 关键发现

- **幻觉率仅由记忆预算决定**：$q^* = 2^{-\mathrm{KL}(\mu_K^*\|\mu_N^*)}$，与具体如何分配 $\varepsilon_K$ vs $\varepsilon_N$ 无关——这意味着提升对随机事实的精度只能靠加 memory，不能靠改 loss 权重。
- **abstention 在 log-loss 下不是最优**：任何把概率推向 $1/2$ 的"我不知道"行为，在 log-loss 下比直接幻觉一个高置信值更亏。这解释了为什么 SFT "教模型说 IDK"经验上效果有限。
- **RAG/外部记忆从根本上改变 frontier**：当 non-parametric memory 介入时 $B(\mathcal{M})$ 不再受参数预算限制，幻觉下界整体下移。

## 亮点与洞察

- **把两条历史悠久但平行的理论线缝在一起**：Bloom filter 空间下界（70 年代起）与 LLM 幻觉理论（2024 起）首次共享一个 master theorem。这种"把工程 data structure 下界搬进 LLM 理论"的视角本身就值得借鉴。
- **"幻觉是最优误差形态"是非常反直觉的论断**：人类直觉总觉得"模型应该在不知道时回 IDK"，本论文用一行 KKT 证明在 log-loss 下这是次优——给 LLM 安全设计提出了硬约束：要让 IDK 是最优，必须改 loss 函数本身（如 abstention-aware loss）或加 external memory。
- **可迁移的 trick**：用 permutation-invariance 把任意非对称 memory tester 规约到 permutation-invariant 子类的论证（Remark 2.2），是几乎任何 information-theoretic 下界证明都能复用的标准化技巧。
- **对 long-tail memorization 的辩护**：Feldman 等人提出的"memorize long-tail 必要"假说在本文中获得 information-theoretic 解释——并非"长尾偶然需要 memorization"，而是有限 budget 下不 memorize 就无法达到目标 error。

## 局限与展望

- **closed-world 假设**：所有事实都在 $\mathcal{K}$ 内且训练见过，与真实 LLM 面对 open-world（用户问没见过的事实）有显著差距；现实里 generative hallucination 还涉及 calibration 假设（Kalai & Vempala 2024 中讨论），本文显式跳过这块。
- **permutation-invariance 是工具不是现实**：真实 LLM 高度 position-dependent，作者承认下界仍成立但 tightness 可能松一些。
- **稀疏极限 $|\mathcal{K}|/|\mathcal{U}|\to 0$**：现实中"plausible claims"宇宙 $\mathcal{U}$ 的大小很难量化，把界量化到具体 LLM 容量需要额外假设。
- **没给出"如何修"的算法**：理论刻画了 frontier，没给出 frontier 上某点对应的训练算法；当前所有 LLM SFT pipeline 都在沿 frontier 滑动，没有方法学跳出 frontier。

## 相关工作与启发

- **vs Kalai et al. (2025)**: Kalai 等人从 calibration 出发证 generative hallucination rate 下界等于 induced classifier error rate，但 calibration 假设强制开放世界；本文用 closed-world 与 information theory 给出独立解释，二者互补。
- **vs Feldman (2020), Feldman & Zhang (2020)**: Feldman 等人提出"long-tail memorization 必要"，本文把这一假说升级为 information-theoretic 必然性。
- **vs Bloom-filter 经典下界（Carter et al. 1978, Pagh & Rodler 2001）**: 本文恢复并细化所有这些下界，关闭 Pagh-Rodler 的 $\Theta(1)$ 加性 gap，并把 Hurley & Waldvogel 的 mutual-info 界推广到 $p\to 0$ 极限。
- **vs 压缩派幻觉理论 (Mohsin et al. 2025; Shi et al. 2025; Kim 2025)**: 之前"压缩→出错"是非形式化或假设无限事实，本文在有限 closed-world 下给出严格率失真定理。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一次把 Bloom-filter 空间下界与 LLM 幻觉理论统一为 $\min\mathrm{KL}$ 框架，并给出 log-loss 下"幻觉通道"闭式解。
- 实验充分度: ⭐⭐⭐ 实验是理论的 sanity check 而非论文核心，覆盖 synthetic + 真实 ISBN 两类设置，足以验证关键预测但规模有限。
- 写作质量: ⭐⭐⭐⭐ 数学严谨，引理-定理结构清晰，但需要相当的 information theory 背景。
- 价值: ⭐⭐⭐⭐⭐ 给"为什么 LLM 必然幻觉"提供了第一性原理解释，对 LLM 安全、abstention 设计、RAG 价值评估都有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)
- [\[ICML 2025\] RADIO: Rate-Distortion Optimization for Large Language Model Compression](../../ICML2025/model_compression/radio_rate-distortion_optimization_for_large_language_model_compression.md)
- [\[ICML 2026\] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding](lk_losses_direct_acceptance_rate_optimization_for_speculative_decoding.md)
- [\[ICML 2026\] Exploiting Weight-Space Symmetries for Approximating Curvature](exploiting_weight-space_symmetries_for_approximating_curvature.md)

</div>

<!-- RELATED:END -->
