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

这篇论文要回答的是"一个有限记忆的模型，至少要花多少比特才能把事实记住、记不全时会以什么形态出错"。作者的做法是先把 LLM、Bloom filter 这些看似不相干的东西抽象成同一个对象——**成员测试器** $\mathcal{M}=(\text{Init},\text{Query})$：Init 吃进 key 集合 $\mathcal{K}$ 吐出一段 memory $W$，Query 拿到查询 $(i,W)$ 给出置信分数 $\hat x_i\in[0,1]$。把记忆开销定义成 $B(\mathcal{M})=I(W;\mathcal{K})$（memory 里关于 key set 的互信息，下界于物理比特数 $H(W)$），再用两个损失函数 $d^K,d^N:[0,1]\to[0,\infty]$ 分别度量 key 与非 key 在置信 $\hat x$ 上的错误，约束在 $\varepsilon_K,\varepsilon_N$ 以内，整个问题就变成"给定误差预算求最小 $B(\mathcal{M})$"。剩下的推导分三步落地：先证一个非渐进 per-key 下界，再在事实稀疏极限下把它收成一个干净的 $\min\mathrm{KL}$，最后把这个界套到 log-loss 上读出幻觉的形态。

### 关键设计

**1. 率失真主定理（Theorem 3.1）：把"记多少比特"从离散组合问题变成一个连续凸优化。**

经典 Bloom-filter 空间下界（Carter et al. 1978、Pagh & Rodler 2001、Hurley & Waldvogel 2007）都是 case-by-case 地用计数或互信息硬算，彼此割裂、还各自留着说不清的常数。作者的统一招法是给任意 permutation-invariant 测试器引入一个辅助 Bernoulli 变量 $X\sim\text{Bern}(p)$（$p=n/u$），并令 $\hat X|X{=}1\sim\mu_K$、$\hat X|X{=}0\sim\mu_N$，于是互信息 $I(X;\hat X)$ 正好刻画了 $n$ 个 key 与 $u-n$ 个非 key 之间的"区分难度"。Lemma 3.4 给出非渐进下界 $B(\mathcal{M})/n\ge F_p(\mu_K,\mu_N)-\log(8n)/(2n)$，其中 $F_p=I(X;\hat X)/p$；Lemma 3.5 证 $F_p$ 下半连续且 $\partial F_p/\partial p=-\mathrm{KL}(\mu_N\|p\mu_K+(1-p)\mu_N)/p^2$，于是在 $p\to 0$ 稀疏极限下 $F_p\to\mathrm{KL}(\mu_K\|\mu_N)$，整个下界塌缩成

$$B(\mathcal{M})/n \;\ge\; \min_{\mu_K\in\mathcal{C}_K,\ \mu_N\in\mathcal{C}_N}\ \mathrm{KL}(\mu_K\|\mu_N).$$

可达性由 Lemma 3.7 的一个 hash-based 构造补上，Theorem 3.3 再给出 finite-$p$ 的修正项 $-\chi^2(\mu_K^*\|\mu_N^*)/(2\ln 2)\cdot p+o(p)$。这一步之所以有力，是因为它把所有历史下界收编进同一个 $\min\mathrm{KL}$ 表达式，并精确补齐了 Pagh-Rodler 一直留着的 $\Theta(1)$ 加性常数。

**2. log-loss 下的"幻觉通道"最优解（Theorem 4.1）：固定记忆下，最优策略既不是遗忘也不是弃答，而是幻觉。**

主定理只给了抽象的 $\min\mathrm{KL}$，真正反直觉的结论要把 LLM 自己的损失代进去才看得见。对 LLM 的概率估计，$d^K(\hat x)=-\ln\hat x$、$d^N(\hat x)=-\ln(1-\hat x)$ 恰好就是 log-loss / binary cross-entropy，与 maximum likelihood 训练严格一致。在非平凡情形 $\varepsilon_K,\varepsilon_N>0$ 且 $e^{-\varepsilon_K}+e^{-\varepsilon_N}>1$ 下，作者用变分法配 KKT 求出唯一最优解 $\mu_K^*=\delta_{x^*}$、$\mu_N^*=(1-q^*)\delta_0+q^*\delta_{x^*}$，其中 $x^*=e^{-\varepsilon_K}$、$q^*=\varepsilon_N/[-\ln(1-x^*)]$——也就是说，事实全压在一个高置信点 $x^*$ 上，而非事实有一部分比例 $q^*$ 偏偏跟事实挤在同一个 $x^*$、剩下 $1-q^*$ 投到 0。代入得 per-key 最小记忆量 $\mathrm{KL}(\mu_K^*\|\mu_N^*)=\log(1/q^*)$，于是幻觉概率 $q^*=2^{-\mathrm{KL}}$ 完全由记忆容量决定，与你怎么在 $\varepsilon_K$、$\varepsilon_N$ 之间分配 trade-off 无关。这个解之所以重要，是因为它把"幻觉是数学上必然最优"这件事钉死在了 log-loss 本身，不依赖任何具体训练算法：遗忘（把 $\hat x$ 都推向 $1/2$）和 abstention（专门留一个"我不知道"标签）在 log-loss 下都严格次优。

**3. 两侧 filter 的阈值不变性：把结论推广到任意阈值机制，并顺势解释 RAG 与 long-tail 微调为何有效。**

第二个结论是关于概率估计的，但现实里很多缓解幻觉的手段是在置信分数上做阈值判断。作者证明任何对 $\hat x$ 阈值化的下游过程（包括 Kalai et al. 2025 的 generative classifier）都突不破 $\mathrm{KL}(\mu_K\|\mu_N)$ 这条下界——这是允许同时有 FP 与 FN 的"两侧 filter"的一般化。直接推论是：想消除 FP（幻觉）就必然抬高 FN（遗忘 / over-refusal），后处理只能沿同一条 frontier 滑动，跳不出去。真正能动 frontier 的只有改记忆预算本身：RAG（Lewis et al. 2020）接入 non-parametric 外部 memory，等于把 $B(\mathcal{M})$ 整体做大，frontier 随之外推；对 long-tail 事实额外 SFT 则是把更多参数容量显式分给随机事实，对应同一条 frontier 上"多花比特换更小 $q^*$"。这一视角一次性解释了三个经验现象：大模型上"教模型说 IDK"的 SFT 为何收效有限、Feldman 等人"memorize long-tail 是必要的"假说为何在信息论上成立、以及 MDL 视角下"有效记忆预算远小于参数数"为何合理——结构知识与随机事实本就在同一段 budget 里竞争。

### 实验设置

这是一篇理论论文，没有标准意义上的"训练目标"，实验只是对 Theorem 4.1 的经验 sanity check。作者用两类设置直接监测最优分布 $\mu_K,\mu_N$ 的形态：一是 **synthetic random strings**，从零训一个小 transformer 去记一组随机字符串；二是 **real-world ISBN + synthetic ID**，对预训练 LLM 做 LoRA 微调让它学一组随机 fact。两种设置都去看非事实输出是否真的呈现理论预测的 $\delta_0$ 与 $\delta_{x^*}$ 双峰。

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
