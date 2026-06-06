---
title: >-
  [论文解读] Provably Learning Attention with Queries
description: >-
  [ICML 2026][模型压缩][model stealing] 作者证明单头 softmax attention 在 value-query 访问下可以惊人简洁地被精确恢复 —— 只需 $O(d^2)$ 次查询，比同等结构的 ReLU MLP 容易得多…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "model stealing"
  - "value query"
  - "注意力机制"
  - "parameter recovery"
  - "compressed sensing"
---

# Provably Learning Attention with Queries

**会议**: ICML 2026  
**arXiv**: [2601.16873](https://arxiv.org/abs/2601.16873)  
**代码**: 无  
**领域**: 学习理论 / 模型抽取 / Transformer 可学习性  
**关键词**: model stealing, value query, single-head attention, parameter recovery, compressed sensing

## 一句话总结
作者证明单头 softmax attention 在 value-query 访问下可以惊人简洁地被精确恢复 —— 只需 $O(d^2)$ 次查询，比同等结构的 ReLU MLP 容易得多；当头维 $r\ll d$ 时还能借压缩感知降到 $O(rd)$，并把结论扩展到带噪 oracle、membership query 以及多头不可识别性。

## 研究背景与动机
**领域现状**：Transformer 已是工业部署主力，模型抽取攻击（model stealing）由此成为安全研究核心。Tramèr 2016 起对前馈网的提取已经有大量实证与理论成果，最近 Carlini 2024 甚至能从生产级 LLM 抓取嵌入矩阵和宽度。但「能否在 API 黑盒访问下证明恢复 softmax attention 的参数」这一最基础的问题，居然没人正面回答过。

**现有痛点**：(1) 现有 query learning 理论几乎全部集中在 ReLU FFN，且需要 Gaussian input、参数线性无关、general position 等强假设；(2) 在 attention 上，无 query 的被动学习问题（passive learning）由于 softmax + 双线性 score 是非凸的，本身已经非常难，作者们经常要靠 token-selection / max-margin / SVM 等工具且只在受限情形下证明；(3) softmax 把 token-pair 的双线性 $x_i^\top W x_N$ 与跨位置加权耦合在一起，朴素方法看不到「怎么把 $W$ 一项一项解出来」。

**核心矛盾**：attention 的非线性看似比 MLP 复杂（多了 softmax 归一化 + 序列长度可变），但作者发现这种「相对软」的非线性其实给攻击者送了把好刀 —— 只要能控制序列长度 $N$，就能让 softmax 退化为可逆 sigmoid，从而把 attention 还原成线性方程组；这是 ReLU MLP 完全没有的便利。

**本文目标**：(1) 给出第一份单头 attention 参数恢复的多项式算法与查询复杂度；(2) 把它接到 ReLU FFN learner 上得到一层 Transformer 的可学习性；(3) 对低秩、噪声 oracle、membership query、多头不可识别性等更现实场景全部给出结果。

**切入角度**：从「长度可控」这一 attention 特有优势出发 —— $N=1$ 时 softmax 权重恒为 1，直接得到 $f(X)=x^\top v$，可独立恢复 $v$；$N=2$ 时 softmax 退化为 sigmoid，可由 oracle 输出 $y$ 反解出 $\sigma^{-1}(\cdot)$ 得到关于 $W$ 列的线性方程。

**核心 idea**：用「单 token 查询恢复 $v$」+「双 token 查询通过 sigmoid 反演逐列恢复 $W$」共 $d^2+d$ 次 query 精确恢复 $(W^\star,v^\star)$，并把同一思路与压缩感知 / Lipschitz clipping / antisymmetric query 等手段组合，覆盖低秩、带噪、含 ReLU FFN 等所有变体。

## 方法详解

### 整体框架
攻击者面对 $f_{W^\star,v^\star}(X)=\text{softmax}(x_1^\top W^\star x_N,\dots,x_N^\top W^\star x_N)^\top(Xv^\star)$ 的 value oracle，目标是恢复 $(W^\star,v^\star)\in\mathbb R^{d\times d}\times\mathbb R^d$。算法分两阶段：(1) 长度-1 输入 $X=[e_i^\top]$ 直接读出 $v^\star_i$，$d$ 次 query；(2) 对每一列 $w_j=W^\star e_j$，构造长度-2 输入 $X=[(u+e_j)^\top; e_j^\top]$，让 softmax 退化为 $\sigma(u^\top w_j)$，再从 $y=v^\star_j+\sigma(u^\top w_j)\cdot u^\top v^\star$ 反解 $u^\top w_j=\sigma^{-1}((y-v^\star_j)/(u^\top v^\star))$，取 $d$ 个线性独立 $u$ 即可解出 $w_j$。一层 Transformer 通过 antisymmetric query $\widetilde{\text{VQ}}(X)=\text{VQ}(X)-\text{VQ}(-X)$ 消掉 ReLU，再调用上面的 attention learner；FFN 部分调用已有 $\mathcal A_{\text{FFN}}$（如 Milli 2019 或 Daniely-Granot 2023）。低秩、噪声、membership query 都是同一思路加工具加 patch。

### 关键设计

1. **长度-2 探针 + sigmoid 反演（Thm 4.1）**:

    - 功能：把 attention 这一非线性结构精确还原为线性方程组，逐列恢复 $W^\star$。
    - 核心思路：固定列 $j$，写 $X=[(u+e_j)^\top; e_j^\top]$。两个 score 分别为 $s_1=(u+e_j)^\top W^\star e_j$ 与 $s_2=e_j^\top W^\star e_j$，差值 $s_1-s_2=u^\top w_j$ 恰好把 $e_j^\top W^\star e_j$ 项消掉。位置 1 的注意力权重 $\alpha=\sigma(u^\top w_j)$。oracle 返回 $y=v^\star_j+\alpha\,(u^\top v^\star)$，只要 $u^\top v^\star\neq 0$ 就能由 $\alpha=(y-v^\star_j)/(u^\top v^\star)\in(0,1)$ 反推 $u^\top w_j=\sigma^{-1}(\alpha)$。取 $d$ 个线性无关 $u$（i.i.d. Gaussian 几乎必然满足）即可解出整列。
    - 设计动机：softmax 的「全长归一」是难点，但只要 $N=2$ 它就退化为 sigmoid —— 一个全局可逆、光滑的标量函数。先用 $N=1$ 把 $v^\star$ 撇清后，再用 $N=2$ 单独逼出 $W^\star$，复杂度由「列数 $\times$ 每列需要的探针数」给出 $O(d^2)$。

2. **低秩压缩感知恢复 $W^\star$（Thm 5.1）**:

    - 功能：在实践中头维 $r\ll d$（如 128 vs 4096），$W^\star=K^\top Q$ rank-$r$，将 $d^2$ 降到 $O(rd)$。
    - 核心思路：把上面探针改成 i.i.d. Gaussian $a,b\sim\mathcal N(0,I_d)$，$X=[(a+b)^\top; b^\top]$，同样得 $\alpha=\sigma(a^\top W^\star b)$，反解后得到秩-1 测量 $t=\langle ab^\top, W^\star\rangle$。收集 $m=O(rd)$ 个这种 ROP（rank-one projection）测量后解凸程序 $\min\|W\|_\ast\ \text{s.t.}\ \langle a_kb_k^\top,W\rangle=t_k$。由 Cai-Zhang 2015 的 RUB 保证，$m\geq Cr(2d)$ 时高概率精确恢复。
    - 设计动机：避开整 $d\times d$ 矩阵中每一项一次单独查询，让单次 query 给出关于 $W^\star$ 的一个「秩-1 线性快照」，再叠加 compressed sensing 的恢复理论 —— 直接换问题不换算法是这篇工作的方法学一大亮点。

3. **带噪 oracle 下的稳定恢复（Thm 6.1）**:

    - 功能：现实 API 都会给输出加微小 noise，而 $\sigma^{-1}$ 在 0/1 附近不 Lipschitz，朴素算法会爆炸；本文给一组 norm + margin 假设下的 $\epsilon$-精度多项式恢复。
    - 核心思路：将探针缩放为 $a=1/2$, $b=1/W$，构造 $X=[(b u+ae_j)^\top; (ae_j)^\top]$，把 logit 控制在 $|ab\,W^\star_{ij}|\leq 1/2$，使得 $\alpha^\star=\sigma(ab\,W^\star_{ij})\in[\sigma(-1/2),1-\sigma(-1/2)]$ 这一 Lipschitz 区间内；估计阶段对 $\hat\alpha$ 做 $\text{clip}(\hat\alpha;\tau_{\text{clip}},1-\tau_{\text{clip}})$ 再用 $\sigma^{-1}$ 反演，由 Lemma A.1 $|\sigma^{-1}(\text{clip}(\hat\alpha))-\sigma^{-1}(\alpha^\star)|\leq 5|\hat\alpha-\alpha^\star|$ 把误差线性传递，最终给出 $\tau=\mathcal O(\min\{\mu,\epsilon_v/\sqrt d,\mu\epsilon_W/(W^2 d)\})$ 即可达 $\|\hat W-W^\star\|_F\leq\epsilon_W,\|\hat v-v^\star\|_2\leq\epsilon_v$。
    - 设计动机：直接套噪声 oracle 的最大风险是 $\sigma^{-1}$ 把 $\hat\alpha$ 落到 $\{0,1\}$ 附近时误差爆炸；通过把探针尺度 $a,b$ 设计成「天然把 logit 锁在 $[-1/2,1/2]$」，让 $\alpha^\star$ 永远停留在 Lipschitz 区域，再 clip 异常估计 —— 这是把光滑性分析嵌进算法设计的典范。

### 损失函数 / 训练策略
本文不训练，证明全部围绕 query 复杂度 + 概率精度。低秩部分解的凸程序 $\min\|W\|_\ast$ 由 nuclear norm 引导；多头不可识别性由构造两个不同 $\{(W_h,v_h)\}$ 诱导相同输入-输出映射给出 Prop 7.1。

## 实验关键数据
本文为理论文章，无实证实验。主要"数据"是各种情形下的 query / 精度复杂度。

### 主实验

| 设定 | Query 复杂度 | 保证 | 假设 |
|------|-------------|------|------|
| 精确单头 attention 恢复 (Thm 4.1) | $O(d^2)$ | 精确 | $v^\star\neq 0$ |
| 低秩单头 attention (Thm 5.1) | $O(rd)$ | 精确，概率 $1-e^{-\Omega(m)}$ | $\text{rank}(W^\star)\leq r$, $v^\star\neq 0$ |
| 带噪 oracle (Thm 6.1) | $O(d^2)$ | $\|\hat W-W^\star\|_F\leq\epsilon_W$ | $\|W^\star\|_F\leq W$, $\min v^\star\geq\mu$ |
| 一层 Transformer (含 ReLU MLP) | $Q_{\text{FFN}}(d,m)+O(d^2)$ | 精确，依赖 $\mathcal A_{\text{FFN}}$ | $A^\star w_o^\star\neq 0$ |
| Multi-head attention | 不可识别 | 不存在算法 | 无额外结构 |

### 消融实验

| 变体 | 查询数 / 精度 | 备注 |
|------|--------------|------|
| Value query | $O(d^2)$, 精确 | 基线 |
| Membership query (App. B) | poly + bisection | 仅返回 ±1 标签，复杂度更高 |
| Antisymmetric query 消 ReLU | $2\times$ 单 query | 用于一层 Transformer 的 attention 部分 |

### 关键发现
- 同样含「一个非线性 + 一个矩阵 + 一个向量」的 single-head attention 与 single-hidden-layer ReLU MLP，前者 query 学习极其容易，后者至今仍需强假设 —— 来自 softmax 是全局可逆光滑函数而 ReLU 不是。
- 探针尺度的精心选择（low-rank 用 Gaussian、noisy 用 $b=1/W$）决定整套算法是否能闭合，是论文最实用的方法学经验。
- multi-head attention 由于 head 间可任意置换 + 线性叠加，存在无穷多参数化诱导同一函数，必须加结构假设（如 head 间正交）才能识别。
- 长度-1 query 直接读出 $v^\star$ 这一步看似平凡，但它是后续 sigmoid 反演的必要前置：没有 $v^\star$ 知道，从 $y$ 反推 $\alpha$ 缺少分母无从下手。
- 在 membership query（只返回二元标签）场景下，作者用 bisection 把 sigmoid 反演降级为多次比较 query，复杂度仍是多项式但常数显著放大。

## 亮点与洞察
- 「靠序列长度可控让 softmax 退化为 sigmoid」是 attention 才有的攻击面，把一层 Transformer 的安全性放在一个比 MLP 弱得多的位置 —— 对部署 LLM API 的厂商是个明确警示。
- 用 antisymmetric query $f(X)-f(-X)$ 把 ReLU 消掉，转化为线性等价问题，这一 trick 可复用于任何「奇变换 + 偶非线性」的网络结构分析。
- 把 model extraction 写在 PAC-style query complexity 框架下，给安全社区与学习理论社区搭起桥梁 —— 之前的 attack 论文绝大部分是实证派。
- 低秩场景下把 query 从 $O(d^2)$ 压到 $O(rd)$，对现代 LLM 头维 $r\sim 128$ vs 宽度 $d\sim 4096$ 的现实刚刚好 —— 意味着质量上可以抽取 SOTA 规模模型的注意力参数。
- 用探针尺度 $a=1/2, b=1/W$ 把 $\sigma^{-1}$ 锁定在 Lipschitz 区间上是「在算法设计深处嵌入光滑性分析」的典范，对一切涉及不稳定反函数的估计问题都有参考价值。

## 局限与展望
- 单头 + 线性 MLP 是最简版本，真实 Transformer 是多层多头 + LayerNorm + position encoding，理论与实际还差很多层抽象。
- 带噪情形需要 margin $\mu>0$，对 LLM 中常见的稀疏 / 接近零的权重不友好。
- Multi-head 的不可识别性只是给出反例，并未深入研究「正交头 / FFN gate 等结构假设下的可识别性边界」。
- 假设 $v^\star\neq 0$ —— 在 $v^\star=0$ 的退化情形 $W^\star$ 完全不可识别，但这一边界条件在实际抽取时如何检测论文未给出工程指南。

## 相关工作与启发
- **vs Chen et al. 2021 (Gaussian-input ReLU MLP)**：他们恢复 2 层 ReLU MLP 也是 query 模型，但需要 Gaussian input 和 distribution-dependent 论证；本文证明 attention 完全不需要分布假设。
- **vs Daniely-Granot 2023 (general position ReLU)**：本文一层 Transformer 的 FFN 子例程可直接调用其算法，是「现有 FFN learner + 我们的 attention learner」组合用法。
- **vs Carlini et al. 2024**：他们是工业级 LLM 上的实证抽取，本文是同问题的最小、可证明、被动版；两条线互补。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一份 softmax attention 的可证明参数恢复结果，方法漂亮
- 实验充分度: ⭐⭐ 全理论，无任何实证或 toy demo
- 写作质量: ⭐⭐⭐⭐⭐ 主定理证明就在正文一两页内可读完，每个引理动机清楚
- 价值: ⭐⭐⭐⭐ 给 model extraction 安全研究与 attention 理论分析同时打开新工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](../../AAAI2026/model_compression/share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)
- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[CVPR 2026\] BinaryAttention: One-Bit QK-Attention for Vision and Diffusion Transformers](../../CVPR2026/model_compression/binaryattention_one-bit_qk-attention_for_vision_and_diffusion_transformers.md)
- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](../../ICLR2026/model_compression/turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)

</div>

<!-- RELATED:END -->
