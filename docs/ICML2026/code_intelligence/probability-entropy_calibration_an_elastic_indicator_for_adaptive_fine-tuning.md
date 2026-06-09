---
title: >-
  [论文解读] Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning
description: >-
  [ICML 2026][代码智能][SFT] RankTuner 提出 Relative Rank Indicator $I_t$，用「真值 token 的实际排名 $R_t$」对比「模型分布下的期望排名 $\mathbb{E}[R_t]$」作为单一标量信号…
tags:
  - "ICML 2026"
  - "代码智能"
  - "SFT"
  - "token reweighting"
  - "概率-熵校准"
  - "相对排名"
  - "数学推理"
---

# Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning

**会议**: ICML 2026  
**arXiv**: [2602.01745](https://arxiv.org/abs/2602.01745)  
**代码**: https://github.com/LvAoAo/Ranktuner_VERL  
**领域**: LLM效率 / 监督微调 / Token 重加权  
**关键词**: SFT, token reweighting, 概率-熵校准, 相对排名, 数学推理

## 一句话总结
RankTuner 提出 Relative Rank Indicator $I_t$，用「真值 token 的实际排名 $R_t$」对比「模型分布下的期望排名 $\mathbb{E}[R_t]$」作为单一标量信号，把概率 $p_t$（任务对齐）和熵 $H_t$（内禀不确定性）拧成一个 token 级权重，在数学推理 SFT 上 Pass@1 普遍超过纯概率/纯熵的重加权 baseline。

## 研究背景与动机

**领域现状**：LLM 微调里，"每个 token 一视同仁"的标准 SFT 已被各种 token-level reweighting 改进，主流分两派——**Prob-Dominant** 用真值概率 $p_t$（如 DFT、TALR、OverTone），**Entropy-Dominant** 用预测熵 $H_t$（如 EAFT），都希望把梯度集中到"重要"token 上。

**现有痛点**：两派都是**一维**信号。Entropy-Dominant 会把 "umm"、"essentially" 这类 filler/可替换词错认成"高不确定 = 重要"而上权重，反而强化噪声；Prob-Dominant 又会狠狠惩罚所有低 $p_t$ 的位置，把那些本来就有多种合理同义词的 token 当成错误去硬拧，破坏预训练给的语言灵活性。论文用一个"故意注入噪声 token"的诊断（Tab. 1）显示：在 top-10% 高权重 token 里，Entropy 派召回了 55% 的噪声，Prob 派召回 40%，而 RankTuner 只召回 26%——一维信号确实在"误伤"。

**核心矛盾**：$p_t$ 度量"下游对齐"，$H_t$ 度量"上游预训练先验给的难度"，两者是正交维度；任何只看一边的方案都会把"难但不该硬学"和"容易但学错了"混在一起。

**本文目标**：构造一个**同时**反映 $p_t$ 和 $H_t$ 的标量 token 权重，且要可比、可解释、训练稳定。

**切入角度**：概率和熵单位不同没法直接相除，但**排名**（rank）是两边共通的量纲——真值的实际排名 $R_t$ 受 $1/p_t$ 上界约束，模型分布下的期望排名 $\mathbb{E}[R_t]$ 被熵 $H_t$ 下界约束（Guessing Problem 经典结论）。换到排名空间，两边就能放进同一个比值里。

**核心 idea**：用 $I_t = 2^{f(R_t)-f(\mathbb{E}[R_t])}$（$f(x)=1/\log_2(x{+}1)$）这一相对排名信号刻画"在这个难度下你猜得有多差"，再取倒数 $S_t = I_t^{-1}$ 作为 SFT 损失的 token 权重，把更新集中到"真正欠学"而非"本来就高熵"的位置。

## 方法详解

### 整体框架
RankTuner 要解决的是"一维 token 权重会误伤"这个问题：它不动模型结构、不引入新参数，只在 SFT 的加权 NLL 损失 $\mathcal{L} = -\mathbb{E}[\sum_t w_t \log p_t]$ 里把基础权重 $w_t$ 换成 $\tilde{w}_t = w_t \cdot S_t$。每个 target token $y_t$ 在前向时已经有了完整词表分布 $\pi_\theta(\cdot|y_{<t},x)$，于是顺手算出真值的实际排名 $R_t$ 和模型分布下的期望排名 $\mathbb{E}[R_t]$，用这两个排名拧出一个把概率与熵都装进去的标量 $S_t$，再乘回 token 损失。数学任务沿用 $w_t=p_t$（兼容 DFT 系），通用任务取 $w_t=1$。

### 关键设计

**1. Relative Rank Indicator $I_t$：把概率和熵翻译到同一个"猜测代价"量纲**

痛点在于 $p_t$ 和 $H_t$ 单位不同、范围不同，没法直接相除融合，硬凑出来的比值也缺乏意义。RankTuner 的破题点是 Guessing Problem 这一经典视角：真值的实际排名 $R_t$ 就是"沿模型降序词表一路猜到真值要花几次"，期望排名 $\mathbb{E}[R_t]=\sum_{\hat i} \hat i \cdot p_{t,\hat i}$ 则是"按模型分布随机猜的期望次数"——而排名恰好被概率与熵分别从两侧夹住：$R_t \le 1/p_t$，$\mathbb{E}[R_t] \ge \tfrac{1}{4}2^{H_t}{+}1$（$H_t\ge 2$ 时），两条紧界把概率、熵都桥接进了同一个 rank 空间，比值因此天然可比。

在此基础上定义 $I_t = g\big(f(R_t)-f(\mathbb{E}[R_t])\big)$，取 $f(x)=1/\log_2(x{+}1)$（NDCG 里常用的对数压缩）、$g(x)=2^x$（把零差归一到 $I_t=1$）。直觉很清楚：$R_t$ 越大（猜得越差）$I_t$ 越小，$\mathbb{E}[R_t]$ 越大（位置本就难）$I_t$ 越大——同样错一个 token，在高难度位置惩罚轻、低难度位置惩罚重。而当 $R_t$ 和 $\mathbb{E}[R_t]$ 都很大时 $I_t$ 饱和到 1 附近，自然形成一块 "Noise Region"，把那些高熵又低概率的可替换/噪声 token 中性化，这正是一维信号最容易误伤的地带。

**2. Relative Competence 模板与 CMVT 推导：给 $I_t$ 一个概率论解释**

为了说明 $I_t$ 不是手工凑的启发式、而是在近似一个有意义的能力比值，论文先写出抽象的 token 能力分 $C_t = \rho(p_t)/\kappa(H_t)$（$\rho$ 随 $p_t$ 单调增、$\kappa$ 随 $H_t$ 单调减），类比条件概率 $\Pr(A|U)$——把 $p_t$ 视作"对齐与先验支持的联合"，把 $H_t$ 映射成有效先验支持。接着用 Cauchy 中值定理把差分 $f(R_t)-f(\mathbb{E}[R_t])$ 改写成对数比形式，得到 $I_t = (\mathbb{E}[R_t]/R_t)^{K(\xi_t)}$，其中推理 token 的典型区间 $K(\xi_t) \approx 0.5$。把 rank 的上下界代回去就有 $\hat\rho(p_t)=p_t^{K(\xi_t)}$、$\hat\kappa(H_t)=s(H_t)^{-K(\xi_t)}$，从而 $I_t \gtrsim \hat C_t = (p_t \cdot s(H_t))^{K(\xi_t)}$，正好坐实"$I_t$ 是能力比值的下界估计"。

这套桥接的价值在于把 $(f,g)$ 的选择"降级"成一个具体的 CMVT 实例：附录进一步证明换其他单调 $(f,g)$ 结果都稳定，说明真正带来增益的是"用 rank 把概率和熵校准在一起"这条原则，而非这一对特定的对数-指数函数——这也是它比经验启发式 reweighting 更可解释、更好调参的根本原因。

**3. Relative Scale $S_t = I_t^{-1}$ 与训练接入：欠学的位置压更多梯度**

最后要把指示子变成能直接乘到损失上的权重。这里关键是取倒数而非直接用 $I_t$ 当奖励：若直接用 $I_t$，那些"已经学得好"的 token 会被加权过大，反而过拟合到容易的位置；取倒数后 $S_t = (p_t \cdot s(H_t))^{-K(\xi_t)}$ 等价于"哪儿欠学就压更多梯度"，已掌握的 token 被自然 down-weight。工程上为了稳训练，实践令 $\xi_t = \max(R_t, s(H_t))$、$K(\xi_t) = (\log_2(\xi_t{+}1))^{-2}$（省掉 $\xi/(\xi{+}1)$ 因子），最终 $\tilde w_t = w_t \cdot S_t$。整个计算只是在已有 forward 的 logits 上加一次排序加期望排名累加，不引入额外网络，推理阶段完全不用，零推理开销；对 PPO/GRPO 这类 RL 后训练也兼容（论文留作 future work）。

### 损失函数 / 训练策略
基础损失沿用加权 NLL；数学推理 SFT 用 $w_t = p_t$（与 DFT 同基），其它通用任务 $w_t = 1$。训练在 verl 框架上 4×A800 完成，10k 条 NuminaMath-CoT，AdamW lr=5e-5，cosine + 0.1 warmup，batch 256，max len 2048，生成时温度 1.0、最长 4096。

## 实验关键数据

### 主实验：数学推理 Pass@1 / Pass@16（Qwen3-8B，节选）

| 数据集 | 指标 | RankTuner | 最强 baseline | $\Delta$Best | 原模型 |
|---|---|---|---|---|---|
| MATH-OAI | P@1 | **72.38** | 70.92 (DFT) | +1.46 | 65.14 |
| MATH-OAI | P@16 | **90.20** | 90.20 (EAFT) | +0.00 | 87.40 |
| Minerva Math | P@1 | 38.26 | 40.46 (TALR) | -2.20 | 31.39 |
| Minerva Math | P@16 | **65.44** | 63.60 (EAFT) | +1.84 | 48.53 |
| OlympiadBench | P@1 | **36.25** | 35.07 (DFT) | +1.18 | 27.19 |
| OlympiadBench | P@16 | **64.00** | 60.00 (TALR) | +4.00 | 51.11 |
| AIME24 | P@1 | **10.21** | 8.75 (DFT) | +1.46 | 6.04 |
| AIME24 | P@16 | **26.67** | 26.67 (TALR) | +0.00 | 26.67 |
| AMC23 | P@1 | **46.56** | 45.78 (DFT) | +0.78 | 35.62 |
| AMC23 | P@16 | **85.00** | 80.00 (EAFT/TALR) | +5.00 | 75.00 |

在 Qwen2.5-Math-7B 上趋势一致：6/10 项 best 或并列 best，AIME24 P@16 维持原模型水平的同时 P@1 +0.83，明显比那些"P@1 涨 P@16 崩"的 baseline 更稳。

### 噪声敏感度诊断（Tab. 1）

| 方法 | TOK PREC@10% ↓ | TOK REC@10% ↓ | SEQ HIT@10% ↓ |
|---|---|---|---|
| Entropy-Dominant | 4.54% | 55.33% | 77% |
| Prob-Dominant | 3.25% | 39.65% | 77% |
| **RankTuner** | **2.16%** | **26.39%** | **9%** |

人工往 SFT 数据里注入噪声 token，看 top-10% 高权重里覆盖了多少噪声：RankTuner 的"误把噪声当重点"比例显著低于两派一维方案，序列级命中率从 77% 暴跌到 9%，这是把"高熵 + 低概率"压到 Noise Region 的直接证据。

### 关键发现
- **OOD 推理迁移**：在 ARC-C / GPQA 两个非数学推理 benchmark 上 RankTuner 同样最优，说明 calibration 信号本身不绑死数学任务；DFT 那种"已自信再加权"的策略反而出现 over-sharpening、迁移变差。
- **可替换 token 自动去权**：CoT 上的可视化里，"them"、"all"这类代词稳定落在 $I\approx 1$ 中性区，而 "frac"、"0"、"{" 这些计算关键 token 落在 $I < 1$ 的深红区——表明这套信号确实把"语言灵活性"和"计算正确性"分了开。
- **理论界经验吻合**：Qwen3-8B 在 Minerva Math 上 $R$-$p$ 散点贴 $R=1/p$ 上包络，$\mathbb{E}[R]$-$H$ 散点贴熵下界，验证 rank 作为概率/熵代理是紧的，校准不是空中楼阁。
- **对 $(f,g)$ 不敏感**：附录消融换不同单调 $(f,g)$ 结果稳定，说明收益来源是"用 rank 把概率和熵桥起来"这个原则，而非具体对数指数形式。

## 亮点与洞察
- **维度选对了**：把无法直接比的 $p_t$ 和 $H_t$ 都翻译到"猜测代价"这一共通量纲上，再做一次比值——这种"先找等价空间再融合"的思路在多信号融合里很可复用（比如 RLHF 里 reward 与 KL 项的耦合）。
- **诊断设计漂亮**：注入噪声 token 然后看 top-k 权重的 precision/recall，给"哪种 reweighting 真在选对位置"提供了量化指标，比单看下游 Pass@k 更接近 reweighting 的本职目的；这套 protocol 可直接搬到任何 token-level 加权方法的评测里。
- **零推理开销 + 框架友好**：只需要 forward 之后多一次排序与累加，权重乘到 loss 上即可；论文已经在 verl 上落地，按住"工程上能用 / 不改架构"这条线，比那些要改 attention 或采样的方法门槛低得多。

## 局限与展望
- 计算 $\mathbb{E}[R_t]=\sum_{\hat i}\hat i \cdot p_{t,\hat i}$ 需要完整词表分布上的排序，词表大（>100k）时虽然每 token 是 $O(|V|\log|V|)$ 可接受但累计开销不可忽略；论文未给出长上下文 + 大词表场景下的训练时延比。
- 主结果只在 NuminaMath-CoT 10k 上跑，且 backbone 集中在 Qwen 家族（Qwen2.5-Math-7B、Qwen3-8B），跨家族 / 跨任务（如 code 生成虽然摘要提了但表里未呈现）实验偏薄。
- 与 RL 后训练（PPO / GRPO）的结合仅是 future work，appendix 给了一种把 $S_t$ 注入 token-level policy ratio 的形式但未实测；这其实是当前 reasoning 模型最关心的场景。
- $K(\xi_t)$ 用了"省掉 $\xi/(\xi{+}1)$ 因子"的近似，论文用稳训练来辩护，但没有展示该近似在长序列上对梯度有效尺度的影响。

## 相关工作与启发
- **vs DFT / TALR / OverTone（Prob-Dominant）**：它们靠 $p_t$ 的单调函数决定权重，在 Pass@1 上常拿不错增益但 Pass@16 容易掉（AIME24、AMC23 上明显），且 OOD 泛化差；本文证明"加上熵的上下文化"就能在 P@1 和 P@16 都保住。
- **vs EAFT（Entropy-Dominant）**：纯熵会把 filler token 上权重，在噪声诊断里召回率最高，本文用 Noise Region 的形成机制把它"反着用"——熵高时反而中性化，而不是上权重。
- **启发**：把 ranking metric（NDCG 里的对数衰减）从"评估指标"反向用作"训练信号"的思路有意思；任何有"真值 + 模型分布"的 token-level 任务（多标签、检索蒸馏）都能模仿这套 rank 桥接来做更稳的加权。

## 评分
- 新颖性: ⭐⭐⭐⭐ rank 桥接 + CMVT 解释让"概率-熵融合"第一次有了清晰的理论叙事，而非拼装的启发式。
- 实验充分度: ⭐⭐⭐ 5 个数学 benchmark + 2 个 OOD + 噪声诊断够说服力，但 backbone 集中、缺 code/通用 SFT 的硬数据。
- 写作质量: ⭐⭐⭐⭐ 动机→反例→理论→实现→实验链条清晰，Fig. 1 的四象限示意图把"为什么一维不行"讲得很直观。
- 价值: ⭐⭐⭐⭐ 零结构改动、零推理开销、可即插即用到现有 SFT 流水线，对 reasoning 微调有直接工程价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Entropy-informed Decoding: Adaptive Information-Driven Branching](entropy-informed_decoding_adaptive_information-driven_branching.md)
- [\[AAAI 2026\] Unintended Misalignment from Agentic Fine-Tuning: Risks and Mitigation](../../AAAI2026/code_intelligence/unintended_misalignment_from_agentic_fine-tuning_risks_and_m.md)
- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](../../ACL2025/code_intelligence/gift_gibbs_fine_tuning_code_gen.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ICLR 2026\] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](../../ICLR2026/code_intelligence/imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
