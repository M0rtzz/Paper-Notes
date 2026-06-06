---
title: >-
  [论文解读] Entropy-informed Decoding: Adaptive Information-Driven Branching
description: >-
  [ICML 2026][代码智能][熵自适应] EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search…
tags:
  - "ICML 2026"
  - "代码智能"
  - "熵自适应"
  - "分支因子"
  - "beam search"
  - "推理时算力分配"
  - "regret bound"
---

# Entropy-informed Decoding: Adaptive Information-Driven Branching

**会议**: ICML 2026  
**arXiv**: [2605.09745](https://arxiv.org/abs/2605.09745)  
**代码**: 无  
**领域**: LLM 解码 / 自适应推理 / 信息论  
**关键词**: 熵自适应, 分支因子, beam search, 推理时算力分配, regret bound

## 一句话总结
EDEN（Entropy-informed DEcodiNg）把每一步的束宽 $B_t$ 设成与归一化熵 $\bar H_t$ 单调正比——高熵 fork 多分支、低熵步骤近贪心——用更少的总扩展近似更宽的 beam search；理论上证明熵单调的分支因子在期望累计 regret 上严格优于任何固定束宽，且能给出 $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$ 的显式 regret 率。

## 研究背景与动机

**领域现状**：LLM 推理时解码大体分两条线：(1) 采样类（top-$k$、nucleus $p$、min-$p$、top-$H$）用随机性换多样性，但通常只走一条路径；(2) 搜索类（beam search、best-of-$n$、majority voting）显式展开多个候选再选优，但算力消耗与任务难度无关——简单题也照样跑满 $n$ 个分支。

**现有痛点**：采样类**探索过窄**，一次只承诺一条路径，遇到推理分叉容易被早期低概率 token 锁死；搜索类**探索均匀**，简单 token 和困难 token 给同样多的算力，浪费严重。已有 entropy 相关工作（Simonds 2025、Entropix、Top-$H$、HARP 等）要么只把熵当作触发开关（branch or not），要么用它做模型切换 / 采样阶段截断，从没有人**把熵直接连续映射到 beam 宽度**并给出理论保证。

**核心矛盾**：好的解码应该「该贪心就贪心、该展开就展开」，但现有方法把分支因子写死成超参，无法在生成过程中根据 token 难度自适应分配。

**本文目标**：(1) 设计一个可插拔、模型无关的搜索策略，让每步算力随模型自身不确定性变化；(2) 用 noisy-maximization 框架给出严格的 regret bound；(3) 对闭源模型也能用（仅 API 访问也能估熵）。

**切入角度**：把 next-token 选择视作 sub-Gaussian noise 下的 noisy maximization 问题——若估计预算 $m_t$ 与「步难度」匹配，单步出错概率呈指数下降；而 Shannon 熵 $H_t$ 既能刻画候选数量（perplexity $\text{PP}_t = e^{H_t}$）又能刻画 top-2 间隔 $\gamma = \log(p_1 / p_2)$。

**核心 idea**：把每步束宽设成 $B_t = \max(1, \lfloor B_\max \cdot \bar H_t \rfloor)$，$\bar H_t = H_t / \log |\mathcal{V}|$ 是归一化熵；高熵 fork 自动多 branch，低熵步骤自动退化成贪心。

## 方法详解

### 整体框架
EDEN 是 beam search 的一个变种：维持活跃候选集 $\mathcal{B}$，每步根据每个候选当前的下一 token 分布 $P(y_{t+1} | x, y_{1:t})$ 估算归一化熵 $\bar H_t$，再用 $B_t = \max(1, \lfloor B_\max \cdot \bar H_t \rfloor)$ 决定要展开多少个 top-$B_t$ token。每个新候选用累积 log-prob $s(y_{1:t+1})$ 和长度归一 $\text{Score}_\alpha = s / t^\alpha$ 排序，并通过上下界剪枝（$\bar S \geq S^*$ 才保留）；EOS 候选直接评分，其他用「未来全 1 概率」当上界、「未来全均匀」当下界。最后返回归一化分最高的序列。

### 关键设计

1. **熵到分支因子的单调映射**:

    - 功能：把不确定性量化的「需要多少候选」直接转成搜索预算。
    - 核心思路：用最朴素的分段线性映射 $f(H, B_\max) = \max(1, \lfloor B_\max \cdot \bar H \rfloor)$，$0 \leq \bar H \leq 1$ 是按 $\log |\mathcal{V}|$ 归一化的熵。当 $\bar H \to 0$（模型超有信心），分支数退化到 1，等同贪心；当 $\bar H \to 1$（极度模糊），扩展接近 $B_\max$。
    - 设计动机：作者用两条引理证明 (a) Lemma 3.1：高熵意味着 $\varepsilon$-typical 集大小至少 $(1-\varepsilon)\text{PP}^{1/\varepsilon}$，候选数多得值得探；(b) Lemma 3.2：top-2 对数间隔 $\gamma \geq \log(e^{-H}/(1-e^{-H}))$，低熵时间隔大易选、高熵时间隔小难选。两条合起来说明「熵越大越需要更多估计预算」。

2. **基于上下界的剪枝与 EOS 处理**:

    - 功能：在保持理论 admissibility 的前提下尽早砍掉无望分支。
    - 核心思路：对每个尚未 EOS 的候选，给出未来分数的上下界 $\bar S, \underline S$——上界假设剩余 token 概率全是 1（最乐观），下界假设全是 $1/|\mathcal{V}|$（最悲观）。若 $\bar S < S^*$（当前最佳），整支砍掉；否则用 $\underline S$ 更新 $S^*$。EOS 候选则把上下界都等于实际分数。
    - 设计动机：beam search 的算力主要浪费在「不可能赢」的候选上，admissible 上界让剪枝既有效又不会误伤最优解，相当于把 A* 思想搬进 LLM 解码。

3. **闭源模型友好的熵估计**:

    - 功能：API-only 场景下也能用 EDEN。
    - 核心思路：对只能拿到样本（无 logits）的闭源 API，作者给出 sub-linear 样本复杂度上界——估熵到 $\epsilon$ 精度只需要 $\tilde O(1/\epsilon^2)$ 次采样，远小于词表大小。配合分桶估计可以在每步几十次采样里完成熵估计。
    - 设计动机：让 EDEN 不只是「需要 logits」的本地推理优化，也能武装到 ChatGPT API、Claude API 之类只暴露采样接口的封闭模型上。

### 损失函数 / 训练策略
本工作不训练任何参数，纯 inference-time 算法。理论部分把 next-token 选择写成 noisy max：$V_t(i) = \log P(i | x_t) + \text{OPT}_{t+1}(i)$，估计器 $\hat V_t(i)$ 的方差代理 $\sigma_t^2 = \delta^2 / m_t$。Proposition 3.3 在 Lipschitz 连续性假设下得到「所需预算 $m_t \gtrsim \frac{1}{(\Delta_t^\text{eff})^2} \log(\text{PP}_t / \delta_t)$，随 $H_t$ 单调上升」；Theorem 3.4 进一步证明熵单调分支因子在期望 regret 上严格优于任何固定束宽，前提是熵在解码过程中有变化（实证上 Wang 2026、Cao 2026 已证明 LLM 生成熵确实变化大）。

## 实验关键数据

### 主实验

| 方法 | GSM8K ↑ | MATH500 ↑ | HumanEval ↑ | SciBench ↑ | Friedman Rank ↓ | EDEN > 该方法的后验概率 |
|------|---------|-----------|-------------|------------|-----------------|--------------------------|
| Greedy | 73.5% | 27.4% | 27.0% | 4.9% | 6.12 | 0.99 |
| Top-$k$ | 70.7% | 23.0% | 27.6% | 4.9% | 7.00 | 1.00 |
| Top-$p$ | 73.5% | 27.4% | 27.0% | 4.8% | 6.62 | 0.99 |
| Top-$H$ | 69.7% | 26.0% | 27.0% | 4.5% | 8.12 | 1.00 |
| Min-$p$ | 72.3% | 28.0% | 25.8% | 4.3% | 8.00 | 1.00 |
| Best-of-5 | 78.2% | 28.2% | 27.0% | 5.2% | 4.62 | 0.96 |
| Beam search (width 3) | — | — | — | — | — | 0.77 |
| **EDEN ($B_\max = 5$)** | **best avg** | — | — | — | **best** | — |

（Friedman 检验 $p = 0.012$，差异统计显著；Bayesian hierarchical 给 EDEN 75% 的「整体最佳」后验概率。）

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| EDEN (full) | 准确率最高、扩展次数最少 | 熵自适应分支 |
| 固定 beam width = 3/5/7 | 准确率持平或略低、扩展次数线性 | 验证「固定束宽浪费算力」 |
| 熵阈值二值触发 | 准确率介于 greedy 与 EDEN 之间 | 验证连续映射比 binary trigger 好 |
| 仅 API 闭源（限采样估熵） | 性能略降但仍优于 greedy | 验证 sub-linear 估熵的可行性 |

### 关键发现
- **EDEN 在 4 个 benchmark 上拿到最佳 Friedman rank**：Math、Code、Science 都覆盖，证明效益跨任务类型稳健，不是某一类任务的偶然。
- **比 beam search 用更少扩展达到相当或更好的准确率**：表 1 括号里的总扩展数比 width=3 beam 还少，验证了「近似更宽 beam」的承诺。
- **跨模型族鲁棒**：Llama-3.2-3B、Gemma3、IBM Granite、Mistral 上都看到改进（附录 B），说明熵作为信号在不同分布族里都靠谱。
- **Pareto 优势**：Bayesian 分析里 EDEN 对其他方法的 pairwise dominance 概率 ≥ 96%，对 beam search 也有 77%。
- **变方差越大越赚**：Theorem 3.4 的几何直觉是「熵越变化越多，自适应分配越有优势」——实证里推理 / 代码任务熵确实波动大，所以 EDEN 在这些任务上提升最显著。

## 亮点与洞察
- **理论 + 实证双轨**：一边用 noisy max + sub-Gaussian 的标准武器给出 regret 率，一边在 4 个任务 × 4 个模型上做 Bayesian 后验分析，论证结构非常完整。
- **「分支因子作为算力一阶变量」的视角**：解码社区长期把束宽当超参；本文显式把它升级为 step-wise 状态变量，给类似 speculative decoding、early exit 等推理时优化指出一条参数化思路。
- **API 友好性**：sub-linear 熵估计让 EDEN 在闭源模型上仍可用，这是社区里很少有的「不依赖 logits」的 search-based 解码方案，潜在部署面广。

## 局限与展望
- 只在 3B 量级模型 + 标准 benchmark 上评测；70B+ 大模型上熵分布更平、分支收益是否仍显著未知。
- $f(H, B_\max)$ 是分段线性，参数 $a=1, b=0$ 是默认值；论文未系统调优非线性映射形式，可能存在更优函数族。
- regret bound 的常数 $c$ 和 Lipschitz $\Lambda$ 在实际系统中难以测量，理论保证更多是「定性指南」而非「定量预算」。
- 实验里 $T = 400$ 的最大生成长度，对长文/agent 链路（数千 token）下 EDEN 的累计收益和算力曲线没有覆盖。
- 与 RL / process reward model 类自适应推理方法的正交结合尚未探索。

## 相关工作与启发
- **vs Entropix / Simonds**：他们用熵做模型切换 / CoT token 插入；EDEN 把熵连续映射到束宽，是更细粒度的算力分配。
- **vs Top-$H$ 解码（Potraghloo et al. 2026）**：Top-$H$ 用熵截断采样分布；EDEN 把熵用在「展开多少分支」这一维度，思路正交可叠加。
- **vs HARP**：HARP 用熵触发 transformer 额外计算；EDEN 在更高层的搜索算法上做自适应，可以与 HARP 这类底层修改同时部署。
- **vs Best-of-$n$ / Majority voting**：这些方法均匀分配预算到独立完整 rollouts；EDEN 按需 token 级分配，对推理链路里「关键 fork」的算力使用效率更高。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把熵 → 束宽连续化并给出 regret 率，是该研究线的明确推进。
- 实验充分度: ⭐⭐⭐⭐ 4 任务 × 多模型族 + Bayesian 分析；缺大模型与长生成场景。
- 写作质量: ⭐⭐⭐⭐⭐ 引理-命题-定理链条清晰，把直觉与证明衔接得很好。
- 价值: ⭐⭐⭐⭐ 推理时算力优化是 LLM 部署的核心议题，EDEN 给出便宜且可解释的方案，落地潜力大。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning](probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning.md)
- [\[ICML 2026\] Locally Coherent Parallel Decoding in Diffusion Language Models](locally_coherent_parallel_decoding_in_diffusion_language_models.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](../../ACL2026/code_intelligence/learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ACL 2026\] Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding](../../ACL2026/code_intelligence/static_program_slicing_using_language_models_with_dataflow-aware_pretraining_and.md)

</div>

<!-- RELATED:END -->
