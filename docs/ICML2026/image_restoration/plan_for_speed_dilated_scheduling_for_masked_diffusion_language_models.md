---
title: >-
  [论文解读] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
description: >-
  [ICML 2026][图像恢复][扩散模型] 本文提出 Dilated Unmasking Scheduler (DUS)：用「等距空隙」预定义不依赖模型置信度的 unmask 顺序，把每块 $B$ 个 token 的 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$，在 LLaDA / Dream / DiffuCoder 上拿到 5.8× wall-clock 加速且质量优于基于置信度的并行 planner。
tags:
  - "ICML 2026"
  - "图像恢复"
  - "扩散模型"
  - "Dilated Scheduling"
  - "Joint Entropy"
  - "Parallel Decoding"
  - "Inference-only"
---

# Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models

**会议**: ICML 2026  
**arXiv**: [2506.19037](https://arxiv.org/abs/2506.19037)  
**代码**: [github.com/omerlux/DUS](https://github.com/omerlux/DUS)  
**领域**: 离散扩散语言模型 / 推理加速 / 信息论调度  
**关键词**: Masked Diffusion、Dilated Scheduling、Joint Entropy、Parallel Decoding、Inference-only

## 一句话总结
本文提出 Dilated Unmasking Scheduler (DUS)：用「等距空隙」预定义不依赖模型置信度的 unmask 顺序，把每块 $B$ 个 token 的 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$，在 LLaDA / Dream / DiffuCoder 上拿到 5.8× wall-clock 加速且质量优于基于置信度的并行 planner。

## 研究背景与动机
**领域现状**：Masked Diffusion 语言模型（MDLM，如 LLaDA-8B、Dream-7B、DiffuCoder-7B）允许「任意顺序、并行」解码，理论上可以打破自回归 $\mathcal O(G)$ 的延迟。但高质量生成通常仍需要每 token 一次 denoiser 调用，主流推理用半自回归（semi-AR）分块：序列切成长度 $B$ 的块，块内多步去噪。

**现有痛点**：现有 planner 选哪些 mask 位置揭开时都用置信度或熵这种 per-token 分数，结果是「高置信度的 token 通常彼此相邻」→ 自回归般从左到右一路填，并行性形同虚设；或者过激进一次揭一大片，但忽略 token 间的强依赖，质量崩盘。Path Planning 引入外部 BERT 评分，靠额外模型 call 换质量。

**核心矛盾**：并行 unmask 的真实目标是降低 $H(X_{\mathcal I_t}\mid\mathcal S_t)$（联合条件熵），但 planner 只能优化可分解的 $\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)$（边缘熵之和）；二者之差 $\Delta(\mathcal I_t;\mathcal S_t):=\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)-H(X_{\mathcal I_t}\mid\mathcal S_t)$ 总是 $\ge 0$，且当被选中的 token 互相强相关（通常空间相邻）时这个 gap 变大，并行采样退化为不连贯的输出。

**本文目标**：(1) 设计无需训练、无需外部 planner、无需 denoiser 评分的揭面调度；(2) 让 denoiser 调用次数从 $\mathcal O(B)$ 降到 $\mathcal O(\log B)$；(3) 同时保持/提升任务精度。

**切入角度**：从信息论上看，要让边缘熵之和近似联合熵，只需保证被并行选中的 token 在序列空间上尽量分散。在 fast-mixing VLMC 假设下，相距 $d$ 的 token 之间互信息按 $C\rho^d$ 指数衰减，因此「显式拉开空间间隔」就能直接收紧 entropy gap。

**核心 idea**：用预定义的对数等距（dilated）调度——第一轮揭间隔 $B/a$ 的位置、第二轮揭间隔 $B/a^2$、…—在 $\lceil\log_a B\rceil$ 轮内填满一块；planner 完全是预先确定的，不依赖任何模型输出。

## 方法详解

### 整体框架
DUS 要解决的是 MDLM 半自回归分块解码里「一块 $B$ 个 token 该怎么并行揭面才不伤质量」的问题。它把推理抽象成「固定 denoiser $\mathcal D_\theta$ + 可选 planner $\mathcal P$」的交互：每轮 $t$ 的状态 $\mathcal S_t$ 记录已 unmask 的 token，planner 选出本轮要揭开的索引集 $\mathcal I_t\subseteq\{b,\dots,b+B-1\}$，denoiser 给出每个位置的分布 $p_\theta(X_i\mid\mathcal S_t)$，从中并行采样填上 $\mathcal I_t$ 后进入下一轮。DUS 的做法是把基于置信度的动态 planner 换成一组完全预定义、与模型输出无关的调度 $\{\mathcal I_t\}_{t=1}^{R}$，让整块在 $R=\lceil\log_a B\rceil$ 轮内填满，把每块调用次数从 $\mathcal O(B)$ 压到 $\mathcal O(\log B)$。

### 关键设计

**1. 熵差最小化原则：把并行揭面是否伤质量数学化**

并行揭面想优化的真实目标是联合条件熵 $H(X_{\mathcal I_t}\mid\mathcal S_t)$，但 planner 实际只能优化可分解的边缘熵之和 $\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)$，二者之差就是 entropy gap $\Delta(\mathcal I_t;\mathcal S_t)$。本文用 Lemma 3.3 把这层关系夹起来：对任意分组方案有 $H(X_{\mathcal B}\mid\mathcal S_1)\le\sum_t\mathcal L(\mathcal I_t;\mathcal S_1,X_{\mathcal I_{<t}})\le\sum_{i\in\mathcal B}H(X_i\mid\mathcal S_1)$，左边等号当且仅当每组只揭一个 token（即退化成 token-by-token 自回归），右边对应一次全揭。Corollary 3.4 据此指出 $\Delta(\mathcal I_t;\mathcal S_t)\ge 0$ 且 planner 设计应显式最小化它。这一步之所以重要，是因为以前所有启发式都默认 per-token 分数和联合最优是一回事，而这个 gap 恰好解释了为什么置信度调度「分数明明最高、生成却不连贯」——被同时选中的高分 token 往往彼此相邻、强相关，gap 被撑大。

**2. 对数对偶 dilation 调度：用预定义间隔强制每轮揭开的 token 互相远离**

既然 gap 在被选 token 空间相邻时变大，DUS 干脆用一组预定义的间隔序列保证每轮揭开的位置尽量分散。给定块长 $B$ 和基数 $a>1$，迭代轮数 $R=\lceil\log_a B\rceil$，第 $t$ 轮步长 $s_t=\lfloor B/a^t\rfloor$，本轮选中 $\mathcal I_t=\{k\in\{1,\dots,B\}\setminus\mathcal U_{t-1}\mid (k-1)\bmod s_t=0\}$，并把已揭集合更新为 $\mathcal U_t=\mathcal U_{t-1}\cup\mathcal I_t$。以 $B=8,a=2$ 为例，三轮分别揭 $\mathcal I_1=\{1,5\}$、$\mathcal I_2=\{3,7\}$、$\mathcal I_3=\{2,4,6,8\}$ 就填满整块，共 $\mathcal O(\log_a B)$ 次 denoiser 调用。这种「早 sparse 晚 dense」的 coarse-to-fine 安排同时照顾了两头：早轮远距离揭开时，在 fast-mixing 假设下 token 间互信息小、gap 小，独立采样近似联合采样；晚轮虽然要填相邻空位、token 间相关性强，但此时 $\mathcal S_t$ 已含周围被揭开的丰富 context，per-token 条件熵已经很低，质量依然有保证。

**3. fast-mixing 下的指数收敛保证：dilation 调度为什么真能让 gap 任意小**

DUS 给 dilation 调度配了一个 VLMC（变长马尔可夫链）框架下的理论支撑。Lemma 3.6 证明 fast-mixing 的平稳遍历 VLMC 满足互信息按距离指数衰减 $I(X_i;X_{i+d},\dots,X_{i+(M+1)d})\le C\rho^d$；Lemma 3.5 与之联立后给出：只要被选中 token 两两距离 $\ge D_\varepsilon$，就有 $H(X_{i_1},\dots,X_{i_k}\mid\mathcal S_t)\ge\sum_j H(X_{i_j}\mid\mathcal S_t)-\varepsilon$，即边缘熵之和与联合熵之差被压到 $\le\varepsilon$。作者诚实地把 fast-mixing VLMC 只当成分析载体——真实文本未必满足，但 DUS 真正需要的只是「平均上空间相距 $d$ 的 token 不太相关」这一更弱条件，并在 HumanEval/MATH500 这类明显违反 fast-mixing 的代码/数学任务上实测仍稳定加速，佐证了这一放宽是合理的。

### 损失函数 / 训练策略
DUS 完全是 inference-only 的：不改 denoiser、不训新模块，调度静态可算。它还提供一个可选的 skip 启发式——每轮揭面前用 denoiser 算分，超过阈值的不确定位置延后到下一轮再揭（阈值在附录 B.2 扫描）。由于 dilation 思想与具体采样器正交，DUS 也能直接叠在 EB/CB 等自适应采样器上充当「dilated-spacing post-filter」。

## 实验关键数据

### 主实验
5 个 MDLM 变体（LLaDA-B/I 8B、Dream-I 7B、DiffuCoder-B/I 7B）、7 个 benchmark（GSM8K、MATH500、HumanEval、MBPP、BBH、MMLU-Pro、IFEval），block size $B\in\{8,16,32,64\}$，对比 self-confidence (Conf.)：

| 模型 | Benchmark | Token-by-token (×1) | Conf. B=32 (×6.4) | DUS B=32 (×6.4) |
|------|-----------|---------------------|--------------------|------------------|
| LLaDA-I | GSM8K | 80.29 | 38.74 | 65.73 |
| LLaDA-I | MATH500 | 28.80 | 10.8 | 19.2 |
| LLaDA-I | HumanEval | 39.02 | 9.76 | 10.37 |
| LLaDA-I | MBPP | 39.4 | 14.0 | 23.2 |
| Dream-I | GSM8K | 77.10 | 27.60 | 44.66 |
| LLaDA-I | BBH | 53.89 | 44.26 | 50.93 |
| LLaDA-B | MMLU-Pro | 39.82 | 24.11 | 32.50 |

DUS 在所有任务和所有 block size 下都击败 Conf.，尤其在 $B=32/64$ 的高加速档：GSM8K 上 LLaDA-I 用 DUS 在 6.4× 速度下仍保留 82% 单步精度，Conf. 只剩 48%。

### 消融实验
**Block size 与加速比的预测关系**（DUS 走的是 $R=\log_a B$，加速比与 NFE 严格挂钩）：

| B | DUS NFE/block | 速度比 | LLaDA-I GSM8K | Conf. 对应 |
|---|-----|--------|---------------|-----------|
| 8 | 3 | 2.7× | 73.24 | 69.22 |
| 16 | 4 | 4× | 70.66 | 61.41 |
| 32 | 5 | 6.4× | 65.73 | 38.74 |
| 64 | 6 | 10.7× | 57.09 | 18.73 |

DUS 的精度下降随 $B$ 缓慢、可预测；Conf. 在 $B=64$ 上直接崩塌（−61 分），说明它越往大块越被强依赖 token 害死。

### 关键发现
- DUS 是「确定性 trade-off」：选定 $B$ 就锁死速度和质量损失上界，工程部署友好；Conf. 的实际加速比与质量都是随输入波动的「期望值」。
- Coarse-to-fine 是关键：作者扫了固定 $k$ 的随机/置信度 planner 和 DUS-like 增量 $k$ 调度，固定 $k$ 永远比不上对数增长，验证了「早期 sparse、晚期 dense」是必要的。
- DUS 在违反 fast-mixing 的代码/数学领域仍稳定，平均 token 间距是其他 planner 的 2-3 倍，说明 spacing 是首要因素，互信息衰减只是理论辅助说明。
- 把 DUS 作为后处理 post-filter 挂在 EB/CB 上，能进一步提升它们的精度（Section 4.4），证明 dilation 思想是正交可加的。

## 亮点与洞察
- **把 unmask 顺序从 ML 问题变回信息论问题**：跳出「学一个 planner」的范式，用信息论 gap 给出确定性最优解的结构，是漂亮的「先理解再设计」案例。
- **predefined > learned**：完全不依赖模型置信度，反而稳健、可解释、可对账（用户能精确预知 NFE）；这一逆向直觉非常值得在其他场景借鉴。
- **theoretical-empirical mismatch 被诚实处理**：作者明确说 VLMC fast-mixing 假设在代码/诗歌里不成立，但实证有效，并解释 spacing 只需「平均上互信息小」即可——这种诚实的免责让理论用得更广。
- **post-filter 的可叠加性**：和 EB/CB/KV-cache 都是正交工具，工业部署可以一次加无数倍。

## 局限与展望
- VLMC 假设是宽容版本但仍存在：当文本含强长程依赖（代码的全局变量约束、押韵诗）时 entropy gap 不可控，DUS 的精度下降幅度会增加。
- 大 $B$ 上 DUS 精度仍随之下降，只是衰减比 Conf. 慢；不存在「免费午餐」加速。
- 调度完全 hand-crafted（基数 $a$ 在 $\{2,3,4\}$ 间扫），还未与可学习调度做端到端对比。
- 仅适用 masked diffusion，对 score-matching 类离散扩散（如 SEDD）不直接适用。

## 相关工作与启发
- **vs Self-confidence / Self-entropy planner**：Conf. 选 top-$k$ 高分 token，相邻 token 易被同时选中→大 entropy gap；DUS 强制 spacing，gap 接近 0。
- **vs Path Planning P2 (peng2025)**：P2 用 BERT 作为外部 planner，需要额外模型调用；DUS 不需要任何评分，纯静态调度。
- **vs Fast-dLLM / dKV-Cache**：KV-cache 是正交的内存优化，DUS 是计算调度优化，可叠加。
- **vs Block Diffusion (arriola2025)**：semi-AR 块内仍是 $\mathcal O(B)$ 调用；DUS 是块内的 $\mathcal O(\log B)$ 调度，可视为块内 planner 的最优替代。
- **vs EB/CB sampler (ben-hamu2025/wu2025)**：它们用熵/置信度阈值动态选 $k$ 但不控空间分布；DUS 给它们做 post-filter 能再提点。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 信息论 gap 分析 + dilated 静态调度，整个 framing 非常新
- 实验充分度: ⭐⭐⭐⭐⭐ 5 模型 × 7 benchmark × 4 block sizes + post-filter 验证 + 多个 ablation
- 写作质量: ⭐⭐⭐⭐⭐ 从 motivation 到定理到伪代码到实验组织都极清晰
- 价值: ⭐⭐⭐⭐⭐ MDLM 推理加速的新基线，可直接被任何开源 diffusion LLM 工具链吸收

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)

</div>

<!-- RELATED:END -->
