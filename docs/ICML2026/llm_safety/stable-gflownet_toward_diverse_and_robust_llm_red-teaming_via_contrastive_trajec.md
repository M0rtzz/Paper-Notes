---
title: >-
  [论文解读] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance
description: >-
  [ICML 2026][LLM安全][红队攻击] 本文指出现有 GFlowNet 红队的两大不稳定来源——partition function $Z_\theta$ 估计带来的高方差…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "红队攻击"
  - "GFlowNet"
  - "Trajectory Balance"
  - "对比目标"
  - "噪声梯度剪枝"
---

# Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

**会议**: ICML 2026  
**arXiv**: [2605.00553](https://arxiv.org/abs/2605.00553)  
**代码**: 论文未公开链接  
**领域**: LLM 安全 / Red-Teaming / GFlowNet  
**关键词**: 红队攻击, GFlowNet, Trajectory Balance, 对比目标, 噪声梯度剪枝

## 一句话总结
本文指出现有 GFlowNet 红队的两大不稳定来源——partition function $Z_\theta$ 估计带来的高方差，与 toxicity classifier 给 OOD gibberish 文本的噪声 reward 引发的 mode collapse——并用三件简单组件（pairwise 对比目标 CTB 消除 $Z$、Noisy Gradient Pruning 过滤无信息 pair、Min-K Fluency Stabilizer 卡掉 gibberish）让红队攻击在 Qwen2.5-1.5B 上独特攻击数从 17 飙到 134（约 7×），ASR 维持 92%，且跨模型/跨防御迁移性全面碾压 baseline。

## 研究背景与动机

**领域现状**：LLM 红队攻击在 deployment 前找出 safety 漏洞，分三派：(1) RL-based（PPO、PPO+Curiosity、Jailbreak-R1）追 reward 最大化，能找高毒性 prompt 但 mode collapse 严重；(2) Quality-Diversity（Rainbow Teaming、Ruby Teaming）靠预定义 style/topic 矩阵 + 进化策略保多样性，但依赖 frozen LLM 的 instruction following，攻击成功率低；(3) GFN-based（Lee et al. 2024）把红队看成分布匹配——sample 概率 $\propto$ reward，理论上可同时拿到 high toxicity 和 diversity。

**现有痛点**：直接把 Trajectory Balance（TB）这种 GFN 目标搬到 LLM 上有两大坑：
- TB loss $\mathcal{L}_{TB}(y; \theta) = (\log Z_\theta + \log \pi_\theta(y) - \log R(y))^2$ 需要学一个标量 $Z_\theta$ 估计 $Z \simeq \sum_{y \in \mathcal{Y}} R(y)$。LLM 的 token 序列空间 $\mathcal{Y}$ 组合爆炸，$Z_\theta$ 难以准确估计，导致梯度高方差、训练崩或仍 mode collapse。
- 红队 reward 来自 toxicity classifier，对 gibberish-like OOD 文本会随机给 0.2~0.3 的伪 reward；attacker 一旦发现这种 reward hacking 路径，会迅速 collapse 到生成 gibberish 的局部最优。

**核心矛盾**：GFN 的 lossless distribution matching 性质本应保多样性，但 $Z$ 估计的实践不稳让 TB 退化成接近 RL 的窄分布拟合；而保护 fluency 的标准方法 KL-divergence 正则 $R_{ref}(y) = \pi_{KL}(y)^\alpha \cdot R(y)^\beta$ 又会扭曲目标分布（让 sampled 分布偏向 reference 而非 reward），与 GFN 的理论假设冲突。

**本文目标**：(1) 设计一个不需要 $Z_\theta$ 的 GFN 替代目标，最优解仍等价于 TB；(2) 给噪声 reward 一个 saliency-based 过滤策略，避免被随机伪 reward 污染；(3) 防止 attacker hack 到 gibberish 区域，但不能用 KL 那种扭曲目标分布的做法。

**切入角度**：作者注意到，如果对同一个 policy 取两条轨迹 $y_1, y_2$ 做 ratio 对比，partition function $Z_\theta$ 会自然抵消——这就是 contrastive 目标的标准动机。同时，"reward 噪声"问题本质是 pair-wise 比较时低对比度 pair 提供错误梯度信号，所以可以用一个 contrast-aware 的 indicator 当 hard filter。"gibberish 反复出现"则可用 Min-K probability（least-likely tokens 的平均 log-prob）作为流畅度 proxy，给一个 hard threshold。

**核心 idea**：用 ratio 形式的 Contrastive Trajectory Balance (CTB) 消 $Z_\theta$ + Noisy Gradient Pruning (NGP) 按 reward 对比度过滤 pair + Min-K Fluency Stabilizer (MKS) 卡 gibberish，三件套合成 Stable-GFN。

## 方法详解

### 整体框架

输入：attacker LLM $\pi_\theta$（Qwen2.5-1.5B SFT 过 Safety-Dataset + AdvBench）、victim LLM $\pi_\phi$、toxicity classifier $\pi_\psi$、fixed meta-prompt。每个训练 step：(1) attacker 用当前 policy 采 $N$ 条候选 attack prompts $\{y_n\}$；(2) victim 对每条给 response $z_n$，classifier 算 toxicity $R(y_n) = \mathbb{E}_{z \sim \pi_\phi(\cdot|y)}[T(y, z)]$；(3) MKS 用 reference model 算每条 prompt 的 Min-K 流畅度，低于阈值的直接 mask；(4) NGP 在 batch 内枚举 $N^2$ 个 pair，过滤 $|\log R(y_1) - \log R(y_2)| \le \sigma$ 的低 saliency pair；(5) 剩下 pair 算 CTB loss 更新 $\theta$。整个 pipeline 不再有外部参数 $Z_\theta$、不维护 archive、不需要 reference policy 强约束。

### 关键设计

1. **Contrastive Trajectory Balance (CTB)**：

    - 功能：用 pair-wise 比较替代 absolute matching，从公式上消掉 $Z_\theta$，得到与 TB 同等最优策略但低方差的目标。
    - 核心思路：对一对独立采样 $y_1, y_2 \sim \pi_\theta$，定义 $\mathcal{L}_{CTB}(y_1, y_2; \theta) = (\log \tfrac{\pi_\theta(y_1)}{\pi_\theta(y_2)} - \log \tfrac{R(y_1)}{R(y_2)})^2$。令 $f(y) = \log \pi_\theta(y) - \log R(y)$，当 $y_1, y_2$ 是 i.i.d. 取样时，目标等价于 $2 \cdot \mathrm{Var}_{\pi_\theta}(f(y))$，最小化到 0 等价于 $f$ 在 support 上恒为常数 $C$，进而（结合归一化条件）$\pi_\theta(y) = R(y)/Z$——即回到 TB 的最优解（Theorem 4.1）。梯度 $\nabla_\theta \mathcal{L}_{CTB} = 2(f(y_1) - f(y_2))(\nabla_\theta f(y_1) - \nabla_\theta f(y_2))$ 中，每个样本被另一个样本的 log-flow error 当作 stochastic baseline，与 RLOO/Williams 的 variance reduction 同构。
    - 设计动机：从根上抹掉 $Z_\theta$ 这个高方差源；同时 batch 内 $N$ 条样本可以枚举 $N^2$ 个标量 pair-wise loss（无需额外 forward），训练复杂度仍是 $O(N)$ 前后向。

2. **Noisy Gradient Pruning (NGP)**：

    - 功能：CTB 把两个样本的 reward noise 加在一起，低对比度 pair 反而放大噪声；NGP 用 hard mask 把低 saliency pair 的梯度清零。
    - 核心思路：$\mathcal{L}_{NGP}(y_1, y_2; \theta) = \mathbb{1}[|\log R(y_1) - \log R(y_2)| > \sigma] \cdot \mathcal{L}_{CTB}(y_1, y_2; \theta)$。$\sigma$ 是 saliency threshold 超参。理论上构造 saliency graph $G_\sigma = (\mathcal{Y}, E_\sigma)$（边定义为对比度 > $\sigma$ 的样本对），若 $G_\sigma$ 连通，则 $\mathcal{L}_{NGP}(\theta) = 0$ 仍等价于 $\pi_\theta(y) \propto R(y)$（Proposition 4.2）。实践中作者用 high-reward replay buffer 做"global anchors"，提供跨高/低 reward 区的对比 pair 保持连通。
    - 设计动机：toxicity classifier 对相近 reward 的样本之间是随机噪声主导，过 filter 掉这些"信息量为零但 noise 不为零"的 pair，让梯度只从有真实 reward 差异的对里来；既保 GFN 的目标性质（在连通假设下），又显著降梯度方差。

3. **Min-K Fluency Stabilizer (MKS)**：

    - 功能：阻止 attacker hack 到 gibberish 区域，但不扭曲目标分布。
    - 核心思路：用 reference model $\pi_{ref}$ 对生成 prompt $y$ 算每个 token 的 log-prob，取最低 $k$ 个 token 的平均：$M_k(y) = \tfrac{1}{|K|}\sum_{w \in K} \log \pi_{ref}(y_w | y_{<w})$。然后 reward 改为 $R_{MKS}(y) = \mathbb{1}[M_k(y) \ge T_{MKS}] \cdot R(y)$——低于流畅度阈值 $T_{MKS}$ 的直接 reward 清零。$\pi_{ref}$ 的梯度不参与 reward 计算。
    - 设计动机：与 KL 全局正则不同，MKS 只惩罚"最不流畅段"的样本（最容易暴露 OOD gibberish），保留正常 prompts 探索的自由度；不修改 target distribution 的形状（reward 内 hard cutoff 而非 reshape），与 GFN 的 distribution matching 假设兼容。

### 损失函数 / 训练策略

总目标 $J_{CTB}(\theta) = \mathbb{E}_{y_1, y_2 \sim \pi_\theta}[\mathcal{L}_{NGP}(y_1, y_2; \theta)]$，外面套 MKS 修改的 reward。Batch 内 $N = 1024$ 条样本枚举 pair。Attacker：Qwen2.5-1.5B SFT；Victim：Qwen2.5-1.5B-Instruct；Toxic classifier：Meta-Llama-Guard-3-8B；Diversity：all-MiniLM-L6-v2 + greedy clustering threshold 0.7；reward >0.5 算 ASR。

## 实验关键数据

### 主实验

| 方法 | UA (#) | ASR (%) | 备注 |
|------|--------|---------|------|
| PPO | 3.00 | **91.70** | 高 ASR 但极度 mode collapse |
| PPO + Curiosity | 4.00 | 36.75 | 仍 collapse |
| Rainbow Teaming | 33.00 | 66.11 | QD 多样性高但 ASR 低 |
| Jailbreak R1 (8B) | 75.33 | 7.36 | 多样但毒性低 |
| GFN (TB) | 17.67 | 93.75 | 高 ASR 但 UA 远低于理论预期 |
| **S-GFN (Ours)** | **134.00** | 92.55 | 同档 ASR、UA 提升 7× |

Cross-Attack 防御传递（在 GFN-defended victim 上仍能攻）：

| Attack 方 | GFN-defended victim ASR | 说明 |
|----------|---------|------|
| GFN | 4.69% | 自家攻击被自家防御挡住 |
| Jailbreak R1 | 2.96% | – |
| **S-GFN** | **22.53%** | 攻击模式更广，跨防御迁移强 |

### 消融实验

| 配置 | UA (#) | ASR (%) | 说明 |
|------|--------|---------|------|
| GFN-TB + KL ref | 14 | – | reference KL 扭曲分布 |
| GFN-TB + LogProb | 65 | – | 替代正则 |
| GFN-TB + MKS | 67 | 85.8 | TB + 流畅度卡口 |
| **GFN-CTB + MKS** | 108 | 82.9 | 加 CTB 后 UA +60% |
| **GFN-CTB + MKS + NGP** | **121** | **92.2** | 完整 S-GFN，ASR 也回升 |

### 关键发现

- **CTB > TB 的核心贡献是稳定**：单独把 TB 换成 CTB（保持 MKS）就把 UA 从 67 提到 108，证明 $Z_\theta$ 估计是 mode collapse 的主因之一。
- **NGP 同时提升 UA 与 ASR**：从 108 到 121 UA、82.9% 到 92.2% ASR，说明过滤低 saliency pair 既降噪又让梯度信号更强，"少而精"胜过"多而噪"。
- **Cross-Attack 不对称性极显著**：S-GFN 攻 GFN-defended 模型 22.53%，反过来 GFN 攻 S-GFN-defended 模型仅 0.03%——这种"我能破你你破不了我"的不对称说明 S-GFN 找到的攻击模式真正多样而非 GFN 攻击的超集换皮。
- **Transfer attack 到完全 unseen victim** (Gemma3, Llama3.2, Qwen3, GPT-OSS-20B) 上 S-GFN 在所有模型上 UA 和 ASR 同时拿第一，说明这些攻击不是过拟合到训练 victim 的"特定 jailbreak"。
- **MKS 的必要性**：没有 MKS 时 reward 为 0（全 hack 到 gibberish），加上后 UA 立刻从 0 跳到 67——直接挽救了整个训练过程。

## 亮点与洞察

- "$Z_\theta$ 估计抵消"是一个看起来很简单但意义重大的洞察——文本侧 GFN 长期被 $Z$ 估计困扰，CTB 直接用 ratio 形式让 $Z$ 自然消失，类似于 contrastive learning 把 normalizing constant 消掉。等价性证明（Theorem 4.1）确保不牺牲分布匹配的理论性质。
- NGP 的"saliency graph 连通性"分析很优雅——它把"我能 prune 多少 pair 还保留 GFN 收敛性"形式化成图连通性条件，并指出 replay buffer 实际起 anchor 作用。
- MKS 用 Min-K probability（来自 LLM membership inference 文献）做 fluency 检测是巧妙的跨域借用——比传统 perplexity 更聚焦"最弱环节"，因此对 partial gibberish 的检测更敏感。
- 整套方法的实现复杂度极低：CTB 是 $N^2$ 标量操作，NGP 是 indicator mask，MKS 是 reward cutoff——三件都是"加一个 hard filter / 改 loss 写法"，完全不增 forward 次数。

## 局限与展望

- $\sigma$（NGP）和 $T_{MKS}$（MKS）是固定超参，没探索 task-adaptive 自适应。reward 分布在训练中变化，固定阈值可能在不同 stage 表现不同。
- 连通性假设在分布 mode 数量很多时未必成立；作者承认实际中 "high-reward replay buffer" 是经验性 anchor，没给非渐近收敛速度界。
- 只在 Qwen2.5-1.5B attacker 上做主实验，没探索 attacker scaling（如 7B/13B），更大 attacker 是否仍能维持 CTB 的方差减少效果存疑。
- 与 multi-stage iterative GFN（Yun et al. 2025）的组合没探索；CTB 是否能融入 iterative 框架进一步提升 diversity 是自然的下一步。
- 红队伦理问题：方法越好越能找漏洞，但论文未深入讨论 disclosure 流程；ASR 92%、UA 134 这种结果对开源 victim 模型有直接风险，需 responsible release。

## 相关工作与启发

- **vs GFN-TB (Lee et al. 2024)**：原始 TB 把 $Z_\theta$ 当可学参数，方差大导致 mode collapse；CTB 用 pairwise ratio 抹掉 $Z$，最优策略等价但训练稳。
- **vs PPO + Curiosity (Hong et al. 2024)**：RL + 多样性 reward 项，仍是单点 reward 优化，UA 只到 4；S-GFN 是分布匹配派，UA 跳到 134。
- **vs Rainbow Teaming (Samvelyan et al. 2024)**：QD 用预定义 style/topic 矩阵硬保多样性，但 ASR 低（66%）。S-GFN 不需要预定义 archive，端到端用 reward 信号自动找多样模式。
- **vs DPO with replay**：DPO 在红队上 UA 仅 5.33；其偏好对比目标与 CTB 表面相似但 DPO 优化 preference 排序而非 distribution matching，目标性质不同。
- **vs DB / SubTB (Bengio et al. 2023; Madan et al. 2023)**：DB / SubTB 避开 $Z$ 估计但 token-level 计算昂贵难以 scale 到 LLM；CTB 只在序列级做 pair-wise，计算友好。

## 评分
- 新颖性: ⭐⭐⭐⭐ pairwise contrastive 消 $Z$ 的思路虽借鉴自 contrastive learning，但首次系统应用到 LLM-scale GFN 并配套噪声/流畅度处理；CTB-TB 等价证明扎实。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 5 个 baseline、cross-attack defense、4 个 transfer victim、3 个消融模块，量化清晰；缺 attacker scaling 实验。
- 写作质量: ⭐⭐⭐⭐ 动机-理论-算法-实验对应清楚，每条命题都有 appendix 证明位置；图 1 综述图直观。
- 价值: ⭐⭐⭐⭐ 把 GFN 实际推到 LLM red-teaming 可用水平，并给出可推广的"稳定 GFN"工具箱，对 alignment 安全社区有用，但对开源 victim 风险需注意。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)
- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ICML 2026\] STARE: Step-wise Temporal Alignment and Red-teaming Engine for Multi-modal Toxicity Attack](stare_step-wise_temporal_alignment_and_red-teaming_engine_for_multi-modal_toxici.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](../../ICLR2026/llm_safety/tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)

</div>

<!-- RELATED:END -->
