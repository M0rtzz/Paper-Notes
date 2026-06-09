---
title: >-
  [论文解读] LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws
description: >-
  [ICML 2026][模型压缩][Shannon 容量] 本文把 LLM 训练重新解释为 Shannon-Hartley 噪声信道——参数量对应带宽、训练 token 对应信号功率、数据/模型噪声对应信道噪声…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "Shannon 容量"
  - "扩展定律"
  - "U 形损失"
  - "灾难性过训练"
  - "量化退化"
---

# LLMs as Noisy Channels: A Shannon Perspective on Model Capacity and Scaling Laws

**会议**: ICML 2026  
**arXiv**: [2605.23901](https://arxiv.org/abs/2605.23901)  
**代码**: 待确认  
**领域**: 模型压缩 / 扩展定律 / 信息论  
**关键词**: Shannon 容量, 扩展定律, U 形损失, 灾难性过训练, 量化退化

## 一句话总结
本文把 LLM 训练重新解释为 Shannon-Hartley 噪声信道——参数量对应带宽、训练 token 对应信号功率、数据/模型噪声对应信道噪声；从该框架推出 Shannon Scaling Law $C_{\text{LLM}} = aN^\alpha \log_2(1 + bD^\beta / (c(DN)^\gamma + dD^\delta + e))$，能统一解释经典单调 scaling 与近期发现的 U 形退化（catastrophic overtraining、quantization-induced degradation），并在 Pythia/OLMo2 上从 ≤6.9B 数据外推到 12B 模型 307B token 上 $R^2 = 0.847$。

## 研究背景与动机

**领域现状**：LLM 发展靠 scaling law 指导——OpenAI law 用 $L = [(N_c/N)^{\alpha_N/\alpha_D} + D_c/D]^{\alpha_D}$，Chinchilla law 用加法形式 $L = A/N^\alpha + B/D^\beta + E$，都假设性能随计算单调提升。这驱动了万亿参数模型（DeepSeek-V4 1.6T、Kimi K2.6 1T）和海量预训练数据。

**现有痛点**：近期一系列发现挑战单调假设：
- **Catastrophic Overtraining**（Springer 2025）：过量预训练反而损害下游 SFT
- **Quantization-induced Degradation (QiD)**（Kumar 2024、Ouyang 2024）：更大/更长训练的模型对量化反而更敏感
这些产生 **U 形 loss 曲线**——先降后升。已有的 perturbation-aware scaling law（Ouyang 2024、Kumar 2024）只是给基础 law 加退化项 $\Delta_q L$ 或 $\delta_{\text{PTQ}}$，缺乏统一理论。

**核心矛盾**：单调 power law 假设信号无限提升，但实际 LLM 训练同时存在信号（学到知识）和噪声（数据噪、模型噪、量化扰动等）；当 noise 增长跟不上 signal 时改善持续，但超临界点 noise 反占主导 → U 形。需要一个能同时表达 signal 与 noise 互动的统一框架。

**本文目标**：（1）建立一个统一 scaling law 既能描述单调收益（高 SNR 区）也能描述 U 形退化（低 SNR 区）；（2）从信息论第一性原理推导而非凑形式；（3）能外推到训练分布外的更大模型/更长训练。

**切入角度**：把 LLM 训练类比 Shannon 通信——pretraining = channel modulation（把信息调进权重）、inference = transmission（从 context 传到 output）。Shannon-Hartley 定理 $C = B \log_2(1 + S/\mathcal{N})$ 天然描述"信号 vs 噪声"的容量上界；当 $S/\mathcal{N}$ 高时容量近似 $B \log_2(S/\mathcal{N})$ 单调升，当噪声大时容量被压缩——恰好对应 U 形。

**核心 idea**：参数 $N$ → 带宽 $B$、训练 token $D$ → 信号功率 $S$、数据噪声 + 模型噪声 + 扰动 → 噪声 $\mathcal{N}$；得 Shannon Scaling Law $C_{\text{LLM}} = aN^\alpha \log_2(1 + bD^\beta / (c(DN)^\gamma + dD^\delta + e))$，统一覆盖单调和 U 形两种行为。

## 方法详解

### 整体框架

本文把一次 LLM 训练当成一条 Shannon-Hartley 噪声信道来看：参数量 $N$ 是信道带宽，训练 token $D$ 是注入的信号功率，数据与模型自身的各种噪声合成信道噪声 $\mathcal{N}$，而模型最终能容纳多少知识就是信道容量 $C_{\text{LLM}}$。具体映射是带宽 $B_{\text{LLM}}\propto N^\alpha$、信号 $S_{\text{LLM}}\propto D^\beta$、噪声拆成数据噪 $dD^\delta$ + 模型交互噪 $c(DN)^\gamma$ + 不可约扰动 $e$ 三源，套进 $C=B\log_2(1+S/\mathcal{N})$ 就得到全篇的核心公式：

$$C_{\text{LLM}} = aN^\alpha \log_2\left(1 + \frac{bD^\beta}{c(DN)^\gamma + dD^\delta + e}\right)$$

它既要在高信噪比区还原出经典单调 scaling，又要在噪声反超信号时自然长出 U 形退化。

### 关键设计

**1. 三源噪声分解：让一个 $\mathcal{N}$ 同时解释过训练和量化两种退化**

经典 scaling law 只有"信号"没有"噪声"，所以一旦出现先降后升的 U 形就束手无策。本文把信道噪声 $\mathcal{N}=c(DN)^\gamma + dD^\delta + e$ 拆成三层，每层对准一种真实的退化机制。数据噪 $dD^\delta$ 来自训练语料本身的 typo、歧义、自相矛盾，token 喂得越多累积越多，正好解释 catastrophic overtraining——过量预训练把这部分噪越堆越高，最终压过信号收益。模型交互噪 $c(DN)^\gamma$ 把训练看成一个 denoising 过程中模型与数据相互作用产生的拟合噪，关键是 $DN$ 用联乘而非相加，刻画"大模型 × 大数据"的耦合，于是大模型在大数据上对量化更脆弱（QiD）这件反直觉的事就有了出处。最后 $e$ 是硬件/算法层不可约的固有扰动。三层拆开的好处是 fit 完之后 $c,d,e$ 各有物理含义，能定量回答某次失败训练究竟是数据噪主导还是模型噪主导，而单一 noise 项做不到这种归因。

**2. U 形与单调统一：单调收益和退化是同一公式在两个 SNR 区的极限**

以往的 perturbation-aware law（Ouyang、Kumar）做法是"基础 law + 退化项加和"，本质是把两段曲线硬拼起来，接缝处缺乏理论依据。本文不打补丁，全靠 $\log_2(1+S/\mathcal{N})$ 这一个函数的形状说话：当信号远大于噪声 $S\gg\mathcal{N}$ 时，它退化成 $\log_2(S/\mathcal{N})$，容量随 $D$ 单调上升，复现经典 power law；而当噪声增长追上甚至反超信号——比如数据噪 $dD^\delta$ 这一项随 token 主导起来——容量转头下降，U 形不是额外加项凑出来的，而是同一公式在低信噪比极限下自己长出来的。单调与退化共用一套参数、同一信息论原理，这种统一正是它外推力更强的根源。

**3. 多源扰动的统一编码：所有扰动都落到噪声项上，只换参数不换公式**

加性高斯噪声、量化、不同领域的 SFT，看似互不相干，本文把它们一律编码成对噪声项 $\mathcal{N}$ 的影响：高斯噪声直接抬高 $e$ 或 $d$，量化等效于压低有效 SNR，而 SFT 在 OOD 数据上引入一项新的交互噪 $c'(DN)^{\gamma'}$。这样面对一种新扰动不必像经典 scaling law 那样重新设计一套退化建模、让参数量级膨胀，只需在同一公式里重新拟合几个参数即可覆盖，框架的通用性也因此体现在后面六类 perturbation 场景全部能 fit 上。

## 实验关键数据

### 主实验：拟合质量（$R^2$）

| 实验设置 | OpenAI law | Chinchilla | Perturbation-aware (Ouyang) | **Shannon** |
|--------|------|---------|---------|-------|
| Pythia 标准预训练 | 0.91 | 0.93 | 0.93 | **0.95** |
| Pythia + Gaussian 噪声 | 0.42 | 0.45 | 0.78 | **0.91** |
| Pythia + 量化 | 0.38 | 0.41 | 0.82 | **0.93** |
| OLMo2 + math SFT | 0.31 | 0.34 | 0.67 | **0.89** |
| OLMo2 + QA SFT | 0.28 | 0.30 | 0.69 | **0.88** |
| OLMo2 + code SFT | 0.35 | 0.37 | 0.71 | **0.90** |

Shannon law 在所有 perturbation 场景下都大幅领先，特别是经典 law 完全崩溃的 SFT 场景。

### 外推能力（关键测试）

| 训练数据 | 测试模型 | 测试 token 数 | OpenAI $R^2$ | Chinchilla $R^2$ | **Shannon $R^2$** |
|--------|------|----------|---------|----------|---------|
| Pythia ≤6.9B, ≤180B token | Pythia 12B | 180B | 0.65 | 0.71 | **0.93** |
| Pythia ≤6.9B, ≤180B token | Pythia 12B | 240B | 0.42 | 0.48 | **0.89** |
| Pythia ≤6.9B, ≤180B token | Pythia 12B | 307B | **−0.18** | **−0.05** | **0.847** |

经典 law 在 307B token 外推上 $R^2$ 变负（fit 比常数还差），Shannon law 保持 0.847——真正能外推到训练分布外。

### U 形捕捉

在 Pythia 量化实验里，量化 bit-width 从 4 降到 2 过程中 loss 先稳后突增，OpenAI/Chinchilla 完全无法捕捉这个临界点，Ouyang perturbation-aware 部分能但偏移，Shannon law 精准捕捉 loss basin 位置。

### 关键发现
- **统一 framing 比拼接公式更强**：Shannon law 不只是"另一个 perturbation-aware law"，而是从 first principle 推出，单调和 U 形是同一公式两个极限——这种深度统一带来更好的外推性
- **三源噪声分解可解释**：fit 出的 $c, d, e$ 让用户能定量分析各种退化来源的贡献（如某次失败训练是数据噪主导还是模型噪主导）
- **外推到 12B / 307B 不崩**：训练只到 6.9B / 180B token 仍能预测远超规模的实验，证明 Shannon law 抓到的是底层规律而非曲线拟合 trick
- **跨扰动类型一致**：Gaussian / 量化 / 三种 SFT 都涨，说明 framework 通用

## 亮点与洞察
- **信息论统一 scaling 与退化是真正深刻的视角**：以往 scaling law 是"凑形式拟合"，Shannon law 是"从通信理论推出"；这种 first-principle 思路给后续 scaling 研究提供模板
- **三源噪声分解的可解释性**：分别对应数据质量、模型-数据交互、不可约项，给训练失败的诊断提供量化手段
- **外推 $R^2 = 0.847$ 的工程意义**：scaling law 的最大价值是预测——本文外推到训练分布外仍准，说明可以用小模型实验预测大模型表现，节省海量算力
- **类比 Shannon-Weaver model**：把 Inference 也看作 channel transmission（从 context 传到 output），这种 framing 让 inference 优化（如 KV cache、speculative decoding）也可纳入同一框架分析

## 局限性 / 可改进方向
- 仍是经验拟合 law——9 个参数（$a, b, c, d, e, \alpha, \beta, \gamma, \delta$）需要数据估计；理论上可否进一步从 channel coding 理论推出参数值得探索
- 仅在 Pythia / OLMo2 两个模型族验证；闭源模型（GPT / Claude）的拟合质量未测
- 没把架构差异（如 attention head 数、layer width）显式建模，只用总参数 $N$
- 推理时的 capacity（context length、KV cache 等）未纳入；推理 scaling law 是个独立有趣方向
- Shannon-Hartley 假设 AWGN，LLM 训练 noise 实际分布更复杂；可考虑 generalized capacity 公式

## 相关工作与启发
- **vs OpenAI / Chinchilla**：那些假设单调，被 U 形现象推翻；Shannon law 是它们的严格超集（高 SNR 区 reduce 到 power law）
- **vs Ouyang QiD law / Kumar precision law**：那些给基础 law 加退化项 patch，不统一；Shannon law 从同一公式自然涌现
- **vs Information Bottleneck theory（Tishby）**：那个用互信息分析 DNN 表示；本文用信道容量分析 LLM 训练，是平行视角
- **启发**：把任何"经验拟合的 scaling law"放到信息论框架下重审，可能发现统一原理；这套"first principle scaling"思路对 RL / multi-modal / agent 领域都值得尝试

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统用 Shannon-Hartley 统一 LLM scaling 与退化，理论深度独到
- 实验充分度: ⭐⭐⭐⭐⭐ 6 类 perturbation 场景拟合对比 + 外推到训练外，证据扎实
- 写作质量: ⭐⭐⭐⭐⭐ 信息论类比清晰，Figure 3 通信结构对照直观；三源噪声分解解释力强
- 价值: ⭐⭐⭐⭐⭐ scaling law 是 LLM 研究的基础工具；统一 + 外推性提供更可靠的规划依据，省算力意义重大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ACL 2026\] Task-Stratified Knowledge Scaling Laws for Post-Training Quantized LLMs](../../ACL2026/model_compression/task-stratified_knowledge_scaling_laws_for_post-training_quantized_large_languag.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICML 2026\] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment](mic_maximizing_informational_capacity_in_adaptive_representations_via_isotropic_.md)
- [\[ICLR 2026\] A universal compression theory for lottery ticket hypothesis and neural scaling laws](../../ICLR2026/model_compression/a_universal_compression_theory_for_lottery_ticket_hypothesis_and_neural_scaling_.md)

</div>

<!-- RELATED:END -->
