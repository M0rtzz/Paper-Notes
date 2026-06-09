---
title: >-
  [论文解读] Do Transformers Need Three Projections？三选一/二的 QKV 共享系统研究
description: >-
  [ICML 2026][LLM效率][QKV 共享] 论文系统比较三种 QKV 投影共享方案——Q=K-V（共享 query 和 key）、Q-K=V（共享 key 和 value）、Q=K=V（三者共享），发现 Q-K=V 在 LM 上 PPL 仅升 3.1% 但 KV cache 减 50%…
tags:
  - "ICML 2026"
  - "LLM效率"
  - "QKV 共享"
  - "KV cache"
  - "GQA/MQA"
  - "注意力机制"
  - "weight tying"
---

# Do Transformers Need Three Projections？三选一/二的 QKV 共享系统研究

**会议**: ICML 2026  
**arXiv**: [2606.04032](https://arxiv.org/abs/2606.04032)  
**代码**: https://github.com/anushamadan02/Do-Transformers-Need-3-Projections  
**领域**: LLM 高效推理 / Attention 架构 / KV cache 优化  
**关键词**: QKV 共享, KV cache, GQA/MQA, attention 共享, weight tying

## 一句话总结
论文系统比较三种 QKV 投影共享方案——Q=K-V（共享 query 和 key）、Q-K=V（共享 key 和 value）、Q=K=V（三者共享），发现 Q-K=V 在 LM 上 PPL 仅升 3.1% 但 KV cache 减 50%，与 GQA/MQA 正交可叠加得 87.5%-96.9% cache 减少；为 edge inference 提供 quantifiable memory benefit。

## 研究背景与动机

**领域现状**：Transformer 自 attention 已是 modern AI 标配，但 context window 扩张和 real-time inference 需求让架构效率成焦点。已有工作分两路——sub-quadratic 复杂度（Performer/Linformer）和 head sharing（GQA/MQA 减少 KV 头数）。但 QKV 三投影本身是否必要这个 fundamental question 一直被绕开。

**现有痛点**：CNN 和 State Space Model 用更 unified 内部表示，但 Transformer 一直保持 Q/K/V 三独立投影，是 persistent redundancy；尤其在 PEFT 和 inference cache 紧张时这种冗余直接吃 memory。Multi-head Latent Attention（MLA）压缩 K/V 但仍 functionally independent；其他工作（linear attention、attention-free）替换整个 mechanism 而非保持 attention 的 flexibility。

**核心矛盾**：要 attention 灵活性就要 Q/K/V 三独立；要省 memory 就要削投影；中间需要细粒度系统比较"哪些共享有损/无损"。

**本文目标**：(1) 系统评估 Q=K-V、Q-K=V、Q=K=V 三种共享方案在 synthetic / vision / language 上的表现；(2) 给出 KV cache 节省数字 + 与 GQA/MQA 正交叠加；(3) 提供"哪些共享 OK、为什么"的架构 insight。

**切入角度**：把 weight tying（语言模型 input embedding = output embedding 是经典共享）思想推广到 attention 投影；用 2D positional encoding 解决 Q=K 引入的 symmetric attention problem。

**核心 idea**：Q-K=V（共享 K 和 V）保 attention 不对称性同时减半 KV cache，是 quality-efficiency sweet spot；可与 GQA/MQA 正交叠加。Q=K-V（共享 Q 和 K）让 attention 对称，需 2D pos encoding 救；Q=K=V 最激进但语言任务质量掉得多。

## 方法详解

### 整体框架

论文不提一个新模型，而是把"标准 Transformer 必须有 Q、K、V 三个独立投影"这个被默认了八年的假设拆开做系统对照：固定 attention 主干，只改投影的共享方式，量化每种共享在 synthetic / vision / language 上的质量代价和 KV cache 收益。它枚举三种共享方案——共享 query 和 key 的 **Q=K-V**（$A = \mathrm{Softmax}(\alpha K K^\top) V$，K 复用 Q，但 V 独立，attention 矩阵 $K K^\top$ 变对称）、共享 key 和 value 的 **Q-K=V**（$A = \mathrm{Softmax}(\alpha Q K^\top) K$，V 复用 K，但 Q 独立，attention 仍非对称，cache 只存 K）、以及三者全合一的 **Q=K=V**（$A = \mathrm{Softmax}(\alpha K K^\top) K$，单投影）。对其中引入对称 attention 的两个方案，再配一个带 2D positional encoding 的 (X)+ 变体，把方向信息重新注回去（仅用于 non-causal 任务）。

### 关键设计

**1. Q-K=V：让 V 复用 K，KV cache 直接减半而质量几乎不掉**

LLM serving 的主要 memory 瓶颈是 KV cache，尤其 long context 下 cache 线性涨满显存。Q-K=V 的做法是让 key 和 value 共用同一个投影矩阵（$V = K$ 的 weight tying），于是推理时 cache 里只存 K、不存 V——V 直接从 K 复用，cache 砍掉 $50\%$；同时因为 Q 仍是独立投影，attention 分数 $Q K^\top$ 保持非对称，不破坏 sequential 任务依赖的方向性。这之所以几乎不掉质量，作者给的机制解释是：K 和 V 本就可以占用相似的 representational space，而 attention 实际工作在 low-rank regime，把 K 当 V 用并不会显著压缩有效表达力。实测 300M 模型在 SlimPajama 10B tokens 上 PPL 仅升 $3.1\%$，1.2B 模型叠 MQA 后也只升 $1.06\%$。相比 MLA 把 K/V 压成 compressed latent、推理时还要 expand 回来，Q-K=V 用一个 hard equality 就拿到同量级收益，实现上更简单。

**2. (X)+：用 2D positional encoding 把 Q=K 丢掉的方向性补回来**

Q=K-V 让 query 和 key 共用投影后，attention 矩阵退化成对称的 $K K^\top$，对依赖前后方向的 sequential 任务不利——这正是 symmetric attention 过去只见于 graph NN 和 relational reasoning、被序列任务避开的原因。(X)+ 变体的补救是构造一个固定的 2D sinusoidal positional encoding $P \in \mathbb{R}^{n \times n \times m}$，把对称的 attention map 广播到 $m$ 个 channel 上加进 $P$，再用一个 $1 \times 1$ 卷积投回二维 attention 矩阵，从而在不放弃投影共享的前提下重新引入 directional bias，思路与 relative positional encoding 和 vision Transformer 的 2D pos embedding 一脉相承。要注意 causal LM 本身已被 causal mask 强制成非对称，不需要这个补丁，所以 (X)+ 只用在 vision、synthetic 这类 non-causal 任务上。

**3. 与 GQA/MQA 正交叠加：投影共享和 head 共享是两个维度，收益可乘**

GQA/MQA 走的是另一条省 cache 的路——head sharing，GQA-$g$ 把 $H$ 个 query head 共享到 $g < H$ 个 KV head，cache 减少比例为 $1 - g/H$。Q-K=V 削的是投影维度而非 head 维度，两者互不冲突，可以在每个 GQA group 内部再 enforce $K = V$ 把该 group 的 cache 又减半，于是收益相乘：Q-GQA-4（$H=16, g=4$）总 cache reduction 为 $1 - g/(2H) = 87.5\%$，Q-MQA 更进一步到 $96.9\%$，逼近 cache-based Transformer 的理论极限。由于 MQA/GQA 已是 PaLM/Llama/Mistral 等的标配，Q-K=V 作为 orthogonal complement 对工业部署是直接 actionable 的；Pareto frontier 上 Q-MQA 给出 $97\%$ cache 减少叠 near-parity 质量，对 edge / on-device inference 是真实可落地的 enabler。

下表汇总三种 variant 相对标准 QKV 的计算、参数、cache 代价——Q-K=V 在拿到 $50\%$ cache 减少的同时还顺带省了 $33\%$ 的计算和参数：

| Variant | Computation | Parameters | KV Cache |
|---|---|---|---|
| QKV | $3nd^2$ | $3d^2$ | K + V |
| Q=K-V / Q-K=V | $2nd^2$ | $2d^2$ | K only (Q-K=V) |
| (Q=K-V)+ | $2nd^2 + n^2 m$ | $2d^2 + m$ | K + V |
| Q=K=V | $nd^2$ | $d^2$ | K only |

## 实验关键数据

### Synthetic tasks（5 任务平均）

| 方法 | Reverse | Sort | Sub | Swap | Copy | Avg. |
|------|---------|------|-----|------|------|------|
| QKV | 0.698 | 0.971 | 1.0 | 0.588 | 1.0 | 0.851 |
| Q=K-V | 0.705 | 0.967 | 1.0 | 0.597 | 1.0 | 0.854 |
| (Q=K-V)+ | **0.718** | 0.963 | 1.0 | **0.671** | 1.0 | **0.870** |
| Q-K=V | 0.701 | 0.958 | 1.0 | 0.590 | 1.0 | 0.850 |
| Q=K=V | 0.514 | 0.939 | 1.0 | 0.446 | 1.0 | 0.780 |
| (Q=K=V)+ | 0.581 | 0.957 | 1.0 | 0.576 | 1.0 | 0.823 |

(Q=K-V)+ 用 2D pos encoding 反超 QKV（0.870 vs 0.851），Q=K=V 掉点明显但加 2D pos 后回到 0.823。

### Vision tasks（5 任务平均）

| 方法 | MNIST | FMNIST | CIFAR-10 | CIFAR-100 | TinyImgNet | Anomaly | Avg. |
|------|-------|--------|----------|-----------|------------|---------|------|
| QKV | 0.981 | 0.887 | 0.663 | 0.363 | 0.229 | 0.942 | 0.767 |
| Q=K-V | 0.981 | 0.885 | 0.666 | 0.369 | 0.236 | 0.954 | 0.771 |
| (Q=K-V)+ | 0.982 | 0.884 | 0.662 | 0.366 | - | **0.966** | 0.772 |
| Q-K=V | 0.976 | 0.883 | 0.659 | 0.358 | - | 0.949 | 0.767 |
| Q=K=V | 0.978 | 0.877 | **0.672** | **0.376** | **0.266** | 0.933 | 0.767 |

Vision 上 Q=K=V 在 CIFAR/TinyImageNet 上反超 QKV——证明 symmetric attention 在 non-causal 任务上 perfectly fine。

### Language Modeling（300M 参数，SlimPajama 10B tokens）

| Model | Train Loss | Train PPL | Val Loss | Val PPL | Speed (tok/s) |
|---|---|---|---|---|---|
| QKV (Baseline) | 1.73 | 5.64 | 1.63 | 5.11 | 423k |
| **Q-K=V** | (~1.74) | (~5.70) | (~1.64) | (~5.27) | (~) |

300M 模型 Q-K=V PPL 仅升 3.1% vs QKV，cache 减半；1.2B 模型 MQA 配 1.06% PPL 升 + 97% cache 减。

### 关键发现

- **Q-K=V 是 LM 上 quality-efficiency sweet spot**：3.1% PPL 升换 50% cache 减是 favorable trade，比 MLA 简单。
- **Symmetric attention 在 vision/sets 上 fine**：Q=K=V 在 CIFAR/TinyImageNet 反超 QKV，验证"sequential tasks 需 asymmetry, set/image 不需要"。
- **2D Pos Encoding 救 symmetric in non-causal**：(Q=K-V)+ 在 synthetic Reverse/Swap 任务上超 QKV，证明 lost asymmetry 可由 spatial encoding 补回。
- **Q-K=V 与 GQA/MQA 正交可乘**：Q-GQA-4 = 87.5% cache reduction; Q-MQA = 96.9%，对 on-device LLM 是 enabler。
- **理论 insight**：Q-K=V 工作因为 K 和 V 占相似 representational space 且 attention 在 low-rank regime；Q=K-V 失败因为 breaks attention directionality（symmetric K K^T 不利于 causal LM）。
- **1.2B 上 ranking stable**：scale 起来 relative quality ranking 不变，说明 conclusions generalize 到工业规模。

## 亮点与洞察

- **系统比较填补 fundamental gap**：QKV 三投影必要性是个被绕了 8 年的问题，本文给出 12 任务跨 domain 的系统答案——Q-K=V 是 LM 的 free lunch。
- **Quantifiable cache benefit**：50% cache 直接换算 2× context window 或 2× 并发用户，对 LLM serving 经济性是真实数字。
- **正交叠加性**：projection sharing 和 head sharing 不冲突，Q-MQA 97% cache 给 edge LLM 部署一条新路。
- **Asymmetric vs Symmetric 任务依赖性**：sequential 任务用 Q-K=V（保 asymmetric），non-causal 任务可用 Q=K=V + 2D pos——给"什么时候用什么 variant"清晰决策树。
- **Theoretical explanation 不空洞**：从"K/V 共享 representational space"和"attention low-rank regime"解释为什么 Q-K=V 工作而 Q=K-V 不工作，是 mechanism-level insight 而非黑盒数据。
- **2D Pos Encoding 的优雅 trick**：把 symmetric attention 通过 spatial encoding 重新 asymmetric 是 reusable 设计模式。

## 局限与展望

- **1.2B 是 scale 上限**：实验最大 1.2B，70B+ 模型上 Q-K=V 是否仍 sweet spot 未验证；可能 large model 上 K-V coupling 影响放大。
- **Long context 上验证不足**：所有实验用 ~2K context；Q-K=V 在 100K context 下 cache reduction 价值更大但 attention pattern 是否还 well-behaved 未充分测。
- **MLA 对比不全面**：作者提到 MLA 用 compressed latent 不等价 Q-K=V 的 hard equality，但没在 main results 直接对比 MLA vs Q-K=V 的 quality-efficiency Pareto。
- **(X)+ 变体只在 non-causal**：causal LM 已 enforce asymmetry，(X)+ 不适用；但其实 prefix/non-causal blocks（如 LM 的 system prompt prefix）能否用 (X)+ 加速没探索。
- **没考虑 SwiGLU/RMSNorm 等现代架构差异**：实验用 standard transformer，但现代 LLM（Llama/Mistral）有 SwiGLU/RMSNorm/RoPE 等改动，Q-K=V 在这些 setting 下效果未验证。
- **训练稳定性没大规模测**：300M/1.2B 训练稳定，但 7B/30B 训练是否会出现 K-V coupling 引起的训练不稳没数据。

## 相关工作与启发

- **vs MLA (DeepSeek-V2)**：MLA 把 K/V 压成 latent vector，cache 小但 inference 时 expand 回；Q-K=V 用 hard equality 直接 cache K 不需 expand，更简单但表达力略弱。
- **vs GQA / MQA (Ainslie 2023, Shazeer 2019)**：head sharing 是 orthogonal axis；本文证明 Q-K=V + GQA/MQA 可叠加。
- **vs Linear Attention / Performer / Linformer**：sub-quadratic 复杂度降低，但 sacrifice attention flexibility；Q-K=V 保 attention mechanism 同时减 cache。
- **vs Weight Tying (Press & Wolf 2017)**：input/output embedding 共享是经典 weight tying；Q-K=V 把它推广到 attention 投影。
- **vs Borji 2023 (first author 先前工作)**：本文是其 follow-up + scale up；Kowsher 2025 后续相似 idea，本文 timestamp 更早。
- **启发**：(1) Transformer 架构里还有未被探索的 weight tying 维度，应继续 systematic study；(2) projection sharing 和 head sharing 是 orthogonal 应组合用；(3) symmetric vs asymmetric attention 应基于任务 modality（sequential vs non-causal）做架构决策。

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性问题 + 解释 mechanism 是方法学贡献；single new architecture trick 较少但组合应用价值高。
- 实验充分度: ⭐⭐⭐⭐ 12 任务跨 3 domain + 300M/1.2B 双 scale + GQA/MQA 正交叠加分析；缺 7B+ scale 和 long-context 验证。
- 写作质量: ⭐⭐⭐⭐⭐ Pareto frontier 清晰、taxonomy 整齐、theoretical explanation + when-to-use 决策树都给出，工程师 readable。
- 价值: ⭐⭐⭐⭐⭐ 直接服务于 LLM 部署 KV cache 紧张痛点（50%-97% reduction），与 industry 标准 GQA/MQA 兼容，开源代码 + 直接可用 recipe。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Tensor Product Attention Is All You Need](../../NeurIPS2025/llm_efficiency/tensor_product_attention_is_all_you_need.md)
- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](../../ICLR2026/llm_efficiency/universe_routing_why_self-evolving_agents_need_epistemic_control.md)
- [\[ICLR 2026\] Efficient Resource-Constrained Training of Transformers via Subspace Optimization](../../ICLR2026/llm_efficiency/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)
- [\[NeurIPS 2025\] Constant Bit-Size Transformers Are Turing Complete](../../NeurIPS2025/llm_efficiency/constant_bit-size_transformers_are_turing_complete.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](../../NeurIPS2025/llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)

</div>

<!-- RELATED:END -->
