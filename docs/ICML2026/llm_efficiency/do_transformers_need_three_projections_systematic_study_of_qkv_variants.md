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

三个 variant：(1) **Q=K-V**: $A = \mathrm{Softmax}(\alpha K K^\top) V$，K=Q 但 V 独立——attention 矩阵 $K K^\top$ symmetric；(2) **Q-K=V**: $A = \mathrm{Softmax}(\alpha Q K^\top) K$，K=V 但 Q 独立——attention 仍 asymmetric，cache 只存 K；(3) **Q=K=V**: $A = \mathrm{Softmax}(\alpha K K^\top) K$，单投影。后两 variant 配 2D positional encoding 的 (X)+ 变体让 attention 重新 asymmetric（只用于 non-causal task）。

### 关键设计

1. **Q-K=V 是 LM 的 sweet spot**:

    - 功能：减半 KV cache、保 attention 不对称、quality 几乎不掉。
    - 核心思路：$V = K$ 单一投影矩阵实现 weight tying；inference 时 cache 只存 K 不存 V（V 从 K 复用），$50\%$ cache reduction。Attention 仍 $Q K^\top$ asymmetric，因 Q 独立。Theoretical insight：因为 K 和 V 可以占用相似 representational space 且 attention 在 low-rank regime 下工作，K 当 V 用 quality 不会大掉。300M 模型在 SlimPajama 10B tokens 上 PPL 仅升 $3.1\%$，1.2B 模型 MQA 配 +1.06%。
    - 设计动机：cache 是 LLM serving 的主要 memory bottleneck（特别是 long context）；Q-K=V 给出 $50\%$ 减少 + 几乎无质量损失的 elegant 方案，比 MLA 简单（hard equality 替代 compressed latent expansion）。

2. **Q=K-V 用 2D Positional Encoding 救 symmetric attention**:

    - 功能：Q=K 时 attention $K K^\top$ symmetric 不利于 sequential task；2D pos encoding 注入 directional bias。
    - 核心思路：构造 fixed 2D sinusoidal positional encoding $P \in \mathbb{R}^{n \times n \times m}$，把 attention map 广播到 channel 加 $P$，再 $1 \times 1$ conv 投回 2D attention matrix。这跟 relative positional encoding 和 vision Transformer 的 2D pos embedding 思想一致。Causal LM 已通过 causal mask enforce asymmetry，(X)+ 只用于 non-causal task（vision、synthetic）。
    - 设计动机：以前 symmetric attention 多在 graph NN 和 relational reasoning 出现，sequential task 因 directionality 重要被避开；2D pos encoding 是 elegant 的"打破对称同时保 efficiency"方案。

3. **与 GQA/MQA 正交叠加**:

    - 功能：projection sharing 和 head sharing 是不同维度的 efficiency，组合可乘性减 cache。
    - 核心思路：GQA-$g$ 是把 $H$ query heads 共享 $g < H$ KV heads，减 cache 比 $1 - g/H$；Q-K=V 进一步在每个 GQA group 内 enforce $K = V$，cache 减半。Q-GQA-4 总 cache reduction $1 - g/(2H) = 87.5\%$（$H=16, g=4$）；Q-MQA cache reduction $96.9\%$，接近 cache-based Transformer 理论极限。
    - 设计动机：MQA/GQA 已被 PaLM/Llama/Mistral 等广泛部署；本文证明 Q-K=V 是 orthogonal complement，对 industry 部署直接 actionable。Pareto frontier 上 Q-MQA combined 给 $97\%$ cache + near-parity quality 是 edge inference 的真实 enabler。

### 复杂度对比表

| Variant | Computation | Parameters | KV Cache |
|---|---|---|---|
| QKV | $3nd^2$ | $3d^2$ | K + V |
| Q=K-V / Q-K=V | $2nd^2$ | $2d^2$ | K only (Q-K=V) |
| (Q=K-V)+ | $2nd^2 + n^2 m$ | $2d^2 + m$ | K + V |
| Q=K=V | $nd^2$ | $d^2$ | K only |

Q-K=V 33% 计算/参数减少 + 50% cache 减少。

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
- [\[NeurIPS 2025\] Constant Bit-Size Transformers Are Turing Complete](../../NeurIPS2025/llm_efficiency/constant_bit-size_transformers_are_turing_complete.md)
- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](../../NeurIPS2025/llm_efficiency/zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[NeurIPS 2025\] From Shortcut to Induction Head: How Data Diversity Shapes Algorithm Selection in Transformers](../../NeurIPS2025/llm_efficiency/from_shortcut_to_induction_head_how_data_diversity_shapes_algorithm_selection_in.md)

</div>

<!-- RELATED:END -->
