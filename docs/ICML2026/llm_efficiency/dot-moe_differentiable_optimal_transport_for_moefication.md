---
title: >-
  [论文解读] DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE
description: >-
  [ICML 2026][LLM效率][MoEfication] DOT-MoE 把"dense FFN 转成 MoE 时怎么分配神经元到专家"建模成 differentiable optimal transport——Sinkhorn-Knopp 迭代解 entropic-regularized balanc…
tags:
  - "ICML 2026"
  - "LLM效率"
  - "MoEfication"
  - "神经元分配"
  - "Sinkhorn-Knopp"
  - "Straight-Through Estimator"
  - "dense-to-MoE"
---

# DOT-MoE: 用可微 optimal transport 把 dense LLM 转成 MoE

**会议**: ICML 2026  
**arXiv**: [2606.01666](https://arxiv.org/abs/2606.01666)  
**代码**: 论文未提供  
**领域**: 模型压缩 / MoE / Optimal Transport  
**关键词**: MoEfication, 神经元分配, Sinkhorn-Knopp, Straight-Through Estimator, dense-to-MoE

## 一句话总结
DOT-MoE 把"dense FFN 转成 MoE 时怎么分配神经元到专家"建模成 differentiable optimal transport——Sinkhorn-Knopp 迭代解 entropic-regularized balanced transport + Straight-Through Estimator 让 neuron-to-expert assignment 和 router 联合 end-to-end 学习；在 LLaMA-2/3 + Qwen2.5 上 50% 激活参数下保留 90% dense 性能，超过 structured pruning / random / 聚类等所有 baseline。

## 研究背景与动机

**领域现状**：LLM 缩放带来性能飞跃但 inference 代价巨大。Dense Transformer 每 token 激活全部参数导致 latency 爆炸。MoE（Switch、GShard、Mixtral、Qwen3-30B-A3B）通过 sparse routing 把"模型大小"和"inference 成本"解耦——Qwen3-30B-A3B 总 30.5B 参数但每 token 只激活 3.3B。但 from-scratch 训练 MoE 数据饥饿、需要复杂 load-balancing。MoEfication（Zhang 2022）走"把 dense 转成 MoE"路线 leverage 已有 dense checkpoint。

**现有痛点**：现有 MoEfication 方法分配神经元到专家的策略都是 heuristic——(1) Random（LLaMA-MoE）随机分配 + 大量 continued pretraining 救；(2) Weight-based clustering（LTE/MoEfication）按 $W_{\text{gate}}/W_{\text{up}}$ 权重相似度聚类；(3) Activation-based clustering（LLaMA-MoE-v2、CMoE）按激活/梯度 importance 聚类。共同 limitation：optimize 的都是 intermediate representation 的 proxy（input weights / activations / co-activation），不是 actual FFN output。看 $\text{FFN}(\mathbf{x}) = \mathbf{H} \mathbf{W}_{\text{down}}$ —— output 取决于 intermediate $\mathbf{H}$ 和 $\mathbf{W}_{\text{down}}$ 的 interaction，proxy 方法没 capture。

**核心矛盾**：分配神经元 + 训 router 必须 jointly 优化（neuron 分配变了 → 该 route 到 expert 的 token 变了 → router 也得变），但 discrete assignment 不可微，所以现有方法 frozen assignment 再 train router——两阶段不优化整体 output reconstruction。

**本文目标**：建一个 (a) jointly 优化 neuron assignment + router、(b) 保证 expert capacity balance、(c) output-aware 而非 proxy-aware 的 framework。

**切入角度**：神经元分配 = mass transport（每个神经元 carry unit mass 到 expert，每个 expert 接 $s$ 个 mass）—— 正是 optimal transport 问题。OT 有 Sinkhorn 解析解（差分可微），entropic regularization 让 solution 唯一且 closed-form。Straight-Through Estimator 让 discrete decision 反向传播 work。

**核心 idea**：(1) 把 neuron 分配 framed 为 balanced OT：source $\mathbf{r} = \mathbf{1}_{d_{\text{ffn}}}$（每神经元一次）、target $\mathbf{c} = s \cdot \mathbf{1}_E$（每 expert 收 $s$ 个）、learnable cost matrix；(2) Sinkhorn-Knopp 解 entropic-regularized OT 得 soft assignment；(3) Greedy rounding 转 hard assignment，STE 让梯度通过；(4) 联合 train assignment + router + reconstruct dense output 的 KL divergence loss。

## 方法详解

### 整体框架

输入：dense pretrained LLM 的 FFN $\text{FFN}(\mathbf{x}) = (\sigma(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}})) \mathbf{W}_{\text{down}}$，$d_{\text{ffn}}$ 个 intermediate 神经元。目标：分成 $E$ experts 每个含 $s = d_{\text{ffn}}/E$ 神经元，每 token 路由到 $k < E$ experts。三组件：(1) **OT-based neuron assignment** 用 learnable affinity matrix $\mathbf{A}$ + Sinkhorn 解 soft assignment；(2) **Token routing** 用 router $\mathbf{W}_r$ + top-$k$ select；(3) **Alignment phase** 用 KL divergence between dense teacher 和 sparse student output + auxiliary load balance + router z-loss 联合训。Training 完后提 $\mathbf{M}$ 转标准 MoE 架构。

### 关键设计

1. **Neuron Assignment 作 Balanced Optimal Transport**:

    - 功能：保证 expert capacity 严格 balanced 同时允许 learnable affinity 决定 assignment。
    - 核心思路：定义 transport problem $\mathbf{M}^* = \arg\max_{\mathbf{M} \in \mathcal{U}(\mathbf{r}, \mathbf{c})} \langle \mathbf{A}, \mathbf{M} \rangle$，$\mathbf{A} \in \mathbb{R}^{d_{\text{ffn}} \times E}$ 是 learnable affinity，$\mathcal{U}$ 是 transportation polytope（marginal $\mathbf{r} = \mathbf{1}_{d_{\text{ffn}}}$, $\mathbf{c} = s \cdot \mathbf{1}_E$）。解析解在 polytope 顶点（$\{0,1\}$-matrix）—— 不可微。加 entropic regularization $-\tau H(\mathbf{M})$ 让解 unique 且在 polytope 内部：$M_{i,e}^* = u_i \cdot \exp(A_{i,e}/\tau) \cdot v_e$，Sinkhorn-Knopp 算法 alternate row/column normalization 求 $\mathbf{u}, \mathbf{v}$，linear convergence。Log-domain 数值稳定。
    - 设计动机：以前 MoEfication 用 random / clustering，没法保证 expert capacity 严格 balance；OT 通过 marginal constraint 强制 balance。Entropic regularization + Sinkhorn 让"解 OT"从 LP（intractable）变成 differentiable iterations，可与 router 联合训。

2. **Straight-Through Estimator 让 discrete decision 可微**:

    - 功能：forward 用 hard assignment（部署需要），backward 用 soft assignment（梯度传播）。
    - 核心思路：Sinkhorn 给 soft assignment $\mathbf{M}_{\text{soft}}$；greedy rounding 转 hard $\mathbf{M} \in \{0,1\}^{d_{\text{ffn}} \times E}$（按 $\mathbf{M}_{\text{soft}}$ entries 降序排，依次满足 capacity）；STE $\mathbf{M}_{\text{STE}} = \mathbf{M} + (\mathbf{M}_{\text{soft}} - \text{sg}(\mathbf{M}_{\text{soft}}))$，forward 用 $\mathbf{M}$ backward 用 $\mathbf{M}_{\text{soft}}$。Router top-$k$ selection 同理 $\mathbf{R}_{\text{STE}} = \mathbf{R} + (\mathbf{P} - \text{sg}(\mathbf{P}))$。
    - 设计动机：MoE 部署需要 hard expert assignment；training 时如果直接用 hard 会断梯度。STE 是 BinaryNet 等量化网络的经典 trick，本文把它移植到 neuron-to-expert 和 token-to-expert 双层 discrete decision，是 end-to-end joint training 的关键 enabler。

3. **Output-Aware KL Divergence Alignment**:

    - 功能：训练目标直接对齐 dense teacher 和 sparse student 的 output，而非 intermediate proxy。
    - 核心思路：Forward 时模拟 sparse MoE 计算：$\hat{\mathbf{Y}} = (\mathbf{H} \odot (\mathbf{R} \mathbf{M}^\top)) \mathbf{W}_{\text{down}}$，只激活 $k \cdot s$ 个神经元贡献 output。Loss 是 KL divergence between dense $\mathbf{Y}$ 和 sparse $\hat{\mathbf{Y}}$ + cross-entropy LM loss + router z-loss（防 router logits 爆炸）+ load balancing loss（防 expert collapse）。整套训练 update affinity $\mathbf{A}$ 和 router $\mathbf{W}_r$ + 全网络。
    - 设计动机：以前方法 optimize input weight 相似度或 activation co-occurrence 等 proxy；本文直接看 output reconstruction MSE（appendix 实验显示 proxy-based 方法 MSE 是 DOT-MoE 的 $2\times$ 到 $41\times$，结构性失败）。Output-aware 让训练目标和 deployment 目标对齐。

### Multi-head Attention 扩展

同样的 balanced transport 框架可推广到 attention heads（把 heads 当 neurons 分到 expert），详见 Appendix G。

## 实验关键数据

### Perplexity & HellaSwag at 50% Parametric Budget（LLaMA-2 7B）

| Method | WikiText PPL ↓ | HellaSwag acc-n ↑ |
|--------|----------------|---------------------|
| **Structured Pruning** | | |
| LLM-Pruner | 31.05 | – |
| LLM Surgeon | 15.38 | 40.3 |
| ShortGPT | 268.11 | 43.7 |
| SliceGPT | 24.82 | 33.0 |
| ModeGPT | 11.88 | – |
| DISP-LLM | 9.84 | 46.3 |
| **Semi-Structured Pruning (2:4)** | | |
| SparseGPT | 10.17 | 43.3 |
| Wanda | 11.02 | 40.9 |
| Pruner-Zero | 10.52 | 54.7 |
| **DOT-MoE** | **7.99** | 53.9 |

DOT-MoE WikiText PPL 7.99 是全场最低，比最强 structured pruning（DISP-LLM 9.84）低 1.85；HellaSwag 53.9 跟最强 Pruner-Zero（54.7）持平。

### Common-Sense Reasoning（多 benchmark）

| Method | Active Params | FT Tokens | BoolQ | SciQ | PIQA | WinoG. | ARC-C | HellaS. | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| **LLaMA-2 7B** | | | | | | | | | |
| Dense | 6.74B | 2T* | 82.0 | 94.0 | 78.1 | 74.3 | 52.5 | 78.9 | 76.6 |
| LLaMA-MoE (Random) | 3.49B | 1.2B | 37.8 | 20.0 | 49.7 | 50.1 | 25.8 | 26.2 | 34.9 |
| LLaMA-MoE-v2 | 3.49B | 1.2B | 51.3 | 67.0 | 56.6 | 52.9 | 25.7 | 35.1 | 48.1 |
| CMoE | 3.49B | 1.2B | 55.0 | 77.5 | 57.1 | 54.1 | 27.6 | 38.8 | 51.7 |
| **DOT-MoE** | 3.49B | 1.2B | **72.5** | **94.3** | **69.3** | **62.5** | **40.9** | **60.2** | **66.6** |
| **LLaMA-3 8B** | | | | | | | | | |
| Dense | 8.03B | 15T* | 83.2 | 96.2 | 79.6 | 77.3 | 58.3 | 82.1 | 79.4 |
| CMoE | 3.80B | 1.2B | 71.1 | 94.4 | 69.5 | 59.5 | 38.2 | 55.3 | 64.7 |
| **DOT-MoE** | 3.80B | 1.2B | **75.0** | 94.2 | **70.2** | **63.8** | **42.4** | **61.1** | **67.8** |
| LLaMA-MoE-v2 (7B FT) | 3.80B | 7B | 74.6 | 94.5 | 69.3 | 60.5 | 42.8 | 59.0 | 66.8 |
| **DOT-MoE (7B FT)** | 3.80B | 7B | 75.4 | **96.2** | **73.3** | **66.1** | **49.1** | **66.0** | **71.0** |

50% 激活参数下：LLaMA-2 7B Dense 76.6 → DOT-MoE 66.6（保留 87%），相比次优 CMoE 51.7 高 +14.9 个点。LLaMA-3 8B 同样 +3.1 over CMoE。Qwen2.5 7B 也类似 trend。

### 关键发现

- **保留 90% dense 性能 at 50% active params**：DOT-MoE 把 dense → MoE 的 quality-efficiency tradeoff 推到接近 Pareto optimal；structured pruning 方法在 50% budget 下通常掉到 40-60%。
- **OT-based >> heuristic clustering**：相比 LLaMA-MoE（random）、LLaMA-MoE-v2（activation）、CMoE（balanced k-means activation），DOT-MoE 平均涨 14.9-30.7 个点，证明 jointly optimizing assignment + router 比 frozen heuristic 数量级好。
- **小数据 FT 也行**：1.2B FT tokens（相比 dense pretrain 2T，0.06%）就能恢复到 87%，说明 OT-based assignment 直接给了好初始化。
- **更多 FT tokens 进一步靠近 dense**：LLaMA-3 8B 用 7B FT tokens 平均 71.0，比 dense 79.4 只差 8.4 个点。
- **MSE 实验验证 output-aware**：appendix 显示 proxy-based 方法 MSE 是 DOT-MoE 的 $2\times$-$41\times$，证明 output reconstruction objective 比 intermediate proxy 更准确。
- **跨架构 robust**：LLaMA-2 / LLaMA-3 / Qwen2.5 三个模型族都 work，方法不挑架构。
- **比 structured pruning 完胜**：WikiText PPL 7.99 vs DISP-LLM 9.84（structured pruning 最强 baseline）+ HellaSwag 53.9 vs Pruner-Zero 54.7，证明 MoEfication 路线在 50% budget 下比 pruning 路线优。

## 亮点与洞察

- **OT framing 是优雅的方法学创新**：把"神经元分配"看成"质量运输"是直观又数学严格的形式化，让 balanced capacity + learnable affinity 自然兼容。
- **Sinkhorn + STE 双管齐下**：Sinkhorn 解决"discrete OT 怎么变 differentiable"，STE 解决"discrete deployment 怎么 backward"——两个 trick 协同让 end-to-end joint training 可行。
- **Output-aware KL > proxy alignment**：明确指出"existing methods optimize proxies"是 fundamental limitation，并用 single-layer reconstruction 实验定量验证（$2\times$-$41\times$ MSE 差距）。
- **Joint vs Sequential 训练范式**：以前"先 freeze assignment 再 train router"是 sequential；DOT-MoE 让 assignment 和 router co-adapt 是 fundamental shift，比 sequential 数量级好。
- **Dynamic structural pruning 视角**：把 MoEfication 看成 dynamic pruning（保 full param 但 conditional 激活），相对 static pruning（永久删 param 损失 long-tail knowledge）有 capacity 优势，文章 framing 也优雅。
- **可扩展到 attention heads**：同样 balanced transport 框架可分 heads，给"压 attention"提供 OT 工具。

## 局限与展望

- **OT + Sinkhorn 计算成本**：每个 FFN layer 需 Sinkhorn 迭代到收敛 + STE 反向传播，比 simple clustering 计算成本高几倍；对超大模型（70B+）每层 expensive。
- **Greedy rounding 的 mismatch**：soft → hard 用 greedy 可能跟 OT 最优 vertex 不同，理论上有 gap 但实验没量化。
- **依赖小数据 FT**：DOT-MoE 仍需 1.2B-7B FT tokens 恢复性能；如果 zero FT 表现如何没单独消融。
- **$k$ 和 $E$ 选择**：实验用 $E=8, k=1\text{-}2$；不同 (E, k) 的 Pareto 没系统扫。
- **跟 MoE pretrained model 对比**：Mixtral / Qwen3-MoE 等 from-scratch MoE 的性能 vs DOT-MoE post-hoc 的对比没给（虽然 from-scratch 更贵）。
- **70B+ scaling 未验证**：实验最大 8B，70B+ 模型的 FFN 神经元更多 ($d_{\text{ffn}} \approx 28K$)，Sinkhorn 在更大 $E$ 下 convergence 速度需要更多 measurement。
- **Inference Latency 实测缺**：理论 FLOPs 减半但 actual GPU latency 还依赖 MoE kernel 实现，没给 wall-clock speedup 数字。

## 相关工作与启发

- **vs MoEfication / LTE (Zhang 2022)**：他们用 weight-based clustering，本文证明 weight similarity 不等于 output contribution；OT framework 比 clustering 数量级好。
- **vs LLaMA-MoE (Random)**：random 分配靠 1.2B continued pretrain 救，DOT-MoE 用 OT 直接给好初始化 + 联合 train router，66.6 vs 34.9 差 31.7 个点。
- **vs LLaMA-MoE-v2 / CMoE (Activation Clustering)**：activation 是 proxy，DOT-MoE output-aware 直接对齐 dense output，差 12-15 个点。
- **vs Structured Pruning (DISP-LLM / SliceGPT / ShortGPT)**：pruning 永久删 param 损 long-tail knowledge；DOT-MoE 保 full param 但 sparse 激活，capacity 优势让 PPL 低 1.85+。
- **vs Semi-Structured (SparseGPT / Wanda / Pruner-Zero)**：2:4 sparsity 硬件友好但 quality 受限；DOT-MoE PPL 7.99 vs SparseGPT 10.17。
- **vs Mixtral / from-scratch MoE**：from-scratch 更贵但 quality 上限高；DOT-MoE 给"用现成 dense → MoE"的 cost-effective 路线。
- **启发**：(1) 任何"discrete assignment + balanced capacity"问题都可试 OT + Sinkhorn + STE 框架；(2) output-aware vs proxy-aware 的对比应推广到其他 model compression（pruning 也常 optimize proxy）；(3) joint vs sequential training 对所有 discrete-continuous 联合问题有指导意义。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ OT framing + Sinkhorn + STE + output-aware objective 四件组合是真原创，把 MoEfication 从 heuristic 推到 principled。
- 实验充分度: ⭐⭐⭐⭐⭐ 3 model families (LLaMA-2/3/Qwen2.5) + 6 benchmarks + 比 structured pruning 完胜 + appendix 详细 ablation；缺 70B scale 和 inference latency 实测。
- 写作质量: ⭐⭐⭐⭐ Section 2.3 揭示"proxy vs output"是关键 insight；OT 公式严格但 readable；alignment phase 细节略压缩到 appendix。
- 价值: ⭐⭐⭐⭐⭐ 直接服务 LLM 部署痛点（50% active params 保 90% performance），对工业 cost-saving 是 actionable recipe；method 跨架构 robust 适合广泛应用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Variational Routing: 校准 MoE Transformer 的可扩展贝叶斯框架](variational_routing_a_scalable_bayesian_framework_for_calibrated_mixture-of-expe.md)
- [\[ICML 2026\] ReMoE: Boosting Expert Reuse through Router Fine-Tuning in Memory-Constrained MoE LLM Inference](remoe_boosting_expert_reuse_through_router_fine-tuning_in_memory-constrained_moe.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[ACL 2025\] DIVE into MoE: Diversity-Enhanced Reconstruction of Large Language Models from Dense into Mixture-of-Experts](../../ACL2025/llm_efficiency/dive_moe_reconstruction.md)
- [\[ICML 2026\] Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers](optimal_bayesian_stopping_for_efficient_inference_of_consistent_llm_answers.md)

</div>

<!-- RELATED:END -->
