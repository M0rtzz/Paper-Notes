---
title: >-
  [论文解读] Stochastic Sparse Attention for Memory-Bound Inference
description: >-
  [ICML 2026][模型压缩][稀疏注意力] SANTA 把 attention 的 value 聚合 $AV$ 看作 "按 softmax 概率 $A$ 对值行 $V$ 做加权求和", 改成 "从 $A$ 中无放回采样 $S\ll n_k$ 个索引、直接平均对应 $V$ 行"的无偏估计…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "稀疏注意力"
  - "随机采样"
  - "KV-cache"
  - "Stratified Sampling"
  - "GPU kernel"
---

# Stochastic Sparse Attention for Memory-Bound Inference

**会议**: ICML 2026  
**arXiv**: [2605.01910](https://arxiv.org/abs/2605.01910)  
**代码**: <https://github.com/OPUSLab/SANTA.git>  
**领域**: 模型压缩 / LLM 推理加速 / 注意力优化  
**关键词**: 稀疏注意力, 随机采样, KV-cache, Stratified Sampling, GPU kernel

## 一句话总结
SANTA 把 attention 的 value 聚合 $AV$ 看作 "按 softmax 概率 $A$ 对值行 $V$ 做加权求和", 改成 "从 $A$ 中无放回采样 $S\ll n_k$ 个索引、直接平均对应 $V$ 行"的无偏估计, 用 stratified / systematic 采样降方差, 再写成 GPU kernel 与 FlashDecoding 对齐——在 32k context 下端到端比 FlashInfer / FlashDecoding 快 1.5× 且精度不掉。

## 研究背景与动机

**领域现状**: 长上下文自回归解码是 LLM 部署的痛, 每生成一个 token 都要把整个 KV cache 流过一遍, 带宽成为瓶颈 (Llama-3.1-8B 32k context 每层每 token 要传 ~128 MB)。现有缓解手段分四类: KV 量化压缩 (KIVI 等)、cache 管理 (Quest, H2O)、结构化稀疏注意力 (Longformer, BigBird)、内核优化 (FlashAttention, FlashDecoding)——再叠 GQA。但即使最优 exact kernel, 每步仍要碰整个 KV state, 带宽墙仍在。

**现有痛点**: top-$k$ / threshold-based 稀疏方法**是有偏估计**, 且通常需要排序; 量化/压缩破坏 KV 数值精度; 结构化稀疏 (sliding window 等) 牺牲表达力; FlashDecoding 已经几乎榨干 IO 局部性, 进一步加速需要直接**减少要读的 V 行数**, 而不是优化怎么读。

**核心矛盾**: attention 输出 $AV$ 是一个**期望**——$A$ 本身就是概率分布, 那为什么要把它当确定性权重和值矩阵乘? 完全可以用蒙特卡洛只算样本和。但 GPU 上随机采样会破坏并行性 (需要全局 CDF), 这正是工程难点。

**本文目标**: (a) 把 $AV$ 改写成无偏蒙特卡洛估计, 把 V 行访问从 $n_k$ 降到 $S\ll n_k$, 顺带消掉 softmax 后的所有乘法; (b) 降方差到能匹配 SDPA 精度; (c) 写一个 GPU kernel 让它真正跑出 wall-clock 加速; (d) 顺便给 score 阶段也提供一个稀疏化方案 (Bernoulli $qK^T$)。

**切入角度**: 从概率视角看 attention——把 $A$ 看成 categorical 分布, 用采样替代矩阵乘; 把 "每个 head 一个独立 CDF"和 FlashDecoding 的 tile 化策略结合, 用两种方案 (proportional / flash) 解决 "全局 CDF vs 全局同步"矛盾。

**核心 idea**: $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, $i_s\sim A$ i.i.d., 这是 $AV$ 的无偏估计, 方差为 $O(1/S)$; 配合 stratified / systematic sampling 进一步降方差; GPU 上用 "全局轻量 sync + 按 tile 概率质量分配采样预算"避免 CDF 串行依赖。

## 方法详解

### 整体框架
SANTA 是一个**解码阶段**的注意力替换方案 (prefill 也能用但收益小)。核心三件套: (1) 数学层面的无偏估计 SANTA / S²ANTA-strat / S²ANTA-sys; (2) GPU kernel 层面的两种实现 S²ANTA-prop (全局 sync 精确分配) 与 S²ANTA-flash (speculative 局部采样); (3) Bernoulli $qK^T$, 对 score 阶段也做稀疏。集成方式: prefill 仍用 SDPA, 仅 decode-step 用 SANTA, 与 GQA / FlashInfer / 量化等正交可叠加。

### 关键设计

1. **SANTA 无偏估计 + Stratified/Systematic 降方差**:

    - 功能: 把 dense $AV$ 替换为 sparse $\widehat{AV}=\frac1S\sum_{s=1}^S V_{i_s}$, 其中 $i_s$ 从 categorical $A$ 中独立采样; V 行读取从 $n_k$ 降到 $S$, 且 softmax 后只剩加法, 没有乘法。
    - 核心思路: 基础 SANTA 用 i.i.d. 采样, $\mathbb E[\widehat{AV}]=AV$, $\mathrm{Var}\propto 1/S$ (附录 A.1, A.2)。 为降方差, **S²ANTA-strat** 把 CDF 等概率划成 $S$ 段, 每段独立采一个: $T_m\sim\mathrm{Unif}(I_m)$, $J_m=F_q^{-1}(T_m)$, $\widehat{AV}=\frac1S\sum V_{J_m}$。 **S²ANTA-sys** 用 systematic sampling, 只采一个偏移 $U\sim\mathrm{Unif}[0,1/S)$, 阈值 $T_m=U+m/S$——硬件友好 (一个随机数即可生成 $S$ 个样本), 实践中和 strat 一样降方差但没有理论保证。 当 $S$ 是 2 的幂时, 归一化是 bit-shift。
    - 设计动机: 把概率视角应用到一个早已成为推理瓶颈的算子, 既消乘法又减读数, 一举两得; 用 stratified 保留无偏性且天然适合并行 (每个 stratum 独立)。

2. **S²ANTA-prop: 全局轻量 sync 的精确预算分配**:

    - 功能: 在 GPU 上把 attention 切成 $T$ 个 tile, 每个 tile 通过两遍 kernel 完成 "精确按概率质量分配采样预算 → 并行采样并 gather V 行"。
    - 核心思路: Pass 1 计算 scores 并把 exponentiated scores ($1\times n_k$, 占 $1/d_k$ 带宽) 与 tile-local partition function $Z_{tile}$ 写到 global memory; 全局 reducer 把 $Z=\sum Z_{tile}$ 加总, 然后分配 $S_{tile}\propto S\cdot(Z_{tile}/Z)$; Pass 2 用 stashed scores + 分配到的 $S_{tile}$ 系统采样 + gather $V$ 行。 低概率 tile 拿到 $S_{tile}=0$ 直接跳过昂贵的 V-read。
    - 设计动机: 全局 CDF 本质是串行依赖, 但把它 "轻量化": 只 sync $T$ 个 scalar 而不是整个 score 矩阵, 同步成本可忽略; 32k context 下用 $S=128$ 即可对齐 SDPA, V 行访问下降到 < 1.56%。

3. **S²ANTA-flash: speculative 采样 + 延迟归一化**:

    - 功能: 完全去掉 sync, 让每个 tile 都按 "平均预算 $S/T$"独立采样, 最后再 merge 时根据真实 $Z$ 缩放——直接 mirror FlashDecoding 的设计哲学。
    - 核心思路: 每个 tile 假定自己持有全部概率质量, 采 $S/T$ 个样本得到本地部分和; 第二遍 reducer 算真实 $Z$ 与每个 tile 的真实 $Z_{tile}/Z$, 把 "低概率 tile"的部分和缩成接近 0。 因此低概率 tile 的采样和 V 读其实是 "被浪费的" (sample waste)。
    - 设计动机: 在不能容忍任何全局 barrier 的场景给一个可选方案; 代价是要更大的总样本数 ($S=2048$ vs prop 的 $S=128$ 才能对齐 SDPA 精度), 但 wall-clock 速度仍能拿到 1.51×。

### 损失函数 / 训练策略
本文是**纯推理时方法**, 不改训练, 不引损失。所有方法 plug-and-play 替换 attention 算子。Bernoulli $qK^T$ 作为 complementary score-stage 稀疏化 (Sec 5): 把 query 归一到 $[-1,1]$ 当 Bernoulli 概率采 $\{-1,0,+1\}$ 三值, 形成 sparse ternary query, 实现对 K 矩阵的 feature-wise 稀疏访问。

## 实验关键数据

### 主实验

**32k 长上下文 RULER (Llama-3.1-8B-Instruct)** Table 1: SDPA 用于 prefill, 仅 decode 替换。

| Kernel | $S$ | FWE | NIAH | QA1 | QA2 |
|---|---|---|---|---|---|
| SDPA (baseline) | – | 95.60 | 98.35 | 64.00 | 58.80 |
| **S²ANTA-prop** | **128** | **95.40** | **98.25** | **64.40** | **60.20** |
| S²ANTA-prop | 256 | 95.47 | 98.50 | 63.40 | 60.60 |
| **S²ANTA-flash** | **2048** | **94.13** | **98.25** | **64.60** | **60.00** |
| S²ANTA-flash | 256 | 66.20 | 88.95 | 63.00 | 57.20 |

prop 在 $S=128$ (= $n_k$ 的 0.39%) 就拿到 SDPA 同档精度, flash 需要 $S=2048$ (= 6.25%)。 Kernel 延迟 (Fig 4): prop 1.50× / flash 1.51× speedup vs FlashInfer。

**GSM8K (Llama 8B)** Table 2 (节选): 比较 SANTA / S²ANTA-strat / S²ANTA-sys 在不同 $S$ 下的精度。

| $S$ | S²ANTA-sys | S²ANTA-strat | SANTA |
|---|---|---|---|
| 16 | 44.63 | 39.12 | 5.51 |
| 32 | 68.59 | 67.00 | 38.26 |
| 64 | 76.42 | 74.43 | 63.63 |
| 128 | **77.33** | 75.64 | 70.23 |
| 256 | 77.56 | **78.17** | 75.61 |
| SDPA | – | – | 78.06 |

方差降低带来巨大差距: $S=16$ 时 sys 比基础 SANTA 高 39 个点。

**MMLU** Table 3: 同样 stratified 系列在小 $S$ 下显著优于 SANTA, $S=256$ 时三者均回到 SDPA ±1% 内 (49.86 baseline)。

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| SANTA vs S²ANTA-strat vs S²ANTA-sys | $S\le 64$ 时 stratified 系列大幅领先 | 验证降方差关键 |
| prop vs flash kernel | 同 wall-clock speedup, prop 用 1/16 的 $S$ | sync 成本可忽略, 显著省样本浪费 |
| Bernoulli $qK^T$ on BitNet 2B (GSM8K) | $B=4$ 时只读 67.5% K 特征, 精度 64.5% (SDPA 65.7%) | score 阶段也能稀疏化, 与 SANTA 正交 |
| Mean group query | $B=4$ K 访问 84.7% (单独时 97.9%) | 缓解 GQA 共享带来的 union 爆炸 |

### 关键发现
- **采样不仅消乘法**: 在 long context decode 阶段, 真正赚的是 V 读带宽下降 (32k 上 < 2%); 而消乘法 (1.1 pJ → 0.4 pJ per op) 是 "等加法器优化的未来硬件"才能完全兑现的红利。
- **stratified 降方差是必须项**: 不带降方差的 SANTA 在 $S=16$ 时 GSM8K 只有 5.5%, 完全不能用; 加 stratified/systematic 后立刻可用——说明朴素蒙特卡洛在 attention 上方差爆炸。
- **systematic vs stratified**: 实测精度几乎一样, 但 systematic 只要 1 个随机数, 极其硬件友好——这是非常 production-friendly 的设计。
- **flash kernel 的 "sample waste"是真实存在的**: 同 wall-clock speedup 下 flash 需要 16× 更多样本, 说明在 attention 这种概率分布极不均匀的场景, 全局 sync 反而更经济。

## 亮点与洞察
- **概率视角看 attention**这一动作非常简洁——既然 softmax 已经给了一个概率分布, 那直接采样就好。这一思想可推广到所有 softmax-based 操作 (mixture-of-experts gating, retrieval ranking)。
- **"消乘法"对应未来硬件**: 加法器和乘法器的能耗比悬殊 (~0.36×), 论文明确指向 sparse, adder-centric accelerator——这与近年 BitNet / 1-bit LLM 的硬件趋势完美对接。
- **systematic sampling 用 1 个随机数生成 $S$ 个样本**, 在嵌入式或定制 silicon 场景里把 "采样"做成 cheap operation 是巨大优势。
- **prop kernel 用 "轻量 sync"打破 CDF 串行**: 这种 "先算 scalar reduction 再分配预算"的设计可以套到任何 "需要全局归一化的稀疏化"任务上, 例如 sparse softmax MoE 路由。
- **方法是 plug-and-play**, 不需要重训, 不破坏精度, 不冲突其它已有手段 (量化、GQA、cache 压缩), 可叠加。

## 局限与展望
- 当前 GPU kernel 的 wall-clock 加速主要来自带宽下降, 乘法消除的红利在 NVIDIA 矩阵 FMA 优化下不显著, 需要等加法器导向的新硬件。
- prefill 阶段几乎无收益——因为 $n_q=n_k$, V 行读的稀疏性被并集吃掉; 论文也没声称在 prefill 上 wall-clock 受益。
- 采样质量依赖 softmax 分布的 "良态性", 如果 attention 分布极平坦 (无明显 hotspot), 即使 stratified 也可能不够; 论文没分析这种 worst-case。
- Bernoulli $qK^T$ 在非 BitNet 模型上效果未知, 普通 fp16 模型对 query ternary 化的容忍度可能更差。
- 与 cache 管理类方法 (Quest, H2O) 的组合实验没做, 实际部署中需要测两者叠加的精度。

## 相关工作与启发
- **vs FlashDecoding / FlashInfer (Dao 2023, Ye 2025)**: 它们是 exact attention 的 IO 优化, 已经摸到带宽天花板; SANTA 是正交方向 (减少需要访问的行), 论文直接以它们为 baseline 比较出 1.5× speedup。
- **vs top-$k$ 注意力 (Quest, H2O 等)**: top-$k$ 是有偏的, 需要排序, 大 $k$ 时仍需读多数 V 行; SANTA 是无偏的, 用 stratified 即可在 $S=128$ 拿到 32k context 的 SDPA 精度。
- **vs Sparse Transformer / Longformer (Child 2019, Beltagy 2020)**: 这些是结构化稀疏, 训练时就要写死 pattern; SANTA 推理时随机, 不动训练。
- **vs KV 量化 (KIVI, Hooper 2024)**: 量化降低每元素 bytes, SANTA 降低被读元素数, 两者完全互补可叠加。
- **vs MoE gating / sparse softmax**: 同样面临 "需要按概率稀疏化"的问题, SANTA 的 prop kernel 设计可直接迁移。

## 评分
- 新颖性: ⭐⭐⭐⭐ 用 Monte Carlo 重新解读 attention value 阶段, 配套 stratified / systematic + GPU kernel, 不算革命性但非常 elegant。
- 实验充分度: ⭐⭐⭐⭐ GSM8K / MMLU / 长上下文 RULER + 真实 GPU kernel 延迟 + Bernoulli $qK^T$ 副实验都全。
- 写作质量: ⭐⭐⭐⭐⭐ 概念清晰, 公式 Eq.(4) 一句话讲完核心估计器, prop / flash 的对比图直观。
- 价值: ⭐⭐⭐⭐⭐ 直接开源 kernel, plug-and-play 提供长上下文 1.5× 加速, 长 context LLM 推理团队必看。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection](token_sparse_attention_efficient_long-context_inference_with_interleaved_token_s.md)
- [\[NeurIPS 2025\] SpecAttn: Speculating Sparse Attention](../../NeurIPS2025/model_compression/specattn_speculating_sparse_attention.md)
- [\[ICML 2025\] FloE: On-the-Fly MoE Inference on Memory-constrained GPU](../../ICML2025/model_compression/floe_on-the-fly_moe_inference_on_memory-constrained_gpu.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](../../ICML2025/model_compression/mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)
- [\[ICML 2026\] Test-Time Training with KV Binding Is Secretly Linear Attention](test-time_training_with_kv_binding_is_secretly_linear_attention.md)

</div>

<!-- RELATED:END -->
