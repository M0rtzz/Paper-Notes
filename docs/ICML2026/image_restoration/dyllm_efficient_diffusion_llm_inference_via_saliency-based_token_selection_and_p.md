---
title: >-
  [论文解读] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
description: >-
  [ICML2026][图像恢复][扩散语言模型] DyLLM 是一种 training-free 的扩散 LLM 推理加速框架，利用相邻去噪步骤之间注意力上下文的余弦相似度识别"显著 token"，只对这部分 token 重算 FFN 和注意力，配合显著感知的近似注意力…
tags:
  - "ICML2026"
  - "图像恢复"
  - "扩散语言模型"
  - "推理加速"
  - "时间稀疏性"
  - "显著 token 选择"
  - "近似注意力"
---

# DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention

**会议**: ICML2026  
**arXiv**: [2603.08026](https://arxiv.org/abs/2603.08026)  
**代码**: https://github.com/scale-snu/DyLLM.git (有)  
**领域**: LLM效率  
**关键词**: 扩散语言模型, 推理加速, 时间稀疏性, 显著 token 选择, 近似注意力

## 一句话总结
DyLLM 是一种 training-free 的扩散 LLM 推理加速框架，利用相邻去噪步骤之间注意力上下文的余弦相似度识别"显著 token"，只对这部分 token 重算 FFN 和注意力，配合显著感知的近似注意力，在 LLaDA / Dream 上把吞吐推到 7.6× / 9.6× 而几乎不掉点。

## 研究背景与动机

**领域现状**：Masked Diffusion Language Models (MDLM，如 LLaDA / Dream / Gemini Diffusion) 用双向注意力把全段 mask 序列一次性"填空"，在 GSM8K、MBPP 等基准上已经逼近 Llama 等 AR LLM，并具备并行解码、突破 token-by-token 顺序限制的潜力。

**现有痛点**：MDLM 每一步去噪都把**整段序列**重算一遍 ("repeated prefill")，因为双向注意力让位置间没有固定因果序，KV cache 无法像 AR 那样直接增量复用。结果是 FFN 在每一步都占绝对主导，整体吞吐反而被一堆"重复前缀"吃掉，相对 vLLM 加持的 AR 解码并无优势。

**核心矛盾**：并行解码（每步多 token、双向注意力）与缓存复用（要求位置/上下文稳定）天然冲突。已有方案要么按固定块/固定调度刷新（Fast-dLLM 的 PrefixCache / DualCache、dKV-Cache），要么按全局阈值挑 token（dLLM-Cache、Elastic-Cache），都没刻画"每一层在每一步只有少量 token 真的在变"这一更细粒度结构。

**本文目标**：在不重训、不改模型权重的前提下，把每步 FFN + Attention 的计算量从全序列压到一个**层自适应、token 自适应**的小子集，同时保证生成质量。

**切入角度**：作者在 LLaDA 上画出相邻步注意力上下文 $C_{t,l}^{(i)}$ 的余弦相似度 $s_{t,l}^{(i)}$ 分布（Fig. 2）—— 多数 token 在所有层都 $s\approx 1$，但深层尾部更"胖"，说明：(1) **时间稀疏性**真实存在；(2) 稀疏度**随层变化**，深层更需要更新。这正好提供了一个 per-layer 的稀疏 selector。

**核心 idea**：用相邻步注意力上下文的 cosine similarity 当显著性度量，对每层每步只重算"显著 token"，并把非显著 token 的注意力上下文更新近似成"只从显著列收集 ΔV"，从而把 FFN 和 attention 的稀疏性同时利用起来。

## 方法详解

### 整体框架
DyLLM 要解决的是 MDLM 每步把整段序列重算一遍的"重复前缀"瓶颈，做法是在标准解码循环外套一层"显著性调度器"：先用前 $T_{full}=4$ 个 full step 把各层的 attention / FFN 输出 cache 灌满，之后进入 "Salient only" 阶段——每一步 $t$、每一层 $l$ 先用相邻步上下文的余弦相似度算出一小撮"真的在动"的显著 token 子集 $\mathcal{A}_{t,l}$，FFN 与注意力都只对这个子集精确重算，其余 token 直接命中 cache。整体上等价于把 AR 推理的 "prefill → decode" 两阶段思想搬到了 diffusion 解码上，但 decode 阶段的 active set 是层自适应、步自适应的。

### 关键设计

**1. 层自适应显著 token 选择：用上下文方向漂移当 proxy**

针对"固定阈值挑 token 浪费或误剪"的痛点，DyLLM 在每层每步把 token 的注意力上下文 $C_{t,l}^{(i)}$（即 $\mathrm{softmax}(QK^T)V$ 的结果）与上一步做余弦相似度 $s_{t,l}^{(i)}=\cos(C_{t,l}^{(i)},C_{t-1,l}^{(i)})$，凡是 $s_{t,l}^{(i)}<\tau$ 的进入显著集 $\mathcal{A}_{t,l}=\{i\mid s_{t,l}^{(i)}<\tau\}$，门限 $\tau$ 取 0.99（LLaDA）/ 0.995（Dream）；非显著 token 不算 FFN、直接复用上一步的 FFN 输出 cache，连带省掉对应的注意力行重算。之所以"方向没怎么变就敢跳 FFN"，由两条命题托底：Prop 3.1 说明 RMSNorm 在线性投影下对尺度不敏感，Prop 3.2 进一步给出 FFN 输入扰动上界 $\delta\le\kappa(W_o)\sqrt{2(1-s_{t,l})}$——当 $s\to 1$ 时跳过 FFN 引入的误差趋近于 0。用 per-layer 门限而非全局阈值，正是因为作者在 Fig. 2 里观察到层间稀疏度差异显著（早期层 $s$ 普遍近 1、深层分布尾部更胖），层内阈值化天然让早期层激进剪枝、深层自动保守，把"误差预算"按层摊开。

**2. 显著感知近似注意力：行稀疏 + 列稀疏的双路径**

只跳 FFN 不够，注意力的二次开销 $O(N^2 d)$ 是另一座大山，DyLLM 把上下文增量展开成 $\Delta C_{t,l}=S_{t,l}\Delta V_{t,l}+(\Delta S)V_{t-1,l}$ 后做两条路径处理。对显著 query $i\in\mathcal{A}_{t,l-1}$，老老实实重算整行注意力（行稀疏路径）；对非显著 query，其注意力分数几乎没变（$\Delta S^{(i,\cdot)}\approx 0$），更新退化为 $\Delta C_{t,l}^{(i)}\approx S_{t,l}^{(i,\cdot)}\Delta V_{t,l}$，而 $\Delta V_{t,l}$ 又只在显著列上非零，于是只需 gather 显著列上的注意力分数即可（列稀疏路径）。这样整体复杂度压到 $O(N\cdot|\mathcal{A}_{t,l-1}|d)$，更妙的是同一个显著集既决定"哪些 query 行重算"又决定"哪些 KV 列要被聚合"，注意力与 FFN 共享同一张稀疏掩码，从而能落到一个 FlashAttention-like 的融合 kernel 上。

**3. Response-only Step 调度：彻底取消周期性全量刷新**

已有 cache 工作（dKV-Cache / Fast-dLLM / dLLM-Cache）都带一个"全 token refresh"的硬性周期，refresh 步会一脚把吞吐打回原始水准。DyLLM 借助 RoPE 相对距离衰减带来的 locality bias——显著 token 本就倾向集中在 response 区段——让大多数步只把 response token 喂入模型，每隔 4 步才把 prompt 一并送进去；而且即便是含 prompt 的步骤也仍按显著集稀疏算，prompt 那段稳定上下文几乎全命中 cache，所以根本不存在传统意义上的 full refresh 步。这直接吃掉了 prior work 的最大瓶颈：当每步并行解出的 token 数 $n_u$ 增大时，Fast-dLLM 的 full refresh 频率随之上升、急剧吞噬吞吐，而 DyLLM 仍能保持近线性的加速增益。

### 损失函数 / 训练策略
完全 training-free，无任何 fine-tune 或额外 loss；超参只有显著门限 $\tau$（model-dependent、task-agnostic）和暖启 full step 数 $T_{full}=4$。工程上在 CUDA 端写了自定义稀疏 attention / cache kernel（FlashAttention-like fused 设计），因为 PyTorch 原生稀疏算子在这种细粒度 token 稀疏下开销过大。

## 实验关键数据

### 主实验

在 LLaDA 8B Instruct / Dream 7B Instruct 上、$n_u=1$、单卡 H100 80GB PCIe，对比 Original / Fast-dLLM (Prefix & Dual) / dLLM-Cache（吞吐单位：tokens/s）：

| 模型 | Bench | Original (acc / tput) | DyLLM 最优 (acc / tput / speedup) | Fast-dLLM Dual | dLLM-Cache |
|------|-------|----------------------|----------------------------------|----------------|------------|
| LLaDA 8B | GSM8K | 77.79 / 11.47 | **79.08 / 87.21 / ×7.60** | 78.24 / 75.24 (×6.56) | 77.18 / 36.77 (×3.21) |
| LLaDA 8B | MATH | 33.22 / 15.81 | **38.68 / 96.98 / ×6.13** | 32.36 / 93.26 (×5.90) | 24.70 / 36.56 (×2.31) |
| LLaDA 8B | MBPP | 29.20 / 33.11 | **30.00 / 169.62 / ×5.12** | 25.40 / 165.44 (×5.00) | 29.00 / 93.04 (×2.81) |
| Dream 7B | GSM8K | 75.59 / 12.57 | **79.30 / 111.79 / ×8.89** ($\tau$=0.9975) | 68.39 / 153.21 (×12.19) | 72.40 / 46.19 (×3.67) |
| Dream 7B | MATH | 37.60 / 17.64 | **45.12 / 130.57 / ×7.40** | 36.06 / 191.56 (×10.86) | 44.98 / 48.76 (×2.76) |
| Dream 7B | MMLU-Pro | 47.94 / 12.60 | 47.45 / 83.10 / ×6.60 | 46.73 / 128.52 (×10.20) | 49.30 / 27.19 (×2.16) |

注：Fast-dLLM DualCache 在 Dream 上吞吐更高，但 GSM8K 直接掉到 68.39（−7.2 acc），这是它周期性 full refresh + 块缓存误剪的代价；DyLLM 在保 acc 前提下吞吐普遍领先 dLLM-Cache 2.16–3.67×。

### 消融实验

| 配置 (LLaDA, GSM8K) | Acc | 说明 |
|--------------------|-----|------|
| Original | 77.79 | 基线 |
| Salient-only FFN (τ=0.995) | 78.09 | 仅按显著性跳 FFN，注意力仍 full | 
| Salient + Approx. (τ=0.995) | 78.01 | 完整 DyLLM (FFN + 近似 attention) |
| τ=0.99 | 79.08 | 阈值更激进，准确率不降反升（softmax 噪声被压低）|
| τ=0.985 | 78.62 | 进一步降阈值开始略掉 |

加上 confidence-aware parallel decoding（Tab 3, Dream GSM8K）：DyLLM (τ=0.9975) 平均 $n_u$ 提到 **3.92**、acc 77.10；Fast-dLLM Dual 同设置下 $n_u$=3.68 但 acc 砸到 67.85 —— 说明稀疏更新和并行解码并不冲突，关键在于保住显著 token 的精确更新。

### 关键发现
- **softmax 噪声"被动去噪"**：低阈值（τ=0.99）反而轻微涨点，作者归因于 softmax 一定会给所有 token 分配正权重，序列变长时低相关 token 的累积贡献会引入噪声；DyLLM 把这些贡献按显著性显式截断，相当于 sparse attention 的隐式正则。
- **GQA 让 DyLLM 收益更大**：Dream 用 GQA，attention 相对便宜、FFN 占 70%+ 运行时，DyLLM 的 FFN 稀疏直接打在主力上，所以 Dream 上加速比比 LLaDA 高近 2×。
- **$n_u$ 扩展性是真正的杀手锏**：Fast-dLLM 每 $B=32$ 步必须做一次 full refresh（约 1280 token），$n_u$ 增大 refresh 频率上升，平均每步算的 token 数从 71（DualCache, $n_u=1$）迅速恶化；DyLLM 完全没有 full refresh 步，$n_u\in\{1,2,4\}$ 时吞吐曲线斜率明显更陡且 acc 不塌。
- **τ model-dependent, task-agnostic**：每个模型只需调一次（LLaDA=0.99 / Dream=0.995），跨 GSM8K / MBPP / MATH / MMLU-Pro 都泛化。

## 亮点与洞察
- 把"加速 diffusion LLM"从"缓存 KV / 缓存激活"的工程视角，重构成"每层每步只更新真正在移动的 token"这一更接近 MDLM 物理本质的视角；显著性度量直接取 attention 上下文 cosine，定义简洁且可证明（Prop 3.2）。
- **同一稀疏掩码同时驱动 FFN 跳过与 attention 双路径**，工程上把 sparse 模式收敛到一个 mask，能在 FlashAttention-like fused kernel 里高效实现；这一"FFN 与 attention 共享 active set"的思路可以直接迁移到任何"每步全序列重算"的迭代模型（如 image diffusion / world model 的多步 refinement）。
- 用"取消 full refresh"破解了 cache 类方法在 $n_u$ 增大时的 Achilles' heel —— 这一观察对所有想结合并行解码 + cache 的工作都有借鉴价值：refresh 步的代价应当被显式建模，否则 throughput 会被它单点拖死。

## 局限与展望
- **额外显存**：除 KV cache 外还要存 attention output cache 和 FFN output cache，按 (2d/g + 2d)/(2d/g)（$g$ 为 GQA 共享头数）膨胀；MHA 模型下扩到 2×，对超大 batch 推理有压力。作者论证扩散 LLM 本就单步 token 数 10× 于 AR、batch 通常较小，所以实际影响可控；但在端侧或 KV cache 已经吃满显存的部署里仍需斟酌。
- **τ 仍需 per-model 标定**：虽然 task-agnostic，但模型一换（譬如未来出的 Gemini Diffusion 或 MoE 变体）就得重新扫一次 τ；缺少自动门限或层间自适应 τ 的机制。
- **依赖时间稀疏性**：方法对早期 denoising 步（cache 尚未稳定，前 4 步 full）和稀疏性不显著的模型可能退化；若未来 MDLM 改用更"动态"的采样策略导致 $s_{t,l}$ 分布扁平，DyLLM 收益会缩水。
- **未覆盖在线服务场景**：所有结果均为 offline batched H100，未讨论变长 batch、prefill/decode 解耦或 KV cache 跨请求复用，能否在 vLLM 这种生产 stack 里达到同样加速比仍是开放问题。

## 相关工作与启发
- **vs Fast-dLLM (PrefixCache / DualCache)**：Fast-dLLM 按块固定缓存 + 周期性 full refresh，块外重要 token 会被忽略、refresh 步吞噬吞吐。DyLLM 用 per-layer per-step 自适应 active set，**完全不做 full refresh**，因此 $n_u$ 扩展性显著更好。
- **vs dKV-Cache**：dKV-Cache 周期性刷新 KV cache 纠错，本质是"时间窗口"的 cache，没有 token 维度的 selectivity；DyLLM 在 token + layer 两个维度都稀疏。
- **vs dLLM-Cache / Elastic-Cache**：两者都按激活相似度选 token，但用全局阈值或全局 $K_P, K_R$，需要 per-model + per-dataset 重调；DyLLM 用层内门限，超参更少、跨任务直接迁移，吞吐快 2.16–3.67×。
- **vs sparse attention 经典工作 (SparseD 等)**：传统 sparse attention 关心"哪些 KV 列要看"，DyLLM 关心"哪些 query 行真的在变"，对 diffusion 这种"全序列反复 refine"的迭代生成范式更契合。
- **可迁移的启发**：(1) 把"每步 vs 上一步的上下文方向漂移"作为通用稀疏 selector，可用于 image diffusion 的 latent token 重算调度；(2) FFN cache + active-set-driven attention 的双路径模式，可移植到 RNN-like decoder 的迭代 refinement 阶段。

## 评分
- 新颖性: ⭐⭐⭐⭐ 时间稀疏性 + 显著 token 概念在 diffusion LLM 上不算第一个，但首次把 per-layer per-step 自适应 selector 同时用到 FFN 跳过和 attention 双路径，并给出 RMSNorm 误差上界证明，组合性创新扎实。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 大基准 × 2 模型 × 3 baseline，含阈值扫、$n_u$ 扩展性、parallel decoding 兼容性、salient token 层分布、近似 attention 误差分布，几乎所有该问的都问了；但未做 B200 / 多卡 / vLLM 集成等部署侧实验（部分在附录 D.5）。
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰（观察 → 度量 → 命题 → 方法 → 实验），Figure 1 / 3 / 5 直击痛点；少量章节（如 Sec 2 背景）信息密度偏低。
- 价值: ⭐⭐⭐⭐⭐ Diffusion LLM 推理加速正处在"AR 能不能被替代"的关键节点，DyLLM 实测把 LLaDA / Dream 推到 ×7.6 / ×9.6 throughput 且几乎无损 —— 这种"训练免费、即插即用、scale 友好"的工作对整个 diffusion LLM 阵营至关重要，已开源代码也利于复现。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Encoder-Decoder Diffusion Language Models for Efficient Training and Inference](../../NeurIPS2025/image_restoration/encoder-decoder_diffusion_language_models_for_efficient_training_and_inference.md)
- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](../../ECCV2024/image_restoration/efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)
- [\[ICLR 2026\] Beyond Scattered Acceptance: Fast and Coherent Inference for DLMs via Longest Stable Prefixes](../../ICLR2026/image_restoration/beyond_scattered_acceptance_fast_and_coherent_inference_for_dlms_via_longest_sta.md)

</div>

<!-- RELATED:END -->
