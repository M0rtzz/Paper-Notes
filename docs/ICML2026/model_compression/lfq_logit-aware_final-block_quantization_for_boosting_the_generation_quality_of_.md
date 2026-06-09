---
title: >-
  [论文解读] LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs
description: >-
  [ICML 2026][模型压缩][低比特量化] 针对 block-wise PTQ 在生成任务上的质量退化问题，LFQ 将最后一个 Transformer block 的量化目标从 MSE 替换为 logit 级交叉熵损失，使量化模型的 token 分布与全精度模型对齐…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "低比特量化"
  - "后训练量化"
  - "交叉熵对齐"
  - "生成质量"
  - "block-wise PTQ"
---

# LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs

**会议**: ICML 2026  
**arXiv**: [2605.29756](https://arxiv.org/abs/2605.29756)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 低比特量化, 后训练量化, 交叉熵对齐, 生成质量, block-wise PTQ  

## 一句话总结
针对 block-wise PTQ 在生成任务上的质量退化问题，LFQ 将最后一个 Transformer block 的量化目标从 MSE 替换为 logit 级交叉熵损失，使量化模型的 token 分布与全精度模型对齐，在 IFEval/GSM8K/MATH500/AIME 等生成基准上一致提升精度。

## 研究背景与动机

**领域现状**：低比特 weight-only PTQ（如 GPTQ、FlexRound、OmniQuant、Block-AP）是 LLM 内存压缩的主流手段。Block-wise PTQ 通过逐块最小化量化块与全精度块输出之间的 MSE，在语言建模（WikiText2）和理解（MMLU）等任务上已能逼近全精度基线。

**现有痛点**：当评测场景转向长文本生成（IFEval）和复杂推理（MATH500、AIME）时，block-wise PTQ 的精度显著下降。尤其是大推理模型（DeepSeek-R1、L1-Max）依赖长链思维来提升任务准确率，量化后的生成轨迹容易偏离正确路径。

**核心矛盾**：问题根源有两个——(1) block-wise PTQ 完全忽略了 unembedding 层（LM head），而最终 block 的输出需要经过 LM head 才生成 token 分布；(2) 即使把 LM head 纳入考虑，MSE 最小化也不保证 token 排序一致。作者用一个 2-token 例子清晰证明：MSE 更小的量化方案反而预测了错误的 top-1 token，而 MSE 更大的方案却保留了正确预测。

**本文目标**：在不改变量化方案和推理内核的前提下，让 block-wise PTQ 的生成质量接近全精度基线。

**切入角度**：最小化交叉熵等价于最小化 KL 散度，而 KL 散度为零当且仅当两个分布完全一致，因此交叉熵比 MSE 更直接地对齐 token 概率分布。

**核心 idea**：只修改最后一个 Transformer block 的量化损失——从 MSE 替换为 logit 级交叉熵，让量化模型的 next-token 分布与全精度模型对齐。

## 方法详解

### 整体框架
LFQ 想解决的是 block-wise PTQ 在生成任务上掉点的问题，办法却只动一处：从第 1 个到倒数第 2 个 Transformer block 照旧逐块最小化 MSE，唯独最后一个 block 把 LM head 接进前向路径、把损失从 MSE 换成 logit 级交叉熵，让量化模型的 next-token 分布对齐全精度模型。它是一个即插即用增强件，量化方案、packing/unpacking 内核、单 GPU 可运行特性全部不变，所以能挂在任意 block-wise PTQ 方法之上。

### 关键设计

**1. 把最后一个 block 的目标从激活 MSE 换成 logit 交叉熵：直接对齐 token 概率**

痛点在于，传统 block-wise PTQ 在激活空间最小化 MSE，但激活上的小误差经 LM head 投影后可能翻转 top-1 token 的排序——作者用一个 2-token 反例证明，MSE 更小的量化方案反而预测错了 top-1，MSE 更大的反而预测对了。LFQ 对最终 block 改为计算全精度 logit $\sigma(X W_{\text{FP}} W_{\text{Head}})$ 与量化 logit $\sigma(X W_q W_{\text{Head}})$ 之间的交叉熵：

$$\mathcal{L}_{\text{CE}} = -\frac{1}{L}\sum_{i,j} \sigma(X W_{\text{FP}} W_{\text{Head}})_{i,j} \log \sigma(X W_q W_{\text{Head}})_{i,j}$$

这之所以有效，是因为最小化交叉熵等价于最小化两个分布的 KL 散度，而 KL 为零当且仅当分布完全一致；相比对绝对数值敏感的 MSE，交叉熵对概率排序敏感，天然保护 top-k 顺序，正好对上"生成任务靠 token 排序而非激活数值"的需求。

**2. 把 LM head 接进最终 block 的前向路径：让优化器看见 unembedding 的影响**

要算上面那个 logit 交叉熵，前提是优化时能"看到" LM head。而标准 block-wise PTQ 只盯着 block 输出激活的 MSE，完全忽略了后续的 LM head 投影，于是优化器对"激活误差会被 head 放大成怎样的 token 分布变化"一无所知。LFQ 让 LM head 权重 $W_{\text{Head}}$ 固定参与前向传播（本身不量化），使梯度能一路传到最终 token 分布。这一步既是交叉熵目标的载体，也单独贡献增益——消融里只加 LM head、损失仍用 MSE 时 IFEval 已经回升，说明"让优化器看见 head"本身就有价值。

**3. 方法无关、只换损失不换参数化：与现有量化生态完全兼容**

正因为 LFQ 只替换最后一个 block 的损失函数、不碰量化参数化方式，它可以无缝套在不同方法上：对 FlexRound 优化 $(s_1, S_2, s_3)$、对 OmniQuant 优化 $(\gamma, \beta)$、对 Block-AP 优化 $(s, W_{\text{FP}})$，都只需把那一处的 MSE 换成 $\mathcal{L}_{\text{CE}}$ 即可。这样换来的是零代价的兼容性——内存开销、推理内核、单 GPU 可运行特性全部和原方法一致，部署侧不需要任何额外适配。

## 实验关键数据

### 主实验（Qwen2.5-7B-Instruct）

| 方法 | 比特 | WikiText2↓ | MMLU↑ | IFEval↑ | MATH500↑ |
|------|------|-----------|-------|---------|----------|
| BF16 基线 | 16 | 6.85 | 73.49 | 70.79 | 74.2 |
| FlexRound | W4 | 7.23 | 72.50 | 69.50 | 72.6 |
| FlexRound+LFQ | W4 | **7.21** | 72.48 | **71.35** | **73.4** |
| FlexRound | W3g128 | 7.63 | 70.13 | 66.54 | 65.6 |
| FlexRound+LFQ | W3g128 | **7.58** | **70.26** | **67.84** | **68.0** |
| OmniQuant | W4 | 7.73 | **71.00** | 68.21 | 69.8 |
| OmniQuant+LFQ | W4 | **7.53** | 70.99 | **69.50** | **71.6** |

### 消融实验（Llama 3.1 8B Instruct, W4）

| 方法 | LM Head | 交叉熵 | WikiText2↓ | IFEval↑ | GSM8K↑ |
|------|---------|--------|-----------|---------|--------|
| FlexRound | ✗ | ✗ | 7.06 | 70.24 | 81.35 |
| +LM Head | ✓ | ✗ | 7.08 | 71.53 | 81.58 |
| +LM Head+CE (LFQ) | ✓ | ✓ | **7.06** | **72.09** | **81.80** |
| OmniQuant | ✗ | ✗ | 7.49 | 70.61 | 78.17 |
| +LM Head | ✓ | ✗ | 7.48 | 71.35 | 78.32 |
| +LM Head+CE (LFQ) | ✓ | ✓ | **7.47** | **71.35** | **79.76** |

### 关键发现
- LFQ 在生成任务（IFEval、MATH500、AIME）上一致提升，而在理解任务（WikiText2、MMLU）上不降反升
- 消融显示两个组件（LM Head 纳入 + 交叉熵损失）各自贡献增量收益，组合效果最佳
- 在推理模型 L1-Qwen-7B-Max 的 W3g128 设置下，LFQ 在 AIME'24 上将 greedy 准确率从 23.33% 恢复到 30.00%（BF16 为 46.67%），恢复了近一半的量化退化

## 亮点与洞察
- 用 2-token 构造性反例直观证明 MSE 与 token 预测一致性之间的脱节，这个分析思路简洁有力，可以推广到其他需要保持离散决策一致性的压缩场景
- 只改最后一个 block 的损失函数就能全面提升生成质量，改动极小但效果显著——体现了"在正确位置施加正确约束"的设计哲学
- 对推理模型"aha moment"（如 "Wait"/"But" token）概率分布的分析揭示了量化模型在关键思维转折点的过度自信问题，这为理解量化对 chain-of-thought 推理的影响提供了新视角
- 在 MoE 模型 Qwen3-30B-A3B 上的验证表明 LFQ 对稀疏激活架构同样有效，AIME'25 greedy 准确率从 53.33% 提升到 60.00%

## 局限与展望，论文未报告 2-bit 结果
- 交叉熵计算需要对全词表做 softmax，词表很大时（如 150K+）可能增加优化开销，论文未讨论此开销
- 只关注 weight-only 量化，未探索 activation 量化场景下 logit 对齐的效果
- 未来可以考虑将交叉熵目标扩展到多个靠后的 block（而非仅最后一个），或结合 temperature scaling 进一步控制分布对齐精度销
- 只关注 weight-only 量化，未探索 activation 量化场景下 logit 对齐的效果

## 相关工作与启发
- **vs FlexRound/OmniQuant/Block-AP**: 这些方法在中间 block 用 MSE 优化已足够好，LFQ 证明只需在最后一个 block 切换损失函数就能补上生成质量的短板
- **vs 知识蒸馏 QAT**: QAT 在整个模型上做端到端的 KL 对齐但需要大量计算，LFQ 以极低成本在最后一个 block 实现了类似的概率对齐效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 观察精准、方法简单但切中要害
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 6 个模型家族、3 种 PTQ 方法、两种比特宽度、多个生成基准
- 写作质量: ⭐⭐⭐⭐⭐ 构造性反例和推理轨迹可视化极具说服力
- 价值: ⭐⭐⭐⭐ 即插即用、零额外推理成本，对 LLM 部署有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] NeUQI: Near-Optimal Uniform Quantization Parameter Initialization for Low-Bit LLMs](neuqi_near-optimal_uniform_quantization_parameter_initialization_for_low-bit_llm.md)
- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/model_compression/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)
- [\[AAAI 2026\] SpecQuant: Spectral Decomposition and Adaptive Truncation for Ultra-Low-Bit LLMs Quantization](../../AAAI2026/model_compression/specquant_spectral_decomposition_and_adaptive_truncation_for_ultra-low-bit_llms_.md)

</div>

<!-- RELATED:END -->
