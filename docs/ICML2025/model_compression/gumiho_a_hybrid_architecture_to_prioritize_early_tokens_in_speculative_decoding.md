---
title: >-
  [论文解读] Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding
description: >-
  [ICML 2025][模型压缩][推测解码] 提出 Gumiho，一种用于推测解码的混合 draft 模型架构：前两个 token 使用串行 Transformer 以确保精度，后续 token 使用并行 MLP heads 以提升效率，并通过 Full Tree Attention 机制进一步增加接受长度，在 Vicuna/LLaMA 上实现了最高 3.65x 加速。
tags:
  - ICML 2025
  - 模型压缩
  - 推测解码
  - 混合架构
  - 草稿模型
  - 推理加速
  - 注意力机制
---

# Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding

**会议**: ICML 2025  
**arXiv**: [2503.10135](https://arxiv.org/abs/2503.10135)  
**代码**: [有](https://github.com/AMD-AIG-AIMA/Gumiho)  
**领域**: 模型压缩  
**关键词**: 推测解码, 混合架构, 草稿模型, 推理加速, tree attention

## 一句话总结

提出 Gumiho，一种用于推测解码的混合 draft 模型架构：前两个 token 使用串行 Transformer 以确保精度，后续 token 使用并行 MLP heads 以提升效率，并通过 Full Tree Attention 机制进一步增加接受长度，在 Vicuna/LLaMA 上实现了最高 3.65x 加速。

## 研究背景与动机

**现状**：推测解码（speculative decoding）通过小模型快速生成 draft tokens，大模型并行验证来加速 LLM 推理。现有方法分为两派：Medusa 式并行（快但精度低，因为看不到同轮未验证 token）和 Eagle 式串行（准但慢）。

**痛点**：现有方法对序列中所有位置的 token 一视同仁，使用相同的模型结构和生成范式。但实际上，当第一个错误 token 出现时，后续所有 draft tokens 都被丢弃。因此位置靠前的 token 对整体接受长度的影响远大于靠后的。

**核心矛盾**：串行方法（Eagle）精度高但推理慢；并行方法（Medusa）速度快但精度低。需要在精度与效率之间取得更好的平衡。

**切入点**：数学证明在总准确率不变的情况下，将准确率从后面的 token "搬到"前面的 token 可以提升平均接受长度 $\tau$。

**核心 idea**：对前几个关键 token 使用大模型串行生成（高精度），对后续 token 使用轻量并行生成（高效率）——混合架构。

## 方法详解

### 整体框架 (pipeline)

1. LLM 输出隐状态 $h_t$ 和 token embedding $e(y_t)$，拼接后通过 FC 层降维
2. **串行阶段**：2 层 Transformer $\mathcal{M}_T$ 自回归生成前 2 个 draft tokens
3. **并行阶段**：5 个 MLP heads $\{\mathcal{M}_M^i\}_{i=1}^5$ 以串行输出为输入，并行生成后续 5 个 tokens
4. 总共产生 7 个 draft tokens → Eagle-2 风格 dynamic tree 构建候选 → Full Tree Attention 验证

### 关键设计

1. **Serial-Parallel Hybrid Heads**：前 2 个 token 使用 2 层 Transformer 串行生成，输入为 $o_t = \text{FC}(\text{cat}(e(y_t), h_t))$。Transformer 通过 attention 捕获 token 间依赖。后 5 个 token 使用 5 个独立 MLP（2 层 FC + ReLU），共享输入 $\text{cat}(\hat{o}_{t+1}, \hat{o}_{t+2})$。设计动机：Theorem 3.1 证明了前面 token 的准确率提升对 $\tau$ 的贡献大于后面 token 的同等提升，因此值得为前面 token 投入更多计算。

2. **Theorem 3.1（前置 token 优先定理）**：给定 draft 长度 $D$，原始接受概率 $\{p_i\}$（递减），将前 $d$ 个位置概率增加 $\zeta_i$、后面位置减少等量 $\zeta_i$（总和守恒），则 $\mathbb{E}[L]_{\text{improved}} \geq \mathbb{E}[L]_{\text{original}}$。核心思路：$\mathbb{E}[L] = \sum_{i=1}^D \prod_{j=1}^i p_j$，前面的 $p_j$ 出现在更多乘积项中，因此提升前面的 $p_j$ 收益更大。

3. **Full Tree Attention (FTA)**：并行 heads 生成的 token 之间相互独立，因此可以任意组合。例如 3 个并行 heads 各生成 $s$ 个候选时，有 $s^3$ 种路径但只需计算 $3s$ 个 token 的 QKV。FTA 允许较短候选路径从较长路径借用对应位置的 token，增加接受长度。设计动机：Eagle-2 的 dynamic tree 可能丢弃后续 token 导致候选路径变短；FTA 利用并行 heads 的独立性无代价地补全短路径。

### 损失函数/训练策略

遵循 Eagle-2 的训练方式，使用 LLM 的隐状态和 token 作为监督信号训练 draft model。串行 Transformer 和并行 MLP heads 联合训练。

## 实验关键数据

### 主实验 — Speedup Ratio（Temperature=0）

| 模型 | 方法 | MT-Bench | HumanEval | GSM8K | Alpaca | CNN/DM | Natural Q. | Mean Speedup | Mean τ |
|------|-----|----------|-----------|-------|--------|--------|------------|-------------|--------|
| Vicuna 7B | Medusa | 1.96x | 2.15x | 2.01x | 1.94x | 1.60x | 1.68x | 1.89x | 2.39 |
| Vicuna 7B | Eagle-2 | 2.88x | 3.27x | 2.93x | 2.71x | 2.45x | 2.24x | 2.74x | 4.68 |
| Vicuna 7B | **Gumiho** | **3.15x** | **3.65x** | **3.10x** | **2.83x** | **2.73x** | **2.34x** | **2.97x** | **4.89** |
| Vicuna 13B | Eagle-2 | 3.16x | 3.68x | 3.19x | 3.01x | 2.79x | 2.41x | 2.74x | - |
| Vicuna 13B | **Gumiho** | - | - | - | - | - | - | **提升** | **提升** |

### 多模型结果 — Speedup Ratio（Temperature=0）

| 模型 | 方法 | Mean Speedup | Mean τ |
|------|------|-------------|--------|
| Vicuna 7B | Eagle-2 | 2.74x | 4.68 |
| Vicuna 7B | **Gumiho** | **2.97x** | **4.89** |
| Vicuna 13B | Eagle-2 | 3.04x | 4.67 |
| Vicuna 13B | **Gumiho** | **3.23x** | **4.87** |
| LLaMA2-Chat 7B | Eagle-2 | 2.82x | 4.64 |
| LLaMA2-Chat 7B | **Gumiho** | **2.95x** | **4.70** |

### 消融实验 — Draft Time 对比

| 方法 | 结构 | Draft 时间(相对) | τ |
|------|------|-----------------|-----|
| Medusa | 全并行 MLP | 最快 | 最低 |
| Eagle | 全串行 Transformer | 较慢 | 较高 |
| Eagle-2 | 全串行 Transformer + dynamic tree | 较慢 | 高 |
| Gumiho | 串行 Transformer + 并行 MLP + FTA | 中等 | 最高 |

### 关键发现

- Gumiho 在 6 个数据集上平均加速比 2.97x，超过 Eagle-2 的 2.74x（Vicuna-7B）
- Vicuna 13B 上 Gumiho 达到 3.23x 加速比，Eagle-2 为 3.04x（提升 6.3%）
- LLaMA2-Chat 7B 上 Gumiho 达到 2.95x，Eagle-2 为 2.82x
- HumanEval 上加速最显著：Vicuna 7B Gumiho 3.65x vs Eagle-2 3.27x（τ=5.77 vs 5.35）
- FTA 进一步提升接受长度，尤其在短候选路径上效果明显
- 并行 MLP heads 比 Transformer heads 快约 30-40%，但精度损失可接受（因为是后续 token）
- 温度 >0 时 Gumiho 的优势同样保持

## 亮点与洞察

- 数学定理驱动的架构设计——先证明前面 token 更重要，再据此分配计算资源
- 混合串行-并行的思路简洁有效，容易理解和实现
- FTA 利用并行 heads 的独立性是巧妙的"免费午餐"——不增加 QKV 计算
- 与 Eagle-2 的 dynamic tree 无缝兼容，可视为 Eagle-2 的直接升级版

## 局限与展望

- 串行部分固定为 2 层 Transformer，最优配置可能因模型/任务而异
- 串行-并行的分界点（前 2 个 vs 前 3 个）的选择缺乏自适应机制
- 仅在 Vicuna/LLaMA 系列评估，未验证对 decoder-only 以外架构的适用性
- 并行 MLP heads 对长距离依赖的建模能力有限
- FTA 的候选路径数 $s^K$ 可能随并行 heads 数指数增长，实际使用需限制 $s$ 的大小
- 属于 self-drafting 类方法，需要访问目标 LLM 隐状态，不适用于黑盒 API 场景

## 相关工作与启发

- **Medusa**（Cai et al., 2024）：并行 MLP heads 的原型，Gumiho 的并行部分延续此设计
- **Eagle/Eagle-2**（Li et al., 2024）：串行 Transformer draft model，Gumiho 的串行部分以此为基础
- **Hydra**（Ankner et al., 2024）：串行 MLP，证明串行优于并行但代价是速度
- 启发：可将此混合思路推广到更多次 token 预测场景（如 multi-token prediction pre-training）

## 评分

- 新颖性: ⭐⭐⭐⭐ 混合串行-并行是新的，但各组件为已有方法的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 6 数据集、多模型、多温度、含消融和 draft time 分析
- 写作质量: ⭐⭐⭐⭐ 定理→方法→实验逻辑清晰
- 价值: ⭐⭐⭐⭐ 对推测解码社区有直接实用价值

<!-- RELATED:START -->

## 相关论文

- [VocabTrim: Vocabulary Pruning for Efficient Speculative Decoding in LLMs](vocabtrim_vocabulary_pruning_for_efficient_speculative_decoding_in_llms.md)
- [Reject Only Critical Tokens: Pivot-Aware Speculative Decoding](../../NeurIPS2025/model_compression/reject_only_critical_tokens_pivot-aware_speculative_decoding.md)
- [Traversal Verification for Speculative Tree Decoding](../../NeurIPS2025/model_compression/traversal_verification_for_speculative_tree_decoding.md)
- [Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput](speculative_decoding_in_decentralized_llm_inference_turning_communication_latenc.md)
- [Steering Pretrained Drafters during Speculative Decoding](../../AAAI2026/model_compression/steering_pretrained_drafters_during_speculative_decoding.md)

<!-- RELATED:END -->
