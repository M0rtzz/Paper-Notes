---
title: >-
  [论文解读] From Language Models over Tokens to Language Models over Characters
description: >-
  [ICML 2025][模型压缩][tokenization] 提出将 token 级语言模型精确转换为字符级语言模型的算法框架，通过定义 covering（最小前缀编码集合）并基于 beam search 近似求解，解决了 prompt boundary 等 token 化导致的用户端问题，同时改善了压缩率（bits/byte）。
tags:
  - ICML 2025
  - 模型压缩
  - tokenization
  - 字符级语言模型
  - 概率流
  - covering
  - 束搜索
---

# From Language Models over Tokens to Language Models over Characters

**会议**: ICML 2025  
**arXiv**: [2412.03719](https://arxiv.org/abs/2412.03719)  
**代码**: [有](https://github.com/genlm/genlm-bytes)  
**领域**: 模型压缩  
**关键词**: tokenization, 字符级语言模型, 概率流, covering, 束搜索

## 一句话总结

提出将 token 级语言模型精确转换为字符级语言模型的算法框架，通过定义 covering（最小前缀编码集合）并基于 beam search 近似求解，解决了 prompt boundary 等 token 化导致的用户端问题，同时改善了压缩率（bits/byte）。

## 研究背景与动机

**现状**：现代 LLM 在数学上是 token 串上的概率分布，而非字符串上的分布。用户通过字符串与模型交互，中间经 tokenizer 转换。

**痛点 — Prompt Boundary Problem**：由于 BPE 等 tokenizer 的非双射性，prompt 末尾一个空格就会导致完全不同的续写结果。例如 GPT-2 对 "the" 和 "the " 的续写完全不同（概率从 0.98 骤降到 5e-7）。Token healing 只能回退一个 token，无法处理更复杂的情况（如 "Hello, worl" 回退一个 token 仍产生 "worlwide"）。

**核心矛盾**：token 级模型的条件概率依赖于 tokenization 方式，而用户实际需要的是字符级的条件概率。直接用典型 tokenization $\tau(\sigma)$ 来条件化是错误的——正确的做法是考虑所有可能的 token 编码。

**切入点**：利用 "概率 101" 的条件概率公式 $p_{\Delta|\Sigma}(\delta|\sigma) = P[Y=\delta | \kappa(Y) \succeq \sigma]$，正确地在字符串上条件化 token 级模型。

**核心 idea**：定义 **covering**（最小前缀编码集合），将无穷求和转化为有限求和，再通过 beam search 高效近似。

## 方法详解

### 整体框架 (pipeline)

1. 给定字符串 $\sigma$，枚举所有 token 串的 covering $\mathcal{C}(\sigma)$
2. 对 covering 中的每个 token 串计算前缀概率
3. 求和得到字符级前缀概率 $\overrightarrow{p_\Sigma}(\sigma) = \sum_{\delta \in \mathcal{C}(\sigma)} \overrightarrow{p_\Delta}(\delta)$
4. 通过前缀概率的比值得到条件概率和字符级分布

### 关键设计

1. **Covering 的定义与性质**：Covering $\mathcal{C}(\sigma)$ 是所有最小地覆盖字符串 $\sigma$ 的 token 串集合。一个 token 串 $\delta$ 最小覆盖 $\sigma$ 当且仅当 $\kappa(\delta_{1..M-1}) \prec \sigma \preceq \kappa(\delta_{1..M})$。核心定理（Proposition 1）证明：在 strict-prefix monotone 的解码器 $\kappa$ 下，前缀概率可用 covering 上的有限求和精确计算。设计动机：将无穷集 $\mathcal{P}(\sigma)$ 的求和归约到有限集 $\mathcal{C}(\sigma)$。

2. **enum_cover 递归枚举算法**：逐字符递归构建 covering。对每个新字符，遍历词汇表中所有可能的 token 扩展，过滤出与目标字符匹配的候选，维护 (概率, 解码串, token 串) 三元组。时间复杂度为 $O(|\Delta| \cdot |\mathcal{C}(\sigma)|)$，最坏情况指数级。设计动机：利用 strict-prefix monotonicity 在枚举过程中剪枝无效路径。

3. **Top-K Bucket Beam Search 剪枝**：将 covering 成员按去掉最后一个 token 后的前缀分桶，保留概率最高的 $K$ 个桶。每步工作量为 $O(K \cdot |\Delta|)$，总时间 $O(N \cdot K \cdot |\Delta|)$，线性于字符串长度。设计动机：covering 的高概率元素通常较少，beam search 可以在小 $K$ 下准确近似。

4. **Bundled Beam 实现优化**：将同一桶中的 token 序列打包为 bundle，用 trie 高效过滤下一个 token 的字符匹配。将 trie 视为局部语言模型，逐字符生成下一个 token。设计动机：减少常数因子，提升实际运行速度。

5. **条件生成算法 (conditional_token_generation)**：从 covering 中按前缀概率比例采样一个 token 串作为 prompt 的 token 级表示，然后从 token 级模型直接续写。Proposition 2 证明此过程等价于正确的条件分布 $p_{\Delta|\Sigma}(\cdot|\sigma)$。与逐字符生成相比，采样到 covering 元素后即可利用 token 级模型的高效自回归生成。

6. **next_character_probability 算法**：复用 enum_cover 输出，分三种情况处理：(1) 完全匹配 → 累加 EOS 概率和所有单 token 扩展概率；(2) 过匹配 → 解码串下一字符直接累加概率。最终除以总前缀概率 $Z$ 即得字符级条件分布。

### 损失函数/训练策略

本文不涉及训练——方法直接在已训练好的 token 级 LLM 上运行，是一个推理时的转换算法。核心优化目标是最大化 beam search 的近似质量（最小化 JSD/byte）。实验基于 wikitext-103-v1 测试集前 4000 字节评估，参考分布使用 $K=128$ 的大 beam size，误差棒为 bootstrapped 95% 置信区间。

## 实验关键数据

### 主实验 — 近似质量 vs 速度

| 模型 | K=8 JSD/byte | K=64 JSD/byte | K=8 速度(bytes/s) | K=64 速度(bytes/s) |
|------|-------------|--------------|------------------|-------------------|
| Llama-3.2-1B | ~1e-4 | ~1e-6 | ~250 | ~30 |
| Meta-Llama-3.1-8B | ~1e-4 | ~1e-6 | ~80 | ~10 |
| DeepSeek-R1-Distill-8B | ~1e-4 | ~1e-6 | ~80 | ~10 |
| phi-4 (14B) | ~1e-4 | ~1e-6 | ~50 | ~8 |

### 压缩率改善（bits/byte）

| 模型 | Token级 bits/byte | 字符级 bits/byte | 改善 |
|------|-------------------|------------------|------|
| Llama-3.2-1B | ~1.05 | ~0.95 | 显著 |
| Meta-Llama-3.1-8B | ~0.85 | ~0.78 | 显著 |
| phi-4 (14B) | ~0.80 | ~0.74 | 显著 |

### 关键发现

- 即使 $K=8$ 的小 beam size，JSD/byte 已在 $10^{-4}$ 量级，近似质量优秀
- 字符级模型的 bits/byte 显著优于 token 级模型的 canonical tokenization
- 高概率 covering 元素中 canonical tokenization 占主导（如 "Hello, world" 的 78 种编码中 top-1 即 canonical）
- 方法在 4 个不同规模的公开模型上均表现一致
- BPE 和 WordPiece 均满足 strict-prefix monotone 条件，方法适用于当前主流 tokenizer
- Error (JSD/byte) 与速度之间存在清晰的 Pareto 曲线，用户可按需求选择 $K$ 值
- 字符串 "Hello, world" 共有 78 种 token 编码，但概率高度集中于 canonical tokenization（top-1 占 >99%）

## 亮点与洞察

- 将 token-to-character 转换问题形式化为概率论问题，covering 的数学定义清晰优美
- 揭示了 "$\tau$ 在推理时无用"的深刻观察：训练后只需要解码器 $\kappa$，而非编码器 $\tau$
- Beam search 的 bucket 策略巧妙——按去掉最后 token 的前缀分桶，允许同一桶内不同的部分匹配 token
- 对 prompt boundary problem 提供了原理性而非启发式的解决方案

## 局限与展望

- 推理速度较慢（~250 bytes/s at K=8 for 1B model），不适合实时应用
- covering 大小最坏情况指数级，虽然实践中高概率元素少但无理论保证
- 仅验证了 4 个模型，缺乏对更大规模模型（>14B）的评估
- 未探索将方法集成到实际 serving 系统中的工程方案

## 相关工作与启发

- **Token healing**（Lundberg & Ribeiro, 2023）：启发式地回退一个 token，本文给出了完整的原理性方案
- **Gastaldi et al., 2025**：tokenization 的一致性估计理论，本文的 covering 框架与之互补
- **心理语言学应用**：计算任意字符子串的 contextual surprisal 可预测人类阅读时间
- 对 constrained decoding（如结构化输出）领域有潜在价值——字符级约束可以直接施加

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次给出 token→character 模型转换的完整数学框架
- 实验充分度: ⭐⭐⭐⭐ 4 个模型、多种 beam size，但缺乏下游任务评估
- 写作质量: ⭐⭐⭐⭐⭐ 数学叙述极为严谨清晰，示例生动
- 价值: ⭐⭐⭐⭐ 对整个 LLM 接口层有基础性影响

<!-- RELATED:START -->

## 相关论文

- [Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)
- [DLP: Dynamic Layerwise Pruning in Large Language Models](dlp_dynamic_layerwise_pruning_in_large_language_models.md)
- [DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](../../CVPR2025/model_compression/dycoke_dynamic_compression_of_tokens_for_fast_video_large_language_models.md)
- [Instruction-Following Pruning for Large Language Models](instruction-following_pruning_for_large_language_models.md)

<!-- RELATED:END -->
