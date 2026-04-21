---
title: >-
  [论文解读] Accurate KV Cache Quantization with Outlier Tokens Tracing
description: >-
  [ACL 2025][模型压缩][KV Cache] 发现 KV Cache 的 outlier channel 中存在少量异常 token 偏离先前假设的均匀分布，提出 OTT（Outlier Tokens Tracing）方法，在量化过程中动态追踪并排除这些 token，在 2-bit 量化下实现 6.4x 内存压缩和 2.3x 吞吐提升，同时显著提高精度。
tags:
  - ACL 2025
  - 模型压缩
  - KV Cache
  - 量化
  - Outlier Tokens
  - LLM Inference
  - Memory Efficiency
---

# Accurate KV Cache Quantization with Outlier Tokens Tracing

**会议**: ACL 2025  
**arXiv**: [2505.10938](https://arxiv.org/abs/2505.10938)  
**代码**: [https://github.com/yisunlp/OTT](https://github.com/yisunlp/OTT)  
**领域**: Model Compression / LLM Efficiency  
**关键词**: KV Cache, quantization, Outlier Tokens, LLM Inference, Memory Efficiency

## 一句话总结

发现 KV Cache 的 outlier channel 中存在少量异常 token 偏离先前假设的均匀分布，提出 OTT（Outlier Tokens Tracing）方法，在量化过程中动态追踪并排除这些 token，在 2-bit 量化下实现 6.4x 内存压缩和 2.3x 吞吐提升，同时显著提高精度。

## 研究背景与动机

LLM 的自回归推理中，KV Cache 虽然将计算复杂度从 $O(n^2)$ 降低到 $O(n)$，但引入了大量 GPU 内存开销。以 LLaMA-3-8B 为例，batch size=64，序列长度=8192 时，KV Cache 需要 256GB 内存。

KV Cache 量化是一种有效的压缩方法。先前研究（KIVI）发现：
- **Keys 按通道分布**：某些通道具有极大的幅度，同一通道内分布相对均匀 → 适合 channel-wise 量化
- **Values 按 token 分布** → 适合 token-wise 量化

然而，作者进一步探索发现：**在 outlier channel 中，少数 token 的 Key 值非常小，严重偏离了"同一通道内均匀分布"的假设**。这些异常 token 会极大地扩大量化范围 $(X_{max} - X_{min})$，严重损害量化精度。

## 方法详解

### 整体框架

OTT 在 KIVI 的 channel-wise Key + token-wise Value 量化基础上，增加了一个 **outlier pool**，用于存储被识别的异常 token 的全精度 KV 表示。推理时维护三种 KV Cache：量化的、全精度的（group token + recent token）、outlier pool 中的。

### 关键设计

**1. Outlier Token 的发现与特征**

- 在 outlier channel（幅度极大的通道）中，少数 token 的 Key 值异常小，打破了原本均匀的分布
- 排序后可见明显的"断崖式"跳变：几个极低值突然升至很高值
- 这些 token 在 **所有通道上** 的 Key 幅度也倾向于较小 → 可用 Key 的整体幅度作为识别标准
- **浅层（前 1-2 层）无 outlier token** 现象

**2. Outlier Token 的识别（Quantization 阶段）**

- 定义固定大小的 outlier pool（容量为 `outlier_num`，默认 3）
- 每 $G$ 步（group size）执行一次量化
- 对待量化的 $G$ 个 token 和 outlier pool 中的 token，计算每个 token 的 Key 幅度
- **幅度最小的 token 被选入 outlier pool**（因为它们是导致量化误差最大的）
- 被选中的 outlier token 的 Key 和 Value 被替换为所有 token 的均值，消除对量化的影响
- 额外维护一个 pool（大小 32）存储被淘汰出 outlier pool 的 token，pool 满后停止追踪

**3. 解码阶段**

- Query 分别与三种 Keys 相乘，拼接得到注意力分数
- 注意力分数与对应的三种 Values 相乘后求和得到最终输出
- 使用 **CUDA fused kernel** 加速全精度和量化矩阵的混合乘法

**4. 与 KIVI 的关键差异**

- KIVI 每步压缩 Values，每 $G$ 步压缩 Keys
- OTT 每 $G$ 步同时压缩 Keys 和 Values，处理更一致
- OTT 的 residual length（recent token 滑动窗口）设为 32（KIVI 为 128），因为 outlier pool 已补偿了精度

### 损失函数 / 训练策略

OTT 是 **training-free** 的方法，无需任何训练或微调。所有操作在推理时在线进行，核心是基于 Key 幅度的简单排序和分桶。

## 实验关键数据

### 主实验

**正常上下文评测（GSM8K, BBH, HumanEval）**：

| 数据集 | LLaMA-3-8B FP16 | KIVI 2-bit | OTT 2-bit |
|--------|-----------------|------------|-----------|
| GSM8K (8-shot) | 74.91 | 63.15 | **72.55** |
| BBH (3-shot CoT) | 68.18 | 47.38 | **60.31** |
| HumanEval (p@1) | 40.24 | 28.05 | **40.85** |

在 BBH (3-CoT, LLaMA-3-8B) 上 OTT 比 KIVI 提升了 **12.93%**，几乎恢复到 FP16 水平。

**长上下文评测（LongBench）**：

| 模型 | KIVI Avg | OTT Avg | FP16 Avg |
|------|---------|---------|----------|
| LLaMA-3-8B-Instruct | 47.13 | **49.18** | 50.22 |
| LLaMA-2-13B-chat | 43.06 | **44.10** | 44.30 |

KIVI 在 LLaMA-3-8B 的 LCC 任务上出现显著性能下降（56.58→44.42），而 OTT（52.37）避免了此类崩溃。

**效率对比**：
- 内存压缩比：OTT 和 KIVI 均达 **6.4x**（序列足够长时）
- 吞吐提升：OTT 达 **2.3x**（大 batch size 时优势明显）
- OTT 在任何 batch size 下都比 KIVI 更快，因为不需要每步压缩 Values

### 消融实验

**1. Group Size 与 Residual Length**

- Residual length 增大 → 精度明显提升（固定 G=128，R 从 0 到 128，GSM8K 8-shot 从 70.96 升至 73.24）
- Group size 变化影响不大（分布足够均匀时增大 G 影响有限）
- 主实验选择 G=128, R=32 平衡性能和压缩比

**2. Outlier Num**

| outlier_num | GSM8K (8-shot) | GSM8K (8-CoT) |
|-------------|---------------|--------------|
| 0 | 62.09 | 68.31 |
| 1 | 71.80 | 75.74 |
| 3 | 72.55 | 75.06 |
| 6 | 72.18 | 75.89 |

仅保留 1 个 outlier token 即可带来约 10% 的提升，后续边际收益递减。

**3. 浅层处理**

前 1-2 层设 outlier_num=0 效果最好（浅层确实没有 outlier token 现象）。设置更多层为 0 反而损害性能。

### 关键发现

1. Outlier token 的 Key 在 outlier channel 中的幅度极小，与其他 token 形成"断崖"
2. 这些 token 的**全通道 Key 幅度之和**也最小，因此用 Key magnitude 可以高效精确地识别
3. 保留最小 Key 幅度 token 为全精度效果最好（远优于保留最大的）
4. 浅层（前 2 层）无 outlier token 现象

## 亮点与洞察

1. **观察驱动的简洁设计**：从对 outlier channel 中 token 分布的细致观察出发，方法极为简洁（仅需追踪 3 个 token），但效果惊人
2. **Training-free**：无需任何额外训练，可直接部署
3. **硬件友好**：使用 CUDA fused kernel，实际推理中 outlier 处理的开销微乎其微
4. **互补性强**：可与权重量化、token eviction 等方法组合使用

## 局限与展望

1. 当序列很短（< group size）且 batch size 很大时，压缩比下降
2. 在某些困难数据集上 2-bit 量化仍有微小损失，长生成时误差可能累积
3. 未探索 3-bit 或 4-bit 量化下的效果（可能已足够接近 FP16 无需此方法）
4. outlier pool 满后停止追踪的策略较为简单

## 相关工作与启发

- **KIVI**：channel-wise Key + token-wise Value 量化的奠基工作，OTT 直接建立在此之上
- **StreamingLLM**：保留首尾 token 的思路与 outlier pool 有相似之处
- **Massive Activations**：Sun et al. 发现的大激活模式与 outlier token 现象有潜在联系
- **GPTQ / AWQ**：权重量化方法，可与 KV Cache 量化互补

## 评分

- **创新性**: ★★★★☆ — 对 outlier token 的观察细致且有价值，方法简洁有效
- **实用性**: ★★★★★ — training-free、硬件友好、内存节省显著，工程价值很高
- **实验充分度**: ★★★★☆ — 4 个模型家族、多种基准、充分的消融实验和效率对比
- **写作质量**: ★★★★☆ — 结构清晰，动机自然，图表丰富

<!-- RELATED:START -->

## 相关论文

- [TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)
- [LookaheadKV: Fast and Accurate KV Cache Eviction by Glimpsing into the Future without Generation](../../ICLR2026/model_compression/lookaheadkv_fast_and_accurate_kv_cache_eviction_by_glimpsing_into_the_future_wit.md)

<!-- RELATED:END -->
