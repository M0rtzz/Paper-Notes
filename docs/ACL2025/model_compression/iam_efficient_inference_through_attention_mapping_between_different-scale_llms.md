---
title: >-
  [论文解读] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs
description: >-
  [ACL 2025][模型压缩] 发现不同规模 LLM 的注意力矩阵具有高度相似性，提出 IAM 框架——在 prefill 阶段建立小模型与大模型注意力头之间的余弦相似度映射，decode 阶段用小模型的注意力矩阵替代大模型映射层的注意力计算，实现 KV cache 减少 22% 和推理加速 11%，且与现有 KV cache 压缩方法正交。
tags:
  - ACL 2025
  - 模型压缩
---

# IAM: Efficient Inference through Attention Mapping between Different-scale LLMs

**会议**: ACL 2025  
**arXiv**: [2507.11953](https://arxiv.org/abs/2507.11953)  
**代码**: 无  
**领域**: 模型压缩  

## 一句话总结

发现不同规模 LLM 的注意力矩阵具有高度相似性，提出 IAM 框架——在 prefill 阶段建立小模型与大模型注意力头之间的余弦相似度映射，decode 阶段用小模型的注意力矩阵替代大模型映射层的注意力计算，实现 KV cache 减少 22% 和推理加速 11%，且与现有 KV cache 压缩方法正交。

## 背景与动机

1. **长上下文推理资源消耗激增**：推理模型（如 DeepSeek-R1）的长思维链和 RAG 场景导致 KV cache 成为长上下文推理的主要瓶颈。
2. **现有方法只利用内部稀疏性**：KV cache eviction（如 H₂O、StreamingLLM）只利用模型自身注意力的稀疏性做 token 级裁剪，未利用外部信息优化。
3. **不同规模模型注意力模式相似**：BERT 时代已有研究发现大小模型注意力模式相似，作者进一步验证这一特性在 decoder-only LLM 中同样成立。
4. **利用相似性的新优化视角**：如果能用小模型的注意力矩阵替代大模型的部分层，可同时节省计算和 KV cache 存储。

## 方法详解

### 整体框架

IAM 在 **prefill 阶段**建立映射关系，**decode 阶段**利用映射进行高效推理。

### 1. 相似度度量选择

比较了余弦相似度、Minkowski 距离、Pearson 相关系数等度量方式。将 SLM 和 LLM 的注意力矩阵展平为向量后计算相似度。

实验结果：**余弦相似度**效果最佳（Qwen2-7B → 72B: log PPL 2.414 vs 原始 2.136），Pearson 不适用（PPL 5.376）。

### 2. 映射层选择策略

将 Qwen2-72B 的 80 层分为 10 个子块（每块 8 层），逐块测试映射后在 MMLU 上的表现：

- **最优映射区域**：最后 2 个子块（64-80 层）和第 2-4 子块（16-40 层）
- **禁止映射区域**：最前 2 个子块（0-16 层），映射后性能崩溃
- 不同任务间的映射效果高度一致（Pearson 相关 0.74-0.95）

**映射策略**：按用户指定映射比例，优先从最后一层向前映射至 64 层，若不够再从 16 层开始向后映射。

### 3. 映射一致性验证

prefill 阶段建立的映射关系在 decode 阶段保持高度一致（consistency rate >90%），证明映射可以一次建立、全程复用。

### 4. 映射与推理

- **Prefill**：SLM 和 LLM 同时处理 prompt，计算所有注意力矩阵的成对余弦相似度，为 LLM 每个映射层的每个头找到 SLM 中最相似的注意力头
- **Decode**：映射层直接使用 SLM 的注意力矩阵，跳过 Q/K 投影和 QK 矩阵乘法，不存储 K cache
- **微调**：用 Alpaca 数据集对 SLM 做 instruction tuning 以适配映射场景
- **阈值机制**：prompt 过短（<τ_e）时先正常生成到阈值再建立映射；过长（>τ_t）时截断计算相似度

## 实验结果

### 性能保持（Qwen2-72B，mapping ratio 0-50%）

30% 映射比例下各任务几乎无损：
- MMLU：~81%（原始 ~82%）
- WikiText PPL：~2.4（原始 2.14）
- HotpotQA F1：~38（原始 ~40）

50% 映射比例下仍保持较高水平。

### 效率评估（Qwen2-72B + Qwen2-0.5B，50% 映射）

| 场景 | KV Cache 减少 | TPOT 加速 | TTFT 加速 | 吞吐提升 |
|:---|:---:|:---:|:---:|:---:|
| 多用户并发（bs=64, 512+512） | 21.7% | 1.11× | 1.12× | 1.11× |
| 长上下文（bs=4, 8192+512） | 22.5% | 1.10× | 1.17× | 1.10× |

长上下文场景 TTFT 提升更显著（17%），因为 QK 矩阵乘法复杂度随序列长度二次增长。

### 跨系列泛化

LLaMA 3.2-1B → LLaMA 3.1-70B 同样有效，但最优映射区域仅在模型末端（与 Qwen 的两段式不同）。

### 与 KV cache 压缩方法兼容

IAM + H₂O（80% budget）联用后，PPL 几乎无额外退化，两者正交互补。

## 亮点

- **利用外部信息的新视角**：首次系统性地利用大小模型注意力相似性进行推理加速，不同于所有现有的内部稀疏性方法
- **双重收益**：同时减少计算量（跳过 QK 计算）和 KV cache（不存 K cache），在 prefill 和 decode 阶段都受益
- **与现有方法正交**：可与 H₂O、StreamingLLM 等 token 级 KV cache 方法叠加使用
- **映射一致性好**：prefill 建立的映射在 decode 全程稳定，无需动态更新

## 局限性

- **需要同系列小模型**：SLM 和 LLM 必须是同一系列（如都是 Qwen2 或都是 LLaMA3），跨系列效果未知
- **不兼容 FlashAttention**：IAM 需要显式计算注意力分数，与 FlashAttention 的融合 kernel 不兼容，这在实际部署中是较大限制
- **层级粒度较粗**：当前以整层为单位映射，若细化到注意力头级别可进一步提升
- **非完全无损**：50% 映射下各任务仍有可感知的性能下降

## 相关工作对比

| 维度 | IAM | H₂O | StreamingLLM | KVQuant |
|:---|:---|:---|:---|:---|
| 优化粒度 | 层级 | Token 级 | Token 级 | 数值精度 |
| 信息来源 | 外部（小模型） | 内部（注意力分数） | 内部（位置） | 内部（数值分布） |
| KV cache 节省 | ~22%（层级跳过） | ~20-40%（token 裁剪） | 固定窗口 | 2-4×（量化） |
| 计算节省 | 有（跳过 QK） | 无 | 无 | 有（低精度） |
| 正交性 | 与上述均正交 | 与 IAM 正交 | 与 IAM 正交 | 与 IAM 正交 |

## 评分

- ⭐⭐⭐⭐ 新颖性：利用大小模型注意力相似性做推理优化是全新视角，区别于所有现有 KV cache 方法
- ⭐⭐⭐⭐ 实用性：与现有方法正交可叠加使用，但不兼容 FlashAttention 是部署硬伤
- ⭐⭐⭐⭐ 实验充分度：4 类场景评估 + 跨系列验证 + 兼容性验证 + 效率分析，比较全面
- ⭐⭐⭐⭐ 写作质量：系统性的三步分析（度量→层选择→一致性）逻辑清晰，图表丰富

<!-- RELATED:START -->

## 相关论文

- [Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [Sci-LoRA: Mixture of Scientific LoRAs for Cross-Domain Lay Paraphrasing](sci-lora_mixture_of_scientific_loras_for_cross-domain_lay_paraphrasing.md)
- [TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)

<!-- RELATED:END -->
