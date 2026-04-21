---
title: >-
  [论文解读] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension
description: >-
  [ICLR 2026][模型压缩][KV缓存压缩] 提出 FreqKV，一种无参数、架构无关的 KV 缓存压缩方法，通过在频域中迭代压缩 KV 状态（保留低频丢弃高频），仅需 8K 长度的少量微调即可将 LLaMA-2-7B 的上下文窗口扩展至 256K，同时保持稳定的困惑度。
tags:
  - ICLR 2026
  - 模型压缩
  - KV缓存压缩
  - 频域变换
  - 上下文窗口扩展
  - DCT
  - 长文本推理
---

# FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension

**会议**: ICLR 2026  
**arXiv**: [2505.00570](https://arxiv.org/abs/2505.00570)  
**代码**: [GitHub](https://github.com/LUMIA-Group/FreqKV)  
**领域**: 模型压缩 / LLM效率  
**关键词**: KV缓存压缩, 频域变换, 上下文窗口扩展, DCT, 长文本推理

## 一句话总结

提出 FreqKV，一种无参数、架构无关的 KV 缓存压缩方法，通过在频域中迭代压缩 KV 状态（保留低频丢弃高频），仅需 8K 长度的少量微调即可将 LLaMA-2-7B 的上下文窗口扩展至 256K，同时保持稳定的困惑度。

## 研究背景与动机

LLM 的推理能力受限于预训练时设定的上下文窗口大小，超出窗口后性能急剧下降。现有解决方案各有局限：

**位置编码方法**（ALiBi、PI、LongRoPE）：依赖完整自注意力，二次计算成本过高

**KV 缓存驱逐方法**（SnapKV、PyramidKV、FastKV）：根据注意力分数丢弃不重要的 token，但被丢弃 token 的信息永久丢失，导致后续解码性能下降，且无法外推超出上下文窗口

**Token 合并方法**（CaM、KVMerger、D2O）：虽保留更多信息但不微调时效果不佳

**额外模块方法**（LoCoCo、Activation Beacon）：引入额外参数压缩 KV 状态，增加内存开销

**关键观察**：作者发现 LLM 的 KV 状态在频域中展现出强烈的能量集中特性——能量主要集中在低频分量中。虽然第一层的初始嵌入没有明显的低频偏置，但随着解码过程推进，后续层逐渐将能量转移至低频。这意味着高频分量基本是冗余的，可以丢弃而损失极小。

进一步的扰动实验证实：低频分量编码全局语义信息和长程依赖，对输入扰动具有鲁棒性；高频分量捕获局部细节，对扰动敏感。在摘要任务上，保留低频分量的性能远优于保留高频分量（如 GovReport: 25.51 vs 14.21）。

## 方法详解

### 整体框架

FreqKV 的核心流程分为三步：

1. 对序列维度上的 KV 缓存执行离散余弦变换（DCT）转换到频域
2. 保留低频分量、丢弃高频分量实现压缩
3. 通过逆 DCT（IDCT）转回时域得到压缩后的 KV 缓存

通过迭代压缩策略，FreqKV 在缓存增长时反复执行上述过程，实现上下文窗口的无限扩展。

### 关键设计

**1. 频域 KV 压缩**

对 KV 缓存沿序列维度执行 DCT：

Z_K = DCT(K), Z_V = DCT(V)

给定保留比率 gamma，缓存大小从 N 压缩为 L = gamma * N，过滤掉 N-L 个高频分量。然后通过 IDCT 转回时域，并乘以缩放因子 sqrt(L/N) 恢复原始幅度：

K_compressed = sqrt(L/N) * IDCT(Z_K[0:L-1])

压缩后的 KV 缓存直接替代原始缓存用于注意力计算。

**2. 迭代压缩策略**

这是 FreqKV 实现上下文扩展的关键。当 KV 缓存达到最大容量 N 时触发压缩，压缩后释放空间接收新 token。流程如下：

- 窗口内的 token 执行标准注意力
- 缓存满时→频域压缩→释放 N-L 个位置
- 新 token 填充到释放的空间→缓存再次满→再次压缩
- 循环往复

效果：越早的 token 经历越多轮压缩，越近的 token 压缩越少——这与自回归 LLM 的自然特性一致（近期 token 更重要）。压缩只在每 N-L 个 token 后触发一次，复杂度为 O(N log N)，计算开销可忽略。

**3. Attention Sink 保留**

遵循 attention sink 现象（LLM 倾向于给初始 token 分配高注意力分数），FreqKV 保留 S 个初始 token 不参与压缩（实验中 S=4）。压缩仅作用于 sink token 之后的缓存内容。

**4. RoPE 集成**

Key 状态在应用 RoPE 之前进行压缩和缓存，注意力计算时再编码 RoPE。压缩后的 key 使用缓存内的位置索引而非原始序列位置，因此无需位置外推或插值即可实现上下文扩展。

### 损失函数 / 训练策略

FreqKV 本身是**无参数方法**，不引入任何额外参数。仅需少量微调即可适配频域压缩模式：

- 对 LLaMA-2：使用 RedPajama 预训练数据集在 8K 长度上微调用于语言建模；使用 LongAlpaca 指令数据集微调 chat 模型
- 对 LLaMA-3：类似地在 16K 长度上微调
- 训练中使用与推理相同的 chunk-wise 处理流程（交替注意力计算和频域压缩）

## 实验关键数据

### 主实验

**表1：PG-19 困惑度评估（LLaMA-2-7B）**

| 训练长度 | 方法 | 推理缓存 | 2K | 4K | 8K | 16K | 32K |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 8K | Full FT | Full | 7.55 | 7.21 | 6.98 | - | - |
| 8K | LoCoCo | Comp. | 8.15 | 8.08 | 7.27 | - | - |
| 8K | LongLoRA | Full | 7.70 | 7.35 | 7.14 | - | - |
| 8K | **FreqKV** | Comp. | **7.45** | **7.12** | **7.04** | **7.02** | **7.02** |
| 32K | LongLoRA | Full | 8.29 | 7.83 | 7.54 | 7.35 | 7.22 |
| 32K | **FreqKV** | Comp. | **7.47** | **7.14** | **7.04** | **7.00** | **6.98** |

FreqKV 仅需 8K 训练就能将上下文扩展到 32K（PPL 7.02），甚至优于 LongLoRA 在 32K 训练后的结果（7.22）。FreqKV 在短上下文（2K/4K）上也不损失性能，甚至优于 Full FT。

**表2：LongBench 长文本理解基准（LLaMA-2-chat-7B, 50% 保留率）**

| 方法 | 单文档QA | 多文档QA | 摘要 | Few-shot | 代码 | 平均 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Cache | 24.9 | 22.5 | 24.6 | 60.0 | 48.1 | 35.17 |
| SnapKV | 25.4 | 22.3 | 24.0 | 59.1 | 48.0 | 36.81 |
| FastKV | 25.5 | 22.9 | 23.7 | 57.6 | 54.5 | 36.33 |
| PyramidKV | 25.3 | 21.3 | 23.9 | 59.8 | 48.0 | 36.84 |
| **FreqKV** | **24.2** | **27.9** | **24.7** | 56.0 | **58.8** | **37.85** |

FreqKV 在 50% 压缩率下超越所有 KV 驱逐和合并方法，尤其在多文档 QA（+5.4 vs SnapKV）和代码任务（+10.8 vs Full Cache）上优势显著。

### 消融实验

论文在附录中提供了多组消融：

- **保留频率分量选择**：仅保留低频远优于仅保留高频（证实低频信息的重要性）
- **保留比率 gamma**：gamma=0.5 是较好的权衡点
- **Sink token 数量 S**：S=4 即可，增大 S 带来的收益递减
- **LLaMA-2-13B**：在更大模型上同样有效，PPL 从 6.95（Full FT at 8K）降至 6.41（FreqKV at 32K）

### 关键发现

1. **频域压缩无信息永久丢失**：与驱逐方法不同，FreqKV 保留了所有 token 的信息（通过低频分量），只是降低了表示精度
2. **少量训练即可大幅扩展**：仅在 8K 训练就可以维持 256K 的稳定 PPL，远优于需要 32K 训练的 LongLoRA
3. **短上下文不退化**：在原始窗口内（2K/4K）性能甚至略有提升
4. **MHA 和 GQA 通用**：在 LLaMA-2（MHA）和 LLaMA-3（GQA）上都有效
5. **迭代压缩的自然对齐**：越早的 token 被压缩越多次，与自回归 LLM 中近期 token 更重要的直觉一致

## 亮点与洞察

- 从频域视角审视 KV 缓存压缩是一个全新且优雅的思路，DCT 的低频能量集中性质为方法提供了坚实的数学基础
- 完全无参数，不需要修改模型架构，即插即用
- 迭代压缩策略自然实现了"近期 token 高精度、远期 token 低精度"的渐进降质
- 通过在 RoPE 之前压缩 key 并重新编码位置，巧妙绕过了位置外推问题

## 局限与展望

1. DCT 假设信号在边界处的对称延拓可能在某些极端 KV 分布下不成立
2. 固定的保留比率 gamma 可能不是所有层和头的最优选择，自适应 gamma 可能带来进一步提升
3. 压缩后的"token"不再对应真实的 token 位置，可能影响某些需要精确位置信息的任务
4. 未在更大规模模型（70B+）上验证
5. 与 Token 驱逐方法正交，未探索两者结合的可能

## 相关工作与启发

FreqKV 首次将频域学习的思想引入 decoder-only LLM 的 KV 缓存压缩。此前频域方法主要用于 CNN 图像处理和 Transformer encoder（如 FNet）。该方法可以与 token 驱逐方法互补使用，也可以启发其他信号处理技术（如小波变换）在 LLM 推理优化中的应用。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首次在 decoder-only LLM 中使用频域 KV 压缩）
- 技术深度: ⭐⭐⭐⭐（DCT/IDCT 理论清晰，迭代压缩设计精巧）
- 实验充分度: ⭐⭐⭐⭐⭐（PPL + LongBench + RULER + 大海捞针 + LongGenBench，多模型多任务）
- 实用性: ⭐⭐⭐⭐⭐（无参数、架构无关、少量训练即可）
- 写作质量: ⭐⭐⭐⭐（频域动机阐述清楚，实验全面）

<!-- RELATED:START -->

## 相关论文

- [QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](../../NeurIPS2025/model_compression/qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](../../AAAI2026/model_compression/earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)
- [Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](cross_domain_lossy_compression_optimal_transport.md)
- [In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)
- [COMI: Coarse-to-fine Context Compression via Marginal Information Gain](comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)

<!-- RELATED:END -->
